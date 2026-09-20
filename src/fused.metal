#include <metal_stdlib>
using namespace metal;

// A&S 7.1.26, also used by Candle's GELU-erf kernel. Keep the intermediate
// activation's dtype rounding before multiplying the gate, including float16.
float laya_erf(float x) {
    float sign=x<0.0f ? -1.0f : 1.0f;
    x=abs(x);
    float t=1.0f/(1.0f+0.3275911f*x);
    float poly=(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t;
    return sign*(1.0f-poly*exp(-x*x));
}
template<typename T>
kernel void geglu(constant size_t &count,constant size_t &width,device const T *x,device T *out,uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint row=group.y,w=uint(width),first=group.x*1024u+tid*4u;
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;j++) {
        uint col=first+j;
        if(col>=w)continue;
        float value=float(x[row*2u*w+col]);
        T activated=T(value*(1.0f+laya_erf(value*0.7071067811865475f))/2.0f);
        out[row*w+col]=activated*x[row*2u*w+w+col];
    }
}

// Convert [B,L,3,H,D] directly to contiguous [3,B,H,L,D], rotating Q/K.
template<typename T>
kernel void qkv_rope(constant size_t &count,constant size_t &batch,constant size_t &length,constant size_t &heads,constant size_t &dim,
                     device const T *x,device const T *cos,device const T *sin,device T *out,uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint d=uint(dim),len=uint(length),nh=uint(heads),nb=uint(batch);
    uint index=group.x*256u+tid;
    if(index>=len*d)return;
    uint t=d==64u ? index>>6 : d==16u ? index>>4 : index/d;
    uint c=d==64u ? index&63u : d==16u ? index&15u : index%d;
    uint h=group.y,b=group.z%nb,q=group.z/nb;
    uint i=((q*nb+b)*nh+h)*len*d+index;
    uint base=((b*len+t)*3u+q)*nh*d+h*d;
    if(q==2u){out[i]=x[base+c];return;}
    uint halfdim=d/2u,partner=c<halfdim ? c+halfdim : c-halfdim;
    float a=float(x[base+c]),other=float(x[base+partner]);
    float co=float(cos[t*halfdim+c%halfdim]),si=float(sin[t*halfdim+c%halfdim]);
    out[i]=T(a*co+(c<halfdim ? -other : other)*si);
}

template [[host_name("geglu_f32")]] kernel void geglu<float>(constant size_t&,constant size_t&,device const float*,device float*,uint,uint3);
template [[host_name("geglu_f16")]] kernel void geglu<half>(constant size_t&,constant size_t&,device const half*,device half*,uint,uint3);
template [[host_name("qkv_rope_f32")]] kernel void qkv_rope<float>(constant size_t&,constant size_t&,constant size_t&,constant size_t&,constant size_t&,device const float*,device const float*,device const float*,device float*,uint,uint3);
template [[host_name("qkv_rope_f16")]] kernel void qkv_rope<half>(constant size_t&,constant size_t&,constant size_t&,constant size_t&,constant size_t&,device const half*,device const half*,device const half*,device half*,uint,uint3);

kernel void scores_features(constant size_t &batch,constant size_t &width,device const float *scores,device const uint *counts,device float *out,uint row [[thread_position_in_grid]]) {
    if(row>=batch)return;
    uint count=counts[row];
    float maximum=-INFINITY;
    for(uint i=0;i<width;i++)maximum=max(maximum,i<count?scores[row*width+i]:-1e4f);
    float sum=0.0f;
    for(uint i=0;i<width;i++)sum+=exp((i<count?scores[row*width+i]:-1e4f)-maximum);
    float first=0.0f,second=0.0f,entropy=0.0f;
    for(uint i=0;i<width;i++){
        float z=i<count?scores[row*width+i]:-1e4f;
        out[row*(width+4)+i]=z;
        float p=exp(z-maximum)/sum;
        entropy-=p*log(max(p,1e-9f));
        if(p>first){second=first;first=p;}else{second=max(second,p);}
    }
    float k=float(max(count,2u));
    out[row*(width+4)+width]=first;
    out[row*(width+4)+width+1]=first-second;
    out[row*(width+4)+width+2]=entropy/log(k);
    out[row*(width+4)+width+3]=k/255.0f;
}

template<typename T>
kernel void merge_heads(constant size_t &heads,constant size_t &length,constant size_t &dim,device const T *x,device T *out,uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint nh=uint(heads),len=uint(length),d=uint(dim),t=group.y,b=group.z;
    uint first=group.x*1024u+tid*4u;
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;j++) {
        uint col=first+j;
        if(col>=nh*d)continue;
        uint h=d==64u ? col>>6 : d==16u ? col>>4 : col/d;
        uint c=d==64u ? col&63u : d==16u ? col&15u : col%d;
        out[(b*len+t)*nh*d+col]=x[((b*nh+h)*len+t)*d+c];
    }
}
template [[host_name("merge_heads_f32")]] kernel void merge_heads<float>(constant size_t&,constant size_t&,constant size_t&,device const float*,device float*,uint,uint3);
template [[host_name("merge_heads_f16")]] kernel void merge_heads<half>(constant size_t&,constant size_t&,constant size_t&,device const half*,device half*,uint,uint3);

template<typename T>
kernel void pack_heads(constant size_t &batch,constant size_t &length,constant size_t &heads,constant size_t &dim,constant size_t &planes,
                      device const T *x,device T *out,uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint nb=uint(batch),len=uint(length),nh=uint(heads),d=uint(dim),np=uint(planes);
    uint index=group.x*256u+tid;
    if(index>=len*d)return;
    uint t=d==64u ? index>>6 : d==16u ? index>>4 : index/d;
    uint c=d==64u ? index&63u : d==16u ? index&15u : index%d;
    uint h=group.y,b=group.z%nb,p=group.z/nb;
    out[((p*nb+b)*nh+h)*len*d+index]=x[((b*len+t)*np+p)*nh*d+h*d+c];
}
template [[host_name("pack_heads_f32")]] kernel void pack_heads<float>(constant size_t&,constant size_t&,constant size_t&,constant size_t&,constant size_t&,device const float*,device float*,uint,uint3);
template [[host_name("pack_heads_f16")]] kernel void pack_heads<half>(constant size_t&,constant size_t&,constant size_t&,constant size_t&,constant size_t&,device const half*,device half*,uint,uint3);

template<typename T>
kernel void add_rows(constant size_t &rows,constant size_t &cols,device const T *x,device const T *bias,device T *out,uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint width=uint(cols),row=group.z*uint(rows)+group.y,first=group.x*1024u+tid*4u;
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;j++) {
        uint col=first+j;
        if(col<width)out[row*width+col]=x[row*width+col]+bias[group.z*width+col];
    }
}
template [[host_name("add_rows_f32")]] kernel void add_rows<float>(constant size_t&,constant size_t&,device const float*,device const float*,device float*,uint,uint3);
template [[host_name("add_rows_f16")]] kernel void add_rows<half>(constant size_t&,constant size_t&,device const half*,device const half*,device half*,uint,uint3);
