use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub(crate) struct Linear {
    inner: candle_nn::Linear,
    #[cfg(feature = "metal")]
    tuned_m1_pro: bool,
}
impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        #[cfg(feature = "metal")]
        let tuned_m1_pro = {
            use objc2_metal::MTLDevice;
            weight
                .device()
                .as_metal_device()
                .ok()
                .map(|d| d.metal_device().as_ref().name().to_string() == "Apple M1 Pro")
                .unwrap_or(false)
        };
        Self {
            inner: candle_nn::Linear::new(weight, bias),
            #[cfg(feature = "metal")]
            tuned_m1_pro,
        }
    }
    pub fn weight(&self) -> &Tensor {
        self.inner.weight()
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.inner.bias()
    }
}
impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if x.device().is_metal()
            && x.is_contiguous()
            && self.weight().is_contiguous()
            && matches!(x.dtype(), candle_core::DType::F32 | candle_core::DType::F16)
        {
            // M1 Pro identifies as applegpu_g13s (the same suffix as Max), so
            // the generic BasePro enum is not an appropriate tuning guard.
            if self.tuned_m1_pro {
                let rows = x.elem_count() / x.dim(candle_core::D::Minus1)?;
                let tile = if rows <= 128 { "32x32" } else { "64x64" };
                let y = crate::metal::linear(x, self.weight(), tile)?;
                return match self.bias() {
                    Some(bias) => crate::metal::add_rows(&y, bias),
                    None => Ok(y),
                };
            }
            let y = candle_nn::Linear::new(self.weight().clone(), None).forward(x)?;
            return match self.bias() {
                Some(bias) => crate::metal::add_rows(&y, bias),
                None => Ok(y),
            };
        }
        self.inner.forward(x)
    }
}

pub(crate) fn linear(input: usize, output: usize, vb: VarBuilder) -> Result<Linear> {
    let layer = candle_nn::linear(input, output, vb)?;
    Ok(Linear::new(layer.weight().clone(), layer.bias().cloned()))
}
pub(crate) fn linear_b(input: usize, output: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let layer = candle_nn::linear_b(input, output, bias, vb)?;
    Ok(Linear::new(layer.weight().clone(), layer.bias().cloned()))
}
