# Security

Security fixes target the latest release and the default development branch.

Use GitHub's **Security → Report a vulnerability** on
[b0xtch/laya-candle](https://github.com/b0xtch/laya-candle/security/advisories/new)
when private reporting is enabled. If that option is unavailable, open an issue asking
for a private contact without disclosing the vulnerability, credentials, or exploit
details. Maintainers should enable private reporting before the public launch.

Include the affected version, backend, a minimal synthetic reproducer, impact, and
any proposed fix. Do not attach access tokens, private checkpoints, or user data.

Load checkpoints from trusted sources and pin a Hub commit with `--revision` when
reproducibility matters. `--offline` disables Hub downloads. Safetensors avoids
pickle deserialization, but model files still control allocations and kernel shapes;
this library does not sandbox untrusted checkpoints. Applications accepting remote
requests should impose their own input, batch, memory, and concurrency limits.

The CLI's `inspect` command emits prepared token IDs and raw model outputs. Treat
these as input-derived data when collecting logs. Hub downloads can use `HF_TOKEN`;
never commit that token or an environment file.

The lockfile audit on 2026-09-20 found no known vulnerabilities. Two transitive
maintenance advisories remain visible in `cargo audit`:

- [`number_prefix`, RUSTSEC-2025-0119](https://rustsec.org/advisories/RUSTSEC-2025-0119.html),
  through `hf-hub → indicatif`.
- [`paste`, RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436.html),
  through Candle's GEMM stack and tokenizers.

These are unmaintained-crate notices, not reported vulnerabilities. Track upstream
replacements when updating dependencies. CI audits the lockfile without advisory
exemptions; the dated result above is not a continuing security guarantee.

The optional Python reference environment is audited separately, including its
resolved dependencies. Transformers is pinned to patched release 5.10.4; the
5.0.0 version recorded in the golden fixture describes historical provenance.
