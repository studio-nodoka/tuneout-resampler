# Infinite design notes

See the [filter design overview](docs/design.md) and [usage guide](README.md).

The legacy modal initialization uses the fitted autoregressive continuation to
initialize recursive state. Endpoint prediction uses conventional Burg/Levinson
fits with residual, stability and amplitude checks. See [NOTICE.md](NOTICE.md)
for algorithm references and attribution.
