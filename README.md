# reverse

[![Crates.io](https://img.shields.io/crates/v/reverse.svg?style=for-the-badge&color=fc8d62&logo=rust)](https://crates.io/crates/reverse)
[![Documentation](https://img.shields.io/badge/docs.rs-reverse-5E81AC?style=for-the-badge&labelColor=555555&logoColor=white)](https://docs.rs/reverse)
![License](https://img.shields.io/crates/l/reverse?label=License&style=for-the-badge&color=62a69b)

Zero-dependency crate for reverse mode automatic differentiation in Rust.

To use this in your crate, add the following to `Cargo.toml`:

```rust
[dependencies]
reverse = "0.2"
```

## Examples

```rust
use reverse::*;

fn main() {
  let tape = Tape::new();
  let a = tape.add_var(2.5);
  let b = tape.add_var(14.);
  let c = (a.sin().powi(2) + b.ln() * 3.) - 5.;
  let gradients = c.grad();

  assert_eq!(gradients.wrt(&a), (2. * 2.5).sin());
  assert_eq!(gradients.wrt(&b), 3. / 14.);
}
```

The main type is `Var<'a>`, so you can define functions that take this as an input (possibly along with other `f64` arguments) and also returns this as an output, and the function will be differentiable. For example:

```rust
use reverse::*;

fn main() {
    let tape = Tape::new();
    let params = tape.add_vars(&[5., 2., 0.]);
    let data = [1., 2.];
    let result = diff_fn(&params, &data);
    let gradients = result.grad();
    println!("{:?}", gradients.wrt(&params));
}

fn diff_fn<'a>(params: &[Var<'a>], data: &[f64]) -> Var<'a> {
    params[0].powf(params[1]) + data[0].sin() - params[2].asinh() / data[1]
}
```
