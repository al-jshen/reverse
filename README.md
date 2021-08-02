# reverse

[![Crates.io](https://img.shields.io/crates/v/reverse.svg?style=for-the-badge&color=fc8d62&logo=rust)](https://crates.io/crates/reverse)
[![Documentation](https://img.shields.io/badge/docs.rs-reverse-5E81AC?style=for-the-badge&labelColor=555555&logoColor=white)](https://docs.rs/reverse)
![License](https://img.shields.io/crates/l/reverse?label=License&style=for-the-badge)

Reverse mode automatic differentiation in Rust.

To use this in your crate, add the following to `Cargo.toml`:

```rust
[dependencies]
reverse = "0.1"
```

## Examples

```rust
use reverse::*;

fn main() {
  let graph = Graph::new();
  let a = graph.add_var(2.5);
  let b = graph.add_var(14.);
  let c = (a.sin().powi(2) + b.ln() * 3.) - 5.;
  let gradients = c.backward();

  assert_eq!(gradients.wrt(&a), (2. * 2.5).sin());
  assert_eq!(gradients.wrt(&b), 3. / 14.);
}
```

## Differentiable Functions

There is an optional `diff` feature that activates a macro to transform functions to the right type so that they are differentiable. That is, functions that act on `f64`s can be used on differentiable variables without change, and without needing to specify the (not simple) correct type.

To use this, add the following to `Cargo.toml`:

```rust
reverse = { version = "0.1", features = ["diff"] }
```

Functions must have the type `Fn(&[f64], &[&[f64]]) -> f64`, where the first argument contains the differentiable parameters and the second argument contains arbitrary arrays of data.

### Example

Here is an example of what the feature allows you to do:

```rust
use reverse::*;

fn main() {
    let graph = Graph::new();
    let a = graph.add_var(5.);
    let b = graph.add_var(2.);

    // you can track gradients through the function as usual!
    let res = addmul(&[a, b], &[&[4.]]);
    let grad = res.backward();

    assert_eq!(grad.wrt(&a), 1.);
    assert_eq!(grad.wrt(&b), 4.);
}

// function must have these argument types but can be arbitrarily complex
#[differentiable]
fn addmul(params: &[f64], data: &[&[f64]]) -> f64 {
    params[0] + data[0][0] * params[1]
}
```
