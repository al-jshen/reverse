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
  let gradients = c.grad();

  assert_eq!(gradients.wrt(&a), (2. * 2.5).sin());
  assert_eq!(gradients.wrt(&b), 3. / 14.);
}
```

The main type is `Var<'a>`, so you can define functions that take this as an input (possibly along with other `f64` arguments) and also returns this as an output, and the function will be differentiable. For example:

```rust
use reverse::*;

fn main() {
    let graph = Graph::new();
    let params = graph.add_vars(&[5., 2., 0., 1.]);
    let result = diff_fn(&params);
    let gradients = result.grad();
    println!("{:?}", gradients.wrt(&params));
}

// in this case there are no other input arguments, but there could be
fn diff_fn<'a>(params: &[Var<'a>]) -> Var<'a> {
    params[0].powf(params[1]) + params[2].sin() - params[3].asinh() / 2.
}
```

## Differentiable Functions

There is an optional `diff` feature that activates a convenience macro to transform certain functions so that they are differentiable. That is, functions that act on `f64`s can be used without change on `Var`s, and without needing to specify the type.

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
    let grad = res.grad();

    assert_eq!(grad.wrt(&a), 1.);
    assert_eq!(grad.wrt(&b), 4.);
}

// function must have these argument types but can be arbitrarily complex
// apply computations to params and data as if they were f64s
#[differentiable]
fn addmul(params: &[f64], data: &[&[f64]]) -> f64 {
    params[0] + data[0][0] * params[1]
}
```
