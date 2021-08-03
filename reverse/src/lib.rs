#[cfg(feature = "diff")]
#[macro_use]
extern crate reverse_differentiable;
#[cfg(feature = "diff")]
#[doc(hidden)]
pub use reverse_differentiable::differentiable;

use std::{
    cell::RefCell,
    fmt::Display,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct Node {
    weights: [f64; 2],
    dependencies: [usize; 2],
}

#[derive(Debug, Clone, Copy)]
pub struct Var<'a> {
    val: f64,
    location: usize,
    tape: &'a Tape,
}

#[derive(Debug, Clone)]
pub struct Tape {
    nodes: RefCell<Vec<Node>>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes: RefCell::new(vec![]),
        }
    }
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }
    pub(crate) fn add_node(&self, loc1: usize, loc2: usize, grad1: f64, grad2: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let n = nodes.len();
        nodes.push(Node {
            weights: [grad1, grad2],
            dependencies: [loc1, loc2],
        });
        n
    }
    pub fn add_var<'a>(&'a self, val: f64) -> Var<'a> {
        let len = self.len();
        Var {
            val,
            location: self.add_node(len, len, 0., 0.),
            tape: self,
        }
    }
    pub fn add_vars<'a>(&'a self, vals: &[f64]) -> Vec<Var<'a>> {
        vals.iter().map(|&x| self.add_var(x)).collect()
    }
    pub fn zero_grad(&self) {
        self.nodes
            .borrow_mut()
            .iter_mut()
            .for_each(|n| n.weights = [0., 0.]);
    }
    pub fn clear(&self) {
        self.nodes.borrow_mut().clear();
    }
}

impl<'a> Var<'a> {
    pub fn val(&self) -> f64 {
        self.val
    }
    pub fn grad(&self) -> Vec<f64> {
        let n = self.tape.len();
        let mut derivs = vec![0.; n];
        derivs[self.location] = 1.;

        for (idx, n) in self.tape.nodes.borrow().iter().enumerate().rev() {
            derivs[n.dependencies[0]] += n.weights[0] * derivs[idx];
            derivs[n.dependencies[1]] += n.weights[1] * derivs[idx];
        }

        derivs
    }
    pub fn recip(&self) -> Self {
        Self {
            val: self.val.recip(),
            location: self.tape.add_node(
                self.location,
                self.location,
                -1. / (self.val.powi(2)),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn sin(&self) -> Self {
        Self {
            val: self.val.sin(),
            location: self
                .tape
                .add_node(self.location, self.location, self.val.cos(), 0.),
            tape: self.tape,
        }
    }
    pub fn cos(&self) -> Self {
        Self {
            val: self.val.cos(),
            location: self
                .tape
                .add_node(self.location, self.location, -self.val.sin(), 0.),
            tape: self.tape,
        }
    }
    pub fn tan(&self) -> Self {
        Self {
            val: self.val.tan(),
            location: self.tape.add_node(
                self.location,
                self.location,
                1. / self.val.cos().powi(2),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn ln(&self) -> Self {
        Self {
            val: self.val.ln(),
            location: self
                .tape
                .add_node(self.location, self.location, 1. / self.val, 0.),
            tape: self.tape,
        }
    }
    pub fn log(&self, base: f64) -> Self {
        Self {
            val: self.val.log(base),
            location: self.tape.add_node(
                self.location,
                self.location,
                1. / (self.val * base.ln()),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn log10(&self) -> Self {
        self.log(10.)
    }
    pub fn log2(&self) -> Self {
        self.log(2.)
    }
    pub fn ln_1p(&self) -> Self {
        Self {
            val: self.val.ln_1p(),
            location: self
                .tape
                .add_node(self.location, self.location, 1. / (1. + self.val), 0.),
            tape: self.tape,
        }
    }
    pub fn asin(&self) -> Self {
        Self {
            val: self.val.asin(),
            location: self.tape.add_node(
                self.location,
                self.location,
                1. / (1. - self.val.powi(2)).sqrt(),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn acos(&self) -> Self {
        Self {
            val: self.val.acos(),
            location: self.tape.add_node(
                self.location,
                self.location,
                -1. / (1. - self.val.powi(2)).sqrt(),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn atan(&self) -> Self {
        Self {
            val: self.val.atan(),
            location: self.tape.add_node(
                self.location,
                self.location,
                1. / (1. + self.val.powi(2)),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn sinh(&self) -> Self {
        Self {
            val: self.val.sinh(),
            location: self
                .tape
                .add_node(self.location, self.location, self.val.cosh(), 0.),
            tape: self.tape,
        }
    }
    pub fn cosh(&self) -> Self {
        Self {
            val: self.val.cosh(),
            location: self
                .tape
                .add_node(self.location, self.location, self.val.sinh(), 0.),
            tape: self.tape,
        }
    }
    pub fn tanh(&self) -> Self {
        Self {
            val: self.val.tanh(),
            location: self.tape.add_node(
                self.location,
                self.location,
                1. / (self.val.cosh().powi(2)),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn asinh(&self) -> Self {
        Self {
            val: self.val.asinh(),
            location: self.tape.add_node(
                self.location,
                self.location,
                1. / (1. + self.val.powi(2)).sqrt(),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn acosh(&self) -> Self {
        Self {
            val: self.val.acosh(),
            location: self.tape.add_node(
                self.location,
                self.location,
                1. / (self.val.powi(2) - 1.).sqrt(),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn atanh(&self) -> Self {
        Self {
            val: self.val.atanh(),
            location: self.tape.add_node(
                self.location,
                self.location,
                1. / (1. - self.val.powi(2)),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn exp(&self) -> Self {
        Self {
            val: self.val.exp(),
            location: self
                .tape
                .add_node(self.location, self.location, self.val.exp(), 0.),
            tape: self.tape,
        }
    }
    pub fn exp2(self) -> Self {
        Self {
            val: self.val.exp2(),
            location: self.tape.add_node(
                self.location,
                self.location,
                self.val.exp2() * 2_f64.ln(),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn sqrt(&self) -> Self {
        Self {
            val: self.val.sqrt(),
            location: self.tape.add_node(
                self.location,
                self.location,
                1. / (2. * self.val.sqrt()),
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn cbrt(&self) -> Self {
        self.powf(1. / 3.)
    }
    pub fn abs(&self) -> Self {
        let val = self.val.abs();
        Self {
            val,
            location: self.tape.add_node(
                self.location,
                self.location,
                if self.val == 0. {
                    f64::NAN
                } else {
                    self.val / val
                },
                0.,
            ),
            tape: self.tape,
        }
    }
    pub fn powi(&self, n: i32) -> Self {
        Self {
            val: self.val.powi(n),
            location: self.tape.add_node(
                self.location,
                self.location,
                n as f64 * self.val.powi(n - 1),
                0.,
            ),
            tape: self.tape,
        }
    }
}

impl<'a> Display for Var<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.val)
    }
}

impl<'a> PartialEq for Var<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.val.eq(&other.val)
    }
}

impl<'a> PartialOrd for Var<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl<'a> PartialEq<f64> for Var<'a> {
    fn eq(&self, other: &f64) -> bool {
        self.val.eq(other)
    }
}

impl<'a> PartialOrd<f64> for Var<'a> {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(other)
    }
}

pub trait Gradient<T, S> {
    fn wrt(&self, v: T) -> S;
}

impl<'a> Gradient<&Var<'a>, f64> for Vec<f64> {
    fn wrt(&self, v: &Var) -> f64 {
        self[v.location]
    }
}

impl<'a> Gradient<&Vec<Var<'a>>, Vec<f64>> for Vec<f64> {
    fn wrt(&self, v: &Vec<Var<'a>>) -> Vec<f64> {
        let mut jac = vec![];
        for i in v {
            jac.push(self.wrt(i));
        }
        jac
    }
}

impl<'a> Gradient<&[Var<'a>], Vec<f64>> for Vec<f64> {
    fn wrt(&self, v: &[Var<'a>]) -> Vec<f64> {
        let mut jac = vec![];
        for i in v {
            jac.push(self.wrt(i));
        }
        jac
    }
}
impl<'a, const N: usize> Gradient<[Var<'a>; N], Vec<f64>> for Vec<f64> {
    fn wrt(&self, v: [Var<'a>; N]) -> Vec<f64> {
        let mut jac = vec![];
        for i in v {
            jac.push(self.wrt(&i));
        }
        jac
    }
}
impl<'a, const N: usize> Gradient<&[Var<'a>; N], Vec<f64>> for Vec<f64> {
    fn wrt(&self, v: &[Var<'a>; N]) -> Vec<f64> {
        let mut jac = vec![];
        for i in v {
            jac.push(self.wrt(i));
        }
        jac
    }
}

impl<'a> Neg for Var<'a> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * -1.
    }
}

impl<'a> Add<Var<'a>> for Var<'a> {
    type Output = Self;
    fn add(self, rhs: Var<'a>) -> Self::Output {
        assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
        Self::Output {
            val: self.val + rhs.val,
            location: self.tape.add_node(self.location, rhs.location, 1., 1.),
            tape: self.tape,
        }
    }
}

impl<'a> Add<f64> for Var<'a> {
    type Output = Self;
    fn add(self, rhs: f64) -> Self::Output {
        Self::Output {
            val: self.val + rhs,
            location: self.tape.add_node(self.location, self.location, 1., 0.),
            tape: self.tape,
        }
    }
}

impl<'a> Add<Var<'a>> for f64 {
    type Output = Var<'a>;
    fn add(self, rhs: Var<'a>) -> Self::Output {
        rhs + self
    }
}

impl<'a> Sub<Var<'a>> for Var<'a> {
    type Output = Self;
    fn sub(self, rhs: Var<'a>) -> Self::Output {
        self.add(rhs.neg())
    }
}

impl<'a> Sub<f64> for Var<'a> {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self::Output {
        self.add(rhs.neg())
    }
}

impl<'a> Sub<Var<'a>> for f64 {
    type Output = Var<'a>;
    fn sub(self, rhs: Var<'a>) -> Self::Output {
        Self::Output {
            val: self - rhs.val,
            location: rhs.tape.add_node(rhs.location, rhs.location, 0., -1.),
            tape: rhs.tape,
        }
    }
}

impl<'a> Mul<Var<'a>> for Var<'a> {
    type Output = Self;
    fn mul(self, rhs: Var<'a>) -> Self::Output {
        assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
        Self::Output {
            val: self.val * rhs.val,
            location: self
                .tape
                .add_node(self.location, rhs.location, rhs.val, self.val),
            tape: self.tape,
        }
    }
}

impl<'a> Mul<f64> for Var<'a> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self::Output {
            val: self.val * rhs,
            location: self.tape.add_node(self.location, self.location, rhs, 0.),
            tape: self.tape,
        }
    }
}

impl<'a> Mul<Var<'a>> for f64 {
    type Output = Var<'a>;
    fn mul(self, rhs: Var<'a>) -> Self::Output {
        rhs * self
    }
}

impl<'a> Div<Var<'a>> for Var<'a> {
    type Output = Self;
    fn div(self, rhs: Var<'a>) -> Self::Output {
        self * rhs.recip()
    }
}

impl<'a> Div<f64> for Var<'a> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        self * rhs.recip()
    }
}

impl<'a> Div<Var<'a>> for f64 {
    type Output = Var<'a>;
    fn div(self, rhs: Var<'a>) -> Self::Output {
        Self::Output {
            val: self / rhs.val,
            location: rhs
                .tape
                .add_node(rhs.location, rhs.location, 0., -1. / rhs.val),
            tape: rhs.tape,
        }
    }
}

pub trait Powf<T> {
    type Output;
    fn powf(&self, other: T) -> Self::Output;
}

impl<'a> Powf<Var<'a>> for Var<'a> {
    type Output = Var<'a>;
    fn powf(&self, rhs: Var<'a>) -> Self::Output {
        assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
        Self {
            val: self.val.powf(rhs.val),
            location: self.tape.add_node(
                self.location,
                rhs.location,
                rhs.val * f64::powf(self.val, rhs.val - 1.),
                f64::powf(self.val, rhs.val) * f64::ln(self.val),
            ),
            tape: self.tape,
        }
    }
}

impl<'a> Powf<f64> for Var<'a> {
    type Output = Var<'a>;
    fn powf(&self, n: f64) -> Self::Output {
        Self {
            val: f64::powf(self.val, n),
            location: self.tape.add_node(
                self.location,
                self.location,
                n * f64::powf(self.val, n - 1.),
                0.,
            ),
            tape: self.tape,
        }
    }
}

impl<'a> Powf<Var<'a>> for f64 {
    type Output = Var<'a>;
    fn powf(&self, rhs: Var<'a>) -> Self::Output {
        Self::Output {
            val: f64::powf(*self, rhs.val),
            location: rhs.tape.add_node(
                rhs.location,
                rhs.location,
                0.,
                rhs.val * f64::powf(*self, rhs.val - 1.),
            ),
            tape: rhs.tape,
        }
    }
}

impl<'a> Sum<Var<'a>> for Var<'a> {
    fn sum<I: Iterator<Item = Var<'a>>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_ad0() {
        let g = Tape::new();
        let a = g.add_var(2.);
        let b = a.exp() / 5.;
        let c = a.exp2() / 5.;
        let gradb = b.grad().wrt(&a);
        let gradc = c.grad().wrt(&a);
        assert_eq!(gradb, 2_f64.exp() / 5.);
        assert_eq!(gradc, 1. / 5. * 2_f64.exp2() * 2_f64.ln());
    }

    #[test]
    fn test_ad1() {
        let tape = Tape::new();
        let vars = (0..6).map(|x| tape.add_var(x as f64)).collect::<Vec<_>>();
        let res =
            -vars[0] + vars[1].sin() * vars[2].ln() - vars[3] / vars[4] + 1.5 * vars[5].sqrt();
        let grads = res.grad();
        let est_grads = vars.iter().map(|v| grads.wrt(v)).collect::<Vec<_>>();
        let true_grads = vec![
            -1.,
            2_f64.ln() * 1_f64.cos(),
            1_f64.sin() / 2.,
            -1. / 4.,
            3. / 4_f64.powi(2),
            0.75 / 5_f64.sqrt(),
        ];
        for i in 0..6 {
            assert_approx_eq!(est_grads[i], true_grads[i]);
        }
    }

    #[test]
    fn test_ad2() {
        fn f<'a>(a: Var<'a>, b: Var<'a>) -> Var<'a> {
            (a / b - a) * (b / a + a + b) * (a - b)
        }

        let g = Tape::new();
        let a = g.add_var(230.3);
        let b = g.add_var(33.2);
        let y = f(a, b);
        let grads = y.grad();
        assert_approx_eq!(grads.wrt(&a), -153284.83150602411);
        assert_approx_eq!(grads.wrt(&b), 3815.0389441500993);
    }

    #[test]
    fn test_ad3() {
        let g = Tape::new();
        let a = g.add_var(10.1);
        let b = g.add_var(2.5);
        let c = g.add_var(4.0);
        let x = g.add_var(1.0);
        let y = g.add_var(2.0);
        let res = a.powf(b) - c * x / y;
        let grads = res.grad();
        assert_approx_eq!(grads.wrt(&a), 2.5 * 10.1_f64.powf(2.5 - 1.));
        assert_approx_eq!(grads.wrt(&b), 10.1_f64.powf(2.5) * 10.1_f64.ln());
        assert_approx_eq!(grads.wrt(&c), -1. / 2.);
        assert_approx_eq!(grads.wrt(&x), -4. / 2.);
        assert_approx_eq!(grads.wrt(&y), 4. * 1. / (2_f64.powi(2)));
    }

    #[test]
    fn test_ad4() {
        let g = Tape::new();
        let params = (0..5).map(|x| g.add_var(x as f64)).collect::<Vec<_>>();
        let sum = params.iter().copied().sum::<Var>();
        let derivs = sum.grad();
        for i in derivs.wrt(&params) {
            assert_approx_eq!(i, 1.);
        }
    }

    #[test]
    fn test_ad5() {
        let g = Tape::new();
        let a = g.add_var(2.);
        let b = g.add_var(3.2);
        let c = g.add_var(-4.5);
        let res = a.exp2() / (b.powf(c) + 5.).sqrt();
        let est_grads = res.grad().wrt(&[a, b, c]);
        let true_grads = vec![
            2_f64.exp2() * 2_f64.ln() / ((3.2_f64).powf(-4.5) + 5.).sqrt(),
            -((2. - 1_f64).exp2() * (-4.5) * (3.2_f64).powf(-4.5 - 1.))
                / ((3.2_f64.powf(-4.5) + 5.).powf(1.5)),
            -((2. - 1_f64).exp2() * (3.2_f64).powf(-4.5) * (3.2_f64).ln())
                / ((3.2_f64).powf(-4.5) + 5.).powf(1.5),
        ];
        for i in 0..3 {
            assert_approx_eq!(est_grads[i], true_grads[i]);
        }
    }

    #[test]
    fn test_ad6() {
        let g = Tape::new();
        let a = g.add_var(10.1);
        let b = g.add_var(2.5);
        let c = g.add_var(4.0);
        let x = g.add_var(-1.0);
        let y = g.add_var(2.0);
        let z = g.add_var(-5.);
        let params = [a, b, c, x, y, z];
        let res = a.tan() * b.log2() + c.exp() / (x.powi(2) + 2.) - y.powf(z);
        let est_grads = res.grad().wrt(&params);
        let true_grads = vec![
            2.5_f64.ln() / (2_f64.ln() * 10.1_f64.cos().powi(2)),
            10.1_f64.tan() / (2.5 * 2_f64.ln()),
            4_f64.exp() / ((-1_f64).powi(2) + 2.),
            -2. * 4_f64.exp() * (-1_f64) / ((-1_f64).powi(2) + 2.).powi(2),
            -5_f64 * -2_f64.powf(-5. - 1.),
            -2_f64.powf(-5.) * 2_f64.ln(),
        ];
        for i in 0..6 {
            assert_approx_eq!(est_grads[i], true_grads[i]);
        }
    }

    #[test]
    fn test_ad7() {
        let g = Tape::new();
        let v = g.add_var(0.5);

        let res = v.powi(2) + 5.;
        let grad = res.grad().wrt(&v);
        assert_approx_eq!(grad, 2. * 0.5);

        let res = (v.powi(2) + 5.).powi(2);
        let grad = res.grad().wrt(&v);
        assert_approx_eq!(grad, 4. * 0.5 * (0.5_f64.powi(2) + 5.));

        let res = (v.powi(2) + 5.).powi(2) / 2.;
        let grad = res.grad().wrt(&v);
        assert_approx_eq!(grad, 2. * 0.5 * (0.5_f64.powi(2) + 5.));

        let res = (v.powi(2) + 5.).powi(2) / 2. - v;
        let grad = res.grad().wrt(&v);
        assert_approx_eq!(grad, 2. * 0.5 * (0.5_f64.powi(2) + 5.) - 1.);

        let res = (v.powi(2) + 5.).powi(2) / 2. - v.powi(3);
        let grad = res.grad().wrt(&v);
        assert_approx_eq!(grad, 0.5 * (2. * 0.5_f64.powi(2) - 3. * 0.5 + 10.));

        let res = ((v.powi(2) + 5.).powi(2) / 2. - v.powi(3)).powi(2);
        let grad = res.grad().wrt(&v);
        assert_approx_eq!(
            grad,
            0.5 * (2. * 0.5_f64.powi(2) - 3. * 0.5 + 10.)
                * (0.5_f64.powi(4) - 2. * 0.5_f64.powi(3) + 10. * 0.5_f64.powi(2) + 25.)
        );
    }

    #[test]
    fn test_rosenbrock() {
        let g = Tape::new();
        let x = g.add_var(5.);
        let y = g.add_var(-2.);

        let res = (1. - x).powi(2);
        let grad = res.grad().wrt(&[x, y]);
        assert_approx_eq!(grad[0], -2. * (1. - 5.));
        assert_approx_eq!(grad[1], 0.);

        let res = 100. * (y - x.powi(2)).powi(2);
        let grad = res.grad().wrt(&[x, y]);
        assert_approx_eq!(grad[0], -400. * 5. * (-2. - 5_f64.powi(2)));
        assert_approx_eq!(grad[1], 200. * (-2. - 5_f64.powi(2)));

        let res = (1. - x).powi(2) + 100. * (y - x.powi(2)).powi(2);
        let grad = res.grad().wrt(&[x, y]);
        assert_approx_eq!(
            grad[0],
            2. * (200. * 5_f64.powi(3) - 200. * 5. * -2. + 5. - 1.)
        );
        assert_approx_eq!(grad[1], 200. * (-2. - 5_f64.powi(2)));
    }
}
