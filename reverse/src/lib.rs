use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Neg, Sub},
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct Node {
    gradients: [f64; 2],
    dependencies: [usize; 2],
}

#[derive(Debug, Clone, Copy)]
pub struct Var<'a> {
    val: f64,
    location: usize,
    graph: &'a Graph,
}

#[derive(Debug, Clone)]
pub struct Graph {
    nodes: RefCell<Vec<Node>>,
}

#[derive(Debug, Clone)]
pub struct Gradient {
    gradients: Vec<f64>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: RefCell::new(vec![]),
        }
    }
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }
    pub fn add_node(&self, loc1: usize, loc2: usize, grad1: f64, grad2: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let n = nodes.len();
        nodes.push(Node {
            gradients: [grad1, grad2],
            dependencies: [loc1, loc2],
        });
        n
    }
    pub fn add_var<'a>(&'a self, val: f64) -> Var<'a> {
        let len = self.len();
        Var {
            val,
            location: self.add_node(len, len, 0., 0.),
            graph: self,
        }
    }
}

impl<'a> Var<'a> {
    pub fn val(&self) -> f64 {
        self.val
    }
    pub fn backward(&self) -> Gradient {
        let n = self.graph.len();
        let mut derivs = vec![0.; n];
        derivs[self.location] = 1.;

        for (idx, n) in self.graph.nodes.borrow().iter().enumerate().rev() {
            derivs[n.dependencies[0]] += n.gradients[0] * derivs[idx];
            derivs[n.dependencies[1]] += n.gradients[1] * derivs[idx];
        }

        Gradient { gradients: derivs }
    }
    pub fn recip(&self) -> Self {
        let val = self.val.recip();
        let m = self.graph.add_var(val);
        Self {
            val,
            location: self
                .graph
                .add_node(self.location, m.location, -1. / (self.val.powi(2)), 0.),
            graph: self.graph,
        }
    }
    pub fn sin(&self) -> Self {
        let val = self.val.sin();
        let m = self.graph.add_var(val);
        Self {
            val,
            location: self
                .graph
                .add_node(self.location, m.location, self.val.cos(), 0.),
            graph: self.graph,
        }
    }
    pub fn cos(&self) -> Self {
        let val = self.val.cos();
        let m = self.graph.add_var(val);
        Self {
            val,
            location: self
                .graph
                .add_node(self.location, m.location, -self.val.sin(), 0.),
            graph: self.graph,
        }
    }
    pub fn ln(&self) -> Self {
        let val = self.val.ln();
        let m = self.graph.add_var(val);
        Self {
            val,
            location: self
                .graph
                .add_node(self.location, m.location, 1. / self.val, 0.),
            graph: self.graph,
        }
    }
    pub fn exp(&self) -> Self {
        let val = self.val.exp();
        let m = self.graph.add_var(val);
        Self {
            val,
            location: self.graph.add_node(self.location, m.location, self.val, 0.),
            graph: self.graph,
        }
    }
    pub fn sqrt(&self) -> Self {
        let val = self.val.sqrt();
        let m = self.graph.add_var(val);
        Self {
            val,
            location: self.graph.add_node(
                self.location,
                m.location,
                1. / (2. * self.val.sqrt()),
                0.,
            ),
            graph: self.graph,
        }
    }
    pub fn abs(&self) -> Self {
        let val = self.val.abs();
        let m = self.graph.add_var(val);
        Self {
            val,
            location: self.graph.add_node(
                self.location,
                m.location,
                if self.val == 0. {
                    f64::NAN
                } else {
                    self.val / val
                },
                0.,
            ),
            graph: self.graph,
        }
    }
}

impl Gradient {
    pub fn wrt(&self, v: &Var) -> f64 {
        self.gradients[v.location]
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
        assert_eq!(self.graph as *const Graph, rhs.graph as *const Graph);
        Self::Output {
            val: self.val + rhs.val,
            location: self.graph.add_node(self.location, rhs.location, 1., 1.),
            graph: self.graph,
        }
    }
}

impl<'a> Add<f64> for Var<'a> {
    type Output = Self;
    fn add(self, rhs: f64) -> Self::Output {
        let rhs_var = self.graph.add_var(rhs);
        Self::Output {
            val: self.val + rhs,
            location: self.graph.add_node(self.location, rhs_var.location, 1., 0.),
            graph: self.graph,
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
        let lhs_var = rhs.graph.add_var(self);
        Self::Output {
            val: self - rhs.val,
            location: rhs.graph.add_node(lhs_var.location, rhs.location, 0., 1.),
            graph: rhs.graph,
        }
    }
}

impl<'a> Mul<Var<'a>> for Var<'a> {
    type Output = Self;
    fn mul(self, rhs: Var<'a>) -> Self::Output {
        assert_eq!(self.graph as *const Graph, rhs.graph as *const Graph);
        Self::Output {
            val: self.val * rhs.val,
            location: self
                .graph
                .add_node(self.location, rhs.location, rhs.val, self.val),
            graph: self.graph,
        }
    }
}

impl<'a> Mul<f64> for Var<'a> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        let rhs_var = self.graph.add_var(rhs);
        Self::Output {
            val: self.val * rhs,
            location: self
                .graph
                .add_node(self.location, rhs_var.location, rhs, 0.),
            graph: self.graph,
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
        let lhs_var = rhs.graph.add_var(self);
        Self::Output {
            val: self / rhs.val,
            location: rhs
                .graph
                .add_node(lhs_var.location, rhs.location, 0., -1. / rhs.val),
            graph: rhs.graph,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ad1() {
        let graph = Graph::new();
        let vars = (0..6).map(|x| graph.add_var(x as f64)).collect::<Vec<_>>();
        let res =
            -vars[0] + vars[1].sin() * vars[2].ln() - vars[3] / vars[4] + 1.5 * vars[5].sqrt();
        let grads = res.backward();
        let est_grads = vars.iter().map(|v| grads.wrt(v)).collect::<Vec<_>>();
        let true_grads = vec![
            -1.,
            2_f64.ln() * 1_f64.cos(),
            1_f64.sin() / 2.,
            -1. / 4.,
            3. / 4_f64.powi(2),
            0.75 / 5_f64.sqrt(),
        ];
        assert_eq!(est_grads, true_grads);
    }

    #[test]
    fn test_ad2() {
        fn f<'a>(a: Var<'a>, b: Var<'a>) -> Var<'a> {
            (a / b - a) * (b / a + a + b) * (a - b)
        }

        let g = Graph::new();
        let a = g.add_var(230.3);
        let b = g.add_var(33.2);
        let y = f(a, b);
        let grads = y.backward();
        assert!((grads.wrt(&a) - -153284.83150602411).abs() < 1e-8);
        assert!((grads.wrt(&b) - 3815.0389441500993).abs() < 1e-8);
    }
}
