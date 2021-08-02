use std::borrow::Borrow;

use reverse::*;

fn main() {
    // let a = Var::new(1.3);
    // let b = Var::new(0.9);
    // let c = 5.;
    // let res = a + b * c;
    // res.grad(a);

    let graph = Graph::new();
    let a = graph.add_var(2.5);
    let b = graph.add_var(14.);
    let c = (a.sin() + b.ln() * 3.) - 5.;
    let gradients = c.backward();

    assert_eq!(gradients.wrt(&a), 2.5_f64.cos());
    assert_eq!(gradients.wrt(&b), 3. / 14.);
}
