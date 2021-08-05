use crate::Var;

macro_rules! impl_fn_helper1 {
    ( $($fn: tt),+ ) => {
        $(
            fn $fn(self) -> Self;
        )+
    };
}

macro_rules! impl_fn_helper2a {
    ( $($fn: tt),+ ) => {
        $(
            fn $fn(self) -> Self {
                Self::$fn(self)
            }
        )+
    };
}

macro_rules! impl_fn_helper2b {
    ( $($fn: tt),+ ) => {
        $(
            fn $fn(self) -> Self {
                Self::$fn(&self)
            }
        )+
    };
}

pub trait Numeric {
    impl_fn_helper1!(
        ln, ln_1p, abs, sin, cos, tan, asin, acos, atan, asinh, acosh, atanh, sinh, cosh, tanh,
        exp, exp2, log10, log2, recip, sqrt, cbrt
    );
    fn powi(self, n: i32) -> Self;
}

impl Numeric for f64 {
    impl_fn_helper2a!(
        ln, ln_1p, abs, sin, cos, tan, asin, acos, atan, asinh, acosh, atanh, sinh, cosh, tanh,
        exp, exp2, log10, log2, recip, sqrt, cbrt
    );
    fn powi(self, n: i32) -> Self {
        Self::powi(self, n)
    }
}

impl<'a> Numeric for Var<'a> {
    impl_fn_helper2b!(
        ln, ln_1p, abs, sin, cos, tan, asin, acos, atan, asinh, acosh, atanh, sinh, cosh, tanh,
        exp, exp2, log10, log2, recip, sqrt, cbrt
    );
    fn powi(self, n: i32) -> Self {
        Self::powi(&self, n)
    }
}
