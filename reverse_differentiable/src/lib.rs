use std::ops::Deref;

use proc_macro::TokenStream;
use quote::ToTokens;
use syn::parse_macro_input;

#[proc_macro_attribute]
pub fn differentiable(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::ItemFn);

    let visibility = input.vis.to_token_stream().to_string();
    let sig = input.sig;
    let body = input.block.to_token_stream().to_string();

    let function_name = sig.ident.to_string();

    assert!(
        sig.generics.params.is_empty(),
        "Function must not have generic parameters."
    );

    match sig.output {
        syn::ReturnType::Default => panic!("Function must have f64 output type."),
        syn::ReturnType::Type(_, bp) => match bp.deref() {
            syn::Type::Path(p) => {
                let segments = &p.path.segments;
                assert!(segments.len() == 1, "Function must have f64 output type.");
                let out_type = &segments[0].ident;
                assert!(
                    out_type.to_string() == "f64",
                    "Function must have f64 output type."
                );
            }
            _ => panic!("Function must have f64 output type."),
        },
    }

    let args = sig.inputs;
    assert!(
        args.len() == 2,
        "Function must have exactly two input arguments (with types &[f64] and &[&[f64]], in that order)."
    );

    let mut identifiers: Vec<String> = Vec::with_capacity(2);

    match &args[0] {
        syn::FnArg::Receiver(_) => panic!("Function must not take `self` as an argument"),
        syn::FnArg::Typed(pat_type) => {
            match pat_type.pat.deref() {
                syn::Pat::Ident(i) => identifiers.push(i.ident.to_string()),
                _ => {
                    panic!("First function argument must be &[f64].");
                }
            }
            assert!(
                pat_type.ty.deref().to_token_stream().to_string() == "& [f64]",
                "First function argument must be &[f64]"
            );
        }
    }

    match &args[1] {
        syn::FnArg::Receiver(_) => panic!("Function must not take `self` as an argument"),
        syn::FnArg::Typed(pat_type) => {
            match pat_type.pat.deref() {
                syn::Pat::Ident(i) => identifiers.push(i.ident.to_string()),
                _ => {
                    panic!("Second function argument must be & [& [f64]].");
                }
            }
            assert!(
                pat_type.ty.deref().to_token_stream().to_string() == "& [& [f64]]",
                "First function argument must be & [& [f64]]"
            );
        }
    }

    assert!(
        identifiers.len() == 2,
        "Function must have exactly two input arguments (with types &[f64] and &[&[f64]], in that order)."
    );

    let params_ident = &identifiers[0];
    let data_ident = &identifiers[1];

    let replacement = format!(
        "
        use reverse::Var;
        {} fn {}<'a>({}: &[Var<'a>], {}: &[&[Var<'a>]]) -> Var<'a> {{
            {}
        }} ",
        visibility, function_name, params_ident, data_ident, body
    );

    replacement.parse::<TokenStream>().unwrap()
}
