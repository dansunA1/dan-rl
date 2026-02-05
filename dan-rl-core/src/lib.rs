//! Core traits, types, and utilities for HFTRL
//! 
//! This crate provides the fundamental abstractions for environments, actors, and actions
//! without any compile-time configuration dependencies.

#![feature(associated_type_defaults)]
#![feature(generic_const_exprs)]
#![feature(generic_const_items)]

pub mod error;
pub mod types;
pub mod traits;
pub mod sampler;
pub mod prelude;

pub use error::*;
pub use types::*;
pub use traits::*;
pub use sampler::*;
pub use prelude::*;

