#![feature(generic_const_exprs)]
#![feature(generic_const_items)]
pub mod prelude;
pub mod types;
pub mod server;
pub mod codegen;

pub use types::*;
pub use server::*;

use tracing::{Level};
use tracing_subscriber::FmtSubscriber;

pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    // Force DEBUG level, ignoring RUST_LOG if set
    let filter = EnvFilter::new("debug")
        // .add_directive("hftrl=debug".parse().unwrap())
        .add_directive("iceoryx2=debug".parse().unwrap());
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .with_env_filter(filter)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
}