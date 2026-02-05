pub use dan_config::{
    FromConfig,
};
pub use dan_rl_core::traits::{
    Env,
    Action,
    RuleActor,
};

pub type F = f64;
pub trait PayloadScheme: 'static {
    const SDIM: usize;
    const ADIM: usize;
    const RDIM: usize;
    const N: usize;
    const N_INFO: usize;
    const INFO_KEYS: [&str; Self::N_INFO] where [();Self::N_INFO]: Sized;
}

pub trait WeightUpdateScheme: 'static {
    const MAX_PARAMS: usize;
}