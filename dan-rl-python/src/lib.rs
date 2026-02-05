#![feature(generic_const_exprs)]

#[macro_use]
pub extern crate dan_macros;
pub mod prelude;

use dan_rl_core::{
    sampler::{Sampler, RuleSampler, Sampling},
    traits::{Env, Action},
};
use crate::prelude::PySampling;
use std::sync::mpsc::{TryRecvError, TrySendError};

#[macro_export]
macro_rules! make_module {
    (
        action: $action:ty,
        env_factories: [$($env_name:ident($($env_arg:ident: $env_arg_ty:ty),* $(,)?) => $env:expr),* $(,)?],
        actor_factories: [$($actor_name:ident($($actor_arg:ident: $actor_arg_ty:ty),* $(,)?) => $actor:expr),* $(,)?]
        $(,)?
    ) => {

mod _dan_rl_python_prelude {
    pub use pyo3::{
        prelude::*,
        exceptions::PyRuntimeError,
    };

    pub use dan_rl_core::{
        sampler::{
            Sampler,
            Sampling,
            RuleSampler,
            NeuralSampler,
        },
    };

    pub use dan_rl_python::prelude::{
        PySampling,
        IntoPySampling,
    };

    pub use std::sync::mpsc;
}

use crate::_dan_rl_python_prelude::*;


#[derive(Clone)]
#[pyclass]
pub enum EnvConfig {
    $(
    $env_name {
        $(
        $env_arg: $env_arg_ty,
        )*
    },
    )*
}

#[pymethods]
impl EnvConfig {
    $(
    #[staticmethod]
    #[allow(non_snake_case)]
    fn $env_name($($env_arg: $env_arg_ty),*) -> Self {
        Self::$env_name {
            $(
            $env_arg: $env_arg,
            )*
        }
    }
    )*
}

#[derive(Clone)]
#[pyclass]
pub enum ActorConfig {
    $(
    $actor_name {
        $(
        $actor_arg: $actor_arg_ty,
        )*
    },
    )*
}

#[pymethods]
impl ActorConfig {
    $(
    #[staticmethod]
    #[allow(non_snake_case)]
    fn $actor_name($($actor_arg: $actor_arg_ty),*) -> Self {
        Self::$actor_name {
            $(
            $actor_arg: $actor_arg,
            )*
        }
    }
    )*
}

#[pyclass(unsendable)]
pub struct RuleSamplingWorker {
    env_config: EnvConfig,
    actor_config: ActorConfig,
    buffer: (mpsc::SyncSender<Box<dyn IntoPySampling + Send>>, mpsc::Receiver<Box<dyn IntoPySampling + Send>>),
}

#[pymethods]
impl RuleSamplingWorker {
    #[new]
    fn new(env_config: EnvConfig, actor_config: ActorConfig) -> Self {
        let (sender, receiver) = mpsc::sync_channel(0);
        Self { env_config, actor_config, buffer: (sender, receiver) }
    }


    fn start(&self, py: Python, horizon: usize) -> PyResult<()> {
        // Clone the configs so we can move them into the detached thread
        let env_config = self.env_config.clone();
        let actor_config = self.actor_config.clone();
        let sender = self.buffer.0.clone();
        // py.detach(move || {
        std::thread::spawn(move || {
            dan_rl_python::dan_macros::single_match!((env_config => env in [
                $(
                    EnvConfig::$env_name {
                        $(
                        $env_arg,
                        )*
                    } => $env;
                )*
            ]),{
            dan_rl_python::dan_macros::single_match!((actor_config => actor in [
                $(
                    ActorConfig::$actor_name {
                        $(
                        $actor_arg,
                        )*
                    } => $actor;
                )*
            ]),{
            
            let mut sampler = RuleSampler::new(env, actor);
            dan_rl_python::dan_macros::single_match!((horizon => make_sampling in [
                128 => ||Sampling::<_,_,128>::new();
                256 => ||Sampling::<_,_,256>::new();
                512 => ||Sampling::<_,_,512>::new();
                1024 => ||Sampling::<_,_,1024>::new();
                2048 => ||Sampling::<_,_,2048>::new();
                4096 => ||Sampling::<_,_,4096>::new();
                8192 => ||Sampling::<_,_,8192>::new();
                16384 => ||Sampling::<_,_,16384>::new();
                32768 => ||Sampling::<_,_,32768>::new();
                65536 => ||Sampling::<_,_,65536>::new();
                131072 => ||Sampling::<_,_,131072>::new();
                262144 => ||Sampling::<_,_,262144>::new();
                _ => ||Sampling::<_,_,128>::new();
            ]),
                loop {
                    let mut sampling = make_sampling();
                    sampler.sample(&mut sampling);
                    if sender.send(Box::new(sampling)).is_err() {
                        break;
                    }
                }
            )})});
        });
        Ok(())
    }

    fn recv(&self, py: Python) -> PyResult<PySampling> {
        self.buffer.1.try_recv()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to receive sampling: {}", e)))
        .map(|data|data.into_pysampling(py))
    }

}

#[pymodule]
fn _test_dan_rl_python(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EnvConfig>()?;
    m.add_class::<ActorConfig>()?;
    m.add_class::<RuleSamplingWorker>()?;
    Ok(())
}
}}
