use crate::{
    types::{
        F
    },
};


pub trait Info: Sized {
    const N: usize;
    const KEYS: [&str; Self::N] where [();Self::N]: Sized;
    fn write(&self, info: &mut [F;Self::N]) where [();Self::N]: Sized;
}

pub trait Action<E:Env>: Sized {
    const DIM: usize;
    fn encode(&self, out: &mut [F;Self::DIM]);
    fn decode(from: &[F;Self::DIM]) -> Self;
    fn perform(&self, env: &mut E) -> (bool,bool);
}

pub trait Env: Sized {
    const SDIM: usize;
    const RDIM: usize;
    type Info: Info;
    fn info(&self) -> &Self::Info;
    fn write_state(&self, state: &mut [F;Self::SDIM]) where [();Self::SDIM]: Sized;
    fn write_reward(&self, reward: &mut [F;Self::RDIM]) where [();Self::RDIM]: Sized;
}

pub trait RuleActor<E:Env,A: Action<E>> {
    fn act(&mut self, env: &E) -> (A,F);
    fn reset(&mut self);
}

pub trait NeuralActor<const SDIM: usize, const ADIM: usize> {
    fn act(&mut self, state: &[F;SDIM]) -> ([F;ADIM],F) where [();SDIM]: Sized, [();ADIM]: Sized;
    fn reset(&mut self);
    fn update_weights(&mut self, weights: &[F]);
}