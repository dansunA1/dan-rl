use std::fmt::Debug;
use crate::{
    types::{F, Info},
};

pub trait Env: Sized + Debug {
    fn reset(&mut self);
}

pub trait Action<E>: Sized + Debug {
    const DIM: usize;
    fn encode(&self, out: &mut [F]);
    fn decode(from: &[F]) -> Self;
    fn perform(&self, env: &mut E, info: Option<&mut Info>) -> bool;
}

pub trait Tracker<E,A:Action<E>>: Sized + Debug {
    const SDIM: usize;
    const RDIM: usize;
    fn calibrate(&mut self, env: &E);
    fn step(&mut self, action: &A, env: &mut E, info: Option<&mut Info>) -> bool;
    fn write_state(&mut self, state: &mut [F]);
    fn write_reward(&mut self, reward: &mut [F]);
}

pub trait RuleActor<E:Env,A: Action<E>> {
    fn act(&mut self, env: &E) -> (A,F);
}

pub trait NeuralActor<E:Env,const SDIM: usize, const ADIM: usize> {
    fn act(&mut self, state: &[F;SDIM]) -> ([F;ADIM],F);
}