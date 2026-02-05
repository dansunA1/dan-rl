//! Sampling functionality for RL environments and actors
use std::{marker::PhantomData, fmt::Debug};
use crate::{
    types::{
        F
    },
    traits::{
        Info,
        Env, 
        Action, 
        RuleActor, 
        NeuralActor
    },
};

pub struct Sampling<E:Env,A:Action<E>,const N: usize> 
where
    [();A::DIM]: Sized,
    [();E::SDIM]: Sized,
    [();E::RDIM]: Sized,
    [();<E::Info as Info>::N]: Sized,
{
    pub state: [[F;E::SDIM];N],
    pub action: [[F;A::DIM];N],
    pub logprob: [F;N],
    pub reward: [[F;E::RDIM];N],
    pub info: [[F;<E::Info as Info>::N];N],
    pub truncate: [bool;N],
    pub terminate: [bool;N],
}

impl<E:Env,A:Action<E>,const N: usize> Sampling<E,A,N> 
where
    [();A::DIM]: Sized,
    [();E::SDIM]: Sized,
    [();E::RDIM]: Sized,
    [();<E::Info as Info>::N]: Sized,
{
    pub fn new() -> Self {
        Self {
            state: [[F::NAN;E::SDIM];N],
            action: [[F::NAN;A::DIM];N],
            logprob: [F::NAN;N],
            reward: [[F::NAN;E::RDIM];N],
            info: [[F::NAN;<E::Info as Info>::N];N],
            truncate: [false;N],
            terminate: [false;N],
        }
    }
}

pub trait Sampler<E:Env,A:Action<E>>
where
    [();A::DIM]: Sized,
    [();E::SDIM]: Sized,
    [();E::RDIM]: Sized,
    [();<E::Info as Info>::N]: Sized,
{
    fn sample<const N: usize>(&mut self, output: &mut Sampling<E,A,N>);
}

#[derive(Debug)]
pub struct RuleSampler<E:Env,A:Action<E>,Ac:RuleActor<E,A>> {
    env: E,
    actor: Ac,
    phantom: PhantomData<A>,
}

impl<E:Env,A:Action<E>,Ac:RuleActor<E,A>> RuleSampler<E,A,Ac> {
    pub fn new(env:E,actor:Ac) -> Self {
        Self {
            env,
            actor,
            phantom: PhantomData,
        }
    }
}

impl<E:Env,A:Action<E>,Ac:RuleActor<E,A>> Sampler<E,A> for RuleSampler<E,A,Ac> 
where
    [();A::DIM]: Sized,
    [();E::SDIM]: Sized,
    [();E::RDIM]: Sized,
    [();<E::Info as Info>::N]: Sized,
{
    fn sample<const N: usize>(&mut self, r: &mut Sampling<E,A,N>) {
        for i in 0..N {
            self.env.write_state(&mut r.state[i]);
            let (action, logprob) = self.actor.act(&self.env);
            action.encode(&mut r.action[i]);
            r.logprob[i] = logprob;
            let (truncate, terminate) = action.perform(&mut self.env);
            r.truncate[i] = truncate;
            if truncate {
                continue;
            }
            self.env.write_reward(&mut r.reward[i]);
            self.env.info().write(&mut r.info[i]);
            r.terminate[i] = terminate;
        }
        r.truncate[N-1] = true;
    }
}

#[derive(Debug)]
pub struct NeuralSampler<E:Env,A:Action<E>,Ac:NeuralActor<{E::SDIM},{A::DIM}>> {
    env: E,
    actor: Ac,
    phantom: PhantomData<A>,
}

impl<E:Env,A:Action<E>,Ac:NeuralActor<{E::SDIM},{A::DIM}>> NeuralSampler<E,A,Ac> {
    pub fn new(env:E,actor:Ac) -> Self 
    where
        [();E::SDIM]: Sized,
        [();A::DIM]: Sized,
    {
        Self {
            env,
            actor,
            phantom: PhantomData,
        }
    }
    pub fn update_weights(&mut self, weights: &[F]) {
        self.actor.update_weights(weights);
    }
}

impl<E:Env,A:Action<E>,Ac:NeuralActor<{E::SDIM},{A::DIM}>> Sampler<E,A> for NeuralSampler<E,A,Ac> 
where
    [();A::DIM]: Sized,
    [();E::SDIM]: Sized,
    [();E::RDIM]: Sized,
    [();<E::Info as Info>::N]: Sized,
{
    fn sample<const N: usize>(&mut self, r: &mut Sampling<E,A,N>) {
        for i in 0..N {
            self.env.write_state(&mut r.state[i]);
            let (action_emb, logprob) = self.actor.act(&r.state[i]);
            r.action[i] = action_emb;
            r.logprob[i] = logprob;
            let action = A::decode(&mut r.action[i]);
            let (truncate, terminate) = action.perform(&mut self.env);
            r.truncate[i] = truncate;
            if truncate {
                continue;
            }
            self.env.write_reward(&mut r.reward[i]);
            self.env.info().write(&mut r.info[i]);
            r.terminate[i] = terminate;
        }
        r.truncate[N-1] = true;
    }
}
