//! Sampling functionality for RL environments and actors
use std::{marker::PhantomData, fmt::Debug};
use crate::{
    types::{F, InfoFrameMapMut},
    traits::{
        Env, 
        Tracker, 
        Action, 
        RuleActor, 
        NeuralActor
    },
};

#[derive(Debug)]
pub struct RuleSampler<E:Env,T:Tracker<E,A>,A:Action<E>,Ac:RuleActor<E,A>> {
    env: E,
    tracker: T,
    actor: Ac,
    phantom: PhantomData<A>,
}

impl<E:Env,T:Tracker<E,A>,A:Action<E>,Ac:RuleActor<E,A>> RuleSampler<E,T,A,Ac> {
    pub fn new(env:E,mut tracker:T,actor:Ac) -> Self {
        tracker.calibrate(&env);
        Self {
            env,
            tracker,
            actor,
            phantom: PhantomData,
        }
    }

    fn write_state(&mut self, state_out: &mut [F;T::SDIM]) {
        self.tracker.write_state(state_out);
    }

    pub fn sample_rule<const N: usize>(
        &mut self,
        actor: &mut Ac,
        state_out:&mut [[F;T::SDIM];N],
        action_out:&mut [[F;A::DIM];N],
        logprob_out:&mut [F;N],
        reward_out:&mut [[F;T::RDIM];N],
        truncate_out:&mut [bool;N],
        terminate_out:&mut [bool;N],
        info_out:Option<&mut InfoFrameMapMut<N>>) 
    {
        match info_out {
            Some(info) => 
            for i in 0..N {
                let (action, logprob) = actor.act(&self.env);
                action.encode(&mut action_out[i]);
                logprob_out[i] = logprob;
                let truncate = action.perform(&mut self.env, Some(info[i]));
                truncate_out[i] = truncate;
                if truncate {
                    self.env.reset();
                    self.tracker.calibrate(&self.env);
                    continue;
                }
                let terminate = !self.tracker.step(&action,&mut self.env, Some(info[i]));
                if terminate {
                    self.env.reset();
                    self.tracker.calibrate(&self.env);
                }
                self.tracker.write_state(&mut state_out[i]);
                self.tracker.write_reward(&mut reward_out[i]);
                terminate_out[i] = terminate;
            }
        None => 
            for i in 0..N {
                let (action, logprob) = actor.act(&self.env);
                logprob_out[i] = logprob;
                action.encode(&mut action_out[i]);
                let truncate = action.perform(&mut self.env, None);
                truncate_out[i] = truncate;
                if truncate {
                    self.tracker.calibrate(&self.env);
                    continue;
                }
                let terminate = self.tracker.step(&action,&mut self.env, None);
                self.tracker.write_state(&mut state_out[i]);
                self.tracker.write_reward(&mut reward_out[i]);
                terminate_out[i] = terminate;
            }
        };
    }
}
