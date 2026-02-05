use thiserror::Error;
use iceoryx2::prelude::ZeroCopySend;
use crate::{
    prelude::{
        F,
        PayloadScheme,
        WeightUpdateScheme,
    }
};

#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Invalid name {0}: {1}")]
    InvalidName(String, Box<dyn std::error::Error>),
    #[error("Node creation failure: {0}")]
    NodeCreationFailure(#[from] iceoryx2::node::NodeCreationFailure),
    #[error("Service creation failure: {0:?}")]
    ServiceCreationFailure(#[from] iceoryx2::service::builder::publish_subscribe::PublishSubscribeOpenOrCreateError),
    #[error("Subscriber creation failure: {0}")]
    SubscriberCreationFailure(#[from] iceoryx2::port::subscriber::SubscriberCreateError),
    #[error("Publisher creation failure: {0}")]
    PublisherCreationFailure(#[from] iceoryx2::port::publisher::PublisherCreateError),
    #[error("Receive error: {0}")]
    ReceiveError(#[from] iceoryx2::port::ReceiveError),
    #[error("Send error: {0}")]
    SendError(#[from] iceoryx2::port::SendError),
    #[error("Wait failure: {0}")]
    WaitFailure(#[from] iceoryx2::node::NodeWaitFailure),
    #[error("Timeout")]
    Timeout,
    #[error("Loan error: {0}")]
    LoanError(#[from] iceoryx2::port::LoanError),
}

#[repr(C)]
#[derive(ZeroCopySend)]
pub struct SamplingPayload<S:PayloadScheme>
where
    [();S::SDIM]: Sized,
    [();S::ADIM]: Sized,
    [();S::RDIM]: Sized,
    [();S::N_INFO]: Sized,
    [();S::N]: Sized,
{
    pub version: u64,
    pub state: [[F; S::SDIM]; S::N],
    pub action: [[F; S::ADIM]; S::N],
    pub logprob: [F; S::N],
    pub reward: [F; S::N],
    pub truncated: [bool; S::N],
    pub terminate: [bool; S::N],
    pub first_state: [F; S::SDIM],
    pub info: [[F; S::N]; S::N_INFO]
}

impl<S:PayloadScheme> std::fmt::Debug for SamplingPayload<S> 
where
    [();S::SDIM]: Sized,
    [();S::ADIM]: Sized,
    [();S::RDIM]: Sized,
    [();S::N_INFO]: Sized,
    [();S::N]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SamplingPayload {{ version: {}, state: {:?}, action: {:?}, logprob: {:?}, reward: {:?}, truncated: {:?}, terminate: {:?}, first_state: {:?}, info: {:?} }}", self.version, self.state, self.action, self.logprob, self.reward, self.truncated, self.terminate, self.first_state, self.info)
    }
}

impl<S:PayloadScheme> Default for SamplingPayload<S>
where
    [();S::SDIM]: Sized,
    [();S::ADIM]: Sized,
    [();S::RDIM]: Sized,
    [();S::N_INFO]: Sized,
    [();S::N]: Sized,
{
    fn default() -> Self {
        Self {
            version: 0,
            state: [[f64::NAN; S::SDIM]; S::N],
            action: [[f64::NAN; S::ADIM]; S::N],
            logprob: [f64::NAN; S::N],
            reward: [f64::NAN; S::N],
            truncated: [false; S::N],
            terminate: [false; S::N],
            first_state: [f64::NAN; S::SDIM],
            info: [[f64::NAN; S::N]; S::N_INFO]
        }
    }

}


#[repr(C)]
#[derive(ZeroCopySend)]
pub struct WeightUpdatePayload<S:WeightUpdateScheme>
where
    [();S::MAX_PARAMS]: Sized,
{
    pub version: u64,
    pub params: [F;S::MAX_PARAMS]
}
impl<S:WeightUpdateScheme> WeightUpdatePayload<S> 
where
    [();S::MAX_PARAMS]: Sized,
{
    pub fn from_vec(version: u64, vec: Vec<F>) -> Self
    where
        [();S::MAX_PARAMS]: Sized,
    {
        let mut params = [0.0; S::MAX_PARAMS];
        params[..vec.len()].copy_from_slice(&vec);
        Self {
            version: version,
            params,
        }
    }
}

impl<S:WeightUpdateScheme> std::fmt::Debug for WeightUpdatePayload<S> 
where
    [();S::MAX_PARAMS]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WeightUpdatePayload {{ version: {}, params: {:?} }}", self.version, self.params)
    }
}

impl<S:WeightUpdateScheme> Default for WeightUpdatePayload<S>
where
    [();S::MAX_PARAMS]: Sized,
{
    fn default() -> Self {
        Self { version: 0, params: [0.0; S::MAX_PARAMS] }
    }
}