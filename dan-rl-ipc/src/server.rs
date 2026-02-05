use iceoryx2::{
    node::NodeBuilder,
    node::Node,
    prelude::ipc,
    port::subscriber::Subscriber,
    port::publisher::Publisher,
    prelude::ZeroCopySend,
    sample_mut_uninit::SampleMutUninit,
};
use std::mem::MaybeUninit;
use std::marker::PhantomData;
use crate::types::{ServerError, WeightUpdatePayload, SamplingPayload};
use crate::prelude::{PayloadScheme, WeightUpdateScheme};

pub struct Server {
    node: Node<ipc::Service>,
}
impl Server {
    pub fn new() -> Result<Self, ServerError> {
        let node = NodeBuilder::new().create::<ipc::Service>()?;
        Ok(Self { node })
    }

    pub fn sampling_sender<S:PayloadScheme>(&self,name: String) -> Result<Sender<SamplingPayload<S>>, ServerError> 
    where
        [();S::SDIM]: Sized,
        [();S::ADIM]: Sized,
        [();S::RDIM]: Sized,
        [();S::N_INFO]: Sized,
        [();S::N]: Sized,
    {
        Sender::new(&self.node, format!("sampling/{}", name), 64)
    }

    pub fn sampling_receiver<S:PayloadScheme>(&self,name: String) -> Result<Receiver<SamplingPayload<S>>, ServerError> 
    where
        [();S::SDIM]: Sized,
        [();S::ADIM]: Sized,
        [();S::RDIM]: Sized,
        [();S::N_INFO]: Sized,
        [();S::N]: Sized,
    {
        Receiver::new(&self.node, format!("sampling/{}", name), 64)
    }

    pub fn weight_update_receiver<S:WeightUpdateScheme>(&self,name: String) -> Result<Receiver<WeightUpdatePayload<S>>, ServerError> 
    where
        [();S::MAX_PARAMS]: Sized,
    {
        Receiver::new(&self.node, format!("weight_update/{}", name), 2)
    }

    pub fn weight_update_sender<S:WeightUpdateScheme>(&self,name: String) -> Result<Sender<WeightUpdatePayload<S>>, ServerError> 
    where
        [();S::MAX_PARAMS]: Sized,
    {
        Sender::new(&self.node, format!("weight_update/{}", name), 2)
    }
}

pub struct Receiver<T: std::fmt::Debug + ZeroCopySend + 'static> {
    subscriber: Subscriber<ipc::Service, T, ()>,
}
impl<T: std::fmt::Debug + ZeroCopySend + 'static> Receiver<T> {
    pub fn new(node: &Node<ipc::Service>,branch:String, buffer_size: usize) -> Result<Self, ServerError> {
        let service_name = branch
        .as_str().try_into().map_err(|e|ServerError::InvalidName(branch, Box::new(e)))?;

        let service = node
        .service_builder(&service_name)
        .publish_subscribe::<T>()
        .max_publishers(256)
        .max_nodes(256)
        .max_subscribers(256)
        .enable_safe_overflow(true)
        .subscriber_max_buffer_size(buffer_size)
        .open_or_create()?;
        let subscriber = service.subscriber_builder().create()?;
        Ok(Self { subscriber })
    }
    pub fn receive<F: FnOnce(&T) -> R, R>(&self,f:F) -> Result<Option<R>, ServerError> {
        Ok(self.subscriber.receive()?.map(|sample| f(sample.payload())))
    }
    pub fn drain<F: FnOnce(&T) -> R, R>(&self,f:F) -> Result<Option<R>, ServerError> {
        if let Some(mut last) = self.subscriber.receive()?{
            while let Some(sample) = self.subscriber.receive()? {
                last = sample;
            }
            return Ok(Some(f(&last.payload())));
        }
        Ok(None)
    }
}


pub struct Sender<T: std::fmt::Debug + ZeroCopySend + 'static> {
    publisher: Publisher<ipc::Service, T, ()>,
}
impl<T: std::fmt::Debug + ZeroCopySend + 'static> Sender<T> {
    pub fn new(node: &Node<ipc::Service>,branch:String, buffer_size: usize) -> Result<Self, ServerError> {
        let service_name = branch
        .as_str().try_into().map_err(|e|ServerError::InvalidName(branch, Box::new(e)))?;

        let service = node
        .service_builder(&service_name)
        .publish_subscribe::<T>()
        .max_publishers(256)
        .max_nodes(256)
        .max_subscribers(256)
        .enable_safe_overflow(true)
        .subscriber_max_buffer_size(buffer_size)
        .open_or_create()?;
        let publisher = service.publisher_builder().create()?;
        Ok(Self { publisher })
    }
    pub fn loan(&self) -> Result<SampleMutUninit<ipc::Service, MaybeUninit<T>, ()>, ServerError> {
        Ok(self.publisher.loan_uninit()?)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SamplingPayload, WeightUpdatePayload};
    use crate::server::{Receiver, Sender};
    use std::thread;
    use std::sync::{Arc, atomic::AtomicBool};
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    impl PayloadScheme for () {
        const SDIM: usize = 5;
        const ADIM: usize = 2;
        const RDIM: usize = 2;
        const N_INFO: usize = 5;
        const N: usize = 512;
        const INFO_KEYS: [&str; Self::N_INFO] = ["cash", "quantity", "price", "info1", "info2"];
    }
    impl WeightUpdateScheme for () {
        const MAX_PARAMS: usize = 1024;
    }
    // type Payload = SamplingPayload<3, 2, 5>;

    #[test]
    fn test_sampling() {
        let name = "test_sampling";
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let send_handle = thread::spawn(move || {
            let server = Server::new().unwrap();
            let sender = server.sampling_sender::<()>(name.to_string()).unwrap();
            while !stop_clone.load(Ordering::Relaxed) {
                let buffer = sender.loan().unwrap();
                let buffer = buffer.write_payload(Default::default());
                buffer.send().unwrap();
            }
        });
        let stop_clone = stop.clone();
        let receive_handle = thread::spawn(move || {
            let server = Server::new().unwrap();
            let receiver = server.sampling_receiver::<()>(name.to_string()).unwrap();
            while !stop_clone.load(Ordering::Relaxed) {
                let _ = receiver.receive(
                    |sample| {
                    println!("Received sample: {:?}", sample);
                }).unwrap();
            }
        });
        thread::sleep(Duration::from_secs(5));
        stop.store(true, Ordering::Relaxed);
        send_handle.join().unwrap();
        receive_handle.join().unwrap();
    }
}