//! Python bindings for HFTRL
//!
//! This module provides Python API for:
//! - SamplingClient: Retrieve sampling data from running samplers
//! - Coach: Send weight updates and reload signals to neural actors

use pyo3::{
    prelude::*,
    exceptions::PyRuntimeError,
};
use numpy::{PyArray2, PyArray1, PyReadonlyArray1};

use std::sync::LazyLock;

use crate::{
    prelude::{
        F,
    },
    server::{
        Receiver,
        Sender,
    },
};

mod pre {
    pub use crate::server::{
        Server as ServerRs,
    };
    pub use crate::types::{
        WeightUpdatePayload, 
        SamplingPayload
    };
}

#[pyfunction]
fn init_tracing() {
    crate::init_tracing();
}

/// Sampling data structure containing state, action, reward, and termination flags
#[pyclass(unsendable)]
pub struct Sampling {
    #[pyo3(get)]
    pub version: u64,
    #[pyo3(get)]
    pub state: Py<PyArray2<F>>,
    #[pyo3(get)]
    pub action: Py<PyArray2<F>>,
    #[pyo3(get)]
    pub logprob: Py<PyArray1<F>>,
    #[pyo3(get)]
    pub reward: Py<PyArray1<F>>,
    #[pyo3(get)]
    pub terminate: Py<PyArray1<bool>>,
    #[pyo3(get)]
    pub truncated: Py<PyArray1<bool>>,
    #[pyo3(get)]
    pub first_state: Py<PyArray1<F>>,
    
    #[pyo3(get)]
    pub info: Py<PyArray2<F>>,
}

#[macro_export]
macro_rules! make_module { ($S:ty) => {

type S = $S;
type ServerRs = pre::ServerRs<S>;
type SamplingPayload = pre::SamplingPayload<S>;
type WeightUpdatePayload = pre::WeightUpdatePayload<S>;
const MAX_PARAMS: usize = S::MAX_PARAMS;

#[pyclass]
struct Server {
    server: ServerRs,
}

#[pymethods]
impl Server {
    #[new]
    fn new() -> Self {
        Self { server: ServerRs::new().unwrap() }
    }
    fn sampling_receiver(&self, name: String) -> PyResult<SamplingClient> {
        SamplingClient::new(name, self)
    }
}
/// Client for retrieving sampling data from a running sampler
#[pyclass(unsendable)]
pub struct SamplingClient {
    sampling_receiver: Receiver<SamplingPayload>,
}

#[pymethods]
impl SamplingClient {
    /// Create a new sampling client
    ///
    /// Args:
    ///     name: Name of the sampler (must match the --name argument when starting the sampler)
    ///
    /// Returns:
    ///     SamplingClient instance
    #[new]
    fn new(name: String, server: &Server) -> PyResult<Self> {
        let sampling_receiver = server.server.sampling_receiver(name.clone())
            .map_err(|e| PyRuntimeError::new_err(
                format!("Failed to create sampling receiver: {}", e)))?;

        Ok(Self { 
            sampling_receiver 
        })
    }

    /// Retrieve a sampling batch
    ///
    /// Returns:
    ///     Sampling object if available, None if no samples are available
    ///
    /// Raises:
    ///     RuntimeError: If there's an error receiving samples
    fn retrieve(&self, py: Python) -> PyResult<Option<Sampling>> {
        self.sampling_receiver.receive(
            |sample: &SamplingPayload| -> PyResult<Sampling> {
                // Convert state: [[F; SDIM]; N] -> PyArray2
                // Create 2D vector structure for from_vec2
                let state_2d: Vec<Vec<F>> = sample.state.iter()
                    .map(|row| row.to_vec())
                    .collect();
                let state_array = PyArray2::from_vec2(py, &state_2d)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create state array: {}", e)))?;
                
                // Convert action: [[F; ADIM]; N] -> PyArray2
                let action_2d: Vec<Vec<F>> = sample.action.iter()
                    .map(|row| row.to_vec())
                    .collect();
                let action_array = PyArray2::from_vec2(py, &action_2d)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create action array: {}", e)))?;
                
                // Convert logprob: [F; N] -> PyArray1
                let logprob_array = PyArray1::from_vec(py, sample.logprob.to_vec());
                
                // Convert reward: [F; N] -> PyArray1
                let reward_array = PyArray1::from_vec(py, sample.reward.to_vec());
                
                // Convert truncated: [bool; N] -> PyArray1
                let truncated_array = PyArray1::from_vec(py, sample.truncated.to_vec());

                // Convert terminate: [bool; N] -> PyArray1
                let terminate_array = PyArray1::from_vec(py, sample.terminate.to_vec());
                
                // Convert last_state: [F; SDIM] -> PyArray1
                let first_state_array = PyArray1::from_vec(py, sample.first_state.to_vec());

                let info_2d: Vec<Vec<F>> = sample.info.iter()
                    .map(|row| row.to_vec())
                    .collect();
                let info_array = PyArray2::from_vec2(py, &info_2d)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create info array: {}", e)))?;
                
                Ok(Sampling {
                    version: sample.version,
                    state: state_array.unbind().into(),
                    action: action_array.unbind().into(),
                    logprob: logprob_array.unbind().into(),
                    reward: reward_array.unbind().into(),
                    truncated: truncated_array.unbind().into(),
                    terminate: terminate_array.unbind().into(),
                    first_state: first_state_array.unbind().into(),
                    info: info_array.unbind().into(),
                })
                
            }
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to receive sampling: {}", e)))?
        .map(|result| result.map_err(|e| PyRuntimeError::new_err(format!("Failed to convert sampling: {}", e))))
        .transpose()
    }
}

/// Client for sending weight updates and reload signals to neural actors
#[pyclass(unsendable)]
pub struct Coach {
    #[pyo3(get)]
    version: u64,
    weight_update_sender: Sender<WeightUpdatePayload>,
}

#[pymethods]
impl Coach {
    /// Create a new coach client
    ///
    /// Args:
    ///     name: Name of the sampler (must match the --name argument when starting the sampler)
    ///
    /// Returns:
    ///     Coach instance
    #[new]
    fn new(name: String, server: &Server) -> PyResult<Self> {
        let weight_update_sender = server.server.weight_update_sender(name.clone())
            .map_err(|e| PyRuntimeError::new_err(
                format!("Failed to create weight update sender: {}", e)))?;
        Ok(Self {
            version: 0,
            weight_update_sender,
        })
    }

    /// Send weight update to the sampler
    ///
    /// Args:
    ///     weights: List of weight values to update
    ///
    /// Raises:
    ///     RuntimeError: If there's an error sending weight update
    fn update_weights(&mut self, weights: PyReadonlyArray1<F>) -> PyResult<()> {
        let len = weights.len()?;
        if len > MAX_PARAMS {
            return Err(PyRuntimeError::new_err(
                format!("Weights length {} exceeds max params length {}", len, MAX_PARAMS)));
        }
        self.version += 1;
        let loan = self.weight_update_sender.loan()
            .map_err(|e| PyRuntimeError::new_err(
                format!("Failed to loan weight update buffer: {}", e)))?;
        
        let mut loan = unsafe { loan.assume_init() };
        let payload = loan.payload_mut();
        
        payload.version = self.version;
        payload.params[..len].copy_from_slice(&weights.as_slice()?);
        
        loan.send()
            .map_err(|e| PyRuntimeError::new_err(
                format!("Failed to send weight update: {}", e)))?;
        
        Ok(())
    }

    /// Get the current version number
    fn get_version(&self) -> u64 {
        self.version
    }
}

/// Python module initialization
#[pymodule]
fn _hftrl(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<> {
    m.add_function(wrap_pyfunction!(init_tracing, m)?)?;
    m.add_class::<Sampling>()?;
    m.add_class::<SamplingClient>()?;
    m.add_class::<Coach>()?;
    m.add("STATE_DIM", SDIM)?;
    m.add("ACTION_DIM", ADIM)?;
    m.add("INFO_KEYS", info_keys())?;
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}

}}