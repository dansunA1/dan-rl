use dan_rl_core::{
    sampler::{
        Sampling,
    },
    traits::{
        Info,
        Env,
        Action,
    },
    types::{
        F,
    },
};

use pyo3::{
    prelude::*,
    exceptions::PyRuntimeError,
};
use numpy::{PyArray2, PyArray1, PyReadonlyArray1, IntoPyArray, PyArrayMethods};

#[pyclass(unsendable)]
pub struct PySampling {
    #[pyo3(get)]
    state: Py<PyArray2<F>>,
    #[pyo3(get)]
    action: Py<PyArray2<F>>,
    #[pyo3(get)]
    logprob: Py<PyArray1<F>>,
    #[pyo3(get)]
    reward: Py<PyArray2<F>>,
    #[pyo3(get)]
    truncate: Py<PyArray1<bool>>,
    #[pyo3(get)]
    terminate: Py<PyArray1<bool>>,
    #[pyo3(get)]
    info: Py<PyArray2<F>>,
}

fn to_pyarray2<T: numpy::Element,const N: usize,const M: usize>(py: Python, data: [[T;N];M]) -> Py<PyArray2<T>> {
    let arr = std::mem::ManuallyDrop::new(data);

    let ptr = arr.as_ptr() as *mut T;
    let len = M * N;

    let data = unsafe {
        Vec::from_raw_parts(ptr, len, len)
    };

    data.into_pyarray(py).reshape([M,N]).unwrap().unbind()
}

fn to_pyarray1<T: numpy::Element,const N: usize>(py: Python, data: [T;N]) -> Py<PyArray1<T>> {
    let arr = std::mem::ManuallyDrop::new(data);

    let ptr = arr.as_ptr() as *mut T;
    let len = N;

    let data = unsafe {
        Vec::from_raw_parts(ptr, len, len)
    };

    data.into_pyarray(py).unbind()
}

// fn to_pyarray2<T: numpy::Element + Copy, const N: usize, const M: usize>(py: Python, data: [[T;N];M]) -> Py<PyArray2<T>> {
//     // Convert to Vec<Vec<T>> for safe conversion
//     let data_vec: Vec<Vec<T>> = data.iter().map(|row| row.to_vec()).collect();
//     PyArray2::from_vec2(py, &data_vec).unwrap().unbind()
// }

// fn to_pyarray1<T: numpy::Element + Copy, const N: usize>(py: Python, data: [T;N]) -> Py<PyArray1<T>> {
//     // Convert to Vec for safe conversion
//     PyArray1::from_vec(py, data.to_vec()).unbind()
// }

pub trait IntoPySampling {
    fn into_pysampling(self: Box<Self>, py: Python) -> PySampling;
}

impl<E:Env,A:Action<E>,const N: usize> IntoPySampling for Sampling<E,A,N>
where
    [();A::DIM]: Sized,
    [();E::SDIM]: Sized,
    [();E::RDIM]: Sized,
    [();<E::Info as Info>::N]: Sized,
    [();N]: Sized,
{
    fn into_pysampling(self: Box<Self>, py: Python) -> PySampling {
        PySampling {
            state: to_pyarray2(py, self.state),
            action: to_pyarray2(py, self.action),
            logprob: to_pyarray1(py, self.logprob),
            reward: to_pyarray2(py, self.reward),
            truncate: to_pyarray1(py, self.truncate),
            terminate: to_pyarray1(py, self.terminate),
            info: to_pyarray2(py, self.info),
        }
    }
}
