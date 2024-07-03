use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple, PyBool};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::PyResult;

#[pyclass(name = "SimpleSliceRust")]
struct SimpleSliceRust {
    args: (Option<PyObject>, Option<PyObject>, Option<PyObject>),
}

#[pymethods]
impl SimpleSliceRust {
    #[new]
    #[pyo3(signature = (start, stop = None, step = None))]
    fn new<'py>(py: Python<'py>, start: Option<&'py PyAny>, stop: Option<&'py PyAny>, step: Option<&'py PyAny>) -> PyResult<Self> {
        if let Some(start) = start {
            if let Ok(slice) = start.extract::<PyRef<SimpleSliceRust>>() {
                return Ok(Self {
                    args: (
                        slice.args.0.clone(),
                        slice.args.1.clone(),
                        slice.args.2.clone(),
                    ),
                });
            }

            if let Ok(slice) = start.downcast::<PySlice>() {
                return Ok(Self {
                    args: (
                        Some(slice.getattr("start")?.into_py(py)),
                        Some(slice.getattr("stop")?.into_py(py)),
                        Some(slice.getattr("step")?.into_py(py)),
                    ),
                });
            }
        }

        let (mut start, mut stop) = (start, stop);

        if stop.is_none() && start.is_some() {
            std::mem::swap(&mut start, &mut stop);
        }

        let py_index = |obj: &PyAny| -> PyResult<i64> {
            if obj.is_none() {
                return Err(PyTypeError::new_err("Cannot convert None to integer index"));
            }
            if obj.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err("'bool' object cannot be interpreted as an integer"));
            }
            if obj.hasattr("__class__")? && obj.getattr("__class__")?.getattr("__name__")?.extract::<String>()? == "bool_" {
                return Err(PyTypeError::new_err("'numpy.bool_' object cannot be interpreted as an integer"));
            }
            match obj.call_method0("__index__") {
                Ok(index_obj) => index_obj.extract::<i64>(),
                Err(_) => Err(PyTypeError::new_err(format!("'{}' object cannot be interpreted as an integer", obj.get_type().name()?)))
            }
        };

        let check_and_convert = |obj: Option<&PyAny>| -> PyResult<Option<PyObject>> {
            match obj {
                Some(o) => {
                    py_index(o)?;
                    Ok(Some(o.into_py(py)))
                },
                None => Ok(None),
            }
        };

        let start_obj = check_and_convert(start)?;
        let stop_obj = check_and_convert(stop)?;
        let step_obj = if let Some(step) = step {
            let step_value = py_index(step)?;
            if step_value == 0 {
                return Err(PyValueError::new_err("slice step cannot be zero"));
            }
            Some(step.into_py(py))
        } else {
            None
        };

        Ok(Self {
            args: (start_obj, stop_obj, step_obj),
        })
    }

    #[getter]
    fn start<'py>(&self, py: Python<'py>) -> PyObject {
        self.args.0.as_ref().map_or_else(|| py.None(), |obj| obj.clone_ref(py))
    }

    #[getter]
    fn stop<'py>(&self, py: Python<'py>) -> PyObject {
        self.args.1.as_ref().map_or_else(|| py.None(), |obj| obj.clone_ref(py))
    }

    #[getter]
    fn step<'py>(&self, py: Python<'py>) -> PyObject {
        self.args.2.as_ref().map_or_else(|| py.None(), |obj| obj.clone_ref(py))
    }

    #[getter]
    fn args<'py>(&self, py: Python<'py>) -> PyObject {
        PyTuple::new(py, &[self.start(py), self.stop(py), self.step(py)]).into()
    }

    #[getter]
    fn raw<'py>(&self, py: Python<'py>) -> PyResult<Py<PySlice>> {
        let start = self.start(py).extract::<Option<isize>>(py)?;
        let stop = self.stop(py).extract::<Option<isize>>(py)?;
        let step = self.step(py).extract::<Option<isize>>(py)?;
        Ok(PySlice::new(py, start.unwrap_or(0), stop.unwrap_or(-1), step.unwrap_or(1)).into())
    }

    fn __eq__(&self, other: &PyAny, py: Python<'_>) -> PyResult<bool> {
        if let Ok(other) = other.extract::<PyRef<SimpleSliceRust>>() {
            let self_tuple = PyTuple::new(py, &[self.start(py), self.stop(py), self.step(py)]);
            let other_tuple = PyTuple::new(py, &[other.start(py), other.stop(py), other.step(py)]);
            self_tuple.eq(other_tuple)
        } else {
            Ok(false)
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let args_repr = PyTuple::new(py, &[self.start(py), self.stop(py), self.step(py)]).repr()?.extract::<String>()?;
        Ok(format!("SimpleSliceRust{}", args_repr))
    }
}


#[pymodule]
fn simple_slice_rust(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SimpleSliceRust>()?;

    let simple_slice_rust_type = m.getattr("SimpleSliceRust")?;
    simple_slice_rust_type.setattr("__module__", "simple_slice_rust")?;

    Ok(())
}
