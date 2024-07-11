use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple, PyBool};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::PyResult;

#[pyclass(name = "SimpleSliceRust")]
struct SimpleSliceRust {
    _start: isize,
    _stop: isize,
    _step: isize,
    _has_start: bool,
    _has_stop: bool,
    _has_step: bool,
}

#[pymethods]
impl SimpleSliceRust {
    #[new]
    #[pyo3(signature = (start, stop = None, step = None))]
    fn new(py: Python<'_>, start: Option<&PyAny>, stop: Option<&PyAny>, step: Option<&PyAny>) -> PyResult<Self> {
        if let Some(start) = start {
            if let Ok(slice) = start.extract::<PyRef<SimpleSliceRust>>() {
                return Ok(Self {
                    _start: slice._start,
                    _stop: slice._stop,
                    _step: slice._step,
                    _has_start: slice._has_start,
                    _has_stop: slice._has_stop,
                    _has_step: slice._has_step,
                });
            }

            if let Ok(slice) = start.downcast::<PySlice>() {
                return Self::new(py,
                    Some(slice.getattr("start")?),
                    Some(slice.getattr("stop")?),
                    Some(slice.getattr("step")?),
                );
            }
        }

        let (mut start, mut stop) = (start, stop);

        if stop.is_none() && start.is_some() {
            std::mem::swap(&mut start, &mut stop);
        }

        let py_index = |obj: &PyAny| -> PyResult<isize> {
            if obj.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err("'bool' object cannot be interpreted as an integer"));
            }
            if obj.hasattr("__class__")? && obj.getattr("__class__")?.getattr("__name__")?.extract::<String>()? == "bool_" {
                return Err(PyTypeError::new_err("'numpy.bool_' object cannot be interpreted as an integer"));
            }
            match obj.call_method0("__index__") {
                Ok(index_obj) => index_obj.extract::<isize>(),
                Err(_) => Err(PyTypeError::new_err(format!("'{}' object cannot be interpreted as an integer", obj.get_type().name()?)))
            }
        };

        let _has_start = start.is_some();
        let _has_stop = stop.is_some();
        let _has_step = step.is_some();

        let _start = start.map(py_index).transpose()?.unwrap_or(0);
        let _stop = stop.map(py_index).transpose()?.unwrap_or(0);
        let _step = if let Some(step) = step {
            let step_value = py_index(step)?;
            if step_value == 0 {
                return Err(PyValueError::new_err("slice step cannot be zero"));
            }
            step_value
        } else { 1 };

        Ok(Self {
            _start,
            _stop,
            _step,
            _has_start,
            _has_stop,
            _has_step,
        })
    }

    #[getter]
    fn start(&self, py: Python<'_>) -> PyObject {
        if self._has_start {
            self._start.into_py(py)
        } else {
            py.None()
        }
    }

    #[getter]
    fn stop(&self, py: Python<'_>) -> PyObject {
        if self._has_stop {
            self._stop.into_py(py)
        } else {
            py.None()
        }
    }

    #[getter]
    fn step(&self, py: Python<'_>) -> PyObject {
        if self._has_step {
            self._step.into_py(py)
        } else {
            py.None()
        }
    }

    #[getter]
    fn args(&self, py: Python<'_>) -> PyObject {
        PyTuple::new(py, &[self.start(py), self.stop(py), self.step(py)]).into()
    }

    fn __eq__(&self, other: &PyAny) -> PyResult<bool> {
        if let Ok(other) = other.extract::<PyRef<SimpleSliceRust>>() {
            Ok(self._start == other._start && self._stop == other._stop && self._step == other._step &&
               self._has_start == other._has_start && self._has_stop == other._has_stop && self._has_step == other._has_step)
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
fn simple_slice_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SimpleSliceRust>()?;
    Ok(())
}
