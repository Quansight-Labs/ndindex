#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <tuple>
#include <stdexcept>
#include <limits>

namespace py = pybind11;

class SimpleSlicePybind11 {
private:
    py::ssize_t _start;
    py::ssize_t _stop;
    py::ssize_t _step;
    bool _has_start;
    bool _has_stop;
    bool _has_step;

    static py::ssize_t py_index(const py::handle& obj) {
        if (py::isinstance<py::bool_>(obj)) {
            throw py::type_error("'bool' object cannot be interpreted as an integer");
        }
        if (py::hasattr(obj, "__class__") && 
            py::str(obj.attr("__class__").attr("__name__")).cast<std::string>() == "bool_") {
            throw py::type_error("'numpy.bool_' object cannot be interpreted as an integer");
        }
        PyObject* index = PyNumber_Index(obj.ptr());
        if (!index) {
            throw py::error_already_set();
        }
        py::ssize_t result = PyLong_AsSsize_t(index);
        Py_DECREF(index);
        if (result == -1 && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        return result;
    }

public:
    SimpleSlicePybind11(py::object start, py::object stop = py::none(), py::object step = py::none()) {
        if (py::isinstance<SimpleSlicePybind11>(start)) {
            auto& other = start.cast<SimpleSlicePybind11&>();
            _start = other._start;
            _stop = other._stop;
            _step = other._step;
            _has_start = other._has_start;
            _has_stop = other._has_stop;
            _has_step = other._has_step;
            return;
        }

        if (py::isinstance<py::slice>(start)) {
            py::slice s = start.cast<py::slice>();
            *this = SimpleSlicePybind11(s.attr("start"), s.attr("stop"), s.attr("step"));
            return;
        }

        if (stop.is_none() && !start.is_none()) {
            std::swap(start, stop);
        }

        _has_start = !start.is_none();
        _has_stop = !stop.is_none();
        _has_step = !step.is_none();

        _start = _has_start ? py_index(start) : 0;
        _stop = _has_stop ? py_index(stop) : 0;
        _step = _has_step ? py_index(step) : 1;

        if (_has_step && _step == 0) {
            throw py::value_error("slice step cannot be zero");
        }
    }

    py::object get_start() const { 
        return _has_start ? py::cast(_start) : py::none();
    }
    py::object get_stop() const { 
        return _has_stop ? py::cast(_stop) : py::none();
    }
    py::object get_step() const { 
        return _has_step ? py::cast(_step) : py::none();
    }
    py::tuple get_args() const { 
        return py::make_tuple(get_start(), get_stop(), get_step());
    }

    py::object raw() const {
        return py::reinterpret_steal<py::object>(PySlice_New(
            get_start().ptr(),
            get_stop().ptr(),
            get_step().ptr()
        ));
    }

    bool operator==(const SimpleSlicePybind11& other) const {
        return _start == other._start && _stop == other._stop && _step == other._step &&
               _has_start == other._has_start && _has_stop == other._has_stop && _has_step == other._has_step;
    }
};

PYBIND11_MODULE(simple_slice_pybind11, m) {
    py::class_<SimpleSlicePybind11>(m, "SimpleSlicePybind11")
        .def(py::init<py::object, py::object, py::object>(),
             py::arg("start"), py::arg("stop") = py::none(), py::arg("step") = py::none())
        .def_property_readonly("start", &SimpleSlicePybind11::get_start)
        .def_property_readonly("stop", &SimpleSlicePybind11::get_stop)
        .def_property_readonly("step", &SimpleSlicePybind11::get_step)
        .def_property_readonly("args", &SimpleSlicePybind11::get_args)
        .def_property_readonly("raw", &SimpleSlicePybind11::raw)
        .def(py::self == py::self)
        .def("__repr__", [](const SimpleSlicePybind11& self) {
            return "SimpleSlicePybind11" + py::str(self.get_args()).cast<std::string>();
        });
}