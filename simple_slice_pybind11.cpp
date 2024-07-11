#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <limits>

namespace py = pybind11;

class SimpleSlicePybind11 {
private:
    py::ssize_t _start;
    py::ssize_t _stop;
    py::ssize_t _step;
    uint8_t _flags; // Use bit flags instead of separate booleans

    static constexpr uint8_t HAS_START = 1;
    static constexpr uint8_t HAS_STOP = 2;
    static constexpr uint8_t HAS_STEP = 4;

    static inline py::ssize_t py_index(const py::handle& obj) {
        if (PyLong_Check(obj.ptr())) {
            return PyLong_AsSsize_t(obj.ptr());
        }
        if (PyBool_Check(obj.ptr()) || (PyObject_HasAttrString(obj.ptr(), "__class__") &&
            strcmp(PyUnicode_AsUTF8(PyObject_GetAttrString(PyObject_GetAttrString(obj.ptr(), "__class__"), "__name__")), "bool_") == 0)) {
            throw py::type_error("boolean value cannot be interpreted as an integer");
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
    SimpleSlicePybind11(py::handle start, py::handle stop = py::none(), py::handle step = py::none()) : _flags(0) {
        if (py::isinstance<SimpleSlicePybind11>(start)) {
            auto& other = start.cast<SimpleSlicePybind11&>();
            _start = other._start;
            _stop = other._stop;
            _step = other._step;
            _flags = other._flags;
            return;
        }

        if (PySlice_Check(start.ptr())) {
            Py_ssize_t slice_start, slice_stop, slice_step;
            PySlice_Unpack(start.ptr(), &slice_start, &slice_stop, &slice_step);
            _start = slice_start;
            _stop = slice_stop;
            _step = slice_step;
            _flags = HAS_START | HAS_STOP | HAS_STEP;
            return;
        }

        if (stop.is_none() && !start.is_none()) {
            std::swap(start, stop);
        }

        if (!start.is_none()) {
            _start = py_index(start);
            _flags |= HAS_START;
        }

        if (!stop.is_none()) {
            _stop = py_index(stop);
            _flags |= HAS_STOP;
        }

        if (!step.is_none()) {
            _step = py_index(step);
            if (_step == 0) {
                throw py::value_error("slice step cannot be zero");
            }
            _flags |= HAS_STEP;
        } else {
            _step = 1;
        }
    }

    py::object get_start() const { 
        return _flags & HAS_START ? py::cast(_start) : py::none();
    }
    py::object get_stop() const { 
        return _flags & HAS_STOP ? py::cast(_stop) : py::none();
    }
    py::object get_step() const { 
        return _flags & HAS_STEP ? py::cast(_step) : py::none();
    }
    py::tuple get_args() const { 
        return py::make_tuple(get_start(), get_stop(), get_step());
    }

    py::object raw() const {
        return py::reinterpret_steal<py::object>(PySlice_New(
            (_flags & HAS_START) ? PyLong_FromSsize_t(_start) : Py_None,
            (_flags & HAS_STOP) ? PyLong_FromSsize_t(_stop) : Py_None,
            (_flags & HAS_STEP) ? PyLong_FromSsize_t(_step) : Py_None
        ));
    }

    bool operator==(const SimpleSlicePybind11& other) const {
        return _start == other._start && _stop == other._stop && _step == other._step && _flags == other._flags;
    }
};

PYBIND11_MODULE(simple_slice_pybind11, m) {
    py::class_<SimpleSlicePybind11>(m, "SimpleSlicePybind11")
        .def(py::init<py::handle, py::handle, py::handle>(),
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