#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <tuple>
#include <stdexcept>

namespace py = pybind11;

class SimpleSlicePybind11 {
private:
    py::tuple args;

    static int64_t py_index(const py::handle& obj) {
        if (obj.is_none()) {
            throw py::type_error("Cannot convert None to integer index");
        }
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
        int64_t result = PyLong_AsLongLong(index);
        Py_DECREF(index);
        if (result == -1 && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        return result;
    }

public:
    SimpleSlicePybind11(py::object start, py::object stop = py::none(), py::object step = py::none()) {
        if (py::isinstance<SimpleSlicePybind11>(start)) {
            args = py::cast<SimpleSlicePybind11&>(start).args;
            return;
        }

        if (py::isinstance<py::slice>(start)) {
            py::slice s = start.cast<py::slice>();
            args = py::make_tuple(s.attr("start"), s.attr("stop"), s.attr("step"));
            return;
        }

        if (stop.is_none() && !start.is_none()) {
            std::swap(start, stop);
        }

        // Type checking
        if (!start.is_none()) py_index(start);
        if (!stop.is_none()) py_index(stop);
        if (!step.is_none()) {
            int64_t step_value = py_index(step);
            if (step_value == 0) {
                throw py::value_error("slice step cannot be zero");
            }
        }

        args = py::make_tuple(start, stop, step);
    }

    py::object get_start() const { return args[0]; }
    py::object get_stop() const { return args[1]; }
    py::object get_step() const { return args[2]; }
    py::tuple get_args() const { return args; }

    py::slice raw() const {
        return py::slice(get_start(), get_stop(), get_step());
    }

    bool operator==(const SimpleSlicePybind11& other) const {
        return args.equal(other.args);
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