#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FLIP_opti_10.cpp"
#include <vector>


namespace py = pybind11;
PYBIND11_MODULE(pyflip, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

	m.def("computeFlip", &computeFLIP, "computeFLIP");

}