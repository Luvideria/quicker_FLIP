#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FLIP_opti_10.cpp"
#include <vector>

std::vector<float> psquare(const std::vector<float>& in){
    std::vector<float> s;
    s.resize(in.size());
    #pragma omp parallel for
    for(int i = 0 ; i < in.size(); i++){
        s[i] = in[i]*in[i];
    }
    printf("sizeof float: %d",int(sizeof(float)));
    return s;
}

std::vector<float> square(const std::vector<float>& in){
    std::vector<float> s;
    s.resize(in.size());
    for(int i = 0 ; i < in.size(); i++){
        s[i] = in[i]*in[i];
    }
    return s;
}

namespace py = pybind11;
PYBIND11_MODULE(pyflip, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

	m.def("computeFlip", &computeFLIP, "computeFLIP");
    m.def("psquare", &psquare, "psquare");
    m.def("square", &square, "square");
}