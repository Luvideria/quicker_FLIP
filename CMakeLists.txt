#libtuning https://code.ipd.kit.edu/pfaffe/libtuning/tree/master
cmake_minimum_required(VERSION 3.14)
#todo : include nlohmann json, see lightmetrica
project(pyFLIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)

FIND_PACKAGE( OpenMP REQUIRED)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()


set(_INTERFACE_DEFINED 1)
add_library(flip_interface INTERFACE)
add_library(flip::interface ALIAS flip_interface)
    
set(JSON_BuildTests OFF CACHE INTERNAL "") #prevent building tests
#add_subdirectory(json)
add_library(flip SHARED  "FLIP_opti_10.cpp" "commandline.h" "pooling.h" "stb_image_write.h" "stb_image.h")
target_link_libraries(flip
    PRIVATE
        #nlohmann_json::nlohmann_json
        $<${_INTERFACE_DEFINED}:flip_interface>)
target_compile_definitions(flip PRIVATE -DLM_FLIP_EXPORTS)
set_target_properties(flip PROPERTIES PREFIX "")
set_target_properties(flip PROPERTIES DEBUG_POSTFIX "-debug")
set_target_properties(flip PROPERTIES FOLDER "./")
set_target_properties(flip PROPERTIES
ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(pybind11 REQUIRED)
add_library(pyflip MODULE "py_flip.cpp")# "pybind11_json.hpp")
target_link_libraries(pyflip PRIVATE pybind11::module flip)
set_target_properties(pyflip PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
set_target_properties(pyflip PROPERTIES FOLDER "./")
set_target_properties(pyflip PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")    
