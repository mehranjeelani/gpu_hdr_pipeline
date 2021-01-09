cmake_minimum_required(VERSION 3.18)

if (CUDAToolkit_FOUND)

project(CUDA_utils CXX)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(CUDA_utils STATIC
	"${SOURCE_DIR}/error.h"
	"${SOURCE_DIR}/error.cpp"
	"${SOURCE_DIR}/device.h"
	"${SOURCE_DIR}/device.cpp"
	"${SOURCE_DIR}/event.h"
	"${SOURCE_DIR}/memory.h"
	"${SOURCE_DIR}/array.h"
	"${SOURCE_DIR}/graphics_interop.h"
	"${SOURCE_DIR}/graphics_gl_interop.h"
)

target_include_directories(CUDA_utils PUBLIC "${SOURCE_DIR}/../../")

configure_project(CUDA_utils)

target_link_libraries(CUDA_utils PUBLIC CUDA::cudart)


source_group("source/CUDA" REGULAR_EXPRESSION "${SOURCE_DIR}/.*")

set_target_properties(CUDA_utils PROPERTIES FOLDER "dependencies")

endif ()
