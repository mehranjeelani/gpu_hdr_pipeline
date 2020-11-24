cmake_minimum_required(VERSION 3.18)

project(utils CXX CUDA)


add_subdirectory("${DEPENDENCIES_DIR}/zlib" zlib)
add_subdirectory("${DEPENDENCIES_DIR}/libpng" libpng)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(utils STATIC
	"${SOURCE_DIR}/argparse.h"
	"${SOURCE_DIR}/argparse.cpp"
	"${SOURCE_DIR}/image_io.h"
	"${SOURCE_DIR}/radiance.h"
	"${SOURCE_DIR}/radiance.cpp"
	"${SOURCE_DIR}/pfm.h"
	"${SOURCE_DIR}/pfm.cpp"
	"${SOURCE_DIR}/png.h"
	"${SOURCE_DIR}/png.cpp"
	"${SOURCE_DIR}/Camera.h"
	"${SOURCE_DIR}/PerspectiveCamera.h"
	"${SOURCE_DIR}/PerspectiveCamera.cpp"
	"${SOURCE_DIR}/CUDA/error.h"
	"${SOURCE_DIR}/CUDA/error.cpp"
	"${SOURCE_DIR}/CUDA/event.h"
	"${SOURCE_DIR}/CUDA/memory.h"
	"${SOURCE_DIR}/CUDA/array.h"
	"${SOURCE_DIR}/CUDA/graphics_interop.h"
	"${SOURCE_DIR}/CUDA/graphics_gl_interop.h"
)

if (INTERACTIVE)
	target_sources(utils PRIVATE
		"${SOURCE_DIR}/Navigator.h"
		"${SOURCE_DIR}/OrbitalNavigator.h"
		"${SOURCE_DIR}/OrbitalNavigator.cpp"
	)
endif ()

target_include_directories(utils PUBLIC "${SOURCE_DIR}/../" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(utils PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
	POSITION_INDEPENDENT_CODE ON
)

if (MSVC)
	target_compile_definitions(utils PRIVATE -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
	target_compile_options(utils PRIVATE /WX /permissive-)

	if (CMAKE_GENERATOR MATCHES "Visual Studio")
		target_compile_options(utils PRIVATE /MP /Gm-)
	endif ()
endif ()

target_link_libraries(utils PUBLIC libpng)
if (INTERACTIVE)
	target_link_libraries(utils PUBLIC GL_platform_tools GL_core_tools)
endif ()


source_group("source" REGULAR_EXPRESSION "${SOURCE_DIR}/.*")
source_group("source/CUDA" REGULAR_EXPRESSION "${SOURCE_DIR}/CUDA/.*")
