cmake_minimum_required(VERSION 3.18)

project(utils CXX)

add_subdirectory("${DEPENDENCIES_DIR}/zlib" zlib)
add_subdirectory("${DEPENDENCIES_DIR}/libpng" libpng)

add_subdirectory(CUDA)
add_subdirectory(GLSL)


set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(utils STATIC
	"${SOURCE_DIR}/io.h"
	"${SOURCE_DIR}/io/compression.h"
	"${SOURCE_DIR}/io/compression.cpp"
	"${SOURCE_DIR}/io/image_io.h"
	"${SOURCE_DIR}/io/radiance.h"
	"${SOURCE_DIR}/io/radiance.cpp"
	"${SOURCE_DIR}/io/pfm.h"
	"${SOURCE_DIR}/io/pfm.cpp"
	"${SOURCE_DIR}/io/png.h"
	"${SOURCE_DIR}/io/png.cpp"
	"${SOURCE_DIR}/io/obj_stream.h"
	"${SOURCE_DIR}/io/obj_reader.h"
	"${SOURCE_DIR}/io/obj.h"
	"${SOURCE_DIR}/io/obj.cpp"
	"${SOURCE_DIR}/dynamic_library.h"
	"${SOURCE_DIR}/argparse.cpp"
	"${SOURCE_DIR}/argparse.h"
	"${SOURCE_DIR}/screenshot.h"
	"${SOURCE_DIR}/screenshot.cpp"
	"${SOURCE_DIR}/Camera.h"
	"${SOURCE_DIR}/PerspectiveCamera.h"
	"${SOURCE_DIR}/PerspectiveCamera.cpp"
)

if (INTERACTIVE)
	target_sources(utils PRIVATE
		"${SOURCE_DIR}/Navigator.h"
		"${SOURCE_DIR}/OrbitalNavigator.h"
		"${SOURCE_DIR}/OrbitalNavigator.cpp"
	)
	target_link_libraries(utils PUBLIC GL_platform_tools GL_core_tools)
endif ()

target_include_directories(utils PUBLIC "${SOURCE_DIR}/../")

configure_project(utils)

target_link_libraries(utils PUBLIC zlib libpng ${CMAKE_DL_LIBS})
if (WIN32)
	target_link_libraries(utils PUBLIC Win32_core_tools)
endif ()


if (WIN32)
	set_target_properties(Win32_core_tools PROPERTIES FOLDER "dependencies")
endif ()
if (INTERACTIVE)
	if (WIN32)
		set_target_properties(glcore PROPERTIES FOLDER "dependencies")
	endif ()
	set_target_properties(GL_platform_tools PROPERTIES FOLDER "dependencies")
	set_target_properties(GL_core_tools PROPERTIES FOLDER "dependencies")
	set_target_properties(glsl2cpp PROPERTIES FOLDER "dependencies")
endif ()
set_target_properties(zlib PROPERTIES FOLDER "dependencies")
set_target_properties(libpng PROPERTIES FOLDER "dependencies")
set_target_properties(utils PROPERTIES FOLDER "dependencies")
