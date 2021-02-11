cmake_minimum_required(VERSION 3.18)

project(framework CXX)


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<IF:$<BOOL:${WIN32}>,bin,lib>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
option(BUILD_TOOLS "build tools" OFF)
option(INTERACTIVE "interactive build" OFF)


set(DEPENDENCIES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../deps")

if (WIN32)
	add_subdirectory("${DEPENDENCIES_DIR}/Win32_core_tools/build/cmake" Win32_core_tools)
endif ()

if (INTERACTIVE)
	add_subdirectory("${DEPENDENCIES_DIR}/GL_platform_tools/build/cmake" GL_platform_tools)
	add_subdirectory("${DEPENDENCIES_DIR}/GL_core_tools/build/cmake" GL_core_tools)
	include("${DEPENDENCIES_DIR}/GLSL_build_tools/build/cmake/add_glsl_sources.cmake")
endif ()


find_package(CUDAToolkit)

if (NOT CUDAToolkit_FOUND)
	message("WARNING: CUDA Toolkit not found, will only build part of the framework!")
endif ()


function(configure_project name)
	set_target_properties(${name} PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
		CXX_EXTENSIONS OFF
		CXX_VISIBILITY_PRESET hidden
		CUDA_STANDARD 14
		CUDA_STANDARD_REQUIRED ON
		CUDA_EXTENSIONS OFF
		CUDA_SEPARABLE_COMPILATION ON
		POSITION_INDEPENDENT_CODE ON
	)

	if (FULL_VERSION)
		target_compile_definitions(${name} PRIVATE -DFULL_VERSION)
	endif ()

	if (MSVC)
		target_compile_definitions(${name} PRIVATE -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
		target_compile_options(${name} PRIVATE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/permissive->)

		if (CMAKE_GENERATOR MATCHES "Visual Studio")
			target_compile_options(${name} PRIVATE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/MP /Gm->)
		endif ()
	endif ()

	get_target_property(SOURCES ${name} SOURCES)

	source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCES})

	list(FILTER SOURCES INCLUDE REGEX ".*\.glsl")

	if (SOURCES)
		add_glsl_sources(${name}_shaders ${SOURCES})
		target_link_libraries(${name} ${name}_shaders)
	endif ()

	set_property(TARGET ${name} APPEND PROPERTY BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

	set_target_properties(${name} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/../")
endfunction ()

add_subdirectory(utils)
add_subdirectory(hdr_pipeline)
add_subdirectory(particles)
if (BUILD_TOOLS)
	add_subdirectory(tools)
endif ()

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT hdr_pipeline)
