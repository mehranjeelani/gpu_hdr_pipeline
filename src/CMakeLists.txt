cmake_minimum_required(VERSION 3.18)

project(framework CXX CUDA)


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_DEBUG_POSTFIX D)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<1:lib>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<IF:$<BOOL:${WIN32}>,bin,lib>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


option(INTERACTIVE "interactive build" OFF)


set(DEPENDENCIES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../deps")

if (INTERACTIVE)
	if (WIN32)
		add_subdirectory("${DEPENDENCIES_DIR}/Win32_core_tools/build/cmake" Win32_core_tools)
	endif ()
	add_subdirectory("${DEPENDENCIES_DIR}/GL_platform_tools/build/cmake" GL_platform_tools)
	add_subdirectory("${DEPENDENCIES_DIR}/GL_core_tools/build/cmake" GL_core_tools)
endif ()


function(configure_project name)
	set_target_properties(${name} PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
		CXX_EXTENSIONS OFF
		CUDA_STANDARD 14
		CUDA_STANDARD_REQUIRED ON
		CUDA_EXTENSIONS OFF
		CUDA_SEPARABLE_COMPILATION ON
		POSITION_INDEPENDENT_CODE ON
	)

	# set_property(TARGET ${name} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

	if (FULL_VERSION)
		target_compile_definitions(${name} PRIVATE -DFULL_VERSION)
	endif ()

	if (MSVC)
		target_compile_definitions(${name} PRIVATE -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
		# target_compile_options(${name} PRIVATE /permissive-)

		# if (CMAKE_GENERATOR MATCHES "Visual Studio")
		# 	target_compile_options(${name} PRIVATE /MP /Gm-)
		# endif ()
	endif ()

	source_group("source" REGULAR_EXPRESSION ".*")

	set_target_properties(${name} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/../")
endfunction ()

add_subdirectory(utils)
add_subdirectory(hdr_pipeline)


if (INTERACTIVE)
	if (WIN32)
		set_target_properties(Win32_core_tools PROPERTIES FOLDER "dependencies")
		set_target_properties(glcore PROPERTIES FOLDER "dependencies")
	endif ()
	set_target_properties(GL_platform_tools PROPERTIES FOLDER "dependencies")
	set_target_properties(GL_core_tools PROPERTIES FOLDER "dependencies")
endif ()
set_target_properties(zlib PROPERTIES FOLDER "dependencies")
set_target_properties(libpng PROPERTIES FOLDER "dependencies")
set_target_properties(utils PROPERTIES FOLDER "dependencies")

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT hdr_pipeline)
