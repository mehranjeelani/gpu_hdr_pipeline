cmake_minimum_required(VERSION 3.10)

project(Win32_core_tools)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_DEBUG_POSTFIX D)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<1:lib>")


set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../source")

add_library(Win32_core_tools STATIC 
	"${SOURCE_DIR}/error.cpp"
	"${SOURCE_DIR}/event.cpp"
	"${SOURCE_DIR}/file.cpp"
	"${SOURCE_DIR}/handle.cpp"
	"${SOURCE_DIR}/memory.cpp"
	"${SOURCE_DIR}/module.cpp"
	"${SOURCE_DIR}/unicode.cpp"
	"${SOURCE_DIR}/window.cpp"
)

target_include_directories(Win32_core_tools PUBLIC 
	"${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

if (MSVC)
	target_compile_definitions(Win32_core_tools PRIVATE -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
	target_compile_options(Win32_core_tools PRIVATE /WX /permissive-)

	if (CMAKE_GENERATOR MATCHES "Visual Studio")
		target_compile_options(Win32_core_tools PRIVATE /MP /Gm-)
	endif ()
endif ()

set_target_properties(Win32_core_tools PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
	POSITION_INDEPENDENT_CODE ON
)
