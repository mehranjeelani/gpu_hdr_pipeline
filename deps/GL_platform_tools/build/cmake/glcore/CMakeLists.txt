cmake_minimum_required(VERSION 3.18)

project(glcore)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../source/win32/glcore")


add_library(glcore STATIC 
	"${SOURCE_DIR}/context.cpp"
)

target_include_directories(glcore PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/../../../source/win32/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
)

target_compile_definitions(glcore PUBLIC -DGLCORE_STATIC)

if (MSVC)
	target_compile_options(glcore PRIVATE /WX /permissive-)

	if (CMAKE_GENERATOR MATCHES "Visual Studio")
		target_compile_options(glcore PRIVATE /MP /Gm-)
	endif ()
endif ()

set_target_properties(glcore PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
	POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(glcore PUBLIC Win32_core_tools)
