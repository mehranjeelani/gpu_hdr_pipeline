cmake_minimum_required(VERSION 3.18)

project(GL_core_tools)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_DEBUG_POSTFIX D)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<1:lib>")


set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../source")

add_library(GL_core_tools STATIC 
	"${SOURCE_DIR}/buffer.cpp"
	"${SOURCE_DIR}/error.cpp"
	"${SOURCE_DIR}/event.cpp"
	"${SOURCE_DIR}/framebuffer.cpp"
	"${SOURCE_DIR}/shader.cpp"
	"${SOURCE_DIR}/texture.cpp"
	"${SOURCE_DIR}/transform_feedback.cpp"
	"${SOURCE_DIR}/vertex_array.cpp"
)

target_include_directories(GL_core_tools PUBLIC 
	"${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

if (MSVC)
	target_compile_definitions(GL_core_tools PRIVATE -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
	target_compile_options(GL_core_tools PRIVATE /WX /permissive-)

	if (CMAKE_GENERATOR MATCHES "Visual Studio")
		target_compile_options(GL_core_tools PRIVATE /MP /Gm-)
	endif ()
endif ()

set_target_properties(GL_core_tools PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
	POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(GL_core_tools PUBLIC GL_platform_tools)
