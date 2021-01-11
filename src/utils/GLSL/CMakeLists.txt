cmake_minimum_required(VERSION 3.18)


project(GLSL_utils)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(GLSL_utils INTERFACE
	# "${SOURCE_DIR}/camera"
)

target_include_directories(GLSL_utils INTERFACE "${SOURCE_DIR}/")

# source_group("source" REGULAR_EXPRESSION "${SOURCE_DIR}/.*")

# set_target_properties(GLSL_utils PROPERTIES FOLDER "dependencies")
