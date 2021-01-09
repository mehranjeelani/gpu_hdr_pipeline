cmake_minimum_required(VERSION 3.18)

project(GL_platform_tools)

find_package(OpenGL REQUIRED)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


if (WIN32)
	add_subdirectory(glcore)
else ()
	find_package(X11 REQUIRED)
endif ()

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../source")

add_library(GL_platform_tools STATIC 
	"${SOURCE_DIR}/DefaultDisplayHandler.cpp"
	"${SOURCE_DIR}/DefaultInputHandler.cpp"
)

if (WIN32)
	set(PLATFORM_SOURCE_DIR "${SOURCE_DIR}/win32")
	target_sources(GL_platform_tools PRIVATE
		"${PLATFORM_SOURCE_DIR}/Win32GLApplication.cpp"
		"${PLATFORM_SOURCE_DIR}/Win32GLContext.cpp"
		"${PLATFORM_SOURCE_DIR}/Win32GLWindow.cpp"
	)
else ()
	set(PLATFORM_SOURCE_DIR "${SOURCE_DIR}/x11")
	target_sources(GL_platform_tools PRIVATE
		"${PLATFORM_SOURCE_DIR}/X11Display.cpp"
		"${PLATFORM_SOURCE_DIR}/X11GLApplication.cpp"
		"${PLATFORM_SOURCE_DIR}/X11GLContext.cpp"
		"${PLATFORM_SOURCE_DIR}/X11GLWindow.cpp"
	)
endif ()

target_include_directories(GL_platform_tools PUBLIC 
	"${CMAKE_CURRENT_SOURCE_DIR}/../../include"
	"${PLATFORM_SOURCE_DIR}/include"
)

if (NOT WIN32)
	target_include_directories(GL_platform_tools PUBLIC "${SOURCE_DIR}")
endif ()

if (MSVC)
	target_compile_definitions(GL_platform_tools PRIVATE -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
	target_compile_options(GL_platform_tools PRIVATE /WX /permissive-)

	if (CMAKE_GENERATOR MATCHES "Visual Studio")
		target_compile_options(GL_platform_tools PRIVATE /MP /Gm-)
	endif ()
endif ()

set_target_properties(GL_platform_tools PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
	POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(GL_platform_tools PUBLIC OpenGL::GL)
if (WIN32)
	target_link_libraries(GL_platform_tools PUBLIC Win32_core_tools glcore)
else ()
	target_link_libraries(GL_platform_tools PUBLIC OpenGL::GLX X11::X11)
endif ()
