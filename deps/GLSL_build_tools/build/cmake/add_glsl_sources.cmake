cmake_minimum_required(VERSION 3.18)


add_executable(glsl2cpp ${CMAKE_CURRENT_LIST_DIR}/../../source/glsl2cpp.cpp)


macro(glsl_add_definitions sources defs)
	set_property(SOURCE ${sources} APPEND PROPERTY GLSL_DEFINITIONS ${defs})
endmacro()

macro(glsl_add_include_directories sources dirs)
	set_property(SOURCE ${sources} APPEND PROPERTY GLSL_INCLUDE_DIRS ${dirs})
endmacro()

function(glsl_source output source tgt)
	get_property(idirs SOURCE ${source} PROPERTY GLSL_INCLUDE_DIRS)
	foreach (i ${idirs})
		list(APPEND cmdline -I ${i})
	endforeach()
	
	get_property(defs SOURCE ${source} PROPERTY GLSL_DEFINITIONS)
	foreach (d ${defs})
		list(APPEND cmdline -D${d})
	endforeach()
	
	add_custom_command(
		OUTPUT  ${output}
		COMMAND glsl2cpp
		ARGS    $<$<BOOL:$<TARGET_PROPERTY:${tgt},INCLUDE_DIRECTORIES>>:-I$<JOIN:$<TARGET_GENEX_EVAL:${tgt},$<TARGET_PROPERTY:${tgt},INCLUDE_DIRECTORIES>>,$<SEMICOLON>-I>> ${cmdline} -o ${output} ${source}
		DEPENDS ${source}
		COMMAND_EXPAND_LISTS
	)
endfunction()

function(add_glsl_sources tgt)
	foreach (f ${ARGN})
		get_filename_component(n ${f} NAME)
		glsl_source(${n}.cpp ${f} ${tgt})
		list(APPEND shader_files ${n}.cpp)
	endforeach()
	add_library(${tgt} STATIC ${shader_files})
endfunction()
