include_guard()

macro(gdj_escape_separator variable output_variable)
	string(REPLACE ";" $<SEMICOLON> ${output_variable} "${${variable}}")
endmacro()

macro(gdj_duplicate_config src dst)
	string(TOUPPER ${src} src_upper)
	string(TOUPPER ${dst} dst_upper)

	set(CMAKE_CXX_FLAGS_${dst_upper} ${CMAKE_CXX_FLAGS_${src_upper}} CACHE STRING
		"Flags used by the CXX compiler during ${dst_upper} builds."
	)

	set(CMAKE_EXE_LINKER_FLAGS_${dst_upper} ${CMAKE_EXE_LINKER_FLAGS_${src_upper}} CACHE STRING
		"Flags used by the linker during ${dst_upper} builds."
	)

	set(CMAKE_MODULE_LINKER_FLAGS_${dst_upper} ${CMAKE_MODULE_LINKER_FLAGS_${src_upper}} CACHE STRING
		"Flags used by the linker during the creation of modules during ${dst_upper} builds."
	)

	set(CMAKE_SHARED_LINKER_FLAGS_${dst_upper} ${CMAKE_SHARED_LINKER_FLAGS_${src_upper}} CACHE STRING
		"Flags used by the linker during the creation of shared libraries during ${dst_upper} builds."
	)

	set(CMAKE_STATIC_LINKER_FLAGS_${dst_upper} ${CMAKE_STATIC_LINKER_FLAGS_${src_upper}} CACHE STRING
		"Flags used by the linker during the creation of static libraries during ${dst_upper} builds."
	)

	mark_as_advanced(
		CMAKE_CXX_FLAGS_${dst_upper}
		CMAKE_EXE_LINKER_FLAGS_${dst_upper}
		CMAKE_MODULE_LINKER_FLAGS_${dst_upper}
		CMAKE_SHARED_LINKER_FLAGS_${dst_upper}
		CMAKE_STATIC_LINKER_FLAGS_${dst_upper}
	)

	if(MSVC)
		set(CMAKE_RC_FLAGS_${dst_upper} ${CMAKE_RC_FLAGS_${src_upper}} CACHE STRING
			"Flags for Windows Resource Compiler during ${dst_upper} builds."
		)

		mark_as_advanced(CMAKE_RC_FLAGS_${dst_upper})
	endif()
endmacro()

macro(gdj_remove_config name)
	string(TOUPPER ${name} name_upper)

	unset(CMAKE_CXX_FLAGS_${name_upper} CACHE)
	unset(CMAKE_EXE_LINKER_FLAGS_${name_upper} CACHE)
	unset(CMAKE_MODULE_LINKER_FLAGS_${name_upper} CACHE)
	unset(CMAKE_SHARED_LINKER_FLAGS_${name_upper} CACHE)
	unset(CMAKE_STATIC_LINKER_FLAGS_${name_upper} CACHE)

	if(MSVC)
		unset(CMAKE_RC_FLAGS_${name_upper} CACHE)
	endif()
endmacro()

function(gdj_args_to_script variable args)
	set(arg_pattern [[^(.+)=(.*)$]])

	set(script_content "")
	set(script_line "")
	set(arg_rest "")

	foreach(element IN LISTS args)
		if(element MATCHES [[^-D(.*)]])
			set(arg ${CMAKE_MATCH_1})

			if(NOT script_line STREQUAL "")
				string(APPEND script_line "${arg_rest}\" CACHE INTERNAL \"\")")
				string(APPEND script_content "${script_line}\n")

				set(script_line "")
				set(arg_rest "")
			endif()

			if(arg MATCHES ${arg_pattern})
				set(arg_name ${CMAKE_MATCH_1})
				set(arg_value ${CMAKE_MATCH_2})
				set(script_line "set(${arg_name} \"${arg_value}")
			endif()
		else()
			string(APPEND arg_rest "\\\;${element}")
		endif()
	endforeach()

	if(NOT script_line STREQUAL "")
		string(APPEND script_line "${arg_rest}\" CACHE INTERNAL \"\")")
		string(APPEND script_content "${script_line}\n")
	endif()

	set(${variable} ${script_content} PARENT_SCOPE)
endfunction()

function(gdj_get_build_id output_variable)
	find_package(Git REQUIRED)

	execute_process(
		COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
		ERROR_QUIET
		OUTPUT_STRIP_TRAILING_WHITESPACE
		OUTPUT_VARIABLE git_hash
		RESULT_VARIABLE git_exit_code
	)

	if(git_exit_code EQUAL 0)
		string(SUBSTRING ${git_hash} 0 10 build_id)
	else()
		set(build_id "custom")
	endif()

	set(${output_variable} ${build_id} PARENT_SCOPE)
endfunction()

function(gdj_generate_rc_file output_variable)
	set(rc_in_file ${CMAKE_CURRENT_BINARY_DIR}/info.rc.in)
	configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/templates/info.rc.in ${rc_in_file})

	set(rc_file ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/info.rc)
	file(GENERATE OUTPUT ${rc_file} INPUT ${rc_in_file})

	set(${output_variable} ${rc_file} PARENT_SCOPE)
endfunction()

function(gdj_generate_info_file output_variable)
	set(info_file ${CMAKE_CURRENT_BINARY_DIR}/info.cpp)
	configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/templates/info.cpp.in ${info_file})
	set(${output_variable} ${info_file} PARENT_SCOPE)
endfunction()

function(gdj_generate_exports_ld output_variable)
	set(exports_file ${CMAKE_CURRENT_BINARY_DIR}/exports.map)
	configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/templates/exports_ld.map.in ${exports_file})

	set(${output_variable} ${exports_file} PARENT_SCOPE)
endfunction()

function(gdj_generate_exports_ld64 output_variable)
	set(exports_file ${CMAKE_CURRENT_BINARY_DIR}/exports_list)
	configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/templates/exports_ld64.in ${exports_file})

	set(${output_variable} ${exports_file} PARENT_SCOPE)
endfunction()

function(gdj_generate_gdextension_file output_variable)
	set(gdextension_file ${CMAKE_CURRENT_BINARY_DIR}/godot-jolt.gdextension)
	configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/templates/gdextension.in ${gdextension_file})

	set(${output_variable} ${gdextension_file} PARENT_SCOPE)
endfunction()
