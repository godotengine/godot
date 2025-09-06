# This file is part of CMake-codecov.
#
# Copyright (c)
#   2015-2017 RWTH Aachen University, Federal Republic of Germany
#
# See the LICENSE file in the package base directory for details
#
# Written by Alexander Haase, alexander.haase@rwth-aachen.de
#


# Add an option to choose, if coverage should be enabled or not. If enabled
# marked targets will be build with coverage support and appropriate targets
# will be added. If disabled coverage will be ignored for *ALL* targets.
option(ENABLE_COVERAGE "Enable coverage build." OFF)

set(COVERAGE_FLAG_CANDIDATES
	# gcc and clang
	"-O0 -g -fprofile-arcs -ftest-coverage"

	# gcc and clang fallback
	"-O0 -g --coverage"
)


# Add coverage support for target ${TNAME} and register target for coverage
# evaluation. If coverage is disabled or not supported, this function will
# simply do nothing.
#
# Note: This function is only a wrapper to define this function always, even if
#   coverage is not supported by the compiler or disabled. This function must
#   be defined here, because the module will be exited, if there is no coverage
#   support by the compiler or it is disabled by the user.
function (add_coverage TNAME)
	# only add coverage for target, if coverage is support and enabled.
	if (ENABLE_COVERAGE)
		foreach (TNAME ${ARGV})
			add_coverage_target(${TNAME})
		endforeach ()
	endif ()
endfunction (add_coverage)


# Add global target to gather coverage information after all targets have been
# added. Other evaluation functions could be added here, after checks for the
# specific module have been passed.
#
# Note: This function is only a wrapper to define this function always, even if
#   coverage is not supported by the compiler or disabled. This function must
#   be defined here, because the module will be exited, if there is no coverage
#   support by the compiler or it is disabled by the user.
function (coverage_evaluate)
	# add lcov evaluation
	if (LCOV_FOUND)
		lcov_capture_initial()
		lcov_capture()
	endif (LCOV_FOUND)
endfunction ()


# Exit this module, if coverage is disabled. add_coverage is defined before this
# return, so this module can be exited now safely without breaking any build-
# scripts.
if (NOT ENABLE_COVERAGE)
	return()
endif ()




# Find the reuired flags foreach language.
set(CMAKE_REQUIRED_QUIET_SAVE ${CMAKE_REQUIRED_QUIET})
set(CMAKE_REQUIRED_QUIET ${codecov_FIND_QUIETLY})

get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
foreach (LANG ${ENABLED_LANGUAGES})
	# Coverage flags are not dependent on language, but the used compiler. So
	# instead of searching flags foreach language, search flags foreach compiler
	# used.
	set(COMPILER ${CMAKE_${LANG}_COMPILER_ID})
	if (NOT COVERAGE_${COMPILER}_FLAGS)
		foreach (FLAG ${COVERAGE_FLAG_CANDIDATES})
			if(NOT CMAKE_REQUIRED_QUIET)
				message(STATUS "Try ${COMPILER} code coverage flag = [${FLAG}]")
			endif()

			set(CMAKE_REQUIRED_FLAGS "${FLAG}")
			unset(COVERAGE_FLAG_DETECTED CACHE)

			if (${LANG} STREQUAL "C")
				include(CheckCCompilerFlag)
				check_c_compiler_flag("${FLAG}" COVERAGE_FLAG_DETECTED)

			elseif (${LANG} STREQUAL "CXX")
				include(CheckCXXCompilerFlag)
				check_cxx_compiler_flag("${FLAG}" COVERAGE_FLAG_DETECTED)

			elseif (${LANG} STREQUAL "Fortran")
				# CheckFortranCompilerFlag was introduced in CMake 3.x. To be
				# compatible with older Cmake versions, we will check if this
				# module is present before we use it. Otherwise we will define
				# Fortran coverage support as not available.
				include(CheckFortranCompilerFlag OPTIONAL
					RESULT_VARIABLE INCLUDED)
				if (INCLUDED)
					check_fortran_compiler_flag("${FLAG}"
						COVERAGE_FLAG_DETECTED)
				elseif (NOT CMAKE_REQUIRED_QUIET)
					message("-- Performing Test COVERAGE_FLAG_DETECTED")
					message("-- Performing Test COVERAGE_FLAG_DETECTED - Failed"
						" (Check not supported)")
				endif ()
			endif()

			if (COVERAGE_FLAG_DETECTED)
				set(COVERAGE_${COMPILER}_FLAGS "${FLAG}"
					CACHE STRING "${COMPILER} flags for code coverage.")
				mark_as_advanced(COVERAGE_${COMPILER}_FLAGS)
				break()
			else ()
				message(WARNING "Code coverage is not available for ${COMPILER}"
				        " compiler. Targets using this compiler will be "
				        "compiled without it.")
			endif ()
		endforeach ()
	endif ()
endforeach ()

set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_SAVE})




# Helper function to get the language of a source file.
function (codecov_lang_of_source FILE RETURN_VAR)
	get_filename_component(FILE_EXT "${FILE}" EXT)
	string(TOLOWER "${FILE_EXT}" FILE_EXT)
	string(SUBSTRING "${FILE_EXT}" 1 -1 FILE_EXT)

	get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
	foreach (LANG ${ENABLED_LANGUAGES})
		list(FIND CMAKE_${LANG}_SOURCE_FILE_EXTENSIONS "${FILE_EXT}" TEMP)
		if (NOT ${TEMP} EQUAL -1)
			set(${RETURN_VAR} "${LANG}" PARENT_SCOPE)
			return()
		endif ()
	endforeach()

	set(${RETURN_VAR} "" PARENT_SCOPE)
endfunction ()


# Helper function to get the relative path of the source file destination path.
# This path is needed by FindGcov and FindLcov cmake files to locate the
# captured data.
function (codecov_path_of_source FILE RETURN_VAR)
	string(REGEX MATCH "TARGET_OBJECTS:([^ >]+)" _source ${FILE})

	# If expression was found, SOURCEFILE is a generator-expression for an
	# object library. Currently we found no way to call this function automatic
	# for the referenced target, so it must be called in the directoryso of the
	# object library definition.
	if (NOT "${_source}" STREQUAL "")
		set(${RETURN_VAR} "" PARENT_SCOPE)
		return()
	endif ()


	string(REPLACE "${CMAKE_CURRENT_BINARY_DIR}/" "" FILE "${FILE}")
	if(IS_ABSOLUTE ${FILE})
		file(RELATIVE_PATH FILE ${CMAKE_CURRENT_SOURCE_DIR} ${FILE})
	endif()

	# get the right path for file
	string(REPLACE ".." "__" PATH "${FILE}")

	set(${RETURN_VAR} "${PATH}" PARENT_SCOPE)
endfunction()




# Add coverage support for target ${TNAME} and register target for coverage
# evaluation.
function(add_coverage_target TNAME)
	# Check if all sources for target use the same compiler. If a target uses
	# e.g. C and Fortran mixed and uses different compilers (e.g. clang and
	# gfortran) this can trigger huge problems, because different compilers may
	# use different implementations for code coverage.
	get_target_property(TSOURCES ${TNAME} SOURCES)
	set(TARGET_COMPILER "")
	set(ADDITIONAL_FILES "")
	foreach (FILE ${TSOURCES})
		# If expression was found, FILE is a generator-expression for an object
		# library. Object libraries will be ignored.
		string(REGEX MATCH "TARGET_OBJECTS:([^ >]+)" _file ${FILE})
		if ("${_file}" STREQUAL "")
			codecov_lang_of_source(${FILE} LANG)
			if (LANG)
				list(APPEND TARGET_COMPILER ${CMAKE_${LANG}_COMPILER_ID})

				list(APPEND ADDITIONAL_FILES "${FILE}.gcno")
				list(APPEND ADDITIONAL_FILES "${FILE}.gcda")
			endif ()
		endif ()
	endforeach ()

	list(REMOVE_DUPLICATES TARGET_COMPILER)
	list(LENGTH TARGET_COMPILER NUM_COMPILERS)

	if (NUM_COMPILERS GREATER 1)
		message(WARNING "Can't use code coverage for target ${TNAME}, because "
		        "it will be compiled by incompatible compilers. Target will be "
		        "compiled without code coverage.")
		return()

	elseif (NUM_COMPILERS EQUAL 0)
		message(WARNING "Can't use code coverage for target ${TNAME}, because "
		        "it uses an unknown compiler. Target will be compiled without "
		        "code coverage.")
		return()

	elseif (NOT DEFINED "COVERAGE_${TARGET_COMPILER}_FLAGS")
		# A warning has been printed before, so just return if flags for this
		# compiler aren't available.
		return()
	endif()


	# enable coverage for target
	set_property(TARGET ${TNAME} APPEND_STRING
		PROPERTY COMPILE_FLAGS " ${COVERAGE_${TARGET_COMPILER}_FLAGS}")
	set_property(TARGET ${TNAME} APPEND_STRING
		PROPERTY LINK_FLAGS " ${COVERAGE_${TARGET_COMPILER}_FLAGS}")


	# Add gcov files generated by compiler to clean target.
	set(CLEAN_FILES "")
	foreach (FILE ${ADDITIONAL_FILES})
		codecov_path_of_source(${FILE} FILE)
		list(APPEND CLEAN_FILES "CMakeFiles/${TNAME}.dir/${FILE}")
	endforeach()

	set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES
		"${CLEAN_FILES}")


	add_gcov_target(${TNAME})
	add_lcov_target(${TNAME})
endfunction(add_coverage_target)




# Include modules for parsing the collected data and output it in a readable
# format (like gcov and lcov).
find_package(Gcov)
find_package(Lcov)
