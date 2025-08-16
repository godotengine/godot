# This file is part of CMake-codecov.
#
# Copyright (c)
#   2015-2017 RWTH Aachen University, Federal Republic of Germany
#
# See the LICENSE file in the package base directory for details
#
# Written by Alexander Haase, alexander.haase@rwth-aachen.de
#


# include required Modules
include(FindPackageHandleStandardArgs)


# Search for gcov binary.
set(CMAKE_REQUIRED_QUIET_SAVE ${CMAKE_REQUIRED_QUIET})
set(CMAKE_REQUIRED_QUIET ${codecov_FIND_QUIETLY})

get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
foreach (LANG ${ENABLED_LANGUAGES})
	# Gcov evaluation is dependent on the used compiler. Check gcov support for
	# each compiler that is used. If gcov binary was already found for this
	# compiler, do not try to find it again.
	if (NOT GCOV_${CMAKE_${LANG}_COMPILER_ID}_BIN)
		get_filename_component(COMPILER_PATH "${CMAKE_${LANG}_COMPILER}" PATH)

		if ("${CMAKE_${LANG}_COMPILER_ID}" STREQUAL "GNU")
			# Some distributions like OSX (homebrew) ship gcov with the compiler
			# version appended as gcov-x. To find this binary we'll build the
			# suggested binary name with the compiler version.
			string(REGEX MATCH "^[0-9]+" GCC_VERSION
				"${CMAKE_${LANG}_COMPILER_VERSION}")

			find_program(GCOV_BIN NAMES gcov-${GCC_VERSION} gcov
				HINTS ${COMPILER_PATH})

		elseif ("${CMAKE_${LANG}_COMPILER_ID}" STREQUAL "Clang")
			# Some distributions like Debian ship llvm-cov with the compiler
			# version appended as llvm-cov-x.y. To find this binary we'll build
			# the suggested binary name with the compiler version.
			string(REGEX MATCH "^[0-9]+.[0-9]+" LLVM_VERSION
				"${CMAKE_${LANG}_COMPILER_VERSION}")

			# llvm-cov prior version 3.5 seems to be not working with coverage
			# evaluation tools, but these versions are compatible with the gcc
			# gcov tool.
			if(LLVM_VERSION VERSION_GREATER 3.4)
				find_program(LLVM_COV_BIN NAMES "llvm-cov-${LLVM_VERSION}"
					"llvm-cov" HINTS ${COMPILER_PATH})
				mark_as_advanced(LLVM_COV_BIN)

				if (LLVM_COV_BIN)
					find_program(LLVM_COV_WRAPPER "llvm-cov-wrapper" PATHS
						${CMAKE_MODULE_PATH})
					if (LLVM_COV_WRAPPER)
						set(GCOV_BIN "${LLVM_COV_WRAPPER}" CACHE FILEPATH "")

						# set additional parameters
						set(GCOV_${CMAKE_${LANG}_COMPILER_ID}_ENV
							"LLVM_COV_BIN=${LLVM_COV_BIN}" CACHE STRING
							"Environment variables for llvm-cov-wrapper.")
						mark_as_advanced(GCOV_${CMAKE_${LANG}_COMPILER_ID}_ENV)
					endif ()
				endif ()
			endif ()

			if (NOT GCOV_BIN)
				# Fall back to gcov binary if llvm-cov was not found or is
				# incompatible. This is the default on OSX, but may crash on
				# recent Linux versions.
				find_program(GCOV_BIN gcov HINTS ${COMPILER_PATH})
			endif ()
		endif ()


		if (GCOV_BIN)
			set(GCOV_${CMAKE_${LANG}_COMPILER_ID}_BIN "${GCOV_BIN}" CACHE STRING
				"${LANG} gcov binary.")

			if (NOT CMAKE_REQUIRED_QUIET)
				message("-- Found gcov evaluation for "
				"${CMAKE_${LANG}_COMPILER_ID}: ${GCOV_BIN}")
			endif()

			unset(GCOV_BIN CACHE)
		endif ()
	endif ()
endforeach ()




# Add a new global target for all gcov targets. This target could be used to
# generate the gcov files for the whole project instead of calling <TARGET>-gcov
# for each target.
if (NOT TARGET gcov)
	add_custom_target(gcov)
endif (NOT TARGET gcov)



# This function will add gcov evaluation for target <TNAME>. Only sources of
# this target will be evaluated and no dependencies will be added. It will call
# Gcov on any source file of <TNAME> once and store the gcov file in the same
# directory.
function (add_gcov_target TNAME)
	set(TDIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TNAME}.dir)

	# We don't have to check, if the target has support for coverage, thus this
	# will be checked by add_coverage_target in Findcoverage.cmake. Instead we
	# have to determine which gcov binary to use.
	get_target_property(TSOURCES ${TNAME} SOURCES)
	set(SOURCES "")
	set(TCOMPILER "")
	foreach (FILE ${TSOURCES})
		codecov_path_of_source(${FILE} FILE)
		if (NOT "${FILE}" STREQUAL "")
			codecov_lang_of_source(${FILE} LANG)
			if (NOT "${LANG}" STREQUAL "")
				list(APPEND SOURCES "${FILE}")
				set(TCOMPILER ${CMAKE_${LANG}_COMPILER_ID})
			endif ()
		endif ()
	endforeach ()

	# If no gcov binary was found, coverage data can't be evaluated.
	if (NOT GCOV_${TCOMPILER}_BIN)
		message(WARNING "No coverage evaluation binary found for ${TCOMPILER}.")
		return()
	endif ()

	set(GCOV_BIN "${GCOV_${TCOMPILER}_BIN}")
	set(GCOV_ENV "${GCOV_${TCOMPILER}_ENV}")


	set(BUFFER "")
	foreach(FILE ${SOURCES})
		get_filename_component(FILE_PATH "${TDIR}/${FILE}" PATH)

		# call gcov
		add_custom_command(OUTPUT ${TDIR}/${FILE}.gcov
			COMMAND ${GCOV_ENV} ${GCOV_BIN} ${TDIR}/${FILE}.gcno > /dev/null
			DEPENDS ${TNAME} ${TDIR}/${FILE}.gcno
			WORKING_DIRECTORY ${FILE_PATH}
		)

		list(APPEND BUFFER ${TDIR}/${FILE}.gcov)
	endforeach()


	# add target for gcov evaluation of <TNAME>
	add_custom_target(${TNAME}-gcov DEPENDS ${BUFFER})

	# add evaluation target to the global gcov target.
	add_dependencies(gcov ${TNAME}-gcov)
endfunction (add_gcov_target)
