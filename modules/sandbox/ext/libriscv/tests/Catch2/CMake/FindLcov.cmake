# This file is part of CMake-codecov.
#
# Copyright (c)
#   2015-2017 RWTH Aachen University, Federal Republic of Germany
#
# See the LICENSE file in the package base directory for details
#
# Written by Alexander Haase, alexander.haase@rwth-aachen.de
#


# configuration
set(LCOV_DATA_PATH "${CMAKE_BINARY_DIR}/lcov/data")
set(LCOV_DATA_PATH_INIT "${LCOV_DATA_PATH}/init")
set(LCOV_DATA_PATH_CAPTURE "${LCOV_DATA_PATH}/capture")
set(LCOV_HTML_PATH "${CMAKE_BINARY_DIR}/lcov/html")




# Search for Gcov which is used by Lcov.
find_package(Gcov)




# This function will add lcov evaluation for target <TNAME>. Only sources of
# this target will be evaluated and no dependencies will be added. It will call
# geninfo on any source file of <TNAME> once and store the info file in the same
# directory.
#
# Note: This function is only a wrapper to define this function always, even if
#   coverage is not supported by the compiler or disabled. This function must
#   be defined here, because the module will be exited, if there is no coverage
#   support by the compiler or it is disabled by the user.
function (add_lcov_target TNAME)
	if (LCOV_FOUND)
		# capture initial coverage data
		lcov_capture_initial_tgt(${TNAME})

		# capture coverage data after execution
		lcov_capture_tgt(${TNAME})
	endif ()
endfunction (add_lcov_target)




# include required Modules
include(FindPackageHandleStandardArgs)

# Search for required lcov binaries.
find_program(LCOV_BIN lcov)
find_program(GENINFO_BIN geninfo)
find_program(GENHTML_BIN genhtml)
find_package_handle_standard_args(lcov
	REQUIRED_VARS LCOV_BIN GENINFO_BIN GENHTML_BIN
)

# enable genhtml C++ demangeling, if c++filt is found.
set(GENHTML_CPPFILT_FLAG "")
find_program(CPPFILT_BIN c++filt)
if (NOT CPPFILT_BIN STREQUAL "")
	set(GENHTML_CPPFILT_FLAG "--demangle-cpp")
endif (NOT CPPFILT_BIN STREQUAL "")

# enable no-external flag for lcov, if available.
if (GENINFO_BIN AND NOT DEFINED GENINFO_EXTERN_FLAG)
	set(FLAG "")
	execute_process(COMMAND ${GENINFO_BIN} --help OUTPUT_VARIABLE GENINFO_HELP)
	string(REGEX MATCH "external" GENINFO_RES "${GENINFO_HELP}")
	if (GENINFO_RES)
		set(FLAG "--no-external")
	endif ()

	set(GENINFO_EXTERN_FLAG "${FLAG}"
		CACHE STRING "Geninfo flag to exclude system sources.")
endif ()

# If Lcov was not found, exit module now.
if (NOT LCOV_FOUND)
	return()
endif (NOT LCOV_FOUND)




# Create directories to be used.
file(MAKE_DIRECTORY ${LCOV_DATA_PATH_INIT})
file(MAKE_DIRECTORY ${LCOV_DATA_PATH_CAPTURE})

set(LCOV_REMOVE_PATTERNS "")

# This function will merge lcov files to a single target file. Additional lcov
# flags may be set with setting LCOV_EXTRA_FLAGS before calling this function.
function (lcov_merge_files OUTFILE ...)
	# Remove ${OUTFILE} from ${ARGV} and generate lcov parameters with files.
	list(REMOVE_AT ARGV 0)

	# Generate merged file.
	string(REPLACE "${CMAKE_BINARY_DIR}/" "" FILE_REL "${OUTFILE}")
	add_custom_command(OUTPUT "${OUTFILE}.raw"
		COMMAND cat ${ARGV} > ${OUTFILE}.raw
		DEPENDS ${ARGV}
		COMMENT "Generating ${FILE_REL}"
	)

	add_custom_command(OUTPUT "${OUTFILE}"
		COMMAND ${LCOV_BIN} --quiet -a ${OUTFILE}.raw --output-file ${OUTFILE}
			--base-directory ${PROJECT_SOURCE_DIR} ${LCOV_EXTRA_FLAGS}
		COMMAND ${LCOV_BIN} --quiet -r ${OUTFILE} ${LCOV_REMOVE_PATTERNS}
			--output-file ${OUTFILE} ${LCOV_EXTRA_FLAGS}
		DEPENDS ${OUTFILE}.raw
		COMMENT "Post-processing ${FILE_REL}"
	)
endfunction ()




# Add a new global target to generate initial coverage reports for all targets.
# This target will be used to generate the global initial info file, which is
# used to gather even empty report data.
if (NOT TARGET lcov-capture-init)
	add_custom_target(lcov-capture-init)
	set(LCOV_CAPTURE_INIT_FILES "" CACHE INTERNAL "")
endif (NOT TARGET lcov-capture-init)


# This function will add initial capture of coverage data for target <TNAME>,
# which is needed to get also data for objects, which were not loaded at
# execution time. It will call geninfo for every source file of <TNAME> once and
# store the info file in the same directory.
function (lcov_capture_initial_tgt TNAME)
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


	set(TDIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TNAME}.dir)
	set(GENINFO_FILES "")
	foreach(FILE ${SOURCES})
		# generate empty coverage files
		set(OUTFILE "${TDIR}/${FILE}.info.init")
		list(APPEND GENINFO_FILES ${OUTFILE})

		add_custom_command(OUTPUT ${OUTFILE} COMMAND ${GCOV_ENV} ${GENINFO_BIN}
				--quiet --base-directory ${PROJECT_SOURCE_DIR} --initial
				--gcov-tool ${GCOV_BIN} --output-filename ${OUTFILE}
				${GENINFO_EXTERN_FLAG} ${TDIR}/${FILE}.gcno
			DEPENDS ${TNAME}
			COMMENT "Capturing initial coverage data for ${FILE}"
		)
	endforeach()

	# Concatenate all files generated by geninfo to a single file per target.
	set(OUTFILE "${LCOV_DATA_PATH_INIT}/${TNAME}.info")
	set(LCOV_EXTRA_FLAGS "--initial")
	lcov_merge_files("${OUTFILE}" ${GENINFO_FILES})
	add_custom_target(${TNAME}-capture-init ALL DEPENDS ${OUTFILE})

	# add geninfo file generation to global lcov-geninfo target
	add_dependencies(lcov-capture-init ${TNAME}-capture-init)
	set(LCOV_CAPTURE_INIT_FILES "${LCOV_CAPTURE_INIT_FILES}"
		"${OUTFILE}" CACHE INTERNAL ""
	)
endfunction (lcov_capture_initial_tgt)


# This function will generate the global info file for all targets. It has to be
# called after all other CMake functions in the root CMakeLists.txt file, to get
# a full list of all targets that generate coverage data.
function (lcov_capture_initial)
	# Skip this function (and do not create the following targets), if there are
	# no input files.
	if ("${LCOV_CAPTURE_INIT_FILES}" STREQUAL "")
		return()
	endif ()

	# Add a new target to merge the files of all targets.
	set(OUTFILE "${LCOV_DATA_PATH_INIT}/all_targets.info")
	lcov_merge_files("${OUTFILE}" ${LCOV_CAPTURE_INIT_FILES})
	add_custom_target(lcov-geninfo-init ALL	DEPENDS ${OUTFILE}
		lcov-capture-init
	)
endfunction (lcov_capture_initial)




# Add a new global target to generate coverage reports for all targets. This
# target will be used to generate the global info file.
if (NOT TARGET lcov-capture)
	add_custom_target(lcov-capture)
	set(LCOV_CAPTURE_FILES "" CACHE INTERNAL "")
endif (NOT TARGET lcov-capture)


# This function will add capture of coverage data for target <TNAME>, which is
# needed to get also data for objects, which were not loaded at execution time.
# It will call geninfo for every source file of <TNAME> once and store the info
# file in the same directory.
function (lcov_capture_tgt TNAME)
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


	set(TDIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TNAME}.dir)
	set(GENINFO_FILES "")
	foreach(FILE ${SOURCES})
		# Generate coverage files. If no .gcda file was generated during
		# execution, the empty coverage file will be used instead.
		set(OUTFILE "${TDIR}/${FILE}.info")
		list(APPEND GENINFO_FILES ${OUTFILE})

		add_custom_command(OUTPUT ${OUTFILE}
			COMMAND test -f "${TDIR}/${FILE}.gcda"
				&& ${GCOV_ENV} ${GENINFO_BIN} --quiet --base-directory
					${PROJECT_SOURCE_DIR} --gcov-tool ${GCOV_BIN}
					--output-filename ${OUTFILE} ${GENINFO_EXTERN_FLAG}
					${TDIR}/${FILE}.gcda
				|| cp ${OUTFILE}.init ${OUTFILE}
			DEPENDS ${TNAME} ${TNAME}-capture-init
			COMMENT "Capturing coverage data for ${FILE}"
		)
	endforeach()

	# Concatenate all files generated by geninfo to a single file per target.
	set(OUTFILE "${LCOV_DATA_PATH_CAPTURE}/${TNAME}.info")
	lcov_merge_files("${OUTFILE}" ${GENINFO_FILES})
	add_custom_target(${TNAME}-geninfo DEPENDS ${OUTFILE})

	# add geninfo file generation to global lcov-capture target
	add_dependencies(lcov-capture ${TNAME}-geninfo)
	set(LCOV_CAPTURE_FILES "${LCOV_CAPTURE_FILES}" "${OUTFILE}" CACHE INTERNAL
		""
	)

	# Add target for generating html output for this target only.
	file(MAKE_DIRECTORY ${LCOV_HTML_PATH}/${TNAME})
	add_custom_target(${TNAME}-genhtml
		COMMAND ${GENHTML_BIN} --quiet --sort --prefix ${PROJECT_SOURCE_DIR}
			--baseline-file ${LCOV_DATA_PATH_INIT}/${TNAME}.info
			--output-directory ${LCOV_HTML_PATH}/${TNAME}
			--title "${CMAKE_PROJECT_NAME} - target ${TNAME}"
			${GENHTML_CPPFILT_FLAG} ${OUTFILE}
		DEPENDS ${TNAME}-geninfo ${TNAME}-capture-init
	)
endfunction (lcov_capture_tgt)


# This function will generate the global info file for all targets. It has to be
# called after all other CMake functions in the root CMakeLists.txt file, to get
# a full list of all targets that generate coverage data.
function (lcov_capture)
	# Skip this function (and do not create the following targets), if there are
	# no input files.
	if ("${LCOV_CAPTURE_FILES}" STREQUAL "")
		return()
	endif ()

	# Add a new target to merge the files of all targets.
	set(OUTFILE "${LCOV_DATA_PATH_CAPTURE}/all_targets.info")
	lcov_merge_files("${OUTFILE}" ${LCOV_CAPTURE_FILES})
	add_custom_target(lcov-geninfo DEPENDS ${OUTFILE} lcov-capture)

	# Add a new global target for all lcov targets. This target could be used to
	# generate the lcov html output for the whole project instead of calling
	# <TARGET>-geninfo and <TARGET>-genhtml for each target. It will also be
	# used to generate a html site for all project data together instead of one
	# for each target.
	if (NOT TARGET lcov)
		file(MAKE_DIRECTORY ${LCOV_HTML_PATH}/all_targets)
		add_custom_target(lcov
			COMMAND ${GENHTML_BIN} --quiet --sort
				--baseline-file ${LCOV_DATA_PATH_INIT}/all_targets.info
				--output-directory ${LCOV_HTML_PATH}/all_targets
				--title "${CMAKE_PROJECT_NAME}" --prefix "${PROJECT_SOURCE_DIR}"
				${GENHTML_CPPFILT_FLAG} ${OUTFILE}
			DEPENDS lcov-geninfo-init lcov-geninfo
		)
	endif ()
endfunction (lcov_capture)




# Add a new global target to generate the lcov html report for the whole project
# instead of calling <TARGET>-genhtml for each target (to create an own report
# for each target). Instead of the lcov target it does not require geninfo for
# all targets, so you have to call <TARGET>-geninfo to generate the info files
# the targets you'd like to have in your report or lcov-geninfo for generating
# info files for all targets before calling lcov-genhtml.
file(MAKE_DIRECTORY ${LCOV_HTML_PATH}/selected_targets)
if (NOT TARGET lcov-genhtml)
	add_custom_target(lcov-genhtml
		COMMAND ${GENHTML_BIN}
			--quiet
			--output-directory ${LCOV_HTML_PATH}/selected_targets
			--title \"${CMAKE_PROJECT_NAME} - targets  `find
				${LCOV_DATA_PATH_CAPTURE} -name \"*.info\" ! -name
				\"all_targets.info\" -exec basename {} .info \\\;`\"
			--prefix ${PROJECT_SOURCE_DIR}
			--sort
			${GENHTML_CPPFILT_FLAG}
			`find ${LCOV_DATA_PATH_CAPTURE} -name \"*.info\" ! -name
				\"all_targets.info\"`
	)
endif (NOT TARGET lcov-genhtml)
