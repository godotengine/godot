include_guard()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
	message(FATAL_ERROR
		"Please use a subdirectory for CMake's binary directory. "
		"Preferably use the ready-made presets found in 'CMakePresets.json'. "
		"If not, there is a Git ignore rule set up for a 'build' directory."
	)
endif()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
	set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/examples CACHE PATH
		"Install path prefix, prepended onto install directories."
	)

	if(CMAKE_INSTALL_PREFIX STREQUAL CMAKE_CURRENT_SOURCE_DIR)
		message(FATAL_ERROR
			"CMAKE_INSTALL_PREFIX is currently set to the project root directory. Please choose a "
			"different install directory.\nIf you're not sure what this means then delete the "
			"build directory, generate a new one and build again."
		)
	endif()

	set(CMAKE_ERROR_DEPRECATED TRUE CACHE BOOL
		"Whether to issue errors for deprecated CMake functionality."
	)

	mark_as_advanced(CMAKE_ERROR_DEPRECATED)
endif()

if(NOT DEFINED MSVC)
	set(MSVC FALSE)
endif()
