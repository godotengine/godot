# A simple FindBrotli package for Cmake's find_package function.
# Note: This find package doesn't have version support, as the version file doesn't seem to be installed on most systems.
#
# If you want to find the static packages instead of shared (the default), define BROTLI_USE_STATIC_LIBS as TRUE.
# The targets will have the same names, but it will use the static libs.
#
# Valid find_package COMPONENTS names: "decoder", "encoder", and "common"
# Note that if you're requiring "decoder" or "encoder", then "common" will be automatically added as required.
#
# Defines the libraries (if found): Brotli::decoder, Brotli::encoder, Brotli::common
# and the includes path variable: Brotli_INCLUDE_DIR
#
# If it's failing to find the libraries, try setting BROTLI_ROOT_DIR to the folder containing your library & include dir.

# If they asked for a specific version, warn/fail since we don't support it.
# TODO: if they start distributing the version somewhere, implement finding it.
# See https://github.com/google/brotli/issues/773#issuecomment-579133187
if(Brotli_FIND_VERSION)
	set(_brotli_version_error_msg "FindBrotli.cmake doesn't have version support!")
	# If the package is required, throw a fatal error
	# Otherwise, if not running quietly, we throw a warning
	if(Brotli_FIND_REQUIRED)
		message(FATAL_ERROR "${_brotli_version_error_msg}")
	elseif(NOT Brotli_FIND_QUIETLY)
		message(WARNING "${_brotli_version_error_msg}")
	endif()
endif()

# Since both decoder & encoder require the common lib, force its requirement..
# if the user is requiring either of those other libs.
if(Brotli_FIND_REQUIRED_decoder OR Brotli_FIND_REQUIRED_encoder)
	set(Brotli_FIND_REQUIRED_common TRUE)
endif()

# Support preference of static libs by adjusting CMAKE_FIND_LIBRARY_SUFFIXES
# Credit to FindOpenSSL.cmake for this
if(BROTLI_USE_STATIC_LIBS)
  set(_brotli_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  endif()
endif()

# Make PkgConfig optional, since some users (mainly Windows) don't have it.
# But it's a lot more clean than manually using find_library.
find_package(PkgConfig QUIET)

# Only used if the PkgConfig libraries aren't used.
find_path(Brotli_INCLUDE_DIR
	NAMES
		"brotli/decode.h"
		"brotli/encode.h"
	HINTS
		${BROTLI_ROOT_DIR}
	PATH_SUFFIXES
		"include"
		"includes"
	DOC "The path to Brotli's include directory."
)
# Hides this var from the GUI
mark_as_advanced(Brotli_INCLUDE_DIR)

# Just used for PkgConfig stuff in the loop below
set(_brotli_stat_str "")
if(BROTLI_USE_STATIC_LIBS)
	set(_brotli_stat_str "_STATIC")
endif()

# Each string here is "ComponentName;LiteralName" (the semi-colon is a delimiter)
foreach(_listvar "common;common" "decoder;dec" "encoder;enc")
	# Split the component name and literal library name from the listvar
	list(GET _listvar 0 _component_name)
	list(GET _listvar 1 _libname)

	# NOTE: We can't rely on PkgConf for static libs since the upstream static lib support is broken
	# See https://github.com/google/brotli/issues/795
	# TODO: whenever their issue is fixed upstream, remove this "AND NOT BROTLI_USE_STATIC_LIBS" check
	if(PKG_CONFIG_FOUND AND NOT BROTLI_USE_STATIC_LIBS)
		# These need to be GLOBAL for MinGW when making ALIAS libraries against them.
		# Have to postfix _STATIC on the name to tell PkgConfig to find the static libs.
		pkg_check_modules(Brotli_${_component_name}${_brotli_stat_str} QUIET GLOBAL IMPORTED_TARGET libbrotli${_libname})
	endif()

	# Check if the target was already found by Pkgconf
	if(TARGET PkgConfig::Brotli_${_component_name}${_brotli_stat_str})
		# ALIAS since we don't want the PkgConfig namespace on the Cmake library (for end-users)
		add_library(Brotli::${_component_name} ALIAS PkgConfig::Brotli_${_component_name}${_brotli_stat_str})

		# Tells HANDLE_COMPONENTS we found the component
		set(Brotli_${_component_name}_FOUND TRUE)
		if(Brotli_FIND_REQUIRED_${_component_name})
			# If the lib is required, we can add its literal path as a required var for FindPackageHandleStandardArgs
			# Since it won't accept the PkgConfig targets
			if(BROTLI_USE_STATIC_LIBS)
				list(APPEND _brotli_req_vars "Brotli_${_component_name}_STATIC_LIBRARIES")
			else()
				list(APPEND _brotli_req_vars "Brotli_${_component_name}_LINK_LIBRARIES")
			endif()
		endif()

		# Skip searching for the libs with find_library since it was already found by Pkgconf
		continue()
	endif()

	if(Brotli_FIND_REQUIRED_${_component_name})
		# If it's required, we can set the name used in find_library as a required var for FindPackageHandleStandardArgs
		list(APPEND _brotli_req_vars "Brotli_${_component_name}")
	endif()

	list(APPEND _brotli_lib_names
		"brotli${_libname}"
		"libbrotli${_libname}"
	)
	if(BROTLI_USE_STATIC_LIBS)
		# Postfix "-static" to the libnames since we're looking for static libs
		list(TRANSFORM _brotli_lib_names APPEND "-static")
	endif()

	find_library(Brotli_${_component_name}
		NAMES ${_brotli_lib_names}
		HINTS ${BROTLI_ROOT_DIR}
		PATH_SUFFIXES
			"lib"
			"lib64"
			"libs"
			"libs64"
			"lib/x86_64-linux-gnu"
	)
	# Hide the library variable from the Cmake GUI
	mark_as_advanced(Brotli_${_component_name})

	# Unset since otherwise it'll stick around for the next loop and break things
	unset(_brotli_lib_names)

	# Check if find_library found the library
	if(Brotli_${_component_name})
		# Tells HANDLE_COMPONENTS we found the component
		set(Brotli_${_component_name}_FOUND TRUE)

		add_library("Brotli::${_component_name}" UNKNOWN IMPORTED)
		# Attach the literal library and include dir to the IMPORTED target for the end-user
		set_target_properties("Brotli::${_component_name}" PROPERTIES
			INTERFACE_INCLUDE_DIRECTORIES "${Brotli_INCLUDE_DIR}"
			IMPORTED_LOCATION "${Brotli_${_component_name}}"
		)
	else()
		# Tells HANDLE_COMPONENTS we found the component
		set(Brotli_${_component_name}_FOUND FALSE)
	endif()
endforeach()

include(FindPackageHandleStandardArgs)
# Sets Brotli_FOUND, and fails the find_package(Brotli) call if it was REQUIRED but missing libs.
find_package_handle_standard_args(Brotli
	FOUND_VAR
		Brotli_FOUND
	REQUIRED_VARS
		Brotli_INCLUDE_DIR
		${_brotli_req_vars}
	HANDLE_COMPONENTS
)

# Restore the original find library ordering
if(BROTLI_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_brotli_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()
