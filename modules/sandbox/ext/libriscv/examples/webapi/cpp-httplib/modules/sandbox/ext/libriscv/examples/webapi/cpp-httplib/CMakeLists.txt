#[[
	Build options:
	* BUILD_SHARED_LIBS (default off) builds as a shared library (if HTTPLIB_COMPILE is ON)
	* HTTPLIB_USE_OPENSSL_IF_AVAILABLE (default on)
	* HTTPLIB_USE_ZLIB_IF_AVAILABLE (default on)
	* HTTPLIB_USE_BROTLI_IF_AVAILABLE (default on)
	* HTTPLIB_USE_ZSTD_IF_AVAILABLE (default on)
	* HTTPLIB_REQUIRE_OPENSSL (default off)
	* HTTPLIB_REQUIRE_ZLIB (default off)
	* HTTPLIB_REQUIRE_BROTLI (default off)
	* HTTPLIB_REQUIRE_ZSTD (default off)
	* HTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN (default on)
	* HTTPLIB_USE_NON_BLOCKING_GETADDRINFO (default on)
	* HTTPLIB_COMPILE (default off)
	* HTTPLIB_INSTALL (default on)
	* HTTPLIB_TEST (default off)
	* BROTLI_USE_STATIC_LIBS - tells Cmake to use the static Brotli libs (only works if you have them installed).
	* OPENSSL_USE_STATIC_LIBS - tells Cmake to use the static OpenSSL libs (only works if you have them installed).

	-------------------------------------------------------------------------------

	After installation with Cmake, a find_package(httplib COMPONENTS OpenSSL ZLIB Brotli zstd) is available.
	This creates a httplib::httplib target (if found and if listed components are supported).
	It can be linked like so:

	target_link_libraries(your_exe httplib::httplib)

	The following will build & install for later use.

	Linux/macOS:

	mkdir -p build
	cd build
	cmake -DCMAKE_BUILD_TYPE=Release ..
	sudo cmake --build . --target install

	Windows:

	mkdir build
	cd build
	cmake ..
	runas /user:Administrator "cmake --build . --config Release --target install"

	-------------------------------------------------------------------------------

	These variables are available after you run find_package(httplib)
	* HTTPLIB_HEADER_PATH - this is the full path to the installed header (e.g. /usr/include/httplib.h).
	* HTTPLIB_IS_USING_OPENSSL - a bool for if OpenSSL support is enabled.
	* HTTPLIB_IS_USING_ZLIB - a bool for if ZLIB support is enabled.
	* HTTPLIB_IS_USING_BROTLI - a bool for if Brotli support is enabled.
	* HTTPLIB_IS_USING_ZSTD - a bool for if ZSTD support is enabled.
	* HTTPLIB_IS_USING_CERTS_FROM_MACOSX_KEYCHAIN - a bool for if support of loading system certs from the Apple Keychain is enabled.
	* HTTPLIB_IS_USING_NON_BLOCKING_GETADDRINFO - a bool for if nonblocking getaddrinfo is enabled.
	* HTTPLIB_IS_COMPILED - a bool for if the library is compiled, or otherwise header-only.
	* HTTPLIB_INCLUDE_DIR - the root path to httplib's header (e.g. /usr/include).
	* HTTPLIB_LIBRARY - the full path to the library if compiled (e.g. /usr/lib/libhttplib.so).
	* httplib_VERSION or HTTPLIB_VERSION - the project's version string.
	* HTTPLIB_FOUND - a bool for if the target was found.

	Want to use precompiled headers (Cmake feature since v3.16)?
	It's as simple as doing the following (before linking):

	target_precompile_headers(httplib::httplib INTERFACE "${HTTPLIB_HEADER_PATH}")

	-------------------------------------------------------------------------------

	ARCH_INDEPENDENT option of write_basic_package_version_file() requires Cmake v3.14
]]
cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)

# Get the CPPHTTPLIB_VERSION value and use it as a version
# This gets the string with the CPPHTTPLIB_VERSION value from the header.
# This is so the maintainer doesn't actually need to update this manually.
file(STRINGS httplib.h _raw_version_string REGEX "CPPHTTPLIB_VERSION \"([0-9]+\\.[0-9]+\\.[0-9]+)\"")

# Extracts just the version string itself from the whole string contained in _raw_version_string
# since _raw_version_string would contain the entire line of code where it found the version string
string(REGEX MATCH "([0-9]+\\.?)+" _httplib_version "${_raw_version_string}")

project(httplib
	VERSION ${_httplib_version}
	LANGUAGES CXX
	DESCRIPTION "A C++ header-only HTTP/HTTPS server and client library."
	HOMEPAGE_URL "https://github.com/yhirose/cpp-httplib"
)

# Change as needed to set an OpenSSL minimum version.
# This is used in the installed Cmake config file.
set(_HTTPLIB_OPENSSL_MIN_VER "3.0.0")

# Lets you disable C++ exception during CMake configure time.
# The value is used in the install CMake config file.
option(HTTPLIB_NO_EXCEPTIONS "Disable the use of C++ exceptions" OFF)
# Allow for a build to require OpenSSL to pass, instead of just being optional
option(HTTPLIB_REQUIRE_OPENSSL "Requires OpenSSL to be found & linked, or fails build." OFF)
option(HTTPLIB_REQUIRE_ZLIB "Requires ZLIB to be found & linked, or fails build." OFF)
# Allow for a build to casually enable OpenSSL/ZLIB support, but silently continue if not found.
# Make these options so their automatic use can be specifically disabled (as needed)
option(HTTPLIB_USE_OPENSSL_IF_AVAILABLE "Uses OpenSSL (if available) to enable HTTPS support." ON)
option(HTTPLIB_USE_ZLIB_IF_AVAILABLE "Uses ZLIB (if available) to enable Zlib compression support." ON)
# Lets you compile the program as a regular library instead of header-only
option(HTTPLIB_COMPILE "If ON, uses a Python script to split the header into a compilable header & source file (requires Python v3)." OFF)
# Lets you disable the installation (useful when fetched from another CMake project)
option(HTTPLIB_INSTALL "Enables the installation target" ON)
option(HTTPLIB_TEST "Enables testing and builds tests" OFF)
option(HTTPLIB_REQUIRE_BROTLI "Requires Brotli to be found & linked, or fails build." OFF)
option(HTTPLIB_USE_BROTLI_IF_AVAILABLE "Uses Brotli (if available) to enable Brotli decompression support." ON)
option(HTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN "Enable feature to load system certs from the Apple Keychain." ON)
option(HTTPLIB_USE_NON_BLOCKING_GETADDRINFO "Enables the non-blocking alternatives for getaddrinfo." ON)
option(HTTPLIB_REQUIRE_ZSTD "Requires ZSTD to be found & linked, or fails build." OFF)
option(HTTPLIB_USE_ZSTD_IF_AVAILABLE "Uses ZSTD (if available) to enable zstd support." ON)
# Defaults to static library
option(BUILD_SHARED_LIBS "Build the library as a shared library instead of static. Has no effect if using header-only." OFF)
if(BUILD_SHARED_LIBS AND WIN32 AND HTTPLIB_COMPILE)
	# Necessary for Windows if building shared libs
	# See https://stackoverflow.com/a/40743080
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows" AND ${CMAKE_SYSTEM_VERSION} VERSION_LESS "10.0.0")
	message(SEND_ERROR "Windows ${CMAKE_SYSTEM_VERSION} or lower is not supported. Please use Windows 10 or later.")
endif()
if(CMAKE_SIZEOF_VOID_P LESS 8)
	message(WARNING "Pointer size ${CMAKE_SIZEOF_VOID_P} is not supported. Please use a 64-bit compiler.")
endif()

# Set some variables that are used in-tree and while building based on our options
set(HTTPLIB_IS_COMPILED ${HTTPLIB_COMPILE})
set(HTTPLIB_IS_USING_CERTS_FROM_MACOSX_KEYCHAIN ${HTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN})
set(HTTPLIB_IS_USING_NON_BLOCKING_GETADDRINFO ${HTTPLIB_USE_NON_BLOCKING_GETADDRINFO})

# Threads needed for <thread> on some systems, and for <pthread.h> on Linux
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
# Since Cmake v3.11, Crypto & SSL became optional when not specified as COMPONENTS.
if(HTTPLIB_REQUIRE_OPENSSL)
	find_package(OpenSSL ${_HTTPLIB_OPENSSL_MIN_VER} COMPONENTS Crypto SSL REQUIRED)
	set(HTTPLIB_IS_USING_OPENSSL TRUE)
elseif(HTTPLIB_USE_OPENSSL_IF_AVAILABLE)
	find_package(OpenSSL ${_HTTPLIB_OPENSSL_MIN_VER} COMPONENTS Crypto SSL QUIET)
	# Avoid a rare circumstance of not finding all components but the end-user did their
	# own call for OpenSSL, which might trick us into thinking we'd otherwise have what we wanted
	if (TARGET OpenSSL::SSL AND TARGET OpenSSL::Crypto)
		set(HTTPLIB_IS_USING_OPENSSL ${OPENSSL_FOUND})
	else()
		set(HTTPLIB_IS_USING_OPENSSL FALSE)
	endif()
endif()

if(HTTPLIB_REQUIRE_ZLIB)
	find_package(ZLIB REQUIRED)
	set(HTTPLIB_IS_USING_ZLIB TRUE)
elseif(HTTPLIB_USE_ZLIB_IF_AVAILABLE)
	find_package(ZLIB QUIET)
	# FindZLIB doesn't have a ZLIB_FOUND variable, so check the target.
	if(TARGET ZLIB::ZLIB)
		set(HTTPLIB_IS_USING_ZLIB TRUE)
	endif()
endif()

# Adds our cmake folder to the search path for find_package
# This is so we can use our custom FindBrotli.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(HTTPLIB_REQUIRE_BROTLI)
	find_package(Brotli COMPONENTS encoder decoder common REQUIRED)
	set(HTTPLIB_IS_USING_BROTLI TRUE)
elseif(HTTPLIB_USE_BROTLI_IF_AVAILABLE)
	find_package(Brotli COMPONENTS encoder decoder common QUIET)
	set(HTTPLIB_IS_USING_BROTLI ${Brotli_FOUND})
endif()

if(HTTPLIB_REQUIRE_ZSTD)
	find_package(zstd)
	if(NOT zstd_FOUND)
		find_package(PkgConfig REQUIRED)
		pkg_check_modules(zstd REQUIRED IMPORTED_TARGET libzstd)
		add_library(zstd::libzstd ALIAS PkgConfig::zstd)
	endif()
	set(HTTPLIB_IS_USING_ZSTD TRUE)
elseif(HTTPLIB_USE_ZSTD_IF_AVAILABLE)
	find_package(zstd QUIET)
	if(NOT zstd_FOUND)
		find_package(PkgConfig QUIET)
		if(PKG_CONFIG_FOUND)
			pkg_check_modules(zstd QUIET IMPORTED_TARGET libzstd)

			if(TARGET PkgConfig::zstd)
				add_library(zstd::libzstd ALIAS PkgConfig::zstd)
			endif()
		endif()
	endif()
	# Both find_package and PkgConf set a XXX_FOUND var
	set(HTTPLIB_IS_USING_ZSTD ${zstd_FOUND})
endif()

# Used for default, common dirs that the end-user can change (if needed)
# like CMAKE_INSTALL_INCLUDEDIR or CMAKE_INSTALL_DATADIR
include(GNUInstallDirs)

if(HTTPLIB_COMPILE)
	# Put the split script into the build dir
	configure_file(split.py "${CMAKE_CURRENT_BINARY_DIR}/split.py"
		COPYONLY
	)
	# Needs to be in the same dir as the python script
	configure_file(httplib.h "${CMAKE_CURRENT_BINARY_DIR}/httplib.h"
		COPYONLY
	)

	# Used outside of this if-else
	set(_INTERFACE_OR_PUBLIC PUBLIC)
	# Brings in the Python3_EXECUTABLE path we can use.
	find_package(Python3 REQUIRED)
	# Actually split the file
	# Keeps the output in the build dir to not pollute the main dir
	execute_process(COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/split.py"
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		ERROR_VARIABLE _httplib_split_error
	)
	if(_httplib_split_error)
		message(FATAL_ERROR "Failed when trying to split cpp-httplib with the Python script.\n${_httplib_split_error}")
	endif()

	# split.py puts output in "out"
	set(_httplib_build_includedir "${CMAKE_CURRENT_BINARY_DIR}/out")
	# This will automatically be either static or shared based on the value of BUILD_SHARED_LIBS
	add_library(${PROJECT_NAME} "${_httplib_build_includedir}/httplib.cc")
	target_sources(${PROJECT_NAME}
		PUBLIC
			$<BUILD_INTERFACE:${_httplib_build_includedir}/httplib.h>
			$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/httplib.h>
	)
	set_target_properties(${PROJECT_NAME}
		PROPERTIES
			VERSION ${${PROJECT_NAME}_VERSION}
			SOVERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}"
			OUTPUT_NAME cpp-httplib
	)
else()
	# This is for header-only.
	set(_INTERFACE_OR_PUBLIC INTERFACE)
	add_library(${PROJECT_NAME} INTERFACE)
	set(_httplib_build_includedir "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
# Lets you address the target with httplib::httplib
# Only useful if building in-tree, versus using it from an installation.
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Require C++11
target_compile_features(${PROJECT_NAME} ${_INTERFACE_OR_PUBLIC} cxx_std_11)

target_include_directories(${PROJECT_NAME} SYSTEM ${_INTERFACE_OR_PUBLIC}
	$<BUILD_INTERFACE:${_httplib_build_includedir}>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Always require threads
target_link_libraries(${PROJECT_NAME} ${_INTERFACE_OR_PUBLIC}
		Threads::Threads
		# Needed for Windows libs on Mingw, as the pragma comment(lib, "xyz") aren't triggered.
		$<$<PLATFORM_ID:Windows>:ws2_32>
		$<$<PLATFORM_ID:Windows>:crypt32>
		# Needed for API from MacOS Security framework
		"$<$<AND:$<PLATFORM_ID:Darwin>,$<BOOL:${HTTPLIB_IS_USING_OPENSSL}>,$<BOOL:${HTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN}>>:-framework CoreFoundation -framework Security>"
		# Can't put multiple targets in a single generator expression or it bugs out.
		$<$<BOOL:${HTTPLIB_IS_USING_BROTLI}>:Brotli::common>
		$<$<BOOL:${HTTPLIB_IS_USING_BROTLI}>:Brotli::encoder>
		$<$<BOOL:${HTTPLIB_IS_USING_BROTLI}>:Brotli::decoder>
		$<$<BOOL:${HTTPLIB_IS_USING_ZLIB}>:ZLIB::ZLIB>
		$<$<BOOL:${HTTPLIB_IS_USING_ZSTD}>:zstd::libzstd>
		$<$<BOOL:${HTTPLIB_IS_USING_OPENSSL}>:OpenSSL::SSL>
		$<$<BOOL:${HTTPLIB_IS_USING_OPENSSL}>:OpenSSL::Crypto>
)

# Set the definitions to enable optional features
target_compile_definitions(${PROJECT_NAME} ${_INTERFACE_OR_PUBLIC}
	$<$<BOOL:${HTTPLIB_NO_EXCEPTIONS}>:CPPHTTPLIB_NO_EXCEPTIONS>
	$<$<BOOL:${HTTPLIB_IS_USING_BROTLI}>:CPPHTTPLIB_BROTLI_SUPPORT>
	$<$<BOOL:${HTTPLIB_IS_USING_ZLIB}>:CPPHTTPLIB_ZLIB_SUPPORT>
	$<$<BOOL:${HTTPLIB_IS_USING_ZSTD}>:CPPHTTPLIB_ZSTD_SUPPORT>
	$<$<BOOL:${HTTPLIB_IS_USING_OPENSSL}>:CPPHTTPLIB_OPENSSL_SUPPORT>
	$<$<AND:$<PLATFORM_ID:Darwin>,$<BOOL:${HTTPLIB_IS_USING_OPENSSL}>,$<BOOL:${HTTPLIB_IS_USING_CERTS_FROM_MACOSX_KEYCHAIN}>>:CPPHTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN>
	$<$<BOOL:${HTTPLIB_USE_NON_BLOCKING_GETADDRINFO}>:CPPHTTPLIB_USE_NON_BLOCKING_GETADDRINFO>
)

# CMake configuration files installation directory
set(_TARGET_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

include(CMakePackageConfigHelpers)

# Configures the meta-file httplibConfig.cmake.in to replace variables with paths/values/etc.
configure_package_config_file("cmake/${PROJECT_NAME}Config.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
	INSTALL_DESTINATION "${_TARGET_INSTALL_CMAKEDIR}"
	# Passes the includedir install path
	PATH_VARS CMAKE_INSTALL_FULL_INCLUDEDIR
)

if(HTTPLIB_COMPILE)
	write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
		# Example: if you find_package(httplib 0.5.4)
		# then anything >= 0.5.4 and < 0.6 is accepted
		COMPATIBILITY SameMinorVersion
	)
else()
	write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
		# Example: if you find_package(httplib 0.5.4)
		# then anything >= 0.5.4 and < 0.6 is accepted
		COMPATIBILITY SameMinorVersion
		# Tells Cmake that it's a header-only lib
		# Mildly useful for end-users :)
		ARCH_INDEPENDENT
	)
endif()

if(HTTPLIB_INSTALL)
	# Creates the export httplibTargets.cmake
	# This is strictly what holds compilation requirements
	# and linkage information (doesn't find deps though).
	install(TARGETS ${PROJECT_NAME} EXPORT httplibTargets)

	install(FILES "${_httplib_build_includedir}/httplib.h" TYPE INCLUDE)

	install(FILES
		"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
		"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
		# Install it so it can be used later by the httplibConfig.cmake file.
		# Put it in the same dir as our config file instead of a global path so we don't potentially stomp on other packages.
		"${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindBrotli.cmake"
		DESTINATION ${_TARGET_INSTALL_CMAKEDIR}
	)

	# NOTE: This path changes depending on if it's on Windows or Linux
	install(EXPORT httplibTargets
		# Puts the targets into the httplib namespace
		# So this makes httplib::httplib linkable after doing find_package(httplib)
		NAMESPACE ${PROJECT_NAME}::
		DESTINATION ${_TARGET_INSTALL_CMAKEDIR}
	)

	# Install documentation & license
	# ex: /usr/share/doc/httplib/README.md and /usr/share/licenses/httplib/LICENSE
	install(FILES "README.md" DESTINATION "${CMAKE_INSTALL_DOCDIR}")
	install(FILES "LICENSE" DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/${PROJECT_NAME}")

	include(CPack)
endif()

if(HTTPLIB_TEST)
	include(CTest)
	add_subdirectory(test)
endif()
