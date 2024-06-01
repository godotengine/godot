include_guard()

include(GodotJoltExternalLibrary)

set(configurations
	Debug
	RelWithDebInfo
)

if(WIN32)
	set(output_name mimalloc-static)
else()
	set(output_name mimalloc)
endif()

set(cflags "")

if(DEFINED ENV{CFLAGS})
	set(cflags "${cflags} $ENV{CFLAGS}")
endif()

if(MSVC)
	set(cflags "${cflags} /W0")
else()
	set(cflags "${cflags} -w")
endif()

set(use_ipo $<NOT:$<CXX_COMPILER_ID:GNU>>)

gdj_add_external_library(mimalloc "${configurations}"
	GIT_REPOSITORY https://github.com/godot-jolt/mimalloc.git
	GIT_COMMIT a1b284de0aecc0ea0e8846656fd01043253b84bb
	LANGUAGE C
	OUTPUT_NAME ${output_name}
	INCLUDE_DIRECTORIES
		<SOURCE_DIR>/include
	ENVIRONMENT
		CFLAGS=${cflags}
	CMAKE_CACHE_ARGS
		-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO=${use_ipo}
		-DMI_OVERRIDE=FALSE
		-DMI_USE_CXX=FALSE
		-DMI_OSX_INTERPOSE=FALSE
		-DMI_OSX_ZONE=FALSE
		-DMI_WIN_REDIRECT=FALSE
		-DMI_BUILD_SHARED=FALSE
		-DMI_BUILD_OBJECT=FALSE
		-DMI_BUILD_TESTS=FALSE
		-DMI_SKIP_COLLECT_ON_EXIT=TRUE
	LIBRARY_CONFIG_DEBUG Debug
	LIBRARY_CONFIG_DEVELOPMENT RelWithDebInfo
	LIBRARY_CONFIG_DISTRIBUTION RelWithDebInfo
	LIBRARY_CONFIG_EDITORDEBUG Debug
	LIBRARY_CONFIG_EDITORDEVELOPMENT RelWithDebInfo
	LIBRARY_CONFIG_EDITORDISTRIBUTION RelWithDebInfo
)
