include_guard()

if(NOT APPLE AND NOT ANDROID)
	set(GDJ_STATIC_RUNTIME_LIBRARY TRUE
		CACHE BOOL
		"Use static C++ runtime library."
	)
else()
	set(GDJ_STATIC_RUNTIME_LIBRARY FALSE)
endif()

if(NOT IOS AND NOT ANDROID)
	set(GDJ_USE_MIMALLOC TRUE
		CACHE BOOL
		"Use mimalloc as the general-purpose memory allocator."
	)
else()
	set(GDJ_USE_MIMALLOC FALSE)
endif()

set(GDJ_INTERPROCEDURAL_OPTIMIZATION TRUE
	CACHE BOOL
	"Enable interprocedural optimizations for any optimized builds."
)

set(GDJ_PRECOMPILE_HEADERS TRUE
	CACHE BOOL
	"Precompile header files that don't change often, like external ones."
)

set(GDJ_DOUBLE_PRECISION FALSE
	CACHE BOOL
	"Compile with 64-bit floating-point precision."
)

if(NOT APPLE AND NOT ANDROID)
	set(GDJ_X86_INSTRUCTION_SET SSE2
		CACHE STRING
		"Minimum required CPU instruction set when compiling for x86."
	)

	set(x86_instruction_sets NONE SSE2 AVX AVX2 AVX512)
	set_property(CACHE GDJ_X86_INSTRUCTION_SET PROPERTY STRINGS ${x86_instruction_sets})

	if(NOT GDJ_X86_INSTRUCTION_SET IN_LIST x86_instruction_sets)
		message(FATAL_ERROR "Unsupported CPU instruction set: '${GDJ_X86_INSTRUCTION_SET}'.")
	endif()
else()
	set(GDJ_X86_INSTRUCTION_SET NONE)
endif()

set(GDJ_INSTALL_DEBUG_SYMBOLS FALSE
	CACHE BOOL
	"Install debug symbols along with the binaries."
)
