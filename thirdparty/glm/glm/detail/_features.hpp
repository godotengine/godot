#pragma once

// #define GLM_CXX98_EXCEPTIONS
// #define GLM_CXX98_RTTI

// #define GLM_CXX11_RVALUE_REFERENCES
// Rvalue references - GCC 4.3
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2118.html

// GLM_CXX11_TRAILING_RETURN
// Rvalue references for *this - GCC not supported
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2439.htm

// GLM_CXX11_NONSTATIC_MEMBER_INIT
// Initialization of class objects by rvalues - GCC any
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1610.html

// GLM_CXX11_NONSTATIC_MEMBER_INIT
// Non-static data member initializers - GCC 4.7
// http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2008/n2756.htm

// #define GLM_CXX11_VARIADIC_TEMPLATE
// Variadic templates - GCC 4.3
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf

//
// Extending variadic template template parameters - GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2555.pdf

// #define GLM_CXX11_GENERALIZED_INITIALIZERS
// Initializer lists - GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm

// #define GLM_CXX11_STATIC_ASSERT
// Static assertions - GCC 4.3
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1720.html

// #define GLM_CXX11_AUTO_TYPE
// auto-typed variables - GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1984.pdf

// #define GLM_CXX11_AUTO_TYPE
// Multi-declarator auto - GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1737.pdf

// #define GLM_CXX11_AUTO_TYPE
// Removal of auto as a storage-class specifier - GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2546.htm

// #define GLM_CXX11_AUTO_TYPE
// New function declarator syntax - GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2541.htm

// #define GLM_CXX11_LAMBDAS
// New wording for C++0x lambdas - GCC 4.5
// http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2927.pdf

// #define GLM_CXX11_DECLTYPE
// Declared type of an expression - GCC 4.3
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2343.pdf

//
// Right angle brackets - GCC 4.3
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html

//
// Default template arguments for function templates	DR226	GCC 4.3
// http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#226

//
// Solving the SFINAE problem for expressions	DR339	GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html

// #define GLM_CXX11_ALIAS_TEMPLATE
// Template aliases	N2258	GCC 4.7
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf

//
// Extern templates	N1987	Yes
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1987.htm

// #define GLM_CXX11_NULLPTR
// Null pointer constant	N2431	GCC 4.6
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2431.pdf

// #define GLM_CXX11_STRONG_ENUMS
// Strongly-typed enums	N2347	GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf

//
// Forward declarations for enums	N2764	GCC 4.6
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf

//
// Generalized attributes	N2761	GCC 4.8
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2761.pdf

//
// Generalized constant expressions	N2235	GCC 4.6
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2235.pdf

//
// Alignment support	N2341	GCC 4.8
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2341.pdf

// #define GLM_CXX11_DELEGATING_CONSTRUCTORS
// Delegating constructors	N1986	GCC 4.7
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1986.pdf

//
// Inheriting constructors	N2540	GCC 4.8
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2540.htm

// #define GLM_CXX11_EXPLICIT_CONVERSIONS
// Explicit conversion operators	N2437	GCC 4.5
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf

//
// New character types	N2249	GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2249.html

//
// Unicode string literals	N2442	GCC 4.5
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm

//
// Raw string literals	N2442	GCC 4.5
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm

//
// Universal character name literals	N2170	GCC 4.5
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2170.html

// #define GLM_CXX11_USER_LITERALS
// User-defined literals		N2765	GCC 4.7
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2765.pdf

//
// Standard Layout Types	N2342	GCC 4.5
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2342.htm

// #define GLM_CXX11_DEFAULTED_FUNCTIONS
// #define GLM_CXX11_DELETED_FUNCTIONS
// Defaulted and deleted functions	N2346	GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm

//
// Extended friend declarations	N1791	GCC 4.7
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1791.pdf

//
// Extending sizeof	N2253	GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2253.html

// #define GLM_CXX11_INLINE_NAMESPACES
// Inline namespaces	N2535	GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2535.htm

// #define GLM_CXX11_UNRESTRICTED_UNIONS
// Unrestricted unions	N2544	GCC 4.6
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2544.pdf

// #define GLM_CXX11_LOCAL_TYPE_TEMPLATE_ARGS
// Local and unnamed types as template arguments	N2657	GCC 4.5
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm

// #define GLM_CXX11_RANGE_FOR
// Range-based for	N2930	GCC 4.6
// http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2930.html

// #define GLM_CXX11_OVERRIDE_CONTROL
// Explicit virtual overrides	N2928 N3206 N3272	GCC 4.7
// http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2928.htm
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3206.htm
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3272.htm

//
// Minimal support for garbage collection and reachability-based leak detection	N2670	No
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2670.htm

// #define GLM_CXX11_NOEXCEPT
// Allowing move constructors to throw [noexcept]	N3050	GCC 4.6 (core language only)
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html

//
// Defining move special member functions	N3053	GCC 4.6
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3053.html

//
// Sequence points	N2239	Yes
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2239.html

//
// Atomic operations	N2427	GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2239.html

//
// Strong Compare and Exchange	N2748	GCC 4.5
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2427.html

//
// Bidirectional Fences	N2752	GCC 4.8
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2752.htm

//
// Memory model	N2429	GCC 4.8
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2429.htm

//
// Data-dependency ordering: atomics and memory model	N2664	GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2664.htm

//
// Propagating exceptions	N2179	GCC 4.4
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2179.html

//
// Abandoning a process and at_quick_exit	N2440	GCC 4.8
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2440.htm

//
// Allow atomics use in signal handlers	N2547	Yes
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2547.htm

//
// Thread-local storage	N2659	GCC 4.8
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm

//
// Dynamic initialization and destruction with concurrency	N2660	GCC 4.3
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm

//
// __func__ predefined identifier	N2340	GCC 4.3
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2340.htm

//
// C99 preprocessor	N1653	GCC 4.3
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1653.htm

//
// long long	N1811	GCC 4.3
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1811.pdf

//
// Extended integral types	N1988	Yes
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1988.pdf

#if(GLM_COMPILER & GLM_COMPILER_GCC)

#	define GLM_CXX11_STATIC_ASSERT

#elif(GLM_COMPILER & GLM_COMPILER_CLANG)
#	if(__has_feature(cxx_exceptions))
#		define GLM_CXX98_EXCEPTIONS
#	endif

#	if(__has_feature(cxx_rtti))
#		define GLM_CXX98_RTTI
#	endif

#	if(__has_feature(cxx_access_control_sfinae))
#		define GLM_CXX11_ACCESS_CONTROL_SFINAE
#	endif

#	if(__has_feature(cxx_alias_templates))
#		define GLM_CXX11_ALIAS_TEMPLATE
#	endif

#	if(__has_feature(cxx_alignas))
#		define GLM_CXX11_ALIGNAS
#	endif

#	if(__has_feature(cxx_attributes))
#		define GLM_CXX11_ATTRIBUTES
#	endif

#	if(__has_feature(cxx_constexpr))
#		define GLM_CXX11_CONSTEXPR
#	endif

#	if(__has_feature(cxx_decltype))
#		define GLM_CXX11_DECLTYPE
#	endif

#	if(__has_feature(cxx_default_function_template_args))
#		define GLM_CXX11_DEFAULT_FUNCTION_TEMPLATE_ARGS
#	endif

#	if(__has_feature(cxx_defaulted_functions))
#		define GLM_CXX11_DEFAULTED_FUNCTIONS
#	endif

#	if(__has_feature(cxx_delegating_constructors))
#		define GLM_CXX11_DELEGATING_CONSTRUCTORS
#	endif

#	if(__has_feature(cxx_deleted_functions))
#		define GLM_CXX11_DELETED_FUNCTIONS
#	endif

#	if(__has_feature(cxx_explicit_conversions))
#		define GLM_CXX11_EXPLICIT_CONVERSIONS
#	endif

#	if(__has_feature(cxx_generalized_initializers))
#		define GLM_CXX11_GENERALIZED_INITIALIZERS
#	endif

#	if(__has_feature(cxx_implicit_moves))
#		define GLM_CXX11_IMPLICIT_MOVES
#	endif

#	if(__has_feature(cxx_inheriting_constructors))
#		define GLM_CXX11_INHERITING_CONSTRUCTORS
#	endif

#	if(__has_feature(cxx_inline_namespaces))
#		define GLM_CXX11_INLINE_NAMESPACES
#	endif

#	if(__has_feature(cxx_lambdas))
#		define GLM_CXX11_LAMBDAS
#	endif

#	if(__has_feature(cxx_local_type_template_args))
#		define GLM_CXX11_LOCAL_TYPE_TEMPLATE_ARGS
#	endif

#	if(__has_feature(cxx_noexcept))
#		define GLM_CXX11_NOEXCEPT
#	endif

#	if(__has_feature(cxx_nonstatic_member_init))
#		define GLM_CXX11_NONSTATIC_MEMBER_INIT
#	endif

#	if(__has_feature(cxx_nullptr))
#		define GLM_CXX11_NULLPTR
#	endif

#	if(__has_feature(cxx_override_control))
#		define GLM_CXX11_OVERRIDE_CONTROL
#	endif

#	if(__has_feature(cxx_reference_qualified_functions))
#		define GLM_CXX11_REFERENCE_QUALIFIED_FUNCTIONS
#	endif

#	if(__has_feature(cxx_range_for))
#		define GLM_CXX11_RANGE_FOR
#	endif

#	if(__has_feature(cxx_raw_string_literals))
#		define GLM_CXX11_RAW_STRING_LITERALS
#	endif

#	if(__has_feature(cxx_rvalue_references))
#		define GLM_CXX11_RVALUE_REFERENCES
#	endif

#	if(__has_feature(cxx_static_assert))
#		define GLM_CXX11_STATIC_ASSERT
#	endif

#	if(__has_feature(cxx_auto_type))
#		define GLM_CXX11_AUTO_TYPE
#	endif

#	if(__has_feature(cxx_strong_enums))
#		define GLM_CXX11_STRONG_ENUMS
#	endif

#	if(__has_feature(cxx_trailing_return))
#		define GLM_CXX11_TRAILING_RETURN
#	endif

#	if(__has_feature(cxx_unicode_literals))
#		define GLM_CXX11_UNICODE_LITERALS
#	endif

#	if(__has_feature(cxx_unrestricted_unions))
#		define GLM_CXX11_UNRESTRICTED_UNIONS
#	endif

#	if(__has_feature(cxx_user_literals))
#		define GLM_CXX11_USER_LITERALS
#	endif

#	if(__has_feature(cxx_variadic_templates))
#		define GLM_CXX11_VARIADIC_TEMPLATES
#	endif

#endif//(GLM_COMPILER & GLM_COMPILER_CLANG)
