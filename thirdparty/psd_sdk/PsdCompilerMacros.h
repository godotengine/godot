// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


/// \def PSD_ABSTRACT
/// \ingroup Platform
/// \brief Marks member functions as being abstract.
/// \sa PSD_OVERRIDE
#define PSD_ABSTRACT									= 0


/// \def PSD_OVERRIDE
/// \ingroup Platform
/// \brief Marks member functions as being an override of a base class virtual function.
/// \sa PSD_ABSTRACT
#if PSD_USE_MSVC && PSD_USE_MSVC_VER <= 2010
	// VS2008 and VS2010 understand override, but only as a non-standard language extension.
	#define PSD_OVERRIDE
#else
	#define PSD_OVERRIDE								override
#endif


/// \def PSD_PRAGMA
/// \ingroup Platform
/// \brief Allows to emit #pragma directives from within macros.
#if PSD_USE_MSVC
	#define PSD_PRAGMA(pragma)							__pragma(pragma)
#else
	#define PSD_PRAGMA(pragma)							_Pragma(pragma)
#endif


/// \def PSD_PUSH_WARNING_LEVEL
/// \ingroup Platform
/// \brief Pushes a certain compiler warning level.
/// \sa PSD_POP_WARNING_LEVEL
#if PSD_USE_MSVC
	#define PSD_PUSH_WARNING_LEVEL(level)				PSD_PRAGMA(warning(push, level))
#else
	#define PSD_PUSH_WARNING_LEVEL(level)
#endif


/// \def PSD_POP_WARNING_LEVEL
/// \ingroup Platform
/// \brief Pops the old compiler warning level.
/// \sa PSD_PUSH_WARNING_LEVEL
#if PSD_USE_MSVC
	#define PSD_POP_WARNING_LEVEL						PSD_PRAGMA(warning(pop))
#else
	#define PSD_POP_WARNING_LEVEL
#endif


/// \def PSD_RESTRICT
/// \ingroup Platform
/// \brief Support for the C99 \c restrict keyword.
#if PSD_USE_MSVC
	#define PSD_RESTRICT								__restrict
#else
	#define PSD_RESTRICT								__restrict__
#endif


/// \def PSD_UNUSED
/// \ingroup Platform
/// \brief Signals to the compiler that a symbol/expression is not used, and that no warning should be generated. Does \b not generate any instructions.
#define PSD_UNUSED(symbol)								(void)(symbol)


/// \def PSD_INLINE
/// \ingroup Platform
/// \brief Forces a function to be inlined.
#if PSD_USE_MSVC
	#define PSD_INLINE									__forceinline
#else
	#define PSD_INLINE									inline __attribute__((always_inline))
#endif


/// \def PSD_ALIGN_OF
/// \ingroup Platform
/// \brief Returns the alignment requirement of a given type.
#if PSD_USE_MSVC
	#define PSD_ALIGN_OF(type)							__alignof(type)
#else
	#define PSD_ALIGN_OF(type)							alignof(type)
#endif


/// \def PSD_DISABLE_WARNING
/// \ingroup Platform
/// \brief Saves the current warning status, and disables a certain compiler warning.
/// \details Compiler warnings can safely be disabled and enabled again by putting the affected code between the
/// respective macros, shown in the following example:
/// \code
///   ME_DISABLE_WARNING(4640);
///   static MyClass myInstance;
///   ME_ENABLE_WARNING(4640);
/// \endcode
/// \sa PSD_ENABLE_WARNING
#if PSD_USE_MSVC
	#define PSD_DISABLE_WARNING(number)					PSD_PRAGMA(warning(push)) PSD_PRAGMA(warning(disable:number))
#else
	#define PSD_DISABLE_WARNING(number)
#endif


/// \def PSD_ENABLE_WARNING
/// \ingroup Platform
/// \brief Enables a certain compiler warning by restoring the old warning status.
/// \sa PSD_DISABLE_WARNING
#if PSD_USE_MSVC
	#define PSD_ENABLE_WARNING(number)					PSD_PRAGMA(warning(pop))
#else
	#define PSD_ENABLE_WARNING(number)
#endif


/// \def PSD_MULTILINE_MACRO_BEGIN
/// \ingroup Platform
/// \brief Begins a multi-line macro, ensuring that the macro can be used like an ordinary function call, and properly works inside if/else clauses.
/// \details Warning 4127 is temporarily disabled because the MSVC compiler complains about the conditional expression being
/// constant (while (0)), and we want to keep the warning enabled in other cases.
/// \sa PSD_MULTILINE_MACRO_END
#define PSD_MULTILINE_MACRO_BEGIN						PSD_DISABLE_WARNING(4127) do {


/// \def PSD_MULTILINE_MACRO_END
/// \ingroup Platform
/// \brief Ends a multi-line macro, ensuring that the macro can be used like an ordinary function call, and properly works inside if/else clauses.
/// \sa PSD_MULTILINE_MACRO_BEGIN
#define PSD_MULTILINE_MACRO_END							} while (0) PSD_ENABLE_WARNING(4127)


#if PSD_USE_MSVC && PSD_USE_MSVC_VER <= 2008
	#define PSD_JOIN2(a, b)								a##b
	#define PSD_JOIN(a, b)								PSD_JOIN2(a, b)

	// VS2008 doesn't have those, so emulate them.
	#define static_assert(condition, message)			typedef char PSD_JOIN(static_assert_impl_, __LINE__)[(condition) ? 1 : -1]
	#define nullptr										NULL
#endif
