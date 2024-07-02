// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include <assert.h>
#include <cstdio>


/// \def PSD_ENABLE_ASSERTIONS 
/// \ingroup Platform
/// Enables/disables the use of PSD_ASSERT* macros. If disabled, the macros will not generate any instructions.
/// \sa PSD_ASSERT PSD_ASSERT_NOT_NULL
#define PSD_ENABLE_ASSERTIONS 1


/// \def PSD_ASSERT
/// \ingroup Platform
/// \brief Custom assertion macro.
/// \details This macro allows asserting that a certain condition holds true, and allows outputting a formatted message
/// to the user (along with additional information) in case the condition does not hold.
///
/// Note that in its default implementation, the macro simply uses the standard assert() macro. However, it can easily be changed
/// into a more sophisticated implementation, e.g. by using the approach described here:
/// http://www.altdevblogaday.com/2011/10/12/upgrading-assert-using-the-preprocessor/.
///
/// \remark Code generation is enabled/disabled via the preprocessor option \ref PSD_ENABLE_ASSERTIONS. If disabled,
/// a call to \ref PSD_ASSERT will not generate any instructions, reducing the executable's size and generally improving
/// performance. It is recommended to disable assertions in retail builds.
/// \sa PSD_ENABLE_ASSERTIONS PSD_ASSERT_NOT_NULL


/// \def PSD_ASSERT_NOT_NULL
/// \ingroup Platform
/// \brief Asserts that a given pointer is not null.
/// \details This macro is a convenience macro that asserts that a given pointer is not null, and uses the assertion
/// facility internally.
/// \remark Code generation is enabled/disabled via the preprocessor option \ref PSD_ENABLE_ASSERTIONS. If disabled,
/// a call to \ref PSD_ASSERT_NOT_NULL will not generate any instructions, reducing the executable's size and generally
/// improving performance. It is recommended to disable assertions in retail builds.
/// \sa PSD_ENABLE_ASSERTIONS PSD_ASSERT

#if PSD_ENABLE_ASSERTIONS
	#define PSD_ASSERT(condition, ...)			PSD_MULTILINE_MACRO_BEGIN (condition) ? true : printf("\n***ASSERT FAILED*** " __VA_ARGS__); assert(condition); PSD_MULTILINE_MACRO_END
#else
	#define PSD_ASSERT(condition, ...)			PSD_UNUSED(condition)
#endif

#define PSD_ASSERT_NOT_NULL(ptr)				PSD_ASSERT(ptr != nullptr, "Pointer is null.")
