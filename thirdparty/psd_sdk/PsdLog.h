// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include <cstdio>


/// \def PSD_ENABLE_LOGGING 
/// \ingroup Platform
/// Enables/disables the use of the \ref PSD_WARNING and \ref PSD_ERROR macros. If disabled, these macros will not generate any instructions.
/// \sa PSD_WARNING PSD_ERROR
#define PSD_ENABLE_LOGGING 1


/// \def PSD_WARNING
/// \ingroup Platform
/// \brief Custom macro for emitting a warning at run-time.
/// \details This macro is used to emit a warning in cases where some condition should not occur in a correct program, but
/// the program can handle the condition and carry on.
/// \remark Code generation is enabled/disabled via the preprocessor option \ref PSD_ENABLE_LOGGING. If disabled,
/// a call to \ref PSD_WARNING will not generate any instructions, reducing the executable's size and generally improving
/// performance. It is recommended to disable logging in retail builds.
/// \sa PSD_ENABLE_LOGGING PSD_ERROR


/// \def PSD_ERROR
/// \ingroup Platform
/// \brief Custom macro for emitting an error at run-time.
/// \details This macro is used to emit an error in cases where a serious error condition is met, but
/// the program can potentially still carry on.
/// \remark Code generation is enabled/disabled via the preprocessor option \ref PSD_ENABLE_LOGGING. If disabled,
/// a call to \ref PSD_ERROR will not generate any instructions, reducing the executable's size and generally improving
/// performance. It is recommended to disable logging in retail builds.
/// \sa PSD_ENABLE_LOGGING PSD_WARNING

#if PSD_ENABLE_LOGGING
	#define PSD_WARNING(channel, ...)		printf("\n***WARNING*** " "[" channel "] " __VA_ARGS__)
	#define PSD_ERROR(channel, ...)			printf("\n***ERROR*** " "[" channel "] " __VA_ARGS__)
#else
	#define PSD_WARNING(channel, ...)		PSD_UNUSED(channel)
	#define PSD_ERROR(channel, ...)			PSD_UNUSED(channel)
#endif
