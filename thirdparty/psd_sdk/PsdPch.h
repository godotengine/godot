// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


// This file is used internally by all translation units in the PSD library.
// First work out which compiler we are using.
#if defined(__clang__)
	#define PSD_USE_CLANG 1
	#define PSD_USE_GCC 0
	#define PSD_USE_MSVC 0
#elif defined(__GNUG__)
	#define PSD_USE_CLANG 0
	#define PSD_USE_GCC 1
	#define PSD_USE_MSVC 0
#elif defined(_MSC_VER)
	#define PSD_USE_CLANG 0
	#define PSD_USE_GCC 0
	#define PSD_USE_MSVC 1

	#if _MSC_VER >= 1920
		#define PSD_USE_MSVC_VER 2019
	#elif _MSC_VER >= 1910
		#define PSD_USE_MSVC_VER 2017
	#elif _MSC_VER >= 1900
		#define PSD_USE_MSVC_VER 2015
	#elif _MSC_VER >= 1800
		#define PSD_USE_MSVC_VER 2013
	#elif _MSC_VER >= 1700
		#define PSD_USE_MSVC_VER 2012
	#elif _MSC_VER >= 1600
		#define PSD_USE_MSVC_VER 2010
	#elif _MSC_VER >= 1500
		#define PSD_USE_MSVC_VER 2008
	#else _MSC_VER < 1500
		#define PSD_USE_MSVC_VER 2005
	#endif
#endif


#if PSD_USE_CLANG
	// Clang will complain about "static_assert declarations are incompatible with C++98", even when compiling for C++11.
	// Similarly, Clang will complain about "variadic macros are incompatible with C++98".
	#pragma clang diagnostic ignored "-Wc++98-compat"
	#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"

	// Clang gets confused by our interfaces Allocator and File, and complains about
	// "has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit" which is not true -
	// all classes in question have at least one virtual method (the destructor) implemented in a .cpp.
	#pragma clang diagnostic ignored "-Wweak-vtables"
#elif PSD_USE_MSVC
	// the following warnings are informational warnings triggered by /Wall. We don't need them.
	#pragma warning(disable : 4711)		// function 'name' selected for automatic inline expansion
	#pragma warning(disable : 4514)		// 'function' : unreferenced inline function has been removed
	#pragma warning(disable : 4820)		// 'bytes' bytes padding added after construct 'member_name'
	#pragma warning(disable : 4710)		// 'function' : function not inlined
	#pragma warning(disable : 4350)		// behavior change : 'std::_Wrap_alloc<std::allocator<char>>::_Wrap_alloc(const std::_Wrap_alloc<std::allocator<char>> &)' called instead of 'std::_Wrap_alloc<std::allocator<char>>::_Wrap_alloc<std::_Wrap_alloc<std::allocator<char>>>(_Other &)'

	// VS 2017 specific warnings
	#if PSD_USE_MSVC_VER == 2017
		#pragma warning(disable : 4577)		// warning C4577: 'noexcept' used with no exception handling mode specified; termination on exception is not guaranteed. Specify /EHsc
	#endif
#endif

#include "PsdCompilerMacros.h"
#include "PsdTypes.h"
#include "PsdNamespace.h"
#include <stddef.h>
