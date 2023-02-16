// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


// Pull in standard 8-bit, 16-bit, 32-bit and 64-bit types.
#if PSD_USE_MSVC && PSD_USE_MSVC_VER <= 2008
	// VS2008 does not ship with the stdint.h header.
	typedef signed __int8		int8_t;
	typedef signed __int16		int16_t;
	typedef signed __int32		int32_t;
	typedef unsigned __int8		uint8_t;
	typedef unsigned __int16	uint16_t;
	typedef unsigned __int32	uint32_t;
	typedef signed __int64		int64_t;
	typedef unsigned __int64	uint64_t;

	#ifndef SIZE_MAX
		#ifdef _WIN64
			#define SIZE_MAX  _UI64_MAX
		#else
			#define SIZE_MAX  _UI32_MAX
		#endif
	#endif
#else
	PSD_PUSH_WARNING_LEVEL(0)
	#include <stdint.h>
    #if defined(__APPLE__)
        #include <assert.h>
    #endif
	PSD_POP_WARNING_LEVEL
#endif
