// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


#if PSD_USE_MSVC && PSD_USE_MSVC_VER <= 2012
	// VS2008, VS2010 and VS2012 don't provide inttypes.h
	#define PRIu64 "I64u"
	#define PRId64 "I64d"
#else
	#include <inttypes.h>
#endif
