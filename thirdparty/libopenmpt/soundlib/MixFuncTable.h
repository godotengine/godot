/*
 * MixFuncTable.h
 * --------------
 * Purpose: Table containing all mixer functions.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "MixerInterface.h"

OPENMPT_NAMESPACE_BEGIN

namespace MixFuncTable
{
	// Table index:
	//	[b1-b0]	format (8-bit-mono, 16-bit-mono, 8-bit-stereo, 16-bit-stereo)
	//	[b2]	ramp
	//	[b3]	filter
	//	[b6-b4]	src type

	// Sample type / processing type index
	enum FunctionIndex
	{
		ndx16Bit		= 0x01,
		ndxStereo		= 0x02,
		ndxRamp			= 0x04,
		ndxFilter		= 0x08,
	};

	// SRC index
	enum ResamplingIndex
	{
		ndxNoInterpolation	= 0x00,
		ndxLinear			= 0x10,
		ndxFastSinc			= 0x20,
		ndxKaiser			= 0x30,
		ndxFIRFilter		= 0x40,
		ndxAmigaBlep		= 0x50,
	};

	extern const MixFuncInterface Functions[6 * 16];

	ResamplingIndex ResamplingModeToMixFlags(ResamplingMode resamplingMode);
}

OPENMPT_NAMESPACE_END
