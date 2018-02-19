/*
 * MixFuncTable.cpp
 * ----------------
 * Purpose: Table containing all mixer functions.
 * Notes  : The Visual Studio project settings for this file have been adjusted
 *          to force function inlining, so that the mixer has a somewhat acceptable
 *          performance in debug mode. If you need to debug anything here, be sure
 *          to disable those optimizations if needed.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#include "Mixer.h"
#include "Snd_defs.h"
#include "ModChannel.h"
#include "MixFuncTable.h"

#ifdef MPT_INTMIXER
#include "IntMixer.h"
#else
#include "FloatMixer.h"
#endif // MPT_INTMIXER

OPENMPT_NAMESPACE_BEGIN

namespace MixFuncTable
{
#ifdef MPT_INTMIXER
	typedef Int8MToIntS I8M;
	typedef Int16MToIntS I16M;
	typedef Int8SToIntS I8S;
	typedef Int16SToIntS I16S;
#else
	typedef Int8MToFloatS I8M;
	typedef Int16MToFloatS I16M;
	typedef Int8SToFloatS I8S;
	typedef Int16SToFloatS I16S;
#endif // MPT_INTMIXER

// Build mix function table for given resampling, filter and ramping settings: One function each for 8-Bit / 16-Bit Mono / Stereo
#define BuildMixFuncTableRamp(resampling, filter, ramp) \
	SampleLoop<I8M, resampling<I8M>, filter<I8M>, MixMono ## ramp<I8M> >, \
	SampleLoop<I16M, resampling<I16M>, filter<I16M>, MixMono ## ramp<I16M> >, \
	SampleLoop<I8S, resampling<I8S>, filter<I8S>, MixStereo ## ramp<I8S> >, \
	SampleLoop<I16S, resampling<I16S>, filter<I16S>, MixStereo ## ramp<I16S> >

// Build mix function table for given resampling, filter settings: With and without ramping
#define BuildMixFuncTableFilter(resampling, filter) \
	BuildMixFuncTableRamp(resampling, filter, NoRamp), \
	BuildMixFuncTableRamp(resampling, filter, Ramp)

// Build mix function table for given resampling settings: With and without filter
#define BuildMixFuncTable(resampling) \
	BuildMixFuncTableFilter(resampling, NoFilter), \
	BuildMixFuncTableFilter(resampling, ResonantFilter)

const MixFuncInterface Functions[6 * 16] =
{
	BuildMixFuncTable(NoInterpolation),			// No SRC
	BuildMixFuncTable(LinearInterpolation),		// Linear SRC
	BuildMixFuncTable(FastSincInterpolation),	// Fast Sinc (Cubic Spline) SRC
	BuildMixFuncTable(PolyphaseInterpolation),	// Kaiser SRC
	BuildMixFuncTable(FIRFilterInterpolation),	// FIR SRC
	BuildMixFuncTable(AmigaBlepInterpolation),	// Amiga emulation
};


#undef BuildMixFuncTableRamp
#undef BuildMixFuncTableFilter
#undef BuildMixFuncTable


ResamplingIndex ResamplingModeToMixFlags(ResamplingMode resamplingMode)
{
	switch(resamplingMode)
	{
	case SRCMODE_NEAREST:   return ndxNoInterpolation;
	case SRCMODE_LINEAR:    return ndxLinear;
	case SRCMODE_SPLINE:    return ndxFastSinc;
	case SRCMODE_POLYPHASE: return ndxKaiser;
	case SRCMODE_FIRFILTER: return ndxFIRFilter;
	case SRCMODE_AMIGA:     return ndxAmigaBlep;
	default:                MPT_ASSERT_NOTREACHED();
	}
	return ndxNoInterpolation;
}

} // namespace MixFuncTable

OPENMPT_NAMESPACE_END
