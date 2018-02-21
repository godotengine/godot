/*
 * Mixer.h
 * -------
 * Purpose: Basic mixer constants
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

OPENMPT_NAMESPACE_BEGIN

#define MPT_INTMIXER

#ifdef MPT_INTMIXER
typedef int32 mixsample_t;
enum { MIXING_FILTER_PRECISION = 16 };	// Fixed point resonant filter bits
#else
typedef float mixsample_t;
#endif

#define MIXBUFFERSIZE 512

#define VOLUMERAMPPRECISION 12	// Fractional bits in volume ramp variables

enum { MIXING_ATTENUATION = 4 };
enum { MIXING_FRACTIONAL_BITS = 32 - 1 - MIXING_ATTENUATION };

enum
{
	MIXING_CLIPMAX = ((1<<MIXING_FRACTIONAL_BITS)-1),
	MIXING_CLIPMIN = -(MIXING_CLIPMAX),
};

const float MIXING_SCALEF = static_cast<float>(1 << MIXING_FRACTIONAL_BITS);

// The absolute maximum number of sampling points any interpolation algorithm is going to look at in any direction from the current sampling point
// Currently, the maximum is 4 sampling points forwards and 3 sampling points backwards (Polyphase / FIR algorithms).
// Hence, this value must be at least 4.
// Note that choosing a higher value (e.g. 16) will reduce CPU usage when using many extremely short (length < 16) samples.
#define InterpolationMaxLookahead	16u

// Maximum size of a sampling point of a sample, in bytes.
// The biggest sampling point size is currently 16-bit stereo = 2 * 2 bytes.
#define MaxSamplingPointSize		4u

OPENMPT_NAMESPACE_END
