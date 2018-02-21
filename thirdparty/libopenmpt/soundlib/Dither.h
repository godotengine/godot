/*
 * Dither.h
 * --------
 * Purpose: Dithering when converting to lower resolution sample formats.
 * Notes  : (currently none)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


#include "../common/mptRandom.h"


OPENMPT_NAMESPACE_BEGIN


struct DitherModPlugState
{
	uint32 rng_a;
	uint32 rng_b;
	DitherModPlugState()
	{
		rng_a = 0;
		rng_b = 0;
	}
};

struct DitherSimpleState
{
	int32 error[4];
	DitherSimpleState() {
		error[0] = 0;
		error[1] = 0;
		error[2] = 0;
		error[3] = 0;
	}
};

struct DitherState
{
	DitherModPlugState modplug;
	DitherSimpleState  simple;
	mpt::fast_prng prng;
	void Reset()
	{
		modplug = DitherModPlugState();
		simple = DitherSimpleState();
	}
	template <typename Trd>
	DitherState(Trd & rd)
		: prng(mpt::make_prng<mpt::fast_prng>(rd))
	{
		return;
	}
};

enum DitherMode
{
	DitherNone       = 0,
	DitherDefault    = 1, // chosen by OpenMPT code, might change
	DitherModPlug    = 2, // rectangular, 0.5 bit depth, no noise shaping (original ModPlug Tracker)
	DitherSimple     = 3, // rectangular, 1 bit depth, simple 1st order noise shaping
	NumDitherModes
};

class Dither
{
private:
	DitherState state;
	DitherMode mode;
public:
	template <typename Trd>
	Dither(Trd & rd)
		: state(rd)
	{
		mode = DitherDefault;
	}
	void SetMode(DitherMode mode_);
	DitherMode GetMode() const;
	void Reset();
	void Process(int *mixbuffer, std::size_t count, std::size_t channels, int bits);
	static mpt::ustring GetModeName(DitherMode mode);
};


OPENMPT_NAMESPACE_END
