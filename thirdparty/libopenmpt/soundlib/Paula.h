/*
* Paula.h
* -------
* Purpose: Emulating the Amiga's sound chip, Paula, by implementing resampling using band-limited steps (BLEPs)
* Notes  : (currently none)
* Authors: OpenMPT Devs
*          Antti S. Lankila
* The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
*/

#pragma once

#include "Snd_defs.h"

OPENMPT_NAMESPACE_BEGIN

namespace Paula
{

const int PAULA_HZ = 3546895;
const int MINIMUM_INTERVAL = 16;
const int BLEP_SCALE = 17;
const int BLEP_SIZE = 2048;
const int MAX_BLEPS = (BLEP_SIZE / MINIMUM_INTERVAL);

class State
{
	struct Blep
	{
		int16 level;
		uint16 age;
	};

public:
	SamplePosition remainder, stepRemainder;
	int numSteps;				// Number of full-length steps
private:
	uint16 activeBleps;			// Count of simultaneous bleps to keep track of
	int16 globalOutputLevel;	// The instantenous value of Paula output
	Blep blepState[MAX_BLEPS];

public:
	State(uint32 sampleRate = 48000);

	void Reset();
	void InputSample(int16 sample);
	int OutputSample(bool filter);
	void Clock(int cycles);
};

}

OPENMPT_NAMESPACE_END
