/*
 * EQ.h
 * ----
 * Purpose: Mixing code for equalizer.
 * Notes  : Ugh... This should really be removed at some point.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "../soundlib/Mixer.h"	// For MIXBUFFERSIZE

OPENMPT_NAMESPACE_BEGIN

#ifndef NO_EQ

#define MAX_EQ_BANDS	6

typedef struct _EQBANDSTRUCT
{
	float32 a0;
	float32 a1;
	float32 a2;
	float32 b1;
	float32 b2;
	float32 x1;
	float32 x2;
	float32 y1;
	float32 y2;
	float32 Gain;
	float32 CenterFrequency;
	bool bEnable;
} EQBANDSTRUCT;

class CEQ
{
private:
	EQBANDSTRUCT gEQ[MAX_EQ_BANDS*2];
public:
	CEQ();
public:
	void Initialize(bool bReset, DWORD MixingFreq);
	void ProcessStereo(int *pbuffer, float *MixFloatBuffer, UINT nCount);
	void ProcessMono(int *pbuffer, float *MixFloatBuffer, UINT nCount);
	void SetEQGains(const UINT *pGains, UINT nGains, const UINT *pFreqs, bool bReset, DWORD MixingFreq);
};


class CQuadEQ
{
private:
	CEQ front;
	CEQ rear;
	float EQTempFloatBuffer[MIXBUFFERSIZE * 2];
public:
	void Initialize(bool bReset, DWORD MixingFreq);
	void Process(int *frontBuffer, int *rearBuffer, UINT nCount, UINT nChannels);
	void SetEQGains(const UINT *pGains, UINT nGains, const UINT *pFreqs, bool bReset, DWORD MixingFreq);
};


#endif // !NO_EQ

OPENMPT_NAMESPACE_END
