/*
 * DSP.cpp
 * -----------
 * Purpose: Mixing code for various DSPs (EQ, Mega-Bass, ...)
 * Notes  : Ugh... This should really be removed at some point.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "../soundlib/Sndfile.h"
#include "../sounddsp/DSP.h"
#include <math.h>

OPENMPT_NAMESPACE_BEGIN

#ifndef NO_DSP


// Bass Expansion
#define DEFAULT_XBASS_RANGE		14	// (x+2)*20 Hz (320Hz)
#define DEFAULT_XBASS_DEPTH		6	// 1+(3>>(x-4)) (+6dB)


////////////////////////////////////////////////////////////////////
// DSP Effects internal state

static void X86_StereoDCRemoval(int *, uint32 count, int32 &nDCRFlt_Y1l, int32 &nDCRFlt_X1l, int32 &nDCRFlt_Y1r, int32 &nDCRFlt_X1r);
static void X86_MonoDCRemoval(int *, uint32 count, int32 &nDCRFlt_Y1l, int32 &nDCRFlt_X1l);

///////////////////////////////////////////////////////////////////////////////////
//
// Biquad setup
//


#define PI	3.14159265358979323f
static inline float Sgn(float x) { return (x >= 0) ? 1.0f : -1.0f; }
static void ShelfEQ(int32 scale,
	int32 &outA1, int32 &outB0, int32 &outB1,
	int32 F_c, int32 F_s, float gainDC, float gainFT, float gainPI)
{
	float a1, b0, b1;
	float gainFT2, gainDC2, gainPI2;
	float alpha, beta0, beta1, rho;
	float wT, quad;

	wT = PI * F_c / F_s;
	gainPI2 = gainPI * gainPI;
	gainFT2 = gainFT * gainFT;
	gainDC2 = gainDC * gainDC;

	quad = gainPI2 + gainDC2 - (gainFT2*2);

	alpha = 0;

	if (quad != 0)
	{
		float lambda = (gainPI2 - gainDC2) / quad;
		alpha  = (float)(lambda - Sgn(lambda)*sqrt(lambda*lambda - 1.0f));
	}

	beta0 = 0.5f * ((gainDC + gainPI) + (gainDC - gainPI) * alpha);
	beta1 = 0.5f * ((gainDC - gainPI) + (gainDC + gainPI) * alpha);
	rho   = (float)((sin((wT*0.5f) - (PI/4.0f))) / (sin((wT*0.5f) + (PI/4.0f))));

	quad  = 1.0f / (1.0f + rho*alpha);

	b0 = ((beta0 + rho*beta1) * quad);
	b1 = ((beta1 + rho*beta0) * quad);
	a1 = - ((rho + alpha) * quad);

	outA1 = Util::Round<int32>(a1 * scale);
	outB0 = Util::Round<int32>(b0 * scale);
	outB1 = Util::Round<int32>(b1 * scale);
}


CSurroundSettings::CSurroundSettings() : m_nProLogicDepth(12), m_nProLogicDelay(20)
{

}


CMegaBassSettings::CMegaBassSettings() : m_nXBassDepth(DEFAULT_XBASS_DEPTH), m_nXBassRange(DEFAULT_XBASS_RANGE)
{

}


CSurround::CSurround()
{
	// Surround Encoding: 1 delay line + low-pass filter + high-pass filter
	nSurroundSize = 0;
	nSurroundPos = 0;
	nDolbyDepth = 0;

	// Surround Biquads
	nDolbyHP_Y1 = 0;
	nDolbyHP_X1 = 0;
	nDolbyLP_Y1 = 0;
	nDolbyHP_B0 = 0;
	nDolbyHP_B1 = 0;
	nDolbyHP_A1 = 0;
	nDolbyLP_B0 = 0;
	nDolbyLP_B1 = 0;
	nDolbyLP_A1 = 0;

	MemsetZero(SurroundBuffer);

}


CMegaBass::CMegaBass()
{

	// Bass Expansion: low-pass filter
	nXBassFlt_Y1 = 0;
	nXBassFlt_X1 = 0;
	nXBassFlt_B0 = 0;
	nXBassFlt_B1 = 0;
	nXBassFlt_A1 = 0;

	// DC Removal Biquad
	nDCRFlt_Y1lf = 0;
	nDCRFlt_X1lf = 0;
	nDCRFlt_Y1rf = 0;
	nDCRFlt_X1rf = 0;
	nDCRFlt_Y1lb = 0;
	nDCRFlt_X1lb = 0;
	nDCRFlt_Y1rb = 0;
	nDCRFlt_X1rb = 0;

}


void CSurround::Initialize(bool bReset, DWORD MixingFreq)
{
	MPT_UNREFERENCED_PARAMETER(bReset);
	if (!m_Settings.m_nProLogicDelay) m_Settings.m_nProLogicDelay = 20;

	// Pro-Logic Surround
	nSurroundPos = nSurroundSize = 0;
	{
		memset(SurroundBuffer, 0, sizeof(SurroundBuffer));
		nSurroundSize = (MixingFreq * m_Settings.m_nProLogicDelay) / 1000;
		if (nSurroundSize > SURROUNDBUFFERSIZE) nSurroundSize = SURROUNDBUFFERSIZE;
		nDolbyDepth = m_Settings.m_nProLogicDepth;
		if (nDolbyDepth < 1) nDolbyDepth = 1;
		if (nDolbyDepth > 16) nDolbyDepth = 16;
		// Setup biquad filters
		ShelfEQ(1024, nDolbyHP_A1, nDolbyHP_B0, nDolbyHP_B1, 200, MixingFreq, 0, 0.5f, 1);
		ShelfEQ(1024, nDolbyLP_A1, nDolbyLP_B0, nDolbyLP_B1, 7000, MixingFreq, 1, 0.75f, 0);
		nDolbyHP_X1 = nDolbyHP_Y1 = 0;
		nDolbyLP_Y1 = 0;
		// Surround Level
		nDolbyHP_B0 = (nDolbyHP_B0 * nDolbyDepth) >> 5;
		nDolbyHP_B1 = (nDolbyHP_B1 * nDolbyDepth) >> 5;
		// +6dB
		nDolbyLP_B0 *= 2;
		nDolbyLP_B1 *= 2;
	}
}


void CMegaBass::Initialize(bool bReset, DWORD MixingFreq)
{
	// Bass Expansion Reset
	{
		int32 a1 = 0, b0 = 1024, b1 = 0;
		int nXBassCutOff = 50 + (m_Settings.m_nXBassRange+2) * 20;
		int nXBassGain = m_Settings.m_nXBassDepth;
		nXBassGain = mpt::clamp(nXBassGain, 2, 8);
		nXBassCutOff = mpt::clamp(nXBassCutOff, 60, 600);
		ShelfEQ(1024, a1, b0, b1, nXBassCutOff, MixingFreq,
				1.0f + (1.0f/16.0f) * (0x300 >> nXBassGain),
				1.0f,
				0.0000001f);
		if (nXBassGain > 5)
		{
			b0 >>= (nXBassGain-5);
			b1 >>= (nXBassGain-5);
		}
		nXBassFlt_A1 = a1;
		nXBassFlt_B0 = b0;
		nXBassFlt_B1 = b1;
		//Log("b0=%d b1=%d a1=%d\n", b0, b1, a1);
	}
	if (bReset)
	{
		nXBassFlt_X1 = 0;
		nXBassFlt_Y1 = 0;
		nDCRFlt_X1lf = 0;
		nDCRFlt_X1rf = 0;
		nDCRFlt_Y1lf = 0;
		nDCRFlt_Y1rf = 0;
		nDCRFlt_X1lb = 0;
		nDCRFlt_X1rb = 0;
		nDCRFlt_Y1lb = 0;
		nDCRFlt_Y1rb = 0;
	}
}


// 2-channel surround
void CSurround::ProcessStereoSurround(int * MixSoundBuffer, int count)
{
	int *pr = MixSoundBuffer, hy1 = nDolbyHP_Y1;
	for (int r=count; r; r--)
	{
		// Delay
		int secho = SurroundBuffer[nSurroundPos];
		SurroundBuffer[nSurroundPos] = (pr[0]+pr[1]+256) >> 9;
		// High-pass
		int v0 = (nDolbyHP_B0 * secho + nDolbyHP_B1 * nDolbyHP_X1 + nDolbyHP_A1 * hy1) >> 10;
		nDolbyHP_X1 = secho;
		// Low-pass
		int v = (nDolbyLP_B0 * v0 + nDolbyLP_B1 * hy1 + nDolbyLP_A1 * nDolbyLP_Y1) >> (10-8);
		hy1 = v0;
		nDolbyLP_Y1 = v >> 8;
		// Add echo
		pr[0] += v;
		pr[1] -= v;
		if (++nSurroundPos >= nSurroundSize) nSurroundPos = 0;
		pr += 2;
	}
	nDolbyHP_Y1 = hy1;
}


// 4-channels surround
void CSurround::ProcessQuadSurround(int * MixSoundBuffer, int * MixRearBuffer, int count)
{
	int *pr = MixSoundBuffer, hy1 = nDolbyHP_Y1;
	for (int r=count; r; r--)
	{
		int vl = pr[0] >> 1;
		int vr = pr[1] >> 1;
		pr[(uint32)(MixRearBuffer-MixSoundBuffer)] += vl;
		pr[((uint32)(MixRearBuffer-MixSoundBuffer))+1] += vr;
		// Delay
		int secho = SurroundBuffer[nSurroundPos];
		SurroundBuffer[nSurroundPos] = (vr+vl+256) >> 9;
		// High-pass
		int v0 = (nDolbyHP_B0 * secho + nDolbyHP_B1 * nDolbyHP_X1 + nDolbyHP_A1 * hy1) >> 10;
		nDolbyHP_X1 = secho;
		// Low-pass
		int v = (nDolbyLP_B0 * v0 + nDolbyLP_B1 * hy1 + nDolbyLP_A1 * nDolbyLP_Y1) >> (10-8);
		hy1 = v0;
		nDolbyLP_Y1 = v >> 8;
		// Add echo
		pr[(uint32)(MixRearBuffer-MixSoundBuffer)] += v;
		pr[((uint32)(MixRearBuffer-MixSoundBuffer))+1] += v;
		if (++nSurroundPos >= nSurroundSize) nSurroundPos = 0;
		pr += 2;
	}
	nDolbyHP_Y1 = hy1;
}


void CSurround::Process(int * MixSoundBuffer, int * MixRearBuffer, int count, uint32 nChannels)
{

	if(nChannels >= 2)

	// Dolby Pro-Logic Surround
	{
		if (nChannels > 2) ProcessQuadSurround(MixSoundBuffer, MixRearBuffer, count); else
		ProcessStereoSurround(MixSoundBuffer, count);
	}

}


void CMegaBass::Process(int * MixSoundBuffer, int * MixRearBuffer, int count, uint32 nChannels)
{

	if(nChannels >= 2)
	{
		X86_StereoDCRemoval(MixSoundBuffer, count, nDCRFlt_Y1lf, nDCRFlt_X1lf, nDCRFlt_Y1rf, nDCRFlt_X1rf);
		if(nChannels > 2) X86_StereoDCRemoval(MixSoundBuffer, count, nDCRFlt_Y1lb, nDCRFlt_X1lb, nDCRFlt_Y1rb, nDCRFlt_X1rb);
		int *px = MixSoundBuffer;
		int *py = MixRearBuffer;
		int x1 = nXBassFlt_X1;
		int y1 = nXBassFlt_Y1;
		if(nChannels > 2) for (int x=count; x; x--)
		{
			int x_m = (px[0]+px[1]+py[0]+py[1]+0x100)>>9;

			y1 = (nXBassFlt_B0 * x_m + nXBassFlt_B1 * x1 + nXBassFlt_A1 * y1) >> (10-8);
			x1 = x_m;
			px[0] += y1;
			px[1] += y1;
			py[0] += y1;
			py[1] += y1;
			y1 = (y1+0x80) >> 8;
			px += 2;
			py += 2;
		} else for (int x=count; x; x--)
		{
			int x_m = (px[0]+px[1]+0x100)>>9;

			y1 = (nXBassFlt_B0 * x_m + nXBassFlt_B1 * x1 + nXBassFlt_A1 * y1) >> (10-8);
			x1 = x_m;
			px[0] += y1;
			px[1] += y1;
			y1 = (y1+0x80) >> 8;
			px += 2;
		}
		nXBassFlt_X1 = x1;
		nXBassFlt_Y1 = y1;
	} else
	{
		X86_MonoDCRemoval(MixSoundBuffer, count, nDCRFlt_Y1lf, nDCRFlt_X1lf);
		int *px = MixSoundBuffer;
		int x1 = nXBassFlt_X1;
		int y1 = nXBassFlt_Y1;
		for (int x=count; x; x--)
		{
			int x_m = (px[0]+0x80)>>8;

			y1 = (nXBassFlt_B0 * x_m + nXBassFlt_B1 * x1 + nXBassFlt_A1 * y1) >> (10-8);
			x1 = x_m;
			px[0] += y1;
			y1 = (y1+0x40) >> 8;
			px++;
		}
		nXBassFlt_X1 = x1;
		nXBassFlt_Y1 = y1;
	}

}



//////////////////////////////////////////////////////////////////////////
//
// DC Removal
//

#define DCR_AMOUNT		9

static void X86_StereoDCRemoval(int *pBuffer, uint32 nSamples, int32 &nDCRFlt_Y1l, int32 &nDCRFlt_X1l, int32 &nDCRFlt_Y1r, int32 &nDCRFlt_X1r)
{
	int y1l = nDCRFlt_Y1l, x1l = nDCRFlt_X1l;
	int y1r = nDCRFlt_Y1r, x1r = nDCRFlt_X1r;

	while(nSamples--)
	{
		int inL = pBuffer[0];
		int inR = pBuffer[1];
		int diffL = x1l - inL;
		int diffR = x1r - inR;
		x1l = inL;
		x1r = inR;
		int outL = diffL / (1 << (DCR_AMOUNT + 1)) - diffL + y1l;
		int outR = diffR / (1 << (DCR_AMOUNT + 1)) - diffR + y1r;
		pBuffer[0] = outL;
		pBuffer[1] = outR;
		pBuffer += 2;
		y1l = outL - outL / (1 << DCR_AMOUNT);
		y1r = outR - outR / (1 << DCR_AMOUNT);
	}

	nDCRFlt_Y1l = y1l;
	nDCRFlt_X1l = x1l;
	nDCRFlt_Y1r = y1r;
	nDCRFlt_X1r = x1r;
}


static void X86_MonoDCRemoval(int *pBuffer, uint32 nSamples, int32 &nDCRFlt_Y1l, int32 &nDCRFlt_X1l)
{
	int y1l = nDCRFlt_Y1l, x1l = nDCRFlt_X1l;
	while(nSamples--)
	{
		int inM = pBuffer[0];
		int diff = x1l - inM;
		x1l = inM;
		pBuffer[0] = inM = diff / (1 << (DCR_AMOUNT + 1)) - diff + y1l;
		pBuffer++;
		y1l = inM - inM / (1 << DCR_AMOUNT);
	}

	nDCRFlt_Y1l = y1l;
	nDCRFlt_X1l = x1l;
}


/////////////////////////////////////////////////////////////////
// Clean DSP Effects interface

// [XBass level 0(quiet)-100(loud)], [cutoff in Hz 20-100]
void CMegaBass::SetXBassParameters(uint32 nDepth, uint32 nRange)
{
	if (nDepth > 100) nDepth = 100;
	uint32 gain = nDepth / 20;
	if (gain > 4) gain = 4;
	m_Settings.m_nXBassDepth = 8 - gain;	// filter attenuation 1/256 .. 1/16
	uint32 range = nRange / 5;
	if (range > 5) range -= 5; else range = 0;
	if (nRange > 16) nRange = 16;
	m_Settings.m_nXBassRange = 21 - range;	// filter average on 0.5-1.6ms
}


// [Surround level 0(quiet)-100(heavy)] [delay in ms, usually 5-50ms]
void CSurround::SetSurroundParameters(uint32 nDepth, uint32 nDelay)
{
	uint32 gain = (nDepth * 16) / 100;
	if (gain > 16) gain = 16;
	if (gain < 1) gain = 1;
	m_Settings.m_nProLogicDepth = gain;
	if (nDelay < 4) nDelay = 4;
	if (nDelay > 50) nDelay = 50;
	m_Settings.m_nProLogicDelay = nDelay;
}


#else


MPT_MSVC_WORKAROUND_LNK4221(DSP)


#endif // NO_DSP


OPENMPT_NAMESPACE_END
