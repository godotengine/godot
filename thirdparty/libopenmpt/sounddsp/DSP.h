/*
 * DSP.h
 * -----
 * Purpose: Mixing code for various DSPs (EQ, Mega-Bass, ...)
 * Notes  : Ugh... This should really be removed at some point.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


OPENMPT_NAMESPACE_BEGIN


#ifndef NO_DSP

// Buffer Sizes
#define SURROUNDBUFFERSIZE		2048	// 50ms @ 48kHz


class CSurroundSettings
{
public:
	uint32 m_nProLogicDepth;
	uint32 m_nProLogicDelay;
public:
	CSurroundSettings();
};


class CMegaBassSettings
{
public:
	uint32 m_nXBassDepth;
	uint32 m_nXBassRange;
public:
	CMegaBassSettings();
};


class CSurround
{
public:
	CSurroundSettings m_Settings;

	// Surround Encoding: 1 delay line + low-pass filter + high-pass filter
	int32 nSurroundSize;
	int32 nSurroundPos;
	int32 nDolbyDepth;

	// Surround Biquads
	int32 nDolbyHP_Y1;
	int32 nDolbyHP_X1;
	int32 nDolbyLP_Y1;
	int32 nDolbyHP_B0;
	int32 nDolbyHP_B1;
	int32 nDolbyHP_A1;
	int32 nDolbyLP_B0;
	int32 nDolbyLP_B1;
	int32 nDolbyLP_A1;

	int32 SurroundBuffer[SURROUNDBUFFERSIZE];

public:
	CSurround();
public:
	void SetSettings(const CSurroundSettings &settings) { m_Settings = settings; }
	// [XBass level 0(quiet)-100(loud)], [cutoff in Hz 10-100]
	bool SetXBassParameters(uint32 nDepth, uint32 nRange);
	// [Surround level 0(quiet)-100(heavy)] [delay in ms, usually 5-40ms]
	void SetSurroundParameters(uint32 nDepth, uint32 nDelay);
	void Initialize(bool bReset, DWORD MixingFreq);
	void Process(int * MixSoundBuffer, int * MixRearBuffer, int count, uint32 nChannels);
private:
	void ProcessStereoSurround(int * MixSoundBuffer, int count);
	void ProcessQuadSurround(int * MixSoundBuffer, int * MixRearBuffer, int count);
};



class CMegaBass
{
public:
	CMegaBassSettings m_Settings;

	// Bass Expansion: low-pass filter
	int32 nXBassFlt_Y1;
	int32 nXBassFlt_X1;
	int32 nXBassFlt_B0;
	int32 nXBassFlt_B1;
	int32 nXBassFlt_A1;

	// DC Removal Biquad
	int32 nDCRFlt_Y1lf;
	int32 nDCRFlt_X1lf;
	int32 nDCRFlt_Y1rf;
	int32 nDCRFlt_X1rf;
	int32 nDCRFlt_Y1lb;
	int32 nDCRFlt_X1lb;
	int32 nDCRFlt_Y1rb;
	int32 nDCRFlt_X1rb;

public:
	CMegaBass();
public:
	void SetSettings(const CMegaBassSettings &settings) { m_Settings = settings; }
	// [XBass level 0(quiet)-100(loud)], [cutoff in Hz 10-100]
	void SetXBassParameters(uint32 nDepth, uint32 nRange);
	void Initialize(bool bReset, DWORD MixingFreq);
	void Process(int * MixSoundBuffer, int * MixRearBuffer, int count, uint32 nChannels);
};


#endif // NO_DSP


OPENMPT_NAMESPACE_END
