/*
 * Reverb.h
 * --------
 * Purpose: Mixing code for reverb.
 * Notes  : Ugh... This should really be removed at some point.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

#ifndef NO_REVERB

#include "../soundlib/Mixer.h"	// For MIXBUFFERSIZE

OPENMPT_NAMESPACE_BEGIN

////////////////////////////////////////////////////////////////////////
// Reverberation

#define NUM_REVERBTYPES			29

const char *GetReverbPresetName(uint32 nPreset);

/////////////////////////////////////////////////////////////////////////////
//
// SW Reverb structures
//

// Length-1 (in samples) of the reflections delay buffer: 32K, 371ms@22kHz
#define SNDMIX_REFLECTIONS_DELAY_MASK	0x1fff
#define SNDMIX_PREDIFFUSION_DELAY_MASK	0x7f	// 128 samples
#define SNDMIX_REVERB_DELAY_MASK		0xfff	// 4K samples (92ms @ 44kHz)

union LR16
{
	struct { int16 l, r; } c;
	int32 lr;
};

struct SWRvbReflection
{
	uint32 Delay, DelayDest;
	LR16   Gains[2];	// g_ll, g_rl, g_lr, g_rr
};

struct SWRvbRefDelay
{
	uint32 nDelayPos, nPreDifPos, nRefOutPos;
	int32  lMasterGain;			// reflections linear master gain
	LR16   nCoeffs;				// room low-pass coefficients
	LR16   History;				// room low-pass history
	LR16   nPreDifCoeffs;		// prediffusion coefficients
	LR16   ReflectionsGain;		// master reflections gain
	SWRvbReflection Reflections[8];	// Up to 8 SW Reflections
	LR16   RefDelayBuffer[SNDMIX_REFLECTIONS_DELAY_MASK + 1]; // reflections delay buffer
	LR16   PreDifBuffer[SNDMIX_PREDIFFUSION_DELAY_MASK + 1]; // pre-diffusion
	LR16   RefOut[SNDMIX_REVERB_DELAY_MASK + 1]; // stereo output of reflections
};


// Late reverberation
// Tank diffusers lengths
#define RVBDIF1L_LEN		(149*2)	// 6.8ms
#define RVBDIF1R_LEN		(223*2)	// 10.1ms
#define RVBDIF2L_LEN		(421*2)	// 19.1ms
#define RVBDIF2R_LEN		(647*2)	// 29.3ms
// Tank delay lines lengths
#define RVBDLY1L_LEN		(683*2)	// 30.9ms
#define RVBDLY1R_LEN	    (811*2) // 36.7ms
#define RVBDLY2L_LEN		(773*2)	// 35.1ms
#define RVBDLY2R_LEN	    (1013*2) // 45.9ms
// Tank delay lines mask
#define RVBDLY_MASK			2047

// Min/Max reflections delay
#define RVBMINREFDELAY		96		// 96 samples
#define RVBMAXREFDELAY		7500	// 7500 samples
// Min/Max reverb delay
#define RVBMINRVBDELAY		128		// 256 samples (11.6ms @ 22kHz)
#define RVBMAXRVBDELAY		3800	// 1900 samples (86ms @ 24kHz)

struct SWLateReverb
{
	uint32 nReverbDelay;		// Reverb delay (in samples)
	uint32 nDelayPos;			// Delay line position
	LR16   nDifCoeffs[2];		// Reverb diffusion
	LR16   nDecayDC[2];			// Reverb DC decay
	LR16   nDecayLP[2];			// Reverb HF decay
	LR16   LPHistory[2];		// Low-pass history
	LR16   Dif2InGains[2];		// 2nd diffuser input gains
	LR16   RvbOutGains[2];		// 4x2 Reverb output gains
	int32  lMasterGain;			// late reverb master gain
	int32  lDummyAlign;
	// Tank Delay lines
	LR16   Diffusion1[RVBDLY_MASK + 1];	// {dif1_l, dif1_r}
	LR16   Diffusion2[RVBDLY_MASK + 1];	// {dif2_l, dif2_r}
	LR16   Delay1[RVBDLY_MASK + 1];		// {dly1_l, dly1_r}
	LR16   Delay2[RVBDLY_MASK + 1];		// {dly2_l, dly2_r}
};

#define ENVIRONMENT_NUMREFLECTIONS		8

struct EnvironmentReflection
{
	int16  GainLL, GainRR, GainLR, GainRL;	// +/- 32K scale
	uint32 Delay;							// In samples
};

struct EnvironmentReverb
{
	int32  ReverbLevel;			// Late reverb gain (mB)
	int32  ReflectionsLevel;	// Master reflections gain (mB)
	int32  RoomHF;				// Room gain HF (mB)
	uint32 ReverbDecay;			// Reverb tank decay (0-7fff scale)
	int32  PreDiffusion;		// Reverb pre-diffusion amount (+/- 32K scale)
	int32  TankDiffusion;		// Reverb tank diffusion (+/- 32K scale)
	uint32 ReverbDelay;			// Reverb delay (in samples)
	float  flReverbDamping;		// HF tank gain [0.0, 1.0]
	int32  ReverbDecaySamples;	// Reverb decay time (in samples)
	EnvironmentReflection Reflections[ENVIRONMENT_NUMREFLECTIONS];
};


class CReverbSettings
{
public:
	uint32 m_nReverbDepth;
	uint32 m_nReverbType;
public:
	CReverbSettings();
};


class CReverb
{
public:
	CReverbSettings m_Settings;

	// Shared reverb state
private:
	mixsample_t MixReverbBuffer[MIXBUFFERSIZE * 2];
public:
	mixsample_t gnRvbROfsVol, gnRvbLOfsVol;

private:

	uint32 gnReverbSend;

	uint32 gnReverbSamples;
	uint32 gnReverbDecaySamples;

	// Internal reverb state
	bool g_bLastInPresent;
	bool g_bLastOutPresent;
	int g_nLastRvbIn_xl;
	int g_nLastRvbIn_xr;
	int g_nLastRvbIn_yl;
	int g_nLastRvbIn_yr;
	int g_nLastRvbOut_xl;
	int g_nLastRvbOut_xr;
	int32 gnDCRRvb_Y1[2];
	int32 gnDCRRvb_X1[2];

	// Reverb mix buffers
	SWRvbRefDelay g_RefDelay;
	SWLateReverb g_LateReverb;

public:
	CReverb();
public:
	void Initialize(bool bReset, uint32 MixingFreq);

	// can be called multiple times or never (if no data is sent to reverb)
	mixsample_t *GetReverbSendBuffer(uint32 nSamples);

	// call once after all data has been sent.
	void Process(mixsample_t *MixSoundBuffer, uint32 nSamples);

private:
	void Shutdown();
	// Pre/Post resampling and filtering
	uint32 ReverbProcessPreFiltering1x(int32 *pWet, uint32 nSamples);
	uint32 ReverbProcessPreFiltering2x(int32 *pWet, uint32 nSamples);
	void ReverbProcessPostFiltering1x(const int32 *pRvb, int32 *pDry, uint32 nSamples);
	void ReverbProcessPostFiltering2x(const int32 *pRvb, int32 *pDry, uint32 nSamples);
	void ReverbDCRemoval(int32 *pBuffer, uint32 nSamples);
	void ReverbDryMix(int32 *pDry, int32 *pWet, int lDryVol, uint32 nSamples);
	// Process pre-diffusion and pre-delay
	static void ProcessPreDelay(SWRvbRefDelay *pPreDelay, const int32 *pIn, uint32 nSamples);
	// Process reflections
	static void ProcessReflections(SWRvbRefDelay *pPreDelay, LR16 *pRefOut, int32 *pMixOut, uint32 nSamples);
	// Process Late Reverb (SW Reflections): stereo reflections output, 32-bit reverb output, SW reverb gain
	static void ProcessLateReverb(SWLateReverb *pReverb, LR16 *pRefOut, int32 *pMixOut, uint32 nSamples);
};


/////////////////////////////////////////////////////////////////////////////////
//
// I3DL2 reverb presets
//

#define SNDMIX_REVERB_PRESET_DEFAULT \
-10000,    0, 1.00f,0.50f,-10000,0.020f,-10000,0.040f,100.0f,100.0f

#define SNDMIX_REVERB_PRESET_GENERIC \
 -1000, -100, 1.49f,0.83f, -2602,0.007f,   200,0.011f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_PADDEDCELL \
 -1000,-6000, 0.17f,0.10f, -1204,0.001f,   207,0.002f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_ROOM \
 -1000, -454, 0.40f,0.83f, -1646,0.002f,    53,0.003f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_BATHROOM \
 -1000,-1200, 1.49f,0.54f,  -370,0.007f,  1030,0.011f,100.0f, 60.0f
#define SNDMIX_REVERB_PRESET_LIVINGROOM \
 -1000,-6000, 0.50f,0.10f, -1376,0.003f, -1104,0.004f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_STONEROOM \
 -1000, -300, 2.31f,0.64f,  -711,0.012f,    83,0.017f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_AUDITORIUM \
 -1000, -476, 4.32f,0.59f,  -789,0.020f,  -289,0.030f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_CONCERTHALL \
 -1000, -500, 3.92f,0.70f, -1230,0.020f,    -2,0.029f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_CAVE \
 -1000,    0, 2.91f,1.30f,  -602,0.015f,  -302,0.022f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_ARENA \
 -1000, -698, 7.24f,0.33f, -1166,0.020f,    16,0.030f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_HANGAR \
 -1000,-1000,10.05f,0.23f,  -602,0.020f,   198,0.030f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_CARPETEDHALLWAY \
 -1000,-4000, 0.30f,0.10f, -1831,0.002f, -1630,0.030f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_HALLWAY \
 -1000, -300, 1.49f,0.59f, -1219,0.007f,   441,0.011f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_STONECORRIDOR \
 -1000, -237, 2.70f,0.79f, -1214,0.013f,   395,0.020f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_ALLEY \
 -1000, -270, 1.49f,0.86f, -1204,0.007f,    -4,0.011f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_FOREST \
 -1000,-3300, 1.49f,0.54f, -2560,0.162f,  -613,0.088f, 79.0f,100.0f
#define SNDMIX_REVERB_PRESET_CITY \
 -1000, -800, 1.49f,0.67f, -2273,0.007f, -2217,0.011f, 50.0f,100.0f
#define SNDMIX_REVERB_PRESET_MOUNTAINS \
 -1000,-2500, 1.49f,0.21f, -2780,0.300f, -2014,0.100f, 27.0f,100.0f
#define SNDMIX_REVERB_PRESET_QUARRY \
 -1000,-1000, 1.49f,0.83f,-10000,0.061f,   500,0.025f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_PLAIN \
 -1000,-2000, 1.49f,0.50f, -2466,0.179f, -2514,0.100f, 21.0f,100.0f
#define SNDMIX_REVERB_PRESET_PARKINGLOT \
 -1000,    0, 1.65f,1.50f, -1363,0.008f, -1153,0.012f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_SEWERPIPE \
 -1000,-1000, 2.81f,0.14f,   429,0.014f,   648,0.021f, 80.0f, 60.0f
#define SNDMIX_REVERB_PRESET_UNDERWATER \
 -1000,-4000, 1.49f,0.10f,  -449,0.007f,  1700,0.011f,100.0f,100.0f

// Examples simulating General MIDI 2'musical' reverb presets
//
// Name  (Decay time)  Description
//
// Small Room  (1.1s)  A small size room with a length of 5m or so.
// Medium Room (1.3s)  A medium size room with a length of 10m or so.
// Large Room  (1.5s)  A large size room suitable for live performances.
// Medium Hall (1.8s)  A medium size concert hall.
// Large Hall  (1.8s)  A large size concert hall suitable for a full orchestra.
// Plate       (1.3s)  A plate reverb simulation.

#define SNDMIX_REVERB_PRESET_SMALLROOM \
 -1000, -600, 1.10f,0.83f,  -400,0.005f,   500,0.010f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_MEDIUMROOM \
 -1000, -600, 1.30f,0.83f, -1000,0.010f,  -200,0.020f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_LARGEROOM \
 -1000, -600, 1.50f,0.83f, -1600,0.020f, -1000,0.040f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_MEDIUMHALL \
 -1000, -600, 1.80f,0.70f, -1300,0.015f,  -800,0.030f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_LARGEHALL \
 -1000, -600, 1.80f,0.70f, -2000,0.030f, -1400,0.060f,100.0f,100.0f
#define SNDMIX_REVERB_PRESET_PLATE \
 -1000, -200, 1.30f,0.90f,     0,0.002f,     0,0.010f,100.0f, 75.0f

OPENMPT_NAMESPACE_END

#endif // NO_REVERB
