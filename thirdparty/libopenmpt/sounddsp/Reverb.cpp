/*
 * Reverb.cpp
 * ----------
 * Purpose: Mixing code for reverb.
 * Notes  : Ugh... This should really be removed at some point.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#ifndef NO_REVERB
#include "Reverb.h"
#include "../soundlib/MixerLoops.h"

#ifdef ENABLE_MMX
#include <mmintrin.h>
#endif
#ifdef ENABLE_SSE2
#include <emmintrin.h>
#endif

#endif // NO_REVERB


OPENMPT_NAMESPACE_BEGIN


#ifndef NO_REVERB


#ifdef ENABLE_MMX
// Load two 32-bit values
static MPT_FORCEINLINE __m64 Load64MMX(const int32 *x) { return _mm_set_pi32(x[1], x[0]); }
// Load four 16-bit values
static MPT_FORCEINLINE __m64 Load64MMX(const LR16 (&x)[2]) { return Load64MMX(&x->lr); }
// Store 64-bit value from register (MSVC does not have_mm_cvtsi64_si64x) - macro to avoid emms warnings
#define Store64MMX(dst, src) \
	MPT_DO \
	{ \
		STATIC_ASSERT(sizeof((dst)[0]) == 4); \
		(dst)[0] = _mm_cvtsi64_si32(src); \
		(dst)[1] = _mm_cvtsi64_si32(_mm_unpackhi_pi32(src, src)); \
	} MPT_WHILE_0
#endif
#ifdef ENABLE_SSE2
// Load two 32-bit values
static MPT_FORCEINLINE __m128i Load64SSE(const int32 *x) { return _mm_loadl_epi64(reinterpret_cast<const __m128i *>(x)); }
// Load four 16-bit values
static MPT_FORCEINLINE __m128i Load64SSE(const LR16 (&x)[2]) { return _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&x)); }
// Store two 32-bit or four 16-bit values from register
static MPT_FORCEINLINE void Store64SSE(int32 *dst, __m128i src) { return _mm_storel_epi64(reinterpret_cast<__m128i *>(dst), src); }
static MPT_FORCEINLINE void Store64SSE(LR16 *dst, __m128i src) { return _mm_storel_epi64(reinterpret_cast<__m128i *>(dst), src); }
#endif

CReverbSettings::CReverbSettings()
{
	m_nReverbType = 0;
	m_nReverbDepth = 8; // 50%
}


CReverb::CReverb()
{
	// Shared reverb state
	InitMixBuffer(MixReverbBuffer, static_cast<uint32>(mpt::size(MixReverbBuffer)));
	gnRvbROfsVol = 0;
	gnRvbLOfsVol = 0;

	gnReverbSend = 0;

	gnReverbSamples = 0;
	gnReverbDecaySamples = 0;

	// Internal reverb state
	g_bLastInPresent = 0;
	g_bLastOutPresent = 0;
	g_nLastRvbIn_xl = 0;
	g_nLastRvbIn_xr = 0;
	g_nLastRvbIn_yl = 0;
	g_nLastRvbIn_yr = 0;
	g_nLastRvbOut_xl = 0;
	g_nLastRvbOut_xr = 0;
	MemsetZero(gnDCRRvb_Y1);
	MemsetZero(gnDCRRvb_X1);

	// Reverb mix buffers
	MemsetZero(g_RefDelay);
	MemsetZero(g_LateReverb);

}


static int32 OnePoleLowPassCoef(int32 scale, float g, float F_c, float F_s)
{
	if(g > 0.999999f) return 0;

	g *= g;
	double scale_over_1mg = scale / (1.0 - g);
	double cosw = std::cos(2.0 * M_PI * F_c / F_s);
	return Util::Round<int32>((1.0 - (std::sqrt((g + g) * (1.0 - cosw) - g * g * (1.0 - cosw * cosw)) + g * cosw)) * scale_over_1mg);
}

static float mBToLinear(int32 value_mB)
{
	if(!value_mB) return 1;
	if(value_mB <= -100000) return 0;

	const double val = value_mB * 3.321928094887362304 / (100.0 * 20.0);	// log2(10)/(100*20)
	return static_cast<float>(std::pow(2.0, val - static_cast<int32>(0.5 + val)));
}

static int32 mBToLinear(int32 scale, int32 value_mB)
{
	return Util::Round<int32>(mBToLinear(value_mB) * scale);
}


struct SNDMIX_REVERB_PROPERTIES
{
	int32 lRoom;                   // [-10000, 0]      default: -10000 mB
	int32 lRoomHF;                 // [-10000, 0]      default: 0 mB
	float flDecayTime;             // [0.1, 20.0]      default: 1.0 s
	float flDecayHFRatio;          // [0.1, 2.0]       default: 0.5
	int32 lReflections;            // [-10000, 1000]   default: -10000 mB
	float flReflectionsDelay;      // [0.0, 0.3]       default: 0.02 s
	int32 lReverb;                 // [-10000, 2000]   default: -10000 mB
	float flReverbDelay;           // [0.0, 0.1]       default: 0.04 s
	float flDiffusion;             // [0.0, 100.0]     default: 100.0 %
	float flDensity;               // [0.0, 100.0]     default: 100.0 %
};

struct SNDMIX_RVBPRESET
{
	SNDMIX_REVERB_PROPERTIES Preset;
	const char *name;
};


static SNDMIX_RVBPRESET gRvbPresets[NUM_REVERBTYPES] =
{
	{{ SNDMIX_REVERB_PRESET_PLATE },			"GM Plate"},
	{{ SNDMIX_REVERB_PRESET_SMALLROOM },		"GM Small Room"},
	{{ SNDMIX_REVERB_PRESET_MEDIUMROOM },		"GM Medium Room"},
	{{ SNDMIX_REVERB_PRESET_LARGEROOM },		"GM Large Room"},
	{{ SNDMIX_REVERB_PRESET_MEDIUMHALL },		"GM Medium Hall"},
	{{ SNDMIX_REVERB_PRESET_LARGEHALL },		"GM Large Hall"},
	{{ SNDMIX_REVERB_PRESET_GENERIC },			"Generic"},
	{{ SNDMIX_REVERB_PRESET_PADDEDCELL },		"Padded Cell"},
	{{ SNDMIX_REVERB_PRESET_ROOM },				"Room"},
	{{ SNDMIX_REVERB_PRESET_BATHROOM },			"Bathroom"},
	{{ SNDMIX_REVERB_PRESET_LIVINGROOM },		"Living Room"},
	{{ SNDMIX_REVERB_PRESET_STONEROOM },		"Stone Room"},
	{{ SNDMIX_REVERB_PRESET_AUDITORIUM },		"Auditorium"},
	{{ SNDMIX_REVERB_PRESET_CONCERTHALL },		"Concert Hall"},
	{{ SNDMIX_REVERB_PRESET_CAVE },				"Cave"},
	{{ SNDMIX_REVERB_PRESET_ARENA },			"Arena"},
	{{ SNDMIX_REVERB_PRESET_HANGAR },			"Hangar"},
	{{ SNDMIX_REVERB_PRESET_CARPETEDHALLWAY },	"Carpeted Hallway"},
	{{ SNDMIX_REVERB_PRESET_HALLWAY },			"Hallway"},
	{{ SNDMIX_REVERB_PRESET_STONECORRIDOR },	"Stone Corridor"},
	{{ SNDMIX_REVERB_PRESET_ALLEY },			"Alley"},
	{{ SNDMIX_REVERB_PRESET_FOREST },			"Forest"},
	{{ SNDMIX_REVERB_PRESET_CITY },				"City"},
	{{ SNDMIX_REVERB_PRESET_MOUNTAINS },		"Mountains"},
	{{ SNDMIX_REVERB_PRESET_QUARRY },			"Quarry"},
	{{ SNDMIX_REVERB_PRESET_PLAIN },			"Plain"},
	{{ SNDMIX_REVERB_PRESET_PARKINGLOT },		"Parking Lot"},
	{{ SNDMIX_REVERB_PRESET_SEWERPIPE },		"Sewer Pipe"},
	{{ SNDMIX_REVERB_PRESET_UNDERWATER },		"Underwater"},
};

const char *GetReverbPresetName(uint32 nPreset)
{
	return (nPreset < NUM_REVERBTYPES) ? gRvbPresets[nPreset].name : nullptr;
}

//////////////////////////////////////////////////////////////////////////
//
// I3DL2 environmental reverb support
//

struct REFLECTIONPRESET
{
	int32 lDelayFactor;
	int16 sGainLL, sGainRR, sGainLR, sGainRL;
};

const REFLECTIONPRESET gReflectionsPreset[ENVIRONMENT_NUMREFLECTIONS] =
{
	// %Delay, ll,    rr,   lr,    rl
	{0,    9830,   6554,	  0,     0},
	{10,   6554,  13107,	  0,     0},
	{24,  -9830,  13107,	  0,     0},
	{36,  13107,  -6554,      0,     0},
	{54,  16384,  16384,  -1638, -1638},
	{61, -13107,   8192,   -328,  -328},
	{73, -11468, -11468,  -3277,  3277},
	{87,  13107,  -9830,   4916, -4916}
};

////////////////////////////////////////////////////////////////////////////////////
//
// Implementation
//

MPT_FORCEINLINE int32 ftol(float f) { return static_cast<int32>(f); }

static void I3dl2_to_Generic(
				const SNDMIX_REVERB_PROPERTIES *pReverb,
				EnvironmentReverb *pRvb,
				float flOutputFreq,
				int32 lMinRefDelay,
				int32 lMaxRefDelay,
				int32 lMinRvbDelay,
				int32 lMaxRvbDelay,
				int32 lTankLength)
{
	float flDelayFactor, flDelayFactorHF, flDecayTimeHF;
	int32 lDensity, lTailDiffusion;

	// Common parameters
	pRvb->ReverbLevel = pReverb->lReverb;
	pRvb->ReflectionsLevel = pReverb->lReflections;
	pRvb->RoomHF = pReverb->lRoomHF;

	// HACK: Somewhat normalize the reverb output level
	int32 lMaxLevel = (pRvb->ReverbLevel > pRvb->ReflectionsLevel) ? pRvb->ReverbLevel : pRvb->ReflectionsLevel;
	if (lMaxLevel < -600)
	{
		lMaxLevel += 600;
		pRvb->ReverbLevel -= lMaxLevel;
		pRvb->ReflectionsLevel -= lMaxLevel;
	}

	// Pre-Diffusion factor (for both reflections and late reverb)
	lDensity = 8192 + ftol(79.31f * pReverb->flDensity);
	pRvb->PreDiffusion = lDensity;

	// Late reverb diffusion
	lTailDiffusion = ftol((0.15f + pReverb->flDiffusion * (0.36f*0.01f)) * 32767.0f);
	if (lTailDiffusion > 0x7f00) lTailDiffusion = 0x7f00;
	pRvb->TankDiffusion = lTailDiffusion;

	// Verify reflections and reverb delay parameters
	float flRefDelay = pReverb->flReflectionsDelay;
	if (flRefDelay > 0.100f) flRefDelay = 0.100f;
	int32 lReverbDelay = ftol(pReverb->flReverbDelay * flOutputFreq);
	int32 lReflectionsDelay = ftol(flRefDelay * flOutputFreq);
	int32 lReverbDecayTime = ftol(pReverb->flDecayTime * flOutputFreq);
	if (lReflectionsDelay < lMinRefDelay)
	{
		lReverbDelay -= (lMinRefDelay - lReflectionsDelay);
		lReflectionsDelay = lMinRefDelay;
	}
	if (lReflectionsDelay > lMaxRefDelay)
	{
		lReverbDelay += (lReflectionsDelay - lMaxRefDelay);
		lReflectionsDelay = lMaxRefDelay;
	}
	// Adjust decay time when adjusting reverb delay
	if (lReverbDelay < lMinRvbDelay)
	{
		lReverbDecayTime -= (lMinRvbDelay - lReverbDelay);
		lReverbDelay = lMinRvbDelay;
	}
	if (lReverbDelay > lMaxRvbDelay)
	{
		lReverbDecayTime += (lReverbDelay - lMaxRvbDelay);
		lReverbDelay = lMaxRvbDelay;
	}
	pRvb->ReverbDelay = lReverbDelay;
	pRvb->ReverbDecaySamples = lReverbDecayTime;
	// Setup individual reflections delay and gains
	for (uint32 iRef=0; iRef<ENVIRONMENT_NUMREFLECTIONS; iRef++)
	{
		EnvironmentReflection &ref = pRvb->Reflections[iRef];
		ref.Delay = lReflectionsDelay + (gReflectionsPreset[iRef].lDelayFactor * lReverbDelay + 50)/100;
		ref.GainLL = gReflectionsPreset[iRef].sGainLL;
		ref.GainRL = gReflectionsPreset[iRef].sGainRL;
		ref.GainLR = gReflectionsPreset[iRef].sGainLR;
		ref.GainRR = gReflectionsPreset[iRef].sGainRR;
	}

	// Late reverb decay time
	if (lTankLength < 10) lTankLength = 10;
	flDelayFactor = (lReverbDecayTime <= lTankLength) ? 1.0f : ((float)lTankLength / (float)lReverbDecayTime);
	pRvb->ReverbDecay = ftol(std::pow(0.001f, flDelayFactor) * 32768.0f);

	// Late Reverb Decay HF
	flDecayTimeHF = (float)lReverbDecayTime * pReverb->flDecayHFRatio;
	flDelayFactorHF = (flDecayTimeHF <= (float)lTankLength) ? 1.0f : ((float)lTankLength / flDecayTimeHF);
	pRvb->flReverbDamping = std::pow(0.001f, flDelayFactorHF);
}


void CReverb::Shutdown()
{
	gnReverbSend = 0;

	gnRvbLOfsVol = 0;
	gnRvbROfsVol = 0;

	// Clear out all reverb state
	g_bLastInPresent = false;
	g_bLastOutPresent = false;
	g_nLastRvbIn_xl = g_nLastRvbIn_xr = 0;
	g_nLastRvbIn_yl = g_nLastRvbIn_yr = 0;
	g_nLastRvbOut_xl = g_nLastRvbOut_xr = 0;
	MemsetZero(gnDCRRvb_X1);
	MemsetZero(gnDCRRvb_Y1);

	// Zero internal buffers
	MemsetZero(g_LateReverb.Diffusion1);
	MemsetZero(g_LateReverb.Diffusion2);
	MemsetZero(g_LateReverb.Delay1);
	MemsetZero(g_LateReverb.Delay2);
	MemsetZero(g_RefDelay.RefDelayBuffer);
	MemsetZero(g_RefDelay.PreDifBuffer);
	MemsetZero(g_RefDelay.RefOut);
}


void CReverb::Initialize(bool bReset, uint32 MixingFreq)
{
	if (m_Settings.m_nReverbType >= NUM_REVERBTYPES) m_Settings.m_nReverbType = 0;
	static SNDMIX_REVERB_PROPERTIES *spCurrentPreset = nullptr;
	SNDMIX_REVERB_PROPERTIES *pRvbPreset = &gRvbPresets[m_Settings.m_nReverbType].Preset;

	if ((pRvbPreset != spCurrentPreset) || (bReset))
	{
		// Reverb output frequency is half of the dry output rate
		float flOutputFrequency = (float)MixingFreq;
		EnvironmentReverb rvb;

		// Reset reverb parameters
		spCurrentPreset = pRvbPreset;
		I3dl2_to_Generic(pRvbPreset, &rvb, flOutputFrequency,
							RVBMINREFDELAY, RVBMAXREFDELAY,
							RVBMINRVBDELAY, RVBMAXRVBDELAY,
							( RVBDIF1L_LEN + RVBDIF1R_LEN
							+ RVBDIF2L_LEN + RVBDIF2R_LEN
							+ RVBDLY1L_LEN + RVBDLY1R_LEN
							+ RVBDLY2L_LEN + RVBDLY2R_LEN) / 2);

		// Store reverb decay time (in samples) for reverb auto-shutdown
		gnReverbDecaySamples = rvb.ReverbDecaySamples;

		// Room attenuation at high frequencies
		int32 nRoomLP;
		nRoomLP = OnePoleLowPassCoef(32768, mBToLinear(rvb.RoomHF), 5000, flOutputFrequency);
		g_RefDelay.nCoeffs.c.l = (int16)nRoomLP;
		g_RefDelay.nCoeffs.c.r = (int16)nRoomLP;

		// Pre-Diffusion factor (for both reflections and late reverb)
		g_RefDelay.nPreDifCoeffs.c.l = (int16)(rvb.PreDiffusion*2);
		g_RefDelay.nPreDifCoeffs.c.r = (int16)(rvb.PreDiffusion*2);

		// Setup individual reflections delay and gains
		for (uint32 iRef=0; iRef<8; iRef++)
		{
			SWRvbReflection &ref = g_RefDelay.Reflections[iRef];
			ref.DelayDest = rvb.Reflections[iRef].Delay;
			ref.Delay = ref.DelayDest;
			ref.Gains[0].c.l = rvb.Reflections[iRef].GainLL;
			ref.Gains[0].c.r = rvb.Reflections[iRef].GainRL;
			ref.Gains[1].c.l = rvb.Reflections[iRef].GainLR;
			ref.Gains[1].c.r = rvb.Reflections[iRef].GainRR;
		}
		g_LateReverb.nReverbDelay = rvb.ReverbDelay;

		// Reflections Master Gain
		uint32 lReflectionsGain = 0;
		if (rvb.ReflectionsLevel > -9000)
		{
			lReflectionsGain = mBToLinear(32768, rvb.ReflectionsLevel);
		}
		g_RefDelay.lMasterGain = lReflectionsGain;

		// Late reverb master gain
		uint32 lReverbGain = 0;
		if (rvb.ReverbLevel > -9000)
		{
			lReverbGain = mBToLinear(32768, rvb.ReverbLevel);
		}
		g_LateReverb.lMasterGain = lReverbGain;

		// Late reverb diffusion
		uint32 nTailDiffusion = rvb.TankDiffusion;
		if (nTailDiffusion > 0x7f00) nTailDiffusion = 0x7f00;
		g_LateReverb.nDifCoeffs[0].c.l = (int16)nTailDiffusion;
		g_LateReverb.nDifCoeffs[0].c.r = (int16)nTailDiffusion;
		g_LateReverb.nDifCoeffs[1].c.l = (int16)nTailDiffusion;
		g_LateReverb.nDifCoeffs[1].c.r = (int16)nTailDiffusion;
		g_LateReverb.Dif2InGains[0].c.l = 0x7000;
		g_LateReverb.Dif2InGains[0].c.r = 0x1000;
		g_LateReverb.Dif2InGains[1].c.l = 0x1000;
		g_LateReverb.Dif2InGains[1].c.r = 0x7000;

		// Late reverb decay time
		int32 nReverbDecay = rvb.ReverbDecay;
		Limit(nReverbDecay, 0, 0x7ff0);
		g_LateReverb.nDecayDC[0].c.l = (int16)nReverbDecay;
		g_LateReverb.nDecayDC[0].c.r = 0;
		g_LateReverb.nDecayDC[1].c.l = 0;
		g_LateReverb.nDecayDC[1].c.r = (int16)nReverbDecay;

		// Late Reverb Decay HF
		float fReverbDamping = rvb.flReverbDamping * rvb.flReverbDamping;
		int32 nDampingLowPass;

		nDampingLowPass = OnePoleLowPassCoef(32768, fReverbDamping, 5000, flOutputFrequency);
		Limit(nDampingLowPass, 0x100, 0x7f00);
		
		g_LateReverb.nDecayLP[0].c.l = (int16)nDampingLowPass;
		g_LateReverb.nDecayLP[0].c.r = 0;
		g_LateReverb.nDecayLP[1].c.l = 0;
		g_LateReverb.nDecayLP[1].c.r = (int16)nDampingLowPass;
	}
	if (bReset)
	{
		gnReverbSamples = 0;
		Shutdown();
	}
	// Wait at least 5 seconds before shutting down the reverb
	if (gnReverbDecaySamples < MixingFreq*5)
	{
		gnReverbDecaySamples = MixingFreq*5;
	}
}


mixsample_t *CReverb::GetReverbSendBuffer(uint32 nSamples)
{
	if(!gnReverbSend)
	{ // and we did not clear the buffer yet, do it now because we will get new data
		StereoFill(MixReverbBuffer, nSamples, gnRvbROfsVol, gnRvbLOfsVol);
	}
	gnReverbSend = 1; // we will have to process reverb
	return MixReverbBuffer;
}


// Reverb
void CReverb::Process(mixsample_t *MixSoundBuffer, uint32 nSamples)
{
	if((!gnReverbSend) && (!gnReverbSamples))
	{ // no data is sent to reverb and reverb decayed completely
		return;
	}
	if(!gnReverbSend)
	{ // no input data in MixReverbBuffer, so the buffer got not cleared in GetReverbSendBuffer(), do it now for decay
		StereoFill(MixReverbBuffer, nSamples, gnRvbROfsVol, gnRvbLOfsVol);
	}

	uint32 nIn, nOut;
	// Dynamically adjust reverb master gains
	int32 lMasterGain;
	lMasterGain = ((g_RefDelay.lMasterGain * m_Settings.m_nReverbDepth) >> 4);
	if (lMasterGain > 0x7fff) lMasterGain = 0x7fff;
	g_RefDelay.ReflectionsGain.c.l = (int16)lMasterGain;
	g_RefDelay.ReflectionsGain.c.r = (int16)lMasterGain;
	lMasterGain = ((g_LateReverb.lMasterGain * m_Settings.m_nReverbDepth) >> 4);
	if (lMasterGain > 0x10000) lMasterGain = 0x10000;
	g_LateReverb.RvbOutGains[0].c.l = (int16)((lMasterGain+0x7f) >> 3);	// l->l
	g_LateReverb.RvbOutGains[0].c.r = (int16)((lMasterGain+0xff) >> 4);	// r->l
	g_LateReverb.RvbOutGains[1].c.l = (int16)((lMasterGain+0xff) >> 4);	// l->r
	g_LateReverb.RvbOutGains[1].c.r = (int16)((lMasterGain+0x7f) >> 3);	// r->r
	// Process Dry/Wet Mix
	int32 lMaxRvbGain = (g_RefDelay.lMasterGain > g_LateReverb.lMasterGain) ? g_RefDelay.lMasterGain : g_LateReverb.lMasterGain;
	if (lMaxRvbGain > 32768) lMaxRvbGain = 32768;
	int32 lDryVol = (36 - m_Settings.m_nReverbDepth)>>1;
	if (lDryVol < 8) lDryVol = 8;
	if (lDryVol > 16) lDryVol = 16;
	lDryVol = 16 - (((16-lDryVol) * lMaxRvbGain) >> 15);
	ReverbDryMix(MixSoundBuffer, MixReverbBuffer, lDryVol, nSamples);
	// Downsample 2x + 1st stage of lowpass filter
	nIn = ReverbProcessPreFiltering1x(MixReverbBuffer, nSamples);
	nOut = nIn;
	// Main reverb processing: split into small chunks (needed for short reverb delays)
	// Reverb Input + Low-Pass stage #2 + Pre-diffusion
	if (nIn > 0) ProcessPreDelay(&g_RefDelay, MixReverbBuffer, nIn);
	// Process Reverb Reflections and Late Reverberation
	int32 *pRvbOut = MixReverbBuffer;
	uint32 nRvbSamples = nOut, nCount = 0;
	while (nRvbSamples > 0)
	{
		uint32 nPosRef = g_RefDelay.nRefOutPos & SNDMIX_REVERB_DELAY_MASK;
		uint32 nPosRvb = (nPosRef - g_LateReverb.nReverbDelay) & SNDMIX_REVERB_DELAY_MASK;
		uint32 nmax1 = (SNDMIX_REVERB_DELAY_MASK+1) - nPosRef;
		uint32 nmax2 = (SNDMIX_REVERB_DELAY_MASK+1) - nPosRvb;
		nmax1 = (nmax1 < nmax2) ? nmax1 : nmax2;
		uint32 n = nRvbSamples;
		if (n > nmax1) n = nmax1;
		if (n > 64) n = 64;
		// Reflections output + late reverb delay
		ProcessReflections(&g_RefDelay, &g_RefDelay.RefOut[nPosRef], pRvbOut, n);
		// Late Reverberation
		ProcessLateReverb(&g_LateReverb, &g_RefDelay.RefOut[nPosRvb], pRvbOut, n);
		// Update delay positions
		g_RefDelay.nRefOutPos = (g_RefDelay.nRefOutPos + n) & SNDMIX_REVERB_DELAY_MASK;
		g_RefDelay.nDelayPos = (g_RefDelay.nDelayPos + n) & SNDMIX_REFLECTIONS_DELAY_MASK;
		nCount += n*2;
		pRvbOut += n*2;
		nRvbSamples -= n;
	}
	// Adjust nDelayPos, in case nIn != nOut
	g_RefDelay.nDelayPos = (g_RefDelay.nDelayPos - nOut + nIn) & SNDMIX_REFLECTIONS_DELAY_MASK;
	// Upsample 2x
	ReverbProcessPostFiltering1x(MixReverbBuffer, MixSoundBuffer, nSamples);
	// Automatically shut down if needed
	if(gnReverbSend) gnReverbSamples = gnReverbDecaySamples; // reset decay counter
	else if(gnReverbSamples > nSamples) gnReverbSamples -= nSamples; // decay
	else // decayed
	{
		Shutdown();
		gnReverbSamples = 0;
	}
	gnReverbSend = 0; // no input data in MixReverbBuffer
}


void CReverb::ReverbDryMix(int32 * MPT_RESTRICT pDry, int32 * MPT_RESTRICT pWet, int lDryVol, uint32 nSamples)
{
	for (uint32 i=0; i<nSamples; i++)
	{
		pDry[i*2] += (pWet[i*2]>>4) * lDryVol;
		pDry[i*2+1] += (pWet[i*2+1]>>4) * lDryVol;
	}
}


uint32 CReverb::ReverbProcessPreFiltering2x(int32 * MPT_RESTRICT pWet, uint32 nSamples)
{
	uint32 nOutSamples = 0;
	int lowpass = g_RefDelay.nCoeffs.c.l;
	int y1_l = g_nLastRvbIn_yl, y1_r = g_nLastRvbIn_yr;
	uint32 n = nSamples;

	if (g_bLastInPresent)
	{
		int x1_l = g_nLastRvbIn_xl, x1_r = g_nLastRvbIn_xr;
		int x2_l = pWet[0], x2_r = pWet[1];
		x1_l = (x1_l+x2_l)>>13;
		x1_r = (x1_r+x2_r)>>13;
		y1_l = x1_l + (((x1_l - y1_l)*lowpass)>>15);
		y1_r = x1_r + (((x1_r - y1_r)*lowpass)>>15);
		pWet[0] = y1_l;
		pWet[1] = y1_r;
		pWet+=2;
		n--;
		nOutSamples = 1;
		g_bLastInPresent = false;
	}
	if (n & 1)
	{
		n--;
		g_nLastRvbIn_xl = pWet[n*2];
		g_nLastRvbIn_xr = pWet[n*2+1];
		g_bLastInPresent = true;
	}
	n >>= 1;
	for (uint32 i=0; i<n; i++)
	{
		int x1_l = pWet[i*4];
		int x2_l = pWet[i*4+2];
		x1_l = (x1_l+x2_l)>>13;
		int x1_r = pWet[i*4+1];
		int x2_r = pWet[i*4+3];
		x1_r = (x1_r+x2_r)>>13;
		y1_l = x1_l + (((x1_l - y1_l)*lowpass)>>15);
		y1_r = x1_r + (((x1_r - y1_r)*lowpass)>>15);
		pWet[i*2] = y1_l;
		pWet[i*2+1] = y1_r;
	}
	g_nLastRvbIn_yl = y1_l;
	g_nLastRvbIn_yr = y1_r;
	return nOutSamples + n;
}


uint32 CReverb::ReverbProcessPreFiltering1x(int32 * MPT_RESTRICT pWet, uint32 nSamples)
{
	int lowpass = g_RefDelay.nCoeffs.c.l;
	int y1_l = g_nLastRvbIn_yl, y1_r = g_nLastRvbIn_yr;

	for (uint32 i=0; i<nSamples; i++)
	{
		int x_l = pWet[i*2] >> 12;
		int x_r = pWet[i*2+1] >> 12;
		y1_l = x_l + (((x_l - y1_l)*lowpass)>>15);
		y1_r = x_r + (((x_r - y1_r)*lowpass)>>15);
		pWet[i*2] = y1_l;
		pWet[i*2+1] = y1_r;
	}
	g_nLastRvbIn_yl = y1_l;
	g_nLastRvbIn_yr = y1_r;
	return nSamples;
}


void CReverb::ReverbProcessPostFiltering2x(const int32 * MPT_RESTRICT pRvb, int32 * MPT_RESTRICT pDry, uint32 nSamples)
{
	uint32 n0 = nSamples, n;
	int x1_l = g_nLastRvbOut_xl, x1_r = g_nLastRvbOut_xr;

	if (g_bLastOutPresent)
	{
		pDry[0] += x1_l;
		pDry[1] += x1_r;
		pDry += 2;
		n0--;
		g_bLastOutPresent = false;
	}
	n  = n0 >> 1;
	for (uint32 i=0; i<n; i++)
	{
		int x_l = pRvb[i*2], x_r = pRvb[i*2+1];
		pDry[i*4] += (x_l + x1_l)>>1;
		pDry[i*4+1] += (x_r + x1_r)>>1;
		pDry[i*4+2] += x_l;
		pDry[i*4+3] += x_r;
		x1_l = x_l;
		x1_r = x_r;
	}
	if (n0 & 1)
	{
		int x_l = pRvb[n*2], x_r = pRvb[n*2+1];
		pDry[n*4] += (x_l + x1_l)>>1;
		pDry[n*4+1] += (x_r + x1_r)>>1;
		x1_l = x_l;
		x1_r = x_r;
		g_bLastOutPresent = true;
	}
	g_nLastRvbOut_xl = x1_l;
	g_nLastRvbOut_xr = x1_r;
}


#define DCR_AMOUNT		9

// Stereo Add + DC removal
void CReverb::ReverbProcessPostFiltering1x(const int32 * MPT_RESTRICT pRvb, int32 * MPT_RESTRICT pDry, uint32 nSamples)
{
#ifdef ENABLE_MMX
	if(GetProcSupport() & PROCSUPPORT_MMX)
	{
		__m64 nDCRRvb_Y1 = Load64MMX(gnDCRRvb_Y1);
		__m64 nDCRRvb_X1 = Load64MMX(gnDCRRvb_X1);
		__m64 in = _mm_set1_pi32(0);
		while(nSamples--)
		{
			in = Load64MMX(pRvb);
			pRvb += 2;
			// x(n-1) - x(n)
			__m64 diff = _mm_sub_pi32(nDCRRvb_X1, in);
			nDCRRvb_X1 = _mm_add_pi32(nDCRRvb_Y1, _mm_sub_pi32(_mm_srai_pi32(diff, DCR_AMOUNT + 1), diff));
			__m64 out = _mm_add_pi32(Load64MMX(pDry), nDCRRvb_X1);
			nDCRRvb_Y1 = _mm_sub_pi32(nDCRRvb_X1, _mm_srai_pi32(nDCRRvb_X1, DCR_AMOUNT));
			nDCRRvb_X1 = in;
			Store64MMX(pDry, out);
			pDry += 2;
		}
		Store64MMX(gnDCRRvb_X1, in);
		Store64MMX(gnDCRRvb_Y1, nDCRRvb_Y1);
		_mm_empty();
		return;
	}
#endif
	int32 X1L = gnDCRRvb_X1[0], X1R = gnDCRRvb_X1[1];
	int32 Y1L = gnDCRRvb_Y1[0], Y1R = gnDCRRvb_Y1[1];
	int32 inL = 0, inR = 0;
	while(nSamples--)
	{
		inL = pRvb[0];
		inR = pRvb[1];
		pRvb += 2;
		int32 outL = pDry[0], outR = pDry[1];

		// x(n-1) - x(n)
		X1L -= inL;
		X1R -= inR;
		X1L = X1L / (1 << (DCR_AMOUNT + 1)) - X1L;
		X1R = X1R / (1 << (DCR_AMOUNT + 1)) - X1R;
		Y1L += X1L;
		Y1R += X1R;
		// add to dry mix
		outL += Y1L;
		outR += Y1R;
		Y1L -= Y1L / (1 << DCR_AMOUNT);
		Y1R -= Y1R / (1 << DCR_AMOUNT);
		X1L = inL;
		X1R = inR;

		pDry[0] = outL;
		pDry[1] = outR;
		pDry += 2;
	}
	gnDCRRvb_Y1[0] = Y1L;
	gnDCRRvb_Y1[1] = Y1R;
	gnDCRRvb_X1[0] = inL;
	gnDCRRvb_X1[1] = inR;
}


void CReverb::ReverbDCRemoval(int32 * MPT_RESTRICT pBuffer, uint32 nSamples)
{
#ifdef ENABLE_MMX
	if(GetProcSupport() & PROCSUPPORT_MMX)
	{
		__m64 nDCRRvb_Y1 = Load64MMX(gnDCRRvb_Y1);
		__m64 nDCRRvb_X1 = Load64MMX(gnDCRRvb_X1);
		while(nSamples--)
		{
			__m64 in = Load64MMX(pBuffer);
			__m64 diff = _mm_sub_pi32(nDCRRvb_X1, in);
			__m64 out = _mm_add_pi32(nDCRRvb_Y1, _mm_sub_pi32(_mm_srai_pi32(diff, DCR_AMOUNT + 1), diff));
			Store64MMX(pBuffer, out);
			pBuffer += 2;
			nDCRRvb_Y1 = _mm_sub_pi32(out, _mm_srai_pi32(out, DCR_AMOUNT));
			nDCRRvb_X1 = in;
		}
		Store64MMX(gnDCRRvb_X1, nDCRRvb_X1);
		Store64MMX(gnDCRRvb_Y1, nDCRRvb_Y1);
		_mm_empty();
		return;
	}
#endif
	int32 X1L = gnDCRRvb_X1[0], X1R = gnDCRRvb_X1[1];
	int32 Y1L = gnDCRRvb_Y1[0], Y1R = gnDCRRvb_Y1[1];
	int32 inL = 0, inR = 0;
	while(nSamples--)
	{
		inL = pBuffer[0];
		inR = pBuffer[1];
		// x(n-1) - x(n)
		X1L -= inL;
		X1R -= inR;
		X1L = X1L / (1 << (DCR_AMOUNT + 1)) - X1L;
		X1R = X1R / (1 << (DCR_AMOUNT + 1)) - X1R;
		Y1L += X1L;
		Y1R += X1R;
		pBuffer[0] = Y1L;
		pBuffer[1] = Y1R;
		pBuffer += 2;
		Y1L -= Y1L / (1 << DCR_AMOUNT);
		Y1R -= Y1R / (1 << DCR_AMOUNT);
		X1L = inL;
		X1R = inR;
	}
	gnDCRRvb_Y1[0] = Y1L;
	gnDCRRvb_Y1[1] = Y1R;
	gnDCRRvb_X1[0] = inL;
	gnDCRRvb_X1[1] = inR;
}


//////////////////////////////////////////////////////////////////////////
//
// Pre-Delay:
//
// 1. Saturate and low-pass the reverb input (stage 2 of roomHF)
// 2. Process pre-diffusion
// 3. Insert the result in the reflections delay buffer
//

// Save some typing
static MPT_FORCEINLINE int32 Clamp16(int32 x) { return Clamp(x, int16_min, int16_max); }

void CReverb::ProcessPreDelay(SWRvbRefDelay * MPT_RESTRICT pPreDelay, const int32 * MPT_RESTRICT pIn, uint32 nSamples)
{
	uint32 preDifPos = pPreDelay->nPreDifPos;
	uint32 delayPos = pPreDelay->nDelayPos - 1;
#ifdef ENABLE_SSE2
	if(GetProcSupport() & PROCSUPPORT_SSE2)
	{
		__m128i coeffs = _mm_cvtsi32_si128(pPreDelay->nCoeffs.lr);
		__m128i history = _mm_cvtsi32_si128(pPreDelay->History.lr);
		__m128i preDifCoeffs = _mm_cvtsi32_si128(pPreDelay->nPreDifCoeffs.lr);
		while(nSamples--)
		{
			__m128i in32 = Load64SSE(pIn);					// 16-bit unsaturated reverb input [  r  |  l  ]
			__m128i inSat = _mm_packs_epi32(in32, in32);	// [ r | l | r | l ] (16-bit saturated)
			pIn += 2;
			// Low-pass
			__m128i lp = _mm_mulhi_epi16(_mm_subs_epi16(history, inSat), coeffs);
			__m128i preDif = _mm_cvtsi32_si128(pPreDelay->PreDifBuffer[preDifPos].lr);
			history = _mm_adds_epi16(_mm_adds_epi16(lp, lp), inSat);
			// Pre-Diffusion
			preDifPos = (preDifPos + 1) & SNDMIX_PREDIFFUSION_DELAY_MASK;
			delayPos = (delayPos + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
			__m128i preDif2 = _mm_subs_epi16(history, _mm_mulhi_epi16(preDif, preDifCoeffs));
			pPreDelay->PreDifBuffer[preDifPos].lr = _mm_cvtsi128_si32(preDif2);
			pPreDelay->RefDelayBuffer[delayPos].lr = _mm_cvtsi128_si32(_mm_adds_epi16(_mm_mulhi_epi16(preDifCoeffs, preDif2), preDif));
		}
		pPreDelay->nPreDifPos = preDifPos;
		pPreDelay->History.lr = _mm_cvtsi128_si32(history);
		return;
	}
#endif
#ifdef ENABLE_MMX
	if(GetProcSupport() & PROCSUPPORT_MMX)
	{
		__m64 coeffs = _mm_cvtsi32_si64(pPreDelay->nCoeffs.lr);
		__m64 history = _mm_cvtsi32_si64(pPreDelay->History.lr);
		__m64 preDifCoeffs = _mm_cvtsi32_si64(pPreDelay->nPreDifCoeffs.lr);
		while(nSamples--)
		{
			__m64 in32 = Load64MMX(pIn);				// 16-bit unsaturated reverb input [  r  |  l  ]
			__m64 inSat = _mm_packs_pi32(in32, in32);	// [ r | l | r | l ] (16-bit saturated)
			pIn += 2;
			// Low-pass
			__m64 lp = _mm_mulhi_pi16(_mm_subs_pi16(history, inSat), coeffs);
			__m64 preDif = _mm_cvtsi32_si64(pPreDelay->PreDifBuffer[preDifPos].lr);
			history = _mm_adds_pi16(_mm_adds_pi16(lp, lp), inSat);
			// Pre-Diffusion
			preDifPos = (preDifPos + 1) & SNDMIX_PREDIFFUSION_DELAY_MASK;
			delayPos = (delayPos + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
			__m64 preDif2 = _mm_subs_pi16(history, _mm_mulhi_pi16(preDif, preDifCoeffs));
			pPreDelay->PreDifBuffer[preDifPos].lr = _mm_cvtsi64_si32(preDif2);
			pPreDelay->RefDelayBuffer[delayPos].lr = _mm_cvtsi64_si32(_mm_adds_pi16(_mm_mulhi_pi16(preDifCoeffs, preDif2), preDif));
		}
		pPreDelay->nPreDifPos = preDifPos;
		pPreDelay->History.lr = _mm_cvtsi64_si32(history);
		_mm_empty();
		return;
	}
#endif
	const int32 coeffsL = pPreDelay->nCoeffs.c.l, coeffsR = pPreDelay->nCoeffs.c.r;
	const int32 preDifCoeffsL = pPreDelay->nPreDifCoeffs.c.l, preDifCoeffsR = pPreDelay->nPreDifCoeffs.c.r;
	int16 historyL = pPreDelay->History.c.l, historyR = pPreDelay->History.c.r;
	while(nSamples--)
	{
		int32 inL = Clamp16(pIn[0]);
		int32 inR = Clamp16(pIn[1]);
		pIn += 2;
		// Low-pass
		int32 lpL = (Clamp16(historyL - inL) * coeffsL) / 65536;
		int32 lpR = (Clamp16(historyR - inR) * coeffsR) / 65536;
		historyL = mpt::saturate_cast<int16>(Clamp16(lpL + lpL) + inL);
		historyR = mpt::saturate_cast<int16>(Clamp16(lpR + lpR) + inR);
		// Pre-Diffusion
		int32 preDifL = pPreDelay->PreDifBuffer[preDifPos].c.l;
		int32 preDifR = pPreDelay->PreDifBuffer[preDifPos].c.r;
		preDifPos = (preDifPos + 1) & SNDMIX_PREDIFFUSION_DELAY_MASK;
		delayPos = (delayPos + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
		int16 preDif2L = mpt::saturate_cast<int16>(historyL - preDifL * preDifCoeffsL / 65536);
		int16 preDif2R = mpt::saturate_cast<int16>(historyR - preDifR * preDifCoeffsR / 65536);
		pPreDelay->PreDifBuffer[preDifPos].c.l = preDif2L;
		pPreDelay->PreDifBuffer[preDifPos].c.r = preDif2R;
		pPreDelay->RefDelayBuffer[delayPos].c.l = mpt::saturate_cast<int16>(preDifCoeffsL * preDif2L / 65536 + preDifL);
		pPreDelay->RefDelayBuffer[delayPos].c.r = mpt::saturate_cast<int16>(preDifCoeffsR * preDif2R / 65536 + preDifR);
	}
	pPreDelay->nPreDifPos = preDifPos;
	pPreDelay->History.c.l = historyL;
	pPreDelay->History.c.r = historyR;
}


////////////////////////////////////////////////////////////////////
//
// ProcessReflections:
// First stage:
//	- process 4 reflections, output to pRefOut
//	- output results to pRefOut
// Second stage:
//	- process another 3 reflections
//	- sum with pRefOut
//	- apply reflections master gain and accumulate in the given output
//

void CReverb::ProcessReflections(SWRvbRefDelay * MPT_RESTRICT pPreDelay, LR16 * MPT_RESTRICT pRefOut, int32 * MPT_RESTRICT pOut, uint32 nSamples)
{
#ifdef ENABLE_SSE2
	if(GetProcSupport() & PROCSUPPORT_SSE2)
	{
		union
		{
			__m128i xmm;
			int16 i[8];
		} pos;
		const LR16 *refDelayBuffer = pPreDelay->RefDelayBuffer;
#define GETDELAY(x) static_cast<int16>(pPreDelay->Reflections[x].Delay)
		__m128i delayPos = _mm_set_epi16(GETDELAY(7), GETDELAY(6), GETDELAY(5), GETDELAY(4), GETDELAY(3), GETDELAY(2), GETDELAY(1), GETDELAY(0));
#undef GETDELAY
		delayPos = _mm_sub_epi16(_mm_set1_epi16(static_cast<int16>(pPreDelay->nDelayPos - 1)), delayPos);
		__m128i gain12 = _mm_unpacklo_epi64(Load64SSE(pPreDelay->Reflections[0].Gains), Load64SSE(pPreDelay->Reflections[1].Gains));
		__m128i gain34 = _mm_unpacklo_epi64(Load64SSE(pPreDelay->Reflections[2].Gains), Load64SSE(pPreDelay->Reflections[3].Gains));
		__m128i gain56 = _mm_unpacklo_epi64(Load64SSE(pPreDelay->Reflections[4].Gains), Load64SSE(pPreDelay->Reflections[5].Gains));
		__m128i gain78 = _mm_unpacklo_epi64(Load64SSE(pPreDelay->Reflections[6].Gains), Load64SSE(pPreDelay->Reflections[7].Gains));
		// For 28-bit final output: 16+15-3 = 28
		__m128i refGain = _mm_srai_epi32(_mm_set_epi32(0, 0, pPreDelay->ReflectionsGain.c.r, pPreDelay->ReflectionsGain.c.l), 3);
		__m128i delayInc = _mm_set1_epi16(1), delayMask = _mm_set1_epi16(SNDMIX_REFLECTIONS_DELAY_MASK);
		while(nSamples--)
		{
			delayPos = _mm_and_si128(_mm_add_epi16(delayInc, delayPos), delayMask);
			_mm_storeu_si128(&pos.xmm, delayPos);
			__m128i ref12 = _mm_set_epi32(refDelayBuffer[pos.i[1]].lr, refDelayBuffer[pos.i[1]].lr, refDelayBuffer[pos.i[0]].lr, refDelayBuffer[pos.i[0]].lr);
			__m128i ref34 = _mm_set_epi32(refDelayBuffer[pos.i[3]].lr, refDelayBuffer[pos.i[3]].lr, refDelayBuffer[pos.i[2]].lr, refDelayBuffer[pos.i[2]].lr);
			__m128i ref56 = _mm_set_epi32(refDelayBuffer[pos.i[5]].lr, refDelayBuffer[pos.i[5]].lr, refDelayBuffer[pos.i[4]].lr, refDelayBuffer[pos.i[4]].lr);
			__m128i ref78 = _mm_set_epi32(0,                           0,                           refDelayBuffer[pos.i[6]].lr, refDelayBuffer[pos.i[6]].lr);
			// First stage
			__m128i refOut1 = _mm_add_epi32(_mm_madd_epi16(ref12, gain12), _mm_madd_epi16(ref34, gain34));
			refOut1 = _mm_srai_epi32(_mm_add_epi32(refOut1, _mm_shuffle_epi32(refOut1, _MM_SHUFFLE(1, 0, 3, 2))), 15);

			// Second stage
			__m128i refOut2 = _mm_add_epi32(_mm_madd_epi16(ref56, gain56), _mm_madd_epi16(ref78, gain78));
			refOut2 = _mm_srai_epi32(_mm_add_epi32(refOut2, _mm_shuffle_epi32(refOut2, _MM_SHUFFLE(1, 0, 3, 2))), 15);

			// Saturate to 16-bit and sum stages
			__m128i refOut = _mm_adds_epi16(_mm_packs_epi32(refOut1, refOut1), _mm_packs_epi32(refOut2, refOut2));
			pRefOut->lr = _mm_cvtsi128_si32(refOut);
			pRefOut++;

			__m128i out = _mm_madd_epi16(_mm_unpacklo_epi16(refOut, refOut), refGain);	// Apply reflections gain
			// At this, point, this is the only output of the reverb
			Store64SSE(pOut, out);
			pOut += 2;
		}
		return;
	}
#endif
#ifdef ENABLE_MMX
	if(GetProcSupport() & PROCSUPPORT_MMX)
	{
		// First stage
		uint32 numSamples = nSamples;
		const LR16 *refDelayBuffer = pPreDelay->RefDelayBuffer;
		int pos1 = pPreDelay->nDelayPos - pPreDelay->Reflections[0].Delay - 1;
		int pos2 = pPreDelay->nDelayPos - pPreDelay->Reflections[1].Delay - 1;
		int pos3 = pPreDelay->nDelayPos - pPreDelay->Reflections[2].Delay - 1;
		int pos4 = pPreDelay->nDelayPos - pPreDelay->Reflections[3].Delay - 1;
		__m64 gain1 = Load64MMX(pPreDelay->Reflections[0].Gains);
		__m64 gain2 = Load64MMX(pPreDelay->Reflections[1].Gains);
		__m64 gain3 = Load64MMX(pPreDelay->Reflections[2].Gains);
		__m64 gain4 = Load64MMX(pPreDelay->Reflections[3].Gains);
		while(numSamples--)
		{
			pos1 = (pos1 + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
			pos2 = (pos2 + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
			pos3 = (pos3 + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
			pos4 = (pos4 + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
			__m64 ref1 = _mm_cvtsi32_si64(refDelayBuffer[pos1].lr);	// [0 | 0 | r | l ]
			__m64 ref2 = _mm_cvtsi32_si64(refDelayBuffer[pos2].lr);
			__m64 ref3 = _mm_cvtsi32_si64(refDelayBuffer[pos3].lr);
			__m64 ref4 = _mm_cvtsi32_si64(refDelayBuffer[pos4].lr);
			__m64 refOut = _mm_srai_pi32(_mm_add_pi32(
				_mm_add_pi32(_mm_madd_pi16(_mm_unpacklo_pi32(ref1, ref1), gain1), _mm_madd_pi16(_mm_unpacklo_pi32(ref2, ref2), gain2)),
				_mm_add_pi32(_mm_madd_pi16(_mm_unpacklo_pi32(ref3, ref3), gain3), _mm_madd_pi16(_mm_unpacklo_pi32(ref4, ref4), gain4))),
				15);
			pRefOut->lr = _mm_cvtsi64_si32(_mm_packs_pi32(refOut, refOut));
			pRefOut++;
		}

		// Second stage
		numSamples = nSamples;
		pRefOut -= nSamples;

		__m64 refGain = _mm_unpacklo_pi16(_mm_cvtsi32_si64(pPreDelay->ReflectionsGain.lr), _mm_cvtsi32_si64(0));	// [0 | g_r | 0 | g_l]
		refGain = _mm_srai_pi32(refGain, 3);	// For 28-bit final output: 16+15-3 = 28
		int pos5 = pPreDelay->nDelayPos - pPreDelay->Reflections[4].Delay - 1;
		int pos6 = pPreDelay->nDelayPos - pPreDelay->Reflections[5].Delay - 1;
		int pos7 = pPreDelay->nDelayPos - pPreDelay->Reflections[6].Delay - 1;
		__m64 gain5 = Load64MMX(pPreDelay->Reflections[4].Gains);
		__m64 gain6 = Load64MMX(pPreDelay->Reflections[5].Gains);
		__m64 gain7 = Load64MMX(pPreDelay->Reflections[6].Gains);
		while(numSamples--)
		{
			pos5 = (pos5 + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
			pos6 = (pos6 + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
			pos7 = (pos7 + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
			__m64 ref5 = _mm_cvtsi32_si64(refDelayBuffer[pos5].lr);	// [0 | 0 | r | l ]
			__m64 ref6 = _mm_cvtsi32_si64(refDelayBuffer[pos6].lr);
			__m64 ref7 = _mm_cvtsi32_si64(refDelayBuffer[pos7].lr);
			__m64 refPrev = _mm_cvtsi32_si64(pRefOut->lr);	// output of previous reflections
			__m64 refOut = _mm_srai_pi32(_mm_add_pi32(
				_mm_add_pi32(_mm_madd_pi16(_mm_unpacklo_pi32(ref5, ref5), gain5), _mm_madd_pi16(_mm_unpacklo_pi32(ref7, ref7), gain7)),
				_mm_madd_pi16(_mm_unpacklo_pi32(ref6, ref6), gain6)),
				15);
			refOut = _mm_adds_pi16(_mm_packs_pi32(refOut, refOut), refPrev);
			pRefOut->lr = _mm_cvtsi64_si32(refOut);	// late reverb stereo input
			pRefOut++;
			__m64 out = _mm_madd_pi16(_mm_unpacklo_pi16(refOut, refOut), refGain);	// Apply reflections gain
			// At this, point, this is the only output of the reverb
			Store64MMX(pOut, out);
			pOut += 2;
		}
		_mm_empty();
		return;
	}
#endif
	int pos[7];
	for(int i = 0; i < 7; i++)
		pos[i] = pPreDelay->nDelayPos - pPreDelay->Reflections[i].Delay - 1;
	// For 28-bit final output: 16+15-3 = 28
	int16 refGain = pPreDelay->ReflectionsGain.c.l / (1 << 3);
	while(nSamples--)
	{
		// First stage
		int32 refOutL = 0, refOutR = 0;
		for(int i = 0; i < 4; i++)
		{
			pos[i] = (pos[i] + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
			int16 refL = pPreDelay->RefDelayBuffer[pos[i]].c.l, refR = pPreDelay->RefDelayBuffer[pos[i]].c.r;
			refOutL += refL * pPreDelay->Reflections[i].Gains[0].c.l + refR * pPreDelay->Reflections[i].Gains[0].c.r;
			refOutR += refL * pPreDelay->Reflections[i].Gains[1].c.l + refR * pPreDelay->Reflections[i].Gains[1].c.r;
		}
		int16 stage1l = mpt::saturate_cast<int16>(refOutL / (1 << 15));
		int16 stage1r = mpt::saturate_cast<int16>(refOutR / (1 << 15));
		// Second stage
		refOutL = 0;
		refOutR = 0;
		for(int i = 4; i < 7; i++)
		{
			pos[i] = (pos[i] + 1) & SNDMIX_REFLECTIONS_DELAY_MASK;
			int16 refL = pPreDelay->RefDelayBuffer[pos[i]].c.l, refR = pPreDelay->RefDelayBuffer[pos[i]].c.r;
			refOutL += refL * pPreDelay->Reflections[i].Gains[0].c.l + refR * pPreDelay->Reflections[i].Gains[0].c.r;
			refOutR += refL * pPreDelay->Reflections[i].Gains[1].c.l + refR * pPreDelay->Reflections[i].Gains[1].c.r;
		}
		pOut[0] = (pRefOut->c.l = mpt::saturate_cast<int16>(stage1l + refOutL / (1 << 15))) * refGain;
		pOut[1] = (pRefOut->c.r = mpt::saturate_cast<int16>(stage1r + refOutR / (1 << 15))) * refGain;
		pRefOut++;
		pOut += 2;
	}
}


//////////////////////////////////////////////////////////////////////////
//
// Late reverberation (with SW reflections)
//

void CReverb::ProcessLateReverb(SWLateReverb * MPT_RESTRICT pReverb, LR16 * MPT_RESTRICT pRefOut, int32 * MPT_RESTRICT pMixOut, uint32 nSamples)
{
	// Calculate delay line offset from current delay position
	#define DELAY_OFFSET(x) ((delayPos - (x)) & RVBDLY_MASK)

#ifdef ENABLE_SSE2
	if(GetProcSupport() & PROCSUPPORT_SSE2)
	{
		int delayPos = pReverb->nDelayPos & RVBDLY_MASK;
		__m128i rvbOutGains = Load64SSE(pReverb->RvbOutGains);
		__m128i difCoeffs = Load64SSE(pReverb->nDifCoeffs);
		__m128i decayLP = Load64SSE(pReverb->nDecayLP);
		__m128i lpHistory = Load64SSE(pReverb->LPHistory);
		while(nSamples--)
		{
			__m128i refIn = _mm_cvtsi32_si128(pRefOut->lr);	// 16-bit stereo input
			pRefOut++;

			__m128i delay2 = _mm_unpacklo_epi32(
				_mm_cvtsi32_si128(pReverb->Delay2[DELAY_OFFSET(RVBDLY2L_LEN)].lr),
				_mm_cvtsi32_si128(pReverb->Delay2[DELAY_OFFSET(RVBDLY2R_LEN)].lr));

			// Unsigned to avoid sign extension
			uint16 diff1L = pReverb->Diffusion1[DELAY_OFFSET(RVBDIF1L_LEN)].c.l;
			uint16 diff1R = pReverb->Diffusion1[DELAY_OFFSET(RVBDIF1R_LEN)].c.r;
			int32 diffusion1 = diff1L | (diff1R << 16);	// diffusion1 history

			uint16 diff2L = pReverb->Diffusion2[DELAY_OFFSET(RVBDIF2L_LEN)].c.l;
			uint16 diff2R = pReverb->Diffusion2[DELAY_OFFSET(RVBDIF2R_LEN)].c.r;
			int32 diffusion2 = diff2L | (diff2R << 16);	// diffusion2 history

			__m128i lpDecay = _mm_mulhi_epi16(_mm_subs_epi16(lpHistory, delay2), decayLP);
			lpHistory = _mm_adds_epi16(_mm_adds_epi16(lpDecay, lpDecay), delay2);	// Low-passed decay

			// Apply decay gain
			__m128i histDecay = _mm_srai_epi32(_mm_madd_epi16(Load64SSE(pReverb->nDecayDC), lpHistory), 15);
			__m128i histDecayPacked = _mm_shuffle_epi32(_mm_packs_epi32(histDecay, histDecay), _MM_SHUFFLE(2, 0, 2, 0));
			__m128i histDecayIn = _mm_adds_epi16(_mm_shuffle_epi32(_mm_packs_epi32(histDecay, histDecay), _MM_SHUFFLE(2, 0, 2, 0)), _mm_srai_epi16(_mm_unpacklo_epi32(refIn, refIn), 2));
			__m128i histDecayInDiff = _mm_subs_epi16(histDecayIn, _mm_mulhi_epi16(_mm_cvtsi32_si128(diffusion1), difCoeffs));
			pReverb->Diffusion1[delayPos].lr = _mm_cvtsi128_si32(histDecayInDiff);

			__m128i delay1Out = _mm_adds_epi16(_mm_mulhi_epi16(difCoeffs, histDecayInDiff), _mm_cvtsi32_si128(diffusion1));
			// Insert the diffusion output in the reverb delay line
			pReverb->Delay1[delayPos].lr = _mm_cvtsi128_si32(delay1Out);
			__m128i histDecayInDelay = _mm_adds_epi16(histDecayIn, _mm_unpacklo_epi32(delay1Out, delay1Out));

			// Input to second diffuser
			__m128i delay1 = _mm_unpacklo_epi32(
				_mm_cvtsi32_si128(pReverb->Delay1[DELAY_OFFSET(RVBDLY1L_LEN)].lr),
				_mm_cvtsi32_si128(pReverb->Delay1[DELAY_OFFSET(RVBDLY1R_LEN)].lr));

			__m128i delay1Gains = _mm_srai_epi32(_mm_madd_epi16(delay1, Load64SSE(pReverb->Dif2InGains)), 15);
			__m128i delay1GainsSat = _mm_shuffle_epi32(_mm_packs_epi32(delay1Gains, delay1Gains), _MM_SHUFFLE(2, 0, 2, 0));
			__m128i histDelay1 = _mm_subs_epi16(_mm_adds_epi16(histDecayInDelay, delay1), delay1GainsSat);	// accumulate with reverb output
			__m128i diff2out = _mm_subs_epi16(delay1GainsSat, _mm_mulhi_epi16(_mm_cvtsi32_si128(diffusion2), difCoeffs));
			__m128i diff2outCoeffs = _mm_mulhi_epi16(difCoeffs, diff2out);
			pReverb->Diffusion2[delayPos].lr = _mm_cvtsi128_si32(diff2out);

			__m128i mixOut = Load64SSE(pMixOut);
			__m128i delay2out = _mm_adds_epi16(diff2outCoeffs, _mm_cvtsi32_si128(diffusion2));
			pReverb->Delay2[delayPos].lr = _mm_cvtsi128_si32(delay2out);
			delayPos = (delayPos + 1) & RVBDLY_MASK;
			// Accumulate with reverb output
			__m128i out = _mm_add_epi32(_mm_madd_epi16(_mm_adds_epi16(histDelay1, delay2out), rvbOutGains), mixOut);
			Store64SSE(pMixOut, out);
			pMixOut += 2;
		}
		Store64SSE(pReverb->LPHistory, lpHistory);
		pReverb->nDelayPos = delayPos;
		return;
	}
#endif
#ifdef ENABLE_MMX
	if(GetProcSupport() & PROCSUPPORT_MMX)
	{
		int delayPos = pReverb->nDelayPos & RVBDLY_MASK;
		__m64 rvbOutGains = Load64MMX(pReverb->RvbOutGains);
		__m64 difCoeffs = Load64MMX(pReverb->nDifCoeffs);
		__m64 decayLP = Load64MMX(pReverb->nDecayLP);
		__m64 lpHistory = Load64MMX(pReverb->LPHistory);
		while(nSamples--)
		{
			__m64 refIn = _mm_cvtsi32_si64(pRefOut->lr);	// 16-bit stereo input
			pRefOut++;

			__m64 delay2 = _mm_unpacklo_pi32(
				_mm_cvtsi32_si64(pReverb->Delay2[DELAY_OFFSET(RVBDLY2L_LEN)].lr),
				_mm_cvtsi32_si64(pReverb->Delay2[DELAY_OFFSET(RVBDLY2R_LEN)].lr));

			// Unsigned to avoid sign extension
			uint16 diff1L = pReverb->Diffusion1[DELAY_OFFSET(RVBDIF1L_LEN)].c.l;
			uint16 diff1R = pReverb->Diffusion1[DELAY_OFFSET(RVBDIF1R_LEN)].c.r;
			int32 diffusion1 = diff1L | (diff1R << 16);	// diffusion1 history

			uint16 diff2L = pReverb->Diffusion2[DELAY_OFFSET(RVBDIF2L_LEN)].c.l;
			uint16 diff2R = pReverb->Diffusion2[DELAY_OFFSET(RVBDIF2R_LEN)].c.r;
			int32 diffusion2 = diff2L | (diff2R << 16);	// diffusion2 history

			__m64 lpDecay = _mm_mulhi_pi16(_mm_subs_pi16(lpHistory, delay2), decayLP);
			lpHistory = _mm_adds_pi16(_mm_adds_pi16(lpDecay, lpDecay), delay2);	// Low-passed decay

			// Apply decay gain
			__m64 histDecay = _mm_srai_pi32(_mm_madd_pi16(Load64MMX(pReverb->nDecayDC), lpHistory), 15);
			__m64 histDecayIn = _mm_adds_pi16(_mm_packs_pi32(histDecay, histDecay), _mm_srai_pi16(_mm_unpacklo_pi32(refIn, refIn), 2));
			__m64 histDecayInDiff = _mm_subs_pi16(histDecayIn, _mm_mulhi_pi16(_mm_cvtsi32_si64(diffusion1), difCoeffs));
			pReverb->Diffusion1[delayPos].lr = _mm_cvtsi64_si32(histDecayInDiff);

			__m64 delay1Out = _mm_adds_pi16(_mm_mulhi_pi16(difCoeffs, histDecayInDiff), _mm_cvtsi32_si64(diffusion1));
			// Insert the diffusion output in the reverb delay line
			pReverb->Delay1[delayPos].lr = _mm_cvtsi64_si32(delay1Out);
			__m64 histDecayInDelay = _mm_adds_pi16(histDecayIn, _mm_unpacklo_pi32(delay1Out, delay1Out));

			// Input to second diffuser
			__m64 delay1 = _mm_unpacklo_pi32(
				_mm_cvtsi32_si64(pReverb->Delay1[DELAY_OFFSET(RVBDLY1L_LEN)].lr),
				_mm_cvtsi32_si64(pReverb->Delay1[DELAY_OFFSET(RVBDLY1R_LEN)].lr));

			__m64 delay1Gains = _mm_srai_pi32(_mm_madd_pi16(delay1, Load64MMX(pReverb->Dif2InGains)), 15);
			__m64 delay1GainsSat = _mm_packs_pi32(delay1Gains, delay1Gains);
			__m64 histDelay1 = _mm_subs_pi16(_mm_adds_pi16(histDecayInDelay, delay1), delay1GainsSat);	// accumulate with reverb output
			__m64 diff2out = _mm_subs_pi16(delay1GainsSat, _mm_mulhi_pi16(_mm_cvtsi32_si64(diffusion2), difCoeffs));
			__m64 diff2outCoeffs = _mm_mulhi_pi16(difCoeffs, diff2out);
			pReverb->Diffusion2[delayPos].lr = _mm_cvtsi64_si32(diff2out);

			__m64 mixOut = Load64MMX(pMixOut);
			__m64 delay2out = _mm_adds_pi16(diff2outCoeffs, _mm_cvtsi32_si64(diffusion2));
			pReverb->Delay2[delayPos].lr = _mm_cvtsi64_si32(delay2out);
			delayPos = (delayPos + 1) & RVBDLY_MASK;
			// Accumulate with reverb output
			__m64 out = _mm_add_pi32(_mm_madd_pi16(_mm_adds_pi16(histDelay1, delay2out), rvbOutGains), mixOut);
			Store64MMX(pMixOut, out);
			pMixOut += 2;
		}
		Store64MMX(&pReverb->LPHistory[0].lr, lpHistory);
		pReverb->nDelayPos = delayPos;
		_mm_empty();
		return;
	}
#endif
	int delayPos = pReverb->nDelayPos & RVBDLY_MASK;
	while(nSamples--)
	{
		int16 refInL = pRefOut->c.l, refInR = pRefOut->c.r;
		pRefOut++;

		int32 delay2LL = pReverb->Delay2[DELAY_OFFSET(RVBDLY2L_LEN)].c.l, delay2LR = pReverb->Delay2[DELAY_OFFSET(RVBDLY2L_LEN)].c.r;
		int32 delay2RL = pReverb->Delay2[DELAY_OFFSET(RVBDLY2R_LEN)].c.l, delay2RR = pReverb->Delay2[DELAY_OFFSET(RVBDLY2R_LEN)].c.r;

		int32 diff1L = pReverb->Diffusion1[DELAY_OFFSET(RVBDIF1L_LEN)].c.l;
		int32 diff1R = pReverb->Diffusion1[DELAY_OFFSET(RVBDIF1R_LEN)].c.r;

		int32 diff2L = pReverb->Diffusion2[DELAY_OFFSET(RVBDIF2L_LEN)].c.l;
		int32 diff2R = pReverb->Diffusion2[DELAY_OFFSET(RVBDIF2R_LEN)].c.r;

		int32 lpDecayLL = Clamp16(pReverb->LPHistory[0].c.l - delay2LL) * pReverb->nDecayLP[0].c.l / 65536;
		int32 lpDecayLR = Clamp16(pReverb->LPHistory[0].c.r - delay2LR) * pReverb->nDecayLP[0].c.r / 65536;
		int32 lpDecayRL = Clamp16(pReverb->LPHistory[1].c.l - delay2RL) * pReverb->nDecayLP[1].c.l / 65536;
		int32 lpDecayRR = Clamp16(pReverb->LPHistory[1].c.r - delay2RR) * pReverb->nDecayLP[1].c.r / 65536;
		// Low-passed decay
		pReverb->LPHistory[0].c.l = mpt::saturate_cast<int16>(Clamp16(lpDecayLL + lpDecayLL) + delay2LL);
		pReverb->LPHistory[0].c.r = mpt::saturate_cast<int16>(Clamp16(lpDecayLR + lpDecayLR) + delay2LR);
		pReverb->LPHistory[1].c.l = mpt::saturate_cast<int16>(Clamp16(lpDecayRL + lpDecayRL) + delay2RL);
		pReverb->LPHistory[1].c.r = mpt::saturate_cast<int16>(Clamp16(lpDecayRR + lpDecayRR) + delay2RR);

		// Apply decay gain
		int32 histDecayL = Clamp16((int32)pReverb->nDecayDC[0].c.l * pReverb->LPHistory[0].c.l / (1 << 15));
		int32 histDecayR = Clamp16((int32)pReverb->nDecayDC[1].c.r * pReverb->LPHistory[1].c.r / (1 << 15));
		int32 histDecayInL = Clamp16(histDecayL + refInL / 4);
		int32 histDecayInR = Clamp16(histDecayR + refInR / 4);
		int32 histDecayInDiffL = Clamp16(histDecayInL - diff1L * pReverb->nDifCoeffs[0].c.l / 65536);
		int32 histDecayInDiffR = Clamp16(histDecayInR - diff1R * pReverb->nDifCoeffs[0].c.r / 65536);
		pReverb->Diffusion1[delayPos].c.l = static_cast<int16>(histDecayInDiffL);
		pReverb->Diffusion1[delayPos].c.r = static_cast<int16>(histDecayInDiffR);

		int32 delay1L = Clamp16(pReverb->nDifCoeffs[0].c.l * histDecayInDiffL / 65536 + diff1L);
		int32 delay1R = Clamp16(pReverb->nDifCoeffs[0].c.r * histDecayInDiffR / 65536 + diff1R);
		// Insert the diffusion output in the reverb delay line
		pReverb->Delay1[delayPos].c.l = static_cast<int16>(delay1L);
		pReverb->Delay1[delayPos].c.r = static_cast<int16>(delay1R);
		int32 histDecayInDelayL = Clamp16(histDecayInL + delay1L);
		int32 histDecayInDelayR = Clamp16(histDecayInR + delay1R);

		// Input to second diffuser
		int32 delay1LL = pReverb->Delay1[DELAY_OFFSET(RVBDLY1L_LEN)].c.l, delay1LR = pReverb->Delay1[DELAY_OFFSET(RVBDLY1L_LEN)].c.r;
		int32 delay1RL = pReverb->Delay1[DELAY_OFFSET(RVBDLY1R_LEN)].c.l, delay1RR = pReverb->Delay1[DELAY_OFFSET(RVBDLY1R_LEN)].c.r;

		int32 delay1GainsL = Clamp16((delay1LL * pReverb->Dif2InGains[0].c.l + delay1LR * pReverb->Dif2InGains[0].c.r) / (1 << 15));
		int32 delay1GainsR = Clamp16((delay1RL * pReverb->Dif2InGains[1].c.l + delay1RR * pReverb->Dif2InGains[1].c.r) / (1 << 15));

		// accumulate with reverb output
		int32 histDelay1LL = Clamp16(Clamp16(histDecayInDelayL + delay1LL) - delay1GainsL);
		int32 histDelay1LR = Clamp16(Clamp16(histDecayInDelayR + delay1LR) - delay1GainsR);
		int32 histDelay1RL = Clamp16(Clamp16(histDecayInDelayL + delay1RL) - delay1GainsL);
		int32 histDelay1RR = Clamp16(Clamp16(histDecayInDelayR + delay1RR) - delay1GainsR);
		int32 diff2outL = Clamp16(delay1GainsL - diff2L * pReverb->nDifCoeffs[0].c.l / 65536);
		int32 diff2outR = Clamp16(delay1GainsR - diff2R * pReverb->nDifCoeffs[0].c.r / 65536);
		int32 diff2outCoeffsL = pReverb->nDifCoeffs[0].c.l * diff2outL / 65536;
		int32 diff2outCoeffsR = pReverb->nDifCoeffs[0].c.r * diff2outR / 65536;
		pReverb->Diffusion2[delayPos].c.l = static_cast<int16>(diff2outL);
		pReverb->Diffusion2[delayPos].c.r = static_cast<int16>(diff2outR);

		int32 delay2outL = Clamp16(diff2outCoeffsL + diff2L);
		int32 delay2outR = Clamp16(diff2outCoeffsR + diff2R);
		pReverb->Delay2[delayPos].c.l = static_cast<int16>(delay2outL);
		pReverb->Delay2[delayPos].c.r = static_cast<int16>(delay2outR);
		delayPos = (delayPos + 1) & RVBDLY_MASK;
		// Accumulate with reverb output
		pMixOut[0] += Clamp16(histDelay1LL + delay2outL) * pReverb->RvbOutGains[0].c.l + Clamp16(histDelay1LR + delay2outR) * pReverb->RvbOutGains[0].c.r;
		pMixOut[1] += Clamp16(histDelay1RL + Clamp16(diff2outCoeffsL)) * pReverb->RvbOutGains[1].c.l + Clamp16(histDelay1RR + Clamp16(diff2outCoeffsR)) * pReverb->RvbOutGains[1].c.r;
		pMixOut += 2;
	}
	pReverb->nDelayPos = delayPos;

	#undef DELAY_OFFSET
}


#else


MPT_MSVC_WORKAROUND_LNK4221(Reverb)


#endif // NO_REVERB


OPENMPT_NAMESPACE_END

