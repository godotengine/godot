/*
 * Resampler.h
 * -----------
 * Purpose: Holds the tables for all available resamplers.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once


#include "WindowedFIR.h"
#include "Mixer.h"
#include "MixerSettings.h"


OPENMPT_NAMESPACE_BEGIN


#ifdef LIBOPENMPT_BUILD
// All these optimizations are not applicable to the tracker
// because cutoff and firtype are configurable there.

// Cache resampler tables across resampler object creation.
// A C++11-style function-static singleton is holding the cached values.
#define MPT_RESAMPLER_TABLES_CACHED

// Prime the tables cache when the library is loaded.
// Caching gets triggered via a global object that primes the cache during
//  construction.
// This is only really useful with MPT_RESAMPLER_TABLES_CACHED.
#define MPT_RESAMPLER_TABLES_CACHED_ONSTARTUP

#endif // LIBOPENMPT_BUILD


#define SINC_WIDTH       8

#define SINC_PHASES_BITS 12
#define SINC_PHASES      (1<<SINC_PHASES_BITS)

#ifdef MPT_INTMIXER
typedef int16 SINC_TYPE;
#define SINC_QUANTSHIFT 15
#else
typedef mixsample_t SINC_TYPE;
#endif // MPT_INTMIXER

#define SINC_MASK (SINC_PHASES-1)
STATIC_ASSERT((SINC_MASK & 0xffff) == SINC_MASK); // exceeding fractional freq


class CResamplerSettings
{
public:
	ResamplingMode SrcMode;
	double gdWFIRCutoff;
	uint8 gbWFIRType;
	bool emulateAmiga;
public:
	CResamplerSettings()
	{
		SrcMode = SRCMODE_POLYPHASE;
		gdWFIRCutoff = 0.97;
		gbWFIRType = WFIR_KAISER4T;
		emulateAmiga = false;
	}
	bool operator == (const CResamplerSettings &cmp) const
	{
		return SrcMode == cmp.SrcMode && gdWFIRCutoff == cmp.gdWFIRCutoff && gbWFIRType == cmp.gbWFIRType && emulateAmiga == cmp.emulateAmiga;
	}
	bool operator != (const CResamplerSettings &cmp) const { return !(*this == cmp); }
};


class CResampler
{
public:
	CResamplerSettings m_Settings;
	CWindowedFIR m_WindowedFIR;
	static const int16 FastSincTable[256 * 4];

#ifdef MODPLUG_TRACKER
	static bool StaticTablesInitialized;
	#define RESAMPLER_TABLE static
#else
	// no global data which has to be initialized by hand in the library
	#define RESAMPLER_TABLE 
#endif // MODPLUG_TRACKER

	RESAMPLER_TABLE SINC_TYPE gKaiserSinc[SINC_PHASES * 8];     // Upsampling
	RESAMPLER_TABLE SINC_TYPE gDownsample13x[SINC_PHASES * 8];  // Downsample 1.333x
	RESAMPLER_TABLE SINC_TYPE gDownsample2x[SINC_PHASES * 8];   // Downsample 2x

#ifndef MPT_INTMIXER
	RESAMPLER_TABLE mixsample_t FastSincTablef[256 * 4];	// Cubic spline LUT
	RESAMPLER_TABLE mixsample_t LinearTablef[256];		// Linear interpolation LUT
#endif // !defined(MPT_INTMIXER)

#undef RESAMPLER_TABLE

private:
	CResamplerSettings m_OldSettings;
public:
	CResampler(bool fresh_generate=false)
	{
		if(fresh_generate)
		{
			InitializeTablesFromScratch(true);
		} else
		{
			InitializeTables();
		}
	}
	void InitializeTables()
	{
		#if defined(MPT_RESAMPLER_TABLES_CACHED)
			InitializeTablesFromCache();
		#else
			InitializeTablesFromScratch(true);
		#endif
	}
	void UpdateTables()
	{
		InitializeTablesFromScratch(false);
	}
	~CResampler() {}
	bool IsHQ() const { return m_Settings.SrcMode >= SRCMODE_SPLINE && m_Settings.SrcMode < SRCMODE_DEFAULT; }
private:
	void InitFloatmixerTables();
	void InitializeTablesFromScratch(bool force=false);
#ifdef MPT_RESAMPLER_TABLES_CACHED
	void InitializeTablesFromCache();
#endif
};


OPENMPT_NAMESPACE_END
