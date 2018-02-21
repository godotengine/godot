/*
 * DLSBank.h
 * ---------
 * Purpose: Sound bank loading.
 * Notes  : Supported sound bank types: DLS (including embedded DLS in MSS & RMI), SF2
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

OPENMPT_NAMESPACE_BEGIN
class CSoundFile;
OPENMPT_NAMESPACE_END
#include "Snd_defs.h"

OPENMPT_NAMESPACE_BEGIN

#ifdef MODPLUG_TRACKER


#define DLSMAXREGIONS		128

// Region Flags
#define DLSREGION_KEYGROUPMASK		0x0F
#define DLSREGION_OVERRIDEWSMP		0x10
#define DLSREGION_PINGPONGLOOP		0x20
#define DLSREGION_SAMPLELOOP		0x40
#define DLSREGION_SELFNONEXCLUSIVE	0x80
#define DLSREGION_SUSTAINLOOP		0x100

struct DLSREGION
{
	uint32 ulLoopStart;
	uint32 ulLoopEnd;
	uint16 nWaveLink;
	uint16 uPercEnv;
	uint16 usVolume;	// 0..256
	uint16 fuOptions;	// flags + key group
	int16  sFineTune;	// +128 = +1 semitone
	uint8  uKeyMin;
	uint8  uKeyMax;
	uint8  uUnityNote;
};

struct DLSENVELOPE
{
	// Volume Envelope
	uint16 wVolAttack;		// Attack Time: 0-1000, 1 = 20ms (1/50s) -> [0-20s]
	uint16 wVolDecay;		// Decay Time: 0-1000, 1 = 20ms (1/50s) -> [0-20s]
	uint16 wVolRelease;		// Release Time: 0-1000, 1 = 20ms (1/50s) -> [0-20s]
	uint8 nVolSustainLevel;	// Sustain Level: 0-128, 128=100%	
	uint8 nDefPan;			// Default Pan
};

// Special Bank bits
#define F_INSTRUMENT_DRUMS		0x80000000

struct DLSINSTRUMENT
{
	uint32 ulBank, ulInstrument;
	uint32 nRegions, nMelodicEnv;
	DLSREGION Regions[DLSMAXREGIONS];
	char szName[32];
	// SF2 stuff (DO NOT USE! -> used internally by the SF2 loader)
	uint16 wPresetBagNdx, wPresetBagNum;
};

struct DLSSAMPLEEX
{
	char   szName[20];
	uint32 dwLen;
	uint32 dwStartloop;
	uint32 dwEndloop;
	uint32 dwSampleRate;
	uint8  byOriginalPitch;
	int8   chPitchCorrection;
};


#define SOUNDBANK_TYPE_INVALID	0
#define SOUNDBANK_TYPE_DLS		0x01
#define SOUNDBANK_TYPE_SF2		0x02

struct SOUNDBANKINFO
{
	std::string szBankName,
		szCopyRight,
		szComments,
		szEngineer,
		szSoftware,		// ISFT: Software
		szDescription;	// ISBJ: Subject
};

struct IFFCHUNK;
struct SF2LOADERINFO;

class CDLSBank
{
protected:
	SOUNDBANKINFO m_BankInfo;
	mpt::PathString m_szFileName;
	uint32 m_nType;
	uint32 m_dwWavePoolOffset;
	// DLS Information
	uint32 m_nMaxWaveLink;
	std::vector<uint32> m_WaveForms;
	std::vector<DLSINSTRUMENT> m_Instruments;
	std::vector<DLSSAMPLEEX> m_SamplesEx;
	std::vector<DLSENVELOPE> m_Envelopes;

public:
	CDLSBank();
	static bool IsDLSBank(const mpt::PathString &filename);
	static uint32 MakeMelodicCode(uint32 bank, uint32 instr) { return ((bank << 16) | (instr));}
	static uint32 MakeDrumCode(uint32 rgn, uint32 instr) { return (0x80000000 | (rgn << 16) | (instr));}

public:
	bool Open(const mpt::PathString &filename);
	bool Open(FileReader file);
	mpt::PathString GetFileName() const { return m_szFileName; }
	uint32 GetBankType() const { return m_nType; }
	const SOUNDBANKINFO &GetBankInfo() const { return m_BankInfo; }

public:
	uint32 GetNumInstruments() const { return static_cast<uint32>(m_Instruments.size()); }
	uint32 GetNumSamples() const { return static_cast<uint32>(m_WaveForms.size()); }
	DLSINSTRUMENT *GetInstrument(uint32 iIns) { return iIns < m_Instruments.size() ? &m_Instruments[iIns] : nullptr; }
	DLSINSTRUMENT *FindInstrument(bool bDrum, uint32 nBank=0xFF, uint32 dwProgram=0xFF, uint32 dwKey=0xFF, uint32 *pInsNo=nullptr);
	uint32 GetRegionFromKey(uint32 nIns, uint32 nKey);
	bool ExtractWaveForm(uint32 nIns, uint32 nRgn, std::vector<uint8> &waveData, uint32 &length);
	bool ExtractSample(CSoundFile &sndFile, SAMPLEINDEX nSample, uint32 nIns, uint32 nRgn, int transpose=0);
	bool ExtractInstrument(CSoundFile &sndFile, INSTRUMENTINDEX nInstr, uint32 nIns, uint32 nDrumRgn);
	const char *GetRegionName(uint32 nIns, uint32 nRgn) const;
	uint8 GetPanning(uint32 ins, uint32 region) const;

// Internal Loader Functions
protected:
	bool UpdateInstrumentDefinition(DLSINSTRUMENT *pDlsIns, const IFFCHUNK *pchunk, uint32 dwMaxLen);
	bool UpdateSF2PresetData(SF2LOADERINFO &sf2info, const IFFCHUNK &header, FileReader &chunk);
	bool ConvertSF2ToDLS(SF2LOADERINFO &sf2info);

public:
	// DLS Unit conversion
	static int32 DLS32BitTimeCentsToMilliseconds(int32 lTimeCents);
	static int32 DLS32BitRelativeGainToLinear(int32 lCentibels);	// 0dB = 0x10000
	static int32 DLS32BitRelativeLinearToGain(int32 lGain);		// 0dB = 0x10000
	static int32 DLSMidiVolumeToLinear(uint32 nMidiVolume);		// [0-127] -> [0-0x10000]
};


#endif // MODPLUG_TRACKER


OPENMPT_NAMESPACE_END
