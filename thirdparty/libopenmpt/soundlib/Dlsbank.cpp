/*
 * DLSBank.cpp
 * -----------
 * Purpose: Sound bank loading.
 * Notes  : Supported sound bank types: DLS (including embedded DLS in MSS & RMI), SF2
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Sndfile.h"
#ifdef MODPLUG_TRACKER
#include "../mptrack/mptrack.h"
#include "../common/mptFileIO.h"
#endif
#include "Dlsbank.h"
#include "../common/StringFixer.h"
#include "../common/FileReader.h"
#include "../common/Endianness.h"
#include "SampleIO.h"
#include "modsmp_ctrl.h"

#include <math.h>

OPENMPT_NAMESPACE_BEGIN

#ifdef MODPLUG_TRACKER

//#define DLSBANK_LOG
//#define DLSINSTR_LOG

#define F_RGN_OPTION_SELFNONEXCLUSIVE	0x0001

///////////////////////////////////////////////////////////////////////////
// Articulation connection graph definitions

// Generic Sources
#define CONN_SRC_NONE              0x0000
#define CONN_SRC_LFO               0x0001
#define CONN_SRC_KEYONVELOCITY     0x0002
#define CONN_SRC_KEYNUMBER         0x0003
#define CONN_SRC_EG1               0x0004
#define CONN_SRC_EG2               0x0005
#define CONN_SRC_PITCHWHEEL        0x0006

#define CONN_SRC_POLYPRESSURE      0x0007
#define CONN_SRC_CHANNELPRESSURE   0x0008
#define CONN_SRC_VIBRATO           0x0009

// Midi Controllers 0-127
#define CONN_SRC_CC1               0x0081
#define CONN_SRC_CC7               0x0087
#define CONN_SRC_CC10              0x008a
#define CONN_SRC_CC11              0x008b

#define CONN_SRC_CC91              0x00db
#define CONN_SRC_CC93              0x00dd

#define CONN_SRC_RPN0              0x0100
#define CONN_SRC_RPN1              0x0101
#define CONN_SRC_RPN2              0x0102

// Generic Destinations
#define CONN_DST_NONE              0x0000
#define CONN_DST_ATTENUATION       0x0001
#define CONN_DST_RESERVED          0x0002
#define CONN_DST_PITCH             0x0003
#define CONN_DST_PAN               0x0004

// LFO Destinations
#define CONN_DST_LFO_FREQUENCY     0x0104
#define CONN_DST_LFO_STARTDELAY    0x0105

#define CONN_DST_KEYNUMBER         0x0005

// EG1 Destinations
#define CONN_DST_EG1_ATTACKTIME    0x0206
#define CONN_DST_EG1_DECAYTIME     0x0207
#define CONN_DST_EG1_RESERVED      0x0208
#define CONN_DST_EG1_RELEASETIME   0x0209
#define CONN_DST_EG1_SUSTAINLEVEL  0x020a

#define CONN_DST_EG1_DELAYTIME     0x020b
#define CONN_DST_EG1_HOLDTIME      0x020c
#define CONN_DST_EG1_SHUTDOWNTIME  0x020d

// EG2 Destinations
#define CONN_DST_EG2_ATTACKTIME    0x030a
#define CONN_DST_EG2_DECAYTIME     0x030b
#define CONN_DST_EG2_RESERVED      0x030c
#define CONN_DST_EG2_RELEASETIME   0x030d
#define CONN_DST_EG2_SUSTAINLEVEL  0x030e

#define CONN_DST_EG2_DELAYTIME     0x030f
#define CONN_DST_EG2_HOLDTIME      0x0310

#define CONN_TRN_NONE              0x0000
#define CONN_TRN_CONCAVE           0x0001


//////////////////////////////////////////////////////////
// Supported DLS1 Articulations

#define MAKE_ART(src, ctl, dst)	( ((dst)<<16) | ((ctl)<<8) | (src) )

// Vibrato / Tremolo
#define ART_LFO_FREQUENCY	MAKE_ART	(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_LFO_FREQUENCY)
#define ART_LFO_STARTDELAY	MAKE_ART	(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_LFO_STARTDELAY)
#define ART_LFO_ATTENUATION	MAKE_ART	(CONN_SRC_LFO,	CONN_SRC_NONE,	CONN_DST_ATTENUATION)
#define ART_LFO_PITCH		MAKE_ART	(CONN_SRC_LFO,	CONN_SRC_NONE,	CONN_DST_PITCH)
#define ART_LFO_MODWTOATTN	MAKE_ART	(CONN_SRC_LFO,	CONN_SRC_CC1,	CONN_DST_ATTENUATION)
#define ART_LFO_MODWTOPITCH	MAKE_ART	(CONN_SRC_LFO,	CONN_SRC_CC1,	CONN_DST_PITCH)

// Volume Envelope
#define ART_VOL_EG_ATTACKTIME	MAKE_ART(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_EG1_ATTACKTIME)
#define ART_VOL_EG_DECAYTIME	MAKE_ART(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_EG1_DECAYTIME)
#define ART_VOL_EG_SUSTAINLEVEL	MAKE_ART(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_EG1_SUSTAINLEVEL)
#define ART_VOL_EG_RELEASETIME	MAKE_ART(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_EG1_RELEASETIME)
#define ART_VOL_EG_DELAYTIME	MAKE_ART(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_EG1_DELAYTIME)
#define ART_VOL_EG_HOLDTIME		MAKE_ART(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_EG1_HOLDTIME)
#define ART_VOL_EG_SHUTDOWNTIME	MAKE_ART(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_EG1_SHUTDOWNTIME)
#define ART_VOL_EG_VELTOATTACK	MAKE_ART(CONN_SRC_KEYONVELOCITY,	CONN_SRC_NONE,	CONN_DST_EG1_ATTACKTIME)
#define ART_VOL_EG_KEYTODECAY	MAKE_ART(CONN_SRC_KEYNUMBER,		CONN_SRC_NONE,	CONN_DST_EG1_DECAYTIME)

// Pitch Envelope
#define ART_PITCH_EG_ATTACKTIME		MAKE_ART(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_EG2_ATTACKTIME)
#define ART_PITCH_EG_DECAYTIME		MAKE_ART(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_EG2_DECAYTIME)
#define ART_PITCH_EG_SUSTAINLEVEL	MAKE_ART(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_EG2_SUSTAINLEVEL)
#define ART_PITCH_EG_RELEASETIME	MAKE_ART(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_EG2_RELEASETIME)
#define ART_PITCH_EG_VELTOATTACK	MAKE_ART(CONN_SRC_KEYONVELOCITY,	CONN_SRC_NONE,	CONN_DST_EG2_ATTACKTIME)
#define ART_PITCH_EG_KEYTODECAY		MAKE_ART(CONN_SRC_KEYNUMBER,		CONN_SRC_NONE,	CONN_DST_EG2_DECAYTIME)

// Default Pan
#define ART_DEFAULTPAN		MAKE_ART	(CONN_SRC_NONE,	CONN_SRC_NONE,	CONN_DST_PAN)

//////////////////////////////////////////////////////////
// DLS IFF Chunk IDs

// Standard IFF chunks IDs
#define IFFID_FORM		0x4d524f46
#define IFFID_RIFF		0x46464952
#define IFFID_LIST		0x5453494C
#define IFFID_INFO		0x4F464E49

// IFF Info fields
#define IFFID_ICOP		0x504F4349
#define IFFID_INAM		0x4D414E49
#define IFFID_ICMT		0x544D4349
#define IFFID_IENG		0x474E4549
#define IFFID_ISFT		0x54465349
#define IFFID_ISBJ		0x4A425349

// Wave IFF chunks IDs
#define IFFID_wave		0x65766177
#define IFFID_wsmp		0x706D7377

#define IFFID_XDLS		0x534c4458
#define IFFID_DLS		0x20534C44
#define IFFID_MLS		0x20534C4D
#define IFFID_RMID		0x44494D52
#define IFFID_colh		0x686C6F63
#define IFFID_ins		0x20736E69
#define IFFID_insh		0x68736E69
#define IFFID_ptbl		0x6C627470
#define IFFID_wvpl		0x6C707677
#define IFFID_rgn		0x206E6772
#define IFFID_rgn2		0x326E6772
#define IFFID_rgnh		0x686E6772
#define IFFID_wlnk		0x6B6E6C77
#define IFFID_art1		0x31747261
#define IFFID_art2		0x32747261

//////////////////////////////////////////////////////////
// DLS Structures definitions

struct IFFCHUNK
{
	uint32le id;
	uint32le len;
};

MPT_BINARY_STRUCT(IFFCHUNK, 8)

struct RIFFCHUNKID
{
	uint32le id_RIFF;
	uint32le riff_len;
	uint32le id_DLS;
};

MPT_BINARY_STRUCT(RIFFCHUNKID, 12)

struct LISTCHUNK
{
	uint32le id;
	uint32le len;
	uint32le listid;
};

MPT_BINARY_STRUCT(LISTCHUNK, 12)

struct DLSRGNRANGE
{
	uint16le usLow;
	uint16le usHigh;
};

MPT_BINARY_STRUCT(DLSRGNRANGE, 4)

struct VERSCHUNK
{
	uint32le id;
	uint32le len;
	uint16le version[4];
};

MPT_BINARY_STRUCT(VERSCHUNK, 16)

struct PTBLCHUNK
{
	uint32le cbSize;
	uint32le cCues;
};

MPT_BINARY_STRUCT(PTBLCHUNK, 8)

struct INSHCHUNK
{
	uint32le id;
	uint32le len;
	uint32le cRegions;
	uint32le ulBank;
	uint32le ulInstrument;
};

MPT_BINARY_STRUCT(INSHCHUNK, 20)

struct RGNHCHUNK
{
	uint32le id;
	uint32le len;
	DLSRGNRANGE RangeKey;
	DLSRGNRANGE RangeVelocity;
	uint16le fusOptions;
	uint16le usKeyGroup;
};

MPT_BINARY_STRUCT(RGNHCHUNK, 20)

struct WLNKCHUNK
{
	uint32le id;
	uint32le len;
	uint16le fusOptions;
	uint16le usPhaseGroup;
	uint32le ulChannel;
	uint32le ulTableIndex;
};

MPT_BINARY_STRUCT(WLNKCHUNK, 20)

struct ART1CHUNK
{
	uint32le id;
	uint32le len;
	uint32le cbSize;
	uint32le cConnectionBlocks;
};

MPT_BINARY_STRUCT(ART1CHUNK, 16)

struct CONNECTIONBLOCK
{
	uint16le usSource;
	uint16le usControl;
	uint16le usDestination;
	uint16le usTransform;
	int32le  lScale;
};

MPT_BINARY_STRUCT(CONNECTIONBLOCK, 12)

struct WSMPCHUNK
{
	uint32le id;
	uint32le len;
	uint32le cbSize;
	uint16le usUnityNote;
	int16le  sFineTune;
	int32le  lAttenuation;
	uint32le fulOptions;
	uint32le cSampleLoops;
};

MPT_BINARY_STRUCT(WSMPCHUNK, 28)

struct WSMPSAMPLELOOP
{
	uint32le cbSize;
	uint32le ulLoopType;
	uint32le ulLoopStart;
	uint32le ulLoopLength;

};

MPT_BINARY_STRUCT(WSMPSAMPLELOOP, 16)


/////////////////////////////////////////////////////////////////////
// SF2 IFF Chunk IDs

#define IFFID_sfbk		0x6b626673
#define IFFID_sdta		0x61746473
#define IFFID_pdta		0x61746470
#define IFFID_phdr		0x72646870
#define IFFID_pbag		0x67616270
#define IFFID_pgen		0x6E656770
#define IFFID_inst		0x74736E69
#define IFFID_ibag		0x67616269
#define IFFID_igen		0x6E656769
#define IFFID_shdr		0x72646873

///////////////////////////////////////////
// SF2 Generators IDs

enum SF2Generators
{
	SF2_GEN_MODENVTOFILTERFC	= 11,
	SF2_GEN_PAN					= 17,
	SF2_GEN_DECAYMODENV			= 28,
	SF2_GEN_ATTACKVOLENV		= 34,
	SF2_GEN_HOLDVOLENV			= 34,
	SF2_GEN_DECAYVOLENV			= 36,
	SF2_GEN_SUSTAINVOLENV		= 37,
	SF2_GEN_RELEASEVOLENV		= 38,
	SF2_GEN_INSTRUMENT			= 41,
	SF2_GEN_KEYRANGE			= 43,
	SF2_GEN_ATTENUATION			= 48,
	SF2_GEN_COARSETUNE			= 51,
	SF2_GEN_FINETUNE			= 52,
	SF2_GEN_SAMPLEID			= 53,
	SF2_GEN_SAMPLEMODES			= 54,
	SF2_GEN_KEYGROUP			= 57,
	SF2_GEN_UNITYNOTE			= 58,
};

/////////////////////////////////////////////////////////////////////
// SF2 Structures Definitions

struct SFPRESETHEADER
{
	char     achPresetName[20];
	uint16le wPreset;
	uint16le wBank;
	uint16le wPresetBagNdx;
	uint32le dwLibrary;
	uint32le dwGenre;
	uint32le dwMorphology;
};

MPT_BINARY_STRUCT(SFPRESETHEADER, 38)

struct SFPRESETBAG
{
	uint16le wGenNdx;
	uint16le wModNdx;
};

MPT_BINARY_STRUCT(SFPRESETBAG, 4)

struct SFGENLIST
{
	uint16le sfGenOper;
	uint16le genAmount;
};

MPT_BINARY_STRUCT(SFGENLIST, 4)

struct SFINST
{
	char     achInstName[20];
	uint16le wInstBagNdx;
};

MPT_BINARY_STRUCT(SFINST, 22)

struct SFINSTBAG
{
	uint16le wGenNdx;
	uint16le wModNdx;
};

MPT_BINARY_STRUCT(SFINSTBAG, 4)

struct SFINSTGENLIST
{
	uint16le sfGenOper;
	uint16le genAmount;
};

MPT_BINARY_STRUCT(SFINSTGENLIST, 4)

struct SFSAMPLE
{
	char     achSampleName[20];
	uint32le dwStart;
	uint32le dwEnd;
	uint32le dwStartloop;
	uint32le dwEndloop;
	uint32le dwSampleRate;
	uint8le  byOriginalPitch;
	int8le   chPitchCorrection;
	uint16le wSampleLink;
	uint16le sfSampleType;
};

MPT_BINARY_STRUCT(SFSAMPLE, 46)

// End of structures definitions
/////////////////////////////////////////////////////////////////////


struct SF2LOADERINFO
{
	uint32 nPresetBags;
	const SFPRESETBAG *pPresetBags;
	uint32 nPresetGens;
	const SFGENLIST *pPresetGens;
	uint32 nInsts;
	const SFINST *pInsts;
	uint32 nInstBags;
	const SFINSTBAG *pInstBags;
	uint32 nInstGens;
	const SFINSTGENLIST *pInstGens;
};


/////////////////////////////////////////////////////////////////////
// Unit conversion

static uint8 DLSSustainLevelToLinear(int32 sustain)
{
	// 0.1% units
	if(sustain >= 0)
	{
		int32 l = sustain / (1000 * 512);
		if(l >= 0 || l <= 128)
			return static_cast<uint8>(l);
	}
	return 128;
}


static uint8 SF2SustainLevelToLinear(int32 sustain)
{
	// 0.1% units
	int32 l = 128 * (1000 - Clamp(sustain, 0, 1000)) / 1000;
	return static_cast<uint8>(l);
}


int32 CDLSBank::DLS32BitTimeCentsToMilliseconds(int32 lTimeCents)
{
	// tc = log2(time[secs]) * 1200*65536
	// time[secs] = 2^(tc/(1200*65536))
	if ((uint32)lTimeCents == 0x80000000) return 0;
	double fmsecs = 1000.0 * pow(2.0, ((double)lTimeCents)/(1200.0*65536.0));
	if (fmsecs < -32767) return -32767;
	if (fmsecs > 32767) return 32767;
	return (int32)fmsecs;
}


// 0dB = 0x10000
int32 CDLSBank::DLS32BitRelativeGainToLinear(int32 lCentibels)
{
	// v = 10^(cb/(200*65536)) * V
	return (int32)(65536.0 * pow(10.0, ((double)lCentibels)/(200*65536.0)) );
}


int32 CDLSBank::DLS32BitRelativeLinearToGain(int32 lGain)
{
	// cb = log10(v/V) * 200 * 65536
	if (lGain <= 0) return -960 * 65536;
	return (int32)( 200*65536.0 * log10( ((double)lGain)/65536.0 ) );
}


int32 CDLSBank::DLSMidiVolumeToLinear(uint32 nMidiVolume)
{
	return (nMidiVolume * nMidiVolume << 16) / (127*127);
}


/////////////////////////////////////////////////////////////////////
// Implementation

CDLSBank::CDLSBank()
{
	m_nMaxWaveLink = 0;
	m_nType = SOUNDBANK_TYPE_INVALID;
}


bool CDLSBank::IsDLSBank(const mpt::PathString &filename)
{
	RIFFCHUNKID riff;
	FILE *f;
	if(filename.empty()) return false;
	if((f = mpt_fopen(filename, "rb")) == nullptr) return false;
	MemsetZero(riff);
	fread(&riff, sizeof(RIFFCHUNKID), 1, f);
	// Check for embedded DLS sections
	if (riff.id_RIFF == IFFID_FORM)
	{
		// Miles Sound System
		do
		{
			uint32 len = riff.riff_len;
			len = SwapBytesBE(len);
			if (len <= 4) break;
			if (riff.id_DLS == IFFID_XDLS)
			{
				fread(&riff, sizeof(RIFFCHUNKID), 1, f);
				break;
			}
			if((len % 2u) != 0)
				len++;
			if (fseek(f, len-4, SEEK_CUR) != 0) break;
		} while (fread(&riff, sizeof(RIFFCHUNKID), 1, f) != 0);
	} else
	if ((riff.id_RIFF == IFFID_RIFF) && (riff.id_DLS == IFFID_RMID))
	{
		for (;;)
		{
			if(!fread(&riff, sizeof(RIFFCHUNKID), 1, f))
				break;
			if (riff.id_DLS == IFFID_DLS)
				break; // found it
			int len = riff.riff_len;
			if((len % 2u) != 0)
				len++;
			if ((len <= 4) || (fseek(f, len-4, SEEK_CUR) != 0)) break;
		}
	}
	fclose(f);
	return ((riff.id_RIFF == IFFID_RIFF)
		&& ((riff.id_DLS == IFFID_DLS) || (riff.id_DLS == IFFID_MLS) || (riff.id_DLS == IFFID_sfbk))
		&& (riff.riff_len >= 256));
}


///////////////////////////////////////////////////////////////
// Find an instrument based on the given parameters

DLSINSTRUMENT *CDLSBank::FindInstrument(bool bDrum, uint32 nBank, uint32 dwProgram, uint32 dwKey, uint32 *pInsNo)
{
	if (m_Instruments.empty()) return NULL;
	for (uint32 iIns=0; iIns<m_Instruments.size(); iIns++)
	{
		DLSINSTRUMENT *pDlsIns = &m_Instruments[iIns];
		uint32 insbank = ((pDlsIns->ulBank & 0x7F00) >> 1) | (pDlsIns->ulBank & 0x7F);
		if ((nBank >= 0x4000) || (insbank == nBank))
		{
			if (bDrum)
			{
				if (pDlsIns->ulBank & F_INSTRUMENT_DRUMS)
				{
					if ((dwProgram >= 0x80) || (dwProgram == (pDlsIns->ulInstrument & 0x7F)))
					{
						for (uint32 iRgn=0; iRgn<pDlsIns->nRegions; iRgn++)
						{
							if ((!dwKey) || (dwKey >= 0x80)
							 || ((dwKey >= pDlsIns->Regions[iRgn].uKeyMin)
							  && (dwKey <= pDlsIns->Regions[iRgn].uKeyMax)))
							{
								if (pInsNo) *pInsNo = iIns;
								return pDlsIns;
							}
						}
					}
				}
			} else
			{
				if (!(pDlsIns->ulBank & F_INSTRUMENT_DRUMS))
				{
					if ((dwProgram >= 0x80) || (dwProgram == (pDlsIns->ulInstrument & 0x7F)))
					{
						if (pInsNo) *pInsNo = iIns;
						return pDlsIns;
					}
				}
			}
		}
	}
	return NULL;
}


///////////////////////////////////////////////////////////////
// Update DLS instrument definition from an IFF chunk

bool CDLSBank::UpdateInstrumentDefinition(DLSINSTRUMENT *pDlsIns, const IFFCHUNK *pchunk, uint32 dwMaxLen)
{
	if ((!pchunk->len) || (pchunk->len+8 > dwMaxLen)) return false;
	if (pchunk->id == IFFID_LIST)
	{
		LISTCHUNK *plist = (LISTCHUNK *)pchunk;
		uint32 dwPos = 12;
		while (dwPos < plist->len)
		{
			const IFFCHUNK *p = (const IFFCHUNK *)(((uint8 *)plist) + dwPos);
			if (!(p->id & 0xFF))
			{
				p = (const IFFCHUNK *)( ((uint8 *)p)+1  );
				dwPos++;
			}
			if (dwPos + p->len + 8 <= plist->len + 12)
			{
				UpdateInstrumentDefinition(pDlsIns, p, p->len+8);
			}
			dwPos += p->len + 8;
		}
		switch(plist->listid)
		{
		case IFFID_rgn:		// Level 1 region
		case IFFID_rgn2:	// Level 2 region
			if (pDlsIns->nRegions < DLSMAXREGIONS) pDlsIns->nRegions++;
			break;
		}
	} else
	{
		switch(pchunk->id)
		{
		case IFFID_insh:
			pDlsIns->ulBank = ((INSHCHUNK *)pchunk)->ulBank;
			pDlsIns->ulInstrument = ((INSHCHUNK *)pchunk)->ulInstrument;
			//Log("%3d regions, bank 0x%04X instrument %3d\n", ((INSHCHUNK *)pchunk)->cRegions, pDlsIns->ulBank, pDlsIns->ulInstrument);
			break;

		case IFFID_rgnh:
			if (pDlsIns->nRegions < DLSMAXREGIONS)
			{
				RGNHCHUNK *p = (RGNHCHUNK *)pchunk;
				DLSREGION *pregion = &pDlsIns->Regions[pDlsIns->nRegions];
				pregion->uKeyMin = (uint8)p->RangeKey.usLow;
				pregion->uKeyMax = (uint8)p->RangeKey.usHigh;
				pregion->fuOptions = (uint8)(p->usKeyGroup & DLSREGION_KEYGROUPMASK);
				if (p->fusOptions & F_RGN_OPTION_SELFNONEXCLUSIVE) pregion->fuOptions |= DLSREGION_SELFNONEXCLUSIVE;
				//Log("  Region %d: fusOptions=0x%02X usKeyGroup=0x%04X ", pDlsIns->nRegions, p->fusOptions, p->usKeyGroup);
				//Log("KeyRange[%3d,%3d] ", p->RangeKey.usLow, p->RangeKey.usHigh);
			}
			break;

		case IFFID_wlnk:
			if (pDlsIns->nRegions < DLSMAXREGIONS)
			{
				DLSREGION *pregion = &pDlsIns->Regions[pDlsIns->nRegions];
				WLNKCHUNK *p = (WLNKCHUNK *)pchunk;
				pregion->nWaveLink = (uint16)p->ulTableIndex;
				if ((pregion->nWaveLink < uint16_max) && (pregion->nWaveLink >= m_nMaxWaveLink)) m_nMaxWaveLink = pregion->nWaveLink + 1;
				//Log("  WaveLink %d: fusOptions=0x%02X usPhaseGroup=0x%04X ", pDlsIns->nRegions, p->fusOptions, p->usPhaseGroup);
				//Log("ulChannel=%d ulTableIndex=%4d\n", p->ulChannel, p->ulTableIndex);
			}
			break;

		case IFFID_wsmp:
			if (pDlsIns->nRegions < DLSMAXREGIONS)
			{
				DLSREGION *pregion = &pDlsIns->Regions[pDlsIns->nRegions];
				WSMPCHUNK *p = (WSMPCHUNK *)pchunk;
				pregion->fuOptions |= DLSREGION_OVERRIDEWSMP;
				pregion->uUnityNote = (uint8)p->usUnityNote;
				pregion->sFineTune = p->sFineTune;
				int32 lVolume = DLS32BitRelativeGainToLinear(p->lAttenuation) / 256;
				if (lVolume > 256) lVolume = 256;
				if (lVolume < 4) lVolume = 4;
				pregion->usVolume = (uint16)lVolume;
				//Log("  WaveSample %d: usUnityNote=%2d sFineTune=%3d ", pDlsEnv->nRegions, p->usUnityNote, p->sFineTune);
				//Log("fulOptions=0x%04X loops=%d\n", p->fulOptions, p->cSampleLoops);
				if ((p->cSampleLoops) && (p->cbSize + sizeof(WSMPSAMPLELOOP) <= p->len))
				{
					WSMPSAMPLELOOP *ploop = (WSMPSAMPLELOOP *)(((uint8 *)p)+8+p->cbSize);
					//Log("looptype=%2d loopstart=%5d loopend=%5d\n", ploop->ulLoopType, ploop->ulLoopStart, ploop->ulLoopLength);
					if (ploop->ulLoopLength > 3)
					{
						pregion->fuOptions |= DLSREGION_SAMPLELOOP;
						//if (ploop->ulLoopType) pregion->fuOptions |= DLSREGION_PINGPONGLOOP;
						pregion->ulLoopStart = ploop->ulLoopStart;
						pregion->ulLoopEnd = ploop->ulLoopStart + ploop->ulLoopLength;
					}
				}
			}
			break;

		case IFFID_art1:
		case IFFID_art2:
			{
				ART1CHUNK *p = (ART1CHUNK *)pchunk;
				if (pDlsIns->ulBank & F_INSTRUMENT_DRUMS)
				{
					if (pDlsIns->nRegions >= DLSMAXREGIONS) break;
				} else
				{
					pDlsIns->nMelodicEnv = m_Envelopes.size() + 1;
				}
				if (p->cbSize+p->cConnectionBlocks*sizeof(CONNECTIONBLOCK) > p->len) break;
				DLSENVELOPE dlsEnv;
				MemsetZero(dlsEnv);
				dlsEnv.nDefPan = 128;
				dlsEnv.nVolSustainLevel = 128;
				//Log("  art1 (%3d bytes): cbSize=%d cConnectionBlocks=%d\n", p->len, p->cbSize, p->cConnectionBlocks);
				CONNECTIONBLOCK *pblk = (CONNECTIONBLOCK *)( ((uint8 *)p)+8+p->cbSize );
				for (uint32 iblk=0; iblk<p->cConnectionBlocks; iblk++, pblk++)
				{
					// [4-bit transform][12-bit dest][8-bit control][8-bit source] = 32-bit ID
					uint32 dwArticulation = pblk->usTransform;
					dwArticulation = (dwArticulation << 12) | (pblk->usDestination & 0x0FFF);
					dwArticulation = (dwArticulation << 8) | (pblk->usControl & 0x00FF);
					dwArticulation = (dwArticulation << 8) | (pblk->usSource & 0x00FF);
					switch(dwArticulation)
					{
					case ART_DEFAULTPAN:
						{
							int32 pan = 128 + pblk->lScale / (65536000/128);
							if (pan < 0) pan = 0;
							if (pan > 255) pan = 255;
							dlsEnv.nDefPan = (uint8)pan;
						}
						break;

					case ART_VOL_EG_ATTACKTIME:
						// 32-bit time cents units. range = [0s, 20s]
						dlsEnv.wVolAttack = 0;
						if (pblk->lScale > -0x40000000)
						{
							int32 l = pblk->lScale - 78743200; // maximum velocity
							if (l > 0) l = 0;
							int32 attacktime = DLS32BitTimeCentsToMilliseconds(l);
							if (attacktime < 0) attacktime = 0;
							if (attacktime > 20000) attacktime = 20000;
							if (attacktime >= 20) dlsEnv.wVolAttack = (uint16)(attacktime / 20);
							//Log("%3d: Envelope Attack Time set to %d (%d time cents)\n", (uint32)(dlsEnv.ulInstrument & 0x7F)|((dlsEnv.ulBank >> 16) & 0x8000), attacktime, pblk->lScale);
						}
						break;

					case ART_VOL_EG_DECAYTIME:
						// 32-bit time cents units. range = [0s, 20s]
						dlsEnv.wVolDecay = 0;
						if (pblk->lScale > -0x40000000)
						{
							int32 decaytime = DLS32BitTimeCentsToMilliseconds(pblk->lScale);
							if (decaytime > 20000) decaytime = 20000;
							if (decaytime >= 20) dlsEnv.wVolDecay = (uint16)(decaytime / 20);
							//Log("%3d: Envelope Decay Time set to %d (%d time cents)\n", (uint32)(dlsEnv.ulInstrument & 0x7F)|((dlsEnv.ulBank >> 16) & 0x8000), decaytime, pblk->lScale);
						}
						break;

					case ART_VOL_EG_RELEASETIME:
						// 32-bit time cents units. range = [0s, 20s]
						dlsEnv.wVolRelease = 0;
						if (pblk->lScale > -0x40000000)
						{
							int32 releasetime = DLS32BitTimeCentsToMilliseconds(pblk->lScale);
							if (releasetime > 20000) releasetime = 20000;
							if (releasetime >= 20) dlsEnv.wVolRelease = (uint16)(releasetime / 20);
							//Log("%3d: Envelope Release Time set to %d (%d time cents)\n", (uint32)(dlsEnv.ulInstrument & 0x7F)|((dlsEnv.ulBank >> 16) & 0x8000), dlsEnv.wVolRelease, pblk->lScale);
						}
						break;

					case ART_VOL_EG_SUSTAINLEVEL:
						// 0.1% units
						if (pblk->lScale >= 0)
						{
							dlsEnv.nVolSustainLevel = DLSSustainLevelToLinear(pblk->lScale);
						}
						break;

					//default:
					//	Log("    Articulation = 0x%08X value=%d\n", dwArticulation, pblk->lScale);
					}
				}
				m_Envelopes.push_back(dlsEnv);
			}
			break;

		case IFFID_INAM:
			mpt::String::CopyN(pDlsIns->szName, ((const char *)pchunk) + 8, pchunk->len);
			break;
	#if 0
		default:
			{
				char sid[5];
				memcpy(sid, &pchunk->id, 4);
				sid[4] = 0;
				Log("    \"%s\": %d bytes\n", (uint32)sid, pchunk->len.get());
			}
	#endif
		}
	}
	return true;
}

///////////////////////////////////////////////////////////////
// Converts SF2 chunks to DLS

bool CDLSBank::UpdateSF2PresetData(SF2LOADERINFO &sf2info, const IFFCHUNK &header, FileReader &chunk)
{
	if (!chunk.IsValid()) return false;
	switch(header.id)
	{
	case IFFID_phdr:
		if (m_Instruments.empty())
		{
			uint32 numIns = chunk.GetLength() / sizeof(SFPRESETHEADER);
			if(numIns <= 1)
				break;
			// The terminal sfPresetHeader record should never be accessed, and exists only to provide a terminal wPresetBagNdx with which to determine the number of zones in the last preset.
			numIns--;
			m_Instruments.resize(numIns);

		#ifdef DLSBANK_LOG
			Log("phdr: %d instruments\n", m_Instruments.size());
		#endif
			SFPRESETHEADER psfh;
			chunk.ReadStruct(psfh);
			for (auto &dlsIns : m_Instruments)
			{
				mpt::String::Copy(dlsIns.szName, psfh.achPresetName);
				dlsIns.ulInstrument = psfh.wPreset & 0x7F;
				dlsIns.ulBank = (psfh.wBank >= 128) ? F_INSTRUMENT_DRUMS : (psfh.wBank << 8);
				dlsIns.wPresetBagNdx = psfh.wPresetBagNdx;
				dlsIns.wPresetBagNum = 1;
				chunk.ReadStruct(psfh);
				if (psfh.wPresetBagNdx > dlsIns.wPresetBagNdx) dlsIns.wPresetBagNum = static_cast<uint16>(psfh.wPresetBagNdx - dlsIns.wPresetBagNdx);
			}
		}
		break;

	case IFFID_pbag:
		if (!m_Instruments.empty())
		{
			uint32 nBags = chunk.GetLength() / sizeof(SFPRESETBAG);
			if (nBags)
			{
				sf2info.nPresetBags = nBags;
				sf2info.pPresetBags = reinterpret_cast<const SFPRESETBAG *>(chunk.GetRawData());
			}
		}
	#ifdef DLSINSTR_LOG
		else Log("pbag: no instruments!\n");
	#endif
		break;

	case IFFID_pgen:
		if (!m_Instruments.empty())
		{
			uint32 nGens = chunk.GetLength() / sizeof(SFGENLIST);
			if (nGens)
			{
				sf2info.nPresetGens = nGens;
				sf2info.pPresetGens = reinterpret_cast<const SFGENLIST *>(chunk.GetRawData());
			}
		}
	#ifdef DLSINSTR_LOG
		else Log("pgen: no instruments!\n");
	#endif
		break;

	case IFFID_inst:
		if (!m_Instruments.empty())
		{
			uint32 nIns = chunk.GetLength() / sizeof(SFINST);
			sf2info.nInsts = nIns;
			sf2info.pInsts = reinterpret_cast<const SFINST *>(chunk.GetRawData());
		}
		break;

	case IFFID_ibag:
		if (!m_Instruments.empty())
		{
			uint32 nBags = chunk.GetLength() / sizeof(SFINSTBAG);
			if (nBags)
			{
				sf2info.nInstBags = nBags;
				sf2info.pInstBags = reinterpret_cast<const SFINSTBAG *>(chunk.GetRawData());
			}
		}
		break;

	case IFFID_igen:
		if (!m_Instruments.empty())
		{
			uint32 nGens = chunk.GetLength() / sizeof(SFINSTGENLIST);
			if (nGens)
			{
				sf2info.nInstGens = nGens;
				sf2info.pInstGens = reinterpret_cast<const SFINSTGENLIST *>(chunk.GetRawData());
			}
		}
		break;

	case IFFID_shdr:
		if (m_SamplesEx.empty())
		{
			uint32 numSmp = chunk.GetLength() / sizeof(SFSAMPLE);
			if (numSmp < 1) break;
			m_SamplesEx.resize(numSmp);
			m_WaveForms.resize(numSmp);
	#ifdef DLSINSTR_LOG
		Log("shdr: %d samples\n", m_SamplesEx.size());
	#endif

			for (uint32 i = 0; i < numSmp; i++)
			{
				SFSAMPLE p;
				chunk.ReadStruct(p);
				DLSSAMPLEEX &dlsSmp = m_SamplesEx[i];
				mpt::String::Copy(dlsSmp.szName, p.achSampleName);
				dlsSmp.dwLen = 0;
				dlsSmp.dwSampleRate = p.dwSampleRate;
				dlsSmp.byOriginalPitch = p.byOriginalPitch;
				dlsSmp.chPitchCorrection = static_cast<int8>(Util::muldivr(p.chPitchCorrection, 128, 100));
				if (((p.sfSampleType & 0x7FFF) <= 4) && (p.dwStart < 0x08000000) && (p.dwEnd >= p.dwStart+8))
				{
					dlsSmp.dwLen = (p.dwEnd - p.dwStart) * 2;
					if ((p.dwEndloop > p.dwStartloop + 7) && (p.dwStartloop >= p.dwStart))
					{
						dlsSmp.dwStartloop = p.dwStartloop - p.dwStart;
						dlsSmp.dwEndloop = p.dwEndloop - p.dwStart;
					}
					m_WaveForms[i] = p.dwStart * 2;
					//Log("  offset[%d]=%d len=%d\n", i, p.dwStart*2, psmp->dwLen);
				}
			}
		}
		break;

	#ifdef DLSINSTR_LOG
	default:
		{
			char sdbg[5];
			memcpy(sdbg, &header.id, 4);
			sdbg[4] = 0;
			Log("Unsupported SF2 chunk: %s (%d bytes)\n", sdbg, header.len.get());
		}
	#endif
	}
	return true;
}


static int16 SF2TimeToDLS(int16 amount)
{
	int32 time = CDLSBank::DLS32BitTimeCentsToMilliseconds(static_cast<int32>(amount) << 16);
	return static_cast<int16>(Clamp(time, 20, 20000) / 20);
}


// Convert all instruments to the DLS format
bool CDLSBank::ConvertSF2ToDLS(SF2LOADERINFO &sf2info)
{
	if (m_Instruments.empty() || m_SamplesEx.empty())
		return false;

	for (auto &dlsIns : m_Instruments)
	{
		DLSENVELOPE dlsEnv;
		uint32 nInstrNdx = 0;
		int32 lAttenuation = 0;
		// Default Envelope Values
		dlsEnv.wVolAttack = 0;
		dlsEnv.wVolDecay = 0;
		dlsEnv.wVolRelease = 0;
		dlsEnv.nVolSustainLevel = 128;
		dlsEnv.nDefPan = 128;
		// Load Preset Bags
		for (uint32 ipbagcnt=0; ipbagcnt<(uint32)dlsIns.wPresetBagNum; ipbagcnt++)
		{
			uint32 ipbagndx = dlsIns.wPresetBagNdx + ipbagcnt;
			if ((ipbagndx+1 >= sf2info.nPresetBags) || (!sf2info.pPresetBags)) break;
			// Load generators for each preset bag
			const SFPRESETBAG *pbag = sf2info.pPresetBags + ipbagndx;
			for (uint32 ipgenndx=pbag[0].wGenNdx; ipgenndx<pbag[1].wGenNdx; ipgenndx++)
			{
				if ((!sf2info.pPresetGens) || (ipgenndx+1 >= sf2info.nPresetGens)) break;
				const SFGENLIST *pgen = sf2info.pPresetGens + ipgenndx;
				switch(pgen->sfGenOper)
				{
				case SF2_GEN_ATTACKVOLENV:
					dlsEnv.wVolAttack = SF2TimeToDLS(pgen->genAmount);
					break;
				case SF2_GEN_DECAYVOLENV:
					dlsEnv.wVolDecay = SF2TimeToDLS(pgen->genAmount);
					break;
				case SF2_GEN_SUSTAINVOLENV:
					// 0.1% units
					if(pgen->genAmount >= 0)
					{
						dlsEnv.nVolSustainLevel = SF2SustainLevelToLinear(pgen->genAmount);
					}
					break;
				case SF2_GEN_RELEASEVOLENV:
					dlsEnv.wVolRelease = SF2TimeToDLS(pgen->genAmount);
					break;
				case SF2_GEN_INSTRUMENT:
					nInstrNdx = pgen->genAmount + 1;
					break;
				case SF2_GEN_ATTENUATION:
					lAttenuation = - (int)(uint16)(pgen->genAmount);
					break;
#if 0
				default:
					Log("Ins %3d: bag %3d gen %3d: ", nIns, ipbagndx, ipgenndx);
					Log("genoper=%d amount=0x%04X ", pgen->sfGenOper, pgen->genAmount);
					Log((pSmp->ulBank & F_INSTRUMENT_DRUMS) ? "(drum)\n" : "\n");
#endif
				}
			}
		}
		// Envelope
		if (!(dlsIns.ulBank & F_INSTRUMENT_DRUMS))
		{
			m_Envelopes.push_back(dlsEnv);
			dlsIns.nMelodicEnv = m_Envelopes.size();
		}
		// Load Instrument Bags
		if ((!nInstrNdx) || (nInstrNdx >= sf2info.nInsts) || (!sf2info.pInsts)) continue;
		nInstrNdx--;
		dlsIns.nRegions = sf2info.pInsts[nInstrNdx+1].wInstBagNdx - sf2info.pInsts[nInstrNdx].wInstBagNdx;
		//Log("\nIns %3d, %2d regions:\n", nIns, pSmp->nRegions);
		if (dlsIns.nRegions > DLSMAXREGIONS) dlsIns.nRegions = DLSMAXREGIONS;
		DLSREGION *pRgn = dlsIns.Regions;
		for (uint32 nRgn = 0; nRgn < dlsIns.nRegions; nRgn++, pRgn++)
		{
			uint32 ibagcnt = sf2info.pInsts[nInstrNdx].wInstBagNdx + nRgn;
			if ((ibagcnt >= sf2info.nInstBags) || (!sf2info.pInstBags)) break;
			// Create a new envelope for drums
			DLSENVELOPE *pDlsEnv = &dlsEnv;
			if (!(dlsIns.ulBank & F_INSTRUMENT_DRUMS) && dlsIns.nMelodicEnv > 0 && dlsIns.nMelodicEnv <= m_Envelopes.size())
			{
				pDlsEnv = &m_Envelopes[dlsIns.nMelodicEnv - 1];
			}
			// Region Default Values
			int32 lAttn = lAttenuation;
			pRgn->uUnityNote = 0xFF;	// 0xFF means undefined -> use sample
			pRgn->sFineTune = 0;
			pRgn->nWaveLink = Util::MaxValueOfType(pRgn->nWaveLink);
			// Load Generators
			const SFINSTBAG *pbag = sf2info.pInstBags + ibagcnt;
			for (uint32 igenndx=pbag[0].wGenNdx; igenndx<pbag[1].wGenNdx; igenndx++)
			{
				if ((igenndx >= sf2info.nInstGens) || (!sf2info.pInstGens)) break;
				const SFINSTGENLIST *pgen = sf2info.pInstGens + igenndx;
				uint16 value = pgen->genAmount;
				switch(pgen->sfGenOper)
				{
				case SF2_GEN_KEYRANGE:
					pRgn->uKeyMin = (uint8)(value & 0xFF);
					pRgn->uKeyMax = (uint8)(value >> 8);
					if (pRgn->uKeyMin > pRgn->uKeyMax)
					{
						std::swap(pRgn->uKeyMin, pRgn->uKeyMax);
					}
					//if (nIns == 9) Log("  keyrange: %d-%d\n", pRgn->uKeyMin, pRgn->uKeyMax);
					break;
				case SF2_GEN_UNITYNOTE:
					if (value < 128) pRgn->uUnityNote = (uint8)value;
					break;
				case SF2_GEN_ATTACKVOLENV:
					pDlsEnv->wVolAttack = SF2TimeToDLS(pgen->genAmount);
					break;
				case SF2_GEN_DECAYVOLENV:
					pDlsEnv->wVolDecay = SF2TimeToDLS(pgen->genAmount);
					break;
				case SF2_GEN_SUSTAINVOLENV:
					// 0.1% units
					if(pgen->genAmount >= 0)
					{
						pDlsEnv->nVolSustainLevel = SF2SustainLevelToLinear(pgen->genAmount);
					}
					break;
				case SF2_GEN_RELEASEVOLENV:
					pDlsEnv->wVolRelease = SF2TimeToDLS(pgen->genAmount);
					break;
				case SF2_GEN_PAN:
					{
						int pan = (short int)value;
						pan = (((pan + 500) * 127) / 500) + 128;
						if (pan < 0) pan = 0;
						if (pan > 255) pan = 255;
						pDlsEnv->nDefPan = (uint8)pan;
					}
					break;
				case SF2_GEN_ATTENUATION:
					lAttn = -(int)value;
					break;
				case SF2_GEN_SAMPLEID:
					if (value < m_SamplesEx.size())
					{
						pRgn->nWaveLink = value;
						pRgn->ulLoopStart = m_SamplesEx[value].dwStartloop;
						pRgn->ulLoopEnd = m_SamplesEx[value].dwEndloop;
					}
					break;
				case SF2_GEN_SAMPLEMODES:
					value &= 3;
					pRgn->fuOptions &= uint16(~(DLSREGION_SAMPLELOOP|DLSREGION_PINGPONGLOOP|DLSREGION_SUSTAINLOOP));
					if (value == 1) pRgn->fuOptions |= DLSREGION_SAMPLELOOP; else
					if (value == 2) pRgn->fuOptions |= DLSREGION_SAMPLELOOP|DLSREGION_PINGPONGLOOP; else
					if (value == 3) pRgn->fuOptions |= DLSREGION_SAMPLELOOP|DLSREGION_SUSTAINLOOP;
					pRgn->fuOptions |= DLSREGION_OVERRIDEWSMP;
					break;
				case SF2_GEN_KEYGROUP:
					pRgn->fuOptions |= (uint8)(value & DLSREGION_KEYGROUPMASK);
					break;
				case SF2_GEN_COARSETUNE:
					pRgn->sFineTune += static_cast<int16>(value) * 128;
					break;
				case SF2_GEN_FINETUNE:
					pRgn->sFineTune += static_cast<int16>(Util::muldiv(static_cast<int8>(value), 128, 100));
					break;
				//default:
				//	Log("    gen=%d value=%04X\n", pgen->sfGenOper, pgen->genAmount);
				}
			}
			int32 lVolume = DLS32BitRelativeGainToLinear((lAttn/10) << 16) / 256;
			if (lVolume < 16) lVolume = 16;
			if (lVolume > 256) lVolume = 256;
			pRgn->usVolume = (uint16)lVolume;
			//Log("\n");
		}
	}
	return true;
}


///////////////////////////////////////////////////////////////
// Open: opens a DLS bank

bool CDLSBank::Open(const mpt::PathString &filename)
{
	if(filename.empty()) return false;
	m_szFileName = filename;
	InputFile f(filename);
	if(!f.IsValid()) return false;
	return Open(GetFileReader(f));
}


bool CDLSBank::Open(FileReader file)
{
	SF2LOADERINFO sf2info;
	uint32 nInsDef;

	if(!file.GetFileName().empty())
		m_szFileName = file.GetFileName();

	file.Rewind();
	const uint8 *lpMemFile = file.GetRawData<uint8>();
	uint32 dwMemLength = file.GetLength();
	uint32 dwMemPos = 0;
	if(!file.CanRead(256))
	{
		return false;
	}

	RIFFCHUNKID riff;
	file.ReadStruct(riff);
	// Check DLS sections embedded in RMI midi files
	if(riff.id_RIFF == IFFID_RIFF && riff.id_DLS == IFFID_RMID)
	{
		while(file.ReadStruct(riff))
		{
			if(riff.id_RIFF == IFFID_RIFF && riff.id_DLS == IFFID_DLS)
			{
				file.SkipBack(sizeof(riff));
				break;
			}
			uint32 len = riff.riff_len;
			if((len % 2u) != 0)
				len++;
			file.SkipBack(4);
			file.Skip(len);
		}
	}

	// Check XDLS sections embedded in big endian IFF files (Miles Sound System)
	if (riff.id_RIFF == IFFID_FORM)
	{
		do
		{
			if(riff.id_DLS == IFFID_XDLS)
			{
				file.ReadStruct(riff);
				break;
			}
			uint32 len = SwapBytesBE(riff.riff_len);
			if((len % 2u) != 0)
				len++;
			file.SkipBack(4);
			file.Skip(len);
		} while(file.ReadStruct(riff));
	}
	if (riff.id_RIFF != IFFID_RIFF
		|| (riff.id_DLS != IFFID_DLS && riff.id_DLS != IFFID_MLS && riff.id_DLS != IFFID_sfbk)
		|| !file.CanRead(riff.riff_len - 4))
	{
	#ifdef DLSBANK_LOG
		Log("Invalid DLS bank!\n");
	#endif
		return false;
	}
	MemsetZero(sf2info);
	m_nType = (riff.id_DLS == IFFID_sfbk) ? SOUNDBANK_TYPE_SF2 : SOUNDBANK_TYPE_DLS;
	m_dwWavePoolOffset = 0;
	m_Instruments.clear();
	m_WaveForms.clear();
	m_Envelopes.clear();
	nInsDef = 0;
	if (dwMemLength > 8 + riff.riff_len + dwMemPos) dwMemLength = 8 + riff.riff_len + dwMemPos;
	while(file.CanRead(sizeof(IFFCHUNK)))
	{
		IFFCHUNK chunkHeader;
		file.ReadStruct(chunkHeader);
		dwMemPos = file.GetPosition();
		FileReader chunk = file.ReadChunk(chunkHeader.len);
		if(chunkHeader.len % 2u)
			file.Skip(1);

		if(!chunk.LengthIsAtLeast(chunkHeader.len))
			break;

		switch(chunkHeader.id)
		{
		// DLS 1.0: Instruments Collection Header
		case IFFID_colh:
		#ifdef DLSBANK_LOG
			Log("colh (%d bytes)\n", chunkHeader.len);
		#endif
			if (m_Instruments.empty())
			{
				m_Instruments.resize(chunk.ReadUint32LE());
			#ifdef DLSBANK_LOG
				Log("  %d instruments\n", m_Instruments.size());
			#endif
			}
			break;

		// DLS 1.0: Instruments Pointers Table
		case IFFID_ptbl:
		#ifdef DLSBANK_LOG
			Log("ptbl (%d bytes)\n", chunkHeader.len);
		#endif
			if (m_WaveForms.empty())
			{
				PTBLCHUNK ptbl;
				chunk.ReadStruct(ptbl);
				chunk.Skip(ptbl.cbSize - 8);
				uint32 cues = std::min(ptbl.cCues.get(), mpt::saturate_cast<uint32>(chunk.BytesLeft() / sizeof(uint32)));
				m_WaveForms.reserve(cues);
				for(uint32 i = 0; i < cues; i++)
				{
					m_WaveForms.push_back(chunk.ReadUint32LE());
				}
			#ifdef DLSBANK_LOG
				Log("  %d waveforms\n", m_WaveForms.size());
			#endif
			}
			break;

		// DLS 1.0: LIST section
		case IFFID_LIST:
		#ifdef DLSBANK_LOG
			Log("LIST\n");
		#endif
			{
				uint32 listid = chunk.ReadUint32LE();
				if (((listid == IFFID_wvpl) && (m_nType & SOUNDBANK_TYPE_DLS))
				 || ((listid == IFFID_sdta) && (m_nType & SOUNDBANK_TYPE_SF2)))
				{
					m_dwWavePoolOffset = dwMemPos + 4;
				#ifdef DLSBANK_LOG
					Log("Wave Pool offset: %d\n", m_dwWavePoolOffset);
				#endif
					break;
				}

				while (chunk.CanRead(12))
				{
					IFFCHUNK listHeader;
					const void *subData = chunk.GetRawData();
					chunk.ReadStruct(listHeader);

					if(!chunk.CanRead(listHeader.len))
						break;

					FileReader listChunk = chunk.ReadChunk(listHeader.len);
					if(listHeader.len % 2u)
						chunk.Skip(1);
					// DLS Instrument Headers
					if (listHeader.id == IFFID_LIST && (m_nType & SOUNDBANK_TYPE_DLS))
					{
						uint32 subID = listChunk.ReadUint32LE();
						if ((subID == IFFID_ins) && (nInsDef < m_Instruments.size()))
						{
							DLSINSTRUMENT *pDlsIns = &m_Instruments[nInsDef];
							//Log("Instrument %d:\n", nInsDef);
							UpdateInstrumentDefinition(pDlsIns, static_cast<const IFFCHUNK *>(subData), listHeader.len + 8);
							nInsDef++;
						}
					} else
					// DLS/SF2 Bank Information
					if (listid == IFFID_INFO && listHeader.len)
					{
						switch(listHeader.id)
						{
						case IFFID_INAM:
							listChunk.ReadString<mpt::String::maybeNullTerminated>(m_BankInfo.szBankName, listChunk.BytesLeft());
							break;
						case IFFID_IENG:
							listChunk.ReadString<mpt::String::maybeNullTerminated>(m_BankInfo.szEngineer, listChunk.BytesLeft());
							break;
						case IFFID_ICOP:
							listChunk.ReadString<mpt::String::maybeNullTerminated>(m_BankInfo.szCopyRight, listChunk.BytesLeft());
							break;
						case IFFID_ICMT:
							listChunk.ReadString<mpt::String::maybeNullTerminated>(m_BankInfo.szComments, listChunk.BytesLeft());
							break;
						case IFFID_ISFT:
							listChunk.ReadString<mpt::String::maybeNullTerminated>(m_BankInfo.szSoftware, listChunk.BytesLeft());
							break;
						case IFFID_ISBJ:
							listChunk.ReadString<mpt::String::maybeNullTerminated>(m_BankInfo.szDescription, listChunk.BytesLeft());
							break;
						}
					} else
					if ((listid == IFFID_pdta) && (m_nType & SOUNDBANK_TYPE_SF2))
					{
						UpdateSF2PresetData(sf2info, listHeader, listChunk);
					}
				}
			}
			break;

		#ifdef DLSBANK_LOG
		default:
			{
				char sdbg[5];
				memcpy(sdbg, &chunkHeader.id, 4);
				sdbg[4] = 0;
				Log("Unsupported chunk: %s (%d bytes)\n", sdbg, chunkHeader.len);
			}
			break;
		#endif
		}
	}
	// Build the ptbl is not present in file
	if ((m_WaveForms.empty()) && (m_dwWavePoolOffset) && (m_nType & SOUNDBANK_TYPE_DLS) && (m_nMaxWaveLink > 0))
	{
	#ifdef DLSBANK_LOG
		Log("ptbl not present: building table (%d wavelinks)...\n", m_nMaxWaveLink);
	#endif
		m_WaveForms.reserve(m_nMaxWaveLink);
		dwMemPos = m_dwWavePoolOffset;
		while (dwMemPos + sizeof(IFFCHUNK) < dwMemLength)
		{
			IFFCHUNK *pchunk = (IFFCHUNK *)(lpMemFile + dwMemPos);
			if (pchunk->id == IFFID_LIST) m_WaveForms.push_back(dwMemPos - m_dwWavePoolOffset);
			dwMemPos += 8 + pchunk->len;
			if (m_WaveForms.size() >= m_nMaxWaveLink) break;
		}
#ifdef DLSBANK_LOG
		Log("Found %d waveforms\n", m_WaveForms.size());
#endif
	}
	// Convert the SF2 data to DLS
	if ((m_nType & SOUNDBANK_TYPE_SF2) && !m_SamplesEx.empty() && !m_Instruments.empty())
	{
		ConvertSF2ToDLS(sf2info);
	}
#ifdef DLSBANK_LOG
	Log("DLS bank closed\n");
#endif
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////
// Extracts the WaveForms from a DLS bank

uint32 CDLSBank::GetRegionFromKey(uint32 nIns, uint32 nKey)
{
	DLSINSTRUMENT *pDlsIns;

	if (nIns >= m_Instruments.size()) return 0;
	pDlsIns = &m_Instruments[nIns];
	for (uint32 rgn=0; rgn<pDlsIns->nRegions; rgn++)
	{
		if ((nKey >= pDlsIns->Regions[rgn].uKeyMin) && (nKey <= pDlsIns->Regions[rgn].uKeyMax))
		{
			return rgn;
		}
	}
	return 0;
}


bool CDLSBank::ExtractWaveForm(uint32 nIns, uint32 nRgn, std::vector<uint8> &waveData, uint32 &length)
{
	waveData.clear();
	length = 0;

	if (nIns >= m_Instruments.size() || !m_dwWavePoolOffset)
	{
	#ifdef DLSBANK_LOG
		Log("ExtractWaveForm(%d) failed: m_Instruments.size()=%d m_dwWavePoolOffset=%d m_WaveForms.size()=%d\n", nIns, m_Instruments.size(), m_dwWavePoolOffset, m_WaveForms.size());
	#endif
		return false;
	}
	DLSINSTRUMENT &dlsIns = m_Instruments[nIns];
	if (nRgn >= dlsIns.nRegions)
	{
	#ifdef DLSBANK_LOG
		Log("invalid waveform region: nIns=%d nRgn=%d pSmp->nRegions=%d\n", nIns, nRgn, pSmp->nRegions);
	#endif
		return false;
	}
	uint32 nWaveLink = dlsIns.Regions[nRgn].nWaveLink;
	if(nWaveLink >= m_WaveForms.size())
	{
	#ifdef DLSBANK_LOG
		Log("Invalid wavelink id: nWaveLink=%d nWaveForms=%d\n", nWaveLink, m_WaveForms.size());
	#endif
		return false;
	}

	uint32 dwOffset = m_WaveForms[nWaveLink] + m_dwWavePoolOffset;
	FILE *f = mpt_fopen(m_szFileName, "rb");
	if(f == nullptr) return false;
	if (fseek(f, dwOffset, SEEK_SET) == 0)
	{
		if (m_nType & SOUNDBANK_TYPE_SF2)
		{
			if (m_SamplesEx[nWaveLink].dwLen)
			{
				if (fseek(f, 8, SEEK_CUR) == 0)
				{
					length = m_SamplesEx[nWaveLink].dwLen;
					try
					{
						waveData.assign(length + 8, 0);
						fread(waveData.data(), 1, length, f);
					} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
					{
						MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
					}
				}
			}
		} else
		{
			LISTCHUNK chunk;
			if (fread(&chunk, 1, 12, f) == 12)
			{
				if ((chunk.id == IFFID_LIST) && (chunk.listid == IFFID_wave) && (chunk.len > 4))
				{
					length = chunk.len + 8;
					try
					{
						waveData.assign(chunk.len + 8, 0);
						memcpy(waveData.data(), &chunk, 12);
						fread(&waveData[12], 1, length - 12, f);
					} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
					{
						MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
					}
				}
			}
		}
	}
	fclose(f);
	return !waveData.empty();
}


bool CDLSBank::ExtractSample(CSoundFile &sndFile, SAMPLEINDEX nSample, uint32 nIns, uint32 nRgn, int transpose)
{
	DLSINSTRUMENT *pDlsIns;
	std::vector<uint8> pWaveForm;
	uint32 dwLen = 0;
	bool bOk, bWaveForm;

	if (nIns >= m_Instruments.size()) return false;
	pDlsIns = &m_Instruments[nIns];
	if (nRgn >= pDlsIns->nRegions) return false;
	if (!ExtractWaveForm(nIns, nRgn, pWaveForm, dwLen)) return false;
	if (dwLen < 16) return false;
	bOk = false;

	FileReader wsmpChunk;
	if (m_nType & SOUNDBANK_TYPE_SF2)
	{
		sndFile.DestroySample(nSample);
		uint32 nWaveLink = pDlsIns->Regions[nRgn].nWaveLink;
		ModSample &sample = sndFile.GetSample(nSample);
		if (sndFile.m_nSamples < nSample) sndFile.m_nSamples = nSample;
		if (nWaveLink < m_SamplesEx.size())
		{
			DLSSAMPLEEX *p = &m_SamplesEx[nWaveLink];
		#ifdef DLSINSTR_LOG
			Log("  SF2 WaveLink #%3d: %5dHz\n", nWaveLink, p->dwSampleRate);
		#endif
			sample.Initialize();
			sample.nLength = dwLen / 2;
			sample.nLoopStart = pDlsIns->Regions[nRgn].ulLoopStart;
			sample.nLoopEnd = pDlsIns->Regions[nRgn].ulLoopEnd;
			sample.nC5Speed = p->dwSampleRate;
			sample.RelativeTone = p->byOriginalPitch;
			sample.nFineTune = p->chPitchCorrection;
			if (p->szName[0])
				mpt::String::Copy(sndFile.m_szNames[nSample], p->szName);
			else if(pDlsIns->szName[0])
				mpt::String::Copy(sndFile.m_szNames[nSample], pDlsIns->szName);

			FileReader chunk(mpt::as_span(pWaveForm.data(), dwLen));
			SampleIO(
				SampleIO::_16bit,
				SampleIO::mono,
				SampleIO::littleEndian,
				SampleIO::signedPCM)
				.ReadSample(sample, chunk);
		}
		bWaveForm = sample.pSample != nullptr;
	} else
	{
		FileReader file(mpt::as_span(pWaveForm.data(), dwLen));
		bWaveForm = sndFile.ReadWAVSample(nSample, file, false, &wsmpChunk);
		if(pDlsIns->szName[0])
			mpt::String::Copy(sndFile.m_szNames[nSample], pDlsIns->szName);
	}
	if (bWaveForm)
	{
		ModSample &sample = sndFile.GetSample(nSample);
		DLSREGION *pRgn = &pDlsIns->Regions[nRgn];
		sample.uFlags.reset(CHN_LOOP | CHN_PINGPONGLOOP | CHN_SUSTAINLOOP | CHN_PINGPONGSUSTAIN);
		if (pRgn->fuOptions & DLSREGION_SAMPLELOOP) sample.uFlags.set(CHN_LOOP);
		if (pRgn->fuOptions & DLSREGION_SUSTAINLOOP) sample.uFlags.set(CHN_SUSTAINLOOP);
		if (pRgn->fuOptions & DLSREGION_PINGPONGLOOP) sample.uFlags.set(CHN_PINGPONGLOOP);
		if (sample.uFlags[CHN_LOOP | CHN_SUSTAINLOOP])
		{
			if (pRgn->ulLoopEnd > pRgn->ulLoopStart)
			{
				if (sample.uFlags[CHN_SUSTAINLOOP])
				{
					sample.nSustainStart = pRgn->ulLoopStart;
					sample.nSustainEnd = pRgn->ulLoopEnd;
				} else
				{
					sample.nLoopStart = pRgn->ulLoopStart;
					sample.nLoopEnd = pRgn->ulLoopEnd;
				}
			} else
			{
				sample.uFlags.reset(CHN_LOOP|CHN_SUSTAINLOOP);
			}
		}
		// WSMP chunk
		{
			uint32 usUnityNote = pRgn->uUnityNote;
			int sFineTune = pRgn->sFineTune;
			int lVolume = pRgn->usVolume;

			WSMPCHUNK wsmp;
			if(!(pRgn->fuOptions & DLSREGION_OVERRIDEWSMP) && wsmpChunk.IsValid() && wsmpChunk.ReadStructPartial(wsmp))
			{
				usUnityNote = wsmp.usUnityNote;
				sFineTune = wsmp.sFineTune;
				lVolume = DLS32BitRelativeGainToLinear(wsmp.lAttenuation) / 256;
				if(wsmp.cSampleLoops)
				{
					WSMPSAMPLELOOP loop;
					wsmpChunk.Skip(8 + wsmp.cbSize);
					wsmpChunk.ReadStruct(loop);
					if(loop.ulLoopLength > 3)
					{
						sample.uFlags.set(CHN_LOOP);
						//if (loop.ulLoopType) sample.uFlags |= CHN_PINGPONGLOOP;
						sample.nLoopStart = loop.ulLoopStart;
						sample.nLoopEnd = loop.ulLoopStart + loop.ulLoopLength;
					}
				}
			} else if (m_nType & SOUNDBANK_TYPE_SF2)
			{
				usUnityNote = (usUnityNote < 0x80) ? usUnityNote : sample.RelativeTone;
				sFineTune += sample.nFineTune;
			}
		#ifdef DLSINSTR_LOG
			Log("WSMP: usUnityNote=%d.%d, %dHz (transp=%d)\n", usUnityNote, sFineTune, sample.nC5Speed, transpose);
		#endif
			if (usUnityNote > 0x7F) usUnityNote = 60;
			int steps = (60 + transpose - usUnityNote) * 128 + sFineTune;
			sample.Transpose(steps * (1.0 / (12.0 * 128.0)));

			Limit(lVolume, 16, 256);
			sample.nGlobalVol = (uint8)(lVolume / 4);	// 0-64
		}
		sample.nPan = GetPanning(nIns, nRgn);

		sample.Convert(MOD_TYPE_IT, sndFile.GetType());
		sample.PrecomputeLoops(sndFile, false);
		bOk = true;
	}
	return bOk;
}


static uint16 ScaleEnvelope(uint32 time, float tempoScale)
{
	return std::max<uint16>(Util::Round<uint16>(time * tempoScale), 1);
}


bool CDLSBank::ExtractInstrument(CSoundFile &sndFile, INSTRUMENTINDEX nInstr, uint32 nIns, uint32 nDrumRgn)
{
	SAMPLEINDEX RgnToSmp[DLSMAXREGIONS];
	DLSINSTRUMENT *pDlsIns;
	ModInstrument *pIns;
	uint32 nRgnMin, nRgnMax, nEnv;

	if (nIns >= m_Instruments.size()) return false;
	pDlsIns = &m_Instruments[nIns];
	if (pDlsIns->ulBank & F_INSTRUMENT_DRUMS)
	{
		if (nDrumRgn >= pDlsIns->nRegions) return false;
		nRgnMin = nDrumRgn;
		nRgnMax = nDrumRgn+1;
		nEnv = pDlsIns->Regions[nDrumRgn].uPercEnv;
	} else
	{
		if (!pDlsIns->nRegions) return false;
		nRgnMin = 0;
		nRgnMax = pDlsIns->nRegions;
		nEnv = pDlsIns->nMelodicEnv;
	}
#ifdef DLSINSTR_LOG
	Log("DLS Instrument #%d: %s\n", nIns, pDlsIns->szName);
	Log("  Bank=0x%04X Instrument=0x%04X\n", pDlsIns->ulBank, pDlsIns->ulInstrument);
	Log("  %2d regions, nMelodicEnv=%d\n", pDlsIns->nRegions, pDlsIns->nMelodicEnv);
	for (uint32 iDbg=0; iDbg<pDlsIns->nRegions; iDbg++)
	{
		DLSREGION *prgn = &pDlsIns->Regions[iDbg];
		Log(" Region %d:\n", iDbg);
		Log("  WaveLink = %d (loop [%5d, %5d])\n", prgn->nWaveLink, prgn->ulLoopStart, prgn->ulLoopEnd);
		Log("  Key Range: [%2d, %2d]\n", prgn->uKeyMin, prgn->uKeyMax);
		Log("  fuOptions = 0x%04X\n", prgn->fuOptions);
		Log("  usVolume = %3d, Unity Note = %d\n", prgn->usVolume, prgn->uUnityNote);
	}
#endif

	pIns = new (std::nothrow) ModInstrument();
	if(pIns == nullptr)
	{
		return false;
	}

	if (sndFile.Instruments[nInstr])
	{
		sndFile.DestroyInstrument(nInstr, deleteAssociatedSamples);
	}
	// Initializes Instrument
	if (pDlsIns->ulBank & F_INSTRUMENT_DRUMS)
	{
		char s[64] = "";
		uint32 key = pDlsIns->Regions[nDrumRgn].uKeyMin;
		if ((key >= 24) && (key <= 84)) lstrcpyA(s, szMidiPercussionNames[key-24]);
		if (pDlsIns->szName[0])
		{
			sprintf(&s[strlen(s)], " (%s", pDlsIns->szName);
			size_t n = strlen(s);
			while ((n) && (s[n-1] == ' '))
			{
				n--;
				s[n] = 0;
			}
			lstrcatA(s, ")");
		}
		mpt::String::Copy(pIns->name, s);
	} else
	{
		mpt::String::Copy(pIns->name, pDlsIns->szName);
	}
	int nTranspose = 0;
	if (pDlsIns->ulBank & F_INSTRUMENT_DRUMS)
	{
		for (uint32 iNoteMap=0; iNoteMap<NOTE_MAX; iNoteMap++)
		{
			if(sndFile.GetType() & (MOD_TYPE_IT|MOD_TYPE_MID|MOD_TYPE_MPT))
			{
				// Formate has instrument note mapping
				if (iNoteMap < pDlsIns->Regions[nDrumRgn].uKeyMin) pIns->NoteMap[iNoteMap] = (uint8)(pDlsIns->Regions[nDrumRgn].uKeyMin + 1);
				if (iNoteMap > pDlsIns->Regions[nDrumRgn].uKeyMax) pIns->NoteMap[iNoteMap] = (uint8)(pDlsIns->Regions[nDrumRgn].uKeyMax + 1);
			} else
			{
				if (iNoteMap == pDlsIns->Regions[nDrumRgn].uKeyMin)
				{
					nTranspose = (pDlsIns->Regions[nDrumRgn].uKeyMin + (pDlsIns->Regions[nDrumRgn].uKeyMax - pDlsIns->Regions[nDrumRgn].uKeyMin) / 2) - 60;
				}
			}
		}
	}
	pIns->nFadeOut = 1024;
	pIns->nMidiProgram = (uint8)(pDlsIns->ulInstrument & 0x7F) + 1;
	pIns->nMidiChannel = (uint8)((pDlsIns->ulBank & F_INSTRUMENT_DRUMS) ? 10 : 0);
	pIns->wMidiBank = (uint16)(((pDlsIns->ulBank & 0x7F00) >> 1) | (pDlsIns->ulBank & 0x7F));
	pIns->nNNA = NNA_NOTEOFF;
	pIns->nDCT = DCT_NOTE;
	pIns->nDNA = DNA_NOTEFADE;
	sndFile.Instruments[nInstr] = pIns;
	uint32 nLoadedSmp = 0;
	SAMPLEINDEX nextSample = 0;
	// Extract Samples
	for (uint32 nRgn=nRgnMin; nRgn<nRgnMax; nRgn++)
	{
		bool bDupRgn = false;
		SAMPLEINDEX nSmp = 0;
		DLSREGION *pRgn = &pDlsIns->Regions[nRgn];
		// Elimitate Duplicate Regions
		uint32 iDup;
		for (iDup=nRgnMin; iDup<nRgn; iDup++)
		{
			DLSREGION *pRgn2 = &pDlsIns->Regions[iDup];
			if (((pRgn2->nWaveLink == pRgn->nWaveLink)
			  && (pRgn2->ulLoopEnd == pRgn->ulLoopEnd)
			  && (pRgn2->ulLoopStart == pRgn->ulLoopStart))
			 || ((pRgn2->uKeyMin == pRgn->uKeyMin)
			  && (pRgn2->uKeyMax == pRgn->uKeyMax)))
			{
				bDupRgn = true;
				nSmp = RgnToSmp[iDup];
				break;
			}
		}
		// Create a new sample
		if (!bDupRgn)
		{
			uint32 nmaxsmp = (m_nType & MOD_TYPE_XM) ? 16 : 32;
			if (nLoadedSmp >= nmaxsmp)
			{
				nSmp = RgnToSmp[nRgn-1];
			} else
			{
				nextSample = sndFile.GetNextFreeSample(nInstr, nextSample + 1);
				if (nextSample == SAMPLEINDEX_INVALID) break;
				if (nextSample > sndFile.GetNumSamples()) sndFile.m_nSamples = nextSample;
				nSmp = nextSample;
				nLoadedSmp++;
			}
		}

		RgnToSmp[nRgn] = nSmp;
		// Map all notes to the right sample
		if (nSmp)
		{
			for (uint32 iKey=0; iKey<NOTE_MAX; iKey++)
			{
				if ((nRgn == nRgnMin) || ((iKey >= pRgn->uKeyMin) && (iKey <= pRgn->uKeyMax)))
				{
					pIns->Keyboard[iKey] = nSmp;
				}
			}
			// Load the sample
			if(!bDupRgn || sndFile.GetSample(nSmp).pSample == nullptr)
			{
				ExtractSample(sndFile, nSmp, nIns, nRgn, nTranspose);
			} else if(sndFile.GetSample(nSmp).GetNumChannels() == 1)
			{
				// Try to combine stereo samples
				uint8 pan1 = GetPanning(nIns, nRgn), pan2 = GetPanning(nIns, iDup);
				if((pan1 == 0 || pan1 == 255) && (pan2 == 0 || pan2 == 255))
				{
					ModSample &sample = sndFile.GetSample(nSmp);
					ctrlSmp::ConvertToStereo(sample, sndFile);
					std::vector<uint8> pWaveForm;
					uint32 dwLen = 0;
					if(ExtractWaveForm(nIns, nRgn, pWaveForm, dwLen) && dwLen >= sample.GetSampleSizeInBytes() / 2)
					{
						SmpLength len = sample.nLength;
						const int16 *src = reinterpret_cast<int16 *>(pWaveForm.data());
						int16 *dst = sample.pSample16 + ((pan1 == 0) ? 0 : 1);
						while(len--)
						{
							*dst = *src;
							src++;
							dst += 2;
						}
					}
				}
			}
		}
	}

	float tempoScale = 1.0f;
	if(sndFile.m_nTempoMode == tempoModeModern)
	{
		uint32 ticksPerBeat = sndFile.m_nDefaultRowsPerBeat * sndFile.m_nDefaultSpeed;
		if(ticksPerBeat == 0)
			ticksPerBeat = 24;
		tempoScale = ticksPerBeat / 24.0f;
	}

	// Initializes Envelope
	if ((nEnv) && (nEnv <= m_Envelopes.size()))
	{
		DLSENVELOPE *part = &m_Envelopes[nEnv-1];
		// Volume Envelope
		if ((part->wVolAttack) || (part->wVolDecay < 20*50) || (part->nVolSustainLevel) || (part->wVolRelease < 20*50))
		{
			pIns->VolEnv.dwFlags.set(ENV_ENABLED);
			// Delay section
			// -> DLS level 2
			// Attack section
			pIns->VolEnv.clear();
			if (part->wVolAttack)
			{
				pIns->VolEnv.push_back(0, (uint8)(ENVELOPE_MAX / (part->wVolAttack / 2 + 2) + 8)); //	/-----
				pIns->VolEnv.push_back(ScaleEnvelope(part->wVolAttack, tempoScale), ENVELOPE_MAX); //	|
			} else
			{
				pIns->VolEnv.push_back(0, ENVELOPE_MAX);
			}
			// Hold section
			// -> DLS Level 2
			// Sustain Level
			if (part->nVolSustainLevel > 0)
			{
				if (part->nVolSustainLevel < 128)
				{
					uint16 lStartTime = pIns->VolEnv.back().tick;
					int32 lSusLevel = - DLS32BitRelativeLinearToGain(part->nVolSustainLevel << 9) / 65536;
					int32 lDecayTime = 1;
					if (lSusLevel > 0)
					{
						lDecayTime = (lSusLevel * (int32)part->wVolDecay) / 960;
						for (uint32 i=0; i<7; i++)
						{
							int32 lFactor = 128 - (1 << i);
							if (lFactor <= part->nVolSustainLevel) break;
							int32 lev = - DLS32BitRelativeLinearToGain(lFactor << 9) / 65536;
							if (lev > 0)
							{
								int32 ltime = (lev * (int32)part->wVolDecay) / 960;
								if ((ltime > 1) && (ltime < lDecayTime))
								{
									uint16 tick = lStartTime + ScaleEnvelope(ltime, tempoScale);
									if(tick > pIns->VolEnv.back().tick)
									{
										pIns->VolEnv.push_back(tick, (uint8)(lFactor / 2));
									}
								}
							}
						}
					}

					uint16 decayEnd = lStartTime + ScaleEnvelope(lDecayTime, tempoScale);
					if (decayEnd > pIns->VolEnv.back().tick)
					{
						pIns->VolEnv.push_back(decayEnd, (uint8)((part->nVolSustainLevel+1) / 2));
					}
				}
				pIns->VolEnv.dwFlags.set(ENV_SUSTAIN);
			} else
			{
				pIns->VolEnv.dwFlags.set(ENV_SUSTAIN);
				pIns->VolEnv.push_back(pIns->VolEnv.back().tick + 1u, pIns->VolEnv.back().value);
			}
			pIns->VolEnv.nSustainStart = pIns->VolEnv.nSustainEnd = (uint8)(pIns->VolEnv.size() - 1);
			// Release section
			if ((part->wVolRelease) && (pIns->VolEnv.back().value > 1))
			{
				int32 lReleaseTime = part->wVolRelease;
				uint16 lStartTime = pIns->VolEnv.back().tick;
				int32 lStartFactor = pIns->VolEnv.back().value;
				int32 lSusLevel = - DLS32BitRelativeLinearToGain(lStartFactor << 10) / 65536;
				int32 lDecayEndTime = (lReleaseTime * lSusLevel) / 960;
				lReleaseTime -= lDecayEndTime;
				for (uint32 i=0; i<5; i++)
				{
					int32 lFactor = 1 + ((lStartFactor * 3) >> (i+2));
					if ((lFactor <= 1) || (lFactor >= lStartFactor)) continue;
					int32 lev = - DLS32BitRelativeLinearToGain(lFactor << 10) / 65536;
					if (lev > 0)
					{
						int32 ltime = (((int32)part->wVolRelease * lev) / 960) - lDecayEndTime;
						if ((ltime > 1) && (ltime < lReleaseTime))
						{
							uint16 tick = lStartTime + ScaleEnvelope(ltime, tempoScale);
							if(tick > pIns->VolEnv.back().tick)
							{
								pIns->VolEnv.push_back(tick, (uint8)lFactor);
							}
						}
					}
				}
				if (lReleaseTime < 1) lReleaseTime = 1;
				auto releaseTicks = ScaleEnvelope(lReleaseTime, tempoScale);
				pIns->VolEnv.push_back(lStartTime + releaseTicks, ENVELOPE_MIN);
				if(releaseTicks > 0)
				{
					pIns->nFadeOut = 32768 / releaseTicks;
				}
			} else
			{
				pIns->VolEnv.push_back(pIns->VolEnv.back().tick + 1u, ENVELOPE_MIN);
			}
		}
	}
	if (pDlsIns->ulBank & F_INSTRUMENT_DRUMS)
	{
		// Create a default envelope for drums
		pIns->VolEnv.dwFlags.reset(ENV_SUSTAIN);
		if(!pIns->VolEnv.dwFlags[ENV_ENABLED])
		{
			pIns->VolEnv.dwFlags.set(ENV_ENABLED);
			pIns->VolEnv.resize(4);
			pIns->VolEnv[0] = EnvelopeNode(0, ENVELOPE_MAX);
			pIns->VolEnv[1] = EnvelopeNode(ScaleEnvelope(5, tempoScale), ENVELOPE_MAX);
			pIns->VolEnv[2] = EnvelopeNode(pIns->VolEnv[1].tick * 2u, ENVELOPE_MID);
			pIns->VolEnv[3] = EnvelopeNode(pIns->VolEnv[2].tick * 2u, ENVELOPE_MIN);	// 1 second max. for drums
		}
	}
	return true;
}


const char *CDLSBank::GetRegionName(uint32 nIns, uint32 nRgn) const
{
	if (nIns >= m_Instruments.size()) return nullptr;
	const DLSINSTRUMENT &dlsIns = m_Instruments[nIns];
	if (nRgn >= dlsIns.nRegions) return nullptr;

	if (m_nType & SOUNDBANK_TYPE_SF2)
	{
		uint32 nWaveLink = dlsIns.Regions[nRgn].nWaveLink;
		if (nWaveLink < m_SamplesEx.size())
		{
			return m_SamplesEx[nWaveLink].szName;
		}
	}
	return nullptr;
}


uint8 CDLSBank::GetPanning(uint32 ins, uint32 region) const
{
	const DLSINSTRUMENT &dlsIns = m_Instruments[ins];
	if(region >= CountOf(dlsIns.Regions))
		return 128;
	const DLSREGION &rgn = dlsIns.Regions[region];
	if(dlsIns.ulBank & F_INSTRUMENT_DRUMS)
	{
		if(rgn.uPercEnv > 0 && rgn.uPercEnv <= m_Envelopes.size())
		{
			return m_Envelopes[rgn.uPercEnv - 1].nDefPan;
		}
	} else
	{
		if(dlsIns.nMelodicEnv > 0 && dlsIns.nMelodicEnv <= m_Envelopes.size())
		{
			return m_Envelopes[dlsIns.nMelodicEnv - 1].nDefPan;
		}
	}
	return 128;
}


#else // !MODPLUG_TRACKER

MPT_MSVC_WORKAROUND_LNK4221(Dlsbank)

#endif // MODPLUG_TRACKER


OPENMPT_NAMESPACE_END
