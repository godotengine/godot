/*
 * InstrumentExtensions.cpp
 * ------------------------
 * Purpose: Instrument properties I/O
 * Notes  : Welcome to the absolutely horrible abominations that are the "extended instrument properties"
 *          which are some of the earliest additions OpenMPT did to the IT / XM format. They are ugly,
 *          and the way they work even differs between IT/XM and ITI/XI/ITP.
 *          Yes, the world would be a better place without this stuff.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"

OPENMPT_NAMESPACE_BEGIN

/*---------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------
MODULAR (in/out) ModInstrument :
-----------------------------------------------------------------------------------------------

* to update:
------------

- both following functions need to be updated when adding a new member in ModInstrument :

void WriteInstrumentHeaderStructOrField(ModInstrument * input, FILE * file, uint32 only_this_code, int16 fixedsize);
bool ReadInstrumentHeaderField(ModInstrument * input, uint32 fcode, int16 fsize, FileReader &file);

- see below for body declaration.


* members:
----------

- 32bit identification CODE tag (must be unique)
- 16bit content SIZE in byte(s)
- member field


* CODE tag naming convention:
-----------------------------

- have a look below in current tag dictionnary
- take the initial ones of the field name
- 4 caracters code (not more, not less)
- must be filled with '.' caracters if code has less than 4 caracters
- for arrays, must include a '[' caracter following significant caracters ('.' not significant!!!)
- use only caracters used in full member name, ordered as they appear in it
- match caracter attribute (small,capital)

Example with "PanEnv.nLoopEnd" , "PitchEnv.nLoopEnd" & "VolEnv.Values[MAX_ENVPOINTS]" members :
- use 'PLE.' for PanEnv.nLoopEnd
- use 'PiLE' for PitchEnv.nLoopEnd
- use 'VE[.' for VolEnv.Values[MAX_ENVPOINTS]


* In use CODE tag dictionary (alphabetical order):
--------------------------------------------------

						!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						!!! SECTION TO BE UPDATED !!!
						!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

		[EXT]	means external (not related) to ModInstrument content

AUTH	[EXT]	Song artist
C...	[EXT]	nChannels
ChnS	[EXT]	IT/MPTM: Channel settings for channels 65-127 if needed (doesn't fit to IT header).
CS..			nCutSwing
CUES	[EXT]	Sample cue points
CWV.	[EXT]	dwCreatedWithVersion
DCT.			nDCT;
dF..			dwFlags;
DGV.	[EXT]	nDefaultGlobalVolume
DT..	[EXT]	nDefaultTempo;
DTFR	[EXT]	Fractional part of default tempo
DNA.			nDNA;
EBIH	[EXT]	embeded instrument header tag (ITP file format)
FM..			nFilterMode;
fn[.			filename[12];
FO..			nFadeOut;
GV..			nGlobalVol;
IFC.			nIFC;
IFR.			nIFR;
K[.				Keyboard[128];
LSWV	[EXT]	Last Saved With Version
MB..			wMidiBank;
MC..			nMidiChannel;
MDK.			nMidiDrumKey;
MIMA	[EXT]									MIdi MApping directives
MiP.			nMixPlug;
MP..			nMidiProgram;
MPTS	[EXT]									Extra song info tag
MPTX	[EXT]									EXTRA INFO tag
MSF.	[EXT]									Mod(Specific)Flags
n[..			name[32];
NNA.			nNNA;
NM[.			NoteMap[128];
P...			nPan;
PE..			PanEnv.nNodes;
PE[.			PanEnv.Values[MAX_ENVPOINTS];
PiE.			PitchEnv.nNodes;
PiE[			PitchEnv.Values[MAX_ENVPOINTS];
PiLE			PitchEnv.nLoopEnd;
PiLS			PitchEnv.nLoopStart;
PiP[			PitchEnv.Ticks[MAX_ENVPOINTS];
PiSB			PitchEnv.nSustainStart;
PiSE			PitchEnv.nSustainEnd;
PLE.			PanEnv.nLoopEnd;
PLS.			PanEnv.nLoopStart;
PMM.	[EXT]	nPlugMixMode;
PP[.			PanEnv.Ticks[MAX_ENVPOINTS];
PPC.			nPPC;
PPS.			nPPS;
PS..			nPanSwing;
PSB.			PanEnv.nSustainStart;
PSE.			PanEnv.nSustainEnd;
PTTL			pitchToTempoLock;
PTTF			pitchToTempoLock (fractional part);
PVEH			nPluginVelocityHandling;
PVOH			nPluginVolumeHandling;
R...			nResampling;
RP..	[EXT]	nRestartPos;
RPB.	[EXT]	nRowsPerBeat;
RPM.	[EXT]	nRowsPerMeasure;
RS..			nResSwing;
RSMP	[EXT]	Global resampling
SEP@	[EXT]	chunk SEPARATOR tag
SPA.	[EXT]	m_nSamplePreAmp;
TM..	[EXT]	nTempoMode;
VE..			VolEnv.nNodes;
VE[.			VolEnv.Values[MAX_ENVPOINTS];
VLE.			VolEnv.nLoopEnd;
VLS.			VolEnv.nLoopStart;
VP[.			VolEnv.Ticks[MAX_ENVPOINTS];
VR..			nVolRampUp;
VS..			nVolSwing;
VSB.			VolEnv.nSustainStart;
VSE.			VolEnv.nSustainEnd;
VSTV	[EXT]	nVSTiVolume;
PERN			PitchEnv.nReleaseNode
AERN			PanEnv.nReleaseNode
VERN			VolEnv.nReleaseNode
PFLG			PitchEnv.dwFlag
AFLG			PanEnv.dwFlags
VFLG			VolEnv.dwFlags
MPWD			MIDI Pitch Wheel Depth
-----------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------*/

#ifndef MODPLUG_NO_FILESAVE

template<typename T, bool is_signed> struct IsNegativeFunctor { bool operator()(T val) const { return val < 0; } };
template<typename T> struct IsNegativeFunctor<T, true> { bool operator()(T val) const { return val < 0; } };
template<typename T> struct IsNegativeFunctor<T, false> { bool operator()(T /*val*/) const { return false; } };

template<typename T>
bool IsNegative(const T &val)
{
	return IsNegativeFunctor<T, std::numeric_limits<T>::is_signed>()(val);
}

// ------------------------------------------------------------------------------------------
// Convenient macro to help WRITE_HEADER declaration for single type members ONLY (non-array)
// ------------------------------------------------------------------------------------------
#define WRITE_MPTHEADER_sized_member(name,type,code) \
	static_assert(sizeof(input->name) == sizeof(type), "Instrument property does match specified type!");\
	fcode = code;\
	fsize = sizeof( type );\
	if(writeAll) \
	{ \
		mpt::IO::WriteIntLE<uint32>(file, fcode); \
		mpt::IO::WriteIntLE<uint16>(file, fsize); \
	} else if(only_this_code == fcode)\
	{ \
		MPT_ASSERT(fixedsize == fsize); \
	} \
	if(only_this_code == fcode || only_this_code == Util::MaxValueOfType(only_this_code)) \
	{ \
		type tmp = input-> name; \
		tmp = SwapBytesLE(tmp); \
		fwrite(&tmp , 1 , fsize , file); \
	} \
/**/

// -----------------------------------------------------------------------------------------------------
// Convenient macro to help WRITE_HEADER declaration for single type members which are written truncated
// -----------------------------------------------------------------------------------------------------
#define WRITE_MPTHEADER_trunc_member(name,type,code) \
	static_assert(sizeof(input->name) > sizeof(type), "Instrument property would not be truncated, use WRITE_MPTHEADER_sized_member instead!");\
	fcode = code;\
	fsize = sizeof( type );\
	if(writeAll) \
	{ \
		mpt::IO::WriteIntLE<uint32>(file, fcode); \
		mpt::IO::WriteIntLE<uint16>(file, fsize); \
		type tmp = (type)(input-> name ); \
		tmp = SwapBytesLE(tmp); \
		fwrite(&tmp , 1 , fsize , file); \
	} else if(only_this_code == fcode)\
	{ \
		/* hackish workaround to resolve mismatched size values: */ \
		/* nResampling was a long time declared as uint32 but these macro tables used uint16 and UINT. */ \
		/* This worked fine on little-endian, on big-endian not so much. Thus support writing size-mismatched fields. */ \
		MPT_ASSERT(fixedsize >= fsize); \
		type tmp = (type)(input-> name ); \
		tmp = SwapBytesLE(tmp); \
		fwrite(&tmp , 1 , fsize , file); \
		if(fixedsize > fsize) \
		{ \
			for(int16 i = 0; i < fixedsize - fsize; ++i) \
			{ \
				uint8 fillbyte = !IsNegative(tmp) ? 0 : 0xff; /* sign extend */ \
				fwrite(&fillbyte, 1, 1, file); \
			} \
		} \
	} \
/**/

// ------------------------------------------------------------------------
// Convenient macro to help WRITE_HEADER declaration for array members ONLY
// ------------------------------------------------------------------------
#define WRITE_MPTHEADER_array_member(name,type,code,arraysize) \
	STATIC_ASSERT(sizeof(type) == sizeof(input-> name [0])); \
	MPT_ASSERT(sizeof(input->name) >= sizeof(type) * arraysize);\
	fcode = code;\
	fsize = sizeof( type ) * arraysize;\
	if(writeAll) \
	{ \
		mpt::IO::WriteIntLE<uint32>(file, fcode); \
		mpt::IO::WriteIntLE<uint16>(file, fsize); \
	} else if(only_this_code == fcode)\
	{ \
		/* MPT_ASSERT(fixedsize <= fsize); */ \
		fsize = fixedsize; /* just trust the size we got passed */ \
	} \
	if(only_this_code == fcode || only_this_code == Util::MaxValueOfType(only_this_code)) \
	{ \
		for(std::size_t i = 0; i < fsize/sizeof(type); ++i) \
		{ \
			type tmp; \
			tmp = input-> name [i]; \
			tmp = SwapBytesLE(tmp); \
			fwrite(&tmp, 1, sizeof(type), file); \
		} \
	} \
/**/

// ------------------------------------------------------------------------
// Convenient macro to help WRITE_HEADER declaration for envelope members ONLY
// ------------------------------------------------------------------------
#define WRITE_MPTHEADER_envelope_member(envType,envField,type,code) \
	{\
		const InstrumentEnvelope &env = input->GetEnvelope(envType); \
		STATIC_ASSERT(sizeof(type) == sizeof(env[0]. envField)); \
		fcode = code;\
		fsize = mpt::saturate_cast<int16>(sizeof( type ) * env.size());\
		MPT_ASSERT(size_t(fsize) == sizeof( type ) * env.size()); \
		\
		if(writeAll) \
		{ \
			mpt::IO::WriteIntLE<uint32>(file, fcode); \
			mpt::IO::WriteIntLE<uint16>(file, fsize); \
		} else if(only_this_code == fcode)\
		{ \
			fsize = fixedsize; /* just trust the size we got passed */ \
		} \
		if(only_this_code == fcode || only_this_code == Util::MaxValueOfType(only_this_code)) \
		{ \
			uint32 maxNodes = std::min<uint32>(fsize/sizeof(type), env.size()); \
			for(uint32 i = 0; i < maxNodes; ++i) \
			{ \
				type tmp; \
				tmp = env[i]. envField; \
				tmp = SwapBytesLE(tmp); \
				fwrite(&tmp, 1, sizeof(type), file); \
			} \
			/* Not every instrument's envelope will be the same length. fill up with zeros. */ \
			for(uint32 i = maxNodes; i < fsize/sizeof(type); ++i) \
			{ \
				type tmp = 0; \
				tmp = SwapBytesLE(tmp); \
				fwrite(&tmp, 1, sizeof(type), file); \
			} \
		} \
	}\
/**/


// Write (in 'file') 'input' ModInstrument with 'code' & 'size' extra field infos for each member
void WriteInstrumentHeaderStructOrField(ModInstrument * input, FILE * file, uint32 only_this_code, uint16 fixedsize)
{
uint32 fcode;
uint16 fsize;
// If true, all extension are written to the file; otherwise only the specified extension is written.
// writeAll is true iff we are saving an instrument (or, hypothetically, the legacy ITP format)
const bool writeAll = only_this_code == Util::MaxValueOfType(only_this_code);

if(!writeAll)
{
	MPT_ASSERT(fixedsize > 0);
}

	WRITE_MPTHEADER_sized_member(	nFadeOut					, uint32	, MAGIC4BE('F','O','.','.')	)
	WRITE_MPTHEADER_sized_member(	nPan						, uint32	, MAGIC4BE('P','.','.','.')	)
	WRITE_MPTHEADER_sized_member(	VolEnv.size()				, uint32	, MAGIC4BE('V','E','.','.')	)
	WRITE_MPTHEADER_sized_member(	PanEnv.size()				, uint32	, MAGIC4BE('P','E','.','.')	)
	WRITE_MPTHEADER_sized_member(	PitchEnv.size()				, uint32	, MAGIC4BE('P','i','E','.')	)
	WRITE_MPTHEADER_sized_member(	wMidiBank					, uint16	, MAGIC4BE('M','B','.','.')	)
	WRITE_MPTHEADER_sized_member(	nMidiProgram				, uint8		, MAGIC4BE('M','P','.','.')	)
	WRITE_MPTHEADER_sized_member(	nMidiChannel				, uint8		, MAGIC4BE('M','C','.','.')	)
	WRITE_MPTHEADER_envelope_member(	ENV_VOLUME	, tick		, uint16	, MAGIC4BE('V','P','[','.')	)
	WRITE_MPTHEADER_envelope_member(	ENV_PANNING	, tick		, uint16	, MAGIC4BE('P','P','[','.')	)
	WRITE_MPTHEADER_envelope_member(	ENV_PITCH	, tick		, uint16	, MAGIC4BE('P','i','P','[')	)
	WRITE_MPTHEADER_envelope_member(	ENV_VOLUME	, value		, uint8		, MAGIC4BE('V','E','[','.')	)
	WRITE_MPTHEADER_envelope_member(	ENV_PANNING	, value		, uint8		, MAGIC4BE('P','E','[','.')	)
	WRITE_MPTHEADER_envelope_member(	ENV_PITCH	, value		, uint8		, MAGIC4BE('P','i','E','[')	)
	WRITE_MPTHEADER_sized_member(	nMixPlug					, uint8		, MAGIC4BE('M','i','P','.')	)
	WRITE_MPTHEADER_sized_member(	nVolRampUp					, uint16	, MAGIC4BE('V','R','.','.')	)
	WRITE_MPTHEADER_trunc_member(	nResampling					, uint16	, MAGIC4BE('R','.','.','.')	)
	WRITE_MPTHEADER_sized_member(	nCutSwing					, uint8		, MAGIC4BE('C','S','.','.')	)
	WRITE_MPTHEADER_sized_member(	nResSwing					, uint8		, MAGIC4BE('R','S','.','.')	)
	WRITE_MPTHEADER_sized_member(	nFilterMode					, uint8		, MAGIC4BE('F','M','.','.')	)
	WRITE_MPTHEADER_sized_member(	nPluginVelocityHandling		, uint8		, MAGIC4BE('P','V','E','H')	)
	WRITE_MPTHEADER_sized_member(	nPluginVolumeHandling		, uint8		, MAGIC4BE('P','V','O','H')	)
	WRITE_MPTHEADER_trunc_member(	pitchToTempoLock.GetInt()	, uint16	, MAGIC4BE('P','T','T','L')	)
	WRITE_MPTHEADER_trunc_member(	pitchToTempoLock.GetFract() , uint16	, MAGIC4LE('P','T','T','F')	)
	WRITE_MPTHEADER_sized_member(	PitchEnv.nReleaseNode		, uint8		, MAGIC4BE('P','E','R','N')	)
	WRITE_MPTHEADER_sized_member(	PanEnv.nReleaseNode			, uint8		, MAGIC4BE('A','E','R','N')	)
	WRITE_MPTHEADER_sized_member(	VolEnv.nReleaseNode			, uint8		, MAGIC4BE('V','E','R','N')	)
	WRITE_MPTHEADER_sized_member(	PitchEnv.dwFlags			, uint32	, MAGIC4BE('P','F','L','G')	)
	WRITE_MPTHEADER_sized_member(	PanEnv.dwFlags				, uint32	, MAGIC4BE('A','F','L','G')	)
	WRITE_MPTHEADER_sized_member(	VolEnv.dwFlags				, uint32	, MAGIC4BE('V','F','L','G')	)
	WRITE_MPTHEADER_sized_member(	midiPWD						, int8		, MAGIC4BE('M','P','W','D')	)
}


// Used only when saving IT, XM and MPTM.
// ITI, ITP saves using Ericus' macros etc...
// The reason is that ITs and XMs save [code][size][ins1.Value][ins2.Value]...
// whereas ITP saves [code][size][ins1.Value][code][size][ins2.Value]...
// too late to turn back....
void CSoundFile::SaveExtendedInstrumentProperties(INSTRUMENTINDEX nInstruments, FILE *f) const
{
	uint32 code = MAGIC4BE('M','P','T','X');	// write extension header code
	mpt::IO::WriteIntLE<uint32>(f, code);

	if (nInstruments == 0)
		return;

	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('V','R','.','.'), sizeof(ModInstrument().nVolRampUp),  f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('M','i','P','.'), sizeof(ModInstrument().nMixPlug),    f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('M','C','.','.'), sizeof(ModInstrument().nMidiChannel),f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('M','P','.','.'), sizeof(ModInstrument().nMidiProgram),f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('M','B','.','.'), sizeof(ModInstrument().wMidiBank),   f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('P','.','.','.'), sizeof(ModInstrument().nPan),        f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('F','O','.','.'), sizeof(ModInstrument().nFadeOut),    f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('R','.','.','.'), sizeof(ModInstrument().nResampling), f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('C','S','.','.'), sizeof(ModInstrument().nCutSwing),   f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('R','S','.','.'), sizeof(ModInstrument().nResSwing),   f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('F','M','.','.'), sizeof(ModInstrument().nFilterMode), f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('P','E','R','N'), sizeof(ModInstrument().PitchEnv.nReleaseNode ), f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('A','E','R','N'), sizeof(ModInstrument().PanEnv.nReleaseNode), f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('V','E','R','N'), sizeof(ModInstrument().VolEnv.nReleaseNode), f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('P','T','T','L'), sizeof(uint16),  f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4LE('P','T','T','F'), sizeof(uint16),  f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('P','V','E','H'), sizeof(ModInstrument().nPluginVelocityHandling),  f, nInstruments);
	WriteInstrumentPropertyForAllInstruments(MAGIC4BE('P','V','O','H'), sizeof(ModInstrument().nPluginVolumeHandling),  f, nInstruments);

	if(!(GetType() & MOD_TYPE_XM))
	{
		// XM instrument headers already have support for this
		WriteInstrumentPropertyForAllInstruments(MAGIC4BE('M','P','W','D'), sizeof(ModInstrument().midiPWD), f, nInstruments);
	}

	if(GetType() & MOD_TYPE_MPT)
	{
		uint32 maxNodes[3] = { 0 };
		for(INSTRUMENTINDEX i = 1; i <= m_nInstruments; i++) if(Instruments[i] != nullptr)
		{
			maxNodes[0] = std::max(maxNodes[0], Instruments[i]->VolEnv.size());
			maxNodes[1] = std::max(maxNodes[1], Instruments[i]->PanEnv.size());
			maxNodes[2] = std::max(maxNodes[2], Instruments[i]->PitchEnv.size());
		}
		// write full envelope information for MPTM files (more env points)
		if(maxNodes[0] > 25)
		{
			WriteInstrumentPropertyForAllInstruments(MAGIC4BE('V','E','.','.'), sizeof(ModInstrument().VolEnv.size()), f, nInstruments);
			WriteInstrumentPropertyForAllInstruments(MAGIC4BE('V','P','[','.'), static_cast<uint16>(maxNodes[0] * sizeof(EnvelopeNode().tick)),  f, nInstruments);
			WriteInstrumentPropertyForAllInstruments(MAGIC4BE('V','E','[','.'), static_cast<uint16>(maxNodes[0] * sizeof(EnvelopeNode().value)), f, nInstruments);
		}
		if(maxNodes[1] > 25)
		{
			WriteInstrumentPropertyForAllInstruments(MAGIC4BE('P','E','.','.'), sizeof(ModInstrument().PanEnv.size()), f, nInstruments);
			WriteInstrumentPropertyForAllInstruments(MAGIC4BE('P','P','[','.'), static_cast<uint16>(maxNodes[1] * sizeof(EnvelopeNode().tick)),  f, nInstruments);
			WriteInstrumentPropertyForAllInstruments(MAGIC4BE('P','E','[','.'), static_cast<uint16>(maxNodes[1] * sizeof(EnvelopeNode().value)), f, nInstruments);
		}
		if(maxNodes[2] > 25)
		{
			WriteInstrumentPropertyForAllInstruments(MAGIC4BE('P','i','E','.'), sizeof(ModInstrument().PitchEnv.size()), f, nInstruments);
			WriteInstrumentPropertyForAllInstruments(MAGIC4BE('P','i','P','['), static_cast<uint16>(maxNodes[2] * sizeof(EnvelopeNode().tick)),  f, nInstruments);
			WriteInstrumentPropertyForAllInstruments(MAGIC4BE('P','i','E','['), static_cast<uint16>(maxNodes[2] * sizeof(EnvelopeNode().value)), f, nInstruments);
		}
	}
}

void CSoundFile::WriteInstrumentPropertyForAllInstruments(uint32 code, uint16 size, FILE *f, INSTRUMENTINDEX nInstruments) const
{
	mpt::IO::WriteIntLE<uint32>(f, code);		//write code
	mpt::IO::WriteIntLE<uint16>(f, size);		//write size
	for(INSTRUMENTINDEX i = 1; i <= nInstruments; i++)	//for all instruments...
	{
		if (Instruments[i])
		{
			WriteInstrumentHeaderStructOrField(Instruments[i], f, code, size);
		} else
		{
			ModInstrument emptyInstrument;
			WriteInstrumentHeaderStructOrField(&emptyInstrument, f, code, size);
		}
	}
}


#endif // !MODPLUG_NO_FILESAVE


// --------------------------------------------------------------------------------------------
// Convenient macro to help GET_HEADER declaration for single type members ONLY (non-array)
// --------------------------------------------------------------------------------------------
#define GET_MPTHEADER_sized_member(name,type,code) \
	case code: \
	{\
		if( fsize <= sizeof( type ) ) \
		{ \
			/* hackish workaround to resolve mismatched size values: */ \
			/* nResampling was a long time declared as uint32 but these macro tables used uint16 and UINT. */ \
			/* This worked fine on little-endian, on big-endian not so much. Thus support reading size-mismatched fields. */ \
			if(file.CanRead(fsize)) \
			{ \
				type tmp; \
				tmp = file.ReadTruncatedIntLE<type>(fsize); \
				STATIC_ASSERT(sizeof(tmp) == sizeof(input-> name )); \
				memcpy(&(input-> name ), &tmp, sizeof(type)); \
				result = true; \
			} \
		} \
	} break;

// --------------------------------------------------------------------------------------------
// Convenient macro to help GET_HEADER declaration for array members ONLY
// --------------------------------------------------------------------------------------------
#define GET_MPTHEADER_array_member(name,type,code) \
	case code: \
	{\
		if( fsize <= sizeof( type ) * CountOf(input-> name) ) \
		{ \
			FileReader arrayChunk = file.ReadChunk(fsize); \
			for(std::size_t i = 0; i < CountOf(input-> name); ++i) \
			{ \
				input-> name [i] = arrayChunk.ReadIntLE<type>(); \
			} \
			result = true; \
		} \
	} break;

// --------------------------------------------------------------------------------------------
// Convenient macro to help GET_HEADER declaration for envelope tick/value members
// --------------------------------------------------------------------------------------------
#define GET_MPTHEADER_envelope_member(envType,envField,type,code) \
	case code: \
	{\
		FileReader arrayChunk = file.ReadChunk(fsize); \
		InstrumentEnvelope &env = input->GetEnvelope(envType); \
		for(uint32 i = 0; i < env.size(); i++) \
		{ \
			env[i]. envField = arrayChunk.ReadIntLE<type>(); \
		} \
		result = true; \
	} break;


// Return a pointer on the wanted field in 'input' ModInstrument given field code & size
bool ReadInstrumentHeaderField(ModInstrument *input, uint32 fcode, uint16 fsize, FileReader &file)
{
	if(input == nullptr) return false;

	bool result = false;

	// Members which can be found in this table but not in the write table are only required in the legacy ITP format.
	switch(fcode)
	{
	GET_MPTHEADER_sized_member(	nFadeOut				, uint32		, MAGIC4BE('F','O','.','.')	)
	GET_MPTHEADER_sized_member(	dwFlags					, uint32		, MAGIC4BE('d','F','.','.')	)
	GET_MPTHEADER_sized_member(	nGlobalVol				, uint32		, MAGIC4BE('G','V','.','.')	)
	GET_MPTHEADER_sized_member(	nPan					, uint32		, MAGIC4BE('P','.','.','.')	)
	GET_MPTHEADER_sized_member(	VolEnv.nLoopStart		, uint8			, MAGIC4BE('V','L','S','.')	)
	GET_MPTHEADER_sized_member(	VolEnv.nLoopEnd			, uint8			, MAGIC4BE('V','L','E','.')	)
	GET_MPTHEADER_sized_member(	VolEnv.nSustainStart	, uint8			, MAGIC4BE('V','S','B','.')	)
	GET_MPTHEADER_sized_member(	VolEnv.nSustainEnd		, uint8			, MAGIC4BE('V','S','E','.')	)
	GET_MPTHEADER_sized_member(	PanEnv.nLoopStart		, uint8			, MAGIC4BE('P','L','S','.')	)
	GET_MPTHEADER_sized_member(	PanEnv.nLoopEnd			, uint8			, MAGIC4BE('P','L','E','.')	)
	GET_MPTHEADER_sized_member(	PanEnv.nSustainStart	, uint8			, MAGIC4BE('P','S','B','.')	)
	GET_MPTHEADER_sized_member(	PanEnv.nSustainEnd		, uint8			, MAGIC4BE('P','S','E','.')	)
	GET_MPTHEADER_sized_member(	PitchEnv.nLoopStart		, uint8			, MAGIC4BE('P','i','L','S')	)
	GET_MPTHEADER_sized_member(	PitchEnv.nLoopEnd		, uint8			, MAGIC4BE('P','i','L','E')	)
	GET_MPTHEADER_sized_member(	PitchEnv.nSustainStart	, uint8			, MAGIC4BE('P','i','S','B')	)
	GET_MPTHEADER_sized_member(	PitchEnv.nSustainEnd	, uint8			, MAGIC4BE('P','i','S','E')	)
	GET_MPTHEADER_sized_member(	nNNA					, uint8			, MAGIC4BE('N','N','A','.')	)
	GET_MPTHEADER_sized_member(	nDCT					, uint8			, MAGIC4BE('D','C','T','.')	)
	GET_MPTHEADER_sized_member(	nDNA					, uint8			, MAGIC4BE('D','N','A','.')	)
	GET_MPTHEADER_sized_member(	nPanSwing				, uint8			, MAGIC4BE('P','S','.','.')	)
	GET_MPTHEADER_sized_member(	nVolSwing				, uint8			, MAGIC4BE('V','S','.','.')	)
	GET_MPTHEADER_sized_member(	nIFC					, uint8			, MAGIC4BE('I','F','C','.')	)
	GET_MPTHEADER_sized_member(	nIFR					, uint8			, MAGIC4BE('I','F','R','.')	)
	GET_MPTHEADER_sized_member(	wMidiBank				, uint16		, MAGIC4BE('M','B','.','.')	)
	GET_MPTHEADER_sized_member(	nMidiProgram			, uint8			, MAGIC4BE('M','P','.','.')	)
	GET_MPTHEADER_sized_member(	nMidiChannel			, uint8			, MAGIC4BE('M','C','.','.')	)
	GET_MPTHEADER_sized_member(	nPPS					, int8			, MAGIC4BE('P','P','S','.')	)
	GET_MPTHEADER_sized_member(	nPPC					, uint8			, MAGIC4BE('P','P','C','.')	)
	GET_MPTHEADER_envelope_member(ENV_VOLUME	, tick	, uint16		, MAGIC4BE('V','P','[','.')	)
	GET_MPTHEADER_envelope_member(ENV_PANNING	, tick	, uint16		, MAGIC4BE('P','P','[','.')	)
	GET_MPTHEADER_envelope_member(ENV_PITCH		, tick	, uint16		, MAGIC4BE('P','i','P','[')	)
	GET_MPTHEADER_envelope_member(ENV_VOLUME	, value	, uint8			, MAGIC4BE('V','E','[','.')	)
	GET_MPTHEADER_envelope_member(ENV_PANNING	, value	, uint8			, MAGIC4BE('P','E','[','.')	)
	GET_MPTHEADER_envelope_member(ENV_PITCH		, value	, uint8			, MAGIC4BE('P','i','E','[')	)
	GET_MPTHEADER_array_member(	NoteMap					, uint8			, MAGIC4BE('N','M','[','.')	)
	GET_MPTHEADER_array_member(	Keyboard				, uint16		, MAGIC4BE('K','[','.','.')	)
	GET_MPTHEADER_array_member(	name					, char			, MAGIC4BE('n','[','.','.')	)
	GET_MPTHEADER_array_member(	filename				, char			, MAGIC4BE('f','n','[','.')	)
	GET_MPTHEADER_sized_member(	nMixPlug				, uint8			, MAGIC4BE('M','i','P','.')	)
	GET_MPTHEADER_sized_member(	nVolRampUp				, uint16		, MAGIC4BE('V','R','.','.')	)
	GET_MPTHEADER_sized_member(	nResampling				, uint32		, MAGIC4BE('R','.','.','.')	)
	GET_MPTHEADER_sized_member(	nCutSwing				, uint8			, MAGIC4BE('C','S','.','.')	)
	GET_MPTHEADER_sized_member(	nResSwing				, uint8			, MAGIC4BE('R','S','.','.')	)
	GET_MPTHEADER_sized_member(	nFilterMode				, uint8			, MAGIC4BE('F','M','.','.')	)
	GET_MPTHEADER_sized_member(	nPluginVelocityHandling	, uint8			, MAGIC4BE('P','V','E','H')	)
	GET_MPTHEADER_sized_member(	nPluginVolumeHandling	, uint8			, MAGIC4BE('P','V','O','H')	)
	GET_MPTHEADER_sized_member(	PitchEnv.nReleaseNode	, uint8			, MAGIC4BE('P','E','R','N')	)
	GET_MPTHEADER_sized_member(	PanEnv.nReleaseNode		, uint8			, MAGIC4BE('A','E','R','N')	)
	GET_MPTHEADER_sized_member(	VolEnv.nReleaseNode		, uint8			, MAGIC4BE('V','E','R','N')	)
	GET_MPTHEADER_sized_member(	PitchEnv.dwFlags		, uint32		, MAGIC4BE('P','F','L','G')	)
	GET_MPTHEADER_sized_member(	PanEnv.dwFlags			, uint32		, MAGIC4BE('A','F','L','G')	)
	GET_MPTHEADER_sized_member(	VolEnv.dwFlags			, uint32		, MAGIC4BE('V','F','L','G')	)
	GET_MPTHEADER_sized_member(	midiPWD					, int8			, MAGIC4BE('M','P','W','D')	)
	case MAGIC4BE('P','T','T','L'):
	{
		// Integer part of pitch/tempo lock
		uint16 tmp = file.ReadTruncatedIntLE<uint16>(fsize);
		input->pitchToTempoLock.Set(tmp, input->pitchToTempoLock.GetFract());
		result = true;
	} break;
	case MAGIC4LE('P','T','T','F'):
	{
		// Fractional part of pitch/tempo lock
		uint16 tmp = file.ReadTruncatedIntLE<uint16>(fsize);
		input->pitchToTempoLock.Set(input->pitchToTempoLock.GetInt(), tmp);
		result = true;
	} break;
	case MAGIC4BE('V','E','.','.'):
		input->VolEnv.resize(std::min<uint32>(MAX_ENVPOINTS, file.ReadTruncatedIntLE<uint32>(fsize)));
		result = true;
		break;
	case MAGIC4BE('P','E','.','.'):
		input->PanEnv.resize(std::min<uint32>(MAX_ENVPOINTS, file.ReadTruncatedIntLE<uint32>(fsize)));
		result = true;
		break;
	case MAGIC4BE('P','i','E','.'):
		input->PitchEnv.resize(std::min<uint32>(MAX_ENVPOINTS, file.ReadTruncatedIntLE<uint32>(fsize)));
		result = true;
		break;
	}

	return result;
}


// Convert instrument flags which were read from 'dF..' extension to proper internal representation.
static void ConvertReadExtendedFlags(ModInstrument *pIns)
{
	// Flags of 'dF..' datafield in extended instrument properties.
	enum
	{
		dFdd_VOLUME 		= 0x0001,
		dFdd_VOLSUSTAIN 	= 0x0002,
		dFdd_VOLLOOP 		= 0x0004,
		dFdd_PANNING 		= 0x0008,
		dFdd_PANSUSTAIN 	= 0x0010,
		dFdd_PANLOOP 		= 0x0020,
		dFdd_PITCH 			= 0x0040,
		dFdd_PITCHSUSTAIN 	= 0x0080,
		dFdd_PITCHLOOP 		= 0x0100,
		dFdd_SETPANNING 	= 0x0200,
		dFdd_FILTER 		= 0x0400,
		dFdd_VOLCARRY 		= 0x0800,
		dFdd_PANCARRY 		= 0x1000,
		dFdd_PITCHCARRY 	= 0x2000,
		dFdd_MUTE 			= 0x4000,
	};

	const uint32 dwOldFlags = pIns->dwFlags.GetRaw();

	pIns->VolEnv.dwFlags.set(ENV_ENABLED, (dwOldFlags & dFdd_VOLUME) != 0);
	pIns->VolEnv.dwFlags.set(ENV_SUSTAIN, (dwOldFlags & dFdd_VOLSUSTAIN) != 0);
	pIns->VolEnv.dwFlags.set(ENV_LOOP, (dwOldFlags & dFdd_VOLLOOP) != 0);
	pIns->VolEnv.dwFlags.set(ENV_CARRY, (dwOldFlags & dFdd_VOLCARRY) != 0);

	pIns->PanEnv.dwFlags.set(ENV_ENABLED, (dwOldFlags & dFdd_PANNING) != 0);
	pIns->PanEnv.dwFlags.set(ENV_SUSTAIN, (dwOldFlags & dFdd_PANSUSTAIN) != 0);
	pIns->PanEnv.dwFlags.set(ENV_LOOP, (dwOldFlags & dFdd_PANLOOP) != 0);
	pIns->PanEnv.dwFlags.set(ENV_CARRY, (dwOldFlags & dFdd_PANCARRY) != 0);

	pIns->PitchEnv.dwFlags.set(ENV_ENABLED, (dwOldFlags & dFdd_PITCH) != 0);
	pIns->PitchEnv.dwFlags.set(ENV_SUSTAIN, (dwOldFlags & dFdd_PITCHSUSTAIN) != 0);
	pIns->PitchEnv.dwFlags.set(ENV_LOOP, (dwOldFlags & dFdd_PITCHLOOP) != 0);
	pIns->PitchEnv.dwFlags.set(ENV_CARRY, (dwOldFlags & dFdd_PITCHCARRY) != 0);
	pIns->PitchEnv.dwFlags.set(ENV_FILTER, (dwOldFlags & dFdd_FILTER) != 0);

	pIns->dwFlags.reset();
	pIns->dwFlags.set(INS_SETPANNING, (dwOldFlags & dFdd_SETPANNING) != 0);
	pIns->dwFlags.set(INS_MUTE, (dwOldFlags & dFdd_MUTE) != 0);
}


void ReadInstrumentExtensionField(ModInstrument* pIns, const uint32 code, const uint16 size, FileReader &file)
{
	if(code == MAGIC4BE('K','[','.','.'))
	{
		// skip keyboard mapping
		file.Skip(size);
		return;
	}

	bool success = ReadInstrumentHeaderField(pIns, code, size, file);

	if(!success)
	{
		file.Skip(size);
		return;
	}

	if(code == MAGIC4BE('n','[','.','.'))
		mpt::String::SetNullTerminator(pIns->name);
	if(code == MAGIC4BE('f','n','[','.'))
		mpt::String::SetNullTerminator(pIns->filename);

	if(code == MAGIC4BE('d','F','.','.')) // 'dF..' field requires additional processing.
		ConvertReadExtendedFlags(pIns);
}


void ReadExtendedInstrumentProperty(ModInstrument* pIns, const uint32 code, FileReader &file)
{
	uint16 size = file.ReadUint16LE();
	if(!file.CanRead(size))
	{
		return;
	}
	ReadInstrumentExtensionField(pIns, code, size, file);
}


void ReadExtendedInstrumentProperties(ModInstrument* pIns, FileReader &file)
{
	if(!file.ReadMagic("XTPM"))	// 'MPTX'
	{
		return;
	}

	while(file.CanRead(7))
	{
		ReadExtendedInstrumentProperty(pIns, file.ReadUint32LE(), file);
	}
}


void CSoundFile::LoadExtendedInstrumentProperties(FileReader &file, bool *pInterpretMptMade)
{
	if(!file.ReadMagic("XTPM"))	// 'MPTX'
	{
		return;
	}

	// Found MPTX, interpret the file MPT made.
	if(pInterpretMptMade != nullptr)
		*pInterpretMptMade = true;

	while(file.CanRead(6))
	{
		uint32 code = file.ReadUint32LE();

		if(code == MAGIC4BE('M','P','T','S')	// Reached song extensions, break out of this loop
			|| code == MAGIC4LE('2','2','8',4)	// Reached MPTM extensions (in case there are no song extensions)
			|| (code & 0x80808080) || !(code & 0x60606060))	// Non-ASCII chunk ID
		{
			file.SkipBack(4);
			return;
		}

		// Read size of this property for *one* instrument
		const uint16 size = file.ReadUint16LE();

		for(INSTRUMENTINDEX i = 1; i <= GetNumInstruments(); i++)
		{
			if(Instruments[i])
			{
				ReadInstrumentExtensionField(Instruments[i], code, size, file);
			}
		}
	}
}


OPENMPT_NAMESPACE_END
