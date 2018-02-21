/*
 * ITTools.h
 * ---------
 * Purpose: Definition of IT file structures and helper functions
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "../soundlib/ModInstrument.h"
#include "../soundlib/ModSample.h"
#include "../soundlib/SampleIO.h"

OPENMPT_NAMESPACE_BEGIN

struct ITFileHeader
{
	// Header Flags
	enum ITHeaderFlags
	{
		useStereoPlayback		= 0x01,
		vol0Optimisations		= 0x02,
		instrumentMode			= 0x04,
		linearSlides			= 0x08,
		itOldEffects			= 0x10,
		itCompatGxx				= 0x20,
		useMIDIPitchController	= 0x40,
		reqEmbeddedMIDIConfig	= 0x80,
		extendedFilterRange		= 0x1000,
	};

	// Special Flags
	enum ITHeaderSpecialFlags
	{
		embedSongMessage		= 0x01,
		embedEditHistory		= 0x02,
		embedPatternHighlights	= 0x04,
		embedMIDIConfiguration	= 0x08,
	};

	char     id[4];				// Magic Bytes (IMPM)
	char     songname[26];		// Song Name, null-terminated (but may also contain nulls)
	uint8le  highlight_minor;	// Rows per Beat highlight
	uint8le  highlight_major;	// Rows per Measure highlight
	uint16le ordnum;			// Number of Orders
	uint16le insnum;			// Number of Instruments
	uint16le smpnum;			// Number of Samples
	uint16le patnum;			// Number of Patterns
	uint16le cwtv;				// "Made With" Tracker
	uint16le cmwt;				// "Compatible With" Tracker
	uint16le flags;				// Header Flags
	uint16le special;			// Special Flags, for embedding extra information
	uint8le  globalvol;			// Global Volume (0...128)
	uint8le  mv;				// Master Volume (0...128), referred to as Sample Volume in OpenMPT
	uint8le  speed;				// Initial Speed (1...255)
	uint8le  tempo;				// Initial Tempo (31...255)
	uint8le  sep;				// Pan Separation (0...128)
	uint8le  pwd;				// Pitch Wheel Depth
	uint16le msglength;			// Length of Song Message
	uint32le msgoffset;			// Offset of Song Message in File (IT crops message after first null)
	uint32le reserved;			// Some IT versions save an edit timer here. ChibiTracker writes "CHBI" here. OpenMPT writes "OMPT" here in some cases, see Load_it.cpp
	uint8le  chnpan[64];		// Initial Channel Panning
	uint8le  chnvol[64];		// Initial Channel Volume
};

MPT_BINARY_STRUCT(ITFileHeader, 192)


struct ITEnvelope
{
	// Envelope Flags
	enum ITEnvelopeFlags
	{
		envEnabled	= 0x01,
		envLoop		= 0x02,
		envSustain	= 0x04,
		envCarry	= 0x08,
		envFilter	= 0x80,
	};

	struct Node
	{
		int8le   value;
		uint16le tick;
	};

	uint8 flags;	// Envelope Flags
	uint8 num;		// Number of Envelope Nodes
	uint8 lpb;		// Loop Start
	uint8 lpe;		// Loop End
	uint8 slb;		// Sustain Start
	uint8 sle;		// Sustain End
	Node  data[25];	// Envelope Node Positions / Values
	uint8 reserved;	// Reserved

	// Convert OpenMPT's internal envelope format to an IT/MPTM envelope.
	void ConvertToIT(const InstrumentEnvelope &mptEnv, uint8 envOffset, uint8 envDefault);
	// Convert IT/MPTM envelope data into OpenMPT's internal envelope format - To be used by ITInstrToMPT()
	void ConvertToMPT(InstrumentEnvelope &mptEnv, uint8 envOffset, uint8 maxNodes) const;
};

MPT_BINARY_STRUCT(ITEnvelope::Node, 3)
MPT_BINARY_STRUCT(ITEnvelope, 82)


// Old Impulse Instrument Format (cmwt < 0x200)
struct ITOldInstrument
{
	enum ITOldInstrFlags
	{
		envEnabled	= 0x01,
		envLoop		= 0x02,
		envSustain	= 0x04,
	};

	char     id[4];			// Magic Bytes (IMPI)
	char     filename[13];	// DOS Filename, null-terminated
	uint8le  flags;			// Volume Envelope Flags
	uint8le  vls;			// Envelope Loop Start
	uint8le  vle;			// Envelope Loop End
	uint8le  sls;			// Envelope Sustain Start
	uint8le  sle;			// Envelope Sustain End
	char     reserved1[2];	// Reserved
	uint16le fadeout;		// Instrument Fadeout (0...128)
	uint8le  nna;			// New Note Action
	uint8le  dnc;			// Duplicate Note Check Type
	uint16le trkvers;		// Tracker ID
	uint8le  nos;			// Number of embedded samples
	char     reserved2;		// Reserved
	char     name[26];		// Instrument Name, null-terminated (but may also contain nulls)
	char     reserved3[6];	// Even more reserved bytes
	uint8le  keyboard[240];	// Sample / Transpose map
	uint8le  volenv[200];	// This appears to be a pre-computed (interpolated) version of the volume envelope data found below.
	uint8le  nodes[25 * 2];	// Volume Envelope Node Positions / Values

	// Convert an ITOldInstrument to OpenMPT's internal instrument representation.
	void ConvertToMPT(ModInstrument &mptIns) const;
};

MPT_BINARY_STRUCT(ITOldInstrument, 554)


// Impulse Instrument Format
struct ITInstrument
{
	enum ITInstrumentFlags
	{
		ignorePanning	= 0x80,
		enableCutoff	= 0x80,
		enableResonance	= 0x80,
	};

	char     id[4];			// Magic Bytes (IMPI)
	char     filename[13];	// DOS Filename, null-terminated
	uint8le  nna;			// New Note Action
	uint8le  dct;			// Duplicate Note Check Type
	uint8le  dca;			// Duplicate Note Check Action
	uint16le fadeout;		// Instrument Fadeout (0...256, although values up to 1024 would be sensible. Up to IT2.07, the limit was 0...128)
	int8le   pps;			// Pitch/Pan Separatation
	uint8le  ppc;			// Pitch/Pan Centre
	uint8le  gbv;			// Global Volume
	uint8le  dfp;			// Panning
	uint8le  rv;			// Vol Swing
	uint8le  rp;			// Pan Swing
	uint16le trkvers;		// Tracker ID
	uint8le  nos;			// Number of embedded samples
	char     reserved1;		// Reserved
	char     name[26];		// Instrument Name, null-terminated (but may also contain nulls)
	uint8le  ifc;			// Filter Cutoff
	uint8le  ifr;			// Filter Resonance
	uint8le  mch;			// MIDI Channel
	uint8le  mpr;			// MIDI Program
	uint8le  mbank[2];		// MIDI Bank
	uint8le  keyboard[240];	// Sample / Transpose map
	ITEnvelope volenv;		// Volume Envelope
	ITEnvelope panenv;		// Pan Envelope
	ITEnvelope pitchenv;	// Pitch / Filter Envelope
	char       dummy[4];	// IT saves some additional padding bytes to match the size of the old instrument format for simplified loading. We use them for some hacks.

	// Convert OpenMPT's internal instrument representation to an ITInstrument. Returns amount of bytes that need to be written.
	uint32 ConvertToIT(const ModInstrument &mptIns, bool compatExport, const CSoundFile &sndFile);
	// Convert an ITInstrument to OpenMPT's internal instrument representation. Returns size of the instrument data that has been read.
	uint32 ConvertToMPT(ModInstrument &mptIns, MODTYPE fromType) const;
};

MPT_BINARY_STRUCT(ITInstrument, 554)


// MPT IT Instrument Extension
struct ITInstrumentEx
{
	ITInstrument iti;		// Normal IT Instrument
	uint8 keyboardhi[120];	// High Byte of Sample map
	
	// Convert OpenMPT's internal instrument representation to an ITInstrumentEx. Returns amount of bytes that need to be written.
	uint32 ConvertToIT(const ModInstrument &mptIns, bool compatExport, const CSoundFile &sndFile);
	// Convert an ITInstrumentEx to OpenMPT's internal instrument representation. Returns size of the instrument data that has been read.
	uint32 ConvertToMPT(ModInstrument &mptIns, MODTYPE fromType) const;
};

MPT_BINARY_STRUCT(ITInstrumentEx, sizeof(ITInstrument) + 120)


// IT Sample Format
struct ITSample
{
	// Magic Bytes
	enum Magic
	{
		magic = 0x53504D49,	// "IMPS" IT Sample Header Magic Bytes
	};

	enum ITSampleFlags
	{
		sampleDataPresent	= 0x01,
		sample16Bit			= 0x02,
		sampleStereo		= 0x04,
		sampleCompressed	= 0x08,
		sampleLoop			= 0x10,
		sampleSustain		= 0x20,
		sampleBidiLoop		= 0x40,
		sampleBidiSustain	= 0x80,

		enablePanning		= 0x80,

		cvtSignedSample		= 0x01,
		cvtExternalSample	= 0x80,		// Keep MPTM sample on disk
		cvtADPCMSample		= 0xFF,		// MODPlugin :(

		// ITTECH.TXT says these convert flags are "safe to ignore". IT doesn't ignore them, though, so why should we? :)
		cvtBigEndian		= 0x02,
		cvtDelta			= 0x04,
		cvtPTM8to16			= 0x08,
	};

	char     id[4];			// Magic Bytes (IMPS)
	char     filename[13];	// DOS Filename, null-terminated
	uint8le  gvl;			// Global Volume
	uint8le  flags;			// Sample Flags
	uint8le  vol;			// Default Volume
	char     name[26];		// Sample Name, null-terminated (but may also contain nulls)
	uint8le  cvt;			// Sample Import Format
	uint8le  dfp;			// Sample Panning
	uint32le length;		// Sample Length (in samples)
	uint32le loopbegin;		// Sample Loop Begin (in samples)
	uint32le loopend;		// Sample Loop End (in samples)
	uint32le C5Speed;		// C-5 frequency
	uint32le susloopbegin;	// Sample Sustain Begin (in samples)
	uint32le susloopend;	// Sample Sustain End (in samples)
	uint32le samplepointer;	// Pointer to sample data
	uint8le  vis;			// Auto-Vibrato Rate (called Sweep in IT)
	uint8le  vid;			// Auto-Vibrato Depth
	uint8le  vir;			// Auto-Vibrato Sweep (called Rate in IT)
	uint8le  vit;			// Auto-Vibrato Type

	// Convert OpenMPT's internal sample representation to an ITSample.
	void ConvertToIT(const ModSample &mptSmp, MODTYPE fromType, bool compress, bool compressIT215, bool allowExternal);
	// Convert an ITSample to OpenMPT's internal sample representation.
	uint32 ConvertToMPT(ModSample &mptSmp) const;
	// Retrieve the internal sample format flags for this instrument.
	SampleIO GetSampleFormat(uint16 cwtv = 0x214) const;
};

MPT_BINARY_STRUCT(ITSample, 80)


struct FileHistory;

// IT Header extension: Save history
struct ITHistoryStruct
{
	uint16le fatdate;	// DOS / FAT date when the file was opened / created in the editor. For details, read http://msdn.microsoft.com/en-us/library/ms724247(VS.85).aspx
	uint16le fattime;	// DOS / FAT time when the file was opened / created in the editor.
	uint32le runtime;	// The time how long the file was open in the editor, in 1/18.2th seconds. (= ticks of the DOS timer)

	// Convert an ITHistoryStruct to OpenMPT's internal edit history representation
	void ConvertToMPT(FileHistory &mptHistory) const;
	// Convert OpenMPT's internal edit history representation to an ITHistoryStruct
	void ConvertToIT(const FileHistory &mptHistory);

};

MPT_BINARY_STRUCT(ITHistoryStruct, 8)


enum IT_ReaderBitMasks
{
	// pattern row parsing, the channel data is read to obtain
	// number of channels active in the pattern. These bit masks are
	// to blank out sections of the byte of data being read.

	IT_bitmask_patternChanField_c   = 0x7f,
	IT_bitmask_patternChanMask_c    = 0x3f,
	IT_bitmask_patternChanEnabled_c = 0x80,
	IT_bitmask_patternChanUsed_c    = 0x0f
};

OPENMPT_NAMESPACE_END
