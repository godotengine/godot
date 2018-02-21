/*
 * XMTools.h
 * ---------
 * Purpose: Definition of XM file structures and helper functions
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


OPENMPT_NAMESPACE_BEGIN


// XM File Header
struct XMFileHeader
{
	enum XMHeaderFlags
	{
		linearSlides			= 0x01,
		extendedFilterRange		= 0x1000,
	};

	char     signature[17];		// "Extended Module: "
	char     songName[20];		// Song Name, not null-terminated (any nulls are treated as spaces)
	uint8le  eof;				// DOS EOF Character (0x1A)
	char     trackerName[20];	// Software that was used to create the XM file
	uint16le version;			// File version (1.02 - 1.04 are supported)
	uint32le size;				// Header Size
	uint16le orders;			// Number of Orders
	uint16le restartPos;		// Restart Position
	uint16le channels;			// Number of Channels
	uint16le patterns;			// Number of Patterns
	uint16le instruments;		// Number of Unstruments
	uint16le flags;				// Song Flags
	uint16le speed;				// Default Speed
	uint16le tempo;				// Default Tempo
};

MPT_BINARY_STRUCT(XMFileHeader, 80)


// XM Instrument Data
struct XMInstrument
{
	// Envelope Flags
	enum XMEnvelopeFlags
	{
		envEnabled	= 0x01,
		envSustain	= 0x02,
		envLoop		= 0x04,
	};

	uint8le  sampleMap[96];		// Note -> Sample assignment
	uint16le volEnv[24];		// Volume envelope nodes / values (0...64)
	uint16le panEnv[24];		// Panning envelope nodes / values (0...63)
	uint8le  volPoints;			// Volume envelope length
	uint8le  panPoints;			// Panning envelope length
	uint8le  volSustain;		// Volume envelope sustain point
	uint8le  volLoopStart;		// Volume envelope loop start point
	uint8le  volLoopEnd;		// Volume envelope loop end point
	uint8le  panSustain;		// Panning envelope sustain point
	uint8le  panLoopStart;		// Panning envelope loop start point
	uint8le  panLoopEnd;		// Panning envelope loop end point
	uint8le  volFlags;			// Volume envelope flags
	uint8le  panFlags;			// Panning envelope flags
	uint8le  vibType;			// Sample Auto-Vibrato Type
	uint8le  vibSweep;			// Sample Auto-Vibrato Sweep
	uint8le  vibDepth;			// Sample Auto-Vibrato Depth
	uint8le  vibRate;			// Sample Auto-Vibrato Rate
	uint16le volFade;			// Volume Fade-Out
	uint8le  midiEnabled;		// MIDI Out Enabled (0 / 1)
	uint8le  midiChannel;		// MIDI Channel (0...15)
	uint16le midiProgram;		// MIDI Program (0...127)
	uint16le pitchWheelRange;	// MIDI Pitch Wheel Range (0...36 halftones)
	uint8le  muteComputer;		// Mute instrument if MIDI is enabled (0 / 1)
	uint8le  reserved[15];		// Reserved

	enum EnvType
	{
		EnvTypeVol,
		EnvTypePan,
	};
	// Convert OpenMPT's internal envelope representation to XM envelope data.
	void ConvertEnvelopeToXM(const InstrumentEnvelope &mptEnv, uint8le &numPoints, uint8le &flags, uint8le &sustain, uint8le &loopStart, uint8le &loopEnd, EnvType env);
	// Convert XM envelope data to an OpenMPT's internal envelope representation.
	void ConvertEnvelopeToMPT(InstrumentEnvelope &mptEnv, uint8 numPoints, uint8 flags, uint8 sustain, uint8 loopStart, uint8 loopEnd, EnvType env) const;

	// Convert OpenMPT's internal sample representation to an XMInstrument.
	uint16 ConvertToXM(const ModInstrument &mptIns, bool compatibilityExport);
	// Convert an XMInstrument to OpenMPT's internal instrument representation.
	void ConvertToMPT(ModInstrument &mptIns) const;
	// Apply auto-vibrato settings from sample to file.
	void ApplyAutoVibratoToXM(const ModSample &mptSmp, MODTYPE fromType);
	// Apply auto-vibrato settings from file to a sample.
	void ApplyAutoVibratoToMPT(ModSample &mptSmp) const;

	// Get a list of samples that should be written to the file.
	std::vector<SAMPLEINDEX> GetSampleList(const ModInstrument &mptIns, bool compatibilityExport) const;
};

MPT_BINARY_STRUCT(XMInstrument, 230)


// XM Instrument Header
struct XMInstrumentHeader
{
	uint32le size;				// Size of XMInstrumentHeader + XMInstrument
	char     name[22];			// Instrument Name, not null-terminated (any nulls are treated as spaces)
	uint8le  type;				// Instrument Type (Apparently FT2 writes some crap here, but it's the same crap for all instruments of the same module!)
	uint16le numSamples;		// Number of Samples associated with instrument
	uint32le sampleHeaderSize;	// Size of XMSample
	XMInstrument instrument;

	// Write stuff to the header that's always necessary (also for empty instruments)
	void Finalise();

	// Convert OpenMPT's internal sample representation to an XMInstrument.
	void ConvertToXM(const ModInstrument &mptIns, bool compatibilityExport);
	// Convert an XMInstrument to OpenMPT's internal instrument representation.
	void ConvertToMPT(ModInstrument &mptIns) const;
};

MPT_BINARY_STRUCT(XMInstrumentHeader, 263)


// XI Instrument Header
struct XIInstrumentHeader
{
	enum
	{
		fileVersion	= 0x102,
	};

	char     signature[21];		// "Extended Instrument: "
	char     name[22];			// Instrument Name, not null-terminated (any nulls are treated as spaces)
	uint8le  eof;				// DOS EOF Character (0x1A)
	char     trackerName[20];	// Software that was used to create the XI file
	uint16le version;			// File Version (1.02)
	XMInstrument instrument;
	uint16le numSamples;		// Number of embedded sample headers + samples

	// Convert OpenMPT's internal sample representation to an XIInstrumentHeader.
	void ConvertToXM(const ModInstrument &mptIns, bool compatibilityExport);
	// Convert an XIInstrumentHeader to OpenMPT's internal instrument representation.
	void ConvertToMPT(ModInstrument &mptIns) const;
};

MPT_BINARY_STRUCT(XIInstrumentHeader, 298)


// XM Sample Header
struct XMSample
{
	enum XMSampleFlags
	{
		sampleLoop			= 0x01,
		sampleBidiLoop		= 0x02,
		sample16Bit			= 0x10,
		sampleStereo		= 0x20,

		sampleADPCM			= 0xAD,		// MODPlugin :(
	};

	uint32le length;		// Sample Length (in bytes)
	uint32le loopStart;		// Loop Start (in bytes)
	uint32le loopLength;	// Loop Length (in bytes)
	uint8le  vol;			// Default Volume
	int8le   finetune;		// Sample Finetune
	uint8le  flags;			// Sample Flags
	uint8le  pan;			// Sample Panning
	int8le   relnote;		// Sample Transpose
	uint8le  reserved;		// Reserved (abused for ModPlug's ADPCM compression)
	char     name[22];		// Sample Name, not null-terminated (any nulls are treated as spaces)

	// Convert OpenMPT's internal sample representation to an XMSample.
	void ConvertToXM(const ModSample &mptSmp, MODTYPE fromType, bool compatibilityExport);
	// Convert an XMSample to OpenMPT's internal sample representation.
	void ConvertToMPT(ModSample &mptSmp) const;
	// Retrieve the internal sample format flags for this instrument.
	SampleIO GetSampleFormat() const;
};

MPT_BINARY_STRUCT(XMSample, 40)


OPENMPT_NAMESPACE_END
