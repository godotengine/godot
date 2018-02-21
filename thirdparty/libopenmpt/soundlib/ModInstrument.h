/*
 * ModInstrument.h
 * ---------------
 * Purpose: Module Instrument header class and helpers
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "tuningbase.h"
#include "Snd_defs.h"
#include "../common/FlagSet.h"
#include "../common/misc_util.h"
#include <set>

OPENMPT_NAMESPACE_BEGIN

// Instrument Nodes
struct EnvelopeNode
{
	typedef uint16 tick_t;
	typedef uint8 value_t;

	tick_t tick;	// Envelope node position (x axis)
	value_t value;	// Envelope node value (y axis)

	EnvelopeNode() : tick(0), value(0) { }
	EnvelopeNode(tick_t tick, value_t value) : tick(tick), value(value) { }

	bool operator== (const EnvelopeNode &other) const { return tick == other.tick && value == other.value; }
};

// Instrument Envelopes
struct InstrumentEnvelope : public std::vector<EnvelopeNode>
{
	FlagSet<EnvelopeFlags> dwFlags;	// Envelope flags
	uint8 nLoopStart;				// Loop start node
	uint8 nLoopEnd;					// Loop end node
	uint8 nSustainStart;			// Sustain start node
	uint8 nSustainEnd;				// Sustain end node
	uint8 nReleaseNode;				// Release node

	InstrumentEnvelope()
	{
		nLoopStart = nLoopEnd = 0;
		nSustainStart = nSustainEnd = 0;
		nReleaseNode = ENV_RELEASE_NODE_UNSET;
	}

	// Convert envelope data between various formats.
	void Convert(MODTYPE fromType, MODTYPE toType);

	// Get envelope value at a given tick. Assumes that the envelope data is in rage [0, rangeIn],
	// returns value in range [0, rangeOut].
	int32 GetValueFromPosition(int position, int32 rangeOut, int32 rangeIn = ENVELOPE_MAX) const;

	// Ensure that ticks are ordered in increasing order and values are within the allowed range.
	void Sanitize(uint8 maxValue = ENVELOPE_MAX);

	uint32 size() const { return static_cast<uint32>(std::vector<EnvelopeNode>::size()); }

	using std::vector<EnvelopeNode>::push_back;
	void push_back(EnvelopeNode::tick_t tick, EnvelopeNode::value_t value) { push_back(EnvelopeNode(tick, value)); }
};

// Instrument Struct
struct ModInstrument
{
	FlagSet<InstrumentFlags> dwFlags;	// Instrument flags
	uint32 nFadeOut;					// Instrument fadeout speed
	uint32 nGlobalVol;					// Global volume (0...64, all sample volumes are multiplied with this - TODO: This is 0...128 in Impulse Tracker)
	uint32 nPan;						// Default pan (0...256), if the appropriate flag is set. Sample panning overrides instrument panning.

	uint16 nVolRampUp;					// Default sample ramping up, 0 = use global default

	uint16 wMidiBank;					// MIDI Bank (1...16384). 0 = Don't send.
	uint8 nMidiProgram;					// MIDI Program (1...128). 0 = Don't send.
	uint8 nMidiChannel;					// MIDI Channel (1...16). 0 = Don't send. 17 = Mapped (Send to tracker channel modulo 16).
	uint8 nMidiDrumKey;					// Drum set note mapping (currently only used by the .MID loader)
	int8 midiPWD;						// MIDI Pitch Wheel Depth in semitones

	uint8 nNNA;							// New note action (NNA_* constants)
	uint8 nDCT;							// Duplicate check type	(i.e. which condition will trigger the duplicate note action, DCT_* constants)
	uint8 nDNA;							// Duplicate note action (DNA_* constants)
	uint8 nPanSwing;					// Random panning factor (0...64)
	uint8 nVolSwing;					// Random volume factor (0...100)
	uint8 nIFC;							// Default filter cutoff (0...127). Used if the high bit is set
	uint8 nIFR;							// Default filter resonance (0...127). Used if the high bit is set

	int8 nPPS;							// Pitch/Pan separation (i.e. how wide the panning spreads, -32...32)
	uint8 nPPC;							// Pitch/Pan centre (zero-based, default is NOTE_MIDDLE_C - 1)

	PLUGINDEX nMixPlug;					// Plugin assigned to this instrument (0 = no plugin, 1 = first plugin)
	uint8 nCutSwing;					// Random cutoff factor (0...64)
	uint8 nResSwing;					// Random resonance factor (0...64)
	uint8 nFilterMode;					// Default filter mode (FLTMODE_* constants)
	uint8 nPluginVelocityHandling;		// How to deal with plugin velocity (PLUGIN_VELOCITYHANDLING_* constants)
	uint8 nPluginVolumeHandling;		// How to deal with plugin volume (PLUGIN_VOLUMEHANDLING_* constants)
	TEMPO pitchToTempoLock;				// BPM at which the samples assigned to this instrument loop correctly (0 = unset)
	uint32 nResampling;					// Resampling mode (SRCMODE_* constants)
	CTuning *pTuning;					// sample tuning assigned to this instrument

	InstrumentEnvelope VolEnv;			// Volume envelope data
	InstrumentEnvelope PanEnv;			// Panning envelope data
	InstrumentEnvelope PitchEnv;		// Pitch / filter envelope data

	uint8 NoteMap[128];					// Note mapping, e.g. C-5 => D-5.
	SAMPLEINDEX Keyboard[128];			// Sample mapping, e.g. C-5 => Sample 1

	char name[MAX_INSTRUMENTNAME];
	char filename[MAX_INSTRUMENTFILENAME];

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// WHEN adding new members here, ALSO update InstrumentExtensions.cpp
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	void SetTuning(CTuning* pT)
	{
		pTuning = pT;
	}

	ModInstrument(SAMPLEINDEX sample = 0);

	// Assign all notes to a given sample.
	void AssignSample(SAMPLEINDEX sample)
	{
		for(size_t n = 0; n < CountOf(Keyboard); n++)
		{
			Keyboard[n] = sample;
		}
	}

	// Reset note mapping (i.e. every note is mapped to itself)
	void ResetNoteMap()
	{
		for(size_t n = 0; n < CountOf(NoteMap); n++)
		{
			NoteMap[n] = static_cast<uint8>(n + 1);
		}
	}

	bool IsCutoffEnabled() const { return (nIFC & 0x80) != 0; }
	bool IsResonanceEnabled() const { return (nIFR & 0x80) != 0; }
	uint8 GetCutoff() const { return (nIFC & 0x7F); }
	uint8 GetResonance() const { return (nIFR & 0x7F); }
	void SetCutoff(uint8 cutoff, bool enable) { nIFC = std::min<uint8>(cutoff, 0x7F) | (enable ? 0x80 : 0x00); }
	void SetResonance(uint8 resonance, bool enable) { nIFR = std::min<uint8>(resonance, 0x7F) | (enable ? 0x80 : 0x00); }

	bool HasValidMIDIChannel() const { return (nMidiChannel >= 1 && nMidiChannel <= 17); }

	// Get a reference to a specific envelope of this instrument
	const InstrumentEnvelope &GetEnvelope(EnvelopeType envType) const
	{
		switch(envType)
		{
		case ENV_VOLUME:
		default:
			return VolEnv;
		case ENV_PANNING:
			return PanEnv;
		case ENV_PITCH:
			return PitchEnv;
		}
	}

	InstrumentEnvelope &GetEnvelope(EnvelopeType envType)
	{
		return const_cast<InstrumentEnvelope &>(static_cast<const ModInstrument &>(*this).GetEnvelope(envType));
	}

	// Get a set of all samples referenced by this instrument
	std::set<SAMPLEINDEX> GetSamples() const;

	// Write sample references into a bool vector. If a sample is referenced by this instrument, true is written.
	// The caller has to initialize the vector.
	void GetSamples(std::vector<bool> &referencedSamples) const;

	// Translate instrument properties between two given formats.
	void Convert(MODTYPE fromType, MODTYPE toType);

	// Sanitize all instrument data.
	void Sanitize(MODTYPE modType);

};

OPENMPT_NAMESPACE_END
