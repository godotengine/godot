/*
 * ModChannel.h
 * ------------
 * Purpose: Module Channel header class and helpers
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "ModSample.h"
#include "ModInstrument.h"
#include "modcommand.h"
#include "Paula.h"

OPENMPT_NAMESPACE_BEGIN

class CSoundFile;

// Mix Channel Struct
struct ModChannel
{
	// Envelope playback info
	struct EnvInfo
	{
		FlagSet<EnvelopeFlags> flags;
		uint32 nEnvPosition;
		int32 nEnvValueAtReleaseJump;

		void Reset()
		{
			nEnvPosition = 0;
			nEnvValueAtReleaseJump = NOT_YET_RELEASED;
		}
	};

	// Information used in the mixer (should be kept tight for better caching)
	SamplePosition position;	// Current play position (fixed point)
	SamplePosition increment;	// Sample speed relative to mixing frequency (fixed point)
	const void *pCurrentSample;	// Currently playing sample (nullptr if no sample is playing)
	int32 leftVol;			// 0...4096 (12 bits, since 16 bits + 12 bits = 28 bits = 0dB in integer mixer, see MIXING_ATTENUATION)
	int32 rightVol;			// Ditto
	int32 leftRamp;			// Ramping delta, 20.12 fixed point (see VOLUMERAMPPRECISION)
	int32 rightRamp;		// Ditto
	int32 rampLeftVol;		// Current ramping volume, 20.12 fixed point (see VOLUMERAMPPRECISION)
	int32 rampRightVol;		// Ditto
	mixsample_t nFilter_Y[2][2];					// Filter memory - two history items per sample channel
	mixsample_t nFilter_A0, nFilter_B0, nFilter_B1;	// Filter coeffs
	mixsample_t nFilter_HP;

	SmpLength nLength;
	SmpLength nLoopStart;
	SmpLength nLoopEnd;
	FlagSet<ChannelFlags> dwFlags;
	mixsample_t nROfs, nLOfs;
	uint32 nRampLength;

	const ModSample *pModSample;			// Currently assigned sample slot (may already be stopped)
	Paula::State paulaState;

	// Information not used in the mixer
	const ModInstrument *pModInstrument;	// Currently assigned instrument slot
	SmpLength proTrackerOffset;				// Offset for instrument-less notes in ProTracker mode
	SmpLength oldOffset;
	FlagSet<ChannelFlags> dwOldFlags;		// Flags from previous tick
	int32 newLeftVol, newRightVol;
	int32 nRealVolume, nRealPan;
	int32 nVolume, nPan, nFadeOutVol;
	int32 nPeriod, nC5Speed, nPortamentoDest;
	int32 cachedPeriod, glissandoPeriod;
	int32 nCalcVolume;								// Calculated channel volume, 14-Bit (without global volume, pre-amp etc applied) - for MIDI macros
	EnvInfo VolEnv, PanEnv, PitchEnv;				// Envelope playback info
	int32 nGlobalVol;	// Channel volume (CV in ITTECH.TXT)
	int32 nInsVol;		// Sample / Instrument volume (SV * IV in ITTECH.TXT)
	int32 nFineTune, nTranspose;
	int32 nPortamentoSlide, nAutoVibDepth;
	uint32 nEFxOffset; // offset memory for Invert Loop (EFx, .MOD only)
	int16 nVolSwing, nPanSwing;
	int16 nCutSwing, nResSwing;
	int16 nRestorePanOnNewNote; //If > 0, nPan should be set to nRestorePanOnNewNote - 1 on new note. Used to recover from panswing.
	int16 nRetrigCount, nRetrigParam;
	ROWINDEX nPatternLoop;
	CHANNELINDEX nMasterChn;
	ModCommand rowCommand;
	// 8-bit members
	uint8 resamplingMode;
	uint8 nRestoreResonanceOnNewNote;	// See nRestorePanOnNewNote
	uint8 nRestoreCutoffOnNewNote;		// ditto
	uint8 nNote, nNNA;
	uint8 nLastNote;				// Last note, ignoring note offs and cuts - for MIDI macros
	uint8 nArpeggioLastNote, nArpeggioBaseNote;	// For plugin arpeggio
	uint8 nNewNote, nNewIns, nOldIns, nCommand, nArpeggio;
	uint8 nOldVolumeSlide, nOldFineVolUpDown;
	uint8 nOldPortaUp, nOldPortaDown, nOldFinePortaUpDown, nOldExtraFinePortaUpDown;
	uint8 nOldPanSlide, nOldChnVolSlide;
	uint8 nOldGlobalVolSlide;
	uint8 nAutoVibPos, nVibratoPos, nTremoloPos, nPanbrelloPos;
	uint8 nVibratoType, nVibratoSpeed, nVibratoDepth;
	uint8 nTremoloType, nTremoloSpeed, nTremoloDepth;
	uint8 nPanbrelloType, nPanbrelloSpeed, nPanbrelloDepth;
	int8  nPanbrelloOffset, nPanbrelloRandomMemory;
	uint8 nOldCmdEx, nOldVolParam, nOldTempo;
	uint8 nOldHiOffset;
	uint8 nCutOff, nResonance;
	uint8 nTremorCount, nTremorParam;
	uint8 nPatternLoopCount;
	uint8 nLeftVU, nRightVU;
	uint8 nActiveMacro, nFilterMode;
	uint8 nEFxSpeed, nEFxDelay;		// memory for Invert Loop (EFx, .MOD only)
	uint8 nNoteSlideCounter, nNoteSlideSpeed, nNoteSlideStep;	// IMF / PTM Note Slide
	uint8 lastZxxParam;	// Memory for \xx slides
	bool isFirstTick : 1;

	//-->Variables used to make user-definable tuning modes work with pattern effects.
	//If true, freq should be recalculated in ReadNote() on first tick.
	//Currently used only for vibrato things - using in other context might be 
	//problematic.
	bool m_ReCalculateFreqOnFirstTick : 1;

	//To tell whether to calculate frequency.
	bool m_CalculateFreq : 1;

	int32 m_PortamentoFineSteps, m_PortamentoTickSlide;

	uint32 m_Freq;
	//<----

	//NOTE_PCs memory.
	float m_plugParamValueStep, m_plugParamTargetValue;
	uint16 m_RowPlugParam;
	PLUGINDEX m_RowPlug;

	void ClearRowCmd() { rowCommand = ModCommand::Empty(); }

	// Get a reference to a specific envelope of this channel
	const EnvInfo &GetEnvelope(EnvelopeType envType) const
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

	EnvInfo &GetEnvelope(EnvelopeType envType)
	{
		return const_cast<EnvInfo &>(static_cast<const ModChannel *>(this)->GetEnvelope(envType));
	}

	void ResetEnvelopes()
	{
		VolEnv.Reset();
		PanEnv.Reset();
		PitchEnv.Reset();
	}

	enum ResetFlags
	{
		resetChannelSettings =	1,		// Reload initial channel settings
		resetSetPosBasic =		2,		// Reset basic runtime channel attributes
		resetSetPosAdvanced =	4,		// Reset more runtime channel attributes
		resetSetPosFull = 		resetSetPosBasic | resetSetPosAdvanced | resetChannelSettings,		// Reset all runtime channel attributes
		resetTotal =			resetSetPosFull,
	};

	void Reset(ResetFlags resetMask, const CSoundFile &sndFile, CHANNELINDEX sourceChannel);
	void Stop();

	bool IsSamplePlaying() const { return !increment.IsZero(); }

	uint32 GetVSTVolume() { return (pModInstrument) ? pModInstrument->nGlobalVol * 4 : nVolume; }

	ModCommand::NOTE GetPluginNote(bool realNoteMapping) const;

	// Check if the channel has a valid MIDI output. This function guarantees that pModInstrument != nullptr.
	bool HasMIDIOutput() const { return pModInstrument != nullptr && pModInstrument->HasValidMIDIChannel(); }

	// Check if currently processed loop is a sustain loop. pModSample is not checked for validity!
	bool InSustainLoop() const { return (dwFlags & (CHN_LOOP | CHN_KEYOFF)) == CHN_LOOP && pModSample->uFlags[CHN_SUSTAINLOOP]; }

	void UpdateInstrumentVolume(const ModSample *smp, const ModInstrument *ins);
};


// Default pattern channel settings
struct ModChannelSettings
{
	FlagSet<ChannelFlags> dwFlags;	// Channel flags
	uint16 nPan;					// Initial pan (0...256)
	uint16 nVolume;					// Initial channel volume (0...64)
	PLUGINDEX nMixPlugin;			// Assigned plugin
	char szName[MAX_CHANNELNAME];	// Channel name

	ModChannelSettings()
	{
		Reset();
	}

	void Reset()
	{
		dwFlags.reset();
		nPan = 128;
		nVolume = 64;
		nMixPlugin = 0;
		szName[0] = '\0';
	}
};

OPENMPT_NAMESPACE_END
