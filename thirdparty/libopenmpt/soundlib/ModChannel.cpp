/*
 * ModChannel.cpp
 * --------------
 * Purpose: Module Channel header class and helpers
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Sndfile.h"
#include "ModChannel.h"

OPENMPT_NAMESPACE_BEGIN

void ModChannel::Reset(ResetFlags resetMask, const CSoundFile &sndFile, CHANNELINDEX sourceChannel)
{
	if(resetMask & resetSetPosBasic)
	{
		nNote = nNewNote = NOTE_NONE;
		nNewIns = nOldIns = 0;
		pModSample = nullptr;
		pModInstrument = nullptr;
		nPortamentoDest = 0;
		nCommand = CMD_NONE;
		nPatternLoopCount = 0;
		nPatternLoop = 0;
		nFadeOutVol = 0;
		dwFlags.set(CHN_KEYOFF | CHN_NOTEFADE);
		dwOldFlags.reset();
		//IT compatibility 15. Retrigger
		if(sndFile.m_playBehaviour[kITRetrigger])
		{
			nRetrigParam = 1;
			nRetrigCount = 0;
		}
		nTremorCount = 0;
		nEFxSpeed = 0;
		proTrackerOffset = 0;
		lastZxxParam = 0xFF;
		isFirstTick = false;
	}

	if(resetMask & resetSetPosAdvanced)
	{
		nPeriod = 0;
		position.Set(0);
		nLength = 0;
		nLoopStart = 0;
		nLoopEnd = 0;
		nROfs = nLOfs = 0;
		pModSample = nullptr;
		pModInstrument = nullptr;
		nCutOff = 0x7F;
		nResonance = 0;
		nFilterMode = 0;
		rightVol = leftVol = 0;
		newRightVol = newLeftVol = 0;
		rightRamp = leftRamp = 0;
		nVolume = 0;	// Needs to be 0 for SMP_NODEFAULTVOLUME flag
		nVibratoPos = nTremoloPos = nPanbrelloPos = 0;
		nOldHiOffset = 0;
		nLeftVU = nRightVU = 0;

		//-->Custom tuning related
		m_ReCalculateFreqOnFirstTick = false;
		m_CalculateFreq = false;
		m_PortamentoFineSteps = 0;
		m_PortamentoTickSlide = 0;
		m_Freq = 0;
		//<--Custom tuning related.
	}

	if(resetMask & resetChannelSettings)
	{
		if(sourceChannel < MAX_BASECHANNELS)
		{
			dwFlags = sndFile.ChnSettings[sourceChannel].dwFlags;
			nPan = sndFile.ChnSettings[sourceChannel].nPan;
			nGlobalVol = sndFile.ChnSettings[sourceChannel].nVolume;
		} else
		{
			dwFlags.reset();
			nPan = 128;
			nGlobalVol = 64;
		}
		nRestorePanOnNewNote = 0;
		nRestoreCutoffOnNewNote = 0;
		nRestoreResonanceOnNewNote = 0;

	}
}


void ModChannel::Stop()
{
	nPeriod = 0;
	increment.Set(0);
	position.Set(0);
	nLeftVU = nRightVU = 0;
	nVolume = 0;
	pCurrentSample = nullptr;
}


void ModChannel::UpdateInstrumentVolume(const ModSample *smp, const ModInstrument *ins)
{
	nInsVol = 64;
	if(smp != nullptr)
		nInsVol = smp->nGlobalVol;
	if(ins != nullptr)
		nInsVol = (nInsVol * ins->nGlobalVol) / 64;
}


ModCommand::NOTE ModChannel::GetPluginNote(bool realNoteMapping) const
{
	if(nArpeggioLastNote != NOTE_NONE)
	{
		// If an arpeggio is playing, this definitely the last playing note, which may be different from the arpeggio base note stored in nNote.
		return nArpeggioLastNote;
	}
	ModCommand::NOTE plugNote = nNote;
	// Caution: When in compatible mode, ModChannel::nNote stores the "real" note, not the mapped note!
	if(realNoteMapping && pModInstrument != nullptr && plugNote >= NOTE_MIN && plugNote < (MPT_ARRAY_COUNT(pModInstrument->NoteMap) + NOTE_MIN))
	{
		plugNote = pModInstrument->NoteMap[plugNote - NOTE_MIN];
	}
	return plugNote;
}


OPENMPT_NAMESPACE_END
