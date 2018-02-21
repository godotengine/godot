/*
 * UpdateModule.cpp
 * ----------------
 * Purpose: CSoundFile functions for correcting modules made with previous versions of OpenMPT.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Sndfile.h"
#include "../common/StringFixer.h"
#include "../common/version.h"


OPENMPT_NAMESPACE_BEGIN


struct UpgradePatternData
{
	UpgradePatternData(CSoundFile &sf)
		: sndFile(sf)
		, chn(0)
		, compatPlay(sf.m_playBehaviour[MSF_COMPATIBLE_PLAY]) { }

	void operator() (ModCommand &m)
	{
		const CHANNELINDEX curChn = chn;
		chn++;
		if(chn >= sndFile.GetNumChannels())
		{
			chn = 0;
		}

		if(m.IsPcNote())
		{
			return;
		}
		const auto version = sndFile.m_dwLastSavedWithVersion;
		const auto modType = sndFile.GetType();

		if(modType == MOD_TYPE_S3M)
		{
			// Out-of-range global volume commands should be ignored in S3M. Fixed in OpenMPT 1.19 (r831).
			// So for tracks made with older versions of OpenMPT, we limit invalid global volume commands.
			if(version < MAKE_VERSION_NUMERIC(1, 19, 00, 00) && m.command == CMD_GLOBALVOLUME)
			{
				LimitMax(m.param, ModCommand::PARAM(64));
			}
		}

		else if(modType & (MOD_TYPE_IT | MOD_TYPE_MPT))
		{
			if(version < MAKE_VERSION_NUMERIC(1, 17, 03, 02) ||
				(!compatPlay && version < MAKE_VERSION_NUMERIC(1, 20, 00, 00)))
			{
				if(m.command == CMD_GLOBALVOLUME)
				{
					// Out-of-range global volume commands should be ignored in IT.
					// OpenMPT 1.17.03.02 fixed this in compatible mode, OpenMPT 1.20 fixes it in normal mode as well.
					// So for tracks made with older versions than OpenMPT 1.17.03.02 or tracks made with 1.17.03.02 <= version < 1.20, we limit invalid global volume commands.
					LimitMax(m.param, ModCommand::PARAM(128));
				}

				// SC0 and SD0 should be interpreted as SC1 and SD1 in IT files.
				// OpenMPT 1.17.03.02 fixed this in compatible mode, OpenMPT 1.20 fixes it in normal mode as well.
				else if(m.command == CMD_S3MCMDEX)
				{
					if(m.param == 0xC0)
					{
						m.command = CMD_NONE;
						m.note = NOTE_NOTECUT;
					} else if(m.param == 0xD0)
					{
						m.command = CMD_NONE;
					}
				}
			}

			// In the IT format, slide commands with both nibbles set should be ignored.
			// For note volume slides, OpenMPT 1.18 fixes this in compatible mode, OpenMPT 1.20 fixes this in normal mode as well.
			const bool noteVolSlide =
				(version < MAKE_VERSION_NUMERIC(1, 18, 00, 00) ||
				(!compatPlay && version < MAKE_VERSION_NUMERIC(1, 20, 00, 00)))
				&&
				(m.command == CMD_VOLUMESLIDE || m.command == CMD_VIBRATOVOL || m.command == CMD_TONEPORTAVOL || m.command == CMD_PANNINGSLIDE);

			// OpenMPT 1.20 also fixes this for global volume and channel volume slides.
			const bool chanVolSlide =
				(version < MAKE_VERSION_NUMERIC(1, 20, 00, 00))
				&&
				(m.command == CMD_GLOBALVOLSLIDE || m.command == CMD_CHANNELVOLSLIDE);

			if(noteVolSlide || chanVolSlide)
			{
				if((m.param & 0x0F) != 0x00 && (m.param & 0x0F) != 0x0F && (m.param & 0xF0) != 0x00 && (m.param & 0xF0) != 0xF0)
				{
					m.param &= 0x0F;
				}
			}

			if(version < MAKE_VERSION_NUMERIC(1, 22, 01, 04)
				&& version != MAKE_VERSION_NUMERIC(1, 22, 00, 00))	// Ignore compatibility export
			{
				// OpenMPT 1.22.01.04 fixes illegal (out of range) instrument numbers; they should do nothing. In previous versions, they stopped the playing sample.
				if(sndFile.GetNumInstruments() && m.instr > sndFile.GetNumInstruments() && !compatPlay)
				{
					m.volcmd = VOLCMD_VOLUME;
					m.vol = 0;
				}
			}
		}

		else if(modType == MOD_TYPE_XM)
		{
			// Something made be believe that out-of-range global volume commands are ignored in XM
			// just like they are ignored in IT, but apparently they are not. Aaaaaargh!
			if(((version >= MAKE_VERSION_NUMERIC(1, 17, 03, 02) && compatPlay) || (version >= MAKE_VERSION_NUMERIC(1, 20, 00, 00)))
				&& version < MAKE_VERSION_NUMERIC(1, 24, 02, 02)
				&& m.command == CMD_GLOBALVOLUME
				&& m.param > 64)
			{
				m.command = CMD_NONE;
			}

			if(version < MAKE_VERSION_NUMERIC(1, 19, 00, 00)
				|| (!compatPlay && version < MAKE_VERSION_NUMERIC(1, 20, 00, 00)))
			{
				if(m.command == CMD_OFFSET && m.volcmd == VOLCMD_TONEPORTAMENTO)
				{
					// If there are both a portamento and an offset effect, the portamento should be preferred in XM files.
					// OpenMPT 1.19 fixed this in compatible mode, OpenMPT 1.20 fixes it in normal mode as well.
					m.command = CMD_NONE;
				}
			}

			if(version < MAKE_VERSION_NUMERIC(1, 20, 01, 10)
				&& m.volcmd == VOLCMD_TONEPORTAMENTO && m.command == CMD_TONEPORTAMENTO
				&& (m.vol != 0 || compatPlay) && m.param != 0)
			{
				// Mx and 3xx on the same row does weird things in FT2: 3xx is completely ignored and the Mx parameter is doubled. Fixed in revision 1312 / OpenMPT 1.20.01.10
				// Previously the values were just added up, so let's fix this!
				m.volcmd = VOLCMD_NONE;
				const uint16 param = static_cast<uint16>(m.param) + static_cast<uint16>(m.vol << 4);
				m.param = mpt::saturate_cast<ModCommand::PARAM>(param);
			}

			if(version < MAKE_VERSION_NUMERIC(1, 22, 07, 09)
				&& m.command == CMD_SPEED && m.param == 0)
			{
				// OpenMPT can emulate FT2's F00 behaviour now.
				m.command = CMD_NONE;
			}
		}

		if(version < MAKE_VERSION_NUMERIC(1, 20, 00, 00))
		{
			// Pattern Delay fixes

			const bool fixS6x = (m.command == CMD_S3MCMDEX && (m.param & 0xF0) == 0x60);
			// We also fix X6x commands in hacked XM files, since they are treated identically to the S6x command in IT/S3M files.
			// We don't treat them in files made with OpenMPT 1.18+ that have compatible play enabled, though, since they are ignored there anyway.
			const bool fixX6x = (m.command == CMD_XFINEPORTAUPDOWN && (m.param & 0xF0) == 0x60
				&& (!(compatPlay && modType == MOD_TYPE_XM) || version < MAKE_VERSION_NUMERIC(1, 18, 00, 00)));

			if(fixS6x || fixX6x)
			{
				// OpenMPT 1.20 fixes multiple fine pattern delays on the same row. Previously, only the last command was considered,
				// but all commands should be added up. Since Scream Tracker 3 itself doesn't support S6x, we also use Impulse Tracker's behaviour here,
				// since we can assume that most S3Ms that make use of S6x were composed with Impulse Tracker.
				for(ModCommand *fixCmd = (&m) - curChn; fixCmd < &m; fixCmd++)
				{
					if((fixCmd->command == CMD_S3MCMDEX || fixCmd->command == CMD_XFINEPORTAUPDOWN) && (fixCmd->param & 0xF0) == 0x60)
					{
						fixCmd->command = CMD_NONE;
					}
				}
			}

			if(m.command == CMD_S3MCMDEX && (m.param & 0xF0) == 0xE0)
			{
				// OpenMPT 1.20 fixes multiple pattern delays on the same row. Previously, only the *last* command was considered,
				// but Scream Tracker 3 and Impulse Tracker only consider the *first* command.
				for(ModCommand *fixCmd = (&m) - curChn; fixCmd < &m; fixCmd++)
				{
					if(fixCmd->command == CMD_S3MCMDEX && (fixCmd->param & 0xF0) == 0xE0)
					{
						fixCmd->command = CMD_NONE;
					}
				}
			}
		}

		if(m.volcmd == VOLCMD_VIBRATODEPTH
			&& version < MAKE_VERSION_NUMERIC(1, 27, 00, 37)
			&& version != MAKE_VERSION_NUMERIC(1, 27, 00, 00))
		{
			// Fix handling of double vibrato commands - previously only one of them was applied at a time
			if(m.command == CMD_VIBRATOVOL && m.vol > 0)
			{
				m.command = CMD_VOLUMESLIDE;
			} else if((m.command == CMD_VIBRATO || m.command == CMD_FINEVIBRATO) && (m.param & 0x0F) == 0)
			{
				m.command = CMD_VIBRATO;
				m.param |= (m.vol & 0x0F);
				m.volcmd = VOLCMD_NONE;
			} else if(m.command == CMD_VIBRATO || m.command == CMD_VIBRATOVOL || m.command == CMD_FINEVIBRATO)
			{
				m.volcmd = VOLCMD_NONE;
			}
		}

		// Volume column offset in IT/XM is bad, mkay?
		if(modType != MOD_TYPE_MPT && m.volcmd == VOLCMD_OFFSET && m.command == CMD_NONE)
		{
			m.command = CMD_OFFSET;
			m.param = m.vol << 3;
			m.volcmd = VOLCMD_NONE;
		}

	}

	const CSoundFile &sndFile;
	CHANNELINDEX chn;
	const bool compatPlay;
};


void CSoundFile::UpgradeModule()
{
	if(m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 17, 02, 46) && m_dwLastSavedWithVersion != MAKE_VERSION_NUMERIC(1, 17, 00, 00))
	{
		// Compatible playback mode didn't exist in earlier versions, so definitely disable it.
		m_playBehaviour.reset(MSF_COMPATIBLE_PLAY);
	}

	const bool compatModeIT = m_playBehaviour[MSF_COMPATIBLE_PLAY] && (GetType() & (MOD_TYPE_IT | MOD_TYPE_MPT));
	const bool compatModeXM = m_playBehaviour[MSF_COMPATIBLE_PLAY] && GetType() == MOD_TYPE_XM;

	if(m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 20, 00, 00))
	{
		for(INSTRUMENTINDEX i = 1; i <= GetNumInstruments(); i++) if(Instruments[i] != nullptr)
		{
			ModInstrument *ins = Instruments[i];
			// Previously, volume swing values ranged from 0 to 64. They should reach from 0 to 100 instead.
			ins->nVolSwing = static_cast<uint8>(std::min<uint32>(ins->nVolSwing * 100 / 64, 100));

			if(!compatModeIT || m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 18, 00, 00))
			{
				// Previously, Pitch/Pan Separation was only half depth.
				// This was corrected in compatible mode in OpenMPT 1.18, and in OpenMPT 1.20 it is corrected in normal mode as well.
				ins->nPPS = (ins->nPPS + (ins->nPPS >= 0 ? 1 : -1)) / 2;
			}

			if(!compatModeIT || m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 17, 03, 02))
			{
				// IT compatibility 24. Short envelope loops
				// Previously, the pitch / filter envelope loop handling was broken, the loop was shortened by a tick (like in XM).
				// This was corrected in compatible mode in OpenMPT 1.17.03.02, and in OpenMPT 1.20 it is corrected in normal mode as well.
				ins->GetEnvelope(ENV_PITCH).Convert(MOD_TYPE_XM, GetType());
			}

			if(m_dwLastSavedWithVersion >= MAKE_VERSION_NUMERIC(1, 17, 00, 00) && m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 17, 02, 50))
			{
				// If there are any plugins that can receive volume commands, enable volume bug emulation.
				if(ins->nMixPlug && ins->HasValidMIDIChannel())
				{
					m_playBehaviour.set(kMIDICCBugEmulation);
				}
			}

			if(m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 17, 02, 50) && (ins->nVolSwing | ins->nPanSwing | ins->nCutSwing | ins->nResSwing))
			{
				// If there are any instruments with random variation, enable the old random variation behaviour.
				m_playBehaviour.set(kMPTOldSwingBehaviour);
				break;
			}
		}

		if((GetType() & (MOD_TYPE_IT | MOD_TYPE_MPT)) && (m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 17, 03, 02) || !compatModeIT))
		{
			// In the IT format, a sweep value of 0 shouldn't apply vibrato at all. Previously, a value of 0 was treated as "no sweep".
			// In OpenMPT 1.17.03.02, this was corrected in compatible mode, in OpenMPT 1.20 it is corrected in normal mode as well,
			// so we have to fix the setting while loading.
			for(SAMPLEINDEX i = 1; i <= GetNumSamples(); i++)
			{
				if(Samples[i].nVibSweep == 0 && (Samples[i].nVibDepth | Samples[i].nVibRate))
				{
					Samples[i].nVibSweep = 255;
				}
			}
		}

		// Fix old nasty broken (non-standard) MIDI configs in files.
		m_MidiCfg.UpgradeMacros();
	}

	if(m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 20, 02, 10)
		&& m_dwLastSavedWithVersion != MAKE_VERSION_NUMERIC(1, 20, 00, 00)
		&& (GetType() & (MOD_TYPE_XM | MOD_TYPE_IT | MOD_TYPE_MPT)))
	{
		bool instrPlugs = false;
		// Old pitch wheel commands were closest to sample pitch bend commands if the PWD is 13.
		for(INSTRUMENTINDEX i = 1; i <= GetNumInstruments(); i++)
		{
			if(Instruments[i] != nullptr && Instruments[i]->nMidiChannel != MidiNoChannel)
			{
				Instruments[i]->midiPWD = 13;
				instrPlugs = true;
			}
		}
		if(instrPlugs)
		{
			m_playBehaviour.set(kOldMIDIPitchBends);
		}
	}

	if(m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 22, 03, 12)
		&& m_dwLastSavedWithVersion != MAKE_VERSION_NUMERIC(1, 22, 00, 00)
		&& (GetType() & (MOD_TYPE_IT | MOD_TYPE_MPT))
		&& (m_playBehaviour[MSF_COMPATIBLE_PLAY] || m_playBehaviour[kMPTOldSwingBehaviour]))
	{
		// The "correct" pan swing implementation did nothing if the instrument also had a pan envelope.
		// If there's a pan envelope, disable pan swing for such modules.
		for(INSTRUMENTINDEX i = 1; i <= GetNumInstruments(); i++)
		{
			if(Instruments[i] != nullptr && Instruments[i]->nPanSwing != 0 && Instruments[i]->PanEnv.dwFlags[ENV_ENABLED])
			{
				Instruments[i]->nPanSwing = 0;
			}
		}
	}

#ifndef NO_PLUGINS
	if(m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 22, 07, 01))
	{
		// Convert ANSI plugin path names to UTF-8 (irrelevant in probably 99% of all cases anyway, I think I've never seen a VST plugin with a non-ASCII file name)
		for(PLUGINDEX i = 0; i < MAX_MIXPLUGINS; i++)
		{
#if defined(MODPLUG_TRACKER)
			const std::string name = mpt::ToCharset(mpt::CharsetUTF8, mpt::CharsetLocale, m_MixPlugins[i].Info.szLibraryName);
#else
			const std::string name = mpt::ToCharset(mpt::CharsetUTF8, mpt::CharsetWindows1252, m_MixPlugins[i].Info.szLibraryName);
#endif
			mpt::String::Copy(m_MixPlugins[i].Info.szLibraryName, name);
		}
	}
#endif // NO_PLUGINS

	// Starting from OpenMPT 1.22.07.19, FT2-style panning was applied in compatible mix mode.
	// Starting from OpenMPT 1.23.01.04, FT2-style panning has its own mix mode instead.
	if(GetType() == MOD_TYPE_XM)
	{
		if(m_dwLastSavedWithVersion >= MAKE_VERSION_NUMERIC(1, 22, 07, 19)
			&& m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 23, 01, 04)
			&& GetMixLevels() == mixLevelsCompatible)
		{
			SetMixLevels(mixLevelsCompatibleFT2);
		}
	}

	if(m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 25, 00, 07) && m_dwLastSavedWithVersion != MAKE_VERSION_NUMERIC(1, 25, 00, 00))
	{
		// Instrument plugins can now receive random volume variation.
		// For old instruments, disable volume swing in case there was no sample associated.
		for(INSTRUMENTINDEX i = 1; i <= GetNumInstruments(); i++)
		{
			if(Instruments[i] != nullptr && Instruments[i]->nVolSwing != 0 && Instruments[i]->nMidiChannel != MidiNoChannel)
			{
				bool hasSample = false;
				for(size_t k = 0; k < CountOf(Instruments[k]->Keyboard); k++)
				{
					if(Instruments[i]->Keyboard[k] != 0)
					{
						hasSample = true;
						break;
					}
				}
				if(!hasSample)
				{
					Instruments[i]->nVolSwing = 0;
				}
			}
		}
	}

	if(m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 26, 00, 00))
	{
		for(INSTRUMENTINDEX i = 1; i <= GetNumInstruments(); i++) if(Instruments[i] != nullptr)
		{
			ModInstrument *ins = Instruments[i];
			// Even after fixing it in OpenMPT 1.18, instrument PPS was only half the depth.
			ins->nPPS = (ins->nPPS + (ins->nPPS >= 0 ? 1 : -1)) / 2;

			// OpenMPT 1.18 fixed the depth of random pan in compatible mode.
			// OpenMPT 1.26 fixes it in normal mode too.
			if(!compatModeIT || m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 18, 00, 00))
			{
				ins->nPanSwing = (ins->nPanSwing + 3) / 4u;
			}
		}
	}

	Patterns.ForEachModCommand(UpgradePatternData(*this));

	// Convert compatibility flags
	// NOTE: Some of these version numbers are just approximations.
	// Sometimes a quirk flag is shared by several code locations which might have been fixed at different times.
	// Sometimes the quirk behaviour has been revised over time, in which case the first version that emulated the quirk enables it.
	struct PlayBehaviourVersion
	{
		PlayBehaviour behaviour;
		MptVersion::VersionNum version;
	};
	
	if(compatModeIT && m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 26, 00, 00))
	{
		// Pre-1.26: Detailed compatibility flags did not exist.
		static constexpr PlayBehaviourVersion behaviours[] =
		{
			{ kTempoClamp,						MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kPerChannelGlobalVolSlide,		MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kPanOverride,						MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kITInstrWithoutNote,				MAKE_VERSION_NUMERIC(1, 17, 02, 46) },
			{ kITVolColFinePortamento,			MAKE_VERSION_NUMERIC(1, 17, 02, 49) },
			{ kITArpeggio,						MAKE_VERSION_NUMERIC(1, 17, 02, 49) },
			{ kITOutOfRangeDelay,				MAKE_VERSION_NUMERIC(1, 17, 02, 49) },
			{ kITPortaMemoryShare,				MAKE_VERSION_NUMERIC(1, 17, 02, 49) },
			{ kITPatternLoopTargetReset,		MAKE_VERSION_NUMERIC(1, 17, 02, 49) },
			{ kITFT2PatternLoop,				MAKE_VERSION_NUMERIC(1, 17, 02, 49) },
			{ kITPingPongNoReset,				MAKE_VERSION_NUMERIC(1, 17, 02, 51) },
			{ kITEnvelopeReset,					MAKE_VERSION_NUMERIC(1, 17, 02, 51) },
			{ kITClearOldNoteAfterCut,			MAKE_VERSION_NUMERIC(1, 17, 02, 52) },
			{ kITVibratoTremoloPanbrello,		MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kITTremor,						MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kITRetrigger,						MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kITMultiSampleBehaviour,			MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kITPortaTargetReached,			MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kITPatternLoopBreak,				MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kITOffset,						MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kITSwingBehaviour,				MAKE_VERSION_NUMERIC(1, 18, 00, 00) },
			{ kITNNAReset,						MAKE_VERSION_NUMERIC(1, 18, 00, 00) },
			{ kITSCxStopsSample,				MAKE_VERSION_NUMERIC(1, 18, 00, 01) },
			{ kITEnvelopePositionHandling,		MAKE_VERSION_NUMERIC(1, 18, 01, 00) },
			{ kITPortamentoInstrument,			MAKE_VERSION_NUMERIC(1, 19, 00, 01) },
			{ kITPingPongMode,					MAKE_VERSION_NUMERIC(1, 19, 00, 21) },
			{ kITRealNoteMapping,				MAKE_VERSION_NUMERIC(1, 19, 00, 30) },
			{ kITHighOffsetNoRetrig,			MAKE_VERSION_NUMERIC(1, 20, 00, 14) },
			{ kITFilterBehaviour,				MAKE_VERSION_NUMERIC(1, 20, 00, 35) },
			{ kITNoSurroundPan,					MAKE_VERSION_NUMERIC(1, 20, 00, 53) },
			{ kITShortSampleRetrig,				MAKE_VERSION_NUMERIC(1, 20, 00, 54) },
			{ kITPortaNoNote,					MAKE_VERSION_NUMERIC(1, 20, 00, 56) },
			{ kRowDelayWithNoteDelay,			MAKE_VERSION_NUMERIC(1, 20, 00, 76) },
			{ kITDontResetNoteOffOnPorta,		MAKE_VERSION_NUMERIC(1, 20, 02, 06) },
			{ kITVolColMemory,					MAKE_VERSION_NUMERIC(1, 21, 01, 16) },
			{ kITPortamentoSwapResetsPos,		MAKE_VERSION_NUMERIC(1, 21, 01, 25) },
			{ kITEmptyNoteMapSlot,				MAKE_VERSION_NUMERIC(1, 21, 01, 25) },
			{ kITFirstTickHandling,				MAKE_VERSION_NUMERIC(1, 22, 07, 09) },
			{ kITSampleAndHoldPanbrello,		MAKE_VERSION_NUMERIC(1, 22, 07, 19) },
			{ kITClearPortaTarget,				MAKE_VERSION_NUMERIC(1, 23, 04, 03) },
			{ kITPanbrelloHold,					MAKE_VERSION_NUMERIC(1, 24, 01, 06) },
			{ kITPanningReset,					MAKE_VERSION_NUMERIC(1, 24, 01, 06) },
			{ kITPatternLoopWithJumps,			MAKE_VERSION_NUMERIC(1, 25, 00, 19) },
		};

		for(const auto &b : behaviours)
		{
			m_playBehaviour.set(b.behaviour, (m_dwLastSavedWithVersion >= b.version || m_dwLastSavedWithVersion == (b.version & 0xFFFF0000)));
		}
	} else if(compatModeXM && m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 26, 00, 00))
	{
		// Pre-1.26: Detailed compatibility flags did not exist.
		static constexpr PlayBehaviourVersion behaviours[] =
		{
			{ kTempoClamp,						MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kPerChannelGlobalVolSlide,		MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kPanOverride,						MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kITFT2PatternLoop,				MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kFT2Arpeggio,						MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kFT2Retrigger,					MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kFT2VolColVibrato,				MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kFT2PortaNoNote,					MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kFT2KeyOff,						MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kFT2PanSlide,						MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kFT2OffsetOutOfRange,				MAKE_VERSION_NUMERIC(1, 17, 03, 02) },
			{ kFT2RestrictXCommand,				MAKE_VERSION_NUMERIC(1, 18, 00, 00) },
			{ kFT2RetrigWithNoteDelay,			MAKE_VERSION_NUMERIC(1, 18, 00, 00) },
			{ kFT2SetPanEnvPos,					MAKE_VERSION_NUMERIC(1, 18, 00, 00) },
			{ kFT2PortaIgnoreInstr,				MAKE_VERSION_NUMERIC(1, 18, 00, 01) },
			{ kFT2VolColMemory,					MAKE_VERSION_NUMERIC(1, 18, 01, 00) },
			{ kFT2LoopE60Restart,				MAKE_VERSION_NUMERIC(1, 18, 02, 01) },
			{ kFT2ProcessSilentChannels,		MAKE_VERSION_NUMERIC(1, 18, 02, 01) },
			{ kFT2ReloadSampleSettings,			MAKE_VERSION_NUMERIC(1, 20, 00, 36) },
			{ kFT2PortaDelay,					MAKE_VERSION_NUMERIC(1, 20, 00, 40) },
			{ kFT2Transpose,					MAKE_VERSION_NUMERIC(1, 20, 00, 62) },
			{ kFT2PatternLoopWithJumps,			MAKE_VERSION_NUMERIC(1, 20, 00, 69) },
			{ kFT2PortaTargetNoReset,			MAKE_VERSION_NUMERIC(1, 20, 00, 69) },
			{ kFT2EnvelopeEscape,				MAKE_VERSION_NUMERIC(1, 20, 00, 77) },
			{ kFT2Tremor,						MAKE_VERSION_NUMERIC(1, 20, 01, 11) },
			{ kFT2OutOfRangeDelay,				MAKE_VERSION_NUMERIC(1, 20, 02, 02) },
			{ kFT2Periods,						MAKE_VERSION_NUMERIC(1, 22, 03, 01) },
			{ kFT2PanWithDelayedNoteOff,		MAKE_VERSION_NUMERIC(1, 22, 03, 02) },
			{ kFT2VolColDelay,					MAKE_VERSION_NUMERIC(1, 22, 07, 19) },
			{ kFT2FinetunePrecision,			MAKE_VERSION_NUMERIC(1, 22, 07, 19) },
		};

		for(const auto &b : behaviours)
		{
			m_playBehaviour.set(b.behaviour, m_dwLastSavedWithVersion >= b.version);
		}
	}
	
	if(GetType() & (MOD_TYPE_IT | MOD_TYPE_MPT))
	{
		// The following behaviours were added in/after OpenMPT 1.26, so are not affected by the upgrade mechanism above.
		static constexpr PlayBehaviourVersion behaviours[] =
		{
			{ kITInstrWithNoteOff,				MAKE_VERSION_NUMERIC(1, 26, 00, 01) },
			{ kITMultiSampleInstrumentNumber,	MAKE_VERSION_NUMERIC(1, 27, 00, 27) },
		};

		for(const auto &b : behaviours)
		{
			if(m_dwLastSavedWithVersion < (b.version & 0xFFFF0000))
				m_playBehaviour.reset(b.behaviour);
			// Full version information available, i.e. not compatibility-exported.
			else if(m_dwLastSavedWithVersion > (b.version & 0xFFFF0000) && m_dwLastSavedWithVersion < b.version)
				m_playBehaviour.reset(b.behaviour);
		}
	} else if(GetType() == MOD_TYPE_XM)
	{
		// The following behaviours were added after OpenMPT 1.26, so are not affected by the upgrade mechanism above.
		static constexpr PlayBehaviourVersion behaviours[] =
		{
			{ kFT2NoteOffFlags,					MAKE_VERSION_NUMERIC(1, 27, 00, 27) },
			{ kRowDelayWithNoteDelay,			MAKE_VERSION_NUMERIC(1, 27, 00, 37) },
			{ kFT2TremoloRampWaveform,			MAKE_VERSION_NUMERIC(1, 27, 00, 37) },
			{ kFT2PortaUpDownMemory,			MAKE_VERSION_NUMERIC(1, 27, 00, 37) },
		};

		for(const auto &b : behaviours)
		{
			if(m_dwLastSavedWithVersion < b.version)
				m_playBehaviour.reset(b.behaviour);
		}
	} else if(GetType() == MOD_TYPE_S3M)
	{
		// We do not store any of these flags in S3M files.
		static constexpr PlayBehaviourVersion behaviours[] =
		{
			{ kST3NoMutedChannels,		MAKE_VERSION_NUMERIC(1, 18, 00, 00) },
			{ kST3EffectMemory,			MAKE_VERSION_NUMERIC(1, 20, 00, 00) },
			{ kRowDelayWithNoteDelay,	MAKE_VERSION_NUMERIC(1, 20, 00, 00) },
			{ kST3PortaSampleChange,	MAKE_VERSION_NUMERIC(1, 22, 00, 00) },
			{ kST3VibratoMemory,		MAKE_VERSION_NUMERIC(1, 26, 00, 00) },
			{ kITPanbrelloHold,			MAKE_VERSION_NUMERIC(1, 26, 00, 00) },
			{ KST3PortaAfterArpeggio,	MAKE_VERSION_NUMERIC(1, 27, 00, 00) },
		};

		for(const auto &b : behaviours)
		{
			if(m_dwLastSavedWithVersion < b.version)
				m_playBehaviour.reset(b.behaviour);
		}
	}

	if(m_dwLastSavedWithVersion >= MAKE_VERSION_NUMERIC(1, 27, 00, 27) && m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 27, 00, 49))
	{
		// OpenMPT 1.27 inserted some IT/FT2 flags before the S3M flags that are never saved to files anyway, to keep the flag IDs a bit more compact.
		// However, it was overlooked that these flags would still be read by OpenMPT 1.26 and thus S3M-specific behaviour would be enabled in IT/XM files.
		// Hence, in OpenMPT 1.27.00.49 the flag IDs got remapped to no longer conflict with OpenMPT 1.26.
		// Files made with the affected pre-release versions of OpenMPT 1.27 are upgraded here to use the new IDs.
		for(int i = 0; i < 5; i++)
		{
			m_playBehaviour.set(kFT2NoteOffFlags + i, m_playBehaviour[kST3NoMutedChannels + i]);
			m_playBehaviour.reset(kST3NoMutedChannels + i);
		}
	}

	if(m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 17, 00, 00))
	{
		// MPT 1.16 has a maximum tempo of 255.
		m_playBehaviour.set(kTempoClamp);
	} else if(m_dwLastSavedWithVersion >= MAKE_VERSION_NUMERIC(1, 17, 00, 00) && m_dwLastSavedWithVersion <= MAKE_VERSION_NUMERIC(1, 20, 01, 03) && m_dwLastSavedWithVersion != MAKE_VERSION_NUMERIC(1, 20, 00, 00))
	{
		// OpenMPT introduced some "fixes" that execute regular portamentos also at speed 1.
		m_playBehaviour.set(kSlidesAtSpeed1);
	}

	if(m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 24, 00, 00))
	{
		// No frequency slides in Hz before OpenMPT 1.24
		m_playBehaviour.reset(kHertzInLinearMode);
	} else if(m_dwLastSavedWithVersion >= MAKE_VERSION_NUMERIC(1, 24, 00, 00) && m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 26, 00, 00) && (GetType() & (MOD_TYPE_IT | MOD_TYPE_MPT)))
	{
		// Frequency slides were always in Hz rather than periods in this version range.
		m_playBehaviour.set(kHertzInLinearMode);
	}
}


OPENMPT_NAMESPACE_END
