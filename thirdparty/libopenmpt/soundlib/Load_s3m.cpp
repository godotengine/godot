/*
 * Load_s3m.cpp
 * ------------
 * Purpose: S3M (ScreamTracker 3) module loader / saver
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "S3MTools.h"
#ifndef MODPLUG_NO_FILESAVE
#include "../common/mptFileIO.h"
#ifdef MODPLUG_TRACKER
#include "../mptrack/TrackerSettings.h"
#endif // MODPLUG_TRACKER
#endif // MODPLUG_NO_FILESAVE
#include "../common/version.h"


OPENMPT_NAMESPACE_BEGIN


void CSoundFile::S3MConvert(ModCommand &m, bool fromIT)
{
	switch(m.command | 0x40)
	{
	case 'A':	m.command = CMD_SPEED; break;
	case 'B':	m.command = CMD_POSITIONJUMP; break;
	case 'C':	m.command = CMD_PATTERNBREAK; if (!fromIT) m.param = (m.param >> 4) * 10 + (m.param & 0x0F); break;
	case 'D':	m.command = CMD_VOLUMESLIDE; break;
	case 'E':	m.command = CMD_PORTAMENTODOWN; break;
	case 'F':	m.command = CMD_PORTAMENTOUP; break;
	case 'G':	m.command = CMD_TONEPORTAMENTO; break;
	case 'H':	m.command = CMD_VIBRATO; break;
	case 'I':	m.command = CMD_TREMOR; break;
	case 'J':	m.command = CMD_ARPEGGIO; break;
	case 'K':	m.command = CMD_VIBRATOVOL; break;
	case 'L':	m.command = CMD_TONEPORTAVOL; break;
	case 'M':	m.command = CMD_CHANNELVOLUME; break;
	case 'N':	m.command = CMD_CHANNELVOLSLIDE; break;
	case 'O':	m.command = CMD_OFFSET; break;
	case 'P':	m.command = CMD_PANNINGSLIDE; break;
	case 'Q':	m.command = CMD_RETRIG; break;
	case 'R':	m.command = CMD_TREMOLO; break;
	case 'S':	m.command = CMD_S3MCMDEX; break;
	case 'T':	m.command = CMD_TEMPO; break;
	case 'U':	m.command = CMD_FINEVIBRATO; break;
	case 'V':	m.command = CMD_GLOBALVOLUME; break;
	case 'W':	m.command = CMD_GLOBALVOLSLIDE; break;
	case 'X':	m.command = CMD_PANNING8; break;
	case 'Y':	m.command = CMD_PANBRELLO; break;
	case 'Z':	m.command = CMD_MIDI; break;
	case '\\':	m.command = static_cast<ModCommand::COMMAND>(fromIT ? CMD_SMOOTHMIDI : CMD_MIDI); break;
	// Chars under 0x40 don't save properly, so map : to ] and # to [.
	case ']':	m.command = CMD_DELAYCUT; break;
	case '[':	m.command = CMD_XPARAM; break;
	default:	m.command = CMD_NONE;
	}
}


void CSoundFile::S3MSaveConvert(uint8 &command, uint8 &param, bool toIT, bool compatibilityExport) const
{
	switch(command)
	{
	case CMD_SPEED:				command = 'A'; break;
	case CMD_POSITIONJUMP:		command = 'B'; break;
	case CMD_PATTERNBREAK:		command = 'C'; if(!toIT) param = ((param / 10) << 4) + (param % 10); break;
	case CMD_VOLUMESLIDE:		command = 'D'; break;
	case CMD_PORTAMENTODOWN:	command = 'E'; if (param >= 0xE0 && (GetType() & (MOD_TYPE_MOD | MOD_TYPE_XM))) param = 0xDF; break;
	case CMD_PORTAMENTOUP:		command = 'F'; if (param >= 0xE0 && (GetType() & (MOD_TYPE_MOD | MOD_TYPE_XM))) param = 0xDF; break;
	case CMD_TONEPORTAMENTO:	command = 'G'; break;
	case CMD_VIBRATO:			command = 'H'; break;
	case CMD_TREMOR:			command = 'I'; break;
	case CMD_ARPEGGIO:			command = 'J'; break;
	case CMD_VIBRATOVOL:		command = 'K'; break;
	case CMD_TONEPORTAVOL:		command = 'L'; break;
	case CMD_CHANNELVOLUME:		command = 'M'; break;
	case CMD_CHANNELVOLSLIDE:	command = 'N'; break;
	case CMD_OFFSET:			command = 'O'; break;
	case CMD_PANNINGSLIDE:		command = 'P'; break;
	case CMD_RETRIG:			command = 'Q'; break;
	case CMD_TREMOLO:			command = 'R'; break;
	case CMD_S3MCMDEX:			command = 'S'; break;
	case CMD_TEMPO:				command = 'T'; break;
	case CMD_FINEVIBRATO:		command = 'U'; break;
	case CMD_GLOBALVOLUME:		command = 'V'; break;
	case CMD_GLOBALVOLSLIDE:	command = 'W'; break;
	case CMD_PANNING8:
		command = 'X';
		if(toIT && !(GetType() & (MOD_TYPE_IT | MOD_TYPE_MPT | MOD_TYPE_XM | MOD_TYPE_MOD)))
		{
			if (param == 0xA4) { command = 'S'; param = 0x91; }
			else if (param == 0x80) { param = 0xFF; }
			else if (param < 0x80) { param <<= 1; }
			else command = 0;
		} else if (!toIT && (GetType() & (MOD_TYPE_IT | MOD_TYPE_MPT | MOD_TYPE_XM | MOD_TYPE_MOD)))
		{
			param >>= 1;
		}
		break;
	case CMD_PANBRELLO:			command = 'Y'; break;
	case CMD_MIDI:				command = 'Z'; break;
	case CMD_SMOOTHMIDI:
		if(compatibilityExport || !toIT)
			command = 'Z';
		else
			command = '\\';
		break;
	case CMD_XFINEPORTAUPDOWN:
		switch(param & 0xF0)
		{
		case 0x10:	command = 'F'; param = (param & 0x0F) | 0xE0; break;
		case 0x20:	command = 'E'; param = (param & 0x0F) | 0xE0; break;
		case 0x90:	command = 'S'; break;
		default:	command = 0;
		}
		break;
	case CMD_MODCMDEX:
		{
			ModCommand m;
			m.command = CMD_MODCMDEX;
			m.param = param;
			m.ExtendedMODtoS3MEffect();
			command = m.command;
			param = m.param;
			S3MSaveConvert(command, param, toIT, compatibilityExport);
		}
		return;
	// Chars under 0x40 don't save properly, so map : to ] and # to [.
	case CMD_DELAYCUT:
		if(compatibilityExport || !toIT)
			command = 0;
		else
			command = ']';
		break;
	case CMD_XPARAM:
		if(compatibilityExport || !toIT)
			command = 0;
		else
			command = '[';
		break;
	default:
		command = 0;
	}
	if(command == 0)
	{
		param = 0;
	}

	command &= ~0x40;
}


// Pattern decoding flags
enum S3MPattern
{
	s3mEndOfRow			= 0x00,
	s3mChannelMask		= 0x1F,
	s3mNotePresent		= 0x20,
	s3mVolumePresent	= 0x40,
	s3mEffectPresent	= 0x80,
	s3mAnyPresent		= 0xE0,

	s3mNoteOff			= 0xFE,
	s3mNoteNone			= 0xFF,
};


static bool ValidateHeader(const S3MFileHeader &fileHeader)
{
	if(std::memcmp(fileHeader.magic, "SCRM", 4)
		|| fileHeader.fileType != S3MFileHeader::idS3MType
		|| (fileHeader.formatVersion != S3MFileHeader::oldVersion && fileHeader.formatVersion != S3MFileHeader::newVersion)
		)
	{
		return false;
	}
	return true;
}


static uint64 GetHeaderMinimumAdditionalSize(const S3MFileHeader &fileHeader)
{
	return fileHeader.ordNum + (fileHeader.smpNum + fileHeader.patNum) * 2;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderS3M(MemoryFileReader file, const uint64 *pfilesize)
{
	S3MFileHeader fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return ProbeWantMoreData;
	}
	if(!ValidateHeader(fileHeader))
	{
		return ProbeFailure;
	}
	return ProbeAdditionalSize(file, pfilesize, GetHeaderMinimumAdditionalSize(fileHeader));
}


bool CSoundFile::ReadS3M(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();

	// Is it a valid S3M file?
	S3MFileHeader fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return false;
	}
	if(!ValidateHeader(fileHeader))
	{
		return false;
	}
	if(!file.CanRead(mpt::saturate_cast<FileReader::off_t>(GetHeaderMinimumAdditionalSize(fileHeader))))
	{
		return false;
	}
	if(loadFlags == onlyVerifyHeader)
	{
		return true;
	}

	InitializeGlobals(MOD_TYPE_S3M);
	m_nMinPeriod = 64;
	m_nMaxPeriod = 32767;

	// ST3 ignored Zxx commands, so if we find that a file was made with ST3, we should erase all MIDI macros.
	bool keepMidiMacros = false;

	mpt::ustring trackerStr;
	bool nonCompatTracker = false;
	bool isST3 = false;
	switch(fileHeader.cwtv & S3MFileHeader::trackerMask)
	{
	case S3MFileHeader::trkScreamTracker:
		if(fileHeader.cwtv == S3MFileHeader::trkST3_20 && fileHeader.special == 0 && (fileHeader.ordNum & 0x0F) == 0 && fileHeader.ultraClicks == 0 && (fileHeader.flags & ~0x50) == 0)
		{
			// MPT 1.16 and older versions of OpenMPT - Simply keep default (filter) MIDI macros
			m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 16, 00, 00);
			m_madeWithTracker = MPT_USTRING("ModPlug Tracker / OpenMPT");
			keepMidiMacros = true;
			nonCompatTracker = true;
			m_playBehaviour.set(kST3LimitPeriod);
		} else if(fileHeader.cwtv == S3MFileHeader::trkST3_20 && fileHeader.special == 0 && fileHeader.ultraClicks == 0 && fileHeader.flags == 0 && fileHeader.usePanningTable == 0)
		{
			m_madeWithTracker = MPT_USTRING("Velvet Studio");
		} else
		{
			trackerStr = MPT_USTRING("Scream Tracker");
			isST3 = true;
		}
		break;
	case S3MFileHeader::trkImagoOrpheus:
		trackerStr = MPT_USTRING("Imago Orpheus");
		nonCompatTracker = true;
		break;
	case S3MFileHeader::trkImpulseTracker:
		if(fileHeader.cwtv <= S3MFileHeader::trkIT2_14)
			trackerStr = MPT_USTRING("Impulse Tracker");
		else
			m_madeWithTracker = mpt::format(MPT_USTRING("Impulse Tracker 2.14p%1"))(fileHeader.cwtv - S3MFileHeader::trkIT2_14);
		nonCompatTracker = true;
		m_nMinPeriod = 1;
		break;
	case S3MFileHeader::trkSchismTracker:
		if(fileHeader.cwtv == S3MFileHeader::trkBeRoTrackerOld)
		{
			m_madeWithTracker = MPT_USTRING("BeRoTracker");
			m_playBehaviour.set(kST3LimitPeriod);
		} else
		{
			m_madeWithTracker = GetSchismTrackerVersion(fileHeader.cwtv);
			m_nMinPeriod = 1;
		}
		nonCompatTracker = true;
		break;
	case S3MFileHeader::trkOpenMPT:
		trackerStr = MPT_USTRING("OpenMPT");
		m_dwLastSavedWithVersion = (fileHeader.cwtv & S3MFileHeader::versionMask) << 16;
		break; 
	case S3MFileHeader::trkBeRoTracker:
		m_madeWithTracker = MPT_USTRING("BeRoTracker");
		m_playBehaviour.set(kST3LimitPeriod);
		break;
	case S3MFileHeader::trkCreamTracker:
		m_madeWithTracker = MPT_USTRING("CreamTracker");
		break;
	}
	if(!trackerStr.empty())
	{
		m_madeWithTracker = mpt::format(MPT_USTRING("%1 %2.%3"))(trackerStr, (fileHeader.cwtv & 0xF00) >> 8, mpt::ufmt::hex0<2>(fileHeader.cwtv & 0xFF));
	}
	if(nonCompatTracker)
	{
		m_playBehaviour.reset(kST3NoMutedChannels);
		m_playBehaviour.reset(kST3EffectMemory);
		m_playBehaviour.reset(kST3PortaSampleChange);
		m_playBehaviour.reset(kST3VibratoMemory);
		m_playBehaviour.reset(KST3PortaAfterArpeggio);
	}

	if((fileHeader.cwtv & S3MFileHeader::trackerMask) > S3MFileHeader::trkScreamTracker)
	{
		// 2xyy - Imago Orpheus, 3xyy - IT, 4xyy - Schism, 5xyy - OpenMPT, 6xyy - BeRoTracker
		if((fileHeader.cwtv & S3MFileHeader::trackerMask) != S3MFileHeader::trkImpulseTracker || fileHeader.cwtv >= S3MFileHeader::trkIT2_14)
		{
			// Keep MIDI macros if this is not an old IT version (BABYLON.S3M by Necros has Zxx commands and was saved with IT 2.05)
			keepMidiMacros = true;
		}
	}

	m_MidiCfg.Reset();
	if(!keepMidiMacros)
	{
		// Remove macros so they don't interfere with tunes made in trackers that don't support Zxx
		m_MidiCfg.ClearZxxMacros();
	}

	mpt::String::Read<mpt::String::nullTerminated>(m_songName, fileHeader.name);

	if(fileHeader.flags & S3MFileHeader::amigaLimits) m_SongFlags.set(SONG_AMIGALIMITS);
	if(fileHeader.flags & S3MFileHeader::st2Vibrato) m_SongFlags.set(SONG_S3MOLDVIBRATO);

	if(fileHeader.cwtv < S3MFileHeader::trkST3_20 || (fileHeader.flags & S3MFileHeader::fastVolumeSlides) != 0)
	{
		m_SongFlags.set(SONG_FASTVOLSLIDES);
	}

	// Speed
	m_nDefaultSpeed = fileHeader.speed;
	if(m_nDefaultSpeed == 0 || (m_nDefaultSpeed == 255 && isST3))
	{
		// Even though ST3 accepts the command AFF as expected, it mysteriously fails to load a default speed of 255...
		m_nDefaultSpeed = 6;
	}

	// Tempo
	m_nDefaultTempo.Set(fileHeader.tempo);
	if(fileHeader.tempo < 33)
	{
		// ST3 also fails to load an otherwise valid default tempo of 32...
		m_nDefaultTempo.Set(isST3 ? 125 : 32);
	}

	// Global Volume
	m_nDefaultGlobalVolume = std::min<uint32>(fileHeader.globalVol, 64) * 4u;
	// The following check is probably not very reliable, but it fixes a few tunes, e.g.
	// DARKNESS.S3M by Purple Motion (ST 3.00) and "Image of Variance" by C.C.Catch (ST 3.01):
	if(m_nDefaultGlobalVolume == 0 && fileHeader.cwtv < S3MFileHeader::trkST3_20)
	{
		m_nDefaultGlobalVolume = MAX_GLOBAL_VOLUME;
	}

	// Bit 8 = Stereo (we always use stereo)
	m_nSamplePreAmp = std::max(fileHeader.masterVolume & 0x7F, 0x10);

	// Channel setup
	m_nChannels = 4;
	for(CHANNELINDEX i = 0; i < 32; i++)
	{
		ChnSettings[i].Reset();

		if(fileHeader.channels[i] != 0xFF)
		{
			m_nChannels = i + 1;
			ChnSettings[i].nPan = (fileHeader.channels[i] & 8) ? 0xCC : 0x33;	// 200 : 56
		}
		if(fileHeader.channels[i] & 0x80)
		{
			ChnSettings[i].dwFlags = CHN_MUTE;
			// Detect Adlib channels here (except for OpenMPT 1.19 and older, which would write wrong channel types for PCM channels 16-32):
			// c = channels[i] ^ 0x80;
			// if(c >= 16 && c < 32) adlibChannel = true;
		}
	}
	if(m_nChannels < 1)
	{
		m_nChannels = 1;
	}

	ReadOrderFromFile<uint8>(Order(), file, fileHeader.ordNum, 0xFF, 0xFE);

	// Read sample header offsets
	std::vector<uint16le> sampleOffsets;
	file.ReadVector(sampleOffsets, fileHeader.smpNum);
	// Read pattern offsets
	std::vector<uint16le> patternOffsets;
	file.ReadVector(patternOffsets, fileHeader.patNum);

	// Read extended channel panning
	if(fileHeader.usePanningTable == S3MFileHeader::idPanning)
	{
		uint8 pan[32];
		file.ReadArray(pan);
		for(CHANNELINDEX i = 0; i < 32; i++)
		{
			if((pan[i] & 0x20) != 0)
			{
				ChnSettings[i].nPan = (static_cast<uint16>(pan[i] & 0x0F) * 256 + 8) / 15;
			}
		}
	}

	bool hasAdlibPatches = false;

	// Reading sample headers
	m_nSamples = std::min<SAMPLEINDEX>(fileHeader.smpNum, MAX_SAMPLES - 1);
	for(SAMPLEINDEX smp = 0; smp < m_nSamples; smp++)
	{
		S3MSampleHeader sampleHeader;

		if(!file.Seek(sampleOffsets[smp] * 16) || !file.ReadStruct(sampleHeader))
		{
			continue;
		}

		sampleHeader.ConvertToMPT(Samples[smp + 1]);
		mpt::String::Read<mpt::String::nullTerminated>(m_szNames[smp + 1], sampleHeader.name);

		if(sampleHeader.sampleType >= S3MSampleHeader::typeAdMel)
		{
			hasAdlibPatches = true;
		}

		const uint32 sampleOffset = (sampleHeader.dataPointer[1] << 4) | (sampleHeader.dataPointer[2] << 12) | (sampleHeader.dataPointer[0] << 20);

		if((loadFlags & loadSampleData) && sampleHeader.length != 0 && file.Seek(sampleOffset))
		{
			sampleHeader.GetSampleFormat((fileHeader.formatVersion == S3MFileHeader::oldVersion)).ReadSample(Samples[smp + 1], file);
		}
	}

	if(hasAdlibPatches)
	{
		AddToLog("This track uses Adlib instruments, which are not supported by OpenMPT.");
	}


	// Try to find out if Zxx commands are supposed to be panning commands (PixPlay).
	// Actually I am only aware of one module that uses this panning style, namely "Crawling Despair" by $volkraq
	// and I have no idea what PixPlay is, so this code is solely based on the sample text of that module.
	// We won't convert if there are not enough Zxx commands, too "high" Zxx commands
	// or there are only "left" or "right" pannings (we assume that stereo should be somewhat balanced),
	// and modules not made with an old version of ST3 were probably made in a tracker that supports panning anyway.
	bool pixPlayPanning = (fileHeader.cwtv < S3MFileHeader::trkST3_20);
	int zxxCountRight = 0, zxxCountLeft = 0;

	// Reading patterns
	if(!(loadFlags & loadPatternData))
	{
		return true;
	}
	// Order list cannot contain pattern indices > 255, so do not even try to load higher patterns
	const PATTERNINDEX readPatterns = std::min<PATTERNINDEX>(fileHeader.patNum, uint8_max);
	Patterns.ResizeArray(readPatterns);
	for(PATTERNINDEX pat = 0; pat < readPatterns; pat++)
	{
		// A zero parapointer indicates an empty pattern.
		if(!Patterns.Insert(pat, 64) || patternOffsets[pat] == 0 || !file.Seek(patternOffsets[pat] * 16))
		{
			continue;
		}

		// Skip pattern length indication.
		// Some modules, for example http://aminet.net/mods/8voic/s3m_hunt.lha seem to have a wrong pattern length -
		// If you strictly adhere the pattern length, you won't read the patterns correctly in that module.
		file.Skip(2);

		// Read pattern data
		ROWINDEX row = 0;
		PatternRow rowBase = Patterns[pat].GetRow(0);

		while(row < 64)
		{
			uint8 info = file.ReadUint8();

			if(info == s3mEndOfRow)
			{
				// End of row
				if(++row < 64)
				{
					rowBase = Patterns[pat].GetRow(row);
				}
				continue;
			}

			CHANNELINDEX channel = (info & s3mChannelMask);
			ModCommand dummy;
			ModCommand &m = (channel < GetNumChannels()) ? rowBase[channel] : dummy;

			if(info & s3mNotePresent)
			{
				uint8 note = file.ReadUint8(), instr = file.ReadUint8();

				if(note < 0xF0)
				{
					// Note
					note = (note & 0x0F) + 12 * (note >> 4) + 12 + NOTE_MIN;
				} else if(note == s3mNoteOff)
				{
					// ^^
					note = NOTE_NOTECUT;
				} else if(note == s3mNoteNone)
				{
					// ..
					note = NOTE_NONE;
				}

				m.note = note;
				m.instr = instr;
			}

			if(info & s3mVolumePresent)
			{
				uint8 volume = file.ReadUint8();
				if(volume >= 128 && volume <= 192)
				{
					m.volcmd = VOLCMD_PANNING;
					m.vol = volume - 128;
				} else
				{
					m.volcmd = VOLCMD_VOLUME;
					m.vol = MIN(volume, 64);
				}
			}

			if(info & s3mEffectPresent)
			{
				uint8 command = file.ReadUint8(), param = file.ReadUint8();

				if(command != 0)
				{
					m.command = command;
					m.param = param;
					S3MConvert(m, false);
				}

				if(m.command == CMD_S3MCMDEX && (m.param & 0xF0) == 0xA0 && fileHeader.cwtv < S3MFileHeader::trkST3_20)
				{
					// Convert old SAx panning to S8x (should only be found in PANIC.S3M by Purple Motion)
					m.param = 0x80 | ((m.param & 0x0F) ^ 8);
				} else if(m.command == CMD_MIDI)
				{
					// PixPlay panning test
					if(m.param > 0x0F)
					{
						// PixPlay has Z00 to Z0F panning, so we ignore this.
						pixPlayPanning = false;
					} else
					{
						if(m.param < 0x08)
						{
							zxxCountLeft++;
						} else if(m.param > 0x08)
						{
							zxxCountRight++;
						}
					}
				}
			}
		}
	}

	if(pixPlayPanning && zxxCountLeft + zxxCountRight >= m_nChannels && (-zxxCountLeft + zxxCountRight) < static_cast<int>(m_nChannels))
	{
		// There are enough Zxx commands, so let's assume this was made to be played with PixPlay
		Patterns.ForEachModCommand([](ModCommand &m)
		{
			if(m.command == CMD_MIDI)
			{
				m.command = CMD_S3MCMDEX;
				m.param |= 0x80;
			}
		});
	}

	return true;
}


#ifndef MODPLUG_NO_FILESAVE

bool CSoundFile::SaveS3M(const mpt::PathString &filename) const
{
	static const uint8 filler[16] =
	{
		0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
		0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	};

	FILE *f;
	if(m_nChannels == 0 || filename.empty()) return false;
	if((f = mpt_fopen(filename, "wb")) == nullptr) return false;

	S3MFileHeader fileHeader;
	MemsetZero(fileHeader);

	mpt::String::Write<mpt::String::nullTerminated>(fileHeader.name, m_songName);
	fileHeader.dosEof = S3MFileHeader::idEOF;
	fileHeader.fileType = S3MFileHeader::idS3MType;

	// Orders
	ORDERINDEX writeOrders = Order().GetLengthTailTrimmed();
	if(writeOrders < 2)
	{
		writeOrders = 2;
	} else if((writeOrders % 2u) != 0)
	{
		// Number of orders should be even
		writeOrders++;
	}
	LimitMax(writeOrders, static_cast<ORDERINDEX>(256));
	fileHeader.ordNum = static_cast<uint16>(writeOrders);

	// Samples
	SAMPLEINDEX writeSamples = static_cast<SAMPLEINDEX>(GetNumInstruments());
	if(fileHeader.smpNum == 0)
	{
		writeSamples = GetNumSamples();
	}
	writeSamples = Clamp(writeSamples, static_cast<SAMPLEINDEX>(1), static_cast<SAMPLEINDEX>(99));
	fileHeader.smpNum = static_cast<uint16>(writeSamples);

	// Patterns
	PATTERNINDEX writePatterns = MIN(Patterns.GetNumPatterns(), 100u);
	fileHeader.patNum = static_cast<uint16>(writePatterns);

	// Flags
	if(m_SongFlags[SONG_FASTVOLSLIDES])
	{
		fileHeader.flags |= S3MFileHeader::fastVolumeSlides;
	}
	if(m_nMaxPeriod < 20000 || m_SongFlags[SONG_AMIGALIMITS])
	{
		fileHeader.flags |= S3MFileHeader::amigaLimits;
	}

	// Version info following: ST3.20 = 0x1320
	// Most significant nibble = Tracker ID, see S3MFileHeader::S3MTrackerVersions
	// Following: One nibble = Major version, one byte = Minor version (hex)
	fileHeader.cwtv = S3MFileHeader::trkOpenMPT | static_cast<uint16>((MptVersion::num >> 16) & S3MFileHeader::versionMask);
	fileHeader.formatVersion = S3MFileHeader::newVersion;
	memcpy(fileHeader.magic, "SCRM", 4);

	// Song Variables
	fileHeader.globalVol = static_cast<uint8>(std::min(m_nDefaultGlobalVolume / 4u, 64u));
	fileHeader.speed = static_cast<uint8>(Clamp(m_nDefaultSpeed, 1u, 254u));
	fileHeader.tempo = static_cast<uint8>(Clamp(m_nDefaultTempo.GetInt(), 33u, 255u));
	fileHeader.masterVolume = static_cast<uint8>(Clamp(m_nSamplePreAmp, 16u, 127u) | 0x80);
	fileHeader.ultraClicks = 8;
	fileHeader.usePanningTable = S3MFileHeader::idPanning;

	// Channel Table
	const uint8 midCh = static_cast<uint8>(std::min(GetNumChannels() / 2, 8));
	for(CHANNELINDEX chn = 0; chn < 32; chn++)
	{
		if(chn < GetNumChannels())
		{
			// ST3 only supports 16 PCM channels, so if channels 17-32 are used,
			// they must be mapped to the same "internal channels" as channels 1-16.
			// The channel indices determine in which order channels are evaluated in ST3.
			// First, the "left" channels (0...7) are evaluated, then the "right" channels (8...15).
			// Previously, an alternating LRLR scheme was written, which would lead to a different
			// effect processing in ST3 than LLL...RRR, but since OpenMPT doesn't care about the
			// channel order and always parses them left to right as they appear in the pattern,
			// we should just write in the LLL...RRR manner.
			uint8 ch = chn & 0x0F;
			if(ch >= midCh)
			{
				ch += 8 - midCh;
			}
#ifdef MODPLUG_TRACKER
			if(TrackerSettings::Instance().MiscSaveChannelMuteStatus)
#endif
			if(ChnSettings[chn].dwFlags[CHN_MUTE])
			{
				ch |= 0x80;
			}
			fileHeader.channels[chn] = ch;
		} else
		{
			fileHeader.channels[chn] = 0xFF;
		}
	}

	fwrite(&fileHeader, sizeof(fileHeader), 1, f);
	Order().WriteAsByte(f, writeOrders);

	// Comment about parapointers stolen from Schism Tracker:
	// The sample data parapointers are 24+4 bits, whereas pattern data and sample headers are only 16+4
	// bits -- so while the sample data can be written up to 268 MB within the file (starting at 0xffffff0),
	// the pattern data and sample headers are restricted to the first 1 MB (starting at 0xffff0). In effect,
	// this practically requires the sample data to be written last in the file, as it is entirely possible
	// (and quite easy, even) to write more than 1 MB of sample data in a file.
	// The "practical standard order" listed in TECH.DOC is sample headers, patterns, then sample data.

	// Calculate offset of first sample header...
	size_t sampleHeaderOffset = ftell(f) + (writeSamples + writePatterns) * 2 + 32;
	// ...which must be a multiple of 16, because parapointers omit the lowest 4 bits.
	sampleHeaderOffset = (sampleHeaderOffset + 15) & ~15;

	std::vector<uint16le> sampleOffsets(writeSamples);
	for(SAMPLEINDEX smp = 0; smp < writeSamples; smp++)
	{
		STATIC_ASSERT((sizeof(S3MSampleHeader) % 16) == 0);
		sampleOffsets[smp] = static_cast<uint16>((sampleHeaderOffset + smp * sizeof(S3MSampleHeader)) / 16);
	}

	if(writeSamples != 0)
	{
		fwrite(sampleOffsets.data(), 2, writeSamples, f);
	}

	size_t patternPointerOffset = ftell(f);
	size_t firstPatternOffset = sampleHeaderOffset + writeSamples * sizeof(S3MSampleHeader);
	std::vector<uint16le> patternOffsets(writePatterns);

	// Need to calculate the real offsets later.
	if(writePatterns != 0)
	{
		fwrite(patternOffsets.data(), 2, writePatterns, f);
	}

	// Write channel panning
	uint8 chnPan[32];
	for(CHANNELINDEX chn = 0; chn < 32; chn++)
	{
		if(chn < GetNumChannels())
			chnPan[chn] = static_cast<uint8>(((ChnSettings[chn].nPan * 15 + 128) / 256)) | 0x20;
		else
			chnPan[chn] = 0x08;
	}
	fwrite(chnPan, 32, 1, f);

	// Do we need to fill up the file with some padding bytes for 16-Byte alignment?
	size_t curPos = ftell(f);
	if(curPos < sampleHeaderOffset)
	{
		MPT_ASSERT(sampleHeaderOffset - curPos < 16);
		fwrite(filler, sampleHeaderOffset - curPos, 1, f);
	}

	// Don't write sample headers for now, we are lacking the sample offset data.
	fseek(f, firstPatternOffset, SEEK_SET);

	// Write patterns
	for(PATTERNINDEX pat = 0; pat < writePatterns; pat++)
	{
		if(Patterns.IsPatternEmpty(pat))
		{
			patternOffsets[pat] = 0;
			continue;
		}

		long patOffset = ftell(f);
		if(patOffset > 0xFFFF0)
		{
			AddToLog(LogError, mpt::format(MPT_USTRING("Too much pattern data! Writing patterns failed starting from pattern %1."))(pat));
			break;
		}
		MPT_ASSERT((patOffset % 16) == 0);
		patternOffsets[pat] = static_cast<uint16>(patOffset / 16);

		std::vector<uint8> buffer;
		buffer.reserve(5 * 1024);
		// Reserve space for length bytes
		buffer.resize(2, 0);

		if(Patterns.IsValidPat(pat))
		{
			for(ROWINDEX row = 0; row < 64; row++)
			{
				if(row >= Patterns[pat].GetNumRows())
				{
					// Invent empty row
					buffer.push_back(s3mEndOfRow);
					continue;
				}

				const PatternRow rowBase = Patterns[pat].GetRow(row);

				CHANNELINDEX writeChannels = MIN(32, GetNumChannels());
				for(CHANNELINDEX chn = 0; chn < writeChannels; chn++)
				{
					const ModCommand &m = rowBase[chn];

					uint8 info = static_cast<uint8>(chn);
					uint8 note = m.note;
					ModCommand::VOLCMD volcmd = m.volcmd;
					uint8 vol = m.vol;
					uint8 command = m.command;
					uint8 param = m.param;

					if(note != NOTE_NONE || m.instr != 0)
					{
						info |= s3mNotePresent;

						if(note == NOTE_NONE)
						{
							// No Note, or note is too low
							note = s3mNoteNone;
						} else if(ModCommand::IsSpecialNote(note))
						{
							// Note Cut
							note = s3mNoteOff;
						} else if(note < 12 + NOTE_MIN)
						{
							// Too low
							note = 0;
						} else if(note <= NOTE_MAX)
						{
							note -= (12 + NOTE_MIN);
							note = (note % 12) + ((note / 12) << 4);
						}
					}

					if(command == CMD_VOLUME)
					{
						command = CMD_NONE;
						volcmd = VOLCMD_VOLUME;
						vol = MIN(param, 64);
					}

					if(volcmd == VOLCMD_VOLUME)
					{
						info |= s3mVolumePresent;
					} else if(volcmd == VOLCMD_PANNING)
					{
						info |= s3mVolumePresent;
						vol |= 0x80;
					}

					if(command != CMD_NONE)
					{
						S3MSaveConvert(command, param, false, true);
						if(command)
						{
							info |= s3mEffectPresent;
						}
					}

					if(info & s3mAnyPresent)
					{
						buffer.push_back(info);
						if(info & s3mNotePresent)
						{
							buffer.push_back(note);
							buffer.push_back(m.instr);
						}
						if(info & s3mVolumePresent)
						{
							buffer.push_back(vol);
						}
						if(info & s3mEffectPresent)
						{
							buffer.push_back(command);
							buffer.push_back(param);
						}
					}
				}

				buffer.push_back(s3mEndOfRow);
			}
		} else
		{
			// Invent empty pattern
			buffer.insert(buffer.end(), 64, s3mEndOfRow);
		}

		uint16 length = mpt::saturate_cast<uint16>(buffer.size());
		buffer[0] = static_cast<uint8>(length & 0xFF);
		buffer[1] = static_cast<uint8>((length >> 8) & 0xFF);

		if((buffer.size() % 16u) != 0)
		{
			// Add padding bytes
			buffer.insert(buffer.end(), 16 - (buffer.size() % 16u), 0);
		}

		fwrite(buffer.data(), buffer.size(), 1, f);
	}

	size_t sampleDataOffset = ftell(f);

	// Write samples
	std::vector<S3MSampleHeader> sampleHeader(writeSamples);

	for(SAMPLEINDEX smp = 0; smp < writeSamples; smp++)
	{
		SAMPLEINDEX realSmp = smp + 1;
		if(GetNumInstruments() != 0 && Instruments[smp] != nullptr)
		{
			// Find some valid sample associated with this instrument.
			for(size_t i = 0; i < CountOf(Instruments[smp]->Keyboard); i++)
			{
				if(Instruments[smp]->Keyboard[i] > 0 && Instruments[smp]->Keyboard[i] <= GetNumSamples())
				{
					realSmp = Instruments[smp]->Keyboard[i];
					break;
				}
			}
		}

		if(realSmp > GetNumSamples())
		{
			continue;
		}

		SmpLength smpLength = sampleHeader[smp].ConvertToS3M(Samples[realSmp]);

		mpt::String::Write<mpt::String::nullTerminated>(sampleHeader[smp].name, m_szNames[realSmp]);

		if(Samples[realSmp].pSample)
		{
			// Write sample data
			if(sampleDataOffset > 0xFFFFFF0)
			{
				AddToLog(LogError, mpt::format(MPT_USTRING("Too much sample data! Writing samples failed starting from sample %1."))(realSmp));
				break;
			}

			sampleHeader[smp].dataPointer[1] = static_cast<uint8>((sampleDataOffset >> 4) & 0xFF);
			sampleHeader[smp].dataPointer[2] = static_cast<uint8>((sampleDataOffset >> 12) & 0xFF);
			sampleHeader[smp].dataPointer[0] = static_cast<uint8>((sampleDataOffset >> 20) & 0xFF);

			size_t writtenLength = sampleHeader[smp].GetSampleFormat(false).WriteSample(f, Samples[realSmp], smpLength);
			sampleDataOffset += writtenLength;
			if((writtenLength % 16u) != 0)
			{
				size_t fillSize = 16 - (writtenLength % 16u);
				fwrite(filler, fillSize, 1, f);
				sampleDataOffset += fillSize;
			}
		}
	}

	// Now we know where the patterns are.
	if(writePatterns != 0)
	{
		fseek(f, patternPointerOffset, SEEK_SET);
		fwrite(patternOffsets.data(), 2, writePatterns, f);
	}

	// And we can finally write the sample headers.
	if(writeSamples != 0)
	{
		fseek(f, sampleHeaderOffset, SEEK_SET);
		fwrite(sampleHeader.data(), sizeof(sampleHeader[0]), writeSamples, f);
	}

	fclose(f);
	return true;
}

#endif // MODPLUG_NO_FILESAVE


OPENMPT_NAMESPACE_END
