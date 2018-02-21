/*
 * Load_itp.cpp
 * ------------
 * Purpose: Impulse Tracker Project (ITP) module loader
 * Notes  : Despite its name, ITP is not a format supported by Impulse Tracker.
 *          In fact, it's a format invented by the OpenMPT team to allow people to work
 *          with the IT format, but keeping the instrument files with big samples separate
 *          from the pattern data, to keep the work files small and handy.
 *          The design of the format is quite flawed, though, so it was superseded by
 *          extra functionality in the MPTM format in OpenMPT 1.24.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "../common/version.h"
#include "Loaders.h"
#include "ITTools.h"
#ifdef MODPLUG_TRACKER
// For loading external instruments
#include "../mptrack/Moddoc.h"
#endif // MODPLUG_TRACKER
#ifdef MPT_EXTERNAL_SAMPLES
#include "../common/mptFileIO.h"
#endif // MPT_EXTERNAL_SAMPLES

OPENMPT_NAMESPACE_BEGIN

// Version changelog:
// v1.03: - Relative unicode instrument paths instead of absolute ANSI paths
//        - Per-path variable string length
//        - Embedded samples are IT-compressed
//        (rev. 3249)
// v1.02: Explicitely updated format to use new instrument flags representation (rev. 483)
// v1.01: Added option to embed instrument headers


struct ITPModCommand
{
	uint8le note;
	uint8le instr;
	uint8le volcmd;
	uint8le command;
	uint8le vol;
	uint8le param;
	operator ModCommand() const
	{
		ModCommand result;
		result.note = (ModCommand::IsNote(note) || ModCommand::IsSpecialNote(note)) ? note : NOTE_NONE;
		result.instr = instr;
		result.command = (command < MAX_EFFECTS) ? static_cast<EffectCommand>(command.get()) : CMD_NONE;
		result.volcmd = (volcmd < MAX_VOLCMDS) ? static_cast<VolumeCommand>(volcmd.get()) : VOLCMD_NONE;
		result.vol = vol;
		result.param = param;
		return result;
	}
};

MPT_BINARY_STRUCT(ITPModCommand, 6)


struct ITPHeader
{
	uint32le magic;
	uint32le version;
};

MPT_BINARY_STRUCT(ITPHeader, 8)


static bool ValidateHeader(const ITPHeader &hdr)
{
	if(hdr.magic != MAGIC4BE('.','i','t','p'))
	{
		return false;
	}
	if(hdr.version > 0x00000103 || hdr.version < 0x00000100)
	{
		return false;
	}
	return true;
}


static uint64 GetHeaderMinimumAdditionalSize(const ITPHeader &hdr)
{
	MPT_UNREFERENCED_PARAMETER(hdr);
	return 12 + 4 + 24 + 4 - sizeof(ITPHeader);
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderITP(MemoryFileReader file, const uint64 *pfilesize)
{
	ITPHeader hdr;
	if(!file.ReadStruct(hdr))
	{
		return ProbeWantMoreData;
	}
	if(!ValidateHeader(hdr))
	{
		return ProbeFailure;
	}
	return ProbeAdditionalSize(file, pfilesize, GetHeaderMinimumAdditionalSize(hdr));
}


bool CSoundFile::ReadITProject(FileReader &file, ModLoadingFlags loadFlags)
{
#if !defined(MPT_EXTERNAL_SAMPLES) && !defined(MPT_FUZZ_TRACKER)
	// Doesn't really make sense to support this format when there's no support for external files...
	MPT_UNREFERENCED_PARAMETER(file);
	MPT_UNREFERENCED_PARAMETER(loadFlags);
	return false;
#else // !MPT_EXTERNAL_SAMPLES && !MPT_FUZZ_TRACKER

	enum ITPSongFlags
	{
		ITP_EMBEDMIDICFG	= 0x00001,	// Embed macros in file
		ITP_ITOLDEFFECTS	= 0x00004,	// Old Impulse Tracker effect implementations
		ITP_ITCOMPATGXX		= 0x00008,	// IT "Compatible Gxx" (IT's flag to behave more like other trackers w/r/t portamento effects)
		ITP_LINEARSLIDES	= 0x00010,	// Linear slides vs. Amiga slides
		ITP_EXFILTERRANGE	= 0x08000,	// Cutoff Filter has double frequency range (up to ~10Khz)
		ITP_ITPROJECT		= 0x20000,	// Is a project file
		ITP_ITPEMBEDIH		= 0x40000,	// Embed instrument headers in project file
	};

	file.Rewind();

	ITPHeader hdr;
	if(!file.ReadStruct(hdr))
	{
		return false;
	}
	if(!ValidateHeader(hdr))
	{
		return false;
	}
	if(!file.CanRead(mpt::saturate_cast<FileReader::off_t>(GetHeaderMinimumAdditionalSize(hdr))))
	{
		return false;
	}
	if(loadFlags == onlyVerifyHeader)
	{
		return true;
	}

	uint32 version, size;
	version = hdr.version;

	InitializeGlobals(MOD_TYPE_IT);
	m_playBehaviour.reset();
	file.ReadSizedString<uint32le, mpt::String::maybeNullTerminated>(m_songName);

	// Song comments
	m_songMessage.Read(file, file.ReadUint32LE(), SongMessage::leCR);

	// Song global config
	const uint32 songFlags = file.ReadUint32LE();
	if(!(songFlags & ITP_ITPROJECT))
	{
		return false;
	}
	if(songFlags & ITP_ITOLDEFFECTS)	m_SongFlags.set(SONG_ITOLDEFFECTS);
	if(songFlags & ITP_ITCOMPATGXX)		m_SongFlags.set(SONG_ITCOMPATGXX);
	if(songFlags & ITP_LINEARSLIDES)	m_SongFlags.set(SONG_LINEARSLIDES);
	if(songFlags & ITP_EXFILTERRANGE)	m_SongFlags.set(SONG_EXFILTERRANGE);

	m_nDefaultGlobalVolume = file.ReadUint32LE();
	m_nSamplePreAmp = file.ReadUint32LE();
	m_nDefaultSpeed = std::max(uint32(1), file.ReadUint32LE());
	m_nDefaultTempo.Set(std::max(uint32(32), file.ReadUint32LE()));
	m_nChannels = static_cast<CHANNELINDEX>(file.ReadUint32LE());
	if(m_nChannels == 0 || m_nChannels > MAX_BASECHANNELS)
	{
		return false;
	}

	// channel name string length (=MAX_CHANNELNAME)
	size = file.ReadUint32LE();

	// Channels' data
	for(CHANNELINDEX chn = 0; chn < m_nChannels; chn++)
	{
		ChnSettings[chn].nPan = std::min(static_cast<uint16>(file.ReadUint32LE()), uint16(256));
		ChnSettings[chn].dwFlags.reset();
		uint32 flags = file.ReadUint32LE();
		if(flags & 0x100) ChnSettings[chn].dwFlags.set(CHN_MUTE);
		if(flags & 0x800) ChnSettings[chn].dwFlags.set(CHN_SURROUND);
		ChnSettings[chn].nVolume = std::min(static_cast<uint16>(file.ReadUint32LE()), uint16(64));
		file.ReadString<mpt::String::maybeNullTerminated>(ChnSettings[chn].szName, size);
	}

	// Song mix plugins
	{
		FileReader plugChunk = file.ReadChunk(file.ReadUint32LE());
		LoadMixPlugins(plugChunk);
	}

	// MIDI Macro config
	file.ReadStructPartial<MIDIMacroConfigData>(m_MidiCfg, file.ReadUint32LE());
	m_MidiCfg.Sanitize();

	// Song Instruments
	m_nInstruments = static_cast<INSTRUMENTINDEX>(file.ReadUint32LE());
	if(m_nInstruments >= MAX_INSTRUMENTS)
	{
		m_nInstruments = 0;
		return false;
	}

	// Instruments' paths
	if(version <= 0x00000102)
	{
		size = file.ReadUint32LE();	// path string length
	}

	std::vector<mpt::PathString> instrPaths(GetNumInstruments());
	for(INSTRUMENTINDEX ins = 0; ins < GetNumInstruments(); ins++)
	{
		if(version > 0x00000102)
		{
			size = file.ReadUint32LE();	// path string length
		}
		std::string path;
		file.ReadString<mpt::String::maybeNullTerminated>(path, size);
#ifdef MODPLUG_TRACKER
		if(version <= 0x00000102)
		{
			instrPaths[ins] = mpt::PathString::FromLocaleSilent(path);
		} else
#endif // MODPLUG_TRACKER
		{
			instrPaths[ins] = mpt::PathString::FromUTF8(path);
		}
#ifdef MODPLUG_TRACKER
		if(!file.GetFileName().empty())
		{
			instrPaths[ins] = instrPaths[ins].RelativePathToAbsolute(file.GetFileName().GetPath());
		} else if(GetpModDoc() != nullptr)
		{
			instrPaths[ins] = instrPaths[ins].RelativePathToAbsolute(GetpModDoc()->GetPathNameMpt().GetPath());
		}
#endif // MODPLUG_TRACKER
	}

	// Song Orders
	size = file.ReadUint32LE();
	ReadOrderFromFile<uint8>(Order(), file, size, 0xFF, 0xFE);

	// Song Patterns
	const PATTERNINDEX numPats = static_cast<PATTERNINDEX>(file.ReadUint32LE());
	const PATTERNINDEX numNamedPats = static_cast<PATTERNINDEX>(file.ReadUint32LE());
	size_t patNameLen = file.ReadUint32LE();	// Size of each pattern name
	FileReader pattNames = file.ReadChunk(numNamedPats * patNameLen);

	// modcommand data length
	size = file.ReadUint32LE();
	if(size != 6)
	{
		return false;
	}

	if(loadFlags & loadPatternData)
		Patterns.ResizeArray(numPats);
	for(PATTERNINDEX pat = 0; pat < numPats; pat++)
	{
		const ROWINDEX numRows = file.ReadUint32LE();
		FileReader patternChunk = file.ReadChunk(numRows * size * GetNumChannels());

		// Allocate pattern
		if(!(loadFlags & loadPatternData) || !Patterns.Insert(pat, numRows))
		{
			pattNames.Skip(patNameLen);
			continue;
		}

		if(pat < numNamedPats)
		{
			char patName[32];
			pattNames.ReadString<mpt::String::maybeNullTerminated>(patName, patNameLen);
			Patterns[pat].SetName(patName);
		}

		// Pattern data
		size_t numCommands = GetNumChannels() * numRows;

		if(patternChunk.CanRead(sizeof(ITPModCommand) * numCommands))
		{
			ModCommand *target = Patterns[pat].GetpModCommand(0, 0);
			while(numCommands-- != 0)
			{
				ITPModCommand data;
				patternChunk.ReadStruct(data);
				*(target++) = data;
			}
		}
	}

	// Load embedded samples

	// Read original number of samples
	m_nSamples = static_cast<SAMPLEINDEX>(file.ReadUint32LE());
	LimitMax(m_nSamples, SAMPLEINDEX(MAX_SAMPLES - 1));

	// Read number of embedded samples - at most as many as there are real samples in a valid file
	uint32 embeddedSamples = file.ReadUint32LE();
	if(embeddedSamples > m_nSamples)
	{
		return false;
	}

	// Read samples
	for(uint32 smp = 0; smp < embeddedSamples && file.CanRead(8 + sizeof(ITSample)); smp++)
	{
		uint32 realSample = file.ReadUint32LE();
		ITSample sampleHeader;
		file.ReadStruct(sampleHeader);
		FileReader sampleData = file.ReadChunk(file.ReadUint32LE());

		if(realSample >= 1 && realSample <= GetNumSamples() && !memcmp(sampleHeader.id, "IMPS", 4) && (loadFlags & loadSampleData))
		{
			sampleHeader.ConvertToMPT(Samples[realSample]);
			mpt::String::Read<mpt::String::nullTerminated>(m_szNames[realSample], sampleHeader.name);

			// Read sample data
			sampleHeader.GetSampleFormat().ReadSample(Samples[realSample], sampleData);
		}
	}

	// Load instruments
	for(INSTRUMENTINDEX ins = 0; ins < GetNumInstruments(); ins++)
	{
		if(instrPaths[ins].empty())
			continue;

#ifdef MPT_EXTERNAL_SAMPLES
		InputFile f(instrPaths[ins]);
		FileReader instrFile = GetFileReader(f);
		if(!ReadInstrumentFromFile(ins + 1, instrFile, true))
		{
			AddToLog(LogWarning, MPT_USTRING("Unable to open instrument: ") + instrPaths[ins].ToUnicode());
		}
#else
		AddToLog(LogWarning, mpt::format(MPT_USTRING("Loading external instrument %1 ('%2') failed: External instruments are not supported."))(ins, instrPaths[ins].ToUnicode()));
#endif // MPT_EXTERNAL_SAMPLES
	}

	// Extra info data
	uint32 code = file.ReadUint32LE();

	// Embed instruments' header [v1.01]
	if(version >= 0x00000101 && (songFlags & ITP_ITPEMBEDIH) && code == MAGIC4BE('E', 'B', 'I', 'H'))
	{
		code = file.ReadUint32LE();

		INSTRUMENTINDEX ins = 1;
		while(ins <= GetNumInstruments() && file.CanRead(4))
		{
			if(code == MAGIC4BE('M', 'P', 'T', 'S'))
			{
				break;
			} else if(code == MAGIC4BE('S', 'E', 'P', '@') || code == MAGIC4BE('M', 'P', 'T', 'X'))
			{
				// jump code - switch to next instrument
				ins++;
			} else
			{
				ReadExtendedInstrumentProperty(Instruments[ins], code, file);
			}

			code = file.ReadUint32LE();
		}
	}

	// Song extensions
	if(code == MAGIC4BE('M', 'P', 'T', 'S'))
	{
		file.SkipBack(4);
		LoadExtendedSongProperties(file);
	}

	m_nMaxPeriod = 0xF000;
	m_nMinPeriod = 8;

	// Before OpenMPT 1.20.01.09, the MIDI macros were always read from the file, even if the "embed" flag was not set.
	if(m_dwLastSavedWithVersion >= MAKE_VERSION_NUMERIC(1,20,01,09) && !(songFlags & ITP_EMBEDMIDICFG))
	{
		m_MidiCfg.Reset();
	}

	m_madeWithTracker = MPT_USTRING("OpenMPT ") + MptVersion::ToUString(m_dwLastSavedWithVersion);

	return true;
#endif // MPT_EXTERNAL_SAMPLES
}

OPENMPT_NAMESPACE_END
