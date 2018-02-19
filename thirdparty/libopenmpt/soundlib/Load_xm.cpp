/*
 * Load_xm.cpp
 * -----------
 * Purpose: XM (FastTracker II) module loader / saver
 * Notes  : (currently none)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "../common/version.h"
#include "../common/misc_util.h"
#include "XMTools.h"
#ifndef MODPLUG_NO_FILESAVE
#include "../common/mptFileIO.h"
#endif
#include <algorithm>
#ifdef MODPLUG_TRACKER
#include "../mptrack/TrackerSettings.h"	// For super smooth ramping option
#endif // MODPLUG_TRACKER


OPENMPT_NAMESPACE_BEGIN


// Allocate samples for an instrument
static std::vector<SAMPLEINDEX> AllocateXMSamples(CSoundFile &sndFile, SAMPLEINDEX numSamples)
{
	LimitMax(numSamples, SAMPLEINDEX(32));

	std::vector<SAMPLEINDEX> foundSlots;
	foundSlots.reserve(numSamples);

	for(SAMPLEINDEX i = 0; i < numSamples; i++)
	{
		SAMPLEINDEX candidateSlot = sndFile.GetNumSamples() + 1;

		if(candidateSlot >= MAX_SAMPLES)
		{
			// If too many sample slots are needed, try to fill some empty slots first.
			for(SAMPLEINDEX j = 1; j <= sndFile.GetNumSamples(); j++)
			{
				if(sndFile.GetSample(j).pSample != nullptr)
				{
					continue;
				}

				if(std::find(foundSlots.begin(), foundSlots.end(), j) == foundSlots.end())
				{
					// Empty sample slot that is not occupied by the current instrument. Yay!
					candidateSlot = j;

					// Remove unused sample from instrument sample assignments
					for(INSTRUMENTINDEX ins = 1; ins <= sndFile.GetNumInstruments(); ins++)
					{
						if(sndFile.Instruments[ins] == nullptr)
						{
							continue;
						}
						for(auto &sample : sndFile.Instruments[ins]->Keyboard)
						{
							if(sample == candidateSlot)
							{
								sample = 0;
							}
						}
					}
					break;
				}
			}
		}

		if(candidateSlot >= MAX_SAMPLES)
		{
			// Still couldn't find any empty sample slots, so look out for existing but unused samples.
			std::vector<bool> usedSamples;
			SAMPLEINDEX unusedSampleCount = sndFile.DetectUnusedSamples(usedSamples);

			if(unusedSampleCount > 0)
			{
				sndFile.RemoveSelectedSamples(usedSamples);
				// Remove unused samples from instrument sample assignments
				for(INSTRUMENTINDEX ins = 1; ins <= sndFile.GetNumInstruments(); ins++)
				{
					if(sndFile.Instruments[ins] == nullptr)
					{
						continue;
					}
					for(auto &sample : sndFile.Instruments[ins]->Keyboard)
					{
						if(sample < usedSamples.size() && !usedSamples[sample])
						{
							sample = 0;
						}
					}
				}

				// New candidate slot is first unused sample slot.
				candidateSlot = static_cast<SAMPLEINDEX>(std::find(usedSamples.begin() + 1, usedSamples.end(), false) - usedSamples.begin());
			} else
			{
				// No unused sampel slots: Give up :(
				break;
			}
		}

		if(candidateSlot < MAX_SAMPLES)
		{
			foundSlots.push_back(candidateSlot);
			if(candidateSlot > sndFile.GetNumSamples())
			{
				sndFile.m_nSamples = candidateSlot;
			}
		}
	}

	return foundSlots;
}


// Read .XM patterns
static void ReadXMPatterns(FileReader &file, const XMFileHeader &fileHeader, CSoundFile &sndFile)
{
	// Reading patterns
	sndFile.Patterns.ResizeArray(fileHeader.patterns);
	for(PATTERNINDEX pat = 0; pat < fileHeader.patterns; pat++)
	{
		FileReader::off_t curPos = file.GetPosition();
		uint32 headerSize = file.ReadUint32LE();
		file.Skip(1);	// Pack method (= 0)

		ROWINDEX numRows = 64;

		if(fileHeader.version == 0x0102)
		{
			numRows = file.ReadUint8() + 1;
		} else
		{
			numRows = file.ReadUint16LE();
		}

		// A packed size of 0 indicates a completely empty pattern.
		const uint16 packedSize = file.ReadUint16LE();

		if(numRows == 0 || numRows > MAX_PATTERN_ROWS)
		{
			numRows = 64;
		}

		file.Seek(curPos + headerSize);
		FileReader patternChunk = file.ReadChunk(packedSize);

		if(!sndFile.Patterns.Insert(pat, numRows) || packedSize == 0)
		{
			continue;
		}

		enum PatternFlags
		{
			isPackByte		= 0x80,
			allFlags		= 0xFF,

			notePresent		= 0x01,
			instrPresent	= 0x02,
			volPresent		= 0x04,
			commandPresent	= 0x08,
			paramPresent	= 0x10,
		};

		for(auto &m : sndFile.Patterns[pat])
		{
			uint8 info = patternChunk.ReadUint8();

			uint8 vol = 0;
			if(info & isPackByte)
			{
				// Interpret byte as flag set.
				if(info & notePresent) m.note = patternChunk.ReadUint8();
			} else
			{
				// Interpret byte as note, read all other pattern fields as well.
				m.note = info;
				info = allFlags;
			}

			if(info & instrPresent) m.instr = patternChunk.ReadUint8();
			if(info & volPresent) vol = patternChunk.ReadUint8();
			if(info & commandPresent) m.command = patternChunk.ReadUint8();
			if(info & paramPresent) m.param = patternChunk.ReadUint8();

			if(m.note == 97)
			{
				m.note = NOTE_KEYOFF;
			} else if(m.note > 0 && m.note < 97)
			{
				m.note += 12;
			} else
			{
				m.note = NOTE_NONE;
			}

			if(m.command | m.param)
			{
				CSoundFile::ConvertModCommand(m);
			} else
			{
				m.command = CMD_NONE;
			}

			if(m.instr == 0xFF)
			{
				m.instr = 0;
			}

			if(vol >= 0x10 && vol <= 0x50)
			{
				m.volcmd = VOLCMD_VOLUME;
				m.vol = vol - 0x10;
			} else if (vol >= 0x60)
			{
				// Volume commands 6-F translation.
				static const ModCommand::VOLCMD volEffTrans[] =
				{
					VOLCMD_VOLSLIDEDOWN, VOLCMD_VOLSLIDEUP, VOLCMD_FINEVOLDOWN, VOLCMD_FINEVOLUP,
					VOLCMD_VIBRATOSPEED, VOLCMD_VIBRATODEPTH, VOLCMD_PANNING, VOLCMD_PANSLIDELEFT,
					VOLCMD_PANSLIDERIGHT, VOLCMD_TONEPORTAMENTO,
				};

				m.volcmd = volEffTrans[(vol - 0x60) >> 4];
				m.vol = vol & 0x0F;

				if(m.volcmd == VOLCMD_PANNING)
				{
					m.vol *= 4;	// FT2 does indeed not scale panning symmetrically.
				}
			}
		}
	}
}


enum TrackerVersions
{
	verUnknown		= 0x00,		// Probably not made with MPT
	verOldModPlug	= 0x01,		// Made with MPT Alpha / Beta
	verNewModPlug	= 0x02,		// Made with MPT (not Alpha / Beta)
	verModPlug1_09	= 0x04,		// Made with MPT 1.09 or possibly other version
	verOpenMPT		= 0x08,		// Made with OpenMPT
	verConfirmed	= 0x10,		// We are very sure that we found the correct tracker version.

	verFT2Generic	= 0x20,		// "FastTracker v2.00", but FastTracker has NOT been ruled out
	verOther		= 0x40,		// Something we don't know, testing for DigiTrakker.
	verFT2Clone		= 0x80,		// NOT FT2: itype changed between instruments, or \0 found in song title
	verDigiTrakker	= 0x100,	// Probably DigiTrakker
	verUNMO3		= 0x200,	// TODO: UNMO3-ed XMs are detected as MPT 1.16
	verEmptyOrders	= 0x400,	// Allow empty order list like in OpenMPT (FT2 just plays pattern 0 if the order list is empty according to the header)
};
DECLARE_FLAGSET(TrackerVersions)


static bool ValidateHeader(const XMFileHeader &fileHeader)
{
	if(fileHeader.channels == 0
		|| fileHeader.channels > MAX_BASECHANNELS
		|| std::memcmp(fileHeader.signature, "Extended Module: ", 17)
		)
	{
		return false;
	}
	return true;
}


static uint64 GetHeaderMinimumAdditionalSize(const XMFileHeader &fileHeader)
{
	return fileHeader.orders + 4 * (fileHeader.patterns + fileHeader.instruments);
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderXM(MemoryFileReader file, const uint64 *pfilesize)
{
	XMFileHeader fileHeader;
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


bool CSoundFile::ReadXM(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();

	XMFileHeader fileHeader;
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
	} else if(loadFlags == onlyVerifyHeader)
	{
		return true;
	}

	InitializeGlobals(MOD_TYPE_XM);
	InitializeChannels();
	m_nMixLevels = mixLevelsCompatible;

	FlagSet<TrackerVersions> madeWith(verUnknown);

	if(!memcmp(fileHeader.trackerName, "FastTracker ", 12))
	{
		if(fileHeader.size == 276 && !memcmp(fileHeader.trackerName + 12, "v2.00   ", 8))
		{
			if(fileHeader.version < 0x0104)
				madeWith = verFT2Generic | verConfirmed;
			else if(memchr(fileHeader.songName, '\0', 20) != nullptr)
				// FT2 pads the song title with spaces, some other trackers use null chars
				madeWith = verFT2Clone | verNewModPlug | verEmptyOrders;
			else
				madeWith = verFT2Generic | verNewModPlug;
		} else if(!memcmp(fileHeader.trackerName + 12, "v 2.00  ", 8))
		{
			// MPT 1.0 (exact version to be determined later)
			madeWith = verOldModPlug;
		} else
		{
			// ???
			madeWith.set(verConfirmed);
			m_madeWithTracker = MPT_USTRING("FastTracker Clone");
		}
	} else
	{
		// Something else!
		madeWith = verUnknown | verConfirmed;

		mpt::String::Read<mpt::String::spacePadded>(m_madeWithTracker, mpt::CharsetCP437, fileHeader.trackerName);

		if(!memcmp(fileHeader.trackerName, "OpenMPT ", 8))
		{
			madeWith = verOpenMPT | verConfirmed | verEmptyOrders;
		} else if(!memcmp(fileHeader.trackerName, "MilkyTracker ", 12))
		{
			// MilkyTracker prior to version 0.90.87 doesn't set a version string.
			// Luckily, starting with v0.90.87, MilkyTracker also implements the FT2 panning scheme.
			if(memcmp(fileHeader.trackerName + 12, "        ", 8))
			{
				m_nMixLevels = mixLevelsCompatibleFT2;
			}
		} else if(!memcmp(fileHeader.trackerName, "MadTracker 2.0\0", 15))
		{
			// Fix channel 2 in m3_cha.xm
			m_playBehaviour.reset(kFT2PortaNoNote);
			// Fix arpeggios in kragle_-_happy_day.xm
			m_playBehaviour.reset(kFT2Arpeggio);
		} else if(!memcmp(fileHeader.trackerName, "Skale Tracker\0", 14))
		{
			m_playBehaviour.reset(kFT2OffsetOutOfRange);
		} else if(!memcmp(fileHeader.trackerName, "*Converted ", 11))
		{
			madeWith = verDigiTrakker;
		}
	}

	mpt::String::Read<mpt::String::spacePadded>(m_songName, fileHeader.songName);

	m_nMinPeriod = 1;
	m_nMaxPeriod = 31999;

	Order().SetRestartPos(fileHeader.restartPos);
	m_nChannels = fileHeader.channels;
	m_nInstruments = std::min<uint16>(fileHeader.instruments, MAX_INSTRUMENTS - 1u);
	if(fileHeader.speed)
		m_nDefaultSpeed = fileHeader.speed;
	if(fileHeader.tempo)
		m_nDefaultTempo.Set(Clamp<uint16, uint16>(fileHeader.tempo, 32, 512));

	m_SongFlags.reset();
	m_SongFlags.set(SONG_LINEARSLIDES, (fileHeader.flags & XMFileHeader::linearSlides) != 0);
	m_SongFlags.set(SONG_EXFILTERRANGE, (fileHeader.flags & XMFileHeader::extendedFilterRange) != 0);
	if(m_SongFlags[SONG_EXFILTERRANGE] && madeWith == (verFT2Generic | verNewModPlug))
	{
		madeWith = verFT2Clone | verNewModPlug | verConfirmed;
	}

	ReadOrderFromFile<uint8>(Order(), file, fileHeader.orders);
	if(fileHeader.orders == 0 && !madeWith[verEmptyOrders])
	{
		// Fix lamb_-_dark_lighthouse.xm, which only contains one pattern and an empty order list
		Order().assign(1, 0);
	}
	file.Seek(fileHeader.size + 60);

	if(fileHeader.version >= 0x0104)
	{
		ReadXMPatterns(file, fileHeader, *this);
	}

	// In case of XM versions < 1.04, we need to memorize the sample flags for all samples, as they are not stored immediately after the sample headers.
	std::vector<SampleIO> sampleFlags;
	uint8 sampleReserved = 0;
	int instrType = -1;

	// Reading instruments
	for(INSTRUMENTINDEX instr = 1; instr <= m_nInstruments; instr++)
	{
		// First, try to read instrument header length...
		uint32 headerSize = file.ReadUint32LE();
		if(headerSize == 0)
		{
			headerSize = sizeof(XMInstrumentHeader);
		}

		// Now, read the complete struct.
		file.SkipBack(4);
		XMInstrumentHeader instrHeader;
		file.ReadStructPartial(instrHeader, headerSize);

		// Time for some version detection stuff.
		if(madeWith == verOldModPlug)
		{
			madeWith.set(verConfirmed);
			if(instrHeader.size == 245)
			{
				// ModPlug Tracker Alpha
				m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 00, 00, A5);
				m_madeWithTracker = MPT_USTRING("ModPlug Tracker 1.0 alpha");
			} else if(instrHeader.size == 263)
			{
				// ModPlug Tracker Beta (Beta 1 still behaves like Alpha, but Beta 3.3 does it this way)
				m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 00, 00, B3);
				m_madeWithTracker = MPT_USTRING("ModPlug Tracker 1.0 beta");
			} else
			{
				// WTF?
				madeWith = (verUnknown | verConfirmed);
			}
		} else if(instrHeader.numSamples == 0)
		{
			// Empty instruments make tracker identification pretty easy!
			if(instrHeader.size == 263 && instrHeader.sampleHeaderSize == 0 && madeWith[verNewModPlug])
				madeWith.set(verConfirmed);
			else if(instrHeader.size != 29 && madeWith[verDigiTrakker])
				madeWith.reset(verDigiTrakker);
			else if(madeWith[verFT2Clone | verFT2Generic] && instrHeader.size != 33)
			{
				// Sure isn't FT2.
				// Note: FT2 NORMALLY writes shdr=40 for all samples, but sometimes it
				// just happens to write random garbage there instead. Surprise!
				// Note: 4-mat's eternity.xm has an instrument header size of 29.
				madeWith = verUnknown;
			}
		}

		if(AllocateInstrument(instr) == nullptr)
		{
			continue;
		}

		instrHeader.ConvertToMPT(*Instruments[instr]);

		if(instrType == -1)
		{
			instrType = instrHeader.type;
		} else if(instrType != instrHeader.type && madeWith[verFT2Generic])
		{
			// FT2 writes some random junk for the instrument type field,
			// but it's always the SAME junk for every instrument saved.
			madeWith.reset(verFT2Generic);
			madeWith.set(verFT2Clone);
		}

		if(instrHeader.numSamples > 0)
		{
			// Yep, there are some samples associated with this instrument.
			if((instrHeader.instrument.midiEnabled | instrHeader.instrument.midiChannel | instrHeader.instrument.midiProgram | instrHeader.instrument.muteComputer) != 0)
			{
				// Definitely not an old MPT.
				madeWith.reset(verOldModPlug | verNewModPlug);
			}

			// Read sample headers
			std::vector<SAMPLEINDEX> sampleSlots = AllocateXMSamples(*this, instrHeader.numSamples);

			// Update sample assignment map
			for(size_t k = 0 + 12; k < 96 + 12; k++)
			{
				if(Instruments[instr]->Keyboard[k] < sampleSlots.size())
				{
					Instruments[instr]->Keyboard[k] = sampleSlots[Instruments[instr]->Keyboard[k]];
				}
			}

			if(fileHeader.version >= 0x0104)
			{
				sampleFlags.clear();
			}
			// Need to memorize those if we're going to skip any samples...
			std::vector<uint32> sampleSize(instrHeader.numSamples);

			// Early versions of Sk@le Tracker set instrHeader.sampleHeaderSize = 0 (IFULOVE.XM)
			// cybernostra weekend has instrHeader.sampleHeaderSize = 0x12, which would leave out the sample name, but FT2 still reads the name.
			MPT_ASSERT(instrHeader.sampleHeaderSize == 0 || instrHeader.sampleHeaderSize == sizeof(XMSample));

			for(SAMPLEINDEX sample = 0; sample < instrHeader.numSamples; sample++)
			{
				XMSample sampleHeader;
				file.ReadStruct(sampleHeader);

				sampleFlags.push_back(sampleHeader.GetSampleFormat());
				sampleSize[sample] = sampleHeader.length;
				sampleReserved |= sampleHeader.reserved;

				if(sample < sampleSlots.size())
				{
					SAMPLEINDEX mptSample = sampleSlots[sample];

					sampleHeader.ConvertToMPT(Samples[mptSample]);
					instrHeader.instrument.ApplyAutoVibratoToMPT(Samples[mptSample]);

					mpt::String::Read<mpt::String::spacePadded>(m_szNames[mptSample], sampleHeader.name);

					if((sampleHeader.flags & 3) == 3 && madeWith[verNewModPlug])
					{
						// MPT 1.09 and maybe newer / older versions set both loop flags for bidi loops.
						madeWith.set(verModPlug1_09);
					}
				}
			}

			// Read samples
			if(fileHeader.version >= 0x0104)
			{
				for(SAMPLEINDEX sample = 0; sample < instrHeader.numSamples; sample++)
				{
					// Sample 15 in dirtysex.xm by J/M/T/M is a 16-bit sample with an odd size of 0x18B according to the header, while the real sample size would be 0x18A.
					// Always read as many bytes as specified in the header, even if the sample reader would probably read less bytes.
					FileReader sampleChunk = file.ReadChunk(sampleFlags[sample].GetEncoding() != SampleIO::ADPCM ? sampleSize[sample] : (16 + (sampleSize[sample] + 1) / 2));
					if(sample < sampleSlots.size() && (loadFlags & loadSampleData))
					{
						sampleFlags[sample].ReadSample(Samples[sampleSlots[sample]], sampleChunk);
					}
				}
			}
		}
	}

	if(sampleReserved == 0 && madeWith[verNewModPlug] && memchr(fileHeader.songName, '\0', sizeof(fileHeader.songName)) != nullptr)
	{
		// Null-terminated song name: Quite possibly MPT. (could really be an MPT-made file resaved in FT2, though)
		madeWith.set(verConfirmed);
	}

	if(fileHeader.version < 0x0104)
	{
		// Load Patterns and Samples (Version 1.02 and 1.03)
		if(loadFlags & (loadPatternData | loadSampleData))
		{
			ReadXMPatterns(file, fileHeader, *this);
		}

		if(loadFlags & loadSampleData)
		{
			for(SAMPLEINDEX sample = 1; sample <= GetNumSamples(); sample++)
			{
				sampleFlags[sample - 1].ReadSample(Samples[sample], file);
			}
		}
	}

	// Read song comments: "text"
	if(file.ReadMagic("text"))
	{
		m_songMessage.Read(file, file.ReadUint32LE(), SongMessage::leCR);
		madeWith.set(verConfirmed);
	}
	
	// Read midi config: "MIDI"
	bool hasMidiConfig = false;
	if(file.ReadMagic("MIDI"))
	{
		file.ReadStructPartial<MIDIMacroConfigData>(m_MidiCfg, file.ReadUint32LE());
		m_MidiCfg.Sanitize();
		hasMidiConfig = true;
		madeWith.set(verConfirmed);
	}

	// Read pattern names: "PNAM"
	if(file.ReadMagic("PNAM"))
	{
		const PATTERNINDEX namedPats = std::min(static_cast<PATTERNINDEX>(file.ReadUint32LE() / MAX_PATTERNNAME), Patterns.Size());
		
		for(PATTERNINDEX pat = 0; pat < namedPats; pat++)
		{
			char patName[MAX_PATTERNNAME];
			file.ReadString<mpt::String::maybeNullTerminated>(patName, MAX_PATTERNNAME);
			Patterns[pat].SetName(patName);
		}
		madeWith.set(verConfirmed);
	}

	// Read channel names: "CNAM"
	if(file.ReadMagic("CNAM"))
	{
		const CHANNELINDEX namedChans = std::min(static_cast<CHANNELINDEX>(file.ReadUint32LE() / MAX_CHANNELNAME), GetNumChannels());
		for(CHANNELINDEX chn = 0; chn < namedChans; chn++)
		{
			file.ReadString<mpt::String::maybeNullTerminated>(ChnSettings[chn].szName, MAX_CHANNELNAME);
		}
		madeWith.set(verConfirmed);
	}

	// Read mix plugins information
	if(file.CanRead(8))
	{
		FileReader::off_t oldPos = file.GetPosition();
		LoadMixPlugins(file);
		if(file.GetPosition() != oldPos)
		{
			madeWith.set(verConfirmed);
		}
	}

	if(madeWith[verConfirmed])
	{
		if(madeWith[verModPlug1_09])
		{
			m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 09, 00, 00);
			m_madeWithTracker = MPT_USTRING("ModPlug Tracker 1.09");
		} else if(madeWith[verNewModPlug])
		{
			m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 16, 00, 00);
			m_madeWithTracker = MPT_USTRING("ModPlug Tracker 1.10 - 1.16");
		}
	}

	if(!memcmp(fileHeader.trackerName, "OpenMPT ", 8))
	{
		// Hey, I know this tracker!
		std::string mptVersion(fileHeader.trackerName + 8, 12);
		m_dwLastSavedWithVersion = MptVersion::ToNum(mptVersion);
		madeWith = verOpenMPT | verConfirmed;

		if(m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 22, 07, 19))
			m_nMixLevels = mixLevelsCompatible;
		else
			m_nMixLevels = mixLevelsCompatibleFT2;
	}

	if(m_dwLastSavedWithVersion != 0 && !madeWith[verOpenMPT])
	{
		m_nMixLevels = mixLevelsOriginal;
		m_playBehaviour.reset();
	}

	if(madeWith[verFT2Generic])
	{
		m_nMixLevels = mixLevelsCompatibleFT2;

		if(!hasMidiConfig)
		{
			// FT2 allows typing in arbitrary unsupported effect letters such as Zxx.
			// Prevent these commands from being interpreted as filter commands by erasing the default MIDI Config.
			m_MidiCfg.ClearZxxMacros();
		}

		if(fileHeader.version >= 0x0104	// Old versions of FT2 didn't have (smooth) ramping. Disable it for those versions where we can be sure that there should be no ramping.
#ifdef MODPLUG_TRACKER
			&& TrackerSettings::Instance().autoApplySmoothFT2Ramping
#endif // MODPLUG_TRACKER
			)
		{
			// apply FT2-style super-soft volume ramping
			m_playBehaviour.set(kFT2VolumeRamping);
		}
	}

	if(m_madeWithTracker.empty())
	{
		if(madeWith[verDigiTrakker] && sampleReserved == 0 && (instrType ? instrType : -1) == -1)
		{
			m_madeWithTracker = MPT_USTRING("DigiTrakker");
		} else if(madeWith[verFT2Generic])
		{
			m_madeWithTracker = MPT_USTRING("FastTracker 2 or compatible");
		} else
		{
			m_madeWithTracker = MPT_USTRING("Unknown");
		}
	}

	// Leave if no extra instrument settings are available (end of file reached)
	if(file.NoBytesLeft()) return true;

	bool interpretOpenMPTMade = false; // specific for OpenMPT 1.17+ (bMadeWithModPlug is also for MPT 1.16)
	if(GetNumInstruments())
	{
		LoadExtendedInstrumentProperties(file, &interpretOpenMPTMade);
	}

	LoadExtendedSongProperties(file, &interpretOpenMPTMade);

	if(interpretOpenMPTMade && m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 17, 00, 00))
	{
		// Up to OpenMPT 1.17.02.45 (r165), it was possible that the "last saved with" field was 0
		// when saving a file in OpenMPT for the first time.
		m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 17, 00, 00);
	}

	if(m_dwLastSavedWithVersion >= MAKE_VERSION_NUMERIC(1, 17, 00, 00))
	{
		m_madeWithTracker = MPT_USTRING("OpenMPT ") + MptVersion::ToUString(m_dwLastSavedWithVersion);
	}

	// We no longer allow any --- or +++ items in the order list now.
	if(m_dwLastSavedWithVersion && m_dwLastSavedWithVersion < MAKE_VERSION_NUMERIC(1, 22, 02, 02))
	{
		if(!Patterns.IsValidPat(0xFE))
			Order().RemovePattern(0xFE);
		if(!Patterns.IsValidPat(0xFF))
			Order().Replace(0xFF, Order.GetInvalidPatIndex());
	}

	return true;
}


#ifndef MODPLUG_NO_FILESAVE

#define str_tooMuchPatternData	("Warning: File format limit was reached. Some pattern data may not get written to file.")
#define str_pattern				("pattern")


bool CSoundFile::SaveXM(const mpt::PathString &filename, bool compatibilityExport)
{
	if(filename.empty())
	{
		return false;
	}
	FILE *f = mpt_fopen(filename, "wb");
	if(!f)
	{
		return false;
	}

	bool addChannel = false; // avoid odd channel count for FT2 compatibility

	XMFileHeader fileHeader;
	MemsetZero(fileHeader);

	memcpy(fileHeader.signature, "Extended Module: ", 17);
	mpt::String::Write<mpt::String::spacePadded>(fileHeader.songName, m_songName);
	fileHeader.eof = 0x1A;
	const std::string openMptTrackerName = MptVersion::GetOpenMPTVersionStr();
	mpt::String::Write<mpt::String::spacePadded>(fileHeader.trackerName, openMptTrackerName);

	// Writing song header
	fileHeader.version = 0x0104;					// XM Format v1.04
	fileHeader.size = sizeof(XMFileHeader) - 60;	// minus everything before this field
	fileHeader.restartPos = Order().GetRestartPos();

	fileHeader.channels = m_nChannels;
	if((m_nChannels % 2u) && m_nChannels < 32)
	{
		// Avoid odd channel count for FT2 compatibility
		fileHeader.channels++;
		addChannel = true;
	} else if(compatibilityExport && fileHeader.channels > 32)
	{
		fileHeader.channels = 32;
	}

	// Find out number of orders and patterns used.
	// +++ and --- patterns are not taken into consideration as FastTracker does not support them.
	
	const ORDERINDEX trimmedLength = Order().GetLengthTailTrimmed();
	std::vector<uint8> orderList(trimmedLength);
	const ORDERINDEX orderLimit = compatibilityExport ? 256 : uint16_max;
	ORDERINDEX numOrders = 0;
	PATTERNINDEX numPatterns = Patterns.GetNumPatterns();
	bool changeOrderList = false;
	for(ORDERINDEX ord = 0; ord < trimmedLength; ord++)
	{
		PATTERNINDEX pat = Order()[ord];
		if(pat == Order.GetIgnoreIndex() || pat == Order.GetInvalidPatIndex() || pat > uint8_max)
		{
			changeOrderList = true;
		} else if(numOrders < orderLimit)
		{
			orderList[numOrders++] = static_cast<uint8>(pat);
			if(pat >= numPatterns)
				numPatterns = pat + 1;
		}
	}
	if(changeOrderList)
	{
		AddToLog("Skip and stop order list items (+++ and ---) are not saved in XM files.");
	}
	orderList.resize(compatibilityExport ? 256 : numOrders);

	fileHeader.orders = numOrders;
	fileHeader.patterns = numPatterns;
	fileHeader.size += static_cast<uint32>(orderList.size());

	uint16 writeInstruments;
	if(m_nInstruments > 0)
		fileHeader.instruments = writeInstruments = m_nInstruments;
	else
		fileHeader.instruments = writeInstruments = m_nSamples;

	if(m_SongFlags[SONG_LINEARSLIDES]) fileHeader.flags |= XMFileHeader::linearSlides;
	if(m_SongFlags[SONG_EXFILTERRANGE] && !compatibilityExport) fileHeader.flags |= XMFileHeader::extendedFilterRange;
	fileHeader.flags = fileHeader.flags;

	// Fasttracker 2 will happily accept any tempo faster than 255 BPM. XMPlay does also support this, great!
	fileHeader.tempo = mpt::saturate_cast<uint16>(m_nDefaultTempo.GetInt());
	fileHeader.speed = static_cast<uint16>(Clamp(m_nDefaultSpeed, 1u, 31u));

	mpt::IO::Write(f, fileHeader);

	// Write processed order list
	mpt::IO::WriteRaw(f, orderList.data(), orderList.size());

	// Writing patterns

#define ASSERT_CAN_WRITE(x) \
	if(len > s.size() - x) /*Buffer running out? Make it larger.*/ \
		s.resize(s.size() + 10 * 1024, 0);
	std::vector<uint8> s(64 * 64 * 5, 0);

	for(PATTERNINDEX pat = 0; pat < numPatterns; pat++)
	{
		uint8 patHead[9] = { 0 };
		patHead[0] = 9;

		if(!Patterns.IsValidPat(pat))
		{
			// There's nothing to write... chicken out.
			patHead[5] = 64;
			mpt::IO::Write(f, patHead);
			continue;
		}

		const uint16 numRows = mpt::saturate_cast<uint16>(Patterns[pat].GetNumRows());
		patHead[5] = static_cast<uint8>(numRows & 0xFF);
		patHead[6] = static_cast<uint8>(numRows >> 8);

		auto p = Patterns[pat].cbegin();
		size_t len = 0;
		// Empty patterns are always loaded as 64-row patterns in FT2, regardless of their real size...
		bool emptyPattern = true;

		for(size_t j = m_nChannels * numRows; j > 0; j--, p++)
		{
			// Don't write more than 32 channels
			if(compatibilityExport && m_nChannels - ((j - 1) % m_nChannels) > 32) continue;

			uint8 note = p->note;
			uint8 command = p->command, param = p->param;
			ModSaveCommand(command, param, true, compatibilityExport);

			if (note >= NOTE_MIN_SPECIAL) note = 97; else
			if ((note <= 12) || (note > 96+12)) note = 0; else
			note -= 12;
			uint8 vol = 0;
			if (p->volcmd != VOLCMD_NONE)
			{
				switch(p->volcmd)
				{
				case VOLCMD_VOLUME:			vol = 0x10 + p->vol; break;
				case VOLCMD_VOLSLIDEDOWN:	vol = 0x60 + (p->vol & 0x0F); break;
				case VOLCMD_VOLSLIDEUP:		vol = 0x70 + (p->vol & 0x0F); break;
				case VOLCMD_FINEVOLDOWN:	vol = 0x80 + (p->vol & 0x0F); break;
				case VOLCMD_FINEVOLUP:		vol = 0x90 + (p->vol & 0x0F); break;
				case VOLCMD_VIBRATOSPEED:	vol = 0xA0 + (p->vol & 0x0F); break;
				case VOLCMD_VIBRATODEPTH:	vol = 0xB0 + (p->vol & 0x0F); break;
				case VOLCMD_PANNING:		vol = 0xC0 + (p->vol / 4); if (vol > 0xCF) vol = 0xCF; break;
				case VOLCMD_PANSLIDELEFT:	vol = 0xD0 + (p->vol & 0x0F); break;
				case VOLCMD_PANSLIDERIGHT:	vol = 0xE0 + (p->vol & 0x0F); break;
				case VOLCMD_TONEPORTAMENTO:	vol = 0xF0 + (p->vol & 0x0F); break;
				}
				// Those values are ignored in FT2. Don't save them, also to avoid possible problems with other trackers (or MPT itself)
				if(compatibilityExport && p->vol == 0)
				{
					switch(p->volcmd)
					{
					case VOLCMD_VOLUME:
					case VOLCMD_PANNING:
					case VOLCMD_VIBRATODEPTH:
					case VOLCMD_TONEPORTAMENTO:
					case VOLCMD_PANSLIDELEFT:	// Doesn't have memory, but does weird things with zero param.
						break;
					default:
						// no memory here.
						vol = 0;
					}
				}
			}

			// no need to fix non-empty patterns
			if(!p->IsEmpty())
				emptyPattern = false;

			// Apparently, completely empty patterns are loaded as empty 64-row patterns in FT2, regardless of their original size.
			// We have to avoid this, so we add a "break to row 0" command in the last row.
			if(j == 1 && emptyPattern && numRows != 64)
			{
				command = 0x0D;
				param = 0;
			}

			if ((note) && (p->instr) && (vol > 0x0F) && (command) && (param))
			{
				s[len++] = note;
				s[len++] = p->instr;
				s[len++] = vol;
				s[len++] = command;
				s[len++] = param;
			} else
			{
				uint8 b = 0x80;
				if (note) b |= 0x01;
				if (p->instr) b |= 0x02;
				if (vol >= 0x10) b |= 0x04;
				if (command) b |= 0x08;
				if (param) b |= 0x10;
				s[len++] = b;
				if (b & 1) s[len++] = note;
				if (b & 2) s[len++] = p->instr;
				if (b & 4) s[len++] = vol;
				if (b & 8) s[len++] = command;
				if (b & 16) s[len++] = param;
			}

			if(addChannel && (j % m_nChannels == 1 || m_nChannels == 1))
			{
				ASSERT_CAN_WRITE(1);
				s[len++] = 0x80;
			}

			ASSERT_CAN_WRITE(5);
		}

		if(emptyPattern && numRows == 64)
		{
			// Be smart when saving empty patterns!
			len = 0;
		}

		// Reaching the limits of file format?
		if(len > uint16_max)
		{
			AddToLog(mpt::format("%1 (%2 %3)")(str_tooMuchPatternData, str_pattern, pat));
			len = uint16_max;
		}

		patHead[7] = static_cast<uint8>(len & 0xFF);
		patHead[8] = static_cast<uint8>(len >> 8);
		mpt::IO::Write(f, patHead);
		if(len) mpt::IO::WriteRaw(f, s.data(), len);
	}

#undef ASSERT_CAN_WRITE

	// Check which samples are referenced by which instruments (for assigning unreferenced samples to instruments)
	std::vector<bool> sampleAssigned(GetNumSamples() + 1, false);
	for(INSTRUMENTINDEX ins = 1; ins <= GetNumInstruments(); ins++)
	{
		if(Instruments[ins] != nullptr)
		{
			Instruments[ins]->GetSamples(sampleAssigned);
		}
	}

	// Writing instruments
	for(INSTRUMENTINDEX ins = 1; ins <= writeInstruments; ins++)
	{
		XMInstrumentHeader insHeader;
		std::vector<SAMPLEINDEX> samples;

		if(GetNumInstruments())
		{
			if(Instruments[ins] != nullptr)
			{
				// Convert instrument
				insHeader.ConvertToXM(*Instruments[ins], compatibilityExport);

				samples = insHeader.instrument.GetSampleList(*Instruments[ins], compatibilityExport);
				if(samples.size() > 0 && samples[0] <= GetNumSamples())
				{
					// Copy over auto-vibrato settings of first sample
					insHeader.instrument.ApplyAutoVibratoToXM(Samples[samples[0]], GetType());
				}

				std::vector<SAMPLEINDEX> additionalSamples;

				// Try to save "instrument-less" samples as well by adding those after the "normal" samples of our sample.
				// We look for unassigned samples directly after the samples assigned to our current instrument, so if
				// e.g. sample 1 is assigned to instrument 1 and samples 2 to 10 aren't assigned to any instrument,
				// we will assign those to sample 1. Any samples before the first referenced sample are going to be lost,
				// but hey, I wrote this mostly for preserving instrument texts in existing modules, where we shouldn't encounter this situation...
				for(auto smp : samples)
				{
					while(++smp <= GetNumSamples()
						&& !sampleAssigned[smp]
						&& insHeader.numSamples < (compatibilityExport ? 16 : 32))
					{
						sampleAssigned[smp] = true;			// Don't want to add this sample again.
						additionalSamples.push_back(smp);
						insHeader.numSamples++;
					}
				}

				samples.insert(samples.end(), additionalSamples.begin(), additionalSamples.end());
			} else
			{
				MemsetZero(insHeader);
			}
		} else
		{
			// Convert samples to instruments
			MemsetZero(insHeader);
			insHeader.numSamples = 1;
			insHeader.instrument.ApplyAutoVibratoToXM(Samples[ins], GetType());
			samples.push_back(ins);
		}

		insHeader.Finalise();
		size_t insHeaderSize = insHeader.size;
		mpt::IO::WritePartial(f, insHeader, insHeaderSize);

		std::vector<SampleIO> sampleFlags(samples.size());

		// Write Sample Headers
		for(SAMPLEINDEX smp = 0; smp < samples.size(); smp++)
		{
			XMSample xmSample;
			if(samples[smp] <= GetNumSamples())
			{
				xmSample.ConvertToXM(Samples[samples[smp]], GetType(), compatibilityExport);
			} else
			{
				MemsetZero(xmSample);
			}
			sampleFlags[smp] = xmSample.GetSampleFormat();

			mpt::String::Write<mpt::String::spacePadded>(xmSample.name, m_szNames[samples[smp]]);

			mpt::IO::Write(f, xmSample);
		}

		// Write Sample Data
		for(SAMPLEINDEX smp = 0; smp < samples.size(); smp++)
		{
			if(samples[smp] <= GetNumSamples())
			{
				sampleFlags[smp].WriteSample(f, Samples[samples[smp]]);
			}
		}
	}

	if(!compatibilityExport)
	{
		// Writing song comments
		if(!m_songMessage.empty())
		{
			uint32 size = mpt::saturate_cast<uint32>(m_songMessage.length());
			mpt::IO::WriteRaw(f, "text", 4);
			mpt::IO::WriteIntLE<uint32>(f, size);
			mpt::IO::WriteRaw(f, m_songMessage.c_str(), size);
		}
		// Writing midi cfg
		if(!m_MidiCfg.IsMacroDefaultSetupUsed())
		{
			mpt::IO::WriteRaw(f, "MIDI", 4);
			mpt::IO::WriteIntLE<uint32>(f, sizeof(MIDIMacroConfigData));
			mpt::IO::Write(f, static_cast<MIDIMacroConfigData &>(m_MidiCfg));
		}
		// Writing Pattern Names
		const PATTERNINDEX numNamedPats = Patterns.GetNumNamedPatterns();
		if(numNamedPats > 0)
		{
			mpt::IO::WriteRaw(f, "PNAM", 4);
			mpt::IO::WriteIntLE<uint32>(f, numNamedPats * MAX_PATTERNNAME);
			for(PATTERNINDEX pat = 0; pat < numNamedPats; pat++)
			{
				char name[MAX_PATTERNNAME];
				mpt::String::Write<mpt::String::maybeNullTerminated>(name, Patterns[pat].GetName());
				mpt::IO::Write(f, name);
			}
		}
		// Writing Channel Names
		{
			CHANNELINDEX numNamedChannels = 0;
			for(CHANNELINDEX chn = 0; chn < m_nChannels; chn++)
			{
				if (ChnSettings[chn].szName[0]) numNamedChannels = chn + 1;
			}
			// Do it!
			if(numNamedChannels)
			{
				mpt::IO::WriteRaw(f, "CNAM", 4);
				mpt::IO::WriteIntLE<uint32>(f, numNamedChannels * MAX_CHANNELNAME);
				for(CHANNELINDEX chn = 0; chn < numNamedChannels; chn++)
				{
					char name[MAX_CHANNELNAME];
					mpt::String::Write<mpt::String::maybeNullTerminated>(name, ChnSettings[chn].szName);
					mpt::IO::Write(f, name);
				}
			}
		}

		//Save hacked-on extra info
		SaveMixPlugins(f);
		if(GetNumInstruments())
		{
			SaveExtendedInstrumentProperties(writeInstruments, f);
		}
		SaveExtendedSongProperties(f);
	}

	fclose(f);
	return true;
}

#endif // MODPLUG_NO_FILESAVE


OPENMPT_NAMESPACE_END
