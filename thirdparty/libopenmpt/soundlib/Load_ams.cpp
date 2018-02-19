/*
 * Load_ams.cpp
 * ------------
 * Purpose: AMS (Extreme's Tracker / Velvet Studio) module loader
 * Notes  : Extreme was renamed to Velvet Development at some point,
 *          and thus they also renamed their tracker from
 *          "Extreme's Tracker" to "Velvet Studio".
 *          While the two programs look rather similiar, the structure of both
 *          programs' "AMS" format is significantly different in some places -
 *          Velvet Studio is a rather advanced tracker in comparison to Extreme's Tracker.
 *          The source code of Velvet Studio has been released into the
 *          public domain in 2013: https://github.com/Patosc/VelvetStudio/commits/master
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"


OPENMPT_NAMESPACE_BEGIN


// Read AMS or AMS2 (newVersion = true) pattern. At least this part of the format is more or less identical between the two trackers...
static void ReadAMSPattern(CPattern &pattern, bool newVersion, FileReader &patternChunk)
{
	enum
	{
		emptyRow		= 0xFF,	// No commands on row
		endOfRowMask	= 0x80,	// If set, no more commands on this row
		noteMask		= 0x40,	// If set, no note+instr in this command
		channelMask		= 0x1F,	// Mask for extracting channel

		// Note flags
		readNextCmd		= 0x80,	// One more command follows
		noteDataMask	= 0x7F,	// Extract note

		// Command flags
		volCommand		= 0x40,	// Effect is compressed volume command
		commandMask		= 0x3F,	// Command or volume mask
	};

	// Effect translation table for extended (non-Protracker) effects
	static const ModCommand::COMMAND effTrans[] =
	{
		CMD_S3MCMDEX,		// Forward / Backward
		CMD_PORTAMENTOUP,	// Extra fine slide up
		CMD_PORTAMENTODOWN,	// Extra fine slide up
		CMD_RETRIG,			// Retrigger
		CMD_NONE,
		CMD_TONEPORTAVOL,	// Toneporta with fine volume slide
		CMD_VIBRATOVOL,		// Vibrato with fine volume slide
		CMD_NONE,
		CMD_PANNINGSLIDE,
		CMD_NONE,
		CMD_VOLUMESLIDE,	// Two times finder volume slide than Axx
		CMD_NONE,
		CMD_CHANNELVOLUME,	// Channel volume (0...127)
		CMD_PATTERNBREAK,	// Long pattern break (in hex)
		CMD_S3MCMDEX,		// Fine slide commands
		CMD_NONE,			// Fractional BPM
		CMD_KEYOFF,			// Key off at tick xx
		CMD_PORTAMENTOUP,	// Porta up, but uses all octaves (?)
		CMD_PORTAMENTODOWN,	// Porta down, but uses all octaves (?)
		CMD_NONE,
		CMD_NONE,
		CMD_NONE,
		CMD_NONE,
		CMD_NONE,
		CMD_NONE,
		CMD_NONE,
		CMD_GLOBALVOLSLIDE,	// Global volume slide
		CMD_NONE,
		CMD_GLOBALVOLUME,	// Global volume (0... 127)
	};

	ModCommand dummy = ModCommand::Empty();

	for(ROWINDEX row = 0; row < pattern.GetNumRows(); row++)
	{
		PatternRow baseRow = pattern.GetRow(row);
		while(patternChunk.CanRead(1))
		{
			const uint8 flags = patternChunk.ReadUint8();
			if(flags == emptyRow)
			{
				break;
			}

			const CHANNELINDEX chn = (flags & channelMask);
			ModCommand &m = chn < pattern.GetNumChannels() ? baseRow[chn] : dummy;
			bool moreCommands = true;
			if(!(flags & noteMask))
			{
				// Read note + instr
				uint8 note = patternChunk.ReadUint8();
				moreCommands = (note & readNextCmd) != 0;
				note &= noteDataMask;

				if(note == 1)
				{
					m.note = NOTE_KEYOFF;
				} else if(note >= 2 && note <= 121 && newVersion)
				{
					m.note = note - 2 + NOTE_MIN;
				} else if(note >= 12 && note <= 108 && !newVersion)
				{
					m.note = note + 12 + NOTE_MIN;
				}
				m.instr = patternChunk.ReadUint8();
			}

			while(moreCommands)
			{
				// Read one more effect command
				ModCommand origCmd = m;
				const uint8 command = patternChunk.ReadUint8(), effect = (command & commandMask);
				moreCommands = (command & readNextCmd) != 0;

				if(command & volCommand)
				{
					m.volcmd = VOLCMD_VOLUME;
					m.vol = effect;
				} else
				{
					m.param = patternChunk.ReadUint8();

					if(effect < 0x10)
					{
						// PT commands
						m.command = effect;
						CSoundFile::ConvertModCommand(m);

						// Post-fix some commands
						switch(m.command)
						{
						case CMD_PANNING8:
							// 4-Bit panning
							m.command = CMD_PANNING8;
							m.param = (m.param & 0x0F) * 0x11;
							break;

						case CMD_VOLUME:
							m.command = CMD_NONE;
							m.volcmd = VOLCMD_VOLUME;
							m.vol = static_cast<ModCommand::VOL>(std::min((m.param + 1) / 2, 64));
							break;

						case CMD_MODCMDEX:
							if(m.param == 0x80)
							{
								// Break sample loop (cut after loop)
								m.command = CMD_NONE;
							} else
							{
								m.ExtendedMODtoS3MEffect();
							}
							break;
						}
					} else if(effect - 0x10 < (int)CountOf(effTrans))
					{
						// Extended commands
						m.command = effTrans[effect - 0x10];

						// Post-fix some commands
						switch(effect)
						{
						case 0x10:
							// Play sample forwards / backwards
							if(m.param <= 0x01)
							{
								m.param |= 0x9E;
							} else
							{
								m.command = CMD_NONE;
							}
							break;

						case 0x11:
						case 0x12:
							// Extra fine slides
							m.param = static_cast<ModCommand::PARAM>(std::min(uint8(0x0F), m.param) | 0xE0);
							break;

						case 0x15:
						case 0x16:
							// Fine slides
							m.param = static_cast<ModCommand::PARAM>((std::min(0x10, m.param + 1) / 2) | 0xF0);
							break;

						case 0x1E:
							// More fine slides
							switch(m.param >> 4)
							{
							case 0x1:
								// Fine porta up
								m.command = CMD_PORTAMENTOUP;
								m.param |= 0xF0;
								break;
							case 0x2:
								// Fine porta down
								m.command = CMD_PORTAMENTODOWN;
								m.param |= 0xF0;
								break;
							case 0xA:
								// Extra fine volume slide up
								m.command = CMD_VOLUMESLIDE;
								m.param = ((((m.param & 0x0F) + 1) / 2) << 4) | 0x0F;
								break;
							case 0xB:
								// Extra fine volume slide down
								m.command = CMD_VOLUMESLIDE;
								m.param = (((m.param & 0x0F) + 1) / 2) | 0xF0;
								break;
							default:
								m.command = CMD_NONE;
								break;
							}
							break;

						case 0x1C:
							// Adjust channel volume range
							m.param = static_cast<ModCommand::PARAM>(std::min((m.param + 1) / 2, 64));
							break;
						}
					}

					// Try merging commands first
					ModCommand::CombineEffects(m.command, m.param, origCmd.command, origCmd.param);

					if(ModCommand::GetEffectWeight(origCmd.command) > ModCommand::GetEffectWeight(m.command))
					{
						if(m.volcmd == VOLCMD_NONE && ModCommand::ConvertVolEffect(m.command, m.param, true))
						{
							// Volume column to the rescue!
							m.volcmd = m.command;
							m.vol = m.param;
						}

						m.command = origCmd.command;
						m.param = origCmd.param;
					}
				}
			}

			if(flags & endOfRowMask)
			{
				// End of row
				break;
			}
		}
	}
}


/////////////////////////////////////////////////////////////////////
// AMS (Extreme's Tracker) 1.x loader

// AMS File Header
struct AMSFileHeader
{
	uint8le  versionLow;
	uint8le  versionHigh;
	uint8le  channelConfig;
	uint8le  numSamps;
	uint16le numPats;
	uint16le numOrds;
	uint8le  midiChannels;
	uint16le extraSize;
};

MPT_BINARY_STRUCT(AMSFileHeader, 11)


// AMS Sample Header
struct AMSSampleHeader
{
	enum SampleFlags
	{
		smp16BitOld	= 0x04,	// AMS 1.0 (at least according to docs, I yet have to find such a file)
		smp16Bit	= 0x80,	// AMS 1.1+
		smpPacked	= 0x03,
	};

	uint32le length;
	uint32le loopStart;
	uint32le loopEnd;
	uint8le  panFinetune;	// High nibble = pan position, low nibble = finetune value
	uint16le sampleRate;
	uint8le  volume;		// 0...127
	uint8le  flags;			// See SampleFlags

	// Convert sample header to OpenMPT's internal format.
	void ConvertToMPT(ModSample &mptSmp) const
	{
		mptSmp.Initialize();

		mptSmp.nLength = length;
		mptSmp.nLoopStart = std::min(loopStart, length);
		mptSmp.nLoopEnd = std::min(loopEnd, length);

		mptSmp.nVolume = (std::min<uint8>(127, volume) * 256 + 64) / 127;
		if(panFinetune & 0xF0)
		{
			mptSmp.nPan = (panFinetune & 0xF0);
			mptSmp.uFlags = CHN_PANNING;
		}

		mptSmp.nC5Speed = 2 * sampleRate;
		if(sampleRate == 0)
		{
			mptSmp.nC5Speed = 2 * 8363;
		}

		uint32 newC4speed = ModSample::TransposeToFrequency(0, MOD2XMFineTune(panFinetune & 0x0F));
		mptSmp.nC5Speed = (mptSmp.nC5Speed * newC4speed) / 8363;

		if(mptSmp.nLoopStart < mptSmp.nLoopEnd)
		{
			mptSmp.uFlags.set(CHN_LOOP);
		}

		if(flags & (smp16Bit | smp16BitOld))
		{
			mptSmp.uFlags.set(CHN_16BIT);
		}
	}
};

MPT_BINARY_STRUCT(AMSSampleHeader, 17)


static bool ValidateHeader(const AMSFileHeader &fileHeader)
{
	if(fileHeader.versionHigh != 0x01)
	{
		return false;
	}
	return true;
}


static uint64 GetHeaderMinimumAdditionalSize(const AMSFileHeader &fileHeader)
{
	return fileHeader.extraSize + 3u + fileHeader.numSamps * (1u + sizeof(AMSSampleHeader)) + fileHeader.numOrds * 2u + fileHeader.numPats * 4u;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderAMS(MemoryFileReader file, const uint64 *pfilesize)
{
	if(!file.CanRead(7))
	{
		return ProbeWantMoreData;
	}
	if(!file.ReadMagic("Extreme"))
	{
		return ProbeFailure;
	}
	AMSFileHeader fileHeader;
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


bool CSoundFile::ReadAMS(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();

	if(!file.ReadMagic("Extreme"))
	{
		return false;
	}
	AMSFileHeader fileHeader;
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
	if(!file.Skip(fileHeader.extraSize))
	{
		return false;
	}
	if(loadFlags == onlyVerifyHeader)
	{
		return true;
	}

	InitializeGlobals(MOD_TYPE_AMS);

	m_SongFlags = SONG_ITCOMPATGXX | SONG_ITOLDEFFECTS;
	m_nChannels = (fileHeader.channelConfig & 0x1F) + 1;
	m_nSamples = fileHeader.numSamps;
	SetupMODPanning(true);
	m_madeWithTracker = mpt::format(MPT_USTRING("Extreme's Tracker %1.%2"))(fileHeader.versionHigh, fileHeader.versionLow);

	std::vector<bool> packSample(fileHeader.numSamps);

	STATIC_ASSERT(MAX_SAMPLES > 255);
	for(SAMPLEINDEX smp = 1; smp <= GetNumSamples(); smp++)
	{
		AMSSampleHeader sampleHeader;
		file.ReadStruct(sampleHeader);
		sampleHeader.ConvertToMPT(Samples[smp]);
		packSample[smp - 1] = (sampleHeader.flags & AMSSampleHeader::smpPacked) != 0;
	}

	// Texts
	file.ReadSizedString<uint8le, mpt::String::spacePadded>(m_songName);

	// Read sample names
	for(SAMPLEINDEX smp = 1; smp <= GetNumSamples(); smp++)
	{
		file.ReadSizedString<uint8le, mpt::String::spacePadded>(m_szNames[smp]);
	}

	// Read channel names
	for(CHANNELINDEX chn = 0; chn < GetNumChannels(); chn++)
	{
		ChnSettings[chn].Reset();
		file.ReadSizedString<uint8le, mpt::String::spacePadded>(ChnSettings[chn].szName);
	}

	// Read pattern names
	Patterns.ResizeArray(fileHeader.numPats);
	for(PATTERNINDEX pat = 0; pat < fileHeader.numPats; pat++)
	{
		char name[11];
		file.ReadSizedString<uint8le, mpt::String::spacePadded>(name);
		// Create pattern now, so name won't be reset later.
		if(Patterns.Insert(pat, 64))
		{
			Patterns[pat].SetName(name);
		}
	}

	// Read packed song message
	const uint16 packedLength = file.ReadUint16LE();
	if(packedLength && file.CanRead(packedLength))
	{
		std::vector<uint8> textIn;
		file.ReadVector(textIn, packedLength);
		std::string textOut;
		textOut.reserve(packedLength);

		for(auto c : textIn)
		{
			if(c & 0x80)
			{
				textOut.insert(textOut.end(), (c & 0x7F), ' ');
			} else
			{
				textOut.push_back(c);
			}
		}

		textOut = mpt::ToCharset(mpt::CharsetCP437, mpt::CharsetCP437AMS, textOut);

		// Packed text doesn't include any line breaks!
		m_songMessage.ReadFixedLineLength(mpt::byte_cast<const mpt::byte*>(textOut.c_str()), textOut.length(), 76, 0);
	}

	// Read Order List
	ReadOrderFromFile<uint16le>(Order(), file, fileHeader.numOrds);

	// Read patterns
	for(PATTERNINDEX pat = 0; pat < fileHeader.numPats; pat++)
	{
		uint32 patLength = file.ReadUint32LE();
		FileReader patternChunk = file.ReadChunk(patLength);

		if(loadFlags & loadPatternData)
		{
			ReadAMSPattern(Patterns[pat], false, patternChunk);
		}
	}

	if(loadFlags & loadSampleData)
	{
		// Read Samples
		for(SAMPLEINDEX smp = 1; smp <= GetNumSamples(); smp++)
		{
			SampleIO(
				Samples[smp].uFlags[CHN_16BIT] ? SampleIO::_16bit : SampleIO::_8bit,
				SampleIO::mono,
				SampleIO::littleEndian,
				packSample[smp - 1] ? SampleIO::AMS : SampleIO::signedPCM)
				.ReadSample(Samples[smp], file);
		}
	}

	return true;
}


/////////////////////////////////////////////////////////////////////
// AMS (Velvet Studio) 2.0 - 2.02 loader

// AMS2 File Header
struct AMS2FileHeader
{
	enum FileFlags
	{
		linearSlides	= 0x40,
	};

	uint8le  versionLow;		// Version of format (Hi = MainVer, Low = SubVer e.g. 0202 = 2.02)
	uint8le  versionHigh;		// ditto
	uint8le  numIns;			// Nr of Instruments (0-255)
	uint16le numPats;			// Nr of Patterns (1-1024)
	uint16le numOrds;			// Nr of Positions (1-65535)
	// Rest of header differs between format revision 2.01 and 2.02
};

MPT_BINARY_STRUCT(AMS2FileHeader, 7)


// AMS2 Instument Envelope
struct AMS2Envelope
{
	uint8 speed;		// Envelope speed (currently not supported, always the same as current BPM)
	uint8 sustainPoint;	// Envelope sustain point
	uint8 loopStart;	// Envelope loop Start
	uint8 loopEnd;		// Envelope loop End
	uint8 numPoints;	// Envelope length

	// Read envelope and do partial conversion.
	void ConvertToMPT(InstrumentEnvelope &mptEnv, FileReader &file)
	{
		file.ReadStruct(*this);

		// Read envelope points
		uint8 data[64][3];
		file.ReadStructPartial(data, numPoints * 3);

		if(numPoints <= 1)
		{
			// This is not an envelope.
			return;
		}

		STATIC_ASSERT(MAX_ENVPOINTS >= CountOf(data));
		mptEnv.resize(std::min(numPoints, uint8(CountOf(data))));
		mptEnv.nLoopStart = loopStart;
		mptEnv.nLoopEnd = loopEnd;
		mptEnv.nSustainStart = mptEnv.nSustainEnd = sustainPoint;

		for(uint32 i = 0; i < mptEnv.size(); i++)
		{
			if(i != 0)
			{
				mptEnv[i].tick = mptEnv[i - 1].tick + static_cast<uint16>(std::max(1, data[i][0] | ((data[i][1] & 0x01) << 8)));
			}
			mptEnv[i].value = data[i][2];
		}
	}
};

MPT_BINARY_STRUCT(AMS2Envelope, 5)


// AMS2 Instrument Data
struct AMS2Instrument
{
	enum EnvelopeFlags
	{
		envLoop		= 0x01,
		envSustain	= 0x02,
		envEnabled	= 0x04,
		
		// Flag shift amounts
		volEnvShift	= 0,
		panEnvShift	= 1,
		vibEnvShift	= 2,

		vibAmpMask	= 0x3000,
		vibAmpShift	= 12,
		fadeOutMask	= 0xFFF,
	};

	uint8le  shadowInstr;	// Shadow Instrument. If non-zero, the value=the shadowed inst.
	uint16le vibampFadeout;	// Vib.Amplify + Volume fadeout in one variable!
	uint16le envFlags;		// See EnvelopeFlags

	void ApplyFlags(InstrumentEnvelope &mptEnv, EnvelopeFlags shift) const
	{
		const int flags = envFlags >> (shift * 3);
		mptEnv.dwFlags.set(ENV_ENABLED, (flags & envEnabled) != 0);
		mptEnv.dwFlags.set(ENV_LOOP, (flags & envLoop) != 0);
		mptEnv.dwFlags.set(ENV_SUSTAIN, (flags & envSustain) != 0);

		// "Break envelope" should stop the envelope loop when encountering a note-off... We can only use the sustain loop to emulate this behaviour.
		if(!(flags & envSustain) && (flags & envLoop) != 0 && (flags & (1 << (9 - shift * 2))) != 0)
		{
			mptEnv.nSustainStart = mptEnv.nLoopStart;
			mptEnv.nSustainEnd = mptEnv.nLoopEnd;
			mptEnv.dwFlags.set(ENV_SUSTAIN);
			mptEnv.dwFlags.reset(ENV_LOOP);
		}
	}

};

MPT_BINARY_STRUCT(AMS2Instrument, 5)


// AMS2 Sample Header
struct AMS2SampleHeader
{
	enum SampleFlags
	{
		smpPacked	= 0x03,
		smp16Bit	= 0x04,
		smpLoop		= 0x08,
		smpBidiLoop	= 0x10,
		smpReverse	= 0x40,
	};

	uint32le length;
	uint32le loopStart;
	uint32le loopEnd;
	uint16le sampledRate;		// Whyyyy?
	uint8le  panFinetune;		// High nibble = pan position, low nibble = finetune value
	uint16le c4speed;			// Why is all of this so redundant?
	int8le   relativeTone;		// q.e.d.
	uint8le  volume;			// 0...127
	uint8le  flags;			// See SampleFlags

	// Convert sample header to OpenMPT's internal format.
	void ConvertToMPT(ModSample &mptSmp) const
	{
		mptSmp.Initialize();

		mptSmp.nLength = length;
		mptSmp.nLoopStart = std::min(loopStart, length);
		mptSmp.nLoopEnd = std::min(loopEnd, length);

		mptSmp.nC5Speed = c4speed * 2;
		if(c4speed == 0)
		{
			mptSmp.nC5Speed = 8363 * 2;
		}
		// Why, oh why, does this format need a c5speed and transpose/finetune at the same time...
		uint32 newC4speed = ModSample::TransposeToFrequency(relativeTone, MOD2XMFineTune(panFinetune & 0x0F));
		mptSmp.nC5Speed = (mptSmp.nC5Speed * newC4speed) / 8363;

		mptSmp.nVolume = (std::min<uint8>(volume, 127) * 256 + 64) / 127;
		if(panFinetune & 0xF0)
		{
			mptSmp.nPan = (panFinetune & 0xF0);
			mptSmp.uFlags = CHN_PANNING;
		}

		if(flags & smp16Bit) mptSmp.uFlags.set(CHN_16BIT);
		if((flags & smpLoop) && mptSmp.nLoopStart < mptSmp.nLoopEnd)
		{
			mptSmp.uFlags.set(CHN_LOOP);
			if(flags & smpBidiLoop) mptSmp.uFlags.set(CHN_PINGPONGLOOP);
			if(flags & smpReverse) mptSmp.uFlags.set(CHN_REVERSE);
		}
	}
};

MPT_BINARY_STRUCT(AMS2SampleHeader, 20)


// AMS2 Song Description Header
struct AMS2Description
{
	uint32le packedLen;		// Including header
	uint32le unpackedLen;
	uint8le  packRoutine;	// 01
	uint8le  preProcessing;	// None!
	uint8le  packingMethod;	// RLE
};

MPT_BINARY_STRUCT(AMS2Description, 11)


static bool ValidateHeader(const AMS2FileHeader &fileHeader)
{
	if(fileHeader.versionHigh != 2 || fileHeader.versionLow > 2)
	{
		return false;
	}
	return true;
}


static uint64 GetHeaderMinimumAdditionalSize(const AMS2FileHeader &fileHeader)
{
	return 36u + sizeof(AMS2Description) + fileHeader.numIns * 2u + fileHeader.numOrds * 2u + fileHeader.numPats * 4u;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderAMS2(MemoryFileReader file, const uint64 *pfilesize)
{
	if(!file.CanRead(7))
	{
		return ProbeWantMoreData;
	}
	if(!file.ReadMagic("AMShdr\x1A"))
	{
		return ProbeFailure;
	}
	if(!file.CanRead(1))
	{
		return ProbeWantMoreData;
	}
	const uint8 songNameLength = file.ReadUint8();
	if(!file.Skip(songNameLength))
	{
		return ProbeWantMoreData;
	}
	AMS2FileHeader fileHeader;
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


bool CSoundFile::ReadAMS2(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();

	if(!file.ReadMagic("AMShdr\x1A"))
	{
		return false;
	}
	std::string songName;
	if(!file.ReadSizedString<uint8le, mpt::String::spacePadded>(songName))
	{
		return false;
	}
	AMS2FileHeader fileHeader;
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
	
	InitializeGlobals(MOD_TYPE_AMS2);
	
	m_songName = songName;

	m_nInstruments = fileHeader.numIns;
	m_nChannels = 32;
	SetupMODPanning(true);
	m_madeWithTracker = mpt::format(MPT_USTRING("Velvet Studio %1.%2"))(fileHeader.versionHigh.get(), mpt::ufmt::dec0<2>(fileHeader.versionLow.get()));

	uint16 headerFlags;
	if(fileHeader.versionLow >= 2)
	{
		uint16 tempo = std::max(uint16(32 << 8), file.ReadUint16LE());	// 8.8 tempo
		m_nDefaultTempo.SetRaw((tempo * TEMPO::fractFact) >> 8);
		m_nDefaultSpeed = std::max(uint8(1), file.ReadUint8());
		file.Skip(3);	// Default values for pattern editor
		headerFlags = file.ReadUint16LE();
	} else
	{
		m_nDefaultTempo.Set(std::max(uint8(32), file.ReadUint8()));
		m_nDefaultSpeed = std::max(uint8(1), file.ReadUint8());
		headerFlags = file.ReadUint8();
	}

	m_SongFlags = SONG_ITCOMPATGXX | SONG_ITOLDEFFECTS | ((headerFlags & AMS2FileHeader::linearSlides) ? SONG_LINEARSLIDES : SongFlags(0));

	// Instruments
	std::vector<SAMPLEINDEX> firstSample;	// First sample of instrument
	std::vector<uint16> sampleSettings;		// Shadow sample map... Lo byte = Instrument, Hi byte, lo nibble = Sample index in instrument, Hi byte, hi nibble = Sample pack status
	enum
	{
		instrIndexMask		= 0xFF,		// Shadow instrument
		sampleIndexMask		= 0x7F00,	// Sample index in instrument
		sampleIndexShift	= 8,
		packStatusMask		= 0x8000,	// If bit is set, sample is packed
	};

	STATIC_ASSERT(MAX_INSTRUMENTS > 255);
	for(INSTRUMENTINDEX ins = 1; ins <= m_nInstruments; ins++)
	{
		ModInstrument *instrument = AllocateInstrument(ins);
		if(instrument == nullptr
			|| !file.ReadSizedString<uint8le, mpt::String::spacePadded>(instrument->name))
		{
			break;
		}

		uint8 numSamples = file.ReadUint8();
		uint8 sampleAssignment[120];
		MemsetZero(sampleAssignment);	// Only really needed for v2.0, where the lowest and highest octave aren't cleared.

		if(numSamples == 0
			|| (fileHeader.versionLow > 0 && !file.ReadArray(sampleAssignment))	// v2.01+: 120 Notes
			|| (fileHeader.versionLow == 0 && !file.ReadRaw(sampleAssignment + 12, 96)))	// v2.0: 96 Notes
		{
			continue;
		}

		STATIC_ASSERT(CountOf(instrument->Keyboard) >= CountOf(sampleAssignment));
		for(size_t i = 0; i < 120; i++)
		{
			instrument->Keyboard[i] = sampleAssignment[i] + GetNumSamples() + 1;
		}

		AMS2Envelope volEnv, panEnv, vibratoEnv;
		volEnv.ConvertToMPT(instrument->VolEnv, file);
		panEnv.ConvertToMPT(instrument->PanEnv, file);
		vibratoEnv.ConvertToMPT(instrument->PitchEnv, file);

		AMS2Instrument instrHeader;
		file.ReadStruct(instrHeader);
		instrument->nFadeOut = (instrHeader.vibampFadeout & AMS2Instrument::fadeOutMask);
		const int16 vibAmp = 1 << ((instrHeader.vibampFadeout & AMS2Instrument::vibAmpMask) >> AMS2Instrument::vibAmpShift);

		instrHeader.ApplyFlags(instrument->VolEnv, AMS2Instrument::volEnvShift);
		instrHeader.ApplyFlags(instrument->PanEnv, AMS2Instrument::panEnvShift);
		instrHeader.ApplyFlags(instrument->PitchEnv, AMS2Instrument::vibEnvShift);

		// Scale envelopes to correct range
		for(auto &p : instrument->VolEnv)
		{
			p.value = std::min(uint8(ENVELOPE_MAX), static_cast<uint8>((p.value * ENVELOPE_MAX + 64u) / 127u));
		}
		for(auto &p : instrument->PanEnv)
		{
			p.value = std::min(uint8(ENVELOPE_MAX), static_cast<uint8>((p.value * ENVELOPE_MAX + 128u) / 255u));
		}
		for(auto &p : instrument->PitchEnv)
		{
#ifdef MODPLUG_TRACKER
			p.value = std::min(uint8(ENVELOPE_MAX), static_cast<uint8>(32 + Util::muldivrfloor(static_cast<int8>(p.value - 128), vibAmp, 255)));
#else
			// Try to keep as much precision as possible... divide by 8 since that's the highest possible vibAmp factor.
			p.value = static_cast<uint8>(128 + Util::muldivrfloor(static_cast<int8>(p.value - 128), vibAmp, 8));
#endif
		}

		// Sample headers - we will have to read them even for shadow samples, and we will have to load them several times,
		// as it is possible that shadow samples use different sample settings like base frequency or panning.
		const SAMPLEINDEX firstSmp = GetNumSamples() + 1;
		for(SAMPLEINDEX smp = 0; smp < numSamples; smp++)
		{
			if(firstSmp + smp >= MAX_SAMPLES)
			{
				file.Skip(sizeof(AMS2SampleHeader));
				break;
			}
			file.ReadSizedString<uint8le, mpt::String::spacePadded>(m_szNames[firstSmp + smp]);

			AMS2SampleHeader sampleHeader;
			file.ReadStruct(sampleHeader);
			sampleHeader.ConvertToMPT(Samples[firstSmp + smp]);

			uint16 settings = (instrHeader.shadowInstr & instrIndexMask)
				| ((smp << sampleIndexShift) & sampleIndexMask)
				| ((sampleHeader.flags & AMS2SampleHeader::smpPacked) ? packStatusMask : 0);
			sampleSettings.push_back(settings);
		}

		firstSample.push_back(firstSmp);
		m_nSamples = static_cast<SAMPLEINDEX>(std::min(MAX_SAMPLES - 1, GetNumSamples() + numSamples));
	}

	// Text

	// Read composer name
	uint8 composerLength = file.ReadUint8();
	if(composerLength)
	{
		std::string str;
		file.ReadString<mpt::String::spacePadded>(str, composerLength);
		m_songArtist = mpt::ToUnicode(mpt::CharsetCP437AMS2, str);
	}

	// Channel names
	for(CHANNELINDEX chn = 0; chn < 32; chn++)
	{
		ChnSettings[chn].Reset();
		file.ReadSizedString<uint8le, mpt::String::spacePadded>(ChnSettings[chn].szName);
	}

	// RLE-Packed description text
	AMS2Description descriptionHeader;
	if(!file.ReadStruct(descriptionHeader))
	{
		return true;
	}
	if(descriptionHeader.packedLen > sizeof(descriptionHeader) && file.CanRead(descriptionHeader.packedLen - sizeof(descriptionHeader)))
	{
		const size_t textLength = descriptionHeader.packedLen - sizeof(descriptionHeader);
		std::vector<uint8> textIn;
		file.ReadVector(textIn, textLength);
		std::string textOut;
		textOut.reserve(descriptionHeader.unpackedLen);

		size_t readLen = 0;
		while(readLen < textLength)
		{
			uint8 c = textIn[readLen++];
			if(c == 0xFF && textLength - readLen >= 2)
			{
				c = textIn[readLen++];
				uint32 count = textIn[readLen++];
				textOut.insert(textOut.end(), count, c);
			} else
			{
				textOut.push_back(c);
			}
		}
		textOut = mpt::ToCharset(mpt::CharsetCP437, mpt::CharsetCP437AMS2, textOut);
		// Packed text doesn't include any line breaks!
		m_songMessage.ReadFixedLineLength(mpt::byte_cast<const mpt::byte*>(textOut.c_str()), textOut.length(), 74, 0);
	}

	// Read Order List
	ReadOrderFromFile<uint16le>(Order(), file, fileHeader.numOrds);

	// Read Patterns
	if(loadFlags & loadPatternData)
		Patterns.ResizeArray(fileHeader.numPats);
	for(PATTERNINDEX pat = 0; pat < fileHeader.numPats; pat++)
	{
		uint32 patLength = file.ReadUint32LE();
		FileReader patternChunk = file.ReadChunk(patLength);

		if(loadFlags & loadPatternData)
		{
			const ROWINDEX numRows = patternChunk.ReadUint8() + 1;
			// We don't need to know the number of channels or commands.
			patternChunk.Skip(1);

			if(!Patterns.Insert(pat, numRows))
			{
				continue;
			}

			char patternName[11];
			patternChunk.ReadSizedString<uint8le, mpt::String::spacePadded>(patternName);
			Patterns[pat].SetName(patternName);

			ReadAMSPattern(Patterns[pat], true, patternChunk);
		}
	}

	if(!(loadFlags & loadSampleData))
	{
		return true;
	}

	// Read Samples
	for(SAMPLEINDEX smp = 0; smp < GetNumSamples(); smp++)
	{
		if((sampleSettings[smp] & instrIndexMask) == 0)
		{
			// Only load samples that aren't part of a shadow instrument
			SampleIO(
				(Samples[smp + 1].uFlags & CHN_16BIT) ? SampleIO::_16bit : SampleIO::_8bit,
				SampleIO::mono,
				SampleIO::littleEndian,
				(sampleSettings[smp] & packStatusMask) ? SampleIO::AMS : SampleIO::signedPCM)
				.ReadSample(Samples[smp + 1], file);
		}
	}

	// Copy shadow samples
	for(SAMPLEINDEX smp = 0; smp < GetNumSamples(); smp++)
	{
		INSTRUMENTINDEX sourceInstr = (sampleSettings[smp] & instrIndexMask);
		if(sourceInstr == 0
			|| --sourceInstr >= firstSample.size())
		{
			continue;
		}

		SAMPLEINDEX sourceSample = ((sampleSettings[smp] & sampleIndexMask) >> sampleIndexShift) + firstSample[sourceInstr];
		if(sourceSample > GetNumSamples() || Samples[sourceSample].pSample == nullptr)
		{
			continue;
		}

		// Copy over original sample
		ModSample &sample = Samples[smp + 1];
		ModSample &source = Samples[sourceSample];
		sample.uFlags.set(CHN_16BIT, source.uFlags[CHN_16BIT]);
		sample.nLength = source.nLength;
		if(sample.AllocateSample())
		{
			memcpy(sample.pSample, source.pSample, source.GetSampleSizeInBytes());
		}
	}

	return true;
}


/////////////////////////////////////////////////////////////////////
// AMS Sample unpacking

void AMSUnpack(const int8 * const source, size_t sourceSize, void * const dest, const size_t destSize, char packCharacter)
{
	std::vector<int8> tempBuf(destSize, 0);
	size_t depackSize = destSize;

	// Unpack Loop
	{
		const int8 *in = source;
		int8 *out = tempBuf.data();

		size_t i = sourceSize, j = destSize;
		while(i != 0 && j != 0)
		{
			int8 ch = *(in++);
			if(--i != 0 && ch == packCharacter)
			{
				uint8 repCount = *(in++);
				repCount = static_cast<uint8>(std::min(static_cast<size_t>(repCount), j));
				if(--i != 0 && repCount)
				{
					ch = *(in++);
					i--;
					while(repCount-- != 0)
					{
						*(out++) = ch;
						j--;
					}
				} else
				{
					*(out++) = packCharacter;
					j--;
				}
			} else
			{
				*(out++) = ch;
				j--;
			}
		}
		// j should only be non-zero for truncated samples
		depackSize -= j;
	}

	// Bit Unpack Loop
	{
		int8 *out = tempBuf.data();
		uint16 bitcount = 0x80;
		size_t k = 0;
		uint8 *dst = static_cast<uint8 *>(dest);
		for(size_t i = 0; i < depackSize; i++)
		{
			uint8 al = *out++;
			uint16 dh = 0;
			for(uint16 count = 0; count < 8; count++)
			{
				uint16 bl = al & bitcount;
				bl = ((bl | (bl << 8)) >> ((dh + 8 - count) & 7)) & 0xFF;
				bitcount = ((bitcount | (bitcount << 8)) >> 1) & 0xFF;
				dst[k++] |= bl;
				if(k >= destSize)
				{
					k = 0;
					dh++;
				}
			}
			bitcount = ((bitcount | (bitcount << 8)) >> dh) & 0xFF;
		}
	}

	// Delta Unpack
	{
		int8 old = 0;
		int8 *out = static_cast<int8 *>(dest);
		for(size_t i = depackSize; i != 0; i--)
		{
			int pos = *reinterpret_cast<uint8 *>(out);
			if(pos != 128 && (pos & 0x80) != 0)
			{
				pos = -(pos & 0x7F);
			}
			old -= static_cast<int8>(pos);
			*(out++) = old;
		}
	}
}


OPENMPT_NAMESPACE_END
