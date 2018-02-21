/*
 * Load_dbm.cpp
 * ------------
 * Purpose: DigiBooster Pro module Loader (DBM)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "ChunkReader.h"
#include "../common/StringFixer.h"
#ifndef NO_PLUGINS
#include "plugins/DigiBoosterEcho.h"
#endif // NO_PLUGINS

#ifdef LIBOPENMPT_BUILD
#define MPT_DBM_USE_REAL_SUBSONGS
#endif

OPENMPT_NAMESPACE_BEGIN

struct DBMFileHeader
{
	char  dbm0[4];
	uint8 trkVerHi;
	uint8 trkVerLo;
	char  reserved[2];
};

MPT_BINARY_STRUCT(DBMFileHeader, 8)


// IFF-style Chunk
struct DBMChunk
{
	// 32-Bit chunk identifiers
	enum ChunkIdentifiers
	{
		idNAME	= MAGIC4BE('N','A','M','E'),
		idINFO	= MAGIC4BE('I','N','F','O'),
		idSONG	= MAGIC4BE('S','O','N','G'),
		idINST	= MAGIC4BE('I','N','S','T'),
		idVENV	= MAGIC4BE('V','E','N','V'),
		idPENV	= MAGIC4BE('P','E','N','V'),
		idPATT	= MAGIC4BE('P','A','T','T'),
		idPNAM	= MAGIC4BE('P','N','A','M'),
		idSMPL	= MAGIC4BE('S','M','P','L'),
		idDSPE	= MAGIC4BE('D','S','P','E'),
		idMPEG	= MAGIC4BE('M','P','E','G'),
	};

	uint32be id;
	uint32be length;

	size_t GetLength() const
	{
		return length;
	}

	ChunkIdentifiers GetID() const
	{
		return static_cast<ChunkIdentifiers>(id.get());
	}
};

MPT_BINARY_STRUCT(DBMChunk, 8)


struct DBMInfoChunk
{
	uint16be instruments;
	uint16be samples;
	uint16be songs;
	uint16be patterns;
	uint16be channels;
};

MPT_BINARY_STRUCT(DBMInfoChunk, 10)


// Instrument header
struct DBMInstrument
{
	enum DBMInstrFlags
	{
		smpLoop			= 0x01,
		smpPingPongLoop	= 0x02,
	};

	char     name[30];
	uint16be sample;		// Sample reference
	uint16be volume;		// 0...64
	uint32be sampleRate;
	uint32be loopStart;
	uint32be loopLength;
	int16be  panning;		// -128...128
	uint16be flags;			// See DBMInstrFlags
};

MPT_BINARY_STRUCT(DBMInstrument, 50)


// Volume or panning envelope
struct DBMEnvelope
{
	enum DBMEnvelopeFlags
	{
		envEnabled	= 0x01,
		envSustain	= 0x02,
		envLoop		= 0x04,
	};

	uint16be instrument;
	uint8be  flags;			// See DBMEnvelopeFlags
	uint8be  numSegments;	// Number of envelope points - 1
	uint8be  sustain1;
	uint8be  loopBegin;
	uint8be  loopEnd;
	uint8be  sustain2;		// Second sustain point
	uint16be data[2 * 32];
};

MPT_BINARY_STRUCT(DBMEnvelope, 136)


// Note: Unlike in MOD, 1Fx, 2Fx, 5Fx / 5xF, 6Fx / 6xF and AFx / AxF are fine slides.
static const ModCommand::COMMAND dbmEffects[] =
{
	CMD_ARPEGGIO, CMD_PORTAMENTOUP, CMD_PORTAMENTODOWN, CMD_TONEPORTAMENTO,
	CMD_VIBRATO, CMD_TONEPORTAVOL, CMD_VIBRATOVOL, CMD_TREMOLO,
	CMD_PANNING8, CMD_OFFSET, CMD_VOLUMESLIDE, CMD_POSITIONJUMP,
	CMD_VOLUME, CMD_PATTERNBREAK, CMD_MODCMDEX, CMD_TEMPO,
	CMD_GLOBALVOLUME, CMD_GLOBALVOLSLIDE, CMD_KEYOFF, CMD_SETENVPOSITION,
	CMD_CHANNELVOLUME, CMD_CHANNELVOLSLIDE, CMD_NONE, CMD_NONE,
	CMD_NONE, CMD_PANNINGSLIDE, CMD_NONE, CMD_NONE,
	CMD_NONE, CMD_NONE, CMD_NONE,
#ifndef NO_PLUGINS
	CMD_DBMECHO,	// Toggle DSP
	CMD_MIDI,		// Wxx Echo Delay
	CMD_MIDI,		// Xxx Echo Feedback
	CMD_MIDI,		// Yxx Echo Mix
	CMD_MIDI,		// Zxx Echo Cross
#endif // NO_PLUGINS
};


static void ConvertDBMEffect(uint8 &command, uint8 &param)
{
	uint8 oldCmd = command;
	if(command < CountOf(dbmEffects))
		command = dbmEffects[command];
	else
		command = CMD_NONE;

	switch(command)
	{
	case CMD_ARPEGGIO:
		if(param == 0)
			command = CMD_NONE;
		break;

#ifdef MODPLUG_TRACKER
	case CMD_VIBRATO:
		if(param & 0x0F)
		{
			// DBM vibrato is half as deep as most other trackers. Convert it to IT fine vibrato range if possible.
			uint8 depth = (param & 0x0F) * 2u;
			param &= 0xF0;
			if(depth < 16)
				command = CMD_FINEVIBRATO;
			else
				depth = (depth + 2u) / 4u;
			param |= depth;
		}
		break;
#endif

	// Volume slide nibble priority - first nibble (slide up) has precedence.
	case CMD_VOLUMESLIDE:
	case CMD_TONEPORTAVOL:
	case CMD_VIBRATOVOL:
		if((param & 0xF0) != 0x00 && (param & 0xF0) != 0xF0 && (param & 0x0F) != 0x0F)
			param &= 0xF0;
		break;

	case CMD_GLOBALVOLUME:
		if(param <= 64)
			param *= 2;
		else
			param = 128;
		break;

	case CMD_MODCMDEX:
		switch(param & 0xF0)
		{
		case 0x00:	// set filter
			command = CMD_NONE;
			break;
		case 0x30:	// play backwards
			command = CMD_S3MCMDEX;
			param = 0x9F;
			break;
		case 0x40:	// turn off sound in channel (volume / portamento commands after this can't pick up the note anymore)
			command = CMD_S3MCMDEX;
			param = 0xC0;
			break;
		case 0x50:	// turn on/off channel
			// TODO: Apparently this should also kill the playing note.
			if((param & 0x0F) <= 0x01)
			{
				command = CMD_CHANNELVOLUME;
				param = (param == 0x50) ? 0x00 : 0x40;
			}
			break;
		case 0x60:	// set loop begin / loop
			// TODO
			break;
		case 0x70:	// set offset
			// TODO
			break;
		default:
			// Rest will be converted later from CMD_MODCMDEX to CMD_S3MCMDEX.
			break;
		}
		break;

	case CMD_TEMPO:
		if(param <= 0x1F) command = CMD_SPEED;
		break;

	case CMD_KEYOFF:
		if (param == 0)
		{
			// TODO key of at tick 0
		}
		break;

	case CMD_MIDI:
		// Encode echo parameters into fixed MIDI macros
		param = 128 + (oldCmd - 32) * 32 + param / 8;
	}
}


// Read a chunk of volume or panning envelopes
static void ReadDBMEnvelopeChunk(FileReader chunk, EnvelopeType envType, CSoundFile &sndFile, bool scaleEnv)
{
	uint16 numEnvs = chunk.ReadUint16BE();
	for(uint16 env = 0; env < numEnvs; env++)
	{
		DBMEnvelope dbmEnv;
		chunk.ReadStruct(dbmEnv);

		uint16 dbmIns = dbmEnv.instrument;
		if(dbmIns > 0 && dbmIns <= sndFile.GetNumInstruments() && (sndFile.Instruments[dbmIns] != nullptr))
		{
			ModInstrument *mptIns = sndFile.Instruments[dbmIns];
			InstrumentEnvelope &mptEnv = mptIns->GetEnvelope(envType);

			if(dbmEnv.numSegments)
			{
				if(dbmEnv.flags & DBMEnvelope::envEnabled) mptEnv.dwFlags.set(ENV_ENABLED);
				if(dbmEnv.flags & DBMEnvelope::envSustain) mptEnv.dwFlags.set(ENV_SUSTAIN);
				if(dbmEnv.flags & DBMEnvelope::envLoop) mptEnv.dwFlags.set(ENV_LOOP);
			}

			uint8 numPoints = std::min<uint8>(dbmEnv.numSegments, 31) + 1;
			mptEnv.resize(numPoints);

			mptEnv.nLoopStart = dbmEnv.loopBegin;
			mptEnv.nLoopEnd = dbmEnv.loopEnd;
			mptEnv.nSustainStart = mptEnv.nSustainEnd = dbmEnv.sustain1;

			for(uint8 i = 0; i < numPoints; i++)
			{
				mptEnv[i].tick = dbmEnv.data[i * 2];
				uint16 val = dbmEnv.data[i * 2 + 1];
				if(scaleEnv)
				{
					// Panning envelopes are -128...128 in DigiBooster Pro 3.x
					val = (val + 128) / 4;
				}
				LimitMax(val, uint16(64));
				mptEnv[i].value = static_cast<uint8>(val);
			}
		}
	}
}


static bool ValidateHeader(const DBMFileHeader &fileHeader)
{
	if(std::memcmp(fileHeader.dbm0, "DBM0", 4)
		|| fileHeader.trkVerHi > 3)
	{
		return false;
	}
	return true;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderDBM(MemoryFileReader file, const uint64 *pfilesize)
{
	DBMFileHeader fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return ProbeWantMoreData;
	}
	if(!ValidateHeader(fileHeader))
	{
		return ProbeFailure;
	}
	MPT_UNREFERENCED_PARAMETER(pfilesize);
	return ProbeSuccess;
}


bool CSoundFile::ReadDBM(FileReader &file, ModLoadingFlags loadFlags)
{

	file.Rewind();
	DBMFileHeader fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return false;
	}
	if(!ValidateHeader(fileHeader))
	{
		return false;
	}
	if(loadFlags == onlyVerifyHeader)
	{
		return true;
	}

	ChunkReader chunkFile(file);
	auto chunks = chunkFile.ReadChunks<DBMChunk>(1);

	// Globals
	DBMInfoChunk infoData;
	if(!chunks.GetChunk(DBMChunk::idINFO).ReadStruct(infoData))
	{
		return false;
	}

	InitializeGlobals(MOD_TYPE_DBM);
	InitializeChannels();
	m_SongFlags = SONG_ITCOMPATGXX | SONG_ITOLDEFFECTS;
	m_nChannels = Clamp<uint16, uint16>(infoData.channels, 1, MAX_BASECHANNELS);	// note: MAX_BASECHANNELS is currently 127, but DBPro 2 supports up to 128 channels, DBPro 3 apparently up to 254.
	m_nInstruments = std::min<INSTRUMENTINDEX>(infoData.instruments, MAX_INSTRUMENTS - 1);
	m_nSamples = std::min<SAMPLEINDEX>(infoData.samples, MAX_SAMPLES - 1);
	m_madeWithTracker = mpt::format(MPT_USTRING("DigiBooster Pro %1.%2"))(mpt::ufmt::hex(fileHeader.trkVerHi), mpt::ufmt::hex(fileHeader.trkVerLo));
	m_playBehaviour.set(kSlidesAtSpeed1);
	m_playBehaviour.reset(kITVibratoTremoloPanbrello);

	// Name chunk
	FileReader nameChunk = chunks.GetChunk(DBMChunk::idNAME);
	nameChunk.ReadString<mpt::String::maybeNullTerminated>(m_songName, nameChunk.GetLength());

	// Song chunk
	FileReader songChunk = chunks.GetChunk(DBMChunk::idSONG);
	Order().clear();
	uint16 numSongs = infoData.songs;
	for(uint16 i = 0; i < numSongs && songChunk.CanRead(46); i++)
	{
		char name[44];
		songChunk.ReadString<mpt::String::maybeNullTerminated>(name, 44);
		if(m_songName.empty())
		{
			m_songName = name;
		}
		uint16 numOrders = songChunk.ReadUint16BE();

#ifdef MPT_DBM_USE_REAL_SUBSONGS
		if(!Order().empty())
		{
			// Add a new sequence for this song
			if(Order.AddSequence(false) == SEQUENCEINDEX_INVALID)
				break;
		}
		Order().SetName(name);
		ReadOrderFromFile<uint16be>(Order(), songChunk, numOrders);
#else
		const ORDERINDEX startIndex = Order().GetLength();
		if(startIndex < MAX_ORDERS && songChunk.CanRead(numOrders * 2u))
		{
			LimitMax(numOrders, static_cast<ORDERINDEX>(MAX_ORDERS - startIndex - 1));
			Order().resize(startIndex + numOrders + 1);
			for(uint16 ord = 0; ord < numOrders; ord++)
			{
				Order()[startIndex + ord] = static_cast<PATTERNINDEX>(songChunk.ReadUint16BE());
			}
		}
#endif // MPT_DBM_USE_REAL_SUBSONGS
	}
#ifdef MPT_DBM_USE_REAL_SUBSONGS
	Order.SetSequence(0);
#endif // MPT_DBM_USE_REAL_SUBSONGS

	// Read instruments
	if(FileReader instChunk = chunks.GetChunk(DBMChunk::idINST))
	{
		for(INSTRUMENTINDEX i = 1; i <= GetNumInstruments(); i++)
		{
			DBMInstrument instrHeader;
			instChunk.ReadStruct(instrHeader);

			ModInstrument *mptIns = AllocateInstrument(i, instrHeader.sample);
			if(mptIns == nullptr || instrHeader.sample >= MAX_SAMPLES)
			{
				continue;
			}
			ModSample &mptSmp = Samples[instrHeader.sample];

			mpt::String::Read<mpt::String::maybeNullTerminated>(mptIns->name, instrHeader.name);
			mpt::String::Read<mpt::String::maybeNullTerminated>(m_szNames[instrHeader.sample], instrHeader.name);

			mptIns->nFadeOut = 0;
			mptIns->nPan = static_cast<uint16>(instrHeader.panning + 128);
			LimitMax(mptIns->nPan, uint32(256));
			mptIns->dwFlags.set(INS_SETPANNING);

			// Sample Info
			mptSmp.Initialize();
			mptSmp.nVolume = std::min<uint16>(instrHeader.volume, 64) * 4u;
			mptSmp.nC5Speed = instrHeader.sampleRate;

			if(instrHeader.loopLength && (instrHeader.flags & (DBMInstrument::smpLoop | DBMInstrument::smpPingPongLoop)))
			{
				mptSmp.nLoopStart = instrHeader.loopStart;
				mptSmp.nLoopEnd = mptSmp.nLoopStart + instrHeader.loopLength;
				mptSmp.uFlags.set(CHN_LOOP);
				if(instrHeader.flags & DBMInstrument::smpPingPongLoop) mptSmp.uFlags.set(CHN_PINGPONGLOOP);
			}
		}

		// Read envelopes
		ReadDBMEnvelopeChunk(chunks.GetChunk(DBMChunk::idVENV), ENV_VOLUME, *this, false);
		ReadDBMEnvelopeChunk(chunks.GetChunk(DBMChunk::idPENV), ENV_PANNING, *this, fileHeader.trkVerHi > 2);

		// Note-Off cuts samples if there's no envelope.
		for(INSTRUMENTINDEX i = 1; i <= GetNumInstruments(); i++)
		{
			if(Instruments[i] != nullptr && !Instruments[i]->VolEnv.dwFlags[ENV_ENABLED])
			{
				Instruments[i]->nFadeOut = 32767;
			}
		}
	}

	// Patterns
	FileReader patternChunk = chunks.GetChunk(DBMChunk::idPATT);
#ifndef NO_PLUGINS
	bool hasEchoEnable = false, hasEchoParams = false;
#endif // NO_PLUGINS
	if(patternChunk.IsValid() && (loadFlags & loadPatternData))
	{
		FileReader patternNameChunk = chunks.GetChunk(DBMChunk::idPNAM);
		patternNameChunk.Skip(1);	// Encoding, should be UTF-8 or ASCII

		Patterns.ResizeArray(infoData.patterns);
		for(PATTERNINDEX pat = 0; pat < infoData.patterns; pat++)
		{
			uint16 numRows = patternChunk.ReadUint16BE();
			uint32 packedSize = patternChunk.ReadUint32BE();
			FileReader chunk = patternChunk.ReadChunk(packedSize);

			if(!Patterns.Insert(pat, numRows))
			{
				continue;
			}

			std::string patName;
			patternNameChunk.ReadSizedString<uint8be, mpt::String::maybeNullTerminated>(patName);
			Patterns[pat].SetName(patName);

			PatternRow patRow = Patterns[pat].GetRow(0);
			ROWINDEX row = 0;
			while(chunk.CanRead(1))
			{
				const uint8 ch = chunk.ReadUint8();

				if(!ch)
				{
					// End Of Row
					if(++row >= numRows)
						break;

					patRow = Patterns[pat].GetRow(row);
					continue;
				}

				ModCommand dummy;
				ModCommand &m = ch <= GetNumChannels() ? patRow[ch - 1] : dummy;

				const uint8 b = chunk.ReadUint8();

				if(b & 0x01)
				{
					uint8 note = chunk.ReadUint8();

					if(note == 0x1F)
						note = NOTE_KEYOFF;
					else if(note > 0 && note < 0xFE)
					{
						note = ((note >> 4) * 12) + (note & 0x0F) + 13;
					}
					m.note = note;
				}
				if(b & 0x02)
				{
					m.instr = chunk.ReadUint8();
				}
				if(b & 0x3C)
				{
					uint8 cmd1 = CMD_NONE, cmd2 = CMD_NONE;
					uint8 param1 = 0, param2 = 0;
					if(b & 0x04) cmd2 = chunk.ReadUint8();
					if(b & 0x08) param2 = chunk.ReadUint8();
					if(b & 0x10) cmd1 = chunk.ReadUint8();
					if(b & 0x20) param1 = chunk.ReadUint8();
					ConvertDBMEffect(cmd1, param1);
					ConvertDBMEffect(cmd2, param2);

					if (cmd2 == CMD_VOLUME || (cmd2 == CMD_NONE && cmd1 != CMD_VOLUME))
					{
						std::swap(cmd1, cmd2);
						std::swap(param1, param2);
					}

					ModCommand::TwoRegularCommandsToMPT(cmd1, param1, cmd2, param2);

					m.volcmd = cmd1;
					m.vol = param1;
					m.command = cmd2;
					m.param = param2;
#ifdef MODPLUG_TRACKER
					m.ExtendedMODtoS3MEffect();
#endif // MODPLUG_TRACKER
#ifndef NO_PLUGINS
					if(m.command == CMD_DBMECHO)
						hasEchoEnable = true;
					else if(m.command == CMD_MIDI)
						hasEchoParams = true;
#endif // NO_PLUGINS
				}
			}
		}
	}

#ifndef NO_PLUGINS
	// Echo DSP
	if(loadFlags & loadPluginData)
	{
		if(hasEchoEnable)
		{
			// If there are any Vxx effects to dynamically enable / disable echo, use the CHN_NOFX flag.
			for(CHANNELINDEX i = 0; i < m_nChannels; i++)
			{
				ChnSettings[i].nMixPlugin = 1;
				ChnSettings[i].dwFlags.set(CHN_NOFX);
			}
		}

		bool anyEnabled = hasEchoEnable;
		// DBP 3 Documentation says that the defaults are 64/128/128/255, but they appear to be 80/150/80/255 in DBP 2.21
		uint8 settings[8] = { 0, 80, 0, 150, 0, 80, 0, 255 };

		if(FileReader dspChunk = chunks.GetChunk(DBMChunk::idDSPE))
		{
			uint16 maskLen = dspChunk.ReadUint16BE();
			for(uint16 i = 0; i < maskLen; i++)
			{
				bool enabled = (dspChunk.ReadUint8() == 0);
				if(i < m_nChannels)
				{
					if(hasEchoEnable)
					{
						// If there are any Vxx effects to dynamically enable / disable echo, use the CHN_NOFX flag.
						ChnSettings[i].dwFlags.set(CHN_NOFX, !enabled);
					} else if(enabled)
					{
						ChnSettings[i].nMixPlugin = 1;
						anyEnabled = true;
					}
				}
			}
			dspChunk.ReadArray(settings);
		}

		if(anyEnabled)
		{
			// Note: DigiBooster Pro 3 has a more versatile per-channel echo effect.
			// In this case, we'd have to create one plugin per channel.
			SNDMIXPLUGIN &plugin = m_MixPlugins[0];
			plugin.Destroy();
			memcpy(&plugin.Info.dwPluginId1, "DBM0", 4);
			memcpy(&plugin.Info.dwPluginId2, "Echo", 4);
			plugin.Info.routingFlags = SNDMIXPLUGININFO::irAutoSuspend;
			plugin.Info.mixMode = 0;
			plugin.Info.gain = 10;
			plugin.Info.reserved = 0;
			plugin.Info.dwOutputRouting = 0;
			std::fill(plugin.Info.dwReserved, plugin.Info.dwReserved + mpt::size(plugin.Info.dwReserved), 0);
			mpt::String::Write<mpt::String::nullTerminated>(plugin.Info.szName, "Echo");
			mpt::String::Write<mpt::String::nullTerminated>(plugin.Info.szLibraryName, "DigiBooster Pro Echo");

			plugin.pluginData.resize(sizeof(DigiBoosterEcho::PluginChunk));
			new (plugin.pluginData.data()) DigiBoosterEcho::PluginChunk(settings[1], settings[3], settings[5], settings[7]);
		}
	}

	// Encode echo parameters into fixed MIDI macros
	if(hasEchoParams)
	{
		for(uint32 i = 0; i < 32; i++)
		{
			uint32 param = (i * 127u) / 32u;
			sprintf(m_MidiCfg.szMidiZXXExt[i     ], "F0F080%02X", param);
			sprintf(m_MidiCfg.szMidiZXXExt[i + 32], "F0F081%02X", param);
			sprintf(m_MidiCfg.szMidiZXXExt[i + 64], "F0F082%02X", param);
			sprintf(m_MidiCfg.szMidiZXXExt[i + 96], "F0F083%02X", param);
		}
	}
#endif // NO_PLUGINS

	// Samples
	FileReader sampleChunk = chunks.GetChunk(DBMChunk::idSMPL);
	if(sampleChunk.IsValid() && (loadFlags & loadSampleData))
	{
		for(SAMPLEINDEX smp = 1; smp <= GetNumSamples(); smp++)
		{
			uint32 sampleFlags = sampleChunk.ReadUint32BE();
			uint32 sampleLength = sampleChunk.ReadUint32BE();

			ModSample &sample = Samples[smp];
			sample.nLength = sampleLength;

			if(sampleFlags & 7)
			{
				SampleIO(
					(sampleFlags & 4) ? SampleIO::_32bit : ((sampleFlags & 2) ? SampleIO::_16bit : SampleIO::_8bit),
					SampleIO::mono,
					SampleIO::bigEndian,
					SampleIO::signedPCM)
					.ReadSample(sample, sampleChunk);
			}
		}
	}

#if defined(MPT_ENABLE_MP3_SAMPLES) && 0
	// Compressed samples - this does not quite work yet...
	FileReader mpegChunk = chunks.GetChunk(DBMChunk::idMPEG);
	if(mpegChunk.IsValid() && (loadFlags & loadSampleData))
	{
		for(SAMPLEINDEX smp = 1; smp <= GetNumSamples(); smp++)
		{
			Samples[smp].nLength = mpegChunk.ReadUint32BE();
		}
		mpegChunk.Skip(2);	// 0x00 0x40

		// Read whole MPEG stream into one sample and then split it up.
		FileReader chunk = mpegChunk.GetChunk(mpegChunk.BytesLeft());
		if(ReadMP3Sample(0, chunk))
		{
			ModSample &srcSample = Samples[0];
			const int8 *smpData = srcSample.pSample8;
			uint32 predelay = Util::muldiv_unsigned(20116, srcSample.nC5Speed, 100000);
			smpData += predelay * srcSample.GetBytesPerSample();
			srcSample.nLength -= predelay;

			for(SAMPLEINDEX smp = 1; smp <= GetNumSamples(); smp++)
			{
				ModSample &sample = Samples[smp];
				sample.uFlags.set(srcSample.uFlags);
				LimitMax(sample.nLength, srcSample.nLength);
				if(sample.nLength)
				{
					sample.AllocateSample();
					memcpy(sample.pSample, smpData, sample.GetSampleSizeInBytes());
					smpData += sample.GetSampleSizeInBytes();
					srcSample.nLength -= sample.nLength;
					uint32 gap = Util::muldiv_unsigned(454, srcSample.nC5Speed, 10000);
					smpData += gap * srcSample.GetBytesPerSample();
					srcSample.nLength -= gap;
				}
			}
			srcSample.FreeSample();
		}
	}
#endif // MPT_ENABLE_MP3_SAMPLES
	
	return true;
}


OPENMPT_NAMESPACE_END
