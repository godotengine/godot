/*
 * Load_dtm.cpp
 * ------------
 * Purpose: Digital Tracker / Digital Home Studio module Loader (DTM)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "ChunkReader.h"

OPENMPT_NAMESPACE_BEGIN

enum PatternFormats : uint32
{
	DTM_PT_PATTERN_FORMAT = 0,
	DTM_204_PATTERN_FORMAT = MAGIC4BE('2', '.', '0', '4'),
	DTM_206_PATTERN_FORMAT = MAGIC4BE('2', '.', '0', '6'),
};


struct DTMFileHeader
{
	char     magic[4];
	uint32be headerSize;
	uint16be type;        // 0 = module
	uint8be  stereoMode;  // FF = panoramic stereo, 00 = old stereo
	uint8be  bitDepth;    // Typically 8, sometimes 16, but is not actually used anywhere?
	uint16be reserved;    // Usually 0, but not in unknown title 1.dtm and unknown title 2.dtm
	uint16be speed;
	uint16be tempo;
	uint32be forcedSampleRate; // Seems to be ignored in newer files
};

MPT_BINARY_STRUCT(DTMFileHeader, 22)


// IFF-style Chunk
struct DTMChunk
{
	// 32-Bit chunk identifiers
	enum ChunkIdentifiers
	{
		idS_Q_ = MAGIC4BE('S', '.', 'Q', '.'),
		idPATT = MAGIC4BE('P', 'A', 'T', 'T'),
		idINST = MAGIC4BE('I', 'N', 'S', 'T'),
		idIENV = MAGIC4BE('I', 'E', 'N', 'V'),
		idDAPT = MAGIC4BE('D', 'A', 'P', 'T'),
		idDAIT = MAGIC4BE('D', 'A', 'I', 'T'),
		idTEXT = MAGIC4BE('T', 'E', 'X', 'T'),
		idPATN = MAGIC4BE('P', 'A', 'T', 'N'),
		idTRKN = MAGIC4BE('T', 'R', 'K', 'N'),
		idVERS = MAGIC4BE('V', 'E', 'R', 'S'),
		idSV19 = MAGIC4BE('S', 'V', '1', '9'),
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

MPT_BINARY_STRUCT(DTMChunk, 8)


struct DTMSample
{
	uint32be reserved;   // 0x204 for first sample, 0x208 for second, etc...
	uint32be length;     // in bytes
	uint8be  finetune;   // -8....7
	uint8be  volume;     // 0...64
	uint32be loopStart;  // in bytes
	uint32be loopLength; // ditto
	char     name[22];
	uint8be  stereo;
	uint8be  bitDepth;
	uint16be transpose;
	uint16be unknown;
	uint32be sampleRate;

	void ConvertToMPT(ModSample &mptSmp, uint32 forcedSampleRate, uint32 formatVersion) const
	{
		mptSmp.Initialize(MOD_TYPE_IT);
		mptSmp.nLength = length;
		mptSmp.nLoopStart = loopStart;
		mptSmp.nLoopEnd = mptSmp.nLoopStart + loopLength;
		// In revolution to come.dtm, the file header says samples rate is 24512 Hz, but samples say it's 50000 Hz
		// Digital Home Studio ignores the header setting in 2.04-/2.06-style modules
		mptSmp.nC5Speed = (formatVersion == DTM_PT_PATTERN_FORMAT && forcedSampleRate > 0) ? forcedSampleRate : sampleRate;
		int32 transposeAmount = MOD2XMFineTune(finetune);
		if(formatVersion == DTM_206_PATTERN_FORMAT && transpose > 0 && transpose != 48)
		{
			// Digital Home Studio applies this unconditionally, but some old songs sound wrong then (delirium.dtm).
			// Maybe this should not be applied for "real" Digital Tracker modules?
			transposeAmount += (48 - transpose) * 128;
		}
		mptSmp.Transpose(transposeAmount * (1.0 / (12.0 * 128.0)));
		mptSmp.nVolume = std::min<uint8>(volume, 64u) * 4u;
		if(stereo & 1)
		{
			mptSmp.uFlags.set(CHN_STEREO);
			mptSmp.nLength /= 2u;
			mptSmp.nLoopStart /= 2u;
			mptSmp.nLoopEnd /= 2u;
		}
		if(bitDepth > 8)
		{
			mptSmp.uFlags.set(CHN_16BIT);
			mptSmp.nLength /= 2u;
			mptSmp.nLoopStart /= 2u;
			mptSmp.nLoopEnd /= 2u;
		}
		if(mptSmp.nLoopEnd > mptSmp.nLoopStart + 1)
		{
			mptSmp.uFlags.set(CHN_LOOP);
		} else
		{
			mptSmp.nLoopStart = mptSmp.nLoopEnd = 0;
		}
	}
};

MPT_BINARY_STRUCT(DTMSample, 50)


struct DTMInstrument
{
	uint16be insNum;
	uint8be  unknown1;
	uint8be  envelope; // 0xFF = none
	uint8be  sustain;  // 0xFF = no sustain point
	uint16be fadeout;
	uint8be  vibRate;
	uint8be  vibDepth;
	uint8be  modulationRate;
	uint8be  modulationDepth;
	uint8be  breathRate;
	uint8be  breathDepth;
	uint8be  volumeRate;
	uint8be  volumeDepth;
};

MPT_BINARY_STRUCT(DTMInstrument, 15)


struct DTMEnvelope
{
	struct DTMEnvPoint
	{
		uint8be value;
		uint8be tick;
	};
	uint16be numPoints;
	DTMEnvPoint points[16];
};

MPT_BINARY_STRUCT(DTMEnvelope::DTMEnvPoint, 2)
MPT_BINARY_STRUCT(DTMEnvelope, 34)


struct DTMText
{
	uint16be textType;	// 0 = pattern, 1 = free, 2 = song
	uint32be textLength;
	uint16be tabWidth;
	uint16be reserved;
	uint16be oddLength;
};

MPT_BINARY_STRUCT(DTMText, 12)


static bool ValidateHeader(const DTMFileHeader &fileHeader)
{
	if(std::memcmp(fileHeader.magic, "D.T.", 4)
		|| fileHeader.headerSize < sizeof(fileHeader) - 8u
		|| fileHeader.headerSize > 256 // Excessively long song title?
		|| fileHeader.type != 0)
	{
		return false;
	}
	return true;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderDTM(MemoryFileReader file, const uint64 *pfilesize)
{
	DTMFileHeader fileHeader;
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


bool CSoundFile::ReadDTM(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();

	DTMFileHeader fileHeader;
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

	InitializeGlobals(MOD_TYPE_DTM);
	InitializeChannels();
	m_SongFlags.set(SONG_ITCOMPATGXX | SONG_ITOLDEFFECTS);
	m_playBehaviour.reset(kITVibratoTremoloPanbrello);
	// Various files have a default speed or tempo of 0
	if(fileHeader.tempo)
		m_nDefaultTempo.Set(fileHeader.tempo);
	if(fileHeader.speed)
		m_nDefaultSpeed = fileHeader.speed;
	if(fileHeader.stereoMode == 0)
		SetupMODPanning(true);

	file.ReadString<mpt::String::maybeNullTerminated>(m_songName, fileHeader.headerSize - (sizeof(fileHeader) - 8u));

	auto chunks = ChunkReader(file).ReadChunks<DTMChunk>(1);

	// Read order list
	if(FileReader chunk = chunks.GetChunk(DTMChunk::idS_Q_))
	{
		uint16 ordLen = chunk.ReadUint16BE();
		uint16 restartPos = chunk.ReadUint16BE();
		chunk.Skip(4);	// Reserved
		ReadOrderFromFile<uint8>(Order(), chunk, ordLen);
		Order().SetRestartPos(restartPos);
	} else
	{
		return false;
	}

	// Read pattern properties
	uint32 patternFormat;
	if(FileReader chunk = chunks.GetChunk(DTMChunk::idPATT))
	{
		m_nChannels = chunk.ReadUint16BE();
		if(m_nChannels < 1 || m_nChannels > 32)
		{
			return false;
		}
		Patterns.ResizeArray(chunk.ReadUint16BE());	// Number of stored patterns, may be lower than highest pattern number
		patternFormat = chunk.ReadUint32BE();
		if(patternFormat != DTM_PT_PATTERN_FORMAT && patternFormat != DTM_204_PATTERN_FORMAT && patternFormat != DTM_206_PATTERN_FORMAT)
		{
			return false;
		}
	} else
	{
		return false;
	}

	// Read global info
	if(FileReader chunk = chunks.GetChunk(DTMChunk::idSV19))
	{
		chunk.Skip(2);	// Ticks per quarter note, typically 24
		uint32 fractionalTempo = chunk.ReadUint32BE();
		m_nDefaultTempo = TEMPO(m_nDefaultTempo.GetInt() + fractionalTempo / 4294967296.0);

		uint16be panning[32];
		chunk.ReadArray(panning);
		for(CHANNELINDEX chn = 0; chn < 32 && chn < GetNumChannels(); chn++)
		{
			// Panning is in range 0...180, 90 = center
			ChnSettings[chn].nPan = static_cast<uint16>(128 + Util::muldivr(std::min<int>(panning[chn], 180) - 90, 128, 90));
		}

		chunk.Skip(146);
		uint16be volume[32];
		if(chunk.ReadArray(volume))
		{
			for(CHANNELINDEX chn = 0; chn < 32 && chn < GetNumChannels(); chn++)
			{
				// Volume is in range 0...128, 64 = normal
				ChnSettings[chn].nVolume = static_cast<uint8>(std::min<int>(volume[chn], 128) / 2);
			}
			m_nSamplePreAmp *= 2;	// Compensate for channel volume range
		}
	}

	// Read song message
	if(FileReader chunk = chunks.GetChunk(DTMChunk::idTEXT))
	{
		DTMText text;
		chunk.ReadStruct(text);
		if(text.oddLength == 0xFFFF)
		{
			chunk.Skip(1);
		}
		m_songMessage.Read(chunk, chunk.BytesLeft(), SongMessage::leCRLF);
	}

	// Read sample headers
	if(FileReader chunk = chunks.GetChunk(DTMChunk::idINST))
	{
		uint16 numSamples = chunk.ReadUint16BE();
		bool newSamples = (numSamples >= 0x8000);
		numSamples &= 0x7FFF;
		if(numSamples >= MAX_SAMPLES || !chunk.CanRead(numSamples * (sizeof(DTMSample) + (newSamples ? 2u : 0u))))
		{
			return false;
		}
		
		m_nSamples = numSamples;
		for(SAMPLEINDEX smp = 1; smp <= numSamples; smp++)
		{
			SAMPLEINDEX realSample = newSamples ? (chunk.ReadUint16BE() + 1u) : smp;
			DTMSample dtmSample;
			chunk.ReadStruct(dtmSample);
			if(realSample < 1 || realSample >= MAX_SAMPLES)
			{
				continue;
			}
			m_nSamples = std::max(m_nSamples, realSample);
			ModSample &mptSmp = Samples[realSample];
			dtmSample.ConvertToMPT(mptSmp, fileHeader.forcedSampleRate, patternFormat);
			mpt::String::Read<mpt::String::maybeNullTerminated>(m_szNames[realSample], dtmSample.name);
		}
	
		if(chunk.ReadUint16BE() == 0x0004)
		{
			// Digital Home Studio instruments
			m_nInstruments = std::min<INSTRUMENTINDEX>(m_nSamples, MAX_INSTRUMENTS - 1);

			FileReader envChunk = chunks.GetChunk(DTMChunk::idIENV);
			while(chunk.CanRead(sizeof(DTMInstrument)))
			{
				DTMInstrument instr;
				chunk.ReadStruct(instr);
				if(instr.insNum < GetNumInstruments())
				{
					Samples[instr.insNum + 1].nVibDepth = instr.vibDepth;
					Samples[instr.insNum + 1].nVibRate = instr.vibRate;
					Samples[instr.insNum + 1].nVibSweep = 255;

					ModInstrument *mptIns = AllocateInstrument(instr.insNum + 1, instr.insNum + 1);
					if(mptIns != nullptr)
					{
						InstrumentEnvelope &mptEnv = mptIns->VolEnv;
						mptIns->nFadeOut = std::min<uint16>(instr.fadeout, 0xFFF);
						if(instr.envelope != 0xFF && envChunk.Seek(2 + sizeof(DTMEnvelope) * instr.envelope))
						{
							DTMEnvelope env;
							envChunk.ReadStruct(env);
							mptEnv.dwFlags.set(ENV_ENABLED);
							mptEnv.resize(std::min<size_t>({ env.numPoints, mpt::size(env.points), MAX_ENVPOINTS }));
							for(size_t i = 0; i < mptEnv.size(); i++)
							{
								mptEnv[i].value = std::min<uint8>(64, env.points[i].value);
								mptEnv[i].tick = env.points[i].tick;
							}

							if(instr.sustain != 0xFF)
							{
								mptEnv.dwFlags.set(ENV_SUSTAIN);
								mptEnv.nSustainStart = mptEnv.nSustainEnd = instr.sustain;
							}
							if(!mptEnv.empty())
							{
								mptEnv.dwFlags.set(ENV_LOOP);
								mptEnv.nLoopStart = mptEnv.nLoopEnd = static_cast<uint8>(mptEnv.size() - 1);
							}
						}
					}
				}
			}
		}
	}

	// Read pattern data
	for(auto &chunk : chunks.GetAllChunks(DTMChunk::idDAPT))
	{
		chunk.Skip(4);	// FF FF FF FF
		PATTERNINDEX patNum = chunk.ReadUint16BE();
		ROWINDEX numRows = chunk.ReadUint16BE();
		if(patternFormat == DTM_206_PATTERN_FORMAT)
		{
			// The stored data is actually not row-based, but tick-based.
			numRows /= m_nDefaultSpeed;
		}
		if(!(loadFlags & loadPatternData) || patNum > 255 || !Patterns.Insert(patNum, numRows))
		{
			continue;
		}

		if(patternFormat == DTM_206_PATTERN_FORMAT)
		{
			chunk.Skip(4);
			for(CHANNELINDEX chn = 0; chn < GetNumChannels(); chn++)
			{
				uint16 length = chunk.ReadUint16BE();
				if(length % 2u) length++;
				FileReader rowChunk = chunk.ReadChunk(length);
				int tick = 0;
				std::div_t position = { 0, 0 };
				while(rowChunk.CanRead(6) && static_cast<ROWINDEX>(position.quot) < numRows)
				{
					ModCommand *m = Patterns[patNum].GetpModCommand(position.quot, chn);

					uint8 data[6];
					rowChunk.ReadArray(data);
					if(data[0] > 0 && data[0] <= 96)
					{
						m->note = data[0] + NOTE_MIN + 12;
						if(position.rem)
						{
							m->command = CMD_MODCMDEX;
							m->param = 0xD0 | static_cast<ModCommand::PARAM>(std::min(position.rem, 15));
						}
					} else if(data[0] & 0x80)
					{
						// Lower 7 bits contain note, probably intended for MIDI-like note-on/note-off events
						if(position.rem)
						{
							m->command = CMD_MODCMDEX;
							m->param = 0xC0 | static_cast<ModCommand::PARAM>(std::min(position.rem, 15));
						} else
						{
							m->note = NOTE_NOTECUT;
						}
					}
					if(data[1])
					{
						m->volcmd = VOLCMD_VOLUME;
						m->vol = std::min(data[1], uint8(64)); // Volume can go up to 255, but we do not support over-amplification at the moment.
					}
					if(data[2])
					{
						m->instr = data[2];
					}
					if(data[3] || data[4])
					{
						m->command = data[3];
						m->param = data[4];
						ConvertModCommand(*m);
#ifdef MODPLUG_TRACKER
						m->Convert(MOD_TYPE_MOD, MOD_TYPE_IT, *this);
#endif
					}
					if(data[5] & 0x80)
						tick += (data[5] & 0x7F) * 0x100 + rowChunk.ReadUint8();
					else
						tick += data[5];
					position = std::div(tick, m_nDefaultSpeed);
				}
			}
		} else
		{
			ModCommand *m = Patterns[patNum].GetpModCommand(0, 0);
			for(ROWINDEX row = 0; row < numRows; row++)
			{
				for(CHANNELINDEX chn = 0; chn < GetNumChannels(); chn++, m++)
				{
					uint8 data[4];
					chunk.ReadArray(data);
					if(patternFormat == DTM_204_PATTERN_FORMAT)
					{
						if(data[0] > 0 && data[0] < 0x80)
						{
							m->note = (data[0] >> 4) * 12 + (data[0] & 0x0F) + NOTE_MIN + 11;
						}
						uint8 vol = data[1] >> 2;
						if(vol)
						{
							m->volcmd = VOLCMD_VOLUME;
							m->vol = vol - 1u;
						}
						m->instr = ((data[1] & 0x03) << 4) | (data[2] >> 4);
						m->command = data[2] & 0x0F;
						m->param = data[3];
					} else
					{
						ReadMODPatternEntry(data, *m);
						m->instr |= data[0] & 0x30;	// Allow more than 31 instruments
					}
					ConvertModCommand(*m);
					// Fix commands without memory and slide nibble precedence
					switch(m->command)
					{
					case CMD_PORTAMENTOUP:
					case CMD_PORTAMENTODOWN:
						if(!m->param)
						{
							m->command = CMD_NONE;
						}
						break;
					case CMD_VOLUMESLIDE:
					case CMD_TONEPORTAVOL:
					case CMD_VIBRATOVOL:
						if(m->param & 0xF0)
						{
							m->param &= 0xF0;
						} else if(!m->param)
						{
							m->command = CMD_NONE;
						}
						break;
					default:
						break;
					}
#ifdef MODPLUG_TRACKER
					m->Convert(MOD_TYPE_MOD, MOD_TYPE_IT, *this);
#endif
				}
			}
		}
	}

	// Read pattern names
	if(FileReader chunk = chunks.GetChunk(DTMChunk::idPATN))
	{
		PATTERNINDEX pat = 0;
		std::string name;
		while(chunk.CanRead(1) && pat < Patterns.Size())
		{
			chunk.ReadNullString(name, 32);
			Patterns[pat].SetName(name);
			pat++;
		}
	}

	// Read channel names
	if(FileReader chunk = chunks.GetChunk(DTMChunk::idTRKN))
	{
		CHANNELINDEX chn = 0;
		std::string name;
		while(chunk.CanRead(1) && chn < GetNumChannels())
		{
			chunk.ReadNullString(name, 32);
			mpt::String::Copy(ChnSettings[chn].szName, name);
			chn++;
		}
	}

	// Read sample data
	for(auto &chunk : chunks.GetAllChunks(DTMChunk::idDAIT))
	{
		PATTERNINDEX smp = chunk.ReadUint16BE() + 1;
		if(smp == 0 || smp > GetNumSamples() || !(loadFlags & loadSampleData))
		{
			continue;
		}
		ModSample &mptSmp = Samples[smp];
		SampleIO(
			mptSmp.uFlags[CHN_16BIT] ? SampleIO::_16bit : SampleIO::_8bit,
			mptSmp.uFlags[CHN_STEREO] ? SampleIO::stereoInterleaved: SampleIO::mono,
			SampleIO::bigEndian,
			SampleIO::signedPCM).ReadSample(mptSmp, chunk);
	}

	// Is this accurate?
	if(patternFormat == DTM_206_PATTERN_FORMAT)
	{
		m_madeWithTracker = MPT_USTRING("Digital Home Studio");
	} else if(FileReader chunk = chunks.GetChunk(DTMChunk::idVERS))
	{
		uint32 version = chunk.ReadUint32BE();
		m_madeWithTracker = mpt::format(MPT_USTRING("Digital Tracker %1.%2"))(version >> 4, version & 0x0F);
	} else
	{
		m_madeWithTracker = MPT_USTRING("Digital Tracker");
	}

	return true;
}

OPENMPT_NAMESPACE_END
