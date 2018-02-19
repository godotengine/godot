/*
 * Load_669.cpp
 * ------------
 * Purpose: 669 Composer / UNIS 669 module loader
 * Notes  : <opinion humble="false">This is better than Schism's 669 loader</opinion> :)
 *          (some of this code is "heavily inspired" by Storlek's code from Schism Tracker, and improvements have been made where necessary.)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"

OPENMPT_NAMESPACE_BEGIN

struct _669FileHeader
{
	char  magic[2];			// 'if' (0x6669, ha ha) or 'JN'
	char  songMessage[108];	// Song Message
	uint8 samples;			// number of samples (1-64)
	uint8 patterns;			// number of patterns (1-128)
	uint8 restartPos;
	uint8 orders[128];
	uint8 tempoList[128];
	uint8 breaks[128];
};

MPT_BINARY_STRUCT(_669FileHeader, 497)


struct _669Sample
{
	char     filename[13];
	uint32le length;
	uint32le loopStart;
	uint32le loopEnd;

	// Convert a 669 sample header to OpenMPT's internal sample header.
	void ConvertToMPT(ModSample &mptSmp) const
	{
		mptSmp.Initialize();

		mptSmp.nC5Speed = 8363;
		mptSmp.nLength = length;
		mptSmp.nLoopStart = loopStart;
		mptSmp.nLoopEnd = loopEnd;

		if(mptSmp.nLoopEnd > mptSmp.nLength && mptSmp.nLoopStart == 0)
		{
			mptSmp.nLoopEnd = 0;
		}
		if(mptSmp.nLoopEnd != 0)
		{
			mptSmp.uFlags = CHN_LOOP;
			mptSmp.SanitizeLoops();
		}
	}
};

MPT_BINARY_STRUCT(_669Sample, 25)


static bool ValidateHeader(const _669FileHeader &fileHeader)
{
	if((std::memcmp(fileHeader.magic, "if", 2) && std::memcmp(fileHeader.magic, "JN", 2))
		|| fileHeader.samples > 64
		|| fileHeader.restartPos >= 128
		|| fileHeader.patterns > 128)
	{
		return false;
	}
	for(std::size_t i = 0; i < CountOf(fileHeader.breaks); i++)
	{
		if(fileHeader.orders[i] >= 128 && fileHeader.orders[i] < 0xFE)
		{
			return false;
		}
		if(fileHeader.orders[i] < 128 && fileHeader.tempoList[i] == 0)
		{
			return false;
		}
		if(fileHeader.breaks[i] >= 64)
		{
			return false;
		}
	}
	return true;
}


static uint64 GetHeaderMinimumAdditionalSize(const _669FileHeader &fileHeader)
{
	return fileHeader.samples * sizeof(_669Sample) + fileHeader.patterns * 1536u;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeader669(MemoryFileReader file, const uint64 *pfilesize)
{
	_669FileHeader fileHeader;
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


bool CSoundFile::Read669(FileReader &file, ModLoadingFlags loadFlags)
{
	_669FileHeader fileHeader;

	file.Rewind();
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
	
	if(!file.CanRead(mpt::saturate_cast<FileReader::off_t>(GetHeaderMinimumAdditionalSize(fileHeader))))
	{
		return false;
	}

	InitializeGlobals(MOD_TYPE_669);
	m_nMinPeriod = 28 << 2;
	m_nMaxPeriod = 1712 << 3;
	m_nDefaultTempo.Set(78);
	m_nDefaultSpeed = 4;
	m_nChannels = 8;
#ifdef MODPLUG_TRACEKR
	// 669 uses frequencies rather than periods, so linear slides mode will sound better in the higher octaves.
	m_SongFlags.set(SONG_LINEARSLIDES);
#endif // MODPLUG_TRACKER

	if(!memcmp(fileHeader.magic, "if", 2))
		m_madeWithTracker = MPT_USTRING("Composer 669");
	else
		m_madeWithTracker = MPT_USTRING("UNIS 669");

	m_nSamples = fileHeader.samples;
	for(SAMPLEINDEX smp = 1; smp <= m_nSamples; smp++)
	{
		_669Sample sampleHeader;
		file.ReadStruct(sampleHeader);
		// Since 669 files have very unfortunate magic bytes ("if") and can
		// hardly be validated, reject any file with far too big samples.
		if(sampleHeader.length >= 0x4000000)
			return false;
		sampleHeader.ConvertToMPT(Samples[smp]);
		mpt::String::Read<mpt::String::maybeNullTerminated>(m_szNames[smp], sampleHeader.filename);
	}

	// Copy first song message line into song title
	mpt::String::Read<mpt::String::spacePadded>(m_songName, fileHeader.songMessage, 36);
	// Song Message
	m_songMessage.ReadFixedLineLength(mpt::byte_cast<const mpt::byte*>(fileHeader.songMessage), 108, 36, 0);

	// Reading Orders
	ReadOrderFromArray(Order(), fileHeader.orders, MPT_ARRAY_COUNT(fileHeader.orders), 0xFF, 0xFE);
	if(Order()[fileHeader.restartPos] < fileHeader.patterns)
		Order().SetRestartPos(fileHeader.restartPos);

	// Set up panning
	for(CHANNELINDEX chn = 0; chn < 8; chn++)
	{
		ChnSettings[chn].Reset();
		ChnSettings[chn].nPan = (chn & 1) ? 0xD0 : 0x30;
	}

	// Reading Patterns
	Patterns.ResizeArray(fileHeader.patterns);
	for(PATTERNINDEX pat = 0; pat < fileHeader.patterns; pat++)
	{
		if(!(loadFlags & loadPatternData) || !Patterns.Insert(pat, 64))
		{
			file.Skip(64 * 8 * 3);
			continue;
		}

		const ModCommand::COMMAND effTrans[] =
		{
			CMD_PORTAMENTOUP,	// Slide up (param * 80) Hz on every tick
			CMD_PORTAMENTODOWN,	// Slide down (param * 80) Hz on every tick
			CMD_TONEPORTAMENTO,	// Slide to note by (param * 40) Hz on every tick
			CMD_S3MCMDEX,		// Add (param * 80) Hz to sample frequency
			CMD_VIBRATO,		// Add (param * 669) Hz on every other tick
			CMD_SPEED,			// Set ticks per row
			CMD_PANNINGSLIDE,	// Extended UNIS 669 effect
			CMD_RETRIG,			// Extended UNIS 669 effect
		};

		uint8 effect[8] = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
		for(ROWINDEX row = 0; row < 64; row++)
		{
			PatternRow m = Patterns[pat].GetRow(row);

			for(CHANNELINDEX chn = 0; chn < 8; chn++, m++)
			{
				uint8 data[3];
				file.ReadArray(data);

				uint8 note = data[0] >> 2;
				uint8 instr = ((data[0] & 0x03) << 4) | (data[1] >> 4);
				uint8 vol = data[1] & 0x0F;
				if(data[0] < 0xFE)
				{
					m->note = note + 36 + NOTE_MIN;
					m->instr = instr + 1;
					effect[chn] = 0xFF;
				}
				if(data[0] <= 0xFE)
				{
					m->volcmd = VOLCMD_VOLUME;
					m->vol = ((vol * 64 + 8) / 15);
				}

				if(data[2] != 0xFF)
				{
					effect[chn] = data[2];
				}
				if((data[2] & 0x0F) == 0 && data[2] != 0x30)
				{
					// A param value of 0 resets the effect.
					effect[chn] = 0xFF;
				}
				if(effect[chn] == 0xFF)
				{
					continue;
				}

				m->param = effect[chn] & 0x0F;

				uint8 command = effect[chn] >> 4;
				if(command < static_cast<uint8>(CountOf(effTrans)))
				{
					m->command = effTrans[command];
				} else
				{
					m->command = CMD_NONE;
					continue;
				}

				// Fix some commands
				switch(command)
				{
				case 3:
					// D - frequency adjust
#ifdef MODPLUG_TRACKER
					// Since we convert to S3M, the finetune command will not quite do what we intend to do (it can adjust the frequency upwards and downwards), so try to approximate it using a fine slide.
					m->command = CMD_PORTAMENTOUP;
					m->param |= 0xF0;
#else
					m->param |= 0x20;
#endif
					effect[chn] = 0xFF;
					break;

				case 4:
					// E - frequency vibrato - almost like an arpeggio, but does not arpeggiate by a given note but by a frequency amount.
#ifdef MODPLUG_TRACKER
					m->command = CMD_ARPEGGIO;
#endif
					m->param |= (m->param << 4);
					break;

				case 5:
					// F - set tempo
					// TODO: param 0 is a "super fast tempo" in Unis 669 mode (?)
					effect[chn] = 0xFF;
					break;

				case 6:
					// G - subcommands (extended)
					switch(m->param)
					{
					case 0:
						// balance fine slide left
						m->param = 0x4F;
						break;
					case 1:
						// balance fine slide right
						m->param = 0xF4;
						break;
					default:
						m->command = CMD_NONE;
					}
					break;
				}
			}
		}

		// Write pattern break
		if(fileHeader.breaks[pat] < 63)
		{
			Patterns[pat].WriteEffect(EffectWriter(CMD_PATTERNBREAK, 0).Row(fileHeader.breaks[pat]).RetryNextRow());
		}
		// And of course the speed...
		Patterns[pat].WriteEffect(EffectWriter(CMD_SPEED, fileHeader.tempoList[pat]).RetryNextRow());
	}

	if(loadFlags & loadSampleData)
	{
		// Reading Samples
		const SampleIO sampleIO(
			SampleIO::_8bit,
			SampleIO::mono,
			SampleIO::littleEndian,
			SampleIO::unsignedPCM);

		for(SAMPLEINDEX n = 1; n <= m_nSamples; n++)
		{
			sampleIO.ReadSample(Samples[n], file);
		}
	}

	return true;
}


OPENMPT_NAMESPACE_END
