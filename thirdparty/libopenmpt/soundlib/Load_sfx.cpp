/*
 * Load_sfx.cpp
 * ------------
 * Purpose: SFX / MMS (SoundFX / MultiMedia Sound) module loader
 * Notes  : Mostly based on the Soundtracker loader, some effect behavior is based on Flod's implementation.
 * Authors: Devin Acker
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#include "stdafx.h"
#include "Loaders.h"
#include "Tables.h"

OPENMPT_NAMESPACE_BEGIN

// File Header
struct SFXFileHeader
{
	uint8be numOrders;
	uint8be restartPos;
	uint8be orderList[128];
};

MPT_BINARY_STRUCT(SFXFileHeader, 130)

// Sample Header
struct SFXSampleHeader
{
	char     name[22];
	char     dummy[2];	// Supposedly sample length, but almost always incorrect
	uint8be  finetune;
	uint8be  volume;
	uint16be loopStart;
	uint16be loopLength;

	// Convert an MOD sample header to OpenMPT's internal sample header.
	void ConvertToMPT(ModSample &mptSmp, uint32 length) const
	{
		mptSmp.Initialize(MOD_TYPE_MOD);
		mptSmp.nLength = length;
		mptSmp.nFineTune = MOD2XMFineTune(finetune);
		mptSmp.nVolume = 4u * std::min<uint8>(volume, 64);

		SmpLength lStart = loopStart;
		SmpLength lLength = loopLength * 2u;

		if(mptSmp.nLength)
		{
			mptSmp.nLoopStart = lStart;
			mptSmp.nLoopEnd = lStart + lLength;

			if(mptSmp.nLoopStart >= mptSmp.nLength)
			{
				mptSmp.nLoopStart = mptSmp.nLength - 1;
			}
			if(mptSmp.nLoopEnd > mptSmp.nLength)
			{
				mptSmp.nLoopEnd = mptSmp.nLength;
			}
			if(mptSmp.nLoopStart > mptSmp.nLoopEnd || mptSmp.nLoopEnd < 4 || mptSmp.nLoopEnd - mptSmp.nLoopStart < 4)
			{
				mptSmp.nLoopStart = 0;
				mptSmp.nLoopEnd = 0;
			}

			if(mptSmp.nLoopEnd > mptSmp.nLoopStart)
			{
				mptSmp.uFlags.set(CHN_LOOP);
			}
		}
	}
};

MPT_BINARY_STRUCT(SFXSampleHeader, 30)

static uint8 ClampSlideParam(uint8 value, uint8 lowNote, uint8 highNote)
{
	uint16 lowPeriod, highPeriod;

	if(lowNote  < highNote &&
	   lowNote  >= 36 + NOTE_MIN &&
	   highNote >= 36 + NOTE_MIN &&
	   lowNote  < CountOf(ProTrackerPeriodTable) + 36 + NOTE_MIN &&
	   highNote < CountOf(ProTrackerPeriodTable) + 36 + NOTE_MIN)
	{
		lowPeriod  = ProTrackerPeriodTable[lowNote - 36 - NOTE_MIN];
		highPeriod = ProTrackerPeriodTable[highNote - 36 - NOTE_MIN];

		// with a fixed speed of 6 ticks/row, and excluding the first row,
		// 1xx/2xx param has a max value of (low-high)/5 to avoid sliding too far
		return std::min<uint8>(value, static_cast<uint8>((lowPeriod - highPeriod) / 5));
	}

	return 0;
}


static bool ValidateHeader(const SFXFileHeader &fileHeader)
{
	if(fileHeader.numOrders > 128)
	{
		return false;
	}
	return true;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderSFX(MemoryFileReader file, const uint64 *pfilesize)
{
	SAMPLEINDEX numSamples = 0;
	if(numSamples == 0)
	{
		file.Rewind();
		if(!file.CanRead(0x40))
		{
			return ProbeWantMoreData;
		}
		if(file.Seek(0x3c) && file.ReadMagic("SONG"))
		{
			numSamples = 15;
		}
	}
	if(numSamples == 0)
	{
		file.Rewind();
		if(!file.CanRead(0x80))
		{
			return ProbeWantMoreData;
		}
		if(file.Seek(0x7c) && file.ReadMagic("SO31"))
		{
			numSamples = 31;
		}
	}
	if(numSamples == 0)
	{
		return ProbeFailure;
	}
	file.Rewind();
	for(SAMPLEINDEX smp = 0; smp < numSamples; smp++)
	{
		if(file.ReadUint32BE() > 131072)
		{
			return ProbeFailure;
		}
	}
	file.Skip(4);
	if(!file.CanRead(2))
	{
		return ProbeWantMoreData;
	}
	uint16 speed = file.ReadUint16BE();
	if(speed < 178)
	{
		return ProbeFailure;
	}
	if(!file.CanRead(sizeof(SFXSampleHeader) * numSamples))
	{
		return ProbeWantMoreData;
	}
	file.Skip(sizeof(SFXSampleHeader) * numSamples);
	SFXFileHeader fileHeader;
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


bool CSoundFile::ReadSFX(FileReader &file, ModLoadingFlags loadFlags)
{
	if(file.Seek(0x3C), file.ReadMagic("SONG"))
	{
		InitializeGlobals(MOD_TYPE_SFX);
		m_nSamples = 15;
	} else if(file.Seek(0x7C), file.ReadMagic("SO31"))
	{
		InitializeGlobals(MOD_TYPE_SFX);
		m_nSamples = 31;
	} else
	{
		return false;
	}

	uint32 sampleLen[31];

	file.Rewind();
	for(SAMPLEINDEX smp = 0; smp < m_nSamples; smp++)
	{
		sampleLen[smp] = file.ReadUint32BE();
		if(sampleLen[smp] > 131072)
			return false;
	}

	m_nChannels = 4;
	m_nInstruments = 0;
	m_nDefaultSpeed = 6;
	m_nMinPeriod = 14 * 4;
	m_nMaxPeriod = 3424 * 4;
	m_nSamplePreAmp = 64;

	// Setup channel pan positions and volume
	SetupMODPanning(true);
	m_playBehaviour.set(kMODIgnorePanning);

	file.Skip(4);
	uint16 speed = file.ReadUint16BE();
	if(speed < 178)
		return false;
	m_nDefaultTempo = TEMPO((14565.0 * 122.0) / speed);

	file.Skip(14);

	uint32 invalidChars = 0;
	for(SAMPLEINDEX smp = 1; smp <= m_nSamples; smp++)
	{
		SFXSampleHeader sampleHeader;

		file.ReadStruct(sampleHeader);
		sampleHeader.ConvertToMPT(Samples[smp], sampleLen[smp - 1]);

		// Get rid of weird characters in sample names.
		for(uint32 i = 0; i < CountOf(sampleHeader.name); i++)
		{
			if(sampleHeader.name[i] > 0 && sampleHeader.name[i] < ' ')
			{
				sampleHeader.name[i] = ' ';
				invalidChars++;
			}
		}
		if(invalidChars >= 128)
			return false;
		mpt::String::Read<mpt::String::spacePadded>(m_szNames[smp], sampleHeader.name);
	}

	SFXFileHeader fileHeader;
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

	PATTERNINDEX numPatterns = 0;
	for(ORDERINDEX ord = 0; ord < fileHeader.numOrders; ord++)
	{
		numPatterns = std::max<PATTERNINDEX>(numPatterns, fileHeader.orderList[ord] + 1u);
	}

	if(fileHeader.restartPos < fileHeader.numOrders)
		Order().SetRestartPos(fileHeader.restartPos);
	else
		Order().SetRestartPos(0);

	ReadOrderFromArray(Order(), fileHeader.orderList, fileHeader.numOrders);

	// SFX v2 / MMS modules have 4 extra bytes here for some reason
	if(m_nSamples == 31)
		file.Skip(4);

	uint8 lastNote[4] = {0};
	uint8 slideTo[4] = {0};
	uint8 slideRate[4] = {0};

	// Reading patterns
	if(loadFlags & loadPatternData)
		Patterns.ResizeArray(numPatterns);
	for(PATTERNINDEX pat = 0; pat < numPatterns; pat++)
	{
		if(!(loadFlags & loadPatternData) || !Patterns.Insert(pat, 64))
		{
			file.Skip(64 * 4 * 4);
			continue;
		}

		for(ROWINDEX row = 0; row < 64; row++)
		{
			PatternRow rowBase = Patterns[pat].GetpModCommand(row, 0);
			for(CHANNELINDEX chn = 0; chn < 4; chn++)
			{
				ModCommand &m = rowBase[chn];
				uint8 data[4];
				file.ReadArray(data);

				if(data[0] == 0xFF)
				{
					lastNote[chn] = slideRate[chn] = 0;

					switch(data[1])
					{
					case 0xFE: // STP (note cut)
						m.command = CMD_VOLUME;
						continue;

					case 0xFD: // PIC (null)
						continue;

					case 0xFC: // BRK (pattern break)
						m.command = CMD_PATTERNBREAK;
						continue;
					}
				}

				ReadMODPatternEntry(data, m);
				if(m.note != NOTE_NONE)
				{
					lastNote[chn] = m.note;
					slideRate[chn] = 0;
				}

				if(m.command || m.param)
				{
					switch(m.command)
					{
					case 0x1: // arpeggio
						m.command = CMD_ARPEGGIO;
						break;

					case 0x2: // portamento (like Ultimate Soundtracker)
						if(m.param & 0xF0)
						{
							m.command = CMD_PORTAMENTODOWN;
							m.param >>= 4;
						} else if(m.param & 0xF)
						{
							m.command = CMD_PORTAMENTOUP;
							m.param &= 0x0F;
						} else
						{
							m.command = m.param = 0;
						}
						break;

					case 0x3: // enable filter/LED
						// give precedence to 7xy/8xy slides
						if(slideRate[chn])
						{
							m.command = m.param = 0;
							break;
						}
						m.command = CMD_MODCMDEX;
						m.param = 0;
						break;

					case 0x4: // disable filter/LED
						// give precedence to 7xy/8xy slides
						if(slideRate[chn])
						{
							m.command = m.param = 0;
							break;
						}
						m.command = CMD_MODCMDEX;
						m.param = 1;
						break;

					case 0x5: // increase volume
						if(m.instr)
						{
							m.command = CMD_VOLUME;
							m.param = std::min(ModCommand::PARAM(0x3F), static_cast<ModCommand::PARAM>((Samples[m.instr].nVolume / 4u) + m.param));

							// give precedence to 7xy/8xy slides (and move this to the volume column)
							if(slideRate[chn])
							{
								m.volcmd = VOLCMD_VOLUME;
								m.vol = m.param;
								m.command = m.param = 0;
								break;
							}
						} else
						{
							m.command = m.param = 0;
						}
						break;

					case 0x6: // decrease volume
						if(m.instr)
						{
							m.command = CMD_VOLUME;
							if((Samples[m.instr].nVolume / 4u) >= m.param)
								m.param = static_cast<ModCommand::PARAM>(Samples[m.instr].nVolume / 4u) - m.param;
							else
								m.param = 0;

							// give precedence to 7xy/8xy slides (and move this to the volume column)
							if(slideRate[chn])
							{
								m.volcmd = VOLCMD_VOLUME;
								m.vol = m.param;
								m.command = m.param = 0;
								break;
							}
						} else
						{
							m.command = m.param = 0;
						}
						break;

					case 0x7: // 7xy: slide down x semitones at speed y
						slideTo[chn] = lastNote[chn] - (m.param >> 4);

						m.command = CMD_PORTAMENTODOWN;
						slideRate[chn] = m.param & 0xF;
						m.param = ClampSlideParam(slideRate[chn], slideTo[chn], lastNote[chn]);
						break;

					case 0x8: // 8xy: slide up x semitones at speed y
						slideTo[chn] = lastNote[chn] + (m.param >> 4);

						m.command = CMD_PORTAMENTOUP;
						slideRate[chn] = m.param & 0xF;
						m.param = ClampSlideParam(slideRate[chn], lastNote[chn], slideTo[chn]);
						break;

					default:
						m.command = CMD_NONE;
						break;
					}
				}

				// continue 7xy/8xy slides if needed
				if(m.command == CMD_NONE && slideRate[chn])
				{
					if(slideTo[chn])
					{
						m.note = lastNote[chn] = slideTo[chn];
						m.param = slideRate[chn];
						slideTo[chn] = 0;
					}
					m.command = CMD_TONEPORTAMENTO;
				}
			}
		}
	}

	// Reading samples
	if(loadFlags & loadSampleData)
	{
		for(SAMPLEINDEX smp = 1; smp <= m_nSamples; smp++) if(Samples[smp].nLength)
		{
			SampleIO(
				SampleIO::_8bit,
				SampleIO::mono,
				SampleIO::littleEndian,
				SampleIO::signedPCM)
				.ReadSample(Samples[smp], file);
		}
	}

	return true;
}

OPENMPT_NAMESPACE_END
