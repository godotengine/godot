/*
 * Load_plm.cpp
 * ------------
 * Purpose: PLM (Disorder Tracker 2) module loader
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"


OPENMPT_NAMESPACE_BEGIN

struct PLMFileHeader
{
	char     magic[4];		// "PLM\x1A"
	uint8le  headerSize;	// Number of bytes in header, including magic bytes
	uint8le  version;		// version code of file format (0x10)
	char     songName[48];
	uint8le  numChannels;
	uint8le  flags;			// unused?
	uint8le  maxVol;		// Maximum volume for vol slides, normally 0x40
	uint8le  amplify;		// SoundBlaster amplify, 0x40 = no amplify
	uint8le  tempo;
	uint8le  speed;
	uint8le  panPos[32];	// 0...15
	uint8le  numSamples;
	uint8le  numPatterns;
	uint16le numOrders;
};

MPT_BINARY_STRUCT(PLMFileHeader, 96)


struct PLMSampleHeader
{
	enum SampleFlags
	{
		smp16Bit = 1,
		smpPingPong = 2,
	};

	char     magic[4];		// "PLS\x1A"
	uint8le  headerSize;	// Number of bytes in header, including magic bytes
	uint8le  version;	
	char     name[32];
	char     filename[12];
	uint8le  panning;		// 0...15, 255 = no pan
	uint8le  volume;		// 0...64
	uint8le  flags;			// See SampleFlags
	uint16le sampleRate;
	char     unused[4];
	uint32le loopStart;
	uint32le loopEnd;
	uint32le length;
};

MPT_BINARY_STRUCT(PLMSampleHeader, 71)


struct PLMPatternHeader
{
	uint32le size;
	uint8le  numRows;
	uint8le  numChannels;
	uint8le  color;
	char     name[25];
};

MPT_BINARY_STRUCT(PLMPatternHeader, 32)


struct PLMOrderItem
{
	uint16le x;		// Starting position of pattern
	uint8le  y;		// Number of first channel
	uint8le  pattern;
};

MPT_BINARY_STRUCT(PLMOrderItem, 4)


static bool ValidateHeader(const PLMFileHeader &fileHeader)
{
	if(std::memcmp(fileHeader.magic, "PLM\x1A", 4)
		|| fileHeader.version != 0x10
		|| fileHeader.numChannels == 0 || fileHeader.numChannels > 32
		|| fileHeader.headerSize < sizeof(PLMFileHeader)
		)
	{
		return false;
	}
	return true;
}


static uint64 GetHeaderMinimumAdditionalSize(const PLMFileHeader &fileHeader)
{
	return fileHeader.headerSize - sizeof(PLMFileHeader) + 4 * (fileHeader.numOrders + fileHeader.numPatterns + fileHeader.numSamples);
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderPLM(MemoryFileReader file, const uint64 *pfilesize)
{
	PLMFileHeader fileHeader;
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


bool CSoundFile::ReadPLM(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();

	PLMFileHeader fileHeader;
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

	if(!file.Seek(fileHeader.headerSize))
	{
		return false;
	}

	InitializeGlobals(MOD_TYPE_PLM);
	InitializeChannels();
	m_SongFlags = SONG_ITOLDEFFECTS;
	m_madeWithTracker = MPT_USTRING("Disorder Tracker 2");
	// Some PLMs use ASCIIZ, some space-padding strings...weird. Oh, and the file browser stops at 0 bytes in the name, the main GUI doesn't.
	mpt::String::Read<mpt::String::spacePadded>(m_songName, fileHeader.songName);
	m_nChannels = fileHeader.numChannels + 1;	// Additional channel for writing pattern breaks
	m_nSamplePreAmp = fileHeader.amplify;
	m_nDefaultTempo.Set(fileHeader.tempo);
	m_nDefaultSpeed = fileHeader.speed;
	for(CHANNELINDEX chn = 0; chn < fileHeader.numChannels; chn++)
	{
		ChnSettings[chn].nPan = fileHeader.panPos[chn] * 0x11;
	}
	m_nSamples = fileHeader.numSamples;

	std::vector<PLMOrderItem> order(fileHeader.numOrders);
	file.ReadVector(order, fileHeader.numOrders);

	std::vector<uint32le> patternPos, samplePos;
	file.ReadVector(patternPos, fileHeader.numPatterns);
	file.ReadVector(samplePos, fileHeader.numSamples);

	for(SAMPLEINDEX smp = 0; smp < fileHeader.numSamples; smp++)
	{
		ModSample &sample = Samples[smp + 1];
		sample.Initialize();

		PLMSampleHeader sampleHeader;
		if(samplePos[smp] == 0
			|| !file.Seek(samplePos[smp])
			|| !file.ReadStruct(sampleHeader))
				continue;

		mpt::String::Read<mpt::String::maybeNullTerminated>(m_szNames[smp + 1], sampleHeader.name);
		mpt::String::Read<mpt::String::maybeNullTerminated>(sample.filename, sampleHeader.filename);
		if(sampleHeader.panning <= 15)
		{
			sample.uFlags.set(CHN_PANNING);
			sample.nPan = sampleHeader.panning * 0x11;
		}
		sample.nGlobalVol = std::min<uint8>(sampleHeader.volume, 64);
		sample.nC5Speed = sampleHeader.sampleRate;
		sample.nLoopStart = sampleHeader.loopStart;
		sample.nLoopEnd = sampleHeader.loopEnd;
		sample.nLength = sampleHeader.length;
		if(sampleHeader.flags & PLMSampleHeader::smp16Bit)
		{
			sample.nLoopStart /= 2;
			sample.nLoopEnd /= 2;
			sample.nLength /= 2;
		}
		if(sample.nLoopEnd > sample.nLoopStart)
		{
			sample.uFlags.set(CHN_LOOP);
			if(sampleHeader.flags & PLMSampleHeader::smpPingPong) sample.uFlags.set(CHN_PINGPONGLOOP);
		}
		sample.SanitizeLoops();
		
		if(loadFlags & loadSampleData)
		{
			file.Seek(samplePos[smp] + sampleHeader.headerSize);
			SampleIO(
				(sampleHeader.flags & PLMSampleHeader::smp16Bit) ? SampleIO::_16bit : SampleIO::_8bit,
				SampleIO::mono,
				SampleIO::littleEndian,
				SampleIO::unsignedPCM)
				.ReadSample(sample, file);
		}
	}

	if(!(loadFlags & loadPatternData))
	{
		return true;
	}

	// PLM is basically one huge continuous pattern, so we split it up into smaller patterns.
	const ROWINDEX rowsPerPat = 64;
	uint32 maxPos = 0;

	static const ModCommand::COMMAND effTrans[] =
	{
		CMD_NONE,
		CMD_PORTAMENTOUP,
		CMD_PORTAMENTODOWN,
		CMD_TONEPORTAMENTO,
		CMD_VOLUMESLIDE,
		CMD_TREMOLO,
		CMD_VIBRATO,
		CMD_S3MCMDEX,		// Tremolo Waveform
		CMD_S3MCMDEX,		// Vibrato Waveform
		CMD_TEMPO,
		CMD_SPEED,
		CMD_POSITIONJUMP,	// Jump to order
		CMD_POSITIONJUMP,	// Break to end of this order
		CMD_OFFSET,
		CMD_S3MCMDEX,		// GUS Panning
		CMD_RETRIG,
		CMD_S3MCMDEX,		// Note Delay
		CMD_S3MCMDEX,		// Note Cut
		CMD_S3MCMDEX,		// Pattern Delay
		CMD_FINEVIBRATO,
		CMD_VIBRATOVOL,
		CMD_TONEPORTAVOL,
		CMD_OFFSETPERCENTAGE,
	};

	Order().clear();
	for(const auto &ord : order)
	{
		if(ord.pattern >= fileHeader.numPatterns
			|| ord.y > fileHeader.numChannels
			|| !file.Seek(patternPos[ord.pattern])) continue;

		PLMPatternHeader patHeader;
		file.ReadStruct(patHeader);
		if(!patHeader.numRows) continue;

		STATIC_ASSERT(ORDERINDEX_MAX >= (MPT_MAX_UNSIGNED_VALUE(ord.x) + 255) / rowsPerPat);
		ORDERINDEX curOrd = static_cast<ORDERINDEX>(ord.x / rowsPerPat);
		ROWINDEX curRow = static_cast<ROWINDEX>(ord.x % rowsPerPat);
		const CHANNELINDEX numChannels = std::min<uint8>(patHeader.numChannels, fileHeader.numChannels - ord.y);
		const uint32 patternEnd = ord.x + patHeader.numRows;
		maxPos = std::max(maxPos, patternEnd);

		ModCommand::NOTE lastNote[32] = { 0 };
		for(ROWINDEX r = 0; r < patHeader.numRows; r++, curRow++)
		{
			if(curRow >= rowsPerPat)
			{
				curRow = 0;
				curOrd++;
			}
			if(curOrd >= Order().size())
			{
				Order().resize(curOrd + 1);
				Order()[curOrd] = Patterns.InsertAny(rowsPerPat);
			}
			PATTERNINDEX pat = Order()[curOrd];
			if(!Patterns.IsValidPat(pat)) break;

			ModCommand *m = Patterns[pat].GetpModCommand(curRow, ord.y);
			for(CHANNELINDEX c = 0; c < numChannels; c++, m++)
			{
				uint8 data[5];
				file.ReadArray(data);
				if(data[0])
					lastNote[c] = m->note = (data[0] >> 4) * 12 + (data[0] & 0x0F) + 12 + NOTE_MIN;
				else
					m->note = NOTE_NONE;
				m->instr = data[1];
				m->volcmd = VOLCMD_VOLUME;
				if(data[2] != 0xFF)
					m->vol = data[2];
				else
					m->volcmd = VOLCMD_NONE;

				if(data[3] < CountOf(effTrans))
				{
					m->command = effTrans[data[3]];
					m->param = data[4];
					// Fix some commands
					switch(data[3])
					{
					case 0x07:	// Tremolo waveform
						m->param = 0x40 | (m->param & 0x03);
						break;
					case 0x08:	// Vibrato waveform
						m->param = 0x30 | (m->param & 0x03);
						break;
					case 0x0B:	// Jump to order
						if(m->param < order.size())
						{
							uint16 target = order[m->param].x;
							m->param = static_cast<ModCommand::PARAM>(target / rowsPerPat);
							ModCommand *mBreak = Patterns[pat].GetpModCommand(curRow, m_nChannels - 1);
							mBreak->command = CMD_PATTERNBREAK;
							mBreak->param = static_cast<ModCommand::PARAM>(target % rowsPerPat);
						}
						break;
					case 0x0C:	// Jump to end of order
						{
							m->param = static_cast<ModCommand::PARAM>(patternEnd / rowsPerPat);
							ModCommand *mBreak = Patterns[pat].GetpModCommand(curRow, m_nChannels - 1);
							mBreak->command = CMD_PATTERNBREAK;
							mBreak->param = static_cast<ModCommand::PARAM>(patternEnd % rowsPerPat);
						}
						break;
					case 0x0E:	// GUS Panning
						m->param = 0x80 | (m->param & 0x0F);
						break;
					case 0x10:	// Delay Note
						m->param = 0xD0 | std::min<ModCommand::PARAM>(m->param, 0x0F);
						break;
					case 0x11:	// Cut Note
						m->param = 0xC0 | std::min<ModCommand::PARAM>(m->param, 0x0F);
						break;
					case 0x12:	// Pattern Delay
						m->param = 0xE0 | std::min<ModCommand::PARAM>(m->param, 0x0F);
						break;
					case 0x04:	// Volume Slide
					case 0x14:	// Vibrato + Volume Slide
					case 0x15:	// Tone Portamento + Volume Slide
						// If both nibbles of a volume slide are set, act as fine volume slide up
						if((m->param & 0xF0) && (m->param & 0x0F) && (m->param & 0xF0) != 0xF0)
						{
							m->param |= 0x0F;
						}
						break;
					case 0x0D:
					case 0x16:
						// Offset without note
						if(m->note == NOTE_NONE)
						{
							m->note = lastNote[c];
						}
						break;
					}
				}
			}
			if(patHeader.numChannels > numChannels)
			{
				file.Skip(5 * (patHeader.numChannels - numChannels));
			}
		}
	}
	// Module ends with the last row of the last order item
	ROWINDEX endPatSize = maxPos % rowsPerPat;
	ORDERINDEX endOrder = static_cast<ORDERINDEX>(maxPos / rowsPerPat);
	if(endPatSize > 0 && Order().IsValidPat(endOrder))
	{
		Patterns[Order()[endOrder]].Resize(endPatSize, false);
	}
	// If there are still any non-existent patterns in our order list, insert some blank patterns.
	PATTERNINDEX blankPat = PATTERNINDEX_INVALID;
	for(auto &pat : Order())
	{
		if(pat == Order.GetInvalidPatIndex())
		{
			if(blankPat == PATTERNINDEX_INVALID)
			{
				blankPat = Patterns.InsertAny(rowsPerPat);
			}
			pat = blankPat;
		}
	}

	return true;
}

OPENMPT_NAMESPACE_END
