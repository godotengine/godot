/*
 * load_j2b.cpp
 * ------------
 * Purpose: RIFF AM and RIFF AMFF (Galaxy Sound System) module loader
 * Notes  : J2B is a compressed variant of RIFF AM and RIFF AMFF files used in Jazz Jackrabbit 2.
 *          It seems like no other game used the AM(FF) format.
 *          RIFF AM is the newer version of the format, generally following the RIFF "standard" closely.
 * Authors: Johannes Schultz (OpenMPT port, reverse engineering + loader implementation of the instrument format)
 *          Chris Moeller (foo_dumb - this is almost a complete port of his code, thanks)
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "ChunkReader.h"

#if defined(MPT_WITH_ZLIB)
#include <zlib.h>
#elif defined(MPT_WITH_MINIZ)
#include <miniz/miniz.h>
#endif


OPENMPT_NAMESPACE_BEGIN


// First off, a nice vibrato translation LUT.
static const uint8 j2bAutoVibratoTrans[] = 
{
	VIB_SINE, VIB_SQUARE, VIB_RAMP_UP, VIB_RAMP_DOWN, VIB_RANDOM,
};


// header for compressed j2b files
struct J2BFileHeader
{
	// Magic Bytes
	// 32-Bit J2B header identifiers
	static const uint32 magicDEADBEAF = 0xAFBEADDEu;
	static const uint32 magicDEADBABE = 0xBEBAADDEu;

	char     signature[4];		// MUSE
	uint32le deadbeaf;			// 0xDEADBEAF (AM) or 0xDEADBABE (AMFF)
	uint32le fileLength;		// complete filesize
	uint32le crc32;				// checksum of the compressed data block
	uint32le packedLength;		// length of the compressed data block
	uint32le unpackedLength;	// length of the decompressed module
};

MPT_BINARY_STRUCT(J2BFileHeader, 24)


// AM(FF) stuff

struct AMFFRiffChunk
{
	// 32-Bit chunk identifiers
	enum ChunkIdentifiers
	{
		idRIFF	= MAGIC4LE('R','I','F','F'),
		idAMFF	= MAGIC4LE('A','M','F','F'),
		idAM__	= MAGIC4LE('A','M',' ',' '),
		idMAIN	= MAGIC4LE('M','A','I','N'),
		idINIT	= MAGIC4LE('I','N','I','T'),
		idORDR	= MAGIC4LE('O','R','D','R'),
		idPATT	= MAGIC4LE('P','A','T','T'),
		idINST	= MAGIC4LE('I','N','S','T'),
		idSAMP	= MAGIC4LE('S','A','M','P'),
		idAI__	= MAGIC4LE('A','I',' ',' '),
		idAS__	= MAGIC4LE('A','S',' ',' '),
	};

	uint32le id;		// See ChunkIdentifiers
	uint32le length;	// Chunk size without header

	size_t GetLength() const
	{
		return length;
	}

	ChunkIdentifiers GetID() const
	{
		return static_cast<ChunkIdentifiers>(id.get());
	}
};

MPT_BINARY_STRUCT(AMFFRiffChunk, 8)


// This header is used for both AM's "INIT" as well as AMFF's "MAIN" chunk
struct AMFFMainChunk
{
	// Main Chunk flags
	enum MainFlags
	{
		amigaSlides = 0x01,
	};

	char     songname[64];
	uint8le  flags;
	uint8le  channels;
	uint8le  speed;
	uint8le  tempo;
	uint16le minPeriod;	// 16x Amiga periods, but we should ignore them - otherwise some high notes in Medivo.j2b won't sound correct.
	uint16le maxPeriod;	// Ditto
	uint8le  globalvolume;
};

MPT_BINARY_STRUCT(AMFFMainChunk, 73)


// AMFF instrument envelope (old format)
struct AMFFEnvelope
{
	// Envelope flags (also used for RIFF AM)
	enum EnvelopeFlags
	{
		envEnabled	= 0x01,
		envSustain	= 0x02,
		envLoop		= 0x04,
	};

	struct EnvPoint
	{
		uint16le tick;
		uint8le  value;	// 0...64
	};

	uint8le envFlags;			// high nibble = pan env flags, low nibble = vol env flags (both nibbles work the same way)
	uint8le envNumPoints;		// high nibble = pan env length, low nibble = vol env length
	uint8le envSustainPoints;	// you guessed it... high nibble = pan env sustain point, low nibble = vol env sustain point
	uint8le envLoopStarts;		// I guess you know the pattern now.
	uint8le envLoopEnds;		// same here.
	EnvPoint volEnv[10];
	EnvPoint panEnv[10];

	// Convert weird envelope data to OpenMPT's internal format.
	void ConvertEnvelope(uint8 flags, uint8 numPoints, uint8 sustainPoint, uint8 loopStart, uint8 loopEnd, const EnvPoint (&points)[10], InstrumentEnvelope &mptEnv) const
	{
		// The buggy mod2j2b converter will actually NOT limit this to 10 points if the envelope is longer.
		mptEnv.resize(std::min(numPoints, static_cast<uint8>(10)));

		mptEnv.nSustainStart = mptEnv.nSustainEnd = sustainPoint;

		mptEnv.nLoopStart = loopStart;
		mptEnv.nLoopEnd = loopEnd;

		for(uint32 i = 0; i < mptEnv.size(); i++)
		{
			mptEnv[i].tick = points[i].tick >> 4;
			if(i == 0)
				mptEnv[0].tick = 0;
			else if(mptEnv[i].tick < mptEnv[i - 1].tick)
				mptEnv[i].tick = mptEnv[i - 1].tick + 1;

			mptEnv[i].value = Clamp<uint8, uint8>(points[i].value, 0, 64);
		}

		mptEnv.dwFlags.set(ENV_ENABLED, (flags & AMFFEnvelope::envEnabled) != 0);
		mptEnv.dwFlags.set(ENV_SUSTAIN, (flags & AMFFEnvelope::envSustain) && mptEnv.nSustainStart <= mptEnv.size());
		mptEnv.dwFlags.set(ENV_LOOP, (flags & AMFFEnvelope::envLoop) && mptEnv.nLoopStart <= mptEnv.nLoopEnd && mptEnv.nLoopStart <= mptEnv.size());
	}

	void ConvertToMPT(ModInstrument &mptIns) const
	{
		// interleaved envelope data... meh. gotta split it up here and decode it separately.
		// note: mod2j2b is BUGGY and always writes ($original_num_points & 0x0F) in the header,
		// but just has room for 10 envelope points. That means that long (>= 16 points)
		// envelopes are cut off, and envelopes have to be trimmed to 10 points, even if
		// the header claims that they are longer.
		// For XM files the number of points also appears to be off by one,
		// but luckily there are no official J2Bs using envelopes anyway.
		ConvertEnvelope(envFlags & 0x0F, envNumPoints & 0x0F, envSustainPoints & 0x0F, envLoopStarts & 0x0F, envLoopEnds & 0x0F, volEnv, mptIns.VolEnv);
		ConvertEnvelope(envFlags >> 4, envNumPoints >> 4, envSustainPoints >> 4, envLoopStarts >> 4, envLoopEnds >> 4, panEnv, mptIns.PanEnv);
	}
};

MPT_BINARY_STRUCT(AMFFEnvelope::EnvPoint, 3)
MPT_BINARY_STRUCT(AMFFEnvelope, 65)


// AMFF instrument header (old format)
struct AMFFInstrumentHeader
{
	uint8le  unknown;		// 0x00
	uint8le  index;			// actual instrument number
	char     name[28];
	uint8le  numSamples;
	uint8le  sampleMap[120];
	uint8le  vibratoType;
	uint16le vibratoSweep;
	uint16le vibratoDepth;
	uint16le vibratoRate;
	AMFFEnvelope envelopes;
	uint16le fadeout;

	// Convert instrument data to OpenMPT's internal format.
	void ConvertToMPT(ModInstrument &mptIns, SAMPLEINDEX baseSample)
	{
		mpt::String::Read<mpt::String::maybeNullTerminated>(mptIns.name, name);

		STATIC_ASSERT(CountOf(sampleMap) <= CountOf(mptIns.Keyboard));
		for(size_t i = 0; i < CountOf(sampleMap); i++)
		{
			mptIns.Keyboard[i] = sampleMap[i] + baseSample + 1;
		}

		mptIns.nFadeOut = fadeout << 5;
		envelopes.ConvertToMPT(mptIns);
	}

};

MPT_BINARY_STRUCT(AMFFInstrumentHeader, 225)


// AMFF sample header (old format)
struct AMFFSampleHeader
{
	// Sample flags (also used for RIFF AM)
	enum SampleFlags
	{
		smp16Bit	= 0x04,
		smpLoop		= 0x08,
		smpPingPong	= 0x10,
		smpPanning	= 0x20,
		smpExists	= 0x80,
		// some flags are still missing... what is e.g. 0x8000?
	};

	uint32le id;		// "SAMP"
	uint32le chunkSize;	// header + sample size
	char     name[28];
	uint8le  pan;
	uint8le  volume;
	uint16le flags;
	uint32le length;
	uint32le loopStart;
	uint32le loopEnd;
	uint32le sampleRate;
	uint32le reserved1;
	uint32le reserved2;

	// Convert sample header to OpenMPT's internal format.
	void ConvertToMPT(AMFFInstrumentHeader &instrHeader, ModSample &mptSmp) const
	{
		mptSmp.Initialize();
		mptSmp.nPan = pan * 4;
		mptSmp.nVolume = volume * 4;
		mptSmp.nGlobalVol = 64;
		mptSmp.nLength = length;
		mptSmp.nLoopStart = loopStart;
		mptSmp.nLoopEnd = loopEnd;
		mptSmp.nC5Speed = sampleRate;

		if(instrHeader.vibratoType < CountOf(j2bAutoVibratoTrans))
			mptSmp.nVibType = j2bAutoVibratoTrans[instrHeader.vibratoType];
		mptSmp.nVibSweep = static_cast<uint8>(instrHeader.vibratoSweep);
		mptSmp.nVibRate = static_cast<uint8>(instrHeader.vibratoRate / 16);
		mptSmp.nVibDepth = static_cast<uint8>(instrHeader.vibratoDepth / 4);
		if((mptSmp.nVibRate | mptSmp.nVibDepth) != 0)
		{
			// Convert XM-style vibrato sweep to IT
			mptSmp.nVibSweep = 255 - mptSmp.nVibSweep;
		}

		if(flags & AMFFSampleHeader::smp16Bit)
			mptSmp.uFlags.set(CHN_16BIT);
		if(flags & AMFFSampleHeader::smpLoop)
			mptSmp.uFlags.set(CHN_LOOP);
		if(flags & AMFFSampleHeader::smpPingPong)
			mptSmp.uFlags.set(CHN_PINGPONGLOOP);
		if(flags & AMFFSampleHeader::smpPanning)
			mptSmp.uFlags.set(CHN_PANNING);
	}

	// Retrieve the internal sample format flags for this sample.
	SampleIO GetSampleFormat() const
	{
		return SampleIO(
			(flags & AMFFSampleHeader::smp16Bit) ? SampleIO::_16bit : SampleIO::_8bit,
			SampleIO::mono,
			SampleIO::littleEndian,
			SampleIO::signedPCM);
	}
};

MPT_BINARY_STRUCT(AMFFSampleHeader, 64)


// AM instrument envelope (new format)
struct AMEnvelope
{
	struct EnvPoint
	{
		uint16le tick;
		uint16le value;
	};

	uint16le flags;
	uint8le  numPoints;		// actually, it's num. points - 1, and 0xFF if there is no envelope
	uint8le  sustainPoint;
	uint8le  loopStart;
	uint8le  loopEnd;
	EnvPoint values[10];
	uint16le fadeout;		// why is this here? it's only needed for the volume envelope...

	// Convert envelope data to OpenMPT's internal format.
	void ConvertToMPT(InstrumentEnvelope &mptEnv, EnvelopeType envType) const
	{
		if(numPoints == 0xFF || numPoints == 0)
			return;

		mptEnv.resize(std::min(numPoints + 1, 10));

		mptEnv.nSustainStart = mptEnv.nSustainEnd = sustainPoint;

		mptEnv.nLoopStart = loopStart;
		mptEnv.nLoopEnd = loopEnd;

		for(uint32 i = 0; i < mptEnv.size(); i++)
		{
			mptEnv[i].tick = values[i].tick >> 4;
			if(i == 0)
				mptEnv[i].tick = 0;
			else if(mptEnv[i].tick < mptEnv[i - 1].tick)
				mptEnv[i].tick = mptEnv[i - 1].tick + 1;

			const uint16 val = values[i].value;
			switch(envType)
			{
			case ENV_VOLUME:	// 0....32767
			default:
				mptEnv[i].value = (uint8)((val + 1) >> 9);
				break;
			case ENV_PITCH:		// -4096....4096
				mptEnv[i].value = (uint8)((((int16)val) + 0x1001) >> 7);
				break;
			case ENV_PANNING:	// -32768...32767
				mptEnv[i].value = (uint8)((((int16)val) + 0x8001) >> 10);
				break;
			}
			Limit(mptEnv[i].value, uint8(ENVELOPE_MIN), uint8(ENVELOPE_MAX));
		}

		mptEnv.dwFlags.set(ENV_ENABLED, (flags & AMFFEnvelope::envEnabled) != 0);
		mptEnv.dwFlags.set(ENV_SUSTAIN, (flags & AMFFEnvelope::envSustain) && mptEnv.nSustainStart <= mptEnv.size());
		mptEnv.dwFlags.set(ENV_LOOP, (flags & AMFFEnvelope::envLoop) && mptEnv.nLoopStart <= mptEnv.nLoopEnd && mptEnv.nLoopStart <= mptEnv.size());
	}
};

MPT_BINARY_STRUCT(AMEnvelope::EnvPoint, 4)
MPT_BINARY_STRUCT(AMEnvelope, 48)


// AM instrument header (new format)
struct AMInstrumentHeader
{
	uint32le headSize;	// Header size (i.e. the size of this struct)
	uint8le  unknown1;	// 0x00
	uint8le  index;		// Actual instrument number
	char     name[32];
	uint8le  sampleMap[128];
	uint8le  vibratoType;
	uint16le vibratoSweep;
	uint16le vibratoDepth;
	uint16le vibratoRate;
	uint8le  unknown2[7];
	AMEnvelope volEnv;
	AMEnvelope pitchEnv;
	AMEnvelope panEnv;
	uint16le numSamples;

	// Convert instrument data to OpenMPT's internal format.
	void ConvertToMPT(ModInstrument &mptIns, SAMPLEINDEX baseSample)
	{
		mpt::String::Read<mpt::String::maybeNullTerminated>(mptIns.name, name);

		STATIC_ASSERT(CountOf(sampleMap) <= CountOf(mptIns.Keyboard));
		for(uint8 i = 0; i < CountOf(sampleMap); i++)
		{
			mptIns.Keyboard[i] = sampleMap[i] + baseSample + 1;
		}

		mptIns.nFadeOut = volEnv.fadeout << 5;

		volEnv.ConvertToMPT(mptIns.VolEnv, ENV_VOLUME);
		pitchEnv.ConvertToMPT(mptIns.PitchEnv, ENV_PITCH);
		panEnv.ConvertToMPT(mptIns.PanEnv, ENV_PANNING);

		if(numSamples == 0)
		{
			MemsetZero(mptIns.Keyboard);
		}
	}
};

MPT_BINARY_STRUCT(AMInstrumentHeader, 326)


// AM sample header (new format)
struct AMSampleHeader
{
	uint32le headSize;		// Header size (i.e. the size of this struct), apparently not including headSize.
	char     name[32];
	uint16le pan;
	uint16le volume;
	uint16le flags;
	uint16le unknown;		// 0x0000 / 0x0080?
	uint32le length;
	uint32le loopStart;
	uint32le loopEnd;
	uint32le sampleRate;

	// Convert sample header to OpenMPT's internal format.
	void ConvertToMPT(AMInstrumentHeader &instrHeader, ModSample &mptSmp) const
	{
		mptSmp.Initialize();
		mptSmp.nPan = std::min<uint16>(pan, 32767) * 256 / 32767;
		mptSmp.nVolume = std::min<uint16>(volume, 32767) * 256 / 32767;
		mptSmp.nGlobalVol = 64;
		mptSmp.nLength = length;
		mptSmp.nLoopStart = loopStart;
		mptSmp.nLoopEnd = loopEnd;
		mptSmp.nC5Speed = sampleRate;

		if(instrHeader.vibratoType < CountOf(j2bAutoVibratoTrans))
			mptSmp.nVibType = j2bAutoVibratoTrans[instrHeader.vibratoType];
		mptSmp.nVibSweep = static_cast<uint8>(instrHeader.vibratoSweep);
		mptSmp.nVibRate = static_cast<uint8>(instrHeader.vibratoRate / 16);
		mptSmp.nVibDepth = static_cast<uint8>(instrHeader.vibratoDepth / 4);
		if((mptSmp.nVibRate | mptSmp.nVibDepth) != 0)
		{
			// Convert XM-style vibrato sweep to IT
			mptSmp.nVibSweep = 255 - mptSmp.nVibSweep;
		}

		if(flags & AMFFSampleHeader::smp16Bit)
			mptSmp.uFlags.set(CHN_16BIT);
		if(flags & AMFFSampleHeader::smpLoop)
			mptSmp.uFlags.set(CHN_LOOP);
		if(flags & AMFFSampleHeader::smpPingPong)
			mptSmp.uFlags.set(CHN_PINGPONGLOOP);
		if(flags & AMFFSampleHeader::smpPanning)
			mptSmp.uFlags.set(CHN_PANNING);
	}

	// Retrieve the internal sample format flags for this sample.
	SampleIO GetSampleFormat() const
	{
		return SampleIO(
			(flags & AMFFSampleHeader::smp16Bit) ? SampleIO::_16bit : SampleIO::_8bit,
			SampleIO::mono,
			SampleIO::littleEndian,
			SampleIO::signedPCM);
	}
};

MPT_BINARY_STRUCT(AMSampleHeader, 60)


// Convert RIFF AM(FF) pattern data to MPT pattern data.
static bool ConvertAMPattern(FileReader chunk, PATTERNINDEX pat, bool isAM, CSoundFile &sndFile)
{
	// Effect translation LUT
	static const EffectCommand amEffTrans[] =
	{
		CMD_ARPEGGIO, CMD_PORTAMENTOUP, CMD_PORTAMENTODOWN, CMD_TONEPORTAMENTO,
		CMD_VIBRATO, CMD_TONEPORTAVOL, CMD_VIBRATOVOL, CMD_TREMOLO,
		CMD_PANNING8, CMD_OFFSET, CMD_VOLUMESLIDE, CMD_POSITIONJUMP,
		CMD_VOLUME, CMD_PATTERNBREAK, CMD_MODCMDEX, CMD_TEMPO,
		CMD_GLOBALVOLUME, CMD_GLOBALVOLSLIDE, CMD_KEYOFF, CMD_SETENVPOSITION,
		CMD_CHANNELVOLUME, CMD_CHANNELVOLSLIDE, CMD_PANNINGSLIDE, CMD_RETRIG,
		CMD_TREMOR, CMD_XFINEPORTAUPDOWN,
	};

	enum
	{
		rowDone		= 0,		// Advance to next row
		channelMask	= 0x1F,		// Mask for retrieving channel information
		volFlag		= 0x20,		// Volume effect present
		noteFlag	= 0x40,		// Note + instr present
		effectFlag	= 0x80,		// Effect information present
		dataFlag	= 0xE0,		// Channel data present
	};

	if(chunk.NoBytesLeft())
	{
		return false;
	}

	ROWINDEX numRows = Clamp(static_cast<ROWINDEX>(chunk.ReadUint8()) + 1, ROWINDEX(1), MAX_PATTERN_ROWS);

	if(!sndFile.Patterns.Insert(pat, numRows))
		return false;

	const CHANNELINDEX channels = sndFile.GetNumChannels();
	if(channels == 0)
		return false;

	PatternRow rowBase = sndFile.Patterns[pat].GetRow(0);
	ROWINDEX row = 0;

	while(row < numRows && chunk.CanRead(1))
	{
		const uint8 flags = chunk.ReadUint8();

		if(flags == rowDone)
		{
			row++;
			rowBase = sndFile.Patterns[pat].GetRow(row);
			continue;
		}

		ModCommand &m = rowBase[std::min<CHANNELINDEX>((flags & channelMask), channels - 1)];

		if(flags & dataFlag)
		{
			if(flags & effectFlag) // effect
			{
				m.param = chunk.ReadUint8();
				uint8 command = chunk.ReadUint8();

				if(command < CountOf(amEffTrans))
				{
					// command translation
					m.command = amEffTrans[command];
				} else
				{
#ifdef DEBUG
					Log(mpt::format("J2B: Unknown command: 0x%1, param 0x%2")(mpt::fmt::HEX0<2>(command), mpt::fmt::HEX0<2>(m.param)));
#endif // DEBUG
					m.command = CMD_NONE;
				}

				// Handling special commands
				switch(m.command)
				{
				case CMD_ARPEGGIO:
					if(m.param == 0) m.command = CMD_NONE;
					break;
				case CMD_VOLUME:
					if(m.volcmd == VOLCMD_NONE)
					{
						m.volcmd = VOLCMD_VOLUME;
						m.vol = Clamp(m.param, uint8(0), uint8(64));
						m.command = CMD_NONE;
						m.param = 0;
					}
					break;
				case CMD_TONEPORTAVOL:
				case CMD_VIBRATOVOL:
				case CMD_VOLUMESLIDE:
				case CMD_GLOBALVOLSLIDE:
				case CMD_PANNINGSLIDE:
					if (m.param & 0xF0) m.param &= 0xF0;
					break;
				case CMD_PANNING8:
					if(m.param <= 0x80) m.param = MIN(m.param << 1, 0xFF);
					else if(m.param == 0xA4) {m.command = CMD_S3MCMDEX; m.param = 0x91;}
					break;
				case CMD_PATTERNBREAK:
					m.param = ((m.param >> 4) * 10) + (m.param & 0x0F);
					break;
				case CMD_MODCMDEX:
					m.ExtendedMODtoS3MEffect();
					break;
				case CMD_TEMPO:
					if(m.param <= 0x1F) m.command = CMD_SPEED;
					break;
				case CMD_XFINEPORTAUPDOWN:
					switch(m.param & 0xF0)
					{
					case 0x10:
						m.command = CMD_PORTAMENTOUP;
						break;
					case 0x20:
						m.command = CMD_PORTAMENTODOWN;
						break;
					}
					m.param = (m.param & 0x0F) | 0xE0;
					break;
				}
			}

			if (flags & noteFlag) // note + ins
			{
				m.instr = chunk.ReadUint8();
				m.note = chunk.ReadUint8();
				if(m.note == 0x80) m.note = NOTE_KEYOFF;
				else if(m.note > 0x80) m.note = NOTE_FADE;	// I guess the support for IT "note fade" notes was not intended in mod2j2b, but hey, it works! :-D
			}

			if (flags & volFlag) // volume
			{
				m.volcmd = VOLCMD_VOLUME;
				m.vol = chunk.ReadUint8();
				if(isAM)
				{
					m.vol = m.vol * 64 / 127;
				}
			}
		}
	}

	return true;
}


struct AMFFRiffChunkFormat
{
	uint32le format;
};

MPT_BINARY_STRUCT(AMFFRiffChunkFormat, 4)


static bool ValidateHeader(const AMFFRiffChunk &fileHeader)
{
	if(fileHeader.id != AMFFRiffChunk::idRIFF)
	{
		return false;
	}
	if(fileHeader.GetLength() < 8 + sizeof(AMFFMainChunk))
	{
		return false;
	}
	return true;
}


static bool ValidateHeader(const AMFFRiffChunkFormat &formatHeader)
{
	if(formatHeader.format != AMFFRiffChunk::idAMFF && formatHeader.format != AMFFRiffChunk::idAM__)
	{
		return false;
	}
	return true;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderAM(MemoryFileReader file, const uint64 *pfilesize)
{
	AMFFRiffChunk fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return ProbeWantMoreData;
	}
	if(!ValidateHeader(fileHeader))
	{
		return ProbeFailure;
	}
	AMFFRiffChunkFormat formatHeader;
	if(!file.ReadStruct(formatHeader))
	{
		return ProbeWantMoreData;
	}
	if(!ValidateHeader(formatHeader))
	{
		return ProbeFailure;
	}
	MPT_UNREFERENCED_PARAMETER(pfilesize);
	return ProbeSuccess;
}


bool CSoundFile::ReadAM(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();
	AMFFRiffChunk fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return false;
	}
	if(!ValidateHeader(fileHeader))
	{
		return false;
	}
	AMFFRiffChunkFormat formatHeader;
	if(!file.ReadStruct(formatHeader))
	{
		return false;
	}
	if(!ValidateHeader(formatHeader))
	{
		return false;
	}

	bool isAM; // false: AMFF, true: AM

	uint32 format = formatHeader.format;
	if(format == AMFFRiffChunk::idAMFF)
		isAM = false; // "AMFF"
	else if(format == AMFFRiffChunk::idAM__)
		isAM = true; // "AM  "
	else
		return false;

	ChunkReader chunkFile(file);

	// The main chunk is almost identical in both formats but uses different chunk IDs.
	// "MAIN" - Song info (AMFF)
	// "INIT" - Song info (AM)
	AMFFRiffChunk::ChunkIdentifiers mainChunkID = isAM ? AMFFRiffChunk::idINIT : AMFFRiffChunk::idMAIN;

	// RIFF AM has a padding byte so that all chunks have an even size.
	ChunkReader::ChunkList<AMFFRiffChunk> chunks;
	if(loadFlags == onlyVerifyHeader)
		chunks = chunkFile.ReadChunksUntil<AMFFRiffChunk>(isAM ? 2 : 1, mainChunkID);
	else
		chunks = chunkFile.ReadChunks<AMFFRiffChunk>(isAM ? 2 : 1);

	FileReader chunkMain(chunks.GetChunk(mainChunkID));
	AMFFMainChunk mainChunk;
	if(!chunkMain.IsValid() 
		|| !chunkMain.ReadStruct(mainChunk)
		|| mainChunk.channels < 1
		|| !chunkMain.CanRead(mainChunk.channels))
	{
		return false;
	} else if(loadFlags == onlyVerifyHeader)
	{
		return true;
	}

	InitializeGlobals(MOD_TYPE_J2B);
	m_SongFlags = SONG_ITOLDEFFECTS | SONG_ITCOMPATGXX;
	m_SongFlags.set(SONG_LINEARSLIDES, !(mainChunk.flags & AMFFMainChunk::amigaSlides));

	m_nChannels = MIN(mainChunk.channels, MAX_BASECHANNELS);
	m_nDefaultSpeed = mainChunk.speed;
	m_nDefaultTempo.Set(mainChunk.tempo);
	m_nDefaultGlobalVolume = mainChunk.globalvolume * 2;

	m_madeWithTracker = MPT_USTRING("Galaxy Sound System (");
	if(isAM)
		m_madeWithTracker += MPT_USTRING("new version)");
	else
		m_madeWithTracker += MPT_USTRING("old version)");
	
	mpt::String::Read<mpt::String::maybeNullTerminated>(m_songName, mainChunk.songname);

	// It seems like there's no way to differentiate between
	// Muted and Surround channels (they're all 0xA0) - might
	// be a limitation in mod2j2b.
	for(CHANNELINDEX nChn = 0; nChn < m_nChannels; nChn++)
	{
		ChnSettings[nChn].Reset();

		uint8 pan = chunkMain.ReadUint8();
		if(isAM)
		{
			if(pan > 128)
				ChnSettings[nChn].dwFlags = CHN_MUTE;
			else
				ChnSettings[nChn].nPan = pan * 2;
		} else
		{
			if(pan >= 128)
				ChnSettings[nChn].dwFlags = CHN_MUTE;
			else
				ChnSettings[nChn].nPan = static_cast<uint16>(std::min(pan * 4, 256));
		}
	}

	if(chunks.ChunkExists(AMFFRiffChunk::idORDR))
	{
		// "ORDR" - Order list
		FileReader chunk(chunks.GetChunk(AMFFRiffChunk::idORDR));
		uint8 numOrders = chunk.ReadUint8() + 1;
		ReadOrderFromFile<uint8>(Order(), chunk, numOrders, 0xFF, 0xFE);
	}

	// "PATT" - Pattern data for one pattern
	if(loadFlags & loadPatternData)
	{
		PATTERNINDEX maxPattern = 0;
		auto pattChunks = chunks.GetAllChunks(AMFFRiffChunk::idPATT);
		Patterns.ResizeArray(static_cast<PATTERNINDEX>(pattChunks.size()));
		for(auto chunk : pattChunks)
		{
			PATTERNINDEX pat = chunk.ReadUint8();
			size_t patternSize = chunk.ReadUint32LE();
			ConvertAMPattern(chunk.ReadChunk(patternSize), pat, isAM, *this);
			maxPattern = std::max(maxPattern, pat);
		}
		for(PATTERNINDEX pat = 0; pat < maxPattern; pat++)
		{
			if(!Patterns.IsValidPat(pat))
				Patterns.Insert(pat, 64);
		}
	}

	if(!isAM)
	{
		// "INST" - Instrument (only in RIFF AMFF)
		auto instChunks = chunks.GetAllChunks(AMFFRiffChunk::idINST);
		for(auto chunk : instChunks)
		{
			AMFFInstrumentHeader instrHeader;
			if(!chunk.ReadStruct(instrHeader))
			{
				continue;
			}

			const INSTRUMENTINDEX instr = instrHeader.index + 1;
			if(instr >= MAX_INSTRUMENTS)
				continue;

			ModInstrument *pIns = AllocateInstrument(instr);
			if(pIns == nullptr)
			{
				continue;
			}

			instrHeader.ConvertToMPT(*pIns, m_nSamples);

			// read sample sub-chunks - this is a rather "flat" format compared to RIFF AM and has no nested RIFF chunks.
			for(size_t samples = 0; samples < instrHeader.numSamples; samples++)
			{
				AMFFSampleHeader sampleHeader;

				if(m_nSamples + 1 >= MAX_SAMPLES || !chunk.ReadStruct(sampleHeader))
				{
					continue;
				}

				const SAMPLEINDEX smp = ++m_nSamples;

				if(sampleHeader.id != AMFFRiffChunk::idSAMP)
				{
					continue;
				}

				mpt::String::Read<mpt::String::maybeNullTerminated>(m_szNames[smp], sampleHeader.name);
				sampleHeader.ConvertToMPT(instrHeader, Samples[smp]);
				if(loadFlags & loadSampleData)
					sampleHeader.GetSampleFormat().ReadSample(Samples[smp], chunk);
				else
					chunk.Skip(Samples[smp].GetSampleSizeInBytes());
			}
		}
	} else
	{
		// "RIFF" - Instrument (only in RIFF AM)
		auto instChunks = chunks.GetAllChunks(AMFFRiffChunk::idRIFF);
		for(ChunkReader chunk : instChunks)
		{
			if(chunk.ReadUint32LE() != AMFFRiffChunk::idAI__)
			{
				continue;
			}

			AMFFRiffChunk instChunk;
			if(!chunk.ReadStruct(instChunk) || instChunk.id != AMFFRiffChunk::idINST)
			{
				continue;
			}

			AMInstrumentHeader instrHeader;
			if(!chunk.ReadStruct(instrHeader))
			{
				continue;
			}
			MPT_ASSERT(instrHeader.headSize + 4 == sizeof(instrHeader));

			const INSTRUMENTINDEX instr = instrHeader.index + 1;
			if(instr >= MAX_INSTRUMENTS)
				continue;

			ModInstrument *pIns = AllocateInstrument(instr);
			if(pIns == nullptr)
			{
				continue;
			}

			instrHeader.ConvertToMPT(*pIns, m_nSamples);

			// Read sample sub-chunks (RIFF nesting ftw)
			auto sampleChunks = chunk.ReadChunks<AMFFRiffChunk>(2).GetAllChunks(AMFFRiffChunk::idRIFF);
			MPT_ASSERT(sampleChunks.size() == instrHeader.numSamples);

			for(auto sampleChunk : sampleChunks)
			{
				if(sampleChunk.ReadUint32LE() != AMFFRiffChunk::idAS__ || m_nSamples + 1 >= MAX_SAMPLES)
				{
					continue;
				}

				// Don't read more samples than the instrument header claims to have.
				if((instrHeader.numSamples--) == 0)
				{
					break;
				}

				const SAMPLEINDEX smp = ++m_nSamples;

				// Aaand even more nested chunks! Great, innit?
				AMFFRiffChunk sampleHeaderChunk;
				if(!sampleChunk.ReadStruct(sampleHeaderChunk) || sampleHeaderChunk.id != AMFFRiffChunk::idSAMP)
				{
					break;
				}

				FileReader sampleFileChunk = sampleChunk.ReadChunk(sampleHeaderChunk.length);

				AMSampleHeader sampleHeader;
				if(!sampleFileChunk.ReadStruct(sampleHeader))
				{
					break;
				}

				mpt::String::Read<mpt::String::maybeNullTerminated>(m_szNames[smp], sampleHeader.name);

				sampleHeader.ConvertToMPT(instrHeader, Samples[smp]);

				if(loadFlags & loadSampleData)
				{
					sampleFileChunk.Seek(sampleHeader.headSize + 4);
					sampleHeader.GetSampleFormat().ReadSample(Samples[smp], sampleFileChunk);
				}
			}
		
		}
	}

	return true;
}


static bool ValidateHeader(const J2BFileHeader &fileHeader)
{
	if(std::memcmp(fileHeader.signature, "MUSE", 4)
		|| (fileHeader.deadbeaf != J2BFileHeader::magicDEADBEAF // 0xDEADBEAF (RIFF AM)
			&& fileHeader.deadbeaf != J2BFileHeader::magicDEADBABE) // 0xDEADBABE (RIFF AMFF)
		)
	{
		return false;
	}
	if(fileHeader.packedLength == 0)
	{
		return false;
	}
	if(fileHeader.fileLength != fileHeader.packedLength + sizeof(J2BFileHeader))
	{
		return false;
	}
	return true;
}


static bool ValidateHeaderFileSize(const J2BFileHeader &fileHeader, uint64 filesize)
{
	if(filesize != fileHeader.fileLength)
	{
		return false;
	}
	return true;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderJ2B(MemoryFileReader file, const uint64 *pfilesize)
{
	J2BFileHeader fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return ProbeWantMoreData;
	}
	if(!ValidateHeader(fileHeader))
	{
		return ProbeFailure;
	}
	if(pfilesize)
	{
		if(!ValidateHeaderFileSize(fileHeader, *pfilesize))
		{
			return ProbeFailure;
		}
	}
	MPT_UNREFERENCED_PARAMETER(pfilesize);
	return ProbeSuccess;
}


bool CSoundFile::ReadJ2B(FileReader &file, ModLoadingFlags loadFlags)
{

#if !defined(MPT_WITH_ZLIB) && !defined(MPT_WITH_MINIZ)

	MPT_UNREFERENCED_PARAMETER(file);
	MPT_UNREFERENCED_PARAMETER(loadFlags);
	return false;

#else

	file.Rewind();
	J2BFileHeader fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return false;
	}
	if(!ValidateHeader(fileHeader))
	{
		return false;
	}
	if(fileHeader.fileLength != file.GetLength()
		|| fileHeader.packedLength != file.BytesLeft()
		)
	{
		return false;
	}
	if(loadFlags == onlyVerifyHeader)
	{
		return true;
	}

	FileReader::PinnedRawDataView filePackedView = file.GetPinnedRawDataView(fileHeader.packedLength);

#ifndef MPT_BUILD_FUZZER
	if(fileHeader.crc32 != crc32(0, mpt::byte_cast<const Bytef*>(filePackedView.data()), static_cast<uint32>(filePackedView.size())))
	{
		return false;
	}
#endif

	// Header is valid, now unpack the RIFF AM file using inflate
	uLongf destSize = fileHeader.unpackedLength;
	std::vector<Bytef> amFileData(destSize);
	int retVal = uncompress(amFileData.data(), &destSize, mpt::byte_cast<const Bytef*>(filePackedView.data()), static_cast<uint32>(filePackedView.size()));

	bool result = false;
#ifndef MPT_BUILD_FUZZER
	if(destSize == fileHeader.unpackedLength && retVal == Z_OK)
#endif
	{
		// Success, now load the RIFF AM(FF) module.
		FileReader amFile(mpt::as_span(amFileData));
		result = ReadAM(amFile, loadFlags);
	}
	return result;

#endif

}


OPENMPT_NAMESPACE_END
