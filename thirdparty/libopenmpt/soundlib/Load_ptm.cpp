/*
 * Load_ptm.cpp
 * ------------
 * Purpose: PTM (PolyTracker) module loader
 * Notes  : (currently none)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"

OPENMPT_NAMESPACE_BEGIN

struct PTMFileHeader
{
	char     songname[28];		// Name of song, asciiz string
	uint8le  dosEOF;			// 26
	uint8le  versionLo;			// 03 version of file, currently 0203h
	uint8le  versionHi;			// 02
	uint8le  reserved1;			// Reserved, set to 0
	uint16le numOrders;			// Number of orders (0..256)
	uint16le numSamples;		// Number of instruments (1..255)
	uint16le numPatterns;		// Number of patterns (1..128)
	uint16le numChannels;		// Number of channels (voices) used (1..32)
	uint16le flags;				// Set to 0
	uint8le  reserved2[2];		// Reserved, set to 0
	char     magic[4];			// Song identification, 'PTMF'
	uint8le  reserved3[16];		// Reserved, set to 0
	uint8le  chnPan[32];		// Channel panning settings, 0..15, 0 = left, 7 = middle, 15 = right
	uint8le  orders[256];		// Order list, valid entries 0..nOrders-1
	uint16le patOffsets[128];	// Pattern offsets (*16)
};

MPT_BINARY_STRUCT(PTMFileHeader, 608)

struct PTMSampleHeader
{
	enum SampleFlags
	{
		smpTypeMask	= 0x03,
		smpPCM		= 0x01,

		smpLoop		= 0x04,
		smpPingPong	= 0x08,
		smp16Bit	= 0x10,
	};

	uint8le  flags;				// Sample type (see SampleFlags)
	char     filename[12];		// Name of external sample file
	uint8le  volume;			// Default volume
	uint16le c4speed;			// C-4 speed (yep, not C-5)
	uint8le  smpSegment[2];		// Sample segment (used internally)
	uint32le dataOffset;		// Offset of sample data
	uint32le length;			// Sample size (in bytes)
	uint32le loopStart;			// Start of loop
	uint32le loopEnd;			// End of loop
	uint8le  gusdata[14];
	char     samplename[28];	// Name of sample, ASCIIZ
	char     magic[4];			// Sample identification, 'PTMS'

	// Convert an PTM sample header to OpenMPT's internal sample header.
	SampleIO ConvertToMPT(ModSample &mptSmp) const
	{
		mptSmp.Initialize(MOD_TYPE_S3M);
		mptSmp.nVolume = std::min<uint8>(volume, 64) * 4;
		mptSmp.nC5Speed = c4speed * 2;

		mpt::String::Read<mpt::String::maybeNullTerminated>(mptSmp.filename, filename);

		SampleIO sampleIO(
			SampleIO::_8bit,
			SampleIO::mono,
			SampleIO::littleEndian,
			SampleIO::deltaPCM);

		if((flags & smpTypeMask) == smpPCM)
		{
			mptSmp.nLength = length;
			mptSmp.nLoopStart = loopStart;
			mptSmp.nLoopEnd = loopEnd;
			if(mptSmp.nLoopEnd > mptSmp.nLoopStart)
				mptSmp.nLoopEnd--;

			if(flags & smpLoop) mptSmp.uFlags.set(CHN_LOOP);
			if(flags & smpPingPong) mptSmp.uFlags.set(CHN_PINGPONGLOOP);
			if(flags & smp16Bit)
			{
				sampleIO |= SampleIO::_16bit;
				sampleIO |= SampleIO::PTM8Dto16;

				mptSmp.nLength /= 2;
				mptSmp.nLoopStart /= 2;
				mptSmp.nLoopEnd /= 2;
			}
		}

		return sampleIO;
	}
};

MPT_BINARY_STRUCT(PTMSampleHeader, 80)


static bool ValidateHeader(const PTMFileHeader &fileHeader)
{
	if(std::memcmp(fileHeader.magic, "PTMF", 4)
		|| fileHeader.dosEOF != 26
		|| fileHeader.versionHi > 2
		|| fileHeader.flags != 0
		|| !fileHeader.numChannels
		|| fileHeader.numChannels > 32
		|| !fileHeader.numOrders || fileHeader.numOrders > 256
		|| !fileHeader.numSamples || fileHeader.numSamples > 255
		|| !fileHeader.numPatterns || fileHeader.numPatterns > 128
		)
	{
		return false;
	}
	return true;
}


static uint64 GetHeaderMinimumAdditionalSize(const PTMFileHeader &fileHeader)
{
	return fileHeader.numSamples * sizeof(PTMSampleHeader);
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderPTM(MemoryFileReader file, const uint64 *pfilesize)
{
	PTMFileHeader fileHeader;
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


bool CSoundFile::ReadPTM(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();

	PTMFileHeader fileHeader;
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

	InitializeGlobals(MOD_TYPE_PTM);

	mpt::String::Read<mpt::String::maybeNullTerminated>(m_songName, fileHeader.songname);

	m_madeWithTracker = mpt::format(MPT_USTRING("PolyTracker %1.%2"))(fileHeader.versionHi.get(), mpt::ufmt::hex0<2>(fileHeader.versionLo.get()));
	m_SongFlags = SONG_ITCOMPATGXX | SONG_ITOLDEFFECTS;
	m_nChannels = fileHeader.numChannels;
	m_nSamples = std::min<SAMPLEINDEX>(fileHeader.numSamples, MAX_SAMPLES - 1);
	ReadOrderFromArray(Order(), fileHeader.orders, fileHeader.numOrders, 0xFF, 0xFE);

	// Reading channel panning
	for(CHANNELINDEX chn = 0; chn < m_nChannels; chn++)
	{
		ChnSettings[chn].Reset();
		ChnSettings[chn].nPan = ((fileHeader.chnPan[chn] & 0x0F) << 4) + 4;
	}

	// Reading samples
	FileReader sampleHeaderChunk = file.ReadChunk(fileHeader.numSamples * sizeof(PTMSampleHeader));
	for(SAMPLEINDEX smp = 0; smp < m_nSamples; smp++)
	{
		PTMSampleHeader sampleHeader;
		sampleHeaderChunk.ReadStruct(sampleHeader);

		ModSample &sample = Samples[smp + 1];
		mpt::String::Read<mpt::String::maybeNullTerminated>(m_szNames[smp + 1], sampleHeader.samplename);
		SampleIO sampleIO = sampleHeader.ConvertToMPT(sample);

		if((loadFlags & loadSampleData) && sample.nLength && file.Seek(sampleHeader.dataOffset))
		{
			sampleIO.ReadSample(sample, file);
		}
	}

	// Reading Patterns
	if(!(loadFlags & loadPatternData))
	{
		return true;
	}

	Patterns.ResizeArray(fileHeader.numPatterns);
	for(PATTERNINDEX pat = 0; pat < fileHeader.numPatterns; pat++)
	{
		if(!Patterns.Insert(pat, 64)
			|| fileHeader.patOffsets[pat] == 0
			|| !file.Seek(fileHeader.patOffsets[pat] << 4))
		{
			continue;
		}

		ModCommand *rowBase = Patterns[pat].GetpModCommand(0, 0);
		ROWINDEX row = 0;
		while(row < 64 && file.CanRead(1))
		{
			uint8 b = file.ReadUint8();

			if(b == 0)
			{
				row++;
				rowBase += m_nChannels;
				continue;
			}
			CHANNELINDEX chn = (b & 0x1F);
			ModCommand dummy = ModCommand();
			ModCommand &m = chn < GetNumChannels() ? rowBase[chn] : dummy;

			if(b & 0x20)
			{
				m.note = file.ReadUint8();
				m.instr = file.ReadUint8();
				if(m.note == 254)
					m.note = NOTE_NOTECUT;
				else if(!m.note || m.note > 120)
					m.note = NOTE_NONE;
			}
			if(b & 0x40)
			{
				m.command = file.ReadUint8();
				m.param = file.ReadUint8();

				static const EffectCommand effTrans[] = { CMD_GLOBALVOLUME, CMD_RETRIG, CMD_FINEVIBRATO, CMD_NOTESLIDEUP, CMD_NOTESLIDEDOWN, CMD_NOTESLIDEUPRETRIG, CMD_NOTESLIDEDOWNRETRIG, CMD_REVERSEOFFSET };
				if(m.command < 0x10)
				{
					// Beware: Effect letters are as in MOD, but portamento and volume slides behave like in S3M (i.e. fine slides share the same effect letters)
					ConvertModCommand(m);
				} else if(m.command < 0x10 + CountOf(effTrans))
				{
					m.command = effTrans[m.command - 0x10];
				} else
				{
					m.command = CMD_NONE;
				}
				switch(m.command)
				{
				case CMD_PANNING8:
					// My observations of this weird command...
					// 800...887 pan from hard left to hard right (whereas the low nibble seems to be ignored)
					// 888...8FF do the same (so 888...88F is hard left, and 890 is identical to 810)
					if(m.param >= 0x88) m.param &= 0x7F;
					else if(m.param > 0x80) m.param = 0x80;
					m.param &= 0x7F;
					m.param = (m.param * 0xFF) / 0x7F;
					break;
				case CMD_GLOBALVOLUME:
					m.param = std::min(m.param, uint8(0x40)) * 2u;
					break;
				}
			}
			if(b & 0x80)
			{
				m.volcmd = VOLCMD_VOLUME;
				m.vol = file.ReadUint8();
			}
		}
	}
	return true;
}


OPENMPT_NAMESPACE_END
