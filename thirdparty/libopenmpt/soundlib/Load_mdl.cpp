/*
 * Load_mdl.cpp
 * ------------
 * Purpose: Digitrakker (MDL) module loader
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "ChunkReader.h"

#include <stdexcept>

OPENMPT_NAMESPACE_BEGIN

// MDL file header
struct MDLFileHeader
{
	char  id[4];	// "DMDL"
	uint8 version;
};

MPT_BINARY_STRUCT(MDLFileHeader, 5)


// RIFF-style Chunk
struct MDLChunk
{
	// 16-Bit chunk identifiers
	enum ChunkIdentifiers
	{
		idInfo			= MAGIC2LE('I','N'),
		idMessage		= MAGIC2LE('M','E'),
		idPats			= MAGIC2LE('P','A'),
		idPatNames		= MAGIC2LE('P','N'),
		idTracks		= MAGIC2LE('T','R'),
		idInstrs		= MAGIC2LE('I','I'),
		idVolEnvs		= MAGIC2LE('V','E'),
		idPanEnvs		= MAGIC2LE('P','E'),
		idFreqEnvs		= MAGIC2LE('F','E'),
		idSampleInfo	= MAGIC2LE('I','S'),
		ifSampleData	= MAGIC2LE('S','A'),
	};

	uint16le id;
	uint32le length;

	size_t GetLength() const
	{
		return length;
	}

	ChunkIdentifiers GetID() const
	{
		return static_cast<ChunkIdentifiers>(id.get());
	}
};

MPT_BINARY_STRUCT(MDLChunk, 6)


struct MDLInfoBlock
{
	char     title[32];
	char     composer[20];
	uint16le numOrders;
	uint16le restartPos;
	uint8le  globalVol;	// 1...255
	uint8le  speed;		// 1...255
	uint8le  tempo;		// 4...255
	uint8le  chnSetup[32];
};

MPT_BINARY_STRUCT(MDLInfoBlock, 91)


// Sample header in II block
struct MDLSampleHeader
{
	uint8le  smpNum;
	uint8le  lastNote;
	uint8le  volume;
	uint8le  volEnvFlags;	// 6 bits env #, 2 bits flags
	uint8le  panning;
	uint8le  panEnvFlags;
	uint16le fadeout;
	uint8le  vibSpeed;
	uint8le  vibDepth;
	uint8le  vibSweep;
	uint8le  vibType;
	uint8le  reserved;		// zero
	uint8le  freqEnvFlags;
};

MPT_BINARY_STRUCT(MDLSampleHeader, 14)


// Part of the sample header that's common between v0 and v1.
struct MDLSampleInfoCommon
{
	uint8le sampleIndex;
	char    name[32];
	char    filename[8];
};

MPT_BINARY_STRUCT(MDLSampleInfoCommon, 41)


struct MDLEnvelope
{
	uint8 envNum;
	struct
	{
		uint8 x;	// Delta value from last point, 0 means no more points defined
		uint8 y;	// 0...63
	} nodes[15];
	uint8 flags;
	uint8 loop;		// Lower 4 bits = start, upper 4 bits = end

	void ConvertToMPT(InstrumentEnvelope &mptEnv) const
	{
		mptEnv.dwFlags.reset();
		mptEnv.clear();
		mptEnv.reserve(15);
		int16 tick = -nodes[0].x;
		for(uint8 n = 0; n < 15; n++)
		{
			if(!nodes[n].x)
				break;
			tick += nodes[n].x;
			mptEnv.push_back(EnvelopeNode(tick, std::min(nodes[n].y, uint8(64)))); // actually 0-63
		}

		mptEnv.nLoopStart = (loop & 0x0F);
		mptEnv.nLoopEnd = (loop >> 4);
		mptEnv.nSustainStart = mptEnv.nSustainEnd = (flags & 0x0F);

		if(flags & 0x10) mptEnv.dwFlags.set(ENV_SUSTAIN);
		if(flags & 0x20) mptEnv.dwFlags.set(ENV_LOOP);
	}
};

MPT_BINARY_STRUCT(MDLEnvelope, 33)


struct MDLPatternHeader
{
	uint8le channels;
	uint8le lastRow;
	char    name[16];
};

MPT_BINARY_STRUCT(MDLPatternHeader, 18)


enum
{
	MDLNOTE_NOTE	= 1 << 0,
	MDLNOTE_SAMPLE	= 1 << 1,
	MDLNOTE_VOLUME	= 1 << 2,
	MDLNOTE_EFFECTS	= 1 << 3,
	MDLNOTE_PARAM1	= 1 << 4,
	MDLNOTE_PARAM2	= 1 << 5,
};


static const uint8 MDLVibratoType[] = { VIB_SINE, VIB_RAMP_DOWN, VIB_SQUARE, VIB_SINE };

static const ModCommand::COMMAND MDLEffTrans[] =
{
	/* 0 */ CMD_NONE,
	/* 1st column only */
	/* 1 */ CMD_PORTAMENTOUP,
	/* 2 */ CMD_PORTAMENTODOWN,
	/* 3 */ CMD_TONEPORTAMENTO,
	/* 4 */ CMD_VIBRATO,
	/* 5 */ CMD_ARPEGGIO,
	/* 6 */ CMD_NONE,
	/* Either column */
	/* 7 */ CMD_TEMPO,
	/* 8 */ CMD_PANNING8,
	/* 9 */ CMD_SETENVPOSITION,
	/* A */ CMD_NONE,
	/* B */ CMD_POSITIONJUMP,
	/* C */ CMD_GLOBALVOLUME,
	/* D */ CMD_PATTERNBREAK,
	/* E */ CMD_S3MCMDEX,
	/* F */ CMD_SPEED,
	/* 2nd column only */
	/* G */ CMD_VOLUMESLIDE, // up
	/* H */ CMD_VOLUMESLIDE, // down
	/* I */ CMD_RETRIG,
	/* J */ CMD_TREMOLO,
	/* K */ CMD_TREMOR,
	/* L */ CMD_NONE,
};


// receive an MDL effect, give back a 'normal' one.
static void ConvertMDLCommand(uint8_t &cmd, uint8_t &param)
{
	if(cmd >= CountOf(MDLEffTrans))
		return;

	uint8 origCmd = cmd;
	cmd = MDLEffTrans[cmd];

	switch(origCmd)
	{
#ifdef MODPLUG_TRACKER
	case 0x07: // Tempo
		// MDL supports any nonzero tempo value, but OpenMPT doesn't
		param = std::max(param, uint8(0x20));
		break;
#endif // MODPLUG_TRACKER
	case 0x08: // Panning
		param = (param & 0x7F) * 2u;
		break;
	case 0x0C:	// Global volume
		param = (param + 1) / 2u;
		break;
	case 0x0D: // Pattern Break
		// Convert from BCD
		param = 10 * (param >> 4) + (param & 0x0F);
		break;
	case 0x0E: // Special
		switch(param >> 4)
		{
		case 0x0: // unused
		case 0x3: // unused
		case 0x5: // Set Finetune
		case 0x8: // Set Samplestatus (loop type)
			cmd = CMD_NONE;
			break;
		case 0x1: // Pan Slide Left
			cmd = CMD_PANNINGSLIDE;
			param = (std::min<uint8>(param & 0x0F, 0x0E) << 4) | 0x0F;
			break;
		case 0x2: // Pan Slide Right
			cmd = CMD_PANNINGSLIDE;
			param = 0xF0 | std::min<uint8>(param & 0x0F, 0x0E);
			break;
		case 0x4: // Vibrato Waveform
			param = 0x30 | (param & 0x0F);
			break;
		case 0x6: // Pattern Loop
			param = 0xB0 | (param & 0x0F);
			break;
		case 0x7: // Tremolo Waveform
			param = 0x40 | (param & 0x0F);
			break;
		case 0x9: // Retrig
			cmd = CMD_RETRIG;
			param &= 0x0F;
			break;
		case 0xA: // Global vol slide up
			cmd = CMD_GLOBALVOLSLIDE;
			param = 0xF0 & (((param & 0x0F) + 1) << 3);
			break;
		case 0xB: // Global vol slide down
			cmd = CMD_GLOBALVOLSLIDE;
			param = ((param & 0x0F) + 1) >> 1;
			break;
		case 0xC: // Note cut
		case 0xD: // Note delay
		case 0xE: // Pattern delay
			// Nothing to change here
			break;
		case 0xF: // Offset -- further mangled later.
			cmd = CMD_OFFSET;
			break;
		}
		break;
	case 0x10: // Volslide up
		if(param < 0xE0)
		{
			// 00...DF regular slide - four times more precise than in XM
			param >>= 2;
			if(param > 0x0F)
				param = 0x0F;
			param <<= 4;
		} else if(param < 0xF0)
		{
			// E0...EF extra fine slide (on first tick, 4 times finer)
			param = (((param & 0x0F) << 2) | 0x0F);
		} else
		{
			// F0...FF regular fine slide (on first tick) - like in XM
			param = ((param << 4) | 0x0F);
		}
		break;
	case 0x11: // Volslide down
		if(param < 0xE0)
		{
			// 00...DF regular slide - four times more precise than in XM
			param >>= 2;
			if(param > 0x0F)
				param = 0x0F;
		} else if(param < 0xF0)
		{
			// E0...EF extra fine slide (on first tick, 4 times finer)
			param = (((param & 0x0F) >> 2) | 0xF0);
		} else
		{
			// F0...FF regular fine slide (on first tick) - like in XM
		}
		break;
	}
}


// Returns true if command was lost
static bool ImportMDLCommands(ModCommand &m, uint8 vol, uint8 e1, uint8 e2, uint8 p1, uint8 p2)
{
	// Map second effect values 1-6 to effects G-L
	if(e2 >= 1 && e2 <= 6)
		e2 += 15;

	ConvertMDLCommand(e1, p1);
	ConvertMDLCommand(e2, p2);
	/* From the Digitrakker documentation:
		* EFx -xx - Set Sample Offset
		This  is a  double-command.  It starts the
		sample at adress xxx*256.
		Example: C-5 01 -- EF1 -23 ->starts sample
		01 at address 12300 (in hex).
	Kind of screwy, but I guess it's better than the mess required to do it with IT (which effectively
	requires 3 rows in order to set the offset past 0xff00). If we had access to the entire track, we
	*might* be able to shove the high offset SAy into surrounding rows (or 2x MPTM #xx), but it wouldn't
	always be possible, it'd make the loader a lot uglier, and generally would be more trouble than
	it'd be worth to implement.

	What's more is, if there's another effect in the second column, it's ALSO processed in addition to the
	offset, and the second data byte is shared between the two effects. */
	if(e1 == CMD_OFFSET)
	{
		// EFy -xx => offset yxx00
		p1 = (p1 & 0x0F) ? 0xFF : p2;
		if(e2 == CMD_OFFSET)
			e2 = CMD_NONE;
	} else if (e2 == CMD_OFFSET)
	{
		// --- EFy => offset y0000 (best we can do without doing a ton of extra work is 0xff00)
		p2 = (p2 & 0x0F) ? 0xFF : 0;
	}

	if(vol)
	{
		m.volcmd = VOLCMD_VOLUME;
		m.vol = (vol + 2) / 4u;
	}

	// If we have Dxx + G00, or Dxx + H00, combine them into Lxx/Kxx.
	ModCommand::CombineEffects(e1, p1, e2, p2);

	bool lostCommand = false;
	// Try to fit the "best" effect into e2.
	if(e1 == CMD_NONE)
	{
		// Easy
	} else if(e2 == CMD_NONE)
	{
		// Almost as easy
		e2 = e1;
		p2 = p1;
		e1 = CMD_NONE;
	} else if(e1 == e2 && e1 != CMD_S3MCMDEX)
	{
		// Digitrakker processes the effects left-to-right, so if both effects are the same, the
		// second essentially overrides the first.
		e1 = CMD_NONE;
	} else if(!vol)
	{
		lostCommand |= !ModCommand::TwoRegularCommandsToMPT(e1, p1, e2, p2);
		m.volcmd = e1;
		m.vol = p1;
	} else
	{
		if(ModCommand::GetEffectWeight((ModCommand::COMMAND)e1) > ModCommand::GetEffectWeight((ModCommand::COMMAND)e2))
		{
			std::swap(e1, e2);
			std::swap(p1, p2);
		}
	}

	m.command = e2;
	m.param = p2;
	return lostCommand;
}


static void MDLReadEnvelopes(FileReader file, std::vector<MDLEnvelope> &envelopes)
{
	if(!file.CanRead(1))
		return;

	envelopes.resize(64);
	uint8 numEnvs = file.ReadUint8();
	while(numEnvs--)
	{
		MDLEnvelope mdlEnv;
		if(!file.ReadStruct(mdlEnv) || mdlEnv.envNum > 63)
			continue;
		envelopes[mdlEnv.envNum] = mdlEnv;
	}
}


static void CopyEnvelope(InstrumentEnvelope &mptEnv, uint8 flags, std::vector<MDLEnvelope> &envelopes)
{
	uint8 envNum = flags & 0x3F;
	if(envNum < envelopes.size())
		envelopes[envNum].ConvertToMPT(mptEnv);
	mptEnv.dwFlags.set(ENV_ENABLED, (flags & 0x80) && !mptEnv.empty());
}


static bool ValidateHeader(const MDLFileHeader &fileHeader)
{
	if(std::memcmp(fileHeader.id, "DMDL", 4)
		|| fileHeader.version >= 0x20)
	{
		return false;
	}
	return true;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderMDL(MemoryFileReader file, const uint64 *pfilesize)
{
	MDLFileHeader fileHeader;
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


bool CSoundFile::ReadMDL(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();
	MDLFileHeader fileHeader;
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
	ChunkReader::ChunkList<MDLChunk> chunks = chunkFile.ReadChunks<MDLChunk>(0);

	// Read global info
	FileReader chunk = chunks.GetChunk(MDLChunk::idInfo);
	MDLInfoBlock info;
	if(!chunk.IsValid() || !chunk.ReadStruct(info))
	{
		return false;
	}

	InitializeGlobals(MOD_TYPE_MDL);
	m_SongFlags = SONG_ITCOMPATGXX;
	m_playBehaviour.set(kPerChannelGlobalVolSlide);
	m_playBehaviour.reset(kITVibratoTremoloPanbrello);
	m_playBehaviour.reset(kITSCxStopsSample);	// Gate effect in underbeat.mdl

	m_madeWithTracker = MPT_USTRING("Digitrakker ") + (
		(fileHeader.version == 0x11) ? MPT_USTRING("3") // really could be 2.99b - close enough
		: (fileHeader.version == 0x10) ? MPT_USTRING("2.3")
		: (fileHeader.version == 0x00) ? MPT_USTRING("2.0 - 2.2b") // there was no 1.x release
		: MPT_USTRING(""));

	mpt::String::Read<mpt::String::spacePadded>(m_songName, info.title);
	{
		std::string artist;
		mpt::String::Read<mpt::String::spacePadded>(artist, info.composer);
		m_songArtist = mpt::ToUnicode(mpt::CharsetCP437, artist);
	}

	m_nDefaultGlobalVolume = info.globalVol + 1;
	m_nDefaultSpeed = Clamp<uint8, uint8>(info.speed, 1, 255);
	m_nDefaultTempo.Set(Clamp<uint8, uint8>(info.tempo, 4, 255));

	ReadOrderFromFile<uint8>(Order(), chunk, info.numOrders);
	Order().SetRestartPos(info.restartPos);

	m_nChannels = 0;
	for(CHANNELINDEX c = 0; c < 32; c++)
	{
		ChnSettings[c].Reset();
		ChnSettings[c].nPan = (info.chnSetup[c] & 0x7F) * 2u;
		if(ChnSettings[c].nPan == 254)
			ChnSettings[c].nPan = 256;
		if(info.chnSetup[c] & 0x80)
			ChnSettings[c].dwFlags.set(CHN_MUTE);
		else
			m_nChannels = c + 1;
		chunk.ReadString<mpt::String::spacePadded>(ChnSettings[c].szName, 8);
	}

	// Read song message
	chunk = chunks.GetChunk(MDLChunk::idMessage);
	m_songMessage.Read(chunk, chunk.GetLength(), SongMessage::leCR);

	// Read sample info and data
	chunk = chunks.GetChunk(MDLChunk::idSampleInfo);
	if(chunk.IsValid())
	{
		FileReader dataChunk = chunks.GetChunk(MDLChunk::ifSampleData);

		uint8 numSamples = chunk.ReadUint8();
		for(uint8 smp = 0; smp < numSamples; smp++)
		{
			MDLSampleInfoCommon header;
			if(!chunk.ReadStruct(header) || header.sampleIndex == 0)
				continue;
			#if 1
				STATIC_ASSERT(MPT_MAX_UNSIGNED_VALUE(header.sampleIndex) < MAX_SAMPLES);
			#else
				MPT_MAYBE_CONSTANT_IF(header.sampleIndex >= MAX_SAMPLES)
					continue;
			#endif

			if(header.sampleIndex > GetNumSamples())
				m_nSamples = header.sampleIndex;

			ModSample &sample = Samples[header.sampleIndex];
			sample.Initialize();

			mpt::String::Read<mpt::String::spacePadded>(m_szNames[header.sampleIndex], header.name);
			mpt::String::Read<mpt::String::spacePadded>(sample.filename, header.filename);

			uint32 c4speed;
			if(fileHeader.version < 0x10)
				c4speed = chunk.ReadUint16LE();
			else
				c4speed = chunk.ReadUint32LE();
			sample.nC5Speed = c4speed * 2u;
			sample.nLength = chunk.ReadUint32LE();
			sample.nLoopStart = chunk.ReadUint32LE();
			sample.nLoopEnd = chunk.ReadUint32LE();
			if(sample.nLoopEnd != 0)
			{
				sample.uFlags.set(CHN_LOOP);
				sample.nLoopEnd += sample.nLoopStart;
			}
			if(fileHeader.version < 0x10)
				sample.nVolume = chunk.ReadUint8();
			else
				chunk.Skip(1);
			uint8 flags = chunk.ReadUint8();

			if(flags & 0x01)
			{
				sample.uFlags.set(CHN_16BIT);
				sample.nLength /= 2u;
				sample.nLoopStart /= 2u;
				sample.nLoopEnd /= 2u;
			}

			sample.uFlags.set(CHN_PINGPONGLOOP, (flags & 0x02) != 0);

			SampleIO sampleIO(
				(flags & 0x01) ? SampleIO::_16bit : SampleIO::_8bit,
				SampleIO::mono,
				SampleIO::littleEndian,
				(flags & 0x0C) ? SampleIO::MDL : SampleIO::signedPCM);

			if(loadFlags & loadSampleData)
			{
				sampleIO.ReadSample(sample, dataChunk);
			}
		}
	}

	chunk = chunks.GetChunk(MDLChunk::idInstrs);
	if(chunk.IsValid())
	{
		std::vector<MDLEnvelope> volEnvs, panEnvs, pitchEnvs;
		MDLReadEnvelopes(chunks.GetChunk(MDLChunk::idVolEnvs), volEnvs);
		MDLReadEnvelopes(chunks.GetChunk(MDLChunk::idPanEnvs), panEnvs);
		MDLReadEnvelopes(chunks.GetChunk(MDLChunk::idFreqEnvs), pitchEnvs);

		uint8 numInstruments = chunk.ReadUint8();
		for(uint8 i = 0; i < numInstruments; i++)
		{
			uint8 ins = chunk.ReadUint8();
			uint8 numSamples = chunk.ReadUint8();
			uint8 firstNote = 0;
			ModInstrument *mptIns = nullptr;
			if(ins == 0
				|| !chunk.CanRead(32 + sizeof(MDLSampleHeader) * numSamples)
				|| (mptIns = AllocateInstrument(ins)) == nullptr)
			{
				chunk.Skip(32 + sizeof(MDLSampleHeader) * numSamples);
				continue;
			}

			chunk.ReadString<mpt::String::spacePadded>(mptIns->name, 32);
			while(numSamples--)
			{
				MDLSampleHeader sampleHeader;
				chunk.ReadStruct(sampleHeader);
				if(sampleHeader.smpNum == 0)
					continue;
				#if 1
					STATIC_ASSERT(MPT_MAX_UNSIGNED_VALUE(sampleHeader.smpNum) < MAX_SAMPLES);
				#else
					MPT_MAYBE_CONSTANT_IF(sampleHeader.smpNum >= MAX_SAMPLES)
						continue;
				#endif

				LimitMax(sampleHeader.lastNote, static_cast<uint8>(CountOf(mptIns->Keyboard)));
				for(uint8 n = firstNote; n <= sampleHeader.lastNote; n++)
				{
					mptIns->Keyboard[n] = sampleHeader.smpNum;
				}
				firstNote = sampleHeader.lastNote + 1;

				CopyEnvelope(mptIns->VolEnv, sampleHeader.volEnvFlags, volEnvs);
				CopyEnvelope(mptIns->PanEnv, sampleHeader.panEnvFlags, panEnvs);
				CopyEnvelope(mptIns->PitchEnv, sampleHeader.freqEnvFlags, pitchEnvs);
				mptIns->nFadeOut = (sampleHeader.fadeout + 1u) / 2u;
#ifdef MODPLUG_TRACKER
				if((mptIns->VolEnv.dwFlags & (ENV_ENABLED | ENV_LOOP)) == ENV_ENABLED)
				{
					// Fade-out is only supposed to happen on key-off, not at the end of a volume envelope.
					// Fake it by putting a loop at the end.
					mptIns->VolEnv.nLoopStart = mptIns->VolEnv.nLoopEnd = static_cast<uint8>(mptIns->VolEnv.size() - 1);
					mptIns->VolEnv.dwFlags.set(ENV_LOOP);
				}
				for(auto &p : mptIns->PitchEnv)
				{
					// Scale pitch envelope
					p.value = (p.value * 6u) / 16u;
				}
#endif // MODPLUG_TRACKER

				// Samples were already initialized above. Let's hope they are not going to be re-used with different volume / panning / vibrato...
				ModSample &mptSmp = Samples[sampleHeader.smpNum];

				// This flag literally enables and disables the default volume of a sample. If you disable this flag,
				// the sample volume of a previously sample is re-used, even if you put an instrument number next to the note.
				if(sampleHeader.volEnvFlags & 0x40)
					mptSmp.nVolume = sampleHeader.volume;
				else
					mptSmp.uFlags.set(SMP_NODEFAULTVOLUME);
				mptSmp.nPan = std::min<uint16>(sampleHeader.panning * 2, 254);
				mptSmp.nVibType = MDLVibratoType[sampleHeader.vibType & 3];
				mptSmp.nVibSweep = sampleHeader.vibSweep;
				mptSmp.nVibDepth = sampleHeader.vibDepth;
				mptSmp.nVibRate = sampleHeader.vibSpeed;
				if(sampleHeader.panEnvFlags & 0x40)
					mptSmp.uFlags.set(CHN_PANNING);
			}
		}
	}

	// Read pattern tracks
	std::vector<FileReader> tracks;
	if((loadFlags & loadPatternData) && (chunk = chunks.GetChunk(MDLChunk::idTracks)).IsValid())
	{
		uint32 numTracks = chunk.ReadUint16LE();
		tracks.resize(numTracks + 1);
		for(uint32 i = 1; i <= numTracks; i++)
		{
			tracks[i] = chunk.ReadChunk(chunk.ReadUint16LE());
		}
	}

	// Read actual patterns
	if((loadFlags & loadPatternData) && (chunk = chunks.GetChunk(MDLChunk::idPats)).IsValid())
	{
		PATTERNINDEX numPats = chunk.ReadUint8();

		// In case any muted channels contain data, be sure that we import them as well.
		for(PATTERNINDEX pat = 0; pat < numPats; pat++)
		{
			CHANNELINDEX numChans = 32;
			if(fileHeader.version >= 0x10)
			{
				MDLPatternHeader patHead;
				chunk.ReadStruct(patHead);
				if(patHead.channels > m_nChannels && patHead.channels <= 32)
					m_nChannels = patHead.channels;
				numChans = patHead.channels;
			}
			for(CHANNELINDEX chn = 0; chn < numChans; chn++)
			{
				if(chunk.ReadUint16LE() > 0 && chn >= m_nChannels)
					m_nChannels = chn + 1;
			}
		}
		chunk.Seek(1);

		Patterns.ResizeArray(numPats);
		for(PATTERNINDEX pat = 0; pat < numPats; pat++)
		{
			CHANNELINDEX numChans = 32;
			ROWINDEX numRows = 64;
			char name[17] = "";
			if(fileHeader.version >= 0x10)
			{
				MDLPatternHeader patHead;
				chunk.ReadStruct(patHead);
				numChans = patHead.channels;
				numRows = patHead.lastRow + 1;
				mpt::String::Read<mpt::String::spacePadded>(name, patHead.name);
			}

			if(!Patterns.Insert(pat, numRows))
			{
				chunk.Skip(2 * numChans);
				continue;
			}
			Patterns[pat].SetName(name);

			for(CHANNELINDEX chn = 0; chn < numChans; chn++)
			{
				uint16 trkNum = chunk.ReadUint16LE();
				if(!trkNum || trkNum >= tracks.size() || chn >= m_nChannels)
					continue;

				FileReader &track = tracks[trkNum];
				track.Rewind();
				ROWINDEX row = 0;
				while(row < numRows && track.CanRead(1))
				{
					ModCommand *m = Patterns[pat].GetpModCommand(row, chn);
					uint8 b = track.ReadUint8();
					uint8 x = (b >> 2), y = (b & 3);
					switch(y)
					{
					case 0:
						// (x + 1) empty notes follow
						row += x + 1;
						break;
					case 1:
						// Repeat previous note (x + 1) times
						if(row > 0)
						{
							ModCommand &orig = *Patterns[pat].GetpModCommand(row - 1, chn);
							do
							{
								*m = orig;
								m += m_nChannels;
								row++;
							} while (row < numRows && x--);
						}
						break;
					case 2:
						// Copy note from row x
						if(row > x)
						{
							*m = *Patterns[pat].GetpModCommand(x, chn);
						}
						row++;
						break;
					case 3:
						// New note data
						if(x & MDLNOTE_NOTE)
						{
							b = track.ReadUint8();
							m->note = (b > 120) ? NOTE_KEYOFF : b;
						}
						if(x & MDLNOTE_SAMPLE)
						{
							m->instr = track.ReadUint8();
						}
						{
							uint8 vol = 0, e1 = 0, e2 = 0, p1 = 0, p2 = 0;
							if(x & MDLNOTE_VOLUME)
							{
								vol = track.ReadUint8();
							}
							if(x & MDLNOTE_EFFECTS)
							{
								b = track.ReadUint8();
								e1 = (b & 0x0F);
								e2 = (b >> 4);
							}
							if(x & MDLNOTE_PARAM1)
								p1 = track.ReadUint8();
							if(x & MDLNOTE_PARAM2)
								p2 = track.ReadUint8();
							ImportMDLCommands(*m, vol, e1, e2, p1, p2);
						}

						row++;
						break;
					}
				}
			}
		}
	}

	if((loadFlags & loadPatternData) && (chunk = chunks.GetChunk(MDLChunk::idPatNames)).IsValid())
	{
		PATTERNINDEX i = 0;
		while(i < Patterns.Size() && chunk.CanRead(16))
		{
			char name[17];
			chunk.ReadString<mpt::String::spacePadded>(name, 16);
			Patterns[i].SetName(name);
		}
	}

	return true;
}


/////////////////////////////////////////////////////////////////////////
// MDL Sample Unpacking

// MDL Huffman ReadBits compression
uint8 MDLReadBits(uint32 &bitbuf, int32 &bitnum, const uint8 *(&ibuf), size_t &bytesLeft, int8 n)
{
	if(bitnum < n)
	{
		if(bytesLeft)
		{
			bitbuf |= (((uint32)(*ibuf++)) << bitnum);
			bitnum += 8;
			bytesLeft--;
		} else
		{
			throw std::range_error("Truncated MDL sample block");
		}
	}

	uint8 v = static_cast<uint8>(bitbuf & ((1 << n) - 1));
	bitbuf >>= n;
	bitnum -= n;
	return v;
}


OPENMPT_NAMESPACE_END
