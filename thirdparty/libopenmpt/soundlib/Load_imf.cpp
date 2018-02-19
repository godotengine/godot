/*
 * Load_imf.cpp
 * ------------
 * Purpose: IMF (Imago Orpheus) module loader
 * Notes  : Reverb and Chorus are not supported.
 * Authors: Storlek (Original author - http://schismtracker.org/ - code ported with permission)
 *			Johannes Schultz (OpenMPT Port, tweaks)
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"

OPENMPT_NAMESPACE_BEGIN

struct IMFChannel
{
	char  name[12];	// Channel name (ASCIIZ-String, max 11 chars)
	uint8 chorus;	// Default chorus
	uint8 reverb;	// Default reverb
	uint8 panning;	// Pan positions 00-FF
	uint8 status;	// Channel status: 0 = enabled, 1 = mute, 2 = disabled (ignore effects!)
};

MPT_BINARY_STRUCT(IMFChannel, 16)

struct IMFFileHeader
{
	enum SongFlags
	{
		linearSlides = 0x01,
	};

	char     title[32];			// Songname (ASCIIZ-String, max. 31 chars)
	uint16le ordNum;			// Number of orders saved
	uint16le patNum;			// Number of patterns saved
	uint16le insNum;			// Number of instruments saved
	uint16le flags;				// See SongFlags
	uint8le  unused1[8];
	uint8le  tempo;				// Default tempo (Axx, 1...255)
	uint8le  bpm;				// Default beats per minute (BPM) (Txx, 32...255)
	uint8le  master;			// Default master volume (Vxx, 0...64)
	uint8le  amp;				// Amplification factor (mixing volume, 4...127)
	uint8le  unused2[8];
	char     im10[4];			// 'IM10'
	IMFChannel channels[32];	// Channel settings
};

MPT_BINARY_STRUCT(IMFFileHeader, 576)

struct IMFEnvelope
{
	enum EnvFlags
	{
		envEnabled	= 0x01,
		envSustain	= 0x02,
		envLoop		= 0x04,
	};

	uint8 points;		// Number of envelope points
	uint8 sustain;		// Envelope sustain point
	uint8 loopStart;	// Envelope loop start point
	uint8 loopEnd;		// Envelope loop end point
	uint8 flags;		// See EnvFlags
	uint8 unused[3];
};

MPT_BINARY_STRUCT(IMFEnvelope, 8)

struct IMFEnvNode
{
	uint16le tick;
	uint16le value;
};

MPT_BINARY_STRUCT(IMFEnvNode, 4)

struct IMFInstrument
{
	enum EnvTypes
	{
		volEnv = 0,
		panEnv = 1,
		filterEnv = 2,
	};

	char        name[32];	// Inst. name (ASCIIZ-String, max. 31 chars)
	uint8le     map[120];	// Multisample settings
	uint8le     unused[8];
	IMFEnvNode  nodes[3][16];
	IMFEnvelope env[3];
	uint16le    fadeout;	// Fadeout rate (0...0FFFH)
	uint16le    smpNum;		// Number of samples in instrument
	char        ii10[4];	// 'II10'

	void ConvertEnvelope(InstrumentEnvelope &mptEnv, EnvTypes e) const
	{
		const int shift = (e == volEnv) ? 0 : 2;

		mptEnv.dwFlags.set(ENV_ENABLED, (env[e].flags & 1) != 0);
		mptEnv.dwFlags.set(ENV_SUSTAIN, (env[e].flags & 2) != 0);
		mptEnv.dwFlags.set(ENV_LOOP, (env[e].flags & 4) != 0);

		mptEnv.resize(Clamp(env[e].points, uint8(2), uint8(16)));
		mptEnv.nLoopStart = env[e].loopStart;
		mptEnv.nLoopEnd = env[e].loopEnd;
		mptEnv.nSustainStart = mptEnv.nSustainEnd = env[e].sustain;

		uint16 minTick = 0; // minimum tick value for next node
		for(uint32 n = 0; n < mptEnv.size(); n++)
		{
			minTick = mptEnv[n].tick = std::max<uint16>(minTick, nodes[e][n].tick);
			minTick++;
			mptEnv[n].value = static_cast<uint8>(std::min(nodes[e][n].value >> shift, ENVELOPE_MAX));
		}
	}

	// Convert an IMFInstrument to OpenMPT's internal instrument representation.
	void ConvertToMPT(ModInstrument &mptIns, SAMPLEINDEX firstSample) const
	{
		mpt::String::Read<mpt::String::nullTerminated>(mptIns.name, name);

		if(smpNum)
		{
			STATIC_ASSERT(CountOf(mptIns.Keyboard) >= CountOf(map));
			for(size_t note = 0; note < CountOf(map); note++)
			{
				mptIns.Keyboard[note] = firstSample + map[note];
			}
		}

		mptIns.nFadeOut = fadeout;

		ConvertEnvelope(mptIns.VolEnv, volEnv);
		ConvertEnvelope(mptIns.PanEnv, panEnv);
		ConvertEnvelope(mptIns.PitchEnv, filterEnv);
		if(mptIns.PitchEnv.dwFlags[ENV_ENABLED])
			mptIns.PitchEnv.dwFlags.set(ENV_FILTER);

		// hack to get === to stop notes (from modplug's xm loader)
		if(!mptIns.VolEnv.dwFlags[ENV_ENABLED] && !mptIns.nFadeOut)
			mptIns.nFadeOut = 8192;
	}
};

MPT_BINARY_STRUCT(IMFInstrument, 384)

struct IMFSample
{
	enum SampleFlags
	{
		smpLoop			= 0x01,
		smpPingPongLoop	= 0x02,
		smp16Bit		= 0x04,
		smpPanning		= 0x08,
	};

	char     filename[13];	// Sample filename (12345678.ABC) */
	uint8le  unused1[3];
	uint32le length;		// Length (in bytes)
	uint32le loopStart;		// Loop start (in bytes)
	uint32le loopEnd;		// Loop end (in bytes)
	uint32le c5Speed;		// Samplerate
	uint8le  volume;		// Default volume (0...64)
	uint8le  panning;		// Default pan (0...255)
	uint8le  unused2[14];
	uint8le  flags;			// Sample flags
	uint8le  unused3[5];
	uint16le ems;			// Reserved for internal usage
	uint32le dram;			// Reserved for internal usage
	char     is10[4];		// 'IS10'

	// Convert an IMFSample to OpenMPT's internal sample representation.
	void ConvertToMPT(ModSample &mptSmp) const
	{
		mptSmp.Initialize();
		mpt::String::Read<mpt::String::nullTerminated>(mptSmp.filename, filename);

		mptSmp.nLength = length;
		mptSmp.nLoopStart = loopStart;
		mptSmp.nLoopEnd = loopEnd;
		mptSmp.nC5Speed = c5Speed;
		mptSmp.nVolume = volume * 4;
		mptSmp.nPan = panning;
		if(flags & smpLoop)
			mptSmp.uFlags.set(CHN_LOOP);
		if(flags & smpPingPongLoop)
			mptSmp.uFlags.set(CHN_PINGPONGLOOP);
		if(flags & smp16Bit)
		{
			mptSmp.uFlags.set(CHN_16BIT);
			mptSmp.nLength /= 2;
			mptSmp.nLoopStart /= 2;
			mptSmp.nLoopEnd /= 2;
		}
		if(flags & smpPanning)
			mptSmp.uFlags.set(CHN_PANNING);
	}
};

MPT_BINARY_STRUCT(IMFSample, 64)


static const EffectCommand imfEffects[] =
{
	CMD_NONE,
	CMD_SPEED,			// 0x01 1xx Set Tempo
	CMD_TEMPO,			// 0x02 2xx Set BPM
	CMD_TONEPORTAMENTO, // 0x03 3xx Tone Portamento
	CMD_TONEPORTAVOL,	// 0x04 4xy Tone Portamento + Volume Slide
	CMD_VIBRATO,		// 0x05 5xy Vibrato
	CMD_VIBRATOVOL,		// 0x06 6xy Vibrato + Volume Slide
	CMD_FINEVIBRATO,	// 0x07 7xy Fine Vibrato
	CMD_TREMOLO,		// 0x08 8xy Tremolo
	CMD_ARPEGGIO,		// 0x09 9xy Arpeggio
	CMD_PANNING8,		// 0x0A Axx Set Pan Position
	CMD_PANNINGSLIDE,	// 0x0B Bxy Pan Slide
	CMD_VOLUME,			// 0x0C Cxx Set Volume
	CMD_VOLUMESLIDE,	// 0x0D Dxy Volume Slide
	CMD_VOLUMESLIDE,	// 0x0E Exy Fine Volume Slide
	CMD_S3MCMDEX,		// 0x0F Fxx Set Finetune
	CMD_NOTESLIDEUP,	// 0x10 Gxy Note Slide Up
	CMD_NOTESLIDEDOWN,	// 0x11 Hxy Note Slide Down
	CMD_PORTAMENTOUP,	// 0x12 Ixx Slide Up
	CMD_PORTAMENTODOWN,	// 0x13 Jxx Slide Down
	CMD_PORTAMENTOUP,	// 0x14 Kxx Fine Slide Up
	CMD_PORTAMENTODOWN,	// 0x15 Lxx Fine Slide Down
	CMD_MIDI,			// 0x16 Mxx Set Filter Cutoff - XXX
	CMD_NONE,			// 0x17 Nxy Filter Slide + Resonance - XXX
	CMD_OFFSET,			// 0x18 Oxx Set Sample Offset
	CMD_NONE,			// 0x19 Pxx Set Fine Sample Offset - XXX
	CMD_KEYOFF,			// 0x1A Qxx Key Off
	CMD_RETRIG,			// 0x1B Rxy Retrig
	CMD_TREMOR,			// 0x1C Sxy Tremor
	CMD_POSITIONJUMP,	// 0x1D Txx Position Jump
	CMD_PATTERNBREAK,	// 0x1E Uxx Pattern Break
	CMD_GLOBALVOLUME,	// 0x1F Vxx Set Mastervolume
	CMD_GLOBALVOLSLIDE,	// 0x20 Wxy Mastervolume Slide
	CMD_S3MCMDEX,		// 0x21 Xxx Extended Effect
							// X1x Set Filter
							// X3x Glissando
							// X5x Vibrato Waveform
							// X8x Tremolo Waveform
							// XAx Pattern Loop
							// XBx Pattern Delay
							// XCx Note Cut
							// XDx Note Delay
							// XEx Ignore Envelope
							// XFx Invert Loop
	CMD_NONE,			// 0x22 Yxx Chorus - XXX
	CMD_NONE,			// 0x23 Zxx Reverb - XXX
};

static void ImportIMFEffect(ModCommand &m)
{
	uint8 n;
	// fix some of them
	switch (m.command)
	{
	case 0xE: // fine volslide
		// hackaround to get almost-right behavior for fine slides (i think!)
		if(m.param == 0)
			/* nothing */;
		else if(m.param == 0xF0)
			m.param = 0xEF;
		else if(m.param == 0x0F)
			m.param = 0xFE;
		else if(m.param & 0xF0)
			m.param |= 0x0F;
		else
			m.param |= 0xF0;
		break;
	case 0xF: // set finetune
		// we don't implement this, but let's at least import the value
		m.param = 0x20 | MIN(m.param >> 4, 0x0F);
		break;
	case 0x14: // fine slide up
	case 0x15: // fine slide down
		// this is about as close as we can do...
		if(m.param >> 4)
			m.param = 0xF0 | MIN(m.param >> 4, 0x0F);
		else
			m.param |= 0xE0;
		break;
	case 0x16: // cutoff
		m.param >>= 1;
		break;
	case 0x1F: // set global volume
		m.param = MIN(m.param << 1, 0xFF);
		break;
	case 0x21:
		n = 0;
		switch (m.param >> 4)
		{
		case 0:
			/* undefined, but since S0x does nothing in IT anyway, we won't care.
			this is here to allow S00 to pick up the previous value (assuming IMF
			even does that -- I haven't actually tried it) */
			break;
		default: // undefined
		case 0x1: // set filter
		case 0xF: // invert loop
			m.command = CMD_NONE;
			break;
		case 0x3: // glissando
			n = 0x20;
			break;
		case 0x5: // vibrato waveform
			n = 0x30;
			break;
		case 0x8: // tremolo waveform
			n = 0x40;
			break;
		case 0xA: // pattern loop
			n = 0xB0;
			break;
		case 0xB: // pattern delay
			n = 0xE0;
			break;
		case 0xC: // note cut
		case 0xD: // note delay
			// Apparently, Imago Orpheus doesn't cut samples on tick 0.
			if(!m.param)
				m.command = CMD_NONE;
			break;
		case 0xE: // ignore envelope
			/* predicament: we can only disable one envelope at a time.
			volume is probably most noticeable, so let's go with that.
			(... actually, orpheus doesn't even seem to implement this at all) */
			m.param = 0x77;
			break;
		case 0x18: // sample offset
			// O00 doesn't pick up the previous value
			if(!m.param)
				m.command = CMD_NONE;
			break;
		}
		if(n)
			m.param = n | (m.param & 0x0F);
		break;
	}
	m.command = (m.command < CountOf(imfEffects)) ? imfEffects[m.command] : CMD_NONE;
	if(m.command == CMD_VOLUME && m.volcmd == VOLCMD_NONE)
	{
		m.volcmd = VOLCMD_VOLUME;
		m.vol = m.param;
		m.command = CMD_NONE;
		m.param = 0;
	}
}


static bool ValidateHeader(const IMFFileHeader &fileHeader)
{
	if(std::memcmp(fileHeader.im10, "IM10", 4)
		|| fileHeader.ordNum > 256
		|| fileHeader.insNum >= MAX_INSTRUMENTS
		)
	{
		return false;
	}
	bool channelFound = false;
	for(uint8 chn = 0; chn < 32; chn++)
	{
		switch(fileHeader.channels[chn].status)
		{
		case 0: // enabled; don't worry about it
			channelFound = true;
			break;
		case 1: // mute
			channelFound = true;
			break;
		case 2: // disabled
			// nothing
			break;
		default: // uhhhh.... freak out
			return false;
		}
	}
	if(!channelFound)
	{
		return false;
	}
	return true;
}


static uint64 GetHeaderMinimumAdditionalSize(const IMFFileHeader &fileHeader)
{
	MPT_UNREFERENCED_PARAMETER(fileHeader);
	return 256;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderIMF(MemoryFileReader file, const uint64 *pfilesize)
{
	IMFFileHeader fileHeader;
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


bool CSoundFile::ReadIMF(FileReader &file, ModLoadingFlags loadFlags)
{
	IMFFileHeader fileHeader;
	file.Rewind();
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

	// Read channel configuration
	std::bitset<32> ignoreChannels; // bit set for each channel that's completely disabled
	uint8 detectedChannels = 0;
	for(uint8 chn = 0; chn < 32; chn++)
	{
		ChnSettings[chn].Reset();
		ChnSettings[chn].nPan = fileHeader.channels[chn].panning * 256 / 255;

		mpt::String::Read<mpt::String::nullTerminated>(ChnSettings[chn].szName, fileHeader.channels[chn].name);

		// TODO: reverb/chorus?
		switch(fileHeader.channels[chn].status)
		{
		case 0: // enabled; don't worry about it
			detectedChannels = chn + 1;
			break;
		case 1: // mute
			ChnSettings[chn].dwFlags = CHN_MUTE;
			detectedChannels = chn + 1;
			break;
		case 2: // disabled
			ChnSettings[chn].dwFlags = CHN_MUTE;
			ignoreChannels[chn] = true;
			break;
		default: // uhhhh.... freak out
			//fprintf(stderr, "imf: channel %d has unknown status %d\n", n, hdr.channels[n].status);
			return false;
		}
	}
	if(!detectedChannels)
	{
		return false;
	}
	
	if(loadFlags == onlyVerifyHeader)
	{
		return true;
	}

	InitializeGlobals(MOD_TYPE_IMF);
	m_nChannels = detectedChannels;

	//From mikmod: work around an Orpheus bug
	if(fileHeader.channels[0].status == 0)
	{
		CHANNELINDEX chn;
		for(chn = 1; chn < 16; chn++)
			if(fileHeader.channels[chn].status != 1)
				break;
		if(chn == 16)
			for(chn = 1; chn < 16; chn++)
				ChnSettings[chn].dwFlags.reset(CHN_MUTE);
	}

	// Song Name
	mpt::String::Read<mpt::String::nullTerminated>(m_songName, fileHeader.title);

	m_SongFlags = (fileHeader.flags & IMFFileHeader::linearSlides) ? SONG_LINEARSLIDES : SongFlags(0);
	m_nDefaultSpeed = fileHeader.tempo;
	m_nDefaultTempo.Set(fileHeader.bpm);
	m_nDefaultGlobalVolume = Clamp<uint8, uint8>(fileHeader.master, 0, 64) * 4;
	m_nSamplePreAmp = Clamp<uint8, uint8>(fileHeader.amp, 4, 127);

	m_nInstruments = fileHeader.insNum;
	m_nSamples = 0; // Will be incremented later

	uint8 orders[256];
	file.ReadArray(orders);
	ReadOrderFromArray(Order(), orders, fileHeader.ordNum, uint16_max, 0xFF);

	// Read patterns
	if(loadFlags & loadPatternData)
		Patterns.ResizeArray(fileHeader.patNum);
	for(PATTERNINDEX pat = 0; pat < fileHeader.patNum; pat++)
	{
		const uint16 length = file.ReadUint16LE(), numRows = file.ReadUint16LE();
		FileReader patternChunk = file.ReadChunk(length - 4);

		if(!(loadFlags & loadPatternData) || !Patterns.Insert(pat, numRows))
		{
			continue;
		}

		ModCommand junkNote;
		ROWINDEX row = 0;
		while(row < numRows)
		{
			uint8 mask = patternChunk.ReadUint8();
			if(mask == 0)
			{
				row++;
				continue;
			}

			uint8 channel = mask & 0x1F;
			ModCommand &m = ignoreChannels[channel] ? junkNote : *Patterns[pat].GetpModCommand(row, channel);

			if(mask & 0x20)
			{
				// Read note/instrument
				m.note = patternChunk.ReadUint8();
				m.instr = patternChunk.ReadUint8();

				if(m.note == 160)
				{
					m.note = NOTE_KEYOFF;
				} else if(m.note == 255)
				{
					m.note = NOTE_NONE;
				} else
				{
					m.note = (m.note >> 4) * 12 + (m.note & 0x0F) + 12 + 1;
					if(!m.IsNoteOrEmpty())
					{
						m.note = NOTE_NONE;
					}
				}
			}
			if((mask & 0xC0) == 0xC0)
			{
				// Read both effects and figure out what to do with them
				uint8 e1c = patternChunk.ReadUint8();	// Command 1
				uint8 e1d = patternChunk.ReadUint8();	// Data 1
				uint8 e2c = patternChunk.ReadUint8();	// Command 2
				uint8 e2d = patternChunk.ReadUint8();	// Data 2

				if(e1c == 0x0C)
				{
					m.vol = MIN(e1d, 0x40);
					m.volcmd = VOLCMD_VOLUME;
					m.command = e2c;
					m.param = e2d;
				} else if(e2c == 0x0C)
				{
					m.vol = MIN(e2d, 0x40);
					m.volcmd = VOLCMD_VOLUME;
					m.command = e1c;
					m.param = e1d;
				} else if(e1c == 0x0A)
				{
					m.vol = e1d * 64 / 255;
					m.volcmd = VOLCMD_PANNING;
					m.command = e2c;
					m.param = e2d;
				} else if(e2c == 0x0A)
				{
					m.vol = e2d * 64 / 255;
					m.volcmd = VOLCMD_PANNING;
					m.command = e1c;
					m.param = e1d;
				} else
				{
					/* check if one of the effects is a 'global' effect
					-- if so, put it in some unused channel instead.
					otherwise pick the most important effect. */
					m.command = e2c;
					m.param = e2d;
				}
			} else if(mask & 0xC0)
			{
				// There's one effect, just stick it in the effect column
				m.command = patternChunk.ReadUint8();
				m.param = patternChunk.ReadUint8();
			}
			if(m.command)
				ImportIMFEffect(m);
		}
	}

	SAMPLEINDEX firstSample = 1; // first sample index of the current instrument

	// read instruments
	for(INSTRUMENTINDEX ins = 0; ins < GetNumInstruments(); ins++)
	{
		ModInstrument *instr = AllocateInstrument(ins + 1);
		IMFInstrument instrumentHeader;
		if(!file.ReadStruct(instrumentHeader) || instr == nullptr)
		{
			continue;
		}

		// Orpheus does not check this!
		//if(memcmp(instrumentHeader.ii10, "II10", 4) != 0)
		//	return false;
		instrumentHeader.ConvertToMPT(*instr, firstSample);

		// Read this instrument's samples
		for(SAMPLEINDEX smp = 0; smp < instrumentHeader.smpNum; smp++)
		{
			IMFSample sampleHeader;
			file.ReadStruct(sampleHeader);

			const SAMPLEINDEX smpID = firstSample + smp;
			if(memcmp(sampleHeader.is10, "IS10", 4) || smpID >= MAX_SAMPLES)
			{
				continue;
			}

			m_nSamples = smpID;
			ModSample &sample = Samples[smpID];

			sampleHeader.ConvertToMPT(sample);
			mpt::String::Copy(m_szNames[smpID], sample.filename);

			if(sampleHeader.length)
			{
				FileReader sampleChunk = file.ReadChunk(sampleHeader.length);
				if(loadFlags & loadSampleData)
				{
					SampleIO(
						sample.uFlags[CHN_16BIT] ? SampleIO::_16bit : SampleIO::_8bit,
						SampleIO::mono,
						SampleIO::littleEndian,
						SampleIO::signedPCM)
						.ReadSample(sample, sampleChunk);
				}
			}
		}
		firstSample += instrumentHeader.smpNum;
	}

	return true;
}


OPENMPT_NAMESPACE_END
