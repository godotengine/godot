/*
 * load_dmf.cpp
 * ------------
 * Purpose: DMF module loader (X-Tracker by D-LUSiON).
 * Notes  : If it wasn't already outdated when the tracker was released, this would be a rather interesting
 *          and in some parts even sophisticated format - effect columns are separated by effect type, an easy to
 *          understand BPM tempo mode, effect durations are always divided into a 256th row, vibrato effects are
 *          specified by period length and the same 8-Bit granularity is used for both volume and panning.
 *          Unluckily, this format does not offer any envelopes or multi-sample instruments, and bidi sample loops
 *          are missing as well, so it was already well behind FT2 and IT back then.
 * Authors: Johannes Schultz (mostly based on DMF.TXT, DMF_EFFC.TXT, trial and error and some invaluable hints by Zatzen)
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "ChunkReader.h"
#include <stdexcept>

OPENMPT_NAMESPACE_BEGIN

// DMF header
struct DMFFileHeader
{
	char   signature[4];	// "DDMF"
	uint8  version;			// 1 - 7 are beta versions, 8 is the official thing, 10 is xtracker32
	char   tracker[8];		// "XTRACKER"
	char   songname[30];
	char   composer[20];
	uint8  creationDay;
	uint8  creationMonth;
	uint8  creationYear;
};

MPT_BINARY_STRUCT(DMFFileHeader, 66)

struct DMFChunk
{
	// 32-Bit chunk identifiers
	enum ChunkIdentifiers
	{
		idCMSG	= MAGIC4LE('C','M','S','G'),	// Song message
		idSEQU	= MAGIC4LE('S','E','Q','U'),	// Order list
		idPATT	= MAGIC4LE('P','A','T','T'),	// Patterns
		idSMPI	= MAGIC4LE('S','M','P','I'),	// Sample headers
		idSMPD	= MAGIC4LE('S','M','P','D'),	// Sample data
		idSMPJ	= MAGIC4LE('S','M','P','J'),	// Sample jump table (XTracker 32 only)
		idENDE	= MAGIC4LE('E','N','D','E'),	// Last four bytes of DMF file
		idSETT	= MAGIC4LE('S','E','T','T'),	// Probably contains GUI settings
	};

	uint32le id;
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

MPT_BINARY_STRUCT(DMFChunk, 8)

// Order list
struct DMFSequence
{
	uint16le loopStart;
	uint16le loopEnd;
	// order list follows here ...
};

MPT_BINARY_STRUCT(DMFSequence, 4)

// Pattern header (global)
struct DMFPatterns
{
	uint16le numPatterns;	// 1..1024 patterns
	uint8le  numTracks;		// 1..32 channels
};

MPT_BINARY_STRUCT(DMFPatterns, 3)

// Pattern header (for each pattern)
struct DMFPatternHeader
{
	uint8le  numTracks;	// 1..32 channels
	uint8le  beat;		// [hi|lo] -> hi = rows per beat, lo = reserved
	uint16le numRows;
	uint32le patternLength;
	// patttern data follows here ...
};

MPT_BINARY_STRUCT(DMFPatternHeader, 8)

// Sample header
struct DMFSampleHeader
{
	enum SampleFlags
	{
		// Sample flags
		smpLoop		= 0x01,
		smp16Bit	= 0x02,
		smpCompMask = 0x0C,
		smpComp1	= 0x04,	// Compression type 1
		smpComp2	= 0x08,	// Compression type 2 (unused)
		smpComp3	= 0x0C,	// Compression type 3 (ditto)
		smpLibrary	= 0x80,	// Sample is stored in a library
	};

	uint32le length;
	uint32le loopStart;
	uint32le loopEnd;
	uint16le c3freq;		// 1000..45000hz
	uint8le  volume;		// 0 = ignore
	uint8le  flags;

	// Convert an DMFSampleHeader to OpenMPT's internal sample representation.
	void ConvertToMPT(ModSample &mptSmp) const
	{
		mptSmp.Initialize();
		mptSmp.nLength = length;
		mptSmp.nSustainStart = loopStart;
		mptSmp.nSustainEnd = loopEnd;

		mptSmp.nC5Speed = c3freq;
		mptSmp.nGlobalVol = 64;
		if(volume)
			mptSmp.nVolume = volume + 1;
		else
			mptSmp.nVolume = 256;
		mptSmp.uFlags.set(SMP_NODEFAULTVOLUME, volume == 0);

		if((flags & smpLoop) != 0 && mptSmp.nSustainEnd > mptSmp.nSustainStart)
		{
			mptSmp.uFlags.set(CHN_SUSTAINLOOP);
		}
		if((flags & smp16Bit) != 0)
		{
			mptSmp.uFlags.set(CHN_16BIT);
			mptSmp.nLength /= 2;
			mptSmp.nSustainStart /= 2;
			mptSmp.nSustainEnd /= 2;
		}
	}
};

MPT_BINARY_STRUCT(DMFSampleHeader, 16)

// Sample header tail (between head and tail, there might be the library name of the sample, depending on the DMF version)
struct DMFSampleHeaderTail
{
	uint16le filler;
	uint32le crc32;
};

MPT_BINARY_STRUCT(DMFSampleHeaderTail, 6)


// Pattern translation memory
struct DMFPatternSettings
{
	struct ChannelState
	{
		ModCommand::NOTE noteBuffer;	// Note buffer
		ModCommand::NOTE lastNote;		// Last played note on channel
		uint8 vibratoType;				// Last used vibrato type on channel
		uint8 tremoloType;				// Last used tremolo type on channel
		uint8 highOffset;				// Last used high offset on channel
		bool playDir;					// Sample play direction... false = forward (default)

		ChannelState()
		{
			noteBuffer = lastNote = NOTE_NONE;
			vibratoType = 8;
			tremoloType = 4;
			highOffset = 6;
			playDir = false;
		}
	};

	std::vector<ChannelState> channels;		// Memory for each channel's state
	bool realBPMmode;						// true = BPM mode
	uint8 beat;								// Rows per beat
	uint8 tempoTicks;						// Tick mode param
	uint8 tempoBPM;							// BPM mode param
	uint8 internalTicks;					// Ticks per row in final pattern

	DMFPatternSettings(CHANNELINDEX numChannels) : channels(numChannels)
	{
		realBPMmode = false;
		beat = 0;
		tempoTicks = 32;
		tempoBPM = 120;
		internalTicks = 6;
	}
};


// Convert portamento value (not very accurate due to X-Tracker's higher granularity, to say the least)
static uint8 DMFporta2MPT(uint8 val, const uint8 internalTicks, const bool hasFine)
{
	if(val == 0)
		return 0;
	else if((val <= 0x0F && hasFine) || internalTicks < 2)
		return (val | 0xF0);
	else
		return std::max<uint8>(1, (val / (internalTicks - 1)));	// no porta on first tick!
}


// Convert portamento / volume slide value (not very accurate due to X-Tracker's higher granularity, to say the least)
static uint8 DMFslide2MPT(uint8 val, const uint8 internalTicks, const bool up)
{
	val = std::max<uint8>(1, val / 4);
	const bool isFine = (val < 0x0F) || (internalTicks < 2);
	if(!isFine)
		val = std::max<uint8>(1, (val + internalTicks - 2) / (internalTicks - 1));	// no slides on first tick! "+ internalTicks - 2" for rounding precision

	if(up)
		return (isFine ? 0x0F : 0x00) | (val << 4);
	else
		return (isFine ? 0xF0 : 0x00) | (val & 0x0F);

}


// Calculate tremor on/off param
static uint8 DMFtremor2MPT(uint8 val, const uint8 internalTicks)
{
	uint8 ontime = (val >> 4);
	uint8 offtime = (val & 0x0F);
	ontime = static_cast<uint8>(Clamp(ontime * internalTicks / 15, 1, 15));
	offtime = static_cast<uint8>(Clamp(offtime * internalTicks / 15, 1, 15));
	return (ontime << 4) | offtime;
}


// Calculate delay parameter for note cuts / delays
static uint8 DMFdelay2MPT(uint8 val, const uint8 internalTicks)
{
	int newval = (int)val * (int)internalTicks / 255;
	Limit(newval, 0, 15);
	return (uint8)newval;
}


// Convert vibrato-style command parameters
static uint8 DMFvibrato2MPT(uint8 val, const uint8 internalTicks)
{
	// MPT: 1 vibrato period == 64 ticks... we have internalTicks ticks per row.
	// X-Tracker: Period length specified in rows!
	const int periodInTicks = MAX(1, (val >> 4)) * internalTicks;
	const uint8 matchingPeriod = (uint8)Clamp((128 / periodInTicks), 1, 15);
	return (matchingPeriod << 4) | MAX(1, (val & 0x0F));
}


// Try using effect memory (zero paramer) to give the effect swapper some optimization hints.
static void ApplyEffectMemory(const ModCommand *m, ROWINDEX row, CHANNELINDEX numChannels, uint8 effect, uint8 &param)
{
	if(effect == CMD_NONE || param == 0)
	{
		return;
	}

	const bool isTonePortaEffect = (effect == CMD_PORTAMENTOUP || effect == CMD_PORTAMENTODOWN || effect == CMD_TONEPORTAMENTO);
	const bool isVolSlideEffect = (effect == CMD_VOLUMESLIDE || effect == CMD_TONEPORTAVOL || effect == CMD_VIBRATOVOL);

	while(row > 0)
	{
		m -= numChannels;
		row--;

		// First, keep some extra rules in mind for portamento, where effect memory is shared between various commands.
		bool isSame = (effect == m->command);
		if(isTonePortaEffect && (m->command == CMD_PORTAMENTOUP || m->command == CMD_PORTAMENTODOWN || m->command == CMD_TONEPORTAMENTO))
		{
			if(m->param < 0xE0)
			{
				// Avoid effect param for fine slides, or else we could accidentally put this command in the volume column, where fine slides won't work!
				isSame = true;
			} else
			{
				return;
			}
		} else if(isVolSlideEffect && (m->command == CMD_VOLUMESLIDE || m->command == CMD_TONEPORTAVOL || m->command == CMD_VIBRATOVOL))
		{
			isSame = true;
		}
		if(isTonePortaEffect
			&& (m->volcmd == VOLCMD_PORTAUP || m->volcmd == VOLCMD_PORTADOWN || m->volcmd == VOLCMD_TONEPORTAMENTO)
			&& m->vol != 0)
		{
			// Uuh... Don't even try
			return;
		} else if(isVolSlideEffect
			&& (m->volcmd == VOLCMD_FINEVOLUP || m->volcmd == VOLCMD_FINEVOLDOWN || m->volcmd == VOLCMD_VOLSLIDEUP || m->volcmd == VOLCMD_VOLSLIDEDOWN)
			&& m->vol != 0)
		{
			// Same!
			return;
		}

		if(isSame)
		{
			if(param != m->param && m->param != 0)
			{
				// No way to optimize this
				return;
			} else if(param == m->param)
			{
				// Yay!
				param = 0;
				return;
			}
		}
	}
}


static PATTERNINDEX ConvertDMFPattern(FileReader &file, DMFPatternSettings &settings, CSoundFile &sndFile)
{
	// Pattern flags
	enum PatternFlags
	{
		// Global Track
		patGlobPack	= 0x80,	// Pack information for global track follows
		patGlobMask	= 0x3F,	// Mask for global effects
		// Note tracks
		patCounter	= 0x80,	// Pack information for current channel follows
		patInstr	= 0x40,	// Instrument number present
		patNote		= 0x20,	// Note present
		patVolume	= 0x10,	// Volume present
		patInsEff	= 0x08,	// Instrument effect present
		patNoteEff	= 0x04,	// Note effect present
		patVolEff	= 0x02,	// Volume effect stored
	};

	file.Rewind();
	
	DMFPatternHeader patHead;
	file.ReadStruct(patHead);

	const ROWINDEX numRows = Clamp(ROWINDEX(patHead.numRows), ROWINDEX(1), MAX_PATTERN_ROWS);
	const PATTERNINDEX pat = sndFile.Patterns.InsertAny(numRows);
	if(pat == PATTERNINDEX_INVALID)
	{
		return pat;
	}

	PatternRow m = sndFile.Patterns[pat].GetRow(0);
	const CHANNELINDEX numChannels = std::min<CHANNELINDEX>(sndFile.GetNumChannels() - 1, patHead.numTracks);

	// When breaking to a pattern with less channels that the previous pattern,
	// all voices in the now unused channels are killed:
	for(CHANNELINDEX chn = numChannels + 1; chn < sndFile.GetNumChannels(); chn++)
	{
		m[chn].note = NOTE_NOTECUT;
	}

	// Initialize tempo stuff
	settings.beat = (patHead.beat >> 4);
	bool tempoChange = settings.realBPMmode;
	uint8 writeDelay = 0;

	// Counters for channel packing (including global track)
	std::vector<uint8> channelCounter(numChannels + 1, 0);

	for(ROWINDEX row = 0; row < numRows; row++)
	{
		// Global track info counter reached 0 => read global track data
		if(channelCounter[0] == 0)
		{
			uint8 globalInfo = file.ReadUint8();
			// 0x80: Packing counter (if not present, counter stays at 0)
			if((globalInfo & patGlobPack) != 0)
			{
				channelCounter[0] = file.ReadUint8();
			}

			globalInfo &= patGlobMask;

			uint8 globalData = 0;
			if(globalInfo != 0)
			{
				globalData = file.ReadUint8();
			}

			switch(globalInfo)
			{
			case 1:		// Set Tick Frame Speed
				settings.realBPMmode = false;
				settings.tempoTicks = std::max(uint8(1), globalData);	// Tempo in 1/4 rows per second
				settings.tempoBPM = 0;									// Automatically updated by X-Tracker
				tempoChange = true;
				break;
			case 2:		// Set BPM Speed (real BPM mode)
				if(globalData)	// DATA = 0 doesn't do anything
				{
					settings.realBPMmode = true;
					settings.tempoBPM = globalData;		// Tempo in real BPM (depends on rows per beat)
					if(settings.beat != 0)
					{
						settings.tempoTicks = (globalData * settings.beat * 15);	// Automatically updated by X-Tracker
					}
					tempoChange = true;
				}
				break;
			case 3:		// Set Beat
				settings.beat = (globalData >> 4);
				if(settings.beat != 0)
				{
					// Tempo changes only if we're in real BPM mode
					tempoChange = settings.realBPMmode;
				} else
				{
					// If beat is 0, change to tick speed mode, but keep current tempo
					settings.realBPMmode = false;
				}
				break;
			case 4:		// Tick Delay
				writeDelay = globalData;
				break;
			case 5:		// Set External Flag
				break;
			case 6:		// Slide Speed Up
				if(globalData > 0)
				{
					uint8 &tempoData = (settings.realBPMmode) ? settings.tempoBPM : settings.tempoTicks;
					if(tempoData < 256 - globalData)
					{
						tempoData += globalData;
					} else
					{
						tempoData = 255;
					}
					tempoChange = true;
				}
				break;
			case 7:		// Slide Speed Down
				if(globalData > 0)
				{
					uint8 &tempoData = (settings.realBPMmode) ? settings.tempoBPM : settings.tempoTicks;
					if(tempoData > 1 + globalData)
					{
						tempoData -= globalData;
					} else
					{
						tempoData = 1;
					}
					tempoChange = true;
				}
				break;
			}
		} else
		{
			channelCounter[0]--;
		}

		// These will eventually be written to the pattern
		int speed = 0, tempo = 0;

		if(tempoChange)
		{
			// Can't do anything if we're in BPM mode and there's no rows per beat set...
			if(!settings.realBPMmode || settings.beat)
			{
				// My approach to convert X-Tracker's "tick speed" (1/4 rows per second):
				// Tempo * 6 / Speed = Beats per Minute
				// => Tempo * 6 / (Speed * 60) = Beats per Second
				// => Tempo * 24 / (Speed * 60) = Rows per Second (4 rows per beat at tempo 6)
				// => Tempo = 60 * Rows per Second * Speed / 24
				// For some reason, using settings.tempoTicks + 1 gives more accurate results than just settings.tempoTicks... (same problem in the old libmodplug DMF loader)
				// Original unoptimized formula:
				//const int tickspeed = (tempoRealBPMmode) ? MAX(1, (tempoData * beat * 4) / 60) : tempoData;
				const int tickspeed = (settings.realBPMmode) ? std::max(1, settings.tempoBPM * settings.beat * 2) : ((settings.tempoTicks + 1) * 30);
				// Try to find matching speed - try higher speeds first, so that effects like arpeggio and tremor work better.
				for(speed = 255; speed > 2; speed--)
				{
					// Original unoptimized formula:
					// tempo = 30 * tickspeed * speed / 48;
					tempo = tickspeed * speed / 48;
					if(tempo >= 32 && tempo <= 255)
					{
						break;
					}
				}
				Limit(tempo, 32, 255);
				settings.internalTicks = (uint8)speed;
			} else
			{
				tempoChange = false;
			}
		}

		m = sndFile.Patterns[pat].GetpModCommand(row, 1);	// Reserve first channel for global effects

		for(CHANNELINDEX chn = 1; chn <= numChannels; chn++, m++)
		{
			// Track info counter reached 0 => read track data
			if(channelCounter[chn] == 0)
			{
				const uint8 channelInfo = file.ReadUint8();
				////////////////////////////////////////////////////////////////
				// 0x80: Packing counter (if not present, counter stays at 0)
				if((channelInfo & patCounter) != 0)
				{
					channelCounter[chn] = file.ReadUint8();
				}

				////////////////////////////////////////////////////////////////
				// 0x40: Instrument
				bool slideNote = true;		// If there is no instrument number next to a note, the note is not retriggered!
				if((channelInfo & patInstr) != 0)
				{
					m->instr = file.ReadUint8();
					if(m->instr != 0)
					{
						slideNote = false;
					}
				}

				////////////////////////////////////////////////////////////////
				// 0x20: Note
				if((channelInfo & patNote) != 0)
				{
					m->note = file.ReadUint8();
					if(m->note >= 1 && m->note <= 108)
					{
						m->note = static_cast<uint8>(Clamp(m->note + 24, NOTE_MIN, NOTE_MAX));
						settings.channels[chn].lastNote = m->note;
					} else if(m->note >= 129 && m->note <= 236)
					{
						// "Buffer notes" for portamento (and other effects?) that are actually not played, but just "queued"...
						m->note = static_cast<uint8>(Clamp((m->note & 0x7F) + 24, NOTE_MIN, NOTE_MAX));
						settings.channels[chn].noteBuffer = m->note;
						m->note = NOTE_NONE;
					} else if(m->note == 255)
					{
						m->note = NOTE_NOTECUT;
					}
				}

				// If there's just an instrument number, but no note, retrigger sample.
				if(m->note == NOTE_NONE && m->instr > 0)
				{
					m->note = settings.channels[chn].lastNote;
					m->instr = 0;
				}

				if(m->IsNote())
				{
					settings.channels[chn].playDir = false;
				}

				uint8 effect1 = CMD_NONE, effect2 = CMD_NONE, effect3 = CMD_NONE;
				uint8 effectParam1 = 0, effectParam2 = 0, effectParam3 = 0;
				bool useMem2 = false, useMem3 = false;	// Effect can use memory if necessary

				////////////////////////////////////////////////////////////////
				// 0x10: Volume
				if((channelInfo & patVolume) != 0)
				{
					m->volcmd = VOLCMD_VOLUME;
					m->vol = (file.ReadUint8() + 2) / 4;	// Should be + 3 instead of + 2, but volume 1 is silent in X-Tracker.
				}

				////////////////////////////////////////////////////////////////
				// 0x08: Instrument effect
				if((channelInfo & patInsEff) != 0)
				{
					effect1 = file.ReadUint8();
					effectParam1 = file.ReadUint8();

					switch(effect1)
					{
					case 1:		// Stop Sample
						m->note = NOTE_NOTECUT;
						effect1 = CMD_NONE;
						break;
					case 2:		// Stop Sample Loop
						m->note = NOTE_KEYOFF;
						effect1 = CMD_NONE;
						break;
					case 3:		// Instrument Volume Override (aka "Restart")
						m->note = settings.channels[chn].lastNote;
						settings.channels[chn].playDir = false;
						effect1 = CMD_NONE;
						break;
					case 4:		// Sample Delay
						effectParam1 = DMFdelay2MPT(effectParam1, settings.internalTicks);
						if(effectParam1)
						{
							effect1 = CMD_S3MCMDEX;
							effectParam1 = 0xD0 | (effectParam1);
						} else
						{
							effect1 = CMD_NONE;
						}
						if(m->note == NOTE_NONE)
						{
							m->note = settings.channels[chn].lastNote;
							settings.channels[chn].playDir = false;
						}
						break;
					case 5:		// Tremolo Retrig Sample (who invented those stupid effect names?)
						effectParam1 = MAX(1, DMFdelay2MPT(effectParam1, settings.internalTicks));
						effect1 = CMD_RETRIG;
						settings.channels[chn].playDir = false;
						break;
					case 6:		// Offset
					case 7:		// Offset + 64k
					case 8:		// Offset + 128k
					case 9:		// Offset + 192k
						// Put high offset on previous row
						if(row > 0 && effect1 != settings.channels[chn].highOffset)
						{
							if(sndFile.Patterns[pat].WriteEffect(EffectWriter(CMD_S3MCMDEX, (0xA0 | (effect1 - 6))).Row(row - 1).Channel(chn).RetryPreviousRow()))
							{
								settings.channels[chn].highOffset = effect1;
							}
						}
						effect1 = CMD_OFFSET;
						if(m->note == NOTE_NONE)
						{
							// Offset without note does also work in DMF.
							m->note = settings.channels[chn].lastNote;
						}
						settings.channels[chn].playDir = false;
						break;
					case 10:	// Invert Sample play direction ("Tekkno Invert")
						effect1 = CMD_S3MCMDEX;
						if(settings.channels[chn].playDir == false)
							effectParam1 = 0x9F;
						else
							effectParam1 = 0x9E;
						settings.channels[chn].playDir = !settings.channels[chn].playDir;
						break;
					default:
						effect1 = CMD_NONE;
						break;
					}
				}

				////////////////////////////////////////////////////////////////
				// 0x04: Note effect
				if((channelInfo & patNoteEff) != 0)
				{
					effect2 = file.ReadUint8();
					effectParam2 = file.ReadUint8();

					switch(effect2)
					{
					case 1:		// Note Finetune
						effect2 = static_cast<ModCommand::COMMAND>(effectParam2 < 128 ? CMD_PORTAMENTOUP : CMD_PORTAMENTODOWN);
						if(effectParam2 > 128) effectParam2 = 255 - effectParam2 + 1;
						effectParam2 = 0xF0 | MIN(0x0F, effectParam2);	// Well, this is not too accurate...
						break;
					case 2:		// Note Delay (wtf is the difference to Sample Delay?)
						effectParam2 = DMFdelay2MPT(effectParam2, settings.internalTicks);
						if(effectParam2)
						{
							effect2 = CMD_S3MCMDEX;
							effectParam2 = 0xD0 | (effectParam2);
						} else
						{
							effect2 = CMD_NONE;
						}
						useMem2 = true;
						break;
					case 3:		// Arpeggio
						effect2 = CMD_ARPEGGIO;
						useMem2 = true;
						break;
					case 4:		// Portamento Up
					case 5:		// Portamento Down
						effectParam2 = DMFporta2MPT(effectParam2, settings.internalTicks, true);
						effect2 = static_cast<ModCommand::COMMAND>(effect2 == 4 ? CMD_PORTAMENTOUP : CMD_PORTAMENTODOWN);
						useMem2 = true;
						break;
					case 6:		// Portamento to Note
						if(m->note == NOTE_NONE)
						{
							m->note = settings.channels[chn].noteBuffer;
						}
						effectParam2 = DMFporta2MPT(effectParam2, settings.internalTicks, false);
						effect2 = CMD_TONEPORTAMENTO;
						useMem2 = true;
						break;
					case 7:		// Scratch to Note (neat! but we don't have such an effect...)
						m->note = static_cast<ModCommand::NOTE>(Clamp(effectParam2 + 25, NOTE_MIN, NOTE_MAX));
						effect2 = CMD_TONEPORTAMENTO;
						effectParam2 = 0xFF;
						useMem2 = true;
						break;
					case 8:		// Vibrato Sine
					case 9:		// Vibrato Triangle (ramp down should be close enough)
					case 10:	// Vibrato Square
						// Put vibrato type on previous row
						if(row > 0 && effect2 != settings.channels[chn].vibratoType)
						{
							if(sndFile.Patterns[pat].WriteEffect(EffectWriter(CMD_S3MCMDEX, (0x30 | (effect2 - 8))).Row(row - 1).Channel(chn).RetryPreviousRow()))
							{
								settings.channels[chn].vibratoType = effect2;
							}
						}
						effect2 = CMD_VIBRATO;
						effectParam2 = DMFvibrato2MPT(effectParam2, settings.internalTicks);
						useMem2 = true;
						break;
					case 11:	 // Note Tremolo
						effectParam2 = DMFtremor2MPT(effectParam2, settings.internalTicks);
						effect2 = CMD_TREMOR;
						useMem2 = true;
						break;
					case 12:	// Note Cut
						effectParam2 = DMFdelay2MPT(effectParam2, settings.internalTicks);
						if(effectParam2)
						{
							effect2 = CMD_S3MCMDEX;
							effectParam2 = 0xC0 | (effectParam2);
						} else
						{
							effect2 = CMD_NONE;
							m->note = NOTE_NOTECUT;
						}
						useMem2 = true;
						break;
					default:
						effect2 = CMD_NONE;
						break;
					}
				}

				////////////////////////////////////////////////////////////////
				// 0x02: Volume effect
				if((channelInfo & patVolEff) != 0)
				{
					effect3 = file.ReadUint8();
					effectParam3 = file.ReadUint8();

					switch(effect3)
					{
					case 1:		// Volume Slide Up
					case 2:		// Volume Slide Down
						effectParam3 = DMFslide2MPT(effectParam3, settings.internalTicks, (effect3 == 1));
						effect3 = CMD_VOLUMESLIDE;
						useMem3 = true;
						break;
					case 3:		// Volume Tremolo (actually this is Tremor)
						effectParam3 = DMFtremor2MPT(effectParam3, settings.internalTicks);
						effect3 = CMD_TREMOR;
						useMem3 = true;
						break;
					case 4:		// Tremolo Sine
					case 5:		// Tremolo Triangle (ramp down should be close enough)
					case 6:		// Tremolo Square
						// Put tremolo type on previous row
						if(row > 0 && effect3 != settings.channels[chn].tremoloType)
						{
							if(sndFile.Patterns[pat].WriteEffect(EffectWriter(CMD_S3MCMDEX, (0x40 | (effect3 - 4))).Row(row - 1).Channel(chn).RetryPreviousRow()))
							{
								settings.channels[chn].tremoloType = effect3;
							}
						}
						effect3 = CMD_TREMOLO;
						effectParam3 = DMFvibrato2MPT(effectParam3, settings.internalTicks);
						useMem3 = true;
						break;
					case 7:		// Set Balance
						effect3 = CMD_PANNING8;
						break;
					case 8:		// Slide Balance Left
					case 9:		// Slide Balance Right
						effectParam3 = DMFslide2MPT(effectParam3, settings.internalTicks, (effect3 == 8));
						effect3 = CMD_PANNINGSLIDE;
						useMem3 = true;
						break;
					case 10:	// Balance Vibrato Left/Right (always sine modulated)
						effect3 = CMD_PANBRELLO;
						effectParam3 = DMFvibrato2MPT(effectParam3, settings.internalTicks);
						useMem3 = true;
						break;
					default:
						effect3 = CMD_NONE;
						break;
					}
				}

				// Let's see if we can help the effect swapper by reducing some effect parameters to "continue" parameters.
				if(useMem2)
				{
					ApplyEffectMemory(m, row, sndFile.GetNumChannels(), effect2, effectParam2);
				}
				if(useMem3)
				{
					ApplyEffectMemory(m, row, sndFile.GetNumChannels(), effect3, effectParam3);
				}

				// I guess this is close enough to "not retriggering the note"
				if(slideNote && m->IsNote())
				{
					if(effect2 == CMD_NONE)
					{
						effect2 = CMD_TONEPORTAMENTO;
						effectParam2 = 0xFF;
					} else if(effect3 == CMD_NONE && effect2 != CMD_TONEPORTAMENTO)	// Tone portamentos normally go in effect #2
					{
						effect3 = CMD_TONEPORTAMENTO;
						effectParam3 = 0xFF;
					}
				}
				// If one of the effects is unused, temporarily put volume commands in there
				if(m->volcmd == VOLCMD_VOLUME)
				{
					if(effect2 == CMD_NONE)
					{
						effect2 = CMD_VOLUME;
						effectParam2 = m->vol;
						m->volcmd = VOLCMD_NONE;
					} else if(effect3 == CMD_NONE)
					{
						effect3 = CMD_VOLUME;
						effectParam3 = m->vol;
						m->volcmd = VOLCMD_NONE;
					}
				}

				ModCommand::TwoRegularCommandsToMPT(effect2, effectParam2, effect3, effectParam3);

				if(m->volcmd == VOLCMD_NONE && effect2 != VOLCMD_NONE)
				{
					m->volcmd = effect2;
					m->vol = effectParam2;
				}
				// Prefer instrument effects over any other effects
				if(effect1 != CMD_NONE)
				{
					m->command = effect1;
					m->param = effectParam1;
				} else if(effect3 != CMD_NONE)
				{
					m->command = effect3;
					m->param = effectParam3;
				}

			} else
			{
				channelCounter[chn]--;
			}
		}	// End for all channels

		// Now we can try to write tempo information.
		if(tempoChange)
		{
			tempoChange = false;
			
			sndFile.Patterns[pat].WriteEffect(EffectWriter(CMD_TEMPO, static_cast<ModCommand::PARAM>(tempo)).Row(row).Channel(0).RetryNextRow());
			sndFile.Patterns[pat].WriteEffect(EffectWriter(CMD_SPEED, static_cast<ModCommand::PARAM>(speed)).Row(row).RetryNextRow());
		}
		// Try to put delay effects somewhere as well
		if(writeDelay & 0xF0)
		{
			sndFile.Patterns[pat].WriteEffect(EffectWriter(CMD_S3MCMDEX, 0xE0 | (writeDelay >> 4)).Row(row).AllowMultiple());
		}
		if(writeDelay & 0x0F)
		{
			const uint8 param = (writeDelay & 0x0F) * settings.internalTicks / 15;
			sndFile.Patterns[pat].WriteEffect(EffectWriter(CMD_S3MCMDEX, 0x60u | Clamp(param, uint8(1), uint8(15))).Row(row).AllowMultiple());
		}
		writeDelay = 0;
	}	// End for all rows

	return pat;
}


static bool ValidateHeader(const DMFFileHeader &fileHeader)
{
	if(std::memcmp(fileHeader.signature, "DDMF", 4)
		|| !fileHeader.version || fileHeader.version > 10)
	{
		return false;
	}
	return true;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderDMF(MemoryFileReader file, const uint64 *pfilesize)
{
	DMFFileHeader fileHeader;
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


bool CSoundFile::ReadDMF(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();

	DMFFileHeader fileHeader;
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

	InitializeGlobals(MOD_TYPE_DMF);
	mpt::String::Read<mpt::String::spacePadded>(m_songName, fileHeader.songname);
	{
		std::string artist;
		mpt::String::Read<mpt::String::spacePadded>(artist, fileHeader.composer);
		m_songArtist = mpt::ToUnicode(mpt::CharsetCP437, artist);
	}

	FileHistory mptHistory;
	MemsetZero(mptHistory);
	mptHistory.loadDate.tm_mday = Clamp(fileHeader.creationDay, uint8(1), uint8(31));
	mptHistory.loadDate.tm_mon = Clamp(fileHeader.creationMonth, uint8(1), uint8(12)) - 1;
	mptHistory.loadDate.tm_year = fileHeader.creationYear;
	m_FileHistory.clear();
	m_FileHistory.push_back(mptHistory);

	// Go through all chunks now
	ChunkReader chunkFile(file);
	ChunkReader::ChunkList<DMFChunk> chunks = chunkFile.ReadChunks<DMFChunk>(1);
	FileReader chunk;

	// Read order list
	DMFSequence seqHeader;
	chunk = chunks.GetChunk(DMFChunk::idSEQU);
	if(!chunk.ReadStruct(seqHeader))
	{
		return false;
	}
	ReadOrderFromFile<uint16le>(Order(), chunk, (chunk.GetLength() - sizeof(DMFSequence)) / 2);

	// Read patterns
	chunk = chunks.GetChunk(DMFChunk::idPATT);
	if(chunk.IsValid() && (loadFlags & loadPatternData))
	{
		DMFPatterns patHeader;
		chunk.ReadStruct(patHeader);
		m_nChannels = Clamp<uint8, uint8>(patHeader.numTracks, 1, 32) + 1;	// + 1 for global track (used for tempo stuff)

		std::vector<FileReader> patternChunks;
		patternChunks.reserve(patHeader.numPatterns);

		// First, find out where all of our patterns are...
		for(PATTERNINDEX pat = 0; pat < patHeader.numPatterns; pat++)
		{
			DMFPatternHeader header;
			chunk.ReadStruct(header);
			chunk.SkipBack(sizeof(header));
			patternChunks.push_back(chunk.ReadChunk(sizeof(header) + header.patternLength));
		}

		// Now go through the order list and load them.
		DMFPatternSettings settings(GetNumChannels());

		Patterns.ResizeArray(Order().GetLength());
		for(ORDERINDEX ord = 0; ord < Order().GetLength(); ord++)
		{
			// Create one pattern for each order item, as the same pattern can be played with different settings
			PATTERNINDEX pat = Order()[ord];
			if(pat < patternChunks.size())
			{
				pat = ConvertDMFPattern(patternChunks[pat], settings, *this);
				Order()[ord] = pat;
				// Loop end?
				if(pat != PATTERNINDEX_INVALID && ord == seqHeader.loopEnd && (seqHeader.loopStart > 0 || ord < Order().GetLastIndex()))
				{
					Patterns[pat].WriteEffect(EffectWriter(CMD_POSITIONJUMP, static_cast<ModCommand::PARAM>(seqHeader.loopStart)).Row(Patterns[pat].GetNumRows() - 1).RetryPreviousRow());
				}
			}
		}
	}

	// Read song message
	chunk = chunks.GetChunk(DMFChunk::idCMSG);
	if(chunk.IsValid())
	{
		// The song message seems to start at a 1 byte offset.
		// The skipped byte seems to always be 0.
		// This also matches how XT 1.03 itself displays the song message.
		chunk.Skip(1);
		m_songMessage.ReadFixedLineLength(chunk, chunk.GetLength() - 1, 40, 0);
	}
	
	// Read sample headers + data
	FileReader sampleDataChunk = chunks.GetChunk(DMFChunk::idSMPD);
	chunk = chunks.GetChunk(DMFChunk::idSMPI);
	m_nSamples = chunk.ReadUint8();

	for(SAMPLEINDEX smp = 1; smp <= GetNumSamples(); smp++)
	{
		chunk.ReadSizedString<uint8le, mpt::String::spacePadded>(m_szNames[smp]);
		DMFSampleHeader sampleHeader;
		ModSample &sample = Samples[smp];
		chunk.ReadStruct(sampleHeader);
		sampleHeader.ConvertToMPT(sample);

		if(fileHeader.version >= 8)
		{
			// Read library name in version 8 files
			chunk.ReadString<mpt::String::spacePadded>(sample.filename, 8);
		}

		// We don't care for the checksum of the sample data...
		chunk.Skip(sizeof(DMFSampleHeaderTail));

		// Now read the sample data from the data chunk
		FileReader sampleData = sampleDataChunk.ReadChunk(sampleDataChunk.ReadUint32LE());
		if(sampleData.IsValid() && (loadFlags & loadSampleData))
		{
			SampleIO(
				sample.uFlags[CHN_16BIT] ? SampleIO::_16bit : SampleIO::_8bit,
				SampleIO::mono,
				SampleIO::littleEndian,
				(sampleHeader.flags & DMFSampleHeader::smpCompMask) == DMFSampleHeader::smpComp1 ? SampleIO::DMF : SampleIO::signedPCM)
				.ReadSample(sample, sampleData);
		}
	}

	InitializeChannels();
	m_SongFlags = SONG_LINEARSLIDES | SONG_ITCOMPATGXX;	// this will be converted to IT format by MPT. SONG_ITOLDEFFECTS is not set because of tremor and vibrato.
	m_nDefaultSpeed = 6;
	m_nDefaultTempo.Set(120);
	m_nDefaultGlobalVolume = 256;
	m_nSamplePreAmp = m_nVSTiVolume = 48;

	return true;
}


///////////////////////////////////////////////////////////////////////
// DMF Compression

struct DMFHNode
{
	int16 left, right;
	uint8 value;
};

struct DMFHTree
{
	const uint8 *ibuf, *ibufmax;
	uint32 bitbuf;
	int bitnum;
	int lastnode, nodecount;
	DMFHNode nodes[256];

	// DMF Huffman ReadBits
	uint8 DMFReadBits(int nbits)
	{
		if(bitnum < nbits)
		{
			if(ibuf < ibufmax)
			{
				bitbuf |= (((uint32)(*ibuf++)) << bitnum);
				bitnum += 8;
			} else
			{
				throw std::range_error("Truncated DMF sample block");
			}
		}

		uint8 v = static_cast<uint8>(bitbuf & ((1 << nbits) - 1));
		bitbuf >>= nbits;
		bitnum -= nbits;
		return v;
	}


	//
	// tree: [8-bit value][12-bit index][12-bit index] = 32-bit
	//

	void DMFNewNode()
	{
		uint8 isleft, isright;
		int actnode;

		actnode = nodecount;
		if(actnode > 255) return;
		nodes[actnode].value = DMFReadBits(7);
		isleft = DMFReadBits(1);
		isright = DMFReadBits(1);
		actnode = lastnode;
		if(actnode > 255) return;
		nodecount++;
		lastnode = nodecount;
		if(isleft)
		{
			nodes[actnode].left = (int16)lastnode;
			DMFNewNode();
		} else
		{
			nodes[actnode].left = -1;
		}
		lastnode = nodecount;
		if(isright)
		{
			nodes[actnode].right = (int16)lastnode;
			DMFNewNode();
		} else
		{
			nodes[actnode].right = -1;
		}
	}
};


uintptr_t DMFUnpack(uint8 *psample, const uint8 *ibuf, const uint8 *ibufmax, uint32 maxlen)
{
	DMFHTree tree;

	MemsetZero(tree);
	tree.ibuf = ibuf;
	tree.ibufmax = ibufmax;
	tree.DMFNewNode();
	uint8 value = 0, delta = 0;

	try
	{
		for(uint32 i = 0; i < maxlen; i++)
		{
			int actnode = 0;
			uint8 sign = tree.DMFReadBits(1);
			do
			{
				if(tree.DMFReadBits(1))
					actnode = tree.nodes[actnode].right;
				else
					actnode = tree.nodes[actnode].left;
				if(actnode > 255) break;
				delta = tree.nodes[actnode].value;
			} while((tree.nodes[actnode].left >= 0) && (tree.nodes[actnode].right >= 0));
			if(sign) delta ^= 0xFF;
			value += delta;
			psample[i] = value;
		}
	} catch(const std::range_error &)
	{
		//AddToLog(LogWarning, "Truncated DMF sample block");
	}
	return tree.ibuf - ibuf;
}


OPENMPT_NAMESPACE_END
