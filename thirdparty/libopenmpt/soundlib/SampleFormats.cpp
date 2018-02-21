/*
 * SampleFormats.cpp
 * -----------------
 * Purpose: Code for loading various more or less common sample and instrument formats.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Sndfile.h"
#include "mod_specifications.h"
#ifdef MODPLUG_TRACKER
#include "../mptrack/Moddoc.h"
#include "../mptrack/TrackerSettings.h"
#include "Dlsbank.h"
#endif //MODPLUG_TRACKER
#include "../soundlib/AudioCriticalSection.h"
#ifndef MODPLUG_NO_FILESAVE
#include "../common/mptFileIO.h"
#endif
#include "../common/misc_util.h"
#include "Tagging.h"
#include "ITTools.h"
#include "XMTools.h"
#include "S3MTools.h"
#include "WAVTools.h"
#include "../common/version.h"
#include "Loaders.h"
#include "ChunkReader.h"
#include "modsmp_ctrl.h"
#include "../soundbase/SampleFormatConverters.h"
#include "../soundbase/SampleFormatCopy.h"
#include "../soundlib/ModSampleCopy.h"


OPENMPT_NAMESPACE_BEGIN


bool CSoundFile::ReadSampleFromFile(SAMPLEINDEX nSample, FileReader &file, bool mayNormalize, bool includeInstrumentFormats)
{
	if(!nSample || nSample >= MAX_SAMPLES) return false;
	if(!ReadWAVSample(nSample, file, mayNormalize)
		&& !(includeInstrumentFormats && ReadXISample(nSample, file))
		&& !(includeInstrumentFormats && ReadITISample(nSample, file))
		&& !ReadAIFFSample(nSample, file, mayNormalize)
		&& !ReadITSSample(nSample, file)
		&& !(includeInstrumentFormats && ReadPATSample(nSample, file))
		&& !ReadIFFSample(nSample, file)
		&& !ReadS3ISample(nSample, file)
		&& !ReadAUSample(nSample, file, mayNormalize)
		&& !ReadFLACSample(nSample, file)
		&& !ReadOpusSample(nSample, file)
		&& !ReadVorbisSample(nSample, file)
		&& !ReadMP3Sample(nSample, file)
		&& !ReadMediaFoundationSample(nSample, file)
		)
	{
		return false;
	}

	if(nSample > GetNumSamples())
	{
		m_nSamples = nSample;
	}
	return true;
}


bool CSoundFile::ReadInstrumentFromFile(INSTRUMENTINDEX nInstr, FileReader &file, bool mayNormalize)
{
	if ((!nInstr) || (nInstr >= MAX_INSTRUMENTS)) return false;
	if(!ReadITIInstrument(nInstr, file)
		&& !ReadXIInstrument(nInstr, file)
		&& !ReadPATInstrument(nInstr, file)
		&& !ReadSFZInstrument(nInstr, file)
		// Generic read
		&& !ReadSampleAsInstrument(nInstr, file, mayNormalize))
	{
		bool ok = false;
#ifdef MODPLUG_TRACKER
		CDLSBank bank;
		if(bank.Open(file))
		{
			ok = bank.ExtractInstrument(*this, nInstr, 0, 0);
		}
#endif // MODPLUG_TRACKER
		if(!ok) return false;
	}

	if(nInstr > GetNumInstruments()) m_nInstruments = nInstr;
	return true;
}


bool CSoundFile::ReadSampleAsInstrument(INSTRUMENTINDEX nInstr, FileReader &file, bool mayNormalize)
{
	// Scanning free sample
	SAMPLEINDEX nSample = GetNextFreeSample(nInstr); // may also return samples which are only referenced by the current instrument
	if(nSample == SAMPLEINDEX_INVALID)
	{
		return false;
	}

	// Loading Instrument
	ModInstrument *pIns = new (std::nothrow) ModInstrument(nSample);
	if(pIns == nullptr)
	{
		return false;
	}
	if(!ReadSampleFromFile(nSample, file, mayNormalize, false))
	{
		delete pIns;
		return false;
	}

	// Remove all samples which are only referenced by the old instrument, except for the one we just loaded our new sample into.
	RemoveInstrumentSamples(nInstr, nSample);

	// Replace the instrument
	DestroyInstrument(nInstr, doNoDeleteAssociatedSamples);
	Instruments[nInstr] = pIns;

#if defined(MPT_ENABLE_FILEIO) && defined(MPT_EXTERNAL_SAMPLES)
	SetSamplePath(nSample, file.GetFileName());
#endif

	return true;
}


bool CSoundFile::DestroyInstrument(INSTRUMENTINDEX nInstr, deleteInstrumentSamples removeSamples)
{
	if(nInstr == 0 || nInstr >= MAX_INSTRUMENTS || !Instruments[nInstr]) return true;

	if(removeSamples == deleteAssociatedSamples)
	{
		RemoveInstrumentSamples(nInstr);
	}

	CriticalSection cs;

	ModInstrument *pIns = Instruments[nInstr];
	Instruments[nInstr] = nullptr;
	for(auto &chn : m_PlayState.Chn)
	{
		if(chn.pModInstrument == pIns)
			chn.pModInstrument = nullptr;
	}
	delete pIns;
	return true;
}


// Remove all unused samples from the given nInstr and keep keepSample if provided
bool CSoundFile::RemoveInstrumentSamples(INSTRUMENTINDEX nInstr, SAMPLEINDEX keepSample)
{
	if(Instruments[nInstr] == nullptr)
	{
		return false;
	}

	std::vector<bool> keepSamples(GetNumSamples() + 1, true);

	// Check which samples are used by the instrument we are going to nuke.
	auto referencedSamples = Instruments[nInstr]->GetSamples();
	for(auto sample : referencedSamples)
	{
		if(sample <= GetNumSamples())
		{
			keepSamples[sample] = false;
		}
	}

	// If we want to keep a specific sample, do so.
	if(keepSample != SAMPLEINDEX_INVALID)
	{
		if(keepSample <= GetNumSamples())
		{
			keepSamples[keepSample] = true;
		}
	}

	// Check if any of those samples are referenced by other instruments as well, in which case we want to keep them of course.
	for(INSTRUMENTINDEX nIns = 1; nIns <= GetNumInstruments(); nIns++) if (Instruments[nIns] != nullptr && nIns != nInstr)
	{
		Instruments[nIns]->GetSamples(keepSamples);
	}

	// Now nuke the selected samples.
	RemoveSelectedSamples(keepSamples);
	return true;
}

////////////////////////////////////////////////////////////////////////////////
//
// I/O From another song
//

bool CSoundFile::ReadInstrumentFromSong(INSTRUMENTINDEX targetInstr, const CSoundFile &srcSong, INSTRUMENTINDEX sourceInstr)
{
	if ((!sourceInstr) || (sourceInstr > srcSong.GetNumInstruments())
		|| (targetInstr >= MAX_INSTRUMENTS) || (!srcSong.Instruments[sourceInstr]))
	{
		return false;
	}
	if (m_nInstruments < targetInstr) m_nInstruments = targetInstr;

	ModInstrument *pIns = new (std::nothrow) ModInstrument();
	if(pIns == nullptr)
	{
		return false;
	}

	DestroyInstrument(targetInstr, deleteAssociatedSamples);

	Instruments[targetInstr] = pIns;
	*pIns = *srcSong.Instruments[sourceInstr];

	std::vector<SAMPLEINDEX> sourceSample;	// Sample index in source song
	std::vector<SAMPLEINDEX> targetSample;	// Sample index in target song
	SAMPLEINDEX targetIndex = 0;		// Next index for inserting sample

	for(auto &sample : pIns->Keyboard)
	{
		const SAMPLEINDEX sourceIndex = sample;
		if(sourceIndex > 0 && sourceIndex <= srcSong.GetNumSamples())
		{
			const auto entry = std::find(sourceSample.cbegin(), sourceSample.cend(), sourceIndex);
			if(entry == sourceSample.end())
			{
				// Didn't consider this sample yet, so add it to our map.
				targetIndex = GetNextFreeSample(targetInstr, targetIndex + 1);
				if(targetIndex <= GetModSpecifications().samplesMax)
				{
					sourceSample.push_back(sourceIndex);
					targetSample.push_back(targetIndex);
					sample = targetIndex;
				} else
				{
					sample = 0;
				}
			} else
			{
				// Sample reference has already been created, so only need to update the sample map.
				sample = *(entry - sourceSample.begin() + targetSample.begin());
			}
		} else
		{
			// Invalid or no source sample
			sample = 0;
		}
	}

#ifdef MODPLUG_TRACKER
	if(!strcmp(pIns->filename, "") && srcSong.GetpModDoc() != nullptr)
	{
		mpt::String::Copy(pIns->filename, srcSong.GetpModDoc()->GetPathNameMpt().GetFullFileName().ToLocale());
	}
#endif
	pIns->Convert(srcSong.GetType(), GetType());

	// Copy all referenced samples over
	for(size_t i = 0; i < targetSample.size(); i++)
	{
		ReadSampleFromSong(targetSample[i], srcSong, sourceSample[i]);
	}

	return true;
}


bool CSoundFile::ReadSampleFromSong(SAMPLEINDEX targetSample, const CSoundFile &srcSong, SAMPLEINDEX sourceSample)
{
	if(!sourceSample
		|| sourceSample > srcSong.GetNumSamples()
		|| (targetSample >= GetModSpecifications().samplesMax && targetSample > GetNumSamples()))
	{
		return false;
	}

	DestroySampleThreadsafe(targetSample);

	const ModSample &sourceSmp = srcSong.GetSample(sourceSample);
	ModSample &targetSmp = GetSample(targetSample);

	if(GetNumSamples() < targetSample) m_nSamples = targetSample;
	targetSmp = sourceSmp;
	strcpy(m_szNames[targetSample], srcSong.m_szNames[sourceSample]);

	if(sourceSmp.pSample)
	{
		targetSmp.pSample = nullptr;	// Don't want to delete the original sample!
		if(targetSmp.AllocateSample())
		{
			SmpLength nSize = sourceSmp.GetSampleSizeInBytes();
			memcpy(targetSmp.pSample, sourceSmp.pSample, nSize);
			targetSmp.PrecomputeLoops(*this, false);
		}
		// Remember on-disk path (for MPTM files), but don't implicitely enable on-disk storage
		// (we really don't want this for e.g. duplicating samples or splitting stereo samples)
#ifdef MPT_EXTERNAL_SAMPLES
		SetSamplePath(targetSample, srcSong.GetSamplePath(sourceSample));
#endif
		targetSmp.uFlags.reset(SMP_KEEPONDISK);
	}

#ifdef MODPLUG_TRACKER
	if(!strcmp(targetSmp.filename, "") && srcSong.GetpModDoc() != nullptr)
	{
		mpt::String::Copy(targetSmp.filename, mpt::ToCharset(GetCharsetInternal(), srcSong.GetpModDoc()->GetTitle()));
	}
#endif

	targetSmp.Convert(srcSong.GetType(), GetType());
	return true;
}


////////////////////////////////////////////////////////////////////////
// IMA ADPCM Support for WAV files


static bool IMAADPCMUnpack16(int16 *target, SmpLength sampleLen, FileReader file, uint16 blockAlign, uint32 numChannels)
{
	static const int32 IMAIndexTab[8] =  { -1, -1, -1, -1, 2, 4, 6, 8 };
	static const int32 IMAUnpackTable[90] =
	{
		7,     8,     9,     10,    11,    12,    13,    14,
		16,    17,    19,    21,    23,    25,    28,    31,
		34,    37,    41,    45,    50,    55,    60,    66,
		73,    80,    88,    97,    107,   118,   130,   143,
		157,   173,   190,   209,   230,   253,   279,   307,
		337,   371,   408,   449,   494,   544,   598,   658,
		724,   796,   876,   963,   1060,  1166,  1282,  1411,
		1552,  1707,  1878,  2066,  2272,  2499,  2749,  3024,
		3327,  3660,  4026,  4428,  4871,  5358,  5894,  6484,
		7132,  7845,  8630,  9493,  10442, 11487, 12635, 13899,
		15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
		32767, 0
	};

	if(target == nullptr || blockAlign < 4u * numChannels)
		return false;

	SmpLength samplePos = 0;
	sampleLen *= numChannels;
	while(file.CanRead(4u * numChannels) && samplePos < sampleLen)
	{
		FileReader block = file.ReadChunk(blockAlign);
		FileReader::PinnedRawDataView blockView = block.GetPinnedRawDataView();
		const mpt::byte *data = blockView.data();
		const uint32 blockSize = static_cast<uint32>(blockView.size());

		for(uint32 chn = 0; chn < numChannels; chn++)
		{
			// Block header
			int32 value = block.ReadInt16LE();
			int32 nIndex = block.ReadUint8();
			Limit(nIndex, 0, 89);
			block.Skip(1);

			SmpLength smpPos = samplePos + chn;
			uint32 dataPos = (numChannels + chn) * 4;
			// Block data
			while(smpPos <= (sampleLen - 8) && dataPos <= (blockSize - 4))
			{
				for(uint32 i = 0; i < 8; i++)
				{
					uint8 delta = data[dataPos];
					if(i & 1)
					{
						delta >>= 4;
						dataPos++;
					} else
					{
						delta &= 0x0F;
					}
					int32 v = IMAUnpackTable[nIndex] >> 3;
					if (delta & 1) v += IMAUnpackTable[nIndex] >> 2;
					if (delta & 2) v += IMAUnpackTable[nIndex] >> 1;
					if (delta & 4) v += IMAUnpackTable[nIndex];
					if (delta & 8) value -= v; else value += v;
					nIndex += IMAIndexTab[delta & 7];
					Limit(nIndex, 0, 88);
					Limit(value, -32768, 32767);
					target[smpPos] = static_cast<int16>(value);
					smpPos += numChannels;
				}
				dataPos += (numChannels - 1) * 4u;
			}
		}
		samplePos += ((blockSize - (numChannels * 4u)) * 2u);
	}

	return true;
}


////////////////////////////////////////////////////////////////////////////////
// WAV Open

bool CSoundFile::ReadWAVSample(SAMPLEINDEX nSample, FileReader &file, bool mayNormalize, FileReader *wsmpChunk)
{
	WAVReader wavFile(file);

	if(!wavFile.IsValid()
		|| wavFile.GetNumChannels() == 0
		|| wavFile.GetNumChannels() > 2
		|| (wavFile.GetBitsPerSample() == 0 && wavFile.GetSampleFormat() != WAVFormatChunk::fmtMP3)
		|| (wavFile.GetBitsPerSample() > 64)
		|| (wavFile.GetSampleFormat() != WAVFormatChunk::fmtPCM && wavFile.GetSampleFormat() != WAVFormatChunk::fmtFloat && wavFile.GetSampleFormat() != WAVFormatChunk::fmtIMA_ADPCM && wavFile.GetSampleFormat() != WAVFormatChunk::fmtMP3 && wavFile.GetSampleFormat() != WAVFormatChunk::fmtALaw && wavFile.GetSampleFormat() != WAVFormatChunk::fmtULaw))
	{
		return false;
	}

	DestroySampleThreadsafe(nSample);
	strcpy(m_szNames[nSample], "");
	ModSample &sample = Samples[nSample];
	sample.Initialize();
	sample.nLength = wavFile.GetSampleLength();
	sample.nC5Speed = wavFile.GetSampleRate();
	wavFile.ApplySampleSettings(sample, m_szNames[nSample]);

	FileReader sampleChunk = wavFile.GetSampleData();

	SampleIO sampleIO(
		SampleIO::_8bit,
		(wavFile.GetNumChannels() > 1) ? SampleIO::stereoInterleaved : SampleIO::mono,
		SampleIO::littleEndian,
		SampleIO::signedPCM);

	if(wavFile.GetSampleFormat() == WAVFormatChunk::fmtIMA_ADPCM && wavFile.GetNumChannels() <= 2)
	{
		// IMA ADPCM 4:1
		LimitMax(sample.nLength, MAX_SAMPLE_LENGTH);
		sample.uFlags.set(CHN_16BIT);
		sample.uFlags.set(CHN_STEREO, wavFile.GetNumChannels() == 2);
		if(!sample.AllocateSample())
		{
			return false;
		}
		IMAADPCMUnpack16(sample.pSample16, sample.nLength, sampleChunk, wavFile.GetBlockAlign(), wavFile.GetNumChannels());
		sample.PrecomputeLoops(*this, false);
	} else if(wavFile.GetSampleFormat() == WAVFormatChunk::fmtMP3)
	{
		// MP3 in WAV
		return ReadMP3Sample(nSample, sampleChunk, true) || ReadMediaFoundationSample(nSample, sampleChunk, true);
	} else if(!wavFile.IsExtensibleFormat() && wavFile.MayBeCoolEdit16_8() && wavFile.GetSampleFormat() == WAVFormatChunk::fmtPCM && wavFile.GetBitsPerSample() == 32 && wavFile.GetBlockAlign() == wavFile.GetNumChannels() * 4)
	{
		// Syntrillium Cool Edit hack to store IEEE 32bit floating point
		// Format is described as 32bit integer PCM contained in 32bit blocks and an WAVEFORMATEX extension size of 2 which contains a single 16 bit little endian value of 1.
		//  (This is parsed in WAVTools.cpp and returned via MayBeCoolEdit16_8()).
		// The data actually stored in this case is little endian 32bit floating point PCM with 2**15 full scale.
		// Cool Edit calls this format "16.8 float".
		sampleIO |= SampleIO::_32bit;
		sampleIO |= SampleIO::floatPCM15;
		sampleIO.ReadSample(sample, sampleChunk);
	} else if(!wavFile.IsExtensibleFormat() && wavFile.GetSampleFormat() == WAVFormatChunk::fmtPCM && wavFile.GetBitsPerSample() == 24 && wavFile.GetBlockAlign() == wavFile.GetNumChannels() * 4)
	{
		// Syntrillium Cool Edit hack to store IEEE 32bit floating point
		// Format is described as 24bit integer PCM contained in 32bit blocks.
		// The data actually stored in this case is little endian 32bit floating point PCM with 2**23 full scale.
		// Cool Edit calls this format "24.0 float".
		sampleIO |= SampleIO::_32bit;
		sampleIO |= SampleIO::floatPCM23;
		sampleIO.ReadSample(sample, sampleChunk);
	} else if(wavFile.GetSampleFormat() == WAVFormatChunk::fmtALaw || wavFile.GetSampleFormat() == WAVFormatChunk::fmtULaw)
	{
		// a-law / u-law
		sampleIO |= SampleIO::_16bit;
		sampleIO |= wavFile.GetSampleFormat() == WAVFormatChunk::fmtALaw ? SampleIO::aLaw : SampleIO::uLaw;
		sampleIO.ReadSample(sample, sampleChunk);
	} else
	{
		// PCM / Float
		SampleIO::Bitdepth bitDepth;
		switch((wavFile.GetBitsPerSample() - 1) / 8u)
		{
		default:
		case 0: bitDepth = SampleIO::_8bit; break;
		case 1: bitDepth = SampleIO::_16bit; break;
		case 2: bitDepth = SampleIO::_24bit; break;
		case 3: bitDepth = SampleIO::_32bit; break;
		case 7: bitDepth = SampleIO::_64bit; break;
		}

		sampleIO |= bitDepth;
		if(wavFile.GetBitsPerSample() <= 8)
			sampleIO |= SampleIO::unsignedPCM;

		if(wavFile.GetSampleFormat() == WAVFormatChunk::fmtFloat)
			sampleIO |= SampleIO::floatPCM;

		if(mayNormalize)
			sampleIO.MayNormalize();

		sampleIO.ReadSample(sample, sampleChunk);
	}

	if(wsmpChunk != nullptr)
	{
		// DLS WSMP chunk
		*wsmpChunk = wavFile.GetWsmpChunk();
	}

	sample.Convert(MOD_TYPE_IT, GetType());
	sample.PrecomputeLoops(*this, false);

	return true;
}


///////////////////////////////////////////////////////////////
// Save WAV


#ifndef MODPLUG_NO_FILESAVE
bool CSoundFile::SaveWAVSample(SAMPLEINDEX nSample, const mpt::PathString &filename) const
{
	mpt::ofstream f(filename, std::ios::binary);
	if(!f)
	{
		return false;
	}

	WAVWriter file(&f);

	if(!file.IsValid())
	{
		return false;
	}

	const ModSample &sample = Samples[nSample];
	file.WriteFormat(sample.GetSampleRate(GetType()), sample.GetElementarySampleSize() * 8, sample.GetNumChannels(), WAVFormatChunk::fmtPCM);

	// Write sample data
	file.StartChunk(RIFFChunk::iddata);
	file.Skip(SampleIO(
		sample.uFlags[CHN_16BIT] ? SampleIO::_16bit : SampleIO::_8bit,
		sample.uFlags[CHN_STEREO] ? SampleIO::stereoInterleaved : SampleIO::mono,
		SampleIO::littleEndian,
		sample.uFlags[CHN_16BIT] ? SampleIO::signedPCM : SampleIO::unsignedPCM)
		.WriteSample(f, sample));

	file.WriteLoopInformation(sample);
	file.WriteExtraInformation(sample, GetType());
	if(sample.HasCustomCuePoints())
	{
		file.WriteCueInformation(sample);
	}

	FileTags tags;
	tags.title = mpt::ToUnicode(GetCharsetInternal(), m_szNames[nSample]);
	tags.encoder = mpt::ToUnicode(mpt::CharsetUTF8, MptVersion::GetOpenMPTVersionStr());
	file.WriteMetatags(tags);

	return true;
}

#endif // MODPLUG_NO_FILESAVE


#ifndef MODPLUG_NO_FILESAVE

///////////////////////////////////////////////////////////////
// Save RAW

bool CSoundFile::SaveRAWSample(SAMPLEINDEX nSample, const mpt::PathString &filename) const
{
	mpt::ofstream f(filename, std::ios::binary);
	if(!f)
	{
		return false;
	}

	const ModSample &sample = Samples[nSample];
	SampleIO(
		sample.uFlags[CHN_16BIT] ? SampleIO::_16bit : SampleIO::_8bit,
		sample.uFlags[CHN_STEREO] ? SampleIO::stereoInterleaved : SampleIO::mono,
		SampleIO::littleEndian,
		SampleIO::signedPCM)
		.WriteSample(f, sample);

	return true;
}

#endif // MODPLUG_NO_FILESAVE

/////////////////////////////////////////////////////////////
// GUS Patches

struct GF1PatchFileHeader
{
	char     magic[8];		// "GF1PATCH"
	char     version[4];	// "100", or "110"
	char     id[10];		// "ID#000002"
	char     copyright[60];	// Copyright
	uint8le  numInstr;		// Number of instruments in patch
	uint8le  voices;		// Number of voices, usually 14
	uint8le  channels;		// Number of wav channels that can be played concurently to the patch
	uint16le numSamples;	// Total number of waveforms for all the .PAT
	uint16le volume;		// Master volume
	uint32le dataSize;
	char     reserved2[36];
};

MPT_BINARY_STRUCT(GF1PatchFileHeader, 129)


struct GF1Instrument
{
	uint16le id;			// Instrument id: 0-65535
	char     name[16];		// Name of instrument. Gravis doesn't seem to use it
	uint32le size;			// Number of bytes for the instrument with header. (To skip to next instrument)
	uint8    layers;		// Number of layers in instrument: 1-4
	char     reserved[40];
};

MPT_BINARY_STRUCT(GF1Instrument, 63)


struct GF1SampleHeader
{
	char     name[7];		// null terminated string. name of the wave.
	uint8le  fractions;		// Start loop point fraction in 4 bits + End loop point fraction in the 4 other bits.
	uint32le length;		// total size of wavesample. limited to 65535 now by the drivers, not the card.
	uint32le loopstart;		// start loop position in the wavesample
	uint32le loopend;		// end loop position in the wavesample
	uint16le freq;			// Rate at which the wavesample has been sampled
	uint32le low_freq, high_freq, root_freq;	// check note.h for the correspondance.
	int16le  finetune;		// fine tune. -512 to +512, EXCLUDING 0 cause it is a multiplier. 512 is one octave off, and 1 is a neutral value
	uint8le  balance;		// Balance: 0-15. 0=full left, 15 = full right
	uint8le  env_rate[6];	// attack rates
	uint8le  env_volume[6];	// attack volumes
	uint8le  tremolo_sweep, tremolo_rate, tremolo_depth;
	uint8le  vibrato_sweep, vibrato_rate, vibrato_depth;
	uint8le  flags;
	int16le  scale_frequency;	// Note
	uint16le scale_factor;		// 0...2048 (1024 is normal) or 0...2
	char     reserved[36];
};

MPT_BINARY_STRUCT(GF1SampleHeader, 96)

// -- GF1 Envelopes --
//
//	It can be represented like this (the envelope is totally bogus, it is
//	just to show the concept):
//
//	|
//	|           /----`               | |
//	|   /------/      `\         | | | | |
//	|  /                 \       | | | | |
//	| /                    \     | | | | |
//	|/                       \   | | | | |
//	---------------------------- | | | | | |
//	<---> attack rate 0          0 1 2 3 4 5 amplitudes
//	     <----> attack rate 1
//		     <> attack rate 2
//			 <--> attack rate 3
//			     <> attack rate 4
//				 <-----> attack rate 5
//
// -- GF1 Flags --
//
// bit 0: 8/16 bit
// bit 1: Signed/Unsigned
// bit 2: off/on looping
// bit 3: off/on bidirectionnal looping
// bit 4: off/on backward looping
// bit 5: off/on sustaining (3rd point in env.)
// bit 6: off/on envelopes
// bit 7: off/on clamped release (6th point, env)


struct GF1Layer
{
	uint8le  previous;		// If !=0 the wavesample to use is from the previous layer. The waveheader is still needed
	uint8le  id;			// Layer id: 0-3
	uint32le size;			// data size in bytes in the layer, without the header. to skip to next layer for example:
	uint8le  samples;		// number of wavesamples
	char     reserved[40];
};

MPT_BINARY_STRUCT(GF1Layer, 47)


static double PatchFreqToNote(uint32 nFreq)
{
	return std::log(nFreq / 2044.0) * (12.0 * 1.44269504088896340736);	// 1.0/std::log(2.0)
}


static int32 PatchFreqToNoteInt(uint32 nFreq)
{
	return Util::Round<int32>(PatchFreqToNote(nFreq));
}


static void PatchToSample(CSoundFile *that, SAMPLEINDEX nSample, GF1SampleHeader &sampleHeader, FileReader &file)
{
	ModSample &sample = that->GetSample(nSample);

	file.ReadStruct(sampleHeader);

	sample.Initialize();
	if(sampleHeader.flags & 4) sample.uFlags.set(CHN_LOOP);
	if(sampleHeader.flags & 8) sample.uFlags.set(CHN_PINGPONGLOOP);
	if(sampleHeader.flags & 16) sample.uFlags.set(CHN_REVERSE);
	sample.nLength = sampleHeader.length;
	sample.nLoopStart = sampleHeader.loopstart;
	sample.nLoopEnd = sampleHeader.loopend;
	sample.nC5Speed = sampleHeader.freq;
	sample.nPan = (sampleHeader.balance * 256 + 8) / 15;
	if(sample.nPan > 256) sample.nPan = 128;
	else sample.uFlags.set(CHN_PANNING);
	sample.nVibType = VIB_SINE;
	sample.nVibSweep = sampleHeader.vibrato_sweep;
	sample.nVibDepth = sampleHeader.vibrato_depth;
	sample.nVibRate = sampleHeader.vibrato_rate / 4;
	if(sampleHeader.scale_factor)
	{
		sample.Transpose((84.0 - PatchFreqToNote(sampleHeader.root_freq)) / 12.0);
	}

	SampleIO sampleIO(
		SampleIO::_8bit,
		SampleIO::mono,
		SampleIO::littleEndian,
		(sampleHeader.flags & 2) ? SampleIO::unsignedPCM : SampleIO::signedPCM);

	if(sampleHeader.flags & 1)
	{
		sampleIO |= SampleIO::_16bit;
		sample.nLength /= 2;
		sample.nLoopStart /= 2;
		sample.nLoopEnd /= 2;
	}
	sampleIO.ReadSample(sample, file);
	sample.Convert(MOD_TYPE_IT, that->GetType());
	sample.PrecomputeLoops(*that, false);

	mpt::String::Read<mpt::String::maybeNullTerminated>(that->m_szNames[nSample], sampleHeader.name);
}


bool CSoundFile::ReadPATSample(SAMPLEINDEX nSample, FileReader &file)
{
	file.Rewind();
	GF1PatchFileHeader fileHeader;
	GF1Instrument instrHeader;	// We only support one instrument
	GF1Layer layerHeader;
	if(!file.ReadStruct(fileHeader)
		|| memcmp(fileHeader.magic, "GF1PATCH", 8)
		|| (memcmp(fileHeader.version, "110\0", 4) && memcmp(fileHeader.version, "100\0", 4))
		|| memcmp(fileHeader.id, "ID#000002\0", 10)
		|| !fileHeader.numInstr || !fileHeader.numSamples
		|| !file.ReadStruct(instrHeader)
		//|| !instrHeader.layers	// DOO.PAT has 0 layers
		|| !file.ReadStruct(layerHeader)
		|| !layerHeader.samples)
	{
		return false;
	}

	DestroySampleThreadsafe(nSample);
	GF1SampleHeader sampleHeader;
	PatchToSample(this, nSample, sampleHeader, file);

	if(instrHeader.name[0] > ' ')
	{
		mpt::String::Read<mpt::String::maybeNullTerminated>(m_szNames[nSample], instrHeader.name);
	}
	return true;
}


// PAT Instrument
bool CSoundFile::ReadPATInstrument(INSTRUMENTINDEX nInstr, FileReader &file)
{
	file.Rewind();
	GF1PatchFileHeader fileHeader;
	GF1Instrument instrHeader;	// We only support one instrument
	GF1Layer layerHeader;
	if(!file.ReadStruct(fileHeader)
		|| memcmp(fileHeader.magic, "GF1PATCH", 8)
		|| (memcmp(fileHeader.version, "110\0", 4) && memcmp(fileHeader.version, "100\0", 4))
		|| memcmp(fileHeader.id, "ID#000002\0", 10)
		|| !fileHeader.numInstr || !fileHeader.numSamples
		|| !file.ReadStruct(instrHeader)
		//|| !instrHeader.layers	// DOO.PAT has 0 layers
		|| !file.ReadStruct(layerHeader)
		|| !layerHeader.samples)
	{
		return false;
	}

	ModInstrument *pIns = new (std::nothrow) ModInstrument();
	if(pIns == nullptr)
	{
		return false;
	}

	DestroyInstrument(nInstr, deleteAssociatedSamples);
	if (nInstr > m_nInstruments) m_nInstruments = nInstr;
	Instruments[nInstr] = pIns;

	mpt::String::Read<mpt::String::maybeNullTerminated>(pIns->name, instrHeader.name);
	pIns->nFadeOut = 2048;
	if(GetType() & (MOD_TYPE_IT | MOD_TYPE_MPT))
	{
		pIns->nNNA = NNA_NOTEOFF;
		pIns->nDNA = DNA_NOTEFADE;
	}

	SAMPLEINDEX nextSample = 0;
	int32 nMinSmpNote = 0xFF;
	SAMPLEINDEX nMinSmp = 0;
	for(uint8 smp = 0; smp < layerHeader.samples; smp++)
	{
		// Find a free sample
		nextSample = GetNextFreeSample(nInstr, nextSample + 1);
		if(nextSample == SAMPLEINDEX_INVALID) break;
		if(m_nSamples < nextSample) m_nSamples = nextSample;
		if(!nMinSmp) nMinSmp = nextSample;
		// Load it
		GF1SampleHeader sampleHeader;
		PatchToSample(this, nextSample, sampleHeader, file);
		int32 nMinNote = (sampleHeader.low_freq > 100) ? PatchFreqToNoteInt(sampleHeader.low_freq) : 0;
		int32 nMaxNote = (sampleHeader.high_freq > 100) ? PatchFreqToNoteInt(sampleHeader.high_freq) : NOTE_MAX;
		int32 nBaseNote = (sampleHeader.root_freq > 100) ? PatchFreqToNoteInt(sampleHeader.root_freq) : -1;
		if(!sampleHeader.scale_factor && layerHeader.samples == 1) { nMinNote = 0; nMaxNote = NOTE_MAX; }
		// Fill Note Map
		for(int32 k = 0; k < NOTE_MAX; k++)
		{
			if(k == nBaseNote || (!pIns->Keyboard[k] && k >= nMinNote && k <= nMaxNote))
			{
				if(!sampleHeader.scale_factor)
					pIns->NoteMap[k] = NOTE_MIDDLEC;

				pIns->Keyboard[k] = nextSample;
				if(k < nMinSmpNote)
				{
					nMinSmpNote = k;
					nMinSmp = nextSample;
				}
			}
		}
	}
	if(nMinSmp)
	{
		// Fill note map and missing samples
		for(uint8 k = 0; k < NOTE_MAX; k++)
		{
			if(!pIns->NoteMap[k]) pIns->NoteMap[k] = k + 1;
			if(!pIns->Keyboard[k])
			{
				pIns->Keyboard[k] = nMinSmp;
			} else
			{
				nMinSmp = pIns->Keyboard[k];
			}
		}
	}

	pIns->Sanitize(MOD_TYPE_IT);
	pIns->Convert(MOD_TYPE_IT, GetType());
	return true;
}


/////////////////////////////////////////////////////////////
// S3I Samples


bool CSoundFile::ReadS3ISample(SAMPLEINDEX nSample, FileReader &file)
{
	file.Rewind();

	S3MSampleHeader sampleHeader;
	if(!file.ReadStruct(sampleHeader)
		|| sampleHeader.sampleType != S3MSampleHeader::typePCM
		|| memcmp(sampleHeader.magic, "SCRS", 4)
		|| !file.Seek((sampleHeader.dataPointer[1] << 4) | (sampleHeader.dataPointer[2] << 12) | (sampleHeader.dataPointer[0] << 20)))
	{
		return false;
	}

	DestroySampleThreadsafe(nSample);

	ModSample &sample = Samples[nSample];
	sampleHeader.ConvertToMPT(sample);
	mpt::String::Read<mpt::String::nullTerminated>(m_szNames[nSample], sampleHeader.name);
	sampleHeader.GetSampleFormat(false).ReadSample(sample, file);
	sample.Convert(MOD_TYPE_S3M, GetType());
	sample.PrecomputeLoops(*this, false);
	return true;
}


/////////////////////////////////////////////////////////////
// XI Instruments


bool CSoundFile::ReadXIInstrument(INSTRUMENTINDEX nInstr, FileReader &file)
{
	file.Rewind();

	XIInstrumentHeader fileHeader;
	if(!file.ReadStruct(fileHeader)
		|| memcmp(fileHeader.signature, "Extended Instrument: ", 21)
		|| fileHeader.version != XIInstrumentHeader::fileVersion
		|| fileHeader.eof != 0x1A)
	{
		return false;
	}

	ModInstrument *pIns = new (std::nothrow) ModInstrument();
	if(pIns == nullptr)
	{
		return false;
	}

	DestroyInstrument(nInstr, deleteAssociatedSamples);
	if(nInstr > m_nInstruments)
	{
		m_nInstruments = nInstr;
	}
	Instruments[nInstr] = pIns;

	fileHeader.ConvertToMPT(*pIns);

	// Translate sample map and find available sample slots
	std::vector<SAMPLEINDEX> sampleMap(fileHeader.numSamples);
	SAMPLEINDEX maxSmp = 0;

	for(size_t i = 0 + 12; i < 96 + 12; i++)
	{
		if(pIns->Keyboard[i] >= fileHeader.numSamples)
		{
			continue;
		}

		if(sampleMap[pIns->Keyboard[i]] == 0)
		{
			// Find slot for this sample
			maxSmp = GetNextFreeSample(nInstr, maxSmp + 1);
			if(maxSmp != SAMPLEINDEX_INVALID)
			{
				sampleMap[pIns->Keyboard[i]] = maxSmp;
			}
		}
		pIns->Keyboard[i] = sampleMap[pIns->Keyboard[i]];
	}

	if(m_nSamples < maxSmp)
	{
		m_nSamples = maxSmp;
	}

	std::vector<SampleIO> sampleFlags(fileHeader.numSamples);

	// Read sample headers
	for(SAMPLEINDEX i = 0; i < fileHeader.numSamples; i++)
	{
		XMSample sampleHeader;
		if(!file.ReadStruct(sampleHeader)
			|| !sampleMap[i])
		{
			continue;
		}

		ModSample &mptSample = Samples[sampleMap[i]];
		sampleHeader.ConvertToMPT(mptSample);
		fileHeader.instrument.ApplyAutoVibratoToMPT(mptSample);
		mptSample.Convert(MOD_TYPE_XM, GetType());
		if(GetType() != MOD_TYPE_XM && fileHeader.numSamples == 1)
		{
			// No need to pan that single sample, thank you...
			mptSample.uFlags &= ~CHN_PANNING;
		}

		mpt::String::Read<mpt::String::spacePadded>(mptSample.filename, sampleHeader.name);
		mpt::String::Read<mpt::String::spacePadded>(m_szNames[sampleMap[i]], sampleHeader.name);

		sampleFlags[i] = sampleHeader.GetSampleFormat();
	}

	// Read sample data
	for(SAMPLEINDEX i = 0; i < fileHeader.numSamples; i++)
	{
		if(sampleMap[i])
		{
			sampleFlags[i].ReadSample(Samples[sampleMap[i]], file);
			Samples[sampleMap[i]].PrecomputeLoops(*this, false);
		}
	}

	pIns->Convert(MOD_TYPE_XM, GetType());

	// Read MPT crap
	ReadExtendedInstrumentProperties(pIns, file);
	pIns->Sanitize(GetType());
	return true;
}


#ifndef MODPLUG_NO_FILESAVE

bool CSoundFile::SaveXIInstrument(INSTRUMENTINDEX nInstr, const mpt::PathString &filename) const
{
	ModInstrument *pIns = Instruments[nInstr];
	if(pIns == nullptr || filename.empty())
	{
		return false;
	}

	FILE *f;
	if((f = mpt_fopen(filename, "wb")) == nullptr)
	{
		return false;
	}

	// Create file header
	XIInstrumentHeader header;
	header.ConvertToXM(*pIns, false);

	const std::vector<SAMPLEINDEX> samples = header.instrument.GetSampleList(*pIns, false);
	if(samples.size() > 0 && samples[0] <= GetNumSamples())
	{
		// Copy over auto-vibrato settings of first sample
		header.instrument.ApplyAutoVibratoToXM(Samples[samples[0]], GetType());
	}

	fwrite(&header, 1, sizeof(XIInstrumentHeader), f);

	std::vector<SampleIO> sampleFlags(samples.size());

	// XI Sample Headers
	for(SAMPLEINDEX i = 0; i < samples.size(); i++)
	{
		XMSample xmSample;
		if(samples[i] <= GetNumSamples())
		{
			xmSample.ConvertToXM(Samples[samples[i]], GetType(), false);
		} else
		{
			MemsetZero(xmSample);
		}
		sampleFlags[i] = xmSample.GetSampleFormat();

		mpt::String::Write<mpt::String::spacePadded>(xmSample.name, m_szNames[samples[i]]);

		fwrite(&xmSample, 1, sizeof(xmSample), f);
	}

	// XI Sample Data
	for(SAMPLEINDEX i = 0; i < samples.size(); i++)
	{
		if(samples[i] <= GetNumSamples())
		{
			sampleFlags[i].WriteSample(f, Samples[samples[i]]);
		}
	}

	// Write 'MPTX' extension tag
	char code[4];
	memcpy(code, "XTPM", 4);
	fwrite(code, 1, 4, f);
	WriteInstrumentHeaderStructOrField(pIns, f);	// Write full extended header.

	fclose(f);
	return true;
}

#endif // MODPLUG_NO_FILESAVE


// Read first sample from XI file into a sample slot
bool CSoundFile::ReadXISample(SAMPLEINDEX nSample, FileReader &file)
{
	file.Rewind();

	XIInstrumentHeader fileHeader;
	if(!file.ReadStruct(fileHeader)
		|| !file.CanRead(sizeof(XMSample))
		|| memcmp(fileHeader.signature, "Extended Instrument: ", 21)
		|| fileHeader.version != XIInstrumentHeader::fileVersion
		|| fileHeader.eof != 0x1A
		|| fileHeader.numSamples == 0)
	{
		return false;
	}

	if(m_nSamples < nSample)
	{
		m_nSamples = nSample;
	}

	uint16 numSamples = fileHeader.numSamples;
	FileReader::off_t samplePos = sizeof(XIInstrumentHeader) + numSamples * sizeof(XMSample);
	// Preferrably read the middle-C sample
	auto sample = fileHeader.instrument.sampleMap[48];
	if(sample >= fileHeader.numSamples)
		sample = 0;
	XMSample sampleHeader;
	while(sample--)
	{
		file.ReadStruct(sampleHeader);
		samplePos += sampleHeader.length;
	}
	file.ReadStruct(sampleHeader);
	// Gotta skip 'em all!
	file.Seek(samplePos);

	DestroySampleThreadsafe(nSample);

	ModSample &mptSample = Samples[nSample];
	sampleHeader.ConvertToMPT(mptSample);
	if(GetType() != MOD_TYPE_XM)
	{
		// No need to pan that single sample, thank you...
		mptSample.uFlags.reset(CHN_PANNING);
	}
	fileHeader.instrument.ApplyAutoVibratoToMPT(mptSample);
	mptSample.Convert(MOD_TYPE_XM, GetType());

	mpt::String::Read<mpt::String::spacePadded>(mptSample.filename, sampleHeader.name);
	mpt::String::Read<mpt::String::spacePadded>(m_szNames[nSample], sampleHeader.name);

	// Read sample data
	sampleHeader.GetSampleFormat().ReadSample(Samples[nSample], file);
	Samples[nSample].PrecomputeLoops(*this, false);

	return true;
}


/////////////////////////////////////////////////////////////////////////////////////////
// SFZ Instrument

#ifdef MPT_EXTERNAL_SAMPLES

struct SFZControl
{
	std::string defaultPath;
	int8 octaveOffset = 0, noteOffset = 0;

	void Parse(const std::string &key, const std::string &value)
	{
		if(key == "default_path")
			defaultPath = value;
		else if(key == "octave_offset")
			octaveOffset = ConvertStrTo<int8>(value);
		else if(key == "note_offset")
			noteOffset = ConvertStrTo<int8>(value);
	}
};

struct SFZEnvelope
{
	float startLevel = 0, delay = 0, attack = 0, hold = 0,
		decay = 0, sustainLevel = 100, release = 0, depth = 0;

	void Parse(std::string key, const std::string &value)
	{
		key.erase(0, key.find('_') + 1);
		float v = ConvertStrTo<float>(value);
		if(key == "depth")
			Limit(v, -12000.0f, 12000.0f);
		else if(key == "start" || key == "sustain")
			Limit(v, -100.0f, 100.0f);
		else
			Limit(v, 0.0f, 100.0f);

		if(key == "start")
			startLevel = v;
		else if(key == "delay")
			delay = v;
		else if(key == "attack")
			attack = v;
		else if(key == "hold")
			hold = v;
		else if(key == "decay")
			decay = v;
		else if(key == "sustain")
			sustainLevel = v;
		else if(key == "release")
			release = v;
		else if(key == "depth")
			depth = v;
	}

	static EnvelopeNode::tick_t ToTicks(float duration, float tickDuration)
	{
		return std::max(EnvelopeNode::tick_t(1), Util::Round<EnvelopeNode::tick_t>(duration / tickDuration));
	}

	EnvelopeNode::value_t ToValue(float value, EnvelopeType envType) const
	{
		value *= (ENVELOPE_MAX / 100.0f);
		if(envType == ENV_PITCH)
		{
			value *= depth / 3200.0f;
			value += ENVELOPE_MID;
		}
		Limit<float, float>(value, ENVELOPE_MIN, ENVELOPE_MAX);
		return Util::Round<EnvelopeNode::value_t>(value);
	}

	void ConvertToMPT(ModInstrument *ins, const CSoundFile &sndFile, EnvelopeType envType) const
	{
		auto &env = ins->GetEnvelope(envType);
		float tickDuration = sndFile.m_PlayState.m_nSamplesPerTick / static_cast<float>(sndFile.GetSampleRate());
		if(tickDuration <= 0)
			return;
		env.clear();
		if(envType != ENV_VOLUME && attack == 0 && delay == 0 && hold == 0 && decay == 0 && sustainLevel == 100 && release == 0 && depth == 0)
		{
			env.dwFlags.reset(ENV_SUSTAIN | ENV_ENABLED);
			return;
		}
		if(attack > 0 || delay > 0)
		{
			env.push_back(0, ToValue(startLevel, envType));
			if(delay > 0)
				env.push_back(ToTicks(delay, tickDuration), env.back().value);
			env.push_back(env.back().tick + ToTicks(attack, tickDuration), ToValue(100, envType));
		}
		if(hold > 0)
		{
			if(env.empty())
				env.push_back(0, ToValue(100, envType));
			env.push_back(env.back().tick + ToTicks(hold, tickDuration), env.back().value);
		}
		if(env.empty())
			env.push_back(0, ToValue(100, envType));
		auto sustain = ToValue(sustainLevel, envType);
		if(env.back().value != sustain)
			env.push_back(env.back().tick + ToTicks(decay, tickDuration), sustain);
		env.nSustainStart = env.nSustainEnd = static_cast<uint8>(env.size() - 1);
		if(sustainLevel != 0)
		{
			env.push_back(env.back().tick + ToTicks(release, tickDuration), ToValue(0, envType));
			env.dwFlags.set(ENV_SUSTAIN);
		}
		env.dwFlags.set(ENV_ENABLED);
	}
};

struct SFZRegion
{
	enum class LoopMode
	{
		kUnspecified,
		kContinuous,
		kOneShot,
		kSustain,
		kNoLoop
	};

	enum class LoopType
	{
		kUnspecified,
		kForward,
		kBackward,
		kAlternate,
	};

	std::string filename;
	SFZEnvelope ampEnv, pitchEnv, filterEnv;
	SmpLength loopStart = 0, loopEnd = 0;
	SmpLength end = MAX_SAMPLE_LENGTH, offset = 0;
	double loopCrossfade = 0.0;
	LoopMode loopMode = LoopMode::kUnspecified;
	LoopType loopType = LoopType::kUnspecified;
	int32 cutoff = 0;			// in Hz
	int32 filterRandom = 0;		// 0...9600 cents
	int16 volume = 0;			// -144dB...+6dB
	int16 pitchBend = 200;		// -9600...9600 cents
	float pitchLfoFade = 0;		// 0...100 seconds
	int16 pitchLfoDepth = 0;	// -1200...12000
	uint8 pitchLfoFreq = 0;		// 0...20 Hz
	int8 panning = -128;		// -100...+100
	int8 transpose = 0;
	int8 finetune = 0;
	uint8 keyLo = 0, keyHi = 127, keyRoot = 60;
	uint8 resonance = 0;		// 0...40dB
	uint8 filterType = FLTMODE_UNCHANGED;
	uint8 polyphony = 255;
	bool useSampleKeyRoot = false;
	bool invertPhase = false;

	template<typename T, typename Tc>
	static void Read(const std::string &valueStr, T &value, Tc valueMin = std::numeric_limits<T>::min(), Tc valueMax = std::numeric_limits<T>::max())
	{
		double valueF = ConvertStrTo<double>(valueStr);
		MPT_CONSTANT_IF(std::numeric_limits<T>::is_integer)
		{
			valueF = Util::Round(valueF);
		}
		Limit(valueF, static_cast<double>(valueMin), static_cast<double>(valueMax));
		value = static_cast<T>(valueF);
	}

	static uint8 ReadKey(const std::string &value, const SFZControl &control)
	{
		if(value.empty())
			return 0;

		int key = 0;
		if(value[0] >= '0' && value[0] <= '9')
		{
			// MIDI key
			key = ConvertStrTo<uint8>(value);
		} else if(value.length() < 2)
		{
			return 0;
		} else
		{
			// Scientific pitch
			static const int8 keys[] = { 9, 11, 0, 2, 4, 5, 7 };
			STATIC_ASSERT(CountOf(keys) == 'g' - 'a' + 1);
			auto keyC = value[0];
			if(keyC >= 'A' && keyC <= 'G')
				key = keys[keyC - 'A'];
			if(keyC >= 'a' && keyC <= 'g')
				key = keys[keyC - 'a'];
			else
				return 0;

			uint8 octaveOffset = 1;
			if(value[1] == '#')
			{
				key++;
				octaveOffset = 2;
			} else if(value[1] == 'b' || value[1] == 'B')
			{
				key--;
				octaveOffset = 2;
			}
			if(octaveOffset >= value.length())
				return 0;

			int8 octave = ConvertStrTo<int8>(value.c_str() + octaveOffset);
			key += (octave + 1) * 12;
		}
		key += control.octaveOffset * 12 + control.noteOffset;
		return static_cast<uint8>(Clamp(key, 0, 127));
}

	void Parse(const std::string &key, const std::string &value, const SFZControl &control)
	{
		if(key == "sample")
			filename = control.defaultPath + value;
		else if(key == "lokey")
			keyLo = ReadKey(value, control);
		else if(key == "hikey")
			keyHi = ReadKey(value, control);
		else if(key == "pitch_keycenter")
		{
			keyRoot = ReadKey(value, control);
			useSampleKeyRoot = (value == "sample");
		}
		else if(key == "key")
		{
			keyLo = keyHi = keyRoot = ReadKey(value, control);
			useSampleKeyRoot = false;
		}
		else if(key == "bend_up" || key == "bendup")
			Read(value, pitchBend, -9600, 9600);
		else if(key == "pitchlfo_fade")
			Read(value, pitchLfoFade, 0.0f, 100.0f);
		else if(key == "pitchlfo_depth")
			Read(value, pitchLfoDepth, -12000, 12000);
		else if(key == "pitchlfo_freq")
			Read(value, pitchLfoFreq, 0, 20);
		else if(key == "volume")
			Read(value, volume, -144, 6);
		else if(key == "pan")
			Read(value, panning, -100, 100);
		else if(key == "transpose")
			Read(value, transpose, -127, 127);
		else if(key == "tune")
			Read(value, finetune, -100, 100);
		else if(key == "end")
			Read(value, end, SmpLength(0), MAX_SAMPLE_LENGTH);
		else if(key == "offset")
			Read(value, offset, SmpLength(0), MAX_SAMPLE_LENGTH);
		else if(key == "loop_start" || key == "loopstart")
			Read(value, loopStart, SmpLength(0), MAX_SAMPLE_LENGTH);
		else if(key == "loop_end" || key == "loopend")
			Read(value, loopEnd, SmpLength(0), MAX_SAMPLE_LENGTH);
		else if(key == "loop_crossfade")
			Read(value, loopCrossfade, 0.0, DBL_MAX);
		else if(key == "loop_mode" || key == "loopmode")
		{
			if(value == "loop_continuous")
				loopMode = LoopMode::kContinuous;
			else if(value == "one_shot")
				loopMode = LoopMode::kOneShot;
			else if(value == "loop_sustain")
				loopMode = LoopMode::kSustain;
			else if(value == "no_loop")
				loopMode = LoopMode::kNoLoop;
		}
		else if(key == "loop_type" || key == "looptype")
		{
			if(value == "forward")
				loopType = LoopType::kForward;
			else if(value == "backward")
				loopType = LoopType::kBackward;
			else if(value == "alternate")
				loopType = LoopType::kAlternate;
		}
		else if(key == "cutoff")
			Read(value, cutoff, 0, 96000);
		else if(key == "fil_random")
			Read(value, filterRandom, 0, 9600);
		else if(key == "resonance")
			Read(value, resonance, 0u, 40u);
		else if(key == "polyphony")
			Read(value, polyphony, 0u, 255u);
		else if(key == "phase")
			invertPhase = (value == "invert");
		else if(key == "fil_type" || key == "filtype")
		{
			if(value == "lpf_1p" || value == "lpf_2p" || value == "lpf_4p" || value == "lpf_6p")
				filterType = FLTMODE_LOWPASS;
			else if(value == "hpf_1p" || value == "hpf_2p" || value == "hpf_4p" || value == "hpf_6p")
				filterType = FLTMODE_HIGHPASS;
			// Alternatives: bpf_2p, brf_2p
		}
		else if(key.substr(0, 6) == "ampeg_")
			ampEnv.Parse(key, value);
		else if(key.substr(0, 6) == "fileg_")
			filterEnv.Parse(key, value);
		else if(key.substr(0, 8) == "pitcheg_")
			pitchEnv.Parse(key, value);
	}
};

bool CSoundFile::ReadSFZInstrument(INSTRUMENTINDEX nInstr, FileReader &file)
{
	file.Rewind();

	enum { kNone, kGlobal, kMaster, kGroup, kRegion, kControl, kUnknown } section = kNone;
	SFZControl control;
	SFZRegion group, master, globals;
	std::vector<SFZRegion> regions;
	std::map<std::string, std::string> macros;

	std::string s;
	while(file.ReadLine(s, 1024))
	{
		// First, terminate line at the start of a comment block
		auto commentPos = s.find("//");
		if(commentPos != std::string::npos)
		{
			s.resize(commentPos);
		}

		// Now, read the tokens.
		// This format is so funky that no general tokenizer approach seems to work here...
		// Consider this jolly good example found at https://stackoverflow.com/questions/5923895/tokenizing-a-custom-text-file-format-file-using-c-sharp
		// <region>sample=piano C3.wav key=48 ampeg_release=0.7 // a comment here
		// <region>key = 49 sample = piano Db3.wav
		// <region>
		// group=1
		// key = 48
		//     sample = piano D3.ogg
		// The original sfz specification claims that spaces around = are not allowed, but a quick look into the real world tells us otherwise.

		while(!s.empty())
		{
			s.erase(0, s.find_first_not_of(" \t"));

			// Replace macros
			for(const auto &m : macros)
			{
				auto &oldStr = m.first;
				auto &newStr = m.second;
				std::string::size_type pos = 0;
				while((pos = s.find(oldStr, pos)) != std::string::npos)
				{
					s.replace(pos, oldStr.length(), newStr);
					pos += newStr.length();
				}
			}

			if(s.empty())
			{
				break;
			}

			std::string::size_type charsRead = 0;

			if(s[0] == '<' && (charsRead = s.find('>')) != std::string::npos)
			{
				// Section header
				std::string sec = s.substr(1, charsRead - 1);
				section = kUnknown;
				if(sec == "global")
				{
					section = kGlobal;
					// Reset global parameters
					globals = SFZRegion();
				} else if(sec == "master")
				{
					section = kMaster;
					// Reset master parameters
					master = globals;
				} else if(sec == "group")
				{
					section = kGroup;
					// Reset group parameters
					group = master;
				} else if(sec == "region")
				{
					section = kRegion;
					regions.push_back(group);
				} else if(sec == "control")
				{
					section = kControl;
				}
				charsRead++;
			} else if(s.substr(0, 8) == "#define " || s.substr(0, 8) == "#define\t")
			{
				// Macro definition
				auto keyStart = s.find_first_not_of(" \t", 8);
				auto keyEnd = s.find_first_of(" \t", keyStart);
				auto valueStart = s.find_first_not_of(" \t", keyEnd);
				std::string key = s.substr(keyStart, keyEnd - keyStart);
				if(valueStart != std::string::npos && key.length() > 1 && key[0] == '$')
				{
					charsRead = s.find_first_of(" \t", valueStart);
					macros[key] = s.substr(valueStart, charsRead - valueStart);
				}
			} else if(s.substr(0, 9) == "#include " || s.substr(0, 9) == "#include\t")
			{
				AddToLog(LogWarning, MPT_USTRING("#include directive is not supported."));
				auto fileStart = s.find("\"", 9);	// Yes, there can be arbitrary characters before the opening quote, at least that's how sforzando does it.
				auto fileEnd = s.find("\"", fileStart + 1);
				if(fileStart != std::string::npos && fileEnd != std::string::npos)
				{
					charsRead = fileEnd + 1;
				} else
				{
					return false;
				}
			} else if(section == kNone)
			{
				// Garbage before any section, probably not an sfz file
				return false;
			} else if(s.find('=') != std::string::npos)
			{
				// Read key=value pair
				auto keyEnd = s.find_first_of(" \t=");
				auto valueStart = s.find_first_not_of(" \t=", keyEnd);
				std::string key = mpt::ToLowerCaseAscii(s.substr(0, keyEnd));
				if(key == "sample" || key == "default_path" || key.substr(0, 8) == "label_cc")
				{
					// Sample / CC name may contain spaces...
					charsRead = s.find_first_of("=\t<", valueStart);
					if(charsRead != std::string::npos && s[charsRead] == '=')
					{
						// Backtrack to end of key
						while(charsRead > valueStart && s[charsRead] == ' ')
							charsRead--;
						// Backtrack to start of key
						while(charsRead > valueStart && s[charsRead] != ' ')
							charsRead--;
					}
				} else
				{
					charsRead = s.find_first_of(" \t<", valueStart);
				}
				std::string value = s.substr(valueStart, charsRead - valueStart);

				switch(section)
				{
				case kGlobal:
					globals.Parse(key, value, control);
					MPT_FALLTHROUGH;
				case kMaster:
					master.Parse(key, value, control);
					MPT_FALLTHROUGH;
				case kGroup:
					group.Parse(key, value, control);
					break;
				case kRegion:
					regions.back().Parse(key, value, control);
					break;
				case kControl:
					control.Parse(key, value);
					break;
				}
			} else
			{
				// Garbage, probably not an sfz file
				MPT_ASSERT(false);
				return false;
			}

			// Remove the token(s) we just read
			s.erase(0, charsRead);
		}
	}

	if(regions.empty())
	{
		return false;
	}


	ModInstrument *pIns = new (std::nothrow) ModInstrument();
	if(pIns == nullptr)
	{
		return false;
	}

	RecalculateSamplesPerTick();
	DestroyInstrument(nInstr, deleteAssociatedSamples);
	if(nInstr > m_nInstruments) m_nInstruments = nInstr;
	Instruments[nInstr] = pIns;

	SAMPLEINDEX prevSmp = 0;
	for(auto &region : regions)
	{
		uint8 keyLo = region.keyLo, keyHi = region.keyHi;
		if(keyLo > keyHi)
			continue;
		Clamp<uint8, uint8>(keyLo, 0, NOTE_MAX - NOTE_MIN);
		Clamp<uint8, uint8>(keyHi, 0, NOTE_MAX - NOTE_MIN);
		SAMPLEINDEX smp = GetNextFreeSample(nInstr, prevSmp + 1);
		if(smp == SAMPLEINDEX_INVALID)
			break;
		prevSmp = smp;

		ModSample &sample = Samples[smp];
		mpt::PathString filename = mpt::PathString::FromUTF8(region.filename);
		if(!filename.empty())
		{
			if(region.filename.find(':') == std::string::npos)
			{
				filename = file.GetFileName().GetPath() + filename;
			}
			SetSamplePath(smp, filename);
			InputFile f(filename);
			FileReader smpFile = GetFileReader(f);
			if(!ReadSampleFromFile(smp, smpFile, false))
			{
				AddToLog(LogWarning, MPT_USTRING("Unable to load sample: ") + filename.ToUnicode());
				prevSmp--;
				continue;
			}
			if(!m_szNames[smp][0])
			{
				mpt::String::Copy(m_szNames[smp], filename.GetFileName().ToLocale());
			}
		}
		sample.uFlags.set(SMP_KEEPONDISK, sample.pSample != nullptr);

		if(region.useSampleKeyRoot)
		{
			if(sample.rootNote != NOTE_NONE)
				region.keyRoot = sample.rootNote - NOTE_MIN;
			else
				region.keyRoot = 60;
		}

		const auto origSampleRate = sample.GetSampleRate(GetType());
		int8 transp = region.transpose + (60 - region.keyRoot);
		for(uint8 i = keyLo; i <= keyHi; i++)
		{
			pIns->Keyboard[i] = smp;
			if(GetType() != MOD_TYPE_XM)
				pIns->NoteMap[i] = NOTE_MIN + i + transp;
		}
		if(GetType() == MOD_TYPE_XM)
			sample.Transpose(transp / 12.0);

		pIns->nFilterMode = region.filterType;
		if(region.cutoff != 0)
			pIns->SetCutoff(FrequencyToCutOff(region.cutoff), true);
		if(region.resonance != 0)
			pIns->SetResonance(mpt::saturate_cast<uint8>(Util::muldivr(region.resonance, 128, 24)), true);
		pIns->nCutSwing = mpt::saturate_cast<uint8>(Util::muldivr(region.filterRandom, m_SongFlags[SONG_EXFILTERRANGE] ? 20 : 24, 1200));
		pIns->midiPWD = static_cast<int8>(region.pitchBend / 100);

		pIns->nNNA = NNA_NOTEOFF;
		if(region.polyphony == 1)
		{
			pIns->nDNA = NNA_NOTECUT;
			pIns->nDCT = DCT_SAMPLE;
		}
		region.ampEnv.ConvertToMPT(pIns, *this, ENV_VOLUME);
		region.pitchEnv.ConvertToMPT(pIns, *this, ENV_PITCH);
		//region.filterEnv.ConvertToMPT(pIns, *this, ENV_PITCH);

		sample.rootNote = region.keyRoot + NOTE_MIN;
		sample.nGlobalVol = Util::Round<decltype(sample.nGlobalVol)>(64 * std::pow(10.0, region.volume / 20.0));
		if(region.panning != -128)
		{
			sample.nPan = static_cast<decltype(sample.nPan)>(Util::muldivr_unsigned(region.panning + 100, 256, 200));
			sample.uFlags.set(CHN_PANNING);
		}
		sample.Transpose(region.finetune / 1200.0);

		if(region.pitchLfoDepth && region.pitchLfoFreq)
		{
			sample.nVibSweep = 255;
			if(region.pitchLfoFade > 0)
				sample.nVibSweep = Util::Round<uint8>(255 / region.pitchLfoFade);
			sample.nVibDepth = static_cast<uint8>(Util::muldivr(region.pitchLfoDepth, 32, 100));
			sample.nVibRate = region.pitchLfoFreq * 4;
		}

		if(region.loopMode != SFZRegion::LoopMode::kUnspecified)
		{
			switch(region.loopMode)
			{
			case SFZRegion::LoopMode::kContinuous:
			case SFZRegion::LoopMode::kOneShot:
				sample.uFlags.set(CHN_LOOP);
				break;
			case SFZRegion::LoopMode::kSustain:
				sample.uFlags.set(CHN_SUSTAINLOOP);
				break;
			case SFZRegion::LoopMode::kNoLoop:
				sample.uFlags.reset(CHN_LOOP | CHN_SUSTAINLOOP);
			}
		}
		if(region.loopEnd > region.loopStart)
		{
			// Loop may also be defined in file, in which case loopStart and loopEnd are unset.
			if(region.loopMode == SFZRegion::LoopMode::kSustain)
			{
				sample.nSustainStart = region.loopStart;
				sample.nSustainEnd = region.loopEnd + 1;
			} else if(region.loopMode == SFZRegion::LoopMode::kContinuous || region.loopMode == SFZRegion::LoopMode::kOneShot)
			{
				sample.nLoopStart = region.loopStart;
				sample.nLoopEnd = region.loopEnd + 1;
			}
		} else if(sample.nLoopEnd <= sample.nLoopStart && region.loopMode != SFZRegion::LoopMode::kUnspecified && region.loopMode != SFZRegion::LoopMode::kNoLoop)
		{
			sample.nLoopEnd = sample.nLength;
		}
		switch(region.loopType)
		{
		case SFZRegion::LoopType::kUnspecified:
			break;
		case SFZRegion::LoopType::kForward:
			sample.uFlags.reset(CHN_PINGPONGLOOP | CHN_PINGPONGSUSTAIN | CHN_REVERSE);
			break;
		case SFZRegion::LoopType::kBackward:
			sample.uFlags.set(CHN_REVERSE);
			break;
		case SFZRegion::LoopType::kAlternate:
			sample.uFlags.set(CHN_PINGPONGLOOP | CHN_PINGPONGSUSTAIN);
			break;
		default:
			break;
		}
		if(sample.nSustainEnd <= sample.nSustainStart && sample.nLoopEnd > sample.nLoopStart && region.loopMode == SFZRegion::LoopMode::kSustain)
		{
			// Turn normal loop (imported from sample) into sustain loop
			std::swap(sample.nSustainStart, sample.nLoopStart);
			std::swap(sample.nSustainEnd, sample.nLoopEnd);
			sample.uFlags.set(CHN_SUSTAINLOOP);
			sample.uFlags.set(CHN_PINGPONGSUSTAIN, sample.uFlags[CHN_PINGPONGLOOP]);
			sample.uFlags.reset(CHN_LOOP | CHN_PINGPONGLOOP);
		}

		// Loop cross-fade
		SmpLength fadeSamples = Util::Round<SmpLength>(region.loopCrossfade * origSampleRate);
		LimitMax(fadeSamples, sample.uFlags[CHN_SUSTAINLOOP] ? sample.nSustainStart : sample.nLoopStart);
		if(fadeSamples > 0)
		{
			ctrlSmp::XFadeSample(sample, fadeSamples, 50000, true, sample.uFlags[CHN_SUSTAINLOOP], *this);
			sample.uFlags.set(SMP_MODIFIED);
		}

		// Sample offset
		if(region.offset && region.offset < sample.nLength)
		{
			auto offset = region.offset * sample.GetBytesPerSample();
			memmove(sample.pSample8, sample.pSample8 + offset, sample.nLength * sample.GetBytesPerSample() - offset);
			if(region.end > region.offset)
				region.end -= region.offset;
			sample.nLength -= region.offset;
			sample.nLoopStart -= region.offset;
			sample.nLoopEnd -= region.offset;
			sample.uFlags.set(SMP_MODIFIED);
		}
		LimitMax(sample.nLength, region.end);

		if(region.invertPhase)
		{
			ctrlSmp::InvertSample(sample, 0, sample.nLength, *this);
			sample.uFlags.set(SMP_MODIFIED);
		}

		sample.PrecomputeLoops(*this, false);
		sample.Convert(MOD_TYPE_MPT, GetType());
	}

	pIns->Sanitize(MOD_TYPE_MPT);
	pIns->Convert(MOD_TYPE_MPT, GetType());
	return true;
}
#else
bool CSoundFile::ReadSFZInstrument(INSTRUMENTINDEX, FileReader &)
{
	return false;
}
#endif // MPT_EXTERNAL_SAMPLES


/////////////////////////////////////////////////////////////////////////////////////////
// AIFF File I/O

// AIFF header
struct AIFFHeader
{
	char     magic[4];	// FORM
	uint32be length;	// Size of the file, not including magic and length
	char     type[4];	// AIFF or AIFC
};

MPT_BINARY_STRUCT(AIFFHeader, 12)


// General IFF Chunk header
struct AIFFChunk
{
	// 32-Bit chunk identifiers
	enum ChunkIdentifiers
	{
		idCOMM	= MAGIC4BE('C','O','M','M'),
		idSSND	= MAGIC4BE('S','S','N','D'),
		idINST	= MAGIC4BE('I','N','S','T'),
		idMARK	= MAGIC4BE('M','A','R','K'),
		idNAME	= MAGIC4BE('N','A','M','E'),
	};

	uint32be id;		// See ChunkIdentifiers
	uint32be length;	// Chunk size without header

	size_t GetLength() const
	{
		return length;
	}

	ChunkIdentifiers GetID() const
	{
		return static_cast<ChunkIdentifiers>(id.get());
	}
};

MPT_BINARY_STRUCT(AIFFChunk, 8)


// "Common" chunk (in AIFC, a compression ID and compression name follows this header, but apart from that it's identical)
struct AIFFCommonChunk
{
	uint16be numChannels;
	uint32be numSampleFrames;
	uint16be sampleSize;
	uint8be  sampleRate[10];		// Sample rate in 80-Bit floating point

	// Convert sample rate to integer
	uint32 GetSampleRate() const
	{
		uint32 mantissa = (sampleRate[2] << 24) | (sampleRate[3] << 16) | (sampleRate[4] << 8) | (sampleRate[5] << 0);
		uint32 last = 0;
		uint8 exp = 30 - sampleRate[1];

		while(exp--)
		{
			last = mantissa;
			mantissa >>= 1;
		}
		if(last & 1) mantissa++;
		return mantissa;
	}
};

MPT_BINARY_STRUCT(AIFFCommonChunk, 18)


// Sound chunk
struct AIFFSoundChunk
{
	uint32be offset;
	uint32be blockSize;
};

MPT_BINARY_STRUCT(AIFFSoundChunk, 8)


// Marker
struct AIFFMarker
{
	uint16be id;
	uint32be position;		// Position in sample
	uint8be  nameLength;	// Not counting eventually existing padding byte in name string
};

MPT_BINARY_STRUCT(AIFFMarker, 7)


// Instrument loop
struct AIFFInstrumentLoop
{
	enum PlayModes
	{
		noLoop		= 0,
		loopNormal	= 1,
		loopBidi	= 2,
	};

	uint16be playMode;
	uint16be beginLoop;	// Marker index
	uint16be endLoop;	// Marker index
};

MPT_BINARY_STRUCT(AIFFInstrumentLoop, 6)


struct AIFFInstrumentChunk
{
	uint8be  baseNote;
	uint8be  detune;
	uint8be  lowNote;
	uint8be  highNote;
	uint8be  lowVelocity;
	uint8be  highVelocity;
	uint16be gain;
	AIFFInstrumentLoop sustainLoop;
	AIFFInstrumentLoop releaseLoop;
};

MPT_BINARY_STRUCT(AIFFInstrumentChunk, 20)


bool CSoundFile::ReadAIFFSample(SAMPLEINDEX nSample, FileReader &file, bool mayNormalize)
{
	file.Rewind();
	ChunkReader chunkFile(file);

	// Verify header
	AIFFHeader fileHeader;
	if(!chunkFile.ReadStruct(fileHeader)
		|| memcmp(fileHeader.magic, "FORM", 4)
		|| (memcmp(fileHeader.type, "AIFF", 4) && memcmp(fileHeader.type, "AIFC", 4)))
	{
		return false;
	}

	auto chunks = chunkFile.ReadChunks<AIFFChunk>(2);

	// Read COMM chunk
	FileReader commChunk(chunks.GetChunk(AIFFChunk::idCOMM));
	AIFFCommonChunk sampleInfo;
	if(!commChunk.ReadStruct(sampleInfo))
	{
		return false;
	}

	// Is this a proper sample?
	if(sampleInfo.numSampleFrames == 0
		|| sampleInfo.numChannels < 1 || sampleInfo.numChannels > 2
		|| sampleInfo.sampleSize < 1 || sampleInfo.sampleSize > 64)
	{
		return false;
	}

	// Read compression type in AIFF-C files.
	uint8 compression[4] = { 'N', 'O', 'N', 'E' };
	SampleIO::Endianness endian = SampleIO::bigEndian;
	if(!memcmp(fileHeader.type, "AIFC", 4))
	{
		if(!commChunk.ReadArray(compression))
		{
			return false;
		}
		if(!memcmp(compression, "twos", 4))
		{
			endian = SampleIO::littleEndian;
		}
	}

	// Read SSND chunk
	FileReader soundChunk(chunks.GetChunk(AIFFChunk::idSSND));
	AIFFSoundChunk sampleHeader;
	if(!soundChunk.ReadStruct(sampleHeader)
		|| !soundChunk.CanRead(sampleHeader.offset))
	{
		return false;
	}

	SampleIO::Bitdepth bitDepth;
	switch((sampleInfo.sampleSize - 1) / 8)
	{
	default:
	case 0: bitDepth = SampleIO::_8bit; break;
	case 1: bitDepth = SampleIO::_16bit; break;
	case 2: bitDepth = SampleIO::_24bit; break;
	case 3: bitDepth = SampleIO::_32bit; break;
	case 7: bitDepth = SampleIO::_64bit; break;
	}

	SampleIO sampleIO(bitDepth,
		(sampleInfo.numChannels == 2) ? SampleIO::stereoInterleaved : SampleIO::mono,
		endian,
		SampleIO::signedPCM);

	if(!memcmp(compression, "fl32", 4) || !memcmp(compression, "FL32", 4) || !memcmp(compression, "fl64", 4))
	{
		sampleIO |= SampleIO::floatPCM;
	} else if(!memcmp(compression, "alaw", 4) || !memcmp(compression, "ALAW", 4))
	{
		sampleIO |= SampleIO::aLaw;
		sampleIO |= SampleIO::_16bit;
	} else if(!memcmp(compression, "ulaw", 4) || !memcmp(compression, "ULAW", 4))
	{
		sampleIO |= SampleIO::uLaw;
		sampleIO |= SampleIO::_16bit;
	}

	if(mayNormalize)
	{
		sampleIO.MayNormalize();
	}

	soundChunk.Skip(sampleHeader.offset);

	ModSample &mptSample = Samples[nSample];
	DestroySampleThreadsafe(nSample);
	mptSample.Initialize();
	mptSample.nLength = sampleInfo.numSampleFrames;
	mptSample.nC5Speed = sampleInfo.GetSampleRate();

	sampleIO.ReadSample(mptSample, soundChunk);

	// Read MARK and INST chunk to extract sample loops
	FileReader markerChunk(chunks.GetChunk(AIFFChunk::idMARK));
	AIFFInstrumentChunk instrHeader;
	if(markerChunk.IsValid() && chunks.GetChunk(AIFFChunk::idINST).ReadStruct(instrHeader))
	{
		uint16 numMarkers = markerChunk.ReadUint16BE();

		std::vector<AIFFMarker> markers;
		markers.reserve(numMarkers);
		for(size_t i = 0; i < numMarkers; i++)
		{
			AIFFMarker marker;
			if(!markerChunk.ReadStruct(marker))
			{
				break;
			}
			markers.push_back(marker);
			markerChunk.Skip(marker.nameLength + ((marker.nameLength % 2u) == 0 ? 1 : 0));
		}

		if(instrHeader.sustainLoop.playMode != AIFFInstrumentLoop::noLoop)
		{
			mptSample.uFlags.set(CHN_SUSTAINLOOP);
			mptSample.uFlags.set(CHN_PINGPONGSUSTAIN, instrHeader.sustainLoop.playMode == AIFFInstrumentLoop::loopBidi);
		}

		if(instrHeader.releaseLoop.playMode != AIFFInstrumentLoop::noLoop)
		{
			mptSample.uFlags.set(CHN_LOOP);
			mptSample.uFlags.set(CHN_PINGPONGLOOP, instrHeader.releaseLoop.playMode == AIFFInstrumentLoop::loopBidi);
		}

		// Read markers
		for(const auto &m : markers)
		{
			if(m.id == instrHeader.sustainLoop.beginLoop)
				mptSample.nSustainStart = m.position;
			if(m.id == instrHeader.sustainLoop.endLoop)
				mptSample.nSustainEnd = m.position;
			if(m.id == instrHeader.releaseLoop.beginLoop)
				mptSample.nLoopStart = m.position;
			if(m.id == instrHeader.releaseLoop.endLoop)
				mptSample.nLoopEnd = m.position;
		}
		mptSample.SanitizeLoops();
	}

	// Extract sample name
	FileReader nameChunk(chunks.GetChunk(AIFFChunk::idNAME));
	if(nameChunk.IsValid())
	{
		nameChunk.ReadString<mpt::String::spacePadded>(m_szNames[nSample], nameChunk.GetLength());
	} else
	{
		strcpy(m_szNames[nSample], "");
	}

	mptSample.Convert(MOD_TYPE_IT, GetType());
	mptSample.PrecomputeLoops(*this, false);
	return true;
}


bool CSoundFile::ReadAUSample(SAMPLEINDEX nSample, FileReader &file, bool mayNormalize)
{
	file.Rewind();

	// Verify header
	if(!file.ReadMagic(".snd"))
		return false;

	uint32 dataOffset = file.ReadUint32BE();
	uint32 dataSize = file.ReadUint32BE();
	uint32 encoding = file.ReadUint32BE();
	uint32 sampleRate = file.ReadUint32BE();
	uint32 channels = file.ReadUint32BE();

	if(channels < 1 || channels > 2)
		return false;

	SampleIO sampleIO(SampleIO::_8bit, channels == 1 ? SampleIO::mono : SampleIO::stereoInterleaved, SampleIO::bigEndian, SampleIO::signedPCM);
	switch(encoding)
	{
	case 1: sampleIO |= SampleIO::_16bit;			// u-law
		sampleIO |= SampleIO::uLaw; break;
	case 2: break;									// 8-bit linear PCM
	case 3: sampleIO |= SampleIO::_16bit; break;	// 16-bit linear PCM
	case 4: sampleIO |= SampleIO::_24bit; break;	// 24-bit linear PCM
	case 5: sampleIO |= SampleIO::_32bit; break;	// 32-bit linear PCM
	case 6: sampleIO |= SampleIO::_32bit;			// 32-bit IEEE floating point
		sampleIO |= SampleIO::floatPCM;
		break;
	case 7: sampleIO |= SampleIO::_64bit;			// 64-bit IEEE floating point
		sampleIO |= SampleIO::floatPCM;
		break;
	case 27: sampleIO |= SampleIO::_16bit;			// a-law
		sampleIO |= SampleIO::aLaw; break;
	default: return false;
	}

	if(!file.Seek(dataOffset))
		return false;

	ModSample &mptSample = Samples[nSample];
	DestroySampleThreadsafe(nSample);
	mptSample.Initialize();
	SmpLength length = mpt::saturate_cast<SmpLength>(file.BytesLeft());
	if(dataSize != 0xFFFFFFFF)
		LimitMax(length, dataSize);
	mptSample.nLength = (length * 8u) / (sampleIO.GetEncodedBitsPerSample() * channels);
	mptSample.nC5Speed = sampleRate;
	strcpy(m_szNames[nSample], "");

	if(mayNormalize)
	{
		sampleIO.MayNormalize();
	}

	sampleIO.ReadSample(mptSample, file);

	mptSample.Convert(MOD_TYPE_IT, GetType());
	mptSample.PrecomputeLoops(*this, false);
	return true;
}


/////////////////////////////////////////////////////////////////////////////////////////
// ITS Samples


bool CSoundFile::ReadITSSample(SAMPLEINDEX nSample, FileReader &file, bool rewind)
{
	if(rewind)
	{
		file.Rewind();
	}

	ITSample sampleHeader;
	if(!file.ReadStruct(sampleHeader)
		|| memcmp(sampleHeader.id, "IMPS", 4))
	{
		return false;
	}
	DestroySampleThreadsafe(nSample);

	ModSample &sample = Samples[nSample];
	file.Seek(sampleHeader.ConvertToMPT(sample));
	mpt::String::Read<mpt::String::spacePaddedNull>(m_szNames[nSample], sampleHeader.name);

	if(!sample.uFlags[SMP_KEEPONDISK])
	{
		sampleHeader.GetSampleFormat().ReadSample(Samples[nSample], file);
	} else
	{
		// External sample
		size_t strLen;
		file.ReadVarInt(strLen);
#ifdef MPT_EXTERNAL_SAMPLES
		std::string filenameU8;
		file.ReadString<mpt::String::maybeNullTerminated>(filenameU8, strLen);
		mpt::PathString filename = mpt::PathString::FromUTF8(filenameU8);

		if(!filename.empty())
		{
			if(!file.GetFileName().empty())
			{
				filename = filename.RelativePathToAbsolute(file.GetFileName().GetPath());
			}
			if(!LoadExternalSample(nSample, filename))
			{
				AddToLog(LogWarning, MPT_USTRING("Unable to load sample: ") + filename.ToUnicode());
			}
		} else
		{
			sample.uFlags.reset(SMP_KEEPONDISK);
		}
#else
		file.Skip(strLen);
#endif // MPT_EXTERNAL_SAMPLES
	}

	sample.Convert(MOD_TYPE_IT, GetType());
	sample.PrecomputeLoops(*this, false);
	return true;
}


bool CSoundFile::ReadITISample(SAMPLEINDEX nSample, FileReader &file)
{
	ITInstrument instrumentHeader;

	file.Rewind();
	if(!file.ReadStruct(instrumentHeader)
		|| memcmp(instrumentHeader.id, "IMPI", 4))
	{
		return false;
	}
	file.Rewind();
	ModInstrument dummy;
	ITInstrToMPT(file, dummy, instrumentHeader.trkvers);
	// Old SchismTracker versions set nos=0
	const SAMPLEINDEX nsamples = std::max(static_cast<SAMPLEINDEX>(instrumentHeader.nos), *std::max_element(std::begin(dummy.Keyboard), std::end(dummy.Keyboard)));
	if(!nsamples)
		return false;

	// Preferrably read the middle-C sample
	auto sample = dummy.Keyboard[NOTE_MIDDLEC - NOTE_MIN];
	if(sample > 0)
		sample--;
	else
		sample = 0;
	file.Seek(file.GetPosition() + sample * sizeof(ITSample));
	return ReadITSSample(nSample, file, false);
}


bool CSoundFile::ReadITIInstrument(INSTRUMENTINDEX nInstr, FileReader &file)
{
	ITInstrument instrumentHeader;
	SAMPLEINDEX smp = 0;

	file.Rewind();
	if(!file.ReadStruct(instrumentHeader)
		|| memcmp(instrumentHeader.id, "IMPI", 4))
	{
		return false;
	}
	if(nInstr > GetNumInstruments()) m_nInstruments = nInstr;

	ModInstrument *pIns = new (std::nothrow) ModInstrument();
	if(pIns == nullptr)
	{
		return false;
	}

	DestroyInstrument(nInstr, deleteAssociatedSamples);

	Instruments[nInstr] = pIns;
	file.Rewind();
	ITInstrToMPT(file, *pIns, instrumentHeader.trkvers);
	// Old SchismTracker versions set nos=0
	const SAMPLEINDEX nsamples = std::max(static_cast<SAMPLEINDEX>(instrumentHeader.nos), *std::max_element(std::begin(pIns->Keyboard), std::end(pIns->Keyboard)));

	// In order to properly compute the position, in file, of eventual extended settings
	// such as "attack" we need to keep the "real" size of the last sample as those extra
	// setting will follow this sample in the file
	FileReader::off_t extraOffset = file.GetPosition();

	// Reading Samples
	std::vector<SAMPLEINDEX> samplemap(nsamples, 0);
	for(SAMPLEINDEX i = 0; i < nsamples; i++)
	{
		smp = GetNextFreeSample(nInstr, smp + 1);
		if(smp == SAMPLEINDEX_INVALID) break;
		samplemap[i] = smp;
		const FileReader::off_t offset = file.GetPosition();
		if(!ReadITSSample(smp, file, false))
			smp--;
		extraOffset = std::max(extraOffset, file.GetPosition());
		file.Seek(offset + sizeof(ITSample));
	}
	if(GetNumSamples() < smp) m_nSamples = smp;

	// Adjust sample assignment
	for(auto &sample : pIns->Keyboard)
	{
		if(sample > 0 && sample <= nsamples)
		{
			sample = samplemap[sample - 1];
		}
	}

	pIns->Convert(MOD_TYPE_IT, GetType());

	if(file.Seek(extraOffset))
	{
		// Read MPT crap
		ReadExtendedInstrumentProperties(pIns, file);
	}
	pIns->Sanitize(GetType());

	return true;
}


#ifndef MODPLUG_NO_FILESAVE

bool CSoundFile::SaveITIInstrument(INSTRUMENTINDEX nInstr, const mpt::PathString &filename, bool compress, bool allowExternal) const
{
	ITInstrument iti;
	ModInstrument *pIns = Instruments[nInstr];
	FILE *f;

	if((!pIns) || filename.empty()) return false;
	if((f = mpt_fopen(filename, "wb")) == nullptr) return false;

	auto instSize = iti.ConvertToIT(*pIns, false, *this);

	// Create sample assignment table
	std::vector<SAMPLEINDEX> smptable;
	std::vector<uint8> smpmap(GetNumSamples(), 0);
	for(size_t i = 0; i < NOTE_MAX; i++)
	{
		const SAMPLEINDEX smp = pIns->Keyboard[i];
		if(smp && smp <= GetNumSamples())
		{
			if(!smpmap[smp - 1])
			{
				// We haven't considered this sample yet.
				smptable.push_back(smp);
				smpmap[smp - 1] = static_cast<uint8>(smptable.size());
			}
			iti.keyboard[i * 2 + 1] = smpmap[smp - 1];
		} else
		{
			iti.keyboard[i * 2 + 1] = 0;
		}
	}
	iti.nos = static_cast<uint8>(smptable.size());
	smpmap.clear();

	uint32 filePos = instSize;
	mpt::IO::WritePartial(f, iti, instSize);

	filePos += mpt::saturate_cast<uint32>(smptable.size() * sizeof(ITSample));

	// Writing sample headers + data
	std::vector<SampleIO> sampleFlags;
	for(auto smp : smptable)
	{
		ITSample itss;
		itss.ConvertToIT(Samples[smp], GetType(), compress, compress, allowExternal);
		const bool isExternal = itss.cvt == ITSample::cvtExternalSample;

		mpt::String::Write<mpt::String::nullTerminated>(itss.name, m_szNames[smp]);

		itss.samplepointer = filePos;
		mpt::IO::Write(f, itss);

		// Write sample
		auto curPos = mpt::IO::TellWrite(f);
		mpt::IO::SeekAbsolute(f, filePos);
		if(!isExternal)
		{
			filePos += mpt::saturate_cast<uint32>(itss.GetSampleFormat(0x0214).WriteSample(f, Samples[smp]));
		} else
		{
#ifdef MPT_EXTERNAL_SAMPLES
			const std::string filenameU8 = GetSamplePath(smp).AbsolutePathToRelative(filename.GetPath()).ToUTF8();
			const size_t strSize = mpt::saturate_cast<uint16>(filenameU8.size());
			size_t intBytes = 0;
			if(mpt::IO::WriteVarInt(f, strSize, &intBytes))
			{
				filePos += intBytes + strSize;
				mpt::IO::WriteRaw(f, filenameU8.data(), strSize);
			}
#endif // MPT_EXTERNAL_SAMPLES
		}
		mpt::IO::SeekAbsolute(f, curPos);
	}

	mpt::IO::SeekEnd(f);
	// Write 'MPTX' extension tag
	mpt::IO::WriteRaw(f, "XTPM", 4);
	WriteInstrumentHeaderStructOrField(pIns, f);	// Write full extended header.

	fclose(f);
	return true;
}

#endif // MODPLUG_NO_FILESAVE


///////////////////////////////////////////////////////////////////////////////////////////////////
// 8SVX / 16SVX Samples

// IFF File Header
struct IFFHeader
{
	char     form[4];	// "FORM"
	uint32be size;
	char     magic[4];	// "8SVX" or "16SV"
};

MPT_BINARY_STRUCT(IFFHeader, 12)


// General IFF Chunk header
struct IFFChunk
{
	// 32-Bit chunk identifiers
	enum ChunkIdentifiers
	{
		idVHDR	= MAGIC4BE('V','H','D','R'),
		idBODY	= MAGIC4BE('B','O','D','Y'),
		idNAME	= MAGIC4BE('N','A','M','E'),
	};

	uint32be id;		// See ChunkIdentifiers
	uint32be length;	// Chunk size without header

	size_t GetLength() const
	{
		if(length == 0)	// Broken files
			return std::numeric_limits<size_t>::max();
		return length;
	}

	ChunkIdentifiers GetID() const
	{
		return static_cast<ChunkIdentifiers>(id.get());
	}
};

MPT_BINARY_STRUCT(IFFChunk, 8)


struct IFFSampleHeader
{
	uint32be oneShotHiSamples;	// Samples in the high octave 1-shot part
	uint32be repeatHiSamples;	// Samples in the high octave repeat part
	uint32be samplesPerHiCycle;	// Samples/cycle in high octave, else 0
	uint16be samplesPerSec;		// Data sampling rate
	uint8be  octave;			// Octaves of waveforms
	uint8be  compression;		// Data compression technique used
	uint32be volume;
};

MPT_BINARY_STRUCT(IFFSampleHeader, 20)


bool CSoundFile::ReadIFFSample(SAMPLEINDEX nSample, FileReader &file)
{
	file.Rewind();

	IFFHeader fileHeader;
	if(!file.ReadStruct(fileHeader)
		|| memcmp(fileHeader.form, "FORM", 4 )
		|| (memcmp(fileHeader.magic, "8SVX", 4) && memcmp(fileHeader.magic, "16SV", 4)))
	{
		return false;
	}

	ChunkReader chunkFile(file);
	ChunkReader::ChunkList<IFFChunk> chunks = chunkFile.ReadChunks<IFFChunk>(2);

	FileReader vhdrChunk = chunks.GetChunk(IFFChunk::idVHDR);
	FileReader bodyChunk = chunks.GetChunk(IFFChunk::idBODY);
	IFFSampleHeader sampleHeader;
	if(!bodyChunk.IsValid()
		|| !vhdrChunk.IsValid()
		|| !vhdrChunk.ReadStruct(sampleHeader))
	{
		return false;
	}

	DestroySampleThreadsafe(nSample);
	// Default values
	const uint8 bytesPerSample = memcmp(fileHeader.magic, "8SVX", 4) ? 2 : 1;
	ModSample &sample = Samples[nSample];
	sample.Initialize();
	sample.nLoopStart = sampleHeader.oneShotHiSamples / bytesPerSample;
	sample.nLoopEnd = sample.nLoopStart + sampleHeader.repeatHiSamples / bytesPerSample;
	sample.nC5Speed = sampleHeader.samplesPerSec;
	sample.nVolume = static_cast<uint16>(sampleHeader.volume >> 8);
	if(!sample.nVolume || sample.nVolume > 256) sample.nVolume = 256;
	if(!sample.nC5Speed) sample.nC5Speed = 22050;

	sample.Convert(MOD_TYPE_IT, GetType());

	FileReader nameChunk = chunks.GetChunk(IFFChunk::idNAME);
	if(nameChunk.IsValid())
	{
		nameChunk.ReadString<mpt::String::maybeNullTerminated>(m_szNames[nSample], nameChunk.GetLength());
	} else
	{
		strcpy(m_szNames[nSample], "");
	}

	sample.nLength = mpt::saturate_cast<SmpLength>(bodyChunk.GetLength() / bytesPerSample);
	if((sample.nLoopStart + 4 < sample.nLoopEnd) && (sample.nLoopEnd <= sample.nLength)) sample.uFlags.set(CHN_LOOP);

	// While this is an Amiga format, the 16SV version appears to be only used on PC, and only with little-endian sample data.
	SampleIO(
		(bytesPerSample == 2) ? SampleIO::_16bit : SampleIO::_8bit,
		SampleIO::mono,
		SampleIO::littleEndian,
		SampleIO::signedPCM)
		.ReadSample(sample, bodyChunk);
	sample.PrecomputeLoops(*this, false);

	return true;
}


OPENMPT_NAMESPACE_END
