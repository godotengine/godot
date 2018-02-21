/*
 * Load_wav.cpp
 * ------------
 * Purpose: WAV importer
 * Notes  : This loader converts each WAV channel into a separate mono sample.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "WAVTools.h"
#include "../soundbase/SampleFormatConverters.h"
#include "../soundbase/SampleFormatCopy.h"


OPENMPT_NAMESPACE_BEGIN


/////////////////////////////////////////////////////////////
// WAV file support


template <typename SampleConversion>
bool CopyWavChannel(ModSample &sample, const FileReader &file, size_t channelIndex, size_t numChannels, SampleConversion conv = SampleConversion())
{
	MPT_ASSERT(sample.GetNumChannels() == 1);
	MPT_ASSERT(sample.GetElementarySampleSize() == sizeof(typename SampleConversion::output_t));

	const size_t offset = channelIndex * sizeof(typename SampleConversion::input_t) * SampleConversion::input_inc;

	if(sample.AllocateSample() == 0 || !file.CanRead(offset))
	{
		return false;
	}

	const mpt::byte *inBuf = file.GetRawData<mpt::byte>();
	CopySample<SampleConversion>(reinterpret_cast<typename SampleConversion::output_t*>(sample.pSample), sample.nLength, 1, inBuf + offset, file.BytesLeft() - offset, numChannels, conv);
	return true;
}


bool CSoundFile::ReadWav(FileReader &file, ModLoadingFlags loadFlags)
{
	WAVReader wavFile(file);

	if(!wavFile.IsValid()
		|| wavFile.GetNumChannels() == 0
		|| wavFile.GetNumChannels() > MAX_BASECHANNELS
		|| wavFile.GetBitsPerSample() == 0
		|| wavFile.GetBitsPerSample() > 32
		|| (wavFile.GetSampleFormat() != WAVFormatChunk::fmtPCM && wavFile.GetSampleFormat() != WAVFormatChunk::fmtFloat))
	{
		return false;
	} else if(loadFlags == onlyVerifyHeader)
	{
		return true;
	}

	InitializeGlobals(MOD_TYPE_MPT);
	m_ContainerType = MOD_CONTAINERTYPE_WAV;
	m_nChannels = std::max(wavFile.GetNumChannels(), uint16(2));
	Patterns.ResizeArray(2);
	if(!Patterns.Insert(0, 64) || !Patterns.Insert(1, 64))
	{
		return false;
	}
	
	const SmpLength sampleLength = wavFile.GetSampleLength();

	// Setting up module length
	// Calculate sample length in ticks at tempo 125
	const uint32 sampleRate = std::max(uint32(1), wavFile.GetSampleRate());
	const uint32 sampleTicks = mpt::saturate_cast<uint32>(((sampleLength * 50) / sampleRate) + 1);
	uint32 ticksPerRow = std::max((sampleTicks + 63u) / 63u, 1u);

	Order().assign(1, 0);
	ORDERINDEX numOrders = 1;
	while(ticksPerRow >= 32 && numOrders < MAX_ORDERS)
	{
		numOrders++;
		ticksPerRow = (sampleTicks + (64 * numOrders - 1)) / (64 * numOrders);
	}
	Order().resize(numOrders, 1);

	m_nSamples = wavFile.GetNumChannels();
	m_nInstruments = 0;
	m_nDefaultSpeed = ticksPerRow;
	m_nDefaultTempo.Set(125);
	m_SongFlags = SONG_LINEARSLIDES;

	for(CHANNELINDEX channel = 0; channel < m_nChannels; channel++)
	{
		ChnSettings[channel].Reset();
		ChnSettings[channel].nPan = (channel % 2u) ? 256 : 0;
	}

	// Setting up pattern
	PatternRow pattern = Patterns[0].GetRow(0);
	pattern[0].note = pattern[1].note = NOTE_MIDDLEC;
	pattern[0].instr = pattern[1].instr = 1;

	const FileReader sampleChunk = wavFile.GetSampleData();

	// Read every channel into its own sample lot.
	for(SAMPLEINDEX channel = 0; channel < GetNumSamples(); channel++)
	{
		pattern[channel].note = pattern[0].note;
		pattern[channel].instr = static_cast<ModCommand::INSTR>(channel + 1);

		ModSample &sample = Samples[channel + 1];
		sample.Initialize();
		sample.uFlags = CHN_PANNING;
		sample.nLength =  sampleLength;
		sample.nC5Speed = wavFile.GetSampleRate();
		strcpy(m_szNames[channel + 1], "");
		wavFile.ApplySampleSettings(sample, m_szNames[channel + 1]);

		if(wavFile.GetNumChannels() > 1)
		{
			// Pan all samples appropriately
			switch(channel)
			{
			case 0:
				sample.nPan = 0;
				break;
			case 1:
				sample.nPan = 256;
				break;
			case 2:
				sample.nPan = (wavFile.GetNumChannels() == 3 ? 128u : 64u);
				pattern[channel].command = CMD_S3MCMDEX;
				pattern[channel].param = 0x91;
				break;
			case 3:
				sample.nPan = 192;
				pattern[channel].command = CMD_S3MCMDEX;
				pattern[channel].param = 0x91;
				break;
			default:
				sample.nPan = 128;
				break;
			}
		}

		if(wavFile.GetBitsPerSample() > 8)
		{
			sample.uFlags.set(CHN_16BIT);
		}

		if(wavFile.GetSampleFormat() == WAVFormatChunk::fmtFloat)
		{
			CopyWavChannel<SC::ConversionChain<SC::Convert<int16, float32>, SC::DecodeFloat32<littleEndian32> > >(sample, sampleChunk, channel, wavFile.GetNumChannels());
		} else
		{
			if(wavFile.GetBitsPerSample() <= 8)
			{
				CopyWavChannel<SC::DecodeUint8>(sample, sampleChunk, channel, wavFile.GetNumChannels());
			} else if(wavFile.GetBitsPerSample() <= 16)
			{
				CopyWavChannel<SC::DecodeInt16<0, littleEndian16> >(sample, sampleChunk, channel, wavFile.GetNumChannels());
			} else if(wavFile.GetBitsPerSample() <= 24)
			{
				CopyWavChannel<SC::ConversionChain<SC::Convert<int16, int32>, SC::DecodeInt24<0, littleEndian24> > >(sample, sampleChunk, channel, wavFile.GetNumChannels());
			} else if(wavFile.GetBitsPerSample() <= 32)
			{
				CopyWavChannel<SC::ConversionChain<SC::Convert<int16, int32>, SC::DecodeInt32<0, littleEndian32> > >(sample, sampleChunk, channel, wavFile.GetNumChannels());
			}
		}
		sample.PrecomputeLoops(*this, false);

	}

	return true;
}


OPENMPT_NAMESPACE_END
