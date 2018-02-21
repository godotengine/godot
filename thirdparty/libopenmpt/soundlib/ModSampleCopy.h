/*
 * ModSampleCopy.h
 * ---------------
 * Purpose: Functions for copying ModSample data.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


#include "../soundbase/SampleFormatCopy.h"


OPENMPT_NAMESPACE_BEGIN


struct ModSample;

// Copy a mono sample data buffer.
template <typename SampleConversion, typename Tbyte>
size_t CopyMonoSample(ModSample &sample, const Tbyte *sourceBuffer, size_t sourceSize, SampleConversion conv = SampleConversion())
{
	MPT_ASSERT(sample.GetNumChannels() == 1);
	MPT_ASSERT(sample.GetElementarySampleSize() == sizeof(typename SampleConversion::output_t));

	const size_t frameSize =  SampleConversion::input_inc;
	const size_t countFrames = std::min<size_t>(sourceSize / frameSize, sample.nLength);
	size_t numFrames = countFrames;
	SampleConversion sampleConv(conv);
	const mpt::byte * MPT_RESTRICT inBuf = mpt::byte_cast<const mpt::byte*>(sourceBuffer);
	typename SampleConversion::output_t * MPT_RESTRICT outBuf = static_cast<typename SampleConversion::output_t *>(sample.pSample);
	while(numFrames--)
	{
		*outBuf = sampleConv(inBuf);
		inBuf += SampleConversion::input_inc;
		outBuf++;
	}
	return frameSize * countFrames;
}


// Copy a stereo interleaved sample data buffer.
template <typename SampleConversion, typename Tbyte>
size_t CopyStereoInterleavedSample(ModSample &sample, const Tbyte *sourceBuffer, size_t sourceSize, SampleConversion conv = SampleConversion())
{
	MPT_ASSERT(sample.GetNumChannels() == 2);
	MPT_ASSERT(sample.GetElementarySampleSize() == sizeof(typename SampleConversion::output_t));

	const size_t frameSize = 2 * SampleConversion::input_inc;
	const size_t countFrames = std::min<size_t>(sourceSize / frameSize, sample.nLength);
	size_t numFrames = countFrames;
	SampleConversion sampleConvLeft(conv);
	SampleConversion sampleConvRight(conv);
	const mpt::byte * MPT_RESTRICT inBuf = mpt::byte_cast<const mpt::byte*>(sourceBuffer);
	typename SampleConversion::output_t * MPT_RESTRICT outBuf = static_cast<typename SampleConversion::output_t *>(sample.pSample);
	while(numFrames--)
	{
		*outBuf = sampleConvLeft(inBuf);
		inBuf += SampleConversion::input_inc;
		outBuf++;
		*outBuf = sampleConvRight(inBuf);
		inBuf += SampleConversion::input_inc;
		outBuf++;
	}
	return frameSize * countFrames;
}


// Copy a stereo split sample data buffer.
template <typename SampleConversion, typename Tbyte>
size_t CopyStereoSplitSample(ModSample &sample, const Tbyte *sourceBuffer, size_t sourceSize, SampleConversion conv = SampleConversion())
{
	MPT_ASSERT(sample.GetNumChannels() == 2);
	MPT_ASSERT(sample.GetElementarySampleSize() == sizeof(typename SampleConversion::output_t));

	const size_t sampleSize = SampleConversion::input_inc;
	const size_t sourceSizeLeft = std::min<size_t>(sample.nLength * SampleConversion::input_inc, sourceSize);
	const size_t sourceSizeRight = std::min<size_t>(sample.nLength * SampleConversion::input_inc, sourceSize - sourceSizeLeft);
	const size_t countSamplesLeft = sourceSizeLeft / sampleSize;
	const size_t countSamplesRight = sourceSizeRight / sampleSize;

	size_t numSamplesLeft = countSamplesLeft;
	SampleConversion sampleConvLeft(conv);
	const mpt::byte * MPT_RESTRICT inBufLeft = mpt::byte_cast<const mpt::byte*>(sourceBuffer);
	typename SampleConversion::output_t * MPT_RESTRICT outBufLeft = static_cast<typename SampleConversion::output_t *>(sample.pSample);
	while(numSamplesLeft--)
	{
		*outBufLeft = sampleConvLeft(inBufLeft);
		inBufLeft += SampleConversion::input_inc;
		outBufLeft += 2;
	}

	size_t numSamplesRight = countSamplesRight;
	SampleConversion sampleConvRight(conv);
	const mpt::byte * MPT_RESTRICT inBufRight = mpt::byte_cast<const mpt::byte*>(sourceBuffer) + sample.nLength * SampleConversion::input_inc;
	typename SampleConversion::output_t * MPT_RESTRICT outBufRight = static_cast<typename SampleConversion::output_t *>(sample.pSample) + 1;
	while(numSamplesRight--)
	{
		*outBufRight = sampleConvRight(inBufRight);
		inBufRight += SampleConversion::input_inc;
		outBufRight += 2;
	}

	return (countSamplesLeft + countSamplesRight) * sampleSize;
}


// Copy a sample data buffer and normalize it. Requires slightly advanced sample conversion functor.
template <typename SampleConversion, typename Tbyte>
size_t CopyAndNormalizeSample(ModSample &sample, const Tbyte *sourceBuffer, size_t sourceSize, typename SampleConversion::peak_t *srcPeak = nullptr, SampleConversion conv = SampleConversion())
{
	const size_t inSize = sizeof(typename SampleConversion::input_t);

	MPT_ASSERT(sample.GetElementarySampleSize() == sizeof(typename SampleConversion::output_t));

	size_t numSamples = sample.nLength * sample.GetNumChannels();
	LimitMax(numSamples, sourceSize / inSize);

	const mpt::byte * inBuf = mpt::byte_cast<const mpt::byte*>(sourceBuffer);
	// Finding max value
	SampleConversion sampleConv(conv);
	for(size_t i = numSamples; i != 0; i--)
	{
		sampleConv.FindMax(inBuf);
		inBuf += SampleConversion::input_inc;
	}

	// If buffer is silent (maximum is 0), don't bother normalizing the sample - just keep the already silent buffer.
	if(!sampleConv.IsSilent())
	{
		inBuf = sourceBuffer;
		// Copying buffer.
		typename SampleConversion::output_t *outBuf = static_cast<typename SampleConversion::output_t *>(sample.pSample);

		for(size_t i = numSamples; i != 0; i--)
		{
			*outBuf = sampleConv(inBuf);
			outBuf++;
			inBuf += SampleConversion::input_inc;
		}
	}

	if(srcPeak)
	{
		*srcPeak = sampleConv.GetSrcPeak();
	}

	return numSamples * inSize;
}


OPENMPT_NAMESPACE_END
