/*
 * ModSmp_Ctrl.cpp
 * ---------------
 * Purpose: Basic sample editing code (resizing, adding silence, normalizing, ...).
 * Notes  : Could be merged with ModSample.h / ModSample.cpp at some point.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "modsmp_ctrl.h"
#include "AudioCriticalSection.h"
#include "Sndfile.h"
#include "../soundbase/SampleFormatConverters.h"
#include "../soundbase/SampleFormatCopy.h"

OPENMPT_NAMESPACE_BEGIN

namespace ctrlSmp
{

void ReplaceSample(ModSample &smp, void *pNewSample, const SmpLength nNewLength, CSoundFile &sndFile)
{
	void * const pOldSmp = smp.pSample;
	FlagSet<ChannelFlags> setFlags, resetFlags;

	setFlags.set(CHN_16BIT, smp.uFlags[CHN_16BIT]);
	resetFlags.set(CHN_16BIT, !smp.uFlags[CHN_16BIT]);

	setFlags.set(CHN_STEREO, smp.uFlags[CHN_STEREO]);
	resetFlags.set(CHN_STEREO, !smp.uFlags[CHN_STEREO]);

	CriticalSection cs;

	ctrlChn::ReplaceSample(sndFile, smp, pNewSample, nNewLength, setFlags, resetFlags);
	smp.pSample = pNewSample;
	smp.nLength = nNewLength;
	ModSample::FreeSample(pOldSmp);
}


SmpLength InsertSilence(ModSample &smp, const SmpLength nSilenceLength, const SmpLength nStartFrom, CSoundFile &sndFile)
{
	if(nSilenceLength == 0 || nSilenceLength > MAX_SAMPLE_LENGTH || smp.nLength > MAX_SAMPLE_LENGTH - nSilenceLength || nStartFrom > smp.nLength)
		return smp.nLength;

	const bool wasEmpty = smp.nLength == 0 || smp.pSample == nullptr;
	const SmpLength newLength = smp.nLength + nSilenceLength;

	char *pNewSmp = nullptr;

	pNewSmp = static_cast<char *>(ModSample::AllocateSample(newLength, smp.GetBytesPerSample()));
	if(pNewSmp == nullptr)
		return smp.nLength; //Sample allocation failed.

	if(!wasEmpty)
	{
		// Copy over old sample
		const SmpLength silenceOffset = nStartFrom * smp.GetBytesPerSample();
		const SmpLength silenceBytes = nSilenceLength * smp.GetBytesPerSample();
		if(nStartFrom > 0)
		{
			memcpy(pNewSmp, smp.pSample, silenceOffset);
		}
		if(nStartFrom < smp.nLength)
		{
			memcpy(pNewSmp + silenceOffset + silenceBytes, static_cast<const char *>(smp.pSample) + silenceOffset, smp.GetSampleSizeInBytes() - silenceOffset);
		}

		// Update loop points if necessary.
		if(smp.nLoopStart >= nStartFrom) smp.nLoopStart += nSilenceLength;
		if(smp.nLoopEnd >= nStartFrom) smp.nLoopEnd += nSilenceLength;
		if(smp.nSustainStart >= nStartFrom) smp.nSustainStart += nSilenceLength;
		if(smp.nSustainEnd >= nStartFrom) smp.nSustainEnd += nSilenceLength;
		for(std::size_t i = 0; i < CountOf(smp.cues); i++)
		{
			if(smp.cues[i] >= nStartFrom) smp.cues[i] += nSilenceLength;
		}
	} else
	{
		// Set loop points automatically
		smp.nLoopStart = 0;
		smp.nLoopEnd = newLength;
		smp.uFlags.set(CHN_LOOP);
	}

	ReplaceSample(smp, pNewSmp, newLength, sndFile);
	PrecomputeLoops(smp, sndFile, true);

	return smp.nLength;
}


namespace
{
	// Update loop points and cues after deleting a sample selection
	static void AdjustLoopPoints(SmpLength selStart, SmpLength selEnd, SmpLength &loopStart, SmpLength &loopEnd, SmpLength length)
	{
		Util::DeleteRange(selStart, selEnd - 1, loopStart, loopEnd);

		LimitMax(loopEnd, length);
		if(loopStart + 2 >= loopEnd)
		{
			loopStart = loopEnd = 0;
		}
	}
}

SmpLength RemoveRange(ModSample &smp, SmpLength selStart, SmpLength selEnd, CSoundFile &sndFile)
{
	LimitMax(selEnd, smp.nLength);
	if(selEnd <= selStart)
	{
		return smp.nLength;
	}
	const uint8 bps = smp.GetBytesPerSample();
	memmove(smp.pSample8 + selStart * bps, smp.pSample8 + selEnd * bps, (smp.nLength - selEnd) * bps);
	smp.nLength -= (selEnd - selStart);

	// Did loops or cue points cover the deleted selection?
	AdjustLoopPoints(selStart, selEnd, smp.nLoopStart, smp.nLoopEnd, smp.nLength);
	AdjustLoopPoints(selStart, selEnd, smp.nSustainStart, smp.nSustainEnd, smp.nLength);

	if(smp.nLoopEnd == 0) smp.uFlags.reset(CHN_LOOP | CHN_PINGPONGLOOP);
	if(smp.nSustainEnd == 0) smp.uFlags.reset(CHN_SUSTAINLOOP | CHN_PINGPONGSUSTAIN);

	for(std::size_t i = 0; i < CountOf(smp.cues); i++)
	{
		Util::DeleteItem(selStart, selEnd - 1, smp.cues[i]);
	}

	smp.PrecomputeLoops(sndFile);
	return smp.nLength;
}


SmpLength ResizeSample(ModSample &smp, const SmpLength nNewLength, CSoundFile &sndFile)
{
	// Invalid sample size
	if(nNewLength > MAX_SAMPLE_LENGTH || nNewLength == smp.nLength)
		return smp.nLength;

	// New sample will be bigger so we'll just use "InsertSilence" as it's already there.
	if(nNewLength > smp.nLength)
		return InsertSilence(smp, nNewLength - smp.nLength, smp.nLength, sndFile);

	// Else: Shrink sample

	const SmpLength nNewSmpBytes = nNewLength * smp.GetBytesPerSample();

	void *pNewSmp = ModSample::AllocateSample(nNewLength, smp.GetBytesPerSample());
	if(pNewSmp == nullptr)
		return smp.nLength; //Sample allocation failed.

	// Copy over old data and replace sample by the new one
	memcpy(pNewSmp, smp.pSample, nNewSmpBytes);
	ReplaceSample(smp, pNewSmp, nNewLength, sndFile);

	// Adjust loops
	if(smp.nLoopStart > nNewLength)
	{
		smp.nLoopStart = smp.nLoopEnd = 0;
		smp.uFlags.reset(CHN_LOOP);
	}
	if(smp.nLoopEnd > nNewLength) smp.nLoopEnd = nNewLength;
	if(smp.nSustainStart > nNewLength)
	{
		smp.nSustainStart = smp.nSustainEnd = 0;
		smp.uFlags.reset(CHN_SUSTAINLOOP);
	}
	if(smp.nSustainEnd > nNewLength) smp.nSustainEnd = nNewLength;

	PrecomputeLoops(smp, sndFile);

	return smp.nLength;
}

namespace // Unnamed namespace for local implementation functions.
{


template<typename T>
class PrecomputeLoop
{
protected:
	T *target;
	const T *sampleData;
	SmpLength loopEnd;
	int numChannels;
	bool pingpong;
	bool ITPingPongMode;

public:
	PrecomputeLoop(T *target, const T *sampleData, SmpLength loopEnd, int numChannels, bool pingpong, bool ITPingPongMode)
		: target(target), sampleData(sampleData), loopEnd(loopEnd), numChannels(numChannels), pingpong(pingpong), ITPingPongMode(ITPingPongMode)
	{
		if(loopEnd > 0)
		{
			CopyLoop(true);
			CopyLoop(false);
		}
	}

	void CopyLoop(bool direction) const
	{
		// Direction: true = start reading and writing forward, false = start reading and writing backward (write direction never changes)
		const int numSamples = 2 * InterpolationMaxLookahead + (direction ? 1 : 0);	// Loop point is included in forward loop expansion
		T *dest = target + numChannels * (2 * InterpolationMaxLookahead - 1);		// Write buffer offset
		SmpLength readPosition = loopEnd - 1;
		const int writeIncrement = direction ? 1 : -1;
		int readIncrement = writeIncrement;

		for(int i = 0; i < numSamples; i++)
		{
			// Copy sample over to lookahead buffer
			for(int c = 0; c < numChannels; c++)
			{
				dest[c] = sampleData[readPosition * numChannels + c];
			}
			dest += writeIncrement * numChannels;

			if(readPosition == loopEnd - 1 && readIncrement > 0)
			{
				// Reached end of loop while going forward
				if(pingpong)
				{
					readIncrement = -1;
					if(ITPingPongMode && readPosition > 0)
					{
						readPosition--;
					}
				} else
				{
					readPosition = 0;
				}
			} else if(readPosition == 0 && readIncrement < 0)
			{
				// Reached start of loop while going backward
				if(pingpong)
				{
					readIncrement = 1;
				} else
				{
					readPosition = loopEnd - 1;
				}
			} else
			{
				readPosition += readIncrement;
			}
		}
	}
};


template<typename T>
void PrecomputeLoopsImpl(ModSample &smp, const CSoundFile &sndFile)
{
	const int numChannels = smp.GetNumChannels();
	const int copySamples = numChannels * InterpolationMaxLookahead;
	
	T *sampleData = static_cast<T *>(smp.pSample);
	T *afterSampleStart = sampleData + smp.nLength * numChannels;
	T *loopLookAheadStart = afterSampleStart + copySamples;
	T *sustainLookAheadStart = loopLookAheadStart + 4 * copySamples;

	// Hold sample on the same level as the last sampling point at the end to prevent extra pops with interpolation.
	// Do the same at the sample start, too.
	for(int i = 0; i < (int)InterpolationMaxLookahead; i++)
	{
		for(int c = 0; c < numChannels; c++)
		{
			afterSampleStart[i * numChannels + c] = afterSampleStart[-numChannels + c];
			sampleData[-(i + 1) * numChannels + c] = sampleData[c];
		}
	}

	if(smp.uFlags[CHN_LOOP])
	{
		PrecomputeLoop<T>(loopLookAheadStart,
			sampleData + smp.nLoopStart * numChannels,
			smp.nLoopEnd - smp.nLoopStart,
			numChannels,
			smp.uFlags[CHN_PINGPONGLOOP],
			sndFile.m_playBehaviour[kITPingPongMode]);
	}
	if(smp.uFlags[CHN_SUSTAINLOOP])
	{
		PrecomputeLoop<T>(sustainLookAheadStart,
			sampleData + smp.nSustainStart * numChannels,
			smp.nSustainEnd - smp.nSustainStart,
			numChannels,
			smp.uFlags[CHN_PINGPONGSUSTAIN],
			sndFile.m_playBehaviour[kITPingPongMode]);
	}
}

} // unnamed namespace.


bool PrecomputeLoops(ModSample &smp, CSoundFile &sndFile, bool updateChannels)
{
	if(smp.nLength == 0 || smp.pSample == nullptr)
		return false;

	smp.SanitizeLoops();

	// Update channels with possibly changed loop values
	if(updateChannels)
	{
		UpdateLoopPoints(smp, sndFile);
	}

	if(smp.GetElementarySampleSize() == 2)
		PrecomputeLoopsImpl<int16>(smp, sndFile);
	else if(smp.GetElementarySampleSize() == 1)
		PrecomputeLoopsImpl<int8>(smp, sndFile);

	return true;
}


// Propagate loop point changes to player
bool UpdateLoopPoints(const ModSample &smp, CSoundFile &sndFile)
{
	if(!smp.HasSampleData())
		return false;

	CriticalSection cs;

	// Update channels with new loop values
	for(auto &chn : sndFile.m_PlayState.Chn) if((chn.pModSample == &smp) && chn.nLength != 0)
	{
		bool looped = false, bidi = false;

		if(smp.nSustainStart < smp.nSustainEnd && smp.nSustainEnd <= smp.nLength && smp.uFlags[CHN_SUSTAINLOOP] && !chn.dwFlags[CHN_KEYOFF])
		{
			// Sustain loop is active
			chn.nLoopStart = smp.nSustainStart;
			chn.nLoopEnd = smp.nSustainEnd;
			chn.nLength = smp.nSustainEnd;
			looped = true;
			bidi = smp.uFlags[CHN_PINGPONGSUSTAIN];
		} else if(smp.nLoopStart < smp.nLoopEnd && smp.nLoopEnd <= smp.nLength && smp.uFlags[CHN_LOOP])
		{
			// Normal loop is active
			chn.nLoopStart = smp.nLoopStart;
			chn.nLoopEnd = smp.nLoopEnd;
			chn.nLength = smp.nLoopEnd;
			looped = true;
			bidi = smp.uFlags[CHN_PINGPONGLOOP];
		}
		chn.dwFlags.set(CHN_LOOP, looped);
		chn.dwFlags.set(CHN_PINGPONGLOOP, looped && bidi);

		if(chn.position.GetUInt() > chn.nLength)
		{
			chn.position.Set(chn.nLoopStart);
			chn.dwFlags.reset(CHN_PINGPONGFLAG);
		}
		if(!bidi)
		{
			chn.dwFlags.reset(CHN_PINGPONGFLAG);
		}
		if(!looped)
		{
			chn.nLength = smp.nLength;
		}
	}

	return true;
}


void ResetSamples(CSoundFile &sndFile, ResetFlag resetflag, SAMPLEINDEX minSample, SAMPLEINDEX maxSample)
{
	if(minSample == SAMPLEINDEX_INVALID)
	{
		minSample = 1;
	}
	if(maxSample == SAMPLEINDEX_INVALID)
	{
		maxSample = sndFile.GetNumSamples();
	}
	Limit(minSample, SAMPLEINDEX(1), SAMPLEINDEX(MAX_SAMPLES - 1));
	Limit(maxSample, SAMPLEINDEX(1), SAMPLEINDEX(MAX_SAMPLES - 1));

	if(minSample > maxSample)
	{
		std::swap(minSample, maxSample);
	}

	for(SAMPLEINDEX i = minSample; i <= maxSample; i++)
	{
		ModSample &sample = sndFile.GetSample(i);
		switch(resetflag)
		{
		case SmpResetInit:
			strcpy(sndFile.m_szNames[i], "");
			strcpy(sample.filename, "");
			sample.nC5Speed = 8363;
			// note: break is left out intentionally. keep this order or c&p the stuff from below if you change anything!
			MPT_FALLTHROUGH;
		case SmpResetCompo:
			sample.nPan = 128;
			sample.nGlobalVol = 64;
			sample.nVolume = 256;
			sample.nVibDepth = 0;
			sample.nVibRate = 0;
			sample.nVibSweep = 0;
			sample.nVibType = 0;
			sample.uFlags.reset(CHN_PANNING | SMP_NODEFAULTVOLUME);
			break;
		case SmpResetVibrato:
			sample.nVibDepth = 0;
			sample.nVibRate = 0;
			sample.nVibSweep = 0;
			sample.nVibType = 0;
			break;
		default:
			break;
		}
	}
}


namespace
{
	struct OffsetData
	{
		double dMax, dMin, dOffset;
	};

	// Returns maximum sample amplitude for given sample type (int8/int16).
	template <class T>
	double GetMaxAmplitude() {return 1.0 + (std::numeric_limits<T>::max)();}

	// Calculates DC offset and returns struct with DC offset, max and min values.
	// DC offset value is average of [-1.0, 1.0[-normalized offset values.
	template<class T>
	OffsetData CalculateOffset(const T *pStart, const SmpLength nLength)
	{
		OffsetData offsetVals = {0,0,0};

		if(nLength < 1)
			return offsetVals;

		const double dMaxAmplitude = GetMaxAmplitude<T>();

		double dMax = -1, dMin = 1, dSum = 0;

		const T *p = pStart;
		for(SmpLength i = 0; i < nLength; i++, p++)
		{
			const double dVal = double(*p) / dMaxAmplitude;
			dSum += dVal;
			if(dVal > dMax) dMax = dVal;
			if(dVal < dMin) dMin = dVal;
		}

		offsetVals.dMax = dMax;
		offsetVals.dMin = dMin;
		offsetVals.dOffset = (-dSum / (double)(nLength));
		return offsetVals;
	}

	template <class T>
	void RemoveOffsetAndNormalize(T *pStart, const SmpLength nLength, const double dOffset, const double dAmplify)
	{
		T *p = pStart;
		for(SmpLength i = 0; i < nLength; i++, p++)
		{
			double dVal = (*p) * dAmplify + dOffset;
			*p = mpt::saturate_cast<T>(dVal);
		}
	}
}


// Remove DC offset
float RemoveDCOffset(ModSample &smp,
					 SmpLength iStart,
					 SmpLength iEnd,
					 const MODTYPE modtype,
					 CSoundFile &sndFile)
{
	if(!smp.HasSampleData())
		return 0;

	if (iEnd > smp.nLength) iEnd = smp.nLength;
	if (iStart > iEnd) iStart = iEnd;
	if (iStart == iEnd)
	{
		iStart = 0;
		iEnd = smp.nLength;
	}

	iStart *= smp.GetNumChannels();
	iEnd *= smp.GetNumChannels();

	const double dMaxAmplitude = (smp.GetElementarySampleSize() == 2) ? GetMaxAmplitude<int16>() : GetMaxAmplitude<int8>();

	// step 1: Calculate offset.
	OffsetData oData = {0,0,0};
	if(smp.GetElementarySampleSize() == 2)
		oData = CalculateOffset(smp.pSample16 + iStart, iEnd - iStart);
	else if(smp.GetElementarySampleSize() == 1)
		oData = CalculateOffset(smp.pSample8 + iStart, iEnd - iStart);

	double dMin = oData.dMin, dMax = oData.dMax, dOffset = oData.dOffset;

	const float fReportOffset = (float)dOffset;

	if((int)(dOffset * dMaxAmplitude) == 0)
		return 0;

	// those will be changed...
	dMax += dOffset;
	dMin += dOffset;

	// ... and that might cause distortion, so we will normalize this.
	const double dAmplify = 1 / std::max(dMax, -dMin);

	// step 2: centralize + normalize sample
	dOffset *= dMaxAmplitude * dAmplify;
	if(smp.GetElementarySampleSize() == 2)
		RemoveOffsetAndNormalize(smp.pSample16 + iStart, iEnd - iStart, dOffset, dAmplify);
	else if(smp.GetElementarySampleSize() == 1)
		RemoveOffsetAndNormalize(smp.pSample8 + iStart, iEnd - iStart, dOffset, dAmplify);

	// step 3: adjust global vol (if available)
	if((modtype & (MOD_TYPE_IT | MOD_TYPE_MPT)) && (iStart == 0) && (iEnd == smp.nLength * smp.GetNumChannels()))
	{
		CriticalSection cs;

		smp.nGlobalVol = std::min(Util::Round<uint16>(smp.nGlobalVol / dAmplify), uint16(64));
		for(auto &chn : sndFile.m_PlayState.Chn)
		{
			if(chn.pModSample == &smp)
			{
				chn.UpdateInstrumentVolume(&smp, chn.pModInstrument);
			}
		}
	}

	PrecomputeLoops(smp, sndFile, false);

	return fReportOffset;
}


template <class T>
static void ReverseSampleImpl(T *pStart, const SmpLength nLength)
{
	for(SmpLength i = 0; i < nLength / 2; i++)
	{
		std::swap(pStart[i], pStart[nLength - 1 - i]);
	}
}

// Reverse sample data
bool ReverseSample(ModSample &smp, SmpLength iStart, SmpLength iEnd, CSoundFile &sndFile)
{
	if(!smp.HasSampleData()) return false;
	if(iEnd == 0 || iStart > smp.nLength || iEnd > smp.nLength)
	{
		iStart = 0;
		iEnd = smp.nLength;
	}

	if(iEnd - iStart < 2) return false;

	STATIC_ASSERT(MaxSamplingPointSize <= 4);
	if(smp.GetBytesPerSample() == 4)	// 16 bit stereo
		ReverseSampleImpl(static_cast<int32 *>(smp.pSample) + iStart, iEnd - iStart);
	else if(smp.GetBytesPerSample() == 2)	// 16 bit mono / 8 bit stereo
		ReverseSampleImpl(smp.pSample16 + iStart, iEnd - iStart);
	else if(smp.GetBytesPerSample() == 1)	// 8 bit mono
		ReverseSampleImpl(smp.pSample8 + iStart, iEnd - iStart);
	else
		return false;

	PrecomputeLoops(smp, sndFile, false);
	return true;
}


template <class T>
static void UnsignSampleImpl(T *pStart, const SmpLength nLength)
{
	const T offset = (std::numeric_limits<T>::min)();
	for(SmpLength i = 0; i < nLength; i++)
	{
		pStart[i] += offset;
	}
}

// Virtually unsign sample data
bool UnsignSample(ModSample &smp, SmpLength iStart, SmpLength iEnd, CSoundFile &sndFile)
{
	if(!smp.HasSampleData()) return false;
	if(iEnd == 0 || iStart > smp.nLength || iEnd > smp.nLength)
	{
		iStart = 0;
		iEnd = smp.nLength;
	}
	iStart *= smp.GetNumChannels();
	iEnd *= smp.GetNumChannels();
	if(smp.GetElementarySampleSize() == 2)
		UnsignSampleImpl(smp.pSample16 + iStart, iEnd - iStart);
	else if(smp.GetElementarySampleSize() == 1)
		UnsignSampleImpl(smp.pSample8 + iStart, iEnd - iStart);
	else
		return false;

	PrecomputeLoops(smp, sndFile, false);
	return true;
}


template <class T>
static void InvertSampleImpl(T *pStart, const SmpLength nLength)
{
	for(SmpLength i = 0; i < nLength; i++)
	{
		pStart[i] = ~pStart[i];
	}
}

// Invert sample data (flip by 180 degrees)
bool InvertSample(ModSample &smp, SmpLength iStart, SmpLength iEnd, CSoundFile &sndFile)
{
	if(!smp.HasSampleData()) return false;
	if(iEnd == 0 || iStart > smp.nLength || iEnd > smp.nLength)
	{
		iStart = 0;
		iEnd = smp.nLength;
	}
	iStart *= smp.GetNumChannels();
	iEnd *= smp.GetNumChannels();
	if(smp.GetElementarySampleSize() == 2)
		InvertSampleImpl(smp.pSample16 + iStart, iEnd - iStart);
	else if(smp.GetElementarySampleSize() == 1)
		InvertSampleImpl(smp.pSample8 + iStart, iEnd - iStart);
	else
		return false;

	PrecomputeLoops(smp, sndFile, false);
	return true;
}


template <class T>
static void XFadeSampleImpl(const T *srcIn, const T *srcOut, T *output, const SmpLength fadeLength, double e)
{
	const double length = 1.0 / static_cast<double>(fadeLength);
	for(SmpLength i = 0; i < fadeLength; i++, srcIn++, srcOut++, output++)
	{
		double fact1 = std::pow(i * length, e);
		double fact2 = std::pow((fadeLength - i) * length, e);
		int32 val = static_cast<int32>(
			static_cast<double>(*srcIn) * fact1 +
			static_cast<double>(*srcOut) * fact2);
		*output = mpt::saturate_cast<T>(val);
	}
}

// X-Fade sample data to create smooth loop transitions
bool XFadeSample(ModSample &smp, SmpLength fadeLength, int fadeLaw, bool afterloopFade, bool useSustainLoop, CSoundFile &sndFile)
{
	if(!smp.HasSampleData()) return false;
	const SmpLength loopStart = useSustainLoop ? smp.nSustainStart : smp.nLoopStart;
	const SmpLength loopEnd = useSustainLoop ? smp.nSustainEnd : smp.nLoopEnd;
	
	if(loopEnd <= loopStart || loopEnd > smp.nLength) return false;
	if(loopStart < fadeLength) return false;

	const SmpLength start = (loopStart - fadeLength) * smp.GetNumChannels();
	const SmpLength end = (loopEnd - fadeLength) * smp.GetNumChannels();
	const SmpLength afterloopStart = loopStart * smp.GetNumChannels();
	const SmpLength afterloopEnd = loopEnd * smp.GetNumChannels();
	const SmpLength afterLoopLength = std::min(smp.nLength - loopEnd, fadeLength) * smp.GetNumChannels();
	fadeLength *= smp.GetNumChannels();

	// e=0.5: constant power crossfade (for uncorrelated samples), e=1.0: constant volume crossfade (for perfectly correlated samples)
	const double e = 1.0 - fadeLaw / 200000.0;

	if(smp.GetElementarySampleSize() == 2)
	{
		XFadeSampleImpl(smp.pSample16 + start, smp.pSample16 + end, smp.pSample16 + end, fadeLength, e);
		if(afterloopFade) XFadeSampleImpl(smp.pSample16 + afterloopEnd, smp.pSample16 + afterloopStart, smp.pSample16 + afterloopEnd, afterLoopLength, e);
	} else if(smp.GetElementarySampleSize() == 1)
	{
		XFadeSampleImpl(smp.pSample8 + start, smp.pSample8 + end, smp.pSample8 + end, fadeLength, e);
		if(afterloopFade) XFadeSampleImpl(smp.pSample8 + afterloopEnd, smp.pSample8 + afterloopStart, smp.pSample8 + afterloopEnd, afterLoopLength, e);
	} else
		return false;

	PrecomputeLoops(smp, sndFile, true);
	return true;
}


template <class T>
static void SilenceSampleImpl(T *p, SmpLength length, SmpLength inc, bool fromStart, bool toEnd)
{
	const int dest = toEnd ? 0 : p[(length - 1) * inc];
	const int base = fromStart ? 0 :p[0];
	const int delta = dest - base;
	const int64 len_m1 = length - 1;
	for(SmpLength i = 0; i < length; i++)
	{
		int n = base + static_cast<int>((static_cast<int64>(delta) * static_cast<int64>(i)) / len_m1);
		*p = static_cast<T>(n);
		p += inc;
	}
}

// X-Fade sample data to create smooth loop transitions
bool SilenceSample(ModSample &smp, SmpLength start, SmpLength end, CSoundFile &sndFile)
{
	LimitMax(end, smp.nLength);
	if(!smp.HasSampleData() || start >= end) return false;

	const SmpLength length = end - start;
	const bool fromStart = start == 0;
	const bool toEnd = end == smp.nLength;
	const uint8 numChn = smp.GetNumChannels();

	for(uint8 chn = 0; chn < numChn; chn++)
	{
		if(smp.GetElementarySampleSize() == 2)
			SilenceSampleImpl(smp.pSample16 + start * numChn + chn, length, numChn, fromStart, toEnd);
		else if(smp.GetElementarySampleSize() == 1)
			SilenceSampleImpl(smp.pSample8 + start * numChn + chn, length, numChn, fromStart, toEnd);
		else
			return false;
	}

	PrecomputeLoops(smp, sndFile, false);
	return true;
}


template <class T>
static void StereoSepSampleImpl(T *p, SmpLength length, int32 separation)
{
	const int32 fac1 = static_cast<int32>(32768 + separation / 2), fac2 = static_cast<int32>(32768 - separation / 2);
	while(length--)
	{
		const int32 l = p[0], r = p[1];
		p[0] = mpt::saturate_cast<T>((Util::mul32to64(l, fac1) + Util::mul32to64(r, fac2)) >> 16);
		p[1] = mpt::saturate_cast<T>((Util::mul32to64(l, fac2) + Util::mul32to64(r, fac1)) >> 16);
		p += 2;
	}
}

// X-Fade sample data to create smooth loop transitions
bool StereoSepSample(ModSample &smp, SmpLength start, SmpLength end, double separation, CSoundFile &sndFile)
{
	LimitMax(end, smp.nLength);
	if(!smp.HasSampleData() || start >= end || smp.GetNumChannels() != 2) return false;

	const SmpLength length = end - start;
	const uint8 numChn = smp.GetNumChannels();
	const int32 sep32 = Util::Round<int32>(separation * (65536.0 / 100.0));

	if(smp.GetElementarySampleSize() == 2)
		StereoSepSampleImpl(smp.pSample16 + start * numChn, length, sep32);
	else if(smp.GetElementarySampleSize() == 1)
		StereoSepSampleImpl(smp.pSample8 + start * numChn, length, sep32);
	else
		return false;

	PrecomputeLoops(smp, sndFile, false);
	return true;
}


template <class T>
static void ConvertStereoToMonoMixImpl(T *pDest, const SmpLength length)
{
	const T *pEnd = pDest + length;
	for(T *pSource = pDest; pDest != pEnd; pDest++, pSource += 2)
	{
		*pDest = (pSource[0] + pSource[1] + 1) >> 1;
	}
}


template <class T>
static void ConvertStereoToMonoOneChannelImpl(T *pDest, const SmpLength length)
{
	const T *pEnd = pDest + length;
	for(T *pSource = pDest; pDest != pEnd; pDest++, pSource += 2)
	{
		*pDest = *pSource;
	}
}


// Convert a multichannel sample to mono (currently only implemented for stereo)
bool ConvertToMono(ModSample &smp, CSoundFile &sndFile, StereoToMonoMode conversionMode)
{
	if(!smp.HasSampleData() || smp.GetNumChannels() != 2) return false;

	// Note: Sample is overwritten in-place! Unused data is not deallocated!
	if(conversionMode == mixChannels)
	{
		if(smp.GetElementarySampleSize() == 2)
			ConvertStereoToMonoMixImpl(smp.pSample16, smp.nLength);
		else if(smp.GetElementarySampleSize() == 1)
			ConvertStereoToMonoMixImpl(smp.pSample8, smp.nLength);
		else
			return false;
	} else
	{
		if(conversionMode == splitSample)
		{
			conversionMode = onlyLeft;
		}
		if(smp.GetElementarySampleSize() == 2)
			ConvertStereoToMonoOneChannelImpl(smp.pSample16 + (conversionMode == onlyLeft ? 0 : 1), smp.nLength);
		else if(smp.GetElementarySampleSize() == 1)
			ConvertStereoToMonoOneChannelImpl(smp.pSample8 + (conversionMode == onlyLeft ? 0 : 1), smp.nLength);
		else
			return false;
	}

	CriticalSection cs;
	smp.uFlags.reset(CHN_STEREO);
	for(auto &chn : sndFile.m_PlayState.Chn)
	{
		if(chn.pModSample == &smp)
		{
			chn.dwFlags.reset(CHN_STEREO);
		}
	}

	PrecomputeLoops(smp, sndFile, false);
	return true;
}


template <class T>
static void ConvertMonoToStereoImpl(const T * MPT_RESTRICT src, T * MPT_RESTRICT dst, SmpLength length)
{
	while(length--)
	{
		dst[0] = *src;
		dst[1] = *src;
		dst += 2;
		src++;
	}
}


// Convert a multichannel sample to mono (currently only implemented for stereo)
bool ConvertToStereo(ModSample &smp, CSoundFile &sndFile)
{
	if(!smp.HasSampleData() || smp.GetNumChannels() != 1) return false;

	void *newSample = ModSample::AllocateSample(smp.nLength, smp.GetBytesPerSample() * 2);
	if(newSample == nullptr)
	{
		return 0;
	}

	if(smp.GetElementarySampleSize() == 2)
		ConvertMonoToStereoImpl(smp.pSample16, (int16 *)newSample, smp.nLength);
	else if(smp.GetElementarySampleSize() == 1)
		ConvertMonoToStereoImpl(smp.pSample8, (int8 *)newSample, smp.nLength);
	else
		return false;

	CriticalSection cs;
	smp.uFlags.set(CHN_STEREO);
	ReplaceSample(smp, newSample, smp.nLength, sndFile);

	PrecomputeLoops(smp, sndFile, false);
	return true;
}


// Convert 16-bit sample to 8-bit
bool ConvertTo8Bit(ModSample &smp, CSoundFile &sndFile)
{
	if(!smp.HasSampleData() || smp.GetElementarySampleSize() != 2)
		return false;

	CopySample<SC::ConversionChain<SC::Convert<int8, int16>, SC::DecodeIdentity<int16> > >(smp.pSample8, smp.nLength * smp.GetNumChannels(), 1, smp.pSample16, smp.GetSampleSizeInBytes(), 1);
	smp.uFlags.reset(CHN_16BIT);
	for(auto &chn : sndFile.m_PlayState.Chn) if(chn.pModSample == &smp)
	{
		chn.dwFlags.reset(CHN_16BIT);
	}

	smp.PrecomputeLoops(sndFile, false);
	return true;
}


// Convert 8-bit sample to 16-bit
bool ConvertTo16Bit(ModSample &smp, CSoundFile &sndFile)
{
	if(!smp.HasSampleData() || smp.GetElementarySampleSize() != 1)
		return false;

	int16 *newSample = static_cast<int16 *>(ModSample::AllocateSample(smp.nLength, 2 * smp.GetNumChannels()));
	if(newSample == nullptr)
		return false;

	CopySample<SC::ConversionChain<SC::Convert<int16, int8>, SC::DecodeIdentity<int8> > >(newSample, smp.nLength * smp.GetNumChannels(), 1, smp.pSample8, smp.GetSampleSizeInBytes(), 1);
	smp.uFlags.set(CHN_16BIT);
	ctrlSmp::ReplaceSample(smp, newSample, smp.nLength, sndFile);
	smp.PrecomputeLoops(sndFile, false);
	return true;
}


} // namespace ctrlSmp



namespace ctrlChn
{

void ReplaceSample( CSoundFile &sndFile,
					const ModSample &sample,
					const void * const pNewSample,
					const SmpLength nNewLength,
					FlagSet<ChannelFlags> setFlags,
					FlagSet<ChannelFlags> resetFlags)
{
	const bool periodIsFreq = sndFile.PeriodsAreFrequencies();
	auto begin = sndFile.m_PlayState.Chn, end = sndFile.m_PlayState.Chn + CountOf(sndFile.m_PlayState.Chn);
	for (auto chn = begin; chn != end; chn++)
	{
		if (chn->pModSample == &sample)
		{
			if (chn->pCurrentSample != nullptr)
				chn->pCurrentSample = pNewSample;
			if (chn->position.GetUInt() > nNewLength)
				chn->position.Set(0);
			if (chn->nLength > 0)
				LimitMax(chn->nLength, nNewLength);
			if(chn->InSustainLoop())
			{
				chn->nLoopStart = sample.nSustainStart;
				chn->nLoopEnd = sample.nSustainEnd;
			} else
			{
				chn->nLoopStart = sample.nLoopStart;
				chn->nLoopEnd = sample.nLoopEnd;
			}
			chn->dwFlags.set(setFlags);
			chn->dwFlags.reset(resetFlags);
			if(chn->nC5Speed && sample.nC5Speed && !sndFile.UseFinetuneAndTranspose())
			{
				if(periodIsFreq)
					chn->nPeriod = Util::muldivr_unsigned(chn->nPeriod, sample.nC5Speed, chn->nC5Speed);
				else
					chn->nPeriod = Util::muldivr_unsigned(chn->nPeriod, chn->nC5Speed, sample.nC5Speed);
			}
			chn->nC5Speed = sample.nC5Speed;
		}
	}
}

} // namespace ctrlChn


OPENMPT_NAMESPACE_END
