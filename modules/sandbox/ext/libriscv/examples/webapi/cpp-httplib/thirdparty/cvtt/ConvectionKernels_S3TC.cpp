/*
Convection Texture Tools
Copyright (c) 2018-2019 Eric Lasota

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject
to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

-------------------------------------------------------------------------------------

Portions based on DirectX Texture Library (DirectXTex)

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

http://go.microsoft.com/fwlink/?LinkId=248926
*/
#include "ConvectionKernels_Config.h"

#if !defined(CVTT_SINGLE_FILE) || defined(CVTT_SINGLE_FILE_IMPL)

#include "ConvectionKernels_S3TC.h"

#include "ConvectionKernels_AggregatedError.h"
#include "ConvectionKernels_BCCommon.h"
#include "ConvectionKernels_EndpointRefiner.h"
#include "ConvectionKernels_EndpointSelector.h"
#include "ConvectionKernels_IndexSelector.h"
#include "ConvectionKernels_UnfinishedEndpoints.h"
#include "ConvectionKernels_S3TC_SingleColor.h"

void cvtt::Internal::S3TCComputer::Init(MFloat& error)
{
    error = ParallelMath::MakeFloat(FLT_MAX);
}

void cvtt::Internal::S3TCComputer::QuantizeTo6Bits(MUInt15& v)
{
    MUInt15 reduced = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(v, ParallelMath::MakeUInt15(253)) + ParallelMath::MakeUInt16(512), 10));
    v = (reduced << 2) | ParallelMath::RightShift(reduced, 4);
}

void cvtt::Internal::S3TCComputer::QuantizeTo5Bits(MUInt15& v)
{
    MUInt15 reduced = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(v, ParallelMath::MakeUInt15(249)) + ParallelMath::MakeUInt16(1024), 11));
    v = (reduced << 3) | ParallelMath::RightShift(reduced, 2);
}

void cvtt::Internal::S3TCComputer::QuantizeTo565(MUInt15 endPoint[3])
{
    QuantizeTo5Bits(endPoint[0]);
    QuantizeTo6Bits(endPoint[1]);
    QuantizeTo5Bits(endPoint[2]);
}

cvtt::ParallelMath::Float cvtt::Internal::S3TCComputer::ParanoidFactorForSpan(const MSInt16& span)
{
    return ParallelMath::Abs(ParallelMath::ToFloat(span)) * 0.03f;
}

cvtt::ParallelMath::Float cvtt::Internal::S3TCComputer::ParanoidDiff(const MUInt15& a, const MUInt15& b, const MFloat& d)
{
    MFloat absDiff = ParallelMath::Abs(ParallelMath::ToFloat(ParallelMath::LosslessCast<MSInt16>::Cast(a) - ParallelMath::LosslessCast<MSInt16>::Cast(b)));
    absDiff = absDiff + d;
    return absDiff * absDiff;
}

void cvtt::Internal::S3TCComputer::TestSingleColor(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], int range, const float* channelWeights,
    MFloat &bestError, MUInt15 bestEndpoints[2][3], MUInt15 bestIndexes[16], MUInt15 &bestRange, const ParallelMath::RoundTowardNearestForScope *rtn)
{
    float channelWeightsSq[3];

    for (int ch = 0; ch < 3; ch++)
        channelWeightsSq[ch] = channelWeights[ch] * channelWeights[ch];

    MUInt15 totals[3] = { ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0) };

    for (int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 3; ch++)
            totals[ch] = totals[ch] + pixels[px][ch];
    }

    MUInt15 average[3];
    for (int ch = 0; ch < 3; ch++)
        average[ch] = ParallelMath::RightShift(totals[ch] + ParallelMath::MakeUInt15(8), 4);

    const Tables::S3TCSC::TableEntry* rbTable = NULL;
    const Tables::S3TCSC::TableEntry* gTable = NULL;
    if (flags & cvtt::Flags::S3TC_Paranoid)
    {
        if (range == 4)
        {
            rbTable = Tables::S3TCSC::g_singleColor5_3_p;
            gTable = Tables::S3TCSC::g_singleColor6_3_p;
        }
        else
        {
            assert(range == 3);
            rbTable = Tables::S3TCSC::g_singleColor5_2_p;
            gTable = Tables::S3TCSC::g_singleColor6_2_p;
        }
    }
    else
    {
        if (range == 4)
        {
            rbTable = Tables::S3TCSC::g_singleColor5_3;
            gTable = Tables::S3TCSC::g_singleColor6_3;
        }
        else
        {
            assert(range == 3);
            rbTable = Tables::S3TCSC::g_singleColor5_2;
            gTable = Tables::S3TCSC::g_singleColor6_2;
        }
    }

    MUInt15 interpolated[3];
    MUInt15 eps[2][3];
    MSInt16 spans[3];
    for (int i = 0; i < ParallelMath::ParallelSize; i++)
    {
        for (int ch = 0; ch < 3; ch++)
        {
            uint16_t avg = ParallelMath::Extract(average[ch], i);
            const Tables::S3TCSC::TableEntry& tableEntry = ((ch == 1) ? gTable[avg] : rbTable[avg]);
            ParallelMath::PutUInt15(eps[0][ch], i, tableEntry.m_min);
            ParallelMath::PutUInt15(eps[1][ch], i, tableEntry.m_max);
            ParallelMath::PutUInt15(interpolated[ch], i, tableEntry.m_actualColor);
            ParallelMath::PutSInt16(spans[ch], i, tableEntry.m_span);
        }
    }

    MFloat error = ParallelMath::MakeFloatZero();
    if (flags & cvtt::Flags::S3TC_Paranoid)
    {
        MFloat spanParanoidFactors[3];
        for (int ch = 0; ch < 3; ch++)
            spanParanoidFactors[ch] = ParanoidFactorForSpan(spans[ch]);

        for (int px = 0; px < 16; px++)
        {
            for (int ch = 0; ch < 3; ch++)
                error = error + ParanoidDiff(interpolated[ch], pixels[px][ch], spanParanoidFactors[ch]) * channelWeightsSq[ch];
        }
    }
    else
    {
        for (int px = 0; px < 16; px++)
        {
            for (int ch = 0; ch < 3; ch++)
                error = error + ParallelMath::ToFloat(ParallelMath::SqDiffUInt8(interpolated[ch], pixels[px][ch])) * channelWeightsSq[ch];
        }
    }

    ParallelMath::FloatCompFlag better = ParallelMath::Less(error, bestError);
    ParallelMath::Int16CompFlag better16 = ParallelMath::FloatFlagToInt16(better);

    if (ParallelMath::AnySet(better16))
    {
        bestError = ParallelMath::Min(bestError, error);
        for (int epi = 0; epi < 2; epi++)
            for (int ch = 0; ch < 3; ch++)
                ParallelMath::ConditionalSet(bestEndpoints[epi][ch], better16, eps[epi][ch]);

        MUInt15 vindexes = ParallelMath::MakeUInt15(1);
        for (int px = 0; px < 16; px++)
            ParallelMath::ConditionalSet(bestIndexes[px], better16, vindexes);

        ParallelMath::ConditionalSet(bestRange, better16, ParallelMath::MakeUInt15(range));
    }
}

void cvtt::Internal::S3TCComputer::TestEndpoints(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const MFloat preWeightedPixels[16][4], const MUInt15 unquantizedEndPoints[2][3], int range, const float* channelWeights,
    MFloat &bestError, MUInt15 bestEndpoints[2][3], MUInt15 bestIndexes[16], MUInt15 &bestRange, EndpointRefiner<3> *refiner, const ParallelMath::RoundTowardNearestForScope *rtn)
{
    float channelWeightsSq[3];

    for (int ch = 0; ch < 3; ch++)
        channelWeightsSq[ch] = channelWeights[ch] * channelWeights[ch];

    MUInt15 endPoints[2][3];

    for (int ep = 0; ep < 2; ep++)
        for (int ch = 0; ch < 3; ch++)
            endPoints[ep][ch] = unquantizedEndPoints[ep][ch];

    QuantizeTo565(endPoints[0]);
    QuantizeTo565(endPoints[1]);

    IndexSelector<3> selector;
    selector.Init<false>(channelWeights, endPoints, range);

    MUInt15 indexes[16];

    MFloat paranoidFactors[3];
    for (int ch = 0; ch < 3; ch++)
        paranoidFactors[ch] = ParanoidFactorForSpan(ParallelMath::LosslessCast<MSInt16>::Cast(endPoints[0][ch]) - ParallelMath::LosslessCast<MSInt16>::Cast(endPoints[1][ch]));

    MFloat error = ParallelMath::MakeFloatZero();
    AggregatedError<3> aggError;
    for (int px = 0; px < 16; px++)
    {
        MUInt15 index = selector.SelectIndexLDR(floatPixels[px], rtn);
        indexes[px] = index;

        if (refiner)
            refiner->ContributeUnweightedPW(preWeightedPixels[px], index);

        MUInt15 reconstructed[3];
        selector.ReconstructLDRPrecise(index, reconstructed);

        if (flags & Flags::S3TC_Paranoid)
        {
            for (int ch = 0; ch < 3; ch++)
                error = error + ParanoidDiff(reconstructed[ch], pixels[px][ch], paranoidFactors[ch]) * channelWeightsSq[ch];
        }
        else
            BCCommon::ComputeErrorLDR<3>(flags, reconstructed, pixels[px], aggError);
    }

    if (!(flags & Flags::S3TC_Paranoid))
        error = aggError.Finalize(flags, channelWeightsSq);

    ParallelMath::FloatCompFlag better = ParallelMath::Less(error, bestError);

    if (ParallelMath::AnySet(better))
    {
        ParallelMath::Int16CompFlag betterInt16 = ParallelMath::FloatFlagToInt16(better);

        ParallelMath::ConditionalSet(bestError, better, error);

        for (int ep = 0; ep < 2; ep++)
            for (int ch = 0; ch < 3; ch++)
                ParallelMath::ConditionalSet(bestEndpoints[ep][ch], betterInt16, endPoints[ep][ch]);

        for (int px = 0; px < 16; px++)
            ParallelMath::ConditionalSet(bestIndexes[px], betterInt16, indexes[px]);

        ParallelMath::ConditionalSet(bestRange, betterInt16, ParallelMath::MakeUInt15(static_cast<uint16_t>(range)));
    }
}

void cvtt::Internal::S3TCComputer::TestCounts(uint32_t flags, const int *counts, int nCounts, const MUInt15 &numElements, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const MFloat preWeightedPixels[16][4], bool alphaTest,
    const MFloat floatSortedInputs[16][4], const MFloat preWeightedFloatSortedInputs[16][4], const float *channelWeights, MFloat &bestError, MUInt15 bestEndpoints[2][3], MUInt15 bestIndexes[16], MUInt15 &bestRange,
    const ParallelMath::RoundTowardNearestForScope* rtn)
{
    UNREFERENCED_PARAMETER(alphaTest);
    UNREFERENCED_PARAMETER(flags);

    EndpointRefiner<3> refiner;

    refiner.Init(nCounts, channelWeights);

    bool escape = false;
    int e = 0;
    for (int i = 0; i < nCounts; i++)
    {
        for (int n = 0; n < counts[i]; n++)
        {
            ParallelMath::Int16CompFlag valid = ParallelMath::Less(ParallelMath::MakeUInt15(static_cast<uint16_t>(n)), numElements);
            if (!ParallelMath::AnySet(valid))
            {
                escape = true;
                break;
            }

            if (ParallelMath::AllSet(valid))
                refiner.ContributeUnweightedPW(preWeightedFloatSortedInputs[e++], ParallelMath::MakeUInt15(static_cast<uint16_t>(i)));
            else
            {
                MFloat weight = ParallelMath::Select(ParallelMath::Int16FlagToFloat(valid), ParallelMath::MakeFloat(1.0f), ParallelMath::MakeFloat(0.0f));
                refiner.ContributePW(preWeightedFloatSortedInputs[e++], ParallelMath::MakeUInt15(static_cast<uint16_t>(i)), weight);
            }
        }

        if (escape)
            break;
    }

    MUInt15 endPoints[2][3];
    refiner.GetRefinedEndpointsLDR(endPoints, rtn);

    TestEndpoints(flags, pixels, floatPixels, preWeightedPixels, endPoints, nCounts, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, NULL, rtn);
}

void cvtt::Internal::S3TCComputer::PackExplicitAlpha(uint32_t flags, const PixelBlockU8* inputs, int inputChannel, uint8_t* packedBlocks, size_t packedBlockStride)
{
    UNREFERENCED_PARAMETER(flags);
    ParallelMath::RoundTowardNearestForScope rtn;

    float weights[1] = { 1.0f };

    MUInt15 pixels[16];
    MFloat floatPixels[16];

    for (int px = 0; px < 16; px++)
    {
        ParallelMath::ConvertLDRInputs(inputs, px, inputChannel, pixels[px]);
        floatPixels[px] = ParallelMath::ToFloat(pixels[px]);
    }

    MUInt15 ep[2][1] = { { ParallelMath::MakeUInt15(0) },{ ParallelMath::MakeUInt15(255) } };

    IndexSelector<1> selector;
    selector.Init<false>(weights, ep, 16);

    MUInt15 indexes[16];

    for (int px = 0; px < 16; px++)
        indexes[px] = selector.SelectIndexLDR(&floatPixels[px], &rtn);

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        for (int px = 0; px < 16; px += 2)
        {
            int index0 = ParallelMath::Extract(indexes[px], block);
            int index1 = ParallelMath::Extract(indexes[px + 1], block);

            packedBlocks[px / 2] = static_cast<uint8_t>(index0 | (index1 << 4));
        }

        packedBlocks += packedBlockStride;
    }
}

void cvtt::Internal::S3TCComputer::PackInterpolatedAlpha(uint32_t flags, const PixelBlockU8* inputs, int inputChannel, uint8_t* packedBlocks, size_t packedBlockStride, bool isSigned, int maxTweakRounds, int numRefineRounds)
{
    if (maxTweakRounds < 1)
        maxTweakRounds = 1;

    if (numRefineRounds < 1)
        numRefineRounds = 1;

    ParallelMath::RoundTowardNearestForScope rtn;

    float oneWeight[1] = { 1.0f };

    MUInt15 pixels[16];
    MFloat floatPixels[16];

    MUInt15 highTerminal = isSigned ? ParallelMath::MakeUInt15(254) : ParallelMath::MakeUInt15(255);
    MUInt15 highTerminalMinusOne = highTerminal - ParallelMath::MakeUInt15(1);

    for (int px = 0; px < 16; px++)
    {
        ParallelMath::ConvertLDRInputs(inputs, px, inputChannel, pixels[px]);

        if (isSigned)
            pixels[px] = ParallelMath::Min(pixels[px], highTerminal);

        floatPixels[px] = ParallelMath::ToFloat(pixels[px]);
    }

    MUInt15 sortedPixels[16];
    for (int px = 0; px < 16; px++)
        sortedPixels[px] = pixels[px];

    for (int sortEnd = 15; sortEnd > 0; sortEnd--)
    {
        for (int sortOffset = 0; sortOffset < sortEnd; sortOffset++)
        {
            MUInt15 a = sortedPixels[sortOffset];
            MUInt15 b = sortedPixels[sortOffset + 1];

            sortedPixels[sortOffset] = ParallelMath::Min(a, b);
            sortedPixels[sortOffset + 1] = ParallelMath::Max(a, b);
        }
    }

    MUInt15 zero = ParallelMath::MakeUInt15(0);
    MUInt15 one = ParallelMath::MakeUInt15(1);

    MUInt15 bestIsFullRange = zero;
    MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);
    MUInt15 bestEP[2] = { zero, zero };
    MUInt15 bestIndexes[16] = {
        zero, zero, zero, zero,
        zero, zero, zero, zero,
        zero, zero, zero, zero,
        zero, zero, zero, zero
    };

    // Full-precision
    {
        MUInt15 minEP = sortedPixels[0];
        MUInt15 maxEP = sortedPixels[15];

        MFloat base[1] = { ParallelMath::ToFloat(minEP) };
        MFloat offset[1] = { ParallelMath::ToFloat(maxEP - minEP) };

        UnfinishedEndpoints<1> ufep = UnfinishedEndpoints<1>(base, offset);

        int numTweakRounds = BCCommon::TweakRoundsForRange(8);
        if (numTweakRounds > maxTweakRounds)
            numTweakRounds = maxTweakRounds;

        for (int tweak = 0; tweak < numTweakRounds; tweak++)
        {
            MUInt15 ep[2][1];

            ufep.FinishLDR(tweak, 8, ep[0], ep[1]);

            for (int refinePass = 0; refinePass < numRefineRounds; refinePass++)
            {
                EndpointRefiner<1> refiner;
                refiner.Init(8, oneWeight);

                if (isSigned)
                    for (int epi = 0; epi < 2; epi++)
                        ep[epi][0] = ParallelMath::Min(ep[epi][0], highTerminal);

                IndexSelector<1> indexSelector;
                indexSelector.Init<false>(oneWeight, ep, 8);

                MUInt15 indexes[16];

                AggregatedError<1> aggError;
                for (int px = 0; px < 16; px++)
                {
                    MUInt15 index = indexSelector.SelectIndexLDR(&floatPixels[px], &rtn);

                    MUInt15 reconstructedPixel;

                    indexSelector.ReconstructLDRPrecise(index, &reconstructedPixel);
                    BCCommon::ComputeErrorLDR<1>(flags, &reconstructedPixel, &pixels[px], aggError);

                    if (refinePass != numRefineRounds - 1)
                        refiner.ContributeUnweightedPW(&floatPixels[px], index);

                    indexes[px] = index;
                }
                MFloat error = aggError.Finalize(flags | Flags::Uniform, oneWeight);

                ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(error, bestError);
                ParallelMath::Int16CompFlag errorBetter16 = ParallelMath::FloatFlagToInt16(errorBetter);

                if (ParallelMath::AnySet(errorBetter16))
                {
                    bestError = ParallelMath::Min(error, bestError);
                    ParallelMath::ConditionalSet(bestIsFullRange, errorBetter16, one);
                    for (int px = 0; px < 16; px++)
                        ParallelMath::ConditionalSet(bestIndexes[px], errorBetter16, indexes[px]);

                    for (int epi = 0; epi < 2; epi++)
                        ParallelMath::ConditionalSet(bestEP[epi], errorBetter16, ep[epi][0]);
                }

                if (refinePass != numRefineRounds - 1)
                    refiner.GetRefinedEndpointsLDR(ep, &rtn);
            }
        }
    }

    // Reduced precision with special endpoints
    {
        MUInt15 bestHeuristicMin = sortedPixels[0];
        MUInt15 bestHeuristicMax = sortedPixels[15];

        ParallelMath::Int16CompFlag canTryClipping;

        // In reduced precision, we want try putting endpoints at the reserved indexes at the ends.
        // The heuristic we use is to assign indexes to the end as long as they aren't off by more than half of the index range.
        // This will usually not find anything, but it's cheap to check.

        {
            MUInt15 largestPossibleRange = bestHeuristicMax - bestHeuristicMin; // Max: 255
            MUInt15 lowestPossibleClearance = ParallelMath::Min(bestHeuristicMin, static_cast<MUInt15>(highTerminal - bestHeuristicMax));

            MUInt15 lowestPossibleClearanceTimes10 = (lowestPossibleClearance << 2) + (lowestPossibleClearance << 4);
            canTryClipping = ParallelMath::LessOrEqual(lowestPossibleClearanceTimes10, largestPossibleRange);
        }

        if (ParallelMath::AnySet(canTryClipping))
        {
            MUInt15 lowClearances[16];
            MUInt15 highClearances[16];
            MUInt15 bestSkipCount = ParallelMath::MakeUInt15(0);

            lowClearances[0] = highClearances[0] = ParallelMath::MakeUInt15(0);

            for (int px = 1; px < 16; px++)
            {
                lowClearances[px] = sortedPixels[px - 1];
                highClearances[px] = highTerminal - sortedPixels[16 - px];
            }

            for (uint16_t firstIndex = 0; firstIndex < 16; firstIndex++)
            {
                uint16_t numSkippedLow = firstIndex;

                MUInt15 lowClearance = lowClearances[firstIndex];

                for (uint16_t lastIndex = firstIndex; lastIndex < 16; lastIndex++)
                {
                    uint16_t numSkippedHigh = 15 - lastIndex;
                    uint16_t numSkipped = numSkippedLow + numSkippedHigh;

                    MUInt15 numSkippedV = ParallelMath::MakeUInt15(numSkipped);

                    ParallelMath::Int16CompFlag areMoreSkipped = ParallelMath::Less(bestSkipCount, numSkippedV);

                    if (!ParallelMath::AnySet(areMoreSkipped))
                        continue;

                    MUInt15 clearance = ParallelMath::Max(highClearances[numSkippedHigh], lowClearance);
                    MUInt15 clearanceTimes10 = (clearance << 2) + (clearance << 4);

                    MUInt15 range = sortedPixels[lastIndex] - sortedPixels[firstIndex];

                    ParallelMath::Int16CompFlag isBetter = (areMoreSkipped & ParallelMath::LessOrEqual(clearanceTimes10, range));
                    ParallelMath::ConditionalSet(bestHeuristicMin, isBetter, sortedPixels[firstIndex]);
                    ParallelMath::ConditionalSet(bestHeuristicMax, isBetter, sortedPixels[lastIndex]);
                }
            }
        }

        MUInt15 bestSimpleMin = one;
        MUInt15 bestSimpleMax = highTerminalMinusOne;

        for (int px = 0; px < 16; px++)
        {
            ParallelMath::ConditionalSet(bestSimpleMin, ParallelMath::Less(zero, sortedPixels[15 - px]), sortedPixels[15 - px]);
            ParallelMath::ConditionalSet(bestSimpleMax, ParallelMath::Less(sortedPixels[px], highTerminal), sortedPixels[px]);
        }

        MUInt15 minEPs[2] = { bestSimpleMin, bestHeuristicMin };
        MUInt15 maxEPs[2] = { bestSimpleMax, bestHeuristicMax };

        int minEPRange = 2;
        if (ParallelMath::AllSet(ParallelMath::Equal(minEPs[0], minEPs[1])))
            minEPRange = 1;

        int maxEPRange = 2;
        if (ParallelMath::AllSet(ParallelMath::Equal(maxEPs[0], maxEPs[1])))
            maxEPRange = 1;

        for (int minEPIndex = 0; minEPIndex < minEPRange; minEPIndex++)
        {
            for (int maxEPIndex = 0; maxEPIndex < maxEPRange; maxEPIndex++)
            {
                MFloat base[1] = { ParallelMath::ToFloat(minEPs[minEPIndex]) };
                MFloat offset[1] = { ParallelMath::ToFloat(maxEPs[maxEPIndex] - minEPs[minEPIndex]) };

                UnfinishedEndpoints<1> ufep = UnfinishedEndpoints<1>(base, offset);

                int numTweakRounds = BCCommon::TweakRoundsForRange(6);
                if (numTweakRounds > maxTweakRounds)
                    numTweakRounds = maxTweakRounds;

                for (int tweak = 0; tweak < numTweakRounds; tweak++)
                {
                    MUInt15 ep[2][1];

                    ufep.FinishLDR(tweak, 8, ep[0], ep[1]);

                    for (int refinePass = 0; refinePass < numRefineRounds; refinePass++)
                    {
                        EndpointRefiner<1> refiner;
                        refiner.Init(6, oneWeight);

                        if (isSigned)
                            for (int epi = 0; epi < 2; epi++)
                                ep[epi][0] = ParallelMath::Min(ep[epi][0], highTerminal);

                        IndexSelector<1> indexSelector;
                        indexSelector.Init<false>(oneWeight, ep, 6);

                        MUInt15 indexes[16];
                        MFloat error = ParallelMath::MakeFloatZero();

                        for (int px = 0; px < 16; px++)
                        {
                            MUInt15 selectedIndex = indexSelector.SelectIndexLDR(&floatPixels[px], &rtn);

                            MUInt15 reconstructedPixel;

                            indexSelector.ReconstructLDRPrecise(selectedIndex, &reconstructedPixel);

                            MFloat zeroError = BCCommon::ComputeErrorLDRSimple<1>(flags | Flags::Uniform, &zero, &pixels[px], 1, oneWeight);
                            MFloat highTerminalError = BCCommon::ComputeErrorLDRSimple<1>(flags | Flags::Uniform, &highTerminal, &pixels[px], 1, oneWeight);
                            MFloat selectedIndexError = BCCommon::ComputeErrorLDRSimple<1>(flags | Flags::Uniform, &reconstructedPixel, &pixels[px], 1, oneWeight);

                            MFloat bestPixelError = zeroError;
                            MUInt15 index = ParallelMath::MakeUInt15(6);

                            ParallelMath::ConditionalSet(index, ParallelMath::FloatFlagToInt16(ParallelMath::Less(highTerminalError, bestPixelError)), ParallelMath::MakeUInt15(7));
                            bestPixelError = ParallelMath::Min(bestPixelError, highTerminalError);

                            ParallelMath::FloatCompFlag selectedIndexBetter = ParallelMath::Less(selectedIndexError, bestPixelError);

                            if (ParallelMath::AllSet(selectedIndexBetter))
                            {
                                if (refinePass != numRefineRounds - 1)
                                    refiner.ContributeUnweightedPW(&floatPixels[px], selectedIndex);
                            }
                            else
                            {
                                MFloat refineWeight = ParallelMath::Select(selectedIndexBetter, ParallelMath::MakeFloat(1.0f), ParallelMath::MakeFloatZero());

                                if (refinePass != numRefineRounds - 1)
                                    refiner.ContributePW(&floatPixels[px], selectedIndex, refineWeight);
                            }

                            ParallelMath::ConditionalSet(index, ParallelMath::FloatFlagToInt16(selectedIndexBetter), selectedIndex);
                            bestPixelError = ParallelMath::Min(bestPixelError, selectedIndexError);

                            error = error + bestPixelError;

                            indexes[px] = index;
                        }

                        ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(error, bestError);
                        ParallelMath::Int16CompFlag errorBetter16 = ParallelMath::FloatFlagToInt16(errorBetter);

                        if (ParallelMath::AnySet(errorBetter16))
                        {
                            bestError = ParallelMath::Min(error, bestError);
                            ParallelMath::ConditionalSet(bestIsFullRange, errorBetter16, zero);
                            for (int px = 0; px < 16; px++)
                                ParallelMath::ConditionalSet(bestIndexes[px], errorBetter16, indexes[px]);

                            for (int epi = 0; epi < 2; epi++)
                                ParallelMath::ConditionalSet(bestEP[epi], errorBetter16, ep[epi][0]);
                        }

                        if (refinePass != numRefineRounds - 1)
                            refiner.GetRefinedEndpointsLDR(ep, &rtn);
                    }
                }
            }
        }
    }

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        int ep0 = ParallelMath::Extract(bestEP[0], block);
        int ep1 = ParallelMath::Extract(bestEP[1], block);
        int isFullRange = ParallelMath::Extract(bestIsFullRange, block);

        if (isSigned)
        {
            ep0 -= 127;
            ep1 -= 127;

            assert(ep0 >= -127 && ep0 <= 127);
            assert(ep1 >= -127 && ep1 <= 127);
        }


        bool swapEndpoints = (isFullRange != 0) != (ep0 > ep1);

        if (swapEndpoints)
            std::swap(ep0, ep1);

        uint16_t dumpBits = 0;
        int dumpBitsOffset = 0;
        int dumpByteOffset = 2;
        packedBlocks[0] = static_cast<uint8_t>(ep0 & 0xff);
        packedBlocks[1] = static_cast<uint8_t>(ep1 & 0xff);

        int maxValue = (isFullRange != 0) ? 7 : 5;

        for (int px = 0; px < 16; px++)
        {
            int index = ParallelMath::Extract(bestIndexes[px], block);

            if (swapEndpoints && index <= maxValue)
                index = maxValue - index;

            if (index != 0)
            {
                if (index == maxValue)
                    index = 1;
                else if (index < maxValue)
                    index++;
            }

            assert(index >= 0 && index < 8);

            dumpBits |= static_cast<uint16_t>(index << dumpBitsOffset);
            dumpBitsOffset += 3;

            if (dumpBitsOffset >= 8)
            {
                assert(dumpByteOffset < 8);
                packedBlocks[dumpByteOffset] = static_cast<uint8_t>(dumpBits & 0xff);
                dumpBits >>= 8;
                dumpBitsOffset -= 8;
                dumpByteOffset++;
            }
        }

        assert(dumpBitsOffset == 0);
        assert(dumpByteOffset == 8);

        packedBlocks += packedBlockStride;
    }
}

void cvtt::Internal::S3TCComputer::PackRGB(uint32_t flags, const PixelBlockU8* inputs, uint8_t* packedBlocks, size_t packedBlockStride, const float channelWeights[4], bool alphaTest, float alphaThreshold, bool exhaustive, int maxTweakRounds, int numRefineRounds)
{
    ParallelMath::RoundTowardNearestForScope rtn;

    if (numRefineRounds < 1)
        numRefineRounds = 1;

    if (maxTweakRounds < 1)
        maxTweakRounds = 1;

    EndpointSelector<3, 8> endpointSelector;

    MUInt15 pixels[16][4];
    MFloat floatPixels[16][4];

    MFloat preWeightedPixels[16][4];

    for (int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 4; ch++)
            ParallelMath::ConvertLDRInputs(inputs, px, ch, pixels[px][ch]);
    }

    for (int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 4; ch++)
            floatPixels[px][ch] = ParallelMath::ToFloat(pixels[px][ch]);
    }

    if (alphaTest)
    {
        MUInt15 threshold = ParallelMath::MakeUInt15(static_cast<uint16_t>(floor(alphaThreshold * 255.0f + 0.5f)));

        for (int px = 0; px < 16; px++)
        {
            ParallelMath::Int16CompFlag belowThreshold = ParallelMath::Less(pixels[px][3], threshold);
            pixels[px][3] = ParallelMath::Select(belowThreshold, ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(255));
        }
    }

    BCCommon::PreWeightPixelsLDR<4>(preWeightedPixels, pixels, channelWeights);

    MUInt15 minAlpha = ParallelMath::MakeUInt15(255);

    for (int px = 0; px < 16; px++)
        minAlpha = ParallelMath::Min(minAlpha, pixels[px][3]);

    MFloat pixelWeights[16];
    for (int px = 0; px < 16; px++)
    {
        pixelWeights[px] = ParallelMath::MakeFloat(1.0f);
        if (alphaTest)
        {
            ParallelMath::Int16CompFlag isTransparent = ParallelMath::Less(pixels[px][3], ParallelMath::MakeUInt15(255));

            ParallelMath::ConditionalSet(pixelWeights[px], ParallelMath::Int16FlagToFloat(isTransparent), ParallelMath::MakeFloatZero());
        }
    }

    for (int pass = 0; pass < NumEndpointSelectorPasses; pass++)
    {
        for (int px = 0; px < 16; px++)
            endpointSelector.ContributePass(preWeightedPixels[px], pass, pixelWeights[px]);

        endpointSelector.FinishPass(pass);
    }

    UnfinishedEndpoints<3> ufep = endpointSelector.GetEndpoints(channelWeights);

    MUInt15 bestEndpoints[2][3];
    MUInt15 bestIndexes[16];
    MUInt15 bestRange = ParallelMath::MakeUInt15(0);
    MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);

    for (int px = 0; px < 16; px++)
        bestIndexes[px] = ParallelMath::MakeUInt15(0);

    for (int ep = 0; ep < 2; ep++)
        for (int ch = 0; ch < 3; ch++)
            bestEndpoints[ep][ch] = ParallelMath::MakeUInt15(0);

    if (exhaustive)
    {
        MSInt16 sortBins[16];

        {
            // Compute an 11-bit index, change it to signed, stuff it in the high bits of the sort bins,
            // and pack the original indexes into the low bits.

            MUInt15 sortEP[2][3];
            ufep.FinishLDR(0, 11, sortEP[0], sortEP[1]);

            IndexSelector<3> sortSelector;
            sortSelector.Init<false>(channelWeights, sortEP, 1 << 11);

            for (int16_t px = 0; px < 16; px++)
            {
                MSInt16 sortBin = ParallelMath::LosslessCast<MSInt16>::Cast(sortSelector.SelectIndexLDR(floatPixels[px], &rtn) << 4);

                if (alphaTest)
                {
                    ParallelMath::Int16CompFlag isTransparent = ParallelMath::Less(pixels[px][3], ParallelMath::MakeUInt15(255));

                    ParallelMath::ConditionalSet(sortBin, isTransparent, ParallelMath::MakeSInt16(-16)); // 0xfff0
                }

                sortBin = sortBin + ParallelMath::MakeSInt16(px);

                sortBins[px] = sortBin;
            }
        }

        // Sort bins
        for (int sortEnd = 1; sortEnd < 16; sortEnd++)
        {
            for (int sortLoc = sortEnd; sortLoc > 0; sortLoc--)
            {
                MSInt16 a = sortBins[sortLoc];
                MSInt16 b = sortBins[sortLoc - 1];

                sortBins[sortLoc] = ParallelMath::Max(a, b);
                sortBins[sortLoc - 1] = ParallelMath::Min(a, b);
            }
        }

        MUInt15 firstElement = ParallelMath::MakeUInt15(0);
        for (uint16_t e = 0; e < 16; e++)
        {
            ParallelMath::Int16CompFlag isInvalid = ParallelMath::Less(sortBins[e], ParallelMath::MakeSInt16(0));
            ParallelMath::ConditionalSet(firstElement, isInvalid, ParallelMath::MakeUInt15(e + 1));
            if (!ParallelMath::AnySet(isInvalid))
                break;
        }

        MUInt15 numElements = ParallelMath::MakeUInt15(16) - firstElement;

        MUInt15 sortedInputs[16][4];
        MFloat floatSortedInputs[16][4];
        MFloat pwFloatSortedInputs[16][4];

        for (int e = 0; e < 16; e++)
        {
            for (int ch = 0; ch < 4; ch++)
                sortedInputs[e][ch] = ParallelMath::MakeUInt15(0);
        }

        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            for (int e = ParallelMath::Extract(firstElement, block); e < 16; e++)
            {
                ParallelMath::ScalarUInt16 sortBin = ParallelMath::Extract(sortBins[e], block);
                int originalIndex = (sortBin & 15);

                for (int ch = 0; ch < 4; ch++)
                    ParallelMath::PutUInt15(sortedInputs[15 - e][ch], block, ParallelMath::Extract(pixels[originalIndex][ch], block));
            }
        }

        for (int e = 0; e < 16; e++)
        {
            for (int ch = 0; ch < 4; ch++)
            {
                MFloat f = ParallelMath::ToFloat(sortedInputs[e][ch]);
                floatSortedInputs[e][ch] = f;
                pwFloatSortedInputs[e][ch] = f * channelWeights[ch];
            }
        }

        for (int n0 = 0; n0 <= 15; n0++)
        {
            int remainingFor1 = 16 - n0;
            if (remainingFor1 == 16)
                remainingFor1 = 15;

            for (int n1 = 0; n1 <= remainingFor1; n1++)
            {
                int remainingFor2 = 16 - n1 - n0;
                if (remainingFor2 == 16)
                    remainingFor2 = 15;

                for (int n2 = 0; n2 <= remainingFor2; n2++)
                {
                    int n3 = 16 - n2 - n1 - n0;

                    if (n3 == 16)
                        continue;

                    int counts[4] = { n0, n1, n2, n3 };

                    TestCounts(flags, counts, 4, numElements, pixels, floatPixels, preWeightedPixels, alphaTest, floatSortedInputs, pwFloatSortedInputs, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, &rtn);
                }
            }
        }

        TestSingleColor(flags, pixels, floatPixels, 4, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, &rtn);

        if (alphaTest)
        {
            for (int n0 = 0; n0 <= 15; n0++)
            {
                int remainingFor1 = 16 - n0;
                if (remainingFor1 == 16)
                    remainingFor1 = 15;

                for (int n1 = 0; n1 <= remainingFor1; n1++)
                {
                    int n2 = 16 - n1 - n0;

                    if (n2 == 16)
                        continue;

                    int counts[3] = { n0, n1, n2 };

                    TestCounts(flags, counts, 3, numElements, pixels, floatPixels, preWeightedPixels, alphaTest, floatSortedInputs, pwFloatSortedInputs, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, &rtn);
                }
            }

            TestSingleColor(flags, pixels, floatPixels, 3, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, &rtn);
        }
    }
    else
    {
        int minRange = alphaTest ? 3 : 4;

        for (int range = minRange; range <= 4; range++)
        {
            int tweakRounds = BCCommon::TweakRoundsForRange(range);
            if (tweakRounds > maxTweakRounds)
                tweakRounds = maxTweakRounds;

            for (int tweak = 0; tweak < tweakRounds; tweak++)
            {
                MUInt15 endPoints[2][3];

                ufep.FinishLDR(tweak, range, endPoints[0], endPoints[1]);

                for (int refine = 0; refine < numRefineRounds; refine++)
                {
                    EndpointRefiner<3> refiner;
                    refiner.Init(range, channelWeights);

                    TestEndpoints(flags, pixels, floatPixels, preWeightedPixels, endPoints, range, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, &refiner, &rtn);

                    if (refine != numRefineRounds - 1)
                        refiner.GetRefinedEndpointsLDR(endPoints, &rtn);
                }
            }
        }
    }

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        ParallelMath::ScalarUInt16 range = ParallelMath::Extract(bestRange, block);
        assert(range == 3 || range == 4);

        ParallelMath::ScalarUInt16 compressedEP[2];
        for (int ep = 0; ep < 2; ep++)
        {
            ParallelMath::ScalarUInt16 endPoint[3];
            for (int ch = 0; ch < 3; ch++)
                endPoint[ch] = ParallelMath::Extract(bestEndpoints[ep][ch], block);

            int compressed = (endPoint[0] & 0xf8) << 8;
            compressed |= (endPoint[1] & 0xfc) << 3;
            compressed |= (endPoint[2] & 0xf8) >> 3;

            compressedEP[ep] = static_cast<ParallelMath::ScalarUInt16>(compressed);
        }

        int indexOrder[4];

        if (range == 4)
        {
            if (compressedEP[0] == compressedEP[1])
            {
                indexOrder[0] = 0;
                indexOrder[1] = 0;
                indexOrder[2] = 0;
                indexOrder[3] = 0;
            }
            else if (compressedEP[0] < compressedEP[1])
            {
                std::swap(compressedEP[0], compressedEP[1]);
                indexOrder[0] = 1;
                indexOrder[1] = 3;
                indexOrder[2] = 2;
                indexOrder[3] = 0;
            }
            else
            {
                indexOrder[0] = 0;
                indexOrder[1] = 2;
                indexOrder[2] = 3;
                indexOrder[3] = 1;
            }
        }
        else
        {
            assert(range == 3);

            if (compressedEP[0] > compressedEP[1])
            {
                std::swap(compressedEP[0], compressedEP[1]);
                indexOrder[0] = 1;
                indexOrder[1] = 2;
                indexOrder[2] = 0;
            }
            else
            {
                indexOrder[0] = 0;
                indexOrder[1] = 2;
                indexOrder[2] = 1;
            }
            indexOrder[3] = 3;
        }

        packedBlocks[0] = static_cast<uint8_t>(compressedEP[0] & 0xff);
        packedBlocks[1] = static_cast<uint8_t>((compressedEP[0] >> 8) & 0xff);
        packedBlocks[2] = static_cast<uint8_t>(compressedEP[1] & 0xff);
        packedBlocks[3] = static_cast<uint8_t>((compressedEP[1] >> 8) & 0xff);

        for (int i = 0; i < 16; i += 4)
        {
            int packedIndexes = 0;
            for (int subi = 0; subi < 4; subi++)
            {
                ParallelMath::ScalarUInt16 index = ParallelMath::Extract(bestIndexes[i + subi], block);
                packedIndexes |= (indexOrder[index] << (subi * 2));
            }

            packedBlocks[4 + i / 4] = static_cast<uint8_t>(packedIndexes);
        }

        packedBlocks += packedBlockStride;
    }
}

#endif
