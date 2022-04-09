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

#include "ConvectionKernels.h"
#include "ConvectionKernels_ETC.h"
#include "ConvectionKernels_ETC1.h"
#include "ConvectionKernels_ETC2.h"
#include "ConvectionKernels_ETC2_Rounding.h"
#include "ConvectionKernels_ParallelMath.h"
#include "ConvectionKernels_FakeBT709_Rounding.h"

#include <cmath>

const int cvtt::Internal::ETCComputer::g_flipTables[2][2][8] =
{
    {
        { 0, 1, 4, 5, 8, 9, 12, 13 },
        { 2, 3, 6, 7, 10, 11, 14, 15 }
    },
    {
        { 0, 1, 2, 3, 4, 5, 6, 7 },
        { 8, 9, 10, 11, 12, 13, 14, 15 }
    },
};

cvtt::ParallelMath::Float cvtt::Internal::ETCComputer::ComputeErrorUniform(const MUInt15 pixelA[3], const MUInt15 pixelB[3])
{
    MSInt16 d0 = ParallelMath::LosslessCast<MSInt16>::Cast(pixelA[0]) - ParallelMath::LosslessCast<MSInt16>::Cast(pixelB[0]);
    MFloat fd0 = ParallelMath::ToFloat(d0);
    MFloat error = fd0 * fd0;
    for (int ch = 1; ch < 3; ch++)
    {
        MSInt16 d = ParallelMath::LosslessCast<MSInt16>::Cast(pixelA[ch]) - ParallelMath::LosslessCast<MSInt16>::Cast(pixelB[ch]);
        MFloat fd = ParallelMath::ToFloat(d);
        error = error + fd * fd;
    }
    return error;
}

cvtt::ParallelMath::Float cvtt::Internal::ETCComputer::ComputeErrorWeighted(const MUInt15 reconstructed[3], const MFloat preWeightedPixel[3], const Options options)
{
    MFloat dr = ParallelMath::ToFloat(reconstructed[0]) * options.redWeight - preWeightedPixel[0];
    MFloat dg = ParallelMath::ToFloat(reconstructed[1]) * options.greenWeight - preWeightedPixel[1];
    MFloat db = ParallelMath::ToFloat(reconstructed[2]) * options.blueWeight - preWeightedPixel[2];

    return dr * dr + dg * dg + db * db;
}

cvtt::ParallelMath::Float cvtt::Internal::ETCComputer::ComputeErrorFakeBT709(const MUInt15 reconstructed[3], const MFloat preWeightedPixel[3])
{
    MFloat yuv[3];
    ConvertToFakeBT709(yuv, reconstructed);

    MFloat dy = yuv[0] - preWeightedPixel[0];
    MFloat du = yuv[1] - preWeightedPixel[1];
    MFloat dv = yuv[2] - preWeightedPixel[2];

    return dy * dy + du * du + dv * dv;
}

void cvtt::Internal::ETCComputer::TestHalfBlock(MFloat &outError, MUInt16 &outSelectors, MUInt15 quantizedPackedColor, const MUInt15 pixels[8][3], const MFloat preWeightedPixels[8][3], const MSInt16 modifiers[4], bool isDifferential, const Options &options)
{
    MUInt15 quantized[3];
    MUInt15 unquantized[3];

    for (int ch = 0; ch < 3; ch++)
    {
        quantized[ch] = (ParallelMath::RightShift(quantizedPackedColor, (ch * 5)) & ParallelMath::MakeUInt15(31));

        if (isDifferential)
            unquantized[ch] = (quantized[ch] << 3) | ParallelMath::RightShift(quantized[ch], 2);
        else
            unquantized[ch] = (quantized[ch] << 4) | quantized[ch];
    }

    MUInt16 selectors = ParallelMath::MakeUInt16(0);
    MFloat totalError = ParallelMath::MakeFloatZero();

    MUInt15 u15_255 = ParallelMath::MakeUInt15(255);
    MSInt16 s16_zero = ParallelMath::MakeSInt16(0);

    MUInt15 unquantizedModified[4][3];
    for (unsigned int s = 0; s < 4; s++)
        for (int ch = 0; ch < 3; ch++)
            unquantizedModified[s][ch] = ParallelMath::Min(ParallelMath::ToUInt15(ParallelMath::Max(ParallelMath::ToSInt16(unquantized[ch]) + modifiers[s], s16_zero)), u15_255);

    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);
    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);

    for (int px = 0; px < 8; px++)
    {
        MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);
        MUInt16 bestSelector = ParallelMath::MakeUInt16(0);

        for (unsigned int s = 0; s < 4; s++)
        {
            MFloat error;
            if (isFakeBT709)
                error = ComputeErrorFakeBT709(unquantizedModified[s], preWeightedPixels[px]);
            else if (isUniform)
                error = ComputeErrorUniform(pixels[px], unquantizedModified[s]);
            else
                error = ComputeErrorWeighted(unquantizedModified[s], preWeightedPixels[px], options);

            ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(error, bestError);
            bestSelector = ParallelMath::Select(ParallelMath::FloatFlagToInt16(errorBetter), ParallelMath::MakeUInt16(s), bestSelector);
            bestError = ParallelMath::Min(error, bestError);
        }

        totalError = totalError + bestError;
        selectors = selectors | (bestSelector << (px * 2));
    }

    outError = totalError;
    outSelectors = selectors;
}

void cvtt::Internal::ETCComputer::TestHalfBlockPunchthrough(MFloat &outError, MUInt16 &outSelectors, MUInt15 quantizedPackedColor, const MUInt15 pixels[8][3], const MFloat preWeightedPixels[8][3], const ParallelMath::Int16CompFlag isTransparent[8], const MUInt15 modifier, const Options &options)
{
    MUInt15 quantized[3];
    MUInt15 unquantized[3];

    for (int ch = 0; ch < 3; ch++)
    {
        quantized[ch] = (ParallelMath::RightShift(quantizedPackedColor, (ch * 5)) & ParallelMath::MakeUInt15(31));
        unquantized[ch] = (quantized[ch] << 3) | ParallelMath::RightShift(quantized[ch], 2);
    }

    MUInt16 selectors = ParallelMath::MakeUInt16(0);
    MFloat totalError = ParallelMath::MakeFloatZero();

    MUInt15 u15_255 = ParallelMath::MakeUInt15(255);
    MSInt16 s16_zero = ParallelMath::MakeSInt16(0);

    MUInt15 unquantizedModified[3][3];
    for (int ch = 0; ch < 3; ch++)
    {
        unquantizedModified[0][ch] = ParallelMath::Max(unquantized[ch], modifier) - modifier;
        unquantizedModified[1][ch] = unquantized[ch];
        unquantizedModified[2][ch] = ParallelMath::Min(unquantized[ch] + modifier, u15_255);
    }

    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);
    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);

    for (int px = 0; px < 8; px++)
    {
        ParallelMath::FloatCompFlag isTransparentFloat = ParallelMath::Int16FlagToFloat(isTransparent[px]);

        MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);
        MUInt15 bestSelector = ParallelMath::MakeUInt15(0);

        for (unsigned int s = 0; s < 3; s++)
        {
            MFloat error;
            if (isFakeBT709)
                error = ComputeErrorFakeBT709(unquantizedModified[s], preWeightedPixels[px]);
            else if (isUniform)
                error = ComputeErrorUniform(pixels[px], unquantizedModified[s]);
            else
                error = ComputeErrorWeighted(unquantizedModified[s], preWeightedPixels[px], options);

            ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(error, bestError);
            bestSelector = ParallelMath::Select(ParallelMath::FloatFlagToInt16(errorBetter), ParallelMath::MakeUInt15(s), bestSelector);
            bestError = ParallelMath::Min(error, bestError);
        }

        // Annoying quirk: The ETC encoding machinery assumes that selectors are in the table order in the spec, which isn't
        // the same as their encoding bits, so the transparent index is actually 1 and the valid indexes are 0, 2, and 3.

        // Remap selector 1 to 2, and 2 to 3
        bestSelector = ParallelMath::Min(ParallelMath::MakeUInt15(3), bestSelector << 1);

        // Mark zero transparent as 
        ParallelMath::ConditionalSet(bestError, isTransparentFloat, ParallelMath::MakeFloatZero());
        ParallelMath::ConditionalSet(bestSelector, isTransparent[px], ParallelMath::MakeUInt15(1));

        totalError = totalError + bestError;
        selectors = selectors | (ParallelMath::LosslessCast<MUInt16>::Cast(bestSelector) << (px * 2));
    }

    outError = totalError;
    outSelectors = selectors;
}

void cvtt::Internal::ETCComputer::FindBestDifferentialCombination(int flip, int d, const ParallelMath::Int16CompFlag canIgnoreSector[2], ParallelMath::Int16CompFlag& bestIsThisMode, MFloat& bestTotalError, MUInt15& bestFlip, MUInt15& bestD, MUInt15 bestColors[2], MUInt16 bestSelectors[2], MUInt15 bestTables[2], DifferentialResolveStorage &drs)
{
    // We do this part scalar because most of the cost benefit of parallelization is in error evaluation,
    // and this code has a LOT of early-outs and disjointed index lookups that vary heavily between blocks
    // and save a lot of time.
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        bool canIgnore[2] = { ParallelMath::Extract(canIgnoreSector[0], block), ParallelMath::Extract(canIgnoreSector[1], block) };
        bool canIgnoreEither = canIgnore[0] || canIgnore[1];
        float blockBestTotalError = ParallelMath::Extract(bestTotalError, block);
        float bestDiffErrors[2] = { FLT_MAX, FLT_MAX };
        uint16_t bestDiffSelectors[2] = { 0, 0 };
        uint16_t bestDiffColors[2] = { 0, 0 };
        uint16_t bestDiffTables[2] = { 0, 0 };
        for (int sector = 0; sector < 2; sector++)
        {
            unsigned int sectorNumAttempts = ParallelMath::Extract(drs.diffNumAttempts[sector], block);
            for (unsigned int i = 0; i < sectorNumAttempts; i++)
            {
                float error = ParallelMath::Extract(drs.diffErrors[sector][i], block);
                if (error < bestDiffErrors[sector])
                {
                    bestDiffErrors[sector] = error;
                    bestDiffSelectors[sector] = ParallelMath::Extract(drs.diffSelectors[sector][i], block);
                    bestDiffColors[sector] = ParallelMath::Extract(drs.diffColors[sector][i], block);
                    bestDiffTables[sector] = ParallelMath::Extract(drs.diffTables[sector][i], block);
                }
            }
        }

        if (canIgnore[0])
            bestDiffColors[0] = bestDiffColors[1];
        else if (canIgnore[1])
            bestDiffColors[1] = bestDiffColors[0];

        // The best differential possibilities must be better than the best total error
        if (bestDiffErrors[0] + bestDiffErrors[1] < blockBestTotalError)
        {
            // Fast path if the best possible case is legal
            if (canIgnoreEither || ETCDifferentialIsLegalScalar(bestDiffColors[0], bestDiffColors[1]))
            {
                ParallelMath::PutBoolInt16(bestIsThisMode, block, true);
                ParallelMath::PutFloat(bestTotalError, block, bestDiffErrors[0] + bestDiffErrors[1]);
                ParallelMath::PutUInt15(bestFlip, block, flip);
                ParallelMath::PutUInt15(bestD, block, d);
                for (int sector = 0; sector < 2; sector++)
                {
                    ParallelMath::PutUInt15(bestColors[sector], block, bestDiffColors[sector]);
                    ParallelMath::PutUInt16(bestSelectors[sector], block, bestDiffSelectors[sector]);
                    ParallelMath::PutUInt15(bestTables[sector], block, bestDiffTables[sector]);
                }
            }
            else
            {
                // Slow path: Sort the possible cases by quality, and search valid combinations
                // TODO: Pre-flatten the error lists so this is nicer to cache
                unsigned int numSortIndexes[2] = { 0, 0 };
                for (int sector = 0; sector < 2; sector++)
                {
                    unsigned int sectorNumAttempts = ParallelMath::Extract(drs.diffNumAttempts[sector], block);

                    for (unsigned int i = 0; i < sectorNumAttempts; i++)
                    {
                        if (ParallelMath::Extract(drs.diffErrors[sector][i], block) < blockBestTotalError)
                            drs.attemptSortIndexes[sector][numSortIndexes[sector]++] = i;
                    }

                    struct SortPredicate
                    {
                        const MFloat *diffErrors;
                        int block;

                        bool operator()(uint16_t a, uint16_t b) const
                        {
                            float errorA = ParallelMath::Extract(diffErrors[a], block);
                            float errorB = ParallelMath::Extract(diffErrors[b], block);

                            if (errorA < errorB)
                                return true;
                            if (errorA > errorB)
                                return false;

                            return a < b;
                        }
                    };

                    SortPredicate sp;
                    sp.diffErrors = drs.diffErrors[sector];
                    sp.block = block;

                    std::sort<uint16_t*, const SortPredicate&>(drs.attemptSortIndexes[sector], drs.attemptSortIndexes[sector] + numSortIndexes[sector], sp);
                }

                int scannedElements = 0;
                for (unsigned int i = 0; i < numSortIndexes[0]; i++)
                {
                    unsigned int attemptIndex0 = drs.attemptSortIndexes[0][i];
                    float error0 = ParallelMath::Extract(drs.diffErrors[0][attemptIndex0], block);

                    scannedElements++;

                    if (error0 >= blockBestTotalError)
                        break;

                    float maxError1 = ParallelMath::Extract(bestTotalError, block) - error0;
                    uint16_t diffColor0 = ParallelMath::Extract(drs.diffColors[0][attemptIndex0], block);

                    if (maxError1 < bestDiffErrors[1])
                        break;

                    for (unsigned int j = 0; j < numSortIndexes[1]; j++)
                    {
                        unsigned int attemptIndex1 = drs.attemptSortIndexes[1][j];
                        float error1 = ParallelMath::Extract(drs.diffErrors[1][attemptIndex1], block);

                        scannedElements++;

                        if (error1 >= maxError1)
                            break;

                        uint16_t diffColor1 = ParallelMath::Extract(drs.diffColors[1][attemptIndex1], block);

                        if (ETCDifferentialIsLegalScalar(diffColor0, diffColor1))
                        {
                            blockBestTotalError = error0 + error1;

                            ParallelMath::PutBoolInt16(bestIsThisMode, block, true);
                            ParallelMath::PutFloat(bestTotalError, block, blockBestTotalError);
                            ParallelMath::PutUInt15(bestFlip, block, flip);
                            ParallelMath::PutUInt15(bestD, block, d);
                            ParallelMath::PutUInt15(bestColors[0], block, diffColor0);
                            ParallelMath::PutUInt15(bestColors[1], block, diffColor1);
                            ParallelMath::PutUInt16(bestSelectors[0], block, ParallelMath::Extract(drs.diffSelectors[0][attemptIndex0], block));
                            ParallelMath::PutUInt16(bestSelectors[1], block, ParallelMath::Extract(drs.diffSelectors[1][attemptIndex1], block));
                            ParallelMath::PutUInt15(bestTables[0], block, ParallelMath::Extract(drs.diffTables[0][attemptIndex0], block));
                            ParallelMath::PutUInt15(bestTables[1], block, ParallelMath::Extract(drs.diffTables[1][attemptIndex1], block));
                            break;
                        }
                    }
                }
            }
        }
    }
}

cvtt::ParallelMath::Int16CompFlag cvtt::Internal::ETCComputer::ETCDifferentialIsLegalForChannel(const MUInt15 &a, const MUInt15 &b)
{
    MSInt16 diff = ParallelMath::LosslessCast<MSInt16>::Cast(b) - ParallelMath::LosslessCast<MSInt16>::Cast(a);

    return ParallelMath::Less(ParallelMath::MakeSInt16(-5), diff) & ParallelMath::Less(diff, ParallelMath::MakeSInt16(4));
}

cvtt::ParallelMath::Int16CompFlag cvtt::Internal::ETCComputer::ETCDifferentialIsLegal(const MUInt15 &a, const MUInt15 &b)
{
    MUInt15 mask = ParallelMath::MakeUInt15(31);

    return ETCDifferentialIsLegalForChannel(ParallelMath::RightShift(a, 10), ParallelMath::RightShift(b, 10))
        & ETCDifferentialIsLegalForChannel(ParallelMath::RightShift(a, 5) & mask, ParallelMath::RightShift(b, 5) & mask)
        & ETCDifferentialIsLegalForChannel(a & mask, b & mask);
}

bool cvtt::Internal::ETCComputer::ETCDifferentialIsLegalForChannelScalar(const uint16_t &a, const uint16_t &b)
{
    int16_t diff = static_cast<int16_t>(b) - static_cast<int16_t>(a);

    return (-4 <= diff) && (diff <= 3);
}

bool cvtt::Internal::ETCComputer::ETCDifferentialIsLegalScalar(const uint16_t &a, const uint16_t &b)
{
    MUInt15 mask = ParallelMath::MakeUInt15(31);

    return ETCDifferentialIsLegalForChannelScalar((a >> 10), (b >> 10))
        & ETCDifferentialIsLegalForChannelScalar((a >> 5) & 31, (b >> 5) & 31)
        & ETCDifferentialIsLegalForChannelScalar(a & 31, b & 31);
}

void cvtt::Internal::ETCComputer::EncodeTMode(uint8_t *outputBuffer, MFloat &bestError, const ParallelMath::Int16CompFlag isIsolated[16], const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], const Options &options)
{
    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);
    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);

    ParallelMath::Int16CompFlag bestIsThisMode = ParallelMath::MakeBoolInt16(false);

    MUInt15 isolatedTotal[3] = { ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0) };
    MUInt15 lineTotal[3] = { ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0) };

    MUInt15 numPixelsIsolated = ParallelMath::MakeUInt15(0);

    // To speed this up, we compute line total as the sum, then subtract out isolated
    for (unsigned int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 3; ch++)
        {
            isolatedTotal[ch] = isolatedTotal[ch] + ParallelMath::SelectOrZero(isIsolated[px], pixels[px][ch]);
            lineTotal[ch] = lineTotal[ch] + pixels[px][ch];
        }
        numPixelsIsolated = numPixelsIsolated + ParallelMath::SelectOrZero(isIsolated[px], ParallelMath::MakeUInt15(1));
    }

    for (int ch = 0; ch < 3; ch++)
        lineTotal[ch] = lineTotal[ch] - isolatedTotal[ch];

    MUInt15 numPixelsLine = ParallelMath::MakeUInt15(16) - numPixelsIsolated;

    MUInt15 isolatedAverageQuantized[3];
    MUInt15 isolatedAverageTargets[3];
    {
        int divisors[ParallelMath::ParallelSize];
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            divisors[block] = ParallelMath::Extract(numPixelsIsolated, block) * 34;

        MUInt15 addend = (numPixelsIsolated << 4) | numPixelsIsolated;
        for (int ch = 0; ch < 3; ch++)
        {
            // isolatedAverageQuantized[ch] = (isolatedTotal[ch] * 2 + numPixelsIsolated * 17) / (numPixelsIsolated * 34);

            MUInt15 numerator = isolatedTotal[ch] + isolatedTotal[ch];
            if (!isFakeBT709)
                numerator = numerator + addend;

            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                int divisor = divisors[block];
                if (divisor == 0)
                    ParallelMath::PutUInt15(isolatedAverageQuantized[ch], block, 0);
                else
                    ParallelMath::PutUInt15(isolatedAverageQuantized[ch], block, ParallelMath::Extract(numerator, block) / divisor);
            }

            isolatedAverageTargets[ch] = numerator;
        }
    }

    if (isFakeBT709)
        ResolveTHFakeBT709Rounding(isolatedAverageQuantized, isolatedAverageTargets, numPixelsIsolated);

    MUInt15 isolatedColor[3];
    for (int ch = 0; ch < 3; ch++)
        isolatedColor[ch] = (isolatedAverageQuantized[ch]) | (isolatedAverageQuantized[ch] << 4);

    MFloat isolatedError[16];
    for (int px = 0; px < 16; px++)
    {
        if (isFakeBT709)
            isolatedError[px] = ComputeErrorFakeBT709(isolatedColor, preWeightedPixels[px]);
        else if (isUniform)
            isolatedError[px] = ComputeErrorUniform(pixels[px], isolatedColor);
        else
            isolatedError[px] = ComputeErrorWeighted(isolatedColor, preWeightedPixels[px], options);
    }

    MSInt32 bestSelectors = ParallelMath::MakeSInt32(0);
    MUInt15 bestTable = ParallelMath::MakeUInt15(0);
    MUInt15 bestLineColor = ParallelMath::MakeUInt15(0);

    MSInt16 maxLine = ParallelMath::LosslessCast<MSInt16>::Cast(numPixelsLine);
    MSInt16 minLine = ParallelMath::MakeSInt16(0) - maxLine;

    int16_t clusterMaxLine = 0;
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        int16_t blockMaxLine = ParallelMath::Extract(maxLine, block);
        if (blockMaxLine > clusterMaxLine)
            clusterMaxLine = blockMaxLine;
    }

    int16_t clusterMinLine = -clusterMaxLine;

    int lineDivisors[ParallelMath::ParallelSize];
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
        lineDivisors[block] = ParallelMath::Extract(numPixelsLine, block) * 34;

    MUInt15 lineAddend = (numPixelsLine << 4) | numPixelsLine;

    for (int table = 0; table < 8; table++)
    {
        int numUniqueColors[ParallelMath::ParallelSize];
        MUInt15 uniqueQuantizedColors[31];

        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            numUniqueColors[block] = 0;

        MUInt15 modifier = ParallelMath::MakeUInt15(cvtt::Tables::ETC2::g_thModifierTable[table]);
        MUInt15 modifierOffset = (modifier + modifier);

        for (int16_t offsetPremultiplier = clusterMinLine; offsetPremultiplier <= clusterMaxLine; offsetPremultiplier++)
        {
            MSInt16 clampedOffsetPremultiplier = ParallelMath::Max(minLine, ParallelMath::Min(maxLine, ParallelMath::MakeSInt16(offsetPremultiplier)));
            MSInt16 modifierAddend = ParallelMath::CompactMultiply(clampedOffsetPremultiplier, modifierOffset);

            MUInt15 quantized[3];
            if (isFakeBT709)
            {
                MUInt15 targets[3];
                for (int ch = 0; ch < 3; ch++)
                {
                    //quantized[ch] = std::min<int16_t>(15, std::max(0, (lineTotal[ch] * 2 + modifierOffset * offsetPremultiplier)) / (numDAIILine * 34));
                    MUInt15 numerator = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(ParallelMath::MakeSInt16(0), ParallelMath::LosslessCast<MSInt16>::Cast(lineTotal[ch] + lineTotal[ch]) + modifierAddend));
                    MUInt15 divided = ParallelMath::MakeUInt15(0);
                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        int divisor = lineDivisors[block];
                        if (divisor == 0)
                            ParallelMath::PutUInt15(divided, block, 0);
                        else
                            ParallelMath::PutUInt15(divided, block, ParallelMath::Extract(numerator, block) / divisor);
                    }
                    quantized[ch] = ParallelMath::Min(ParallelMath::MakeUInt15(15), divided);
                    targets[ch] = numerator;
                }

                ResolveTHFakeBT709Rounding(quantized, targets, numPixelsLine);
            }
            else
            {
                for (int ch = 0; ch < 3; ch++)
                {
                    //quantized[ch] = std::min<int16_t>(15, std::max(0, (lineTotal[ch] * 2 + numDAIILine * 17 + modifierOffset * offsetPremultiplier)) / (numDAIILine * 34));
                    MUInt15 numerator = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(ParallelMath::MakeSInt16(0), ParallelMath::LosslessCast<MSInt16>::Cast(lineTotal[ch] + lineTotal[ch] + lineAddend) + modifierAddend));
                    MUInt15 divided = ParallelMath::MakeUInt15(0);
                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        int divisor = lineDivisors[block];
                        if (divisor == 0)
                            ParallelMath::PutUInt15(divided, block, 0);
                        else
                            ParallelMath::PutUInt15(divided, block, ParallelMath::Extract(numerator, block) / divisor);
                    }
                    quantized[ch] = ParallelMath::Min(ParallelMath::MakeUInt15(15), divided);
                }
            }

            MUInt15 packedColor = quantized[0] | (quantized[1] << 5) | (quantized[2] << 10);

            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                uint16_t blockPackedColor = ParallelMath::Extract(packedColor, block);
                if (numUniqueColors[block] == 0 || blockPackedColor != ParallelMath::Extract(uniqueQuantizedColors[numUniqueColors[block] - 1], block))
                    ParallelMath::PutUInt15(uniqueQuantizedColors[numUniqueColors[block]++], block, blockPackedColor);
            }
        }

        // Stripe unfilled unique colors
        int maxUniqueColors = 0;
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            if (numUniqueColors[block] > maxUniqueColors)
                maxUniqueColors = numUniqueColors[block];
        }

        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            uint16_t fillColor = ParallelMath::Extract(uniqueQuantizedColors[0], block);

            int numUnique = numUniqueColors[block];
            for (int fill = numUnique + 1; fill < maxUniqueColors; fill++)
                ParallelMath::PutUInt15(uniqueQuantizedColors[fill], block, fillColor);
        }

        for (int ci = 0; ci < maxUniqueColors; ci++)
        {
            MUInt15 lineColors[3][3];
            for (int ch = 0; ch < 3; ch++)
            {
                MUInt15 quantizedChannel = (ParallelMath::RightShift(uniqueQuantizedColors[ci], (ch * 5)) & ParallelMath::MakeUInt15(15));

                MUInt15 unquantizedColor = (quantizedChannel << 4) | quantizedChannel;
                lineColors[0][ch] = ParallelMath::Min(ParallelMath::MakeUInt15(255), unquantizedColor + modifier);
                lineColors[1][ch] = unquantizedColor;
                lineColors[2][ch] = ParallelMath::ToUInt15(ParallelMath::Max(ParallelMath::MakeSInt16(0), ParallelMath::LosslessCast<MSInt16>::Cast(unquantizedColor) - ParallelMath::LosslessCast<MSInt16>::Cast(modifier)));
            }

            MSInt32 selectors = ParallelMath::MakeSInt32(0);
            MFloat error = ParallelMath::MakeFloatZero();
            for (int px = 0; px < 16; px++)
            {
                MFloat pixelError = isolatedError[px];

                MUInt15 pixelBestSelector = ParallelMath::MakeUInt15(0);
                for (int i = 0; i < 3; i++)
                {
                    MFloat error = isUniform ? ComputeErrorUniform(lineColors[i], pixels[px]) : ComputeErrorWeighted(lineColors[i], preWeightedPixels[px], options);
                    ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(error, pixelError);
                    pixelError = ParallelMath::Min(error, pixelError);
                    pixelBestSelector = ParallelMath::Select(ParallelMath::FloatFlagToInt16(errorBetter), ParallelMath::MakeUInt15(i + 1), pixelBestSelector);
                }

                error = error + pixelError;
                selectors = selectors | (ParallelMath::ToInt32(pixelBestSelector) << (px * 2));
            }

            ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, bestError));
            bestError = ParallelMath::Min(error, bestError);

            if (ParallelMath::AnySet(errorBetter))
            {
                ParallelMath::ConditionalSet(bestLineColor, errorBetter, uniqueQuantizedColors[ci]);
                ParallelMath::ConditionalSet(bestSelectors, errorBetter, selectors);
                ParallelMath::ConditionalSet(bestTable, errorBetter, ParallelMath::MakeUInt15(table));
                bestIsThisMode = bestIsThisMode | errorBetter;
            }
        }
    }

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        if (ParallelMath::Extract(bestIsThisMode, block))
        {
            uint32_t lowBits = 0;
            uint32_t highBits = 0;

            uint16_t blockBestLineColor = ParallelMath::Extract(bestLineColor, block);
            ParallelMath::ScalarUInt16 blockIsolatedAverageQuantized[3];

            for (int ch = 0; ch < 3; ch++)
                blockIsolatedAverageQuantized[ch] = ParallelMath::Extract(isolatedAverageQuantized[ch], block);

            uint16_t blockBestTable = ParallelMath::Extract(bestTable, block);
            int32_t blockBestSelectors = ParallelMath::Extract(bestSelectors, block);

            ParallelMath::ScalarUInt16 lineColor[3];
            for (int ch = 0; ch < 3; ch++)
                lineColor[ch] = (blockBestLineColor >> (ch * 5)) & 15;

            EmitTModeBlock(outputBuffer + block * 8, lineColor, blockIsolatedAverageQuantized, blockBestSelectors, blockBestTable, true);
        }
    }
}

void cvtt::Internal::ETCComputer::EncodeHMode(uint8_t *outputBuffer, MFloat &bestError, const ParallelMath::Int16CompFlag groupings[16], const MUInt15 pixels[16][3], HModeEval &he, const MFloat preWeightedPixels[16][3], const Options &options)
{
    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);
    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);

    MUInt15 zero15 = ParallelMath::MakeUInt15(0);

    MUInt15 counts[2] = { zero15, zero15 };

    ParallelMath::Int16CompFlag bestIsThisMode = ParallelMath::MakeBoolInt16(false);

    MUInt15 totals[2][3] =
    {
        { zero15, zero15, zero15 },
        { zero15, zero15, zero15 }
    };

    for (unsigned int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 3; ch++)
        {
            totals[0][ch] = totals[0][ch] + pixels[px][ch];
            totals[1][ch] = totals[1][ch] + ParallelMath::SelectOrZero(groupings[px], pixels[px][ch]);
        }
        counts[1] = counts[1] + ParallelMath::SelectOrZero(groupings[px], ParallelMath::MakeUInt15(1));
    }

    for (int ch = 0; ch < 3; ch++)
        totals[0][ch] = totals[0][ch] - totals[1][ch];
    counts[0] = ParallelMath::MakeUInt15(16) - counts[1];

    MUInt16 bestSectorBits = ParallelMath::MakeUInt16(0);
    MUInt16 bestSignBits = ParallelMath::MakeUInt16(0);
    MUInt15 bestColors[2] = { zero15, zero15 };
    MUInt15 bestTable = ParallelMath::MakeUInt15(0);

    for (int table = 0; table < 8; table++)
    {
        MUInt15 numUniqueColors = zero15;

        int modifier = cvtt::Tables::ETC1::g_thModifierTable[table];

        for (int sector = 0; sector < 2; sector++)
        {
            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                int blockNumUniqueColors = 0;
                uint16_t blockUniqueQuantizedColors[31];

                int maxOffsetMultiplier = ParallelMath::Extract(counts[sector], block);
                int minOffsetMultiplier = -maxOffsetMultiplier;

                int modifierOffset = modifier * 2;

                int blockSectorCounts = ParallelMath::Extract(counts[sector], block);
                int blockSectorTotals[3];
                for (int ch = 0; ch < 3; ch++)
                    blockSectorTotals[ch] = ParallelMath::Extract(totals[sector][ch], block);

                for (int offsetPremultiplier = minOffsetMultiplier; offsetPremultiplier <= maxOffsetMultiplier; offsetPremultiplier++)
                {
                    // TODO: This isn't ideal for FakeBT709
                    int16_t quantized[3];
                    for (int ch = 0; ch < 3; ch++)
                    {
                        if (blockSectorCounts == 0)
                            quantized[ch] = 0;
                        else
                            quantized[ch] = std::min<int16_t>(15, std::max<int16_t>(0, (blockSectorTotals[ch] * 2 + blockSectorCounts * 17 + modifierOffset * offsetPremultiplier)) / (blockSectorCounts * 34));
                    }

                    uint16_t packedColor = (quantized[0] << 10) | (quantized[1] << 5) | quantized[2];
                    if (blockNumUniqueColors == 0 || packedColor != blockUniqueQuantizedColors[blockNumUniqueColors - 1])
                    {
                        assert(blockNumUniqueColors < 32);
                        blockUniqueQuantizedColors[blockNumUniqueColors++] = packedColor;
                    }
                }

                ParallelMath::PutUInt15(he.numUniqueColors[sector], block, blockNumUniqueColors);

                int baseIndex = 0;
                if (sector == 1)
                    baseIndex = ParallelMath::Extract(he.numUniqueColors[0], block);

                for (int i = 0; i < blockNumUniqueColors; i++)
                    ParallelMath::PutUInt15(he.uniqueQuantizedColors[baseIndex + i], block, blockUniqueQuantizedColors[i]);
            }
        }

        MUInt15 totalColors = he.numUniqueColors[0] + he.numUniqueColors[1];
        int maxErrorColors = 0;
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            maxErrorColors = std::max<int>(maxErrorColors, ParallelMath::Extract(totalColors, block));

        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            int lastColor = ParallelMath::Extract(totalColors, block);
            uint16_t stripeColor = ParallelMath::Extract(he.uniqueQuantizedColors[0], block);
            for (int i = lastColor; i < maxErrorColors; i++)
                ParallelMath::PutUInt15(he.uniqueQuantizedColors[i], block, stripeColor);
        }

        for (int ci = 0; ci < maxErrorColors; ci++)
        {
            MUInt15 fifteen = ParallelMath::MakeUInt15(15);
            MUInt15 twoFiftyFive = ParallelMath::MakeUInt15(255);
            MSInt16 zeroS16 = ParallelMath::MakeSInt16(0);

            MUInt15 colors[2][3];
            for (int ch = 0; ch < 3; ch++)
            {
                MUInt15 quantizedChannel = ParallelMath::RightShift(he.uniqueQuantizedColors[ci], ((2 - ch) * 5)) & fifteen;

                MUInt15 unquantizedColor = (quantizedChannel << 4) | quantizedChannel;
                colors[0][ch] = ParallelMath::Min(twoFiftyFive, unquantizedColor + modifier);
                colors[1][ch] = ParallelMath::ToUInt15(ParallelMath::Max(zeroS16, ParallelMath::LosslessCast<MSInt16>::Cast(unquantizedColor) - ParallelMath::MakeSInt16(modifier)));
            }

            MUInt16 signBits = ParallelMath::MakeUInt16(0);
            for (int px = 0; px < 16; px++)
            {
                MFloat errors[2];
                for (int i = 0; i < 2; i++)
                {
                    if (isFakeBT709)
                        errors[i] = ComputeErrorFakeBT709(colors[i], preWeightedPixels[px]);
                    else if (isUniform)
                        errors[i] = ComputeErrorUniform(colors[i], pixels[px]);
                    else
                        errors[i] = ComputeErrorWeighted(colors[i], preWeightedPixels[px], options);
                }

                ParallelMath::Int16CompFlag errorOneLess = ParallelMath::FloatFlagToInt16(ParallelMath::Less(errors[1], errors[0]));
                he.errors[ci][px] = ParallelMath::Min(errors[0], errors[1]);
                signBits = signBits | ParallelMath::SelectOrZero(errorOneLess, ParallelMath::MakeUInt16(1 << px));
            }
            he.signBits[ci] = signBits;
        }

        int maxUniqueColorCombos = 0;
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            int numUniqueColorCombos = ParallelMath::Extract(he.numUniqueColors[0], block) * ParallelMath::Extract(he.numUniqueColors[1], block);
            if (numUniqueColorCombos > maxUniqueColorCombos)
                maxUniqueColorCombos = numUniqueColorCombos;
        }

        MUInt15 indexes[2] = { zero15, zero15 };
        MUInt15 maxIndex[2] = { he.numUniqueColors[0] - ParallelMath::MakeUInt15(1), he.numUniqueColors[1] - ParallelMath::MakeUInt15(1) };

        int block1Starts[ParallelMath::ParallelSize];
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            block1Starts[block] = ParallelMath::Extract(he.numUniqueColors[0], block);

        for (int combo = 0; combo < maxUniqueColorCombos; combo++)
        {
            MUInt15 index0 = indexes[0] + ParallelMath::MakeUInt15(1);
            ParallelMath::Int16CompFlag index0Overflow = ParallelMath::Less(maxIndex[0], index0);
            ParallelMath::ConditionalSet(index0, index0Overflow, ParallelMath::MakeUInt15(0));

            MUInt15 index1 = ParallelMath::Min(maxIndex[1], indexes[1] + ParallelMath::SelectOrZero(index0Overflow, ParallelMath::MakeUInt15(1)));
            indexes[0] = index0;
            indexes[1] = index1;

            int ci0[ParallelMath::ParallelSize];
            int ci1[ParallelMath::ParallelSize];
            MUInt15 color0;
            MUInt15 color1;

            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                ci0[block] = ParallelMath::Extract(index0, block);
                ci1[block] = ParallelMath::Extract(index1, block) + block1Starts[block];
                ParallelMath::PutUInt15(color0, block, ParallelMath::Extract(he.uniqueQuantizedColors[ci0[block]], block));
                ParallelMath::PutUInt15(color1, block, ParallelMath::Extract(he.uniqueQuantizedColors[ci1[block]], block));
            }

            MFloat totalError = ParallelMath::MakeFloatZero();
            MUInt16 sectorBits = ParallelMath::MakeUInt16(0);
            MUInt16 signBits = ParallelMath::MakeUInt16(0);
            for (int px = 0; px < 16; px++)
            {
                MFloat errorCI0;
                MFloat errorCI1;
                MUInt16 signBits0;
                MUInt16 signBits1;

                for (int block = 0; block < ParallelMath::ParallelSize; block++)
                {
                    ParallelMath::PutFloat(errorCI0, block, ParallelMath::Extract(he.errors[ci0[block]][px], block));
                    ParallelMath::PutFloat(errorCI1, block, ParallelMath::Extract(he.errors[ci1[block]][px], block));
                    ParallelMath::PutUInt16(signBits0, block, ParallelMath::Extract(he.signBits[ci0[block]], block));
                    ParallelMath::PutUInt16(signBits1, block, ParallelMath::Extract(he.signBits[ci1[block]], block));
                }

                totalError = totalError + ParallelMath::Min(errorCI0, errorCI1);

                MUInt16 bitPosition = ParallelMath::MakeUInt16(1 << px);

                ParallelMath::Int16CompFlag error1Better = ParallelMath::FloatFlagToInt16(ParallelMath::Less(errorCI1, errorCI0));

                sectorBits = sectorBits | ParallelMath::SelectOrZero(error1Better, bitPosition);
                signBits = signBits | (bitPosition & ParallelMath::Select(error1Better, signBits1, signBits0));
            }

            ParallelMath::FloatCompFlag totalErrorBetter = ParallelMath::Less(totalError, bestError);
            ParallelMath::Int16CompFlag totalErrorBetter16 = ParallelMath::FloatFlagToInt16(totalErrorBetter);
            if (ParallelMath::AnySet(totalErrorBetter16))
            {
                bestIsThisMode = bestIsThisMode | totalErrorBetter16;
                ParallelMath::ConditionalSet(bestTable, totalErrorBetter16, ParallelMath::MakeUInt15(table));
                ParallelMath::ConditionalSet(bestColors[0], totalErrorBetter16, color0);
                ParallelMath::ConditionalSet(bestColors[1], totalErrorBetter16, color1);
                ParallelMath::ConditionalSet(bestSectorBits, totalErrorBetter16, sectorBits);
                ParallelMath::ConditionalSet(bestSignBits, totalErrorBetter16, signBits);
                bestError = ParallelMath::Min(totalError, bestError);
            }
        }
    }

    if (ParallelMath::AnySet(bestIsThisMode))
    {
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            if (!ParallelMath::Extract(bestIsThisMode, block))
                continue;

            ParallelMath::ScalarUInt16 blockBestColors[2] = { ParallelMath::Extract(bestColors[0], block), ParallelMath::Extract(bestColors[1], block) };
            ParallelMath::ScalarUInt16 blockBestSectorBits = ParallelMath::Extract(bestSectorBits, block);
            ParallelMath::ScalarUInt16 blockBestSignBits = ParallelMath::Extract(bestSignBits, block);
            ParallelMath::ScalarUInt16 blockBestTable = ParallelMath::Extract(bestTable, block);

            EmitHModeBlock(outputBuffer + block * 8, blockBestColors, blockBestSectorBits, blockBestSignBits, blockBestTable, true);
        }
    }
}

void cvtt::Internal::ETCComputer::EncodeVirtualTModePunchthrough(uint8_t *outputBuffer, MFloat &bestError, const ParallelMath::Int16CompFlag isIsolatedBase[16], const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], const ParallelMath::Int16CompFlag isTransparent[16], const ParallelMath::Int16CompFlag& anyTransparent, const ParallelMath::Int16CompFlag& allTransparent, const Options &options)
{
    // We treat T and H mode as the same mode ("Virtual T mode") with punchthrough, because of how the colors work:
    //
    // T mode: C1, C2+M, Transparent, C2-M
    // H mode: C1+M, C1-M, Transparent, C2-M
    //
    // So in either case, we have 2 colors +/- a modifier, and a third unique color, which is basically T mode except without the middle color.
    // The only thing that matters is whether it's better to store the isolated color as T mode color 1, or store it offset in H mode color 2.
    //
    // Sometimes it won't even be possible to store it in H mode color 2 because the table low bit derives from a numeric comparison of the colors,
    // but unlike opaque blocks, we can't flip them.
    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);
    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);

    ParallelMath::FloatCompFlag isTransparentF[16];
    for (int px = 0; px < 16; px++)
        isTransparentF[px] = ParallelMath::Int16FlagToFloat(isTransparent[px]);

    ParallelMath::Int16CompFlag bestIsThisMode = ParallelMath::MakeBoolInt16(false);
    ParallelMath::Int16CompFlag bestIsHMode = ParallelMath::MakeBoolInt16(false);

    MUInt15 isolatedTotal[3] = { ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0) };
    MUInt15 lineTotal[3] = { ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0) };

    MUInt15 numPixelsIsolated = ParallelMath::MakeUInt15(0);
    MUInt15 numPixelsLine = ParallelMath::MakeUInt15(0);

    ParallelMath::Int16CompFlag isIsolated[16];
    ParallelMath::Int16CompFlag isLine[16];

    for (unsigned int px = 0; px < 16; px++)
    {
        ParallelMath::Int16CompFlag isOpaque = ParallelMath::Not(isTransparent[px]);
        isIsolated[px] = isIsolatedBase[px] & isOpaque;
        isLine[px] = ParallelMath::Not(isIsolatedBase[px]) & isOpaque;
    }

    for (unsigned int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 3; ch++)
        {
            isolatedTotal[ch] = isolatedTotal[ch] + ParallelMath::SelectOrZero(isIsolated[px], pixels[px][ch]);
            lineTotal[ch] = lineTotal[ch] + ParallelMath::SelectOrZero(isLine[px], pixels[px][ch]);
        }
        numPixelsIsolated = numPixelsIsolated + ParallelMath::SelectOrZero(isIsolated[px], ParallelMath::MakeUInt15(1));
        numPixelsLine = numPixelsLine + ParallelMath::SelectOrZero(isLine[px], ParallelMath::MakeUInt15(1));
    }

    MUInt15 isolatedAverageQuantized[3];
    MUInt15 hModeIsolatedQuantized[8][3];
    MUInt15 isolatedAverageTargets[3];
    {
        int divisors[ParallelMath::ParallelSize];
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            divisors[block] = ParallelMath::Extract(numPixelsIsolated, block) * 34;

        MUInt15 addend = (numPixelsIsolated << 4) | numPixelsIsolated;
        for (int ch = 0; ch < 3; ch++)
        {
            // isolatedAverageQuantized[ch] = (isolatedTotal[ch] * 2 + numPixelsIsolated * 17) / (numPixelsIsolated * 34);

            MUInt15 numerator = isolatedTotal[ch] + isolatedTotal[ch];
            if (!isFakeBT709)
                numerator = numerator + addend;

            MUInt15 hModeIsolatedNumerators[8];
            for (int table = 0; table < 8; table++)
            {
                // FIXME: Handle fake BT.709 correctly
                MUInt15 offsetTotal = isolatedTotal[ch] + ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply(ParallelMath::MakeUInt15(cvtt::Tables::ETC2::g_thModifierTable[table]), numPixelsIsolated));

                hModeIsolatedNumerators[table] = (offsetTotal + offsetTotal) + addend;
            }

            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                int divisor = divisors[block];
                if (divisor == 0)
                {
                    ParallelMath::PutUInt15(isolatedAverageQuantized[ch], block, 0);
                    for (int table = 0; table < 8; table++)
                        ParallelMath::PutUInt15(hModeIsolatedQuantized[table][ch], block, 0);
                }
                else
                {
                    ParallelMath::PutUInt15(isolatedAverageQuantized[ch], block, ParallelMath::Extract(numerator, block) / divisor);
                    for (int table = 0; table < 8; table++)
                        ParallelMath::PutUInt15(hModeIsolatedQuantized[table][ch], block, ParallelMath::Extract(hModeIsolatedNumerators[table], block) / divisor);
                }
            }

            isolatedAverageTargets[ch] = numerator;
        }
    }

    if (isFakeBT709)
        ResolveTHFakeBT709Rounding(isolatedAverageQuantized, isolatedAverageTargets, numPixelsIsolated);

    for (int table = 0; table < 8; table++)
        for (int ch = 0; ch < 3; ch++)
            hModeIsolatedQuantized[table][ch] = ParallelMath::Min(ParallelMath::MakeUInt15(15), hModeIsolatedQuantized[table][ch]);

    MUInt15 isolatedColor[3];
    for (int ch = 0; ch < 3; ch++)
        isolatedColor[ch] = (isolatedAverageQuantized[ch]) | (isolatedAverageQuantized[ch] << 4);

    MFloat isolatedError[16];
    for (int px = 0; px < 16; px++)
    {
        if (isFakeBT709)
            isolatedError[px] = ComputeErrorFakeBT709(isolatedColor, preWeightedPixels[px]);
        else if (isUniform)
            isolatedError[px] = ComputeErrorUniform(pixels[px], isolatedColor);
        else
            isolatedError[px] = ComputeErrorWeighted(isolatedColor, preWeightedPixels[px], options);

        ParallelMath::ConditionalSet(isolatedError[px], isTransparentF[px], ParallelMath::MakeFloatZero());
    }

    MSInt32 bestSelectors = ParallelMath::MakeSInt32(0);
    MUInt15 bestTable = ParallelMath::MakeUInt15(0);
    MUInt15 bestLineColor = ParallelMath::MakeUInt15(0);
    MUInt15 bestIsolatedColor = ParallelMath::MakeUInt15(0);
    MUInt15 bestHModeColor2 = ParallelMath::MakeUInt15(0);
    ParallelMath::Int16CompFlag bestUseHMode = ParallelMath::MakeBoolInt16(false);

    MSInt16 maxLine = ParallelMath::LosslessCast<MSInt16>::Cast(numPixelsLine);
    MSInt16 minLine = ParallelMath::MakeSInt16(0) - maxLine;

    int16_t clusterMaxLine = 0;
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        int16_t blockMaxLine = ParallelMath::Extract(maxLine, block);
        if (blockMaxLine > clusterMaxLine)
            clusterMaxLine = blockMaxLine;
    }

    int16_t clusterMinLine = -clusterMaxLine;

    int lineDivisors[ParallelMath::ParallelSize];
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
        lineDivisors[block] = ParallelMath::Extract(numPixelsLine, block) * 34;

    MUInt15 lineAddend = (numPixelsLine << 4) | numPixelsLine;

    for (int table = 0; table < 8; table++)
    {
        int numUniqueColors[ParallelMath::ParallelSize];
        MUInt15 uniqueQuantizedColors[31];

        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            numUniqueColors[block] = 0;

        MUInt15 modifier = ParallelMath::MakeUInt15(cvtt::Tables::ETC2::g_thModifierTable[table]);
        MUInt15 modifierOffset = (modifier + modifier);

        for (int16_t offsetPremultiplier = clusterMinLine; offsetPremultiplier <= clusterMaxLine; offsetPremultiplier += 2)
        {
            MSInt16 clampedOffsetPremultiplier = ParallelMath::Max(minLine, ParallelMath::Min(maxLine, ParallelMath::MakeSInt16(offsetPremultiplier)));
            MSInt16 modifierAddend = ParallelMath::CompactMultiply(clampedOffsetPremultiplier, modifierOffset);

            MUInt15 quantized[3];
            if (isFakeBT709)
            {
                MUInt15 targets[3];
                for (int ch = 0; ch < 3; ch++)
                {
                    //quantized[ch] = std::min<int16_t>(15, std::max(0, (lineTotal[ch] * 2 + modifierOffset * offsetPremultiplier)) / (numDAIILine * 34));
                    MUInt15 numerator = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(ParallelMath::MakeSInt16(0), ParallelMath::LosslessCast<MSInt16>::Cast(lineTotal[ch] + lineTotal[ch]) + modifierAddend));
                    MUInt15 divided = ParallelMath::MakeUInt15(0);
                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        int divisor = lineDivisors[block];
                        if (divisor == 0)
                            ParallelMath::PutUInt15(divided, block, 0);
                        else
                            ParallelMath::PutUInt15(divided, block, ParallelMath::Extract(numerator, block) / divisor);
                    }
                    quantized[ch] = ParallelMath::Min(ParallelMath::MakeUInt15(15), divided);
                    targets[ch] = numerator;
                }

                ResolveTHFakeBT709Rounding(quantized, targets, numPixelsLine);
            }
            else
            {
                for (int ch = 0; ch < 3; ch++)
                {
                    //quantized[ch] = std::min<int16_t>(15, std::max(0, (lineTotal[ch] * 2 + numDAIILine * 17 + modifierOffset * offsetPremultiplier)) / (numDAIILine * 34));
                    MUInt15 numerator = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(ParallelMath::MakeSInt16(0), ParallelMath::LosslessCast<MSInt16>::Cast(lineTotal[ch] + lineTotal[ch] + lineAddend) + modifierAddend));
                    MUInt15 divided = ParallelMath::MakeUInt15(0);
                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        int divisor = lineDivisors[block];
                        if (divisor == 0)
                            ParallelMath::PutUInt15(divided, block, 0);
                        else
                            ParallelMath::PutUInt15(divided, block, ParallelMath::Extract(numerator, block) / divisor);
                    }
                    quantized[ch] = ParallelMath::Min(ParallelMath::MakeUInt15(15), divided);
                }
            }

            MUInt15 packedColor = (quantized[0] << 10) | (quantized[1] << 5) | quantized[2];

            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                uint16_t blockPackedColor = ParallelMath::Extract(packedColor, block);
                if (numUniqueColors[block] == 0 || blockPackedColor != ParallelMath::Extract(uniqueQuantizedColors[numUniqueColors[block] - 1], block))
                    ParallelMath::PutUInt15(uniqueQuantizedColors[numUniqueColors[block]++], block, blockPackedColor);
            }
        }

        // Stripe unfilled unique colors
        int maxUniqueColors = 0;
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            if (numUniqueColors[block] > maxUniqueColors)
                maxUniqueColors = numUniqueColors[block];
        }

        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            uint16_t fillColor = ParallelMath::Extract(uniqueQuantizedColors[0], block);

            int numUnique = numUniqueColors[block];
            for (int fill = numUnique + 1; fill < maxUniqueColors; fill++)
                ParallelMath::PutUInt15(uniqueQuantizedColors[fill], block, fillColor);
        }

        MFloat hModeErrors[16];
        MUInt15 hModeUnquantizedColor[3];
        for (int ch = 0; ch < 3; ch++)
        {
            MUInt15 quantizedChannel = hModeIsolatedQuantized[table][ch];

            MUInt15 unquantizedCh = (quantizedChannel << 4) | quantizedChannel;
            hModeUnquantizedColor[ch] = ParallelMath::ToUInt15(ParallelMath::Max(ParallelMath::MakeSInt16(0), ParallelMath::LosslessCast<MSInt16>::Cast(unquantizedCh) - ParallelMath::LosslessCast<MSInt16>::Cast(modifier)));
        }

        for (int px = 0; px < 16; px++)
        {
            hModeErrors[px] = isUniform ? ComputeErrorUniform(hModeUnquantizedColor, pixels[px]) : ComputeErrorWeighted(hModeUnquantizedColor, preWeightedPixels[px], options);
            ParallelMath::ConditionalSet(hModeErrors[px], isTransparentF[px], ParallelMath::MakeFloatZero());
        }

        MUInt15 packedHModeColor2 = (hModeIsolatedQuantized[table][0] << 10) | (hModeIsolatedQuantized[table][1] << 5) | hModeIsolatedQuantized[table][2];
        ParallelMath::Int16CompFlag tableLowBitIsZero = ((table & 1) == 0) ? ParallelMath::MakeBoolInt16(true) : ParallelMath::MakeBoolInt16(false);

        for (int ci = 0; ci < maxUniqueColors; ci++)
        {
            MUInt15 lineColors[2][3];
            for (int ch = 0; ch < 3; ch++)
            {
                MUInt15 quantizedChannel = (ParallelMath::RightShift(uniqueQuantizedColors[ci], 10 - (ch * 5)) & ParallelMath::MakeUInt15(15));

                MUInt15 unquantizedColor = (quantizedChannel << 4) | quantizedChannel;
                lineColors[0][ch] = ParallelMath::Min(ParallelMath::MakeUInt15(255), unquantizedColor + modifier);
                lineColors[1][ch] = ParallelMath::ToUInt15(ParallelMath::Max(ParallelMath::MakeSInt16(0), ParallelMath::LosslessCast<MSInt16>::Cast(unquantizedColor) - ParallelMath::LosslessCast<MSInt16>::Cast(modifier)));
            }

            MUInt15 bestLineSelector[16];
            MFloat bestLineError[16];
            for (int px = 0; px < 16; px++)
            {
                MFloat lineErrors[2];
                for (int i = 0; i < 2; i++)
                    lineErrors[i] = isUniform ? ComputeErrorUniform(lineColors[i], pixels[px]) : ComputeErrorWeighted(lineColors[i], preWeightedPixels[px], options);

                ParallelMath::Int16CompFlag firstIsBetter = ParallelMath::FloatFlagToInt16(ParallelMath::LessOrEqual(lineErrors[0], lineErrors[1]));
                bestLineSelector[px] = ParallelMath::Select(firstIsBetter, ParallelMath::MakeUInt15(1), ParallelMath::MakeUInt15(3));
                bestLineError[px] = ParallelMath::Min(lineErrors[0], lineErrors[1]);

                ParallelMath::ConditionalSet(bestLineError[px], isTransparentF[px], ParallelMath::MakeFloatZero());
            }

            // One case considered here was if it was possible to force H mode to be valid when the line color is unused.
            // That case isn't actually useful because it's equivalent to the isolated color being unused at maximum offset,
            // which is always checked after a swap.
            MFloat tModeError = ParallelMath::MakeFloatZero();
            MFloat hModeError = ParallelMath::MakeFloatZero();
            for (int px = 0; px < 16; px++)
            {
                tModeError = tModeError + ParallelMath::Min(bestLineError[px], isolatedError[px]);
                hModeError = hModeError + ParallelMath::Min(bestLineError[px], hModeErrors[px]);
            }

            ParallelMath::FloatCompFlag hLessError = ParallelMath::Less(hModeError, tModeError);

            MUInt15 packedHModeColor1 = uniqueQuantizedColors[ci];

            ParallelMath::Int16CompFlag hModeTableLowBitMustBeZero = ParallelMath::Less(packedHModeColor1, packedHModeColor2);

            ParallelMath::Int16CompFlag hModeIsLegal = ParallelMath::Equal(hModeTableLowBitMustBeZero, tableLowBitIsZero);
            ParallelMath::Int16CompFlag useHMode = ParallelMath::FloatFlagToInt16(hLessError) & hModeIsLegal;

            MFloat roundBestError = tModeError;
            ParallelMath::ConditionalSet(roundBestError, ParallelMath::Int16FlagToFloat(useHMode), hModeError);

            ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(roundBestError, bestError));
            ParallelMath::FloatCompFlag useHModeF = ParallelMath::Int16FlagToFloat(useHMode);

            if (ParallelMath::AnySet(errorBetter))
            {
                MSInt32 selectors = ParallelMath::MakeSInt32(0);
                for (int px = 0; px < 16; px++)
                {
                    MUInt15 selector = bestLineSelector[px];

                    MFloat isolatedPixelError = ParallelMath::Select(useHModeF, hModeErrors[px], isolatedError[px]);
                    ParallelMath::Int16CompFlag isolatedBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(isolatedPixelError, bestLineError[px]));

                    ParallelMath::ConditionalSet(selector, isolatedBetter, ParallelMath::MakeUInt15(0));
                    ParallelMath::ConditionalSet(selector, isTransparent[px], ParallelMath::MakeUInt15(2));
                    selectors = selectors | (ParallelMath::ToInt32(selector) << (px * 2));
                }

                bestError = ParallelMath::Min(bestError, roundBestError);
                ParallelMath::ConditionalSet(bestLineColor, errorBetter, uniqueQuantizedColors[ci]);
                ParallelMath::ConditionalSet(bestSelectors, errorBetter, selectors);
                ParallelMath::ConditionalSet(bestTable, errorBetter, ParallelMath::MakeUInt15(table));
                ParallelMath::ConditionalSet(bestIsHMode, errorBetter, useHMode);
                ParallelMath::ConditionalSet(bestHModeColor2, errorBetter, packedHModeColor2);
                
                bestIsThisMode = bestIsThisMode | errorBetter;
            }
        }
    }

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        if (ParallelMath::Extract(bestIsThisMode, block))
        {
            uint32_t lowBits = 0;
            uint32_t highBits = 0;

            uint16_t blockBestLineColor = ParallelMath::Extract(bestLineColor, block);
            ParallelMath::ScalarUInt16 blockIsolatedAverageQuantized[3];

            for (int ch = 0; ch < 3; ch++)
                blockIsolatedAverageQuantized[ch] = ParallelMath::Extract(isolatedAverageQuantized[ch], block);

            uint16_t blockBestTable = ParallelMath::Extract(bestTable, block);
            int32_t blockBestSelectors = ParallelMath::Extract(bestSelectors, block);

            ParallelMath::ScalarUInt16 lineColor[3];
            for (int ch = 0; ch < 3; ch++)
                lineColor[ch] = (blockBestLineColor >> (10 - (ch * 5))) & 15;

            if (ParallelMath::Extract(bestIsHMode, block))
            {
                // T mode: C1, C2+M, Transparent, C2-M
                // H mode: C1+M, C1-M, Transparent, C2-M
                static const ParallelMath::ScalarUInt16 selectorRemapSector[4] = { 1, 0, 1, 0 };
                static const ParallelMath::ScalarUInt16 selectorRemapSign[4] = { 1, 0, 0, 1 };

                // Remap selectors
                ParallelMath::ScalarUInt16 signBits = 0;
                ParallelMath::ScalarUInt16 sectorBits = 0;
                int32_t blockBestSelectors = ParallelMath::Extract(bestSelectors, block);
                for (int px = 0; px < 16; px++)
                {
                    int32_t selector = (blockBestSelectors >> (px * 2)) & 3;
                    sectorBits |= (selectorRemapSector[selector] << px);
                    signBits |= (selectorRemapSign[selector] << px);
                }

                ParallelMath::ScalarUInt16 blockColors[2] = { blockBestLineColor, ParallelMath::Extract(bestHModeColor2, block) };

                EmitHModeBlock(outputBuffer + block * 8, blockColors, sectorBits, signBits, blockBestTable, false);
            }
            else
                EmitTModeBlock(outputBuffer + block * 8, lineColor, blockIsolatedAverageQuantized, blockBestSelectors, blockBestTable, false);
        }
    }
}


cvtt::ParallelMath::UInt15 cvtt::Internal::ETCComputer::DecodePlanarCoeff(const MUInt15 &coeff, int ch)
{
    if (ch == 1)
        return (coeff << 1) | (ParallelMath::RightShift(coeff, 6));
    else
        return (coeff << 2) | (ParallelMath::RightShift(coeff, 4));
}

void cvtt::Internal::ETCComputer::EncodePlanar(uint8_t *outputBuffer, MFloat &bestError, const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], const Options &options)
{
    // NOTE: If it's desired to do this in another color space, the best way to do it would probably be
    // to do everything in that color space and then transform it back to RGB.

    // We compute H = (H-O)/4 and V= (V-O)/4 to simplify the math

    // error = (x*H + y*V + O - C)^2
    MFloat h[3] = { ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero() };
    MFloat v[3] = { ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero() };
    MFloat o[3] = { ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero() };

    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);
    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);

    MFloat totalError = ParallelMath::MakeFloatZero();
    MUInt15 bestCoeffs[3][3];	// [Channel][Coeff]
    for (int ch = 0; ch < 3; ch++)
    {
        float fhh = 0.f;
        float fho = 0.f;
        float fhv = 0.f;
        float foo = 0.f;
        float fov = 0.f;
        float fvv = 0.f;
        MFloat fc = ParallelMath::MakeFloatZero();
        MFloat fh = ParallelMath::MakeFloatZero();
        MFloat fv = ParallelMath::MakeFloatZero();
        MFloat fo = ParallelMath::MakeFloatZero();

        float &foh = fho;
        float &fvh = fhv;
        float &fvo = fov;

        for (int px = 0; px < 16; px++)
        {
            float x = static_cast<float>(px % 4);
            float y = static_cast<float>(px / 4);
            MFloat c = isFakeBT709 ? preWeightedPixels[px][ch] : ParallelMath::ToFloat(pixels[px][ch]);

            // (x*H + y*V + O - C)^2
            fhh += x * x;
            fhv += x * y;
            fho += x;
            fh = fh - c * x;

            fvh += y * x;
            fvv += y * y;
            fvo += y;
            fv = fv - c * y;

            foh += x;
            fov += y;
            foo += 1;
            fo = fo - c;

            fh = fh - c * x;
            fv = fv - c * y;
            fo = fo - c;
            fc = fc + c * c;
        }

        //float totalError = fhh * h * h + fho * h*o + fhv * h*v + foo * o * o + fov * o*v + fvv * v * v + fh * h + fv * v + fo * o + fc;

        // error = fhh*h^2 + fho*h*o + fhv*h*v + foo*o^2 + fov*o*v + fvv*v^2 + fh*h + fv*v + fo*o + fc
        // derror/dh = 2*fhh*h + fho*o + fhv*v + fh
        // derror/dv = fhv*h + fov*o + 2*fvv*v + fv
        // derror/do = fho*h + 2*foo*o + fov*v + fo

        // Solve system of equations
        // h o v 1 = 0
        // -------
        // d e f g  R0
        // i j k l  R1
        // m n p q  R2

        float d = 2.0f * fhh;
        float e = fho;
        float f = fhv;
        MFloat gD = fh;

        float i = fhv;
        float j = fov;
        float k = 2.0f * fvv;
        MFloat lD = fv;

        float m = fho;
        float n = 2.0f * foo;
        float p = fov;
        MFloat qD = fo;

        {
            // Factor out first column from R1 and R2
            float r0to1 = -i / d;
            float r0to2 = -m / d;

            // 0 j1 k1 l1D
            float j1 = j + r0to1 * e;
            float k1 = k + r0to1 * f;
            MFloat l1D = lD + gD * r0to1;

            // 0 n1 p1 q1D
            float n1 = n + r0to2 * e;
            float p1 = p + r0to2 * f;
            MFloat q1D = qD + gD * r0to2;

            // Factor out third column from R2
            float r1to2 = -p1 / k1;

            // 0 n2 0 q2D
            float n2 = n1 + r1to2 * j1;
            MFloat q2D = q1D + l1D * r1to2;

            o[ch] = -q2D / n2;

            // Factor out second column from R1
            // 0 n2 0 q2D

            float r2to1 = -j1 / n2;

            // 0 0 k1 l2D
            // 0 n2 0 q2D
            MFloat l2D = l1D + q2D * r2to1;

            float elim2 = -f / k1;
            float elim1 = -e / n2;

            // d 0 0 g2D
            MFloat g2D = gD + l2D * elim2 + q2D * elim1;

            // n2*o + q2 = 0
            // o = -q2 / n2
            h[ch] = -g2D / d;
            v[ch] = -l2D / k1;
        }

        // Undo the local transformation
        h[ch] = h[ch] * 4.0f + o[ch];
        v[ch] = v[ch] * 4.0f + o[ch];
    }

    if (isFakeBT709)
    {
        MFloat oRGB[3];
        MFloat hRGB[3];
        MFloat vRGB[3];

        ConvertFromFakeBT709(oRGB, o);
        ConvertFromFakeBT709(hRGB, h);
        ConvertFromFakeBT709(vRGB, v);

        // Twiddling in fake BT.607 is a mess, just round off for now (the precision is pretty good anyway)
        {
            ParallelMath::RoundTowardNearestForScope rtn;

            for (int ch = 0; ch < 3; ch++)
            {
                MFloat fcoeffs[3] = { oRGB[ch], hRGB[ch], vRGB[ch] };

                for (int c = 0; c < 3; c++)
                {
                    MFloat coeff = ParallelMath::Max(ParallelMath::MakeFloatZero(), fcoeffs[c]);
                    if (ch == 1)
                        coeff = ParallelMath::Min(ParallelMath::MakeFloat(127.0f), coeff * (127.0f / 255.0f));
                    else
                        coeff = ParallelMath::Min(ParallelMath::MakeFloat(63.0f), coeff * (63.0f / 255.0f));
                    fcoeffs[c] = coeff;
                }

                for (int c = 0; c < 3; c++)
                    bestCoeffs[ch][c] = ParallelMath::RoundAndConvertToU15(fcoeffs[c], &rtn);
            }
        }

        MUInt15 reconstructed[16][3];
        for (int ch = 0; ch < 3; ch++)
        {
            MUInt15 dO = DecodePlanarCoeff(bestCoeffs[ch][0], ch);
            MUInt15 dH = DecodePlanarCoeff(bestCoeffs[ch][1], ch);
            MUInt15 dV = DecodePlanarCoeff(bestCoeffs[ch][2], ch);

            MSInt16 hMinusO = ParallelMath::LosslessCast<MSInt16>::Cast(dH) - ParallelMath::LosslessCast<MSInt16>::Cast(dO);
            MSInt16 vMinusO = ParallelMath::LosslessCast<MSInt16>::Cast(dV) - ParallelMath::LosslessCast<MSInt16>::Cast(dO);

            MFloat error = ParallelMath::MakeFloatZero();

            MSInt16 addend = ParallelMath::LosslessCast<MSInt16>::Cast(dO << 2) + 2;

            for (int px = 0; px < 16; px++)
            {
                MUInt15 pxv = ParallelMath::MakeUInt15(px);
                MSInt16 x = ParallelMath::LosslessCast<MSInt16>::Cast(pxv & ParallelMath::MakeUInt15(3));
                MSInt16 y = ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::RightShift(pxv, 2));

                MSInt16 interpolated = ParallelMath::RightShift(ParallelMath::CompactMultiply(x, hMinusO) + ParallelMath::CompactMultiply(y, vMinusO) + addend, 2);
                MUInt15 clampedLow = ParallelMath::ToUInt15(ParallelMath::Max(ParallelMath::MakeSInt16(0), interpolated));
                reconstructed[px][ch] = ParallelMath::Min(ParallelMath::MakeUInt15(255), clampedLow);
            }
        }

        totalError = ParallelMath::MakeFloatZero();
        for (int px = 0; px < 16; px++)
            totalError = totalError + ComputeErrorFakeBT709(reconstructed[px], preWeightedPixels[px]);
    }
    else
    {
        for (int ch = 0; ch < 3; ch++)
        {
            MFloat fcoeffs[3] = { o[ch], h[ch], v[ch] };
            MUInt15 coeffRanges[3][2];

            for (int c = 0; c < 3; c++)
            {
                MFloat coeff = ParallelMath::Max(ParallelMath::MakeFloatZero(), fcoeffs[c]);
                if (ch == 1)
                    coeff = ParallelMath::Min(ParallelMath::MakeFloat(127.0f), coeff * (127.0f / 255.0f));
                else
                    coeff = ParallelMath::Min(ParallelMath::MakeFloat(63.0f), coeff * (63.0f / 255.0f));
                fcoeffs[c] = coeff;
            }

            {
                ParallelMath::RoundDownForScope rd;
                for (int c = 0; c < 3; c++)
                    coeffRanges[c][0] = ParallelMath::RoundAndConvertToU15(fcoeffs[c], &rd);
            }

            {
                ParallelMath::RoundUpForScope ru;
                for (int c = 0; c < 3; c++)
                    coeffRanges[c][1] = ParallelMath::RoundAndConvertToU15(fcoeffs[c], &ru);
            }

            MFloat bestChannelError = ParallelMath::MakeFloat(FLT_MAX);
            for (int io = 0; io < 2; io++)
            {
                MUInt15 dO = DecodePlanarCoeff(coeffRanges[0][io], ch);

                for (int ih = 0; ih < 2; ih++)
                {
                    MUInt15 dH = DecodePlanarCoeff(coeffRanges[1][ih], ch);
                    MSInt16 hMinusO = ParallelMath::LosslessCast<MSInt16>::Cast(dH) - ParallelMath::LosslessCast<MSInt16>::Cast(dO);

                    for (int iv = 0; iv < 2; iv++)
                    {
                        MUInt15 dV = DecodePlanarCoeff(coeffRanges[2][iv], ch);
                        MSInt16 vMinusO = ParallelMath::LosslessCast<MSInt16>::Cast(dV) - ParallelMath::LosslessCast<MSInt16>::Cast(dO);

                        MFloat error = ParallelMath::MakeFloatZero();

                        MSInt16 addend = ParallelMath::LosslessCast<MSInt16>::Cast(dO << 2) + 2;

                        for (int px = 0; px < 16; px++)
                        {
                            MUInt15 pxv = ParallelMath::MakeUInt15(px);
                            MSInt16 x = ParallelMath::LosslessCast<MSInt16>::Cast(pxv & ParallelMath::MakeUInt15(3));
                            MSInt16 y = ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::RightShift(pxv, 2));

                            MSInt16 interpolated = ParallelMath::RightShift(ParallelMath::CompactMultiply(x, hMinusO) + ParallelMath::CompactMultiply(y, vMinusO) + addend, 2);
                            MUInt15 clampedLow = ParallelMath::ToUInt15(ParallelMath::Max(ParallelMath::MakeSInt16(0), interpolated));
                            MUInt15 dec = ParallelMath::Min(ParallelMath::MakeUInt15(255), clampedLow);

                            MSInt16 delta = ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][ch]) - ParallelMath::LosslessCast<MSInt16>::Cast(dec);

                            MFloat deltaF = ParallelMath::ToFloat(delta);
                            error = error + deltaF * deltaF;
                        }

                        ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, bestChannelError));
                        if (ParallelMath::AnySet(errorBetter))
                        {
                            bestChannelError = ParallelMath::Min(error, bestChannelError);
                            ParallelMath::ConditionalSet(bestCoeffs[ch][0], errorBetter, coeffRanges[0][io]);
                            ParallelMath::ConditionalSet(bestCoeffs[ch][1], errorBetter, coeffRanges[1][ih]);
                            ParallelMath::ConditionalSet(bestCoeffs[ch][2], errorBetter, coeffRanges[2][iv]);
                        }
                    }
                }
            }

            if (!isUniform)
            {
                switch (ch)
                {
                case 0:
                    bestChannelError = bestChannelError * (options.redWeight * options.redWeight);
                    break;
                case 1:
                    bestChannelError = bestChannelError * (options.greenWeight * options.greenWeight);
                    break;
                case 2:
                    bestChannelError = bestChannelError * (options.blueWeight * options.blueWeight);
                    break;
                default:
                    break;
                }
            }

            totalError = totalError + bestChannelError;
        }
    }

    ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(totalError, bestError));
    if (ParallelMath::AnySet(errorBetter))
    {
        bestError = ParallelMath::Min(bestError, totalError);

        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            if (!ParallelMath::Extract(errorBetter, block))
                continue;

            int ro = ParallelMath::Extract(bestCoeffs[0][0], block);
            int rh = ParallelMath::Extract(bestCoeffs[0][1], block);
            int rv = ParallelMath::Extract(bestCoeffs[0][2], block);

            int go = ParallelMath::Extract(bestCoeffs[1][0], block);
            int gh = ParallelMath::Extract(bestCoeffs[1][1], block);
            int gv = ParallelMath::Extract(bestCoeffs[1][2], block);

            int bo = ParallelMath::Extract(bestCoeffs[2][0], block);
            int bh = ParallelMath::Extract(bestCoeffs[2][1], block);
            int bv = ParallelMath::Extract(bestCoeffs[2][2], block);

            int go1 = go >> 6;
            int go2 = go & 63;

            int bo1 = bo >> 5;
            int bo2 = (bo >> 3) & 3;
            int bo3 = bo & 7;

            int rh1 = (rh >> 1);
            int rh2 = rh & 1;

            int fakeR = ro >> 2;
            int fakeDR = go1 | ((ro & 3) << 1);

            int fakeG = (go2 >> 2);
            int fakeDG = ((go2 & 3) << 1) | bo1;

            int fakeB = bo2;
            int fakeDB = bo3 >> 1;

            uint32_t highBits = 0;
            uint32_t lowBits = 0;

            // Avoid overflowing R
            if ((fakeDR & 4) != 0 && fakeR + fakeDR < 8)
                highBits |= 1 << (63 - 32);

            // Avoid overflowing G
            if ((fakeDG & 4) != 0 && fakeG + fakeDG < 8)
                highBits |= 1 << (55 - 32);

            // Overflow B
            if (fakeB + fakeDB < 4)
            {
                // Overflow low
                highBits |= 1 << (42 - 32);
            }
            else
            {
                // Overflow high
                highBits |= 7 << (45 - 32);
            }

            highBits |= ro << (57 - 32);
            highBits |= go1 << (56 - 32);
            highBits |= go2 << (49 - 32);
            highBits |= bo1 << (48 - 32);
            highBits |= bo2 << (43 - 32);
            highBits |= bo3 << (39 - 32);
            highBits |= rh1 << (34 - 32);
            highBits |= 1 << (33 - 32);
            highBits |= rh2 << (32 - 32);

            lowBits |= gh << 25;
            lowBits |= bh << 19;
            lowBits |= rv << 13;
            lowBits |= gv << 6;
            lowBits |= bv << 0;

            for (int i = 0; i < 4; i++)
                outputBuffer[block * 8 + i] = (highBits >> (24 - i * 8)) & 0xff;
            for (int i = 0; i < 4; i++)
                outputBuffer[block * 8 + i + 4] = (lowBits >> (24 - i * 8)) & 0xff;
        }
    }
}

void cvtt::Internal::ETCComputer::CompressETC2Block(uint8_t *outputBuffer, const PixelBlockU8 *pixelBlocks, ETC2CompressionData *compressionData, const Options &options, bool punchthroughAlpha)
{
    ParallelMath::Int16CompFlag pixelIsTransparent[16];
    ParallelMath::Int16CompFlag anyTransparent = ParallelMath::MakeBoolInt16(false);
    ParallelMath::Int16CompFlag allTransparent = ParallelMath::MakeBoolInt16(true);

    if (punchthroughAlpha)
    {
        const float fThreshold = std::max<float>(std::min<float>(1.0f, options.threshold), 0.0f) * 255.0f;

        // +1.0f is intentional, we want to take the next valid integer (even if it's 256) since everything else lower is transparent
        MUInt15 threshold = ParallelMath::MakeUInt15(static_cast<uint16_t>(std::floor(fThreshold + 1.0f)));

        for (int px = 0; px < 16; px++)
        {
            MUInt15 alpha;
            for (int block = 0; block < ParallelMath::ParallelSize; block++)
                ParallelMath::PutUInt15(alpha, block, pixelBlocks[block].m_pixels[px][3]);

            ParallelMath::Int16CompFlag isTransparent = ParallelMath::Less(alpha, threshold);
            anyTransparent = (anyTransparent | isTransparent);
            allTransparent = (allTransparent & isTransparent);
            pixelIsTransparent[px] = isTransparent;
        }
    }
    else
    {
        for (int px = 0; px < 16; px++)
            pixelIsTransparent[px] = ParallelMath::MakeBoolInt16(false);

        allTransparent = anyTransparent = ParallelMath::MakeBoolInt16(false);
    }

    MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);

    ETC2CompressionDataInternal* internalData = static_cast<ETC2CompressionDataInternal*>(compressionData);

    MUInt15 pixels[16][3];
    MFloat preWeightedPixels[16][3];
    ExtractBlocks(pixels, preWeightedPixels, pixelBlocks, options);

    if (ParallelMath::AnySet(anyTransparent))
    {
        for (int px = 0; px < 16; px++)
        {
            ParallelMath::Int16CompFlag flag = pixelIsTransparent[px];
            ParallelMath::FloatCompFlag fflag = ParallelMath::Int16FlagToFloat(flag);

            for (int ch = 0; ch < 3; ch++)
            {
                ParallelMath::ConditionalSet(pixels[px][ch], flag, ParallelMath::MakeUInt15(0));
                ParallelMath::ConditionalSet(preWeightedPixels[px][ch], fflag, ParallelMath::MakeFloat(0.0f));
            }
        }
    }

    if (!ParallelMath::AllSet(allTransparent))
        EncodePlanar(outputBuffer, bestError, pixels, preWeightedPixels, options);

    MFloat chromaDelta[16][2];

    MUInt15 numOpaque = ParallelMath::MakeUInt15(16);
    for (int px = 0; px < 16; px++)
        numOpaque = numOpaque - ParallelMath::SelectOrZero(pixelIsTransparent[px], ParallelMath::MakeUInt15(1));

    if (options.flags & cvtt::Flags::Uniform)
    {
        MSInt16 chromaCoordinates3[16][2];
        for (int px = 0; px < 16; px++)
        {
            chromaCoordinates3[px][0] = ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][0]) - ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][2]);
            chromaCoordinates3[px][1] = ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][0]) - ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][1] << 1) + ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][2]);
        }

        MSInt16 chromaCoordinateCentroid[2] = { ParallelMath::MakeSInt16(0), ParallelMath::MakeSInt16(0) };
        for (int px = 0; px < 16; px++)
        {
            for (int ch = 0; ch < 2; ch++)
                chromaCoordinateCentroid[ch] = chromaCoordinateCentroid[ch] + chromaCoordinates3[px][ch];
        }

        if (punchthroughAlpha)
        {
            for (int px = 0; px < 16; px++)
            {
                for (int ch = 0; ch < 2; ch++)
                {
                    MUInt15 chromaCoordinateMultiplied = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply(chromaCoordinates3[px][ch], numOpaque));
                    MSInt16 delta = ParallelMath::LosslessCast<MSInt16>::Cast(chromaCoordinateMultiplied) - chromaCoordinateCentroid[ch];
                    chromaDelta[px][ch] = ParallelMath::ToFloat(delta);
                }
            }
        }
        else
        {
            for (int px = 0; px < 16; px++)
            {
                for (int ch = 0; ch < 2; ch++)
                    chromaDelta[px][ch] = ParallelMath::ToFloat((chromaCoordinates3[px][ch] << 4) - chromaCoordinateCentroid[ch]);
            }
        }

        const MFloat rcpSqrt3 = ParallelMath::MakeFloat(0.57735026918962576450914878050196f);

        for (int px = 0; px < 16; px++)
            chromaDelta[px][1] = chromaDelta[px][1] * rcpSqrt3;
    }
    else
    {
        const float chromaAxis0[3] = { internalData->m_chromaSideAxis0[0], internalData->m_chromaSideAxis0[1], internalData->m_chromaSideAxis0[2] };
        const float chromaAxis1[3] = { internalData->m_chromaSideAxis1[0], internalData->m_chromaSideAxis1[1], internalData->m_chromaSideAxis1[2] };

        MFloat chromaCoordinates3[16][2];
        for (int px = 0; px < 16; px++)
        {
            const MFloat &px0 = preWeightedPixels[px][0];
            const MFloat &px1 = preWeightedPixels[px][1];
            const MFloat &px2 = preWeightedPixels[px][2];

            chromaCoordinates3[px][0] = px0 * chromaAxis0[0] + px1 * chromaAxis0[1] + px2 * chromaAxis0[2];
            chromaCoordinates3[px][1] = px0 * chromaAxis1[0] + px1 * chromaAxis1[1] + px2 * chromaAxis1[2];
        }

        MFloat chromaCoordinateCentroid[2] = { ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero() };
        for (int px = 0; px < 16; px++)
        {
            for (int ch = 0; ch < 2; ch++)
                chromaCoordinateCentroid[ch] = chromaCoordinateCentroid[ch] + chromaCoordinates3[px][ch];
        }

        if (punchthroughAlpha)
        {
            const MFloat numOpaqueF = ParallelMath::ToFloat(numOpaque);
            for (int px = 0; px < 16; px++)
            {
                for (int ch = 0; ch < 2; ch++)
                {
                    MFloat chromaCoordinateMultiplied = chromaCoordinates3[px][ch] * numOpaqueF;
                    MFloat delta = chromaCoordinateMultiplied - chromaCoordinateCentroid[ch];
                    chromaDelta[px][ch] = delta;
                }
            }
        }
        else
        {
            for (int px = 0; px < 16; px++)
            {
                for (int ch = 0; ch < 2; ch++)
                    chromaDelta[px][ch] = chromaCoordinates3[px][ch] * 16.0f - chromaCoordinateCentroid[ch];
            }
        }
    }


    MFloat covXX = ParallelMath::MakeFloatZero();
    MFloat covYY = ParallelMath::MakeFloatZero();
    MFloat covXY = ParallelMath::MakeFloatZero();

    for (int px = 0; px < 16; px++)
    {
        MFloat nx = chromaDelta[px][0];
        MFloat ny = chromaDelta[px][1];

        covXX = covXX + nx * nx;
        covYY = covYY + ny * ny;
        covXY = covXY + nx * ny;
    }

    MFloat halfTrace = (covXX + covYY) * 0.5f;
    MFloat det = covXX * covYY - covXY * covXY;

    MFloat mm = ParallelMath::Sqrt(ParallelMath::Max(ParallelMath::MakeFloatZero(), halfTrace * halfTrace - det));

    MFloat ev = halfTrace + mm;

    MFloat dx = (covYY - ev + covXY);
    MFloat dy = -(covXX - ev + covXY);

    // If evenly distributed, pick an arbitrary plane
    ParallelMath::FloatCompFlag allZero = ParallelMath::Equal(dx, ParallelMath::MakeFloatZero()) & ParallelMath::Equal(dy, ParallelMath::MakeFloatZero());
    ParallelMath::ConditionalSet(dx, allZero, ParallelMath::MakeFloat(1.f));

    ParallelMath::Int16CompFlag sectorAssignments[16];
    for (int px = 0; px < 16; px++)
        sectorAssignments[px] = ParallelMath::FloatFlagToInt16(ParallelMath::Less(chromaDelta[px][0] * dx + chromaDelta[px][1] * dy, ParallelMath::MakeFloatZero()));

    if (!ParallelMath::AllSet(allTransparent))
    {
        EncodeTMode(outputBuffer, bestError, sectorAssignments, pixels, preWeightedPixels, options);

        // Flip sector assignments
        for (int px = 0; px < 16; px++)
            sectorAssignments[px] = ParallelMath::Not(sectorAssignments[px]);

        EncodeTMode(outputBuffer, bestError, sectorAssignments, pixels, preWeightedPixels, options);

        EncodeHMode(outputBuffer, bestError, sectorAssignments, pixels, internalData->m_h, preWeightedPixels, options);

        CompressETC1BlockInternal(bestError, outputBuffer, pixels, preWeightedPixels, internalData->m_drs, options, true);
    }

    if (ParallelMath::AnySet(anyTransparent))
    {
        if (!ParallelMath::AllSet(allTransparent))
        {
            // Flip sector assignments
            for (int px = 0; px < 16; px++)
                sectorAssignments[px] = ParallelMath::Not(sectorAssignments[px]);
        }

        // Reset the error of any transparent blocks to max and retry with punchthrough modes
        ParallelMath::ConditionalSet(bestError, ParallelMath::Int16FlagToFloat(anyTransparent), ParallelMath::MakeFloat(FLT_MAX));

        EncodeVirtualTModePunchthrough(outputBuffer, bestError, sectorAssignments, pixels, preWeightedPixels, pixelIsTransparent, anyTransparent, allTransparent, options);

        // Flip sector assignments
        for (int px = 0; px < 16; px++)
            sectorAssignments[px] = ParallelMath::Not(sectorAssignments[px]);

        EncodeVirtualTModePunchthrough(outputBuffer, bestError, sectorAssignments, pixels, preWeightedPixels, pixelIsTransparent, anyTransparent, allTransparent, options);

        CompressETC1PunchthroughBlockInternal(bestError, outputBuffer, pixels, preWeightedPixels, pixelIsTransparent, static_cast<ETC2CompressionDataInternal*>(compressionData)->m_drs, options);
    }
}

void cvtt::Internal::ETCComputer::CompressETC2AlphaBlock(uint8_t *outputBuffer, const PixelBlockU8 *pixelBlocks, const Options &options)
{
    MUInt15 pixels[16];

    for (int px = 0; px < 16; px++)
    {
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            ParallelMath::PutUInt15(pixels[px], block, pixelBlocks[block].m_pixels[px][3]);
    }

    CompressETC2AlphaBlockInternal(outputBuffer, pixels, false, false, options);
}

void cvtt::Internal::ETCComputer::CompressETC2AlphaBlockInternal(uint8_t *outputBuffer, const MUInt15 pixels[16], bool is11Bit, bool isSigned, const Options &options)
{
    MUInt15 minAlpha = ParallelMath::MakeUInt15(is11Bit ? 2047 : 255);
    MUInt15 maxAlpha = ParallelMath::MakeUInt15(0);

    for (int px = 0; px < 16; px++)
    {
        minAlpha = ParallelMath::Min(minAlpha, pixels[px]);
        maxAlpha = ParallelMath::Max(maxAlpha, pixels[px]);
    }

    MUInt15 alphaSpan = maxAlpha - minAlpha;
    MUInt15 alphaSpanMidpointTimes2 = maxAlpha + minAlpha;

    MUInt31 bestTotalError = ParallelMath::MakeUInt31(0x7fffffff);
    MUInt15 bestTableIndex = ParallelMath::MakeUInt15(0);
    MUInt15 bestBaseCodeword = ParallelMath::MakeUInt15(0);
    MUInt15 bestMultiplier = ParallelMath::MakeUInt15(0);
    MUInt15 bestIndexes[16];

    for (int px = 0; px < 16; px++)
        bestIndexes[px] = ParallelMath::MakeUInt15(0);

    const int numAlphaRanges = 10;
    for (uint16_t tableIndex = 0; tableIndex < 16; tableIndex++)
    {
        for (int r = 0; r < numAlphaRanges; r++)
        {
            int subrange = r % 3;
            int mainRange = r / 3;

            int16_t maxOffset = Tables::ETC2::g_alphaModifierTablePositive[tableIndex][3 - mainRange - (subrange & 1)];
            int16_t minOffset = -Tables::ETC2::g_alphaModifierTablePositive[tableIndex][3 - mainRange - ((subrange >> 1) & 1)] - 1;
            uint16_t offsetSpan = static_cast<uint16_t>(maxOffset - minOffset);

            MSInt16 vminOffset = ParallelMath::MakeSInt16(minOffset);
            MUInt15 vmaxOffset = ParallelMath::MakeUInt15(maxOffset);
            MUInt15 voffsetSpan = ParallelMath::MakeUInt15(offsetSpan);

            MUInt15 minMultiplier = ParallelMath::MakeUInt15(0);
            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                uint16_t singleAlphaSpan = ParallelMath::Extract(alphaSpan, block);

                uint16_t lowMultiplier = singleAlphaSpan / offsetSpan;
                ParallelMath::PutUInt15(minMultiplier, block, lowMultiplier);
            }

            if (is11Bit)
            {
                // Clamps this to valid multipliers under 15 and rounds down to nearest multiple of 8
                minMultiplier = ParallelMath::Min(minMultiplier, ParallelMath::MakeUInt15(112)) & ParallelMath::MakeUInt15(120);
            }
            else
            {
                // We cap at 1 and 14 so both multipliers are valid and dividable
                // Cases where offset span is 0 should be caught by multiplier 1 of table 13
                minMultiplier = ParallelMath::Max(ParallelMath::Min(minMultiplier, ParallelMath::MakeUInt15(14)), ParallelMath::MakeUInt15(1));
            }

            for (uint16_t multiplierOffset = 0; multiplierOffset < 2; multiplierOffset++)
            {
                MUInt15 multiplier = minMultiplier;

                if (is11Bit)
                {
                    if (multiplierOffset == 1)
                        multiplier = multiplier + ParallelMath::MakeUInt15(8);
                    else
                        multiplier = ParallelMath::Max(multiplier, ParallelMath::MakeUInt15(1));
                }
                else
                {
                    if (multiplierOffset == 1)
                        multiplier = multiplier + ParallelMath::MakeUInt15(1);
                }

                MSInt16 multipliedMinOffset = ParallelMath::CompactMultiply(ParallelMath::LosslessCast<MSInt16>::Cast(multiplier), vminOffset);
                MUInt15 multipliedMaxOffset = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply(multiplier, vmaxOffset));

                // codeword = (maxOffset + minOffset + minAlpha + maxAlpha) / 2
                MSInt16 unclampedBaseAlphaTimes2 = ParallelMath::LosslessCast<MSInt16>::Cast(alphaSpanMidpointTimes2) - ParallelMath::LosslessCast<MSInt16>::Cast(multipliedMaxOffset) - multipliedMinOffset;

                MUInt15 baseAlpha;
                if (is11Bit)
                {
                    // In unsigned, 4 is added to the unquantized alpha, so compensating for that cancels the 4 we have to add to do rounding.
                    if (isSigned)
                        unclampedBaseAlphaTimes2 = unclampedBaseAlphaTimes2 + ParallelMath::MakeSInt16(8);

                    // -128 is illegal for some reason
                    MSInt16 minBaseAlphaTimes2 = isSigned ? ParallelMath::MakeSInt16(16) : ParallelMath::MakeSInt16(0);

                    MUInt15 clampedBaseAlphaTimes2 = ParallelMath::Min(ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(unclampedBaseAlphaTimes2, minBaseAlphaTimes2)), ParallelMath::MakeUInt15(4095));
                    baseAlpha = ParallelMath::RightShift(clampedBaseAlphaTimes2, 1) & ParallelMath::MakeUInt15(2040);

                    if (!isSigned)
                        baseAlpha = baseAlpha + ParallelMath::MakeUInt15(4);
                }
                else
                {
                    MUInt15 clampedBaseAlphaTimes2 = ParallelMath::Min(ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(unclampedBaseAlphaTimes2, ParallelMath::MakeSInt16(0))), ParallelMath::MakeUInt15(510));
                    baseAlpha = ParallelMath::RightShift(clampedBaseAlphaTimes2 + ParallelMath::MakeUInt15(1), 1);
                }

                MUInt15 indexes[16];
                MUInt31 totalError = ParallelMath::MakeUInt31(0);
                for (int px = 0; px < 16; px++)
                {
                    MUInt15 quantizedValues;
                    QuantizeETC2Alpha(tableIndex, pixels[px], baseAlpha, multiplier, is11Bit, isSigned, indexes[px], quantizedValues);

                    if (is11Bit)
                    {
                        MSInt16 delta = ParallelMath::LosslessCast<MSInt16>::Cast(quantizedValues) - ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px]);
                        MSInt32 deltaSq = ParallelMath::XMultiply(delta, delta);
                        totalError = totalError + ParallelMath::LosslessCast<MUInt31>::Cast(deltaSq);
                    }
                    else
                        totalError = totalError + ParallelMath::ToUInt31(ParallelMath::SqDiffUInt8(quantizedValues, pixels[px]));
                }

                ParallelMath::Int16CompFlag isBetter = ParallelMath::Int32FlagToInt16(ParallelMath::Less(totalError, bestTotalError));
                if (ParallelMath::AnySet(isBetter))
                {
                    ParallelMath::ConditionalSet(bestTotalError, isBetter, totalError);
                    ParallelMath::ConditionalSet(bestTableIndex, isBetter, ParallelMath::MakeUInt15(tableIndex));
                    ParallelMath::ConditionalSet(bestBaseCodeword, isBetter, baseAlpha);
                    ParallelMath::ConditionalSet(bestMultiplier, isBetter, multiplier);

                    for (int px = 0; px < 16; px++)
                        ParallelMath::ConditionalSet(bestIndexes[px], isBetter, indexes[px]);
                }

                // TODO: Do one refine pass
            }
        }
    }

    if (is11Bit)
    {
        bestMultiplier = ParallelMath::RightShift(bestMultiplier, 3);

        if (isSigned)
            bestBaseCodeword = bestBaseCodeword ^ ParallelMath::MakeUInt15(0x80);
    }

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        uint8_t *output = outputBuffer + block * 8;

        output[0] = static_cast<uint8_t>(ParallelMath::Extract(bestBaseCodeword, block));

        ParallelMath::ScalarUInt16 multiplier = ParallelMath::Extract(bestMultiplier, block);
        ParallelMath::ScalarUInt16 tableIndex = ParallelMath::Extract(bestTableIndex, block);

        output[1] = static_cast<uint8_t>((multiplier << 4) | tableIndex);

        static const int pixelSelectorOrder[16] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

        ParallelMath::ScalarUInt16 indexes[16];
        for (int px = 0; px < 16; px++)
            indexes[pixelSelectorOrder[px]] = ParallelMath::Extract(bestIndexes[px], block);

        int outputOffset = 2;
        int outputBits = 0;
        int numOutputBits = 0;
        for (int s = 0; s < 16; s++)
        {
            outputBits = (outputBits << 3) | indexes[s];
            numOutputBits += 3;

            if (numOutputBits >= 8)
            {
                output[outputOffset++] = static_cast<uint8_t>(outputBits >> (numOutputBits - 8));
                numOutputBits -= 8;

                outputBits &= ((1 << numOutputBits) - 1);
            }
        }

        assert(outputOffset == 8 && numOutputBits == 0);
    }
}

void cvtt::Internal::ETCComputer::CompressEACBlock(uint8_t *outputBuffer, const PixelBlockScalarS16 *inputBlocks, bool isSigned, const Options &options)
{
    MUInt15 pixels[16];
    for (int px = 0; px < 16; px++)
    {
        MSInt16 adjustedPixel;
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            ParallelMath::PutSInt16(adjustedPixel, block, inputBlocks[block].m_pixels[px]);

        // We use a slightly shifted range here so we can keep the unquantized base color in a UInt15
        // That is, signed range is 1..2047, and unsigned range is 0..2047
        if (isSigned)
        {
            adjustedPixel = ParallelMath::Min(adjustedPixel, ParallelMath::MakeSInt16(1023)) + ParallelMath::MakeSInt16(1024);
            adjustedPixel = ParallelMath::Max(ParallelMath::MakeSInt16(1), adjustedPixel);
        }
        else
        {
            adjustedPixel = ParallelMath::Min(adjustedPixel, ParallelMath::MakeSInt16(2047));
            adjustedPixel = ParallelMath::Max(ParallelMath::MakeSInt16(0), adjustedPixel);
        }


        pixels[px] = ParallelMath::LosslessCast<MUInt15>::Cast(adjustedPixel);
    }

    CompressETC2AlphaBlockInternal(outputBuffer, pixels, true, isSigned, options);
}

void cvtt::Internal::ETCComputer::CompressETC1Block(uint8_t *outputBuffer, const PixelBlockU8 *inputBlocks, ETC1CompressionData *compressionData, const Options &options)
{
    DifferentialResolveStorage &drs = static_cast<ETC1CompressionDataInternal*>(compressionData)->m_drs;
    MFloat bestTotalError = ParallelMath::MakeFloat(FLT_MAX);

    MUInt15 pixels[16][3];
    MFloat preWeightedPixels[16][3];
    ExtractBlocks(pixels, preWeightedPixels, inputBlocks, options);

    CompressETC1BlockInternal(bestTotalError, outputBuffer, pixels, preWeightedPixels, drs, options, false);
}

void cvtt::Internal::ETCComputer::ExtractBlocks(MUInt15 pixels[16][3], MFloat preWeightedPixels[16][3], const PixelBlockU8 *inputBlocks, const Options &options)
{
    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);
    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);

    for (int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 3; ch++)
        {
            for (int block = 0; block < ParallelMath::ParallelSize; block++)
                ParallelMath::PutUInt15(pixels[px][ch], block, inputBlocks[block].m_pixels[px][ch]);
        }

        if (isFakeBT709)
            ConvertToFakeBT709(preWeightedPixels[px], pixels[px]);
        else if (isUniform)
        {
            for (int ch = 0; ch < 3; ch++)
                preWeightedPixels[px][ch] = ParallelMath::ToFloat(pixels[px][ch]);
        }
        else
        {
            preWeightedPixels[px][0] = ParallelMath::ToFloat(pixels[px][0]) * options.redWeight;
            preWeightedPixels[px][1] = ParallelMath::ToFloat(pixels[px][1]) * options.greenWeight;
            preWeightedPixels[px][2] = ParallelMath::ToFloat(pixels[px][2]) * options.blueWeight;
        }
    }
}

void cvtt::Internal::ETCComputer::ResolveHalfBlockFakeBT709RoundingAccurate(MUInt15 quantized[3], const MUInt15 sectorCumulative[3], bool isDifferential)
{
    for (int ch = 0; ch < 3; ch++)
    {
        const MUInt15& cu15 = sectorCumulative[ch];

        if (isDifferential)
        {
            //quantized[ch] = (cu * 31 + (cu >> 3)) >> 11;
            quantized[ch] = ParallelMath::ToUInt15(
                ParallelMath::RightShift(
                (ParallelMath::LosslessCast<MUInt16>::Cast(cu15) << 5) - ParallelMath::LosslessCast<MUInt16>::Cast(cu15) + ParallelMath::LosslessCast<MUInt16>::Cast(ParallelMath::RightShift(cu15, 3))
                    , 11)
            );
        }
        else
        {
            //quantized[ch] = (cu * 30 + (cu >> 3)) >> 12;
            quantized[ch] = ParallelMath::ToUInt15(
                ParallelMath::RightShift(
                (ParallelMath::LosslessCast<MUInt16>::Cast(cu15) << 5) - ParallelMath::LosslessCast<MUInt16>::Cast(cu15 << 1) + ParallelMath::LosslessCast<MUInt16>::Cast(ParallelMath::RightShift(cu15, 3))
                    , 12)
            );
        }
    }

    MFloat lowOctantRGBFloat[3];
    MFloat highOctantRGBFloat[3];

    for (int ch = 0; ch < 3; ch++)
    {
        MUInt15 unquantized;
        MUInt15 unquantizedNext;
        if (isDifferential)
        {
            unquantized = (quantized[ch] << 3) | ParallelMath::RightShift(quantized[ch], 2);
            MUInt15 quantizedNext = ParallelMath::Min(ParallelMath::MakeUInt15(31), quantized[ch] + ParallelMath::MakeUInt15(1));
            unquantizedNext = (quantizedNext << 3) | ParallelMath::RightShift(quantizedNext, 2);
        }
        else
        {
            unquantized = (quantized[ch] << 4) | quantized[ch];
            unquantizedNext = ParallelMath::Min(ParallelMath::MakeUInt15(255), unquantized + ParallelMath::MakeUInt15(17));
        }
        lowOctantRGBFloat[ch] = ParallelMath::ToFloat(unquantized << 3);
        highOctantRGBFloat[ch] = ParallelMath::ToFloat(unquantizedNext << 3);
    }

    MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);
    MUInt15 bestOctant = ParallelMath::MakeUInt15(0);

    MFloat cumulativeYUV[3];
    ConvertToFakeBT709(cumulativeYUV, sectorCumulative);

    for (uint16_t octant = 0; octant < 8; octant++)
    {
        const MFloat &r = (octant & 1) ? highOctantRGBFloat[0] : lowOctantRGBFloat[0];
        const MFloat &g = (octant & 2) ? highOctantRGBFloat[1] : lowOctantRGBFloat[1];
        const MFloat &b = (octant & 4) ? highOctantRGBFloat[2] : lowOctantRGBFloat[2];

        MFloat octantYUV[3];
        ConvertToFakeBT709(octantYUV, r, g, b);

        MFloat delta[3];
        for (int ch = 0; ch < 3; ch++)
            delta[ch] = octantYUV[ch] - cumulativeYUV[ch];

        MFloat error = delta[0] * delta[0] + delta[1] + delta[1] + delta[2] * delta[2];
        ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, bestError));
        ParallelMath::ConditionalSet(bestOctant, errorBetter, ParallelMath::MakeUInt15(octant));
        bestError = ParallelMath::Min(error, bestError);
    }

    for (int ch = 0; ch < 3; ch++)
        quantized[ch] = quantized[ch] + (ParallelMath::RightShift(bestOctant, ch) & ParallelMath::MakeUInt15(1));
}

void cvtt::Internal::ETCComputer::ResolveHalfBlockFakeBT709RoundingFast(MUInt15 quantized[3], const MUInt15 sectorCumulative[3], bool isDifferential)
{
    // sectorCumulative range is 0..2040 (11 bits)
    MUInt15 roundingOffset = ParallelMath::MakeUInt15(0);

    MUInt15 rOffset;
    MUInt15 gOffset;
    MUInt15 bOffset;
    MUInt15 quantizedBase[3];
    MUInt15 upperBound;

    MUInt15 sectorCumulativeFillIn[3];
    for (int ch = 0; ch < 3; ch++)
        sectorCumulativeFillIn[ch] = sectorCumulative[ch] + ParallelMath::RightShift(sectorCumulative[ch], 8);

    if (isDifferential)
    {
        rOffset = (sectorCumulativeFillIn[0] << 6) & ParallelMath::MakeUInt15(0xf00);
        gOffset = (sectorCumulativeFillIn[1] << 4) & ParallelMath::MakeUInt15(0x0f0);
        bOffset = ParallelMath::RightShift(sectorCumulativeFillIn[2], 2) & ParallelMath::MakeUInt15(0x00f);

        for (int ch = 0; ch < 3; ch++)
            quantizedBase[ch] = ParallelMath::RightShift(sectorCumulativeFillIn[ch], 6);

        upperBound = ParallelMath::MakeUInt15(31);
    }
    else
    {
        rOffset = (sectorCumulativeFillIn[0] << 5) & ParallelMath::MakeUInt15(0xf00);
        gOffset = (sectorCumulativeFillIn[1] << 1) & ParallelMath::MakeUInt15(0x0f0);
        bOffset = ParallelMath::RightShift(sectorCumulativeFillIn[2], 3) & ParallelMath::MakeUInt15(0x00f);

        for (int ch = 0; ch < 3; ch++)
            quantizedBase[ch] = ParallelMath::RightShift(sectorCumulativeFillIn[ch], 7);

        upperBound = ParallelMath::MakeUInt15(15);
    }

    MUInt15 lookupIndex = (rOffset | gOffset | bOffset);

    MUInt15 octant;
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
        ParallelMath::PutUInt15(octant, block, Tables::FakeBT709::g_rounding16[ParallelMath::Extract(lookupIndex, block)]);

    quantizedBase[0] = quantizedBase[0] + (octant & ParallelMath::MakeUInt15(1));
    quantizedBase[1] = quantizedBase[1] + (ParallelMath::RightShift(octant, 1) & ParallelMath::MakeUInt15(1));
    quantizedBase[2] = quantizedBase[2] + (ParallelMath::RightShift(octant, 2) & ParallelMath::MakeUInt15(1));

    for (int ch = 0; ch < 3; ch++)
        quantized[ch] = ParallelMath::Min(quantizedBase[ch], upperBound);
}

void cvtt::Internal::ETCComputer::ResolveTHFakeBT709Rounding(MUInt15 quantized[3], const MUInt15 targets[3], const MUInt15 &granularity)
{
    MFloat lowOctantRGBFloat[3];
    MFloat highOctantRGBFloat[3];

    for (int ch = 0; ch < 3; ch++)
    {
        MUInt15 unquantized = (quantized[ch] << 4) | quantized[ch];
        MUInt15 unquantizedNext = ParallelMath::Min(ParallelMath::MakeUInt15(255), unquantized + ParallelMath::MakeUInt15(17));

        lowOctantRGBFloat[ch] = ParallelMath::ToFloat(ParallelMath::CompactMultiply(unquantized, granularity) << 1);
        highOctantRGBFloat[ch] = ParallelMath::ToFloat(ParallelMath::CompactMultiply(unquantizedNext, granularity) << 1);
    }

    MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);
    MUInt15 bestOctant = ParallelMath::MakeUInt15(0);

    MFloat cumulativeYUV[3];
    ConvertToFakeBT709(cumulativeYUV, ParallelMath::ToFloat(targets[0]), ParallelMath::ToFloat(targets[1]), ParallelMath::ToFloat(targets[2]));

    for (uint16_t octant = 0; octant < 8; octant++)
    {
        const MFloat &r = (octant & 1) ? highOctantRGBFloat[0] : lowOctantRGBFloat[0];
        const MFloat &g = (octant & 2) ? highOctantRGBFloat[1] : lowOctantRGBFloat[1];
        const MFloat &b = (octant & 4) ? highOctantRGBFloat[2] : lowOctantRGBFloat[2];

        MFloat octantYUV[3];
        ConvertToFakeBT709(octantYUV, r, g, b);

        MFloat delta[3];
        for (int ch = 0; ch < 3; ch++)
            delta[ch] = octantYUV[ch] - cumulativeYUV[ch];

        MFloat error = delta[0] * delta[0] + delta[1] + delta[1] + delta[2] * delta[2];
        ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, bestError));
        ParallelMath::ConditionalSet(bestOctant, errorBetter, ParallelMath::MakeUInt15(octant));
        bestError = ParallelMath::Min(error, bestError);
    }

    for (int ch = 0; ch < 3; ch++)
        quantized[ch] = quantized[ch] + (ParallelMath::RightShift(bestOctant, ch) & ParallelMath::MakeUInt15(1));
}

void cvtt::Internal::ETCComputer::ConvertToFakeBT709(MFloat yuv[3], const MUInt15 color[3])
{
    MFloat floatRGB[3];
    for (int ch = 0; ch < 3; ch++)
        floatRGB[ch] = ParallelMath::ToFloat(color[ch]);

    ConvertToFakeBT709(yuv, floatRGB);
}

void cvtt::Internal::ETCComputer::ConvertToFakeBT709(MFloat yuv[3], const MFloat color[3])
{
    ConvertToFakeBT709(yuv, color[0], color[1], color[2]);
}

void cvtt::Internal::ETCComputer::ConvertToFakeBT709(MFloat yuv[3], const MFloat &pr, const MFloat &pg, const MFloat &pb)
{
    MFloat r = pr;
    MFloat g = pg;
    MFloat b = pb;

    yuv[0] = r * 0.368233989135369f + g * 1.23876274963149f + b * 0.125054068802017f;
    yuv[1] = r * 0.5f - g * 0.4541529f - b * 0.04584709f;
    yuv[2] = r * -0.081014709086133f - g * 0.272538676238785f + b * 0.353553390593274f;
}

void cvtt::Internal::ETCComputer::ConvertFromFakeBT709(MFloat rgb[3], const MFloat yuv[3])
{
    MFloat yy = yuv[0] * 0.57735026466774571071f;
    MFloat u = yuv[1];
    MFloat v = yuv[2];

    rgb[0] = yy + u * 1.5748000207960953486f;
    rgb[1] = yy - u * 0.46812425854364753669f - v * 0.26491652528157560861f;
    rgb[2] = yy + v * 2.6242146882856944069f;
}


void cvtt::Internal::ETCComputer::QuantizeETC2Alpha(int tableIndex, const MUInt15& value, const MUInt15& baseValue, const MUInt15& multiplier, bool is11Bit, bool isSigned, MUInt15& outIndexes, MUInt15& outQuantizedValues)
{
    MSInt16 offset = ParallelMath::LosslessCast<MSInt16>::Cast(value) - ParallelMath::LosslessCast<MSInt16>::Cast(baseValue);
    MSInt16 offsetTimes2 = offset + offset;

    // ETC2's offset tables all have a reflect about 0.5*multiplier
    MSInt16 offsetAboutReflectorTimes2 = offsetTimes2 + ParallelMath::LosslessCast<MSInt16>::Cast(multiplier);

    MUInt15 absOffsetAboutReflectorTimes2 = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Abs(offsetAboutReflectorTimes2));
    MUInt15 lookupIndex = ParallelMath::RightShift(absOffsetAboutReflectorTimes2, 1);

    MUInt15 positiveIndex;
    MUInt15 positiveOffsetUnmultiplied;
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        uint16_t blockLookupIndex = ParallelMath::Extract(lookupIndex, block) / ParallelMath::Extract(multiplier, block);
        if (blockLookupIndex >= Tables::ETC2::g_alphaRoundingTableWidth)
            blockLookupIndex = Tables::ETC2::g_alphaRoundingTableWidth - 1;
        uint16_t index = Tables::ETC2::g_alphaRoundingTables[tableIndex][blockLookupIndex];
        ParallelMath::PutUInt15(positiveIndex, block, index);
        ParallelMath::PutUInt15(positiveOffsetUnmultiplied, block, Tables::ETC2::g_alphaModifierTablePositive[tableIndex][index]);

        // TODO: This is suboptimal when the offset is capped.  We should detect 0 and 255 values and always map them to the maximum offsets.
        // Doing that will also affect refinement though.
    }

    MSInt16 signBits = ParallelMath::RightShift(offsetAboutReflectorTimes2, 15);
    MSInt16 offsetUnmultiplied = ParallelMath::LosslessCast<MSInt16>::Cast(positiveOffsetUnmultiplied) ^ signBits;
    MSInt16 quantizedOffset = ParallelMath::CompactMultiply(offsetUnmultiplied, multiplier);

    MSInt16 offsetValue = ParallelMath::LosslessCast<MSInt16>::Cast(baseValue) + quantizedOffset;

    if (is11Bit)
    {
        if (isSigned)
            outQuantizedValues = ParallelMath::Min(ParallelMath::MakeUInt15(2047), ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(ParallelMath::MakeSInt16(1), offsetValue)));
        else
            outQuantizedValues = ParallelMath::Min(ParallelMath::MakeUInt15(2047), ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(ParallelMath::MakeSInt16(0), offsetValue)));
    }
    else
        outQuantizedValues = ParallelMath::Min(ParallelMath::MakeUInt15(255), ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(ParallelMath::MakeSInt16(0), offsetValue)));

    MUInt15 indexSub = ParallelMath::LosslessCast<MUInt15>::Cast(signBits) & ParallelMath::MakeUInt15(4);

    outIndexes = positiveIndex + ParallelMath::MakeUInt15(4) - indexSub;
}


void cvtt::Internal::ETCComputer::EmitTModeBlock(uint8_t *outputBuffer, const ParallelMath::ScalarUInt16 lineColor[3], const ParallelMath::ScalarUInt16 isolatedColor[3], int32_t packedSelectors, ParallelMath::ScalarUInt16 table, bool opaque)
{
    static const int selectorOrder[] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

    uint32_t lowBits = 0;
    uint32_t highBits = 0;

    int rh = ((isolatedColor[0] >> 2) & 3);
    int rl = (isolatedColor[0] & 3);

    if (rh + rl < 4)
    {
        // Overflow low
        highBits |= 1 << (58 - 32);
    }
    else
    {
        // Overflow high
        highBits |= 7 << (61 - 32);
    }

    highBits |= rh << (59 - 32);
    highBits |= rl << (56 - 32);
    highBits |= isolatedColor[1] << (52 - 32);
    highBits |= isolatedColor[2] << (48 - 32);
    highBits |= lineColor[0] << (44 - 32);
    highBits |= lineColor[1] << (40 - 32);
    highBits |= lineColor[2] << (36 - 32);
    highBits |= ((table >> 1) & 3) << (34 - 32);
    if (opaque)
        highBits |= 1 << (33 - 32);
    highBits |= (table & 1) << (32 - 32);

    for (int px = 0; px < 16; px++)
    {
        int sel = (packedSelectors >> (2 * selectorOrder[px])) & 3;
        if ((sel & 0x1) != 0)
            lowBits |= (1 << px);
        if ((sel & 0x2) != 0)
            lowBits |= (1 << (16 + px));
    }

    for (int i = 0; i < 4; i++)
        outputBuffer[i] = (highBits >> (24 - i * 8)) & 0xff;
    for (int i = 0; i < 4; i++)
        outputBuffer[i + 4] = (lowBits >> (24 - i * 8)) & 0xff;
}

void cvtt::Internal::ETCComputer::EmitHModeBlock(uint8_t *outputBuffer, const ParallelMath::ScalarUInt16 blockColors[2], ParallelMath::ScalarUInt16 sectorBits, ParallelMath::ScalarUInt16 signBits, ParallelMath::ScalarUInt16 table, bool opaque)
{
    if (blockColors[0] == blockColors[1])
    {
        // Base colors are the same.
        // If the table low bit isn't 1, then we can't encode this, because swapping the block colors will have no effect
        // on their order.
        // Instead, we encode this as T mode where all of the indexes are on the line.

        ParallelMath::ScalarUInt16 lineColor[3];
        ParallelMath::ScalarUInt16 isolatedColor[3];

        lineColor[0] = isolatedColor[0] = (blockColors[0] >> 10) & 0x1f;
        lineColor[1] = isolatedColor[1] = (blockColors[0] >> 5) & 0x1f;
        lineColor[2] = isolatedColor[2] = (blockColors[0] >> 0) & 0x1f;

        int32_t packedSelectors = 0x55555555;
        for (int px = 0; px < 16; px++)
            packedSelectors |= ((signBits >> px) & 1) << ((px * 2) + 1);

        EmitTModeBlock(outputBuffer, lineColor, isolatedColor, packedSelectors, table, opaque);
        return;
    }

    static const int selectorOrder[] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

    int16_t colors[2][3];
    for (int sector = 0; sector < 2; sector++)
    {
        for (int ch = 0; ch < 3; ch++)
            colors[sector][ch] = (blockColors[sector] >> ((2 - ch) * 5)) & 15;
    }

    uint32_t lowBits = 0;
    uint32_t highBits = 0;

    if (((table & 1) == 1) != (blockColors[0] > blockColors[1]))
    {
        for (int ch = 0; ch < 3; ch++)
            std::swap(colors[0][ch], colors[1][ch]);
        sectorBits ^= 0xffff;
    }

    int r1 = colors[0][0];
    int g1a = colors[0][1] >> 1;
    int g1b = (colors[0][1] & 1);
    int b1a = colors[0][2] >> 3;
    int b1b = colors[0][2] & 7;
    int r2 = colors[1][0];
    int g2 = colors[1][1];
    int b2 = colors[1][2];

    // Avoid overflowing R
    if ((g1a & 4) != 0 && r1 + g1a < 8)
        highBits |= 1 << (63 - 32);

    int fakeDG = b1b >> 1;
    int fakeG = b1a | (g1b << 1);

    if (fakeG + fakeDG < 4)
    {
        // Overflow low
        highBits |= 1 << (50 - 32);
    }
    else
    {
        // Overflow high
        highBits |= 7 << (53 - 32);
    }

    int da = (table >> 2) & 1;
    int db = (table >> 1) & 1;

    highBits |= r1 << (59 - 32);
    highBits |= g1a << (56 - 32);
    highBits |= g1b << (52 - 32);
    highBits |= b1a << (51 - 32);
    highBits |= b1b << (47 - 32);
    highBits |= r2 << (43 - 32);
    highBits |= g2 << (39 - 32);
    highBits |= b2 << (35 - 32);
    highBits |= da << (34 - 32);
    if (opaque)
        highBits |= 1 << (33 - 32);
    highBits |= db << (32 - 32);

    for (int px = 0; px < 16; px++)
    {
        int sectorBit = (sectorBits >> selectorOrder[px]) & 1;
        int signBit = (signBits >> selectorOrder[px]) & 1;

        lowBits |= (signBit << px);
        lowBits |= (sectorBit << (16 + px));
    }

    uint8_t *output = outputBuffer;

    for (int i = 0; i < 4; i++)
        output[i] = (highBits >> (24 - i * 8)) & 0xff;
    for (int i = 0; i < 4; i++)
        output[i + 4] = (lowBits >> (24 - i * 8)) & 0xff;
}

void cvtt::Internal::ETCComputer::EmitETC1Block(uint8_t *outputBuffer, int blockBestFlip, int blockBestD, const int blockBestColors[2][3], const int blockBestTables[2], const ParallelMath::ScalarUInt16 blockBestSelectors[2], bool transparent)
{
    uint32_t highBits = 0;
    uint32_t lowBits = 0;

    if (blockBestD == 0)
    {
        highBits |= blockBestColors[0][0] << 28;
        highBits |= blockBestColors[1][0] << 24;
        highBits |= blockBestColors[0][1] << 20;
        highBits |= blockBestColors[1][1] << 16;
        highBits |= blockBestColors[0][2] << 12;
        highBits |= blockBestColors[1][2] << 8;
    }
    else
    {
        highBits |= blockBestColors[0][0] << 27;
        highBits |= ((blockBestColors[1][0] - blockBestColors[0][0]) & 7) << 24;
        highBits |= blockBestColors[0][1] << 19;
        highBits |= ((blockBestColors[1][1] - blockBestColors[0][1]) & 7) << 16;
        highBits |= blockBestColors[0][2] << 11;
        highBits |= ((blockBestColors[1][2] - blockBestColors[0][2]) & 7) << 8;
    }

    highBits |= (blockBestTables[0] << 5);
    highBits |= (blockBestTables[1] << 2);
    if (!transparent)
        highBits |= (blockBestD << 1);
    highBits |= blockBestFlip;

    const uint8_t modifierCodes[4] = { 3, 2, 0, 1 };

    uint8_t unpackedSelectors[16];
    uint8_t unpackedSelectorCodes[16];
    for (int sector = 0; sector < 2; sector++)
    {
        int blockSectorBestSelectors = blockBestSelectors[sector];

        for (int px = 0; px < 8; px++)
        {
            int selector = (blockSectorBestSelectors >> (2 * px)) & 3;
            unpackedSelectorCodes[g_flipTables[blockBestFlip][sector][px]] = modifierCodes[selector];
            unpackedSelectors[g_flipTables[blockBestFlip][sector][px]] = selector;
        }
    }

    const int pixelSelectorOrder[16] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

    int lowBitOffset = 0;
    for (int sb = 0; sb < 2; sb++)
        for (int px = 0; px < 16; px++)
            lowBits |= ((unpackedSelectorCodes[pixelSelectorOrder[px]] >> sb) & 1) << (px + sb * 16);

    for (int i = 0; i < 4; i++)
        outputBuffer[i] = (highBits >> (24 - i * 8)) & 0xff;
    for (int i = 0; i < 4; i++)
        outputBuffer[i + 4] = (lowBits >> (24 - i * 8)) & 0xff;
}

void cvtt::Internal::ETCComputer::CompressETC1BlockInternal(MFloat &bestTotalError, uint8_t *outputBuffer, const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], DifferentialResolveStorage &drs, const Options &options, bool punchthrough)
{
	int numTries = 0;

    MUInt15 zeroU15 = ParallelMath::MakeUInt15(0);
    MUInt16 zeroU16 = ParallelMath::MakeUInt16(0);

    MUInt15 bestColors[2] = { zeroU15, zeroU15 };
    MUInt16 bestSelectors[2] = { zeroU16, zeroU16 };
    MUInt15 bestTables[2] = { zeroU15, zeroU15 };
    MUInt15 bestFlip = zeroU15;
    MUInt15 bestD = zeroU15;

    MUInt15 sectorPixels[2][2][8][3];
    MFloat sectorPreWeightedPixels[2][2][8][3];
    MUInt15 sectorCumulative[2][2][3];

    ParallelMath::Int16CompFlag bestIsThisMode = ParallelMath::MakeBoolInt16(false);

    for (int flip = 0; flip < 2; flip++)
	{
		for (int sector = 0; sector < 2; sector++)
		{
			for (int ch = 0; ch < 3; ch++)
				sectorCumulative[flip][sector][ch] = zeroU15;

			for (int px = 0; px < 8; px++)
			{
				for (int ch = 0; ch < 3; ch++)
				{
					MUInt15 pixelChannelValue = pixels[g_flipTables[flip][sector][px]][ch];
					sectorPixels[flip][sector][px][ch] = pixelChannelValue;
                    sectorPreWeightedPixels[flip][sector][px][ch] = preWeightedPixels[g_flipTables[flip][sector][px]][ch];
					sectorCumulative[flip][sector][ch] = sectorCumulative[flip][sector][ch] + pixelChannelValue;
				}
			}
		}
	}

	static const MSInt16 modifierTables[8][4] =
	{
		{ ParallelMath::MakeSInt16(-8), ParallelMath::MakeSInt16(-2), ParallelMath::MakeSInt16(2), ParallelMath::MakeSInt16(8) },
		{ ParallelMath::MakeSInt16(-17), ParallelMath::MakeSInt16(-5), ParallelMath::MakeSInt16(5), ParallelMath::MakeSInt16(17) },
		{ ParallelMath::MakeSInt16(-29), ParallelMath::MakeSInt16(-9), ParallelMath::MakeSInt16(9), ParallelMath::MakeSInt16(29) },
		{ ParallelMath::MakeSInt16(-42), ParallelMath::MakeSInt16(-13), ParallelMath::MakeSInt16(13), ParallelMath::MakeSInt16(42) },
		{ ParallelMath::MakeSInt16(-60), ParallelMath::MakeSInt16(-18), ParallelMath::MakeSInt16(18), ParallelMath::MakeSInt16(60) },
		{ ParallelMath::MakeSInt16(-80), ParallelMath::MakeSInt16(-24), ParallelMath::MakeSInt16(24), ParallelMath::MakeSInt16(80) },
		{ ParallelMath::MakeSInt16(-106), ParallelMath::MakeSInt16(-33), ParallelMath::MakeSInt16(33), ParallelMath::MakeSInt16(106) },
		{ ParallelMath::MakeSInt16(-183), ParallelMath::MakeSInt16(-47), ParallelMath::MakeSInt16(47), ParallelMath::MakeSInt16(183) },
	};

    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);

    int minD = punchthrough ? 1 : 0;

	for (int flip = 0; flip < 2; flip++)
	{
		drs.diffNumAttempts[0] = drs.diffNumAttempts[1] = zeroU15;

		MFloat bestIndError[2] = { ParallelMath::MakeFloat(FLT_MAX), ParallelMath::MakeFloat(FLT_MAX) };
		MUInt16 bestIndSelectors[2] = { ParallelMath::MakeUInt16(0), ParallelMath::MakeUInt16(0) };
		MUInt15 bestIndColors[2] = { zeroU15, zeroU15 };
		MUInt15 bestIndTable[2] = { zeroU15, zeroU15 };

		for (int d = minD; d < 2; d++)
		{
			for (int sector = 0; sector < 2; sector++)
			{
				const int16_t *potentialOffsets = cvtt::Tables::ETC1::g_potentialOffsets4;

				for (int table = 0; table < 8; table++)
				{
					int16_t numOffsets = *potentialOffsets++;

					MUInt15 possibleColors[cvtt::Tables::ETC1::g_maxPotentialOffsets];

                    MUInt15 quantized[3];
                    for (int oi = 0; oi < numOffsets; oi++)
                    {
                        if (!isFakeBT709)
                        {
						    for (int ch = 0; ch < 3; ch++)
						    {
                                // cu is in range 0..2040
                                MUInt15 cu15 = ParallelMath::Min(
                                    ParallelMath::MakeUInt15(2040),
                                    ParallelMath::ToUInt15(
                                        ParallelMath::Max(
                                            ParallelMath::MakeSInt16(0),
                                            ParallelMath::LosslessCast<MSInt16>::Cast(sectorCumulative[flip][sector][ch]) + ParallelMath::MakeSInt16(potentialOffsets[oi])
                                        )
                                    )
                                );

                                if (d == 1)
                                {
                                    //quantized[ch] = (cu * 31 + (cu >> 3) + 1024) >> 11;
                                    quantized[ch] = ParallelMath::ToUInt15(
                                        ParallelMath::RightShift(
                                            (ParallelMath::LosslessCast<MUInt16>::Cast(cu15) << 5) - ParallelMath::LosslessCast<MUInt16>::Cast(cu15) + ParallelMath::LosslessCast<MUInt16>::Cast(ParallelMath::RightShift(cu15, 3)) + ParallelMath::MakeUInt16(1024)
                                            , 11)
                                        );
                                }
                                else
                                {
                                    //quantized[ch] = (cu * 30 + (cu >> 3) + 2048) >> 12;
                                    quantized[ch] = ParallelMath::ToUInt15(
                                        ParallelMath::RightShift(
                                        (ParallelMath::LosslessCast<MUInt16>::Cast(cu15) << 5) - ParallelMath::LosslessCast<MUInt16>::Cast(cu15 << 1) + ParallelMath::LosslessCast<MUInt16>::Cast(ParallelMath::RightShift(cu15, 3)) + ParallelMath::MakeUInt16(2048)
                                            , 12)
                                    );
                                }
						    }
                        }
                        else
                        {
                            MUInt15 offsetCumulative[3];
						    for (int ch = 0; ch < 3; ch++)
						    {
                                // cu is in range 0..2040
                                MUInt15 cu15 = ParallelMath::Min(
                                    ParallelMath::MakeUInt15(2040),
                                    ParallelMath::ToUInt15(
                                        ParallelMath::Max(
                                            ParallelMath::MakeSInt16(0),
                                            ParallelMath::LosslessCast<MSInt16>::Cast(sectorCumulative[flip][sector][ch]) + ParallelMath::MakeSInt16(potentialOffsets[oi])
                                        )
                                    )
                                );

                                offsetCumulative[ch] = cu15;
						    }

                            if ((options.flags & cvtt::Flags::ETC_FakeBT709Accurate) != 0)
                                ResolveHalfBlockFakeBT709RoundingAccurate(quantized, offsetCumulative, d == 1);
                            else
                                ResolveHalfBlockFakeBT709RoundingFast(quantized, offsetCumulative, d == 1);
                        }

						possibleColors[oi] = quantized[0] | (quantized[1] << 5) | (quantized[2] << 10);
					}

					potentialOffsets += numOffsets;

                    ParallelMath::UInt15 numUniqueColors;
                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        uint16_t blockNumUniqueColors = 1;
                        for (int i = 1; i < numOffsets; i++)
                        {
                            uint16_t color = ParallelMath::Extract(possibleColors[i], block);
                            if (color != ParallelMath::Extract(possibleColors[blockNumUniqueColors - 1], block))
                                ParallelMath::PutUInt15(possibleColors[blockNumUniqueColors++], block, color);
                        }

                        ParallelMath::PutUInt15(numUniqueColors, block, blockNumUniqueColors);
                    }

                    int maxUniqueColors = ParallelMath::Extract(numUniqueColors, 0);
                    for (int block = 1; block < ParallelMath::ParallelSize; block++)
                        maxUniqueColors = std::max<int>(maxUniqueColors, ParallelMath::Extract(numUniqueColors, block));

                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        uint16_t fillColor = ParallelMath::Extract(possibleColors[0], block);
                        for (int i = ParallelMath::Extract(numUniqueColors, block); i < maxUniqueColors; i++)
                            ParallelMath::PutUInt15(possibleColors[i], block, fillColor);
                    }

					for (int i = 0; i < maxUniqueColors; i++)
					{
						MFloat error = ParallelMath::MakeFloatZero();
						MUInt16 selectors = ParallelMath::MakeUInt16(0);
                        MUInt15 quantized = possibleColors[i];
						TestHalfBlock(error, selectors, quantized, sectorPixels[flip][sector], sectorPreWeightedPixels[flip][sector], modifierTables[table], d == 1, options);

						if (d == 0)
						{
                            ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, bestIndError[sector]));
							if (ParallelMath::AnySet(errorBetter))
							{
								bestIndError[sector] = ParallelMath::Min(error, bestIndError[sector]);
								ParallelMath::ConditionalSet(bestIndSelectors[sector], errorBetter, selectors);
                                ParallelMath::ConditionalSet(bestIndColors[sector], errorBetter, quantized);
                                ParallelMath::ConditionalSet(bestIndTable[sector], errorBetter, ParallelMath::MakeUInt15(table));
							}
						}
						else
						{
                            ParallelMath::Int16CompFlag isInBounds = ParallelMath::Less(ParallelMath::MakeUInt15(i), numUniqueColors);

							MUInt15 storageIndexes = drs.diffNumAttempts[sector];
                            drs.diffNumAttempts[sector] = drs.diffNumAttempts[sector] + ParallelMath::SelectOrZero(isInBounds, ParallelMath::MakeUInt15(1));

                            for (int block = 0; block < ParallelMath::ParallelSize; block++)
                            {
                                int storageIndex = ParallelMath::Extract(storageIndexes, block);

                                ParallelMath::PutFloat(drs.diffErrors[sector][storageIndex], block, ParallelMath::Extract(error, block));
                                ParallelMath::PutUInt16(drs.diffSelectors[sector][storageIndex], block, ParallelMath::Extract(selectors, block));
                                ParallelMath::PutUInt15(drs.diffColors[sector][storageIndex], block, ParallelMath::Extract(quantized, block));
                                ParallelMath::PutUInt15(drs.diffTables[sector][storageIndex], block, table);
                            }
						}
					}
				}
			}

			if (d == 0)
			{
				MFloat bestIndErrorTotal = bestIndError[0] + bestIndError[1];
                ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(bestIndErrorTotal, bestTotalError));
				if (ParallelMath::AnySet(errorBetter))
				{
                    bestIsThisMode = bestIsThisMode | errorBetter;

					bestTotalError = ParallelMath::Min(bestTotalError, bestIndErrorTotal);
					ParallelMath::ConditionalSet(bestFlip, errorBetter, ParallelMath::MakeUInt15(flip));
                    ParallelMath::ConditionalSet(bestD, errorBetter, ParallelMath::MakeUInt15(d));
					for (int sector = 0; sector < 2; sector++)
					{
                        ParallelMath::ConditionalSet(bestColors[sector], errorBetter, bestIndColors[sector]);
                        ParallelMath::ConditionalSet(bestSelectors[sector], errorBetter, bestIndSelectors[sector]);
                        ParallelMath::ConditionalSet(bestTables[sector], errorBetter, bestIndTable[sector]);
					}
				}
			}
			else
			{
                ParallelMath::Int16CompFlag canIgnoreSector[2] = { ParallelMath::MakeBoolInt16(false), ParallelMath::MakeBoolInt16(false) };
                FindBestDifferentialCombination(flip, d, canIgnoreSector, bestIsThisMode, bestTotalError, bestFlip, bestD, bestColors, bestSelectors, bestTables, drs);
			}
		}
	}

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        if (!ParallelMath::Extract(bestIsThisMode, block))
            continue;

        uint32_t highBits = 0;
        uint32_t lowBits = 0;

        int blockBestFlip = ParallelMath::Extract(bestFlip, block);
        int blockBestD = ParallelMath::Extract(bestD, block);
        int blockBestTables[2] = { ParallelMath::Extract(bestTables[0], block), ParallelMath::Extract(bestTables[1], block) };
        ParallelMath::ScalarUInt16 blockBestSelectors[2] = { ParallelMath::Extract(bestSelectors[0], block), ParallelMath::Extract(bestSelectors[1], block) };

        int colors[2][3];
        for (int sector = 0; sector < 2; sector++)
        {
            int sectorColor = ParallelMath::Extract(bestColors[sector], block);
            for (int ch = 0; ch < 3; ch++)
                colors[sector][ch] = (sectorColor >> (ch * 5)) & 31;
        }

        EmitETC1Block(outputBuffer + block * 8, blockBestFlip, blockBestD, colors, blockBestTables, blockBestSelectors, false);
    }
}


void cvtt::Internal::ETCComputer::CompressETC1PunchthroughBlockInternal(MFloat &bestTotalError, uint8_t *outputBuffer, const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], const ParallelMath::Int16CompFlag isTransparent[16], DifferentialResolveStorage &drs, const Options &options)
{
	int numTries = 0;

    MUInt15 zeroU15 = ParallelMath::MakeUInt15(0);
    MUInt16 zeroU16 = ParallelMath::MakeUInt16(0);

    MUInt15 bestColors[2] = { zeroU15, zeroU15 };
    MUInt16 bestSelectors[2] = { zeroU16, zeroU16 };
    MUInt15 bestTables[2] = { zeroU15, zeroU15 };
    MUInt15 bestFlip = zeroU15;

    MUInt15 sectorPixels[2][2][8][3];
    ParallelMath::Int16CompFlag sectorTransparent[2][2][8];
    MFloat sectorPreWeightedPixels[2][2][8][3];
    MUInt15 sectorCumulative[2][2][3];

    ParallelMath::Int16CompFlag bestIsThisMode = ParallelMath::MakeBoolInt16(false);

    for (int flip = 0; flip < 2; flip++)
	{
		for (int sector = 0; sector < 2; sector++)
		{
			for (int ch = 0; ch < 3; ch++)
				sectorCumulative[flip][sector][ch] = zeroU15;

			for (int px = 0; px < 8; px++)
			{
				for (int ch = 0; ch < 3; ch++)
				{
					MUInt15 pixelChannelValue = pixels[g_flipTables[flip][sector][px]][ch];
					sectorPixels[flip][sector][px][ch] = pixelChannelValue;
                    sectorPreWeightedPixels[flip][sector][px][ch] = preWeightedPixels[g_flipTables[flip][sector][px]][ch];
					sectorCumulative[flip][sector][ch] = sectorCumulative[flip][sector][ch] + pixelChannelValue;
				}

                sectorTransparent[flip][sector][px] = isTransparent[g_flipTables[flip][sector][px]];
			}
		}
	}

	static const MUInt15 modifiers[8] =
	{
		ParallelMath::MakeUInt15(8),
		ParallelMath::MakeUInt15(17),
		ParallelMath::MakeUInt15(29),
		ParallelMath::MakeUInt15(42),
		ParallelMath::MakeUInt15(60),
		ParallelMath::MakeUInt15(80),
		ParallelMath::MakeUInt15(106),
		ParallelMath::MakeUInt15(183),
	};

    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);

    const int maxSectorCumulativeOffsets = 17;

	for (int flip = 0; flip < 2; flip++)
	{
        ParallelMath::Int16CompFlag canIgnoreSector[2] = { ParallelMath::MakeBoolInt16(true), ParallelMath::MakeBoolInt16(false) };

        for (int sector = 0; sector < 2; sector++)
            for (int px = 0; px < 8; px++)
                canIgnoreSector[sector] = canIgnoreSector[sector] & sectorTransparent[flip][sector][px];

		drs.diffNumAttempts[0] = drs.diffNumAttempts[1] = zeroU15;

		for (int sector = 0; sector < 2; sector++)
		{
            MUInt15 sectorNumOpaque = ParallelMath::MakeUInt15(0);
            for (int px = 0; px < 8; px++)
                sectorNumOpaque = sectorNumOpaque + ParallelMath::SelectOrZero(sectorTransparent[flip][sector][px], ParallelMath::MakeUInt15(1));

            int sectorMaxOpaque = 0;
            for (int block = 0; block < ParallelMath::ParallelSize; block++)
                sectorMaxOpaque = std::max<int>(sectorMaxOpaque, ParallelMath::Extract(sectorNumOpaque, block));

            int sectorNumOpaqueMultipliers = sectorMaxOpaque * 2 + 1;

            MUInt15 sectorNumOpaqueDenominator = ParallelMath::Max(ParallelMath::MakeUInt15(1), sectorNumOpaque) << 8;
            MUInt15 sectorNumOpaqueAddend = sectorNumOpaque << 7;

            MSInt16 sectorNumOpaqueSigned = ParallelMath::LosslessCast<MSInt16>::Cast(sectorNumOpaque);
            MSInt16 negSectorNumOpaqueSigned = ParallelMath::MakeSInt16(0) - sectorNumOpaqueSigned;

            MUInt15 sectorCumulativeMax = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply(ParallelMath::MakeUInt15(255), sectorNumOpaque));

			for (int table = 0; table < 8; table++)
			{
				MUInt15 possibleColors[maxSectorCumulativeOffsets];

                MUInt15 quantized[3];
                for (int om = -sectorMaxOpaque; om <= sectorMaxOpaque; om++)
                {
                    MSInt16 clampedOffsetMult = ParallelMath::Max(ParallelMath::Min(ParallelMath::MakeSInt16(om), sectorNumOpaqueSigned), negSectorNumOpaqueSigned);
                    MSInt16 offset = ParallelMath::CompactMultiply(clampedOffsetMult, modifiers[table]);

                    for (int ch = 0; ch < 3; ch++)
                    {
                        // cu is in range 0..255*numOpaque (at most 0..2040)
                        MUInt15 cu15 = ParallelMath::Min(
                            sectorCumulativeMax,
                            ParallelMath::ToUInt15(
                                ParallelMath::Max(
                                    ParallelMath::MakeSInt16(0),
                                    ParallelMath::LosslessCast<MSInt16>::Cast(sectorCumulative[flip][sector][ch]) + offset
                                )
                            )
                        );

                        //quantized[ch] = (cu * 31 + (cu >> 3) + (numOpaque * 128)) / (numOpaque * 256)
                        MUInt16 cuTimes31 = (ParallelMath::LosslessCast<MUInt16>::Cast(cu15) << 5) - ParallelMath::LosslessCast<MUInt16>::Cast(cu15);
                        MUInt15 cuDiv8 = ParallelMath::RightShift(cu15, 3);
                        MUInt16 numerator = cuTimes31 + ParallelMath::LosslessCast<MUInt16>::Cast(cuDiv8 + sectorNumOpaqueAddend);
                        for (int block = 0; block < ParallelMath::ParallelSize; block++)
                            ParallelMath::PutUInt15(quantized[ch], block, ParallelMath::Extract(numerator, block) / ParallelMath::Extract(sectorNumOpaqueDenominator, block));
                    }

					possibleColors[om + sectorMaxOpaque] = quantized[0] | (quantized[1] << 5) | (quantized[2] << 10);
				}

                ParallelMath::UInt15 numUniqueColors;
                for (int block = 0; block < ParallelMath::ParallelSize; block++)
                {
                    uint16_t blockNumUniqueColors = 1;
                    for (int i = 1; i < sectorNumOpaqueMultipliers; i++)
                    {
                        uint16_t color = ParallelMath::Extract(possibleColors[i], block);
                        if (color != ParallelMath::Extract(possibleColors[blockNumUniqueColors - 1], block))
                            ParallelMath::PutUInt15(possibleColors[blockNumUniqueColors++], block, color);
                    }

                    ParallelMath::PutUInt15(numUniqueColors, block, blockNumUniqueColors);
                }

                int maxUniqueColors = ParallelMath::Extract(numUniqueColors, 0);
                for (int block = 1; block < ParallelMath::ParallelSize; block++)
                    maxUniqueColors = std::max<int>(maxUniqueColors, ParallelMath::Extract(numUniqueColors, block));

                for (int block = 0; block < ParallelMath::ParallelSize; block++)
                {
                    uint16_t fillColor = ParallelMath::Extract(possibleColors[0], block);
                    for (int i = ParallelMath::Extract(numUniqueColors, block); i < maxUniqueColors; i++)
                        ParallelMath::PutUInt15(possibleColors[i], block, fillColor);
                }

				for (int i = 0; i < maxUniqueColors; i++)
				{
					MFloat error = ParallelMath::MakeFloatZero();
					MUInt16 selectors = ParallelMath::MakeUInt16(0);
                    MUInt15 quantized = possibleColors[i];
					TestHalfBlockPunchthrough(error, selectors, quantized, sectorPixels[flip][sector], sectorPreWeightedPixels[flip][sector], sectorTransparent[flip][sector], modifiers[table], options);

                    ParallelMath::Int16CompFlag isInBounds = ParallelMath::Less(ParallelMath::MakeUInt15(i), numUniqueColors);

					MUInt15 storageIndexes = drs.diffNumAttempts[sector];
                    drs.diffNumAttempts[sector] = drs.diffNumAttempts[sector] + ParallelMath::SelectOrZero(isInBounds, ParallelMath::MakeUInt15(1));

                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        int storageIndex = ParallelMath::Extract(storageIndexes, block);

                        ParallelMath::PutFloat(drs.diffErrors[sector][storageIndex], block, ParallelMath::Extract(error, block));
                        ParallelMath::PutUInt16(drs.diffSelectors[sector][storageIndex], block, ParallelMath::Extract(selectors, block));
                        ParallelMath::PutUInt15(drs.diffColors[sector][storageIndex], block, ParallelMath::Extract(quantized, block));
                        ParallelMath::PutUInt15(drs.diffTables[sector][storageIndex], block, table);
                    }
                }
            }
        }

        MUInt15 bestDDummy = ParallelMath::MakeUInt15(0);
        FindBestDifferentialCombination(flip, 1, canIgnoreSector, bestIsThisMode, bestTotalError, bestFlip, bestDDummy, bestColors, bestSelectors, bestTables, drs);
	}

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        if (!ParallelMath::Extract(bestIsThisMode, block))
            continue;

        int blockBestColors[2][3];
        int blockBestTables[2];
        ParallelMath::ScalarUInt16 blockBestSelectors[2];
        for (int sector = 0; sector < 2; sector++)
        {
            int sectorColor = ParallelMath::Extract(bestColors[sector], block);
            for (int ch = 0; ch < 3; ch++)
                blockBestColors[sector][ch] = (sectorColor >> (ch * 5)) & 31;

            blockBestTables[sector] = ParallelMath::Extract(bestTables[sector], block);
            blockBestSelectors[sector] = ParallelMath::Extract(bestSelectors[sector], block);
        }

        EmitETC1Block(outputBuffer + block * 8, ParallelMath::Extract(bestFlip, block), 1, blockBestColors, blockBestTables, blockBestSelectors, true);
    }
}


cvtt::ETC1CompressionData *cvtt::Internal::ETCComputer::AllocETC1Data(cvtt::Kernels::allocFunc_t allocFunc, void *context)
{
    void *buffer = allocFunc(context, sizeof(cvtt::Internal::ETCComputer::ETC1CompressionDataInternal));
    if (!buffer)
        return NULL;
    new (buffer) cvtt::Internal::ETCComputer::ETC1CompressionDataInternal(context);
    return static_cast<ETC1CompressionData*>(buffer);
}

void cvtt::Internal::ETCComputer::ReleaseETC1Data(ETC1CompressionData *compressionData, cvtt::Kernels::freeFunc_t freeFunc)
{
    cvtt::Internal::ETCComputer::ETC1CompressionDataInternal* internalData = static_cast<cvtt::Internal::ETCComputer::ETC1CompressionDataInternal*>(compressionData);
    void *context = internalData->m_context;
    internalData->~ETC1CompressionDataInternal();
    freeFunc(context, compressionData, sizeof(cvtt::Internal::ETCComputer::ETC1CompressionDataInternal));
}

cvtt::ETC2CompressionData *cvtt::Internal::ETCComputer::AllocETC2Data(cvtt::Kernels::allocFunc_t allocFunc, void *context, const cvtt::Options &options)
{
    void *buffer = allocFunc(context, sizeof(cvtt::Internal::ETCComputer::ETC2CompressionDataInternal));
    if (!buffer)
        return NULL;
    new (buffer) cvtt::Internal::ETCComputer::ETC2CompressionDataInternal(context, options);
    return static_cast<ETC2CompressionData*>(buffer);
}

void cvtt::Internal::ETCComputer::ReleaseETC2Data(ETC2CompressionData *compressionData, cvtt::Kernels::freeFunc_t freeFunc)
{
    cvtt::Internal::ETCComputer::ETC2CompressionDataInternal* internalData = static_cast<cvtt::Internal::ETCComputer::ETC2CompressionDataInternal*>(compressionData);
    void *context = internalData->m_context;
    internalData->~ETC2CompressionDataInternal();
    freeFunc(context, compressionData, sizeof(cvtt::Internal::ETCComputer::ETC2CompressionDataInternal));
}

cvtt::Internal::ETCComputer::ETC2CompressionDataInternal::ETC2CompressionDataInternal(void *context, const cvtt::Options &options)
    : m_context(context)
{
    const float cd[3] = { options.redWeight, options.greenWeight, options.blueWeight };
    const float rotCD[3] = { cd[1], cd[2], cd[0] };

    const float offs = -(rotCD[0] * cd[0] + rotCD[1] * cd[1] + rotCD[2] * cd[2]) / (cd[0] * cd[0] + cd[1] * cd[1] + cd[2] * cd[2]);

    const float chromaAxis0[3] = { rotCD[0] + cd[0] * offs, rotCD[1] + cd[1] * offs, rotCD[2] + cd[2] * offs };

    const float chromaAxis1Unnormalized[3] =
    {
        chromaAxis0[1] * cd[2] - chromaAxis0[2] * cd[1],
        chromaAxis0[2] * cd[0] - chromaAxis0[0] * cd[2],
        chromaAxis0[0] * cd[1] - chromaAxis0[1] * cd[0]
    };

    const float ca0LengthSq = (chromaAxis0[0] * chromaAxis0[0] + chromaAxis0[1] * chromaAxis0[1] + chromaAxis0[2] * chromaAxis0[2]);
    const float ca1UNLengthSq = (chromaAxis1Unnormalized[0] * chromaAxis1Unnormalized[0] + chromaAxis1Unnormalized[1] * chromaAxis1Unnormalized[1] + chromaAxis1Unnormalized[2] * chromaAxis1Unnormalized[2]);
    const float lengthRatio = static_cast<float>(std::sqrt(ca0LengthSq / ca1UNLengthSq));

    const float chromaAxis1[3] = { chromaAxis1Unnormalized[0] * lengthRatio, chromaAxis1Unnormalized[1] * lengthRatio, chromaAxis1Unnormalized[2] * lengthRatio };

    for (int i = 0; i < 3; i++)
    {
        m_chromaSideAxis0[i] = chromaAxis0[i];
        m_chromaSideAxis1[i] = chromaAxis1[i];
    }
}

#endif
