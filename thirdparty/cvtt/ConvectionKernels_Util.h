#pragma once

#include "ConvectionKernels_ParallelMath.h"

namespace cvtt
{
    struct PixelBlockU8;
    struct PixelBlockS8;
    struct Options;
}

namespace cvtt
{
    namespace Util
    {
        // Signed input blocks are converted into unsigned space, with the maximum value being 254
        void BiasSignedInput(PixelBlockU8 inputNormalized[ParallelMath::ParallelSize], const PixelBlockS8 inputSigned[ParallelMath::ParallelSize]);
        void FillWeights(const Options &options, float channelWeights[4]);
        void ComputeTweakFactors(int tweak, int range, float *outFactors);
    }
}
