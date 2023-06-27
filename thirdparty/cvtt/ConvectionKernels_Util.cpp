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
#include "ConvectionKernels_ParallelMath.h"

#include <algorithm>

namespace cvtt
{
    namespace Util
    {
        // Signed input blocks are converted into unsigned space, with the maximum value being 254
        void BiasSignedInput(PixelBlockU8 inputNormalized[ParallelMath::ParallelSize], const PixelBlockS8 inputSigned[ParallelMath::ParallelSize])
        {
            for (size_t block = 0; block < ParallelMath::ParallelSize; block++)
            {
                const PixelBlockS8& inputSignedBlock = inputSigned[block];
                PixelBlockU8& inputNormalizedBlock = inputNormalized[block];

                for (size_t px = 0; px < 16; px++)
                {
                    for (size_t ch = 0; ch < 4; ch++)
                        inputNormalizedBlock.m_pixels[px][ch] = static_cast<uint8_t>(std::max<int>(inputSignedBlock.m_pixels[px][ch], -127) + 127);
                }
            }
        }

        void FillWeights(const Options &options, float channelWeights[4])
        {
            if (options.flags & Flags::Uniform)
                channelWeights[0] = channelWeights[1] = channelWeights[2] = channelWeights[3] = 1.0f;
            else
            {
                channelWeights[0] = options.redWeight;
                channelWeights[1] = options.greenWeight;
                channelWeights[2] = options.blueWeight;
                channelWeights[3] = options.alphaWeight;
            }
        }

        void ComputeTweakFactors(int tweak, int range, float *outFactors)
        {
            int totalUnits = range - 1;
            int minOutsideUnits = ((tweak >> 1) & 1);
            int maxOutsideUnits = (tweak & 1);
            int insideUnits = totalUnits - minOutsideUnits - maxOutsideUnits;

            outFactors[0] = -static_cast<float>(minOutsideUnits) / static_cast<float>(insideUnits);
            outFactors[1] = static_cast<float>(maxOutsideUnits) / static_cast<float>(insideUnits) + 1.0f;
        }
    }
}

#endif
