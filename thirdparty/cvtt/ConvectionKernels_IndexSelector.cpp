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

#include "ConvectionKernels_IndexSelector.h"

namespace cvtt
{
    namespace Internal
    {
        const ParallelMath::UInt16 g_weightReciprocals[17] =
        {
            ParallelMath::MakeUInt16(0),        // -1 
            ParallelMath::MakeUInt16(0),        // 0
            ParallelMath::MakeUInt16(32768),    // 1
            ParallelMath::MakeUInt16(16384),    // 2
            ParallelMath::MakeUInt16(10923),    // 3
            ParallelMath::MakeUInt16(8192),     // 4
            ParallelMath::MakeUInt16(6554),     // 5
            ParallelMath::MakeUInt16(5461),     // 6
            ParallelMath::MakeUInt16(4681),     // 7
            ParallelMath::MakeUInt16(4096),     // 8
            ParallelMath::MakeUInt16(3641),     // 9
            ParallelMath::MakeUInt16(3277),     // 10
            ParallelMath::MakeUInt16(2979),     // 11
            ParallelMath::MakeUInt16(2731),     // 12
            ParallelMath::MakeUInt16(2521),     // 13
            ParallelMath::MakeUInt16(2341),     // 14
            ParallelMath::MakeUInt16(2185),     // 15
        };
    }
}

#endif
