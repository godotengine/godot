// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FFX_OPTICALFLOW_FILTER_OPTICAL_FLOW_V5_H
#define FFX_OPTICALFLOW_FILTER_OPTICAL_FLOW_V5_H

void FilterOpticalFlow(FfxInt32x2 iGlobalId, FfxInt32x2 iLocalId, FfxInt32x2 iGroupId, FfxInt32 iLocalIndex)
{
    FfxInt32x2 tmpMV[9];
    FfxInt32 idx = 0;
    for (FfxInt32 xx = -1; xx < 2; xx++)
    {
        for (FfxInt32 yy = -1; yy < 2; yy++)
        {

            tmpMV[idx] = LoadPreviousOpticalFlow(iGlobalId + FfxInt32x2(xx, yy));
            idx++;
        }
    }

    FfxUInt32  ret = 0xFFFFFFFF;
    for (FfxInt32 i = 0; i < 9; ++i)
    {
        FfxUInt32 tmp = 0;
        for (FfxInt32 j = 0; j < 9; ++j)
        {
            FfxInt32x2 delta = tmpMV[i] - tmpMV[j];
            tmp = delta.x * delta.x + (delta.y * delta.y + tmp);
        }

        ret = min(((tmp) << 4) | i, ret);
    }

    FfxUInt32 minIdx = ret & 0xF;

    StoreOpticalFlow(iGlobalId, tmpMV[minIdx]);
}

#endif // FFX_OPTICALFLOW_FILTER_OPTICAL_FLOW_V5_H
