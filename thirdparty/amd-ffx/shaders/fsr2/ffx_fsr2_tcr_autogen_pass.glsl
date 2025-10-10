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

//#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require

#define FSR2_BIND_SRV_INPUT_OPAQUE_ONLY                     0
#define FSR2_BIND_SRV_INPUT_COLOR                           1
#define FSR2_BIND_SRV_INPUT_MOTION_VECTORS                  2
#define FSR2_BIND_SRV_PREV_PRE_ALPHA_COLOR                  3
#define FSR2_BIND_SRV_PREV_POST_ALPHA_COLOR                 4
#define FSR2_BIND_SRV_REACTIVE_MASK                         5
#define FSR2_BIND_SRV_TRANSPARENCY_AND_COMPOSITION_MASK     6

#define FSR2_BIND_UAV_AUTOREACTIVE                       2007
#define FSR2_BIND_UAV_AUTOCOMPOSITION                    2008
#define FSR2_BIND_UAV_PREV_PRE_ALPHA_COLOR               2009
#define FSR2_BIND_UAV_PREV_POST_ALPHA_COLOR              2010

#define FSR2_BIND_CB_FSR2								 3000
#define FSR2_BIND_CB_AUTOREACTIVE                        3001

// GODOT BEGINS
#if FFX_FSR2_OPTION_GODOT_DERIVE_INVALID_MOTION_VECTORS
#define FSR2_BIND_SRV_INPUT_DEPTH                           13
#endif
// GODOT ENDS

#include "../../gpu/fsr2/ffx_fsr2_callbacks_glsl.h"
#include "../../gpu/fsr2/ffx_fsr2_common.h"
#include "../../gpu/fsr2/ffx_fsr2_tcr_autogen.h"

#ifndef FFX_FSR2_THREAD_GROUP_WIDTH
#define FFX_FSR2_THREAD_GROUP_WIDTH 8
#endif // #ifndef FFX_FSR2_THREAD_GROUP_WIDTH
#ifndef FFX_FSR2_THREAD_GROUP_HEIGHT
#define FFX_FSR2_THREAD_GROUP_HEIGHT 8
#endif // FFX_FSR2_THREAD_GROUP_HEIGHT
#ifndef FFX_FSR2_THREAD_GROUP_DEPTH
#define FFX_FSR2_THREAD_GROUP_DEPTH 1
#endif // #ifndef FFX_FSR2_THREAD_GROUP_DEPTH
#ifndef FFX_FSR2_NUM_THREADS
#define FFX_FSR2_NUM_THREADS layout (local_size_x = FFX_FSR2_THREAD_GROUP_WIDTH, local_size_y = FFX_FSR2_THREAD_GROUP_HEIGHT, local_size_z = FFX_FSR2_THREAD_GROUP_DEPTH) in;
#endif // #ifndef FFX_FSR2_NUM_THREADS

FFX_FSR2_NUM_THREADS
void main()
{
    FFX_MIN16_I2 uDispatchThreadId = FFX_MIN16_I2(gl_GlobalInvocationID.xy);

    // ToDo: take into account jitter (i.e. add delta of previous jitter and current jitter to previous UV
    // fetch pre- and post-alpha color values
    FFX_MIN16_F2 fUv = ( FFX_MIN16_F2(uDispatchThreadId) + FFX_MIN16_F2(0.5f, 0.5f) ) / FFX_MIN16_F2( RenderSize() );
    FFX_MIN16_F2 fPrevUV = fUv + FFX_MIN16_F2( LoadInputMotionVector(uDispatchThreadId) );
    FFX_MIN16_I2 iPrevIdx = FFX_MIN16_I2(fPrevUV * FFX_MIN16_F2(RenderSize()) - 0.5f);

    FFX_MIN16_F3 colorPreAlpha  = FFX_MIN16_F3( LoadOpaqueOnly( uDispatchThreadId ) );
    FFX_MIN16_F3 colorPostAlpha = FFX_MIN16_F3( LoadInputColor( uDispatchThreadId ) );

    FFX_MIN16_F2 outReactiveMask = FFX_MIN16_F2( 0.f, 0.f );

    outReactiveMask.y = ComputeTransparencyAndComposition(uDispatchThreadId, iPrevIdx);

    if (outReactiveMask.y > 0.5f)
    {
        outReactiveMask.x = ComputeReactive(uDispatchThreadId, iPrevIdx);
        outReactiveMask.x *= FFX_MIN16_F(ReactiveScale());
        outReactiveMask.x = outReactiveMask.x < ReactiveMax() ? outReactiveMask.x : FFX_MIN16_F( ReactiveMax() );
    }

    outReactiveMask.y *= FFX_MIN16_F(TcScale());

    outReactiveMask.x = ffxMax(outReactiveMask.x, FFX_MIN16_F(LoadReactiveMask(uDispatchThreadId)));
    outReactiveMask.y = ffxMax(outReactiveMask.y, FFX_MIN16_F(LoadTransparencyAndCompositionMask(uDispatchThreadId)));

    StoreAutoReactive(uDispatchThreadId, outReactiveMask);

    StorePrevPreAlpha(uDispatchThreadId, colorPreAlpha);
    StorePrevPostAlpha(uDispatchThreadId, colorPostAlpha);
}
