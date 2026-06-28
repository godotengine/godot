// This file is part of the FidelityFX SDK.
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
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

// @internal

#pragma once

static const int FFX_FSR2_MAXIMUM_BIAS_TEXTURE_WIDTH = 16;
static const int FFX_FSR2_MAXIMUM_BIAS_TEXTURE_HEIGHT = 16;
static const float ffxFsr2MaximumBias[] = {
	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	1.876f,	1.809f,	1.772f,	1.753f,	1.748f,
	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	1.869f,	1.801f,	1.764f,	1.745f,	1.739f,
	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	1.976f,	1.841f,	1.774f,	1.737f,	1.716f,	1.71f,
	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	1.914f,	1.784f,	1.716f,	1.673f,	1.649f,	1.641f,
	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	1.793f,	1.676f,	1.604f,	1.562f,	1.54f,	1.533f,
	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	1.802f,	1.619f,	1.536f,	1.492f,	1.467f,	1.454f,	1.449f,
	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	1.812f,	1.575f,	1.496f,	1.456f,	1.432f,	1.416f,	1.408f,	1.405f,
	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	1.555f,	1.479f,	1.438f,	1.413f,	1.398f,	1.387f,	1.381f,	1.379f,
	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	1.812f,	1.555f,	1.474f,	1.43f,	1.404f,	1.387f,	1.376f,	1.368f,	1.363f,	1.362f,
	2.0f,	2.0f,	2.0f,	2.0f,	2.0f,	1.802f,	1.575f,	1.479f,	1.43f,	1.401f,	1.382f,	1.369f,	1.36f,	1.354f,	1.351f,	1.35f,
	2.0f,	2.0f,	1.976f,	1.914f,	1.793f,	1.619f,	1.496f,	1.438f,	1.404f,	1.382f,	1.367f,	1.357f,	1.349f,	1.344f,	1.341f,	1.34f,
	1.876f,	1.869f,	1.841f,	1.784f,	1.676f,	1.536f,	1.456f,	1.413f,	1.387f,	1.369f,	1.357f,	1.347f,	1.341f,	1.336f,	1.333f,	1.332f,
	1.809f,	1.801f,	1.774f,	1.716f,	1.604f,	1.492f,	1.432f,	1.398f,	1.376f,	1.36f,	1.349f,	1.341f,	1.335f,	1.33f,	1.328f,	1.327f,
	1.772f,	1.764f,	1.737f,	1.673f,	1.562f,	1.467f,	1.416f,	1.387f,	1.368f,	1.354f,	1.344f,	1.336f,	1.33f,	1.326f,	1.323f,	1.323f,
	1.753f,	1.745f,	1.716f,	1.649f,	1.54f,	1.454f,	1.408f,	1.381f,	1.363f,	1.351f,	1.341f,	1.333f,	1.328f,	1.323f,	1.321f,	1.32f,
	1.748f,	1.739f,	1.71f,	1.641f,	1.533f,	1.449f,	1.405f,	1.379f,	1.362f,	1.35f,	1.34f,	1.332f,	1.327f,	1.323f,	1.32f,	1.319f,

};
