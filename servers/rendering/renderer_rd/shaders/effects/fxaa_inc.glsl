// FXAA 3.11 compact, Ported from https://github.com/kosua20/Rendu/blob/master/resources/common/shaders/screens/fxaa.frag
///////////////////////////////////////////////////////////////////////////////////
// MIT License
//
// Copyright (c) 2017 Simon Rodriguez
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////

// Nvidia Original FXAA 3.11 License
//----------------------------------------------------------------------------------
// File:        es3-kepler\FXAA/FXAA3_11.h
// SDK Version: v3.00
// Email:       gameworks@nvidia.com
// Site:        http://developer.nvidia.com/
//
// Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//----------------------------------------------------------------------------------
//
//                    NVIDIA FXAA 3.11 by TIMOTHY LOTTES
//
//----------------------------------------------------------------------------------

float QUALITY(float q) {
	return (q < 5 ? 1.0 : (q > 5 ? (q < 10 ? 2.0 : (q < 11 ? 4.0 : 8.0)) : 1.5));
}

float rgb2luma(vec3 rgb) {
	return sqrt(dot(rgb, vec3(0.299, 0.587, 0.114)));
}

vec3 do_fxaa(vec3 color, float exposure, vec2 uv_interp, float luminance_multiplier, vec2 pixel_size,
#ifdef USE_MULTIVIEW
		sampler2DArray source_color, int view_index
#else
		sampler2D source_color
#endif
) {
	const float EDGE_THRESHOLD_MIN = 0.0312;
	const float EDGE_THRESHOLD_MAX = 0.125;
	const int ITERATIONS = 12;
	const float SUBPIXEL_QUALITY = 0.75;

#ifdef USE_MULTIVIEW
	float lumaUp = rgb2luma(textureLodOffset(source_color, vec3(uv_interp, view_index), 0.0, ivec2(0, 1)).xyz * exposure * luminance_multiplier);
	float lumaDown = rgb2luma(textureLodOffset(source_color, vec3(uv_interp, view_index), 0.0, ivec2(0, -1)).xyz * exposure * luminance_multiplier);
	float lumaLeft = rgb2luma(textureLodOffset(source_color, vec3(uv_interp, view_index), 0.0, ivec2(-1, 0)).xyz * exposure * luminance_multiplier);
	float lumaRight = rgb2luma(textureLodOffset(source_color, vec3(uv_interp, view_index), 0.0, ivec2(1, 0)).xyz * exposure * luminance_multiplier);

	float lumaCenter = rgb2luma(color);

	float lumaMin = min(lumaCenter, min(min(lumaUp, lumaDown), min(lumaLeft, lumaRight)));
	float lumaMax = max(lumaCenter, max(max(lumaUp, lumaDown), max(lumaLeft, lumaRight)));

	float lumaRange = lumaMax - lumaMin;

	if (lumaRange < max(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD_MAX)) {
		return color;
	}

	float lumaDownLeft = rgb2luma(textureLodOffset(source_color, vec3(uv_interp, view_index), 0.0, ivec2(-1, -1)).xyz * exposure * luminance_multiplier);
	float lumaUpRight = rgb2luma(textureLodOffset(source_color, vec3(uv_interp, view_index), 0.0, ivec2(1, 1)).xyz * exposure * luminance_multiplier);
	float lumaUpLeft = rgb2luma(textureLodOffset(source_color, vec3(uv_interp, view_index), 0.0, ivec2(-1, 1)).xyz * exposure * luminance_multiplier);
	float lumaDownRight = rgb2luma(textureLodOffset(source_color, vec3(uv_interp, view_index), 0.0, ivec2(1, -1)).xyz * exposure * luminance_multiplier);

	float lumaDownUp = lumaDown + lumaUp;
	float lumaLeftRight = lumaLeft + lumaRight;

	float lumaLeftCorners = lumaDownLeft + lumaUpLeft;
	float lumaDownCorners = lumaDownLeft + lumaDownRight;
	float lumaRightCorners = lumaDownRight + lumaUpRight;
	float lumaUpCorners = lumaUpRight + lumaUpLeft;

	float edgeHorizontal = abs(-2.0 * lumaLeft + lumaLeftCorners) + abs(-2.0 * lumaCenter + lumaDownUp) * 2.0 + abs(-2.0 * lumaRight + lumaRightCorners);
	float edgeVertical = abs(-2.0 * lumaUp + lumaUpCorners) + abs(-2.0 * lumaCenter + lumaLeftRight) * 2.0 + abs(-2.0 * lumaDown + lumaDownCorners);

	bool isHorizontal = (edgeHorizontal >= edgeVertical);

	float stepLength = isHorizontal ? pixel_size.y : pixel_size.x;

	float luma1 = isHorizontal ? lumaDown : lumaLeft;
	float luma2 = isHorizontal ? lumaUp : lumaRight;
	float gradient1 = luma1 - lumaCenter;
	float gradient2 = luma2 - lumaCenter;

	bool is1Steepest = abs(gradient1) >= abs(gradient2);

	float gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

	float lumaLocalAverage = 0.0;
	if (is1Steepest) {
		stepLength = -stepLength;
		lumaLocalAverage = 0.5 * (luma1 + lumaCenter);
	} else {
		lumaLocalAverage = 0.5 * (luma2 + lumaCenter);
	}

	vec2 currentUv = uv_interp;
	if (isHorizontal) {
		currentUv.y += stepLength * 0.5;
	} else {
		currentUv.x += stepLength * 0.5;
	}

	vec2 offset = isHorizontal ? vec2(pixel_size.x, 0.0) : vec2(0.0, pixel_size.y);
	vec3 uv1 = vec3(currentUv - offset * QUALITY(0), view_index);
	vec3 uv2 = vec3(currentUv + offset * QUALITY(0), view_index);

	float lumaEnd1 = rgb2luma(textureLod(source_color, uv1, 0.0).xyz * exposure * luminance_multiplier);
	float lumaEnd2 = rgb2luma(textureLod(source_color, uv2, 0.0).xyz * exposure * luminance_multiplier);
	lumaEnd1 -= lumaLocalAverage;
	lumaEnd2 -= lumaLocalAverage;

	bool reached1 = abs(lumaEnd1) >= gradientScaled;
	bool reached2 = abs(lumaEnd2) >= gradientScaled;
	bool reachedBoth = reached1 && reached2;

	if (!reached1) {
		uv1 -= vec3(offset * QUALITY(1), 0.0);
	}
	if (!reached2) {
		uv2 += vec3(offset * QUALITY(1), 0.0);
	}

	if (!reachedBoth) {
		for (int i = 2; i < ITERATIONS; i++) {
			if (!reached1) {
				lumaEnd1 = rgb2luma(textureLod(source_color, uv1, 0.0).xyz * exposure * luminance_multiplier);
				lumaEnd1 = lumaEnd1 - lumaLocalAverage;
			}
			if (!reached2) {
				lumaEnd2 = rgb2luma(textureLod(source_color, uv2, 0.0).xyz * exposure * luminance_multiplier);
				lumaEnd2 = lumaEnd2 - lumaLocalAverage;
			}
			reached1 = abs(lumaEnd1) >= gradientScaled;
			reached2 = abs(lumaEnd2) >= gradientScaled;
			reachedBoth = reached1 && reached2;
			if (!reached1) {
				uv1 -= vec3(offset * QUALITY(i), 0.0);
			}
			if (!reached2) {
				uv2 += vec3(offset * QUALITY(i), 0.0);
			}
			if (reachedBoth) {
				break;
			}
		}
	}

	float distance1 = isHorizontal ? (uv_interp.x - uv1.x) : (uv_interp.y - uv1.y);
	float distance2 = isHorizontal ? (uv2.x - uv_interp.x) : (uv2.y - uv_interp.y);

	bool isDirection1 = distance1 < distance2;
	float distanceFinal = min(distance1, distance2);

	float edgeThickness = (distance1 + distance2);

	bool isLumaCenterSmaller = lumaCenter < lumaLocalAverage;

	bool correctVariation1 = (lumaEnd1 < 0.0) != isLumaCenterSmaller;
	bool correctVariation2 = (lumaEnd2 < 0.0) != isLumaCenterSmaller;

	bool correctVariation = isDirection1 ? correctVariation1 : correctVariation2;

	float pixelOffset = -distanceFinal / edgeThickness + 0.5;

	float finalOffset = correctVariation ? pixelOffset : 0.0;

	float lumaAverage = (1.0 / 12.0) * (2.0 * (lumaDownUp + lumaLeftRight) + lumaLeftCorners + lumaRightCorners);

	float subPixelOffset1 = clamp(abs(lumaAverage - lumaCenter) / lumaRange, 0.0, 1.0);
	float subPixelOffset2 = (-2.0 * subPixelOffset1 + 3.0) * subPixelOffset1 * subPixelOffset1;

	float subPixelOffsetFinal = subPixelOffset2 * subPixelOffset2 * SUBPIXEL_QUALITY;

	finalOffset = max(finalOffset, subPixelOffsetFinal);

	vec3 finalUv = vec3(uv_interp, view_index);
	if (isHorizontal) {
		finalUv.y += finalOffset * stepLength;
	} else {
		finalUv.x += finalOffset * stepLength;
	}

	vec3 finalColor = textureLod(source_color, finalUv, 0.0).xyz * exposure * luminance_multiplier;
	return finalColor;
#else
	float lumaUp = rgb2luma(textureLodOffset(source_color, uv_interp, 0.0, ivec2(0, 1)).xyz * exposure * luminance_multiplier);
	float lumaDown = rgb2luma(textureLodOffset(source_color, uv_interp, 0.0, ivec2(0, -1)).xyz * exposure * luminance_multiplier);
	float lumaLeft = rgb2luma(textureLodOffset(source_color, uv_interp, 0.0, ivec2(-1, 0)).xyz * exposure * luminance_multiplier);
	float lumaRight = rgb2luma(textureLodOffset(source_color, uv_interp, 0.0, ivec2(1, 0)).xyz * exposure * luminance_multiplier);

	float lumaCenter = rgb2luma(color);

	float lumaMin = min(lumaCenter, min(min(lumaUp, lumaDown), min(lumaLeft, lumaRight)));
	float lumaMax = max(lumaCenter, max(max(lumaUp, lumaDown), max(lumaLeft, lumaRight)));

	float lumaRange = lumaMax - lumaMin;

	if (lumaRange < max(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD_MAX)) {
		return color;
	}

	float lumaDownLeft = rgb2luma(textureLodOffset(source_color, uv_interp, 0.0, ivec2(-1, -1)).xyz * exposure * luminance_multiplier);
	float lumaUpRight = rgb2luma(textureLodOffset(source_color, uv_interp, 0.0, ivec2(1, 1)).xyz * exposure * luminance_multiplier);
	float lumaUpLeft = rgb2luma(textureLodOffset(source_color, uv_interp, 0.0, ivec2(-1, 1)).xyz * exposure * luminance_multiplier);
	float lumaDownRight = rgb2luma(textureLodOffset(source_color, uv_interp, 0.0, ivec2(1, -1)).xyz * exposure * luminance_multiplier);

	float lumaDownUp = lumaDown + lumaUp;
	float lumaLeftRight = lumaLeft + lumaRight;

	float lumaLeftCorners = lumaDownLeft + lumaUpLeft;
	float lumaDownCorners = lumaDownLeft + lumaDownRight;
	float lumaRightCorners = lumaDownRight + lumaUpRight;
	float lumaUpCorners = lumaUpRight + lumaUpLeft;

	float edgeHorizontal = abs(-2.0 * lumaLeft + lumaLeftCorners) + abs(-2.0 * lumaCenter + lumaDownUp) * 2.0 + abs(-2.0 * lumaRight + lumaRightCorners);
	float edgeVertical = abs(-2.0 * lumaUp + lumaUpCorners) + abs(-2.0 * lumaCenter + lumaLeftRight) * 2.0 + abs(-2.0 * lumaDown + lumaDownCorners);

	bool isHorizontal = (edgeHorizontal >= edgeVertical);

	float stepLength = isHorizontal ? pixel_size.y : pixel_size.x;

	float luma1 = isHorizontal ? lumaDown : lumaLeft;
	float luma2 = isHorizontal ? lumaUp : lumaRight;
	float gradient1 = luma1 - lumaCenter;
	float gradient2 = luma2 - lumaCenter;

	bool is1Steepest = abs(gradient1) >= abs(gradient2);

	float gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

	float lumaLocalAverage = 0.0;
	if (is1Steepest) {
		stepLength = -stepLength;
		lumaLocalAverage = 0.5 * (luma1 + lumaCenter);
	} else {
		lumaLocalAverage = 0.5 * (luma2 + lumaCenter);
	}

	vec2 currentUv = uv_interp;
	if (isHorizontal) {
		currentUv.y += stepLength * 0.5;
	} else {
		currentUv.x += stepLength * 0.5;
	}

	vec2 offset = isHorizontal ? vec2(pixel_size.x, 0.0) : vec2(0.0, pixel_size.y);
	vec2 uv1 = currentUv - offset * QUALITY(0);
	vec2 uv2 = currentUv + offset * QUALITY(0);

	float lumaEnd1 = rgb2luma(textureLod(source_color, uv1, 0.0).xyz * exposure * luminance_multiplier);
	float lumaEnd2 = rgb2luma(textureLod(source_color, uv2, 0.0).xyz * exposure * luminance_multiplier);
	lumaEnd1 -= lumaLocalAverage;
	lumaEnd2 -= lumaLocalAverage;

	bool reached1 = abs(lumaEnd1) >= gradientScaled;
	bool reached2 = abs(lumaEnd2) >= gradientScaled;
	bool reachedBoth = reached1 && reached2;

	if (!reached1) {
		uv1 -= offset * QUALITY(1);
	}
	if (!reached2) {
		uv2 += offset * QUALITY(1);
	}

	if (!reachedBoth) {
		for (int i = 2; i < ITERATIONS; i++) {
			if (!reached1) {
				lumaEnd1 = rgb2luma(textureLod(source_color, uv1, 0.0).xyz * exposure * luminance_multiplier);
				lumaEnd1 = lumaEnd1 - lumaLocalAverage;
			}
			if (!reached2) {
				lumaEnd2 = rgb2luma(textureLod(source_color, uv2, 0.0).xyz * exposure * luminance_multiplier);
				lumaEnd2 = lumaEnd2 - lumaLocalAverage;
			}
			reached1 = abs(lumaEnd1) >= gradientScaled;
			reached2 = abs(lumaEnd2) >= gradientScaled;
			reachedBoth = reached1 && reached2;
			if (!reached1) {
				uv1 -= offset * QUALITY(i);
			}
			if (!reached2) {
				uv2 += offset * QUALITY(i);
			}
			if (reachedBoth) {
				break;
			}
		}
	}

	float distance1 = isHorizontal ? (uv_interp.x - uv1.x) : (uv_interp.y - uv1.y);
	float distance2 = isHorizontal ? (uv2.x - uv_interp.x) : (uv2.y - uv_interp.y);

	bool isDirection1 = distance1 < distance2;
	float distanceFinal = min(distance1, distance2);

	float edgeThickness = (distance1 + distance2);

	bool isLumaCenterSmaller = lumaCenter < lumaLocalAverage;

	bool correctVariation1 = (lumaEnd1 < 0.0) != isLumaCenterSmaller;
	bool correctVariation2 = (lumaEnd2 < 0.0) != isLumaCenterSmaller;

	bool correctVariation = isDirection1 ? correctVariation1 : correctVariation2;

	float pixelOffset = -distanceFinal / edgeThickness + 0.5;

	float finalOffset = correctVariation ? pixelOffset : 0.0;

	float lumaAverage = (1.0 / 12.0) * (2.0 * (lumaDownUp + lumaLeftRight) + lumaLeftCorners + lumaRightCorners);

	float subPixelOffset1 = clamp(abs(lumaAverage - lumaCenter) / lumaRange, 0.0, 1.0);
	float subPixelOffset2 = (-2.0 * subPixelOffset1 + 3.0) * subPixelOffset1 * subPixelOffset1;

	float subPixelOffsetFinal = subPixelOffset2 * subPixelOffset2 * SUBPIXEL_QUALITY;

	finalOffset = max(finalOffset, subPixelOffsetFinal);

	vec2 finalUv = uv_interp;
	if (isHorizontal) {
		finalUv.y += finalOffset * stepLength;
	} else {
		finalUv.x += finalOffset * stepLength;
	}

	vec3 finalColor = textureLod(source_color, finalUv, 0.0).xyz * exposure * luminance_multiplier;
	return finalColor;
#endif
}
