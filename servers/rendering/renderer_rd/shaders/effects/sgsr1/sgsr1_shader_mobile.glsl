#[vertex]

#version 450

const vec2 positions[3] = vec2[](
		vec2(-1.0, -1.0),
		vec2(3.0, -1.0),
		vec2(-1.0, 3.0));

const vec2 uvs[3] = vec2[](
		vec2(0.0, 0.0),
		vec2(2.0, 0.0),
		vec2(0.0, 2.0));

layout(location = 0) out vec2 out_uv;

void main() {
	gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
	out_uv = uvs[gl_VertexIndex];
}

#[fragment]

#version 450

//============================================================================================================
//
//
//                  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

////////////////////////
// USER CONFIGURATION //
////////////////////////

/*
 * Operation modes:
 * RGBA -> 1
 * RGBY -> 3
 * LERP -> 4
 */
#define OperationMode 1

/*
 * If set, will use edge direction to improve visual quality
 * Expect a minimal cost increase
 */
#define UseEdgeDirection

#define EdgeThreshold 8.0 / 255.0

#define EdgeSharpness 1.0

#define UseUniformBlock

////////////////////////
////////////////////////
////////////////////////

#if defined(UseUniformBlock)
layout(binding = 0) uniform UniformBlock {
	vec4 ViewportInfo[1];
};
layout(binding = 1) uniform sampler2D ps0;
#else
uniform vec4 ViewportInfo[1];
uniform sampler2D ps0;
#endif

layout(location = 0) in vec4 in_TEXCOORD0;
layout(location = 0) out vec4 out_Target0;

float fastLanczos2(float x) {
	float wA = x - 4.0;
	float wB = x * wA - wA;
	wA *= wA;
	return wB * wA;
}

#if defined(UseEdgeDirection)
vec2 weightY(float dx, float dy, float c, vec3 data)
#else
vec2 weightY(float dx, float dy, float c, float data)
#endif
{
#if defined(UseEdgeDirection)
	float std = data.x;
	vec2 dir = data.yz;

	float edgeDis = ((dx * dir.y) + (dy * dir.x));
	float x = (((dx * dx) + (dy * dy)) + ((edgeDis * edgeDis) * ((clamp(((c * c) * std), 0.0, 1.0) * 0.7) + -1.0)));
#else
	float std = data;
	float x = ((dx * dx) + (dy * dy)) * 0.55 + clamp(abs(c) * std, 0.0, 1.0);
#endif

	float w = fastLanczos2(x);
	return vec2(w, w * c);
}

vec2 edgeDirection(vec4 left, vec4 right) {
	vec2 dir;
	float RxLz = (right.x + (-left.z));
	float RwLy = (right.w + (-left.y));
	vec2 delta;
	delta.x = (RxLz + RwLy);
	delta.y = (RxLz + (-RwLy));
	float lengthInv = inversesqrt((delta.x * delta.x + 3.075740e-05) + (delta.y * delta.y));
	dir.x = (delta.x * lengthInv);
	dir.y = (delta.y * lengthInv);
	return dir;
}

void main() {
	vec4 color;
	if (OperationMode == 1) {
		color.xyz = textureLod(ps0, in_TEXCOORD0.xy, 0.0).xyz;
	} else {
		color.xyzw = textureLod(ps0, in_TEXCOORD0.xy, 0.0).xyzw;
	}

	float xCenter;
	xCenter = abs(in_TEXCOORD0.x + -0.5);
	float yCenter;
	yCenter = abs(in_TEXCOORD0.y + -0.5);

	//todo: config the SR region based on needs
	//if ( OperationMode!=4 && xCenter*xCenter+yCenter*yCenter<=0.4 * 0.4)
	if (OperationMode != 4) {
		vec2 imgCoord = ((in_TEXCOORD0.xy * ViewportInfo[0].zw) + vec2(-0.5, 0.5));
		vec2 imgCoordPixel = floor(imgCoord);
		vec2 coord = (imgCoordPixel * ViewportInfo[0].xy);
		vec2 pl = (imgCoord + (-imgCoordPixel));
		vec4 left = textureGather(ps0, coord, OperationMode);

		float edgeVote = abs(left.z - left.y) + abs(color[OperationMode] - left.y) + abs(color[OperationMode] - left.z);
		if (edgeVote > EdgeThreshold) {
			coord.x += ViewportInfo[0].x;

			vec4 right = textureGather(ps0, coord + vec2(ViewportInfo[0].x, 0.0), OperationMode);
			vec4 upDown;
			upDown.xy = textureGather(ps0, coord + vec2(0.0, -ViewportInfo[0].y), OperationMode).wz;
			upDown.zw = textureGather(ps0, coord + vec2(0.0, ViewportInfo[0].y), OperationMode).yx;

			float mean = (left.y + left.z + right.x + right.w) * 0.25;
			left = left - vec4(mean);
			right = right - vec4(mean);
			upDown = upDown - vec4(mean);
			color.w = color[OperationMode] - mean;

			float sum = (((((abs(left.x) + abs(left.y)) + abs(left.z)) + abs(left.w)) + (((abs(right.x) + abs(right.y)) + abs(right.z)) + abs(right.w))) + (((abs(upDown.x) + abs(upDown.y)) + abs(upDown.z)) + abs(upDown.w)));
			float sumMean = 1.014185e+01 / sum;
			float std = (sumMean * sumMean);

#if defined(UseEdgeDirection)
			vec3 data = vec3(std, edgeDirection(left, right));
#else
			float data = std;
#endif
			vec2 aWY = weightY(pl.x, pl.y + 1.0, upDown.x, data);
			aWY += weightY(pl.x - 1.0, pl.y + 1.0, upDown.y, data);
			aWY += weightY(pl.x - 1.0, pl.y - 2.0, upDown.z, data);
			aWY += weightY(pl.x, pl.y - 2.0, upDown.w, data);
			aWY += weightY(pl.x + 1.0, pl.y - 1.0, left.x, data);
			aWY += weightY(pl.x, pl.y - 1.0, left.y, data);
			aWY += weightY(pl.x, pl.y, left.z, data);
			aWY += weightY(pl.x + 1.0, pl.y, left.w, data);
			aWY += weightY(pl.x - 1.0, pl.y - 1.0, right.x, data);
			aWY += weightY(pl.x - 2.0, pl.y - 1.0, right.y, data);
			aWY += weightY(pl.x - 2.0, pl.y, right.z, data);
			aWY += weightY(pl.x - 1.0, pl.y, right.w, data);

			float finallyY = aWY.y / aWY.x;
			float maxY = max(max(left.y, left.z), max(right.x, right.w));
			float minY = min(min(left.y, left.z), min(right.x, right.w));
			float deltaY = clamp(EdgeSharpness * finallyY, minY, maxY) - color.w;

			//smooth high contrast input
			deltaY = clamp(deltaY, -23.0 / 255.0, 23.0 / 255.0);

			color.x = clamp((color.x + deltaY), 0.0, 1.0);
			color.y = clamp((color.y + deltaY), 0.0, 1.0);
			color.z = clamp((color.z + deltaY), 0.0, 1.0);
		}
	}

	color.w = 1.0; //assume alpha channel is not used
	out_Target0.xyzw = color;
}
