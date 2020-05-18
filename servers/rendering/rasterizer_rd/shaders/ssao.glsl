#[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define TWO_PI 6.283185307179586476925286766559

#ifdef SSAO_QUALITY_HIGH
#define NUM_SAMPLES (20)
#endif

#ifdef SSAO_QUALITY_ULTRA
#define NUM_SAMPLES (48)
#endif

#ifdef SSAO_QUALITY_LOW
#define NUM_SAMPLES (8)
#endif

#if !defined(SSAO_QUALITY_LOW) && !defined(SSAO_QUALITY_HIGH) && !defined(SSAO_QUALITY_ULTRA)
#define NUM_SAMPLES (12)
#endif

// If using depth mip levels, the log of the maximum pixel offset before we need to switch to a lower
// miplevel to maintain reasonable spatial locality in the cache
// If this number is too small (< 3), too many taps will land in the same pixel, and we'll get bad variance that manifests as flashing.
// If it is too high (> 5), we'll get bad performance because we're not using the MIP levels effectively
#define LOG_MAX_OFFSET (3)

// This must be less than or equal to the MAX_MIP_LEVEL defined in SSAO.cpp
#define MAX_MIP_LEVEL (4)

// This is the number of turns around the circle that the spiral pattern makes.  This should be prime to prevent
// taps from lining up.  This particular choice was tuned for NUM_SAMPLES == 9

const int ROTATIONS[] = int[](
		1, 1, 2, 3, 2, 5, 2, 3, 2,
		3, 3, 5, 5, 3, 4, 7, 5, 5, 7,
		9, 8, 5, 5, 7, 7, 7, 8, 5, 8,
		11, 12, 7, 10, 13, 8, 11, 8, 7, 14,
		11, 11, 13, 12, 13, 19, 17, 13, 11, 18,
		19, 11, 11, 14, 17, 21, 15, 16, 17, 18,
		13, 17, 11, 17, 19, 18, 25, 18, 19, 19,
		29, 21, 19, 27, 31, 29, 21, 18, 17, 29,
		31, 31, 23, 18, 25, 26, 25, 23, 19, 34,
		19, 27, 21, 25, 39, 29, 17, 21, 27);

//#define NUM_SPIRAL_TURNS (7)
const int NUM_SPIRAL_TURNS = ROTATIONS[NUM_SAMPLES - 1];

layout(set = 0, binding = 0) uniform sampler2D source_depth_mipmaps;
layout(r8, set = 1, binding = 0) uniform restrict writeonly image2D dest_image;

#ifndef USE_HALF_SIZE
layout(set = 2, binding = 0) uniform sampler2D source_depth;
#endif

layout(set = 3, binding = 0) uniform sampler2D source_normal;

layout(push_constant, binding = 1, std430) uniform Params {
	ivec2 screen_size;
	float z_far;
	float z_near;

	bool orthogonal;
	float intensity_div_r6;
	float radius;
	float bias;

	vec4 proj_info;
	vec2 pixel_size;
	float proj_scale;
	uint pad;
}
params;

vec3 reconstructCSPosition(vec2 S, float z) {
	if (params.orthogonal) {
		return vec3((S.xy * params.proj_info.xy + params.proj_info.zw), z);
	} else {
		return vec3((S.xy * params.proj_info.xy + params.proj_info.zw) * z, z);
	}
}

vec3 getPosition(ivec2 ssP) {
	vec3 P;
#ifdef USE_HALF_SIZE
	P.z = texelFetch(source_depth_mipmaps, ssP, 0).r;
	P.z = -P.z;
#else
	P.z = texelFetch(source_depth, ssP, 0).r;

	P.z = P.z * 2.0 - 1.0;
	if (params.orthogonal) {
		P.z = ((P.z + (params.z_far + params.z_near) / (params.z_far - params.z_near)) * (params.z_far - params.z_near)) / 2.0;
	} else {
		P.z = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near - P.z * (params.z_far - params.z_near));
	}
	P.z = -P.z;
#endif
	// Offset to pixel center
	P = reconstructCSPosition(vec2(ssP) + vec2(0.5), P.z);
	return P;
}

/** Returns a unit vector and a screen-space radius for the tap on a unit disk (the caller should scale by the actual disk radius) */
vec2 tapLocation(int sampleNumber, float spinAngle, out float ssR) {
	// Radius relative to ssR
	float alpha = (float(sampleNumber) + 0.5) * (1.0 / float(NUM_SAMPLES));
	float angle = alpha * (float(NUM_SPIRAL_TURNS) * 6.28) + spinAngle;

	ssR = alpha;
	return vec2(cos(angle), sin(angle));
}

/** Read the camera-space position of the point at screen-space pixel ssP + unitOffset * ssR.  Assumes length(unitOffset) == 1 */
vec3 getOffsetPosition(ivec2 ssP, float ssR) {
	// Derivation:
	//  mipLevel = floor(log(ssR / MAX_OFFSET));

	int mipLevel = clamp(int(floor(log2(ssR))) - LOG_MAX_OFFSET, 0, MAX_MIP_LEVEL);

	vec3 P;

	// We need to divide by 2^mipLevel to read the appropriately scaled coordinate from a MIP-map.
	// Manually clamp to the texture size because texelFetch bypasses the texture unit
	ivec2 mipP = clamp(ssP >> mipLevel, ivec2(0), (params.screen_size >> mipLevel) - ivec2(1));

#ifdef USE_HALF_SIZE
	P.z = texelFetch(source_depth_mipmaps, mipP, mipLevel).r;
	P.z = -P.z;
#else
	if (mipLevel < 1) {
		//read from depth buffer
		P.z = texelFetch(source_depth, mipP, 0).r;
		P.z = P.z * 2.0 - 1.0;
		if (params.orthogonal) {
			P.z = ((P.z + (params.z_far + params.z_near) / (params.z_far - params.z_near)) * (params.z_far - params.z_near)) / 2.0;
		} else {
			P.z = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near - P.z * (params.z_far - params.z_near));
		}
		P.z = -P.z;

	} else {
		//read from mipmaps
		P.z = texelFetch(source_depth_mipmaps, mipP, mipLevel - 1).r;
		P.z = -P.z;
	}
#endif

	// Offset to pixel center
	P = reconstructCSPosition(vec2(ssP) + vec2(0.5), P.z);

	return P;
}

/** Compute the occlusion due to sample with index \a i about the pixel at \a ssC that corresponds
	to camera-space point \a C with unit normal \a n_C, using maximum screen-space sampling radius \a ssDiskRadius

	Note that units of H() in the HPG12 paper are meters, not
	unitless.  The whole falloff/sampling function is therefore
	unitless.  In this implementation, we factor out (9 / radius).

	Four versions of the falloff function are implemented below
*/
float sampleAO(in ivec2 ssC, in vec3 C, in vec3 n_C, in float ssDiskRadius, in float p_radius, in int tapIndex, in float randomPatternRotationAngle) {
	// Offset on the unit disk, spun for this pixel
	float ssR;
	vec2 unitOffset = tapLocation(tapIndex, randomPatternRotationAngle, ssR);
	ssR *= ssDiskRadius;

	ivec2 ssP = ivec2(ssR * unitOffset) + ssC;

	if (any(lessThan(ssP, ivec2(0))) || any(greaterThanEqual(ssP, params.screen_size))) {
		return 0.0;
	}

	// The occluding point in camera space
	vec3 Q = getOffsetPosition(ssP, ssR);

	vec3 v = Q - C;

	float vv = dot(v, v);
	float vn = dot(v, n_C);

	const float epsilon = 0.01;
	float radius2 = p_radius * p_radius;

	// A: From the HPG12 paper
	// Note large epsilon to avoid overdarkening within cracks
	//return float(vv < radius2) * max((vn - bias) / (epsilon + vv), 0.0) * radius2 * 0.6;

	// B: Smoother transition to zero (lowers contrast, smoothing out corners). [Recommended]
	float f = max(radius2 - vv, 0.0);
	return f * f * f * max((vn - params.bias) / (epsilon + vv), 0.0);

	// C: Medium contrast (which looks better at high radii), no division.  Note that the
	// contribution still falls off with radius^2, but we've adjusted the rate in a way that is
	// more computationally efficient and happens to be aesthetically pleasing.
	// return 4.0 * max(1.0 - vv * invRadius2, 0.0) * max(vn - bias, 0.0);

	// D: Low contrast, no division operation
	// return 2.0 * float(vv < radius * radius) * max(vn - bias, 0.0);
}

void main() {
	// Pixel being shaded
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(ssC, params.screen_size))) { //too large, do nothing
		return;
	}

	// World space point being shaded
	vec3 C = getPosition(ssC);

#ifdef USE_HALF_SIZE
	vec3 n_C = texelFetch(source_normal, ssC << 1, 0).xyz * 2.0 - 1.0;
#else
	vec3 n_C = texelFetch(source_normal, ssC, 0).xyz * 2.0 - 1.0;
#endif
	n_C = normalize(n_C);
	n_C.y = -n_C.y; //because this code reads flipped

	// Hash function used in the HPG12 AlchemyAO paper
	float randomPatternRotationAngle = mod(float((3 * ssC.x ^ ssC.y + ssC.x * ssC.y) * 10), TWO_PI);

	// Reconstruct normals from positions. These will lead to 1-pixel black lines
	// at depth discontinuities, however the blur will wipe those out so they are not visible
	// in the final image.

	// Choose the screen-space sample radius
	// proportional to the projected area of the sphere

	float ssDiskRadius = -params.proj_scale * params.radius;
	if (!params.orthogonal) {
		ssDiskRadius = -params.proj_scale * params.radius / C.z;
	}
	float sum = 0.0;
	for (int i = 0; i < NUM_SAMPLES; ++i) {
		sum += sampleAO(ssC, C, n_C, ssDiskRadius, params.radius, i, randomPatternRotationAngle);
	}

	float A = max(0.0, 1.0 - sum * params.intensity_div_r6 * (5.0 / float(NUM_SAMPLES)));

	imageStore(dest_image, ssC, vec4(A));
}
