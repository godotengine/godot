#[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_ssao;
layout(set = 1, binding = 0) uniform sampler2D source_depth;
#ifdef MODE_UPSCALE
layout(set = 2, binding = 0) uniform sampler2D source_depth_mipmaps;
#endif

layout(r8, set = 3, binding = 0) uniform restrict writeonly image2D dest_image;

//////////////////////////////////////////////////////////////////////////////////////////////
// Tunable Parameters:

layout(push_constant, binding = 1, std430) uniform Params {
	float edge_sharpness; /** Increase to make depth edges crisper. Decrease to reduce flicker. */
	int filter_scale;
	float z_far;
	float z_near;
	bool orthogonal;
	uint pad0;
	uint pad1;
	uint pad2;
	ivec2 axis; /** (1, 0) or (0, 1) */
	ivec2 screen_size;
}
params;

/** Filter radius in pixels. This will be multiplied by SCALE. */
#define R (4)

//////////////////////////////////////////////////////////////////////////////////////////////

// Gaussian coefficients
const float gaussian[R + 1] =
		//float[](0.356642, 0.239400, 0.072410, 0.009869);
		//float[](0.398943, 0.241971, 0.053991, 0.004432, 0.000134);  // stddev = 1.0
		float[](0.153170, 0.144893, 0.122649, 0.092902, 0.062970); // stddev = 2.0
//float[](0.111220, 0.107798, 0.098151, 0.083953, 0.067458, 0.050920, 0.036108); // stddev = 3.0

void main() {
	// Pixel being shaded
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(ssC, params.screen_size))) { //too large, do nothing
		return;
	}

#ifdef MODE_UPSCALE

	//closest one should be the same pixel, but check nearby just in case
	float depth = texelFetch(source_depth, ssC, 0).r;

	depth = depth * 2.0 - 1.0;
	if (params.orthogonal) {
		depth = ((depth + (params.z_far + params.z_near) / (params.z_far - params.z_near)) * (params.z_far - params.z_near)) / 2.0;
	} else {
		depth = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near - depth * (params.z_far - params.z_near));
	}

	vec2 pixel_size = 1.0 / vec2(params.screen_size);
	vec2 closest_uv = vec2(ssC) * pixel_size + pixel_size * 0.5;
	vec2 from_uv = closest_uv;
	vec2 ps2 = pixel_size; // * 2.0;

	float closest_depth = abs(textureLod(source_depth_mipmaps, closest_uv, 0.0).r - depth);

	vec2 offsets[4] = vec2[](vec2(ps2.x, 0), vec2(-ps2.x, 0), vec2(0, ps2.y), vec2(0, -ps2.y));
	for (int i = 0; i < 4; i++) {
		vec2 neighbour = from_uv + offsets[i];
		float neighbour_depth = abs(textureLod(source_depth_mipmaps, neighbour, 0.0).r - depth);
		if (neighbour_depth < closest_depth) {
			closest_uv = neighbour;
			closest_depth = neighbour_depth;
		}
	}

	float visibility = textureLod(source_ssao, closest_uv, 0.0).r;
	imageStore(dest_image, ssC, vec4(visibility));
#else

	float depth = texelFetch(source_depth, ssC, 0).r;

#ifdef MODE_FULL_SIZE
	depth = depth * 2.0 - 1.0;

	if (params.orthogonal) {
		depth = ((depth + (params.z_far + params.z_near) / (params.z_far - params.z_near)) * (params.z_far - params.z_near)) / 2.0;
	} else {
		depth = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near - depth * (params.z_far - params.z_near));
	}

#endif
	float depth_divide = 1.0 / params.z_far;

	//depth *= depth_divide;

	/*
	if (depth > params.z_far * 0.999) {
		discard; //skybox
	}
	*/

	float sum = texelFetch(source_ssao, ssC, 0).r;

	// Base weight for depth falloff.  Increase this for more blurriness,
	// decrease it for better edge discrimination
	float BASE = gaussian[0];
	float totalWeight = BASE;
	sum *= totalWeight;

	ivec2 clamp_limit = params.screen_size - ivec2(1);

	for (int r = -R; r <= R; ++r) {
		// We already handled the zero case above.  This loop should be unrolled and the static branch optimized out,
		// so the IF statement has no runtime cost
		if (r != 0) {
			ivec2 ppos = ssC + params.axis * (r * params.filter_scale);
			float value = texelFetch(source_ssao, clamp(ppos, ivec2(0), clamp_limit), 0).r;
			ivec2 rpos = clamp(ppos, ivec2(0), clamp_limit);

			float temp_depth = texelFetch(source_depth, rpos, 0).r;
#ifdef MODE_FULL_SIZE
			temp_depth = temp_depth * 2.0 - 1.0;
			if (params.orthogonal) {
				temp_depth = ((temp_depth + (params.z_far + params.z_near) / (params.z_far - params.z_near)) * (params.z_far - params.z_near)) / 2.0;
			} else {
				temp_depth = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near - temp_depth * (params.z_far - params.z_near));
			}
			//temp_depth *= depth_divide;
#endif
			// spatial domain: offset gaussian tap
			float weight = 0.3 + gaussian[abs(r)];
			//weight *= max(0.0, dot(temp_normal, normal));

			// range domain (the "bilateral" weight). As depth difference increases, decrease weight.
			weight *= max(0.0, 1.0 - params.edge_sharpness * abs(temp_depth - depth));

			sum += value * weight;
			totalWeight += weight;
		}
	}

	const float epsilon = 0.0001;
	float visibility = sum / (totalWeight + epsilon);

	imageStore(dest_image, ssC, vec4(visibility));
#endif
}
