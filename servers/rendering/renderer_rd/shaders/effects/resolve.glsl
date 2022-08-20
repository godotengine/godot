#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#ifdef MODE_RESOLVE_DEPTH
layout(set = 0, binding = 0) uniform sampler2DMS source_depth;
layout(r32f, set = 1, binding = 0) uniform restrict writeonly image2D dest_depth;
#endif

#ifdef MODE_RESOLVE_GI
layout(set = 0, binding = 0) uniform sampler2DMS source_depth;
layout(set = 0, binding = 1) uniform sampler2DMS source_normal_roughness;

layout(r32f, set = 1, binding = 0) uniform restrict writeonly image2D dest_depth;
layout(rgba8, set = 1, binding = 1) uniform restrict writeonly image2D dest_normal_roughness;

#ifdef VOXEL_GI_RESOLVE
layout(set = 2, binding = 0) uniform usampler2DMS source_voxel_gi;
layout(rg8ui, set = 3, binding = 0) uniform restrict writeonly uimage2D dest_voxel_gi;
#endif

#endif

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	int sample_count;
	uint pad;
}
params;

void main() {
	// Pixel being shaded
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pos, params.screen_size))) { //too large, do nothing
		return;
	}

#ifdef MODE_RESOLVE_DEPTH

	float depth_avg = 0.0;
	for (int i = 0; i < params.sample_count; i++) {
		depth_avg += texelFetch(source_depth, pos, i).r;
	}
	depth_avg /= float(params.sample_count);
	imageStore(dest_depth, pos, vec4(depth_avg));

#endif

#ifdef MODE_RESOLVE_GI

	float best_depth = 1e20;
	vec4 best_normal_roughness = vec4(0.0);
#ifdef VOXEL_GI_RESOLVE
	uvec2 best_voxel_gi;
#endif

#if 0

	for(int i=0;i<params.sample_count;i++) {
		float depth = texelFetch(source_depth,pos,i).r;
		if (depth < best_depth) { //use the depth closest to camera
			best_depth = depth;
			best_normal_roughness = texelFetch(source_normal_roughness,pos,i);

#ifdef VOXEL_GI_RESOLVE
			best_voxel_gi = texelFetch(source_voxel_gi,pos,i).rg;
#endif
		}
	}

#else

#if 1

	vec4 group1;
	vec4 group2;
	vec4 group3;
	vec4 group4;
	int best_index = 0;

	//2X
	group1.x = texelFetch(source_depth, pos, 0).r;
	group1.y = texelFetch(source_depth, pos, 1).r;

	//4X
	if (params.sample_count >= 4) {
		group1.z = texelFetch(source_depth, pos, 2).r;
		group1.w = texelFetch(source_depth, pos, 3).r;
	}
	//8X
	if (params.sample_count >= 8) {
		group2.x = texelFetch(source_depth, pos, 4).r;
		group2.y = texelFetch(source_depth, pos, 5).r;
		group2.z = texelFetch(source_depth, pos, 6).r;
		group2.w = texelFetch(source_depth, pos, 7).r;
	}
	//16X
	if (params.sample_count >= 16) {
		group3.x = texelFetch(source_depth, pos, 8).r;
		group3.y = texelFetch(source_depth, pos, 9).r;
		group3.z = texelFetch(source_depth, pos, 10).r;
		group3.w = texelFetch(source_depth, pos, 11).r;

		group4.x = texelFetch(source_depth, pos, 12).r;
		group4.y = texelFetch(source_depth, pos, 13).r;
		group4.z = texelFetch(source_depth, pos, 14).r;
		group4.w = texelFetch(source_depth, pos, 15).r;
	}

	if (params.sample_count == 2) {
		best_index = (pos.x & 1) ^ ((pos.y >> 1) & 1); //not much can be done here
	} else if (params.sample_count == 4) {
		vec4 freq = vec4(equal(group1, vec4(group1.x)));
		freq += vec4(equal(group1, vec4(group1.y)));
		freq += vec4(equal(group1, vec4(group1.z)));
		freq += vec4(equal(group1, vec4(group1.w)));

		float min_f = freq.x;
		best_index = 0;
		if (freq.y < min_f) {
			best_index = 1;
			min_f = freq.y;
		}
		if (freq.z < min_f) {
			best_index = 2;
			min_f = freq.z;
		}
		if (freq.w < min_f) {
			best_index = 3;
		}
	} else if (params.sample_count == 8) {
		vec4 freq0 = vec4(equal(group1, vec4(group1.x)));
		vec4 freq1 = vec4(equal(group2, vec4(group1.x)));
		freq0 += vec4(equal(group1, vec4(group1.y)));
		freq1 += vec4(equal(group2, vec4(group1.y)));
		freq0 += vec4(equal(group1, vec4(group1.z)));
		freq1 += vec4(equal(group2, vec4(group1.z)));
		freq0 += vec4(equal(group1, vec4(group1.w)));
		freq1 += vec4(equal(group2, vec4(group1.w)));
		freq0 += vec4(equal(group1, vec4(group2.x)));
		freq1 += vec4(equal(group2, vec4(group2.x)));
		freq0 += vec4(equal(group1, vec4(group2.y)));
		freq1 += vec4(equal(group2, vec4(group2.y)));
		freq0 += vec4(equal(group1, vec4(group2.z)));
		freq1 += vec4(equal(group2, vec4(group2.z)));
		freq0 += vec4(equal(group1, vec4(group2.w)));
		freq1 += vec4(equal(group2, vec4(group2.w)));

		float min_f0 = freq0.x;
		int best_index0 = 0;
		if (freq0.y < min_f0) {
			best_index0 = 1;
			min_f0 = freq0.y;
		}
		if (freq0.z < min_f0) {
			best_index0 = 2;
			min_f0 = freq0.z;
		}
		if (freq0.w < min_f0) {
			best_index0 = 3;
			min_f0 = freq0.w;
		}

		float min_f1 = freq1.x;
		int best_index1 = 4;
		if (freq1.y < min_f1) {
			best_index1 = 5;
			min_f1 = freq1.y;
		}
		if (freq1.z < min_f1) {
			best_index1 = 6;
			min_f1 = freq1.z;
		}
		if (freq1.w < min_f1) {
			best_index1 = 7;
			min_f1 = freq1.w;
		}

		best_index = mix(best_index0, best_index1, min_f0 < min_f1);
	}

#else
	float depths[16];
	int depth_indices[16];
	int depth_amount[16];
	int depth_count = 0;

	for (int i = 0; i < params.sample_count; i++) {
		float depth = texelFetch(source_depth, pos, i).r;
		int depth_index = -1;
		for (int j = 0; j < depth_count; j++) {
			if (abs(depths[j] - depth) < 0.000001) {
				depth_index = j;
				break;
			}
		}

		if (depth_index == -1) {
			depths[depth_count] = depth;
			depth_indices[depth_count] = i;
			depth_amount[depth_count] = 1;
			depth_count += 1;
		} else {
			depth_amount[depth_index] += 1;
		}
	}

	int depth_least = 0xFFFF;
	int best_index = 0;
	for (int j = 0; j < depth_count; j++) {
		if (depth_amount[j] < depth_least) {
			best_index = depth_indices[j];
			depth_least = depth_amount[j];
		}
	}
#endif
	best_depth = texelFetch(source_depth, pos, best_index).r;
	best_normal_roughness = texelFetch(source_normal_roughness, pos, best_index);
#ifdef VOXEL_GI_RESOLVE
	best_voxel_gi = texelFetch(source_voxel_gi, pos, best_index).rg;
#endif

#endif

	imageStore(dest_depth, pos, vec4(best_depth));
	imageStore(dest_normal_roughness, pos, vec4(best_normal_roughness));
#ifdef VOXEL_GI_RESOLVE
	imageStore(dest_voxel_gi, pos, uvec4(best_voxel_gi, 0, 0));
#endif

#endif
}
