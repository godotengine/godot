#[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#ifdef MODE_RESOLVE_GI
layout(set = 0, binding = 0) uniform sampler2DMS source_depth;
layout(set = 0, binding = 1) uniform sampler2DMS source_normal_roughness;

layout(r32f, set = 1, binding = 0) uniform restrict writeonly image2D dest_depth;
layout(rgba8, set = 1, binding = 1) uniform restrict writeonly image2D dest_normal_roughness;

#ifdef GIPROBE_RESOLVE
layout(set = 2, binding = 0) uniform usampler2DMS source_giprobe;
layout(rg8ui, set = 3, binding = 0) uniform restrict writeonly uimage2D dest_giprobe;
#endif

#endif

layout(push_constant, binding = 16, std430) uniform Params {
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

#ifdef MODE_RESOLVE_GI

	float best_depth = 1e20;
	vec4 best_normal_roughness = vec4(0.0);
#ifdef GIPROBE_RESOLVE
	uvec2 best_giprobe;
#endif

#if 0

	for(int i=0;i<params.sample_count;i++) {
		float depth = texelFetch(source_depth,pos,i).r;
		if (depth < best_depth) { //use the depth closest to camera
			best_depth = depth;
			best_normal_roughness = texelFetch(source_normal_roughness,pos,i);

#ifdef GIPROBE_RESOLVE
			best_giprobe = texelFetch(source_giprobe,pos,i).rg;
#endif
		}
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

	best_depth = texelFetch(source_depth, pos, best_index).r;
	best_normal_roughness = texelFetch(source_normal_roughness, pos, best_index);
#ifdef GIPROBE_RESOLVE
	best_giprobe = texelFetch(source_giprobe, pos, best_index).rg;
#endif

#endif

	imageStore(dest_depth, pos, vec4(best_depth));
	imageStore(dest_normal_roughness, pos, vec4(best_normal_roughness));
#ifdef GIPROBE_RESOLVE
	imageStore(dest_giprobe, pos, uvec4(best_giprobe, 0, 0));
#endif

#endif
}
