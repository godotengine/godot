#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rg32ui, set = 0, binding = 0) uniform readonly uimage2D edges_in;
layout(rg32ui, set = 0, binding = 1) uniform writeonly uimage2D edges_out;
layout(rg16f, set = 0, binding = 2) uniform readonly image2D mesh_mask;

layout(push_constant, std430) uniform Params {
	ivec2 resolution;
	int spread;
	int pad;
}
params;

const ivec2 neighbor_offsets[8] = ivec2[8](
		ivec2(-1, 0),
		ivec2(1, 0),
		ivec2(0, -1),
		ivec2(0, 1),
		ivec2(-1, -1),
		ivec2(-1, 1),
		ivec2(1, -1),
		ivec2(1, 1));

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.resolution))) {
		return;
	}

	vec2 current_mask = imageLoad(mesh_mask, pixel).xy;
	float current_id = current_mask.x;

	uvec2 best_edge = imageLoad(edges_in, pixel).xy;
	vec2 best_diff = vec2(pixel - ivec2(best_edge));
	float best_dist = dot(best_diff, best_diff);

	for (int i = 0; i < 8; i++) {
		ivec2 sample_pixel = pixel + neighbor_offsets[i] * params.spread;
		sample_pixel = clamp(sample_pixel, ivec2(0), params.resolution - ivec2(1));

		uvec2 edge_candidate = imageLoad(edges_in, sample_pixel).xy;
		ivec2 edge_coord = ivec2(edge_candidate);

		if (any(lessThan(edge_coord, ivec2(0))) || any(greaterThanEqual(edge_coord, params.resolution))) {
			continue;
		}

		float edge_id = imageLoad(mesh_mask, edge_coord).x;
		if (edge_id == 0.0 || edge_id == current_id) {
			continue;
		}

		vec2 diff = vec2(pixel - edge_coord);
		float dist = dot(diff, diff);
		if (dist < best_dist) {
			best_dist = dist;
			best_edge = edge_candidate;
		}
	}

	imageStore(edges_out, pixel, uvec4(best_edge, 0u, 0u));
}
