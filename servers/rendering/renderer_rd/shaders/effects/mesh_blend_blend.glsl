/* clang-format off */
#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

void main() {
	vec2 vertex_base;
	if (gl_VertexIndex == 0) {
		vertex_base = vec2(-1.0, -1.0);
	} else if (gl_VertexIndex == 1) {
		vertex_base = vec2(-1.0, 3.0);
	} else {
		vertex_base = vec2(3.0, -1.0);
	}

	gl_Position = vec4(vertex_base, 0.0, 1.0);
	uv_interp = clamp(vertex_base, vec2(0.0), vec2(1.0));
}

/* clang-format off */
#[fragment]

#version 450

#VERSION_DEFINES

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;
/* clang-format on */

#define MAX_VIEWS 2

layout(set = 0, binding = 0, std140) uniform MeshBlendCamera {
	mat4 inv_view_projection[MAX_VIEWS];
}
camera_data;

layout(push_constant, std430) uniform Params {
	ivec2 resolution;		// 8
	float edge_radius;		// 12
	int view_index;			// 16
	int use_world_radius;	// 20
	float neighbor_blend;	// 24
	float pad_pc0;			// 28
	float pad_pc1;			// 32
} params;

layout(set = 1, binding = 0) uniform sampler2D source_color;
layout(set = 1, binding = 1) uniform sampler2D source_depth;
layout(set = 1, binding = 2) uniform sampler2D mask_tex;   // .x = id, .y = material_scale
layout(set = 1, binding = 3) uniform usampler2D edge_tex;

vec3 reconstruct_position(vec2 uv, float depth) {
	vec4 ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);
	vec4 position = camera_data.inv_view_projection[params.view_index] * ndc;
	return position.xyz / position.w;
}

void main() {
	ivec2 pixel = ivec2(gl_FragCoord.xy);
	ivec2 resolution = params.resolution;
	if (any(greaterThanEqual(pixel, resolution))) {
		discard;
	}

	vec2 resolution_rcp = vec2(1.0) / vec2(resolution);
	vec2 pixel_uv = (vec2(pixel) + 0.5) * resolution_rcp;

	vec2 mask_value = texelFetch(mask_tex, pixel, 0).xy;
	float current_id = mask_value.x;

	if (current_id == 0.0) {
		frag_color = texelFetch(source_color, pixel, 0);
		return;
	}

	uvec2 edge_coord_u = texelFetch(edge_tex, pixel, 0).xy;
	ivec2 edge_coord = ivec2(edge_coord_u);
	if (edge_coord == ivec2(0)) {
		frag_color = texelFetch(source_color, pixel, 0);
		return;
	}

	vec2 edge_mask = texelFetch(mask_tex, edge_coord, 0).xy;
	float edge_id = edge_mask.x;
	if (edge_id == 0.0 || edge_id == current_id) {
		frag_color = texelFetch(source_color, pixel, 0);
		return;
	}

	float material_scale = mask_value.y;
	float neighbor_scale = edge_mask.y;

	float base_scale_max;
	if (material_scale < 0.0) {
		base_scale_max = max(0.0, neighbor_scale + material_scale);
	} else {
		base_scale_max = max(material_scale, neighbor_scale);
	}

	float base_scale_avg = 0.5 * (max(material_scale, 0.0) + max(neighbor_scale, 0.0));

	float nb = clamp(params.neighbor_blend, -1.0, 1.0);
	float distance_scale = mix(base_scale_max, base_scale_avg, nb);

	float distance_falloff = distance_scale;
	if (distance_falloff <= 0.0) {
		frag_color = texelFetch(source_color, pixel, 0);
		return;
	}

	vec2 best_offset = vec2(edge_coord) - vec2(pixel);

	vec2 seam_uv = (vec2(pixel) + best_offset * 2.0 + 0.5) * resolution_rcp;
	seam_uv = clamp(seam_uv, vec2(0.0), vec2(1.0));

	vec4 gather_r = textureGather(source_color, seam_uv, 0);
	vec4 gather_g = textureGather(source_color, seam_uv, 1);
	vec4 gather_b = textureGather(source_color, seam_uv, 2);
	vec4 gather_mask = textureGather(mask_tex, seam_uv, 0);
	vec4 gather_depth = textureGather(source_depth, seam_uv, 0);

	float depth_current = texelFetch(source_depth, pixel, 0).r;
	vec3 current_pos = reconstruct_position(pixel_uv, depth_current);
	float edge_depth = texelFetch(source_depth, edge_coord, 0).r;
	vec3 edge_pos = reconstruct_position((vec2(edge_coord) + 0.5) * resolution_rcp, edge_depth);

	vec3 other_color = vec3(0.0);
	float sum = 0.0;
	float dweight = 0.0;

	for (int i = 0; i < 4; i++) {
		float sample_id = gather_mask[i];
		if (sample_id != current_id && sample_id == edge_id) {
			vec3 sample_pos = reconstruct_position(seam_uv, gather_depth[i]);
			float diff = length(sample_pos - current_pos);
			dweight += clamp(1.0 - diff / distance_falloff, 0.0, 1.0);
			other_color += vec3(gather_r[i], gather_g[i], gather_b[i]);
			sum += 1.0;
		}
	}

	vec4 original = texelFetch(source_color, pixel, 0);
	if (sum <= 0.0) {
		frag_color = original;
		return;
	}

	other_color /= sum;
	dweight /= sum;

	float world_best_dist = length(edge_pos - current_pos);
	float radius = params.edge_radius;
	if (params.use_world_radius == 0) {
		float pixel_world = 0.0;
		ivec2 offsets[2] = ivec2[](ivec2(1, 0), ivec2(0, 1));
		for (int i = 0; i < 2 && pixel_world <= 0.0; i++) {
			ivec2 neighbor_px = clamp(pixel + offsets[i], ivec2(0), resolution - ivec2(1));
			float neighbor_depth = texelFetch(source_depth, neighbor_px, 0).r;
			vec3 neighbor_pos = reconstruct_position((vec2(neighbor_px) + 0.5) * resolution_rcp, neighbor_depth);
			pixel_world = length(neighbor_pos - current_pos);
		}
		if (pixel_world <= 0.0) {
			pixel_world = 1.0;
		}
		radius = params.edge_radius * pixel_world;
	}

	radius = max(radius, 0.0001);
	float weight = clamp(0.5 - world_best_dist / radius, 0.0, 1.0) * dweight;
	frag_color = vec4(mix(original.rgb, other_color, weight), 1.0);
}
