///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016, Intel Corporation
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// File changes (yyyy-mm-dd)
// 2016-09-07: filip.strugar@intel.com: first commit
// 2020-12-05: clayjohn: convert to Vulkan and Godot
// 2021-05-27: clayjohn: convert SSAO to SSIL
// 2025-12-16: GT-VBAO implementation for SSIL
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2DArray source_depth_mipmaps;
layout(rgba8, set = 0, binding = 1) uniform restrict readonly image2D source_normal;
layout(set = 0, binding = 2) uniform Constants { //get into a lower set
	vec4 rotation_matrices[20];
}
constants;

#ifdef ADAPTIVE
layout(rgba16, set = 1, binding = 0) uniform restrict readonly image2DArray source_ssil;
layout(set = 1, binding = 1) uniform sampler2D source_importance;
layout(set = 1, binding = 2, std430) buffer Counter {
	uint sum;
}
counter;
#endif

layout(rgba16, set = 2, binding = 0) uniform restrict writeonly image2D dest_image;
layout(r8, set = 2, binding = 1) uniform image2D edges_weights_image;

layout(set = 3, binding = 0) uniform sampler2D last_frame;
layout(set = 3, binding = 1) uniform ProjectionConstants {
	mat4 reprojection;
}
projection_constants;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	int pass;
	int quality;

	vec2 half_screen_pixel_size;
	vec2 half_screen_pixel_size_x025;

	vec2 NDC_to_view_mul;
	vec2 NDC_to_view_add;

	float vb_flag; // use visibility bitmask
	float z_near;
	float z_far;
	float pad_flags;

	float radius;
	float intensity;
	int size_multiplier;

	float fade_out_mul;
	float fade_out_add;
	float normal_rejection_amount;
	float inv_radius_near_limit;

	bool is_orthogonal;
	float neg_inv_radius;
	float load_counter_avg_div;
	float adaptive_sample_limit;

	ivec2 pass_coord_offset;
	vec2 pass_uv_offset;
}
params;

float pack_edges(vec4 p_edgesLRTB) {
	p_edgesLRTB = round(clamp(p_edgesLRTB, 0.0, 1.0) * 3.05);
	return dot(p_edgesLRTB, vec4(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

vec3 NDC_to_view_space(vec2 p_pos, float p_viewspace_depth) {
	if (params.is_orthogonal) {
		return vec3((params.NDC_to_view_mul * p_pos.xy + params.NDC_to_view_add), p_viewspace_depth);
	} else {
		return vec3((params.NDC_to_view_mul * p_pos.xy + params.NDC_to_view_add) * p_viewspace_depth, p_viewspace_depth);
	}
}

vec3 load_normal(ivec2 p_pos) {
	vec3 encoded_normal = normalize(imageLoad(source_normal, p_pos).xyz * 2.0 - 1.0);
	encoded_normal.z = -encoded_normal.z;
	return encoded_normal;
}

vec4 calculate_edges(const float p_center_z, const float p_left_z, const float p_right_z, const float p_top_z, const float p_bottom_z) {
	// slope-sensitive depth-based edge detection
	vec4 edgesLRTB = vec4(p_left_z, p_right_z, p_top_z, p_bottom_z) - p_center_z;
	vec4 edgesLRTB_slope_adjusted = edgesLRTB + edgesLRTB.yxwz;
	edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTB_slope_adjusted));
	return clamp((1.3 - edgesLRTB / (p_center_z * 0.040)), 0.0, 1.0);
}

// Interleaved Gradient Noise for jitter
float interleaved_gradient_noise(vec2 n) {
	return fract(52.9829189 * fract(0.06711056 * n.x + 0.00583715 * n.y));
}

// Slice-based SSIL gather (GTAO-style for indirect light)
void integrate_slice_ssil(vec2 direction, float viewspace_radius, vec2 screen_uv, vec3 view_pos, vec3 view_normal, float pixels_per_meter, int num_steps, inout vec3 total_color, inout float total_obscurance, inout uint mask_bits) {
	vec2 dir_uv = direction * params.half_screen_pixel_size * 2.0; 
	float pixel_radius = viewspace_radius * pixels_per_meter;
	pixel_radius = min(pixel_radius, 256.0); 
	if (pixel_radius < 1.0) return;

	float step_size_px = pixel_radius / float(num_steps + 1.0);
	vec2 step_vec = direction * step_size_px * params.half_screen_pixel_size * 2.0; 

	vec2 current_uv = screen_uv + step_vec;

	float h_angle_max = -10.0; // Start unclamped to find best hit
	vec2 best_sample_uv = vec2(-1.0);
	float best_sample_depth = 0.0;
	bool found_hit = false;
	
	vec3 V = normalize(-view_pos);
	
	// Normal angle in slice plane
	float n_angle_slice = atan(view_normal.z, dot(view_normal.xy, direction)); 
	float tangent_angle = n_angle_slice - 3.14159265359/2.0;

	for (int i = 0; i < num_steps; i++) {
		float sample_depth = textureLod(source_depth_mipmaps, vec3(current_uv, params.pass), 0.0).x;
		vec3 sample_pos = NDC_to_view_space(current_uv, sample_depth);
		
		vec3 diff = sample_pos - view_pos;
		float dist_sq = dot(diff, diff);
		
		if (dist_sq < 0.0001) { current_uv += step_vec; continue; }
		
		float dist_xy = length(diff.xy);
		float angle = atan(diff.z, dist_xy); 
		
		if (angle > h_angle_max) {
			h_angle_max = angle;
			best_sample_uv = current_uv;
			best_sample_depth = sample_depth;
			found_hit = true;
		}
		
		current_uv += step_vec;
	}
	
	// Clamp horizon to tangent for integral calculation
	float h_angle_clamped = clamp(h_angle_max, tangent_angle, n_angle_slice + 3.14159265359/2.0);

	// Calculate geometry weight (integral over [tangent, horizon])
	// sin(h - n) - sin(tangent - n)  => sin(h - n) - (-1) => sin(h - n) + 1
	float geometry_weight = 0.25 * (sin(h_angle_clamped - n_angle_slice) + 1.0);
	float occlusion_slice = clamp(geometry_weight, 0.0, 1.0);
	
	// Accumulate geometry visibility (occlusion of sky)
	total_obscurance += occlusion_slice;
	
	if (found_hit && occlusion_slice > 0.001) {
		// Sample color from best hit (horizon)
		vec4 sample_pos_clip = projection_constants.reprojection * vec4(best_sample_uv * 2.0 - 1.0, (best_sample_depth - params.z_near) / (params.z_far - params.z_near) * 2.0 - 1.0, 1.0);
		vec2 reprojected_uv = (sample_pos_clip.xy / sample_pos_clip.w) * 0.5 + 0.5;
		
		vec3 sample_color = textureLod(last_frame, reprojected_uv, 0.0).rgb; // Level 0 for sharpness
		sample_color /= (1.0 + dot(sample_color, vec3(0.299, 0.587, 0.114))); // Tone map
		
		// Weight by geometry factor
		total_color += sample_color * occlusion_slice;
	}
	
	// Bitmask update (using the raw horizon found)
	// Check against surface tangent to capture slope occlusion correctly
	if (h_angle_max > n_angle_slice + 0.05) {
		float dir_ang = atan(direction.y, direction.x);
		float ang01 = dir_ang * 0.15915494 + 0.5;
		uint bit = uint(clamp(floor(ang01 * 32.0), 0.0, 31.0));
		mask_bits |= (1u << bit);
	}
}


void generate_SSIL(out vec3 r_color, out vec4 r_edges, out float r_obscurance, out float r_weight, const vec2 p_pos, int p_quality_level, bool p_adaptive_base) {
	vec2 pos_rounded = trunc(p_pos);
	uvec2 upos = uvec2(pos_rounded);

	vec4 valuesUL = textureGather(source_depth_mipmaps, vec3(pos_rounded * params.half_screen_pixel_size, params.pass));
	vec4 valuesBR = textureGather(source_depth_mipmaps, vec3((pos_rounded + vec2(1.0)) * params.half_screen_pixel_size, params.pass));

	float pix_z = valuesUL.y;
	float pix_left_z = valuesUL.x;
	float pix_top_z = valuesUL.z;
	float pix_right_z = valuesBR.z;
	float pix_bottom_z = valuesBR.x;

	vec2 normalized_screen_pos = pos_rounded * params.half_screen_pixel_size + params.half_screen_pixel_size_x025;
	vec3 pix_center_pos = NDC_to_view_space(normalized_screen_pos, pix_z);
	
	uvec2 full_res_coord = upos * 2 * params.size_multiplier + params.pass_coord_offset.xy;
	vec3 pixel_normal = load_normal(ivec2(full_res_coord));
	const vec2 pixel_size_at_center = NDC_to_view_space(normalized_screen_pos.xy + params.half_screen_pixel_size, pix_center_pos.z).xy - pix_center_pos.xy;

	float viewspace_radius;
	float pixel_lookup_radius;
	float fallof_sq;
	{
		viewspace_radius = params.radius;
		const float too_close_limit = clamp(length(pix_center_pos) * params.inv_radius_near_limit, 0.0, 1.0) * 0.8 + 0.2;
		viewspace_radius *= too_close_limit;
		pixel_lookup_radius = (0.85 * viewspace_radius) / pixel_size_at_center.x;
		fallof_sq = -1.0 / (viewspace_radius * viewspace_radius);
	}

	vec4 edgesLRTB = vec4(1.0);
	if (!p_adaptive_base && (p_quality_level >= 1)) {
		edgesLRTB = calculate_edges(pix_z, pix_left_z, pix_right_z, pix_top_z, pix_bottom_z);
	}

	vec3 total_color = vec3(0.0);
	float total_obscurance = 0.0;
	uint mask_bits = 0u;

	int num_slices = (p_quality_level == 0) ? 1 : ((p_quality_level == 1) ? 2 : ((p_quality_level == 2) ? 3 : 4));
	int steps_per_slice = (p_quality_level == 0) ? 2 : ((p_quality_level == 1) ? 3 : ((p_quality_level == 2) ? 4 : 6));
	
	float noise_val = interleaved_gradient_noise(vec2(gl_GlobalInvocationID.xy) + vec2(params.pass * 5.588));
	float spatial_offset = noise_val;
	float pixels_per_meter = 1.0 / pixel_size_at_center.x;

	for (int i = 0; i < num_slices; i++) {
		float slice_angle = (float(i) + spatial_offset) * (3.14159265359 / float(num_slices));
		vec2 direction = vec2(cos(slice_angle), sin(slice_angle));
		
		integrate_slice_ssil(direction, viewspace_radius, normalized_screen_pos, pix_center_pos, pixel_normal, pixels_per_meter, steps_per_slice, total_color, total_obscurance, mask_bits);
		integrate_slice_ssil(-direction, viewspace_radius, normalized_screen_pos, pix_center_pos, pixel_normal, pixels_per_meter, steps_per_slice, total_color, total_obscurance, mask_bits);
	}
	
	// Normalize
	float norm_factor = 2.0 / float(num_slices); // Logic from SSAO
	// Actually, for color sum, we just want average color intensity?
	// Original code: average weighted samples.
	// Here total_color is sum of (color * weight).
	// We need to divide by sum of weights?
	// GTAO produces 'total_obscurance' which is sum of occlusion weights.
	// If we divide by total_obscurance, we get average color.
	// Then we multiply by total_obscurance (AO factor) to apply it.
	// So effectively total_color is already what we want? 
	// But we need to normalize for the number of slices.
	
	total_color /= float(num_slices) * 2.0; 
	total_obscurance /= float(num_slices) * 2.0;

	// Untonemap?
	// Original did: color /= 1.0 - dot(color, ...);
	// We need to match that.
	vec3 color = total_color;
	color /= max(0.0001, 1.0 - dot(color, vec3(0.299, 0.587, 0.114)));
	
	float fade_out = clamp(pix_center_pos.z * params.fade_out_mul + params.fade_out_add, 0.0, 1.0);
	if (!p_adaptive_base && (p_quality_level >= 1)) {
		float edge_fadeout_factor = clamp((1.0 - edgesLRTB.x - edgesLRTB.y) * 0.35, 0.0, 1.0) + clamp((1.0 - edgesLRTB.z - edgesLRTB.w) * 0.35, 0.0, 1.0);
		fade_out *= clamp(1.0 - edge_fadeout_factor, 0.0, 1.0);
	}

	color = params.intensity * color;
	color *= fade_out;

	if (params.vb_flag > 0.5) {
		float vb_visibility = 1.0 - float(bitCount(mask_bits)) * (1.0 / 32.0);
		vb_visibility = clamp(vb_visibility, 0.0, 1.0);
		
		bool vb_bi = params.vb_flag > 1.5;
		float vis_applied = mix(1.0, vb_visibility, vb_bi ? 0.9 : 0.6);
		
		color *= vis_applied;
		total_obscurance *= vis_applied;
	}

	r_color = color;
	r_edges = edgesLRTB;
	r_obscurance = clamp(total_obscurance * params.intensity, 0.0, 1.0);
	r_weight = 1.0;
}

void main() {
	vec3 out_color;
	float out_obscurance;
	float out_weight;
	vec4 out_edges;
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(ssC, params.screen_size))) { //too large, do nothing
		return;
	}

	vec2 uv = vec2(gl_GlobalInvocationID) + vec2(0.5);
#ifdef SSIL_BASE
	generate_SSIL(out_color, out_edges, out_obscurance, out_weight, uv, params.quality, true);

	imageStore(dest_image, ssC, vec4(out_color, out_obscurance));
	imageStore(edges_weights_image, ssC, vec4(1.0)); // Weight placeholder
#else
	generate_SSIL(out_color, out_edges, out_obscurance, out_weight, uv, params.quality, false); 

	imageStore(dest_image, ssC, vec4(out_color, out_obscurance));
	imageStore(edges_weights_image, ssC, vec4(pack_edges(out_edges)));
#endif
}
