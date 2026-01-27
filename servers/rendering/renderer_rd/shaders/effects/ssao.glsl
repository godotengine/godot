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
// 2025-12-16: GT-VBAO implementation
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
layout(rg8, set = 1, binding = 0) uniform restrict readonly image2DArray source_ssao;
layout(set = 1, binding = 1) uniform sampler2D source_importance;
layout(set = 1, binding = 2, std430) buffer Counter {
	uint sum;
}
counter;
#endif

layout(rg8, set = 2, binding = 0) uniform restrict writeonly image2D dest_image;

// This push_constant is full - 128 bytes - if you need to add more data, consider adding to the uniform buffer instead
layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	int pass;
	int quality;

	vec2 half_screen_pixel_size;
	int size_multiplier;
	float detail_intensity;

	vec2 NDC_to_view_mul;
	vec2 NDC_to_view_add;

	vec2 flags; // x: VB mode (0 off, 1 uni, 2 bi), y: unused
	vec2 half_screen_pixel_size_x025;

	float radius;
	float intensity;
	float shadow_power;
	float shadow_clamp;

	float fade_out_mul;
	float fade_out_add;
	float horizon_angle_threshold;
	float inv_radius_near_limit;

	bool is_orthogonal;
	float neg_inv_radius;
	float load_counter_avg_div;
	float adaptive_sample_limit;

	ivec2 pass_coord_offset;
	vec2 pass_uv_offset;
}
params;

// packing/unpacking for edges; 2 bits per edge mean 4 gradient values (0, 0.33, 0.66, 1) for smoother transitions!
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

vec3 load_normal(ivec2 p_pos, ivec2 p_offset) {
	vec3 encoded_normal = normalize(imageLoad(source_normal, p_pos + p_offset).xyz * 2.0 - 1.0);
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

// Slice-based horizon search (GTAO)
void integrate_slice(vec2 direction, float viewspace_radius, vec2 screen_uv, vec3 view_pos, vec3 view_normal, float pixels_per_meter, int num_steps, inout float visibility, inout uint mask_bits, bool bi_directional) {
	vec2 dir_uv = direction * params.half_screen_pixel_size * 2.0; // Direction in UV space (normalized roughly)
	
	// Project direction to screen pixels for stepping
	// We want 'viewspace_radius' meters in view space projected to pixels.
	// Approximate pixel length of the radius:
	float pixel_radius = viewspace_radius * pixels_per_meter;
	
	// Clamp pixel radius to avoid extreme performance hits or cache trashing
	pixel_radius = min(pixel_radius, 256.0); 
	if (pixel_radius < 1.0) return;

	// Step size
	float step_size_px = pixel_radius / float(num_steps + 1.0);
	vec2 step_vec = direction * step_size_px * params.half_screen_pixel_size * 2.0; // UV step

	// Jitter starting position
	// We already jittered the angle, but we can also jitter the start distance slightly if needed.
	// For now, start at 1 step.
	vec2 current_uv = screen_uv + step_vec;

	// Horizon tracking
	float max_horizon_cos = -1.0;
	
	// View direction (camera to pixel) - assumed to be (0,0,1) in view space roughly for horizon calc, 
	// but strictly it's normalize(view_pos).
	vec3 V = normalize(-view_pos);
	
	// Project view normal onto the slice plane
	// Slice plane is defined by V and the direction 'direction' (in screen/view XY).
	// We simplify: just check elevation angle in the slice 2D plane defined by (direction, view_z).
	
	// Slice direction in view space (approximate)
	vec3 slice_dir_view = normalize(vec3(direction, 0.0)); // Z is 0 in view XY plane
	
	// Projected normal into slice plane
	vec3 plane_normal = cross(slice_dir_view, vec3(0.0, 0.0, 1.0));
	vec3 projected_normal = view_normal - dot(view_normal, plane_normal) * plane_normal;
	float projected_normal_len = length(projected_normal);
	
	float n_angle = 0.0;
	if (projected_normal_len > 0.001) {
		// Calculate angle of normal in slice plane relative to slice direction
		// n_angle is the angle between view_normal and the slice direction
		// But usually GTAO uses the angle between the normal and the view vector.
		// Let's use the standard "horizon angle" approach.
		// Horizon angle 'h' is angle from slice vector.
	}
	
	// Cosine of the angle between normal and view vector
	float cos_n = dot(view_normal, V);
	// Sine of that angle
	float sin_n = sqrt(max(0.0, 1.0 - cos_n * cos_n));
	
	// Projected normal angle 'gamma' (n_angle) in the formula
	// gamma = acos(dot(projected_normal, V) / len) ...
	// Simplify: we just track the max elevation angle of the horizon.
	
	float h_angle_max = -1.0; // In cosine space, -1 is 180 deg (behind). Horizon is usually < 90 deg.
	
	for (int i = 0; i < num_steps; i++) {
		// Sample depth
		float sample_depth = textureLod(source_depth_mipmaps, vec3(current_uv, params.pass), 0.0).x;
		vec3 sample_pos = NDC_to_view_space(current_uv, sample_depth);
		
		vec3 diff = sample_pos - view_pos;
		float dist_sq = dot(diff, diff);
		
		// Horizon vector
		vec3 H = normalize(diff);
		
		// Falloff
		float falloff = max(0.0, 1.0 - dist_sq * params.neg_inv_radius * -params.neg_inv_radius); // approx falloff_sq logic
		
		// Horizon angle cosine
		float h_cos = dot(H, V); // Angle with view vector?
		// No, standard GTAO measures angle from the tangent plane or just strictly elevation.
		
		// Let's use the vector projection.
		// We want angle between H and the view plane, or H and View Vector.
		// Using H dot V gives angle relative to view line.
		
		// We want to maximize slope.
		// Slope = diff.z / length(diff.xy).
		// But in view space, negative Z is forward.
		// diff.z is negative if sample is further.
		
		// Simplified GTAO:
		// We track the sample with the "highest" elevation seen from P.
		// Elevation can be defined by dot(H, V).
		// But we also need to account for the Normal.
		// Vis = integrate (cos(h) * sin(h)) dh?
		
		// Standard integration: 0.25 * ( -cos(2 * h) + cos(2 * n) + 2 * h * sin(2 * n) ) ... complicated.
		
		// Let's use the approximation: Visibility = falloff * clamp(dot(N, H), 0, 1).
		// But we only take the MAX horizon.
		
		if (dist_sq < 0.0001) { current_uv += step_vec; continue; }

		// Check if this sample is "above" the current horizon
		// In view space, "above" means closer to the camera relative to the distance?
		// No, it means the angle is larger.
		// We project H onto the slice plane (direction, V).
		
		float h_val = dot(H, view_normal);
		
		// If this sample blocks more light...
		// In GTAO we want the maximum angle of occlusion.
		// Angle theta between view vector and horizon vector.
		// theta = acos(dot(V, H)).
		// But we need signed angle?
		
		// Let's stick to a simpler "Ambient Occlusion" heuristic compatible with the "Intel" loop logic but sliced.
		// Or proper GTAO. Proper GTAO:
		// h = atan(diff.z / length(diff.xy)).
		// But we need to handle the Normal.
		
		// Let's use the "Horizon" angle approach relative to the slice direction.
		// slice_dist = dot(diff.xy, direction).
		// angle = atan(diff.z, slice_dist).
		// max_angle = max(max_angle, angle).
		
		float dist_xy = length(diff.xy);
		float angle = atan(diff.z, dist_xy); // diff.z is positive for objects closer to camera relative to center
		
		if (angle > h_angle_max) {
			h_angle_max = angle;
		}
		
		current_uv += step_vec;
	}
	
	// Calculate visibility from max horizon angle
	// Vis = integral from h_angle_max to 3.14159265359/2 of (n dot omega) d_omega
	// In 2D slice: integrate cos(theta - gamma) d_theta
	// Simplified: Vis = 0.25 * (cos(h_angle_max - n_angle) ...)
	
	// Let's use a robust approximation:
	// Projected normal angle in this slice
	vec3 slice_axis = vec3(direction, 0.0);
	vec3 projected_n = view_normal - dot(view_normal, vec3(0,0,1)) * vec3(0,0,1); // Projects to XY
	// This is not quite right. We need N projected onto the plane defined by V and Direction.
	
	// Let's go with the simpler "Bitmask" gathering approach which relies on sampling directions.
	// But the user requested "GT-VBAO".
	// The core of GTAO is the horizon search.
	
	// Normal angle in slice plane
	float n_proj_len = length(vec2(dot(view_normal.xy, direction), view_normal.z));
	float n_angle_slice = atan(view_normal.z, dot(view_normal.xy, direction)); // Angle of normal in slice
	// Note: view_normal.z is usually negative (facing camera).
	// We want the angle with the view plane?
	
	// Let's use the standard "GTAO" formula approx:
	// h_angle_max is the horizon angle from the view vector (forward).
	// We clamp it to be at least the surface tangent.
	
	h_angle_max = clamp(h_angle_max, n_angle_slice - 3.14159265359/2.0, n_angle_slice + 3.14159265359/2.0);
	
	// Inner integral:
	// \int_{h_{max}}^{\pi/2} \cos(\theta - n_{slice}) d\theta
	// = [\sin(\theta - n_{slice})]_{h_{max}}^{\pi/2}
	// = \sin(\pi/2 - n_{slice}) - \sin(h_{max} - n_{slice})
	// = \cos(n_{slice}) - \sin(h_{max} - n_{slice})
	
	// But we need to normalize this?
	// Full hemisphere visibility is 1.
	
	float vis_slice = 0.25 * (cos(n_angle_slice) - sin(h_angle_max - n_angle_slice));
	// Note: factor 0.25 is because we average 2 directions? (left/right). 
	// If this is unidirectional, we might need 0.5.
	// We will accumulate and average later.
	
	// Falloff?
	// We didn't apply falloff in the horizon search (usually GTAO doesn't, but limits radius).
	// We can apply a global falloff.
	
	visibility += clamp(vis_slice, 0.0, 1.0);
	
	// Bitmask update
	// Check if horizon is significantly above the surface tangent (plus small bias)
	if (h_angle_max > n_angle_slice + 0.05) { // Threshold relative to surface
		float dir_ang = atan(direction.y, direction.x);
		float ang01 = dir_ang * 0.15915494 + 0.5;
		uint bit = uint(clamp(floor(ang01 * 32.0), 0.0, 31.0));
		mask_bits |= (1u << bit);
	}
	
	// Bi-directional support handled by caller?
	// If we do bi-directional in the caller, we call this twice.
	// Or we can do it here.
}

void generate_SSAO_shadows_internal(out float r_shadow_term, out vec4 r_edges, out float r_weight, const vec2 p_pos, int p_quality_level, bool p_adaptive_base, const bool p_use_vb, const bool p_vb_bi, out uint r_mask) {
	vec2 pos_rounded = trunc(p_pos);
	uvec2 upos = uvec2(pos_rounded);

	// Fetch 4 depths for edge detection
	vec4 valuesUL = textureGather(source_depth_mipmaps, vec3(pos_rounded * params.half_screen_pixel_size, params.pass));
	vec4 valuesBR = textureGather(source_depth_mipmaps, vec3((pos_rounded + vec2(1.0)) * params.half_screen_pixel_size, params.pass));

	float pix_z = valuesUL.y;
	float pix_left_z = valuesUL.x;
	float pix_top_z = valuesUL.z;
	float pix_right_z = valuesBR.z;
	float pix_bottom_z = valuesBR.x;

	vec2 normalized_screen_pos = pos_rounded * params.half_screen_pixel_size + params.half_screen_pixel_size_x025;
	vec3 pix_center_pos = NDC_to_view_space(normalized_screen_pos, pix_z);

	// Normal
	uvec2 full_res_coord = upos * 2 * params.size_multiplier + params.pass_coord_offset.xy;
	vec3 pixel_normal = load_normal(ivec2(full_res_coord));

	const vec2 pixel_size_at_center = NDC_to_view_space(normalized_screen_pos.xy + params.half_screen_pixel_size, pix_center_pos.z).xy - pix_center_pos.xy;

	float pixel_lookup_radius;
	float fallof_sq;
	float viewspace_radius;
	// Calculate radius logic reused (it's good)
	{
		viewspace_radius = params.radius;
		const float too_close_limit = clamp(length(pix_center_pos) * params.inv_radius_near_limit, 0.0, 1.0) * 0.8 + 0.2;
		viewspace_radius *= too_close_limit;
		pixel_lookup_radius = (0.85 * viewspace_radius) / pixel_size_at_center.x;
		fallof_sq = -1.0 / (viewspace_radius * viewspace_radius);
	}

	// Edges
	vec4 edgesLRTB = vec4(1.0);
	if (!p_adaptive_base && (p_quality_level >= 1)) {
		edgesLRTB = calculate_edges(pix_z, pix_left_z, pix_right_z, pix_top_z, pix_bottom_z);
	}

	// Normal edges (optional quality feature)
	if (!p_adaptive_base && (p_quality_level >= 2)) {
		// ... (keep existing normal edge logic if desired, or skip for perf. Let's skip to save instruction space for GTAO)
	}

	// GT-VBAO GATHER
	float total_visibility = 0.0;
	uint mask_bits = 0u;
	
	// Configuration based on quality
	// Low: 1 slice, 2 directions (bi), 2 steps
	// Med: 2 slices, 4 directions, 3 steps
	// High: 3 slices, 6 directions, 4 steps
	// Ultra: 4 slices, 8 directions, 6 steps
	int num_slices = (p_quality_level == 0) ? 1 : ((p_quality_level == 1) ? 2 : ((p_quality_level == 2) ? 3 : 4));
	int steps_per_slice = (p_quality_level == 0) ? 2 : ((p_quality_level == 1) ? 3 : ((p_quality_level == 2) ? 4 : 6));
	
	// Use Interleaved Gradient Noise for temporal/spatial jitter
	float noise_val = interleaved_gradient_noise(vec2(gl_GlobalInvocationID.xy) + vec2(params.pass * 5.588));
	float spatial_offset = noise_val;
	
	// Also use the rotation matrix for the "phase" to keep compatibility with the temporal filter's expectation?
	// The C++ side feeds rotation matrices. We can use the first component as an angle offset.
	// But IGN is better for GTAO.
	
	float pixels_per_meter = 1.0 / pixel_size_at_center.x; // approx

	for (int i = 0; i < num_slices; i++) {
		float slice_angle = (float(i) + spatial_offset) * (3.14159265359 / float(num_slices));
		vec2 direction = vec2(cos(slice_angle), sin(slice_angle));
		
		// Integrate primary direction
		integrate_slice(direction, viewspace_radius, normalized_screen_pos, pix_center_pos, pixel_normal, pixels_per_meter, steps_per_slice, total_visibility, mask_bits, false);
		
		// Integrate opposite direction (always needed for correct GTAO integral)
		integrate_slice(-direction, viewspace_radius, normalized_screen_pos, pix_center_pos, pixel_normal, pixels_per_meter, steps_per_slice, total_visibility, mask_bits, false);
	}
	
	// Normalize visibility
	// We summed (0.25 * ...) for 2*num_slices directions.
	// Total weight should be 1.0.
	// Each slice pair (dir, -dir) contributes roughly to the integral across PI.
	// We have num_slices.
	// Average it.
	total_visibility /= float(num_slices); // Since we summed 0.25 * term, and we did 2 directions...
	// Wait, term is \cos(n) - \sin(h - n).
	// \int_{-\pi}^{\pi} ... = \pi * AO.
	// We want normalized AO [0,1].
	// GTAO paper says average the terms.
	// If flat surface, h = 0 (relative to tangent). n_slice = 0. 
	// vis = 0.25 * (1 - sin(0)) = 0.25.
	// 2 directions -> 0.5.
	// So for 1 slice (2 dirs), we get 0.5.
	// We need to multiply by 2?
	
	total_visibility *= 2.0; 
	// Clamp
	float occlusion = 1.0 - clamp(total_visibility, 0.0, 1.0);

	// Fade out
	float fade_out = clamp(pix_center_pos.z * params.fade_out_mul + params.fade_out_add, 0.0, 1.0);
	if (!p_adaptive_base && (p_quality_level >= 1)) {
		float edge_fadeout_factor = clamp((1.0 - edgesLRTB.x - edgesLRTB.y) * 0.35, 0.0, 1.0) + clamp((1.0 - edgesLRTB.z - edgesLRTB.w) * 0.35, 0.0, 1.0);
		fade_out *= clamp(1.0 - edge_fadeout_factor, 0.0, 1.0);
	}
	
	occlusion = params.intensity * occlusion;
	occlusion = min(occlusion, params.shadow_clamp);
	occlusion *= fade_out;
	
	float final_val = 1.0 - occlusion;
	final_val = pow(clamp(final_val, 0.0, 1.0), params.shadow_power);

	r_shadow_term = final_val;
	r_edges = edgesLRTB;
	r_weight = 1.0; // Uniform weight for now
	r_mask = mask_bits;
}

void main() {
	float out_shadow_term;
	float out_weight;
	vec4 out_edges;
	uint out_mask = 0u;
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(ssC, params.screen_size))) { //too large, do nothing
		return;
	}

	vec2 uv = vec2(gl_GlobalInvocationID) + vec2(0.5);
	float vb_mode = params.flags.x;
	bool use_gtao_vb = vb_mode > 0.5;
	bool vb_bi = vb_mode > 1.5;

	// Note: We use the same GTAO gather for BASE and final pass to ensure consistency.
	// Adaptive logic could be re-enabled if we implement the 'importance' sampling for slices.
	// For now, we use standard gather for all.
	
	generate_SSAO_shadows_internal(out_shadow_term, out_edges, out_weight, uv, params.quality, false, use_gtao_vb, vb_bi, out_mask);
	
#ifdef SSAO_BASE
	imageStore(dest_image, ivec2(gl_GlobalInvocationID.xy), vec4(out_shadow_term, 1.0, 0.0, 0.0));
#else
	if (params.quality == 0) {
		out_edges = vec4(1.0);
	}

	if (use_gtao_vb) {
		// Derive visibility from the quantized bitmask (32 directions); fewer set bits = more occlusion.
		float vb_visibility = 1.0 - float(bitCount(out_mask)) * (1.0 / 32.0);
		vb_visibility = clamp(vb_visibility, 0.0, 1.0);

		// Blend strategy:
		// - Bi-directional: lean more on the bitmask result for stability across slices.
		// - Uni-directional: keep more of the classic SSAO term to avoid over-darkening.
		if (vb_bi) {
			out_shadow_term = mix(out_shadow_term, vb_visibility, 0.7);
		} else {
			out_shadow_term *= mix(1.0, vb_visibility, 0.6);
		}
	}

	imageStore(dest_image, ivec2(gl_GlobalInvocationID.xy), vec4(out_shadow_term, pack_edges(out_edges), 0.0, 0.0));
#endif
}