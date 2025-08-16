#ifdef ALPHA_HASH_USED

float hash_2d(vec2 p) {
	return fract(1.0e4 * sin(17.0 * p.x + 0.1 * p.y) *
			(0.1 + abs(sin(13.0 * p.y + p.x))));
}

float hash_3d(vec3 p) {
	return hash_2d(vec2(hash_2d(p.xy), p.z));
}

half compute_alpha_hash_threshold(vec3 pos, float hash_scale) {
	vec3 dx = dFdx(pos);
	vec3 dy = dFdy(pos);

	float delta_max_sqr = max(length(dx), length(dy));
	float pix_scale = 1.0 / (hash_scale * delta_max_sqr);

	vec2 pix_scales =
			vec2(exp2(floor(log2(pix_scale))), exp2(ceil(log2(pix_scale))));

	vec2 a_thresh = vec2(hash_3d(floor(pix_scales.x * pos.xyz)),
			hash_3d(floor(pix_scales.y * pos.xyz)));

	float lerp_factor = fract(log2(pix_scale));

	float a_interp = (1.0 - lerp_factor) * a_thresh.x + lerp_factor * a_thresh.y;

	float min_lerp = min(lerp_factor, 1.0 - lerp_factor);

	vec3 cases = vec3(a_interp * a_interp / (2.0 * min_lerp * (1.0 - min_lerp)),
			(a_interp - 0.5 * min_lerp) / (1.0 - min_lerp),
			1.0 - ((1.0 - a_interp) * (1.0 - a_interp) / (2.0 * min_lerp * (1.0 - min_lerp))));

	float alpha_hash_threshold =
			(a_interp < (1.0 - min_lerp)) ? ((a_interp < min_lerp) ? cases.x : cases.y) : cases.z;

	return half(clamp(alpha_hash_threshold, 0.00001, 1.0));
}

#endif // ALPHA_HASH_USED

#ifdef ALPHA_ANTIALIASING_EDGE_USED

half calc_mip_level(vec2 texture_coord) {
	vec2 dx = dFdx(texture_coord);
	vec2 dy = dFdy(texture_coord);
	float delta_max_sqr = max(dot(dx, dx), dot(dy, dy));
	return max(half(0.0), half(0.5) * half(log2(delta_max_sqr)));
}

half compute_alpha_antialiasing_edge(half input_alpha, vec2 texture_coord, half alpha_edge) {
	input_alpha *= half(1.0) + calc_mip_level(texture_coord) * half(0.25);
	input_alpha = (input_alpha - alpha_edge) / max(fwidth(input_alpha), half(0.0001)) + half(0.5);
	return clamp(input_alpha, half(0.0), half(1.0));
}

#endif // ALPHA_ANTIALIASING_USED
