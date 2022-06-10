#ifdef ALPHA_HASH_USED

float hash_2d(vec2 p) {
	return fract(1.0e4 * sin(17.0 * p.x + 0.1 * p.y) *
			(0.1 + abs(sin(13.0 * p.y + p.x))));
}

float hash_3d(vec3 p) {
	return hash_2d(vec2(hash_2d(p.xy), p.z));
}

// Interleaved Gradient Noise
// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
float compute_alpha_hash_threshold(vec2 pos) {
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	return fract(magic.z * fract(dot(pos, magic.xy)));
}

#endif // ALPHA_HASH_USED

#ifdef ALPHA_ANTIALIASING_EDGE_USED

float calc_mip_level(vec2 texture_coord) {
	vec2 dx = dFdx(texture_coord);
	vec2 dy = dFdy(texture_coord);
	float delta_max_sqr = max(dot(dx, dx), dot(dy, dy));
	return max(0.0, 0.5 * log2(delta_max_sqr));
}

float compute_alpha_antialiasing_edge(float input_alpha, vec2 texture_coord, float alpha_edge) {
	input_alpha *= 1.0 + max(0, calc_mip_level(texture_coord)) * 0.25; // 0.25 mip scale, magic number
	input_alpha = (input_alpha - alpha_edge) / max(fwidth(input_alpha), 0.0001) + 0.5;
	return clamp(input_alpha, 0.0, 1.0);
}

#endif // ALPHA_ANTIALIASING_USED
