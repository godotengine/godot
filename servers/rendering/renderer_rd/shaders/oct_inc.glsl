
vec3 oct_to_vec3(vec2 e) {
	vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
	float t = max(-v.z, 0.0);
	v.xy += t * -sign(v.xy);
	return normalize(v);
}

vec3 oct_to_vec3_with_border(vec2 uv, vec2 pixel_size) {
	if (uv.x < pixel_size.x || uv.x > (1.0 - pixel_size.x)) {
		// Flip vertically if it's the left or right border.
		uv.y = 1.0 - uv.y;
	}

	if (uv.y < pixel_size.y || uv.y > (1.0 - pixel_size.y)) {
		// Flip horizontally if it's the top or bottom border.
		uv.x = 1.0 - uv.x;
	}

	vec2 texture_size = vec2(1.0) - pixel_size * 3.0;
	uv -= pixel_size * 1.5;
	uv /= texture_size;
	return oct_to_vec3(clamp(uv * 2.0 - 1.0, -1.0, 1.0));
}

vec2 oct_wrap(vec2 v) {
	vec2 signVal;
	signVal.x = v.x >= 0.0 ? 1.0 : -1.0;
	signVal.y = v.y >= 0.0 ? 1.0 : -1.0;
	return (1.0 - abs(v.yx)) * signVal;
}

vec2 vec3_to_oct(vec3 n) {
	// Reference: https://twitter.com/Stubbesaurus/status/937994790553227264
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = (n.z >= 0.0) ? n.xy : oct_wrap(n.xy);
	n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
}

vec2 vec3_to_oct_with_border(vec3 n, vec2 border_size) {
	vec2 uv = vec3_to_oct(n);
	vec2 texture_size = vec2(1.0) - border_size * 3.0;
	return uv * texture_size + border_size * 1.5;
}

float vec3_to_oct_lod(vec3 n_ddx, vec3 n_ddy, vec2 pixel_size) {
	// Approximate UV space derivatives by a factor of 0.5 because
	// vec3_to_oct maps from [-1,1] to [0,1].
	vec2 pixel_size_sqr = 4.0 * pixel_size * pixel_size;
	float ddx = dot(n_ddx, n_ddx) / pixel_size_sqr.x;
	float ddy = dot(n_ddy, n_ddy) / pixel_size_sqr.y;
	float dd_sqr = max(ddx, ddy);
	return 0.25 * log2(dd_sqr + 1e-6f);
}
