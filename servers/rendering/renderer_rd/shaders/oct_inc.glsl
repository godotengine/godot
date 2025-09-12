
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

vec2 vec3_to_oct_with_gradient(vec3 n, bool gradient) {
	// Reference: https://twitter.com/Stubbesaurus/status/937994790553227264
	// The gradient argument allows the shader to compute the gradient for the function without introducing seams.
	// It is meant to be used with dFdx/dFdy and not as an actual UV coordinate for sampling when gradient is set to true.
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = (n.z >= 0.0 || gradient) ? n.xy : oct_wrap(n.xy);
	n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
}

vec2 vec3_to_oct(vec3 n) {
	return vec3_to_oct_with_gradient(n, false);
}

vec2 vec3_to_oct_with_border_and_gradient(vec3 n, vec2 border_size, bool gradient) {
	vec2 uv = vec3_to_oct_with_gradient(n, gradient);
	vec2 texture_size = vec2(1.0) - border_size * 3.0;
	return uv * texture_size + border_size * 1.5;
}

vec2 vec3_to_oct_with_border(vec3 n, vec2 border_size) {
	return vec3_to_oct_with_border_and_gradient(n, border_size, false);
}
