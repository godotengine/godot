
vec3 srgb_to_linear(vec3 color) {
	// This expects 0-1 range input, outside that range it behaves poorly.
	// Approximation from http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
	return color * (color * (color * 0.305306011 + 0.682171111) + 0.012522878);
}

vec3 linear_to_srgb(vec3 color) {
	//if going to srgb, clamp from 0 to 1.
	color = clamp(color, vec3(0.0), vec3(1.0));
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
}

vec3 linear_to_st2084(vec3 color, float max_luminance) {
	color = clamp(color, vec3(0.0), vec3(1.0));

	// Keep the output from blowing out the display
	// max_luminance is the display's peak luminance in nits
	// we map it here to the native 10000 nits range of ST2084
	float adjustment = max_luminance * (1.0f / 10000.0f);

	// Apply ST2084 curve
	const float c1 = 0.8359375;
	const float c2 = 18.8515625;
	const float c3 = 18.6875;
	const float m1 = 0.1593017578125;
	const float m2 = 78.84375;
	vec3 cp = pow(abs(color.rgb * adjustment), vec3(m1));

	return pow((c1 + c2 * cp) / (1 + c3 * cp), vec3(m2));
}
