// Copyright © 2024 Cory Petkovsek, Roope Palmroos, and Contributors.

// This shader reads the screen and returns absolute depth encoded in Albedo.rg
// It is not used as an INSERT

R"(
shader_type spatial;
render_mode unshaded;

uniform sampler2D depth_texture : source_color, hint_depth_texture, filter_nearest, repeat_disable;

uniform float camera_far = 100000.0;

// Mobile renderer HDR mode has limit of 1 or 2. Pack full range depth to RG
// https://gamedev.stackexchange.com/questions/201151/24bit-float-to-rgb
vec3 encode_rg(float value) {
    vec2 kEncodeMul = vec2(1.0, 255.0);
    float kEncodeBit = 1.0 / 255.0;
    vec2 color = kEncodeMul * value / camera_far;
    color = fract(color);
    color.x -= color.y * kEncodeBit;
	return vec3(color, 0.);
}
	
void fragment() {
	float depth = textureLod(depth_texture, SCREEN_UV, 0.).x;
	vec3 ndc = vec3(SCREEN_UV * 2.0 - 1.0, depth);
	vec4 view = INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
	view.xyz /= view.w;
	float depth_linear = -view.z;
	ALBEDO = encode_rg(depth_linear); // Encoded value for Mobile
}

)"