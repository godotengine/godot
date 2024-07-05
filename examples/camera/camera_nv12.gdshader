shader_type canvas_item;

void fragment() {
	vec3 color;
	color.r = texture(TEXTURE, UV).r;
	color.gb = texture(NORMAL_TEXTURE, UV).rg - vec2(0.5, 0.5);
	
	color.rgb = mat3(
					vec3(1.00000,  1.00000, 1.00000),
					vec3(0.00000, -0.18732, 1.85560),
					vec3(1.57481, -0.46813, 0.00000)
				) * color.rgb;
	
	COLOR = vec4(color, 1.0);
}
