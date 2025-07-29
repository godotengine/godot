shader_type spatial;

global uniform vec3 offset;
instance uniform vec3 emission : source_color = vec3(1.0, 0.0, 0.0);
uniform vec3 color : source_color = vec3(1.0, 0.0, 0.0);

void vertex() {
	VERTEX.xyz += offset;
}

void fragment() {
	ALBEDO = color.rgb;
	EMISSION = emission.rgb;
}
