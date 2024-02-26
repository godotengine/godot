shader_type spatial;
render_mode unshaded;

uniform sampler2D lod_num_tex;


float rand(float seed)
{
    return fract((sin(seed + 78.233)) * 43758.5453);
}

void fragment() {
	ALBEDO = texture(lod_num_tex, UV).rgb;
	ALBEDO += vec3(0.3,0.3,0.0);
}
