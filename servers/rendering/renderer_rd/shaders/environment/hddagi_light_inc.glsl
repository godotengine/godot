struct Light {
	vec3 color;
	float energy;

	vec3 direction;
	bool has_shadow;

	vec3 position;
	float attenuation;

	uint type;
	float cos_spot_angle;
	float inv_spot_attenuation;
	float radius;

	vec4 area_width;
	vec4 area_height;
	vec4 area_projector_rect;
};
