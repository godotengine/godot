#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#ifdef MODE_PROCESS
layout(rgba16f, set = 0, binding = 0) uniform restrict readonly image3D density_map;
layout(rgba16f, set = 0, binding = 1) uniform restrict writeonly image3D fog_map;
#endif

#ifdef MODE_FILTER
layout(rgba16f, set = 0, binding = 0) uniform restrict readonly image3D source_map;
layout(rgba16f, set = 0, binding = 1) uniform restrict writeonly image3D dest_map;
#endif

layout(push_constant, binding = 2, std430) uniform Params {
	float fog_frustum_end;
	float detail_spread;
	vec2 pad;

	ivec3 fog_volume_size;
	int filter_axis;
}
params;

float get_depth_at_pos(float cell_depth_size, int z) {
	float d = float(z) * cell_depth_size + cell_depth_size * 0.5; //center of voxels
	d = pow(d, params.detail_spread);
	return params.fog_frustum_end * d;
}

void main() {
	vec3 fog_cell_size = 1.0 / vec3(params.fog_volume_size);

#ifdef MODE_PROCESS

	ivec3 pos = ivec3(gl_GlobalInvocationID.xy, 0);

	if (any(greaterThanEqual(pos, params.fog_volume_size))) {
		return; //do not compute
	}

	vec4 fog_accum = vec4(0.0);
	float prev_z = 0.0;

	float t = 1.0;

	for (int i = 0; i < params.fog_volume_size.z; i++) {
		//compute fog position
		ivec3 fog_pos = pos + ivec3(0, 0, i);
		//get fog value
		vec4 fog = imageLoad(density_map, fog_pos);

		//get depth at cell pos
		float z = get_depth_at_pos(fog_cell_size.z, i);
		//get distance from previous pos
		float d = abs(prev_z - z);
		//compute exinction based on beer's
		float extinction = t * exp(-d * fog.a);
		//compute alpha based on different of extinctions
		float alpha = t - extinction;
		//update extinction
		t = extinction;

		fog_accum += vec4(fog.rgb * alpha, alpha);
		prev_z = z;

		vec4 fog_value;

		if (fog_accum.a > 0.0) {
			fog_value = vec4(fog_accum.rgb / fog_accum.a, 1.0 - t);
		} else {
			fog_value = vec4(0.0);
		}

		imageStore(fog_map, fog_pos, fog_value);
	}

#endif

#ifdef MODE_FILTER

	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);

	const float gauss[7] = float[](0.071303, 0.131514, 0.189879, 0.214607, 0.189879, 0.131514, 0.071303);

	const ivec3 filter_dir[3] = ivec3[](ivec3(1, 0, 0), ivec3(0, 1, 0), ivec3(0, 0, 1));
	ivec3 offset = filter_dir[params.filter_axis];

	vec4 accum = vec4(0.0);
	for (int i = -3; i <= 3; i++) {
		accum += imageLoad(source_map, clamp(pos + offset * i, ivec3(0), params.fog_volume_size - ivec3(1))) * gauss[i + 3];
	}

	imageStore(dest_map, pos, accum);

#endif
}
