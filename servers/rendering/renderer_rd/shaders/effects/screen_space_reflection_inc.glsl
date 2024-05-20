layout(constant_id = 0) const bool sc_multiview = false;

layout(set = 4, binding = 0, std140) uniform SceneData {
	mat4x4 projection[2];
	mat4x4 inv_projection[2];
	vec4 eye_offset[2];
}
scene_data;

vec3 reconstructCSPosition(vec2 screen_pos, float z) {
	if (sc_multiview) {
		vec4 pos;
		pos.xy = (2.0 * vec2(screen_pos) / vec2(params.screen_size)) - 1.0;
		pos.z = z * 2.0 - 1.0;
		pos.w = 1.0;

		pos = scene_data.inv_projection[params.view_index] * pos;
		pos.xyz /= pos.w;

		return pos.xyz;
	} else {
		if (params.orthogonal) {
			return vec3((screen_pos.xy * params.proj_info.xy + params.proj_info.zw), z);
		} else {
			return vec3((screen_pos.xy * params.proj_info.xy + params.proj_info.zw) * z, z);
		}
	}
}
