/* clang-format off */
#[modes]

mode_default =

#[specializations]

#[vertex]

#define M_PI 3.14159265359

#if defined(TANGENT_USED)
#ifndef NORMAL_USED
#define NORMAL_USED
#endif
#endif


layout(std140) uniform GlobalShaderUniformData { //ubo:0
	vec4 global_shader_uniforms[MAX_GLOBAL_SHADER_UNIFORMS];
};

/* clang-format on */

/* INPUT ATTRIBS */

// Always contains vertex position in XYZ, can contain tangent angle in W.
layout(location = 0) in vec4 vertex_angle_attrib;

//only for pure render depth when normal is not used

#ifdef NORMAL_USED
// Contains Normal/Axis in RG, can contain tangent in BA.
layout(location = 1) in vec4 axis_tangent_attrib;
#endif

// Location 2 is unused.

#if defined(COLOR_USED)
layout(location = 3) in vec4 color_attrib;
#endif

#ifdef UV_USED
layout(location = 4) in vec2 uv_attrib;
#endif

#if defined(UV2_USED)
layout(location = 5) in vec2 uv2_attrib;
#endif

#if defined(CUSTOM0_USED)
layout(location = 6) in vec4 custom0_attrib;
#endif

#if defined(CUSTOM1_USED)
layout(location = 7) in vec4 custom1_attrib;
#endif

#if defined(CUSTOM2_USED)
layout(location = 8) in vec4 custom2_attrib;
#endif

#if defined(CUSTOM3_USED)
layout(location = 9) in vec4 custom3_attrib;
#endif

#if defined(BONES_USED)
layout(location = 10) in uvec4 bone_attrib;
#endif

#if defined(WEIGHTS_USED)
layout(location = 11) in vec4 weight_attrib;
#endif

vec3 oct_to_vec3(vec2 e) {
	vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
	float t = max(-v.z, 0.0);
	v.xy += t * -sign(v.xy);
	return normalize(v);
}

void axis_angle_to_tbn(vec3 axis, float angle, out vec3 tangent, out vec3 binormal, out vec3 normal) {
	float c = cos(angle);
	float s = sin(angle);
	vec3 omc_axis = (1.0 - c) * axis;
	vec3 s_axis = s * axis;
	tangent = omc_axis.xxx * axis + vec3(c, -s_axis.z, s_axis.y);
	binormal = omc_axis.yyy * axis + vec3(s_axis.z, c, -s_axis.x);
	normal = omc_axis.zzz * axis + vec3(-s_axis.y, s_axis.x, c);
}

void _unpack_vertex_attributes(vec4 p_vertex_in,
#if defined(NORMAL_USED) || defined(TANGENT_USED)
		vec4 p_normal_in,
#ifdef NORMAL_USED
		out vec3 r_normal,
#endif
		out vec3 r_tangent,
		out vec3 r_binormal,
#endif
		out vec3 r_vertex) {

	r_vertex = vertex_angle_attrib.xyz;

#ifdef NORMAL_USED
	r_normal = oct_to_vec3(p_normal_in.xy * 2.0 - 1.0);
#endif

#if defined(NORMAL_USED) || defined(TANGENT_USED)

	float binormal_sign;

	// This works because the oct value (0, 1) maps onto (0, 0, -1) which encodes to (1, 1).
	// Accordingly, if p_normal_in.z contains octahedral values, it won't equal (0, 1).
	if (p_normal_in.z > 0.0 || p_normal_in.w < 1.0) {
		// Uncompressed format.
		vec2 signed_tangent_attrib = p_normal_in.zw * 2.0 - 1.0;
		r_tangent = oct_to_vec3(vec2(signed_tangent_attrib.x, abs(signed_tangent_attrib.y) * 2.0 - 1.0));
		binormal_sign = sign(signed_tangent_attrib.y);
		r_binormal = normalize(cross(r_normal, r_tangent) * binormal_sign);
	} else {
		// Compressed format.
		float angle = p_vertex_in.w;
		binormal_sign = angle > 0.5 ? 1.0 : -1.0; // 0.5 does not exist in UNORM16, so values are either greater or smaller.
		angle = abs(angle * 2.0 - 1.0) * M_PI; // 0.5 is basically zero, allowing to encode both signs reliably.
		vec3 axis = r_normal;
		axis_angle_to_tbn(axis, angle, r_tangent, r_binormal, r_normal);
		r_binormal *= binormal_sign;
	}
#endif
}

/* Varyings */

out vec3 vertex_interp;

#ifdef NORMAL_USED
out vec3 normal_interp;
#endif

#if defined(COLOR_USED)
out vec4 color_interp;
#endif

#ifdef UV_USED
out vec2 uv_interp;
#endif

#if defined(UV2_USED)
out vec2 uv2_interp;
#endif

#if defined(TANGENT_USED)
out vec3 tangent_interp;
out vec3 binormal_interp;
#endif

#ifdef MATERIAL_UNIFORMS_USED
/* clang-format off */
layout(std140) uniform MaterialUniforms { // ubo:1

#MATERIAL_UNIFORMS

};
/* clang-format on */
#endif

#GLOBALS

invariant gl_Position;

void main() {
#if defined(NORMAL_USED) || defined(TANGENT_USED)
#if defined(TANGENT_USED)
	vec3 binormal = binormal_interp;
	vec3 tangent = tangent_interp;
#else
	vec3 binormal = vec3(0.0);
	vec3 tangent = vec3(0.0);
#endif
#endif

	_unpack_vertex_attributes(
			vertex_angle_attrib,
#if defined(NORMAL_USED) || defined(TANGENT_USED)
			axis_tangent_attrib,
#ifdef NORMAL_USED
			normal_interp,
#endif
			tangent,
			binormal,
#endif
			vertex_interp);

#ifdef COLOR_USED
	color_interp = color_attrib;
#endif
#ifdef UV_USED
	uv_interp = uv_attrib;
#endif
#ifdef UV2_USED
	uv2_interp = uv2_attrib;
#endif

#if defined(OVERRIDE_POSITION)
	vec4 position = vec4(vertex_interp, 1.0);
	position.y = -position.y;
#endif

	{
#CODE : VERTEX
	}

#ifdef NORMAL_USED
	normal_interp = normalize(normal_interp);
#endif

#ifdef TANGENT_USED
	tangent_interp = normalize(tangent);
	binormal_interp = normalize(binormal);
#endif

#if defined(OVERRIDE_POSITION)
	gl_Position = position;
#else
	gl_Position = vec4(vertex_interp, 1.0);
	gl_Position.y = -gl_Position.y;
#endif
	// Remap z ([0,w]->[-w,w]) to be consistent with vulkan.
	gl_Position.z = (gl_Position.z * 2.0 - 1.0) * gl_Position.w;
}

#[fragment]

layout(std140) uniform GlobalShaderUniformData { //ubo:0
	vec4 global_shader_uniforms[MAX_GLOBAL_SHADER_UNIFORMS];
};

/* Varyings */

in vec3 vertex_interp;

#ifdef NORMAL_USED
in vec3 normal_interp;
#endif

#if defined(COLOR_USED)
in vec4 color_interp;
#endif

#ifdef UV_USED
in vec2 uv_interp;
#endif

#if defined(UV2_USED)
in vec2 uv2_interp;
#endif

#if defined(TANGENT_USED)
in vec3 tangent_interp;
in vec3 binormal_interp;
#endif

#ifdef MATERIAL_UNIFORMS_USED
/* clang-format off */
layout(std140) uniform MaterialUniforms { // ubo:1

#MATERIAL_UNIFORMS

};
/* clang-format on */
#endif

#GLOBALS

layout(location = 0) out vec4 frag_color;

void main() {
	vec4 output_color = vec4(1.0);

	{
#CODE : FRAGMENT
	}

	frag_color = output_color;
}
