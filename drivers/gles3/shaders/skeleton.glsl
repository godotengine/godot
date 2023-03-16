/* clang-format off */
#[modes]

mode_base_pass =
mode_blend_pass = #define MODE_BLEND_PASS

#[specializations]

MODE_2D = true
USE_BLEND_SHAPES = false
USE_SKELETON = false
USE_NORMAL = false
USE_TANGENT = false
FINAL_PASS = false
USE_EIGHT_WEIGHTS = false

#[vertex]

#include "stdlib_inc.glsl"

#ifdef MODE_2D
#define VFORMAT vec2
#else
#define VFORMAT vec3
#endif

#ifdef FINAL_PASS
#define OFORMAT vec2
#else
#define OFORMAT uvec2
#endif

// These come from the source mesh and the output from previous passes.
layout(location = 0) in highp VFORMAT in_vertex;
#ifdef MODE_BLEND_PASS
#ifdef USE_NORMAL
layout(location = 1) in highp uvec2 in_normal;
#endif
#ifdef USE_TANGENT
layout(location = 2) in highp uvec2 in_tangent;
#endif
#else // MODE_BLEND_PASS
#ifdef USE_NORMAL
layout(location = 1) in highp vec2 in_normal;
#endif
#ifdef USE_TANGENT
layout(location = 2) in highp vec2 in_tangent;
#endif
#endif // MODE_BLEND_PASS

#ifdef USE_SKELETON
#ifdef USE_EIGHT_WEIGHTS
layout(location = 10) in highp uvec4 in_bone_attrib;
layout(location = 11) in highp uvec4 in_bone_attrib2;
layout(location = 12) in mediump vec4 in_weight_attrib;
layout(location = 13) in mediump vec4 in_weight_attrib2;
#else
layout(location = 10) in highp uvec4 in_bone_attrib;
layout(location = 11) in mediump vec4 in_weight_attrib;
#endif

uniform mediump sampler2D skeleton_texture; // texunit:0
#endif

/* clang-format on */
#ifdef MODE_BLEND_PASS
layout(location = 3) in highp VFORMAT blend_vertex;
#ifdef USE_NORMAL
layout(location = 4) in highp vec2 blend_normal;
#endif
#ifdef USE_TANGENT
layout(location = 5) in highp vec2 blend_tangent;
#endif
#endif // MODE_BLEND_PASS

out highp VFORMAT out_vertex; //tfb:

#ifdef USE_NORMAL
flat out highp OFORMAT out_normal; //tfb:USE_NORMAL
#endif
#ifdef USE_TANGENT
flat out highp OFORMAT out_tangent; //tfb:USE_TANGENT
#endif

#ifdef USE_BLEND_SHAPES
uniform highp float blend_weight;
uniform lowp float blend_shape_count;
#endif

#ifdef USE_SKELETON
uniform mediump vec2 skeleton_transform_x;
uniform mediump vec2 skeleton_transform_y;
uniform mediump vec2 skeleton_transform_offset;

uniform mediump vec2 inverse_transform_x;
uniform mediump vec2 inverse_transform_y;
uniform mediump vec2 inverse_transform_offset;
#endif

vec2 signNotZero(vec2 v) {
	return mix(vec2(-1.0), vec2(1.0), greaterThanEqual(v.xy, vec2(0.0)));
}

vec3 oct_to_vec3(vec2 oct) {
	oct = oct * 2.0 - 1.0;
	vec3 v = vec3(oct.xy, 1.0 - abs(oct.x) - abs(oct.y));
	if (v.z < 0.0) {
		v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
	}
	return normalize(v);
}

vec2 vec3_to_oct(vec3 e) {
	e /= abs(e.x) + abs(e.y) + abs(e.z);
	vec2 oct = e.z >= 0.0f ? e.xy : (vec2(1.0f) - abs(e.yx)) * signNotZero(e.xy);
	return oct * 0.5f + 0.5f;
}

vec4 oct_to_tang(vec2 oct_sign_encoded) {
	// Binormal sign encoded in y component
	vec2 oct = vec2(oct_sign_encoded.x, abs(oct_sign_encoded.y) * 2.0 - 1.0);
	return vec4(oct_to_vec3(oct), sign(oct_sign_encoded.y));
}

vec2 tang_to_oct(vec4 base) {
	vec2 oct = vec3_to_oct(base.xyz);
	// Encode binormal sign in y component
	oct.y = oct.y * 0.5f + 0.5f;
	oct.y = base.w >= 0.0f ? oct.y : 1.0 - oct.y;
	return oct;
}

// Our original input for normals and tangents is 2 16-bit floats.
// Transform Feedback has to write out 32-bits per channel.
// Octahedral compression requires normalized vectors, but we need to store
// non-normalized vectors until the very end.
// Therefore, we will compress our normals into 16 bits using signed-normalized
// fixed point precision. This works well, because we know that each normal
// is no larger than |1| so we can normalize by dividing by the number of blend
// shapes.
uvec2 vec4_to_vec2(vec4 p_vec) {
	return uvec2(packSnorm2x16(p_vec.xy), packSnorm2x16(p_vec.zw));
}

vec4 vec2_to_vec4(uvec2 p_vec) {
	return vec4(unpackSnorm2x16(p_vec.x), unpackSnorm2x16(p_vec.y));
}

void main() {
#ifdef MODE_2D
	out_vertex = in_vertex;

#ifdef USE_BLEND_SHAPES
#ifdef MODE_BLEND_PASS
	out_vertex = in_vertex + blend_vertex * blend_weight;
#else
	out_vertex = in_vertex * blend_weight;
#endif
#ifdef FINAL_PASS
	out_vertex = normalize(out_vertex);
#endif
#endif // USE_BLEND_SHAPES

#ifdef USE_SKELETON

#define TEX(m) texelFetch(skeleton_texture, ivec2(m % 256u, m / 256u), 0)
#define GET_BONE_MATRIX(a, b, w) mat2x4(TEX(a), TEX(b)) * w

	uvec4 bones = in_bone_attrib * uvec4(2u);
	uvec4 bones_a = bones + uvec4(1u);

	highp mat2x4 m = GET_BONE_MATRIX(bones.x, bones_a.x, in_weight_attrib.x);
	m += GET_BONE_MATRIX(bones.y, bones_a.y, in_weight_attrib.y);
	m += GET_BONE_MATRIX(bones.z, bones_a.z, in_weight_attrib.z);
	m += GET_BONE_MATRIX(bones.w, bones_a.w, in_weight_attrib.w);

	mat4 skeleton_matrix = mat4(vec4(skeleton_transform_x, 0.0, 0.0), vec4(skeleton_transform_y, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(skeleton_transform_offset, 0.0, 1.0));
	mat4 inverse_matrix = mat4(vec4(inverse_transform_x, 0.0, 0.0), vec4(inverse_transform_y, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(inverse_transform_offset, 0.0, 1.0));
	mat4 bone_matrix = mat4(m[0], m[1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));

	bone_matrix = skeleton_matrix * transpose(bone_matrix) * inverse_matrix;

	out_vertex = (bone_matrix * vec4(out_vertex, 0.0, 1.0)).xy;
#endif // USE_SKELETON

#else // MODE_2D

#ifdef USE_BLEND_SHAPES
#ifdef MODE_BLEND_PASS
	out_vertex = in_vertex + blend_vertex * blend_weight;

#ifdef USE_NORMAL
	vec3 normal = vec2_to_vec4(in_normal).xyz * blend_shape_count;
	vec3 normal_blend = oct_to_vec3(blend_normal) * blend_weight;
#ifdef FINAL_PASS
	out_normal = vec3_to_oct(normalize(normal + normal_blend));
#else
	out_normal = vec4_to_vec2(vec4(normal + normal_blend, 0.0) / blend_shape_count);
#endif
#endif // USE_NORMAL

#ifdef USE_TANGENT
	vec4 tangent = vec2_to_vec4(in_tangent) * blend_shape_count;
	vec4 tangent_blend = oct_to_tang(blend_tangent) * blend_weight;
#ifdef FINAL_PASS
	out_tangent = tang_to_oct(vec4(normalize(tangent.xyz + tangent_blend.xyz), tangent.w));
#else
	out_tangent = vec4_to_vec2(vec4((tangent.xyz + tangent_blend.xyz) / blend_shape_count, tangent.w));
#endif
#endif // USE_TANGENT

#else // MODE_BLEND_PASS
	out_vertex = in_vertex * blend_weight;

#ifdef USE_NORMAL
	vec3 normal = oct_to_vec3(in_normal);
	out_normal = vec4_to_vec2(vec4(normal * blend_weight / blend_shape_count, 0.0));
#endif
#ifdef USE_TANGENT
	vec4 tangent = oct_to_tang(in_tangent);
	out_tangent = vec4_to_vec2(vec4(tangent.rgb * blend_weight / blend_shape_count, tangent.w));
#endif
#endif // MODE_BLEND_PASS
#else // USE_BLEND_SHAPES

	// Make attributes available to the skeleton shader if not written by blend shapes.
	out_vertex = in_vertex;
#ifdef USE_NORMAL
	out_normal = in_normal;
#endif
#ifdef USE_TANGENT
	out_tangent = in_tangent;
#endif
#endif // USE_BLEND_SHAPES

#ifdef USE_SKELETON

#define TEX(m) texelFetch(skeleton_texture, ivec2(m % 256u, m / 256u), 0)
#define GET_BONE_MATRIX(a, b, c, w) mat4(TEX(a), TEX(b), TEX(c), vec4(0.0, 0.0, 0.0, 1.0)) * w

	uvec4 bones = in_bone_attrib * uvec4(3);
	uvec4 bones_a = bones + uvec4(1);
	uvec4 bones_b = bones + uvec4(2);

	highp mat4 m;
	m = GET_BONE_MATRIX(bones.x, bones_a.x, bones_b.x, in_weight_attrib.x);
	m += GET_BONE_MATRIX(bones.y, bones_a.y, bones_b.y, in_weight_attrib.y);
	m += GET_BONE_MATRIX(bones.z, bones_a.z, bones_b.z, in_weight_attrib.z);
	m += GET_BONE_MATRIX(bones.w, bones_a.w, bones_b.w, in_weight_attrib.w);

#ifdef USE_EIGHT_WEIGHTS
	bones = in_bone_attrib2 * uvec4(3);
	bones_a = bones + uvec4(1);
	bones_b = bones + uvec4(2);

	m += GET_BONE_MATRIX(bones.x, bones_a.x, bones_b.x, in_weight_attrib2.x);
	m += GET_BONE_MATRIX(bones.y, bones_a.y, bones_b.y, in_weight_attrib2.y);
	m += GET_BONE_MATRIX(bones.z, bones_a.z, bones_b.z, in_weight_attrib2.z);
	m += GET_BONE_MATRIX(bones.w, bones_a.w, bones_b.w, in_weight_attrib2.w);
#endif

	// Reverse order because its transposed.
	out_vertex = (vec4(out_vertex, 1.0) * m).xyz;
#ifdef USE_NORMAL
	vec3 vertex_normal = oct_to_vec3(out_normal);
	out_normal = vec3_to_oct(normalize((vec4(vertex_normal, 0.0) * m).xyz));
#endif // USE_NORMAL
#ifdef USE_TANGENT
	vec4 vertex_tangent = oct_to_tang(out_tangent);
	out_tangent = tang_to_oct(vec4(normalize((vec4(vertex_tangent.xyz, 0.0) * m).xyz), vertex_tangent.w));
#endif // USE_TANGENT
#endif // USE_SKELETON
#endif // MODE_2D
}

/* clang-format off */
#[fragment]

void main() {

}
/* clang-format on */
