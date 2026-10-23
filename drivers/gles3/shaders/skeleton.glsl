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

uniform highp sampler2D skeleton_texture; // texunit:0
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

uniform uint skinning_method;
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

struct DualQuat {
	vec4 real;
	vec4 dual;
	mat3 S;
};

DualQuat bone_to_dual_quat(mat4 M) {
	mat3 B = mat3(M);

	// column-major GLSL version of orthonormalize() in basis.cpp (Gram-Schmidt)
	vec3 x = vec3(B[0][0], B[1][0], B[2][0]);
	vec3 y = vec3(B[0][1], B[1][1], B[2][1]);
	vec3 z = vec3(B[0][2], B[1][2], B[2][2]);
	x = normalize(x);
	y = normalize(y - x * dot(x, y));
	z = normalize(z - x * dot(x, z) - y * dot(y, z));
	mat3 R = mat3(x, y, z);
	if (dot(R[0], cross(R[1], R[2])) < 0.0) {
		R = -R;
	}

	// Scale component of B using polar decomposition
	mat3 S = mat3(1.0);
	if (skinning_method == 2) {
		S = B * R;
	}

	// column-major GLSL version of get_quaternion() in basis.cpp
	float trace = R[0][0] + R[1][1] + R[2][2];
	vec4 real = vec4(0.0);

	if (trace > 0.0) {
        float s = sqrt(trace + 1.0);
        real.w = 0.5 * s;
		s = 0.5 / s;
        real.x = (R[1][2] - R[2][1]) * s;
        real.y = (R[2][0] - R[0][2]) * s;
        real.z = (R[0][1] - R[1][0]) * s;
    } else {
		uint i = R[0][0] < R[1][1] ?
			(R[1][1] < R[2][2] ? 2 : 1) :
			(R[0][0] < R[2][2] ? 2 : 0);
		uint j = (i + 1) % 3;
		uint k = (i + 2) % 3;

		float s = sqrt(R[i][i] - R[j][j] - R[k][k] + 1.0);
		real[i] = 0.5 * s;
		s = 0.5 / s;

		real.w = (R[j][k] - R[k][j]) * s;
		real[j] = (R[i][j] + R[j][i]) * s;
		real[k] = (R[i][k] + R[k][i]) * s;
	}

	// Credit goes to original version at https://users.cs.utah.edu/~ladislav/dq/dqconv.c
	vec3 t = vec3(M[0][3], M[1][3], M[2][3]); // M is still transposed
	vec4 dual = vec4(
		0.5 * (t.x * real.w + t.y * real.z - t.z * real.y),
		0.5 * (-t.x * real.z + t.y * real.w + t.z * real.x),
		0.5 * (t.x * real.y - t.y * real.x + t.z * real.w),
		-0.5 * (t.x * real.x + t.y * real.y + t.z * real.z));
	return DualQuat(real, dual, S);
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
#define GET_BONE_MATRIX(a, b, c, w) mat4(TEX(a), TEX(b), TEX(c), vec4(0.0, 0.0, 0.0, 1.0))

	uvec4 bones = in_bone_attrib * uvec4(3);
	uvec4 bones_a = bones + uvec4(1);
	uvec4 bones_b = bones + uvec4(2);

	highp mat4 m;
	if (skinning_method == 0) {
		m = GET_BONE_MATRIX(bones.x, bones_a.x, bones_b.x) * in_weight_attrib.x;
		m += GET_BONE_MATRIX(bones.y, bones_a.y, bones_b.y) * in_weight_attrib.y;
		m += GET_BONE_MATRIX(bones.z, bones_a.z, bones_b.z) * in_weight_attrib.z;
		m += GET_BONE_MATRIX(bones.w, bones_a.w, bones_b.w) * in_weight_attrib.w;

	#ifdef USE_EIGHT_WEIGHTS
		bones = in_bone_attrib2 * uvec4(3);
		bones_a = bones + uvec4(1);
		bones_b = bones + uvec4(2);

		m += GET_BONE_MATRIX(bones.x, bones_a.x, bones_b.x) * in_weight_attrib2.x;
		m += GET_BONE_MATRIX(bones.y, bones_a.y, bones_b.y) * in_weight_attrib2.y;
		m += GET_BONE_MATRIX(bones.z, bones_a.z, bones_b.z) * in_weight_attrib2.z;
		m += GET_BONE_MATRIX(bones.w, bones_a.w, bones_b.w) * in_weight_attrib2.w;
	#endif

	} else if (skinning_method == 1 || skinning_method == 2) {
		DualQuat dq0 = bone_to_dual_quat(GET_BONE_MATRIX(bones.x, bones_a.x, bones_b.x));
		DualQuat dq1 = bone_to_dual_quat(GET_BONE_MATRIX(bones.y, bones_a.y, bones_b.y));
		DualQuat dq2 = bone_to_dual_quat(GET_BONE_MATRIX(bones.z, bones_a.z, bones_b.z));
		DualQuat dq3 = bone_to_dual_quat(GET_BONE_MATRIX(bones.w, bones_a.w, bones_b.w));

		if (dot(dq0.real, dq1.real) < 0.0) {
			dq1.real = -dq1.real;
			dq1.dual = -dq1.dual;
		}
		if (dot(dq0.real, dq2.real) < 0.0) {
			dq2.real = -dq2.real;
			dq2.dual = -dq2.dual;
		}
		if (dot(dq0.real, dq3.real) < 0.0) {
			dq3.real = -dq3.real;
			dq3.dual = -dq3.dual;
		}

		vec4 real = dq0.real * in_weight_attrib.x;
		real += dq1.real * in_weight_attrib.y;
		real += dq2.real * in_weight_attrib.z;
		real += dq3.real * in_weight_attrib.w;

		vec4 dual = dq0.dual * in_weight_attrib.x;
		dual += dq1.dual * in_weight_attrib.y;
		dual += dq2.dual * in_weight_attrib.z;
		dual += dq3.dual * in_weight_attrib.w;

		mat3 S = mat3(1.0);
		if (skinning_method == 2) {
			S = dq0.S * in_weight_attrib.x;
			S += dq1.S * in_weight_attrib.y;
			S += dq2.S * in_weight_attrib.z;
			S += dq3.S * in_weight_attrib.w;
		}

	#ifdef USE_EIGHT_WEIGHTS
		bones = in_bone_attrib2 * uvec4(3);
		bones_a = bones + uvec4(1);
		bones_b = bones + uvec4(2);

		DualQuat dq4 = bone_to_dual_quat(GET_BONE_MATRIX(bones.x, bones_a.x, bones_b.x));
		DualQuat dq5 = bone_to_dual_quat(GET_BONE_MATRIX(bones.y, bones_a.y, bones_b.y));
		DualQuat dq6 = bone_to_dual_quat(GET_BONE_MATRIX(bones.z, bones_a.z, bones_b.z));
		DualQuat dq7 = bone_to_dual_quat(GET_BONE_MATRIX(bones.w, bones_a.w, bones_b.w));

		if (dot(dq0.real, dq4.real) < 0.0) {
			dq4.real = -dq4.real;
			dq4.dual = -dq4.dual;
		}
		if (dot(dq0.real, dq5.real) < 0.0) {
			dq5.real = -dq5.real;
			dq5.dual = -dq5.dual;
		}
		if (dot(dq0.real, dq6.real) < 0.0) {
			dq6.real = -dq6.real;
			dq6.dual = -dq6.dual;
		}
		if (dot(dq0.real, dq7.real) < 0.0) {
			dq7.real = -dq7.real;
			dq7.dual = -dq7.dual;
		}

		real += dq4.real * in_weight_attrib2.x;
		real += dq5.real * in_weight_attrib2.y;
		real += dq6.real * in_weight_attrib2.z;
		real += dq7.real * in_weight_attrib2.w;

		dual += dq4.dual * in_weight_attrib2.x;
		dual += dq5.dual * in_weight_attrib2.y;
		dual += dq6.dual * in_weight_attrib2.z;
		dual += dq7.dual * in_weight_attrib2.w;

		if (skinning_method == 2) {
			S += dq4.S * in_weight_attrib2.x;
			S += dq5.S * in_weight_attrib2.y;
			S += dq6.S * in_weight_attrib2.z;
			S += dq7.S * in_weight_attrib2.w;
		}
	#endif

		float len = length(real);
		if (len >= 1e-8) {
			real /= len;
			dual /= len;

			// Credit goes to original version at https://users.cs.utah.edu/~ladislav/dq/dqs.cg
			// Transposed here since the LBS version is transposed
			m[0][0] = real.w * real.w + real.x * real.x - real.y * real.y - real.z * real.z;
			m[0][1] = 2.0 * real.x * real.y - 2.0 * real.w * real.z;
			m[0][2] = 2.0 * real.x * real.z + 2.0 * real.w * real.y;

			m[1][0] = (2.0 * real.x * real.y + 2.0 * real.w * real.z);
			m[1][1] = (real.w * real.w + real.y * real.y - real.x * real.x - real.z * real.z);
			m[1][2] = (2.0 * real.y * real.z - 2.0 * real.w * real.x);

			m[2][0] = 2.0 * real.x * real.z - 2.0 * real.w * real.y;
			m[2][1] = 2.0 * real.y * real.z + 2.0 * real.w * real.x;
			m[2][2] = real.w * real.w + real.z * real.z - real.x * real.x - real.y * real.y;

			m[0][3] = -2.0 * dual.w * real.x + 2.0 * real.w * dual.x - 2.0 * dual.y * real.z + 2.0 * real.y * dual.z;
			m[1][3] = -2.0 * dual.w * real.y + 2.0 * dual.x * real.z - 2.0 * real.x * dual.z + 2.0 * real.w * dual.y;
			m[2][3] = -2.0 * dual.w * real.z + 2.0 * real.x * dual.y + 2.0 * real.w * dual.z - 2.0 * dual.x * real.y;

			m[3][0] = 0.0;
			m[3][1] = 0.0;
			m[3][2] = 0.0;
			m[3][3] = 1.0;

			if (skinning_method == 2) {
				// reintroduce scale
				mat3 B = mat3(m);
				B = S * B;
				m[0].xyz = B[0].xyz;
				m[1].xyz = B[1].xyz;
				m[2].xyz = B[2].xyz;
			}
		} else {
			m = mat4(1.0);
		}
	}

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
