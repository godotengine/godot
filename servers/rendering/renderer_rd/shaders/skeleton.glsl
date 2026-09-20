#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 1, std430) buffer restrict writeonly DstVertexData {
	uint data[];
}
dst_vertices;

layout(set = 0, binding = 2, std430) buffer restrict readonly BlendShapeWeights {
	float data[];
}
blend_shape_weights;

layout(set = 1, binding = 0, std430) buffer restrict readonly SrcVertexData {
	uint data[];
}
src_vertices;

layout(set = 1, binding = 1, std430) buffer restrict readonly BoneWeightData {
	uint data[];
}
src_bone_weights;

layout(set = 1, binding = 2, std430) buffer restrict readonly BlendShapeData {
	uint data[];
}
src_blend_shapes;

layout(set = 2, binding = 0, std430) buffer restrict readonly SkeletonData {
	vec4 data[];
}
bone_transforms;

layout(push_constant, std430) uniform Params {
	bool has_normal;
	bool has_tangent;
	bool has_skeleton;
	bool has_blend_shape;

	uint vertex_count;
	uint vertex_stride;
	uint skin_stride;
	uint skin_weight_offset;

	uint blend_shape_count;
	bool normalized_blend_shapes;
	uint normal_tangent_stride;
	uint skinning_method;

	vec2 skeleton_transform_x;
	vec2 skeleton_transform_y;

	vec2 skeleton_transform_offset;
	vec2 inverse_transform_x;

	vec2 inverse_transform_y;
	vec2 inverse_transform_offset;
}
params;

vec2 uint_to_vec2(uint base) {
	uvec2 decode = (uvec2(base) >> uvec2(0, 16)) & uvec2(0xFFFF, 0xFFFF);
	return vec2(decode) / vec2(65535.0, 65535.0) * 2.0 - 1.0;
}

vec3 oct_to_vec3(vec2 oct) {
	vec3 v = vec3(oct.xy, 1.0 - abs(oct.x) - abs(oct.y));
	float t = max(-v.z, 0.0);
	v.xy += t * -sign(v.xy);
	return normalize(v);
}

vec3 decode_uint_oct_to_norm(uint base) {
	return oct_to_vec3(uint_to_vec2(base));
}

vec4 decode_uint_oct_to_tang(uint base) {
	vec2 oct_sign_encoded = uint_to_vec2(base);
	// Binormal sign encoded in y component
	vec2 oct = vec2(oct_sign_encoded.x, abs(oct_sign_encoded.y) * 2.0 - 1.0);
	return vec4(oct_to_vec3(oct), sign(oct_sign_encoded.y));
}

vec2 signNotZero(vec2 v) {
	return mix(vec2(-1.0), vec2(1.0), greaterThanEqual(v.xy, vec2(0.0)));
}

uint vec2_to_uint(vec2 base) {
	uvec2 enc = uvec2(clamp(ivec2(base * vec2(65535, 65535)), ivec2(0), ivec2(0xFFFF, 0xFFFF))) << uvec2(0, 16);
	return enc.x | enc.y;
}

vec2 vec3_to_oct(vec3 e) {
	e /= abs(e.x) + abs(e.y) + abs(e.z);
	vec2 oct = e.z >= 0.0f ? e.xy : (vec2(1.0f) - abs(e.yx)) * signNotZero(e.xy);
	return oct * 0.5f + 0.5f;
}

uint encode_norm_to_uint_oct(vec3 base) {
	return vec2_to_uint(vec3_to_oct(base));
}

uint encode_tang_to_uint_oct(vec4 base) {
	vec2 oct = vec3_to_oct(base.xyz);
	// Encode binormal sign in y component
	oct.y = oct.y * 0.5f + 0.5f;
	oct.y = base.w >= 0.0f ? oct.y : 1 - oct.y;

	if (oct.x == 0.0 && oct.y == 1.0) {
		// (1, 1) and (0, 1) decode to the same value, but (0, 1) messes with our compression detection.
		// So we sanitize here.
		oct.x = 1.0;
	}

	return vec2_to_uint(oct);
}

struct OffsetWeight {
	uvec4 offsets;
	vec4 weights;
};

OffsetWeight get_offsets_weights(uint skin_offset) {
	uvec2 bones = uvec2(src_bone_weights.data[skin_offset + 0], src_bone_weights.data[skin_offset + 1]);
	uvec2 bones_01 = uvec2(bones.x & 0xFFFF, bones.x >> 16) * 3; //pre-add xform offset
	uvec2 bones_23 = uvec2(bones.y & 0xFFFF, bones.y >> 16) * 3;

	skin_offset += params.skin_weight_offset;

	uvec2 weights = uvec2(src_bone_weights.data[skin_offset + 0], src_bone_weights.data[skin_offset + 1]);

	vec2 weights_01 = unpackUnorm2x16(weights.x);
	vec2 weights_23 = unpackUnorm2x16(weights.y);

	return OffsetWeight(uvec4(bones_01, bones_23), vec4(weights_01, weights_23));
}

mat4 bone_matrix(uint offset) {
	return mat4(bone_transforms.data[offset], bone_transforms.data[offset + 1], bone_transforms.data[offset + 2], vec4(0.0, 0.0, 0.0, 1.0));
}

struct DualQuat {
	vec4 real;
	vec4 dual;
	mat3 S;
};

DualQuat bone_to_dual_quat(uint offset) {
	// Convert single bone matrix to dual quaternion
	mat4 M = bone_matrix(offset);
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
	if (params.skinning_method == 2) {
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

void main() {
	uint index = gl_GlobalInvocationID.x;
	if (index >= params.vertex_count) {
		return;
	}

	uint src_offset = index * params.vertex_stride;

#ifdef MODE_2D
	vec2 vertex = uintBitsToFloat(uvec2(src_vertices.data[src_offset + 0], src_vertices.data[src_offset + 1]));

	if (params.has_blend_shape) {
		float blend_total = 0.0;
		vec2 blend_vertex = vec2(0.0);

		for (uint i = 0; i < params.blend_shape_count; i++) {
			float w = blend_shape_weights.data[i];
			if (abs(w) > 0.0001) {
				uint base_offset = (params.vertex_count * i + index) * params.vertex_stride;

				blend_vertex += uintBitsToFloat(uvec2(src_blend_shapes.data[base_offset + 0], src_blend_shapes.data[base_offset + 1])) * w;

				base_offset += 2;

				blend_total += w;
			}
		}

		if (params.normalized_blend_shapes) {
			vertex = (1.0 - blend_total) * vertex;
		}

		vertex += blend_vertex;
	}

	if (params.has_skeleton) {
		uint skin_offset = params.skin_stride * index;

		uvec2 bones = uvec2(src_bone_weights.data[skin_offset + 0], src_bone_weights.data[skin_offset + 1]);
		uvec2 bones_01 = uvec2(bones.x & 0xFFFF, bones.x >> 16) * 2; //pre-add xform offset
		uvec2 bones_23 = uvec2(bones.y & 0xFFFF, bones.y >> 16) * 2;

		skin_offset += params.skin_weight_offset;

		uvec2 weights = uvec2(src_bone_weights.data[skin_offset + 0], src_bone_weights.data[skin_offset + 1]);

		vec2 weights_01 = unpackUnorm2x16(weights.x);
		vec2 weights_23 = unpackUnorm2x16(weights.y);

		mat4 M = mat4(bone_transforms.data[bones_01.x], bone_transforms.data[bones_01.x + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.x;
		M += mat4(bone_transforms.data[bones_01.y], bone_transforms.data[bones_01.y + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.y;
		M += mat4(bone_transforms.data[bones_23.x], bone_transforms.data[bones_23.x + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.x;
		M += mat4(bone_transforms.data[bones_23.y], bone_transforms.data[bones_23.y + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.y;

		mat4 skeleton_matrix = mat4(vec4(params.skeleton_transform_x, 0.0, 0.0), vec4(params.skeleton_transform_y, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(params.skeleton_transform_offset, 0.0, 1.0));
		mat4 inverse_matrix = mat4(vec4(params.inverse_transform_x, 0.0, 0.0), vec4(params.inverse_transform_y, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(params.inverse_transform_offset, 0.0, 1.0));

		M = skeleton_matrix * transpose(M) * inverse_matrix;

		vertex = (M * vec4(vertex, 0.0, 1.0)).xy;
	}

	uint dst_offset = index * params.vertex_stride;

	uvec2 uvertex = floatBitsToUint(vertex);
	dst_vertices.data[dst_offset + 0] = uvertex.x;
	dst_vertices.data[dst_offset + 1] = uvertex.y;

#else
	vec3 vertex;
	vec3 normal;
	vec4 tangent;

	vertex = uintBitsToFloat(uvec3(src_vertices.data[src_offset + 0], src_vertices.data[src_offset + 1], src_vertices.data[src_offset + 2]));

	uint src_normal = params.vertex_count * params.vertex_stride + index * params.normal_tangent_stride;

	if (params.has_normal) {
		normal = decode_uint_oct_to_norm(src_vertices.data[src_normal]);
		src_normal++;
	}

	if (params.has_tangent) {
		tangent = decode_uint_oct_to_tang(src_vertices.data[src_normal]);
	}

	if (params.has_blend_shape) {
		float blend_total = 0.0;
		vec3 blend_vertex = vec3(0.0);
		vec3 blend_normal = vec3(0.0);
		vec3 blend_tangent = vec3(0.0);

		for (uint i = 0; i < params.blend_shape_count; i++) {
			float w = blend_shape_weights.data[i];
			if (abs(w) > 0.0001) {
				uint base_offset = params.vertex_count * i * (params.vertex_stride + params.normal_tangent_stride) + index * params.vertex_stride;

				blend_vertex += uintBitsToFloat(uvec3(src_blend_shapes.data[base_offset + 0], src_blend_shapes.data[base_offset + 1], src_blend_shapes.data[base_offset + 2])) * w;

				uint base_normal = params.vertex_count * i * (params.vertex_stride + params.normal_tangent_stride) + params.vertex_count * params.vertex_stride + index * params.normal_tangent_stride;

				if (params.has_normal) {
					blend_normal += decode_uint_oct_to_norm(src_blend_shapes.data[base_normal]) * w;
					base_normal++;
				}

				if (params.has_tangent) {
					blend_tangent += decode_uint_oct_to_tang(src_blend_shapes.data[base_normal]).rgb * w;
				}

				blend_total += w;
			}
		}

		if (params.normalized_blend_shapes) {
			vertex = (1.0 - blend_total) * vertex;
			normal = (1.0 - blend_total) * normal;
			tangent.rgb = (1.0 - blend_total) * tangent.rgb;
		}

		vertex += blend_vertex;
		normal = normalize(normal + blend_normal);
		tangent.rgb = normalize(tangent.rgb + blend_tangent);
	}

	if (params.has_skeleton) {
		uint skin_offset = params.skin_stride * index;

		mat4 M = mat4(1.0);
		if (params.skinning_method == 0) {
			// linear blend skinning (LBS)
			OffsetWeight ow = get_offsets_weights(skin_offset);
			M = bone_matrix(ow.offsets.x) * ow.weights.x;
			M += bone_matrix(ow.offsets.y) * ow.weights.y;
			M += bone_matrix(ow.offsets.z) * ow.weights.z;
			M += bone_matrix(ow.offsets.w) * ow.weights.w;

			if (params.skin_weight_offset == 4) {
				//using 8 bones/weights
				OffsetWeight ow2 = get_offsets_weights(skin_offset + 2);
				M += bone_matrix(ow2.offsets.x) * ow2.weights.x;
				M += bone_matrix(ow2.offsets.y) * ow2.weights.y;
				M += bone_matrix(ow2.offsets.z) * ow2.weights.z;
				M += bone_matrix(ow2.offsets.w) * ow2.weights.w;
			}

		} else if (params.skinning_method == 1 || params.skinning_method == 2) {
			// Dual Quaternion Skinning (DQS)
			OffsetWeight ow = get_offsets_weights(skin_offset);

			DualQuat dq0 = bone_to_dual_quat(ow.offsets.x);
			DualQuat dq1 = bone_to_dual_quat(ow.offsets.y);
			DualQuat dq2 = bone_to_dual_quat(ow.offsets.z);
			DualQuat dq3 = bone_to_dual_quat(ow.offsets.w);

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

			vec4 real = dq0.real * ow.weights.x;
			real += dq1.real * ow.weights.y;
			real += dq2.real * ow.weights.z;
			real += dq3.real * ow.weights.w;
			vec4 dual = dq0.dual * ow.weights.x;
			dual += dq1.dual * ow.weights.y;
			dual += dq2.dual * ow.weights.z;
			dual += dq3.dual * ow.weights.w;

			mat3 S = mat3(1.0);
			if (params.skinning_method == 2) {
				S = dq0.S * ow.weights.x;
				S += dq1.S * ow.weights.y;
				S += dq2.S * ow.weights.z;
				S += dq3.S * ow.weights.w;
			}

			if (params.skin_weight_offset == 4) {
				//using 8 bones/weights
				OffsetWeight ow2 = get_offsets_weights(skin_offset + 2);

				DualQuat dq4 = bone_to_dual_quat(ow2.offsets.x);
				DualQuat dq5 = bone_to_dual_quat(ow2.offsets.y);
				DualQuat dq6 = bone_to_dual_quat(ow2.offsets.z);
				DualQuat dq7 = bone_to_dual_quat(ow2.offsets.w);

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

				real += dq4.real * ow2.weights.x;
				real +=	dq5.real * ow2.weights.y; 
				real +=	dq6.real * ow2.weights.z;
				real +=	dq7.real * ow2.weights.w;

				dual += dq4.dual * ow2.weights.x;
				dual += dq5.dual * ow2.weights.y;
				dual += dq6.dual * ow2.weights.z; 
				dual += dq7.dual * ow2.weights.w;

				if (params.skinning_method == 2) {
					S += dq4.S * ow2.weights.x;
					S += dq5.S * ow2.weights.y;
					S += dq6.S * ow2.weights.z;
					S += dq7.S * ow2.weights.w;
				}
			}

			float len = length(real);
			if (len >= 1e-8) {
				real /= len;
				dual /= len;

				// Credit goes to original version at https://users.cs.utah.edu/~ladislav/dq/dqs.cg
				// Transposed here since the LBS version is transposed
				M[0][0] = real.w * real.w + real.x * real.x - real.y * real.y - real.z * real.z;
				M[0][1] = 2.0 * real.x * real.y - 2.0 * real.w * real.z;
				M[0][2] = 2.0 * real.x * real.z + 2.0 * real.w * real.y;

				M[1][0] = (2.0 * real.x * real.y + 2.0 * real.w * real.z);
				M[1][1] = (real.w * real.w + real.y * real.y - real.x * real.x - real.z * real.z);
				M[1][2] = (2.0 * real.y * real.z - 2.0 * real.w * real.x);

				M[2][0] = 2.0 * real.x * real.z - 2.0 * real.w * real.y;
				M[2][1] = 2.0 * real.y * real.z + 2.0 * real.w * real.x;
				M[2][2] = real.w * real.w + real.z * real.z - real.x * real.x - real.y * real.y;

				M[0][3] = -2.0 * dual.w * real.x + 2.0 * real.w * dual.x - 2.0 * dual.y * real.z + 2.0 * real.y * dual.z;
				M[1][3] = -2.0 * dual.w * real.y + 2.0 * dual.x * real.z - 2.0 * real.x * dual.z + 2.0 * real.w * dual.y;
				M[2][3] = -2.0 * dual.w * real.z + 2.0 * real.x * dual.y + 2.0 * real.w * dual.z - 2.0 * dual.x * real.y;

				M[3][0] = 0.0;
				M[3][1] = 0.0;
				M[3][2] = 0.0;
				M[3][3] = 1.0;

				if (params.skinning_method == 2) {
					// reintroduce scale
					mat3 B = mat3(M);
					B = S * B;
					M[0].xyz = B[0].xyz;
					M[1].xyz = B[1].xyz;
					M[2].xyz = B[2].xyz;
				}
			} else {
				M = mat4(1.0);
			}
		}

		//reverse order because its transposed
		vertex = (vec4(vertex, 1.0) * M).xyz;
		normal = normalize((vec4(normal, 0.0) * M).xyz);
		tangent.xyz = normalize((vec4(tangent.xyz, 0.0) * M).xyz);
	}

	uint dst_offset = index * params.vertex_stride;

	uvec3 uvertex = floatBitsToUint(vertex);
	dst_vertices.data[dst_offset + 0] = uvertex.x;
	dst_vertices.data[dst_offset + 1] = uvertex.y;
	dst_vertices.data[dst_offset + 2] = uvertex.z;

	uint dst_normal = params.vertex_count * params.vertex_stride + index * params.normal_tangent_stride;

	if (params.has_normal) {
		dst_vertices.data[dst_normal] = encode_norm_to_uint_oct(normal);
		dst_normal++;
	}

	if (params.has_tangent) {
		dst_vertices.data[dst_normal] = encode_tang_to_uint_oct(tangent);
	}

#endif
}
