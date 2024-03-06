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
	uint pad1;

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
	return vec2_to_uint(oct);
}


mat4 dual_quaternion_to_matrix(vec4 Qn, vec4 Qd)
{
	//Original version of this function is in https://users.cs.utah.edu/~ladislav/dq/dqs.cg
	mat4 M = mat4(0.0);
	float len2 = dot(Qn, Qn);
	float w = Qn.w, x = Qn.x, y = Qn.y, z = Qn.z;
	
	float t0 = Qd.w, t1 = Qd.x, t2 = Qd.y, t3 = Qd.z;

	M[0][0] = w*w + x*x - y*y - z*z;
	M[1][0] = 2.0*x*y - 2.0*w*z; 
	M[2][0] = 2.0*x*z + 2.0*w*y;
	M[0][1] = (2.0*x*y + 2.0*w*z); 
	M[1][1] = (w*w + y*y - x*x - z*z); 
	M[2][1] = (2.0*y*z - 2.0*w*x); 
	M[0][2] = 2.0*x*z - 2.0*w*y; 
	M[1][2] = 2.0*y*z + 2.0*w*x; 
	M[2][2] = w*w + z*z - x*x - y*y;
	
	M[3][0] = -2.0*t0*x + 2.0*w*t1 - 2.0*t2*z + 2.0*y*t3;
	M[3][1] = -2.0*t0*y + 2.0*t1*z - 2.0*x*t3 + 2.0*w*t2;
	M[3][2] = -2.0*t0*z + 2.0*x*t2 + 2.0*w*t3 - 2.0*t1*y;
	
	M /= len2;
	
	return M;
}

vec4 unpack_q0(uint index) {
	mat4 mat_data = mat4(bone_transforms.data[index], bone_transforms.data[index + 1], bone_transforms.data[index + 2], vec4(0.0, 0.0, 0.0, 1.0));
	vec4 row1 = mat_data[0];
	vec4 row2 = mat_data[1];
	return vec4(row1.x, row1.y, row1.z, row2.x);
}

vec4 unpack_q1(uint index) {
	mat4 mat_data = mat4(bone_transforms.data[index], bone_transforms.data[index + 1], bone_transforms.data[index + 2], vec4(0.0, 0.0, 0.0, 1.0));
	vec4 row2 = mat_data[1];
	vec4 row3 = mat_data[2];
	return vec4(row2.y, row2.z, row3.x, row3.y);
}

vec3 unpack_scale(uint index) {
	mat4 mat_data = mat4(bone_transforms.data[index], bone_transforms.data[index + 1], bone_transforms.data[index + 2], vec4(0.0, 0.0, 0.0, 1.0));
	vec4 row1 = mat_data[0];
	vec4 row2 = mat_data[1];
	vec4 row3 = mat_data[2];
	return vec3(row1.w, row2.w, row3.w);
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

		mat4 m = mat4(bone_transforms.data[bones_01.x], bone_transforms.data[bones_01.x + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.x;
		m += mat4(bone_transforms.data[bones_01.y], bone_transforms.data[bones_01.y + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.y;
		m += mat4(bone_transforms.data[bones_23.x], bone_transforms.data[bones_23.x + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.x;
		m += mat4(bone_transforms.data[bones_23.y], bone_transforms.data[bones_23.y + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.y;

		mat4 skeleton_matrix = mat4(vec4(params.skeleton_transform_x, 0.0, 0.0), vec4(params.skeleton_transform_y, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(params.skeleton_transform_offset, 0.0, 1.0));
		mat4 inverse_matrix = mat4(vec4(params.inverse_transform_x, 0.0, 0.0), vec4(params.inverse_transform_y, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(params.inverse_transform_offset, 0.0, 1.0));

		m = skeleton_matrix * transpose(m) * inverse_matrix;

		vertex = (m * vec4(vertex, 0.0, 1.0)).xy;
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

		uvec2 bones = uvec2(src_bone_weights.data[skin_offset + 0], src_bone_weights.data[skin_offset + 1]);
		uvec2 bones_01 = uvec2(bones.x & 0xFFFF, bones.x >> 16) * 3; //pre-add xform offset
		uvec2 bones_23 = uvec2(bones.y & 0xFFFF, bones.y >> 16) * 3;

		skin_offset += params.skin_weight_offset;

		uvec2 weights = uvec2(src_bone_weights.data[skin_offset + 0], src_bone_weights.data[skin_offset + 1]);

		vec2 weights_01 = unpackUnorm2x16(weights.x);
		vec2 weights_23 = unpackUnorm2x16(weights.y);

		//mat4 m = mat4(bone_transforms.data[bones_01.x], bone_transforms.data[bones_01.x + 1], bone_transforms.data[bones_01.x + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.x;
		//m += mat4(bone_transforms.data[bones_01.y], bone_transforms.data[bones_01.y + 1], bone_transforms.data[bones_01.y + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.y;
		//m += mat4(bone_transforms.data[bones_23.x], bone_transforms.data[bones_23.x + 1], bone_transforms.data[bones_23.x + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.x;
		//m += mat4(bone_transforms.data[bones_23.y], bone_transforms.data[bones_23.y + 1], bone_transforms.data[bones_23.y + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.y;


		vec4 blend_q0_0 = weights_01.x * unpack_q0(bones_01.x);
		vec4 blend_q0_1 = weights_01.y * unpack_q0(bones_01.y);
		vec4 blend_q0_2 = weights_23.x * unpack_q0(bones_23.x);
		vec4 blend_q0_3 = weights_23.y * unpack_q0(bones_23.y);

		vec4 blend_q1_0 = weights_01.x * unpack_q1(bones_01.x);
		vec4 blend_q1_1 = weights_01.y * unpack_q1(bones_01.y);
		vec4 blend_q1_2 = weights_23.x * unpack_q1(bones_23.x);
		vec4 blend_q1_3 = weights_23.y * unpack_q1(bones_23.y);

		if (dot(blend_q0_0, blend_q0_1) < 0.0) {
			blend_q0_1 *= -1.0;
			blend_q1_1 *= -1.0;
		}
		if (dot(blend_q0_0, blend_q0_2) < 0.0) {
			blend_q0_2 *= -1.0;
			blend_q1_2 *= -1.0;
		}
		if (dot(blend_q0_0, blend_q0_3) < 0.0) {
			blend_q0_3 *= -1.0;
			blend_q1_3 *= -1.0;
		}

		vec4 blend_q0 = weights_01.x * blend_q0_0 + weights_01.y * blend_q0_1 + weights_23.x * blend_q0_2 + weights_23.y * blend_q0_3;
		vec4 blend_q1 = weights_01.x * blend_q1_0 + weights_01.y * blend_q1_1 + weights_23.x * blend_q1_2 + weights_23.y * blend_q1_3;

		//vec3 scale = weights_01.x * unpack_scale(bones_01.x) + weights_01.y * unpack_scale(bones_01.y) + weights_23.x * unpack_scale(bones_23.x) + weights_23.y * unpack_scale(bones_23.y);

		if (params.skin_weight_offset == 4) {
			//using 8 bones/weights
			skin_offset = params.skin_stride * index + 2;

			bones = uvec2(src_bone_weights.data[skin_offset + 0], src_bone_weights.data[skin_offset + 1]);
			bones_01 = uvec2(bones.x & 0xFFFF, bones.x >> 16) * 3; //pre-add xform offset
			bones_23 = uvec2(bones.y & 0xFFFF, bones.y >> 16) * 3;

			skin_offset += params.skin_weight_offset;

			weights = uvec2(src_bone_weights.data[skin_offset + 0], src_bone_weights.data[skin_offset + 1]);

			weights_01 = unpackUnorm2x16(weights.x);
			weights_23 = unpackUnorm2x16(weights.y);

			//m += mat4(bone_transforms.data[bones_01.x], bone_transforms.data[bones_01.x + 1], bone_transforms.data[bones_01.x + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.x;
			//m += mat4(bone_transforms.data[bones_01.y], bone_transforms.data[bones_01.y + 1], bone_transforms.data[bones_01.y + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.y;
			//m += mat4(bone_transforms.data[bones_23.x], bone_transforms.data[bones_23.x + 1], bone_transforms.data[bones_23.x + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.x;
			//m += mat4(bone_transforms.data[bones_23.y], bone_transforms.data[bones_23.y + 1], bone_transforms.data[bones_23.y + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.y;
			//TODO!!!!!!!!!!!!!!!!
			//blend_q0 += weights_01.x * unpack_q0(bones_01.x) + weights_01.y * unpack_q0(bones_01.y) + weights_23.x * unpack_q0(bones_23.x) + weights_23.y * unpack_q0(bones_23.y);
			//blend_q1 += weights_01.x * unpack_q1(bones_01.x) + weights_01.y * unpack_q1(bones_01.y) + weights_23.x * unpack_q1(bones_23.x) + weights_23.y * unpack_q1(bones_23.y);
			//scale += weights_01.x * unpack_scale(bones_01.x) + weights_01.y * unpack_scale(bones_01.y) + weights_23.x * unpack_scale(bones_23.x) + weights_23.y * unpack_scale(bones_23.y);
		}

		mat4 m = dual_quaternion_to_matrix(blend_q0, blend_q1);
		
		// TODO how to conserve local scale?
		//m[0].xyz *= scale.x;
		//m[1].xyz *= scale.y;
		//m[2].xyz *= scale.z;

		//reverse order because its transposed
		//dual quaternion however is not transposed. To restore this, swap the * operator ordering
		vertex = (m * vec4(vertex, 1.0)).xyz;
		normal = normalize(m * vec4(normal, 0.0)).xyz;
		tangent.xyz = normalize(m * vec4(tangent.xyz, 0.0)).xyz;
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
