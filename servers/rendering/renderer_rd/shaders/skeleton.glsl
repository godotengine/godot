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

layout(push_constant, binding = 0, std430) uniform Params {
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
	uint pad0;
	uint pad1;
}
params;

vec4 decode_abgr_2_10_10_10(uint base) {
	uvec4 abgr_2_10_10_10 = (uvec4(base) >> uvec4(0, 10, 20, 30)) & uvec4(0x3FF, 0x3FF, 0x3FF, 0x3);
	return vec4(abgr_2_10_10_10) / vec4(1023.0, 1023.0, 1023.0, 3.0) * 2.0 - 1.0;
}

uint encode_abgr_2_10_10_10(vec4 base) {
	uvec4 abgr_2_10_10_10 = uvec4(clamp(ivec4((base * 0.5 + 0.5) * vec4(1023.0, 1023.0, 1023.0, 3.0)), ivec4(0), ivec4(0x3FF, 0x3FF, 0x3FF, 0x3))) << uvec4(0, 10, 20, 30);
	return abgr_2_10_10_10.x | abgr_2_10_10_10.y | abgr_2_10_10_10.z | abgr_2_10_10_10.w;
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
		uvec2 bones_01 = uvec2(bones.x & 0xFFFF, bones.x >> 16) * 3; //pre-add xform offset
		uvec2 bones_23 = uvec2(bones.y & 0xFFFF, bones.y >> 16) * 3;

		skin_offset += params.skin_weight_offset;

		uvec2 weights = uvec2(src_bone_weights.data[skin_offset + 0], src_bone_weights.data[skin_offset + 1]);

		vec2 weights_01 = unpackUnorm2x16(weights.x);
		vec2 weights_23 = unpackUnorm2x16(weights.y);

		mat4 m = mat4(bone_transforms.data[bones_01.x], bone_transforms.data[bones_01.x + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.x;
		m += mat4(bone_transforms.data[bones_01.y], bone_transforms.data[bones_01.y + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.y;
		m += mat4(bone_transforms.data[bones_23.x], bone_transforms.data[bones_23.x + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.x;
		m += mat4(bone_transforms.data[bones_23.y], bone_transforms.data[bones_23.y + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.y;

		//reverse order because its transposed
		vertex = (vec4(vertex, 0.0, 1.0) * m).xy;
	}
#else
	vec3 vertex;
	vec3 normal;
	vec4 tangent;

	vertex = uintBitsToFloat(uvec3(src_vertices.data[src_offset + 0], src_vertices.data[src_offset + 1], src_vertices.data[src_offset + 2]));

	src_offset += 3;

	if (params.has_normal) {
		normal = decode_abgr_2_10_10_10(src_vertices.data[src_offset]).rgb;
		src_offset++;
	}

	if (params.has_tangent) {
		tangent = decode_abgr_2_10_10_10(src_vertices.data[src_offset]);
	}

	if (params.has_blend_shape) {
		float blend_total = 0.0;
		vec3 blend_vertex = vec3(0.0);
		vec3 blend_normal = vec3(0.0);
		vec3 blend_tangent = vec3(0.0);

		for (uint i = 0; i < params.blend_shape_count; i++) {
			float w = blend_shape_weights.data[i];
			if (abs(w) > 0.0001) {
				uint base_offset = (params.vertex_count * i + index) * params.vertex_stride;

				blend_vertex += uintBitsToFloat(uvec3(src_blend_shapes.data[base_offset + 0], src_blend_shapes.data[base_offset + 1], src_blend_shapes.data[base_offset + 2])) * w;

				base_offset += 3;

				if (params.has_normal) {
					blend_normal += decode_abgr_2_10_10_10(src_blend_shapes.data[base_offset]).rgb * w;
					base_offset++;
				}

				if (params.has_tangent) {
					blend_tangent += decode_abgr_2_10_10_10(src_blend_shapes.data[base_offset]).rgb;
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
		normal += normalize(normal + blend_normal);
		tangent.rgb += normalize(tangent.rgb + blend_tangent);
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

		mat4 m = mat4(bone_transforms.data[bones_01.x], bone_transforms.data[bones_01.x + 1], bone_transforms.data[bones_01.x + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.x;
		m += mat4(bone_transforms.data[bones_01.y], bone_transforms.data[bones_01.y + 1], bone_transforms.data[bones_01.y + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.y;
		m += mat4(bone_transforms.data[bones_23.x], bone_transforms.data[bones_23.x + 1], bone_transforms.data[bones_23.x + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.x;
		m += mat4(bone_transforms.data[bones_23.y], bone_transforms.data[bones_23.y + 1], bone_transforms.data[bones_23.y + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.y;

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

			m += mat4(bone_transforms.data[bones_01.x], bone_transforms.data[bones_01.x + 1], bone_transforms.data[bones_01.x + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.x;
			m += mat4(bone_transforms.data[bones_01.y], bone_transforms.data[bones_01.y + 1], bone_transforms.data[bones_01.y + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.y;
			m += mat4(bone_transforms.data[bones_23.x], bone_transforms.data[bones_23.x + 1], bone_transforms.data[bones_23.x + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.x;
			m += mat4(bone_transforms.data[bones_23.y], bone_transforms.data[bones_23.y + 1], bone_transforms.data[bones_23.y + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.y;
		}

		//reverse order because its transposed
		vertex = (vec4(vertex, 1.0) * m).xyz;
		normal = normalize((vec4(normal, 0.0) * m).xyz);
		tangent.xyz = normalize((vec4(tangent.xyz, 0.0) * m).xyz);
	}

	uint dst_offset = index * params.vertex_stride;

	uvec3 uvertex = floatBitsToUint(vertex);
	dst_vertices.data[dst_offset + 0] = uvertex.x;
	dst_vertices.data[dst_offset + 1] = uvertex.y;
	dst_vertices.data[dst_offset + 2] = uvertex.z;

	dst_offset += 3;

	if (params.has_normal) {
		dst_vertices.data[dst_offset] = encode_abgr_2_10_10_10(vec4(normal, 0.0));
		dst_offset++;
	}

	if (params.has_tangent) {
		dst_vertices.data[dst_offset] = encode_abgr_2_10_10_10(tangent);
	}

#endif
}
