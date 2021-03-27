#[compute]

#version 450

#VERSION_DEFINES

#define GROUP_SIZE 8

layout(local_size_x = GROUP_SIZE, local_size_y = GROUP_SIZE, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform samplerCube source_cube;

layout(rgba16f, set = 1, binding = 0) uniform restrict writeonly imageCube dest_cubemap;

#include "cubemap_roughness_inc.glsl"

void main() {
	uvec3 id = gl_GlobalInvocationID;
	id.z += params.face_id;

	vec2 uv = ((vec2(id.xy) * 2.0 + 1.0) / (params.face_size) - 1.0);
	vec3 N = texelCoordToVec(uv, id.z);

	//vec4 color = color_interp;

	if (params.use_direct_write) {
		imageStore(dest_cubemap, ivec3(id), vec4(texture(source_cube, N).rgb, 1.0));
	} else {
		vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);

		for (uint sampleNum = 0u; sampleNum < params.sample_count; sampleNum++) {
			vec2 xi = Hammersley(sampleNum, params.sample_count);

			vec3 H = ImportanceSampleGGX(xi, params.roughness, N);
			vec3 V = N;
			vec3 L = (2.0 * dot(V, H) * H - V);

			float ndotl = clamp(dot(N, L), 0.0, 1.0);

			if (ndotl > 0.0) {
				sum.rgb += textureLod(source_cube, L, 0.0).rgb * ndotl;
				sum.a += ndotl;
			}
		}
		sum /= sum.a;

		imageStore(dest_cubemap, ivec3(id), vec4(sum.rgb, 1.0));
	}
}
