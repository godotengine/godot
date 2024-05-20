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

	if (params.use_direct_write) {
		imageStore(dest_cubemap, ivec3(id), vec4(texture(source_cube, N).rgb, 1.0));
	} else {
		vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);

		float solid_angle_texel = 4.0 * M_PI / (6.0 * params.face_size * params.face_size);
		float roughness2 = params.roughness * params.roughness;
		float roughness4 = roughness2 * roughness2;
		vec3 UpVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
		mat3 T;
		T[0] = normalize(cross(UpVector, N));
		T[1] = cross(N, T[0]);
		T[2] = N;

		for (uint sampleNum = 0u; sampleNum < params.sample_count; sampleNum++) {
			vec2 xi = Hammersley(sampleNum, params.sample_count);

			vec3 H = T * ImportanceSampleGGX(xi, roughness4);
			float NdotH = dot(N, H);
			vec3 L = (2.0 * NdotH * H - N);

			float ndotl = clamp(dot(N, L), 0.0, 1.0);

			if (ndotl > 0.0) {
				float D = DistributionGGX(NdotH, roughness4);
				float pdf = D * NdotH / (4.0 * NdotH) + 0.0001;

				float solid_angle_sample = 1.0 / (float(params.sample_count) * pdf + 0.0001);

				float mipLevel = params.roughness == 0.0 ? 0.0 : 0.5 * log2(solid_angle_sample / solid_angle_texel);

				sum.rgb += textureLod(source_cube, L, mipLevel).rgb * ndotl;
				sum.a += ndotl;
			}
		}
		sum /= sum.a;

		imageStore(dest_cubemap, ivec3(id), vec4(sum.rgb, 1.0));
	}
}
