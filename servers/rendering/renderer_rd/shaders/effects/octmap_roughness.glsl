#[compute]

#version 450

#VERSION_DEFINES

#define GROUP_SIZE 8

layout(local_size_x = GROUP_SIZE, local_size_y = GROUP_SIZE, local_size_z = 1) in;

shared vec4 samples[GROUP_SIZE * GROUP_SIZE * 4]; // Up to 256 samples supported. Should never need more than that.

layout(set = 0, binding = 0) uniform sampler2D source_oct;

layout(OCTMAP_FORMAT, set = 1, binding = 0) uniform restrict writeonly image2D dest_octmap;

#include "../oct_inc.glsl"
#include "octmap_roughness_inc.glsl"

void main() {
	uvec2 id = gl_GlobalInvocationID.xy;
	vec2 inv_source_size = 1.0 / vec2(params.source_size);
	vec2 inv_dest_size = 1.0 / vec2(params.dest_size);
	vec2 uv = (vec2(id.xy) + 0.5) * inv_dest_size;
	if (params.use_direct_write) {
		if (id.x < params.dest_size && id.y < params.dest_size) {
			imageStore(dest_octmap, ivec2(id), vec4(texture(source_oct, uv).rgb, 1.0));
		}
	} else {
		float solid_angle_texel = 4.0 * M_PI / float(params.dest_size * params.dest_size);
		float roughness2 = params.roughness * params.roughness;
		float roughness4 = roughness2 * roughness2;

		uint scaled_samples = max(uint(float(params.sample_count * 4) * params.roughness), 4);

		// This effectively rounds the sample count up to the nearest (GROUP_SIZE * GROUP_SIZE).
		uint samples_per_thread = max(1, ((scaled_samples) / (GROUP_SIZE * GROUP_SIZE)));
		uint total_samples = samples_per_thread * (GROUP_SIZE * GROUP_SIZE);

		for (uint local_sample = 0; local_sample < samples_per_thread; local_sample++) {
			uint sample_idx = local_sample * (GROUP_SIZE * GROUP_SIZE) + gl_LocalInvocationIndex;
			vec2 xi = Hammersley(sample_idx, total_samples);
			vec3 H_local = ImportanceSampleGGX(xi, roughness4);
			float NdotH = H_local.z;
			vec3 L_local = 2.0 * NdotH * H_local - vec3(0.0, 0.0, 1.0);

			float ndotl = L_local.z;
			if (ndotl > 0.0) {
				float D = DistributionGGX(NdotH, roughness4);
				float pdf = D * NdotH / (4.0 * NdotH) + 0.0001;

				float solid_angle_sample = 1.0 / (float(total_samples) * pdf + 0.0001);

				float mipLevel = 0.5 * log2(solid_angle_sample / solid_angle_texel);
				samples[sample_idx] = vec4(L_local, mipLevel);
			} else {
				samples[sample_idx] = vec4(-1.0);
			}
		}

		memoryBarrierShared();
		barrier();

		if (id.x < params.dest_size && id.y < params.dest_size) {
			vec3 N = oct_to_vec3_with_border(uv, params.border_size.y);
			vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
			vec3 UpVector = abs(N.y) < 0.99999 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
			mat3 T;
			T[0] = normalize(cross(UpVector, N));
			T[1] = cross(N, T[0]);
			T[2] = N;

			for (uint i = 0; i < total_samples; i++) {
				vec4 s = samples[i];
				float ndotl = s.z;
				if (ndotl > 0.0) {
					vec3 L_world = T * s.xyz;
					vec2 sample_uv = vec3_to_oct_with_border(L_world, params.border_size);
					sum.rgb += textureLod(source_oct, sample_uv, s.w).rgb * ndotl;
					sum.a += ndotl;
				}
			}

			imageStore(dest_octmap, ivec2(id), vec4(sum.rgb / sum.a, 1.0));
		}
	}
}
