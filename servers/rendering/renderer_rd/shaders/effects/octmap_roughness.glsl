#[compute]

#version 450

#VERSION_DEFINES

#define GROUP_SIZE 8

layout(local_size_x = GROUP_SIZE, local_size_y = GROUP_SIZE, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_oct;

layout(rgba16f, set = 1, binding = 0) uniform restrict writeonly image2D dest_octmap;

#include "../oct_inc.glsl"
#include "octmap_roughness_inc.glsl"

void main() {
	uvec2 id = gl_GlobalInvocationID.xy;
	if (id.x < params.dest_size && id.y < params.dest_size) {
		vec2 inv_source_size = 1.0 / vec2(params.source_size);
		vec2 inv_dest_size = 1.0 / vec2(params.dest_size);
		vec2 uv = (vec2(id.xy) + 0.5) * inv_dest_size;
		if (params.use_direct_write) {
			imageStore(dest_octmap, ivec2(id), vec4(texture(source_oct, uv).rgb, 1.0));
		} else {
			vec3 N = oct_to_vec3_with_border(uv, params.border_size.y);
			vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
			float solid_angle_texel = 4.0 * M_PI / float(params.dest_size * params.dest_size);
			float roughness2 = params.roughness * params.roughness;
			float roughness4 = roughness2 * roughness2;

			// https://jcgt.org/published/0006/01/01/
			float side = N.z >= 0.0f ? 1.0f : -1.0f;
			float a = -1.0f / (side + N.z);
			float b = N.x * N.y * a;
			mat3 T;
			T[0] = vec3(1.0f + side * N.x * N.x * a, side * b, -side * N.x);
			T[1] = vec3(b, side + N.y * N.y * a, -N.y);
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

					vec2 sample_uv = vec3_to_oct_with_border(L, params.border_size);
					sum.rgb += textureLod(source_oct, sample_uv, mipLevel).rgb * ndotl;
					sum.a += ndotl;
				}
			}

			imageStore(dest_octmap, ivec2(id), vec4(sum.rgb / sum.a, 1.0));
		}
	}
}
