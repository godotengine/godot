/* clang-format off */
#[vertex]

#version 450

#VERSION_DEFINES

#include "cubemap_roughness_inc.glsl"

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

void main() {
	vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
	gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
	uv_interp = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
}

/* clang-format off */
#[fragment]

#version 450

#VERSION_DEFINES

#include "cubemap_roughness_inc.glsl"

layout(location = 0) in vec2 uv_interp;

layout(set = 0, binding = 0) uniform samplerCube source_cube;

layout(location = 0) out vec4 frag_color;
/* clang-format on */

void main() {
	vec3 N = texelCoordToVec(uv_interp * 2.0 - 1.0, params.face_id);

	//vec4 color = color_interp;

	if (params.use_direct_write) {
		frag_color = vec4(texture(source_cube, N).rgb, 1.0);
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

		frag_color = vec4(sum.rgb, 1.0);
	}
}
