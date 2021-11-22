/* clang-format off */
#[vertex]

#version 450

#VERSION_DEFINES

#include "cubemap_roughness_inc.glsl"

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

void main() {
	vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
	uv_interp = base_arr[gl_VertexIndex];
	gl_Position = vec4(uv_interp * 2.0 - 1.0, 0.0, 1.0);
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

		frag_color = vec4(sum.rgb, 1.0);
	}
}
