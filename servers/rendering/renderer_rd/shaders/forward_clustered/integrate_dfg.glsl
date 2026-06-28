#[compute]
#version 450

// References:
// https://www.gamedevs.org/uploads/real-shading-in-unreal-engine-4.pdf
// https://google.github.io/filament/Filament.html
// https://learnopengl.com/PBR/IBL/Specular-IBL

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 0) uniform restrict writeonly image2D current_image;

#define M_PI 3.14159265359
#define SAMPLE_COUNT 1024
#define SIZE 128

#define saturate(x) clamp(x, 0, 1)

// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation
float radical_inverse_vdc(uint bits) {
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint i, float n) {
	return vec2(float(i) / n, radical_inverse_vdc(i));
}

vec3 importance_sample_ggx(vec2 Xi, vec3 N, float roughness) {
	float a = roughness * roughness;

	float phi = 2.0 * M_PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

	// from spherical coordinates to cartesian coordinates - halfway vector
	vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;

	// from tangent-space H vector to world-space sample vector
	vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);

	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}

float geometry_schlick_ggx(float NdotV, float roughness) {
	// note that we use a different k for IBL
	float a = roughness;
	float k = (a * a) / 2.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom / denom;
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
	float NdotV = saturate(dot(N, V));
	float NdotL = saturate(dot(N, L));
	float ggx2 = geometry_schlick_ggx(NdotV, roughness);
	float ggx1 = geometry_schlick_ggx(NdotL, roughness);

	return ggx1 * ggx2;
}

vec3 importance_uniform_sample(vec2 u) {
	float phi = 2.0f * M_PI * u.x;
	float cosTheta = 1 - u.y;
	float sinTheta = sqrt(1 - cosTheta * cosTheta);
	return vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

float distribution_charlie(float NoH, float roughness) {
	// Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF"
	float a = roughness * roughness;
	float invAlpha = 1 / a;
	float cos2h = NoH * NoH;
	float sin2h = 1 - cos2h;
	return (2.0f + invAlpha) * pow(sin2h, invAlpha * 0.5f) / (2.0f * M_PI);
}

float visibility_ashikhmin(float NoV, float NoL) {
	// Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
	return 1 / (4 * (NoL + NoV - NoL * NoV));
}

void integrate_brdfs(float n_dot_v, float roughness, out vec2 brdf, out float cloth_brdf) {
	vec3 v = vec3(sqrt(1.0 - n_dot_v * n_dot_v), 0, n_dot_v);
	vec3 n = vec3(0.0f, 0.0f, 1.0f);
	float A = 0.0f;
	float B = 0.0f;
	float C = 0.0f;

	for (uint i = 0; i < SAMPLE_COUNT; ++i) {
		vec2 Xi = hammersley(i, SAMPLE_COUNT);
		vec3 h = importance_sample_ggx(Xi, n, roughness);
		vec3 l = normalize(2.0 * dot(v, h) * h - v);

		float n_dot_l = saturate(l.z);
		float n_dot_h = saturate(h.z);
		float v_dot_h = saturate(dot(v, h));

		if (n_dot_l > 0.0) {
			float G = geometry_smith(n, v, l, roughness);
			float G_Vis = (G * v_dot_h) / (n_dot_h * n_dot_v);
			float Fc = pow(1.0 - v_dot_h, 5.0);

			// LDFG term for multiscattering
			// https://google.github.io/filament/Filament.html#toc5.3.4.7
			A += Fc * G_Vis;
			B += G_Vis;
		}

		// Cloth BRDF calculations
		// https://github.com/google/filament/blob/main/libs/ibl/src/CubemapIBL.cpp#L856-L874
		vec3 h_cloth = importance_uniform_sample(Xi);
		vec3 l_cloth = normalize(2.0 * dot(v, h_cloth) * h_cloth - v);
		float n_dot_l_cloth = saturate(l_cloth.z);
		float n_dot_h_cloth = saturate(h_cloth.z);
		float v_dot_h_cloth = saturate(dot(v, h_cloth));

		if (n_dot_l_cloth > 0.0) {
			float v_cloth = visibility_ashikhmin(n_dot_v, n_dot_l_cloth);
			float d_cloth = distribution_charlie(n_dot_h_cloth, roughness);
			C += v_cloth * d_cloth * n_dot_l_cloth * v_dot_h_cloth;
		}
	}

	A /= float(SAMPLE_COUNT);
	B /= float(SAMPLE_COUNT);
	C *= (4.0 * 2.0 * M_PI / SAMPLE_COUNT);

	brdf = vec2(A, B);
	cloth_brdf = C;
}

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	float roughness = float(pos.y + 0.5f) / SIZE;
	float NdotV = float(pos.x + 0.5f) / SIZE;
	vec2 brdf;
	float cloth_brdf;
	integrate_brdfs(NdotV, roughness, brdf, cloth_brdf);
	ivec2 out_pos = ivec2(pos.x, (SIZE - 1) - pos.y);
	imageStore(current_image, out_pos, vec4(brdf, cloth_brdf, 1.0));
}
