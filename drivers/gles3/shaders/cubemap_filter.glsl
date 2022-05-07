/* clang-format off */
[vertex]

layout(location = 0) in highp vec2 vertex;
/* clang-format on */

layout(location = 4) in highp vec2 uv;

out highp vec2 uv_interp;

void main() {
	uv_interp = uv;
	gl_Position = vec4(vertex, 0, 1);
}

/* clang-format off */
[fragment]

precision highp float;
/* clang-format on */
precision highp int;

#ifdef USE_SOURCE_PANORAMA
uniform sampler2D source_panorama; //texunit:0
uniform float source_resolution;
#endif

#ifdef USE_SOURCE_DUAL_PARABOLOID_ARRAY
uniform sampler2DArray source_dual_paraboloid_array; //texunit:0
uniform int source_array_index;
#endif

#ifdef USE_SOURCE_DUAL_PARABOLOID
uniform sampler2D source_dual_paraboloid; //texunit:0
#endif

#if defined(USE_SOURCE_DUAL_PARABOLOID) || defined(COMPUTE_IRRADIANCE)
uniform float source_mip_level;
#endif

#if !defined(USE_SOURCE_DUAL_PARABOLOID_ARRAY) && !defined(USE_SOURCE_PANORAMA) && !defined(USE_SOURCE_DUAL_PARABOLOID)
uniform samplerCube source_cube; //texunit:0
#endif

uniform int face_id;
uniform float roughness;

in highp vec2 uv_interp;

layout(location = 0) out vec4 frag_color;

#define M_PI 3.14159265359

vec3 texelCoordToVec(vec2 uv, int faceID) {
	mat3 faceUvVectors[6];
	/*
	// -x
	faceUvVectors[1][0] = vec3(0.0, 0.0, 1.0);  // u -> +z
	faceUvVectors[1][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[1][2] = vec3(-1.0, 0.0, 0.0); // -x face

	// +x
	faceUvVectors[0][0] = vec3(0.0, 0.0, -1.0); // u -> -z
	faceUvVectors[0][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[0][2] = vec3(1.0, 0.0, 0.0);  // +x face

	// -y
	faceUvVectors[3][0] = vec3(1.0, 0.0, 0.0);  // u -> +x
	faceUvVectors[3][1] = vec3(0.0, 0.0, -1.0); // v -> -z
	faceUvVectors[3][2] = vec3(0.0, -1.0, 0.0); // -y face

	// +y
	faceUvVectors[2][0] = vec3(1.0, 0.0, 0.0);  // u -> +x
	faceUvVectors[2][1] = vec3(0.0, 0.0, 1.0);  // v -> +z
	faceUvVectors[2][2] = vec3(0.0, 1.0, 0.0);  // +y face

	// -z
	faceUvVectors[5][0] = vec3(-1.0, 0.0, 0.0); // u -> -x
	faceUvVectors[5][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[5][2] = vec3(0.0, 0.0, -1.0); // -z face

	// +z
	faceUvVectors[4][0] = vec3(1.0, 0.0, 0.0);  // u -> +x
	faceUvVectors[4][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[4][2] = vec3(0.0, 0.0, 1.0);  // +z face
	*/

	// -x
	faceUvVectors[0][0] = vec3(0.0, 0.0, 1.0); // u -> +z
	faceUvVectors[0][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[0][2] = vec3(-1.0, 0.0, 0.0); // -x face

	// +x
	faceUvVectors[1][0] = vec3(0.0, 0.0, -1.0); // u -> -z
	faceUvVectors[1][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[1][2] = vec3(1.0, 0.0, 0.0); // +x face

	// -y
	faceUvVectors[2][0] = vec3(1.0, 0.0, 0.0); // u -> +x
	faceUvVectors[2][1] = vec3(0.0, 0.0, -1.0); // v -> -z
	faceUvVectors[2][2] = vec3(0.0, -1.0, 0.0); // -y face

	// +y
	faceUvVectors[3][0] = vec3(1.0, 0.0, 0.0); // u -> +x
	faceUvVectors[3][1] = vec3(0.0, 0.0, 1.0); // v -> +z
	faceUvVectors[3][2] = vec3(0.0, 1.0, 0.0); // +y face

	// -z
	faceUvVectors[4][0] = vec3(-1.0, 0.0, 0.0); // u -> -x
	faceUvVectors[4][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[4][2] = vec3(0.0, 0.0, -1.0); // -z face

	// +z
	faceUvVectors[5][0] = vec3(1.0, 0.0, 0.0); // u -> +x
	faceUvVectors[5][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[5][2] = vec3(0.0, 0.0, 1.0); // +z face

	// out = u * s_faceUv[0] + v * s_faceUv[1] + s_faceUv[2].
	vec3 result = (faceUvVectors[faceID][0] * uv.x) + (faceUvVectors[faceID][1] * uv.y) + faceUvVectors[faceID][2];
	return normalize(result);
}

vec3 ImportanceSampleGGX(vec2 Xi, float Roughness, vec3 N) {
	float a = Roughness * Roughness; // DISNEY'S ROUGHNESS [see Burley'12 siggraph]

	// Compute distribution direction
	float Phi = 2.0 * M_PI * Xi.x;
	float CosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
	float SinTheta = sqrt(1.0 - CosTheta * CosTheta);

	// Convert to spherical direction
	vec3 H;
	H.x = SinTheta * cos(Phi);
	H.y = SinTheta * sin(Phi);
	H.z = CosTheta;

	vec3 UpVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 TangentX = normalize(cross(UpVector, N));
	vec3 TangentY = cross(N, TangentX);

	// Tangent to world space
	return TangentX * H.x + TangentY * H.y + N * H.z;
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = M_PI * denom * denom;

	return nom / denom;
}

// http://graphicrants.blogspot.com.au/2013/08/specular-brdf-reference.html
float GGX(float NdotV, float a) {
	float k = a / 2.0;
	return NdotV / (NdotV * (1.0 - k) + k);
}

// http://graphicrants.blogspot.com.au/2013/08/specular-brdf-reference.html
float G_Smith(float a, float nDotV, float nDotL) {
	return GGX(nDotL, a * a) * GGX(nDotV, a * a);
}

float radicalInverse_VdC(uint bits) {
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint N) {
	return vec2(float(i) / float(N), radicalInverse_VdC(i));
}

#ifdef LOW_QUALITY

#define SAMPLE_COUNT 64u
#define SAMPLE_DELTA 0.1

#else

#define SAMPLE_COUNT 512u
#define SAMPLE_DELTA 0.03

#endif

uniform bool z_flip;

#ifdef USE_SOURCE_PANORAMA

vec4 texturePanorama(vec3 normal, sampler2D pano, float mipLevel) {
	vec2 st = vec2(
			atan(normal.x, normal.z),
			acos(normal.y));

	if (st.x < 0.0)
		st.x += M_PI * 2.0;

	st /= vec2(M_PI * 2.0, M_PI);

	return textureLod(pano, st, mipLevel);
}

#endif

#ifdef USE_SOURCE_DUAL_PARABOLOID_ARRAY

vec4 textureDualParaboloidArray(vec3 normal) {
	vec3 norm = normalize(normal);
	norm.xy /= 1.0 + abs(norm.z);
	norm.xy = norm.xy * vec2(0.5, 0.25) + vec2(0.5, 0.25);
	if (norm.z < 0.0) {
		norm.y = 0.5 - norm.y + 0.5;
	}
	return textureLod(source_dual_paraboloid_array, vec3(norm.xy, float(source_array_index)), 0.0);
}

#endif

#ifdef USE_SOURCE_DUAL_PARABOLOID
vec4 textureDualParaboloid(vec3 normal) {
	vec3 norm = normalize(normal);
	norm.xy /= 1.0 + abs(norm.z);
	norm.xy = norm.xy * vec2(0.5, 0.25) + vec2(0.5, 0.25);
	if (norm.z < 0.0) {
		norm.y = 0.5 - norm.y + 0.5;
	}
	return textureLod(source_dual_paraboloid, norm.xy, source_mip_level);
}

#endif

void main() {
#ifdef USE_DUAL_PARABOLOID

	vec3 N = vec3(uv_interp * 2.0 - 1.0, 0.0);
	N.z = 0.5 - 0.5 * ((N.x * N.x) + (N.y * N.y));
	N = normalize(N);

	if (z_flip) {
		N.y = -N.y; //y is flipped to improve blending between both sides
		N.z = -N.z;
	}

#else
	vec2 uv = (uv_interp * 2.0) - 1.0;
	vec3 N = texelCoordToVec(uv, face_id);
#endif
	//vec4 color = color_interp;

#ifdef USE_DIRECT_WRITE

#ifdef USE_SOURCE_PANORAMA

	frag_color = vec4(texturePanorama(N, source_panorama, 0.0).rgb, 1.0);
#endif

#ifdef USE_SOURCE_DUAL_PARABOLOID_ARRAY

	frag_color = vec4(textureDualParaboloidArray(N).rgb, 1.0);
#endif

#ifdef USE_SOURCE_DUAL_PARABOLOID

	frag_color = vec4(textureDualParaboloid(N).rgb, 1.0);
#endif

#if !defined(USE_SOURCE_DUAL_PARABOLOID_ARRAY) && !defined(USE_SOURCE_PANORAMA) && !defined(USE_SOURCE_DUAL_PARABOLOID)

	N.y = -N.y;
	frag_color = vec4(texture(N, source_cube).rgb, 1.0);
#endif

#else // USE_DIRECT_WRITE

#ifdef COMPUTE_IRRADIANCE

	vec3 irradiance = vec3(0.0);

	// tangent space calculation from origin point
	vec3 UpVector = vec3(0.0, 1.0, 0.0);
	vec3 TangentX = cross(UpVector, N);
	vec3 TangentY = cross(N, TangentX);

	float num_samples = 0.0f;

	for (float phi = 0.0; phi < 2.0 * M_PI; phi += SAMPLE_DELTA) {
		for (float theta = 0.0; theta < 0.5 * M_PI; theta += SAMPLE_DELTA) {
			// Calculate sample positions
			vec3 tangentSample = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
			// Find world vector of sample position
			vec3 H = tangentSample.x * TangentX + tangentSample.y * TangentY + tangentSample.z * N;

			vec2 st = vec2(atan(H.x, H.z), acos(H.y));
			if (st.x < 0.0) {
				st.x += M_PI * 2.0;
			}
			st /= vec2(M_PI * 2.0, M_PI);

			irradiance += textureLod(source_panorama, st, source_mip_level).rgb * cos(theta) * sin(theta);
			num_samples++;
		}
	}
	irradiance = M_PI * irradiance * (1.0 / float(num_samples));

	frag_color = vec4(irradiance, 1.0);

#else

	vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);

	for (uint sampleNum = 0u; sampleNum < SAMPLE_COUNT; sampleNum++) {
		vec2 xi = Hammersley(sampleNum, SAMPLE_COUNT);

		vec3 H = normalize(ImportanceSampleGGX(xi, roughness, N));
		vec3 V = N;
		vec3 L = normalize(2.0 * dot(V, H) * H - V);

		float ndotl = max(dot(N, L), 0.0);

		if (ndotl > 0.0) {

#ifdef USE_SOURCE_PANORAMA
			float D = DistributionGGX(N, H, roughness);
			float ndoth = max(dot(N, H), 0.0);
			float hdotv = max(dot(H, V), 0.0);
			float pdf = D * ndoth / (4.0 * hdotv) + 0.0001;

			float saTexel = 4.0 * M_PI / (6.0 * source_resolution * source_resolution);
			float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);

			float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel);

			sum.rgb += texturePanorama(L, source_panorama, mipLevel).rgb * ndotl;
#endif

#ifdef USE_SOURCE_DUAL_PARABOLOID_ARRAY
			sum.rgb += textureDualParaboloidArray(L).rgb * ndotl;
#endif

#ifdef USE_SOURCE_DUAL_PARABOLOID
			sum.rgb += textureDualParaboloid(L).rgb * ndotl;
#endif

#if !defined(USE_SOURCE_DUAL_PARABOLOID_ARRAY) && !defined(USE_SOURCE_PANORAMA) && !defined(USE_SOURCE_DUAL_PARABOLOID)
			L.y = -L.y;
			sum.rgb += textureLod(source_cube, L, 0.0).rgb * ndotl;
#endif
			sum.a += ndotl;
		}
	}
	sum /= sum.a;

	frag_color = vec4(sum.rgb, 1.0);

#endif // COMPUTE_IRRADIANCE
#endif // USE_DIRECT_WRITE
}
