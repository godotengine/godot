/* clang-format off */
[vertex]

#ifdef USE_GLES_OVER_GL
#define lowp
#define mediump
#define highp
#else
precision highp float;
precision highp int;
#endif

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

#ifdef USE_GLES_OVER_GL
#define lowp
#define mediump
#define highp
#else
#if defined(USE_HIGHP_PRECISION)
precision highp float;
precision highp int;
#else
precision mediump float;
precision mediump int;
#endif

#endif

#ifdef USE_SOURCE_PANORAMA
uniform sampler2D source_panorama; //texunit:0
#else
uniform samplerCube source_cube; //texunit:0
#endif
/* clang-format on */

uniform int face_id;
uniform float roughness;
in highp vec2 uv_interp;

uniform sampler2D radical_inverse_vdc_cache; // texunit:1

#define M_PI 3.14159265359

#ifdef LOW_QUALITY

#define SAMPLE_COUNT 64

#else

#define SAMPLE_COUNT 512

#endif

#ifdef USE_SOURCE_PANORAMA

vec4 texturePanorama(sampler2D pano, vec3 normal) {
	vec2 st = vec2(
			atan(normal.x, normal.z),
			acos(normal.y));

	if (st.x < 0.0)
		st.x += M_PI * 2.0;

	st /= vec2(M_PI * 2.0, M_PI);

	return textureLod(pano, st, 0.0);
}

#endif

vec3 texelCoordToVec(vec2 uv, int faceID) {
	mat3 faceUvVectors[6];

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
	vec3 result;
	for (int i = 0; i < 6; i++) {
		if (i == faceID) {
			result = (faceUvVectors[i][0] * uv.x) + (faceUvVectors[i][1] * uv.y) + faceUvVectors[i][2];
			break;
		}
	}
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

float radical_inverse_VdC(int i) {
	return texture(radical_inverse_vdc_cache, vec2(float(i) / 512.0, 0.0)).x;
}

vec2 Hammersley(int i, int N) {
	return vec2(float(i) / float(N), radical_inverse_VdC(i));
}

uniform bool z_flip;

layout(location = 0) out vec4 frag_color;

void main() {
	vec3 color = vec3(0.0);

	vec2 uv = (uv_interp * 2.0) - 1.0;
	vec3 N = texelCoordToVec(uv, face_id);

#ifdef USE_DIRECT_WRITE

#ifdef USE_SOURCE_PANORAMA

	frag_color = vec4(texturePanorama(source_panorama, N).rgb, 1.0);
#else

	frag_color = vec4(textureCube(source_cube, N).rgb, 1.0);
#endif //USE_SOURCE_PANORAMA

#else

	vec4 sum = vec4(0.0);

	for (int sample_num = 0; sample_num < SAMPLE_COUNT; sample_num++) {
		vec2 xi = Hammersley(sample_num, SAMPLE_COUNT);

		vec3 H = ImportanceSampleGGX(xi, roughness, N);
		vec3 V = N;
		vec3 L = (2.0 * dot(V, H) * H - V);

		float NdotL = clamp(dot(N, L), 0.0, 1.0);

		if (NdotL > 0.0) {

#ifdef USE_SOURCE_PANORAMA
			vec3 val = texturePanorama(source_panorama, L).rgb;
#else
			vec3 val = textureCubeLod(source_cube, L, 0.0).rgb;
#endif
			//mix using Linear, to approximate high end back-end
			val = mix(pow((val + vec3(0.055)) * (1.0 / (1.0 + 0.055)), vec3(2.4)), val * (1.0 / 12.92), vec3(lessThan(val, vec3(0.04045))));

			sum.rgb += val * NdotL;

			sum.a += NdotL;
		}
	}

	sum /= sum.a;

	vec3 a = vec3(0.055);
	sum.rgb = mix((vec3(1.0) + a) * pow(sum.rgb, vec3(1.0 / 2.4)) - a, 12.92 * sum.rgb, vec3(lessThan(sum.rgb, vec3(0.0031308))));

	frag_color = vec4(sum.rgb, 1.0);
#endif
}
