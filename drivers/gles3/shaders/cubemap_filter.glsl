/* clang-format off */
#[modes]

mode_default =
mode_copy = #define MODE_DIRECT_WRITE

#[specializations]

#[vertex]

layout(location = 0) in highp vec2 vertex_attrib;
/* clang-format on */

out highp vec2 uv_interp;

void main() {
	uv_interp = vertex_attrib;
	gl_Position = vec4(uv_interp, 0.0, 1.0);
}

/* clang-format off */
#[fragment]


#define M_PI 3.14159265359

uniform samplerCube source_cube; //texunit:0

/* clang-format on */

uniform int face_id;
uniform float roughness;
uniform float face_size;
uniform int sample_count;

//Todo, profile on low end hardware to see if fixed loop is faster
#ifdef USE_FIXED_SAMPLES
#define FIXED_SAMPLE_COUNT 32
#endif

in highp vec2 uv_interp;

uniform sampler2D radical_inverse_vdc_cache; // texunit:1

layout(location = 0) out vec4 frag_color;

#define M_PI 3.14159265359

// Don't include tonemap_inc.glsl because all we want is these functions, we don't want the uniforms
vec3 linear_to_srgb(vec3 color) {
	return max(vec3(1.055) * pow(color, vec3(0.416666667)) - vec3(0.055), vec3(0.0));
}

vec3 srgb_to_linear(vec3 color) {
	return color * (color * (color * 0.305306011 + 0.682171111) + 0.012522878);
}

vec3 texelCoordToVec(vec2 uv, int faceID) {
	mat3 faceUvVectors[6];

	// -x
	faceUvVectors[1][0] = vec3(0.0, 0.0, 1.0); // u -> +z
	faceUvVectors[1][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[1][2] = vec3(-1.0, 0.0, 0.0); // -x face

	// +x
	faceUvVectors[0][0] = vec3(0.0, 0.0, -1.0); // u -> -z
	faceUvVectors[0][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[0][2] = vec3(1.0, 0.0, 0.0); // +x face

	// -y
	faceUvVectors[3][0] = vec3(1.0, 0.0, 0.0); // u -> +x
	faceUvVectors[3][1] = vec3(0.0, 0.0, -1.0); // v -> -z
	faceUvVectors[3][2] = vec3(0.0, -1.0, 0.0); // -y face

	// +y
	faceUvVectors[2][0] = vec3(1.0, 0.0, 0.0); // u -> +x
	faceUvVectors[2][1] = vec3(0.0, 0.0, 1.0); // v -> +z
	faceUvVectors[2][2] = vec3(0.0, 1.0, 0.0); // +y face

	// -z
	faceUvVectors[5][0] = vec3(-1.0, 0.0, 0.0); // u -> -x
	faceUvVectors[5][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[5][2] = vec3(0.0, 0.0, -1.0); // -z face

	// +z
	faceUvVectors[4][0] = vec3(1.0, 0.0, 0.0); // u -> +x
	faceUvVectors[4][1] = vec3(0.0, -1.0, 0.0); // v -> -y
	faceUvVectors[4][2] = vec3(0.0, 0.0, 1.0); // +z face

	// out = u * s_faceUv[0] + v * s_faceUv[1] + s_faceUv[2].
	vec3 result = (faceUvVectors[faceID][0] * uv.x) + (faceUvVectors[faceID][1] * uv.y) + faceUvVectors[faceID][2];
	return normalize(result);
}

vec3 ImportanceSampleGGX(vec2 xi, float roughness4) {
	// Compute distribution direction
	float Phi = 2.0 * M_PI * xi.x;
	float CosTheta = sqrt((1.0 - xi.y) / (1.0 + (roughness4 - 1.0) * xi.y));
	float SinTheta = sqrt(1.0 - CosTheta * CosTheta);

	// Convert to spherical direction
	vec3 H;
	H.x = SinTheta * cos(Phi);
	H.y = SinTheta * sin(Phi);
	H.z = CosTheta;

	return H;
}

float DistributionGGX(float NdotH, float roughness4) {
	float NdotH2 = NdotH * NdotH;
	float denom = (NdotH2 * (roughness4 - 1.0) + 1.0);
	denom = M_PI * denom * denom;

	return roughness4 / denom;
}

// https://graphicrants.blogspot.com.au/2013/08/specular-brdf-reference.html
float GGX(float NdotV, float a) {
	float k = a / 2.0;
	return NdotV / (NdotV * (1.0 - k) + k);
}

// https://graphicrants.blogspot.com.au/2013/08/specular-brdf-reference.html
float G_Smith(float a, float nDotV, float nDotL) {
	return GGX(nDotL, a * a) * GGX(nDotV, a * a);
}

float radical_inverse_VdC(int i) {
	return texture(radical_inverse_vdc_cache, vec2(float(i) / 512.0, 0.0)).x;
}

vec2 Hammersley(int i, int N) {
	return vec2(float(i) / float(N), radical_inverse_VdC(i));
}

void main() {
	vec3 color = vec3(0.0);
	vec2 uv = uv_interp;
	vec3 N = texelCoordToVec(uv, face_id);

#ifdef MODE_DIRECT_WRITE
	frag_color = vec4(textureLod(source_cube, N, 0.0).rgb, 1.0);
#else

	vec4 sum = vec4(0.0);
	float solid_angle_texel = 4.0 * M_PI / (6.0 * face_size * face_size);
	float roughness2 = roughness * roughness;
	float roughness4 = roughness2 * roughness2;
	vec3 UpVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	mat3 T;
	T[0] = normalize(cross(UpVector, N));
	T[1] = cross(N, T[0]);
	T[2] = N;

	for (int sample_num = 0; sample_num < sample_count; sample_num++) {
		vec2 xi = Hammersley(sample_num, sample_count);

		vec3 H = T * ImportanceSampleGGX(xi, roughness4);
		float NdotH = dot(N, H);
		vec3 L = (2.0 * NdotH * H - N);

		float NdotL = clamp(dot(N, L), 0.0, 1.0);

		if (NdotL > 0.0) {
			float D = DistributionGGX(NdotH, roughness4);
			float pdf = D * NdotH / (4.0 * NdotH) + 0.0001;

			float solid_angle_sample = 1.0 / (float(sample_count) * pdf + 0.0001);

			float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(solid_angle_sample / solid_angle_texel);

			vec3 val = textureLod(source_cube, L, mipLevel).rgb;
			// Mix using linear
			val = srgb_to_linear(val);

			sum.rgb += val * NdotL;
			sum.a += NdotL;
		}
	}

	sum /= sum.a;

	sum.rgb = linear_to_srgb(sum.rgb);
	frag_color = vec4(sum.rgb, 1.0);
#endif
}
