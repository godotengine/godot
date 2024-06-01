#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;

void main() {
	vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
	gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
	uv_interp = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
}

#[fragment]

#version 450

#VERSION_DEFINES

layout(location = 0) in vec2 uv_interp;

layout(set = 0, binding = 0) uniform sampler2D source_color_0;
layout(set = 1, binding = 0) uniform sampler2D source_color_1;
layout(set = 2, binding = 0) uniform sampler2D source_color_2;
layout(set = 3, binding = 0) uniform sampler2D source_color_3;
layout(set = 4, binding = 0) uniform sampler2D source_color_4;
layout(set = 5, binding = 0) uniform sampler2D source_color_5;

layout(push_constant, std430) uniform Params {
	float fovx;
}
params;

layout(location = 0) out vec4 frag_color;

const float subcamera_fov = 100.0f;

vec3 latlon_to_ray(vec2 latlon) {
	float lat = latlon.x;
	float lon = latlon.y;
	return vec3(sin(lon) * cos(lat), sin(lat), -cos(lon) * cos(lat));
}

vec3 panini_inverse(vec2 p) {
	float d = 1.0;
	float k = p.x * p.x / ((d + 1.0) * (d + 1.0));
	float dscr = k * k * d * d - (k + 1.0) * (k * d * d - 1.0);
	float clon = (-k * d + sqrt(dscr)) / (k + 1.0);
	float s = (d + 1.0) / (d + clon);
	float lon = atan(p.x, (s * clon));
	float lat = atan(p.y, s);
	return latlon_to_ray(vec2(lat, lon));
}
vec2 panini_forward(vec2 latlon) {
	float d = 1.0;
	float s = (d + 1.0) / (d + cos(latlon.y));
	float x = s * sin(latlon.y);
	float y = s * tan(latlon.x);
	return vec2(x, y);
}
vec3 panini_ray(vec2 p) {
	float scale = panini_forward(vec2(0.0, radians(params.fovx) / 2.0)).x;
	return panini_inverse(p * scale);
}

void main() {
	vec2 viewport_size = vec2(textureSize(source_color_0, 0));
	vec3 albedo = vec3(0.0);

	float view_ratio = viewport_size.x / viewport_size.y;
	vec2 uv = uv_interp * viewport_size / min(viewport_size.x, viewport_size.y);
	vec3 pos = vec3(0.0, 0.0, 0.0);
	if (view_ratio > 1.0) {
		pos = panini_ray(vec2(uv.x - 0.5 * view_ratio, uv.y - 0.5) * 1.0);
	} else {
		pos = panini_ray(vec2(uv.x - 0.5, uv.y - 0.5 / view_ratio) * 1.0);
	}
	if (pos == vec3(0.0, 0.0, 0.0)) {
		albedo = pos;
	} else {
		float ax = abs(pos.x);
		float ay = abs(pos.y);
		float az = abs(pos.z);

		float fov_fix = 1.0 / tan(radians(subcamera_fov / 2.0));
		if (az >= ax && az >= ay) {
			if (pos.z < 0.0) {
				uv = vec2(fov_fix * 0.5 * (vec2(pos.x / az, pos.y / az)) + 0.5);
				albedo = (texture(source_color_0, uv).rgb);
			} else {
				uv = vec2(fov_fix * 0.5 * (vec2(-pos.x / az, pos.y / az)) + 0.5);
				albedo = (texture(source_color_5, uv).rgb);
			}
		} else if (ax >= ay && ax >= az) {
			if (pos.x < 0.0) {
				uv = vec2(fov_fix * 0.5 * (vec2(-pos.z / ax, pos.y / ax)) + 0.5);
				albedo = (texture(source_color_1, uv).rgb);
			} else {
				uv = vec2(fov_fix * 0.5 * (vec2(pos.z / ax, pos.y / ax)) + 0.5);
				albedo = (texture(source_color_2, uv).rgb);
			}
		} else if (ay >= ax && ay >= az) {
			if (pos.y > 0.0) {
				uv = vec2(fov_fix * 0.5 * (vec2(pos.x / ay, pos.z / ay)) + 0.5);
				albedo = (texture(source_color_3, uv).rgb);
			} else {
				uv = vec2(fov_fix * 0.5 * (vec2(pos.x / ay, -pos.z / ay)) + 0.5);
				albedo = (texture(source_color_4, uv).rgb);
			}
		} else {
			albedo = vec3(0.0, 0.0, 0.0);
		}
	}
	frag_color = vec4(albedo, 1.0);
}
