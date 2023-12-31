#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(r8, set = 0, binding = 0) uniform restrict writeonly image2D current_image;

// This shader is used to generate a "best fit normal texture" as described by:
// https://advances.realtimerendering.com/s2010/Kaplanyan-CryEngine3(SIGGRAPH%202010%20Advanced%20RealTime%20Rendering%20Course).pdf
// This texture tells you what length of normal can be used to store a unit vector
// with the lest amount of error.

vec3 quantize(vec3 c) {
	return round(clamp(c * 0.5 + 0.5, 0.0, 1.0) * 255.0) * (1.0 / 255.0) * 2.0 - 1.0;
}

float find_minimum_error(vec3 normal) {
	float min_error = 100000.0;
	float t_best = 0.0;
	for (float nstep = 1.5; nstep < 127.5; ++nstep) {
		float t = nstep / 127.5;
		vec3 vp = normal * t;
		vec3 quantizedp = quantize(vp);
		vec3 vdiff = (quantizedp - vp) / t;
		float error = max(abs(vdiff.x), max(abs(vdiff.y), abs(vdiff.z)));
		if (error < min_error) {
			min_error = error;
			t_best = t;
		}
	}
	return t_best;
}

void main() {
	vec2 uv = vec2(gl_GlobalInvocationID.xy) * vec2(1.0 / 1024.0) + vec2(0.5 / 1024.0);
	uv.y *= uv.x;

	vec3 dir = vec3(uv.x, uv.y, 1.0);
	imageStore(current_image, ivec2(gl_GlobalInvocationID.xy), vec4(find_minimum_error(dir), 1.0, 1.0, 1.0));
}
