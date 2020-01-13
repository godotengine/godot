/* clang-format off */
[compute]
/* clang-format on */
#version 450

VERSION_DEFINES

#define BLOCK_SIZE 8

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;


#ifdef MODE_GEN_BLUR_SIZE
layout(rgba16f, set = 0, binding = 0) uniform restrict image2D color_image;
layout(set = 1, binding = 0) uniform sampler2D source_depth;
#endif

#ifdef MODE_GEN_BOKEH
layout(set = 2, binding = 0) uniform sampler2D color_texture;
layout(set = 1, binding = 0) uniform sampler2D source_depth;
layout(rgba16f, set = 0, binding = 0) uniform restrict writeonly image2D bokeh_image;
#endif

#ifdef MODE_COMPOSITE_BOKEH
layout(rgba16f, set = 0, binding = 0) uniform restrict image2D color_image;
layout(set = 1, binding = 0) uniform sampler2D source_bokeh;
#endif




layout(push_constant, binding = 1, std430) uniform Params {
	ivec2 size;
	float z_far;
	float z_near;

	bool orthogonal;
	float blur_size;
	float blur_scale;
	uint pad;

	bool blur_near_active;
	float blur_near_begin;
	float blur_near_end;
	bool blur_far_active;
	float blur_far_begin;
	float blur_far_end;
	uint pad2[2];

} params;

#ifndef MODE_COMPOSITE_BOKEH

float get_depth_at_pos(vec2 uv) {
	float depth = textureLod(source_depth,uv,0.0).x;
	if (params.orthogonal) {
		depth = ((depth + (params.z_far + params.z_near) / (params.z_far - params.z_near)) * (params.z_far - params.z_near)) / 2.0;
	} else {
		depth = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near - depth * (params.z_far - params.z_near));
	}
	return depth;
}

float get_blur_size(float depth) {

	if (params.blur_near_active && depth < params.blur_near_begin) {
		return smoothstep(params.blur_near_end,params.blur_near_begin,depth) * params.blur_size;
	}

	if (params.blur_far_active && depth > params.blur_far_begin) {
		return smoothstep(params.blur_far_begin,params.blur_far_end,depth) * params.blur_size;
	}

	return 0.0;
}

#endif

const float GOLDEN_ANGLE = 2.39996323;

void main() {

	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThan(pos,params.size))) { //too large, do nothing
		return;
	}

	vec2 uv = vec2(pos) / vec2(params.size);

#ifdef MODE_GEN_BLUR_SIZE
	//precompute size in alpha channel
	float depth = get_depth_at_pos(uv);
	float size = get_blur_size(depth);

	vec4 color = imageLoad(color_image,pos);
	color.a = size;
	imageStore(color_image,pos,color);
#endif

#ifdef MODE_GEN_BOKEH

	float depth = get_depth_at_pos(uv);
	float size = get_blur_size(depth);
	vec4 color = texture(color_texture,uv);
	float accum = 1.0;
	float radius = params.blur_scale;
	vec2 pixel_size = 1.0/vec2(params.size);

	for (float ang = 0.0; radius < params.blur_size; ang += GOLDEN_ANGLE) {

		vec2 suv = uv + vec2(cos(ang), sin(ang)) * pixel_size * radius;
		vec4 sample_color = texture(color_texture, suv);
		float sample_depth = get_depth_at_pos(suv);
		if (sample_depth > depth) {
			sample_color.a = clamp(sample_color.a, 0.0, size*2.0);
		}

		float m = smoothstep(radius-0.5, radius+0.5, sample_color.a);
		color += mix(color/accum, sample_color, m);
		accum += 1.0;
		radius += params.blur_size/radius;
	}

	color /= accum;

	imageStore(bokeh_image,pos,color);
#endif

#ifdef MODE_COMPOSITE_BOKEH

	vec4 color = imageLoad(color_image,pos);
	vec4 bokeh = texture(source_bokeh,uv);
	if (max(color.a,bokeh.a) > 0.5) { //there is some blur in this pixel, so use bokeh
		color = bokeh;
	}
	color.a=0; //reset alpha
	imageStore(color_image,pos,color);
#endif

}
