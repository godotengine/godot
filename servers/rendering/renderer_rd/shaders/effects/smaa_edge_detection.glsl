/**
 * Copyright (C) 2013 Jorge Jimenez (jorge@iryoku.com)
 * Copyright (C) 2013 Jose I. Echevarria (joseignacioechevarria@gmail.com)
 * Copyright (C) 2013 Belen Masia (bmasia@unizar.es)
 * Copyright (C) 2013 Fernando Navarro (fernandn@microsoft.com)
 * Copyright (C) 2013 Diego Gutierrez (diegog@unizar.es)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to
 * do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software. As clarification, there
 * is no requirement that the copyright notice and permission be included in
 * binary distributions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#[vertex]
#version 450

layout(location = 0) out vec2 tex_coord;
#ifdef RENDER_DRIVER_WEBGPU
layout(location = 1) out vec4 offset0;
layout(location = 2) out vec4 offset1;
layout(location = 3) out vec4 offset2;
#else
layout(location = 1) out vec4 offset[3];
#endif

layout(push_constant, std430) uniform Params {
	vec2 inv_size;
	float threshold;
	float reserved;
}
params;

void main() {
	vec2 vertex_base;
	if (gl_VertexIndex == 0) {
		vertex_base = vec2(-1.0, -1.0);
	} else if (gl_VertexIndex == 1) {
		vertex_base = vec2(-1.0, 3.0);
	} else {
		vertex_base = vec2(3.0, -1.0);
	}
	gl_Position = vec4(vertex_base, 0.0, 1.0);
	tex_coord = clamp(vertex_base, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0

#ifdef RENDER_DRIVER_WEBGPU
	offset0 = fma(params.inv_size.xyxy, vec4(-1.0, 0.0, 0.0, -1.0), tex_coord.xyxy);
	offset1 = fma(params.inv_size.xyxy, vec4(1.0, 0.0, 0.0, 1.0), tex_coord.xyxy);
	offset2 = fma(params.inv_size.xyxy, vec4(-2.0, 0.0, 0.0, -2.0), tex_coord.xyxy);
#else
	offset[0] = fma(params.inv_size.xyxy, vec4(-1.0, 0.0, 0.0, -1.0), tex_coord.xyxy);
	offset[1] = fma(params.inv_size.xyxy, vec4(1.0, 0.0, 0.0, 1.0), tex_coord.xyxy);
	offset[2] = fma(params.inv_size.xyxy, vec4(-2.0, 0.0, 0.0, -2.0), tex_coord.xyxy);
#endif
}

#[fragment]
#version 450

layout(location = 0) in vec2 tex_coord;
#ifdef RENDER_DRIVER_WEBGPU
layout(location = 1) in vec4 offset0;
layout(location = 2) in vec4 offset1;
layout(location = 3) in vec4 offset2;
#else
layout(location = 1) in vec4 offset[3];
#endif

layout(set = 0, binding = 0) uniform sampler2D color_tex;

layout(location = 0) out vec2 edges;

layout(push_constant, std430) uniform Params {
	vec2 inv_size;
	float threshold;
	float reserved;
}
params;

#define SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR 2.0

void main() {
	vec2 threshold = vec2(params.threshold);

	vec4 delta;
	vec3 C = texture(color_tex, tex_coord).rgb;

#ifdef RENDER_DRIVER_WEBGPU
	vec3 Cleft = texture(color_tex, offset0.xy).rgb;
#else
	vec3 Cleft = texture(color_tex, offset[0].xy).rgb;
#endif
	vec3 t = abs(C - Cleft);
	delta.x = max(max(t.r, t.g), t.b);

#ifdef RENDER_DRIVER_WEBGPU
	vec3 Ctop = texture(color_tex, offset0.zw).rgb;
#else
	vec3 Ctop = texture(color_tex, offset[0].zw).rgb;
#endif
	t = abs(C - Ctop);
	delta.y = max(max(t.r, t.g), t.b);

	edges = step(threshold, delta.xy);

	if (dot(edges, vec2(1.0, 1.0)) == 0.0) {
		discard;
	}

#ifdef RENDER_DRIVER_WEBGPU
	vec3 Cright = texture(color_tex, offset1.xy).rgb;
#else
	vec3 Cright = texture(color_tex, offset[1].xy).rgb;
#endif
	t = abs(C - Cright);
	delta.z = max(max(t.r, t.g), t.b);

#ifdef RENDER_DRIVER_WEBGPU
	vec3 Cbottom = texture(color_tex, offset1.zw).rgb;
#else
	vec3 Cbottom = texture(color_tex, offset[1].zw).rgb;
#endif
	t = abs(C - Cbottom);
	delta.w = max(max(t.r, t.g), t.b);

	vec2 max_delta = max(delta.xy, delta.zw);

#ifdef RENDER_DRIVER_WEBGPU
	vec3 Cleftleft = texture(color_tex, offset2.xy).rgb;
#else
	vec3 Cleftleft = texture(color_tex, offset[2].xy).rgb;
#endif
	t = abs(Cleft - Cleftleft);
	delta.z = max(max(t.r, t.g), t.b);

#ifdef RENDER_DRIVER_WEBGPU
	vec3 Ctoptop = texture(color_tex, offset2.zw).rgb;
#else
	vec3 Ctoptop = texture(color_tex, offset[2].zw).rgb;
#endif
	t = abs(Ctop - Ctoptop);
	delta.w = max(max(t.r, t.g), t.b);

	max_delta = max(max_delta.xy, delta.zw);
	float final_delta = max(max_delta.x, max_delta.y);

	edges.xy *= step(final_delta, SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR * delta.xy);
}
