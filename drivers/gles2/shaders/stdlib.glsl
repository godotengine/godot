
vec2 select2(vec2 a, vec2 b, bvec2 c) {
	vec2 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;

	return ret;
}

vec3 select3(vec3 a, vec3 b, bvec3 c) {
	vec3 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;
	ret.z = c.z ? b.z : a.z;

	return ret;
}

vec4 select4(vec4 a, vec4 b, bvec4 c) {
	vec4 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;
	ret.z = c.z ? b.z : a.z;
	ret.w = c.w ? b.w : a.w;

	return ret;
}

highp vec4 texel2DFetch(highp sampler2D tex, ivec2 size, ivec2 coord) {
	float x_coord = float(2 * coord.x + 1) / float(size.x * 2);
	float y_coord = float(2 * coord.y + 1) / float(size.y * 2);

	return texture2DLod(tex, vec2(x_coord, y_coord), 0.0);
}

#ifndef USE_GLES_OVER_GL
highp mat4 transpose(highp mat4 src) {
	return mat4(
			vec4(src[0].x, src[1].x, src[2].x, src[3].x),
			vec4(src[0].y, src[1].y, src[2].y, src[3].y),
			vec4(src[0].z, src[1].z, src[2].z, src[3].z),
			vec4(src[0].w, src[1].w, src[2].w, src[3].w));
}
#endif
