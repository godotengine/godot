
vec2 select2(vec2 a, vec2 b, bvec2 c)
{
	vec2 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;

	return ret;
}

vec3 select3(vec3 a, vec3 b, bvec3 c)
{
	vec3 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;
	ret.z = c.z ? b.z : a.z;

	return ret;
}

vec4 select4(vec4 a, vec4 b, bvec4 c)
{
	vec4 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;
	ret.z = c.z ? b.z : a.z;
	ret.w = c.w ? b.w : a.w;

	return ret;
}


highp vec4 texel2DFetch(highp sampler2D tex, ivec2 size, ivec2 coord)
{
	float x_coord = float(2 * coord.x + 1) / float(size.x * 2);
	float y_coord = float(2 * coord.y + 1) / float(size.y * 2);

	x_coord = float(coord.x) / float(size.x);
	y_coord = float(coord.y) / float(size.y);

	return texture2DLod(tex, vec2(x_coord, y_coord), 0.0);
}
