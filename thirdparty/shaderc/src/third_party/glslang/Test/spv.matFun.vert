#version 400

uniform bl {
    uniform mat4 m4;
    uniform mat3 m3;
} bName;

in vec3 v3;

vec3 xf(mat3 m, vec3 v)
{
    return v * m;
}

mat3 Mat3(mat4 m)
{
	return mat3(m[0].xyz, m[1].xyz, m[2].xyz);
}

vec3 mxv(mat4 m4, vec3 v)
{
	return v * Mat3(m4);
}

void main()
{
    gl_Position = vec4(mxv(bName.m4, v3) + xf(bName.m3, v3), 1.0);
}
