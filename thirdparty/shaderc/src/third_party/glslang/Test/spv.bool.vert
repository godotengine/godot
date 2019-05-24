#version 450

const bool condition = false;

uniform ubname {
    bool b;
} ubinst;

bool foo(bool b)
{
	return b != condition;
}

void main()
{
    gl_Position = foo(ubinst.b) ? vec4(0.0) : vec4(1.0);
}
