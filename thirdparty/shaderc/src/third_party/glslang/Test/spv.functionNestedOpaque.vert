#version 450

uniform struct S {
     sampler2D s;
} si;

void foo(sampler2D t)
{
    texture(t, vec2(0.5));
}

void barc(const S p)
{
    foo(p.s);
}

void bar(S p)
{
    foo(p.s);
}

void main()
{
    barc(si);
    bar(si);
}
