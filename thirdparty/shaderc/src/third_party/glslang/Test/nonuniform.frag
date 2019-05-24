#version 450

int nonuniformEXT;

#extension GL_EXT_nonuniform_qualifier : enable

nonuniformEXT in vec4 nu_inv4;
nonuniformEXT float nu_gf;

nonuniformEXT out vec4 nu_outv4;           // ERROR, out
nonuniformEXT uniform vec4 nu_uv4;         // ERROR, uniform
nonuniformEXT const float nu_constf = 1.0; // ERROR, const

nonuniformEXT int foo(nonuniformEXT int nupi, nonuniformEXT out int f)
{
    return nupi;
}

void main()
{
    nonuniformEXT int nu_li;
    nonuniformEXT const int nu_ci = 2; // ERROR, const

    foo(nu_li, nu_li);

    int a;
    nu_li = nonuniformEXT(a) + nonuniformEXT(a * 2);
    nu_li = nonuniformEXT(a, a);       // ERROR, too many arguments
    nu_li = nonuniformEXT();           // ERROR, no arguments
}

layout(location=1) in struct S { float a; nonuniformEXT float b; } ins;  // ERROR, not on member
layout(location=3) in inbName { float a; nonuniformEXT float b; } inb;   // ERROR, not on member
