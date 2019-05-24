#version 450

#extension GL_EXT_nonuniform_qualifier : enable

layout(location=0) nonuniformEXT in vec4 nu_inv4;
nonuniformEXT float nu_gf;
layout(location=1) in nonuniformEXT flat int nu_ii;

layout(binding=0, input_attachment_index = 0) uniform subpassInput        inputAttachmentDyn[];
layout(binding=1)                             uniform samplerBuffer       uniformTexelBufferDyn[];
layout(binding=2, r32f)                       uniform imageBuffer         storageTexelBufferDyn[];
layout(binding=3)                             uniform uname { float a; }  uniformBuffer[];
layout(binding=4)                             buffer  bname { float b; }  storageBuffer[];
layout(binding=5)                             uniform sampler2D           sampledImage[];
layout(binding=6, r32f)                       uniform image2D             storageImage[];
layout(binding=7, input_attachment_index = 1) uniform subpassInput        inputAttachment[];
layout(binding=8)                             uniform samplerBuffer       uniformTexelBuffer[];
layout(binding=9, r32f)                       uniform imageBuffer         storageTexelBuffer[];

nonuniformEXT int foo(nonuniformEXT int nupi, nonuniformEXT out int f)
{
    return nupi;
}

void main()
{
    nonuniformEXT int nu_li;
    int dyn_i;

    int a = foo(nu_li, nu_li);
    nu_li = nonuniformEXT(a) + nonuniformEXT(a * 2);

    float b;
    b = nu_inv4.x * nu_gf;
    b += subpassLoad(inputAttachmentDyn[dyn_i]).x;
    b += texelFetch(uniformTexelBufferDyn[dyn_i], 1).x;
    b += imageLoad(storageTexelBufferDyn[dyn_i], 1).x;
    b += uniformBuffer[nu_ii].a;
    b += storageBuffer[nu_ii].b;
    b += texture(sampledImage[nu_ii], vec2(0.5)).x;
    b += imageLoad(storageImage[nu_ii], ivec2(1)).x;
    b += subpassLoad(inputAttachment[nu_ii]).x;
    b += texelFetch(uniformTexelBuffer[nu_ii], 1).x;
    b += imageLoad(storageTexelBuffer[nu_ii], 1).x;

    nonuniformEXT ivec4 v;
    nonuniformEXT mat4 m;
    nonuniformEXT struct S { int a; } s;
    ivec4 uv;
    b += uniformBuffer[v.y].a;
    b += uniformBuffer[v[2]].a;
    b += uniformBuffer[uv[nu_ii]].a;
    b += uniformBuffer[int(m[2].z)].a;
    b += uniformBuffer[s.a].a;
}
