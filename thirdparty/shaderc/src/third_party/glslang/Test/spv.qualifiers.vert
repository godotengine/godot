#version 430 core

in vec4 inV;

centroid out vec4 outVc;
smooth out vec4 outVs;
flat out vec4 outVf;
noperspective out vec4 outVn;

centroid noperspective out vec4 outVcn;

void main()
{
    outVc = inV;
    outVs = inV;
    outVf = inV;
    outVn = inV;
    outVcn = inV;
}
