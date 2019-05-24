#version 450 core

#define PASTER2(type, suffix) type##suffix
#define PASTER3(type, suffix) type## suffix
#define MAKE_TYPE1 image1D dest ## 1;
#define MAKE_TYPE2(type, suffix) PASTER2(type, suffix)
#define MAKE_TYPE3(type, suffix) PASTER3(type, suffix)

#define PREFIX image
#define PREFIX3 imag
#define SUFFIX2 1D
#define SUFFIX3 e1 D

#define RESOURCE_TYPE1 MAKE_TYPE1
#define RESOURCE_TYPE2 MAKE_TYPE2(PREFIX, SUFFIX2)
#define RESOURCE_TYPE3 MAKE_TYPE3(PREFIX3, SUFFIX3)

layout (set = 0, binding = 0) uniform writeonly RESOURCE_TYPE1
layout (set = 0, binding = 0) uniform writeonly RESOURCE_TYPE2 dest2;
layout (set = 0, binding = 0) uniform writeonly RESOURCE_TYPE3 dest3;

void main()
{
}