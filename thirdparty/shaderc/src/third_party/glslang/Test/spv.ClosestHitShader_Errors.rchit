#version 460
#extension GL_NV_ray_tracing : enable
hitAttributeNV vec4 payload;
layout(binding = 0, set = 0) uniform accelerationStructureNV accNV;

void main()
{
    payload.x = 1.0f;                                       // ERROR, cannot write to hitattributeNV in stage
    reportIntersectionNV(1.0, 1U);                          // ERROR, unsupported builtin in stage 
    terminateRayNV();
    ignoreIntersectionNV();
}
