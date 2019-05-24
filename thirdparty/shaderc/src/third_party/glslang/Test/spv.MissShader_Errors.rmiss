#version 460
#extension GL_NV_ray_tracing : enable
hitAttributeNV vec4 payload;                               // ERROR, hitattributeNV unsupported in this stage 
void main()
{
    int e0 = gl_PrimitiveID;                               // ERROR, unsupported builtin in stage
    int e1 = gl_InstanceID;                                // ERROR, unsupported builtin in stage
    int e3 = gl_InstanceCustomIndexNV;                     // ERROR, unsupported builtin in stage
    mat4x3 e10 = gl_ObjectToWorldNV;                       // ERROR, unsupported builtin in stage
    mat4x3 e11 = gl_WorldToObjectNV;                       // ERROR, unsupported builtin in stage
    float e12 = gl_HitTNV;                                 // ERROR, unsupported builtin in stage
    float e13 = gl_HitKindNV;                              // ERROR, unsupported builtin in stage
    reportIntersectionNV(1.0, 1U);                         // ERROR, unsupported builtin in stage
    ignoreIntersectionNV();                                // ERROR, unsupported builtin in stage
    terminateRayNV();                                      // ERROR, unsupported builtin in stage
}
