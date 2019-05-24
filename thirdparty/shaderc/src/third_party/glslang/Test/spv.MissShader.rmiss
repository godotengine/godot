#version 460
#extension GL_NV_ray_tracing : enable
layout(binding = 0, set = 0) uniform accelerationStructureNV accNV;
layout(location = 0) rayPayloadNV vec4 localPayload;
layout(location = 1) rayPayloadInNV vec4 incomingPayload;
void main()
{
	uvec3 v0 = gl_LaunchIDNV;
	uvec3 v1 = gl_LaunchSizeNV;
	vec3 v2 = gl_WorldRayOriginNV;
	vec3 v3 = gl_WorldRayDirectionNV;
	vec3 v4 = gl_ObjectRayOriginNV;
	vec3 v5 = gl_ObjectRayDirectionNV;
	float v6 = gl_RayTminNV;
	float v7 = gl_RayTmaxNV;
	traceNV(accNV, 0u, 1u, 2u, 3u, 0u, vec3(0.5f), 0.5f, vec3(1.0f), 0.75f, 1);
}
