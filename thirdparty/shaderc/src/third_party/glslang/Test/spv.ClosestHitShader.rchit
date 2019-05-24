#version 460
#extension GL_NV_ray_tracing : enable
layout(binding = 0, set = 0) uniform accelerationStructureNV accNV;
layout(location = 0) rayPayloadNV vec4 localPayload;
layout(location = 1) rayPayloadInNV vec4 incomingPayload;
void main()
{
	uvec3 v0 = gl_LaunchIDNV;
	uvec3 v1 = gl_LaunchSizeNV;
	int v2 = gl_PrimitiveID;
	int v3 = gl_InstanceID;
	int v4 = gl_InstanceCustomIndexNV;
	vec3 v5 = gl_WorldRayOriginNV;
	vec3 v6 = gl_WorldRayDirectionNV;
	vec3 v7 = gl_ObjectRayOriginNV;
	vec3 v8 = gl_ObjectRayDirectionNV;
	float v9 = gl_RayTminNV;
	float v10 = gl_RayTmaxNV;
	float v11 = gl_HitTNV;
	uint v12 = gl_HitKindNV;
	mat4x3 v13 = gl_ObjectToWorldNV;
	mat4x3 v14 = gl_WorldToObjectNV;
	traceNV(accNV, 0u, 1u, 2u, 3u, 0u, vec3(0.5f), 0.5f, vec3(1.0f), 0.75f, 1);
}
