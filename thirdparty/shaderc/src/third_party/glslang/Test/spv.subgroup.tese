#version 450
#extension GL_KHR_shader_subgroup_basic: enable
layout(isolines) in;
layout(set = 0, binding = 0, std430) buffer Output
{
  uvec4 result[];
};

void main (void)
{
  result[gl_PrimitiveID] = uvec4(gl_SubgroupSize, gl_SubgroupInvocationID, 0, 0);
}
