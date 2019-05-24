#version 420

layout(local_size_x = 2) in;  // ERROR, no compute

#extension GL_ARB_compute_shader : enable

layout(local_size_x = 2, local_size_y = 4, local_size_z = 6) in;

shared vec3 sfoo;

void main()
{
    sfoo = vec3(gl_WorkGroupSize.x, gl_WorkGroupSize.y, gl_WorkGroupSize.z);
    sfoo += gl_WorkGroupSize + gl_NumWorkGroups + gl_WorkGroupID + gl_LocalInvocationID + gl_GlobalInvocationID;
    sfoo *= gl_LocalInvocationIndex;
    sfoo += gl_MaxComputeWorkGroupCount + gl_MaxComputeWorkGroupSize;
    sfoo *= gl_MaxComputeUniformComponents +
            gl_MaxComputeTextureImageUnits +
            gl_MaxComputeImageUniforms +
            gl_MaxComputeAtomicCounters +
            gl_MaxComputeAtomicCounterBuffers;

    barrier();
    memoryBarrier();
    memoryBarrierAtomicCounter();
    memoryBarrierBuffer();
    memoryBarrierImage();
    memoryBarrierShared();
    groupMemoryBarrier();
}