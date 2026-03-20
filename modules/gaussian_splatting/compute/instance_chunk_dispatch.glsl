#[compute]

#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer CounterBuffer {
    uint visible_chunk_count;
    uint overflowed_chunks;
} counters;

layout(set = 0, binding = 1, std430) buffer DispatchBuffer {
    uint dispatch_xyz[3];
} dispatch;

layout(set = 0, binding = 2, std140) uniform Params {
    mat4 view_matrix;
    uint max_visible_chunks; // Uses InstanceDepthParamsGPU.visible_chunk_count slot.
    uint max_visible_splats;
    uint dispatch_group_x; // Uses InstanceDepthParamsGPU.pad0 slot.
    uint pad1;
} params;

// Build indirect dispatch counts for chunk-level processing.
void main() {
    uint raw_count = counters.visible_chunk_count;
    uint clamped = min(raw_count, params.max_visible_chunks);

    dispatch.dispatch_xyz[0] = params.dispatch_group_x;
    dispatch.dispatch_xyz[1] = clamped;
    dispatch.dispatch_xyz[2] = 1u;

    // Clear counters for Stage B splat counting.
    counters.visible_chunk_count = 0u;
    counters.overflowed_chunks = 0u;
}
