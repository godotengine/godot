#[compute]

#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "../shaders/includes/platform_compat.glsl"

#ifndef GS_DISPATCH_LOCAL_SIZE_X
#define GS_DISPATCH_LOCAL_SIZE_X 256
#endif

layout(set = 0, binding = 0, std430) buffer CounterBuffer {
    uint visible_splat_count;
    uint overflowed_splats;
} counters;

layout(set = 0, binding = 1, std430) buffer IndirectDispatch {
    uint dispatch_xyz[3];
    uint element_count;
    uint overflow_flag;
    uint unclamped_total;
} indirect_dispatch;

layout(set = 0, binding = 3, std430) buffer InstanceCount {
    uint dispatch_xyz[3];
    uint element_count;
    uint overflow_flag;
    uint unclamped_total;
} instance_count;

layout(set = 0, binding = 2, std140) uniform Params {
    mat4 view_matrix;
    uint visible_chunk_count;
    uint max_visible_splats;
    uint pad0;
    uint pad1;
} params;

void main() {
    uint raw_count = counters.visible_splat_count;
    uint max_visible = params.max_visible_splats;
    uint clamped = min(raw_count, max_visible);

    uint dispatch_x = (clamped + GS_DISPATCH_LOCAL_SIZE_X - 1u) / GS_DISPATCH_LOCAL_SIZE_X;
    indirect_dispatch.dispatch_xyz[0] = dispatch_x;
    indirect_dispatch.dispatch_xyz[1] = 1u;
    indirect_dispatch.dispatch_xyz[2] = 1u;
    indirect_dispatch.element_count = clamped;
    indirect_dispatch.unclamped_total = raw_count;
    indirect_dispatch.overflow_flag = (raw_count > max_visible || counters.overflowed_splats > 0u) ? 1u : 0u;

    instance_count.dispatch_xyz[0] = dispatch_x;
    instance_count.dispatch_xyz[1] = 1u;
    instance_count.dispatch_xyz[2] = 1u;
    instance_count.element_count = clamped;
    instance_count.unclamped_total = raw_count;
    instance_count.overflow_flag = (raw_count > max_visible || counters.overflowed_splats > 0u) ? 1u : 0u;
}
