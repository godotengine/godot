#[compute]

#version 450

#VERSION_DEFINES

// Subgroup extensions are enabled via variant defines (GS_ENABLE_SUBGROUPS)
// The ShaderRD variant system injects the appropriate defines based on GPU capabilities.
#ifdef GS_ENABLE_SUBGROUPS
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#define GS_SUBGROUP_AVAILABLE 1
#else
#define GS_SUBGROUP_AVAILABLE 0
#endif

#include "../shaders/includes/platform_compat.glsl"
#include "../shaders/includes/gs_instance_layout.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer InstanceBuffer {
    InstanceDataGPU instances[];
} instance_buffer;

layout(set = 0, binding = 1, std430) readonly buffer AssetMetaBuffer {
    AssetMetaGPU assets[];
} asset_meta_buffer;

layout(set = 0, binding = 2, std430) readonly buffer AssetChunkIndexBuffer {
    uint chunk_ids[];
} asset_chunk_index_buffer;

layout(set = 0, binding = 3, std430) readonly buffer ChunkMetaBuffer {
    ChunkMetaGPU chunks[];
} chunk_meta_buffer;

layout(set = 0, binding = 4, std430) writeonly buffer VisibleChunkBuffer {
    VisibleChunkRefGPU visible_chunks[];
} visible_chunk_buffer;

layout(set = 0, binding = 5, std430) buffer CounterBuffer {
    uint visible_chunk_count;
    uint overflowed_chunks;
} counters;

layout(set = 0, binding = 6, std140) uniform Params {
    mat4 view_matrix;
    mat4 proj_matrix;
    vec4 frustum_planes[6];
    float frustum_plane_slack;
    uint instance_count;
    uint max_visible_chunks;
    uint enable_frustum;
} params;

vec3 quat_rotate(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

bool sphere_frustum_visible(vec3 position, float radius) {
    if (params.enable_frustum == 0u) {
        return true;
    }
    float effective_radius = radius * max(params.frustum_plane_slack, 1.0);
    for (int i = 0; i < 6; i++) {
        vec4 plane = params.frustum_planes[i];
        float distance = dot(plane.xyz, position) - plane.w;
        // Projection::get_projection_planes() returns planes where points inside the frustum
        // have negative distances (RendererSceneCull::InstanceBounds::in_frustum()). Cull only
        // when the sphere is fully outside on the positive side.
        if (distance > effective_radius) {
            return false;
        }
    }
    return true;
}

void main() {
    uint chunk_index_in_lod = gl_GlobalInvocationID.x;
    uint instance_id = gl_GlobalInvocationID.y;

    if (instance_id >= params.instance_count) {
        return;
    }

    InstanceDataGPU instance = instance_buffer.instances[instance_id];
    uint asset_id = instance.ids.x;
    AssetMetaGPU asset = asset_meta_buffer.assets[asset_id];

    if (asset.lod_count == 0u) {
        return;
    }
    uint lod_level = instance.lod.x;
    // Clamp requested LOD to the asset's available range.
    // This prevents instances from being culled when CPU LOD resolves > available.
    lod_level = min(lod_level, asset.lod_count - 1u);

    AssetLodRangeGPU lod_range = asset.lod_ranges[lod_level];
    if (chunk_index_in_lod >= lod_range.count) {
        return;
    }

    uint chunk_index = lod_range.base + chunk_index_in_lod;
    uint chunk_id = asset_chunk_index_buffer.chunk_ids[chunk_index];
    ChunkMetaGPU chunk = chunk_meta_buffer.chunks[chunk_id];
    if (chunk.splat_count == 0u) {
        return;
    }

    uint instance_flags = instance.ids.y;
    float uniform_scale = abs(instance.translation_scale.w);
    vec3 local_center = (instance_flags & GS_INSTANCE_FLAG_SCALE_IDENTITY) != 0u
            ? chunk.bounds_center_local
            : chunk.bounds_center_local * uniform_scale;
    vec3 world_center = (instance_flags & GS_INSTANCE_FLAG_ROTATION_IDENTITY) != 0u
            ? local_center
            : quat_rotate(instance.rotation, local_center);
    if ((instance_flags & GS_INSTANCE_FLAG_TRANSLATION_ZERO) == 0u) {
        world_center += instance.translation_scale.xyz;
    }
    float radius = (instance_flags & GS_INSTANCE_FLAG_SCALE_IDENTITY) != 0u
            ? chunk.bounds_radius_local
            : chunk.bounds_radius_local * uniform_scale;

    if (!sphere_frustum_visible(world_center, radius)) {
        return;
    }

    uint write_index = atomicAdd(counters.visible_chunk_count, 1u);
    if (write_index >= params.max_visible_chunks) {
        atomicAdd(counters.overflowed_chunks, 1u);
        return;
    }

    visible_chunk_buffer.visible_chunks[write_index].instance_id = instance_id;
    visible_chunk_buffer.visible_chunks[write_index].chunk_id = chunk_id;
}
