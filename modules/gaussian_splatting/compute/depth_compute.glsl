#[compute]

#version 450

#VERSION_DEFINES

#include "../shaders/includes/gs_instance_layout.glsl"
#include "../shaders/includes/gs_sort_key.glsl"
#include "../shaders/includes/gs_deformation.glsl"
#include "../shaders/includes/quantization_dequant.glsl"

#if defined(USE_FLOAT16_GAUSSIANS)
// Float16 storage relies on per-chunk center data (QuantizationChunk) which is not
// bound in the instance pipeline yet (binding 1 is ChunkQuantizationGPU).
#error "USE_FLOAT16_GAUSSIANS is not supported for the instance pipeline."
#endif

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

#if defined(USE_QUANTIZED_GAUSSIANS)
    #define GS_GAUSSIAN_STRUCT GaussianQuantized
#else
struct PackedGaussian {
    vec3 position;
    float opacity;
    vec3 scale;
    float area;
    vec4 rotation;
    vec4 sh_dc;
    float sh_encoded[12];
    vec3 normal;
    float stroke_age;
    vec2 brush_axes;
    uint painterly_meta;
    uint sh_metadata;
};
    #define GS_GAUSSIAN_STRUCT PackedGaussian
#endif

layout(set = 0, binding = 0, std430) readonly buffer AtlasGaussianBuffer {
    GS_GAUSSIAN_STRUCT gaussians[];
} atlas_gaussian_buffer;

#if defined(USE_QUANTIZED_GAUSSIANS)
layout(set = 0, binding = 1, std430) readonly buffer QuantizationChunkBuffer {
    ChunkQuantization chunks[];
} quantization_buffer;
#endif

layout(set = 0, binding = 2, std430) readonly buffer InstanceBuffer {
    InstanceDataGPU instances[];
} instance_buffer;

layout(set = 0, binding = 3, std430) readonly buffer ChunkMetaBuffer {
    ChunkMetaGPU chunks[];
} chunk_meta_buffer;

layout(set = 0, binding = 4, std430) readonly buffer VisibleChunkBuffer {
    VisibleChunkRefGPU visible_chunks[];
} visible_chunk_buffer;

layout(set = 0, binding = 5, std430) writeonly buffer SplatRefBuffer {
    SplatRefGPU splat_refs[];
} splat_ref_buffer;

layout(set = 0, binding = 6, std430) writeonly buffer SortKeyBuffer {
    uvec2 keys[];
} sort_key_buffer;

layout(set = 0, binding = 7, std430) writeonly buffer SortValueBuffer {
    uint values[];
} sort_value_buffer;

layout(set = 0, binding = 8, std430) buffer CounterBuffer {
    uint visible_splat_count;
    uint overflowed_splats;
} counters;

layout(set = 0, binding = 9, std140) uniform Params {
    mat4 view_matrix;
    uint visible_chunk_count;
    uint max_visible_splats;
    uint pad0;
    uint pad1;
    vec4 wind_dir_strength;
    vec4 wind_time_config;
    vec4 effector_sphere;
    vec4 effector_config;
    vec4 frustum_planes[6];
    vec4 camera_position_ortho; // xyz = camera position, w = orthographic flag
    vec4 cull_screen_distance; // x = pixel_scale_y, y = tiny_splat_radius_px, z = min_screen_threshold_px, w = max_distance_sq
    vec4 cull_frustum_radius; // x = radius_multiplier, y = frustum_plane_slack, z = enable_frustum, w = reserved
} params;

vec3 quat_rotate(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

bool gs_sphere_frustum_visible(vec3 position, float radius) {
    if (params.cull_frustum_radius.z < 0.5) {
        return true;
    }
    float effective_radius = radius * max(params.cull_frustum_radius.y, 1.0);
    for (int i = 0; i < 6; i++) {
        vec4 plane = params.frustum_planes[i];
        float distance = dot(plane.xyz, position) - plane.w;
        if (distance > effective_radius) {
            return false;
        }
    }
    return true;
}

float gs_compute_screen_size(float depth, float radius) {
    if (params.camera_position_ortho.w > 0.5) {
        return (radius * params.cull_screen_distance.x) * 2.0;
    }
    if (depth + radius <= 0.0) {
        return 0.0;
    }
    float screen_depth = max(depth, 0.0001);
    float screen_radius = (radius * params.cull_screen_distance.x) / screen_depth;
    return screen_radius * 2.0;
}

void main() {
    uint splat_index_in_chunk = gl_GlobalInvocationID.x;
    uint visible_chunk_index = gl_GlobalInvocationID.y;

    if (visible_chunk_index >= params.visible_chunk_count) {
        return;
    }

    VisibleChunkRefGPU visible_chunk = visible_chunk_buffer.visible_chunks[visible_chunk_index];
    ChunkMetaGPU chunk = chunk_meta_buffer.chunks[visible_chunk.chunk_id];
    if (splat_index_in_chunk >= chunk.splat_count) {
        return;
    }

    uint atlas_index = chunk.atlas_base + splat_index_in_chunk;
    GS_GAUSSIAN_STRUCT gaussian = atlas_gaussian_buffer.gaussians[atlas_index];

#if defined(USE_QUANTIZED_GAUSSIANS)
    // ChunkMetaGPU.quant_base is the authoritative absolute index into QuantizationChunkBuffer.
    uint quant_id = chunk.quant_base;
    ChunkQuantization quant = quantization_buffer.chunks[quant_id];
    vec3 local_position = LOAD_POSITION_QUANTIZED(gaussian, quant);
    vec3 local_scale = LOAD_SCALE_QUANTIZED(gaussian, quant);
#else
    vec3 local_position = gaussian.position;
    vec3 local_scale = gaussian.scale;
#endif

    InstanceDataGPU instance = instance_buffer.instances[visible_chunk.instance_id];
    uint instance_flags = instance.ids.y;
    float uniform_scale = abs(instance.translation_scale.w);
    vec3 scaled_position = (instance_flags & GS_INSTANCE_FLAG_SCALE_IDENTITY) != 0u
            ? local_position
            : local_position * uniform_scale;
    vec3 world_position = (instance_flags & GS_INSTANCE_FLAG_ROTATION_IDENTITY) != 0u
            ? scaled_position
            : quat_rotate(instance.rotation, scaled_position);
    if ((instance_flags & GS_INSTANCE_FLAG_TRANSLATION_ZERO) == 0u) {
        world_position += instance.translation_scale.xyz;
    }
    float instance_intensity = max(instance.params.z, 0.0);
    float instance_wind_mode = instance.params.w;
    uint stable_seed = atlas_index ^ (visible_chunk.instance_id * 0x9e3779b9u);
    world_position = gs_apply_wind_deformation(world_position,
            stable_seed,
            gaussian.opacity,
            instance_intensity,
            instance_wind_mode,
            instance.wind_params,
            params.wind_dir_strength,
            params.wind_time_config,
            params.effector_sphere,
            params.effector_config);

    float local_radius = max(max(abs(local_scale.x), abs(local_scale.y)), abs(local_scale.z));
    if (local_radius <= 0.0) {
        local_radius = 1.0;
    }
    float radius_multiplier = params.cull_frustum_radius.x > 0.0 ? params.cull_frustum_radius.x : 1.0;
    float splat_radius = local_radius * radius_multiplier;
    if ((instance_flags & GS_INSTANCE_FLAG_SCALE_IDENTITY) == 0u) {
        splat_radius *= uniform_scale;
    }

    if (!gs_sphere_frustum_visible(world_position, splat_radius)) {
        return;
    }

    vec4 view_pos = params.view_matrix * vec4(world_position, 1.0);
    float depth = -view_pos.z;

    float screen_size = gs_compute_screen_size(depth, splat_radius);
    float tiny_radius = params.cull_screen_distance.y;
    if (tiny_radius > 0.0 && screen_size < tiny_radius * 2.0) {
        return;
    }
    float min_screen_threshold = params.cull_screen_distance.z;
    if (min_screen_threshold > 0.0 && screen_size < min_screen_threshold) {
        return;
    }
    float max_distance_sq = params.cull_screen_distance.w;
    if (max_distance_sq > 0.0) {
        vec3 to_camera = world_position - params.camera_position_ortho.xyz;
        float distance_sq = dot(to_camera, to_camera);
        if (distance_sq > max_distance_sq) {
            return;
        }
    }

    // Allocate a write slot via atomicAdd (single-instruction, no retry loop).
    // The count_clamp shader that runs after this pass clamps the final count.
    uint write_index = atomicAdd(counters.visible_splat_count, 1u);
    if (write_index >= params.max_visible_splats) {
        atomicAdd(counters.overflowed_splats, 1u);
        return;
    }

    splat_ref_buffer.splat_refs[write_index].instance_id = visible_chunk.instance_id;
    splat_ref_buffer.splat_refs[write_index].atlas_index = atlas_index;
    // Favor atlas index for tie-break to improve cache locality when depths are similar.
    uint tie_break = atlas_index;
    sort_key_buffer.keys[write_index] = gs_pack_sort_key64(depth, tie_break);
    sort_value_buffer.values[write_index] = write_index;
}
