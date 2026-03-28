#include "gaussian_splat_merge_utils.h"

#include "gs_project_settings.h"
#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/config/project_settings.h"
#include "core/templates/hash_set.h"
#include "core/variant/typed_array.h"
#include <cstring>
#include "../logger/gs_logger.h"

namespace {
// Project settings helpers provided by gs_project_settings.h (gs::settings namespace).

static Quaternion _safe_quaternion_from_asset(const TypedArray<Quaternion> &p_rotations, int p_index) {
    if (p_index >= 0 && p_index < p_rotations.size()) {
        Variant value = p_rotations[p_index];
        if (value.get_type() == Variant::QUATERNION) {
            return value;
        }
    }
    return Quaternion();
}

static GaussianDCEncoding _resolve_asset_dc_encoding(const Ref<GaussianSplatAsset> &p_asset) {
    if (p_asset.is_null()) {
        return GAUSSIAN_DC_ENCODING_LEGACY_BIAS;
    }

    const Dictionary import_metadata = p_asset->get_import_metadata();
    if (!import_metadata.has(StringName("dc_encoding"))) {
        return GAUSSIAN_DC_ENCODING_LEGACY_BIAS;
    }

    const String dc_encoding = String(import_metadata[StringName("dc_encoding")]).to_lower();
    if (dc_encoding == "linear_rgb") {
        return GAUSSIAN_DC_ENCODING_LINEAR_RGB;
    }
    return GAUSSIAN_DC_ENCODING_LEGACY_BIAS;
}

static void _rebuild_chunk_cache(const Ref<GaussianData> &p_data, float p_chunk_size,
        Vector<GaussianSplatRenderer::StaticChunk> &r_chunks, AABB &r_bounds) {
    r_chunks.clear();
    r_bounds = AABB();

    if (p_data.is_null()) {
        return;
    }

    const int total = p_data->get_count();
    if (total == 0) {
        return;
    }

    p_data->build_octree();
    AABB bounds = p_data->get_aabb();
    r_bounds = bounds;

    if (bounds.size == Vector3()) {
        GaussianSplatRenderer::StaticChunk chunk;
        chunk.bounds = bounds;
        chunk.center = bounds.get_center();
        chunk.radius = 0.0f;
        for (int i = 0; i < total; i++) {
            chunk.indices.push_back(i);
        }
        r_chunks.push_back(chunk);
        return;
    }

    const float cell = MAX(0.1f, p_chunk_size);
    const Vector3 min_corner = bounds.position;
    const Vector3 max_corner = bounds.position + bounds.size;

    const int x_cells = MAX(1, (int)Math::ceil(bounds.size.x / cell));
    const int y_cells = MAX(1, (int)Math::ceil(bounds.size.y / cell));
    const int z_cells = MAX(1, (int)Math::ceil(bounds.size.z / cell));

    const int total_potential_chunks = x_cells * y_cells * z_cells;
    const bool log_enabled = gs::settings::is_data_log_enabled();
    if (log_enabled) {
        GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[SpatialChunking] Scene bounds: %s (size: %s)", bounds, bounds.size));
        GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[SpatialChunking] Cell size: %.1f, Grid: %dx%dx%d = %d potential chunks",
                cell, x_cells, y_cells, z_cells, total_potential_chunks));
    }

    HashSet<uint32_t> assigned;
    assigned.reserve(total);

    for (int x = 0; x < x_cells; x++) {
        float start_x = min_corner.x + x * cell;
        float end_x = (x == x_cells - 1) ? max_corner.x : start_x + cell;
        float size_x = MAX(0.001f, end_x - start_x);
        for (int y = 0; y < y_cells; y++) {
            float start_y = min_corner.y + y * cell;
            float end_y = (y == y_cells - 1) ? max_corner.y : start_y + cell;
            float size_y = MAX(0.001f, end_y - start_y);
            for (int z = 0; z < z_cells; z++) {
                float start_z = min_corner.z + z * cell;
                float end_z = (z == z_cells - 1) ? max_corner.z : start_z + cell;
                float size_z = MAX(0.001f, end_z - start_z);

                AABB cell_bounds(Vector3(start_x, start_y, start_z), Vector3(size_x, size_y, size_z));
                TypedArray<int> indices = p_data->query_octree(cell_bounds);
                if (indices.is_empty()) {
                    continue;
                }

                GaussianSplatRenderer::StaticChunk chunk;
                chunk.bounds = cell_bounds;
                chunk.center = cell_bounds.get_center();
                chunk.radius = cell_bounds.get_longest_axis_size() * 0.5f;

                for (int i = 0; i < indices.size(); i++) {
                    int64_t value = (int64_t)indices[i];
                    if (value < 0) {
                        continue;
                    }
                    uint32_t idx = (uint32_t)value;
                    if (idx >= (uint32_t)total) {
                        continue;
                    }
                    if (assigned.has(idx)) {
                        continue;
                    }
                    chunk.indices.push_back(idx);
                    assigned.insert(idx);
                }

                if (!chunk.indices.is_empty()) {
                    r_chunks.push_back(chunk);
                }
            }
        }
    }

    if (assigned.size() < (uint32_t)total) {
        GaussianSplatRenderer::StaticChunk catch_all;
        catch_all.bounds = bounds;
        catch_all.center = bounds.get_center();
        catch_all.radius = bounds.get_longest_axis_size() * 0.5f;
        for (int i = 0; i < total; i++) {
            uint32_t idx = (uint32_t)i;
            if (!assigned.has(idx)) {
                catch_all.indices.push_back(idx);
            }
        }
        if (!catch_all.indices.is_empty()) {
            r_chunks.push_back(catch_all);
        }
    }

    if (r_chunks.is_empty()) {
        GaussianSplatRenderer::StaticChunk fallback;
        fallback.bounds = bounds;
        fallback.center = bounds.get_center();
        fallback.radius = bounds.get_longest_axis_size() * 0.5f;
        for (int i = 0; i < total; i++) {
            fallback.indices.push_back(i);
        }
        r_chunks.push_back(fallback);
    }

    if (log_enabled) {
        GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[SpatialChunking] Created %d actual chunks (from %d splats)", r_chunks.size(), total));
    }
}
} // namespace

bool gaussian_splat_merge_sources(const Vector<GaussianSplatMergeSource> &sources,
        float chunk_size, GaussianSplatMergeResult &out) {
    out.data.unref();
    out.chunks.clear();
    out.bounds = AABB();

    if (sources.is_empty()) {
        return false;
    }

    uint32_t total_splats = 0;
    for (int i = 0; i < sources.size(); i++) {
        const GaussianSplatMergeSource &source = sources[i];
        if (source.asset.is_null()) {
            continue;
        }
        total_splats += source.asset->get_splat_count();
    }

    if (total_splats == 0) {
        return false;
    }

    Ref<GaussianData> merged_data;
    merged_data.instantiate();
    merged_data->resize(total_splats);

    PackedVector3Array positions;
    PackedVector3Array scales;
    PackedVector3Array normals;
    PackedVector2Array brush_axes;
    PackedFloat32Array opacities;
    PackedFloat32Array stroke_ages;
    PackedInt32Array palette_ids;
    PackedInt32Array brush_override_ids;
    TypedArray<Quaternion> rotations;

    positions.resize(total_splats);
    scales.resize(total_splats);
    normals.resize(total_splats);
    brush_axes.resize(total_splats);
    opacities.resize(total_splats);
    stroke_ages.resize(total_splats);
    palette_ids.resize(total_splats);
    brush_override_ids.resize(total_splats);
    rotations.resize(total_splats);

    Vector3 *position_ptr = positions.ptrw();
    Vector3 *scale_ptr = scales.ptrw();
    Vector3 *normal_ptr = normals.ptrw();
    Vector2 *brush_ptr = brush_axes.ptrw();
    float *opacity_ptr = opacities.ptrw();
    float *stroke_ptr = stroke_ages.ptrw();
    int32_t *palette_ptr = palette_ids.ptrw();
    int32_t *brush_override_ptr = brush_override_ids.ptrw();

    int sh_terms_per_gaussian = -1;
    bool any_2d_mode = false;
    bool bounds_initialized = false;
    AABB merged_bounds;

    Vector<PackedFloat32Array> sh_buffers;
    Vector<uint32_t> sh_offsets;
    sh_buffers.resize(sources.size());
    sh_offsets.resize(sources.size());

    uint32_t write_offset = 0;
    for (int source_index = 0; source_index < sources.size(); source_index++) {
        const GaussianSplatMergeSource &source = sources[source_index];
        Ref<GaussianSplatAsset> asset = source.asset;
        if (asset.is_null()) {
            continue;
        }
        const uint32_t splat_count = asset->get_splat_count();
        if (splat_count == 0) {
            continue;
        }

        PackedVector3Array asset_positions = asset->get_position_vectors();
        PackedVector3Array asset_scales = asset->get_scale_vectors();
        TypedArray<Quaternion> asset_rotations = asset->get_rotation_quaternions();
        PackedVector3Array asset_normals = asset->get_normal_vectors();
        PackedVector2Array asset_brush_axes = asset->get_brush_axes_vector2();
        PackedFloat32Array asset_opacities = asset->get_opacities();
        PackedFloat32Array asset_stroke_ages = asset->get_stroke_ages_buffer();
        PackedInt32Array asset_palette_ids = asset->get_palette_ids_buffer();
        PackedInt32Array asset_brush_override_ids = asset->get_brush_override_ids_buffer();
        PackedFloat32Array asset_sh = asset->get_spherical_harmonics_buffer();

        sh_buffers.set(source_index, asset_sh);
        sh_offsets.set(source_index, write_offset);
        if (!asset_sh.is_empty() && splat_count > 0) {
            int local_terms = asset_sh.size() / splat_count;
            if (local_terms % 3 != 0) {
                local_terms -= local_terms % 3;
            }
            if (local_terms <= 0) {
                local_terms = 3;
            }
            if (sh_terms_per_gaussian == -1) {
                sh_terms_per_gaussian = local_terms;
            } else if (local_terms != sh_terms_per_gaussian) {
                GS_LOG_WARN_DEFAULT("GaussianSplat merge: mismatched SH term counts across sources, clamping to minimum.");
                sh_terms_per_gaussian = MIN(sh_terms_per_gaussian, local_terms);
            }
        }

        const Transform3D &transform = source.transform;
        Basis basis = transform.basis;
        Basis rotation_basis = basis.orthonormalized();
        Quaternion world_rotation = rotation_basis.get_rotation_quaternion();
        Vector3 world_scale = basis.get_scale_abs();
        const GaussianDCEncoding source_dc_encoding = _resolve_asset_dc_encoding(asset);

        for (uint32_t splat = 0; splat < splat_count; splat++) {
            const uint32_t target_index = write_offset + splat;

            Vector3 local_position = splat < (uint32_t)asset_positions.size() ? asset_positions[splat] : Vector3();
            Vector3 local_scale = splat < (uint32_t)asset_scales.size() ? asset_scales[splat] : Vector3(1, 1, 1);
            Vector3 world_position = transform.xform(local_position);
            Vector3 world_splat_scale = Vector3(world_scale.x * local_scale.x, world_scale.y * local_scale.y, world_scale.z * local_scale.z);

            position_ptr[target_index] = world_position;
            scale_ptr[target_index] = world_splat_scale;

            Quaternion local_rotation = _safe_quaternion_from_asset(asset_rotations, splat);
            rotations.set(target_index, world_rotation * local_rotation);

            Vector3 local_normal = splat < (uint32_t)asset_normals.size() ? asset_normals[splat] : Vector3(0, 1, 0);
            Vector3 world_normal = rotation_basis.xform(local_normal).normalized();
            normal_ptr[target_index] = world_normal;

            Vector2 brush_axis = splat < (uint32_t)asset_brush_axes.size() ? asset_brush_axes[splat] : Vector2(1, 0);
            brush_ptr[target_index] = brush_axis;

            float opacity = splat < (uint32_t)asset_opacities.size() ? asset_opacities[splat] : 1.0f;
            opacity_ptr[target_index] = opacity;

            float stroke = splat < (uint32_t)asset_stroke_ages.size() ? asset_stroke_ages[splat] : 0.0f;
            stroke_ptr[target_index] = stroke;

            int32_t palette = splat < (uint32_t)asset_palette_ids.size() ? asset_palette_ids[splat] : 0;
            palette_ptr[target_index] = palette;

            int32_t brush_override_id = splat < (uint32_t)asset_brush_override_ids.size() ? asset_brush_override_ids[splat] : 0;
            brush_override_ptr[target_index] = brush_override_id;

            float max_extent = MAX(MAX(world_splat_scale.x, world_splat_scale.y), world_splat_scale.z) * 3.0f;
            AABB splat_bounds(world_position - Vector3(max_extent, max_extent, max_extent),
                    Vector3(max_extent, max_extent, max_extent) * 2.0f);
            if (!bounds_initialized) {
                merged_bounds = splat_bounds;
                bounds_initialized = true;
            } else {
                merged_bounds = merged_bounds.merge(splat_bounds);
            }

            Gaussian g = merged_data->get_gaussian((int)target_index);
            g.render_meta = gaussian_set_dc_encoding(g.render_meta, source_dc_encoding);
            merged_data->set_gaussian((int)target_index, g);
        }

        if (source.is_2d) {
            any_2d_mode = true;
        }

        write_offset += splat_count;
    }

    if (sh_terms_per_gaussian <= 0) {
        sh_terms_per_gaussian = 3;
    }

    PackedFloat32Array sh_combined;
    sh_combined.resize(total_splats * sh_terms_per_gaussian);
    float *sh_ptr = sh_combined.ptrw();
    for (int source_index = 0; source_index < sources.size(); source_index++) {
        const GaussianSplatMergeSource &source = sources[source_index];
        Ref<GaussianSplatAsset> asset = source.asset;
        if (asset.is_null()) {
            continue;
        }
        uint32_t splat_count = asset->get_splat_count();
        const PackedFloat32Array &asset_sh = sh_buffers[source_index];
        const float *asset_sh_ptr = asset_sh.ptr();
        uint32_t base_offset = sh_offsets[source_index];
        uint32_t source_stride = 0u;
        if (splat_count > 0) {
            source_stride = uint32_t(asset_sh.size() / int(splat_count));
            if (source_stride % 3u != 0u) {
                source_stride -= source_stride % 3u;
            }
            if (source_stride == 0u) {
                source_stride = 3u;
            }
        }
        const uint32_t target_stride = uint32_t(sh_terms_per_gaussian);
        const uint32_t copy_terms = MIN(source_stride, target_stride);

        const uint32_t asset_sh_size = uint32_t(asset_sh.size());
        if (asset_sh_ptr != nullptr && source_stride == target_stride && splat_count > 0u &&
                uint64_t(asset_sh_size) >= uint64_t(splat_count) * uint64_t(source_stride)) {
            const size_t copy_bytes = size_t(splat_count) * size_t(source_stride) * sizeof(float);
            std::memcpy(sh_ptr + size_t(base_offset) * size_t(target_stride), asset_sh_ptr, copy_bytes);
            continue;
        }

        for (uint32_t splat = 0; splat < splat_count; splat++) {
            const uint32_t target_base = (base_offset + splat) * target_stride;
            const uint32_t source_base = splat * source_stride;
            const uint32_t source_available_terms = source_base < asset_sh_size ? asset_sh_size - source_base : 0u;
            const uint32_t available_terms = MIN(copy_terms, source_available_terms);
            if (asset_sh_ptr != nullptr && available_terms > 0u) {
                std::memcpy(sh_ptr + target_base, asset_sh_ptr + source_base, size_t(available_terms) * sizeof(float));
            }
            for (uint32_t term = available_terms; term < target_stride; term++) {
                sh_ptr[target_base + term] = 0.0f;
            }
        }
    }

    merged_data->set_positions(positions);
    merged_data->set_scales(scales);
    merged_data->set_rotations(rotations);
    merged_data->set_normals(normals);
    merged_data->set_brush_axes(brush_axes);
    merged_data->set_opacities(opacities);
    merged_data->set_stroke_ages(stroke_ages);
    merged_data->set_palette_ids(palette_ids);
    merged_data->set_brush_override_ids(brush_override_ids);
    merged_data->set_spherical_harmonics(sh_combined);
    merged_data->set_2d_mode(any_2d_mode);

    merged_data->build_octree();
    _rebuild_chunk_cache(merged_data, chunk_size, out.chunks, out.bounds);
    out.data = merged_data;

    return out.data.is_valid() && out.data->get_count() > 0;
}
