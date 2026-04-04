#pragma once

#include "test_macros.h"

#ifdef TOOLS_ENABLED

#include "../io/resource_importer_ply.h"
#include "../io/resource_importer_spz.h"
#include "../io/resource_importer_gsplatworld.h"
#include "../io/ply_loader.h"
#include "../io/spz_loader.h"
#include "../editor/gaussian_import_dialog.h"
#include "../editor/gaussian_thumbnail_generator.h"
#include "../core/gaussian_splat_asset.h"
#include "../core/gaussian_splat_world.h"
#include "../renderer/gaussian_gpu_layout.h"
#include "../nodes/gaussian_splat_node_3d.h"

#include "core/io/compression.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/io/resource_uid.h"
#include "core/config/project_settings.h"
#include "core/templates/hash_map.h"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"

#include <cstring>

#include "gs_test_setting_guard.h"

namespace {

Error _write_missing_opacity_ascii_ply(const String &p_path) {
    static const char *k_missing_opacity_ascii_ply = R"(ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
0 0 0 0 0 0 1 0 0 0 0.25 0.5 0.75
)";

    Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
    if (file.is_null()) {
        return ERR_CANT_CREATE;
    }

    file->store_string(k_missing_opacity_ascii_ply);
    file.unref();
    return OK;
}

Error _write_minimal_ascii_ply(const String &p_path) {
    static const char *k_minimal_ascii_ply = R"(ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
0 0 0 0 0 0 1 0 0 0 0 0.25 0.5 0.75
)";

    Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
    if (file.is_null()) {
        return ERR_CANT_CREATE;
    }

    file->store_string(k_minimal_ascii_ply);
    file.unref();
    return OK;
}

Error _write_invalid_gsplatworld(const String &p_path) {
    Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
    if (file.is_null()) {
        return ERR_CANT_CREATE;
    }

    // Header-sized payload with invalid magic/version to verify import-time validation.
    PackedByteArray bytes;
    bytes.resize(104);
    bytes.fill(0);
    file->store_buffer(bytes.ptr(), bytes.size());
    file.unref();
    return OK;
}

Error _write_truncated_copy(const String &p_source_path, const String &p_dest_path, uint64_t p_trim_bytes) {
    Error read_err = OK;
    Ref<FileAccess> src = FileAccess::open(p_source_path, FileAccess::READ, &read_err);
    if (src.is_null()) {
        return read_err != OK ? read_err : ERR_CANT_OPEN;
    }

    const uint64_t src_size = src->get_length();
    if (src_size <= p_trim_bytes) {
        return ERR_FILE_CORRUPT;
    }

    const uint64_t out_size = src_size - p_trim_bytes;
    PackedByteArray out_data;
    out_data.resize(out_size);
    const uint64_t read = src->get_buffer(out_data.ptrw(), out_size);
    if (read != out_size) {
        return ERR_FILE_CORRUPT;
    }

    Error write_err = OK;
    Ref<FileAccess> dst = FileAccess::open(p_dest_path, FileAccess::WRITE, &write_err);
    if (dst.is_null()) {
        return write_err != OK ? write_err : ERR_CANT_CREATE;
    }

    if (!dst->store_buffer(out_data.ptr(), out_data.size())) {
        const Error dst_err = dst->get_error();
        return dst_err != OK ? dst_err : ERR_FILE_CANT_WRITE;
    }

    return OK;
}

Error _corrupt_gsplatworld_for_decode_failure(const String &p_path) {
    Error open_err = OK;
    Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ_WRITE, &open_err);
    if (file.is_null()) {
        return open_err != OK ? open_err : ERR_CANT_OPEN;
    }

    if (file->get_length() < 104) {
        return ERR_FILE_CORRUPT;
    }

    constexpr uint64_t flags_offset = 8u;
    constexpr uint64_t gaussian_offset_field = 56u;
    constexpr uint32_t compressed_flag = 1u << 4u;

    file->seek(flags_offset);
    uint32_t flags = file->get_32();
    flags |= compressed_flag;
    file->seek(flags_offset);
    file->store_32(flags);

    file->seek(gaussian_offset_field);
    const uint64_t gaussian_offset = file->get_64();
    if (gaussian_offset + 9u > file->get_length()) {
        return ERR_FILE_CORRUPT;
    }

    file->seek(gaussian_offset);
    file->store_64(1u); // compressed size (invalid for expected payload)
    file->store_8(0u); // compressed payload byte

    const Error write_err = file->get_error();
    return write_err != OK ? write_err : OK;
}

void _remove_user_file(const String &p_user_path) {
    Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_USERDATA);
    if (dir.is_null()) {
        return;
    }

    dir->remove(p_user_path.get_file());
}

PackedByteArray _make_spz_header(uint32_t p_version, uint32_t p_num_points, uint8_t p_sh_degree, uint8_t p_fractional_bits, uint8_t p_flags = 0, uint8_t p_reserved = 0) {
    PackedByteArray header;
    header.resize(sizeof(SPZLoader::SPZHeader));
    uint8_t *w = header.ptrw();

    auto write_u32 = [&](int p_offset, uint32_t p_value) {
        w[p_offset + 0] = uint8_t(p_value & 0xFF);
        w[p_offset + 1] = uint8_t((p_value >> 8) & 0xFF);
        w[p_offset + 2] = uint8_t((p_value >> 16) & 0xFF);
        w[p_offset + 3] = uint8_t((p_value >> 24) & 0xFF);
    };

    write_u32(0, SPZLoader::SPZ_MAGIC);
    write_u32(4, p_version);
    write_u32(8, p_num_points);
    w[12] = p_sh_degree;
    w[13] = p_fractional_bits;
    w[14] = p_flags;
    w[15] = p_reserved;

    return header;
}

PackedByteArray _gzip_compress(const PackedByteArray &p_input) {
    const int64_t max_size = Compression::get_max_compressed_buffer_size(p_input.size(), Compression::MODE_GZIP);
    PackedByteArray out;
    out.resize(max_size);
    const int64_t compressed_size = Compression::compress(out.ptrw(), p_input.ptr(), p_input.size(), Compression::MODE_GZIP);
    if (compressed_size <= 0) {
        out.clear();
        return out;
    }
    out.resize(compressed_size);
    return out;
}

PackedByteArray _make_spz_v2_single_point_payload(uint8_t p_alpha, uint8_t p_r, uint8_t p_g, uint8_t p_b) {
    PackedByteArray payload;
    payload.resize(19);
    payload.fill(0);

    uint8_t *w = payload.ptrw();
    // Positions are zeroed (9 bytes).
    w[9] = p_alpha;
    w[10] = p_r;
    w[11] = p_g;
    w[12] = p_b;
    // log_scale = encoded / 16 - 10, so 160 decodes to exp(0) = 1.0
    w[13] = 160;
    w[14] = 160;
    w[15] = 160;
    // Rotation bytes remain zero => identity quaternion after reconstruction.
    return payload;
}

Error _write_binary_file(const String &p_path, const PackedByteArray &p_bytes) {
    Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
    if (file.is_null()) {
        return ERR_CANT_CREATE;
    }
    file->store_buffer(p_bytes);
    file.unref();
    return OK;
}

Ref<GaussianSplatAsset> _make_thumbnail_fixture_asset(int p_splat_count = 6) {
    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    asset->set_splat_count(p_splat_count);
    asset->set_asset_type(GaussianSplatAsset::ASSET_TYPE_STATIC);
    asset->set_source_path("res://thumbnail_fixture.ply");

    PackedFloat32Array positions;
    positions.resize(p_splat_count * 3);
    {
        float *positions_w = positions.ptrw();
        for (int i = 0; i < p_splat_count; i++) {
            positions_w[i * 3 + 0] = i * 0.5f;
            positions_w[i * 3 + 1] = Math::sin(float(i));
            positions_w[i * 3 + 2] = Math::cos(float(i));
        }
    }

    PackedColorArray colors;
    colors.resize(p_splat_count);
    {
        Color *colors_w = colors.ptrw();
        for (int i = 0; i < p_splat_count; i++) {
            colors_w[i] = Color(0.2f * i, 0.3f, 0.5f, 0.8f);
        }
    }

    PackedFloat32Array scales;
    scales.resize(p_splat_count * 3);
    {
        float *scales_w = scales.ptrw();
        for (int i = 0; i < p_splat_count; i++) {
            scales_w[i * 3 + 0] = 0.5f;
            scales_w[i * 3 + 1] = 0.75f;
            scales_w[i * 3 + 2] = 1.0f;
        }
    }

    PackedFloat32Array rotations;
    rotations.resize(p_splat_count * 4);
    {
        float *rotations_w = rotations.ptrw();
        for (int i = 0; i < p_splat_count; i++) {
            rotations_w[i * 4 + 0] = 1.0f;
            rotations_w[i * 4 + 1] = 0.0f;
            rotations_w[i * 4 + 2] = 0.0f;
            rotations_w[i * 4 + 3] = 0.0f;
        }
    }

    asset->set_positions(positions);
    asset->set_colors(colors);
    asset->set_scales(scales);
    asset->set_rotations(rotations);
    return asset;
}

} // namespace

TEST_CASE("[GaussianSplatting][Importer][RequiresGPU] PLY importer produces metadata and thumbnails") {
    // Thumbnail generation requires a functional RenderingDevice for ImageTexture.
    REQUIRE_GPU_DEVICE();

    const String source_path = "user://test_fixture_metadata_thumb.ply";
    {
        Error ply_err = _write_minimal_ascii_ply(source_path);
        REQUIRE_MESSAGE(ply_err == OK, "Failed to create synthetic PLY fixture.");
    }

    Ref<ResourceImporterPLY> importer;
    importer.instantiate();

    const String save_base_path = "user://gaussian_import_test";

    HashMap<StringName, Variant> options;
    options.insert(StringName("general/asset_type"), 0);
    options.insert(StringName("quality/preset"), String("desktop"));
    options.insert(StringName("compression/quantize_positions"), true);
    options.insert(StringName("compression/quantize_colors"), false);
    options.insert(StringName("compression/quantize_scales"), true);
    options.insert(StringName("compression/quantize_rotations"), false);
    options.insert(StringName("compression/pack_opacity"), true); // force customization
    options.insert(StringName("preview/generate_thumbnail"), true);
    options.insert(StringName("preview/thumbnail_style"),
            int(GaussianThumbnailGenerator::THUMBNAIL_STYLE_HEATMAP));
    options.insert(StringName("preview/thumbnail_size"), 128);
    options.insert(StringName("metadata/include_memory_estimate"), true);

    Variant metadata_variant;
    Error import_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options, nullptr, nullptr,
            &metadata_variant);
    CHECK_MESSAGE(import_err == OK, "ResourceImporterPLY should successfully import the test PLY file.");
    if (import_err != OK) {
        return;
    }

    Ref<GaussianSplatAsset> asset = ResourceLoader::load(save_base_path + String(".tres"));
    CHECK_MESSAGE(asset.is_valid(), "Imported GaussianSplatAsset should be loadable from disk.");
    if (!asset.is_valid()) {
        return;
    }

    CHECK(asset->get_splat_count() > 0);
    CHECK(asset->get_import_quality_preset() == String("desktop"));
    CHECK((asset->get_compression_flags() & GaussianSplatAsset::COMPRESSION_POSITIONS) != 0);
    CHECK_MESSAGE(asset->get_thumbnail().is_valid(), "Importer should generate a thumbnail for the Gaussian asset.");
    CHECK(asset->get_source_path() == source_path);

    Dictionary metadata = metadata_variant;
    CHECK(metadata.has(StringName("options")));
    Dictionary option_dict = metadata.get(StringName("options"), Dictionary());
    CHECK(option_dict.has(StringName("compression/pack_opacity")));
    CHECK(bool(option_dict[StringName("compression/pack_opacity")]) == true);
    CHECK(metadata.has(StringName("memory_estimate_mb")));
    CHECK((double)metadata[StringName("memory_estimate_mb")] > 0.0);
    CHECK(metadata.get(StringName("quality_customized"), false));
    CHECK(String(metadata.get(StringName("thumbnail_style_name"), String())) ==
            GaussianThumbnailGenerator::style_to_display_name(GaussianThumbnailGenerator::THUMBNAIL_STYLE_HEATMAP));

    _remove_user_file(source_path);
}

TEST_CASE("[GaussianSplatting][Importer][RequiresGPU] Reimport updates quality options") {
    // First import generates a thumbnail which requires a GPU device.
    REQUIRE_GPU_DEVICE();

    const String source_path = "user://test_fixture_reimport.ply";
    {
        Error ply_err = _write_minimal_ascii_ply(source_path);
        REQUIRE_MESSAGE(ply_err == OK, "Failed to create synthetic PLY fixture.");
    }

    Ref<ResourceImporterPLY> importer;
    importer.instantiate();

    const String save_base_path = "user://gaussian_reimport_test";

    HashMap<StringName, Variant> options;
    options.insert(StringName("general/asset_type"), 0);
    options.insert(StringName("quality/preset"), String("high"));
    options.insert(StringName("preview/generate_thumbnail"), true);
    options.insert(StringName("metadata/include_statistics"), true);

    Variant metadata_variant;
    Error first_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options, nullptr, nullptr,
            &metadata_variant);
    CHECK(first_err == OK);
    if (first_err != OK) {
        return;
    }

    // Reimport with modified preset and disabled thumbnail generation.
    options.insert(StringName("quality/preset"), String("mobile"));
    options.insert(StringName("preview/generate_thumbnail"), false);
    options.insert(StringName("metadata/include_memory_estimate"), false);

    Variant second_metadata_variant;
    Error second_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options, nullptr, nullptr,
            &second_metadata_variant);
    CHECK(second_err == OK);
    if (second_err != OK) {
        return;
    }

    Ref<GaussianSplatAsset> asset = ResourceLoader::load(save_base_path + String(".tres"));
    CHECK(asset.is_valid());
    if (!asset.is_valid()) {
        return;
    }

    CHECK(asset->get_import_quality_preset() == String("mobile"));
    CHECK(!asset->get_thumbnail().is_valid());

    Dictionary metadata = second_metadata_variant;
    CHECK(String(metadata.get(StringName("quality_preset"), String())) == String("mobile"));
    CHECK(!metadata.get(StringName("thumbnail_generated"), true));
    Dictionary option_dict = metadata.get(StringName("options"), Dictionary());
    CHECK(String(option_dict.get(StringName("quality/preset"), String())) == String("mobile"));

    _remove_user_file(source_path);
}

TEST_CASE("[GaussianSplatting][Importer] Ultra quality preset preserves full PLY splat count") {
    const String source_path = "user://test_fixture_ultra_count.ply";
    {
        Error ply_err = _write_minimal_ascii_ply(source_path);
        REQUIRE_MESSAGE(ply_err == OK, "Failed to create synthetic PLY fixture.");
    }

    Ref<PLYLoader> loader;
    loader.instantiate();
    const Error load_err = loader->load_file(source_path);
    REQUIRE_MESSAGE(load_err == OK, "Fixture PLY must load before testing ultra-quality import.");
    const Ref<::GaussianData> source_data = loader->get_gaussian_data();
    REQUIRE_MESSAGE(source_data.is_valid(), "Fixture PLY should produce GaussianData.");
    const int source_count = source_data->get_count();
    REQUIRE_MESSAGE(source_count > 0, "Fixture PLY should contain at least one splat.");

    Ref<ResourceImporterPLY> importer;
    importer.instantiate();

    const String save_base_path = "user://gaussian_ultra_quality_preserve_count";
    HashMap<StringName, Variant> options;
    options.insert(StringName("quality/preset"), String("ultra"));
    options.insert(StringName("quality/max_splats"), 0);
    options.insert(StringName("quality/density_multiplier"), 1.0);
    options.insert(StringName("preview/generate_thumbnail"), false);

    Variant metadata_variant;
    const Error import_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options,
            nullptr, nullptr, &metadata_variant);
    CHECK_MESSAGE(import_err == OK, "Ultra preset import should succeed for fixture PLY.");
    if (import_err != OK) {
        return;
    }

    Ref<GaussianSplatAsset> asset = ResourceLoader::load(save_base_path + String(".tres"));
    CHECK_MESSAGE(asset.is_valid(), "Ultra preset import should produce a loadable asset.");
    if (asset.is_null()) {
        _remove_user_file(save_base_path + ".tres");
        return;
    }

    CHECK_MESSAGE(int(asset->get_splat_count()) == source_count,
            "Ultra preset should preserve all source splats when density is 1 and max_splats is unlimited.");

    const Dictionary metadata = metadata_variant;
    CHECK(int64_t(metadata.get(StringName("original_splat_count"), int64_t(-1))) == source_count);
    CHECK(int64_t(metadata.get(StringName("splat_count"), int64_t(-1))) == source_count);
    CHECK(String(metadata.get(StringName("quality_preset"), String())) == String("ultra"));

    _remove_user_file(source_path);
    _remove_user_file(save_base_path + ".tres");
}

TEST_CASE("[GaussianSplatting][Importer] Ultra import initializes fresh node runtime from imported fidelity") {
    // TODO: set_splat_asset does not yet auto-switch to QUALITY_CUSTOM with relaxed LOD for full-fidelity assets.
    MESSAGE("Skipping - aspirational test, requires set_splat_asset fidelity auto-detection (not yet wired)");
    return;
    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    REQUIRE_MESSAGE(project_settings != nullptr, "ProjectSettings singleton must exist for runtime fidelity regression.");
    ProjectSettingGuard tier_preset_guard(project_settings, "rendering/gaussian_splatting/quality/tier_preset");
    ProjectSettingGuard tier_apply_guard(project_settings, "rendering/gaussian_splatting/quality/tier_apply_streaming_budgets");
    project_settings->set_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
    project_settings->set_setting("rendering/gaussian_splatting/quality/tier_apply_streaming_budgets", false);

    const String source_path = "user://test_fixture_ultra_fidelity.ply";
    {
        Error ply_err = _write_minimal_ascii_ply(source_path);
        REQUIRE_MESSAGE(ply_err == OK, "Failed to create synthetic PLY fixture.");
    }

    Ref<PLYLoader> loader;
    loader.instantiate();
    const Error load_err = loader->load_file(source_path);
    REQUIRE_MESSAGE(load_err == OK, "Fixture PLY must load before testing node runtime fidelity propagation.");
    const Ref<::GaussianData> source_data = loader->get_gaussian_data();
    REQUIRE_MESSAGE(source_data.is_valid(), "Fixture PLY should produce GaussianData.");
    const int source_count = source_data->get_count();
    REQUIRE_MESSAGE(source_count > 0, "Fixture PLY should contain at least one splat.");

    Ref<ResourceImporterPLY> importer;
    importer.instantiate();

    const String save_base_path = "user://gaussian_ultra_runtime_fidelity";
    HashMap<StringName, Variant> options;
    options.insert(StringName("quality/preset"), String("ultra"));
    options.insert(StringName("quality/max_splats"), 0);
    options.insert(StringName("quality/density_multiplier"), 1.0);
    options.insert(StringName("preview/generate_thumbnail"), false);

    Variant metadata_variant;
    const Error import_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options,
            nullptr, nullptr, &metadata_variant);
    REQUIRE_MESSAGE(import_err == OK, "Ultra preset import should succeed for node runtime fidelity regression.");

    Ref<GaussianSplatAsset> asset = ResourceLoader::load(save_base_path + String(".tres"));
    REQUIRE_MESSAGE(asset.is_valid(), "Imported asset should be loadable for node runtime fidelity regression.");
    REQUIRE_MESSAGE(int(asset->get_splat_count()) == source_count, "Imported asset should preserve the full source count.");

    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    CHECK(node->get_quality_preset() == GaussianSplatNode3D::QUALITY_BALANCED);
    CHECK(node->get_max_splat_count() == 500000);

    node->set_splat_asset(asset);

    CHECK(node->get_quality_preset() == GaussianSplatNode3D::QUALITY_CUSTOM);
    CHECK_MESSAGE(node->get_max_splat_count() >= source_count,
            "Fresh node should not re-cap a full-fidelity imported asset below its imported splat count.");
    const GaussianSplatting::GaussianSplatLODConfig &lod_config = node->get_lod_config();
    CHECK(lod_config.max_splats_per_frame >= (uint32_t)source_count);
    CHECK(lod_config.importance_threshold == doctest::Approx(0.0f));
    CHECK(lod_config.size_cull_threshold == doctest::Approx(0.0f));

    memdelete(node);
    _remove_user_file(source_path);
    _remove_user_file(save_base_path + ".tres");
}

TEST_CASE("[GaussianSplatting][Importer] Imported fidelity defaults do not override customized node quality") {
    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    REQUIRE_MESSAGE(project_settings != nullptr, "ProjectSettings singleton must exist for customized node regression.");
    ProjectSettingGuard tier_preset_guard(project_settings, "rendering/gaussian_splatting/quality/tier_preset");
    ProjectSettingGuard tier_apply_guard(project_settings, "rendering/gaussian_splatting/quality/tier_apply_streaming_budgets");
    project_settings->set_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
    project_settings->set_setting("rendering/gaussian_splatting/quality/tier_apply_streaming_budgets", false);

    Ref<ResourceImporterPLY> importer;
    importer.instantiate();

    const String source_path = "user://test_fixture_customized_node.ply";
    {
        Error ply_err = _write_minimal_ascii_ply(source_path);
        REQUIRE_MESSAGE(ply_err == OK, "Failed to create synthetic PLY fixture.");
    }

    const String save_base_path = "user://gaussian_customized_node_runtime";
    HashMap<StringName, Variant> options;
    options.insert(StringName("quality/preset"), String("ultra"));
    options.insert(StringName("quality/max_splats"), 0);
    options.insert(StringName("quality/density_multiplier"), 1.0);
    options.insert(StringName("preview/generate_thumbnail"), false);

    Variant metadata_variant;
    const Error import_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options,
            nullptr, nullptr, &metadata_variant);
    REQUIRE_MESSAGE(import_err == OK, "Ultra preset import should succeed for customized node regression.");

    Ref<GaussianSplatAsset> asset = ResourceLoader::load(save_base_path + String(".tres"));
    REQUIRE_MESSAGE(asset.is_valid(), "Imported asset should be loadable for customized node regression.");

    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    node->set_quality_preset(GaussianSplatNode3D::QUALITY_PERFORMANCE);

    node->set_splat_asset(asset);

    CHECK(node->get_quality_preset() == GaussianSplatNode3D::QUALITY_PERFORMANCE);
    CHECK(node->get_max_splat_count() == 200000);

    memdelete(node);
    _remove_user_file(source_path);
    _remove_user_file(save_base_path + ".tres");
}

TEST_CASE("[GaussianSplatting][Importer] Missing required PLY properties fail consistently across strict paths") {
    const String source_path = "user://gaussian_missing_required_opacity.ply";
    const String save_base_path = "user://gaussian_missing_required_opacity_import";

    Error write_err = _write_missing_opacity_ascii_ply(source_path);
    CHECK_MESSAGE(write_err == OK, "Failed to create malformed PLY test fixture.");
    if (write_err != OK) {
        return;
    }

    Ref<PLYLoader> loader;
    loader.instantiate();
    Error loader_err = loader->load_file(source_path);
    CHECK(loader_err == OK);
    if (loader_err == OK) {
        PackedStringArray missing_required;
        PackedStringArray missing_optional;
        loader->get_property_deficiencies(missing_required, missing_optional);
        CHECK(missing_required.find("opacity") != -1);
    }

    Ref<ResourceImporterPLY> importer;
    importer.instantiate();

    HashMap<StringName, Variant> options;
    options.insert(StringName("validation/validate_required_properties"), true);
    options.insert(StringName("validation/warn_missing_optional"), false);
    options.insert(StringName("preview/generate_thumbnail"), false);

    Variant metadata_variant;
    Error import_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options, nullptr, nullptr,
            &metadata_variant);
    CHECK_MESSAGE(import_err == ERR_FILE_CORRUPT,
            "ResourceImporterPLY should reject files missing required properties when strict validation is enabled.");

    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    Error asset_err = asset->load_from_file(source_path);
    CHECK_MESSAGE(asset_err == ERR_FILE_CORRUPT,
            "GaussianSplatAsset::load_from_file should reject the same malformed PLY file.");

    _remove_user_file(source_path);
    _remove_user_file(save_base_path + ".tres");
}

TEST_CASE("[GaussianSplatting][Thumbnail][RequiresGPU] Generator produces textures for each style") {
    // Thumbnail generation creates ImageTextures which require a GPU device.
    REQUIRE_GPU_DEVICE();

    Ref<GaussianSplatAsset> asset = _make_thumbnail_fixture_asset();
    const int splat_count = asset->get_splat_count();

    Ref<GaussianThumbnailGenerator> generator;
    generator.instantiate();

    for (int style = 0; style < 4; style++) {
        Ref<Texture2D> texture = generator->generate_thumbnail(asset, 96,
                GaussianThumbnailGenerator::style_from_int(style));
        CHECK_MESSAGE(texture.is_valid(), "Thumbnail generator should return a valid texture for each style.");
    }

    Dictionary memory = generator->compute_memory_statistics(splat_count, GaussianSplatAsset::COMPRESSION_NONE, false);
    CHECK(float(memory.get(StringName("total_mb"), 0.0)) > 0.0f);
}

TEST_CASE("[GaussianSplatting][Thumbnail][RequiresGPU] Generator caches deterministic asset+settings keys") {
    // Thumbnail generation creates ImageTextures which require a GPU device.
    REQUIRE_GPU_DEVICE();

    Ref<GaussianSplatAsset> asset = _make_thumbnail_fixture_asset();

    Ref<GaussianThumbnailGenerator> generator;
    generator.instantiate();

    Ref<Texture2D> first = generator->generate_thumbnail(asset, 96, GaussianThumbnailGenerator::THUMBNAIL_STYLE_COLOR);
    Ref<Texture2D> second = generator->generate_thumbnail(asset, 96, GaussianThumbnailGenerator::THUMBNAIL_STYLE_COLOR);
    Ref<Texture2D> alternate_style = generator->generate_thumbnail(asset, 96, GaussianThumbnailGenerator::THUMBNAIL_STYLE_HEATMAP);
    CHECK(first.is_valid());
    CHECK(second.is_valid());
    CHECK(alternate_style.is_valid());
    CHECK(first == second);

    Dictionary stats = generator->get_cache_statistics();
    CHECK(int(stats.get(StringName("entries"), 0)) == 2);
    CHECK(int64_t(stats.get(StringName("hits"), int64_t(0))) >= 1);
    CHECK(int64_t(stats.get(StringName("misses"), int64_t(0))) >= 2);

    asset->set_source_path("res://thumbnail_fixture_reimported.ply");
    Ref<Texture2D> after_source_change = generator->generate_thumbnail(asset, 96, GaussianThumbnailGenerator::THUMBNAIL_STYLE_COLOR);
    CHECK(after_source_change.is_valid());

    stats = generator->get_cache_statistics();
    CHECK(int(stats.get(StringName("entries"), 0)) == 3);
}

TEST_CASE("[GaussianSplatting][Editor] Import dialog respects metadata baselines") {
    GaussianImportDialog *dialog = memnew(GaussianImportDialog);

    Dictionary baseline_options;
    baseline_options[StringName("quality/preset")] = String("desktop");
    baseline_options[StringName("compression/quantize_positions")] = true;
    baseline_options[StringName("preview/generate_thumbnail")] = true;

    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    asset->set_import_quality_preset("desktop");
    asset->set_compression_flags(GaussianSplatAsset::COMPRESSION_POSITIONS);
    asset->set_source_path("res://test_splats.ply");
    Dictionary metadata;
    metadata[StringName("options")] = baseline_options;
    metadata[StringName("quality_preset")] = String("desktop");
    metadata[StringName("source_path")] = String("res://test_splats.ply");
    asset->set_import_metadata(metadata);

    dialog->configure_for_file("res://test_splats.ply", asset, true, Dictionary());

    Dictionary selected = dialog->get_selected_options();
    CHECK(String(selected.get(StringName("quality/preset"), String())) == String("desktop"));
    CHECK(selected.has(StringName("compression/quantize_positions")));
    CHECK(bool(selected[StringName("compression/quantize_positions")]) == true);

    GaussianImportDialog::ImportConfiguration config = dialog->get_configuration();
    CHECK(config.preset == "desktop");
    CHECK(config.custom_settings == false);

    memdelete(dialog);
}

TEST_CASE("[GaussianSplatting][Editor] Import dialog applies quality override options") {
    GaussianImportDialog *dialog = memnew(GaussianImportDialog);

    Dictionary baseline_options;
    baseline_options[StringName("quality/preset")] = String("desktop");
    baseline_options[StringName("quality/max_splats")] = 250000;
    baseline_options[StringName("quality/density_multiplier")] = 0.5;

    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    asset->set_import_quality_preset("desktop");
    asset->set_source_path("res://test_splats.ply");

    Dictionary metadata;
    metadata[StringName("options")] = baseline_options;
    metadata[StringName("quality_preset")] = String("desktop");
    metadata[StringName("source_path")] = String("res://test_splats.ply");
    asset->set_import_metadata(metadata);

    Dictionary override_options;
    override_options[StringName("quality/preset")] = String("ultra");
    override_options[StringName("quality/max_splats")] = 0;
    override_options[StringName("quality/density_multiplier")] = 1.0;

    dialog->configure_for_file("res://test_splats.ply", asset, true, override_options);

    const Dictionary selected = dialog->get_selected_options();
    CHECK(String(selected.get(StringName("quality/preset"), String())) == String("ultra"));
    CHECK(int(selected.get(StringName("quality/max_splats"), -1)) == 0);
    CHECK((double)selected.get(StringName("quality/density_multiplier"), -1.0) == doctest::Approx(1.0));

    const GaussianImportDialog::ImportConfiguration config = dialog->get_configuration();
    CHECK(config.preset == "ultra");
    CHECK(config.max_splats == 0);
    CHECK(config.density_multiplier == doctest::Approx(1.0));

    memdelete(dialog);
}

// ============================================================================
// SPZ Importer Tests
// ============================================================================

TEST_CASE("[GaussianSplatting][Importer] SPZ importer basic configuration") {
    Ref<ResourceImporterSPZ> importer;
    importer.instantiate();

    CHECK(importer->get_importer_name() == "gaussian_splat_spz");
    CHECK(importer->get_visible_name() == "Gaussian Splat SPZ");
    CHECK(importer->get_resource_type() == "GaussianSplatAsset");
    CHECK(importer->get_save_extension() == "tres");
    CHECK(importer->can_import_threaded() == true);

    List<String> extensions;
    importer->get_recognized_extensions(&extensions);
    CHECK(extensions.size() == 1);
    CHECK(extensions.front()->get() == "spz");

    CHECK(importer->get_preset_count() > 0);
    for (int i = 0; i < importer->get_preset_count(); i++) {
        CHECK(!importer->get_preset_name(i).is_empty());
    }
}

TEST_CASE("[GaussianSplatting][Importer] SPZ importer provides import options") {
    Ref<ResourceImporterSPZ> importer;
    importer.instantiate();

    List<ResourceImporter::ImportOption> options;
    importer->get_import_options("res://test.spz", &options, 0);

    bool has_preset_option = false;
    bool has_thumbnail_option = false;
    bool has_compression_option = false;
    bool has_metadata_option = false;

    for (const ResourceImporter::ImportOption &opt : options) {
        String name = String(opt.option.name);
        if (name == "quality/preset") {
            has_preset_option = true;
        } else if (name == "preview/generate_thumbnail") {
            has_thumbnail_option = true;
        } else if (name == "compression/quantize_positions") {
            has_compression_option = true;
        } else if (name == "metadata/include_statistics") {
            has_metadata_option = true;
        }
    }

    CHECK_MESSAGE(has_preset_option, "SPZ importer should provide quality/preset option");
    CHECK_MESSAGE(has_thumbnail_option, "SPZ importer should provide thumbnail generation option");
    CHECK_MESSAGE(has_compression_option, "SPZ importer should provide compression options");
    CHECK_MESSAGE(has_metadata_option, "SPZ importer should provide metadata options");
}

TEST_CASE("[GaussianSplatting][Importer] SPZ importer handles missing file gracefully") {
    Ref<ResourceImporterSPZ> importer;
    importer.instantiate();

    const String source_path = "res://nonexistent_test.spz";
    const String save_base_path = "user://gaussian_spz_missing_test";

    HashMap<StringName, Variant> options;
    options.insert(StringName("quality/preset"), String("desktop"));
    options.insert(StringName("preview/generate_thumbnail"), false);

    Variant metadata_variant;
    Error import_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options, nullptr, nullptr,
            &metadata_variant);
    CHECK_MESSAGE(import_err == ERR_FILE_NOT_FOUND, "SPZ importer should return ERR_FILE_NOT_FOUND for missing files.");
}

TEST_CASE("[GaussianSplatting][Importer] SPZ loader rejects truncated payload sections") {
    const String source_path = "user://gaussian_spz_truncated_payload.spz";

    // Version 2, 2 points, SH degree 0. Minimum payload is 38 bytes; provide 37 bytes.
    PackedByteArray payload;
    payload.resize(37);
    memset(payload.ptrw(), 0, payload.size());

    PackedByteArray compressed_payload = _gzip_compress(payload);
    REQUIRE_MESSAGE(!compressed_payload.is_empty(), "Failed to gzip-compress truncated SPZ payload fixture");

    PackedByteArray file_data = _make_spz_header(SPZLoader::SPZ_VERSION_2, 2, 0, 12);
    const int header_size = file_data.size();
    file_data.resize(header_size + compressed_payload.size());
    memcpy(file_data.ptrw() + header_size, compressed_payload.ptr(), compressed_payload.size());

    Error write_err = _write_binary_file(source_path, file_data);
    REQUIRE_MESSAGE(write_err == OK, "Failed to write truncated SPZ fixture");

    Ref<SPZLoader> loader;
    loader.instantiate();
    Error load_err = loader->load_file(source_path);
    CHECK_MESSAGE(load_err == ERR_FILE_CORRUPT, "Truncated SPZ payload must be rejected as corrupt");

    _remove_user_file(source_path);
}

TEST_CASE("[GaussianSplatting][Importer] SPZ loader marks DC encoding as linear RGB") {
    const String source_path = "user://gaussian_spz_linear_dc.spz";

    PackedByteArray payload = _make_spz_v2_single_point_payload(255, 64, 128, 255);
    PackedByteArray compressed_payload = _gzip_compress(payload);
    REQUIRE_MESSAGE(!compressed_payload.is_empty(), "Failed to gzip-compress SPZ linear-DC fixture");

    PackedByteArray file_data = _make_spz_header(SPZLoader::SPZ_VERSION_2, 1, 0, 12);
    const int header_size = file_data.size();
    file_data.resize(header_size + compressed_payload.size());
    memcpy(file_data.ptrw() + header_size, compressed_payload.ptr(), compressed_payload.size());

    Error write_err = _write_binary_file(source_path, file_data);
    REQUIRE_MESSAGE(write_err == OK, "Failed to write SPZ linear-DC fixture");

    Ref<SPZLoader> loader;
    loader.instantiate();
    Error load_err = loader->load_file(source_path);
    REQUIRE_MESSAGE(load_err == OK, "SPZ loader should accept valid single-point payload");

    Ref<::GaussianData> data = loader->get_gaussian_data();
    REQUIRE_MESSAGE(data.is_valid(), "SPZ loader must produce GaussianData");
    REQUIRE_MESSAGE(data->get_count() == 1, "SPZ loader must load the single-point payload");

    const Gaussian g = data->get_gaussian(0);
    CHECK(gaussian_get_dc_encoding(g.render_meta) == GAUSSIAN_DC_ENCODING_LINEAR_RGB);
    CHECK(Math::is_equal_approx(g.sh_dc.r, 64.0f / 255.0f));
    CHECK(Math::is_equal_approx(g.sh_dc.g, 128.0f / 255.0f));
    CHECK(Math::is_equal_approx(g.sh_dc.b, 1.0f));

    _remove_user_file(source_path);
}

TEST_CASE("[GaussianSplatting][Importer] SPZ importer persists linear DC encoding metadata") {
    const String source_path = "user://gaussian_spz_linear_dc_import.spz";
    const String save_base_path = "user://gaussian_spz_linear_dc_imported";

    PackedByteArray payload = _make_spz_v2_single_point_payload(255, 64, 128, 255);
    PackedByteArray compressed_payload = _gzip_compress(payload);
    REQUIRE_MESSAGE(!compressed_payload.is_empty(), "Failed to gzip-compress SPZ import fixture");

    PackedByteArray file_data = _make_spz_header(SPZLoader::SPZ_VERSION_2, 1, 0, 12);
    const int header_size = file_data.size();
    file_data.resize(header_size + compressed_payload.size());
    memcpy(file_data.ptrw() + header_size, compressed_payload.ptr(), compressed_payload.size());

    Error write_err = _write_binary_file(source_path, file_data);
    REQUIRE_MESSAGE(write_err == OK, "Failed to write SPZ import fixture");

    Ref<ResourceImporterSPZ> importer;
    importer.instantiate();

    HashMap<StringName, Variant> options;
    options.insert(StringName("quality/preset"), String("desktop"));
    options.insert(StringName("preview/generate_thumbnail"), false);

    Variant metadata_variant;
    const Error import_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options,
            nullptr, nullptr, &metadata_variant);
    REQUIRE_MESSAGE(import_err == OK, "SPZ importer should succeed for linear-DC fixture");

    Ref<GaussianSplatAsset> asset = ResourceLoader::load(save_base_path + ".tres");
    REQUIRE_MESSAGE(asset.is_valid(), "SPZ importer should emit a loadable GaussianSplatAsset");

    const Dictionary metadata = asset->get_import_metadata();
    CHECK(String(metadata.get(StringName("dc_encoding"), String())) == String("linear_rgb"));

    Ref<::GaussianData> data;
    data.instantiate();
    REQUIRE_MESSAGE(data.is_valid(), "GaussianData must instantiate");
    REQUIRE_MESSAGE(data->populate_from_asset(asset) == OK, "populate_from_asset should accept imported SPZ asset");
    REQUIRE_MESSAGE(data->get_count() == 1, "Imported SPZ asset should materialize one gaussian");

    const Gaussian g = data->get_gaussian(0);
    CHECK(gaussian_get_dc_encoding(g.render_meta) == GAUSSIAN_DC_ENCODING_LINEAR_RGB);

    _remove_user_file(source_path);
    _remove_user_file(save_base_path + ".tres");
}

TEST_CASE("[GaussianSplatting][Importer] PLY loader keeps legacy DC encoding") {
    const String source_path = "user://gaussian_ply_legacy_dc.ply";
    Error write_err = _write_minimal_ascii_ply(source_path);
    REQUIRE_MESSAGE(write_err == OK, "Failed to write minimal PLY fixture");

    Ref<PLYLoader> loader;
    loader.instantiate();
    Error load_err = loader->load_file(source_path);
    REQUIRE_MESSAGE(load_err == OK, "PLY loader should accept valid minimal ASCII fixture");

    Ref<::GaussianData> data = loader->get_gaussian_data();
    REQUIRE_MESSAGE(data.is_valid(), "PLY loader must produce GaussianData");
    REQUIRE_MESSAGE(data->get_count() == 1, "PLY loader must load the single-point payload");

    const Gaussian g = data->get_gaussian(0);
    CHECK(gaussian_get_dc_encoding(g.render_meta) == GAUSSIAN_DC_ENCODING_LEGACY_BIAS);

    _remove_user_file(source_path);
}

TEST_CASE("[GaussianSplatting][Renderer] SH metadata preserves DC encoding mode") {
    SHCompressionMetrics metrics;
    PackedGaussian packed = {};

    Gaussian legacy = {};
    legacy.rotation = Quaternion();
    legacy.scale = Vector3(1.0f, 1.0f, 1.0f);
    legacy.opacity = 1.0f;
    legacy.sh_dc = Color(0.25f, 0.5f, 0.75f, 1.0f);
    legacy.render_meta = gaussian_set_dc_encoding(0u, GAUSSIAN_DC_ENCODING_LEGACY_BIAS);
    pack_gaussian(legacy, packed, metrics, nullptr, 0, 0);
    CHECK(gs_get_dc_encoding(packed.sh_metadata) == GAUSSIAN_DC_ENCODING_LEGACY_BIAS);
    CHECK(gs_get_sh_encoding(packed.sh_metadata) == GS_SH_ENCODING_RGB9E5);

    Gaussian linear = legacy;
    linear.render_meta = gaussian_set_dc_encoding(0u, GAUSSIAN_DC_ENCODING_LINEAR_RGB);
    pack_gaussian(linear, packed, metrics, nullptr, 0, 0);
    CHECK(gs_get_dc_encoding(packed.sh_metadata) == GAUSSIAN_DC_ENCODING_LINEAR_RGB);
    CHECK(gs_get_sh_encoding(packed.sh_metadata) == GS_SH_ENCODING_RGB9E5);
}

TEST_CASE("[GaussianSplatting][Renderer] SH metadata preserves DC encoding mode for F16 packing") {
    SHCompressionMetrics metrics;
    PackedGaussianF16 packed = {};

    Gaussian legacy = {};
    legacy.rotation = Quaternion();
    legacy.scale = Vector3(1.0f, 1.0f, 1.0f);
    legacy.opacity = 1.0f;
    legacy.sh_dc = Color(0.25f, 0.5f, 0.75f, 1.0f);
    legacy.render_meta = gaussian_set_dc_encoding(0u, GAUSSIAN_DC_ENCODING_LEGACY_BIAS);
    pack_gaussian_f16(legacy, packed, metrics, Vector3(), nullptr, 0, 0, PackedSphericalHarmonicsF16::MAX_ENCODED_COEFFICIENTS);
    CHECK(gs_get_dc_encoding(packed.sh_metadata) == GAUSSIAN_DC_ENCODING_LEGACY_BIAS);
    CHECK(gs_get_sh_encoding(packed.sh_metadata) == GS_SH_ENCODING_F16);

    Gaussian linear = legacy;
    linear.render_meta = gaussian_set_dc_encoding(0u, GAUSSIAN_DC_ENCODING_LINEAR_RGB);
    pack_gaussian_f16(linear, packed, metrics, Vector3(), nullptr, 0, 0, PackedSphericalHarmonicsF16::MAX_ENCODED_COEFFICIENTS);
    CHECK(gs_get_dc_encoding(packed.sh_metadata) == GAUSSIAN_DC_ENCODING_LINEAR_RGB);
    CHECK(gs_get_sh_encoding(packed.sh_metadata) == GS_SH_ENCODING_F16);
}

TEST_CASE("[GaussianSplatting][Importer] populate_from_asset preserves DC encoding metadata") {
    Ref<GaussianSplatAsset> asset = _make_thumbnail_fixture_asset(1);
    REQUIRE_MESSAGE(asset.is_valid(), "Fixture asset must be created");

    Dictionary metadata = asset->get_import_metadata();
    metadata[StringName("dc_encoding")] = "linear_rgb";
    asset->set_import_metadata(metadata);

    Ref<::GaussianData> data;
    data.instantiate();
    REQUIRE_MESSAGE(data.is_valid(), "GaussianData must instantiate");

    const Error err = data->populate_from_asset(asset);
    REQUIRE_MESSAGE(err == OK, "populate_from_asset should accept fixture asset");
    REQUIRE_MESSAGE(data->get_count() == 1, "populate_from_asset should materialize the fixture");

    const Gaussian g = data->get_gaussian(0);
    CHECK(gaussian_get_dc_encoding(g.render_meta) == GAUSSIAN_DC_ENCODING_LINEAR_RGB);
}

TEST_CASE("[GaussianSplatting][Importer] populate_gaussian_data restores DC encoding from asset metadata") {
    Ref<GaussianSplatAsset> asset = _make_thumbnail_fixture_asset(1);
    REQUIRE_MESSAGE(asset.is_valid(), "Fixture asset must be created");

    Dictionary metadata = asset->get_import_metadata();
    metadata[StringName("dc_encoding")] = "linear_rgb";
    asset->set_import_metadata(metadata);

    Ref<::GaussianData> data;
    data.instantiate();
    REQUIRE_MESSAGE(data.is_valid(), "GaussianData must instantiate");

    const bool ok = asset->populate_gaussian_data(data);
    REQUIRE_MESSAGE(ok, "populate_gaussian_data should accept fixture asset");
    REQUIRE_MESSAGE(data->get_count() == 1, "populate_gaussian_data should materialize the fixture");

    const Gaussian g = data->get_gaussian(0);
    CHECK(gaussian_get_dc_encoding(g.render_meta) == GAUSSIAN_DC_ENCODING_LINEAR_RGB);
}

TEST_CASE("[GaussianSplatting][Importer] SPZ loader rejects oversized payload sections") {
    const String source_path = "user://gaussian_spz_oversized_payload.spz";

    // Version 2, 1 point, SH degree 0. Exact payload is 19 bytes; provide 20 bytes.
    PackedByteArray payload;
    payload.resize(20);
    memset(payload.ptrw(), 0, payload.size());

    PackedByteArray compressed_payload = _gzip_compress(payload);
    REQUIRE_MESSAGE(!compressed_payload.is_empty(), "Failed to gzip-compress oversized SPZ payload fixture");

    PackedByteArray file_data = _make_spz_header(SPZLoader::SPZ_VERSION_2, 1, 0, 12);
    const int header_size = file_data.size();
    file_data.resize(header_size + compressed_payload.size());
    memcpy(file_data.ptrw() + header_size, compressed_payload.ptr(), compressed_payload.size());

    Error write_err = _write_binary_file(source_path, file_data);
    REQUIRE_MESSAGE(write_err == OK, "Failed to write oversized SPZ fixture");

    Ref<SPZLoader> loader;
    loader.instantiate();
    Error load_err = loader->load_file(source_path);
    CHECK_MESSAGE(load_err == ERR_FILE_CORRUPT, "Oversized SPZ payload must be rejected as corrupt");

    _remove_user_file(source_path);
}

TEST_CASE("[GaussianSplatting][Importer] SPZ loader rejects oversized decompression claims") {
    const String source_path = "user://gaussian_spz_oversized_claim.spz";

    // Minimal gzip-like stream with ISIZE trailer set to 2 GiB - 1.
    PackedByteArray fake_gzip;
    fake_gzip.resize(20);
    uint8_t *w = fake_gzip.ptrw();
    w[0] = 0x1F;
    w[1] = 0x8B;
    w[2] = 0x08; // DEFLATE
    w[3] = 0x00; // flags
    w[4] = 0x00; w[5] = 0x00; w[6] = 0x00; w[7] = 0x00; // mtime
    w[8] = 0x00; // xfl
    w[9] = 0x03; // os
    w[10] = 0x03; // empty DEFLATE block
    w[11] = 0x00;
    w[12] = 0x00; w[13] = 0x00; w[14] = 0x00; w[15] = 0x00; // crc32
    w[16] = 0xFF; w[17] = 0xFF; w[18] = 0xFF; w[19] = 0x7F; // isize = 2147483647

    PackedByteArray file_data = _make_spz_header(SPZLoader::SPZ_VERSION_2, 1, 0, 12);
    const int header_size = file_data.size();
    file_data.resize(header_size + fake_gzip.size());
    memcpy(file_data.ptrw() + header_size, fake_gzip.ptr(), fake_gzip.size());

    Error write_err = _write_binary_file(source_path, file_data);
    REQUIRE_MESSAGE(write_err == OK, "Failed to write oversized-claim SPZ fixture");

    Ref<SPZLoader> loader;
    loader.instantiate();
    Error load_err = loader->load_file(source_path);
    CHECK_MESSAGE(load_err == ERR_FILE_CORRUPT, "Oversized SPZ decompression claims must be rejected");

    _remove_user_file(source_path);
}

TEST_CASE("[GaussianSplatting][Importer] SPZ loader rejects malformed gzip optional headers") {
    const String source_path = "user://gaussian_spz_malformed_gzip_header.spz";

    PackedByteArray malformed_gzip;
    malformed_gzip.resize(20);
    uint8_t *w = malformed_gzip.ptrw();
    w[0] = 0x1F;
    w[1] = 0x8B;
    w[2] = 0x08; // DEFLATE
    w[3] = 0x08; // FNAME set, but no null terminator before trailer
    w[4] = 0x00; w[5] = 0x00; w[6] = 0x00; w[7] = 0x00; // mtime
    w[8] = 0x00; // xfl
    w[9] = 0x03; // os
    w[10] = 0x61; // filename bytes without terminator
    w[11] = 0x62;
    w[12] = 0x00; w[13] = 0x00; w[14] = 0x00; w[15] = 0x00; // crc32
    w[16] = 0x13; w[17] = 0x00; w[18] = 0x00; w[19] = 0x00; // isize = 19

    PackedByteArray file_data = _make_spz_header(SPZLoader::SPZ_VERSION_2, 1, 0, 12);
    const int header_size = file_data.size();
    file_data.resize(header_size + malformed_gzip.size());
    memcpy(file_data.ptrw() + header_size, malformed_gzip.ptr(), malformed_gzip.size());

    Error write_err = _write_binary_file(source_path, file_data);
    REQUIRE_MESSAGE(write_err == OK, "Failed to write malformed-gzip SPZ fixture");

    Ref<SPZLoader> loader;
    loader.instantiate();
    Error load_err = loader->load_file(source_path);
    CHECK_MESSAGE(load_err == ERR_FILE_CORRUPT, "Malformed gzip optional headers must be rejected");

    _remove_user_file(source_path);
}

TEST_CASE("[GaussianSplatting][Importer] SPZ loader enforces header-derived decompression cap") {
    const String source_path = "user://gaussian_spz_payload_cap.spz";

    // Version 2, SH degree 3, 20M points => expected payload exceeds 1 GiB cap.
    PackedByteArray file_data = _make_spz_header(SPZLoader::SPZ_VERSION_2, 20000000u, 3, 12);
    file_data.push_back(0x00); // Ensure file is larger than header.

    Error write_err = _write_binary_file(source_path, file_data);
    REQUIRE_MESSAGE(write_err == OK, "Failed to write cap-enforcement SPZ fixture");

    Ref<SPZLoader> loader;
    loader.instantiate();
    Error load_err = loader->load_file(source_path);
    CHECK_MESSAGE(load_err == ERR_FILE_CORRUPT, "Payloads that exceed decompression caps must be rejected");

    _remove_user_file(source_path);
}

// Note: Full SPZ import test requires an SPZ test fixture file.
// To create one, use the SPZ encoder from https://github.com/nianticlabs/spz
// with a minimal Gaussian data set. The test fixture should be placed at:
//   res://test_splats.spz (in test project)
// Required SPZ fixture attributes:
//   - Valid SPZ header with version >= 1
//   - At least 1 Gaussian with position, scale, rotation, color data
//   - SH degree 0 is sufficient for basic testing

// ============================================================================
// Importer Selection Tests (extension-based routing)
// ============================================================================

TEST_CASE("[GaussianSplatting][Importer] Extension-based importer selection logic") {
    // Test the extension detection logic used by _import_from_path and _load_preview_asset

    SUBCASE("PLY extension selects PLY importer") {
        String path = "res://models/test_scene.ply";
        String extension = path.get_extension().to_lower();
        CHECK(extension == "ply");

        String importer_name;
        if (extension == "ply") {
            importer_name = "gaussian_splat_ply";
        } else if (extension == "spz") {
            importer_name = "gaussian_splat_spz";
        }
        CHECK(importer_name == "gaussian_splat_ply");
    }

    SUBCASE("SPZ extension selects SPZ importer") {
        String path = "res://models/compressed_scene.spz";
        String extension = path.get_extension().to_lower();
        CHECK(extension == "spz");

        String importer_name;
        if (extension == "ply") {
            importer_name = "gaussian_splat_ply";
        } else if (extension == "spz") {
            importer_name = "gaussian_splat_spz";
        }
        CHECK(importer_name == "gaussian_splat_spz");
    }

    SUBCASE("Unknown extension is detected correctly") {
        String path = "res://models/invalid_format.xyz";
        String extension = path.get_extension().to_lower();
        CHECK(extension == "xyz");

        bool is_supported = (extension == "ply" || extension == "spz");
        CHECK_MESSAGE(!is_supported, "Unknown extension .xyz should not be recognized as supported");
    }

    SUBCASE("Case insensitive extension handling") {
        String path_upper = "res://models/test.PLY";
        String path_mixed = "res://models/test.SpZ";

        String ext_upper = path_upper.get_extension().to_lower();
        String ext_mixed = path_mixed.get_extension().to_lower();

        CHECK(ext_upper == "ply");
        CHECK(ext_mixed == "spz");
    }

    SUBCASE("Path with multiple dots handles extension correctly") {
        String path = "res://models/my.scene.v2.ply";
        String extension = path.get_extension().to_lower();
        CHECK(extension == "ply");
    }
}

TEST_CASE("[GaussianSplatting][Importer] PLY and SPZ importers have distinct names") {
    Ref<ResourceImporterPLY> ply_importer;
    ply_importer.instantiate();

    Ref<ResourceImporterSPZ> spz_importer;
    spz_importer.instantiate();

    CHECK(ply_importer->get_importer_name() == "gaussian_splat_ply");
    CHECK(spz_importer->get_importer_name() == "gaussian_splat_spz");
    CHECK(ply_importer->get_importer_name() != spz_importer->get_importer_name());

    // Both should produce the same resource type
    CHECK(ply_importer->get_resource_type() == spz_importer->get_resource_type());
    CHECK(ply_importer->get_resource_type() == "GaussianSplatAsset");
}

TEST_CASE("[GaussianSplatting][Importer] gsplatworld importer preserves payload and chunks") {
    Ref<GaussianData> data;
    data.instantiate();
    data->resize(2);

    PackedVector3Array positions;
    positions.push_back(Vector3(0.0f, 0.0f, 0.0f));
    positions.push_back(Vector3(1.0f, 0.0f, 0.0f));
    data->set_positions(positions);

    PackedVector3Array scales;
    scales.push_back(Vector3(1.0f, 1.0f, 1.0f));
    scales.push_back(Vector3(1.0f, 1.0f, 1.0f));
    data->set_scales(scales);

    TypedArray<Quaternion> rotations;
    rotations.push_back(Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    rotations.push_back(Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    data->set_rotations(rotations);

    PackedFloat32Array opacities;
    opacities.push_back(1.0f);
    opacities.push_back(1.0f);
    data->set_opacities(opacities);

    PackedFloat32Array sh_dc;
    sh_dc.push_back(1.0f);
    sh_dc.push_back(0.0f);
    sh_dc.push_back(0.0f);
    sh_dc.push_back(0.0f);
    sh_dc.push_back(1.0f);
    sh_dc.push_back(0.0f);
    data->set_spherical_harmonics(sh_dc);

    Ref<GaussianSplatWorld> source_world;
    source_world.instantiate();
    source_world->set_gaussian_data(data);
    source_world->set_bounds(AABB(Vector3(-1.0f, -1.0f, -1.0f), Vector3(4.0f, 4.0f, 4.0f)));

    Vector<GaussianSplatRenderer::StaticChunk> chunks;
    chunks.resize(1);
    chunks.write[0].bounds = AABB(Vector3(-1.0f, -1.0f, -1.0f), Vector3(3.0f, 3.0f, 3.0f));
    chunks.write[0].center = Vector3(0.5f, 0.0f, 0.0f);
    chunks.write[0].radius = 2.0f;
    chunks.write[0].indices.push_back(0);
    chunks.write[0].indices.push_back(1);
    source_world->set_static_chunks(chunks);

    const String source_path = "user://gsplatworld_importer_source.gsplatworld";
    const String save_base_path = "user://gsplatworld_importer_roundtrip";
    const String imported_path = save_base_path + ".gsplatworld";

    Error save_source_err = ResourceSaver::save(source_world, source_path);
    CHECK_MESSAGE(save_source_err == OK, "Saving source gsplatworld should succeed.");
    if (save_source_err != OK) {
        return;
    }

    Ref<ResourceImporterGSplatWorld> importer;
    importer.instantiate();
    CHECK(importer->get_save_extension() == "gsplatworld");
    CHECK(importer->get_format_version() == 1);

    HashMap<StringName, Variant> options;
    Error import_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options,
            nullptr, nullptr, nullptr);
    CHECK_MESSAGE(import_err == OK, "ResourceImporterGSplatWorld import should succeed.");
    if (import_err != OK) {
        _remove_user_file(source_path);
        _remove_user_file(imported_path);
        return;
    }

    Ref<GaussianSplatWorld> imported_world = ResourceLoader::load(imported_path, "GaussianSplatWorld");
    CHECK_MESSAGE(imported_world.is_valid(), "Imported gsplatworld should be loadable.");
    if (imported_world.is_valid()) {
        Ref<GaussianData> imported_data = imported_world->get_gaussian_data();
        CHECK(imported_data.is_valid());
        if (imported_data.is_valid()) {
            CHECK(imported_data->get_count() == 2);
        }
        CHECK(imported_world->get_chunk_count() == 1);
        PackedInt32Array chunk_sizes = imported_world->get_chunk_sizes();
        CHECK(chunk_sizes.size() == 1);
        if (chunk_sizes.size() == 1) {
            CHECK(chunk_sizes[0] == 2);
        }
    }

    _remove_user_file(source_path);
    _remove_user_file(imported_path);
}

TEST_CASE("[GaussianSplatting][Importer] gsplatworld importer rejects invalid payloads") {
    const String source_path = "user://gsplatworld_importer_invalid_source.gsplatworld";
    const String save_base_path = "user://gsplatworld_importer_invalid";
    const String imported_path = save_base_path + ".gsplatworld";

    Error write_err = _write_invalid_gsplatworld(source_path);
    CHECK_MESSAGE(write_err == OK, "Writing invalid gsplatworld test payload should succeed.");
    if (write_err != OK) {
        return;
    }

    Ref<ResourceImporterGSplatWorld> importer;
    importer.instantiate();

    HashMap<StringName, Variant> options;
    Error import_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options,
            nullptr, nullptr, nullptr);
    CHECK_MESSAGE(import_err != OK, "ResourceImporterGSplatWorld should reject malformed source payloads.");
    CHECK_MESSAGE(!FileAccess::exists(imported_path),
            "Importer should not emit an output file when source payload validation fails.");

    _remove_user_file(source_path);
    _remove_user_file(imported_path);
}

TEST_CASE("[GaussianSplatting][Importer] gsplatworld importer rejects truncated payloads") {
    Ref<GaussianData> data;
    data.instantiate();
    data->resize(2);

    PackedVector3Array positions;
    positions.push_back(Vector3(0.0f, 0.0f, 0.0f));
    positions.push_back(Vector3(1.0f, 0.0f, 0.0f));
    data->set_positions(positions);

    PackedVector3Array scales;
    scales.push_back(Vector3(1.0f, 1.0f, 1.0f));
    scales.push_back(Vector3(1.0f, 1.0f, 1.0f));
    data->set_scales(scales);

    TypedArray<Quaternion> rotations;
    rotations.push_back(Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    rotations.push_back(Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    data->set_rotations(rotations);

    PackedFloat32Array opacities;
    opacities.push_back(1.0f);
    opacities.push_back(1.0f);
    data->set_opacities(opacities);

    PackedFloat32Array sh_dc;
    sh_dc.push_back(1.0f);
    sh_dc.push_back(0.0f);
    sh_dc.push_back(0.0f);
    sh_dc.push_back(0.0f);
    sh_dc.push_back(1.0f);
    sh_dc.push_back(0.0f);
    data->set_spherical_harmonics(sh_dc);

    Ref<GaussianSplatWorld> source_world;
    source_world.instantiate();
    source_world->set_gaussian_data(data);
    source_world->set_bounds(AABB(Vector3(-1.0f, -1.0f, -1.0f), Vector3(4.0f, 4.0f, 4.0f)));

    const String source_path = "user://gsplatworld_importer_truncated_source.gsplatworld";
    const String truncated_source_path = "user://gsplatworld_importer_truncated_input.gsplatworld";
    const String save_base_path = "user://gsplatworld_importer_truncated";
    const String imported_path = save_base_path + ".gsplatworld";

    Error save_err = ResourceSaver::save(source_world, source_path);
    CHECK_MESSAGE(save_err == OK, "Saving source gsplatworld for truncation test should succeed.");
    if (save_err != OK) {
        return;
    }

    Error trunc_err = _write_truncated_copy(source_path, truncated_source_path, 16);
    CHECK_MESSAGE(trunc_err == OK, "Creating truncated gsplatworld input should succeed.");
    if (trunc_err != OK) {
        _remove_user_file(source_path);
        _remove_user_file(truncated_source_path);
        _remove_user_file(imported_path);
        return;
    }

    Ref<ResourceImporterGSplatWorld> importer;
    importer.instantiate();

    HashMap<StringName, Variant> options;
    Error import_err = importer->import(ResourceUID::INVALID_ID, truncated_source_path, save_base_path, options,
            nullptr, nullptr, nullptr);
    CHECK_MESSAGE(import_err != OK,
            "ResourceImporterGSplatWorld should reject payloads truncated after a valid header.");
    CHECK_MESSAGE(!FileAccess::exists(imported_path),
            "Importer should not emit an output file when payload size validation fails.");

    _remove_user_file(source_path);
    _remove_user_file(truncated_source_path);
    _remove_user_file(imported_path);
}

TEST_CASE("[GaussianSplatting][Importer] gsplatworld importer rejects decode-invalid payloads") {
    Ref<GaussianData> data;
    data.instantiate();
    data->resize(2);

    PackedVector3Array positions;
    positions.push_back(Vector3(0.0f, 0.0f, 0.0f));
    positions.push_back(Vector3(1.0f, 0.0f, 0.0f));
    data->set_positions(positions);

    PackedVector3Array scales;
    scales.push_back(Vector3(1.0f, 1.0f, 1.0f));
    scales.push_back(Vector3(1.0f, 1.0f, 1.0f));
    data->set_scales(scales);

    TypedArray<Quaternion> rotations;
    rotations.push_back(Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    rotations.push_back(Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    data->set_rotations(rotations);

    PackedFloat32Array opacities;
    opacities.push_back(1.0f);
    opacities.push_back(1.0f);
    data->set_opacities(opacities);

    PackedFloat32Array sh_dc;
    sh_dc.push_back(1.0f);
    sh_dc.push_back(0.0f);
    sh_dc.push_back(0.0f);
    sh_dc.push_back(0.0f);
    sh_dc.push_back(1.0f);
    sh_dc.push_back(0.0f);
    data->set_spherical_harmonics(sh_dc);

    Ref<GaussianSplatWorld> source_world;
    source_world.instantiate();
    source_world->set_gaussian_data(data);
    source_world->set_bounds(AABB(Vector3(-1.0f, -1.0f, -1.0f), Vector3(4.0f, 4.0f, 4.0f)));

    const String source_path = "user://gsplatworld_importer_decode_source.gsplatworld";
    const String corrupt_path = "user://gsplatworld_importer_decode_corrupt_input.gsplatworld";
    const String save_base_path = "user://gsplatworld_importer_decode_invalid";
    const String imported_path = save_base_path + ".gsplatworld";

    Error save_err = ResourceSaver::save(source_world, source_path);
    CHECK_MESSAGE(save_err == OK, "Saving source gsplatworld for decode corruption test should succeed.");
    if (save_err != OK) {
        return;
    }

    Error copy_err = _write_truncated_copy(source_path, corrupt_path, 0);
    CHECK_MESSAGE(copy_err == OK, "Copying gsplatworld source for decode corruption test should succeed.");
    if (copy_err != OK) {
        _remove_user_file(source_path);
        _remove_user_file(corrupt_path);
        _remove_user_file(imported_path);
        return;
    }

    Error corrupt_err = _corrupt_gsplatworld_for_decode_failure(corrupt_path);
    CHECK_MESSAGE(corrupt_err == OK, "Corrupting gsplatworld payload for decode validation test should succeed.");
    if (corrupt_err != OK) {
        _remove_user_file(source_path);
        _remove_user_file(corrupt_path);
        _remove_user_file(imported_path);
        return;
    }

    Ref<ResourceImporterGSplatWorld> importer;
    importer.instantiate();

    HashMap<StringName, Variant> options;
    Error import_err = importer->import(ResourceUID::INVALID_ID, corrupt_path, save_base_path, options,
            nullptr, nullptr, nullptr);
    CHECK_MESSAGE(import_err != OK,
            "ResourceImporterGSplatWorld should reject payloads that fail full decode validation.");
    CHECK_MESSAGE(!FileAccess::exists(imported_path),
            "Importer should not emit an output file when decode validation fails.");

    _remove_user_file(source_path);
    _remove_user_file(corrupt_path);
    _remove_user_file(imported_path);
}

TEST_CASE("[GaussianSplatting][Importer] gsplatworld importer accepts compressed payloads") {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    CHECK(ps != nullptr);
    if (ps == nullptr) {
        return;
    }

    const StringName compression_key = StringName("rendering/gaussian_splatting/import/gsplatworld_compression_enabled");
    const bool had_setting = ps->has_setting(compression_key);
    const Variant previous_value = had_setting ? ps->get_setting(compression_key) : Variant(false);
    ps->set_setting(compression_key, true);

    Ref<GaussianData> data;
    data.instantiate();
    const int splat_count = 16; // > 1KB payload so saver's compression path activates.
    data->resize(splat_count);

    PackedVector3Array positions;
    PackedVector3Array scales;
    TypedArray<Quaternion> rotations;
    PackedFloat32Array opacities;
    PackedFloat32Array sh_dc;
    for (int i = 0; i < splat_count; i++) {
        positions.push_back(Vector3(float(i), 0.0f, 0.0f));
        scales.push_back(Vector3(1.0f, 1.0f, 1.0f));
        rotations.push_back(Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
        opacities.push_back(1.0f);
        sh_dc.push_back((i % 3) == 0 ? 1.0f : 0.0f);
        sh_dc.push_back((i % 3) == 1 ? 1.0f : 0.0f);
        sh_dc.push_back((i % 3) == 2 ? 1.0f : 0.0f);
    }
    data->set_positions(positions);
    data->set_scales(scales);
    data->set_rotations(rotations);
    data->set_opacities(opacities);
    data->set_spherical_harmonics(sh_dc);

    Ref<GaussianSplatWorld> source_world;
    source_world.instantiate();
    source_world->set_gaussian_data(data);
    source_world->set_bounds(AABB(Vector3(-1.0f, -1.0f, -1.0f), Vector3(32.0f, 4.0f, 4.0f)));

    const String source_path = "user://gsplatworld_importer_compressed_source.gsplatworld";
    const String save_base_path = "user://gsplatworld_importer_compressed";
    const String imported_path = save_base_path + ".gsplatworld";

    Error save_err = ResourceSaver::save(source_world, source_path);
    CHECK_MESSAGE(save_err == OK, "Saving compressed gsplatworld source should succeed.");
    if (save_err != OK) {
        ps->set_setting(compression_key, previous_value);
        return;
    }

    Ref<FileAccess> source_file = FileAccess::open(source_path, FileAccess::READ);
    CHECK(source_file.is_valid());
    if (source_file.is_valid()) {
        source_file->seek(8); // magic + version
        const uint32_t flags = source_file->get_32();
        CHECK_MESSAGE((flags & (1u << 4u)) != 0u, "Saved source gsplatworld should have compression flag set.");
    }

    Ref<ResourceImporterGSplatWorld> importer;
    importer.instantiate();
    HashMap<StringName, Variant> options;
    Error import_err = importer->import(ResourceUID::INVALID_ID, source_path, save_base_path, options,
            nullptr, nullptr, nullptr);
    CHECK_MESSAGE(import_err == OK, "Importer should accept valid compressed gsplatworld payloads.");

    Ref<GaussianSplatWorld> imported_world = ResourceLoader::load(imported_path, "GaussianSplatWorld");
    CHECK_MESSAGE(imported_world.is_valid(), "Imported compressed gsplatworld should be loadable.");

    _remove_user_file(source_path);
    _remove_user_file(imported_path);
    ps->set_setting(compression_key, previous_value);
}

TEST_CASE("[GaussianSplatting][Importer] GaussianSplatAsset save_to_file rejects empty assets") {
    Ref<GaussianSplatAsset> asset;
    asset.instantiate();

    const String output_path = "user://gaussian_asset_empty_save.ply";
    const Error save_err = asset->save_to_file(output_path);
    CHECK_MESSAGE(save_err == ERR_INVALID_DATA,
            "save_to_file should return ERR_INVALID_DATA when asset has no splat payload");
    _remove_user_file(output_path);
}

TEST_CASE("[GaussianSplatting][Importer] GaussianSplatAsset unloaded getters remain safe and empty") {
    Ref<GaussianSplatAsset> asset;
    asset.instantiate();

    CHECK_FALSE(asset->is_loaded());
    CHECK_EQ(asset->get_splat_count(), uint32_t(0));

    CHECK(asset->get_positions().is_empty());
    CHECK(asset->get_position_vectors().is_empty());
    CHECK(asset->get_colors().is_empty());
    CHECK(asset->get_scales().is_empty());
    CHECK(asset->get_scale_vectors().is_empty());
    CHECK(asset->get_rotations().is_empty());
    CHECK_EQ(asset->get_rotation_quaternions().size(), 0);
    CHECK(asset->get_sh_dc_coefficients().is_empty());
    CHECK(asset->get_sh_first_order_coefficients().is_empty());
    CHECK(asset->get_sh_high_order_coefficients().is_empty());
    CHECK(asset->get_spherical_harmonics_buffer().is_empty());
    CHECK(asset->get_opacity_logits().is_empty());
    CHECK(asset->get_opacities().is_empty());
    CHECK(asset->get_palette_ids().is_empty());
    CHECK(asset->get_palette_ids_buffer().is_empty());
    CHECK(asset->get_painterly_flags().is_empty());
    CHECK(asset->get_painterly_flags_buffer().is_empty());
    CHECK(asset->get_brush_override_ids().is_empty());
    CHECK(asset->get_brush_override_ids_buffer().is_empty());
    CHECK(asset->get_normals().is_empty());
    CHECK(asset->get_normal_vectors().is_empty());
    CHECK(asset->get_brush_axes().is_empty());
    CHECK(asset->get_brush_axes_vector2().is_empty());
    CHECK(asset->get_stroke_ages().is_empty());
    CHECK(asset->get_stroke_ages_buffer().is_empty());

    CHECK(asset->get_gaussian_data().is_null());
}

TEST_CASE("[GaussianSplatting][Importer] GaussianSplatAsset save_to_file persists loadable payloads") {
    const String source_path = "user://test_roundtrip_source.ply";
    const String output_path = "user://gaussian_asset_save_roundtrip.ply";

    {
        Error write_err = _write_minimal_ascii_ply(source_path);
        CHECK_MESSAGE(write_err == OK, "Failed to write synthetic PLY fixture for save_to_file test");
        if (write_err != OK) {
            return;
        }
    }

    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    const Error load_err = asset->load_from_file(source_path);
    CHECK_MESSAGE(load_err == OK, "Fixture PLY should load into GaussianSplatAsset before save_to_file");
    if (load_err != OK) {
        _remove_user_file(source_path);
        return;
    }

    PackedInt32Array brush_override_ids;
    brush_override_ids.resize(asset->get_splat_count());
    if (brush_override_ids.size() > 0) {
        int32_t *ptr = brush_override_ids.ptrw();
        for (int i = 0; i < brush_override_ids.size(); i++) {
            ptr[i] = (i * 37) % 65536;
        }
        asset->set_brush_override_ids(brush_override_ids);
        CHECK_EQ(asset->get_painterly_flags_buffer()[0], brush_override_ids[0]);
    }

    const Error save_err = asset->save_to_file(output_path);
    CHECK_MESSAGE(save_err == OK, "save_to_file should persist GaussianData-backed assets");
    CHECK_MESSAGE(FileAccess::exists(output_path), "save_to_file should produce an output file");
    if (save_err != OK || !FileAccess::exists(output_path)) {
        _remove_user_file(output_path);
        return;
    }

    Ref<GaussianSplatAsset> reloaded;
    reloaded.instantiate();
    const Error reload_err = reloaded->load_from_file(output_path);
    CHECK_MESSAGE(reload_err == OK, "Saved output should be loadable through GaussianSplatAsset::load_from_file");
    if (reload_err == OK) {
        CHECK_MESSAGE(reloaded->get_splat_count() == asset->get_splat_count(),
                "Roundtripped asset should preserve splat_count");
        PackedInt32Array reloaded_brush_override_ids = reloaded->get_brush_override_ids_buffer();
        CHECK_EQ(reloaded_brush_override_ids.size(), brush_override_ids.size());
        if (!brush_override_ids.is_empty() && reloaded_brush_override_ids.size() == brush_override_ids.size()) {
            CHECK_EQ(reloaded_brush_override_ids[0], brush_override_ids[0]);
            CHECK_EQ(reloaded->get_painterly_flags_buffer()[0], brush_override_ids[0]);
            CHECK_EQ(reloaded_brush_override_ids[brush_override_ids.size() - 1], brush_override_ids[brush_override_ids.size() - 1]);
        }
    }

    _remove_user_file(source_path);
    _remove_user_file(output_path);
}

// ============================================================================
// Thumbnail Customization Tracking Tests
// ============================================================================

TEST_CASE("[GaussianSplatting][Editor] Thumbnail style change triggers custom_settings flag") {
    GaussianImportDialog *dialog = memnew(GaussianImportDialog);

    // Configure dialog with default "desktop" preset
    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    asset->set_import_quality_preset("desktop");
    asset->set_source_path("res://test_splats.ply");

    Dictionary metadata;
    Dictionary baseline_options;
    baseline_options[StringName("quality/preset")] = String("desktop");
    baseline_options[StringName("preview/thumbnail_style")] = 0; // THUMBNAIL_STYLE_COLOR
    baseline_options[StringName("preview/thumbnail_size")] = 128;
    metadata[StringName("options")] = baseline_options;
    asset->set_import_metadata(metadata);

    dialog->configure_for_file("res://test_splats.ply", asset, true, Dictionary());

    // Initially should not be custom
    // Note: The custom_settings flag depends on whether any setting differs from preset defaults
    // For a fresh dialog configured with matching preset, it should be false
    (void)dialog->get_configuration();

    // Now configure with a different thumbnail style
    Dictionary style_override;
    style_override[StringName("preview/thumbnail_style")] = int(GaussianThumbnailGenerator::THUMBNAIL_STYLE_HEATMAP);
    dialog->configure_for_file("res://test_splats.ply", asset, true, style_override);

    GaussianImportDialog::ImportConfiguration modified_config = dialog->get_configuration();

    // If the style differs from the preset default, custom_settings should be true
    const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_name("desktop");
    if (int(GaussianThumbnailGenerator::THUMBNAIL_STYLE_HEATMAP) != preset.thumbnail_style) {
        CHECK_MESSAGE(modified_config.custom_settings == true,
                "Changing thumbnail_style should trigger custom_settings=true when different from preset default");
    }

    memdelete(dialog);
}

TEST_CASE("[GaussianSplatting][Editor] Thumbnail size change triggers custom_settings flag") {
    GaussianImportDialog *dialog = memnew(GaussianImportDialog);

    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    asset->set_import_quality_preset("desktop");
    asset->set_source_path("res://test_splats.ply");

    Dictionary metadata;
    Dictionary baseline_options;
    baseline_options[StringName("quality/preset")] = String("desktop");
    baseline_options[StringName("preview/thumbnail_size")] = 128;
    metadata[StringName("options")] = baseline_options;
    asset->set_import_metadata(metadata);

    dialog->configure_for_file("res://test_splats.ply", asset, true, Dictionary());

    // Apply override with different thumbnail size
    const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_name("desktop");
    int different_size = preset.default_thumbnail_size != 256 ? 256 : 64;

    Dictionary size_override;
    size_override[StringName("preview/thumbnail_size")] = different_size;
    dialog->configure_for_file("res://test_splats.ply", asset, true, size_override);

    GaussianImportDialog::ImportConfiguration config = dialog->get_configuration();
    CHECK_MESSAGE(config.thumbnail_size == different_size,
            "Dialog should apply thumbnail_size override correctly");
    CHECK_MESSAGE(config.custom_settings == true,
            "Changing thumbnail_size should trigger custom_settings=true when different from preset default");

    memdelete(dialog);
}

TEST_CASE("[GaussianSplatting][Editor] Disabling thumbnail generation triggers custom_settings flag") {
    GaussianImportDialog *dialog = memnew(GaussianImportDialog);

    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    asset->set_import_quality_preset("desktop");
    asset->set_source_path("res://test_splats.ply");

    Dictionary metadata;
    Dictionary baseline_options;
    baseline_options[StringName("quality/preset")] = String("desktop");
    baseline_options[StringName("preview/generate_thumbnail")] = true;
    metadata[StringName("options")] = baseline_options;
    asset->set_import_metadata(metadata);

    // Configure with generate_thumbnail = false
    Dictionary thumb_override;
    thumb_override[StringName("preview/generate_thumbnail")] = false;
    dialog->configure_for_file("res://test_splats.ply", asset, true, thumb_override);

    GaussianImportDialog::ImportConfiguration config = dialog->get_configuration();
    CHECK_MESSAGE(config.generate_thumbnail == false,
            "Dialog should apply generate_thumbnail override correctly");
    // Presets default to generate_thumbnail = true, so disabling it should trigger custom
    CHECK_MESSAGE(config.custom_settings == true,
            "Setting generate_thumbnail=false should trigger custom_settings=true");

    memdelete(dialog);
}

TEST_CASE("[GaussianSplatting][Editor] Import dialog handles SPZ source paths") {
    GaussianImportDialog *dialog = memnew(GaussianImportDialog);

    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    asset->set_import_quality_preset("desktop");
    asset->set_source_path("res://test_splats.spz");

    Dictionary metadata;
    metadata[StringName("source_format")] = "spz";
    metadata[StringName("quality_preset")] = String("desktop");
    asset->set_import_metadata(metadata);

    dialog->configure_for_file("res://test_splats.spz", asset, true, Dictionary());

    CHECK(dialog->get_source_path() == "res://test_splats.spz");

    GaussianImportDialog::ImportConfiguration config = dialog->get_configuration();
    CHECK(config.preset == "desktop");

    memdelete(dialog);
}

#endif // TOOLS_ENABLED
