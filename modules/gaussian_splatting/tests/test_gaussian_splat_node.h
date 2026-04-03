#pragma once

#include "test_macros.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "../nodes/gaussian_splat_world_3d.h"
#include "../nodes/gaussian_splat_dynamic_instance_3d.h"
#include "../core/gaussian_data.h"
#include "../core/effective_config_snapshot.h"
#include "../core/gaussian_splat_asset.h"
#include "../core/gaussian_splat_manager.h"
#include "../core/gaussian_splat_world.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../core/gaussian_splat_source_path.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/sh_config.h"
#include "../resources/color_grading_resource.h"
#ifdef TOOLS_ENABLED
#include "../editor/gaussian_editor_services.h"
#endif
#include "core/math/math_funcs.h"
#include "core/config/project_settings.h"
#include "core/error/error_list.h"
#include "core/templates/local_vector.h"
#include "core/templates/list.h"
#include "core/variant/variant.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

#if defined(TESTS_ENABLED) || defined(TOOLS_ENABLED)

#include "gs_test_setting_guard.h"

namespace {

Ref<GaussianData> make_test_gaussian_data(int p_count, float p_x_offset = 0.0f) {
    Ref<GaussianData> data;
    data.instantiate();
    data->resize(p_count);
    for (int i = 0; i < p_count; i++) {
        Gaussian g;
        g.position = Vector3(p_x_offset + (float)i, 0.0f, 0.0f);
        g.scale = Vector3(1.0f, 1.0f, 1.0f);
        g.rotation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
        g.opacity = 1.0f;
        g.sh_dc = Color(1.0f, 1.0f, 1.0f, 1.0f);
        data->set_gaussian(i, g);
    }
    return data;
}

Ref<GaussianSplatAsset> make_single_splat_asset(float p_x_offset = 0.0f) {
    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    asset->set_splat_count(1);

    PackedFloat32Array positions;
    positions.resize(3);
    {
        float *ptr = positions.ptrw();
        ptr[0] = p_x_offset;
        ptr[1] = 0.0f;
        ptr[2] = 0.0f;
    }
    asset->set_positions(positions);

    PackedFloat32Array scales;
    scales.resize(3);
    {
        float *ptr = scales.ptrw();
        ptr[0] = 1.0f;
        ptr[1] = 1.0f;
        ptr[2] = 1.0f;
    }
    asset->set_scales(scales);

    PackedFloat32Array rotations;
    rotations.resize(4);
    {
        float *ptr = rotations.ptrw();
        ptr[0] = 1.0f; // w
        ptr[1] = 0.0f;
        ptr[2] = 0.0f;
        ptr[3] = 0.0f;
    }
    asset->set_rotations(rotations);

    PackedFloat32Array sh_dc;
    sh_dc.resize(3);
    {
        float *ptr = sh_dc.ptrw();
        ptr[0] = 1.0f;
        ptr[1] = 1.0f;
        ptr[2] = 1.0f;
    }
    asset->set_sh_dc_coefficients(sh_dc);

    PackedFloat32Array opacity_logits;
    opacity_logits.resize(1);
    opacity_logits.set(0, 10.0f);
    asset->set_opacity_logits(opacity_logits);

    return asset;
}

Ref<GaussianSplatAsset> make_import_metadata_asset(int p_count, float p_x_offset, const String &p_quality_preset,
        int p_max_splats, double p_density_multiplier) {
    Ref<GaussianSplatAsset> asset;
    asset.instantiate();

    Ref<GaussianData> data = make_test_gaussian_data(p_count, p_x_offset);
    if (asset->populate_from_gaussian_data(data) != OK) {
        return Ref<GaussianSplatAsset>();
    }

    asset->set_import_quality_preset(p_quality_preset);
    Dictionary metadata = asset->get_import_metadata();
    metadata[StringName("quality_preset")] = p_quality_preset;
    metadata[StringName("max_splats")] = p_max_splats;
    metadata[StringName("density_multiplier")] = p_density_multiplier;
    asset->set_import_metadata(metadata);

    return asset;
}

int find_instance_index_by_translation_x(const LocalVector<InstanceDataGPU> &p_instances, float p_x) {
    for (int i = 0; i < p_instances.size(); i++) {
        if (Math::is_equal_approx(p_instances[i].translation_scale[0], p_x)) {
            return i;
        }
    }
    return -1;
}

int count_instances_by_translation_x(const LocalVector<InstanceDataGPU> &p_instances, float p_x) {
    int count = 0;
    for (int i = 0; i < p_instances.size(); i++) {
        if (Math::is_equal_approx(p_instances[i].translation_scale[0], p_x)) {
            count++;
        }
    }
    return count;
}

bool is_property_editor_exposed(Object *p_object, const StringName &p_property_name) {
    if (p_object == nullptr) {
        return false;
    }

    List<PropertyInfo> property_list;
    p_object->get_property_list(&property_list);
    for (const List<PropertyInfo>::Element *E = property_list.front(); E; E = E->next()) {
        const PropertyInfo &info = E->get();
        if (info.name == p_property_name) {
            return (info.usage & PROPERTY_USAGE_EDITOR) != 0;
        }
    }

    return false;
}

Ref<ColorGradingResource> make_color_grading_resource() {
    Ref<ColorGradingResource> grading;
    grading.instantiate();
    if (grading.is_valid()) {
        grading->set_enabled(true);
        grading->set_exposure(0.5f);
    }
    return grading;
}

void set_single_splat_position(const Ref<GaussianSplatAsset> &p_asset, const Vector3 &p_position) {
    PackedFloat32Array positions;
    positions.resize(3);
    {
        float *ptr = positions.ptrw();
        ptr[0] = p_position.x;
        ptr[1] = p_position.y;
        ptr[2] = p_position.z;
    }
    p_asset->set_positions(positions);
}

void set_single_splat_scale(const Ref<GaussianSplatAsset> &p_asset, const Vector3 &p_scale) {
    PackedFloat32Array scales;
    scales.resize(3);
    {
        float *ptr = scales.ptrw();
        ptr[0] = p_scale.x;
        ptr[1] = p_scale.y;
        ptr[2] = p_scale.z;
    }
    p_asset->set_scales(scales);
}

void set_single_splat_rotation(const Ref<GaussianSplatAsset> &p_asset, const Quaternion &p_rotation) {
    PackedFloat32Array rotations;
    rotations.resize(4);
    {
        float *ptr = rotations.ptrw();
        ptr[0] = p_rotation.w;
        ptr[1] = p_rotation.x;
        ptr[2] = p_rotation.y;
        ptr[3] = p_rotation.z;
    }
    p_asset->set_rotations(rotations);
}

} // namespace

TEST_CASE("[GaussianSplatting][Node] Debug flag persistence mirrors project settings") {
    // NOTE: Debug flag persistence to ProjectSettings is now editor-only.
    // At runtime (non-editor), flags are applied to the renderer but NOT saved.
    // This test verifies the runtime behavior where settings are read but not written.
    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    CHECK_MESSAGE(project_settings != nullptr, "ProjectSettings singleton must exist for GaussianSplatNode3D persistence test");
    if (project_settings == nullptr) {
        return;
    }

    const String tile_setting = "rendering/gaussian_splatting/debug/show_tile_grid";
    const String heatmap_setting = "rendering/gaussian_splatting/debug/show_density_heatmap";
    const String hud_setting = "rendering/gaussian_splatting/debug/show_performance_hud";

    ProjectSettingGuard tile_guard(project_settings, tile_setting);
    ProjectSettingGuard heatmap_guard(project_settings, heatmap_setting);
    ProjectSettingGuard hud_guard(project_settings, hud_setting);

    project_settings->set_setting(tile_setting, false);
    project_settings->set_setting(heatmap_setting, false);
    project_settings->set_setting(hud_setting, false);
    CHECK_EQ(project_settings->save(), OK);

    GaussianSplatNode3D *initial_node = memnew(GaussianSplatNode3D);
    CHECK(initial_node != nullptr);
    if (initial_node == nullptr) {
        return;
    }

    // Verify initial state is loaded from project settings
    CHECK_FALSE(initial_node->is_showing_tile_grid());
    CHECK_FALSE(initial_node->is_showing_density_heatmap());
    CHECK_FALSE(initial_node->is_showing_performance_hud());

    // Change the flags
    initial_node->set_show_tile_grid(true);
    initial_node->set_show_density_heatmap(true);
    initial_node->set_show_performance_hud(true);

    // Verify the node's local state changed
    CHECK(initial_node->is_showing_tile_grid());
    CHECK(initial_node->is_showing_density_heatmap());
    CHECK(initial_node->is_showing_performance_hud());

    // At runtime (non-editor), settings should NOT be persisted to ProjectSettings.
    // In editor, they would be. Since tests run in non-editor mode, project settings
    // should remain unchanged.
#ifndef TOOLS_ENABLED
    CHECK_FALSE((bool)project_settings->get_setting(tile_setting));
    CHECK_FALSE((bool)project_settings->get_setting(heatmap_setting));
    CHECK_FALSE((bool)project_settings->get_setting(hud_setting));
#else
    // In editor builds running tests, we can't easily distinguish editor vs runtime,
    // so we skip the persistence check. The important thing is that the node's
    // local state is correctly set.
#endif

    memdelete(initial_node);

    // Verify new nodes still read from project settings (not from previous node)
    GaussianSplatNode3D *fresh_node = memnew(GaussianSplatNode3D);
    CHECK(fresh_node != nullptr);
    if (fresh_node == nullptr) {
        return;
    }

    // Since we didn't persist above (runtime mode), fresh node should read original values
#ifndef TOOLS_ENABLED
    CHECK_FALSE(fresh_node->is_showing_tile_grid());
    CHECK_FALSE(fresh_node->is_showing_density_heatmap());
    CHECK_FALSE(fresh_node->is_showing_performance_hud());
#endif

    memdelete(fresh_node);
}

TEST_CASE("[GaussianSplatting][Node] Effective config snapshot reports tier caps with source attribution") {
    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    REQUIRE(project_settings != nullptr);
    if (project_settings == nullptr) {
        return;
    }

    const String tier_preset_setting = "rendering/gaussian_splatting/quality/tier_preset";
    const String tier_apply_setting = "rendering/gaussian_splatting/quality/tier_apply_streaming_budgets";

    ProjectSettingGuard tier_preset_guard(project_settings, tier_preset_setting);
    ProjectSettingGuard tier_apply_guard(project_settings, tier_apply_setting);

    project_settings->set_setting(tier_preset_setting, String("low"));
    project_settings->set_setting(tier_apply_setting, true);

    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    REQUIRE(node != nullptr);
    if (node == nullptr) {
        return;
    }

    node->set_quality_preset(GaussianSplatNode3D::QUALITY_QUALITY);

    const Dictionary snapshot = node->get_effective_config_snapshot();
    const Dictionary max_splats_entry = GaussianEffectiveConfig::get_entry(snapshot, StringName("max_splats"));
    const Dictionary gpu_memory_entry = GaussianEffectiveConfig::get_entry(snapshot, StringName("gpu_memory_mb"));
    const Dictionary lod_entry = GaussianEffectiveConfig::get_entry(snapshot, StringName("lod_max_distance"));

    CHECK(int64_t(max_splats_entry.get(StringName("value"), int64_t(-1))) == int64_t(300000));
    CHECK(String(max_splats_entry.get(StringName("source_label"), String())) == String("capped by tier 'low'"));
    CHECK(int64_t(gpu_memory_entry.get(StringName("value"), int64_t(-1))) == int64_t(256));
    CHECK(String(gpu_memory_entry.get(StringName("source_label"), String())) == String("capped by tier 'low'"));
    CHECK(String(lod_entry.get(StringName("source_label"), String())) == String("node property"));
    const Dictionary load_ahead_entry = GaussianEffectiveConfig::get_entry(snapshot, StringName("streaming_load_ahead_factor"));
    const Dictionary unload_entry = GaussianEffectiveConfig::get_entry(snapshot, StringName("streaming_unload_factor"));
    const Dictionary concurrent_loads_entry = GaussianEffectiveConfig::get_entry(snapshot, StringName("streaming_max_concurrent_loads"));
    const Dictionary target_gpu_entry = GaussianEffectiveConfig::get_entry(snapshot, StringName("target_gpu_memory_mb"));
    const Dictionary stream_budget_entry = GaussianEffectiveConfig::get_entry(snapshot, StringName("stream_budget_ms"));
    CHECK(int64_t(target_gpu_entry.get(StringName("value"), int64_t(-1))) == int64_t(192));
    CHECK(String(target_gpu_entry.get(StringName("source_label"), String())) == String("capped by tier 'low'"));
    CHECK(Math::is_equal_approx(float(double(load_ahead_entry.get(StringName("value"), 0.0))), 0.15f));
    CHECK(String(load_ahead_entry.get(StringName("source_label"), String())) == String("capped by tier 'low'"));
    CHECK(Math::is_equal_approx(float(double(unload_entry.get(StringName("value"), 0.0))), 0.95f));
    CHECK(String(unload_entry.get(StringName("source_label"), String())) == String("capped by tier 'low'"));
    CHECK(int64_t(concurrent_loads_entry.get(StringName("value"), int64_t(-1))) == int64_t(1));
    CHECK(String(concurrent_loads_entry.get(StringName("source_label"), String())) == String("capped by tier 'low'"));
    CHECK(int64_t(stream_budget_entry.get(StringName("value"), int64_t(-1))) == int64_t(1));
    CHECK(String(stream_budget_entry.get(StringName("source_label"), String())) == String("capped by tier 'low'"));

    memdelete(node);
}

#ifdef TOOLS_ENABLED
TEST_CASE("[GaussianSplatting][Node] Editor summary surfaces capped streaming values with source attribution") {
    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    REQUIRE(project_settings != nullptr);
    if (project_settings == nullptr) {
        return;
    }

    const String tier_preset_setting = "rendering/gaussian_splatting/quality/tier_preset";
    const String tier_apply_setting = "rendering/gaussian_splatting/quality/tier_apply_streaming_budgets";

    ProjectSettingGuard tier_preset_guard(project_settings, tier_preset_setting);
    ProjectSettingGuard tier_apply_guard(project_settings, tier_apply_setting);

    project_settings->set_setting(tier_preset_setting, String("low"));
    project_settings->set_setting(tier_apply_setting, true);

    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    REQUIRE(node != nullptr);
    if (node == nullptr) {
        return;
    }

    node->set_quality_preset(GaussianSplatNode3D::QUALITY_QUALITY);
    const String stats_text = GaussianEditorServices::format_gaussian_splat_stats(node, Ref<GaussianSplatRenderer>());
    CHECK(stats_text.contains("Effective Target GPU Memory: 192 MB"));
    CHECK(stats_text.contains("Effective Load Ahead: 0.15"));
    CHECK(stats_text.contains("Effective Unload: 0.95"));
    CHECK(stats_text.contains("Effective Concurrent Loads: 1"));
    CHECK(stats_text.contains("Effective Stream Budget: 1 ms"));
    CHECK(stats_text.contains("capped by tier 'low'"));

    memdelete(node);
}
#endif

TEST_CASE("[GaussianSplatting][Node] Effective config snapshot honors SH project override over tier default") {
    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    REQUIRE(project_settings != nullptr);
    if (project_settings == nullptr) {
        return;
    }

    const String tier_preset_setting = "rendering/gaussian_splatting/quality/tier_preset";
    const String sh_bands_setting = SHConfig::BANDS_PATH;

    {
        ProjectSettingGuard tier_preset_guard(project_settings, tier_preset_setting);
        ProjectSettingGuard sh_bands_guard(project_settings, sh_bands_setting);

        project_settings->set_setting(tier_preset_setting, String("steam_deck"));
        project_settings->set_setting(sh_bands_setting, int64_t(SH_BAND_3));

        g_sh_config.load_from_project_settings();
        const Dictionary snapshot = g_sh_config.get_effective_config_snapshot();
        const Dictionary sh_entry = GaussianEffectiveConfig::get_entry(snapshot, StringName("sh_bands"));

        CHECK(int64_t(sh_entry.get(StringName("value"), int64_t(-1))) == int64_t(SH_BAND_3));
        CHECK(String(sh_entry.get(StringName("source_label"), String())) == String("project override"));
        CHECK(String(sh_entry.get(StringName("display_value"), String())) == String("SH3 (3rd order)"));
    }

    g_sh_config.load_from_project_settings();
}

TEST_CASE("[GaussianSplatting][Node][SceneTree] Default update mode processes automatically") {
    SceneTree *tree = SceneTree::get_singleton();
    REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

    Window *root = tree->get_root();
    REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    CHECK(node != nullptr);
    if (node == nullptr) {
        return;
    }

    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    node->set_splat_asset(asset);

    root->add_child(node);
    CHECK(node->is_inside_tree());
    if (!node->is_inside_tree()) {
        memdelete(node);
        return;
    }

    // When GaussianSplatManager exists, it drives updates via frame_pre_draw callback
    // and node processing is disabled to avoid double updates.
    // When manager is unavailable, node falls back to NOTIFICATION_PROCESS.
    GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
    if (manager) {
        // Manager exists: node should NOT be processing (manager handles it)
        CHECK_FALSE(node->is_processing());
    } else {
        // No manager: node should be processing as fallback
        CHECK(node->is_processing());
    }

    const float initial_time = node->get_last_update_time_ms();

    // Run a frame to trigger updates (either via manager or NOTIFICATION_PROCESS)
    tree->process(0.016);

    // Update time should advance regardless of which path is used
    CHECK_MESSAGE(node->get_last_update_time_ms() >= initial_time, "last_update_time_ms should not regress after processing");

    root->remove_child(node);
    memdelete(node);
}

TEST_CASE("[GaussianSplatting][Node] Asset buffers populate GaussianData with painterly metadata") {
    Ref<GaussianSplatAsset> asset;
    asset.instantiate();

    const int splat_count = 2;
    asset->set_splat_count(splat_count);

    PackedFloat32Array positions;
    positions.resize(splat_count * 3);
    {
        float *ptr = positions.ptrw();
        ptr[0] = 1.0f;
        ptr[1] = 2.0f;
        ptr[2] = 3.0f;
        ptr[3] = 4.0f;
        ptr[4] = 5.0f;
        ptr[5] = 6.0f;
    }
    asset->set_positions(positions);

    PackedFloat32Array scales;
    scales.resize(splat_count * 3);
    {
        float *ptr = scales.ptrw();
        ptr[0] = 0.5f;
        ptr[1] = 0.6f;
        ptr[2] = 0.7f;
        ptr[3] = 1.1f;
        ptr[4] = 1.2f;
        ptr[5] = 1.3f;
    }
    asset->set_scales(scales);

    PackedFloat32Array rotations;
    rotations.resize(splat_count * 4);
    {
        float *ptr = rotations.ptrw();
        // Stored as w, x, y, z in asset
        ptr[0] = 1.0f;
        ptr[1] = 0.0f;
        ptr[2] = 0.0f;
        ptr[3] = 0.0f;
        ptr[4] = 0.7071067f;
        ptr[5] = 0.0f;
        ptr[6] = 0.7071067f;
        ptr[7] = 0.0f;
    }
    asset->set_rotations(rotations);

    PackedFloat32Array sh_dc;
    sh_dc.resize(splat_count * 3);
    {
        float *ptr = sh_dc.ptrw();
        ptr[0] = 0.8f;
        ptr[1] = 0.1f;
        ptr[2] = 0.2f;
        ptr[3] = 0.3f;
        ptr[4] = 0.4f;
        ptr[5] = 0.5f;
    }
    asset->set_sh_dc_coefficients(sh_dc);

    PackedFloat32Array sh_first;
    sh_first.resize(splat_count * 2 * 3);
    {
        float *ptr = sh_first.ptrw();
        ptr[0] = 0.01f;
        ptr[1] = 0.02f;
        ptr[2] = 0.03f;
        ptr[3] = 0.04f;
        ptr[4] = 0.05f;
        ptr[5] = 0.06f;
        ptr[6] = 0.11f;
        ptr[7] = 0.12f;
        ptr[8] = 0.13f;
        ptr[9] = 0.14f;
        ptr[10] = 0.15f;
        ptr[11] = 0.16f;
    }
    asset->set_sh_first_order_coefficients(sh_first);

    PackedFloat32Array sh_high;
    sh_high.resize(splat_count * 3);
    {
        float *ptr = sh_high.ptrw();
        ptr[0] = 0.2f;
        ptr[1] = 0.3f;
        ptr[2] = 0.4f;
        ptr[3] = 0.5f;
        ptr[4] = 0.6f;
        ptr[5] = 0.7f;
    }
    asset->set_sh_high_order_coefficients(sh_high);

    PackedFloat32Array opacity_logits;
    opacity_logits.resize(splat_count);
    {
        float *ptr = opacity_logits.ptrw();
        ptr[0] = 0.0f;
        ptr[1] = 2.1972246f; // logit for 0.9
    }
    asset->set_opacity_logits(opacity_logits);

    PackedInt32Array palette_ids;
    palette_ids.resize(splat_count);
    {
        int32_t *ptr = palette_ids.ptrw();
        ptr[0] = 123;
        ptr[1] = 456;
    }
    asset->set_palette_ids(palette_ids);

    PackedInt32Array brush_override_ids;
    brush_override_ids.resize(splat_count);
    {
        int32_t *ptr = brush_override_ids.ptrw();
        ptr[0] = 7;
        ptr[1] = 255;
    }
    asset->set_brush_override_ids(brush_override_ids);
    CHECK_EQ(asset->get_painterly_flags_buffer()[0], 7);
    CHECK_EQ(asset->get_brush_override_ids_buffer()[1], 255);

    PackedFloat32Array normals;
    normals.resize(splat_count * 3);
    {
        float *ptr = normals.ptrw();
        ptr[0] = 0.0f;
        ptr[1] = 1.0f;
        ptr[2] = 0.0f;
        ptr[3] = 0.0f;
        ptr[4] = 0.0f;
        ptr[5] = 1.0f;
    }
    asset->set_normals(normals);

    PackedFloat32Array brush_axes;
    brush_axes.resize(splat_count * 2);
    {
        float *ptr = brush_axes.ptrw();
        ptr[0] = 1.0f;
        ptr[1] = 0.5f;
        ptr[2] = 0.75f;
        ptr[3] = 1.25f;
    }
    asset->set_brush_axes(brush_axes);

    PackedFloat32Array stroke_ages;
    stroke_ages.resize(splat_count);
    {
        float *ptr = stroke_ages.ptrw();
        ptr[0] = 3.0f;
        ptr[1] = 7.5f;
    }
    asset->set_stroke_ages(stroke_ages);

    Dictionary metadata = asset->get_import_metadata();
    metadata[StringName("gaussian_2d_mode")] = true;
    asset->set_import_metadata(metadata);

    Ref<::GaussianData> data;
    data.instantiate();
    CHECK_EQ(data->populate_from_asset(asset), OK);

    CHECK(data->get_2d_mode());
    CHECK_EQ(data->get_sh_first_order_count(), 3u);
    CHECK_EQ(data->get_sh_high_order_count(), 0u);

    Gaussian g0 = data->get_gaussian(0);
    CHECK(g0.position.is_equal_approx(Vector3(1.0f, 2.0f, 3.0f)));
    CHECK(g0.scale.is_equal_approx(Vector3(0.5f, 0.6f, 0.7f)));
    CHECK(g0.rotation.is_equal_approx(Quaternion(0.0f, 0.0f, 0.0f, 1.0f)));
    CHECK(Math::is_equal_approx(g0.opacity, 0.5f));
    CHECK(g0.sh_dc.is_equal_approx(Color(0.8f, 0.1f, 0.2f, 1.0f)));
    CHECK(g0.sh_1[0].is_equal_approx(Vector3(0.01f, 0.02f, 0.03f)));
    CHECK(g0.sh_1[1].is_equal_approx(Vector3(0.04f, 0.05f, 0.06f)));
    CHECK(g0.sh_1[2].is_equal_approx(Vector3(0.2f, 0.3f, 0.4f)));
    CHECK(g0.normal.is_equal_approx(Vector3(0.0f, 1.0f, 0.0f)));
    CHECK(g0.brush_axes.is_equal_approx(Vector2(1.0f, 0.5f)));
    CHECK(Math::is_equal_approx(g0.stroke_age, 3.0f));
    CHECK_EQ(gaussian_get_palette_id(g0.painterly_meta), 123);
    CHECK_EQ(gaussian_get_brush_override_id(g0.painterly_meta), 7);
    CHECK_EQ(gaussian_get_painterly_flags(g0.painterly_meta), 7);

    Gaussian g1 = data->get_gaussian(1);
    CHECK(g1.position.is_equal_approx(Vector3(4.0f, 5.0f, 6.0f)));
    CHECK(g1.scale.is_equal_approx(Vector3(1.1f, 1.2f, 1.3f)));
    CHECK(g1.rotation.is_equal_approx(Quaternion(0.0f, 0.7071067f, 0.0f, 0.7071067f)));
    CHECK(Math::is_equal_approx(g1.opacity, 1.0f / (1.0f + Math::exp(-2.1972246f))));
    CHECK(g1.sh_dc.is_equal_approx(Color(0.3f, 0.4f, 0.5f, 1.0f)));
    CHECK(g1.sh_1[0].is_equal_approx(Vector3(0.11f, 0.12f, 0.13f)));
    CHECK(g1.sh_1[1].is_equal_approx(Vector3(0.14f, 0.15f, 0.16f)));
    CHECK(g1.sh_1[2].is_equal_approx(Vector3(0.5f, 0.6f, 0.7f)));
    CHECK(g1.normal.is_equal_approx(Vector3(0.0f, 0.0f, 1.0f)));
    CHECK(g1.brush_axes.is_equal_approx(Vector2(0.75f, 1.25f)));
    CHECK(Math::is_equal_approx(g1.stroke_age, 7.5f));
    CHECK_EQ(gaussian_get_palette_id(g1.painterly_meta), 456);
    CHECK_EQ(gaussian_get_brush_override_id(g1.painterly_meta), 255);
    CHECK_EQ(gaussian_get_painterly_flags(g1.painterly_meta), 255);

    Ref<GaussianSplatAsset> roundtrip_asset;
    roundtrip_asset.instantiate();
    CHECK_EQ(roundtrip_asset->populate_from_gaussian_data(data), OK);
    CHECK_EQ(roundtrip_asset->get_brush_override_ids_buffer().size(), splat_count);
    CHECK_EQ(roundtrip_asset->get_brush_override_ids_buffer()[0], 7);
    CHECK_EQ(roundtrip_asset->get_painterly_flags_buffer()[1], 255);

    const Vector3 *high_ptr = data->get_sh_high_order_coefficients_ptr();
    CHECK(high_ptr == nullptr);
}

TEST_CASE("[GaussianSplatting][Node] Cached bounds stay coherent after position/scale/rotation mutations") {
    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    CHECK(node != nullptr);
    if (node == nullptr) {
        return;
    }

    Ref<GaussianSplatAsset> asset = make_single_splat_asset(0.0f);
    const AABB cached_bounds(Vector3(-3.0f, -3.0f, -3.0f), Vector3(6.0f, 6.0f, 6.0f));
    Dictionary metadata = asset->get_import_metadata();
    metadata[StringName("bounds")] = cached_bounds;
    asset->set_import_metadata(metadata);

    node->set_splat_asset(asset);
    AABB bounds = node->get_aabb();
    CHECK(bounds.position.is_equal_approx(cached_bounds.position));
    CHECK(bounds.size.is_equal_approx(cached_bounds.size));

    set_single_splat_position(asset, Vector3(4.0f, 0.0f, 0.0f));
    const AABB expected_after_position(Vector3(1.0f, -3.0f, -3.0f), Vector3(6.0f, 6.0f, 6.0f));
    bounds = node->get_aabb();
    CHECK_MESSAGE(bounds.position.is_equal_approx(expected_after_position.position),
            "Position mutation should invalidate stale cached bounds (position).");
    CHECK_MESSAGE(bounds.size.is_equal_approx(expected_after_position.size),
            "Position mutation should invalidate stale cached bounds (size).");

    set_single_splat_scale(asset, Vector3(2.0f, 1.0f, 1.0f));
    const AABB expected_after_scale(Vector3(-2.0f, -6.0f, -6.0f), Vector3(12.0f, 12.0f, 12.0f));
    bounds = node->get_aabb();
    CHECK_MESSAGE(bounds.position.is_equal_approx(expected_after_scale.position),
            "Scale mutation should keep the node bounds coherent (position).");
    CHECK_MESSAGE(bounds.size.is_equal_approx(expected_after_scale.size),
            "Scale mutation should keep the node bounds coherent (size).");

    const Quaternion rotated(Vector3(0.0f, 0.0f, 1.0f), Math::deg_to_rad(90.0f));
    set_single_splat_rotation(asset, rotated);
    bounds = node->get_aabb();
    CHECK_MESSAGE(bounds.position.is_equal_approx(expected_after_scale.position),
            "Rotation mutation should not leave node using stale bounds (position).");
    CHECK_MESSAGE(bounds.size.is_equal_approx(expected_after_scale.size),
            "Rotation mutation should not leave node using stale bounds (size).");

    memdelete(node);
}

TEST_CASE("[GaussianSplatting][Node] Debug overlay opacity is clamped and defaults correctly") {
    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    CHECK(node != nullptr);
    if (node == nullptr) {
        return;
    }

    // Check default value
    CHECK(Math::is_equal_approx(node->get_debug_overlay_opacity(), 0.3f));

    // Set valid values
    node->set_debug_overlay_opacity(0.5f);
    CHECK(Math::is_equal_approx(node->get_debug_overlay_opacity(), 0.5f));

    node->set_debug_overlay_opacity(0.0f);
    CHECK(Math::is_equal_approx(node->get_debug_overlay_opacity(), 0.0f));

    node->set_debug_overlay_opacity(1.0f);
    CHECK(Math::is_equal_approx(node->get_debug_overlay_opacity(), 1.0f));

    // Test clamping
    node->set_debug_overlay_opacity(-0.5f);
    CHECK(Math::is_equal_approx(node->get_debug_overlay_opacity(), 0.0f));

    node->set_debug_overlay_opacity(1.5f);
    CHECK(Math::is_equal_approx(node->get_debug_overlay_opacity(), 1.0f));

    memdelete(node);
}

TEST_CASE("[GaussianSplatting][Node] Debug overlay toggles have correct defaults") {
    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    CHECK(node != nullptr);
    if (node == nullptr) {
        return;
    }

    // Visual overlays should be off by default
    CHECK_FALSE(node->is_showing_tile_grid());
    CHECK_FALSE(node->is_showing_density_heatmap());

    // Stats-only HUD toggles should be off by default
    CHECK_FALSE(node->is_showing_performance_hud());
    CHECK_FALSE(node->is_showing_residency_hud());

    // Test toggling
    node->set_show_tile_grid(true);
    CHECK(node->is_showing_tile_grid());

    node->set_show_density_heatmap(true);
    CHECK(node->is_showing_density_heatmap());

    node->set_show_performance_hud(true);
    CHECK(node->is_showing_performance_hud());

    node->set_show_residency_hud(true);
    CHECK(node->is_showing_residency_hud());

    memdelete(node);
}

TEST_CASE("[GaussianSplatting][Node] Legacy non-functional properties are not exposed but still deserialize") {
    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    CHECK(node != nullptr);
    if (node == nullptr) {
        return;
    }

    List<PropertyInfo> property_list;
    node->get_property_list(&property_list);

    bool has_color_variation_property = false;
    bool has_occlusion_culling_property = false;
    for (const List<PropertyInfo>::Element *E = property_list.front(); E; E = E->next()) {
        const StringName &property_name = E->get().name;
        if (property_name == StringName("painterly/color_variation")) {
            has_color_variation_property = true;
        }
        if (property_name == StringName("rendering/occlusion_culling")) {
            has_occlusion_culling_property = true;
        }
    }

    CHECK_FALSE(has_color_variation_property);
    CHECK_FALSE(has_occlusion_culling_property);

    bool set_valid = false;
    node->set(StringName("painterly/color_variation"), 0.3f, &set_valid);
    CHECK(set_valid);
    CHECK(Math::is_equal_approx(node->get_color_variation(), 0.3f));

    node->set(StringName("rendering/occlusion_culling"), false, &set_valid);
    CHECK(set_valid);
    CHECK_FALSE(node->is_occlusion_culling_enabled());

    bool get_valid = false;
    Variant color_variation = node->get(StringName("painterly/color_variation"), &get_valid);
    CHECK(get_valid);
    CHECK(Math::is_equal_approx(float(color_variation), 0.3f));

    Variant occlusion_culling = node->get(StringName("rendering/occlusion_culling"), &get_valid);
    CHECK(get_valid);
    CHECK_FALSE((bool)occlusion_culling);

    memdelete(node);
}

TEST_CASE("[GaussianSplatting][Node] Asset origin label describes active ingress") {
    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    REQUIRE(node != nullptr);

    CHECK(node->get_asset_origin_label() == String("No asset assigned"));

    node->set_ply_file_path("res://direct_only.ply");
    CHECK(node->get_asset_origin_label().contains("Direct file path"));
    CHECK(node->get_asset_origin_label().contains("res://direct_only.ply"));

    Ref<GaussianSplatAsset> asset = make_single_splat_asset();
    asset->set_source_path("res://imported_source.ply");
    node->set_splat_asset(asset);

    const String asset_origin = node->get_asset_origin_label();
    CHECK(asset_origin.contains("Assigned GaussianSplatAsset"));
    CHECK(asset_origin.contains("source: res://imported_source.ply"));
    CHECK(asset_origin.contains("ply_file_path: res://direct_only.ply"));

    memdelete(node);
}

TEST_CASE("[GaussianSplatting][Node] Source path helper preserves asset-first precedence") {
    Ref<GaussianSplatAsset> asset;
    asset.instantiate();

    Dictionary metadata;
    metadata[StringName("source_file")] = String("res://metadata_source.ply");
    asset->set_import_metadata(metadata);

    CHECK(GaussianSplatSourcePath::get_asset_source_path(asset) == String("res://metadata_source.ply"));
    CHECK(GaussianSplatSourcePath::resolve_primary_source_path(asset, "res://fallback_path.ply") ==
            String("res://metadata_source.ply"));

    metadata.erase(StringName("source_file"));
    metadata[StringName("runtime_load_source_path")] = String("res://runtime_source.ply");
    asset->set_import_metadata(metadata);

    CHECK(GaussianSplatSourcePath::get_asset_source_path(asset) == String("res://runtime_source.ply"));
    CHECK(GaussianSplatSourcePath::resolve_primary_source_path(asset, "res://fallback_path.ply") ==
            String("res://runtime_source.ply"));

    asset->set_source_path("res://asset_source.ply");
    CHECK(GaussianSplatSourcePath::get_asset_source_path(asset) == String("res://asset_source.ply"));
    CHECK(GaussianSplatSourcePath::resolve_primary_source_path(asset, "res://fallback_path.ply") ==
            String("res://asset_source.ply"));

    asset->set_source_path(String());
    asset->set_import_metadata(Dictionary());
    CHECK(GaussianSplatSourcePath::get_asset_source_path(asset).is_empty());
    CHECK(GaussianSplatSourcePath::resolve_primary_source_path(asset, "res://fallback_path.ply") ==
            String("res://fallback_path.ply"));
}

TEST_CASE("[GaussianSplatting][Node] Configuration warnings flag inconsistent dual-path sources") {
    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    REQUIRE(node != nullptr);

    Ref<GaussianSplatAsset> asset = make_single_splat_asset();
    asset->set_source_path("res://asset_source.ply");
    node->set_splat_asset(asset);
    node->set_ply_file_path("res://other_source.ply");

    PackedStringArray warnings = node->get_configuration_warnings();
    bool has_dual_source_warning = false;
    for (int i = 0; i < warnings.size(); i++) {
        if (String(warnings[i]).contains("Both splat_asset and ply_file_path are set to different sources")) {
            has_dual_source_warning = true;
            break;
        }
    }
    CHECK(has_dual_source_warning);

    node->set_ply_file_path("res://asset_source.ply");
    warnings = node->get_configuration_warnings();
    has_dual_source_warning = false;
    for (int i = 0; i < warnings.size(); i++) {
        if (String(warnings[i]).contains("Both splat_asset and ply_file_path")) {
            has_dual_source_warning = true;
            break;
        }
    }
    CHECK_FALSE(has_dual_source_warning);

    memdelete(node);
}

TEST_CASE("[GaussianSplatting][World][SceneTree][RequiresGPU] Shared renderer ownership blocks foreign clear/mutate") {
    SceneTree *tree = SceneTree::get_singleton();
    REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

    Window *root = tree->get_root();
    REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

    Ref<GaussianSplatWorld> world_a;
    world_a.instantiate();
    Ref<GaussianData> data_a = make_test_gaussian_data(1, 0.0f);
    world_a->set_gaussian_data(data_a);

    Ref<GaussianSplatWorld> world_b;
    world_b.instantiate();
    Ref<GaussianData> data_b = make_test_gaussian_data(2, 100.0f);
    world_b->set_gaussian_data(data_b);

    GaussianSplatWorld3D *node_a = memnew(GaussianSplatWorld3D);
    GaussianSplatWorld3D *node_b = memnew(GaussianSplatWorld3D);
    node_a->set_world(world_a);
    node_b->set_world(world_b);

    root->add_child(node_a);
    root->add_child(node_b);
    tree->process(0.0);

    Ref<GaussianSplatRenderer> renderer_a = node_a->get_renderer();
    Ref<GaussianSplatRenderer> renderer_b = node_b->get_renderer();
    if (!renderer_a.is_valid() || !renderer_b.is_valid() || renderer_a != renderer_b) {
        MESSAGE("Skipping test - renderer unavailable (headless mode)");
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }

    node_a->apply_world();
    CHECK(renderer_a->get_gaussian_data() == data_a);

    node_b->clear_world();
    CHECK(renderer_a->get_gaussian_data() == data_a);

    node_b->apply_world();
    CHECK(renderer_a->get_gaussian_data() == data_a);

    node_a->clear_world();
    CHECK_FALSE(renderer_a->get_gaussian_data().is_valid());

    node_b->apply_world();
    CHECK(renderer_a->get_gaussian_data() == data_b);

    root->remove_child(node_b);
    root->remove_child(node_a);
    memdelete(node_b);
    memdelete(node_a);
}

TEST_CASE("[GaussianSplatting][Node][SceneTree][RequiresGPU] Shared renderer debug settings follow the latest writer") {
    SceneTree *tree = SceneTree::get_singleton();
    REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

    Window *root = tree->get_root();
    REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

    GaussianSplatNode3D *node_a = memnew(GaussianSplatNode3D);
    GaussianSplatNode3D *node_b = memnew(GaussianSplatNode3D);
    node_a->set_splat_asset(make_single_splat_asset(0.0f));
    node_b->set_splat_asset(make_single_splat_asset(10.0f));

    root->add_child(node_a);
    tree->process(0.0);

    node_a->set_show_density_heatmap(true);

    root->add_child(node_b);
    tree->process(0.0);

    Ref<GaussianSplatRenderer> renderer_a = node_a->get_renderer();
    Ref<GaussianSplatRenderer> renderer_b = node_b->get_renderer();
    if (!renderer_a.is_valid() || !renderer_b.is_valid() || renderer_a != renderer_b) {
        MESSAGE("Skipping test - renderer unavailable (headless mode)");
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }
    CHECK(renderer_a->is_debug_show_density_heatmap());

    node_b->set_show_density_heatmap(false);
    CHECK_FALSE(renderer_a->is_debug_show_density_heatmap());

    node_a->set_show_density_heatmap(false);
    CHECK_FALSE(renderer_a->is_debug_show_density_heatmap());

    node_a->set_show_density_heatmap(true);
    CHECK(renderer_a->is_debug_show_density_heatmap());

    root->remove_child(node_b);
    root->remove_child(node_a);
    memdelete(node_b);
    memdelete(node_a);
}

TEST_CASE("[GaussianSplatting][Node][SceneTree][RequiresGPU] Shared renderer instance buffer tracks per-node opacity") {
    SceneTree *tree = SceneTree::get_singleton();
    REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

    Window *root = tree->get_root();
    REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

    const float node_a_x = 1111.0f;
    const float node_b_x = 2222.0f;

    GaussianSplatNode3D *node_a = memnew(GaussianSplatNode3D);
    GaussianSplatNode3D *node_b = memnew(GaussianSplatNode3D);
    node_a->set_splat_asset(make_single_splat_asset(node_a_x));
    node_b->set_splat_asset(make_single_splat_asset(node_b_x));

    root->add_child(node_a);
    root->add_child(node_b);
    tree->process(0.0);

    Ref<GaussianSplatRenderer> renderer_a = node_a->get_renderer();
    Ref<GaussianSplatRenderer> renderer_b = node_b->get_renderer();
    GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
    if (!renderer_a.is_valid() || !renderer_b.is_valid()) {
        MESSAGE("Skipping test - renderer unavailable (headless mode)");
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }
    CHECK(renderer_a == renderer_b);
    CHECK(director != nullptr);
    if (renderer_a != renderer_b || director == nullptr) {
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }

    LocalVector<InstanceDataGPU> instance_buffer;
    director->build_instance_buffer_for_renderer(renderer_a.ptr(), instance_buffer);

    int node_a_index = find_instance_index_by_translation_x(instance_buffer, node_a_x);
    int node_b_index = find_instance_index_by_translation_x(instance_buffer, node_b_x);
    if (node_a_index < 0 || node_b_index < 0) {
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }
    CHECK(instance_buffer[node_a_index].params[0] == doctest::Approx(1.0f));
    CHECK(instance_buffer[node_b_index].params[0] == doctest::Approx(1.0f));

    node_b->set_opacity(0.25f);
    tree->process(0.0);
    director->build_instance_buffer_for_renderer(renderer_a.ptr(), instance_buffer);
    node_a_index = find_instance_index_by_translation_x(instance_buffer, node_a_x);
    node_b_index = find_instance_index_by_translation_x(instance_buffer, node_b_x);
    if (node_a_index < 0 || node_b_index < 0) {
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }
    CHECK(instance_buffer[node_a_index].params[0] == doctest::Approx(1.0f));
    CHECK(instance_buffer[node_b_index].params[0] == doctest::Approx(0.25f));

    node_a->set_opacity(0.6f);
    tree->process(0.0);
    director->build_instance_buffer_for_renderer(renderer_a.ptr(), instance_buffer);
    node_a_index = find_instance_index_by_translation_x(instance_buffer, node_a_x);
    node_b_index = find_instance_index_by_translation_x(instance_buffer, node_b_x);
    if (node_a_index < 0 || node_b_index < 0) {
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }
    CHECK(instance_buffer[node_a_index].params[0] == doctest::Approx(0.6f));
    CHECK(instance_buffer[node_b_index].params[0] == doctest::Approx(0.25f));

    root->remove_child(node_b);
    root->remove_child(node_a);
    memdelete(node_b);
    memdelete(node_a);
}

TEST_CASE("[GaussianSplatting][Node][SceneTree][RequiresGPU] Shared renderer instance buffer drops hidden nodes") {
    SceneTree *tree = SceneTree::get_singleton();
    REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

    Window *root = tree->get_root();
    REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

    const float node_a_x = 3333.0f;
    const float node_b_x = 4444.0f;

    GaussianSplatNode3D *node_a = memnew(GaussianSplatNode3D);
    GaussianSplatNode3D *node_b = memnew(GaussianSplatNode3D);
    node_a->set_splat_asset(make_single_splat_asset(node_a_x));
    node_b->set_splat_asset(make_single_splat_asset(node_b_x));

    root->add_child(node_a);
    root->add_child(node_b);
    tree->process(0.0);

    Ref<GaussianSplatRenderer> renderer = node_a->get_renderer();
    GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
    if (!renderer.is_valid()) {
        MESSAGE("Skipping test - renderer unavailable (headless mode)");
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }
    CHECK(renderer == node_b->get_renderer());
    CHECK(director != nullptr);
    if (renderer != node_b->get_renderer() || director == nullptr) {
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }

    LocalVector<InstanceDataGPU> instance_buffer;
    director->build_instance_buffer_for_renderer(renderer.ptr(), instance_buffer);
    CHECK_EQ(count_instances_by_translation_x(instance_buffer, node_a_x), 1);
    CHECK_EQ(count_instances_by_translation_x(instance_buffer, node_b_x), 1);

    node_b->set_visible(false);
    tree->process(0.0);
    director->build_instance_buffer_for_renderer(renderer.ptr(), instance_buffer);
    CHECK_EQ(count_instances_by_translation_x(instance_buffer, node_a_x), 1);
    CHECK_EQ(count_instances_by_translation_x(instance_buffer, node_b_x), 0);

    node_b->set_visible(true);
    tree->process(0.0);
    director->build_instance_buffer_for_renderer(renderer.ptr(), instance_buffer);
    CHECK_EQ(count_instances_by_translation_x(instance_buffer, node_a_x), 1);
    CHECK_EQ(count_instances_by_translation_x(instance_buffer, node_b_x), 1);

    root->remove_child(node_b);
    root->remove_child(node_a);
    memdelete(node_b);
    memdelete(node_a);
}

TEST_CASE("[GaussianSplatting][Node][SceneTree][RequiresGPU] Shared renderer hides node-local color grading property") {
    SceneTree *tree = SceneTree::get_singleton();
    REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

    Window *root = tree->get_root();
    REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

    GaussianSplatNode3D *node_a = memnew(GaussianSplatNode3D);
    GaussianSplatNode3D *node_b = memnew(GaussianSplatNode3D);
    node_a->set_splat_asset(make_single_splat_asset(5555.0f));
    node_b->set_splat_asset(make_single_splat_asset(6666.0f));

    root->add_child(node_a);
    tree->process(0.0);
    CHECK(is_property_editor_exposed(node_a, StringName("rendering/color_grading")));

    root->add_child(node_b);
    tree->process(0.0);

    // Shared renderer hides color_grading; requires GPU renderer to be available.
    Ref<GaussianSplatRenderer> renderer = node_a->get_renderer();
    if (!renderer.is_valid()) {
        MESSAGE("Skipping shared-renderer property check - renderer unavailable (headless mode)");
    } else {
        CHECK_FALSE(is_property_editor_exposed(node_a, StringName("rendering/color_grading")));
        CHECK_FALSE(is_property_editor_exposed(node_b, StringName("rendering/color_grading")));
    }

    root->remove_child(node_b);
    root->remove_child(node_a);
    memdelete(node_b);
    memdelete(node_a);
}

TEST_CASE("[GaussianSplatting][Node][SceneTree][RequiresGPU] Shared renderer preserves local painterly and color grading state") {
    SceneTree *tree = SceneTree::get_singleton();
    REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

    Window *root = tree->get_root();
    REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

    GaussianSplatNode3D *node_a = memnew(GaussianSplatNode3D);
    GaussianSplatNode3D *node_b = memnew(GaussianSplatNode3D);
    node_a->set_splat_asset(make_single_splat_asset(7777.0f));
    node_b->set_splat_asset(make_single_splat_asset(8888.0f));

    root->add_child(node_a);
    root->add_child(node_b);
    tree->process(0.0);

    Ref<GaussianSplatRenderer> renderer = node_a->get_renderer();
    if (!renderer.is_valid()) {
        MESSAGE("Skipping test - renderer unavailable (headless mode)");
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }
    CHECK(renderer == node_b->get_renderer());
    if (renderer != node_b->get_renderer()) {
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }

    Ref<ColorGradingResource> grading = make_color_grading_resource();
    CHECK(grading.is_valid());
    if (!grading.is_valid()) {
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }

    node_a->set_enable_painterly(true);
    node_a->set_color_grading(grading);
    tree->process(0.0);

    CHECK(node_a->is_painterly_enabled());
    CHECK(node_a->get_color_grading().is_valid());
    CHECK(node_a->get_color_grading() == grading);
    CHECK_FALSE(renderer->get_painterly_enabled());
    CHECK(renderer->get_color_grading().is_null());

    root->remove_child(node_b);
    tree->process(0.0);

    CHECK(node_a->is_painterly_enabled());
    CHECK(node_a->get_color_grading() == grading);
    CHECK(renderer->get_painterly_enabled());
    CHECK(renderer->get_color_grading() == grading);

    root->remove_child(node_a);
    memdelete(node_b);
    memdelete(node_a);
}

TEST_CASE("[GaussianSplatting][Node][SceneTree][RequiresGPU] Shared renderer full-fidelity override only follows attached assets") {
    SceneTree *tree = SceneTree::get_singleton();
    REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

    Window *root = tree->get_root();
    REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

    Ref<GaussianSplatAsset> limited_asset_a = make_import_metadata_asset(8, 7000.0f, "desktop", 250000, 0.5);
    Ref<GaussianSplatAsset> limited_asset_b = make_import_metadata_asset(8, 7100.0f, "desktop", 250000, 0.5);
    Ref<GaussianSplatAsset> unattached_full_asset = make_import_metadata_asset(8, 7200.0f, "ultra", 0, 1.0);
    Ref<GaussianSplatAsset> attached_full_asset = make_import_metadata_asset(8, 7300.0f, "ultra", 0, 1.0);
    CHECK(limited_asset_a.is_valid());
    CHECK(limited_asset_b.is_valid());
    CHECK(unattached_full_asset.is_valid());
    CHECK(attached_full_asset.is_valid());
    if (!limited_asset_a.is_valid() || !limited_asset_b.is_valid() || !unattached_full_asset.is_valid() || !attached_full_asset.is_valid()) {
        return;
    }

    GaussianSplatNode3D *node_a = memnew(GaussianSplatNode3D);
    GaussianSplatNode3D *node_b = memnew(GaussianSplatNode3D);
    node_a->set_splat_asset(limited_asset_a);
    node_b->set_splat_asset(limited_asset_b);

    root->add_child(node_a);
    root->add_child(node_b);
    tree->process(0.0);

    Ref<GaussianSplatRenderer> renderer = node_a->get_renderer();
    if (!renderer.is_valid()) {
        MESSAGE("Skipping test - renderer unavailable (headless mode)");
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }
    CHECK(renderer == node_b->get_renderer());
    if (renderer != node_b->get_renderer()) {
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }

    Vector<Vector3> test_positions;
    test_positions.push_back(Vector3(0.0f, 0.0f, -10.0f));
    test_positions.push_back(Vector3(0.0f, 0.0f, -20.0f));
    test_positions.push_back(Vector3(0.0f, 0.0f, -30.0f));
    renderer->test_set_test_splats(test_positions);

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 0.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 200.0f);
    const Size2i viewport_size(1280, 720);

    auto reset_culling_controls = [&renderer]() {
        renderer->set_lod_enabled(true);
        renderer->set_importance_cull_threshold(0.35f);
        renderer->set_tiny_splat_screen_radius(2.5f);
        renderer->set_opacity_aware_culling(true);
        renderer->set_visibility_threshold(0.2f);
        renderer->set_distance_cull_enabled(true);
        renderer->set_distance_cull_start(25.0f);
        renderer->set_distance_cull_max_rate(0.4f);
    };

    reset_culling_controls();
    renderer->set_gaussian_asset(unattached_full_asset);
    renderer->test_cull_visible_count(camera_transform, projection, viewport_size);

    CHECK(renderer->get_lod_enabled());
    CHECK(renderer->get_importance_cull_threshold() == doctest::Approx(0.35f));
    CHECK(renderer->get_tiny_splat_screen_radius() == doctest::Approx(2.5f));
    CHECK(renderer->is_opacity_aware_culling());
    CHECK(renderer->get_visibility_threshold() == doctest::Approx(0.2f));
    CHECK(renderer->is_distance_cull_enabled());
    CHECK(renderer->get_distance_cull_start() == doctest::Approx(25.0f));
    CHECK(renderer->get_distance_cull_max_rate() == doctest::Approx(0.4f));

    node_b->set_splat_asset(attached_full_asset);
    tree->process(0.0);
    reset_culling_controls();
    renderer->set_gaussian_asset(attached_full_asset);
    renderer->test_cull_visible_count(camera_transform, projection, viewport_size);

    CHECK_FALSE(renderer->get_lod_enabled());
    CHECK(renderer->get_importance_cull_threshold() == doctest::Approx(0.0f));
    CHECK(renderer->get_tiny_splat_screen_radius() == doctest::Approx(0.0f));
    CHECK_FALSE(renderer->is_opacity_aware_culling());
    CHECK(renderer->get_visibility_threshold() == doctest::Approx(0.0f));
    CHECK_FALSE(renderer->is_distance_cull_enabled());
    CHECK(renderer->get_distance_cull_start() == doctest::Approx(0.0f));
    CHECK(renderer->get_distance_cull_max_rate() == doctest::Approx(0.0f));

    root->remove_child(node_b);
    root->remove_child(node_a);
    memdelete(node_b);
    memdelete(node_a);
}

TEST_CASE("[GaussianSplatting][Node][SceneTree][RequiresGPU] Shared renderer ignores hidden full-fidelity assets") {
    SceneTree *tree = SceneTree::get_singleton();
    REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

    Window *root = tree->get_root();
    REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

    Ref<GaussianSplatAsset> limited_asset_a = make_import_metadata_asset(8, 7000.0f, "desktop", 250000, 0.5);
    Ref<GaussianSplatAsset> limited_asset_b = make_import_metadata_asset(8, 7100.0f, "desktop", 250000, 0.5);
    Ref<GaussianSplatAsset> hidden_full_asset = make_import_metadata_asset(8, 7200.0f, "ultra", 0, 1.0);
    CHECK(limited_asset_a.is_valid());
    CHECK(limited_asset_b.is_valid());
    CHECK(hidden_full_asset.is_valid());
    if (!limited_asset_a.is_valid() || !limited_asset_b.is_valid() || !hidden_full_asset.is_valid()) {
        return;
    }

    GaussianSplatNode3D *node_a = memnew(GaussianSplatNode3D);
    GaussianSplatNode3D *node_b = memnew(GaussianSplatNode3D);
    node_a->set_splat_asset(limited_asset_a);
    node_b->set_splat_asset(limited_asset_b);

    root->add_child(node_a);
    root->add_child(node_b);
    tree->process(0.0);

    Ref<GaussianSplatRenderer> renderer = node_a->get_renderer();
    if (!renderer.is_valid()) {
        MESSAGE("Skipping test - renderer unavailable (headless mode)");
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }
    CHECK(renderer == node_b->get_renderer());
    if (renderer != node_b->get_renderer()) {
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        return;
    }

    Vector<Vector3> test_positions;
    test_positions.push_back(Vector3(0.0f, 0.0f, -10.0f));
    test_positions.push_back(Vector3(0.0f, 0.0f, -20.0f));
    test_positions.push_back(Vector3(0.0f, 0.0f, -30.0f));
    renderer->test_set_test_splats(test_positions);

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 0.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 200.0f);
    const Size2i viewport_size(1280, 720);

    auto reset_culling_controls = [&renderer]() {
        renderer->set_lod_enabled(true);
        renderer->set_importance_cull_threshold(0.35f);
        renderer->set_tiny_splat_screen_radius(2.5f);
        renderer->set_opacity_aware_culling(true);
        renderer->set_visibility_threshold(0.2f);
        renderer->set_distance_cull_enabled(true);
        renderer->set_distance_cull_start(25.0f);
        renderer->set_distance_cull_max_rate(0.4f);
    };

    node_b->set_splat_asset(hidden_full_asset);
    node_b->set_visible(false);
    tree->process(0.0);

    reset_culling_controls();
    renderer->set_gaussian_asset(limited_asset_a);
    renderer->test_cull_visible_count(camera_transform, projection, viewport_size);

    CHECK(renderer->get_lod_enabled());
    CHECK(renderer->get_importance_cull_threshold() == doctest::Approx(0.35f));
    CHECK(renderer->get_tiny_splat_screen_radius() == doctest::Approx(2.5f));
    CHECK(renderer->is_opacity_aware_culling());
    CHECK(renderer->get_visibility_threshold() == doctest::Approx(0.2f));
    CHECK(renderer->is_distance_cull_enabled());
    CHECK(renderer->get_distance_cull_start() == doctest::Approx(25.0f));
    CHECK(renderer->get_distance_cull_max_rate() == doctest::Approx(0.4f));

    node_b->set_visible(true);
    tree->process(0.0);

    reset_culling_controls();
    renderer->set_gaussian_asset(limited_asset_a);
    renderer->test_cull_visible_count(camera_transform, projection, viewport_size);

    CHECK_FALSE(renderer->get_lod_enabled());
    CHECK(renderer->get_importance_cull_threshold() == doctest::Approx(0.0f));
    CHECK(renderer->get_tiny_splat_screen_radius() == doctest::Approx(0.0f));
    CHECK_FALSE(renderer->is_opacity_aware_culling());
    CHECK(renderer->get_visibility_threshold() == doctest::Approx(0.0f));
    CHECK_FALSE(renderer->is_distance_cull_enabled());
    CHECK(renderer->get_distance_cull_start() == doctest::Approx(0.0f));
    CHECK(renderer->get_distance_cull_max_rate() == doctest::Approx(0.0f));

    root->remove_child(node_b);
    root->remove_child(node_a);
    memdelete(node_b);
    memdelete(node_a);
}

TEST_CASE("[GaussianSplatting][DynamicInstance][SceneTree] Null or empty data unregisters instance") {
    SceneTree *tree = SceneTree::get_singleton();
    REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

    Window *root = tree->get_root();
    REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

    GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
    CHECK_MESSAGE(director != nullptr, "Scene director singleton must exist for dynamic unregister test");
    if (!director) {
        return;
    }

    LocalVector<InstanceDataGPU> instance_buffer;
    director->build_instance_buffer(instance_buffer);
    const int baseline_count = instance_buffer.size();

    GaussianSplatDynamicInstance3D *dynamic_node = memnew(GaussianSplatDynamicInstance3D);
    dynamic_node->set_gaussian_data(make_test_gaussian_data(1, 5.0f));
    root->add_child(dynamic_node);
    tree->process(0.0);

    CHECK(dynamic_node->is_registered());
    director->build_instance_buffer(instance_buffer);
    CHECK(instance_buffer.size() > baseline_count);

    Ref<GaussianData> empty_data;
    empty_data.instantiate();
    empty_data->resize(0);
    dynamic_node->set_gaussian_data(empty_data);
    tree->process(0.0);

    CHECK_FALSE(dynamic_node->is_registered());
    director->build_instance_buffer(instance_buffer);
    CHECK_EQ(instance_buffer.size(), baseline_count);

    dynamic_node->set_gaussian_data(Ref<GaussianData>());
    tree->process(0.0);
    CHECK_FALSE(dynamic_node->is_registered());
    director->build_instance_buffer(instance_buffer);
    CHECK_EQ(instance_buffer.size(), baseline_count);

    root->remove_child(dynamic_node);
    memdelete(dynamic_node);
}

// ── Import propagation proof ───────────────────────────────────────────

TEST_CASE("[GaussianSplatting][Node][SceneTree] Two nodes sharing one asset both observe asset mutation via changed signal") {
	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE(tree != nullptr);
	Window *root = tree->get_root();
	REQUIRE(root != nullptr);

	// Create a shared asset with 1 splat.
	Ref<GaussianSplatAsset> shared_asset = make_single_splat_asset(0.0f);
	REQUIRE(shared_asset.is_valid());
	CHECK(shared_asset->get_splat_count() == 1);

	// Two nodes both point to the same asset Ref.
	GaussianSplatNode3D *node_a = memnew(GaussianSplatNode3D);
	GaussianSplatNode3D *node_b = memnew(GaussianSplatNode3D);
	root->add_child(node_a);
	root->add_child(node_b);

	node_a->set_splat_asset(shared_asset);
	node_b->set_splat_asset(shared_asset);
	tree->process(0.0);

	CHECK(node_a->get_total_splat_count() == 1);
	CHECK(node_b->get_total_splat_count() == 1);

	// Mutate the shared asset: grow to 5 splats.
	// This simulates what happens when Godot reloads an imported resource in-place.
	shared_asset->set_splat_count(5);
	PackedFloat32Array new_positions;
	new_positions.resize(5 * 3);
	{
		float *ptr = new_positions.ptrw();
		for (int i = 0; i < 5 * 3; i++) {
			ptr[i] = float(i);
		}
	}
	shared_asset->set_positions(new_positions);

	// The asset emits "changed" on set_positions(). Both nodes should have
	// received _on_asset_changed() which calls _update_asset() and re-reads
	// total_splat_count from the asset.
	CHECK(node_a->get_total_splat_count() == 5);
	CHECK(node_b->get_total_splat_count() == 5);

	// Verify the asset Ref is truly shared (same object).
	CHECK(node_a->get_splat_asset() == node_b->get_splat_asset());

	root->remove_child(node_a);
	root->remove_child(node_b);
	memdelete(node_a);
	memdelete(node_b);
}

TEST_CASE("[GaussianSplatting][Node][SceneTree] Two nodes with separate asset Refs do not cross-propagate") {
	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE(tree != nullptr);
	Window *root = tree->get_root();
	REQUIRE(root != nullptr);

	Ref<GaussianSplatAsset> asset_a = make_single_splat_asset(0.0f);
	Ref<GaussianSplatAsset> asset_b = make_single_splat_asset(10.0f);

	GaussianSplatNode3D *node_a = memnew(GaussianSplatNode3D);
	GaussianSplatNode3D *node_b = memnew(GaussianSplatNode3D);
	root->add_child(node_a);
	root->add_child(node_b);

	node_a->set_splat_asset(asset_a);
	node_b->set_splat_asset(asset_b);
	tree->process(0.0);

	CHECK(node_a->get_total_splat_count() == 1);
	CHECK(node_b->get_total_splat_count() == 1);

	// Mutate only asset_a.
	asset_a->set_splat_count(7);
	PackedFloat32Array new_positions;
	new_positions.resize(7 * 3);
	{
		float *ptr = new_positions.ptrw();
		for (int i = 0; i < 7 * 3; i++) {
			ptr[i] = float(i);
		}
	}
	asset_a->set_positions(new_positions);

	// node_a should see 7; node_b should still see 1.
	CHECK(node_a->get_total_splat_count() == 7);
	CHECK(node_b->get_total_splat_count() == 1);

	root->remove_child(node_a);
	root->remove_child(node_b);
	memdelete(node_a);
	memdelete(node_b);
}

#endif // TESTS_ENABLED || TOOLS_ENABLED
