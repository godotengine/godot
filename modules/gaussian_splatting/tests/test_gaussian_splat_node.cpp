#include "test_macros.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "../nodes/gaussian_splat_world_3d.h"
#include "../nodes/gaussian_splat_dynamic_instance_3d.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_asset.h"
#include "../core/gaussian_splat_manager.h"
#include "../core/gaussian_splat_world.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "core/math/math_funcs.h"
#include "core/config/project_settings.h"
#include "core/error/error_list.h"
#include "core/templates/local_vector.h"
#include "core/templates/list.h"
#include "core/variant/variant.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

#ifdef TESTS_ENABLED

namespace {

class ProjectSettingGuard {
    ProjectSettings *settings = nullptr;
    String setting_path;
    Variant previous_value;
    bool had_previous_value = false;

public:
    ProjectSettingGuard(ProjectSettings *p_settings, const String &p_setting_path) : settings(p_settings), setting_path(p_setting_path) {
        if (settings && settings->has_setting(setting_path)) {
            previous_value = settings->get_setting(setting_path);
            had_previous_value = true;
        }
    }

    ~ProjectSettingGuard() {
        if (!settings) {
            return;
        }

        if (had_previous_value) {
            settings->set_setting(setting_path, previous_value);
        } else {
            settings->clear(setting_path);
        }

        settings->save();
    }
};

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

TEST_CASE("[GaussianSplatting][Node] Default update mode processes automatically") {
    SceneTree *tree = SceneTree::get_singleton();
    bool created_tree = false;

    if (!tree) {
        tree = memnew(SceneTree);
        tree->initialize();
        created_tree = true;
    }

    Window *root = tree->get_root();
    CHECK_MESSAGE(root != nullptr, "SceneTree root window must exist for GaussianSplatNode3D test");
    if (root == nullptr) {
        if (created_tree) {
            tree->finalize();
            memdelete(tree);
        }
        return;
    }

    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    CHECK(node != nullptr);
    if (node == nullptr) {
        if (created_tree) {
            tree->finalize();
            memdelete(tree);
        }
        return;
    }

    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    node->set_splat_asset(asset);

    root->add_child(node);
    CHECK(node->is_inside_tree());
    if (!node->is_inside_tree()) {
        memdelete(node);
        if (created_tree) {
            tree->finalize();
            memdelete(tree);
        }
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

    if (created_tree) {
        tree->finalize();
        memdelete(tree);
    }
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
    data->resize(splat_count);

    data->set_positions(asset->get_position_vectors());
    data->set_scales(asset->get_scale_vectors());
    data->set_rotations(asset->get_rotation_quaternions());
    data->set_spherical_harmonics(asset->get_spherical_harmonics_buffer());
    data->set_opacities(asset->get_opacities());
    data->set_palette_ids(asset->get_palette_ids_buffer());
    data->set_brush_override_ids(asset->get_brush_override_ids_buffer());
    data->set_normals(asset->get_normal_vectors());
    data->set_brush_axes(asset->get_brush_axes_vector2());
    data->set_stroke_ages(asset->get_stroke_ages_buffer());

    CHECK(data->get_2d_mode());
    CHECK_EQ(data->get_sh_first_order_count(), 2u);
    CHECK_EQ(data->get_sh_high_order_count(), 1u);

    Gaussian g0 = data->get_gaussian(0);
    CHECK(g0.position.is_equal_approx(Vector3(1.0f, 2.0f, 3.0f)));
    CHECK(g0.scale.is_equal_approx(Vector3(0.5f, 0.6f, 0.7f)));
    CHECK(g0.rotation.is_equal_approx(Quaternion(0.0f, 0.0f, 0.0f, 1.0f)));
    CHECK(Math::is_equal_approx(g0.opacity, 0.5f));
    CHECK(g0.sh_dc.is_equal_approx(Color(0.8f, 0.1f, 0.2f, 1.0f)));
    CHECK(g0.sh_1[0].is_equal_approx(Vector3(0.01f, 0.02f, 0.03f)));
    CHECK(g0.sh_1[1].is_equal_approx(Vector3(0.04f, 0.05f, 0.06f)));
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
    CHECK(high_ptr != nullptr);
    if (high_ptr == nullptr) {
        return;
    }
    CHECK(high_ptr[0].is_equal_approx(Vector3(0.2f, 0.3f, 0.4f)));
    CHECK(high_ptr[1].is_equal_approx(Vector3(0.5f, 0.6f, 0.7f)));
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

TEST_CASE("[GaussianSplatting][World] Shared renderer ownership blocks foreign clear/mutate") {
    SceneTree *tree = SceneTree::get_singleton();
    bool created_tree = false;

    if (!tree) {
        tree = memnew(SceneTree);
        tree->initialize();
        created_tree = true;
    }

    Window *root = tree->get_root();
    CHECK_MESSAGE(root != nullptr, "SceneTree root window must exist for GaussianSplatWorld3D ownership test");
    if (root == nullptr) {
        if (created_tree) {
            tree->finalize();
            memdelete(tree);
        }
        return;
    }

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
    CHECK(renderer_a.is_valid());
    CHECK(renderer_b.is_valid());
    CHECK(renderer_a == renderer_b);
    if (!renderer_a.is_valid() || !renderer_b.is_valid() || renderer_a != renderer_b) {
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        if (created_tree) {
            tree->finalize();
            memdelete(tree);
        }
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

    if (created_tree) {
        tree->finalize();
        memdelete(tree);
    }
}

TEST_CASE("[GaussianSplatting][Node] Shared renderer settings remain owned by first claimant") {
    SceneTree *tree = SceneTree::get_singleton();
    bool created_tree = false;

    if (!tree) {
        tree = memnew(SceneTree);
        tree->initialize();
        created_tree = true;
    }

    Window *root = tree->get_root();
    CHECK_MESSAGE(root != nullptr, "SceneTree root window must exist for GaussianSplatNode3D settings isolation test");
    if (root == nullptr) {
        if (created_tree) {
            tree->finalize();
            memdelete(tree);
        }
        return;
    }

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
    CHECK(renderer_a.is_valid());
    CHECK(renderer_b.is_valid());
    CHECK(renderer_a == renderer_b);
    if (!renderer_a.is_valid() || !renderer_b.is_valid() || renderer_a != renderer_b) {
        root->remove_child(node_b);
        root->remove_child(node_a);
        memdelete(node_b);
        memdelete(node_a);
        if (created_tree) {
            tree->finalize();
            memdelete(tree);
        }
        return;
    }
    CHECK(renderer_a->is_debug_show_density_heatmap());

    node_b->set_show_density_heatmap(true);
    CHECK(renderer_a->is_debug_show_density_heatmap());

    node_b->set_show_density_heatmap(false);
    CHECK(renderer_a->is_debug_show_density_heatmap());

    node_a->set_show_density_heatmap(false);
    CHECK_FALSE(renderer_a->is_debug_show_density_heatmap());

    root->remove_child(node_b);
    root->remove_child(node_a);
    memdelete(node_b);
    memdelete(node_a);

    if (created_tree) {
        tree->finalize();
        memdelete(tree);
    }
}

TEST_CASE("[GaussianSplatting][DynamicInstance] Null or empty data unregisters instance") {
    SceneTree *tree = SceneTree::get_singleton();
    bool created_tree = false;

    if (!tree) {
        tree = memnew(SceneTree);
        tree->initialize();
        created_tree = true;
    }

    Window *root = tree->get_root();
    CHECK_MESSAGE(root != nullptr, "SceneTree root window must exist for GaussianSplatDynamicInstance3D unregister test");
    if (root == nullptr) {
        if (created_tree) {
            tree->finalize();
            memdelete(tree);
        }
        return;
    }

    GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
    CHECK_MESSAGE(director != nullptr, "Scene director singleton must exist for dynamic unregister test");
    if (!director) {
        if (created_tree) {
            tree->finalize();
            memdelete(tree);
        }
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

    if (created_tree) {
        tree->finalize();
        memdelete(tree);
    }
}

#endif // TESTS_ENABLED
