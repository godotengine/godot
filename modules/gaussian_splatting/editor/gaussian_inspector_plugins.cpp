#ifdef TOOLS_ENABLED

#include "gaussian_inspector_plugins.h"
#include "gaussian_editor_plugin.h"
#include "gaussian_editor_services.h"
#include "gaussian_asset_preview_control.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_button.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/container.h"
#include "scene/gui/control.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/spin_box.h"
#include "servers/text_server.h"
#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/object/object.h"
#include "core/string/translation.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_asset.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "../resources/color_grading_resource.h"

namespace {

static Ref<ColorGradingResource> clone_color_grading_resource(const Ref<ColorGradingResource> &p_grading) {
    if (!p_grading.is_valid()) {
        return Ref<ColorGradingResource>();
    }

    Ref<ColorGradingResource> snapshot;
    snapshot.instantiate();
    snapshot->set_enabled(p_grading->get_enabled());
    snapshot->set_exposure(p_grading->get_exposure());
    snapshot->set_contrast(p_grading->get_contrast());
    snapshot->set_saturation(p_grading->get_saturation());
    snapshot->set_temperature(p_grading->get_temperature());
    snapshot->set_tint(p_grading->get_tint());
    snapshot->set_hue_shift(p_grading->get_hue_shift());
    return snapshot;
}

} // namespace

GaussianAssetInspectorPlugin::GaussianAssetInspectorPlugin(GaussianEditorPlugin *p_plugin) {
    editor_plugin = p_plugin;
}

GaussianSplatNodeInspectorPlugin::GaussianSplatNodeInspectorPlugin(GaussianEditorPlugin *p_plugin) {
    editor_plugin = p_plugin;
}

bool GaussianAssetInspectorPlugin::can_handle(Object *p_object) {
    return Object::cast_to<GaussianSplatAsset>(p_object) != nullptr;
}

void GaussianAssetInspectorPlugin::parse_begin(Object *p_object) {
    GaussianSplatAsset *asset = Object::cast_to<GaussianSplatAsset>(p_object);
    if (!asset) {
        return;
    }

    Ref<GaussianSplatAsset> asset_ref(asset);

    VBoxContainer *root = memnew(VBoxContainer);
    root->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    root->add_theme_constant_override("separation", int(Math::round(6.0f * EDSCALE)));

    GaussianAssetPreviewControl *preview = memnew(GaussianAssetPreviewControl);
    preview->set_custom_minimum_size(Size2(0, 220 * EDSCALE));
    preview->set_asset(asset_ref);
    root->add_child(preview);

    HBoxContainer *info_row = memnew(HBoxContainer);
    info_row->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    root->add_child(info_row);

    VBoxContainer *info_column = memnew(VBoxContainer);
    info_column->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    info_row->add_child(info_column);

    Label *summary = memnew(Label);
    summary->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    summary->set_text(GaussianEditorServices::format_asset_metadata_summary(asset_ref, asset->get_import_metadata(), 128));
    info_column->add_child(summary);

    Button *reimport_button = memnew(Button);
    reimport_button->set_text(TTR("Reimport..."));
    reimport_button->set_focus_mode(Control::FOCUS_NONE);
    reimport_button->set_tooltip_text(TTR("Open the Gaussian import dialog with the saved settings."));
    reimport_button->connect("pressed", callable_mp(this, &GaussianAssetInspectorPlugin::_on_reimport_pressed).bind(asset_ref));
    info_row->add_child(reimport_button);

    add_custom_control(root);
}

void GaussianAssetInspectorPlugin::_on_reimport_pressed(const Ref<GaussianSplatAsset> &p_asset) {
    if (!editor_plugin) {
        return;
    }
    editor_plugin->request_asset_reimport(p_asset);
}

// GaussianDataInspectorPlugin implementation

bool GaussianDataInspectorPlugin::can_handle(Object *p_object) {
    return Object::cast_to<::GaussianData>(p_object) != nullptr;
}

void GaussianDataInspectorPlugin::parse_begin(Object *p_object) {
    ::GaussianData *data = Object::cast_to<::GaussianData>(p_object);
    if (!data) {
        return;
    }

    // Add custom properties to the inspector
    // This would add UI for:
    // - Data statistics (count, memory usage, etc.)
    // - Import/export options
    // - Optimization settings
    // - Preview controls

    // For now, just add a simple label
    Label *info_label = memnew(Label);
    info_label->set_text("Gaussian Data: " + itos(data->get_count()) + " splats");
    add_custom_control(info_label);
}

// GaussianRendererInspectorPlugin implementation

bool GaussianRendererInspectorPlugin::can_handle(Object *p_object) {
    return Object::cast_to<GaussianSplatRenderer>(p_object) != nullptr;
}

void GaussianRendererInspectorPlugin::parse_begin(Object *p_object) {
    GaussianSplatRenderer *renderer = Object::cast_to<GaussianSplatRenderer>(p_object);
    if (!renderer) {
        return;
    }

    // Add custom properties to the inspector
    // This would add UI for:
    // - Quality presets dropdown
    // - Performance monitoring
    // - Debug visualization options
    // - Real-time statistics

    // Add quality preset buttons. "Custom" is intentionally omitted:
    // renderer presets only accept named quality tiers and "custom" is a no-op.
    HFlowContainer *preset_container = memnew(HFlowContainer);
    preset_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);

    Button *performance_preset = memnew(Button);
    performance_preset->set_text("Performance");
    performance_preset->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    performance_preset->connect("pressed", callable_mp(renderer, &GaussianSplatRenderer::set_quality_preset).bind("performance"));
    preset_container->add_child(performance_preset);

    Button *balanced_preset = memnew(Button);
    balanced_preset->set_text("Balanced");
    balanced_preset->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    balanced_preset->connect("pressed", callable_mp(renderer, &GaussianSplatRenderer::set_quality_preset).bind("balanced"));
    preset_container->add_child(balanced_preset);

    Button *quality_preset = memnew(Button);
    quality_preset->set_text("Quality");
    quality_preset->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    quality_preset->connect("pressed", callable_mp(renderer, &GaussianSplatRenderer::set_quality_preset).bind("quality"));
    preset_container->add_child(quality_preset);

    add_custom_control(preset_container);

    // Add performance stats
    Dictionary stats = renderer->get_render_stats();
    Label *stats_label = memnew(Label);
    String stats_text = "Performance:\n";
    stats_text += "Visible: " + itos(stats["visible_splats"]) + "/" + itos(stats["total_splats"]) + " splats\n";
    stats_text += "Sort: " + String::num(stats["sort_time_ms"], 1) + " ms | ";
    stats_text += "Render: " + String::num(stats["render_time_ms"], 1) + " ms";
    stats_label->set_text(stats_text);
    add_custom_control(stats_label);
}

void GaussianSplatNodeInspectorPlugin::_add_quality_button(Container *p_container, const String &p_label, ObjectID p_node_id, int p_preset) {
    if (!p_container) {
        return;
    }

    Button *button = memnew(Button);
    button->set_text(p_label);
    button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    button->set_focus_mode(Control::FOCUS_NONE);
    button->connect("pressed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_quality_preset_pressed).bind(p_node_id, p_preset));
    p_container->add_child(button);
}

void GaussianSplatNodeInspectorPlugin::_update_stats_label(Label *p_label, GaussianSplatNode3D *p_node) {
    if (!p_label || !p_node) {
        return;
    }

    Ref<GaussianSplatRenderer> renderer = p_node->get_renderer();
    p_label->set_text(GaussianEditorServices::format_gaussian_splat_stats(p_node, renderer));
}

GaussianSplatNode3D *GaussianSplatNodeInspectorPlugin::_get_node(ObjectID p_node_id) const {
    return Object::cast_to<GaussianSplatNode3D>(ObjectDB::get_instance(p_node_id));
}

void GaussianSplatNodeInspectorPlugin::_commit_node_property_change(ObjectID p_node_id, const String &p_action_name, const StringName &p_property, const Variant &p_new_value, bool p_notify_property_list_changed) {
    GaussianSplatNode3D *node = _get_node(p_node_id);
    if (!node) {
        return;
    }

    const Variant old_value = node->get(p_property);
    if (old_value == p_new_value) {
        return;
    }

    EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
    if (!undo_redo) {
        node->set(p_property, p_new_value);
        if (p_notify_property_list_changed) {
            node->notify_property_list_changed();
        }
        return;
    }

    undo_redo->create_action(p_action_name, UndoRedo::MERGE_DISABLE, node);
    undo_redo->add_do_property(node, p_property, p_new_value);
    undo_redo->add_undo_property(node, p_property, old_value);
    if (p_notify_property_list_changed) {
        undo_redo->add_do_method(node, "notify_property_list_changed");
        undo_redo->add_undo_method(node, "notify_property_list_changed");
    }
    undo_redo->commit_action();
}

void GaussianSplatNodeInspectorPlugin::_on_quality_preset_pressed(ObjectID p_node_id, int p_preset) {
    _commit_node_property_change(p_node_id, TTR("Set Gaussian Quality Preset"), "quality/preset", p_preset, true);
}

void GaussianSplatNodeInspectorPlugin::_on_reload_pressed(ObjectID p_node_id) {
    if (GaussianSplatNode3D *node = _get_node(p_node_id)) {
        node->reload_asset();
    }
}

void GaussianSplatNodeInspectorPlugin::_on_force_update_pressed(ObjectID p_node_id) {
    if (GaussianSplatNode3D *node = _get_node(p_node_id)) {
        node->force_update();
    }
}

void GaussianSplatNodeInspectorPlugin::_on_preview_toggled(bool p_pressed, ObjectID p_node_id) {
    _commit_node_property_change(p_node_id, TTR("Toggle Gaussian Preview"), "debug/preview_enabled", p_pressed, true);
}

void GaussianSplatNodeInspectorPlugin::_on_bounds_toggled(bool p_pressed, ObjectID p_node_id) {
    _commit_node_property_change(p_node_id, TTR("Toggle Gaussian Bounds"), "debug/show_bounds", p_pressed, true);
}

void GaussianSplatNodeInspectorPlugin::_on_stats_toggled(bool p_pressed, ObjectID p_node_id) {
    _commit_node_property_change(p_node_id, TTR("Toggle Gaussian Statistics"), "debug/show_statistics", p_pressed, true);
}

void GaussianSplatNodeInspectorPlugin::_on_tile_grid_toggled(bool p_pressed, ObjectID p_node_id) {
    _commit_node_property_change(p_node_id, TTR("Toggle Gaussian Tile Grid"), "debug/show_tile_grid", p_pressed, true);
}

void GaussianSplatNodeInspectorPlugin::_on_density_heatmap_toggled(bool p_pressed, ObjectID p_node_id) {
    _commit_node_property_change(p_node_id, TTR("Toggle Gaussian Heatmap"), "debug/show_density_heatmap", p_pressed, true);
}

void GaussianSplatNodeInspectorPlugin::_on_performance_hud_toggled(bool p_pressed, ObjectID p_node_id) {
    _commit_node_property_change(p_node_id, TTR("Toggle Gaussian Performance HUD"), "debug/show_performance_hud", p_pressed, true);
}

void GaussianSplatNodeInspectorPlugin::_on_lod_spheres_toggled(bool p_pressed, ObjectID p_node_id) {
    _commit_node_property_change(p_node_id, TTR("Toggle Gaussian LOD Spheres"), "debug/show_lod_spheres", p_pressed, true);
}

void GaussianSplatNodeInspectorPlugin::_on_performance_overlay_toggled(bool p_pressed, ObjectID p_node_id) {
    _commit_node_property_change(p_node_id, TTR("Toggle Gaussian Performance Overlay"), "debug/show_performance_overlay", p_pressed, true);
}

void GaussianSplatNodeInspectorPlugin::_on_debug_draw_mode_selected(int p_index, ObjectID p_node_id, OptionButton *p_source) {
    if (!p_source) {
        return;
    }

    int item_id = p_source->get_item_id(p_index);
    _commit_node_property_change(p_node_id, TTR("Set Gaussian Preview Mode"), "debug/debug_draw_mode", item_id, true);
}

void GaussianSplatNodeInspectorPlugin::_on_runtime_preview_toggled(bool p_pressed, ObjectID p_node_id, OptionButton *p_preview_mode) {
    _commit_node_property_change(p_node_id, TTR("Toggle Gaussian Runtime Preview"), "debug/runtime_preview", p_pressed, true);

    if (p_preview_mode) {
        p_preview_mode->set_disabled(p_pressed);
        p_preview_mode->set_tooltip_text(p_pressed ?
                        TTR("Preview mode is locked while Runtime Preview is enabled.") :
                        String());
    }
}

void GaussianSplatNodeInspectorPlugin::_on_residency_hud_toggled(bool p_pressed, ObjectID p_node_id) {
    _commit_node_property_change(p_node_id, TTR("Toggle Gaussian Residency HUD"), "debug/show_residency_hud", p_pressed, true);
}

void GaussianSplatNodeInspectorPlugin::_on_brush_center_changed(double p_value, int p_axis) {
    brush_session_state.has_center = true;
    switch (p_axis) {
        case 0:
            brush_session_state.center.x = p_value;
            break;
        case 1:
            brush_session_state.center.y = p_value;
            break;
        case 2:
            brush_session_state.center.z = p_value;
            break;
        default:
            break;
    }
}

void GaussianSplatNodeInspectorPlugin::_on_brush_radius_changed(double p_value) {
    brush_session_state.radius = p_value;
}

void GaussianSplatNodeInspectorPlugin::_on_brush_strength_changed(double p_value) {
    brush_session_state.strength = p_value;
}

void GaussianSplatNodeInspectorPlugin::_on_brush_hardness_changed(double p_value) {
    brush_session_state.hardness = p_value;
}

void GaussianSplatNodeInspectorPlugin::_on_brush_color_changed(const Color &p_color) {
    brush_session_state.color = p_color;
}

void GaussianSplatNodeInspectorPlugin::_on_apply_brush_pressed(ObjectID p_node_id, SpinBox *p_cx, SpinBox *p_cy, SpinBox *p_cz, SpinBox *p_radius, SpinBox *p_strength, SpinBox *p_hardness, ColorPickerButton *p_color) {
    if (!editor_plugin) {
        return;
    }
    Vector3 center = brush_session_state.has_center ? brush_session_state.center : Vector3();
    if (p_cx) {
        center.x = p_cx->get_value();
    }
    if (p_cy) {
        center.y = p_cy->get_value();
    }
    if (p_cz) {
        center.z = p_cz->get_value();
    }
    brush_session_state.center = center;
    brush_session_state.has_center = true;

    const double radius = p_radius ? p_radius->get_value() : brush_session_state.radius;
    const double strength = p_strength ? p_strength->get_value() : brush_session_state.strength;
    const double hardness = p_hardness ? p_hardness->get_value() : brush_session_state.hardness;
    const Color color = p_color ? p_color->get_pick_color() : brush_session_state.color;

    brush_session_state.radius = radius;
    brush_session_state.strength = strength;
    brush_session_state.hardness = hardness;
    brush_session_state.color = color;

    editor_plugin->_internal_apply_brush_stroke(p_node_id, center, static_cast<float>(radius), color,
            static_cast<float>(strength), static_cast<float>(hardness));
}

void GaussianSplatNodeInspectorPlugin::_on_commit_brush_pressed(ObjectID p_node_id) {
    if (editor_plugin) {
        editor_plugin->_internal_commit_runtime_modifications(p_node_id);
    }
}

void GaussianSplatNodeInspectorPlugin::_on_revert_brush_pressed(ObjectID p_node_id) {
    if (editor_plugin) {
        editor_plugin->_internal_revert_runtime_modifications(p_node_id);
    }
}

void GaussianSplatNodeInspectorPlugin::_on_bake_color_grading_pressed(ObjectID p_node_id) {
    GaussianSplatNode3D *node = _get_node(p_node_id);
    if (!node) {
        ERR_PRINT("GaussianSplatNode3D not found for color grading bake");
        return;
    }
    if (node->is_color_grading_baked()) {
        return;
    }

    Ref<ColorGradingResource> grading_snapshot = clone_color_grading_resource(node->get_color_grading());
    Error err = node->bake_color_grading_snapshot(grading_snapshot);
    if (err == OK) {
        EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
        if (undo_redo) {
            undo_redo->create_action(TTR("Bake Gaussian Color Grading"), UndoRedo::MERGE_DISABLE, node);
            undo_redo->add_do_method(node, "bake_color_grading_snapshot", grading_snapshot);
            undo_redo->add_do_method(node, "notify_property_list_changed");
            undo_redo->add_undo_method(node, "restore_color_grading");
            undo_redo->add_undo_method(node, "notify_property_list_changed");
            undo_redo->commit_action(false);
        }
        node->notify_property_list_changed();
    } else {
        ERR_PRINT("Failed to bake color grading");
    }
}

void GaussianSplatNodeInspectorPlugin::_on_restore_color_grading_pressed(ObjectID p_node_id) {
    GaussianSplatNode3D *node = _get_node(p_node_id);
    if (!node) {
        ERR_PRINT("GaussianSplatNode3D not found for color grading restore");
        return;
    }
    if (!node->is_color_grading_baked()) {
        return;
    }

    Ref<ColorGradingResource> grading_snapshot = clone_color_grading_resource(node->get_color_grading());
    node->restore_color_grading();
    EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
    if (undo_redo) {
        undo_redo->create_action(TTR("Restore Gaussian Color Grading"), UndoRedo::MERGE_DISABLE, node);
        undo_redo->add_do_method(node, "restore_color_grading");
        undo_redo->add_do_method(node, "notify_property_list_changed");
        undo_redo->add_undo_method(node, "bake_color_grading_snapshot", grading_snapshot);
        undo_redo->add_undo_method(node, "notify_property_list_changed");
        undo_redo->commit_action(false);
    }
    node->notify_property_list_changed();
}

bool GaussianSplatNodeInspectorPlugin::can_handle(Object *p_object) {
    return Object::cast_to<GaussianSplatNode3D>(p_object) != nullptr;
}

void GaussianSplatNodeInspectorPlugin::parse_begin(Object *p_object) {
    GaussianSplatNode3D *node = Object::cast_to<GaussianSplatNode3D>(p_object);
    if (!node) {
        return;
    }

    VBoxContainer *root = memnew(VBoxContainer);
    root->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    root->add_theme_constant_override("separation", int(Math::round(4.0f * EDSCALE)));

    Label *header = memnew(Label);
    header->set_text("Gaussian Splat Overview");
    root->add_child(header);

    Label *origin = memnew(Label);
    origin->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    origin->set_text("Asset Origin: " + node->get_asset_origin_label());
    root->add_child(origin);

    Label *stats = memnew(Label);
    stats->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    _update_stats_label(stats, node);
    root->add_child(stats);

    if (editor_plugin) {
        editor_plugin->register_inspector_stats(node->get_instance_id(), stats);
    }

    HFlowContainer *actions = memnew(HFlowContainer);
    actions->set_h_size_flags(Control::SIZE_EXPAND_FILL);

    Button *reload = memnew(Button);
    reload->set_text("Reload");
    reload->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    reload->connect("pressed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_reload_pressed).bind(node->get_instance_id()));
    actions->add_child(reload);

    Button *force_update = memnew(Button);
    force_update->set_text("Force Update");
    force_update->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    force_update->connect("pressed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_force_update_pressed).bind(node->get_instance_id()));
    actions->add_child(force_update);

    root->add_child(actions);

    Label *quality_label = memnew(Label);
    quality_label->set_text("Quality Presets");
    root->add_child(quality_label);

    HFlowContainer *quality_row = memnew(HFlowContainer);
    quality_row->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    _add_quality_button(quality_row, "Performance", node->get_instance_id(), GaussianSplatNode3D::QUALITY_PERFORMANCE);
    _add_quality_button(quality_row, "Balanced", node->get_instance_id(), GaussianSplatNode3D::QUALITY_BALANCED);
    _add_quality_button(quality_row, "Quality", node->get_instance_id(), GaussianSplatNode3D::QUALITY_QUALITY);
    root->add_child(quality_row);

#ifdef DEBUG_ENABLED
    Label *debug_label = memnew(Label);
    debug_label->set_text("Debug Visualization");
    root->add_child(debug_label);

    HFlowContainer *toggle_row = memnew(HFlowContainer);
    toggle_row->set_h_size_flags(Control::SIZE_EXPAND_FILL);

    CheckButton *preview_toggle = memnew(CheckButton);
    preview_toggle->set_text("Preview");
    preview_toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    preview_toggle->set_pressed(node->is_preview_enabled());
    preview_toggle->connect("toggled", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_preview_toggled).bind(node->get_instance_id()));
    toggle_row->add_child(preview_toggle);

    CheckButton *bounds_toggle = memnew(CheckButton);
    bounds_toggle->set_text("Bounds");
    bounds_toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    bounds_toggle->set_pressed(node->is_showing_bounds());
    bounds_toggle->connect("toggled", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_bounds_toggled).bind(node->get_instance_id()));
    toggle_row->add_child(bounds_toggle);

    CheckButton *stats_toggle = memnew(CheckButton);
    stats_toggle->set_text("Statistics");
    stats_toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    stats_toggle->set_pressed(node->is_showing_statistics());
    stats_toggle->connect("toggled", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_stats_toggled).bind(node->get_instance_id()));
    toggle_row->add_child(stats_toggle);

    root->add_child(toggle_row);

    HFlowContainer *overlay_row = memnew(HFlowContainer);
    overlay_row->set_h_size_flags(Control::SIZE_EXPAND_FILL);

    CheckButton *grid_toggle = memnew(CheckButton);
    grid_toggle->set_text("Tile Grid");
    grid_toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    grid_toggle->set_pressed(node->is_showing_tile_grid());
    grid_toggle->connect("toggled", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_tile_grid_toggled).bind(node->get_instance_id()));
    overlay_row->add_child(grid_toggle);

    CheckButton *heatmap_toggle = memnew(CheckButton);
    heatmap_toggle->set_text("Heatmap");
    heatmap_toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    heatmap_toggle->set_pressed(node->is_showing_density_heatmap());
    heatmap_toggle->connect("toggled", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_density_heatmap_toggled).bind(node->get_instance_id()));
    overlay_row->add_child(heatmap_toggle);

    CheckButton *hud_toggle = memnew(CheckButton);
    hud_toggle->set_text("HUD");
    hud_toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    hud_toggle->set_pressed(node->is_showing_performance_hud());
    hud_toggle->connect("toggled", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_performance_hud_toggled).bind(node->get_instance_id()));
    overlay_row->add_child(hud_toggle);

    root->add_child(overlay_row);

    HFlowContainer *lod_row = memnew(HFlowContainer);
    lod_row->set_h_size_flags(Control::SIZE_EXPAND_FILL);

    CheckButton *lod_toggle = memnew(CheckButton);
    lod_toggle->set_text("LOD Spheres");
    lod_toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    lod_toggle->set_pressed(node->is_showing_lod_spheres());
    lod_toggle->connect("toggled", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_lod_spheres_toggled).bind(node->get_instance_id()));
    lod_row->add_child(lod_toggle);

    CheckButton *performance_overlay_toggle = memnew(CheckButton);
    performance_overlay_toggle->set_text("Performance Overlay");
    performance_overlay_toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    performance_overlay_toggle->set_pressed(node->is_showing_performance_overlay());
    performance_overlay_toggle->connect("toggled", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_performance_overlay_toggled).bind(node->get_instance_id()));
    lod_row->add_child(performance_overlay_toggle);

    root->add_child(lod_row);

    GridContainer *preview_row = memnew(GridContainer);
    preview_row->set_columns(2);
    preview_row->set_h_size_flags(Control::SIZE_EXPAND_FILL);

    Label *preview_label = memnew(Label);
    preview_label->set_text("Preview Mode");
    preview_row->add_child(preview_label);

    OptionButton *preview_mode = memnew(OptionButton);
    preview_mode->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    preview_mode->add_item("Off", GaussianSplatNode3D::DEBUG_DRAW_OFF);
    preview_mode->add_item("Wireframe", GaussianSplatNode3D::DEBUG_DRAW_WIREFRAME);
    preview_mode->add_item("Points", GaussianSplatNode3D::DEBUG_DRAW_POINTS);
    preview_mode->add_item("Heatmap", GaussianSplatNode3D::DEBUG_DRAW_HEATMAP);
    int selected_index = preview_mode->get_item_index(node->get_debug_draw_mode());
    if (selected_index >= 0) {
        preview_mode->select(selected_index);
    }
    if (node->is_runtime_preview_enabled()) {
        preview_mode->set_disabled(true);
        preview_mode->set_tooltip_text(TTR("Preview mode is locked while Runtime Preview is enabled."));
    }
    preview_mode->connect("item_selected", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_debug_draw_mode_selected).bind(node->get_instance_id(), preview_mode));
    preview_row->add_child(preview_mode);

    root->add_child(preview_row);

    HFlowContainer *runtime_row = memnew(HFlowContainer);
    runtime_row->set_h_size_flags(Control::SIZE_EXPAND_FILL);

    CheckButton *runtime_preview_toggle = memnew(CheckButton);
    runtime_preview_toggle->set_text(TTR("Runtime Preview"));
    runtime_preview_toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    runtime_preview_toggle->set_pressed(node->is_runtime_preview_enabled());
    runtime_preview_toggle->connect("toggled", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_runtime_preview_toggled).bind(node->get_instance_id(), preview_mode));
    runtime_row->add_child(runtime_preview_toggle);

    CheckButton *residency_toggle = memnew(CheckButton);
    residency_toggle->set_text(TTR("Residency HUD"));
    residency_toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    residency_toggle->set_pressed(node->is_showing_residency_hud());
    residency_toggle->connect("toggled", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_residency_hud_toggled).bind(node->get_instance_id()));
    runtime_row->add_child(residency_toggle);

    root->add_child(runtime_row);
#endif

    HSeparator *paint_separator = memnew(HSeparator);
    root->add_child(paint_separator);

    Label *paint_label = memnew(Label);
    paint_label->set_text(TTR("Painterly Brush Tools"));
    root->add_child(paint_label);

    Label *paint_hint = memnew(Label);
    paint_hint->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    paint_hint->set_text(TTR("Session-only brush controls: values persist while the editor stays open and are not saved to scenes or resources."));
    root->add_child(paint_hint);

    GridContainer *paint_grid = memnew(GridContainer);
    paint_grid->set_columns(2);
    paint_grid->set_h_size_flags(Control::SIZE_EXPAND_FILL);

    Vector3 node_position = node->get_global_position();
    if (!brush_session_state.has_center) {
        brush_session_state.center = node_position;
        brush_session_state.has_center = true;
    }

    Label *center_label = memnew(Label);
    center_label->set_text(TTR("Center X"));
    paint_grid->add_child(center_label);
    SpinBox *center_x = memnew(SpinBox);
    center_x->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    center_x->set_min(-10000.0);
    center_x->set_max(10000.0);
    center_x->set_step(0.1);
    center_x->set_value(brush_session_state.center.x);
    center_x->connect("value_changed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_brush_center_changed).bind(0));
    paint_grid->add_child(center_x);

    Label *center_y_label = memnew(Label);
    center_y_label->set_text(TTR("Center Y"));
    paint_grid->add_child(center_y_label);
    SpinBox *center_y = memnew(SpinBox);
    center_y->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    center_y->set_min(-10000.0);
    center_y->set_max(10000.0);
    center_y->set_step(0.1);
    center_y->set_value(brush_session_state.center.y);
    center_y->connect("value_changed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_brush_center_changed).bind(1));
    paint_grid->add_child(center_y);

    Label *center_z_label = memnew(Label);
    center_z_label->set_text(TTR("Center Z"));
    paint_grid->add_child(center_z_label);
    SpinBox *center_z = memnew(SpinBox);
    center_z->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    center_z->set_min(-10000.0);
    center_z->set_max(10000.0);
    center_z->set_step(0.1);
    center_z->set_value(brush_session_state.center.z);
    center_z->connect("value_changed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_brush_center_changed).bind(2));
    paint_grid->add_child(center_z);

    Label *radius_label = memnew(Label);
    radius_label->set_text(TTR("Radius"));
    paint_grid->add_child(radius_label);
    SpinBox *radius_spin = memnew(SpinBox);
    radius_spin->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    radius_spin->set_min(0.01);
    radius_spin->set_max(250.0);
    radius_spin->set_step(0.05);
    radius_spin->set_value(brush_session_state.radius);
    radius_spin->connect("value_changed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_brush_radius_changed));
    paint_grid->add_child(radius_spin);

    Label *strength_label = memnew(Label);
    strength_label->set_text(TTR("Strength"));
    paint_grid->add_child(strength_label);
    SpinBox *strength_spin = memnew(SpinBox);
    strength_spin->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    strength_spin->set_min(0.0);
    strength_spin->set_max(1.0);
    strength_spin->set_step(0.01);
    strength_spin->set_value(brush_session_state.strength);
    strength_spin->connect("value_changed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_brush_strength_changed));
    paint_grid->add_child(strength_spin);

    Label *hardness_label = memnew(Label);
    hardness_label->set_text(TTR("Hardness"));
    paint_grid->add_child(hardness_label);
    SpinBox *hardness_spin = memnew(SpinBox);
    hardness_spin->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    hardness_spin->set_min(0.1);
    hardness_spin->set_max(8.0);
    hardness_spin->set_step(0.05);
    hardness_spin->set_value(brush_session_state.hardness);
    hardness_spin->connect("value_changed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_brush_hardness_changed));
    paint_grid->add_child(hardness_spin);

    Label *color_label = memnew(Label);
    color_label->set_text(TTR("Color"));
    paint_grid->add_child(color_label);
    ColorPickerButton *color_button = memnew(ColorPickerButton);
    color_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    color_button->set_custom_minimum_size(Size2(80, 0));
    color_button->set_pick_color(brush_session_state.color);
    color_button->connect("color_changed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_brush_color_changed));
    paint_grid->add_child(color_button);

    root->add_child(paint_grid);

    HFlowContainer *paint_actions = memnew(HFlowContainer);
    paint_actions->set_h_size_flags(Control::SIZE_EXPAND_FILL);

    Ref<GaussianSplatRenderer> node_renderer = node->get_renderer();
    Ref<::GaussianData> node_data = node_renderer.is_valid() ? node_renderer->get_gaussian_data() : Ref<::GaussianData>();
    const bool brush_actions_available = node_data.is_valid() && node_data->get_count() > 0;
    const String brush_disabled_tooltip = TTR("Load a Gaussian asset before using brush actions.");

    Button *apply_brush = memnew(Button);
    apply_brush->set_text(TTR("Apply Brush"));
    apply_brush->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    apply_brush->set_disabled(!brush_actions_available);
    if (!brush_actions_available) {
        apply_brush->set_tooltip_text(brush_disabled_tooltip);
    }
    apply_brush->connect("pressed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_apply_brush_pressed).bind(node->get_instance_id(), center_x, center_y, center_z, radius_spin, strength_spin, hardness_spin, color_button));
    paint_actions->add_child(apply_brush);

    Button *commit_brush = memnew(Button);
    commit_brush->set_text(TTR("Commit"));
    commit_brush->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    commit_brush->set_disabled(!brush_actions_available);
    if (!brush_actions_available) {
        commit_brush->set_tooltip_text(brush_disabled_tooltip);
    }
    commit_brush->connect("pressed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_commit_brush_pressed).bind(node->get_instance_id()));
    paint_actions->add_child(commit_brush);

    Button *revert_brush = memnew(Button);
    revert_brush->set_text(TTR("Revert"));
    revert_brush->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    revert_brush->set_disabled(!brush_actions_available);
    if (!brush_actions_available) {
        revert_brush->set_tooltip_text(brush_disabled_tooltip);
    }
    revert_brush->connect("pressed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_revert_brush_pressed).bind(node->get_instance_id()));
    paint_actions->add_child(revert_brush);

    root->add_child(paint_actions);

    // Color Grading section
    HSeparator *color_grading_separator = memnew(HSeparator);
    root->add_child(color_grading_separator);

    Label *color_grading_label = memnew(Label);
    color_grading_label->set_text(TTR("Color Grading"));
    root->add_child(color_grading_label);

    HFlowContainer *color_grading_actions = memnew(HFlowContainer);
    color_grading_actions->set_h_size_flags(Control::SIZE_EXPAND_FILL);

    Button *bake_button = memnew(Button);
    bake_button->set_text(TTR("Bake Color Grading"));
    bake_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    bake_button->set_tooltip_text(TTR("Permanently applies color grading to splat colors (zero runtime cost)"));
    bake_button->connect("pressed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_bake_color_grading_pressed).bind(node->get_instance_id()));
    color_grading_actions->add_child(bake_button);

    Button *restore_button = memnew(Button);
    restore_button->set_text(TTR("Restore Original"));
    restore_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    restore_button->set_tooltip_text(TTR("Restores original colors before baking"));
    restore_button->connect("pressed", callable_mp(this, &GaussianSplatNodeInspectorPlugin::_on_restore_color_grading_pressed).bind(node->get_instance_id()));
    color_grading_actions->add_child(restore_button);

    root->add_child(color_grading_actions);

    // Display bake status
    if (node->is_color_grading_baked()) {
        Label *bake_status = memnew(Label);
        bake_status->set_text(TTR("Status: Color grading is baked"));
        bake_status->add_theme_color_override("font_color", Color(0.4, 1.0, 0.4));
        root->add_child(bake_status);
    }

    add_custom_control(root);
}

void GaussianSplatNodeInspectorPlugin::parse_category(Object *p_object, const String &p_category) {
    // Use default behavior for categories.
    EditorInspectorPlugin::parse_category(p_object, p_category);
}

bool GaussianSplatNodeInspectorPlugin::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) {
    // Skip default debug toggles since we provide custom controls.
    if (p_path == "debug/preview_enabled" || p_path == "debug/show_bounds" || p_path == "debug/show_statistics" ||
            p_path == "debug/show_tile_grid" || p_path == "debug/show_density_heatmap" ||
            p_path == "debug/show_performance_hud" || p_path == "debug/show_lod_spheres" ||
            p_path == "debug/show_performance_overlay" || p_path == "debug/debug_draw_mode" ||
            p_path == "debug/runtime_preview" || p_path == "debug/show_residency_hud") {
        return true;
    }

    return false;
}

#endif // TOOLS_ENABLED
