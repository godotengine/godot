#ifndef GAUSSIAN_INSPECTOR_PLUGINS_H
#define GAUSSIAN_INSPECTOR_PLUGINS_H

#ifdef TOOLS_ENABLED

#include "editor/inspector/editor_inspector.h"
#include "core/math/color.h"
#include "core/math/vector3.h"
#include "../core/gaussian_splat_asset.h"

class GaussianEditorPlugin;
class GaussianSplatNode3D;
class Label;
class Container;
class SpinBox;
class ColorPickerButton;
class OptionButton;

// Custom property editor for GaussianData
class GaussianDataInspectorPlugin : public EditorInspectorPlugin {
    GDCLASS(GaussianDataInspectorPlugin, EditorInspectorPlugin);

public:
    virtual bool can_handle(Object *p_object) override;
    virtual void parse_begin(Object *p_object) override;
};

// Custom property editor for GaussianSplatRenderer
class GaussianRendererInspectorPlugin : public EditorInspectorPlugin {
    GDCLASS(GaussianRendererInspectorPlugin, EditorInspectorPlugin);

public:
    virtual bool can_handle(Object *p_object) override;
    virtual void parse_begin(Object *p_object) override;
};

class GaussianAssetInspectorPlugin : public EditorInspectorPlugin {
    GDCLASS(GaussianAssetInspectorPlugin, EditorInspectorPlugin);

    GaussianEditorPlugin *editor_plugin = nullptr;

    void _on_reimport_pressed(const Ref<GaussianSplatAsset> &p_asset);

public:
    GaussianAssetInspectorPlugin(GaussianEditorPlugin *p_plugin = nullptr);

    virtual bool can_handle(Object *p_object) override;
    virtual void parse_begin(Object *p_object) override;
};

// Custom property editor for GaussianSplatNode3D
class GaussianSplatNodeInspectorPlugin : public EditorInspectorPlugin {
    GDCLASS(GaussianSplatNodeInspectorPlugin, EditorInspectorPlugin);

    struct BrushSessionState {
        Vector3 center = Vector3();
        double radius = 1.5;
        double strength = 0.5;
        double hardness = 1.0;
        Color color = Color(1.0, 0.8, 0.6, 1.0);
        bool has_center = false;
    };

    GaussianEditorPlugin *editor_plugin = nullptr;
    BrushSessionState brush_session_state;

    void _add_quality_button(Container *p_container, const String &p_label, ObjectID p_node_id, int p_preset);
    void _update_stats_label(Label *p_label, GaussianSplatNode3D *p_node);
    GaussianSplatNode3D *_get_node(ObjectID p_node_id) const;
    void _commit_node_property_change(ObjectID p_node_id, const String &p_action_name, const StringName &p_property, const Variant &p_new_value, bool p_notify_property_list_changed = false);

    void _on_quality_preset_pressed(ObjectID p_node_id, int p_preset);
    void _on_reload_pressed(ObjectID p_node_id);
    void _on_force_update_pressed(ObjectID p_node_id);
    void _on_preview_toggled(bool p_pressed, ObjectID p_node_id);
    void _on_bounds_toggled(bool p_pressed, ObjectID p_node_id);
    void _on_stats_toggled(bool p_pressed, ObjectID p_node_id);
    void _on_tile_grid_toggled(bool p_pressed, ObjectID p_node_id);
    void _on_density_heatmap_toggled(bool p_pressed, ObjectID p_node_id);
    void _on_performance_hud_toggled(bool p_pressed, ObjectID p_node_id);
    void _on_lod_spheres_toggled(bool p_pressed, ObjectID p_node_id);
    void _on_performance_overlay_toggled(bool p_pressed, ObjectID p_node_id);
    void _on_debug_draw_mode_selected(int p_index, ObjectID p_node_id, OptionButton *p_source);
    void _on_runtime_preview_toggled(bool p_pressed, ObjectID p_node_id, OptionButton *p_preview_mode);
    void _on_residency_hud_toggled(bool p_pressed, ObjectID p_node_id);
    void _on_brush_center_changed(double p_value, int p_axis);
    void _on_brush_radius_changed(double p_value);
    void _on_brush_strength_changed(double p_value);
    void _on_brush_hardness_changed(double p_value);
    void _on_brush_color_changed(const Color &p_color);
    void _on_apply_brush_pressed(ObjectID p_node_id, SpinBox *p_cx, SpinBox *p_cy, SpinBox *p_cz, SpinBox *p_radius, SpinBox *p_strength, SpinBox *p_hardness, ColorPickerButton *p_color);
    void _on_commit_brush_pressed(ObjectID p_node_id);
    void _on_revert_brush_pressed(ObjectID p_node_id);
    void _on_bake_color_grading_pressed(ObjectID p_node_id);
    void _on_restore_color_grading_pressed(ObjectID p_node_id);

public:
    GaussianSplatNodeInspectorPlugin(GaussianEditorPlugin *p_plugin = nullptr);

    virtual bool can_handle(Object *p_object) override;
    virtual void parse_begin(Object *p_object) override;
    virtual void parse_category(Object *p_object, const String &p_category) override;
    virtual bool parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) override;
};

#endif // TOOLS_ENABLED

#endif // GAUSSIAN_INSPECTOR_PLUGINS_H
