#ifndef GAUSSIAN_EDITOR_PLUGIN_H
#define GAUSSIAN_EDITOR_PLUGIN_H

#ifdef TOOLS_ENABLED

#include "editor/plugins/editor_plugin.h"
#include "editor/scene/3d/node_3d_editor_gizmos.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/file_system/editor_file_system.h"
#include "core/io/resource_loader.h"
#include "core/templates/hash_map.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../core/gaussian_splat_asset.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "gaussian_splat_gizmo_plugin.h"
#include "gaussian_import_dialog.h"
#include "gaussian_editor_integration.h"
#include "gaussian_inspector_plugins.h"

class Label;
class GaussianSplatNode3D;
class GaussianThumbnailGenerator;
class GaussianSplatAssetPreviewGenerator;
class GaussianImportSettingsDialog;
class SceneTreeTimer;

// Custom editor plugin for Gaussian Splatting
class GaussianEditorPlugin : public EditorPlugin {
    GDCLASS(GaussianEditorPlugin, EditorPlugin);

private:
    EditorFileDialog *import_dialog = nullptr;
    GaussianImportDialog *import_settings_dialog = nullptr;
    GaussianImportSettingsDialog *gaussian_import_settings_dialog = nullptr;

    // Current selection
    Ref<GaussianSplatRenderer> current_renderer;
    GaussianSplatNode3D *current_node = nullptr;
    Ref<GaussianSplatAsset> active_asset;
    String current_source_path;
    Dictionary last_import_options;

    // Gizmo plugin for 3D visualization
    Ref<GaussianSplatGizmoPlugin> gizmo_plugin;

    Ref<GaussianDataInspectorPlugin> data_inspector_plugin;
    Ref<GaussianRendererInspectorPlugin> renderer_inspector_plugin;
    Ref<GaussianSplatNodeInspectorPlugin> node_inspector_plugin;
    Ref<GaussianAssetInspectorPlugin> asset_inspector_plugin;
    Ref<GaussianEditorIntegration> editor_integration;
    Ref<GaussianThumbnailGenerator> runtime_thumbnail_generator;
    Ref<GaussianSplatAssetPreviewGenerator> resource_preview_generator;

    struct HotReloadWatch {
        uint64_t last_modified = 0;
        Dictionary import_options;
        ObjectID node_id;
    };

    HashMap<String, HotReloadWatch> hot_reload_watches;
    Ref<SceneTreeTimer> hot_reload_timer;
    bool hot_reload_processing = false;

    struct InspectorStatsBinding {
        ObjectID node_id = ObjectID();
        Label *label = nullptr;
    };

    Vector<InspectorStatsBinding> inspector_stats_bindings;
    int inspector_stats_frame_accumulator = 0;

    void _on_import_file_selected(const String &p_path);
    void _update_stats();
    void _update_inspector_stats();
    void _on_stats_label_tree_exiting(Node *p_label);
    Error _import_from_path(const String &p_path, const Dictionary &p_options);
    void _sync_ui_from_asset();
    void _clear_selection();
    HashMap<StringName, Variant> _dictionary_to_hashmap(const Dictionary &p_dict) const;
    void _on_import_settings_confirmed(const String &p_source_path, const Dictionary &p_options);
    void _show_reimport_dialog(const Dictionary &p_options);
    void _refresh_active_asset_metadata();
    Ref<Texture2D> _resolve_asset_thumbnail(const Ref<GaussianSplatAsset> &p_asset);
    void _apply_brush_stroke(ObjectID p_node_id, const Vector3 &p_center, float p_radius, const Color &p_color, float p_strength, float p_hardness);
    void _commit_runtime_modifications(ObjectID p_node_id);
    void _revert_runtime_modifications(ObjectID p_node_id);
    void _track_hot_reload_source(const String &p_path, const Dictionary &p_options, ObjectID p_node_id = ObjectID());
    void _schedule_hot_reload_poll();
    void _on_hot_reload_timer_timeout();
    void _process_hot_reload_for_watch(const String &p_path, HotReloadWatch &p_watch);
    void _on_import_dialog_watch(const String &p_path);

protected:
    void _notification(int p_what);
    static void _bind_methods();

public:
    GaussianEditorPlugin();
    ~GaussianEditorPlugin();

    String get_plugin_name() const override { return "GaussianSplatting"; }
    virtual bool has_main_screen() const override { return false; }
    virtual void edit(Object *p_object) override;
    virtual bool handles(Object *p_object) const override;
    virtual void make_visible(bool p_visible) override;

    void register_inspector_stats(ObjectID p_node_id, Label *p_label);
    void unregister_inspector_stats(Label *p_label);

    void request_asset_reimport(const Ref<GaussianSplatAsset> &p_asset, const Dictionary &p_override = Dictionary());

    // Internal API for inspector plugins and editor integration subsystems.
    // These methods are not part of the user-facing API but must be accessible
    // to tightly-coupled editor helper classes.
    Ref<Texture2D> _internal_resolve_asset_thumbnail(const Ref<GaussianSplatAsset> &p_asset) { return _resolve_asset_thumbnail(p_asset); }
    void _internal_apply_brush_stroke(ObjectID p_node_id, const Vector3 &p_center, float p_radius, const Color &p_color, float p_strength, float p_hardness) {
        _apply_brush_stroke(p_node_id, p_center, p_radius, p_color, p_strength, p_hardness);
    }
    void _internal_commit_runtime_modifications(ObjectID p_node_id) { _commit_runtime_modifications(p_node_id); }
    void _internal_revert_runtime_modifications(ObjectID p_node_id) { _revert_runtime_modifications(p_node_id); }
    void _internal_refresh_active_asset_metadata() { _refresh_active_asset_metadata(); }
};

#endif // TOOLS_ENABLED

#endif // GAUSSIAN_EDITOR_PLUGIN_H
