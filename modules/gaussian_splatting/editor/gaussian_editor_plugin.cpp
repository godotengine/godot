#ifdef TOOLS_ENABLED

#include "gaussian_editor_plugin.h"
#include "gaussian_editor_services.h"
#include "gaussian_import_settings_dialog.h"
#include "gaussian_resource_preview_generator.h"
#include "editor/editor_node.h"
#include "editor/editor_interface.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_file_dialog.h"
#include "core/config/project_settings.h"
#include "scene/gui/label.h"
#include "scene/main/scene_tree.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_source_path.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "gaussian_thumbnail_generator.h"
#include "../logger/gs_logger.h"
#include "core/math/math_funcs.h"
#include "core/object/object.h"
#include "core/io/file_access.h"

static const int DEFAULT_THUMBNAIL_SIZE = 128;

namespace {

static int64_t _dict_get_int(const Dictionary &p_dict, const StringName &p_key, int64_t p_default = 0) {
    return GaussianEditorServices::dict_get_int(p_dict, p_key, p_default);
}

static bool _dict_get_bool(const Dictionary &p_dict, const StringName &p_key, bool p_default = false) {
    return GaussianEditorServices::dict_get_bool(p_dict, p_key, p_default);
}

static bool _project_setting_bool(ProjectSettings *p_settings, const StringName &p_path, bool p_default) {
    if (!p_settings || !p_settings->has_setting(p_path)) {
        return p_default;
    }
    const Variant value = p_settings->get_setting_with_override(p_path);
    switch (value.get_type()) {
        case Variant::BOOL:
            return bool(value);
        case Variant::INT:
            return int64_t(value) != 0;
        case Variant::FLOAT:
            return !Math::is_zero_approx(float(double(value)));
        default:
            return p_default;
    }
}

static double _project_setting_double(ProjectSettings *p_settings, const StringName &p_path, double p_default) {
    if (!p_settings || !p_settings->has_setting(p_path)) {
        return p_default;
    }
    const Variant value = p_settings->get_setting_with_override(p_path);
    switch (value.get_type()) {
        case Variant::FLOAT:
            return double(value);
        case Variant::INT:
            return double(int64_t(value));
        default:
            return p_default;
    }
}

static bool _hot_reload_enabled() {
    return _project_setting_bool(ProjectSettings::get_singleton(),
            "rendering/gaussian_splatting/editor/hot_reload_enabled", true);
}

static double _hot_reload_poll_interval_seconds() {
    const double configured = _project_setting_double(ProjectSettings::get_singleton(),
            "rendering/gaussian_splatting/editor/hot_reload_poll_interval_sec", 1.0);
    return CLAMP(configured, 0.1, 10.0);
}

static Ref<::GaussianData> convert_asset_to_gaussian_data(const Ref<GaussianSplatAsset> &p_asset) {
    if (p_asset.is_null()) {
        return Ref<::GaussianData>();
    }

    Ref<::GaussianData> data;
    if (!p_asset->populate_gaussian_data(data)) {
        return Ref<::GaussianData>();
    }

    return data;
}

static String _get_node_source_path(GaussianSplatNode3D *p_node) {
    if (!p_node) {
        return String();
    }

    return GaussianSplatSourcePath::resolve_primary_source_path(
            p_node->get_splat_asset(), p_node->get_ply_file_path());
}

static Ref<GaussianSplatAsset> _load_gaussian_splat_asset(const String &p_path, bool p_force_reload) {
    if (p_path.is_empty()) {
        return Ref<GaussianSplatAsset>();
    }

    return ResourceLoader::load(p_path, "GaussianSplatAsset",
            p_force_reload ? ResourceFormatLoader::CACHE_MODE_REPLACE : ResourceFormatLoader::CACHE_MODE_REUSE);
}

static void _append_unique_hot_reload_node_id(Vector<ObjectID> &r_node_ids, ObjectID p_node_id) {
    if (p_node_id == ObjectID()) {
        return;
    }

    for (int i = 0; i < r_node_ids.size(); i++) {
        if (r_node_ids[i] == p_node_id) {
            return;
        }
    }

    r_node_ids.push_back(p_node_id);
}

static Vector<GaussianSplatNode3D *> _collect_live_hot_reload_nodes(Vector<ObjectID> &r_node_ids) {
    Vector<GaussianSplatNode3D *> live_nodes;
    Vector<ObjectID> live_node_ids;
    for (int i = 0; i < r_node_ids.size(); i++) {
        const ObjectID node_id = r_node_ids[i];
        if (node_id == ObjectID()) {
            continue;
        }

        GaussianSplatNode3D *node = Object::cast_to<GaussianSplatNode3D>(ObjectDB::get_instance(node_id));
        if (!node) {
            continue;
        }

        bool already_added = false;
        for (int j = 0; j < live_node_ids.size(); j++) {
            if (live_node_ids[j] == node_id) {
                already_added = true;
                break;
            }
        }
        if (already_added) {
            continue;
        }

        live_node_ids.push_back(node_id);
        live_nodes.push_back(node);
    }

    r_node_ids = live_node_ids;
    return live_nodes;
}

} // namespace

// GaussianEditorPlugin implementation

GaussianEditorPlugin::GaussianEditorPlugin() {
    runtime_thumbnail_generator.instantiate();
    editor_integration.instantiate();

    import_dialog = memnew(EditorFileDialog);
    import_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
    import_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
    import_dialog->connect("file_selected", callable_mp(this, &GaussianEditorPlugin::_on_import_file_selected));
    add_child(import_dialog);

    import_settings_dialog = memnew(GaussianImportDialog);
    import_settings_dialog->connect("import_requested", callable_mp(this, &GaussianEditorPlugin::_on_import_settings_confirmed));
    import_settings_dialog->connect("watch_path_requested", callable_mp(this, &GaussianEditorPlugin::_on_import_dialog_watch));
    add_child(import_settings_dialog);

    // Advanced import settings dialog (opened by double-clicking .ply/.spz in filesystem).
    gaussian_import_settings_dialog = memnew(GaussianImportSettingsDialog);
    add_child(gaussian_import_settings_dialog);

    editor_integration->setup(this, import_dialog);
}

GaussianEditorPlugin::~GaussianEditorPlugin() {
    if (editor_integration.is_valid()) {
        editor_integration->teardown();
    }
    if (runtime_thumbnail_generator.is_valid()) {
        runtime_thumbnail_generator->clear_cache();
    }
    hot_reload_watches.clear();
    hot_reload_timer.unref();
    hot_reload_processing = false;
}

void GaussianEditorPlugin::_bind_methods() {
    // Bind methods if needed for signals
}

void GaussianEditorPlugin::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE: {
            if (editor_integration.is_valid()) {
                editor_integration->setup(this, import_dialog);
            }

            if (resource_preview_generator.is_null()) {
                resource_preview_generator.instantiate();
            }
            if (resource_preview_generator.is_valid()) {
                EditorInterface *editor_interface = EditorInterface::get_singleton();
                EditorResourcePreview *resource_previewer = editor_interface ? editor_interface->get_resource_previewer() : nullptr;
                if (resource_previewer) {
                    resource_previewer->add_preview_generator(resource_preview_generator);
                }
            }

            if (data_inspector_plugin.is_null()) {
                data_inspector_plugin.instantiate();
                add_inspector_plugin(data_inspector_plugin);
            }

            if (renderer_inspector_plugin.is_null()) {
                renderer_inspector_plugin.instantiate();
                add_inspector_plugin(renderer_inspector_plugin);
            }

            if (asset_inspector_plugin.is_null()) {
                asset_inspector_plugin = Ref<GaussianAssetInspectorPlugin>(memnew(GaussianAssetInspectorPlugin(this)));
                add_inspector_plugin(asset_inspector_plugin);
            }

            if (node_inspector_plugin.is_null()) {
                node_inspector_plugin = Ref<GaussianSplatNodeInspectorPlugin>(memnew(GaussianSplatNodeInspectorPlugin(this)));
                add_inspector_plugin(node_inspector_plugin);
            }

            if (gizmo_plugin.is_null()) {
                gizmo_plugin.instantiate();
                add_node_3d_gizmo_plugin(gizmo_plugin);
            }

            inspector_stats_frame_accumulator = 0;
            set_process(true);
        } break;
        case NOTIFICATION_EXIT_TREE: {
            if (gizmo_plugin.is_valid()) {
                remove_node_3d_gizmo_plugin(gizmo_plugin);
                gizmo_plugin.unref();
            }

            if (node_inspector_plugin.is_valid()) {
                remove_inspector_plugin(node_inspector_plugin);
                node_inspector_plugin.unref();
            }

            if (renderer_inspector_plugin.is_valid()) {
                remove_inspector_plugin(renderer_inspector_plugin);
                renderer_inspector_plugin.unref();
            }

            if (asset_inspector_plugin.is_valid()) {
                remove_inspector_plugin(asset_inspector_plugin);
                asset_inspector_plugin.unref();
            }

            if (data_inspector_plugin.is_valid()) {
                remove_inspector_plugin(data_inspector_plugin);
                data_inspector_plugin.unref();
            }

            if (editor_integration.is_valid()) {
                editor_integration->teardown();
            }
            if (resource_preview_generator.is_valid()) {
                EditorInterface *editor_interface = EditorInterface::get_singleton();
                EditorResourcePreview *resource_previewer = editor_interface ? editor_interface->get_resource_previewer() : nullptr;
                if (resource_previewer) {
                    resource_previewer->remove_preview_generator(resource_preview_generator);
                }
                resource_preview_generator.unref();
            }
            if (runtime_thumbnail_generator.is_valid()) {
                runtime_thumbnail_generator->clear_cache();
            }
            hot_reload_watches.clear();
            hot_reload_timer.unref();
            hot_reload_processing = false;

            inspector_stats_bindings.clear();
            set_process(false);
        } break;
        case NOTIFICATION_PROCESS: {
            if (++inspector_stats_frame_accumulator >= 15) {
                inspector_stats_frame_accumulator = 0;
                _update_inspector_stats();
            }
        } break;
    }
}

void GaussianEditorPlugin::edit(Object *p_object) {
    _clear_selection();

    if (GaussianSplatNode3D *node = Object::cast_to<GaussianSplatNode3D>(p_object)) {
        current_node = node;
        current_renderer = node->get_renderer();
        active_asset = node->get_splat_asset();
        current_source_path = _get_node_source_path(node);
    } else if (GaussianSplatRenderer *renderer = Object::cast_to<GaussianSplatRenderer>(p_object)) {
        current_renderer = Ref<GaussianSplatRenderer>(renderer);
    } else if (GaussianSplatAsset *asset_obj = Object::cast_to<GaussianSplatAsset>(p_object)) {
        Ref<GaussianSplatAsset> asset(asset_obj);
        active_asset = asset;
        current_source_path = GaussianSplatSourcePath::get_asset_source_path(asset);
    }

    if (active_asset.is_valid()) {
        _refresh_active_asset_metadata();
    } else {
        _sync_ui_from_asset();
        _update_stats();
    }
}

bool GaussianEditorPlugin::handles(Object *p_object) const {
    return Object::cast_to<GaussianSplatRenderer>(p_object) != nullptr ||
           Object::cast_to<::GaussianData>(p_object) != nullptr ||
           Object::cast_to<GaussianSplatNode3D>(p_object) != nullptr ||
           Object::cast_to<GaussianSplatAsset>(p_object) != nullptr;
}

void GaussianEditorPlugin::make_visible(bool p_visible) {
    (void)p_visible;
}

void GaussianEditorPlugin::_on_import_file_selected(const String &p_path) {
    if (p_path.is_empty() || !import_settings_dialog) {
        return;
    }

    String resource_path = p_path;
    if (editor_integration.is_valid()) {
        resource_path = editor_integration->normalize_path(p_path);
    } else if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        resource_path = ps->localize_path(p_path);
    }

    if (!resource_path.begins_with("res://")) {
        EditorNode::get_singleton()->show_warning(TTR("Selected file must be inside the project (res://)."));
        return;
    }

    import_settings_dialog->configure_for_file(resource_path, Ref<GaussianSplatAsset>(), false, Dictionary());
    import_settings_dialog->popup_centered_ratio(0.7f);
}

Error GaussianEditorPlugin::_import_from_path(const String &p_path, const Dictionary &p_options) {
    ERR_FAIL_COND_V_MSG(!p_path.begins_with("res://"), ERR_INVALID_PARAMETER, "Gaussian imports require project resource paths.");

    EditorFileSystem *fs = EditorFileSystem::get_singleton();
    ERR_FAIL_NULL_V(fs, ERR_UNCONFIGURED);

    // Select importer based on file extension.
    String extension = p_path.get_extension().to_lower();
    String importer_name;
    if (extension == "ply") {
        importer_name = "gaussian_splat_ply";
    } else if (extension == "spz") {
        importer_name = "gaussian_splat_spz";
    } else {
        ERR_FAIL_V_MSG(ERR_FILE_UNRECOGNIZED, vformat("Unsupported Gaussian splat format: .%s (supported: .ply, .spz)", extension));
    }

    Dictionary effective_options = p_options;
    if (effective_options.is_empty() && current_node) {
        Ref<GaussianSplatAsset> node_asset = current_node->get_splat_asset();
        if (node_asset.is_valid()) {
            Dictionary import_metadata = node_asset->get_import_metadata();
            if (import_metadata.has(StringName("options"))) {
                effective_options = import_metadata[StringName("options")];
            }
        }
    }
    if (effective_options.is_empty() && active_asset.is_valid()) {
        Dictionary import_metadata = active_asset->get_import_metadata();
        if (import_metadata.has(StringName("options"))) {
            effective_options = import_metadata[StringName("options")];
        }
    }
    if (effective_options.is_empty()) {
        effective_options = last_import_options;
    }

    HashMap<StringName, Variant> options = _dictionary_to_hashmap(effective_options);
    fs->reimport_file_with_custom_parameters(p_path, importer_name, options);

    Ref<GaussianSplatAsset> asset = _load_gaussian_splat_asset(p_path, true);
    if (asset.is_null()) {
        return ERR_CANT_OPEN;
    }

    active_asset = asset;
    current_source_path = p_path;
    if (active_asset.is_valid()) {
        active_asset->set_source_path(p_path);
    }

    ObjectID node_id = current_node ? current_node->get_instance_id() : ObjectID();
    _track_hot_reload_source(p_path, effective_options, node_id);
    if (active_asset.is_valid() && !active_asset->get_path().is_empty()) {
        _track_hot_reload_source(active_asset->get_path(), Dictionary(), node_id);
    }

    if (current_node) {
        current_node->set_splat_asset(asset);
    }

    if (!current_renderer.is_valid() && current_node) {
        current_renderer = current_node->get_renderer();
    }

    if (current_renderer.is_valid() && !current_node) {
        // Bucket A decision: editor preview keeps the direct bare-renderer upload as the
        // single supported preview exception. Runtime scene paths still go through the
        // director-owned instance/world submission flow.
        Ref<::GaussianData> splat_data = convert_asset_to_gaussian_data(asset);
        Error upload_err = current_renderer->set_gaussian_data(splat_data);
        if (upload_err != OK) {
            GS_LOG_RENDERER_ERROR(vformat("[GaussianEditor] Failed to upload gaussian data for '%s': %d", p_path, upload_err));
        }
    }

    last_import_options = effective_options;
    _refresh_active_asset_metadata(true);
    return OK;
}

void GaussianEditorPlugin::_show_reimport_dialog(const Dictionary &p_options) {
    if (!import_settings_dialog || current_source_path.is_empty()) {
        return;
    }
    import_settings_dialog->configure_for_file(current_source_path, active_asset, true, p_options);
    import_settings_dialog->popup_centered_ratio(0.7f);
}

void GaussianEditorPlugin::_sync_ui_from_asset() {
    // Keep hot-reload watches aligned with the current selection; all visible UI
    // has moved to the asset inspector and import dialog.
    ObjectID node_id = current_node ? current_node->get_instance_id() : ObjectID();
    if (!current_source_path.is_empty()) {
        _track_hot_reload_source(current_source_path, last_import_options, node_id);
    }
    if (active_asset.is_valid() && !active_asset->get_path().is_empty()) {
        _track_hot_reload_source(active_asset->get_path(), Dictionary(), node_id);
    }
}

void GaussianEditorPlugin::_clear_selection() {
    current_renderer.unref();
    current_node = nullptr;
    active_asset.unref();
    current_source_path.clear();
    last_import_options.clear();
}

HashMap<StringName, Variant> GaussianEditorPlugin::_dictionary_to_hashmap(const Dictionary &p_dict) const {
    HashMap<StringName, Variant> map;
    Array keys = p_dict.keys();
    for (int i = 0; i < keys.size(); i++) {
        Variant key = keys[i];
        StringName name;
        if (key.get_type() == Variant::STRING_NAME) {
            name = key;
        } else {
            name = StringName(String(key));
        }
        map.insert(name, p_dict[keys[i]]);
    }
    return map;
}

void GaussianEditorPlugin::_on_import_settings_confirmed(const String &p_source_path, const Dictionary &p_options) {
    Error err = _import_from_path(p_source_path, p_options);
    if (err != OK) {
        EditorNode::get_singleton()->show_warning(
                GaussianEditorServices::format_import_failure_message(p_source_path, err, p_source_path.get_extension()));
    } else {
        last_import_options = p_options;
    }
}

void GaussianEditorPlugin::_refresh_active_asset_metadata(bool p_force_reload) {
    if (!active_asset.is_valid()) {
        last_import_options.clear();
        _sync_ui_from_asset();
        _update_stats();
        return;
    }

    if (p_force_reload && !active_asset->get_path().is_empty()) {
        Ref<GaussianSplatAsset> refreshed_asset = _load_gaussian_splat_asset(active_asset->get_path(), true);
        if (refreshed_asset.is_valid()) {
            active_asset = refreshed_asset;
        }
    }

    const String asset_source_path = GaussianSplatSourcePath::get_asset_source_path(active_asset);
    if (!asset_source_path.is_empty()) {
        current_source_path = asset_source_path;
    }

    Dictionary asset_metadata = active_asset->get_import_metadata();
    if (asset_metadata.has(StringName("options"))) {
        last_import_options = asset_metadata[StringName("options")];
    } else {
        last_import_options.clear();
    }

    _sync_ui_from_asset();
    _update_stats();
}

Ref<Texture2D> GaussianEditorPlugin::_resolve_asset_thumbnail(const Ref<GaussianSplatAsset> &p_asset) {
    if (p_asset.is_null()) {
        return Ref<Texture2D>();
    }

    Ref<Texture2D> existing = p_asset->get_thumbnail();
    if (existing.is_valid()) {
        return existing;
    }

    Dictionary asset_metadata = p_asset->get_import_metadata();
    Dictionary options = asset_metadata.get(StringName("options"), Dictionary());

    bool should_generate = _dict_get_bool(asset_metadata, StringName("thumbnail_generated"),
            _dict_get_bool(options, StringName("preview/generate_thumbnail"), true));
    if (!should_generate) {
        return Ref<Texture2D>();
    }

    if (p_asset->get_splat_count() == 0) {
        return Ref<Texture2D>();
    }

    int style_index = _dict_get_int(asset_metadata, StringName("thumbnail_style"),
            _dict_get_int(options, StringName("preview/thumbnail_style"), 0));
    int size = _dict_get_int(asset_metadata, StringName("thumbnail_size"),
            _dict_get_int(options, StringName("preview/thumbnail_size"), DEFAULT_THUMBNAIL_SIZE));
    size = CLAMP(size, 32, 512);

    if (runtime_thumbnail_generator.is_null()) {
        runtime_thumbnail_generator.instantiate();
    }

    if (runtime_thumbnail_generator.is_valid()) {
        return runtime_thumbnail_generator->generate_thumbnail(p_asset, size,
                GaussianThumbnailGenerator::style_from_int(style_index));
    }

    return Ref<Texture2D>();
}

void GaussianEditorPlugin::_apply_brush_stroke(ObjectID p_node_id, const Vector3 &p_center, float p_radius, const Color &p_color, float p_strength, float p_hardness) {
    GaussianSplatNode3D *node = Object::cast_to<GaussianSplatNode3D>(ObjectDB::get_instance(p_node_id));
    if (!node) {
        return;
    }

    Ref<GaussianSplatRenderer> renderer = node->get_renderer();
    if (!renderer.is_valid()) {
        return;
    }

    Ref<::GaussianData> splat_data = renderer->get_gaussian_data();
    if (splat_data.is_null()) {
        return;
    }

    Vector3 local_center = node->to_local(p_center);

    // Capture affected gaussian state before applying the stroke.
    Dictionary saved_state = splat_data->capture_brush_affected_state(local_center, p_radius);

    EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
    if (undo_redo) {
        undo_redo->create_action(TTR("Paint Gaussian Splats"), UndoRedo::MERGE_DISABLE, node);
        undo_redo->add_do_method(splat_data.ptr(), "apply_brush_stroke", local_center, p_radius, p_color, p_strength, p_hardness);
        undo_redo->add_undo_method(splat_data.ptr(), "restore_brush_stroke", saved_state);
        undo_redo->commit_action();
    } else {
        // Fallback: apply directly without undo support.
        splat_data->apply_brush_stroke(local_center, p_radius, p_color, p_strength, p_hardness);
    }

    node->force_update();
    _update_stats();
    _update_inspector_stats();
}

void GaussianEditorPlugin::_commit_runtime_modifications(ObjectID p_node_id) {
    GaussianSplatNode3D *node = Object::cast_to<GaussianSplatNode3D>(ObjectDB::get_instance(p_node_id));
    if (!node) {
        return;
    }

    Ref<GaussianSplatRenderer> renderer = node->get_renderer();
    if (!renderer.is_valid()) {
        return;
    }

    Ref<::GaussianData> splat_data = renderer->get_gaussian_data();
    if (splat_data.is_null()) {
        return;
    }

    splat_data->commit_runtime_changes();
    node->force_update();
    _update_stats();
    _update_inspector_stats();
}

void GaussianEditorPlugin::_revert_runtime_modifications(ObjectID p_node_id) {
    GaussianSplatNode3D *node = Object::cast_to<GaussianSplatNode3D>(ObjectDB::get_instance(p_node_id));
    if (!node) {
        return;
    }

    Ref<GaussianSplatRenderer> renderer = node->get_renderer();
    if (!renderer.is_valid()) {
        return;
    }

    Ref<::GaussianData> splat_data = renderer->get_gaussian_data();
    if (splat_data.is_null()) {
        return;
    }

    splat_data->revert_runtime_changes();
    node->force_update();
    _update_stats();
    _update_inspector_stats();
}

static uint64_t _get_file_timestamp(const String &p_path) {
    if (p_path.is_empty()) {
        return 0;
    }
    String normalized = p_path;
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        normalized = ps->globalize_path(p_path);
    }
    return FileAccess::get_modified_time(normalized);
}

void GaussianEditorPlugin::_on_import_dialog_watch(const String &p_path) {
    ObjectID node_id = current_node ? current_node->get_instance_id() : ObjectID();
    _track_hot_reload_source(p_path, Dictionary(), node_id);
}

void GaussianEditorPlugin::_track_hot_reload_source(const String &p_path, const Dictionary &p_options, ObjectID p_node_id) {
    if (p_path.is_empty() || !_hot_reload_enabled()) {
        return;
    }

    HotReloadWatch *existing = hot_reload_watches.getptr(p_path);
    if (existing) {
        existing->last_modified = _get_file_timestamp(p_path);
        if (!p_options.is_empty()) {
            existing->import_options = p_options;
        }
        _append_unique_hot_reload_node_id(existing->node_ids, p_node_id);
        _schedule_hot_reload_poll();
        return;
    }

    HotReloadWatch watch;
    watch.last_modified = _get_file_timestamp(p_path);
    watch.import_options = p_options;
    _append_unique_hot_reload_node_id(watch.node_ids, p_node_id);
    hot_reload_watches[p_path] = watch;
    _schedule_hot_reload_poll();
}

void GaussianEditorPlugin::_schedule_hot_reload_poll() {
    if (!_hot_reload_enabled()) {
        hot_reload_watches.clear();
        hot_reload_timer.unref();
        return;
    }

    if (hot_reload_watches.is_empty()) {
        hot_reload_timer.unref();
        return;
    }

    if (hot_reload_timer.is_valid()) {
        return;
    }

    SceneTree *tree = EditorNode::get_singleton() ? EditorNode::get_singleton()->get_tree() : nullptr;
    if (!tree) {
        return;
    }

    Ref<SceneTreeTimer> timer = tree->create_timer(_hot_reload_poll_interval_seconds(), false);
    if (timer.is_valid()) {
        timer->connect("timeout", callable_mp(this, &GaussianEditorPlugin::_on_hot_reload_timer_timeout));
        hot_reload_timer = timer;
    }
}

void GaussianEditorPlugin::_on_hot_reload_timer_timeout() {
    hot_reload_timer.unref();
    if (!_hot_reload_enabled()) {
        hot_reload_watches.clear();
        return;
    }
    PackedStringArray stale_paths;
    for (KeyValue<String, HotReloadWatch> &E : hot_reload_watches) {
        _collect_live_hot_reload_nodes(E.value.node_ids);
        if (E.value.node_ids.is_empty() && current_source_path != E.key &&
                (!active_asset.is_valid() || active_asset->get_path() != E.key)) {
            stale_paths.push_back(E.key);
            continue;
        }
        _process_hot_reload_for_watch(E.key, E.value);
    }
    for (int i = 0; i < stale_paths.size(); i++) {
        hot_reload_watches.erase(stale_paths[i]);
    }
    _schedule_hot_reload_poll();
}

void GaussianEditorPlugin::_process_hot_reload_for_watch(const String &p_path, HotReloadWatch &p_watch) {
    uint64_t current_stamp = _get_file_timestamp(p_path);
    if (current_stamp == 0 || current_stamp <= p_watch.last_modified) {
        return;
    }

    p_watch.last_modified = current_stamp;

    Vector<GaussianSplatNode3D *> watched_nodes = _collect_live_hot_reload_nodes(p_watch.node_ids);
    GaussianSplatNode3D *target_node = watched_nodes.is_empty() ? nullptr : watched_nodes[0];

    GaussianSplatNode3D *previous_node = current_node;
    Ref<GaussianSplatRenderer> previous_renderer = current_renderer;
    Ref<GaussianSplatAsset> previous_asset = active_asset;
    String previous_source_path = current_source_path;
    Dictionary previous_import_options = last_import_options;
    const bool restore_selection_asset = previous_asset.is_valid() &&
            (previous_asset->get_path() == p_path || previous_source_path == p_path);

    const bool context_swapped = target_node && target_node != current_node;

    auto restore_context = [&]() {
        if (!context_swapped) {
            return;
        }
        current_node = previous_node;
        current_renderer = previous_renderer;
        active_asset = previous_asset;
        current_source_path = previous_source_path;
        last_import_options = previous_import_options;
        if (active_asset.is_valid()) {
            _refresh_active_asset_metadata(restore_selection_asset);
        } else {
            _sync_ui_from_asset();
        }
    };

	const String source_extension = p_path.get_extension().to_lower();
	Ref<GaussianSplatAsset> refreshed_asset;
	if (source_extension == "ply" || source_extension == "spz") {
		Dictionary options = p_watch.import_options;
        if (options.is_empty()) {
            for (int i = 0; i < watched_nodes.size(); i++) {
                Ref<GaussianSplatAsset> target_asset = watched_nodes[i]->get_splat_asset();
                if (target_asset.is_valid()) {
                    Dictionary asset_metadata = target_asset->get_import_metadata();
                    if (asset_metadata.has(StringName("options"))) {
                        options = asset_metadata[StringName("options")];
                        break;
                    }
                }
            }
        }

        if (target_node) {
            current_node = target_node;
            current_renderer = target_node->get_renderer();
            active_asset = target_node->get_splat_asset();
            current_source_path = _get_node_source_path(target_node);
            last_import_options = options;
        }

        if (options.is_empty()) {
            options = last_import_options;
        }

		hot_reload_processing = true;
		const Error import_err = _import_from_path(p_path, options);
		hot_reload_processing = false;
		if (import_err == OK) {
			refreshed_asset = active_asset;
		}
	} else {
		Ref<Resource> res = ResourceLoader::load(p_path, String(), ResourceFormatLoader::CACHE_MODE_REPLACE);
		if (res.is_valid()) {
			res->reload_from_file();
			refreshed_asset = res;
			if (refreshed_asset.is_valid() && active_asset.is_valid() && active_asset->get_path() == p_path) {
				active_asset = refreshed_asset;
			}
		}
	}

	_apply_hot_reload_asset_to_nodes(p_path, watched_nodes, refreshed_asset);
    if (watched_nodes.is_empty() && current_node) {
        current_node->force_update();
    }
    if (!watched_nodes.is_empty() && (current_source_path == p_path ||
            (active_asset.is_valid() && active_asset->get_path() == p_path))) {
        _refresh_active_asset_metadata(true);
    }
    restore_context();
    _update_stats();
    _update_inspector_stats();
}

void GaussianEditorPlugin::_apply_hot_reload_asset_to_nodes(const String &p_path, const Vector<GaussianSplatNode3D *> &p_nodes,
        const Ref<GaussianSplatAsset> &p_refreshed_asset) {
    for (int i = 0; i < p_nodes.size(); i++) {
        GaussianSplatNode3D *node = p_nodes[i];
        if (!node) {
            continue;
        }

        if (p_refreshed_asset.is_valid()) {
            const String node_source_path = _get_node_source_path(node);
            Ref<GaussianSplatAsset> node_asset = node->get_splat_asset();
            const bool same_source = !node_source_path.is_empty() && node_source_path == p_path;
            const bool same_asset_path = node_asset.is_valid() && !p_refreshed_asset->get_path().is_empty() &&
                    node_asset->get_path() == p_refreshed_asset->get_path();
            if ((same_source || same_asset_path) && node_asset != p_refreshed_asset) {
                node->set_splat_asset(p_refreshed_asset);
            }
        }
        node->force_update();
    }
}

bool GaussianEditorPlugin::_test_process_hot_reload_path_now(const String &p_path, const Ref<GaussianSplatAsset> &p_refreshed_asset) {
    HotReloadWatch *watch = hot_reload_watches.getptr(p_path);
    if (!watch) {
        return false;
    }

    Vector<GaussianSplatNode3D *> watched_nodes = _collect_live_hot_reload_nodes(watch->node_ids);
    if (watched_nodes.is_empty()) {
        return false;
    }

    watch->last_modified = 0;
    _apply_hot_reload_asset_to_nodes(p_path, watched_nodes, p_refreshed_asset);
    return true;
}

void GaussianEditorPlugin::request_asset_reimport(const Ref<GaussianSplatAsset> &p_asset, const Dictionary &p_override) {
    if (p_asset.is_null()) {
        EditorNode::get_singleton()->show_warning(TTR("No Gaussian asset available for reimport."));
        return;
    }

    active_asset = p_asset;
    current_source_path = GaussianSplatSourcePath::get_asset_source_path(p_asset);

    if (current_source_path.is_empty()) {
        EditorNode::get_singleton()->show_warning(TTR("The selected asset does not have a recorded source path."));
        return;
    }

    _refresh_active_asset_metadata();

    Dictionary override_options = p_override;
    if (override_options.is_empty()) {
        Dictionary import_metadata = p_asset->get_import_metadata();
        if (import_metadata.has(StringName("options"))) {
            override_options = import_metadata[StringName("options")];
        } else {
            override_options = last_import_options;
        }
    }
    _show_reimport_dialog(override_options);
}

void GaussianEditorPlugin::_update_stats() {
}

void GaussianEditorPlugin::_update_inspector_stats() {
    if (inspector_stats_bindings.is_empty()) {
        return;
    }

    for (int i = inspector_stats_bindings.size() - 1; i >= 0; i--) {
        InspectorStatsBinding &binding = inspector_stats_bindings.write[i];

        if (!binding.label) {
            inspector_stats_bindings.remove_at(i);
            continue;
        }

        Object *obj = ObjectDB::get_instance(binding.node_id);
        GaussianSplatNode3D *node = Object::cast_to<GaussianSplatNode3D>(obj);

        if (!node) {
            binding.label->set_text("Gaussian splat node unavailable.");
            inspector_stats_bindings.remove_at(i);
            continue;
        }

        Ref<GaussianSplatRenderer> renderer = node->get_renderer();
        binding.label->set_text(GaussianEditorServices::format_gaussian_splat_stats(node, renderer));
    }
}

void GaussianEditorPlugin::register_inspector_stats(ObjectID p_node_id, Label *p_label) {
    if (!p_label) {
        return;
    }

    InspectorStatsBinding binding;
    binding.node_id = p_node_id;
    binding.label = p_label;
    inspector_stats_bindings.push_back(binding);

    if (!p_label->is_connected("tree_exiting", callable_mp(this, &GaussianEditorPlugin::_on_stats_label_tree_exiting))) {
        p_label->connect("tree_exiting", callable_mp(this, &GaussianEditorPlugin::_on_stats_label_tree_exiting).bind(p_label));
    }

    GaussianSplatNode3D *node = Object::cast_to<GaussianSplatNode3D>(ObjectDB::get_instance(p_node_id));
    if (node) {
        Ref<GaussianSplatRenderer> renderer = node->get_renderer();
        p_label->set_text(GaussianEditorServices::format_gaussian_splat_stats(node, renderer));
    }
}

void GaussianEditorPlugin::unregister_inspector_stats(Label *p_label) {
    if (!p_label) {
        return;
    }

    for (int i = inspector_stats_bindings.size() - 1; i >= 0; i--) {
        if (inspector_stats_bindings[i].label == p_label) {
            inspector_stats_bindings.remove_at(i);
        }
    }
}

void GaussianEditorPlugin::_on_stats_label_tree_exiting(Node *p_label) {
    unregister_inspector_stats(Object::cast_to<Label>(p_label));
}

#endif // TOOLS_ENABLED
