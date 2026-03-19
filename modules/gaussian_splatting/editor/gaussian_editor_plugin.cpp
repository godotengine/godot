#ifdef TOOLS_ENABLED

#include "gaussian_editor_plugin.h"
#include "gaussian_editor_services.h"
#include "editor/editor_node.h"
#include "editor/editor_interface.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_file_dialog.h"
#include "core/config/project_settings.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/control.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "servers/text_server.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_manager.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "../io/gaussian_import_preset.h"
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

static double _dict_get_double(const Dictionary &p_dict, const StringName &p_key, double p_default = 0.0) {
    return GaussianEditorServices::dict_get_double(p_dict, p_key, p_default);
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

static int find_quality_index(const String &p_name) {
    static const char *QUALITY_ORDER[] = { "mobile", "desktop", "high", "ultra", "development" };
    String lower = p_name.to_lower();
    for (int i = 0; i < 5; i++) {
        if (lower == QUALITY_ORDER[i]) {
            return i;
        }
    }
    if (lower == "custom") {
        return 2; // fallback to high quality for legacy values
    }
    return 2;
}

static String quality_from_index(int p_index) {
    static const char *QUALITY_ORDER[] = { "mobile", "desktop", "high", "ultra", "development" };
    if (p_index < 0 || p_index >= 5) {
        return "high";
    }
    return QUALITY_ORDER[p_index];
}

} // namespace

// GaussianEditorPlugin implementation

GaussianEditorPlugin::GaussianEditorPlugin() {
    _build_panel_ui();

    runtime_thumbnail_generator.instantiate();
    editor_integration.instantiate();
    editor_integration->setup(this, import_dialog);

    _sync_ui_from_asset();
    _update_stats();
}

void GaussianEditorPlugin::_build_panel_ui() {
    panel_container = memnew(VBoxContainer);
    panel_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    panel_container->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
    panel_toggle_button = add_control_to_bottom_panel(panel_container, TTR("Gaussian Splatting"));

    HBoxContainer *action_row = memnew(HBoxContainer);
    panel_container->add_child(action_row);

    import_button = memnew(Button);
    import_button->set_text(TTR("Import Gaussian"));
    import_button->connect("pressed", callable_mp(this, &GaussianEditorPlugin::_import_gaussian_data));
    action_row->add_child(import_button);

    optimize_button = memnew(Button);
    optimize_button->set_text(TTR("Optimize Data"));
    optimize_button->connect("pressed", callable_mp(this, &GaussianEditorPlugin::_optimize_gaussian_data));
    action_row->add_child(optimize_button);

    HBoxContainer *quality_row = memnew(HBoxContainer);
    panel_container->add_child(quality_row);

    Label *quality_label = memnew(Label);
    quality_label->set_text(TTR("Quality Preset:"));
    quality_row->add_child(quality_label);

    quality_selector = memnew(OptionButton);
    quality_selector->add_item(TTR("Mobile"));
    quality_selector->add_item(TTR("Desktop"));
    quality_selector->add_item(TTR("High Quality"));
    quality_selector->add_item(TTR("Ultra Quality"));
    quality_selector->add_item(TTR("Development"));
    quality_selector->select(find_quality_index("high"));
    quality_selector->connect("item_selected", callable_mp(this, &GaussianEditorPlugin::_on_quality_preset_selected));
    quality_row->add_child(quality_selector);

    HBoxContainer *compression_row = memnew(HBoxContainer);
    panel_container->add_child(compression_row);

    Label *compression_label = memnew(Label);
    compression_label->set_text(TTR("Compression:"));
    compression_row->add_child(compression_label);

    VBoxContainer *compression_list = memnew(VBoxContainer);
    compression_row->add_child(compression_list);

    compression_positions = memnew(CheckBox);
    compression_positions->set_text(TTR("Positions"));
    compression_positions->connect("toggled", callable_mp(this, &GaussianEditorPlugin::_on_compression_toggled).bind(compression_positions));
    compression_list->add_child(compression_positions);

    compression_colors = memnew(CheckBox);
    compression_colors->set_text(TTR("Colors"));
    compression_colors->connect("toggled", callable_mp(this, &GaussianEditorPlugin::_on_compression_toggled).bind(compression_colors));
    compression_list->add_child(compression_colors);

    compression_scales = memnew(CheckBox);
    compression_scales->set_text(TTR("Scales"));
    compression_scales->connect("toggled", callable_mp(this, &GaussianEditorPlugin::_on_compression_toggled).bind(compression_scales));
    compression_list->add_child(compression_scales);

    compression_rotations = memnew(CheckBox);
    compression_rotations->set_text(TTR("Rotations"));
    compression_rotations->connect("toggled", callable_mp(this, &GaussianEditorPlugin::_on_compression_toggled).bind(compression_rotations));
    compression_list->add_child(compression_rotations);

    HBoxContainer *preview_row = memnew(HBoxContainer);
    panel_container->add_child(preview_row);

    thumbnail_preview = memnew(TextureRect);
    thumbnail_preview->set_custom_minimum_size(Size2(DEFAULT_THUMBNAIL_SIZE, DEFAULT_THUMBNAIL_SIZE));
    thumbnail_preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
    preview_row->add_child(thumbnail_preview);

    stats_label = memnew(Label);
    stats_label->set_text(TTR("No Gaussian asset selected."));
    stats_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    stats_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
    stats_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    preview_row->add_child(stats_label);

    import_dialog = memnew(EditorFileDialog);
    import_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
    import_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
    import_dialog->connect("file_selected", callable_mp(this, &GaussianEditorPlugin::_on_import_file_selected));
    add_child(import_dialog);

    import_settings_dialog = memnew(GaussianImportDialog);
    import_settings_dialog->connect("import_requested", callable_mp(this, &GaussianEditorPlugin::_on_import_settings_confirmed));
    import_settings_dialog->connect("watch_path_requested", callable_mp(this, &GaussianEditorPlugin::_on_import_dialog_watch));
    add_child(import_settings_dialog);
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

    if (panel_container) {
        remove_control_from_bottom_panel(panel_container);
        panel_container = nullptr;
    }
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
        current_source_path = node->get_ply_file_path();
    } else if (GaussianSplatRenderer *renderer = Object::cast_to<GaussianSplatRenderer>(p_object)) {
        current_renderer = Ref<GaussianSplatRenderer>(renderer);
    } else if (GaussianSplatAsset *asset_obj = Object::cast_to<GaussianSplatAsset>(p_object)) {
        Ref<GaussianSplatAsset> asset(asset_obj);
        active_asset = asset;
        current_source_path = asset->get_source_path();
    }

    if (active_asset.is_valid()) {
        _refresh_active_asset_metadata();
    } else {
        _sync_ui_from_asset();
        _update_thumbnail_preview();
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
    if (panel_toggle_button) {
        panel_toggle_button->set_pressed(p_visible);
    }
    if (p_visible) {
        _update_stats();
    }
}

void GaussianEditorPlugin::_import_gaussian_data() {
    if (!import_dialog) {
        return;
    }

    String initial_dir = current_source_path.is_empty() ? String("res://") : current_source_path.get_base_dir();
    if (import_dialog->get_access() == EditorFileDialog::ACCESS_FILESYSTEM && initial_dir.begins_with("res://")) {
        if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
            initial_dir = ps->globalize_path(initial_dir);
        }
    }
    if (!initial_dir.is_empty()) {
        import_dialog->set_current_dir(initial_dir);
    }

    import_dialog->popup_file_dialog();
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

    pending_reimport = false;
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

    HashMap<StringName, Variant> options = _dictionary_to_hashmap(p_options);
    fs->reimport_file_with_custom_parameters(p_path, importer_name, options);

    Ref<GaussianSplatAsset> asset = ResourceLoader::load(p_path);
    if (asset.is_null()) {
        return ERR_CANT_OPEN;
    }

    active_asset = asset;
    current_source_path = p_path;
    if (active_asset.is_valid()) {
        active_asset->set_source_path(p_path);
    }

    ObjectID node_id = current_node ? current_node->get_instance_id() : ObjectID();
    _track_hot_reload_source(p_path, p_options, node_id);
    if (active_asset.is_valid() && !active_asset->get_path().is_empty()) {
        _track_hot_reload_source(active_asset->get_path(), Dictionary(), node_id);
    }

    if (current_node) {
        current_node->set_splat_asset(asset);
        current_node->set_ply_file_path(p_path);
    }

    if (!current_renderer.is_valid() && current_node) {
        current_renderer = current_node->get_renderer();
    }

    if (current_renderer.is_valid() && !current_node) {
        Ref<::GaussianData> splat_data = convert_asset_to_gaussian_data(asset);
        Error upload_err = current_renderer->set_gaussian_data(splat_data);
        if (upload_err != OK) {
            GS_LOG_RENDERER_ERROR(vformat("[GaussianEditor] Failed to upload gaussian data for '%s': %d", p_path, upload_err));
        }
    }

    last_import_options = p_options;
    _refresh_active_asset_metadata();
    return OK;
}

Dictionary GaussianEditorPlugin::_gather_import_options_dict() const {
    Dictionary options;
    if (!last_import_options.is_empty()) {
        Array keys = last_import_options.keys();
        for (int i = 0; i < keys.size(); i++) {
            Variant key = keys[i];
            options[key] = last_import_options[keys[i]];
        }
    }

    Dictionary asset_metadata = active_asset.is_valid() ? active_asset->get_import_metadata() : Dictionary();

    int selected_index = quality_selector ? quality_selector->get_selected() : find_quality_index("high");
    String preset = quality_from_index(selected_index);
    const GaussianImportPresetDefinition &preset_def = gaussian_get_import_preset_by_name(preset);

    auto ensure_option = [&options](const StringName &p_key, const Variant &p_value) {
        if (!options.has(p_key)) {
            options[p_key] = p_value;
        }
    };

    ensure_option(StringName("general/asset_type"),
            active_asset.is_valid() ? int(active_asset->get_asset_type()) : preset_def.default_asset_type);
    ensure_option(StringName("quality/enable_lod"), preset_def.enable_lod);
    ensure_option(StringName("quality/optimize_for_gpu"), preset_def.optimize_for_gpu);
    ensure_option(StringName("quality/max_splats"), preset_def.max_splats);
    ensure_option(StringName("quality/density_multiplier"), preset_def.density_multiplier);
    ensure_option(StringName("validation/validate_required_properties"), true);
    ensure_option(StringName("validation/warn_missing_optional"), true);
    ensure_option(StringName("processing/normalize_opacity"), true);
    ensure_option(StringName("processing/sort_by_opacity"), false);
    ensure_option(StringName("metadata/include_statistics"), preset_def.include_statistics);
    ensure_option(StringName("metadata/include_memory_estimate"), preset_def.include_memory_estimate);

    if (!options.has(StringName("preview/generate_thumbnail"))) {
        bool generate_default = asset_metadata.has(StringName("thumbnail_generated")) ?
                bool(asset_metadata[StringName("thumbnail_generated")]) :
                (active_asset.is_valid() && active_asset->get_thumbnail().is_valid());
        options[StringName("preview/generate_thumbnail")] = generate_default;
    }

    if (!options.has(StringName("preview/thumbnail_style"))) {
        int style_default = asset_metadata.has(StringName("thumbnail_style")) ? int(asset_metadata[StringName("thumbnail_style")]) :
                                                                     preset_def.thumbnail_style;
        options[StringName("preview/thumbnail_style")] = style_default;
    }

    if (!options.has(StringName("preview/thumbnail_size"))) {
        int size_default = asset_metadata.has(StringName("thumbnail_size")) ? int(asset_metadata[StringName("thumbnail_size")]) :
                                                                  preset_def.default_thumbnail_size;
        options[StringName("preview/thumbnail_size")] = size_default;
    }

    // Apply current UI selections for quality preset and compression toggles.
    options[StringName("quality/preset")] = preset;

    bool quantize_positions = compression_positions && compression_positions->is_pressed();
    bool quantize_colors = compression_colors && compression_colors->is_pressed();
    bool quantize_scales = compression_scales && compression_scales->is_pressed();
    bool quantize_rotations = compression_rotations && compression_rotations->is_pressed();

    options[StringName("compression/quantize_positions")] = quantize_positions;
    options[StringName("compression/quantize_colors")] = quantize_colors;
    options[StringName("compression/quantize_scales")] = quantize_scales;
    options[StringName("compression/quantize_rotations")] = quantize_rotations;
    options.erase(StringName("compression/pack_opacity"));

    bool customized = false;
    customized |= _dict_get_int(options, StringName("general/asset_type"), preset_def.default_asset_type) !=
            preset_def.default_asset_type;
    customized |= _dict_get_int(options, StringName("quality/max_splats"), preset_def.max_splats) != preset_def.max_splats;
    customized |= !Math::is_equal_approx(
            _dict_get_double(options, StringName("quality/density_multiplier"), preset_def.density_multiplier),
            preset_def.density_multiplier);
    customized |= _dict_get_bool(options, StringName("quality/enable_lod"), preset_def.enable_lod) != preset_def.enable_lod;
    customized |= _dict_get_bool(options, StringName("quality/optimize_for_gpu"), preset_def.optimize_for_gpu) !=
            preset_def.optimize_for_gpu;
    customized |= _dict_get_bool(options, StringName("compression/quantize_positions"), preset_def.quantize_positions) !=
            preset_def.quantize_positions;
    customized |= _dict_get_bool(options, StringName("compression/quantize_colors"), preset_def.quantize_colors) !=
            preset_def.quantize_colors;
    customized |= _dict_get_bool(options, StringName("compression/quantize_scales"), preset_def.quantize_scales) !=
            preset_def.quantize_scales;
    customized |= _dict_get_bool(options, StringName("compression/quantize_rotations"), preset_def.quantize_rotations) !=
            preset_def.quantize_rotations;
    customized |= _dict_get_int(options, StringName("preview/thumbnail_style"), preset_def.thumbnail_style) !=
            preset_def.thumbnail_style;
    customized |= _dict_get_bool(options, StringName("metadata/include_statistics"), preset_def.include_statistics) !=
            preset_def.include_statistics;
    customized |= _dict_get_bool(options, StringName("metadata/include_memory_estimate"),
            preset_def.include_memory_estimate) != preset_def.include_memory_estimate;

    options[StringName("quality/customized")] = customized;

    return options;
}

void GaussianEditorPlugin::_on_quality_preset_selected(int p_index) {
    (void)p_index;
    if (updating_ui) {
        return;
    }
    _apply_import_settings();
}

void GaussianEditorPlugin::_on_compression_toggled(bool p_pressed, CheckBox *p_box) {
    (void)p_pressed;
    (void)p_box;
    if (updating_ui) {
        return;
    }
    _apply_import_settings();
}

void GaussianEditorPlugin::_apply_import_settings() {
    if (current_source_path.is_empty()) {
        return;
    }
    Dictionary options = _gather_import_options_dict();
    _show_reimport_dialog(options);
}

void GaussianEditorPlugin::_show_reimport_dialog(const Dictionary &p_options) {
    if (!import_settings_dialog || current_source_path.is_empty()) {
        return;
    }
    pending_reimport = true;
    import_settings_dialog->configure_for_file(current_source_path, active_asset, true, p_options);
    import_settings_dialog->popup_centered_ratio(0.7f);
}

void GaussianEditorPlugin::_sync_ui_from_asset() {
    updating_ui = true;

    const bool has_reimport_source = !current_source_path.is_empty();
    const String reimport_tooltip = TTR("Select an imported Gaussian asset to edit reimport settings.");

    if (quality_selector) {
        String preset = "high";
        if (!last_import_options.is_empty() && last_import_options.has(StringName("quality/preset"))) {
            preset = String(last_import_options[StringName("quality/preset")]);
        } else if (active_asset.is_valid()) {
            preset = active_asset->get_import_quality_preset();
        }
        quality_selector->select(find_quality_index(preset));
        quality_selector->set_disabled(!has_reimport_source);
        quality_selector->set_tooltip_text(has_reimport_source ? String() : reimport_tooltip);
    }

    uint32_t flags = active_asset.is_valid() ? active_asset->get_compression_flags() : static_cast<uint32_t>(GaussianSplatAsset::COMPRESSION_NONE);
    if (compression_positions) {
        compression_positions->set_pressed((flags & GaussianSplatAsset::COMPRESSION_POSITIONS) != 0);
        compression_positions->set_disabled(!has_reimport_source);
        compression_positions->set_tooltip_text(has_reimport_source ? String() : reimport_tooltip);
    }
    if (compression_colors) {
        compression_colors->set_pressed((flags & GaussianSplatAsset::COMPRESSION_COLORS) != 0);
        compression_colors->set_disabled(!has_reimport_source);
        compression_colors->set_tooltip_text(has_reimport_source ? String() : reimport_tooltip);
    }
    if (compression_scales) {
        compression_scales->set_pressed((flags & GaussianSplatAsset::COMPRESSION_SCALES) != 0);
        compression_scales->set_disabled(!has_reimport_source);
        compression_scales->set_tooltip_text(has_reimport_source ? String() : reimport_tooltip);
    }
    if (compression_rotations) {
        compression_rotations->set_pressed((flags & GaussianSplatAsset::COMPRESSION_ROTATIONS) != 0);
        compression_rotations->set_disabled(!has_reimport_source);
        compression_rotations->set_tooltip_text(has_reimport_source ? String() : reimport_tooltip);
    }

    if (optimize_button) {
        const bool has_runtime_data = current_renderer.is_valid() && current_renderer->get_gaussian_data().is_valid();
        optimize_button->set_disabled(!has_runtime_data);
        optimize_button->set_tooltip_text(has_runtime_data ? String() : TTR("Select a Gaussian node or renderer with loaded data first."));
    }

    if (import_button) {
        import_button->set_disabled(!(current_node || current_renderer.is_valid()));
    }

    updating_ui = false;

    ObjectID node_id = current_node ? current_node->get_instance_id() : ObjectID();
    if (!current_source_path.is_empty()) {
        _track_hot_reload_source(current_source_path, last_import_options, node_id);
    }
    if (active_asset.is_valid() && !active_asset->get_path().is_empty()) {
        _track_hot_reload_source(active_asset->get_path(), Dictionary(), node_id);
    }
}

void GaussianEditorPlugin::_update_thumbnail_preview() {
    if (!thumbnail_preview) {
        return;
    }

    Ref<Texture2D> texture;
    if (active_asset.is_valid()) {
        texture = _resolve_asset_thumbnail(active_asset);
    }

    thumbnail_preview->set_texture(texture);
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
    pending_reimport = false;
}

void GaussianEditorPlugin::_refresh_active_asset_metadata() {
    if (!active_asset.is_valid()) {
        last_import_options.clear();
        _sync_ui_from_asset();
        _update_thumbnail_preview();
        _update_stats();
        return;
    }

    Dictionary asset_metadata = active_asset->get_import_metadata();
    if (asset_metadata.has(StringName("options"))) {
        last_import_options = asset_metadata[StringName("options")];
    } else {
        last_import_options.clear();
    }

    _sync_ui_from_asset();
    _update_thumbnail_preview();
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

    HotReloadWatch watch;
    watch.last_modified = _get_file_timestamp(p_path);
    watch.import_options = p_options;
    watch.node_id = p_node_id;
    if (hot_reload_processing) {
        HotReloadWatch *existing = hot_reload_watches.getptr(p_path);
        if (existing) {
            existing->last_modified = watch.last_modified;
            if (!p_options.is_empty()) {
                existing->import_options = p_options;
            }
            if (p_node_id != ObjectID()) {
                existing->node_id = p_node_id;
            }
            return;
        }
    }
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
    for (KeyValue<String, HotReloadWatch> &E : hot_reload_watches) {
        _process_hot_reload_for_watch(E.key, E.value);
    }
    _schedule_hot_reload_poll();
}

void GaussianEditorPlugin::_process_hot_reload_for_watch(const String &p_path, HotReloadWatch &p_watch) {
    uint64_t current_stamp = _get_file_timestamp(p_path);
    if (current_stamp == 0 || current_stamp <= p_watch.last_modified) {
        return;
    }

    p_watch.last_modified = current_stamp;

    GaussianSplatNode3D *target_node = nullptr;
    if (p_watch.node_id != ObjectID()) {
        Object *obj = ObjectDB::get_instance(p_watch.node_id);
        target_node = Object::cast_to<GaussianSplatNode3D>(obj);
        if (!target_node) {
            p_watch.node_id = ObjectID();
        }
    }

    GaussianSplatNode3D *previous_node = current_node;
    Ref<GaussianSplatRenderer> previous_renderer = current_renderer;
    Ref<GaussianSplatAsset> previous_asset = active_asset;
    String previous_source_path = current_source_path;
    Dictionary previous_import_options = last_import_options;

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
            _refresh_active_asset_metadata();
        } else {
            _sync_ui_from_asset();
            _update_thumbnail_preview();
        }
    };

    const String source_extension = p_path.get_extension().to_lower();
    if (source_extension == "ply" || source_extension == "spz") {
        Dictionary options = p_watch.import_options;
        if (options.is_empty() && target_node) {
            Ref<GaussianSplatAsset> target_asset = target_node->get_splat_asset();
            if (target_asset.is_valid()) {
                Dictionary asset_metadata = target_asset->get_import_metadata();
                if (asset_metadata.has(StringName("options"))) {
                    options = asset_metadata[StringName("options")];
                }
            }
        }

        if (target_node) {
            current_node = target_node;
            current_renderer = target_node->get_renderer();
            active_asset = target_node->get_splat_asset();
            current_source_path = target_node->get_ply_file_path();
            last_import_options = options;
        }

        if (options.is_empty()) {
            options = last_import_options;
        }

        hot_reload_processing = true;
        if (options.is_empty()) {
            options = _gather_import_options_dict();
        }
        _import_from_path(p_path, options);
        hot_reload_processing = false;
    } else {
        Ref<Resource> res = ResourceLoader::load(p_path, String(), ResourceFormatLoader::CACHE_MODE_REPLACE);
        if (res.is_valid()) {
            res->reload_from_file();
        }
    }

    GaussianSplatNode3D *node_to_update = target_node ? target_node : current_node;
    if (node_to_update) {
        node_to_update->force_update();
    }
    restore_context();
    _update_stats();
    _update_inspector_stats();
}

void GaussianEditorPlugin::request_asset_reimport(const Ref<GaussianSplatAsset> &p_asset, const Dictionary &p_override) {
    if (p_asset.is_null()) {
        EditorNode::get_singleton()->show_warning(TTR("No Gaussian asset available for reimport."));
        return;
    }

    active_asset = p_asset;
    current_source_path = p_asset->get_source_path();

    if (current_source_path.is_empty()) {
        EditorNode::get_singleton()->show_warning(TTR("The selected asset does not have a recorded source path."));
        return;
    }

    Dictionary override_options = p_override;
    if (override_options.is_empty()) {
        Dictionary asset_metadata = p_asset->get_import_metadata();
        if (asset_metadata.has(StringName("options"))) {
            override_options = asset_metadata[StringName("options")];
        }
    }

    _refresh_active_asset_metadata();

    if (override_options.is_empty()) {
        override_options = _gather_import_options_dict();
    }

    _show_reimport_dialog(override_options);
}

void GaussianEditorPlugin::_optimize_gaussian_data() {
    if (!current_renderer.is_valid() && current_node) {
        current_renderer = current_node->get_renderer();
    }

    if (!current_renderer.is_valid() || !current_renderer->get_gaussian_data().is_valid()) {
        return;
    }

    Ref<::GaussianData> splat_data = current_renderer->get_gaussian_data();
    if (splat_data.is_valid()) {
        splat_data->build_octree(8);
    }

    _update_stats();
}

void GaussianEditorPlugin::_update_stats() {
    if (!stats_label) {
        return;
    }

    String stats_text = TTR("Gaussian Asset Statistics:") + "\n";

    if (active_asset.is_valid()) {
        stats_text += GaussianEditorServices::format_asset_metadata_summary(active_asset, active_asset->get_import_metadata(),
                DEFAULT_THUMBNAIL_SIZE) + "\n";
    } else {
        stats_text += TTR("No Gaussian asset selected.") + "\n";
    }

    if (current_renderer.is_valid() && current_renderer->get_gaussian_data().is_valid()) {
        Ref<::GaussianData> splat_data = current_renderer->get_gaussian_data();
        Dictionary render_stats = current_renderer->get_render_stats();

        stats_text += "\n" + TTR("Renderer Statistics:") + "\n";
        stats_text += vformat(TTR("Total Gaussians: %d"), splat_data->get_count()) + "\n";
        stats_text += vformat(TTR("Visible Splats: %d"),
                int(_dict_get_int(render_stats, StringName("visible_splats"), splat_data->get_count()))) + "\n";
        stats_text += vformat(TTR("Memory Usage: %.2f MB"), splat_data->get_memory_usage() / (1024.0 * 1024.0)) + "\n";
        stats_text += vformat(TTR("Sort Time: %.2f ms"),
                _dict_get_double(render_stats, StringName("sort_time_ms"), 0.0)) + "\n";
        stats_text += vformat(TTR("Render Time: %.2f ms"),
                _dict_get_double(render_stats, StringName("render_time_ms"), 0.0)) + "\n";

        if (render_stats.has(StringName("tile_assignment_ms"))) {
            stats_text += vformat(TTR("Tile Assign: %.2f ms"),
                    _dict_get_double(render_stats, StringName("tile_assignment_ms"), 0.0)) + "\n";
        }
        if (render_stats.has(StringName("tile_rasterization_ms"))) {
            stats_text += vformat(TTR("Tile Raster: %.2f ms"),
                    _dict_get_double(render_stats, StringName("tile_rasterization_ms"), 0.0)) + "\n";
        }

        bool show_grid = _dict_get_bool(render_stats, StringName("debug_show_tile_grid"), false);
        bool show_heatmap = _dict_get_bool(render_stats, StringName("debug_show_density_heatmap"), false);
        bool show_hud = _dict_get_bool(render_stats, StringName("debug_show_performance_hud"), false);
        if (show_grid || show_heatmap || show_hud) {
            stats_text += TTR("Overlays: ");
            stats_text += vformat("Grid=%s", show_grid ? "On" : "Off");
            stats_text += vformat(", Heatmap=%s", show_heatmap ? "On" : "Off");
            stats_text += vformat(", HUD=%s", show_hud ? "On" : "Off") + "\n";
        }

        if (render_stats.has(StringName("debug_preview_mode"))) {
            int64_t mode_value = _dict_get_int(render_stats, StringName("debug_preview_mode"), 0);
            static const char *mode_names[] = { "Off", "Wireframe", "Points", "Depth", "Heatmap", "Runtime Modifications" };
            String mode_label = (mode_value >= 0 && mode_value < 6) ? mode_names[mode_value] : "Unknown";
            stats_text += vformat(TTR("Preview Mode: %s"), mode_label) + "\n";
        }
    }

    if (current_node) {
        stats_text += "\n" + TTR("Node Statistics:") + "\n";
        stats_text += vformat(TTR("Visible Splats: %d"), current_node->get_visible_splat_count()) + "\n";
        stats_text += vformat(TTR("Total Splats: %d"), current_node->get_total_splat_count()) + "\n";
        stats_text += vformat(TTR("GPU Memory: %.2f MB"), current_node->get_gpu_memory_mb()) + "\n";
        stats_text += vformat(TTR("LOD Spheres: %s"), current_node->is_showing_lod_spheres() ? "On" : "Off") + "\n";
        stats_text += vformat(TTR("Performance Overlay: %s"),
                current_node->is_showing_performance_overlay() ? "On" : "Off") + "\n";
    }

    stats_label->set_text(stats_text.strip_edges());
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
