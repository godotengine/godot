#ifdef TOOLS_ENABLED

#include "gaussian_import_dialog.h"

#include "core/io/file_access.h"
#include "core/math/math_funcs.h"
#include "core/os/time.h"
#include "core/string/print_string.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/window.h"
#include "servers/text_server.h"

#include "../core/gaussian_data.h"
#include "../io/ply_loader.h"
#include "../io/spz_loader.h"

#include <cfloat>

namespace {
static Dictionary _dictionary_from_options(const Vector<Pair<StringName, Variant>> &p_pairs) {
    Dictionary dict;
    for (int i = 0; i < p_pairs.size(); i++) {
        dict[p_pairs[i].first] = p_pairs[i].second;
    }
    return dict;
}

static String _format_option_value(const String &p_key, const Variant &p_value) {
    if (p_key == "general/asset_type") {
        int asset_type = int(p_value);
        if (asset_type == 0) {
            return "Static";
        }
        if (asset_type == 1) {
            return "Dynamic";
        }
    } else if (p_key == "preview/thumbnail_style") {
        return GaussianThumbnailGenerator::style_to_display_name(
                GaussianThumbnailGenerator::style_from_int(int(p_value)));
    }

    switch (p_value.get_type()) {
        case Variant::BOOL:
            return bool(p_value) ? "true" : "false";
        case Variant::INT:
            return String::num_int64(int64_t(p_value));
        case Variant::FLOAT:
            return String::num_real(double(p_value));
        default:
            return p_value.operator String();
    }
}
}

GaussianImportDialog *GaussianImportDialog::singleton = nullptr;

GaussianImportDialog *GaussianImportDialog::get_singleton() {
    return singleton;
}

void GaussianImportDialog::_bind_methods() {
    ADD_SIGNAL(MethodInfo("import_requested", PropertyInfo(Variant::STRING, "source_path"),
            PropertyInfo(Variant::DICTIONARY, "options")));
    ADD_SIGNAL(MethodInfo("watch_path_requested", PropertyInfo(Variant::STRING, "path")));
}

GaussianImportDialog::GaussianImportDialog() {
    singleton = this;
    thumbnail_generator.instantiate();
    set_title(TTR("Gaussian Splat Import"));
    set_min_size(Size2(720, 520));
    _build_ui();
}

GaussianImportDialog::~GaussianImportDialog() {
    singleton = nullptr;
}

void GaussianImportDialog::_build_ui() {
    VBoxContainer *root = memnew(VBoxContainer);
    add_child(root);
    root->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
    root->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    root->set_v_size_flags(Control::SIZE_EXPAND_FILL);

    file_path_label = memnew(Label);
    file_path_label->set_text(TTR("Select a Gaussian splat file to import (.ply, .spz)."));
    file_path_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    file_path_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    root->add_child(file_path_label);

    TabContainer *tabs = memnew(TabContainer);
    tabs->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    tabs->set_v_size_flags(Control::SIZE_EXPAND_FILL);
    root->add_child(tabs);

    tabs->add_child(_create_quality_tab());
    tabs->add_child(_create_compression_tab());
    tabs->add_child(_create_preview_tab());
    tabs->add_child(_create_metadata_tab());
}

VBoxContainer *GaussianImportDialog::_create_quality_tab() {
    VBoxContainer *tab = memnew(VBoxContainer);
    tab->set_name(TTR("Quality"));
    tab->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    tab->set_v_size_flags(Control::SIZE_EXPAND_FILL);

    GridContainer *grid = memnew(GridContainer);
    grid->set_columns(2);
    grid->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    tab->add_child(grid);

    Label *preset_label = memnew(Label);
    preset_label->set_text(TTR("Quality Preset"));
    grid->add_child(preset_label);

    preset_selector = memnew(OptionButton);
    preset_selector->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    const Vector<GaussianImportPresetDefinition> &presets = gaussian_get_import_presets();
    for (int i = 0; i < presets.size(); i++) {
        preset_selector->add_item(presets[i].display_name, i);
    }
    custom_preset_index = presets.size();
    preset_selector->add_item(TTR("Custom"), custom_preset_index);
    preset_selector->connect("item_selected", callable_mp(this, &GaussianImportDialog::_on_preset_selected));
    grid->add_child(preset_selector);

    Label *asset_type_label = memnew(Label);
    asset_type_label->set_text(TTR("Asset Type"));
    grid->add_child(asset_type_label);

    asset_type_selector = memnew(OptionButton);
    asset_type_selector->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    asset_type_selector->add_item(TTR("Static"), 0);
    asset_type_selector->add_item(TTR("Dynamic"), 1);
    asset_type_selector->connect("item_selected", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    grid->add_child(asset_type_selector);

    Label *max_splats_label = memnew(Label);
    max_splats_label->set_text(TTR("Max Splats"));
    grid->add_child(max_splats_label);

    max_splats_spin = memnew(SpinBox);
    max_splats_spin->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    max_splats_spin->set_allow_greater(true);
    max_splats_spin->set_max(5000000);
    max_splats_spin->set_min(0);
    max_splats_spin->set_step(5000);
    max_splats_spin->connect("value_changed", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    grid->add_child(max_splats_spin);

    Label *density_label = memnew(Label);
    density_label->set_text(TTR("Density Multiplier"));
    grid->add_child(density_label);

    density_spin = memnew(SpinBox);
    density_spin->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    density_spin->set_min(0.1);
    density_spin->set_max(1.0);
    density_spin->set_step(0.05);
    density_spin->connect("value_changed", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    grid->add_child(density_spin);

    lod_checkbox = memnew(CheckBox);
    lod_checkbox->set_text(TTR("Enable Level of Detail"));
    lod_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(lod_checkbox);

    optimize_checkbox = memnew(CheckBox);
    optimize_checkbox->set_text(TTR("Optimize for GPU Upload"));
    optimize_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(optimize_checkbox);

    validate_checkbox = memnew(CheckBox);
    validate_checkbox->set_text(TTR("Validate Required Properties"));
    validate_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(validate_checkbox);

    warn_checkbox = memnew(CheckBox);
    warn_checkbox->set_text(TTR("Warn about Missing Optional Properties"));
    warn_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(warn_checkbox);

    normalize_checkbox = memnew(CheckBox);
    normalize_checkbox->set_text(TTR("Normalize Opacity Range"));
    normalize_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(normalize_checkbox);

    sort_checkbox = memnew(CheckBox);
    sort_checkbox->set_text(TTR("Sort Splats by Opacity"));
    sort_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(sort_checkbox);

    return tab;
}

VBoxContainer *GaussianImportDialog::_create_compression_tab() {
    VBoxContainer *tab = memnew(VBoxContainer);
    tab->set_name(TTR("Compression"));
    tab->set_h_size_flags(Control::SIZE_EXPAND_FILL);

    compress_positions_checkbox = memnew(CheckBox);
    compress_positions_checkbox->set_text(TTR("Quantize Positions"));
    compress_positions_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(compress_positions_checkbox);

    compress_colors_checkbox = memnew(CheckBox);
    compress_colors_checkbox->set_text(TTR("Quantize Colors"));
    compress_colors_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(compress_colors_checkbox);

    compress_scales_checkbox = memnew(CheckBox);
    compress_scales_checkbox->set_text(TTR("Quantize Scales"));
    compress_scales_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(compress_scales_checkbox);

    compress_rotations_checkbox = memnew(CheckBox);
    compress_rotations_checkbox->set_text(TTR("Quantize Rotations"));
    compress_rotations_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(compress_rotations_checkbox);

    pack_opacity_checkbox = memnew(CheckBox);
    pack_opacity_checkbox->set_text(TTR("Pack Opacity (Deprecated)"));
    pack_opacity_checkbox->set_disabled(true);
    pack_opacity_checkbox->set_tooltip_text(TTR("Deprecated option retained for compatibility. It is ignored during import."));
    pack_opacity_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(pack_opacity_checkbox);

    return tab;
}

VBoxContainer *GaussianImportDialog::_create_preview_tab() {
    VBoxContainer *tab = memnew(VBoxContainer);
    tab->set_name(TTR("Preview"));
    tab->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    tab->set_v_size_flags(Control::SIZE_EXPAND_FILL);

    thumbnail_checkbox = memnew(CheckBox);
    thumbnail_checkbox->set_text(TTR("Generate Thumbnail"));
    thumbnail_checkbox->set_pressed(true);
    thumbnail_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(thumbnail_checkbox);

    GridContainer *thumbnail_options = memnew(GridContainer);
    thumbnail_options->set_columns(2);
    thumbnail_options->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    tab->add_child(thumbnail_options);

    Label *style_label = memnew(Label);
    style_label->set_text(TTR("Style"));
    thumbnail_options->add_child(style_label);

    thumbnail_style_option = memnew(OptionButton);
    thumbnail_style_option->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    thumbnail_style_option->add_item(GaussianThumbnailGenerator::style_to_display_name(GaussianThumbnailGenerator::THUMBNAIL_STYLE_COLOR),
            GaussianThumbnailGenerator::THUMBNAIL_STYLE_COLOR);
    thumbnail_style_option->add_item(GaussianThumbnailGenerator::style_to_display_name(GaussianThumbnailGenerator::THUMBNAIL_STYLE_DENSITY),
            GaussianThumbnailGenerator::THUMBNAIL_STYLE_DENSITY);
    thumbnail_style_option->add_item(GaussianThumbnailGenerator::style_to_display_name(GaussianThumbnailGenerator::THUMBNAIL_STYLE_NORMALS),
            GaussianThumbnailGenerator::THUMBNAIL_STYLE_NORMALS);
    thumbnail_style_option->add_item(GaussianThumbnailGenerator::style_to_display_name(GaussianThumbnailGenerator::THUMBNAIL_STYLE_HEATMAP),
            GaussianThumbnailGenerator::THUMBNAIL_STYLE_HEATMAP);
    thumbnail_style_option->connect("item_selected", callable_mp(this, &GaussianImportDialog::_on_thumbnail_style_selected));
    thumbnail_options->add_child(thumbnail_style_option);

    Label *size_label = memnew(Label);
    size_label->set_text(TTR("Size"));
    thumbnail_options->add_child(size_label);

    thumbnail_size_spin = memnew(SpinBox);
    thumbnail_size_spin->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    thumbnail_size_spin->set_min(32);
    thumbnail_size_spin->set_max(512);
    thumbnail_size_spin->set_step(16);
    thumbnail_size_spin->set_value(128);
    thumbnail_size_spin->connect("value_changed", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    thumbnail_options->add_child(thumbnail_size_spin);

    thumbnail_preview = memnew(TextureRect);
    thumbnail_preview->set_custom_minimum_size(Size2(0, 180));
    thumbnail_preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
    thumbnail_preview->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    thumbnail_preview->set_v_size_flags(Control::SIZE_EXPAND_FILL);
    tab->add_child(thumbnail_preview);

    return tab;
}

VBoxContainer *GaussianImportDialog::_create_metadata_tab() {
    VBoxContainer *tab = memnew(VBoxContainer);
    tab->set_name(TTR("Metadata"));
    tab->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    tab->set_v_size_flags(Control::SIZE_EXPAND_FILL);

    include_stats_checkbox = memnew(CheckBox);
    include_stats_checkbox->set_text(TTR("Include Loader Statistics"));
    include_stats_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(include_stats_checkbox);

    include_memory_checkbox = memnew(CheckBox);
    include_memory_checkbox->set_text(TTR("Include Memory Estimates"));
    include_memory_checkbox->connect("toggled", callable_mp(this, &GaussianImportDialog::_on_settings_changed).unbind(1));
    tab->add_child(include_memory_checkbox);

    splat_summary_label = memnew(Label);
    splat_summary_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    splat_summary_label->set_text(TTR("No data loaded."));
    tab->add_child(splat_summary_label);

    memory_label = memnew(Label);
    memory_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    tab->add_child(memory_label);

    comparison_label = memnew(Label);
    comparison_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    tab->add_child(comparison_label);

    return tab;
}

void GaussianImportDialog::_load_preview_asset() {
    preview_valid = false;
    preview_bounds_valid = false;
    loader_statistics.clear();
    preview_bounds = AABB();

    if (source_path.is_empty()) {
        return;
    }

    // Determine format from extension and load appropriately.
    String extension = source_path.get_extension().to_lower();
    Ref<::GaussianData> import_data;
    Error err = OK;

    if (extension == "ply") {
        Ref<PLYLoader> loader;
        loader.instantiate();
        err = loader->load_file(source_path);
        if (err != OK) {
            splat_summary_label->set_text(vformat(TTR("Failed to load %s (error %d)"), source_path, (int)err));
            return;
        }
        loader_statistics = loader->get_load_statistics();
        import_data = loader->get_gaussian_data();
    } else if (extension == "spz") {
        Ref<SPZLoader> loader;
        loader.instantiate();
        err = loader->load_file(source_path);
        if (err != OK) {
            splat_summary_label->set_text(vformat(TTR("Failed to load %s (error %d)"), source_path, (int)err));
            return;
        }
        loader_statistics = loader->get_load_statistics();
        import_data = loader->get_gaussian_data();
    } else {
        splat_summary_label->set_text(vformat(TTR("Unsupported format: .%s (supported: .ply, .spz)"), extension));
        return;
    }

    if (import_data.is_null() || import_data->get_count() == 0) {
        splat_summary_label->set_text(vformat(TTR("File did not contain Gaussian data: %s"), source_path));
        return;
    }

    preview_asset.instantiate();
    preview_asset->populate_from_gaussian_data(import_data);
    preview_asset->set_source_path(source_path);

    // Compute bounds for metadata display.
    PackedFloat32Array positions = preview_asset->get_positions();
    const int count = preview_asset->get_splat_count();
    if (count > 0 && positions.size() >= count * 3) {
        Vector3 min_pos(FLT_MAX, FLT_MAX, FLT_MAX);
        Vector3 max_pos(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int i = 0; i < count; i++) {
            int base = i * 3;
            Vector3 pos(positions[base], positions[base + 1], positions[base + 2]);
            min_pos = min_pos.min(pos);
            max_pos = max_pos.max(pos);
        }
        preview_bounds = AABB(min_pos, max_pos - min_pos);
        preview_bounds_valid = true;
    }

    preview_valid = true;
}

void GaussianImportDialog::_update_format_specific_controls() {
    // Format-specific UI controls:
    //
    // PLY format features:
    // - Property validation: PLY files can have missing required or optional
    //   properties. The validation checkbox enables strict checking for required
    //   properties (positions, colors, scales, rotations, opacity). The warning
    //   checkbox enables warnings for missing optional properties (spherical
    //   harmonics, normals, etc.).
    //
    // SPZ format features:
    // - SPZ is a fixed format with consistent structure. Validation is built-in
    //   and these options are not applicable.
    // - SPZ already uses internal compression (fixed-point positions, smallest-
    //   three quaternions, 8-bit alphas/colors). Additional quantization options
    //   in this dialog may have diminished benefit.
    //
    const bool show_ply_options = source_is_ply;

    if (validate_checkbox) {
        validate_checkbox->set_visible(show_ply_options);
        if (!show_ply_options) {
            // SPZ validation is automatic; reset to default.
            current_config.validate_required = true;
        }
    }

    if (warn_checkbox) {
        warn_checkbox->set_visible(show_ply_options);
        if (!show_ply_options) {
            // SPZ has a fixed property set; reset to default.
            current_config.warn_missing_optional = true;
        }
    }

    // Update dialog title to reflect the source format.
    String ext_upper = source_is_ply ? "PLY" : "SPZ";
    set_title(vformat(TTR("Gaussian %s Import"), ext_upper));
}

void GaussianImportDialog::_update_thumbnail_controls_state() {
    const bool enable_thumbnail_settings = current_config.generate_thumbnail;
    if (thumbnail_style_option) {
        thumbnail_style_option->set_disabled(!enable_thumbnail_settings);
        thumbnail_style_option->set_tooltip_text(enable_thumbnail_settings ?
                        String() :
                        TTR("Enable \"Generate Thumbnail\" to edit style."));
    }
    if (thumbnail_size_spin) {
        thumbnail_size_spin->set_editable(enable_thumbnail_settings);
        thumbnail_size_spin->set_tooltip_text(enable_thumbnail_settings ?
                        String() :
                        TTR("Enable \"Generate Thumbnail\" to edit size."));
    }
}

void GaussianImportDialog::_apply_preset_defaults(const GaussianImportPresetDefinition &p_preset) {
    current_config.preset = p_preset.id;
    current_config.asset_type = p_preset.default_asset_type;
    current_config.max_splats = p_preset.max_splats;
    current_config.density_multiplier = p_preset.density_multiplier;
    current_config.enable_lod = p_preset.enable_lod;
    current_config.optimize_for_gpu = p_preset.optimize_for_gpu;
    current_config.quantize_positions = p_preset.quantize_positions;
    current_config.quantize_colors = p_preset.quantize_colors;
    current_config.quantize_scales = p_preset.quantize_scales;
    current_config.quantize_rotations = p_preset.quantize_rotations;
    current_config.pack_opacity = false;
    current_config.generate_thumbnail = true;
    current_config.thumbnail_style = p_preset.thumbnail_style;
    current_config.thumbnail_size = p_preset.default_thumbnail_size;
    current_config.include_statistics = p_preset.include_statistics;
    current_config.include_memory_estimate = p_preset.include_memory_estimate;
    current_config.validate_required = true;
    current_config.warn_missing_optional = true;
    current_config.normalize_opacity = true;
    current_config.sort_by_opacity = false;
    current_config.custom_settings = false;
}

void GaussianImportDialog::_apply_configuration_to_ui() {
    updating_ui = true;

    int preset_index = gaussian_find_import_preset_index(current_config.preset);
    if (preset_selector) {
        if (preset_index >= 0 && !current_config.custom_settings) {
            preset_selector->select(preset_index);
        } else {
            preset_selector->select(custom_preset_index);
        }
    }

    if (asset_type_selector) {
        asset_type_selector->select(CLAMP(current_config.asset_type, 0, 1));
    }
    if (max_splats_spin) {
        max_splats_spin->set_value(current_config.max_splats);
    }
    if (density_spin) {
        density_spin->set_value(current_config.density_multiplier);
    }
    if (lod_checkbox) {
        lod_checkbox->set_pressed(current_config.enable_lod);
    }
    if (optimize_checkbox) {
        optimize_checkbox->set_pressed(current_config.optimize_for_gpu);
    }
    if (validate_checkbox) {
        validate_checkbox->set_pressed(current_config.validate_required);
    }
    if (warn_checkbox) {
        warn_checkbox->set_pressed(current_config.warn_missing_optional);
    }
    if (normalize_checkbox) {
        normalize_checkbox->set_pressed(current_config.normalize_opacity);
    }
    if (sort_checkbox) {
        sort_checkbox->set_pressed(current_config.sort_by_opacity);
    }
    if (compress_positions_checkbox) {
        compress_positions_checkbox->set_pressed(current_config.quantize_positions);
    }
    if (compress_colors_checkbox) {
        compress_colors_checkbox->set_pressed(current_config.quantize_colors);
    }
    if (compress_scales_checkbox) {
        compress_scales_checkbox->set_pressed(current_config.quantize_scales);
    }
    if (compress_rotations_checkbox) {
        compress_rotations_checkbox->set_pressed(current_config.quantize_rotations);
    }
    if (pack_opacity_checkbox) {
        pack_opacity_checkbox->set_pressed(current_config.pack_opacity);
    }
    if (thumbnail_checkbox) {
        thumbnail_checkbox->set_pressed(current_config.generate_thumbnail);
    }
    if (thumbnail_style_option) {
        thumbnail_style_option->select(current_config.thumbnail_style);
    }
    if (thumbnail_size_spin) {
        thumbnail_size_spin->set_value(current_config.thumbnail_size);
    }
    if (include_stats_checkbox) {
        include_stats_checkbox->set_pressed(current_config.include_statistics);
    }
    if (include_memory_checkbox) {
        include_memory_checkbox->set_pressed(current_config.include_memory_estimate);
    }
    _update_thumbnail_controls_state();

    updating_ui = false;
}

void GaussianImportDialog::_apply_dictionary_override(const Dictionary &p_options) {
    if (p_options.is_empty()) {
        return;
    }
    _configuration_from_dictionary(current_config, p_options);
}

void GaussianImportDialog::_configuration_from_dictionary(ImportConfiguration &r_config, const Dictionary &p_dict) {
    Array keys = p_dict.keys();
    for (int i = 0; i < keys.size(); i++) {
        String key = String(keys[i]);
        Variant value = p_dict[keys[i]];
        if (key == "quality/preset") {
            r_config.preset = String(value).to_lower();
        } else if (key == "general/asset_type") {
            r_config.asset_type = int(value);
        } else if (key == "quality/max_splats") {
            r_config.max_splats = int(value);
        } else if (key == "quality/density_multiplier") {
            r_config.density_multiplier = double(value);
        } else if (key == "quality/enable_lod") {
            r_config.enable_lod = bool(value);
        } else if (key == "quality/optimize_for_gpu") {
            r_config.optimize_for_gpu = bool(value);
        } else if (key == "validation/validate_required_properties") {
            r_config.validate_required = bool(value);
        } else if (key == "validation/warn_missing_optional") {
            r_config.warn_missing_optional = bool(value);
        } else if (key == "processing/normalize_opacity") {
            r_config.normalize_opacity = bool(value);
        } else if (key == "processing/sort_by_opacity") {
            r_config.sort_by_opacity = bool(value);
        } else if (key == "compression/quantize_positions") {
            r_config.quantize_positions = bool(value);
        } else if (key == "compression/quantize_colors") {
            r_config.quantize_colors = bool(value);
        } else if (key == "compression/quantize_scales") {
            r_config.quantize_scales = bool(value);
        } else if (key == "compression/quantize_rotations") {
            r_config.quantize_rotations = bool(value);
        } else if (key == "compression/pack_opacity") {
            r_config.pack_opacity = bool(value);
        } else if (key == "preview/generate_thumbnail") {
            r_config.generate_thumbnail = bool(value);
        } else if (key == "preview/thumbnail_style") {
            r_config.thumbnail_style = int(value);
        } else if (key == "preview/thumbnail_size") {
            r_config.thumbnail_size = int(value);
        } else if (key == "metadata/include_statistics") {
            r_config.include_statistics = bool(value);
        } else if (key == "metadata/include_memory_estimate") {
            r_config.include_memory_estimate = bool(value);
        } else if (key == "quality/customized") {
            r_config.custom_settings = bool(value);
        }
    }
}

Dictionary GaussianImportDialog::_configuration_to_dictionary(const ImportConfiguration &p_config) const {
    Vector<Pair<StringName, Variant>> pairs;
    pairs.push_back({ StringName("quality/preset"), p_config.preset });
    pairs.push_back({ StringName("general/asset_type"), p_config.asset_type });
    pairs.push_back({ StringName("quality/max_splats"), p_config.max_splats });
    pairs.push_back({ StringName("quality/density_multiplier"), p_config.density_multiplier });
    pairs.push_back({ StringName("quality/enable_lod"), p_config.enable_lod });
    pairs.push_back({ StringName("quality/optimize_for_gpu"), p_config.optimize_for_gpu });
    pairs.push_back({ StringName("validation/validate_required_properties"), p_config.validate_required });
    pairs.push_back({ StringName("validation/warn_missing_optional"), p_config.warn_missing_optional });
    pairs.push_back({ StringName("processing/normalize_opacity"), p_config.normalize_opacity });
    pairs.push_back({ StringName("processing/sort_by_opacity"), p_config.sort_by_opacity });
    pairs.push_back({ StringName("compression/quantize_positions"), p_config.quantize_positions });
    pairs.push_back({ StringName("compression/quantize_colors"), p_config.quantize_colors });
    pairs.push_back({ StringName("compression/quantize_scales"), p_config.quantize_scales });
    pairs.push_back({ StringName("compression/quantize_rotations"), p_config.quantize_rotations });
    pairs.push_back({ StringName("preview/generate_thumbnail"), p_config.generate_thumbnail });
    pairs.push_back({ StringName("preview/thumbnail_style"), p_config.thumbnail_style });
    pairs.push_back({ StringName("preview/thumbnail_size"), p_config.thumbnail_size });
    pairs.push_back({ StringName("metadata/include_statistics"), p_config.include_statistics });
    pairs.push_back({ StringName("metadata/include_memory_estimate"), p_config.include_memory_estimate });
    pairs.push_back({ StringName("quality/customized"), p_config.custom_settings });
    return _dictionary_from_options(pairs);
}

void GaussianImportDialog::_update_configuration_from_ui() {
    if (!preset_selector) {
        return;
    }

    int preset_index = preset_selector->get_selected();
    if (preset_index >= 0 && preset_index < gaussian_get_import_presets().size()) {
        const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_index(preset_index);
        current_config.preset = preset.id;
    }

    if (asset_type_selector) {
        current_config.asset_type = asset_type_selector->get_selected();
    }
    if (max_splats_spin) {
        current_config.max_splats = int(max_splats_spin->get_value());
    }
    if (density_spin) {
        current_config.density_multiplier = density_spin->get_value();
    }
    if (lod_checkbox) {
        current_config.enable_lod = lod_checkbox->is_pressed();
    }
    if (optimize_checkbox) {
        current_config.optimize_for_gpu = optimize_checkbox->is_pressed();
    }
    if (validate_checkbox) {
        current_config.validate_required = validate_checkbox->is_pressed();
    }
    if (warn_checkbox) {
        current_config.warn_missing_optional = warn_checkbox->is_pressed();
    }
    if (normalize_checkbox) {
        current_config.normalize_opacity = normalize_checkbox->is_pressed();
    }
    if (sort_checkbox) {
        current_config.sort_by_opacity = sort_checkbox->is_pressed();
    }
    if (compress_positions_checkbox) {
        current_config.quantize_positions = compress_positions_checkbox->is_pressed();
    }
    if (compress_colors_checkbox) {
        current_config.quantize_colors = compress_colors_checkbox->is_pressed();
    }
    if (compress_scales_checkbox) {
        current_config.quantize_scales = compress_scales_checkbox->is_pressed();
    }
    if (compress_rotations_checkbox) {
        current_config.quantize_rotations = compress_rotations_checkbox->is_pressed();
    }
    current_config.pack_opacity = false;
    if (thumbnail_checkbox) {
        current_config.generate_thumbnail = thumbnail_checkbox->is_pressed();
    }
    if (thumbnail_style_option) {
        current_config.thumbnail_style = thumbnail_style_option->get_selected();
    }
    if (thumbnail_size_spin) {
        current_config.thumbnail_size = int(thumbnail_size_spin->get_value());
    }
    if (include_stats_checkbox) {
        current_config.include_statistics = include_stats_checkbox->is_pressed();
    }
    if (include_memory_checkbox) {
        current_config.include_memory_estimate = include_memory_checkbox->is_pressed();
    }

    _update_customization_flag();
}

void GaussianImportDialog::_update_customization_flag() {
    const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_name(current_config.preset);

    bool differs = false;
    differs |= current_config.asset_type != preset.default_asset_type;
    differs |= (preset.max_splats != 0 && current_config.max_splats != preset.max_splats);
    differs |= !Math::is_equal_approx(current_config.density_multiplier, preset.density_multiplier);
    differs |= current_config.enable_lod != preset.enable_lod;
    differs |= current_config.optimize_for_gpu != preset.optimize_for_gpu;
    differs |= current_config.quantize_positions != preset.quantize_positions;
    differs |= current_config.quantize_colors != preset.quantize_colors;
    differs |= current_config.quantize_scales != preset.quantize_scales;
    differs |= current_config.quantize_rotations != preset.quantize_rotations;
    differs |= current_config.thumbnail_style != preset.thumbnail_style;
    differs |= current_config.thumbnail_size != preset.default_thumbnail_size;
    differs |= current_config.generate_thumbnail != true; // Presets default to generating thumbnails.
    differs |= current_config.include_statistics != preset.include_statistics;
    differs |= current_config.include_memory_estimate != preset.include_memory_estimate;

    current_config.custom_settings = differs;
}

void GaussianImportDialog::_update_preview() {
    if (!thumbnail_preview) {
        return;
    }
    if (!preview_valid) {
        thumbnail_preview->set_texture(Ref<Texture2D>());
        return;
    }

    if (!current_config.generate_thumbnail) {
        thumbnail_preview->set_texture(Ref<Texture2D>());
        return;
    }

    int file_size = MAX(32, current_config.thumbnail_size);
    Ref<Texture2D> texture;
    if (thumbnail_generator.is_valid() && preview_asset.is_valid()) {
        texture = thumbnail_generator->generate_thumbnail(preview_asset, file_size,
                GaussianThumbnailGenerator::style_from_int(current_config.thumbnail_style));
    }

    if (texture.is_valid()) {
        thumbnail_preview->set_texture(texture);
    } else {
        thumbnail_preview->set_texture(Ref<Texture2D>());
    }
}

static int _compute_final_splat_count(int p_original, const GaussianImportDialog::ImportConfiguration &p_config) {
    int final_count = p_original;
    if (p_config.max_splats > 0) {
        final_count = MIN(final_count, p_config.max_splats);
    }
    final_count = MIN(final_count, int(Math::round(p_original * p_config.density_multiplier)));
    final_count = MAX(final_count, 0);
    return final_count;
}

void GaussianImportDialog::_update_statistics() {
    if (!splat_summary_label) {
        return;
    }

    if (!preview_valid) {
        splat_summary_label->set_text(TTR("No preview available."));
        return;
    }

    const int original_count = preview_asset->get_splat_count();
    const int final_count = _compute_final_splat_count(original_count, current_config);
    const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_name(current_config.preset);

    String display_name = preset.display_name;
    if (current_config.custom_settings) {
        display_name += TTR(" (modified)");
    }

    String text = vformat(TTR("Preset: %s\nOriginal Splats: %d\nImported Splats: %d"), display_name, original_count, final_count);
    if (preview_bounds_valid) {
        Vector3 min_pos = preview_bounds.position;
        Vector3 bounds_size = preview_bounds.size;
        text += "\n" + vformat(TTR("Bounds: [%s] size [%s]"), min_pos, bounds_size);
    }
    if (loader_statistics.has(StringName("format"))) {
        text += "\n" + vformat(TTR("Format: %s"), String(loader_statistics[StringName("format")]));
    }
    splat_summary_label->set_text(text);
}

void GaussianImportDialog::_update_memory_estimate() {
    if (!memory_label) {
        return;
    }
    if (!preview_valid || !current_config.include_memory_estimate) {
        memory_label->set_text(String());
        return;
    }

    const int original_count = preview_asset->get_splat_count();
    const int final_count = _compute_final_splat_count(original_count, current_config);
    uint32_t compression_flags = 0;
    if (current_config.quantize_positions) {
        compression_flags |= GaussianSplatAsset::COMPRESSION_POSITIONS;
    }
    if (current_config.quantize_colors) {
        compression_flags |= GaussianSplatAsset::COMPRESSION_COLORS;
    }
    if (current_config.quantize_scales) {
        compression_flags |= GaussianSplatAsset::COMPRESSION_SCALES;
    }
    if (current_config.quantize_rotations) {
        compression_flags |= GaussianSplatAsset::COMPRESSION_ROTATIONS;
    }

    Dictionary stats = thumbnail_generator->compute_memory_statistics(final_count, compression_flags, false);
    double total_mb = stats.get(StringName("total_mb"), 0.0);
    memory_label->set_text(vformat(TTR("Estimated Memory: %.2f MB"), total_mb));
}

void GaussianImportDialog::_update_comparison() {
    if (!comparison_label) {
        return;
    }

    if (!reimport_mode) {
        comparison_label->set_text(String());
        return;
    }

    if (!comparison_metadata.has(StringName("options"))) {
        comparison_label->set_text(TTR("No previous import metadata available."));
        return;
    }

    Dictionary previous_options = comparison_metadata.get(StringName("options"), Dictionary());
    Dictionary new_options = get_selected_options();

    PackedStringArray changes;
    Array keys = new_options.keys();
    PackedStringArray sorted_keys;
    sorted_keys.resize(keys.size());
    for (int i = 0; i < keys.size(); i++) {
        sorted_keys.write[i] = String(keys[i]);
    }
    sorted_keys.sort();

    for (int i = 0; i < sorted_keys.size(); i++) {
        String key = sorted_keys[i];
        StringName key_name = StringName(key);
        Variant new_value = new_options[key_name];
        Variant old_value = previous_options.get(key_name, Variant());
        if (old_value == Variant()) {
            continue;
        }
        if (old_value == new_value) {
            continue;
        }
        changes.push_back(vformat("%s: %s -> %s", key, _format_option_value(key, old_value), _format_option_value(key, new_value)));
    }

    if (changes.is_empty()) {
        comparison_label->set_text(TTR("Settings match previous import."));
    } else {
        comparison_label->set_text(TTR("Changes:\n") + String("\n").join(changes));
    }
}

void GaussianImportDialog::_refresh_all() {
    if (updating_ui) {
        return;
    }
    _apply_configuration_to_ui();
    _update_preview();
    _update_statistics();
    _update_memory_estimate();
    _update_comparison();
}

void GaussianImportDialog::_on_preset_selected(int p_index) {
    if (updating_ui) {
        return;
    }
    if (p_index >= 0 && p_index < gaussian_get_import_presets().size()) {
        const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_index(p_index);
        _apply_preset_defaults(preset);
        _apply_configuration_to_ui();
        _refresh_all();
    } else if (p_index == custom_preset_index) {
        current_config.custom_settings = true;
        _apply_configuration_to_ui();
        _refresh_all();
    }
}

void GaussianImportDialog::_on_thumbnail_style_selected(int p_index) {
    if (updating_ui) {
        return;
    }
    current_config.thumbnail_style = p_index;
    _update_customization_flag();
    _refresh_all();
}

void GaussianImportDialog::_on_settings_changed() {
    if (updating_ui) {
        return;
    }
    _update_configuration_from_ui();
    _refresh_all();
}

void GaussianImportDialog::_on_confirmed() {
    Dictionary options = get_selected_options();
    emit_signal(StringName("import_requested"), source_path, options);
}

void GaussianImportDialog::ok_pressed() {
    _on_confirmed();
    AcceptDialog::ok_pressed();
}

void GaussianImportDialog::configure_for_file(const String &p_source_path, const Ref<GaussianSplatAsset> &p_existing_asset,
        bool p_reimport, const Dictionary &p_override_options) {
    source_path = p_source_path;
    reimport_mode = p_reimport;
    override_options = p_override_options;
    baseline_options.clear();
    comparison_metadata.clear();

    // Detect source format from extension.
    String extension = source_path.get_extension().to_lower();
    source_is_ply = (extension == "ply");

    if (!source_path.is_empty()) {
        emit_signal(StringName("watch_path_requested"), source_path);
    }

    if (p_existing_asset.is_valid()) {
        comparison_metadata = p_existing_asset->get_import_metadata();
        if (comparison_metadata.has(StringName("options"))) {
            baseline_options = comparison_metadata[StringName("options")];
        }
    }

    if (override_options.has(StringName("options"))) {
        baseline_options = override_options[StringName("options")];
    }

    if (file_path_label) {
        file_path_label->set_text(vformat(TTR("Source: %s"), source_path));
    }

    String target_preset;
    if (override_options.has(StringName("quality/preset"))) {
        target_preset = String(override_options[StringName("quality/preset")]).to_lower();
    } else if (baseline_options.has(StringName("quality/preset"))) {
        target_preset = String(baseline_options[StringName("quality/preset")]).to_lower();
    } else {
        target_preset = gaussian_get_import_preset_by_index(1).id; // default to desktop/high
    }

    _apply_preset_defaults(gaussian_get_import_preset_by_name(target_preset));

    if (!baseline_options.is_empty()) {
        _apply_dictionary_override(baseline_options);
    }
    if (!override_options.is_empty()) {
        _apply_dictionary_override(override_options);
    }
    _update_customization_flag();

    _apply_configuration_to_ui();
    _update_format_specific_controls();
    _load_preview_asset();
    _refresh_all();
}

Dictionary GaussianImportDialog::get_selected_options() const {
    return _configuration_to_dictionary(current_config);
}

#endif // TOOLS_ENABLED
