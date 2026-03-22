#ifndef GAUSSIAN_IMPORT_DIALOG_H
#define GAUSSIAN_IMPORT_DIALOG_H

#ifdef TOOLS_ENABLED

#include "scene/gui/dialogs.h"

#include "core/math/aabb.h"
#include "core/object/ref_counted.h"
#include "core/templates/vector.h"
#include "core/variant/dictionary.h"

#include "../core/gaussian_splat_asset.h"
#include "../io/gaussian_import_preset.h"
#include "gaussian_thumbnail_generator.h"

class Button;
class CheckBox;
class Control;
class HBoxContainer;
class Label;
class OptionButton;
class SceneTreeTimer;
class SpinBox;
class TabContainer;
class TextureRect;
class VBoxContainer;

class GaussianImportDialog : public AcceptDialog {
    GDCLASS(GaussianImportDialog, AcceptDialog);

    static GaussianImportDialog *singleton;

public:
    static GaussianImportDialog *get_singleton();

    /**
     * @struct ImportConfiguration
     * @brief Import settings for Gaussian splat files.
     *
     * ## Compression Options (quantize_* and pack_opacity)
     *
     * These options control **precision reduction** for GPU memory optimization,
     * not file-level compression. They reduce the bit depth of stored values:
     *
     * - **quantize_positions**: Reduces position precision from 32-bit float to
     *   16-bit normalized values per chunk. Provides ~2x memory savings with
     *   minimal visual impact for typical scenes.
     *
     * - **quantize_colors**: Reduces color precision from 32-bit float per channel
     *   to 8-bit normalized values. Imperceptible for most content.
     *
     * - **quantize_scales**: Reduces scale precision from 32-bit float to 16-bit.
     *   May cause subtle aliasing on very small or very large splats.
     *
     * - **quantize_rotations**: Reduces rotation quaternion precision from 32-bit
     *   float components to 16-bit. Generally imperceptible.
     *
     * - **pack_opacity**: Deprecated compatibility option. It is ignored by
     *   current import/runtime paths and retained only for metadata migration.
     *
     * These settings affect GPU buffer storage, not the source file format.
     * SPZ files already use internal compression and may not benefit as much
     * from additional quantization.
     */
    struct ImportConfiguration {
        String preset;
        int asset_type = 0;      ///< 0 = static, 1 = dynamic
        int max_splats = 0;      ///< Maximum splats to import (0 = unlimited)
        double density_multiplier = 1.0;  ///< Subsampling factor (1.0 = all splats)
        bool enable_lod = true;           ///< Generate LOD hierarchy
        bool optimize_for_gpu = true;     ///< Reorder data for GPU cache efficiency
        bool validate_required = true;    ///< Check for required PLY properties
        bool warn_missing_optional = true; ///< Warn about missing optional properties
        bool quantize_positions = false;  ///< Reduce position precision (see above)
        bool quantize_colors = false;     ///< Reduce color precision (see above)
        bool quantize_scales = false;     ///< Reduce scale precision (see above)
        bool quantize_rotations = false;  ///< Reduce rotation precision (see above)
        bool pack_opacity = false;        ///< Deprecated compatibility option (ignored)
        bool normalize_opacity = true;    ///< Normalize opacity range to [0,1]
        bool sort_by_opacity = false;     ///< Sort splats by opacity for blending
        bool generate_thumbnail = true;   ///< Generate preview thumbnail
        int thumbnail_style = 0;          ///< Thumbnail visualization style
        int thumbnail_size = 128;         ///< Thumbnail resolution in pixels
        bool include_statistics = true;   ///< Include load statistics in metadata
        bool include_memory_estimate = true; ///< Include memory estimates in metadata
        bool custom_settings = false;     ///< True if settings differ from preset
    };

private:
    Label *file_path_label = nullptr;
    Label *splat_summary_label = nullptr;
    Label *memory_label = nullptr;
    Label *comparison_label = nullptr;
    TextureRect *thumbnail_preview = nullptr;
    OptionButton *preset_selector = nullptr;
    OptionButton *asset_type_selector = nullptr;
    SpinBox *max_splats_spin = nullptr;
    SpinBox *density_spin = nullptr;
    CheckBox *lod_checkbox = nullptr;
    CheckBox *optimize_checkbox = nullptr;
    CheckBox *validate_checkbox = nullptr;
    CheckBox *warn_checkbox = nullptr;
    CheckBox *normalize_checkbox = nullptr;
    CheckBox *sort_checkbox = nullptr;
    CheckBox *compress_positions_checkbox = nullptr;
    CheckBox *compress_colors_checkbox = nullptr;
    CheckBox *compress_scales_checkbox = nullptr;
    CheckBox *compress_rotations_checkbox = nullptr;
    CheckBox *pack_opacity_checkbox = nullptr;
    CheckBox *thumbnail_checkbox = nullptr;
    OptionButton *thumbnail_style_option = nullptr;
    SpinBox *thumbnail_size_spin = nullptr;
    CheckBox *include_stats_checkbox = nullptr;
    CheckBox *include_memory_checkbox = nullptr;
    int custom_preset_index = -1;

    Ref<GaussianSplatAsset> preview_asset;
    Ref<GaussianThumbnailGenerator> thumbnail_generator;
    ImportConfiguration current_config;
    Dictionary override_options;
    Dictionary baseline_options;
    Dictionary loader_statistics;
    Dictionary comparison_metadata;
    String source_path;
    Ref<SceneTreeTimer> preview_debounce_timer;
    AABB preview_bounds;
    int64_t preview_request_serial = 0;
    bool preview_update_pending = false;
    bool preview_generation_in_progress = false;
    bool reimport_mode = false;
    bool updating_ui = false;
    bool preview_valid = false;
    bool preview_bounds_valid = false;
    bool source_is_ply = true; // Track source format for format-aware UI

    void _build_ui();
    VBoxContainer *_create_quality_tab();
    VBoxContainer *_create_compression_tab();
    VBoxContainer *_create_preview_tab();
    VBoxContainer *_create_metadata_tab();

    void _load_preview_asset();
    void _update_format_specific_controls();
    void _update_thumbnail_controls_state();
    void _apply_preset_defaults(const GaussianImportPresetDefinition &p_preset);
    void _apply_configuration_to_ui();
    void _apply_dictionary_override(const Dictionary &p_options);
    void _update_configuration_from_ui();
    void _update_customization_flag();
    void _schedule_preview_update(bool p_immediate = false);
    void _on_preview_debounce_timeout(int64_t p_request_serial);
    void _process_queued_preview(int64_t p_request_serial);
    void _update_preview();
    void _update_statistics();
    void _update_memory_estimate();
    void _update_comparison();
    void _refresh_all();

    Dictionary _configuration_to_dictionary(const ImportConfiguration &p_config) const;
    void _configuration_from_dictionary(ImportConfiguration &r_config, const Dictionary &p_dict);

    void _on_preset_selected(int p_index);
    void _on_thumbnail_style_selected(int p_index);
    void _on_settings_changed();
    void _on_confirmed();

protected:
    static void _bind_methods();

    virtual void ok_pressed() override;

public:
    GaussianImportDialog();

    void configure_for_file(const String &p_source_path, const Ref<GaussianSplatAsset> &p_existing_asset, bool p_reimport,
            const Dictionary &p_override_options = Dictionary());
    Dictionary get_selected_options() const;
    ImportConfiguration get_configuration() const { return current_config; }
    String get_source_path() const { return source_path; }
};

#endif // TOOLS_ENABLED

#endif // GAUSSIAN_IMPORT_DIALOG_H
