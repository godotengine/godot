#ifndef GAUSSIAN_IMPORT_SETTINGS_DIALOG_H
#define GAUSSIAN_IMPORT_SETTINGS_DIALOG_H

#ifdef TOOLS_ENABLED

#include "scene/gui/dialogs.h"

#include "core/math/aabb.h"
#include "core/variant/dictionary.h"

class Camera3D;
class CheckBox;
class DirectionalLight3D;
class GaussianSplatAsset;
class HSplitContainer;
class Label;
class MeshInstance3D;
class MultiMeshInstance3D;
class Node3D;
class OptionButton;
class SubViewport;
class SubViewportContainer;

class GaussianImportSettingsDialog : public ConfirmationDialog {
	GDCLASS(GaussianImportSettingsDialog, ConfirmationDialog);

	static GaussianImportSettingsDialog *singleton;

	// 3D viewport preview.
	SubViewportContainer *viewport_container = nullptr;
	SubViewport *viewport = nullptr;
	Node3D *viewport_root = nullptr;
	Node3D *orbit_root = nullptr;
	Node3D *preview_root = nullptr;
	Camera3D *camera = nullptr;
	DirectionalLight3D *light1 = nullptr;
	DirectionalLight3D *light2 = nullptr;
	MeshInstance3D *bounds_instance = nullptr;
	MultiMeshInstance3D *splat_instance = nullptr;

	// Info labels.
	Label *file_label = nullptr;
	Label *stats_label = nullptr;

	// Import options.
	OptionButton *quality_selector = nullptr;
	CheckBox *compress_positions = nullptr;
	CheckBox *compress_colors = nullptr;
	CheckBox *compress_scales = nullptr;
	CheckBox *compress_rotations = nullptr;

	// State.
	String source_path;
	Ref<GaussianSplatAsset> loaded_asset;
	AABB asset_bounds;
	Dictionary import_options;

	float cam_rot_x = -0.35f;
	float cam_rot_y = 0.7f;
	float cam_zoom = 1.0f;

	void _build_ui();
	void _build_viewport_scene();
	void _clear_viewport_scene();
	void _update_camera();
	void _update_stats();
	void _load_source_asset();
	void _viewport_input(const Ref<InputEvent> &p_input);
	void _re_import();

	AABB _resolve_bounds() const;

protected:
	static void _bind_methods();

public:
	void open_settings(const String &p_path);
	static GaussianImportSettingsDialog *get_singleton();

	GaussianImportSettingsDialog();
};

#endif // TOOLS_ENABLED

#endif // GAUSSIAN_IMPORT_SETTINGS_DIALOG_H
