#ifndef GAUSSIAN_IMPORT_SETTINGS_DIALOG_H
#define GAUSSIAN_IMPORT_SETTINGS_DIALOG_H

#ifdef TOOLS_ENABLED

#include "scene/gui/dialogs.h"

#include "core/math/aabb.h"
#include "core/variant/dictionary.h"
#include "scene/resources/3d/sky_material.h"

class Button;
class Camera3D;
class CameraAttributesPractical;
class DirectionalLight3D;
class EditorInspector;
class Environment;
class GaussianImportSettingsData;
class GaussianSplatAsset;
class GaussianSplatNode3D;
class HSplitContainer;
class Label;
class MeshInstance3D;
class Sky;
class SubViewport;
class SubViewportContainer;

class GaussianImportSettingsDialog : public ConfirmationDialog {
	GDCLASS(GaussianImportSettingsDialog, ConfirmationDialog);

	static GaussianImportSettingsDialog *singleton;

	// 3D viewport preview.
	SubViewportContainer *viewport_container = nullptr;
	SubViewport *viewport = nullptr;
	Camera3D *camera = nullptr;
	Ref<CameraAttributesPractical> camera_attributes;
	Ref<Environment> environment;
	Ref<Sky> sky;
	Ref<ProceduralSkyMaterial> procedural_sky_material;

	DirectionalLight3D *light1 = nullptr;
	DirectionalLight3D *light2 = nullptr;

	// Light toggle buttons (overlaid on viewport).
	Button *light_1_switch = nullptr;
	Button *light_2_switch = nullptr;
	Button *light_rotate_switch = nullptr;

	struct ThemeCache {
		Ref<Texture2D> light_1_icon;
		Ref<Texture2D> light_2_icon;
		Ref<Texture2D> rotate_icon;
	} theme_cache;

	// Scene nodes inside viewport.
	MeshInstance3D *bounds_instance = nullptr;
	GaussianSplatNode3D *splat_node = nullptr;

	// Info labels.
	Label *file_label = nullptr;
	Label *stats_label = nullptr;

	// Import settings inspector.
	EditorInspector *inspector = nullptr;
	GaussianImportSettingsData *settings_data = nullptr;

	// State.
	String source_path;
	Ref<GaussianSplatAsset> loaded_asset;
	AABB asset_bounds;
	Dictionary import_options;

	float cam_rot_x = 0.0f;
	float cam_rot_y = 0.0f;
	float cam_zoom = 0.0f;

	void _build_ui();
	void _build_viewport_scene();
	void _clear_viewport_scene();
	void _update_camera();
	void _update_stats();
	void _load_source_asset();
	void _viewport_input(const Ref<InputEvent> &p_input);
	void _re_import();
	void _on_light_1_switch_pressed();
	void _on_light_2_switch_pressed();
	void _on_light_rotate_switch_pressed();
	void _on_inspector_property_edited(const String &p_name);
	void _populate_settings_data();

	AABB _resolve_bounds() const;

protected:
	virtual void _update_theme_item_cache() override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	void open_settings(const String &p_path);
	static GaussianImportSettingsDialog *get_singleton();

	GaussianImportSettingsDialog();
	~GaussianImportSettingsDialog();
};

#endif // TOOLS_ENABLED

#endif // GAUSSIAN_IMPORT_SETTINGS_DIALOG_H
