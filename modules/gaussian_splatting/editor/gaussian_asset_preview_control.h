#ifndef GAUSSIAN_ASSET_PREVIEW_CONTROL_H
#define GAUSSIAN_ASSET_PREVIEW_CONTROL_H

#ifdef TOOLS_ENABLED

#include "core/math/vector3.h"
#include "scene/gui/box_container.h"

#include "../core/gaussian_splat_asset.h"

class Camera3D;
class DirectionalLight3D;
class Label;
class MeshInstance3D;
class MultiMeshInstance3D;
class SubViewport;
class SubViewportContainer;
class TextureRect;
class Node3D;
class GaussianThumbnailGenerator;

class GaussianAssetPreviewControl : public VBoxContainer {
	GDCLASS(GaussianAssetPreviewControl, VBoxContainer);

private:
	SubViewportContainer *viewport_container = nullptr;
	SubViewport *viewport = nullptr;
	Node3D *viewport_root = nullptr;
	Node3D *orbit_root = nullptr;
	Node3D *preview_root = nullptr;
	MeshInstance3D *bounds_instance = nullptr;
	MultiMeshInstance3D *splat_instance = nullptr;
	Camera3D *camera = nullptr;
	DirectionalLight3D *key_light = nullptr;
	DirectionalLight3D *fill_light = nullptr;
	VBoxContainer *fallback_container = nullptr;
	TextureRect *fallback_texture = nullptr;
	Label *fallback_label = nullptr;

	Ref<GaussianSplatAsset> asset;
	Ref<GaussianThumbnailGenerator> thumbnail_generator;

	Vector3 orbit_center = Vector3();
	float orbit_distance = 3.0f;
	float orbit_yaw = 0.7f;
	float orbit_pitch = -0.35f;
	bool orbit_dragging = false;
	Vector2 orbit_drag_last = Vector2();
	bool interactive_preview_ready = false;

	void _build_ui();
	void _clear_viewport_scene();
	void _update_preview_visibility();
	void _update_fallback_texture();
	void _update_camera_transform();
	void _rebuild_scene_from_asset();
	void _build_bounds_mesh(const AABB &p_bounds);
	void _build_sample_cloud(const AABB &p_bounds);
	AABB _resolve_asset_bounds() const;
	void _set_placeholder_text(const String &p_text);

protected:
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	static void _bind_methods();

public:
	GaussianAssetPreviewControl();
	void set_asset(const Ref<GaussianSplatAsset> &p_asset);
	Ref<GaussianSplatAsset> get_asset() const { return asset; }
};

#endif // TOOLS_ENABLED

#endif // GAUSSIAN_ASSET_PREVIEW_CONTROL_H
