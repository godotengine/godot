#pragma once

#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/main/viewport.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/subviewport_container.h"
#include "scene/gui/button.h"
#include "scene/gui/box_container.h"
#include "scene/resources/camera_attributes.h"
#include "scene/resources/material.h"


class SceneViewPanel : public SubViewportContainer {
    GDCLASS(SceneViewPanel, SubViewportContainer);

	float rot_x;
	float rot_y;

	SubViewport *viewport = nullptr;
	Node3D  *mesh_instance = nullptr;
	Node3D *rotation = nullptr;
	DirectionalLight3D *light1 = nullptr;
	DirectionalLight3D *light2 = nullptr;
	Camera3D *camera = nullptr;
	Ref<CameraAttributesPractical> camera_attributes;


	Button *light_1_switch = nullptr;
	Button *light_2_switch = nullptr;

	struct ThemeCache {
		Ref<Texture2D> light_1_icon;
		Ref<Texture2D> light_2_icon;
	} theme_cache;

	void _on_light_1_switch_pressed();
	void _on_light_2_switch_pressed();
	void _update_rotation();

protected:
	virtual void _update_theme_item_cache() override;
	void _notification(int p_what);
	void gui_input(const Ref<InputEvent> &p_event) override;

    void update_aabb(AABB& aabb,Node3D* p_mesh);

public:
	void edit(Node3D * p_mesh);
    void set_scene_path(const String& p_scene_path);
    SceneViewPanel();
};
