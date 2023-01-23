/**************************************************************************/
/*  scene_preview.h                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef SCENE_PREVIEW_H
#define SCENE_PREVIEW_H

#include "editor/editor_inspector.h"
#include "editor/editor_plugin.h"
#include "scene/2d/camera_2d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/subviewport_container.h"
#include "scene/resources/camera_attributes.h"
#include "scene/resources/material.h"

class SubViewport;
class TextureButton;

class Scene3DPreview : public SubViewportContainer {
	GDCLASS(Scene3DPreview, SubViewportContainer);

	float rot_x;
	float rot_y;

	SubViewport *viewport = nullptr;
	Node3D *current = nullptr;
	Node3D *rotation = nullptr;
	DirectionalLight3D *light1 = nullptr;
	DirectionalLight3D *light2 = nullptr;
	Camera3D *camera = nullptr;
	Ref<CameraAttributesPractical> camera_attributes;

	TextureButton *light_1_switch = nullptr;
	TextureButton *light_2_switch = nullptr;

	struct ThemeCache {
		Ref<Texture2D> light_1_on;
		Ref<Texture2D> light_1_off;
		Ref<Texture2D> light_2_on;
		Ref<Texture2D> light_2_off;
	} theme_cache;

	void _button_pressed(Node *p_button);
	void _update_rotation();

	AABB _calculate_aabb(Node3D *p_node);

protected:
	virtual void _update_theme_item_cache() override;
	void _notification(int p_what);
	void gui_input(const Ref<InputEvent> &p_event) override;

public:
	void edit(Node3D *p_node);
	Scene3DPreview();
};

class Scene2DPreview : public SubViewportContainer {
	GDCLASS(Scene2DPreview, SubViewportContainer);

	Node2D *current = nullptr;
	SubViewport *viewport = nullptr;

protected:
	void _notification(int p_what);
	void gui_input(const Ref<InputEvent> &p_event) override;

public:
	void edit(Node2D *p_node);
	Scene2DPreview();
};

class SceneControlPreview : public SubViewportContainer {
	GDCLASS(SceneControlPreview, SubViewportContainer);

	Control *current = nullptr;
	SubViewport *viewport = nullptr;

protected:
	void _notification(int p_what);

public:
	void edit(Control *p_node);
	SceneControlPreview();
};

#endif // SCENE_PREVIEW_H
