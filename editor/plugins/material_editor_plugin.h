/*************************************************************************/
/*  material_editor_plugin.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef MATERIAL_EDITOR_PLUGIN_H
#define MATERIAL_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/resources/material.h"

#if 0
class MaterialEditor : public Control {

	GDCLASS(MaterialEditor, Control);


	Viewport *viewport;
	MeshInstance *sphere_instance;
	MeshInstance *box_instance;
	DirectionalLight *light1;
	DirectionalLight *light2;
	Camera *camera;

	Ref<Mesh> sphere_mesh;
	Ref<Mesh> box_mesh;

	TextureButton *sphere_switch;
	TextureButton *box_switch;

	TextureButton *light_1_switch;
	TextureButton *light_2_switch;


	Ref<Material> material;


	void _button_pressed(Node* p_button);
	bool first_enter;

protected:
	void _notification(int p_what);
	void _gui_input(InputEvent p_event);
	static void _bind_methods();
public:

	void edit(Ref<Material> p_material);
	MaterialEditor();
};


class MaterialEditorPlugin : public EditorPlugin {

	GDCLASS( MaterialEditorPlugin, EditorPlugin );

	MaterialEditor *material_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "Material"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	MaterialEditorPlugin(EditorNode *p_node);
	~MaterialEditorPlugin();

};

#endif // MATERIAL_EDITOR_PLUGIN_H
#endif
