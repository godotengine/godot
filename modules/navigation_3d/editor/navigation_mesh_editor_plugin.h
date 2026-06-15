/**************************************************************************/
/*  navigation_mesh_editor_plugin.h                                       */
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

#pragma once

#include "editor/inspector/editor_inspector.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/gui/subviewport_container.h"

class SubViewport;
class Camera3D;
class Label;
class Mesh;
class MeshInstance3D;
class NavigationMesh;

class NavigationMeshEditor : public SubViewportContainer {
	GDCLASS(NavigationMeshEditor, SubViewportContainer);

	float rot_x;
	float rot_y;

	SubViewport *viewport = nullptr;
	MeshInstance3D *mesh_instance = nullptr;
	Node3D *rotation = nullptr;
	Camera3D *camera = nullptr;
	Label *metadata_label = nullptr;

	Ref<Mesh> mesh;
	Ref<NavigationMesh> navigation_mesh;

	void _update_rotation();
	void _navigation_mesh_changed();

protected:
	void _notification(int p_what);
	virtual void _update_theme_item_cache() override;
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

public:
	void edit(Ref<NavigationMesh> p_navigation_mesh);
	NavigationMeshEditor();
};

class EditorInspectorPluginNavigationMesh : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginNavigationMesh, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
};

class NavigationMeshEditorPlugin : public EditorPlugin {
	GDCLASS(NavigationMeshEditorPlugin, EditorPlugin);

public:
	virtual String get_plugin_name() const override { return "NavigationMesh"; }

	NavigationMeshEditorPlugin();
};
