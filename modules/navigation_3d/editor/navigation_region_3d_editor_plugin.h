/**************************************************************************/
/*  navigation_region_3d_editor_plugin.h                                  */
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

#include "editor/plugins/editor_plugin.h"

#include "navigation_region_3d_gizmo_plugin.h"

class AcceptDialog;
class Button;
class ConfirmationDialog;
class HBoxContainer;
class Label;
class NavigationRegion3D;

class NavigationRegion3DEditor : public Control {
	friend class NavigationRegion3DEditorPlugin;

	GDCLASS(NavigationRegion3DEditor, Control);

	AcceptDialog *err_dialog = nullptr;
	ConfirmationDialog *multibake_dialog = nullptr;

	HBoxContainer *bake_hbox = nullptr;
	Button *button_bake = nullptr;
	Button *button_reset = nullptr;
	Label *bake_info = nullptr;

	LocalVector<NavigationRegion3D *> selected_regions;

	LocalVector<NavigationRegion3D *> regions_to_bake;
	LocalVector<NavigationRegion3D *> regions_with_navmesh_to_bake;

	int processed_regions_to_bake_count = 0;
	int processed_regions_to_bake_count_max = 0;
	bool region_baking_canceled = false;
	NavigationRegion3D *currently_baking_region = nullptr;

	bool bake_in_process = false;

	void _bake_pressed();
	void _clear_pressed();

	void _on_navmesh_multibake_confirmed();
	void _on_navmesh_multibake_canceled();
	void _process_regions_to_bake();

protected:
	void _node_removed(Node *p_node);
	void _notification(int p_what);

public:
	void edit(LocalVector<NavigationRegion3D *> p_regions);
	NavigationRegion3DEditor();
};

class NavigationRegion3DEditorPlugin : public EditorPlugin {
	GDCLASS(NavigationRegion3DEditorPlugin, EditorPlugin);

	NavigationRegion3DEditor *navigation_region_editor = nullptr;

	Ref<NavigationRegion3DGizmoPlugin> gizmo_plugin;

public:
	virtual String get_plugin_name() const override { return "NavigationRegion3D"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	NavigationRegion3DEditorPlugin();
};
