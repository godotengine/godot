/**************************************************************************/
/*  gpu_particles_3d_editor_plugin.h                                      */
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

#ifndef GPU_PARTICLES_3D_EDITOR_PLUGIN_H
#define GPU_PARTICLES_3D_EDITOR_PLUGIN_H

#include "editor/plugins/editor_plugin.h"
#include "scene/3d/gpu_particles_3d.h"
#include "scene/gui/spin_box.h"

class ConfirmationDialog;
class HBoxContainer;
class MenuButton;
class OptionButton;
class SceneTreeDialog;

class GPUParticles3DEditorBase : public Control {
	GDCLASS(GPUParticles3DEditorBase, Control);

protected:
	Node3D *base_node = nullptr;
	Panel *panel = nullptr;
	MenuButton *options = nullptr;
	HBoxContainer *particles_editor_hb = nullptr;

	SceneTreeDialog *emission_tree_dialog = nullptr;

	ConfirmationDialog *emission_dialog = nullptr;
	SpinBox *emission_amount = nullptr;
	OptionButton *emission_fill = nullptr;

	Vector<Face3> geometry;

	bool _generate(Vector<Vector3> &points, Vector<Vector3> &normals);
	virtual void _generate_emission_points(){};
	void _node_selected(const NodePath &p_path);

	static void _bind_methods();

public:
	GPUParticles3DEditorBase();
};

class GPUParticles3DEditor : public GPUParticles3DEditorBase {
	GDCLASS(GPUParticles3DEditor, GPUParticles3DEditorBase);

	ConfirmationDialog *generate_aabb = nullptr;
	SpinBox *generate_seconds = nullptr;
	GPUParticles3D *node = nullptr;

	enum Menu {
		MENU_OPTION_GENERATE_AABB,
		MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE,
		MENU_OPTION_CLEAR_EMISSION_VOLUME,
		MENU_OPTION_CONVERT_TO_CPU_PARTICLES,
		MENU_OPTION_RESTART,

	};

	void _generate_aabb();

	void _menu_option(int);

	friend class GPUParticles3DEditorPlugin;

	virtual void _generate_emission_points() override;

protected:
	void _notification(int p_notification);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	void edit(GPUParticles3D *p_particles);
	GPUParticles3DEditor();
};

class GPUParticles3DEditorPlugin : public EditorPlugin {
	GDCLASS(GPUParticles3DEditorPlugin, EditorPlugin);

	GPUParticles3DEditor *particles_editor = nullptr;

public:
	virtual String get_name() const override { return "GPUParticles3D"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	GPUParticles3DEditorPlugin();
	~GPUParticles3DEditorPlugin();
};

#endif // GPU_PARTICLES_3D_EDITOR_PLUGIN_H
