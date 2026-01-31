/**************************************************************************/
/*  particles_3d_editor_plugin.h                                          */
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

#include "editor/scene/particles_editor_plugin.h"

class Particles3DEditorPlugin : public ParticlesEditorPlugin {
	GDCLASS(Particles3DEditorPlugin, ParticlesEditorPlugin);

	enum {
		MENU_OPTION_GENERATE_AABB = 300,
		MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE,
	};

	ConfirmationDialog *generate_aabb = nullptr;
	SpinBox *generate_seconds = nullptr;

	SceneTreeDialog *emission_tree_dialog = nullptr;
	ConfirmationDialog *emission_dialog = nullptr;
	SpinBox *emission_amount = nullptr;
	OptionButton *emission_fill = nullptr;

	void _generate_aabb();
	void _node_selected(const NodePath &p_path);

protected:
	Vector<Face3> geometry;

	virtual void _menu_callback(int p_idx) override;
	virtual void _add_menu_options(PopupMenu *p_menu) override;

	bool _generate(Vector<Vector3> &r_points, Vector<Vector3> &r_normals);
	virtual bool _can_generate_points() const = 0;
	virtual void _generate_emission_points() = 0;

public:
	Particles3DEditorPlugin();
};

class GPUParticles3DEditorPlugin : public Particles3DEditorPlugin {
	GDCLASS(GPUParticles3DEditorPlugin, Particles3DEditorPlugin);

protected:
	Node *_convert_particles() override;

	bool _can_generate_points() const override;
	void _generate_emission_points() override;

public:
	GPUParticles3DEditorPlugin();
};

class CPUParticles3DEditorPlugin : public Particles3DEditorPlugin {
	GDCLASS(CPUParticles3DEditorPlugin, Particles3DEditorPlugin);

protected:
	Node *_convert_particles() override;

	bool _can_generate_points() const override { return true; }
	void _generate_emission_points() override;

public:
	CPUParticles3DEditorPlugin();
};
