/**************************************************************************/
/*  particles_2d_editor_plugin.h                                          */
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

class EditorFileDialog;

class Particles2DEditorPlugin : public ParticlesEditorPlugin {
	GDCLASS(Particles2DEditorPlugin, ParticlesEditorPlugin);

protected:
	enum {
		MENU_LOAD_EMISSION_MASK = 100,
	};

	HashSet<Node *> selected_particles;

	enum EmissionMode {
		EMISSION_MODE_SOLID,
		EMISSION_MODE_BORDER,
		EMISSION_MODE_BORDER_DIRECTED
	};

	EditorFileDialog *file = nullptr;
	ConfirmationDialog *emission_mask = nullptr;
	OptionButton *emission_mask_mode = nullptr;
	CheckBox *emission_mask_centered = nullptr;
	CheckBox *emission_colors = nullptr;
	String source_emission_file;

	virtual void _menu_callback(int p_idx) override;
	virtual void _add_menu_options(PopupMenu *p_menu) override;

	void _file_selected(const String &p_file);
	void _get_base_emission_mask(PackedVector2Array &r_valid_positions, PackedVector2Array &r_valid_normals, PackedByteArray &r_valid_colors, Vector2i &r_image_size);
	virtual void _generate_emission_mask() = 0;
	void _notification(int p_what);
	void _set_show_gizmos(Node *p_node, bool p_show);
	void _selection_changed();
	void _node_removed(Node *p_node);

public:
	Particles2DEditorPlugin();
};

class GPUParticles2DEditorPlugin : public Particles2DEditorPlugin {
	GDCLASS(GPUParticles2DEditorPlugin, Particles2DEditorPlugin);

	enum {
		MENU_GENERATE_VISIBILITY_RECT = 200,
	};

	ConfirmationDialog *generate_visibility_rect = nullptr;
	SpinBox *generate_seconds = nullptr;

	void _generate_visibility_rect();

protected:
	void _menu_callback(int p_idx) override;
	void _add_menu_options(PopupMenu *p_menu) override;

	Node *_convert_particles() override;

	void _generate_emission_mask() override;

public:
	GPUParticles2DEditorPlugin();
};

class CPUParticles2DEditorPlugin : public Particles2DEditorPlugin {
	GDCLASS(CPUParticles2DEditorPlugin, Particles2DEditorPlugin);

protected:
	Node *_convert_particles() override;

	void _generate_emission_mask() override;

public:
	CPUParticles2DEditorPlugin();
};
