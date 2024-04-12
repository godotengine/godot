/**************************************************************************/
/*  gpu_particles_2d_editor_plugin.h                                      */
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

#ifndef GPU_PARTICLES_2D_EDITOR_PLUGIN_H
#define GPU_PARTICLES_2D_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "scene/2d/gpu_particles_2d.h"
#include "scene/2d/physics/collision_polygon_2d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/spin_box.h"

class CheckBox;
class ConfirmationDialog;
class EditorFileDialog;
class MenuButton;
class OptionButton;

class GPUParticles2DEditorPlugin : public EditorPlugin {
	GDCLASS(GPUParticles2DEditorPlugin, EditorPlugin);

	enum {
		MENU_GENERATE_VISIBILITY_RECT,
		MENU_LOAD_EMISSION_MASK,
		MENU_CLEAR_EMISSION_MASK,
		MENU_OPTION_CONVERT_TO_CPU_PARTICLES,
		MENU_RESTART
	};

	enum EmissionMode {
		EMISSION_MODE_SOLID,
		EMISSION_MODE_BORDER,
		EMISSION_MODE_BORDER_DIRECTED
	};

	GPUParticles2D *particles = nullptr;
	List<GPUParticles2D *> selected_particles;

	EditorFileDialog *file = nullptr;

	HBoxContainer *toolbar = nullptr;
	MenuButton *menu = nullptr;

	ConfirmationDialog *generate_visibility_rect = nullptr;
	SpinBox *generate_seconds = nullptr;

	ConfirmationDialog *emission_mask = nullptr;
	OptionButton *emission_mask_mode = nullptr;
	CheckBox *emission_mask_centered = nullptr;
	CheckBox *emission_colors = nullptr;

	String source_emission_file;

	void _file_selected(const String &p_file);
	void _menu_callback(int p_idx);
	void _generate_visibility_rect();
	void _generate_emission_mask();
	void _selection_changed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual String get_name() const override { return "GPUParticles2D"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	GPUParticles2DEditorPlugin();
	~GPUParticles2DEditorPlugin();
};

#endif // GPU_PARTICLES_2D_EDITOR_PLUGIN_H
