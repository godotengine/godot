/**************************************************************************/
/*  cpu_particles_3d_editor_plugin.h                                      */
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

#ifndef CPU_PARTICLES_3D_EDITOR_PLUGIN_H
#define CPU_PARTICLES_3D_EDITOR_PLUGIN_H

#include "editor/plugins/gpu_particles_3d_editor_plugin.h"
#include "scene/3d/cpu_particles_3d.h"

class CPUParticles3DEditor : public GPUParticles3DEditorBase {
	GDCLASS(CPUParticles3DEditor, GPUParticles3DEditorBase);

	enum Menu {
		MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE,
		MENU_OPTION_CLEAR_EMISSION_VOLUME,
		MENU_OPTION_RESTART,
		MENU_OPTION_CONVERT_TO_GPU_PARTICLES,
	};

	CPUParticles3D *node = nullptr;

	void _menu_option(int);

	friend class CPUParticles3DEditorPlugin;

	virtual void _generate_emission_points() override;

protected:
	void _notification(int p_notification);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	void edit(CPUParticles3D *p_particles);
	CPUParticles3DEditor();
};

class CPUParticles3DEditorPlugin : public EditorPlugin {
	GDCLASS(CPUParticles3DEditorPlugin, EditorPlugin);

	CPUParticles3DEditor *particles_editor = nullptr;

public:
	virtual String get_name() const override { return "CPUParticles3D"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	CPUParticles3DEditorPlugin();
	~CPUParticles3DEditorPlugin();
};

#endif // CPU_PARTICLES_3D_EDITOR_PLUGIN_H
