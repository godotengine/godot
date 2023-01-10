/**************************************************************************/
/*  cpu_particles_editor_plugin.h                                         */
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

#ifndef CPU_PARTICLES_EDITOR_PLUGIN_H
#define CPU_PARTICLES_EDITOR_PLUGIN_H

#include "editor/plugins/particles_editor_plugin.h"
#include "scene/3d/cpu_particles.h"

class CPUParticlesEditor : public ParticlesEditorBase {
	GDCLASS(CPUParticlesEditor, ParticlesEditorBase);

	enum Menu {

		MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE,
		MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_MESH,
		MENU_OPTION_CLEAR_EMISSION_VOLUME,
		MENU_OPTION_RESTART

	};

	CPUParticles *node;

	void _menu_option(int);

	friend class CPUParticlesEditorPlugin;

	virtual void _generate_emission_points();

protected:
	void _notification(int p_notification);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	void edit(CPUParticles *p_particles);
	CPUParticlesEditor();
};

class CPUParticlesEditorPlugin : public EditorPlugin {
	GDCLASS(CPUParticlesEditorPlugin, EditorPlugin);

	CPUParticlesEditor *particles_editor;
	EditorNode *editor;

public:
	virtual String get_name() const { return "CPUParticles"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	CPUParticlesEditorPlugin(EditorNode *p_node);
	~CPUParticlesEditorPlugin();
};

#endif // CPU_PARTICLES_EDITOR_PLUGIN_H
