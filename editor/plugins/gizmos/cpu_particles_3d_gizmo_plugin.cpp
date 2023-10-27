/**************************************************************************/
/*  cpu_particles_3d_gizmo_plugin.cpp                                     */
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

#include "cpu_particles_3d_gizmo_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/cpu_particles_3d.h"

CPUParticles3DGizmoPlugin::CPUParticles3DGizmoPlugin() {
	create_icon_material("particles_icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GizmoCPUParticles3D"), EditorStringName(EditorIcons)));
}

bool CPUParticles3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<CPUParticles3D>(p_spatial) != nullptr;
}

String CPUParticles3DGizmoPlugin::get_gizmo_name() const {
	return "CPUParticles3D";
}

int CPUParticles3DGizmoPlugin::get_priority() const {
	return -1;
}

bool CPUParticles3DGizmoPlugin::is_selectable_when_hidden() const {
	return true;
}

void CPUParticles3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	Ref<Material> icon = get_material("particles_icon", p_gizmo);
	p_gizmo->add_unscaled_billboard(icon, 0.05);
}
