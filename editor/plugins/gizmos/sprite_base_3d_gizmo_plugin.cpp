/**************************************************************************/
/*  sprite_base_3d_gizmo_plugin.cpp                                       */
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

#include "sprite_base_3d_gizmo_plugin.h"

#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/sprite_3d.h"

SpriteBase3DGizmoPlugin::SpriteBase3DGizmoPlugin() {
}

bool SpriteBase3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<SpriteBase3D>(p_spatial) != nullptr;
}

String SpriteBase3DGizmoPlugin::get_gizmo_name() const {
	return "SpriteBase3D";
}

int SpriteBase3DGizmoPlugin::get_priority() const {
	return -1;
}

bool SpriteBase3DGizmoPlugin::can_be_hidden() const {
	return false;
}

void SpriteBase3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	SpriteBase3D *sprite_base = Object::cast_to<SpriteBase3D>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	Ref<TriangleMesh> tm = sprite_base->generate_triangle_mesh();
	if (tm.is_valid()) {
		p_gizmo->add_collision_triangles(tm);
	}
}
