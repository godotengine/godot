/**************************************************************************/
/*  multimesh_instance_2d.cpp                                             */
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

#include "multimesh_instance_2d.h"

#include "scene/scene_string_names.h"

void MultiMeshInstance2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			if (multimesh.is_valid()) {
				draw_multimesh(multimesh, texture);
			}
		} break;
	}
}

void MultiMeshInstance2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_multimesh", "multimesh"), &MultiMeshInstance2D::set_multimesh);
	ClassDB::bind_method(D_METHOD("get_multimesh"), &MultiMeshInstance2D::get_multimesh);

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &MultiMeshInstance2D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &MultiMeshInstance2D::get_texture);

	ADD_SIGNAL(MethodInfo("texture_changed"));

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "multimesh", PROPERTY_HINT_RESOURCE_TYPE, "MultiMesh"), "set_multimesh", "get_multimesh");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture");
}

void MultiMeshInstance2D::set_multimesh(const Ref<MultiMesh> &p_multimesh) {
	// Cleanup previous connection if any.
	if (multimesh.is_valid()) {
		multimesh->disconnect_changed(callable_mp((CanvasItem *)this, &CanvasItem::queue_redraw));
	}
	multimesh = p_multimesh;

	// Connect to the multimesh so the AABB can update when instance transforms are changed.
	if (multimesh.is_valid()) {
		multimesh->connect_changed(callable_mp((CanvasItem *)this, &CanvasItem::queue_redraw));
	}
	queue_redraw();
}

Ref<MultiMesh> MultiMeshInstance2D::get_multimesh() const {
	return multimesh;
}

void MultiMeshInstance2D::set_texture(const Ref<Texture2D> &p_texture) {
	if (p_texture == texture) {
		return;
	}
	texture = p_texture;
	queue_redraw();
	emit_signal(SceneStringNames::get_singleton()->texture_changed);
}

Ref<Texture2D> MultiMeshInstance2D::get_texture() const {
	return texture;
}

#ifdef TOOLS_ENABLED
Rect2 MultiMeshInstance2D::_edit_get_rect() const {
	if (multimesh.is_valid()) {
		AABB aabb = multimesh->get_aabb();
		return Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);
	}

	return Node2D::_edit_get_rect();
}
#endif

MultiMeshInstance2D::MultiMeshInstance2D() {
}

MultiMeshInstance2D::~MultiMeshInstance2D() {
}
