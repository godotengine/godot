/*************************************************************************/
/*  occluder_shape.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "occluder_shape.h"

#include "core/engine.h"
#include "core/math/transform.h"
#include "servers/visual_server.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#endif

void OccluderShape::_bind_methods() {
}

OccluderShape::OccluderShape(RID p_shape) {
	_shape = p_shape;
}

OccluderShape::~OccluderShape() {
	if (_shape != RID()) {
		VisualServer::get_singleton()->free(_shape);
	}
}

void OccluderShape::update_transform_to_visual_server(const Transform &p_global_xform) {
	VisualServer::get_singleton()->occluder_set_transform(get_shape(), p_global_xform);
}

void OccluderShape::update_active_to_visual_server(bool p_active) {
	VisualServer::get_singleton()->occluder_set_active(get_shape(), p_active);
}

void OccluderShape::notification_exit_world() {
	VisualServer::get_singleton()->occluder_set_scenario(_shape, RID(), VisualServer::OCCLUDER_TYPE_UNDEFINED);
}

//////////////////////////////////////////////

void OccluderShapeSphere::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_spheres", "spheres"), &OccluderShapeSphere::set_spheres);
	ClassDB::bind_method(D_METHOD("get_spheres"), &OccluderShapeSphere::get_spheres);

	ClassDB::bind_method(D_METHOD("set_sphere_position", "index", "position"), &OccluderShapeSphere::set_sphere_position);
	ClassDB::bind_method(D_METHOD("set_sphere_radius", "index", "radius"), &OccluderShapeSphere::set_sphere_radius);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "spheres", PROPERTY_HINT_NONE, itos(Variant::PLANE) + ":"), "set_spheres", "get_spheres");
}

void OccluderShapeSphere::update_shape_to_visual_server() {
	VisualServer::get_singleton()->occluder_spheres_update(get_shape(), _spheres);
}

Transform OccluderShapeSphere::center_node(const Transform &p_global_xform, const Transform &p_parent_xform, real_t p_snap) {
	if (!_spheres.size()) {
		return Transform();
	}

	// make sure world spheres correct
	Vector<Plane> spheres_world_space;
	if (spheres_world_space.size() != _spheres.size()) {
		spheres_world_space.resize(_spheres.size());
	}

	Vector3 scale3 = p_global_xform.basis.get_scale_abs();
	real_t scale = (scale3.x + scale3.y + scale3.z) / 3.0;

	for (int n = 0; n < _spheres.size(); n++) {
		Plane p;
		p.normal = p_global_xform.xform(_spheres[n].normal);
		p.d = _spheres[n].d * scale;
		spheres_world_space.set(n, p);
	}

	// first find the center
	AABB bb;
	bb.set_position(spheres_world_space[0].normal);

	// new positions
	for (int n = 0; n < spheres_world_space.size(); n++) {
		const Plane &sphere = spheres_world_space[n];

		// update aabb
		AABB sphere_bb(sphere.normal, Vector3());
		sphere_bb.grow_by(sphere.d);
		bb.merge_with(sphere_bb);
	}

	Vector3 center = bb.get_center();

	// snapping
	if (p_snap > 0.0001) {
		center.snap(Vector3(p_snap, p_snap, p_snap));
	}

	// new transform with no rotate or scale, centered
	Transform new_local_xform = Transform();
	new_local_xform.translate(center.x, center.y, center.z);

	Transform inv_xform = new_local_xform.affine_inverse();

	// back calculate the new spheres
	for (int n = 0; n < spheres_world_space.size(); n++) {
		Plane p = spheres_world_space[n];

		p.normal = inv_xform.xform(p.normal);

		// assuming uniform scale, otherwise this will go wrong
		Vector3 inv_scale = inv_xform.basis.get_scale_abs();
		p.d *= inv_scale.x;

		spheres_world_space.set(n, p);
	}

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		UndoRedo *undo_redo = EditorNode::get_undo_redo();

		undo_redo->create_action(TTR("OccluderShapeSphere Set Spheres"));
		undo_redo->add_do_method(this, "set_spheres", spheres_world_space);
		undo_redo->add_undo_method(this, "set_spheres", _spheres);
		undo_redo->commit_action();
	} else {
		set_spheres(spheres_world_space);
	}
#else
	set_spheres(spheres_world_space);
#endif

	notify_change_to_owners();

	return new_local_xform;
}

void OccluderShapeSphere::notification_enter_world(RID p_scenario) {
	VisualServer::get_singleton()->occluder_set_scenario(get_shape(), p_scenario, VisualServer::OCCLUDER_TYPE_SPHERE);
}

void OccluderShapeSphere::set_spheres(const Vector<Plane> &p_spheres) {
#ifdef TOOLS_ENABLED
	// try and detect special circumstance of adding a new sphere in the editor
	bool adding_in_editor = false;
	if ((p_spheres.size() == _spheres.size() + 1) && (p_spheres[p_spheres.size() - 1] == Plane())) {
		adding_in_editor = true;
	}
#endif

	_spheres = p_spheres;

	// sanitize radii
	for (int n = 0; n < _spheres.size(); n++) {
		if (_spheres[n].d < _min_radius) {
			Plane p = _spheres[n];
			p.d = _min_radius;
			_spheres.set(n, p);
		}
	}

#ifdef TOOLS_ENABLED
	if (adding_in_editor) {
		_spheres.set(_spheres.size() - 1, Plane(Vector3(), 1.0));
	}
#endif

	notify_change_to_owners();
}

void OccluderShapeSphere::set_sphere_position(int p_idx, const Vector3 &p_position) {
	if ((p_idx >= 0) && (p_idx < _spheres.size())) {
		Plane p = _spheres[p_idx];
		p.normal = p_position;
		_spheres.set(p_idx, p);
		notify_change_to_owners();
	}
}

void OccluderShapeSphere::set_sphere_radius(int p_idx, real_t p_radius) {
	if ((p_idx >= 0) && (p_idx < _spheres.size())) {
		Plane p = _spheres[p_idx];
		p.d = MAX(p_radius, _min_radius);
		_spheres.set(p_idx, p);
		notify_change_to_owners();
	}
}

OccluderShapeSphere::OccluderShapeSphere() :
		OccluderShape(VisualServer::get_singleton()->occluder_create()) {
}
