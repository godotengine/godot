/*************************************************************************/
/*  collision_object_sw.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "collision_object_sw.h"
#include "space_sw.h"

void CollisionObjectSW::add_shape(ShapeSW *p_shape, const Transform &p_transform) {

	Shape s;
	s.shape = p_shape;
	s.xform = p_transform;
	s.xform_inv = s.xform.affine_inverse();
	s.bpid = 0; //needs update
	shapes.push_back(s);
	p_shape->add_owner(this);
	_update_shapes();
	_shapes_changed();
}

void CollisionObjectSW::set_shape(int p_index, ShapeSW *p_shape) {

	ERR_FAIL_INDEX(p_index, shapes.size());
	shapes[p_index].shape->remove_owner(this);
	shapes[p_index].shape = p_shape;

	p_shape->add_owner(this);
	_update_shapes();
	_shapes_changed();
}
void CollisionObjectSW::set_shape_transform(int p_index, const Transform &p_transform) {

	ERR_FAIL_INDEX(p_index, shapes.size());

	shapes[p_index].xform = p_transform;
	shapes[p_index].xform_inv = p_transform.affine_inverse();
	_update_shapes();
	_shapes_changed();
}

void CollisionObjectSW::remove_shape(ShapeSW *p_shape) {

	//remove a shape, all the times it appears
	for (int i = 0; i < shapes.size(); i++) {

		if (shapes[i].shape == p_shape) {
			remove_shape(i);
			i--;
		}
	}
}

void CollisionObjectSW::remove_shape(int p_index) {

	//remove anything from shape to be erased to end, so subindices don't change
	ERR_FAIL_INDEX(p_index, shapes.size());
	for (int i = p_index; i < shapes.size(); i++) {

		if (shapes[i].bpid == 0)
			continue;
		//should never get here with a null owner
		space->get_broadphase()->remove(shapes[i].bpid);
		shapes[i].bpid = 0;
	}
	shapes[p_index].shape->remove_owner(this);
	shapes.remove(p_index);

	_shapes_changed();
}

void CollisionObjectSW::_set_static(bool p_static) {
	if (_static == p_static)
		return;
	_static = p_static;

	if (!space)
		return;
	for (int i = 0; i < get_shape_count(); i++) {
		Shape &s = shapes[i];
		if (s.bpid > 0) {
			space->get_broadphase()->set_static(s.bpid, _static);
		}
	}
}

void CollisionObjectSW::_unregister_shapes() {

	for (int i = 0; i < shapes.size(); i++) {

		Shape &s = shapes[i];
		if (s.bpid > 0) {
			space->get_broadphase()->remove(s.bpid);
			s.bpid = 0;
		}
	}
}

void CollisionObjectSW::_update_shapes() {

	if (!space)
		return;

	for (int i = 0; i < shapes.size(); i++) {

		Shape &s = shapes[i];
		if (s.bpid == 0) {
			s.bpid = space->get_broadphase()->create(this, i);
			space->get_broadphase()->set_static(s.bpid, _static);
		}

		//not quite correct, should compute the next matrix..
		Rect3 shape_aabb = s.shape->get_aabb();
		Transform xform = transform * s.xform;
		shape_aabb = xform.xform(shape_aabb);
		s.aabb_cache = shape_aabb;
		s.aabb_cache = s.aabb_cache.grow((s.aabb_cache.size.x + s.aabb_cache.size.y) * 0.5 * 0.05);

		Vector3 scale = xform.get_basis().get_scale();
		s.area_cache = s.shape->get_area() * scale.x * scale.y * scale.z;

		space->get_broadphase()->move(s.bpid, s.aabb_cache);
	}
}

void CollisionObjectSW::_update_shapes_with_motion(const Vector3 &p_motion) {

	if (!space)
		return;

	for (int i = 0; i < shapes.size(); i++) {

		Shape &s = shapes[i];
		if (s.bpid == 0) {
			s.bpid = space->get_broadphase()->create(this, i);
			space->get_broadphase()->set_static(s.bpid, _static);
		}

		//not quite correct, should compute the next matrix..
		Rect3 shape_aabb = s.shape->get_aabb();
		Transform xform = transform * s.xform;
		shape_aabb = xform.xform(shape_aabb);
		shape_aabb = shape_aabb.merge(Rect3(shape_aabb.pos + p_motion, shape_aabb.size)); //use motion
		s.aabb_cache = shape_aabb;

		space->get_broadphase()->move(s.bpid, shape_aabb);
	}
}

void CollisionObjectSW::_set_space(SpaceSW *p_space) {

	if (space) {

		space->remove_object(this);

		for (int i = 0; i < shapes.size(); i++) {

			Shape &s = shapes[i];
			if (s.bpid) {
				space->get_broadphase()->remove(s.bpid);
				s.bpid = 0;
			}
		}
	}

	space = p_space;

	if (space) {

		space->add_object(this);
		_update_shapes();
	}
}

void CollisionObjectSW::_shape_changed() {

	_update_shapes();
	_shapes_changed();
}

CollisionObjectSW::CollisionObjectSW(Type p_type) {

	_static = true;
	type = p_type;
	space = NULL;
	instance_id = 0;
	layer_mask = 1;
	collision_mask = 1;
	ray_pickable = true;
}
