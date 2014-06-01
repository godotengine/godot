/*************************************************************************/
/*  remote_transform_2d.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "remote_transform_2d.h"
#include "scene/scene_string_names.h"

void RemoteTransform2D::_update_cache() {

	cache=0;
	if (has_node(remote_node)) {
		Node *node = get_node(remote_node);
		if (!node || this==node || node->is_a_parent_of(this) || this->is_a_parent_of(node)) {
			return;
		}

		cache=node->get_instance_ID();
	}
}

void RemoteTransform2D::_update_remote() {


	if (!is_inside_scene())
		return;

	if (!cache)
		return;

	Object *obj = ObjectDB::get_instance(cache);
	if (!obj)
		return;

	Node2D *object = obj->cast_to<Node2D>();
	Node2D *subject = this;

	if (!object)
		return;

	if (!object->is_inside_scene())
		return;

	if (track_backwards) {
		SWAP(object, subject);
	}

	if (track_pos && track_rot && track_scale) object->set_global_transform(subject->get_global_transform()); //todo make faster
	else {
		if (track_pos) object->set_pos(subject->get_pos());
		if (track_rot) object->set_rot(subject->get_rot());
		if (track_scale) object->set_scale(subject->get_scale());
	}

}

void RemoteTransform2D::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_READY: {

			_update_cache();

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (!is_inside_scene())
				break;

			if (cache) {

				_update_remote();

			}

		} break;

	}
}


void RemoteTransform2D::set_remote_node(const NodePath& p_remote_node) {

	remote_node=p_remote_node;
	if (is_inside_scene())
		_update_cache();
		_update_remote();
}

NodePath RemoteTransform2D::get_remote_node() const{
	return remote_node;
}

bool RemoteTransform2D::get_track_pos()	const { return track_pos; }
void RemoteTransform2D::set_track_pos(bool track) {
	track_pos = track; 
	if (is_inside_scene()) _update_remote();
}

bool RemoteTransform2D::get_track_rot()	const { return track_rot; }
void RemoteTransform2D::set_track_rot(bool track) {
	track_rot = track;
	if (is_inside_scene()) _update_remote();
}

bool RemoteTransform2D::get_track_scale() const { return track_scale; }
void RemoteTransform2D::set_track_scale(bool track) {
	track_scale = track;
	if (is_inside_scene()) _update_remote();
}

bool RemoteTransform2D::get_track_method() const { return track_backwards; }
void RemoteTransform2D::set_track_method(bool backward) {
	track_backwards = backward;
	if (is_inside_scene()) _update_remote();
}


void RemoteTransform2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_remote_node","path"),&RemoteTransform2D::set_remote_node);
	ObjectTypeDB::bind_method(_MD("get_remote_node"),&RemoteTransform2D::get_remote_node);

	ObjectTypeDB::bind_method(_MD("set_track_pos", "track_pos"), &RemoteTransform2D::set_track_pos);
	ObjectTypeDB::bind_method(_MD("get_track_pos"), &RemoteTransform2D::get_track_pos);

	ObjectTypeDB::bind_method(_MD("set_track_rot", "track_rot"), &RemoteTransform2D::set_track_rot);
	ObjectTypeDB::bind_method(_MD("get_track_rot"), &RemoteTransform2D::get_track_rot);

	ObjectTypeDB::bind_method(_MD("set_track_scale", "track_scale"), &RemoteTransform2D::set_track_scale);
	ObjectTypeDB::bind_method(_MD("get_track_scale"), &RemoteTransform2D::get_track_scale);

	ObjectTypeDB::bind_method(_MD("set_track_method", "track_backwards"), &RemoteTransform2D::set_track_method);
	ObjectTypeDB::bind_method(_MD("get_track_method"), &RemoteTransform2D::get_track_method);

	ADD_PROPERTY( PropertyInfo(Variant::NODE_PATH,"remote_path"),_SCS("set_remote_node"),_SCS("get_remote_node"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"track_pos"),_SCS("set_track_pos"),_SCS("get_track_pos") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"track_rot"),_SCS("set_track_rot"),_SCS("get_track_rot") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"track_scale"),_SCS("set_track_scale"),_SCS("get_track_scale") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"track_backwards"),_SCS("set_track_method"),_SCS("get_track_method"));
}

RemoteTransform2D::RemoteTransform2D() {

	cache=0;
	track_pos=true;
	track_rot=true;
	track_scale=true;
}


