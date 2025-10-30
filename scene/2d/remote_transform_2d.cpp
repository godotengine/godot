/**************************************************************************/
/*  remote_transform_2d.cpp                                               */
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

#include "remote_transform_2d.h"
#include "scene/scene_string_names.h"

void RemoteTransform2D::_update_cache() {
	cache = ObjectID();
	if (has_node(remote_node)) {
		Node *node = get_node(remote_node);
		if (!node || this == node || node->is_a_parent_of(this) || this->is_a_parent_of(node)) {
			return;
		}

		cache = node->get_instance_id();
	}
}

void RemoteTransform2D::_update_remote() {
	if (!is_inside_tree()) {
		return;
	}

	if (!cache.is_valid()) {
		return;
	}

	Node2D *n = ObjectDB::get_instance<Node2D>(cache);
	if (!n) {
		return;
	}

	if (!n->is_inside_tree()) {
		return;
	}

	//todo make faster
	if (use_global_coordinates) {
		if (update_remote_position && update_remote_rotation && update_remote_scale) {
			n->set_global_transform(get_global_transform());
		} else {
			Transform2D n_trans = n->get_global_transform();
			Transform2D our_trans = get_global_transform();
			Vector2 n_scale = n->get_scale();

			if (!update_remote_position) {
				our_trans.set_origin(n_trans.get_origin());
			}
			if (!update_remote_rotation) {
				our_trans.set_rotation(n_trans.get_rotation());
			}

			n->set_global_transform(our_trans);

			if (update_remote_scale) {
				n->set_scale(get_global_scale());
			} else {
				n->set_scale(n_scale);
			}
		}

	} else {
		if (update_remote_position && update_remote_rotation && update_remote_scale) {
			n->set_transform(get_transform());
		} else {
			Transform2D n_trans = n->get_transform();
			Transform2D our_trans = get_transform();
			Vector2 n_scale = n->get_scale();

			if (!update_remote_position) {
				our_trans.set_origin(n_trans.get_origin());
			}
			if (!update_remote_rotation) {
				our_trans.set_rotation(n_trans.get_rotation());
			}

			n->set_transform(our_trans);

			if (update_remote_scale) {
				n->set_scale(get_scale());
			} else {
				n->set_scale(n_scale);
			}
		}
	}
}

void RemoteTransform2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_cache();

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (!is_inside_tree()) {
				break;
			}

			if (cache.is_valid()) {
				_update_remote();
			}

		} break;
	}
}

void RemoteTransform2D::set_remote_node(const NodePath &p_remote_node) {
	remote_node = p_remote_node;
	if (is_inside_tree()) {
		_update_cache();
		_update_remote();
	}

	update_configuration_warning();
}

NodePath RemoteTransform2D::get_remote_node() const {
	return remote_node;
}

void RemoteTransform2D::set_use_global_coordinates(const bool p_enable) {
	use_global_coordinates = p_enable;
	_update_remote();
}

bool RemoteTransform2D::get_use_global_coordinates() const {
	return use_global_coordinates;
}

void RemoteTransform2D::set_update_position(const bool p_update) {
	update_remote_position = p_update;
	_update_remote();
}

bool RemoteTransform2D::get_update_position() const {
	return update_remote_position;
}

void RemoteTransform2D::set_update_rotation(const bool p_update) {
	update_remote_rotation = p_update;
	_update_remote();
}

bool RemoteTransform2D::get_update_rotation() const {
	return update_remote_rotation;
}

void RemoteTransform2D::set_update_scale(const bool p_update) {
	update_remote_scale = p_update;
	_update_remote();
}

bool RemoteTransform2D::get_update_scale() const {
	return update_remote_scale;
}

void RemoteTransform2D::force_update_cache() {
	_update_cache();
}

String RemoteTransform2D::get_configuration_warning() const {
	String warning = Node2D::get_configuration_warning();
	if (!has_node(remote_node) || !Object::cast_to<Node2D>(get_node(remote_node))) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("Path property must point to a valid Node2D node to work.");
	}

	return warning;
}

void RemoteTransform2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_remote_node", "path"), &RemoteTransform2D::set_remote_node);
	ClassDB::bind_method(D_METHOD("get_remote_node"), &RemoteTransform2D::get_remote_node);
	ClassDB::bind_method(D_METHOD("force_update_cache"), &RemoteTransform2D::force_update_cache);

	ClassDB::bind_method(D_METHOD("set_use_global_coordinates", "use_global_coordinates"), &RemoteTransform2D::set_use_global_coordinates);
	ClassDB::bind_method(D_METHOD("get_use_global_coordinates"), &RemoteTransform2D::get_use_global_coordinates);

	ClassDB::bind_method(D_METHOD("set_update_position", "update_remote_position"), &RemoteTransform2D::set_update_position);
	ClassDB::bind_method(D_METHOD("get_update_position"), &RemoteTransform2D::get_update_position);
	ClassDB::bind_method(D_METHOD("set_update_rotation", "update_remote_rotation"), &RemoteTransform2D::set_update_rotation);
	ClassDB::bind_method(D_METHOD("get_update_rotation"), &RemoteTransform2D::get_update_rotation);
	ClassDB::bind_method(D_METHOD("set_update_scale", "update_remote_scale"), &RemoteTransform2D::set_update_scale);
	ClassDB::bind_method(D_METHOD("get_update_scale"), &RemoteTransform2D::get_update_scale);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "remote_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_remote_node", "get_remote_node");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_global_coordinates"), "set_use_global_coordinates", "get_use_global_coordinates");

	ADD_GROUP("Update", "update_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "update_position"), "set_update_position", "get_update_position");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "update_rotation"), "set_update_rotation", "get_update_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "update_scale"), "set_update_scale", "get_update_scale");
}

RemoteTransform2D::RemoteTransform2D() {
	use_global_coordinates = true;
	update_remote_position = true;
	update_remote_rotation = true;
	update_remote_scale = true;

	set_notify_transform(true);
}
