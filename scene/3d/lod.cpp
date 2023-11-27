/**************************************************************************/
/*  lod.cpp                                                               */
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

#include "lod.h"

#include "core/engine.h"
#include "scene/3d/visual_instance.h"

void LOD::_lod_register() {
	if (!data.registered) {
		Ref<World> world = get_world();
		ERR_FAIL_COND(!world.is_valid());
		world->_register_lod(this, data.queue_id);
		data.registered = true;
	}
}

void LOD::_lod_unregister() {
	if (data.registered) {
		Ref<World> world = get_world();
		ERR_FAIL_COND(!world.is_valid());
		world->_unregister_lod(this, data.queue_id);
		data.registered = false;
	}
}

void LOD::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (is_visible_in_tree()) {
				_lod_register();
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (is_visible_in_tree()) {
				_lod_unregister();
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_inside_tree()) {
				if (is_visible_in_tree()) {
					_lod_register();
				} else {
					_lod_unregister();
				}
			}
		} break;
		default:
			break;
	}
}

void LOD::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_hysteresis", "distance"), &LOD::set_hysteresis);
	ClassDB::bind_method(D_METHOD("get_hysteresis"), &LOD::get_hysteresis);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "hysteresis", PROPERTY_HINT_RANGE, "0,1024,0.01,or_greater"), "set_hysteresis", "get_hysteresis");

	ClassDB::bind_method(D_METHOD("set_lod_priority", "priority"), &LOD::set_lod_priority);
	ClassDB::bind_method(D_METHOD("get_lod_priority"), &LOD::get_lod_priority);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_priority", PROPERTY_HINT_RANGE, "0,4"), "set_lod_priority", "get_lod_priority");
}

void LOD::set_hysteresis(real_t p_distance) {
	data.hysteresis = CLAMP((float)p_distance, 0.0f, 100000.0f);
}

void LOD::set_lod_priority(int p_priority) {
	// We are just using priority as a user facing
	// description. Internally we use queues.
	int queue_id = CLAMP(p_priority, 0, 4);

	if (queue_id == data.queue_id) {
		return;
	}

	if (is_inside_tree()) {
		// If already in the world, we must remove the LOD
		// and re-add in a different queue.
		Ref<World> world = get_world();
		ERR_FAIL_COND(!world.is_valid());
		world->_unregister_lod(this, data.queue_id);

		data.queue_id = queue_id;

		world->_register_lod(this, data.queue_id);
	} else {
		data.queue_id = queue_id;
	}
}

void LOD::_lod_pre_save() {
	// The pre-save is primarily for the editor,
	// to ensure that saved scenes do not have unnecessary
	// diffs because of changes to which LOD child is active.
	// We standardize on just showing the first child in saved scenes.
	_update_child_distances();

	int32_t num_lods = data.lod_children.size();

	// Make first visible, and all others invisible.
	data.current_lod_child = 0;

	for (int32_t n = 0; n < num_lods; n++) {
		uint32_t child_id = data.lod_children[n].child_id;
		Spatial *child = Object::cast_to<Spatial>(get_child(child_id));

		if (child) {
			child->set_visible(n == data.current_lod_child);
		}
	}
}

// Returns whether a visibility change was triggered.
bool LOD::_lod_update(float p_camera_dist_squared) {
	// This should later be done as a one-off, as
	// this is expensive.
	_update_child_distances();

	int32_t num_lods = data.lod_children.size();

	// LOD node has no valid children to update.
	if (!num_lods) {
		return false;
	}

	data.current_lod_child = MIN(data.current_lod_child, num_lods - 1);
	int32_t curr = data.current_lod_child;

	float dist = Math::sqrt(p_camera_dist_squared);

	bool changed = true;

	while (changed) {
		changed = false;
		if ((curr < num_lods - 1) && (dist >= (data.lod_children[curr + 1].distance) + data.hysteresis)) {
			// Lower detail.
			curr += 1;
			changed = true;
		}

		if (curr && (dist < data.lod_children[curr].distance)) {
			// Increase detail.
			curr -= 1;
			changed = true;
		}
	}

	// No change?
	if ((curr == data.current_lod_child) && (data.current_lod_node == get_child(data.lod_children[curr].child_id))) {
		return false;
	}

	data.current_lod_child = curr;

	// Make current visible, and all others invisible.
	for (int32_t n = 0; n < num_lods; n++) {
		uint32_t child_id = data.lod_children[n].child_id;

		Spatial *child = Object::cast_to<Spatial>(get_child(child_id));

		if (child) {
			child->set_visible(n == curr);
			if (n == curr) {
				data.current_lod_node = child;
			}
		}
	}

	return true;
}

void LOD::_update_child_distances() {
	// Reserve enough space for all children, assuming they are all valid.
	LODChild *lod_children = (LODChild *)alloca(sizeof(LODChild) * get_child_count());

	// Reset prior to loop.
	float total_dist = 0.0f;
	uint32_t valid_count = 0;

#ifdef TOOLS_ENABLED
	bool is_editor = Engine::get_singleton()->is_editor_hint();
	uint32_t visible_count = 0;
#endif

	// Check every possible node child, not all will be valid lod children.
	for (int32_t n = 0; n < get_child_count(); n++) {
		// Destination for a valid child.
		LODChild &lod_child = lod_children[valid_count];

		const Spatial *child = Object::cast_to<Spatial>(get_child(n));
		if (child) {
			// Fill the data.
			lod_child.distance = total_dist;
			lod_child.child_id = n;

			// Keep running total of the distance range used by each lod child.
			total_dist += child->get_lod_range();

#ifdef TOOLS_ENABLED
			if (is_editor && child->is_visible()) {
				visible_count++;
			}
#endif
			valid_count++;
		}
	}

	// Size the actual vector, and copy data across.
	data.lod_children.resize(valid_count);
	if (valid_count) {
		memcpy(&data.lod_children[0], lod_children, valid_count * sizeof(LODChild));
	}

#ifdef TOOLS_ENABLED
	// Something external has changed the visibilities of the children,
	// such as the editor.
	if (is_editor && visible_count != 1) {
		// Force the current child to reset.
		data.current_lod_child = -1;
	}
#endif
}
