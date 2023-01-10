/**************************************************************************/
/*  visibility_notifier.h                                                 */
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

#ifndef VISIBILITY_NOTIFIER_H
#define VISIBILITY_NOTIFIER_H

#include "scene/3d/cull_instance.h"

class World;
class Camera;
class VisibilityNotifier : public CullInstance {
	GDCLASS(VisibilityNotifier, CullInstance);

	Ref<World> world;
	Set<Camera *> cameras;

	AABB aabb;
	Vector3 _world_aabb_center;

	// if using rooms and portals
	RID _cull_instance_rid;
	bool _in_gameplay;

	bool _max_distance_active;
	real_t _max_distance;
	real_t _max_distance_squared;

	// This is a first number of frames where distance objects
	// are forced seen as visible, to make sure their animations
	// and physics positions etc are something reasonable.
	uint32_t _max_distance_leadin_counter;

protected:
	virtual void _screen_enter() {}
	virtual void _screen_exit() {}
	virtual void _refresh_portal_mode();

	void _notification(int p_what);
	static void _bind_methods();
	friend struct SpatialIndexer;

	void _enter_camera(Camera *p_camera);
	void _exit_camera(Camera *p_camera);

public:
	void set_aabb(const AABB &p_aabb);
	AABB get_aabb() const;
	bool is_on_screen() const;

	// This is only currently kept up to date if max_distance is active
	const Vector3 &get_world_aabb_center() const { return _world_aabb_center; }

	void set_max_distance(real_t p_max_distance);
	real_t get_max_distance() const { return _max_distance; }
	real_t get_max_distance_squared() const { return _max_distance_squared; }
	bool is_max_distance_active() const { return _max_distance_active; }
	bool inside_max_distance_leadin() {
		if (!_max_distance_leadin_counter) {
			return false;
		}
		_max_distance_leadin_counter--;
		return true;
	}

	VisibilityNotifier();
	~VisibilityNotifier();
};

class VisibilityEnabler : public VisibilityNotifier {
	GDCLASS(VisibilityEnabler, VisibilityNotifier);

public:
	enum Enabler {
		ENABLER_PAUSE_ANIMATIONS,
		ENABLER_FREEZE_BODIES,
		ENABLER_MAX
	};

protected:
	virtual void _screen_enter();
	virtual void _screen_exit();

	bool visible;

	void _find_nodes(Node *p_node);

	Map<Node *, Variant> nodes;
	void _node_removed(Node *p_node);
	bool enabler[ENABLER_MAX];

	void _change_node_state(Node *p_node, bool p_enabled);

	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_enabler(Enabler p_enabler, bool p_enable);
	bool is_enabler_enabled(Enabler p_enabler) const;

	VisibilityEnabler();
};

VARIANT_ENUM_CAST(VisibilityEnabler::Enabler);

#endif // VISIBILITY_NOTIFIER_H
