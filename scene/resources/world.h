/**************************************************************************/
/*  world.h                                                               */
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

#ifndef WORLD_H
#define WORLD_H

#include "core/resource.h"
#include "scene/resources/environment.h"
#include "servers/physics_server.h"
#include "servers/visual_server.h"

class Camera;
class VisibilityNotifier;
struct SpatialIndexer;

class World : public Resource {
	GDCLASS(World, Resource);
	RES_BASE_EXTENSION("world");

private:
	RID space;
	RID scenario;
	RID navigation_map;

	SpatialIndexer *indexer;
	Ref<Environment> environment;
	Ref<Environment> fallback_environment;

protected:
	static void _bind_methods();

	friend class Camera;
	friend class VisibilityNotifier;

	void _register_camera(Camera *p_camera);
	void _update_camera(Camera *p_camera);
	void _remove_camera(Camera *p_camera);

	void _register_notifier(VisibilityNotifier *p_notifier, const AABB &p_rect);
	void _update_notifier(VisibilityNotifier *p_notifier, const AABB &p_rect);
	void _remove_notifier(VisibilityNotifier *p_notifier);
	friend class Viewport;
	void _update(uint64_t p_frame);

public:
	RID get_space() const;
	RID get_scenario() const;
	RID get_navigation_map() const;

	void set_environment(const Ref<Environment> &p_environment);
	Ref<Environment> get_environment() const;

	void set_fallback_environment(const Ref<Environment> &p_environment);
	Ref<Environment> get_fallback_environment() const;

	void get_camera_list(List<Camera *> *r_cameras);

	PhysicsDirectSpaceState *get_direct_space_state();

	World();
	~World();
};

#endif // WORLD_H
