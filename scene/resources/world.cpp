/*************************************************************************/
/*  world.cpp                                                            */
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

#include "world.h"

#include "core/math/camera_matrix.h"
#include "core/math/octree.h"
#include "scene/3d/camera.h"
#include "scene/3d/visibility_notifier.h"
#include "scene/scene_string_names.h"
#include "servers/navigation_server.h"

struct SpatialIndexer {
	Octree<VisibilityNotifier> octree;

	struct NotifierData {
		AABB aabb;
		OctreeElementID id;
	};

	Map<VisibilityNotifier *, NotifierData> notifiers;
	struct CameraData {
		Map<VisibilityNotifier *, uint64_t> notifiers;
	};

	Map<Camera *, CameraData> cameras;

	enum {
		VISIBILITY_CULL_MAX = 32768
	};

	Vector<VisibilityNotifier *> cull;

	bool changed;
	uint64_t pass;
	uint64_t last_frame;

	void _notifier_add(VisibilityNotifier *p_notifier, const AABB &p_rect) {
		ERR_FAIL_COND(notifiers.has(p_notifier));
		notifiers[p_notifier].aabb = p_rect;
		notifiers[p_notifier].id = octree.create(p_notifier, p_rect);
		changed = true;
	}

	void _notifier_update(VisibilityNotifier *p_notifier, const AABB &p_rect) {
		Map<VisibilityNotifier *, NotifierData>::Element *E = notifiers.find(p_notifier);
		ERR_FAIL_COND(!E);
		if (E->get().aabb == p_rect) {
			return;
		}

		E->get().aabb = p_rect;
		octree.move(E->get().id, E->get().aabb);
		changed = true;
	}

	void _notifier_remove(VisibilityNotifier *p_notifier) {
		Map<VisibilityNotifier *, NotifierData>::Element *E = notifiers.find(p_notifier);
		ERR_FAIL_COND(!E);

		octree.erase(E->get().id);
		notifiers.erase(p_notifier);

		List<Camera *> removed;
		for (Map<Camera *, CameraData>::Element *F = cameras.front(); F; F = F->next()) {
			Map<VisibilityNotifier *, uint64_t>::Element *G = F->get().notifiers.find(p_notifier);

			if (G) {
				F->get().notifiers.erase(G);
				removed.push_back(F->key());
			}
		}

		while (!removed.empty()) {
			p_notifier->_exit_camera(removed.front()->get());
			removed.pop_front();
		}

		changed = true;
	}

	void _add_camera(Camera *p_camera) {
		ERR_FAIL_COND(cameras.has(p_camera));
		CameraData vd;
		cameras[p_camera] = vd;
		changed = true;
	}

	void _update_camera(Camera *p_camera) {
		Map<Camera *, CameraData>::Element *E = cameras.find(p_camera);
		ERR_FAIL_COND(!E);
		changed = true;
	}

	void _remove_camera(Camera *p_camera) {
		ERR_FAIL_COND(!cameras.has(p_camera));
		List<VisibilityNotifier *> removed;
		for (Map<VisibilityNotifier *, uint64_t>::Element *E = cameras[p_camera].notifiers.front(); E; E = E->next()) {
			removed.push_back(E->key());
		}

		while (!removed.empty()) {
			removed.front()->get()->_exit_camera(p_camera);
			removed.pop_front();
		}

		cameras.erase(p_camera);
	}

	void _update(uint64_t p_frame) {
		if (p_frame == last_frame) {
			return;
		}
		last_frame = p_frame;

		if (!changed) {
			return;
		}

		for (Map<Camera *, CameraData>::Element *E = cameras.front(); E; E = E->next()) {
			pass++;

			// prepare camera info
			Camera *c = E->key();
			Vector3 cam_pos = c->get_global_transform().origin;
			Vector<Plane> planes = c->get_frustum();
			bool cam_is_ortho = c->get_projection() == Camera::PROJECTION_ORTHOGONAL;

			int culled = octree.cull_convex(planes, cull.ptrw(), cull.size());

			VisibilityNotifier **ptr = cull.ptrw();

			List<VisibilityNotifier *> added;
			List<VisibilityNotifier *> removed;

			for (int i = 0; i < culled; i++) {
				//notifiers in frustum

				// check and remove notifiers that have a max range
				VisibilityNotifier &nt = *ptr[i];
				if (nt.is_max_distance_active() && !cam_is_ortho) {
					Vector3 offset = nt.get_world_aabb_center() - cam_pos;
					if ((offset.length_squared() >= nt.get_max_distance_squared()) && !nt.inside_max_distance_leadin()) {
						// unordered remove
						cull.set(i, cull[culled - 1]);
						culled--;
						i--;
						continue;
					}
				}

				Map<VisibilityNotifier *, uint64_t>::Element *H = E->get().notifiers.find(ptr[i]);
				if (!H) {
					E->get().notifiers.insert(ptr[i], pass);
					added.push_back(ptr[i]);
				} else {
					H->get() = pass;
				}
			}

			for (Map<VisibilityNotifier *, uint64_t>::Element *F = E->get().notifiers.front(); F; F = F->next()) {
				if (F->get() != pass) {
					removed.push_back(F->key());
				}
			}

			while (!added.empty()) {
				added.front()->get()->_enter_camera(E->key());
				added.pop_front();
			}

			while (!removed.empty()) {
				E->get().notifiers.erase(removed.front()->get());
				removed.front()->get()->_exit_camera(E->key());
				removed.pop_front();
			}
		}
		changed = false;
	}

	SpatialIndexer() {
		pass = 0;
		last_frame = 0;
		changed = false;
		cull.resize(VISIBILITY_CULL_MAX);
	}
};

void World::_register_camera(Camera *p_camera) {
#ifndef _3D_DISABLED
	indexer->_add_camera(p_camera);
#endif
}

void World::_update_camera(Camera *p_camera) {
#ifndef _3D_DISABLED
	indexer->_update_camera(p_camera);
#endif
}
void World::_remove_camera(Camera *p_camera) {
#ifndef _3D_DISABLED
	indexer->_remove_camera(p_camera);
#endif
}

void World::_register_notifier(VisibilityNotifier *p_notifier, const AABB &p_rect) {
#ifndef _3D_DISABLED
	indexer->_notifier_add(p_notifier, p_rect);
#endif
}

void World::_update_notifier(VisibilityNotifier *p_notifier, const AABB &p_rect) {
#ifndef _3D_DISABLED
	indexer->_notifier_update(p_notifier, p_rect);
#endif
}

void World::_remove_notifier(VisibilityNotifier *p_notifier) {
#ifndef _3D_DISABLED
	indexer->_notifier_remove(p_notifier);
#endif
}

void World::_update(uint64_t p_frame) {
#ifndef _3D_DISABLED
	indexer->_update(p_frame);
#endif
}

RID World::get_space() const {
	return space;
}

RID World::get_scenario() const {
	return scenario;
}

RID World::get_navigation_map() const {
	return navigation_map;
}

void World::set_environment(const Ref<Environment> &p_environment) {
	if (environment == p_environment) {
		return;
	}

	environment = p_environment;
	if (environment.is_valid()) {
		VS::get_singleton()->scenario_set_environment(scenario, environment->get_rid());
	} else {
		VS::get_singleton()->scenario_set_environment(scenario, RID());
	}

	emit_changed();
}

Ref<Environment> World::get_environment() const {
	return environment;
}

void World::set_fallback_environment(const Ref<Environment> &p_environment) {
	if (fallback_environment == p_environment) {
		return;
	}

	fallback_environment = p_environment;
	if (fallback_environment.is_valid()) {
		VS::get_singleton()->scenario_set_fallback_environment(scenario, p_environment->get_rid());
	} else {
		VS::get_singleton()->scenario_set_fallback_environment(scenario, RID());
	}

	emit_changed();
}

Ref<Environment> World::get_fallback_environment() const {
	return fallback_environment;
}

PhysicsDirectSpaceState *World::get_direct_space_state() {
	return PhysicsServer::get_singleton()->space_get_direct_state(space);
}

void World::get_camera_list(List<Camera *> *r_cameras) {
	for (Map<Camera *, SpatialIndexer::CameraData>::Element *E = indexer->cameras.front(); E; E = E->next()) {
		r_cameras->push_back(E->key());
	}
}

void World::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_space"), &World::get_space);
	ClassDB::bind_method(D_METHOD("get_scenario"), &World::get_scenario);
	ClassDB::bind_method(D_METHOD("get_navigation_map"), &World::get_navigation_map);
	ClassDB::bind_method(D_METHOD("set_environment", "env"), &World::set_environment);
	ClassDB::bind_method(D_METHOD("get_environment"), &World::get_environment);
	ClassDB::bind_method(D_METHOD("set_fallback_environment", "env"), &World::set_fallback_environment);
	ClassDB::bind_method(D_METHOD("get_fallback_environment"), &World::get_fallback_environment);
	ClassDB::bind_method(D_METHOD("get_direct_space_state"), &World::get_direct_space_state);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "environment", PROPERTY_HINT_RESOURCE_TYPE, "Environment"), "set_environment", "get_environment");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "fallback_environment", PROPERTY_HINT_RESOURCE_TYPE, "Environment"), "set_fallback_environment", "get_fallback_environment");
	ADD_PROPERTY(PropertyInfo(Variant::_RID, "space", PROPERTY_HINT_NONE, "", 0), "", "get_space");
	ADD_PROPERTY(PropertyInfo(Variant::_RID, "scenario", PROPERTY_HINT_NONE, "", 0), "", "get_scenario");
	ADD_PROPERTY(PropertyInfo(Variant::_RID, "navigation_map", PROPERTY_HINT_NONE, "", 0), "", "get_navigation_map");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "direct_space_state", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsDirectSpaceState", 0), "", "get_direct_space_state");
}

World::World() {
	space = RID_PRIME(PhysicsServer::get_singleton()->space_create());
	scenario = RID_PRIME(VisualServer::get_singleton()->scenario_create());

	PhysicsServer::get_singleton()->space_set_active(space, true);
	PhysicsServer::get_singleton()->area_set_param(space, PhysicsServer::AREA_PARAM_GRAVITY, GLOBAL_DEF("physics/3d/default_gravity", 9.8));
	PhysicsServer::get_singleton()->area_set_param(space, PhysicsServer::AREA_PARAM_GRAVITY_VECTOR, GLOBAL_DEF("physics/3d/default_gravity_vector", Vector3(0, -1, 0)));
	PhysicsServer::get_singleton()->area_set_param(space, PhysicsServer::AREA_PARAM_LINEAR_DAMP, GLOBAL_DEF("physics/3d/default_linear_damp", 0.1));
	ProjectSettings::get_singleton()->set_custom_property_info("physics/3d/default_linear_damp", PropertyInfo(Variant::REAL, "physics/3d/default_linear_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"));
	PhysicsServer::get_singleton()->area_set_param(space, PhysicsServer::AREA_PARAM_ANGULAR_DAMP, GLOBAL_DEF("physics/3d/default_angular_damp", 0.1));
	ProjectSettings::get_singleton()->set_custom_property_info("physics/3d/default_angular_damp", PropertyInfo(Variant::REAL, "physics/3d/default_angular_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"));

	// Create default navigation map
	navigation_map = NavigationServer::get_singleton()->map_create();
	NavigationServer::get_singleton()->map_set_active(navigation_map, true);
	NavigationServer::get_singleton()->map_set_up(navigation_map, GLOBAL_DEF("navigation/3d/default_map_up", Vector3(0, 1, 0)));
	NavigationServer::get_singleton()->map_set_cell_size(navigation_map, GLOBAL_DEF("navigation/3d/default_cell_size", 0.25));
	NavigationServer::get_singleton()->map_set_cell_height(navigation_map, GLOBAL_DEF("navigation/3d/default_cell_height", 0.25));
	NavigationServer::get_singleton()->map_set_edge_connection_margin(navigation_map, GLOBAL_DEF("navigation/3d/default_edge_connection_margin", 0.25));

#ifdef _3D_DISABLED
	indexer = NULL;
#else
	indexer = memnew(SpatialIndexer);
#endif
}

World::~World() {
	PhysicsServer::get_singleton()->free(space);
	VisualServer::get_singleton()->free(scenario);
	NavigationServer::get_singleton()->free(navigation_map);

#ifndef _3D_DISABLED
	memdelete(indexer);
#endif
}
