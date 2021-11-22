/*************************************************************************/
/*  navigation_region_3d.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "navigation_region_3d.h"

#include "mesh_instance_3d.h"
#include "servers/navigation_server_3d.h"

void NavigationRegion3D::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}
	enabled = p_enabled;

	if (!is_inside_tree()) {
		return;
	}

	if (!enabled) {
		NavigationServer3D::get_singleton()->region_set_map(region, RID());
	} else {
		NavigationServer3D::get_singleton()->region_set_map(region, get_world_3d()->get_navigation_map());
	}

	if (debug_view) {
		MeshInstance3D *dm = Object::cast_to<MeshInstance3D>(debug_view);
		if (is_enabled()) {
			dm->set_material_override(get_tree()->get_debug_navigation_material());
		} else {
			dm->set_material_override(get_tree()->get_debug_navigation_disabled_material());
		}
	}

	update_gizmos();
}

bool NavigationRegion3D::is_enabled() const {
	return enabled;
}

void NavigationRegion3D::set_layers(uint32_t p_layers) {
	NavigationServer3D::get_singleton()->region_set_layers(region, p_layers);
}

uint32_t NavigationRegion3D::get_layers() const {
	return NavigationServer3D::get_singleton()->region_get_layers(region);
}

/////////////////////////////

void NavigationRegion3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (enabled) {
				NavigationServer3D::get_singleton()->region_set_map(region, get_world_3d()->get_navigation_map());
			}

			if (navmesh.is_valid() && get_tree()->is_debugging_navigation_hint()) {
				MeshInstance3D *dm = memnew(MeshInstance3D);
				dm->set_mesh(navmesh->get_debug_mesh());
				if (is_enabled()) {
					dm->set_material_override(get_tree()->get_debug_navigation_material());
				} else {
					dm->set_material_override(get_tree()->get_debug_navigation_disabled_material());
				}
				add_child(dm);
				debug_view = dm;
			}

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			NavigationServer3D::get_singleton()->region_set_transform(region, get_global_transform());

		} break;
		case NOTIFICATION_EXIT_TREE: {
			NavigationServer3D::get_singleton()->region_set_map(region, RID());

			if (debug_view) {
				debug_view->queue_delete();
				debug_view = nullptr;
			}
		} break;
	}
}

void NavigationRegion3D::set_navigation_mesh(const Ref<NavigationMesh> &p_navmesh) {
	if (p_navmesh == navmesh) {
		return;
	}

	if (navmesh.is_valid()) {
		navmesh->disconnect("changed", callable_mp(this, &NavigationRegion3D::_navigation_changed));
	}

	navmesh = p_navmesh;

	if (navmesh.is_valid()) {
		navmesh->connect("changed", callable_mp(this, &NavigationRegion3D::_navigation_changed));
	}

	NavigationServer3D::get_singleton()->region_set_navmesh(region, p_navmesh);

	if (debug_view && navmesh.is_valid()) {
		Object::cast_to<MeshInstance3D>(debug_view)->set_mesh(navmesh->get_debug_mesh());
	}

	emit_signal(SNAME("navigation_mesh_changed"));

	update_gizmos();
	update_configuration_warnings();
}

Ref<NavigationMesh> NavigationRegion3D::get_navigation_mesh() const {
	return navmesh;
}

struct BakeThreadsArgs {
	NavigationRegion3D *nav_region = nullptr;
};

void _bake_navigation_mesh(void *p_user_data) {
	BakeThreadsArgs *args = static_cast<BakeThreadsArgs *>(p_user_data);

	if (args->nav_region->get_navigation_mesh().is_valid()) {
		Ref<NavigationMesh> nav_mesh = args->nav_region->get_navigation_mesh()->duplicate();

		NavigationServer3D::get_singleton()->region_bake_navmesh(nav_mesh, args->nav_region);
		args->nav_region->call_deferred(SNAME("_bake_finished"), nav_mesh);
		memdelete(args);
	} else {
		ERR_PRINT("Can't bake the navigation mesh if the `NavigationMesh` resource doesn't exist");
		args->nav_region->call_deferred(SNAME("_bake_finished"), Ref<NavigationMesh>());
		memdelete(args);
	}
}

void NavigationRegion3D::bake_navigation_mesh() {
	ERR_FAIL_COND_MSG(bake_thread.is_started(), "Unable to start another bake request. The navigation mesh bake thread is already baking a navigation mesh.");

	BakeThreadsArgs *args = memnew(BakeThreadsArgs);
	args->nav_region = this;

	bake_thread.start(_bake_navigation_mesh, args);
}

void NavigationRegion3D::_bake_finished(Ref<NavigationMesh> p_nav_mesh) {
	set_navigation_mesh(p_nav_mesh);
	bake_thread.wait_to_finish();
	emit_signal(SNAME("bake_finished"));
}

TypedArray<String> NavigationRegion3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (is_visible_in_tree() && is_inside_tree()) {
		if (!navmesh.is_valid()) {
			warnings.push_back(TTR("A NavigationMesh resource must be set or created for this node to work."));
		}
	}

	return warnings;
}

void NavigationRegion3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_navigation_mesh", "navmesh"), &NavigationRegion3D::set_navigation_mesh);
	ClassDB::bind_method(D_METHOD("get_navigation_mesh"), &NavigationRegion3D::get_navigation_mesh);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &NavigationRegion3D::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &NavigationRegion3D::is_enabled);

	ClassDB::bind_method(D_METHOD("set_layers", "layers"), &NavigationRegion3D::set_layers);
	ClassDB::bind_method(D_METHOD("get_layers"), &NavigationRegion3D::get_layers);

	ClassDB::bind_method(D_METHOD("bake_navigation_mesh"), &NavigationRegion3D::bake_navigation_mesh);
	ClassDB::bind_method(D_METHOD("_bake_finished", "nav_mesh"), &NavigationRegion3D::_bake_finished);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navmesh", PROPERTY_HINT_RESOURCE_TYPE, "NavigationMesh"), "set_navigation_mesh", "get_navigation_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "layers", PROPERTY_HINT_LAYERS_3D_NAVIGATION), "set_layers", "get_layers");

	ADD_SIGNAL(MethodInfo("navigation_mesh_changed"));
	ADD_SIGNAL(MethodInfo("bake_finished"));
}

void NavigationRegion3D::_navigation_changed() {
	update_gizmos();
	update_configuration_warnings();
}

NavigationRegion3D::NavigationRegion3D() {
	set_notify_transform(true);
	region = NavigationServer3D::get_singleton()->region_create();
}

NavigationRegion3D::~NavigationRegion3D() {
	if (navmesh.is_valid()) {
		navmesh->disconnect("changed", callable_mp(this, &NavigationRegion3D::_navigation_changed));
	}
	NavigationServer3D::get_singleton()->free(region);
}
