/*************************************************************************/
/*  navigation_region_3d.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/os/thread.h"
#include "mesh_instance_3d.h"
#include "navigation_3d.h"
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
		if (navigation) {
			NavigationServer3D::get_singleton()->region_set_map(region, navigation->get_rid());
		}
	}

	if (debug_view) {
		MeshInstance3D *dm = Object::cast_to<MeshInstance3D>(debug_view);
		if (is_enabled()) {
			dm->set_material_override(get_tree()->get_debug_navigation_material());
		} else {
			dm->set_material_override(get_tree()->get_debug_navigation_disabled_material());
		}
	}

	update_gizmo();
}

bool NavigationRegion3D::is_enabled() const {
	return enabled;
}

/////////////////////////////

void NavigationRegion3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			Node3D *c = this;
			while (c) {
				navigation = Object::cast_to<Navigation3D>(c);
				if (navigation) {
					if (enabled) {
						NavigationServer3D::get_singleton()->region_set_map(region, navigation->get_rid());
					}
					break;
				}

				c = c->get_parent_spatial();
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
			if (navigation) {
				NavigationServer3D::get_singleton()->region_set_map(region, RID());
			}

			if (debug_view) {
				debug_view->queue_delete();
				debug_view = nullptr;
			}
			navigation = nullptr;
		} break;
	}
}

void NavigationRegion3D::set_navigation_mesh(const Ref<NavigationMesh> &p_navmesh) {
	if (p_navmesh == navmesh) {
		return;
	}

	if (navmesh.is_valid()) {
		navmesh->remove_change_receptor(this);
	}

	navmesh = p_navmesh;

	if (navmesh.is_valid()) {
		navmesh->add_change_receptor(this);
	}

	NavigationServer3D::get_singleton()->region_set_navmesh(region, p_navmesh);

	if (debug_view && navmesh.is_valid()) {
		Object::cast_to<MeshInstance3D>(debug_view)->set_mesh(navmesh->get_debug_mesh());
	}

	emit_signal("navigation_mesh_changed");

	update_gizmo();
	update_configuration_warning();
}

Ref<NavigationMesh> NavigationRegion3D::get_navigation_mesh() const {
	return navmesh;
}

struct BakeThreadsArgs {
	NavigationRegion3D *nav_region;
};

void _bake_navigation_mesh(void *p_user_data) {
	BakeThreadsArgs *args = static_cast<BakeThreadsArgs *>(p_user_data);

	if (args->nav_region->get_navigation_mesh().is_valid()) {
		Ref<NavigationMesh> nav_mesh = args->nav_region->get_navigation_mesh()->duplicate();

		NavigationServer3D::get_singleton()->region_bake_navmesh(nav_mesh, args->nav_region);
		args->nav_region->call_deferred("_bake_finished", nav_mesh);
		memdelete(args);
	} else {
		ERR_PRINT("Can't bake the navigation mesh if the `NavigationMesh` resource doesn't exist");
		args->nav_region->call_deferred("_bake_finished", Ref<NavigationMesh>());
		memdelete(args);
	}
}

void NavigationRegion3D::bake_navigation_mesh() {
	ERR_FAIL_COND(bake_thread != nullptr);

	BakeThreadsArgs *args = memnew(BakeThreadsArgs);
	args->nav_region = this;

	bake_thread = Thread::create(_bake_navigation_mesh, args);
	ERR_FAIL_COND(bake_thread == nullptr);
}

void NavigationRegion3D::_bake_finished(Ref<NavigationMesh> p_nav_mesh) {
	set_navigation_mesh(p_nav_mesh);
	bake_thread = nullptr;
}

String NavigationRegion3D::get_configuration_warning() const {
	if (!is_visible_in_tree() || !is_inside_tree()) {
		return String();
	}

	String warning = Node3D::get_configuration_warning();

	if (!navmesh.is_valid()) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("A NavigationMesh resource must be set or created for this node to work.");
	}

	const Node3D *c = this;
	while (c) {
		if (Object::cast_to<Navigation3D>(c)) {
			return warning;
		}

		c = Object::cast_to<Node3D>(c->get_parent());
	}

	if (!warning.empty()) {
		warning += "\n\n";
	}
	return warning + TTR("NavigationRegion3D must be a child or grandchild to a Navigation3D node. It only provides navigation data.");
}

void NavigationRegion3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_navigation_mesh", "navmesh"), &NavigationRegion3D::set_navigation_mesh);
	ClassDB::bind_method(D_METHOD("get_navigation_mesh"), &NavigationRegion3D::get_navigation_mesh);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &NavigationRegion3D::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &NavigationRegion3D::is_enabled);

	ClassDB::bind_method(D_METHOD("bake_navigation_mesh"), &NavigationRegion3D::bake_navigation_mesh);
	ClassDB::bind_method(D_METHOD("_bake_finished", "nav_mesh"), &NavigationRegion3D::_bake_finished);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navmesh", PROPERTY_HINT_RESOURCE_TYPE, "NavigationMesh"), "set_navigation_mesh", "get_navigation_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");

	ADD_SIGNAL(MethodInfo("navigation_mesh_changed"));
	ADD_SIGNAL(MethodInfo("bake_finished"));
}

void NavigationRegion3D::_changed_callback(Object *p_changed, const char *p_prop) {
	update_gizmo();
	update_configuration_warning();
}

NavigationRegion3D::NavigationRegion3D() {
	set_notify_transform(true);
	region = NavigationServer3D::get_singleton()->region_create();
}

NavigationRegion3D::~NavigationRegion3D() {
	if (navmesh.is_valid()) {
		navmesh->remove_change_receptor(this);
	}
	NavigationServer3D::get_singleton()->free(region);
}
