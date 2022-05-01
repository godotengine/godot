/*************************************************************************/
/*  navigation_mesh_instance.h                                           */
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

#ifndef NAVIGATION_MESH_INSTANCE_H
#define NAVIGATION_MESH_INSTANCE_H

#include "scene/3d/spatial.h"
#include "scene/resources/mesh.h"
#include "scene/resources/navigation_mesh.h"

class Navigation;

class NavigationMeshInstance : public Spatial {
	GDCLASS(NavigationMeshInstance, Spatial);

	bool enabled;
	RID region;
	Ref<NavigationMesh> navmesh;

	Navigation *navigation = nullptr;
	Node *debug_view = nullptr;
	Thread bake_thread;

protected:
	void _notification(int p_what);
	static void _bind_methods();
	void _changed_callback(Object *p_changed, const char *p_prop);

public:
	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	RID get_region_rid() const;

	void set_navigation_mesh(const Ref<NavigationMesh> &p_navmesh);
	Ref<NavigationMesh> get_navigation_mesh() const;

	/// Bakes the navigation mesh in a dedicated thread; once done, automatically
	/// sets the new navigation mesh and emits a signal
	void bake_navigation_mesh();
	void _bake_finished(Ref<NavigationMesh> p_nav_mesh);

	String get_configuration_warning() const;

	NavigationMeshInstance();
	~NavigationMeshInstance();
};

#endif // NAVIGATION_MESH_INSTANCE_H
