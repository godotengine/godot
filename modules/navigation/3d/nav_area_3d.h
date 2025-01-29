/**************************************************************************/
/*  nav_area_3d.h                                                         */
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

#ifndef NAV_AREA_3D_H
#define NAV_AREA_3D_H

#include "../nav_rid.h"
#include "../nav_utils.h"

#include "core/math/aabb.h"
#include "core/os/rw_lock.h"
#include "core/templates/self_list.h"
#include "servers/navigation_server_3d.h"

class NavMap;

class NavArea3D : public NavRid {
	RWLock area_rwlock;

	NavMap *map = nullptr;

	NavigationServer3D::AreaShapeType3D shape_type = NavigationServer3D::AreaShapeType3D::AREA_SHAPE_NONE;
	bool enabled = true;
	AABB bounds;

	Vector3 position;
	uint32_t navigation_layers = 1;
	int priority = 0;
	float height = 1.0;
	ObjectID owner_id;

	Vector3 size = Vector3(1.0, height, 1.0);
	float radius = 1.0;
	LocalVector<Vector3> vertices;

	bool area_dirty = true;

	SelfList<NavArea3D> sync_dirty_request_list_element;

private:
	void _update_bounds();

public:
	NavArea3D();
	~NavArea3D();

	void set_shape_type(NavigationServer3D::AreaShapeType3D p_shape_type);
	NavigationServer3D::AreaShapeType3D get_shape_type() const { return shape_type; }

	void set_enabled(bool p_enabled);
	bool get_enabled() const { return enabled; }

	void set_map(NavMap *p_map);
	NavMap *get_map() const { return map; }

	void set_position(Vector3 p_position);
	Vector3 get_position() const { return position; }

	void set_height(float p_height);
	float get_height() const { return height; }

	void set_navigation_layers(uint32_t p_navigation_layers);
	uint32_t get_navigation_layers() const { return navigation_layers; }

	void set_priority(int p_priority);
	int get_priority() const { return priority; }

	void set_owner_id(ObjectID p_owner_id) { owner_id = p_owner_id; }
	ObjectID get_owner_id() const { return owner_id; }

	void set_size(Vector3 p_size);
	Vector3 get_size() const { return size; }

	void set_radius(float p_radius);
	float get_radius() const { return radius; }

	void set_vertices(const Vector<Vector3> &p_vertices);
	const LocalVector<Vector3> &get_vertices() const { return vertices; }

	AABB get_bounds() const { return bounds; }

	bool has_point(const Vector3 &p_point);

	bool sync();
	void request_sync();
	void cancel_sync_request();
};

#endif // NAV_AREA_3D_H
