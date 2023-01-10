/**************************************************************************/
/*  nav_region.h                                                          */
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

#ifndef NAV_REGION_H
#define NAV_REGION_H

#include "scene/3d/navigation.h"

#include "nav_rid.h"
#include "nav_utils.h"

#include <vector>

class NavMap;
class NavRegion;

class NavRegion : public NavRid {
	NavMap *map = nullptr;
	Transform transform;
	Ref<NavigationMesh> mesh;
	uint32_t navigation_layers = 1;
	float enter_cost = 0.0;
	float travel_cost = 1.0;
	Vector<gd::Edge::Connection> connections;

	bool polygons_dirty = true;

	/// Cache
	LocalVector<gd::Polygon> polygons;

public:
	void scratch_polygons() {
		polygons_dirty = true;
	}

	void set_map(NavMap *p_map);
	NavMap *get_map() const {
		return map;
	}

	void set_enter_cost(float p_enter_cost) { enter_cost = MAX(p_enter_cost, 0.0); }
	float get_enter_cost() const { return enter_cost; }

	void set_travel_cost(float p_travel_cost) { travel_cost = MAX(p_travel_cost, 0.0); }
	float get_travel_cost() const { return travel_cost; }

	void set_navigation_layers(uint32_t p_navigation_layers);
	uint32_t get_navigation_layers() const;

	void set_transform(Transform transform);
	const Transform &get_transform() const {
		return transform;
	}

	void set_mesh(Ref<NavigationMesh> p_mesh);
	const Ref<NavigationMesh> get_mesh() const {
		return mesh;
	}

	Vector<gd::Edge::Connection> &get_connections() {
		return connections;
	}
	int get_connections_count() const;
	Vector3 get_connection_pathway_start(int p_connection_id) const;
	Vector3 get_connection_pathway_end(int p_connection_id) const;

	LocalVector<gd::Polygon> const &get_polygons() const {
		return polygons;
	}

	bool sync();

	NavRegion();
	~NavRegion();

private:
	void update_polygons();
};

#endif // NAV_REGION_H
