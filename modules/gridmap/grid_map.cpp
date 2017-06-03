/*************************************************************************/
/*  grid_map.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "grid_map.h"
#include "message_queue.h"
#include "scene/3d/light.h"
#include "scene/resources/surface_tool.h"
#include "servers/visual_server.h"

#include "io/marshalls.h"
#include "os/os.h"
#include "scene/resources/mesh_library.h"
#include "scene/scene_string_names.h"

bool GridMap::_set(const StringName &p_name, const Variant &p_value) {

	String name = p_name;

	if (name == "theme") {

		set_theme(p_value);
	} else if (name == "cell_size") {
		set_cell_size(p_value);
	} else if (name == "cell_octant_size") {
		set_octant_size(p_value);
	} else if (name == "cell_center_x") {
		set_center_x(p_value);
	} else if (name == "cell_center_y") {
		set_center_y(p_value);
	} else if (name == "cell_center_z") {
		set_center_z(p_value);
	} else if (name == "cell_scale") {
		set_cell_scale(p_value);
		/*	} else if (name=="cells") {
		PoolVector<int> cells = p_value;
		int amount=cells.size();
		PoolVector<int>::Read r = cells.read();
		ERR_FAIL_COND_V(amount&1,false); // not even
		cell_map.clear();
		for(int i=0;i<amount/3;i++) {


			IndexKey ik;
			ik.key=decode_uint64(&r[i*3]);
			Cell cell;
			cell.cell=uint32_t(r[i*+1]);
			cell_map[ik]=cell;

		}
		_recreate_octant_data();*/
	} else if (name == "data") {

		Dictionary d = p_value;

		if (d.has("cells")) {

			PoolVector<int> cells = d["cells"];
			int amount = cells.size();
			PoolVector<int>::Read r = cells.read();
			ERR_FAIL_COND_V(amount % 3, false); // not even
			cell_map.clear();
			for (int i = 0; i < amount / 3; i++) {

				IndexKey ik;
				ik.key = decode_uint64((const uint8_t *)&r[i * 3]);
				Cell cell;
				cell.cell = decode_uint32((const uint8_t *)&r[i * 3 + 2]);
				cell_map[ik] = cell;
			}
		}
		_recreate_octant_data();

	} else if (name.begins_with("areas/")) {
		int which = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		if (what == "bounds") {
			ERR_FAIL_COND_V(area_map.has(which), false);
			create_area(which, p_value);
			return true;
		}

		ERR_FAIL_COND_V(!area_map.has(which), false);

		if (what == "name")
			area_set_name(which, p_value);
		else if (what == "disable_distance")
			area_set_portal_disable_distance(which, p_value);
		else if (what == "exterior_portal")
			area_set_portal_disable_color(which, p_value);
		else
			return false;
	} else
		return false;

	return true;
}

bool GridMap::_get(const StringName &p_name, Variant &r_ret) const {

	String name = p_name;

	if (name == "theme") {
		r_ret = get_theme();
	} else if (name == "cell_size") {
		r_ret = get_cell_size();
	} else if (name == "cell_octant_size") {
		r_ret = get_octant_size();
	} else if (name == "cell_center_x") {
		r_ret = get_center_x();
	} else if (name == "cell_center_y") {
		r_ret = get_center_y();
	} else if (name == "cell_center_z") {
		r_ret = get_center_z();
	} else if (name == "cell_scale") {
		r_ret = cell_scale;
	} else if (name == "data") {

		Dictionary d;

		PoolVector<int> cells;
		cells.resize(cell_map.size() * 3);
		{
			PoolVector<int>::Write w = cells.write();
			int i = 0;
			for (Map<IndexKey, Cell>::Element *E = cell_map.front(); E; E = E->next(), i++) {

				encode_uint64(E->key().key, (uint8_t *)&w[i * 3]);
				encode_uint32(E->get().cell, (uint8_t *)&w[i * 3 + 2]);
			}
		}

		d["cells"] = cells;

		r_ret = d;
	} else if (name.begins_with("areas/")) {
		int which = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		if (what == "bounds")
			r_ret = area_get_bounds(which);
		else if (what == "name")
			r_ret = area_get_name(which);
		else if (what == "disable_distance")
			r_ret = area_get_portal_disable_distance(which);
		else if (what == "exterior_portal")
			r_ret = area_is_exterior_portal(which);
		else
			return false;
	} else
		return false;

	return true;
}

void GridMap::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back(PropertyInfo(Variant::OBJECT, "theme", PROPERTY_HINT_RESOURCE_TYPE, "MeshLibrary"));
	p_list->push_back(PropertyInfo(Variant::NIL, "Cell", PROPERTY_HINT_NONE, "cell_", PROPERTY_USAGE_GROUP));
	p_list->push_back(PropertyInfo(Variant::REAL, "cell_size", PROPERTY_HINT_RANGE, "0.01,16384,0.01"));
	p_list->push_back(PropertyInfo(Variant::INT, "cell_octant_size", PROPERTY_HINT_RANGE, "1,1024,1"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "cell_center_x"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "cell_center_y"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "cell_center_z"));
	p_list->push_back(PropertyInfo(Variant::REAL, "cell_scale"));

	p_list->push_back(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));

	for (const Map<int, Area *>::Element *E = area_map.front(); E; E = E->next()) {

		String base = "areas/" + itos(E->key()) + "/";
		p_list->push_back(PropertyInfo(Variant::RECT3, base + "bounds", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
		p_list->push_back(PropertyInfo(Variant::STRING, base + "name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
		p_list->push_back(PropertyInfo(Variant::REAL, base + "disable_distance", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
		p_list->push_back(PropertyInfo(Variant::COLOR, base + "disable_color", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
		p_list->push_back(PropertyInfo(Variant::BOOL, base + "exterior_portal", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	}
}

void GridMap::set_theme(const Ref<MeshLibrary> &p_theme) {

	if (!theme.is_null())
		theme->unregister_owner(this);
	theme = p_theme;
	if (!theme.is_null())
		theme->register_owner(this);

	_recreate_octant_data();
	_change_notify("theme");
}

Ref<MeshLibrary> GridMap::get_theme() const {

	return theme;
}

void GridMap::set_cell_size(float p_size) {

	cell_size = p_size;
	_recreate_octant_data();
}
float GridMap::get_cell_size() const {

	return cell_size;
}

void GridMap::set_octant_size(int p_size) {

	octant_size = p_size;
	_recreate_octant_data();
}
int GridMap::get_octant_size() const {

	return octant_size;
}

void GridMap::set_center_x(bool p_enable) {

	center_x = p_enable;
	_recreate_octant_data();
}

bool GridMap::get_center_x() const {
	return center_x;
}

void GridMap::set_center_y(bool p_enable) {

	center_y = p_enable;
	_recreate_octant_data();
}

bool GridMap::get_center_y() const {
	return center_y;
}

void GridMap::set_center_z(bool p_enable) {

	center_z = p_enable;
	_recreate_octant_data();
}

bool GridMap::get_center_z() const {
	return center_z;
}

int GridMap::_find_area(const IndexKey &p_pos) const {

	for (const Map<int, Area *>::Element *E = area_map.front(); E; E = E->next()) {
		//this should somehow be faster...
		const Area &a = *E->get();
		if (p_pos.x >= a.from.x && p_pos.x < a.to.x &&
				p_pos.y >= a.from.y && p_pos.y < a.to.y &&
				p_pos.z >= a.from.z && p_pos.z < a.to.z) {
			return E->key();
		}
	}

	return 0;
}
void GridMap::set_cell_item(int p_x, int p_y, int p_z, int p_item, int p_rot) {

	ERR_FAIL_INDEX(ABS(p_x), 1 << 20);
	ERR_FAIL_INDEX(ABS(p_y), 1 << 20);
	ERR_FAIL_INDEX(ABS(p_z), 1 << 20);

	IndexKey key;
	key.x = p_x;
	key.y = p_y;
	key.z = p_z;

	OctantKey ok;
	ok.x = p_x / octant_size;
	ok.y = p_y / octant_size;
	ok.z = p_z / octant_size;
	ok.area = _find_area(key);

	if (cell_map.has(key)) {

		int prev_item = cell_map[key].item;

		OctantKey octantkey = ok;

		ERR_FAIL_COND(!octant_map.has(octantkey));
		Octant &g = *octant_map[octantkey];
		ERR_FAIL_COND(!g.items.has(prev_item));
		ERR_FAIL_COND(!g.items[prev_item].cells.has(key));

		g.items[prev_item].cells.erase(key);
		if (g.items[prev_item].cells.size() == 0) {
			VS::get_singleton()->free(g.items[prev_item].multimesh_instance);
			g.items.erase(prev_item);
		}
		if (g.items.empty()) {

			PhysicsServer::get_singleton()->free(g.static_body);
			if (g.collision_debug.is_valid()) {
				PhysicsServer::get_singleton()->free(g.collision_debug);
				PhysicsServer::get_singleton()->free(g.collision_debug_instance);
			}

			memdelete(&g);
			octant_map.erase(octantkey);
		} else {

			g.dirty = true;
		}
		cell_map.erase(key);

		_queue_dirty_map();
	}

	if (p_item < 0)
		return;

	OctantKey octantkey = ok;

	//add later
	if (!octant_map.has(octantkey)) {

		Octant *g = memnew(Octant);
		g->dirty = true;
		g->static_body = PhysicsServer::get_singleton()->body_create(PhysicsServer::BODY_MODE_STATIC);
		PhysicsServer::get_singleton()->body_attach_object_instance_ID(g->static_body, get_instance_ID());
		if (is_inside_world())
			PhysicsServer::get_singleton()->body_set_space(g->static_body, get_world()->get_space());

		SceneTree *st = SceneTree::get_singleton();

		if (st && st->is_debugging_collisions_hint()) {

			g->collision_debug = VisualServer::get_singleton()->mesh_create();
			g->collision_debug_instance = VisualServer::get_singleton()->instance_create();
			VisualServer::get_singleton()->instance_set_base(g->collision_debug_instance, g->collision_debug);
			if (is_inside_world()) {
				VisualServer::get_singleton()->instance_set_scenario(g->collision_debug_instance, get_world()->get_scenario());
				VisualServer::get_singleton()->instance_set_transform(g->collision_debug_instance, get_global_transform());
			}
		}

		octant_map[octantkey] = g;
	}

	Octant &g = *octant_map[octantkey];
	if (!g.items.has(p_item)) {

		Octant::ItemInstances ii;
		if (theme.is_valid() && theme->has_item(p_item)) {
			ii.mesh = theme->get_item_mesh(p_item);
			ii.shape = theme->get_item_shape(p_item);
			ii.navmesh = theme->get_item_navmesh(p_item);
		}
		ii.multimesh = Ref<MultiMesh>(memnew(MultiMesh));
		ii.multimesh->set_color_format(MultiMesh::COLOR_NONE);
		ii.multimesh->set_transform_format(MultiMesh::TRANSFORM_3D);
		ii.multimesh->set_mesh(ii.mesh);
		ii.multimesh_instance = VS::get_singleton()->instance_create();
		VS::get_singleton()->instance_set_base(ii.multimesh_instance, ii.multimesh->get_rid());

		g.items[p_item] = ii;
	}

	Octant::ItemInstances &ii = g.items[p_item];
	ii.cells.insert(key);
	g.dirty = true;

	_queue_dirty_map();

	cell_map[key] = Cell();
	Cell &c = cell_map[key];
	c.item = p_item;
	c.rot = p_rot;
}

int GridMap::get_cell_item(int p_x, int p_y, int p_z) const {

	ERR_FAIL_INDEX_V(ABS(p_x), 1 << 20, INVALID_CELL_ITEM);
	ERR_FAIL_INDEX_V(ABS(p_y), 1 << 20, INVALID_CELL_ITEM);
	ERR_FAIL_INDEX_V(ABS(p_z), 1 << 20, INVALID_CELL_ITEM);

	IndexKey key;
	key.x = p_x;
	key.y = p_y;
	key.z = p_z;

	if (!cell_map.has(key))
		return INVALID_CELL_ITEM;
	return cell_map[key].item;
}

int GridMap::get_cell_item_orientation(int p_x, int p_y, int p_z) const {

	ERR_FAIL_INDEX_V(ABS(p_x), 1 << 20, -1);
	ERR_FAIL_INDEX_V(ABS(p_y), 1 << 20, -1);
	ERR_FAIL_INDEX_V(ABS(p_z), 1 << 20, -1);

	IndexKey key;
	key.x = p_x;
	key.y = p_y;
	key.z = p_z;

	if (!cell_map.has(key))
		return -1;
	return cell_map[key].rot;
}

void GridMap::_octant_enter_tree(const OctantKey &p_key) {
	ERR_FAIL_COND(!octant_map.has(p_key));
	if (navigation) {
		Octant &g = *octant_map[p_key];

		Vector3 ofs(cell_size * 0.5 * int(center_x), cell_size * 0.5 * int(center_y), cell_size * 0.5 * int(center_z));
		_octant_clear_navmesh(p_key);

		for (Map<int, Octant::ItemInstances>::Element *E = g.items.front(); E; E = E->next()) {
			Octant::ItemInstances &ii = E->get();

			for (Set<IndexKey>::Element *F = ii.cells.front(); F; F = F->next()) {

				IndexKey ik = F->get();
				Map<IndexKey, Cell>::Element *C = cell_map.find(ik);
				ERR_CONTINUE(!C);

				Vector3 cellpos = Vector3(ik.x, ik.y, ik.z);

				Transform xform;

				if (clip && ((clip_above && cellpos[clip_axis] > clip_floor) || (!clip_above && cellpos[clip_axis] < clip_floor))) {

					xform.basis.set_zero();

				} else {

					xform.basis.set_orthogonal_index(C->get().rot);
				}

				xform.set_origin(cellpos * cell_size + ofs);
				xform.basis.scale(Vector3(cell_scale, cell_scale, cell_scale));
				// add the item's navmesh at given xform to GridMap's Navigation ancestor
				if (ii.navmesh.is_valid()) {
					int nm_id = navigation->navmesh_create(ii.navmesh, xform, this);
					Octant::NavMesh nm;
					nm.id = nm_id;
					nm.xform = xform;
					g.navmesh_ids[ik] = nm;
				}
			}
		}
	}
}

void GridMap::_octant_enter_world(const OctantKey &p_key) {

	ERR_FAIL_COND(!octant_map.has(p_key));
	Octant &g = *octant_map[p_key];
	PhysicsServer::get_singleton()->body_set_state(g.static_body, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());
	PhysicsServer::get_singleton()->body_set_space(g.static_body, get_world()->get_space());
	//print_line("BODYPOS: "+get_global_transform());

	if (g.collision_debug_instance.is_valid()) {
		VS::get_singleton()->instance_set_scenario(g.collision_debug_instance, get_world()->get_scenario());
		VS::get_singleton()->instance_set_transform(g.collision_debug_instance, get_global_transform());
		if (area_map.has(p_key.area)) {
			VS::get_singleton()->instance_set_room(g.collision_debug_instance, area_map[p_key.area]->instance);
		}
	}
	for (Map<int, Octant::ItemInstances>::Element *E = g.items.front(); E; E = E->next()) {

		VS::get_singleton()->instance_set_scenario(E->get().multimesh_instance, get_world()->get_scenario());
		VS::get_singleton()->instance_set_transform(E->get().multimesh_instance, get_global_transform());
		//print_line("INSTANCEPOS: "+get_global_transform());

		if (area_map.has(p_key.area)) {
			VS::get_singleton()->instance_set_room(E->get().multimesh_instance, area_map[p_key.area]->instance);
		}
	}
}

void GridMap::_octant_transform(const OctantKey &p_key) {

	ERR_FAIL_COND(!octant_map.has(p_key));
	Octant &g = *octant_map[p_key];
	PhysicsServer::get_singleton()->body_set_state(g.static_body, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());

	if (g.collision_debug_instance.is_valid()) {
		VS::get_singleton()->instance_set_transform(g.collision_debug_instance, get_global_transform());
	}

	for (Map<int, Octant::ItemInstances>::Element *E = g.items.front(); E; E = E->next()) {

		VS::get_singleton()->instance_set_transform(E->get().multimesh_instance, get_global_transform());
		//print_line("UPDATEPOS: "+get_global_transform());
	}
}

void GridMap::_octant_clear_navmesh(const OctantKey &p_key) {
	Octant &g = *octant_map[p_key];
	if (navigation) {
		for (Map<IndexKey, Octant::NavMesh>::Element *E = g.navmesh_ids.front(); E; E = E->next()) {
			Octant::NavMesh *nvm = &E->get();
			if (nvm && nvm->id) {
				navigation->navmesh_remove(E->get().id);
			}
		}
		g.navmesh_ids.clear();
	}
}

void GridMap::_octant_update(const OctantKey &p_key) {
	ERR_FAIL_COND(!octant_map.has(p_key));
	Octant &g = *octant_map[p_key];
	if (!g.dirty)
		return;

	Ref<Mesh> mesh;

	_octant_clear_navmesh(p_key);
	PhysicsServer::get_singleton()->body_clear_shapes(g.static_body);

	if (g.collision_debug.is_valid()) {

		VS::get_singleton()->mesh_clear(g.collision_debug);
	}

	PoolVector<Vector3> col_debug;

	/*
	 * foreach item in this octant,
	 * set item's multimesh's instance count to number of cells which have this item
	 * and set said multimesh bounding box to one containing all cells which have this item
	 */
	for (Map<int, Octant::ItemInstances>::Element *E = g.items.front(); E; E = E->next()) {

		Octant::ItemInstances &ii = E->get();

		ii.multimesh->set_instance_count(ii.cells.size());

		Rect3 aabb;
		Rect3 mesh_aabb = ii.mesh.is_null() ? Rect3() : ii.mesh->get_aabb();

		Vector3 ofs(cell_size * 0.5 * int(center_x), cell_size * 0.5 * int(center_y), cell_size * 0.5 * int(center_z));

		//print_line("OCTANT, CELLS: "+itos(ii.cells.size()));
		int idx = 0;
		// foreach cell containing this item type
		for (Set<IndexKey>::Element *F = ii.cells.front(); F; F = F->next()) {
			IndexKey ik = F->get();
			Map<IndexKey, Cell>::Element *C = cell_map.find(ik);
			ERR_CONTINUE(!C);

			Vector3 cellpos = Vector3(ik.x, ik.y, ik.z);

			Transform xform;

			if (clip && ((clip_above && cellpos[clip_axis] > clip_floor) || (!clip_above && cellpos[clip_axis] < clip_floor))) {

				xform.basis.set_zero();

			} else {

				xform.basis.set_orthogonal_index(C->get().rot);
			}

			xform.set_origin(cellpos * cell_size + ofs);
			xform.basis.scale(Vector3(cell_scale, cell_scale, cell_scale));

			ii.multimesh->set_instance_transform(idx, xform);
			//ii.multimesh->set_instance_transform(idx,Transform()	);
			//ii.multimesh->set_instance_color(idx,Color(1,1,1,1));
			//print_line("MMINST: "+xform);

			if (idx == 0) {

				aabb = xform.xform(mesh_aabb);
			} else {

				aabb.merge_with(xform.xform(mesh_aabb));
			}

			// add the item's shape at given xform to octant's static_body
			if (ii.shape.is_valid()) {
				// add the item's shape
				PhysicsServer::get_singleton()->body_add_shape(g.static_body, ii.shape->get_rid(), xform);
				if (g.collision_debug.is_valid()) {
					ii.shape->add_vertices_to_array(col_debug, xform);
				}

				//print_line("PHIS x: "+xform);
			}

			// add the item's navmesh at given xform to GridMap's Navigation ancestor
			if (navigation) {
				if (ii.navmesh.is_valid()) {
					int nm_id = navigation->navmesh_create(ii.navmesh, xform, this);
					Octant::NavMesh nm;
					nm.id = nm_id;
					nm.xform = xform;
					g.navmesh_ids[ik] = nm;
				}
			}

			idx++;
		}

		//ii.multimesh->set_aabb(aabb);
	}

	if (col_debug.size()) {

		Array arr;
		arr.resize(VS::ARRAY_MAX);
		arr[VS::ARRAY_VERTEX] = col_debug;

		VS::get_singleton()->mesh_add_surface_from_arrays(g.collision_debug, VS::PRIMITIVE_LINES, arr);
		SceneTree *st = SceneTree::get_singleton();
		if (st) {
			VS::get_singleton()->mesh_surface_set_material(g.collision_debug, 0, st->get_debug_collision_material()->get_rid());
		}
	}

	g.dirty = false;
}

void GridMap::_octant_exit_world(const OctantKey &p_key) {

	ERR_FAIL_COND(!octant_map.has(p_key));
	Octant &g = *octant_map[p_key];
	PhysicsServer::get_singleton()->body_set_state(g.static_body, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());
	PhysicsServer::get_singleton()->body_set_space(g.static_body, RID());

	if (g.collision_debug_instance.is_valid()) {

		VS::get_singleton()->instance_set_room(g.collision_debug_instance, RID());
		VS::get_singleton()->instance_set_scenario(g.collision_debug_instance, RID());
	}

	for (Map<int, Octant::ItemInstances>::Element *E = g.items.front(); E; E = E->next()) {

		VS::get_singleton()->instance_set_scenario(E->get().multimesh_instance, RID());
		//VS::get_singleton()->instance_set_transform(E->get().multimesh_instance,get_global_transform());
		VS::get_singleton()->instance_set_room(E->get().multimesh_instance, RID());
	}
}

void GridMap::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_WORLD: {

			_update_area_instances();

			for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
				//IndexKey ik;
				//ik.key = E->key().indexkey;
				_octant_enter_world(E->key());
				_octant_update(E->key());
			}

			awaiting_update = false;

			last_transform = get_global_transform();

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			Transform new_xform = get_global_transform();
			if (new_xform == last_transform)
				break;
			//update run
			for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
				_octant_transform(E->key());
			}

			last_transform = new_xform;

		} break;
		case NOTIFICATION_EXIT_WORLD: {

			for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
				_octant_exit_world(E->key());
			}

			//_queue_dirty_map(MAP_DIRTY_INSTANCES|MAP_DIRTY_TRANSFORMS);
			//_update_dirty_map_callback();
			//_update_area_instances();

		} break;
		case NOTIFICATION_ENTER_TREE: {

			Spatial *c = this;
			while (c) {
				navigation = c->cast_to<Navigation>();
				if (navigation) {
					break;
				}

				c = c->get_parent()->cast_to<Spatial>();
			}

			if (navigation) {
				for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
					if (navigation) {
						_octant_enter_tree(E->key());
					}
				}
			}

			_queue_dirty_map();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
				if (navigation) {
					_octant_clear_navmesh(E->key());
				}
			}

			navigation = NULL;

		} break;
	}
}

void GridMap::_queue_dirty_map() {

	if (awaiting_update)
		return;

	if (is_inside_world()) {

		MessageQueue::get_singleton()->push_call(this, "_update_dirty_map_callback");
		awaiting_update = true;
	}
}

void GridMap::_recreate_octant_data() {

	Map<IndexKey, Cell> cell_copy = cell_map;
	_clear_internal(true);
	for (Map<IndexKey, Cell>::Element *E = cell_copy.front(); E; E = E->next()) {

		set_cell_item(E->key().x, E->key().y, E->key().z, E->get().item, E->get().rot);
	}
}

void GridMap::_clear_internal(bool p_keep_areas) {

	for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
		if (is_inside_world())
			_octant_exit_world(E->key());

		for (Map<int, Octant::ItemInstances>::Element *F = E->get()->items.front(); F; F = F->next()) {

			VS::get_singleton()->free(F->get().multimesh_instance);
		}

		if (E->get()->collision_debug.is_valid())
			VS::get_singleton()->free(E->get()->collision_debug);
		if (E->get()->collision_debug_instance.is_valid())
			VS::get_singleton()->free(E->get()->collision_debug_instance);

		PhysicsServer::get_singleton()->free(E->get()->static_body);
		memdelete(E->get());
	}

	octant_map.clear();
	cell_map.clear();

	if (p_keep_areas)
		return;

	for (Map<int, Area *>::Element *E = area_map.front(); E; E = E->next()) {

		VS::get_singleton()->free(E->get()->base_portal);
		VS::get_singleton()->free(E->get()->instance);
		for (int i = 0; i < E->get()->portals.size(); i++) {
			VS::get_singleton()->free(E->get()->portals[i].instance);
		}

		memdelete(E->get());
	}
}

void GridMap::clear() {

	_clear_internal();
}

void GridMap::resource_changed(const RES &p_res) {

	_recreate_octant_data();
}

void GridMap::_update_dirty_map_callback() {

	if (!awaiting_update)
		return;

	for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
		_octant_update(E->key());
	}

	awaiting_update = false;
}

void GridMap::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_theme", "theme:MeshLibrary"), &GridMap::set_theme);
	ClassDB::bind_method(D_METHOD("get_theme:MeshLibrary"), &GridMap::get_theme);

	ClassDB::bind_method(D_METHOD("set_cell_size", "size"), &GridMap::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &GridMap::get_cell_size);

	ClassDB::bind_method(D_METHOD("set_octant_size", "size"), &GridMap::set_octant_size);
	ClassDB::bind_method(D_METHOD("get_octant_size"), &GridMap::get_octant_size);

	ClassDB::bind_method(D_METHOD("set_cell_item", "x", "y", "z", "item", "orientation"), &GridMap::set_cell_item, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_cell_item", "x", "y", "z"), &GridMap::get_cell_item);
	ClassDB::bind_method(D_METHOD("get_cell_item_orientation", "x", "y", "z"), &GridMap::get_cell_item_orientation);

	//ClassDB::bind_method(D_METHOD("_recreate_octants"),&GridMap::_recreate_octants);
	ClassDB::bind_method(D_METHOD("_update_dirty_map_callback"), &GridMap::_update_dirty_map_callback);
	ClassDB::bind_method(D_METHOD("resource_changed", "resource"), &GridMap::resource_changed);

	ClassDB::bind_method(D_METHOD("set_center_x", "enable"), &GridMap::set_center_x);
	ClassDB::bind_method(D_METHOD("get_center_x"), &GridMap::get_center_x);
	ClassDB::bind_method(D_METHOD("set_center_y", "enable"), &GridMap::set_center_y);
	ClassDB::bind_method(D_METHOD("get_center_y"), &GridMap::get_center_y);
	ClassDB::bind_method(D_METHOD("set_center_z", "enable"), &GridMap::set_center_z);
	ClassDB::bind_method(D_METHOD("get_center_z"), &GridMap::get_center_z);

	ClassDB::bind_method(D_METHOD("set_clip", "enabled", "clipabove", "floor", "axis"), &GridMap::set_clip, DEFVAL(true), DEFVAL(0), DEFVAL(Vector3::AXIS_X));

	ClassDB::bind_method(D_METHOD("create_area", "id", "area"), &GridMap::create_area);
	ClassDB::bind_method(D_METHOD("area_get_bounds", "area", "bounds"), &GridMap::area_get_bounds);
	ClassDB::bind_method(D_METHOD("area_set_exterior_portal", "area", "enable"), &GridMap::area_set_exterior_portal);
	ClassDB::bind_method(D_METHOD("area_set_name", "area", "name"), &GridMap::area_set_name);
	ClassDB::bind_method(D_METHOD("area_get_name", "area"), &GridMap::area_get_name);
	ClassDB::bind_method(D_METHOD("area_is_exterior_portal", "area"), &GridMap::area_is_exterior_portal);
	ClassDB::bind_method(D_METHOD("area_set_portal_disable_distance", "area", "distance"), &GridMap::area_set_portal_disable_distance);
	ClassDB::bind_method(D_METHOD("area_get_portal_disable_distance", "area"), &GridMap::area_get_portal_disable_distance);
	ClassDB::bind_method(D_METHOD("area_set_portal_disable_color", "area", "color"), &GridMap::area_set_portal_disable_color);
	ClassDB::bind_method(D_METHOD("area_get_portal_disable_color", "area"), &GridMap::area_get_portal_disable_color);
	ClassDB::bind_method(D_METHOD("erase_area", "area"), &GridMap::erase_area);
	ClassDB::bind_method(D_METHOD("get_unused_area_id", "area"), &GridMap::get_unused_area_id);

	ClassDB::bind_method(D_METHOD("clear"), &GridMap::clear);

	ClassDB::bind_method(D_METHOD("get_meshes"), &GridMap::get_meshes);

	BIND_CONSTANT(INVALID_CELL_ITEM);
}

void GridMap::set_clip(bool p_enabled, bool p_clip_above, int p_floor, Vector3::Axis p_axis) {

	if (!p_enabled && !clip)
		return;
	if (clip && p_enabled && clip_floor == p_floor && p_clip_above == clip_above && p_axis == clip_axis)
		return;

	clip = p_enabled;
	clip_floor = p_floor;
	clip_axis = p_axis;
	clip_above = p_clip_above;

	//make it all update
	for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {

		Octant *g = E->get();
		g->dirty = true;
	}
	awaiting_update = true;
	_update_dirty_map_callback();
}

void GridMap::_update_areas() {

	//clear the portals
	for (Map<int, Area *>::Element *E = area_map.front(); E; E = E->next()) {
		//this should somehow be faster...
		Area &a = *E->get();
		a.portals.clear();
		if (a.instance.is_valid()) {
			VisualServer::get_singleton()->free(a.instance);
			a.instance = RID();
		}
	}

	//test all areas against all areas and create portals - this sucks (slow :( )
	for (Map<int, Area *>::Element *E = area_map.front(); E; E = E->next()) {
		Area &a = *E->get();
		if (a.exterior_portal) //that's pretty much all it does... yes it is
			continue;
		Vector3 from_a(a.from.x, a.from.y, a.from.z);
		Vector3 to_a(a.to.x, a.to.y, a.to.z);

		for (Map<int, Area *>::Element *F = area_map.front(); F; F = F->next()) {

			Area &b = *F->get();
			Vector3 from_b(b.from.x, b.from.y, b.from.z);
			Vector3 to_b(b.to.x, b.to.y, b.to.z);

			// initially test intersection and discards
			int axis = -1;
			float sign = 0;
			bool valid = true;
			Vector3 axmin, axmax;

			for (int i = 0; i < 3; i++) {

				if (from_a[i] == to_b[i]) {

					if (axis != -1) {
						valid = false;
						break;
					}

					axis = i;
					sign = -1;
				} else if (from_b[i] == to_a[i]) {

					if (axis != -1) {
						valid = false;
						break;
					}
					axis = i;
					sign = +1;
				}

				if (from_a[i] > to_b[i] || to_a[i] < from_b[i]) {
					valid = false;
					break;
				} else {

					axmin[i] = (from_a[i] > from_b[i]) ? from_a[i] : from_b[i];
					axmax[i] = (to_a[i] < to_b[i]) ? to_a[i] : to_b[i];
				}
			}

			if (axis == -1 || !valid)
				continue;

			Transform xf;

			for (int i = 0; i < 3; i++) {

				int ax = (axis + i) % 3;
				Vector3 axis_vec;
				float scale = (i == 0) ? sign : ((axmax[ax] - axmin[ax]) * cell_size);
				axis_vec[ax] = scale;
				xf.basis.set_axis((2 + i) % 3, axis_vec);
				xf.origin[i] = axmin[i] * cell_size;
			}

			Area::Portal portal;
			portal.xform = xf;
			a.portals.push_back(portal);
		}
	}

	_update_area_instances();
}

void GridMap::_update_area_instances() {

	Transform base_xform;
	if (_in_tree)
		base_xform = get_global_transform();

	for (Map<int, Area *>::Element *E = area_map.front(); E; E = E->next()) {
		//this should somehow be faster...
		Area &a = *E->get();
		if (a.instance.is_valid() != _in_tree) {

			if (!_in_tree) {

				for (int i = 0; i < a.portals.size(); i++) {

					Area::Portal &p = a.portals[i];
					ERR_CONTINUE(!p.instance.is_valid());
					VisualServer::get_singleton()->free(p.instance);
					p.instance = RID();
				}

				VisualServer::get_singleton()->free(a.instance);
				a.instance = RID();

			} else {

				//a.instance = VisualServer::get_singleton()->instance_create2(base_room,get_world()->get_scenario());
				for (int i = 0; i < a.portals.size(); i++) {

					Area::Portal &p = a.portals[i];
					ERR_CONTINUE(p.instance.is_valid());
					p.instance = VisualServer::get_singleton()->instance_create2(a.base_portal, get_world()->get_scenario());
					VisualServer::get_singleton()->instance_set_room(p.instance, a.instance);
				}
			}
		}

		if (a.instance.is_valid()) {
			Transform xform;

			Vector3 from_a(a.from.x, a.from.y, a.from.z);
			Vector3 to_a(a.to.x, a.to.y, a.to.z);

			for (int i = 0; i < 3; i++) {
				xform.origin[i] = from_a[i] * cell_size;
				Vector3 s;
				s[i] = (to_a[i] - from_a[i]) * cell_size;
				xform.basis.set_axis(i, s);
			}

			VisualServer::get_singleton()->instance_set_transform(a.instance, base_xform * xform);

			for (int i = 0; i < a.portals.size(); i++) {

				Area::Portal &p = a.portals[i];
				ERR_CONTINUE(!p.instance.is_valid());

				VisualServer::get_singleton()->instance_set_transform(p.instance, base_xform * xform);
			}
		}
	}
}

Error GridMap::create_area(int p_id, const Rect3 &p_bounds) {

	ERR_FAIL_COND_V(area_map.has(p_id), ERR_ALREADY_EXISTS);
	ERR_EXPLAIN("ID 0 is taken as global area, start from 1");
	ERR_FAIL_COND_V(p_id == 0, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_bounds.has_no_area(), ERR_INVALID_PARAMETER);

	// FIRST VALIDATE AREA
	IndexKey from, to;
	from.x = p_bounds.pos.x;
	from.y = p_bounds.pos.y;
	from.z = p_bounds.pos.z;
	to.x = p_bounds.pos.x + p_bounds.size.x;
	to.y = p_bounds.pos.y + p_bounds.size.y;
	to.z = p_bounds.pos.z + p_bounds.size.z;

	for (Map<int, Area *>::Element *E = area_map.front(); E; E = E->next()) {
		//this should somehow be faster...
		Area &a = *E->get();

		//does it interset with anything else?

		if (from.x >= a.to.x ||
				to.x <= a.from.x ||
				from.y >= a.to.y ||
				to.y <= a.from.y ||
				from.z >= a.to.z ||
				to.z <= a.from.z) {

			// all good
		} else {

			return ERR_INVALID_PARAMETER;
		}
	}

	Area *area = memnew(Area);
	area->from = from;
	area->to = to;
	area->portal_disable_distance = 0;
	area->exterior_portal = false;
	area->name = "Area " + itos(p_id);
	area_map[p_id] = area;
	_recreate_octant_data();
	return OK;
}

Rect3 GridMap::area_get_bounds(int p_area) const {

	ERR_FAIL_COND_V(!area_map.has(p_area), Rect3());

	const Area *a = area_map[p_area];
	Rect3 aabb;
	aabb.pos = Vector3(a->from.x, a->from.y, a->from.z);
	aabb.size = Vector3(a->to.x, a->to.y, a->to.z) - aabb.pos;

	return aabb;
}

void GridMap::area_set_name(int p_area, const String &p_name) {

	ERR_FAIL_COND(!area_map.has(p_area));

	Area *a = area_map[p_area];
	a->name = p_name;
}

String GridMap::area_get_name(int p_area) const {

	ERR_FAIL_COND_V(!area_map.has(p_area), "");

	const Area *a = area_map[p_area];
	return a->name;
}

void GridMap::area_set_exterior_portal(int p_area, bool p_enable) {

	ERR_FAIL_COND(!area_map.has(p_area));

	Area *a = area_map[p_area];
	if (a->exterior_portal == p_enable)
		return;
	a->exterior_portal = p_enable;

	_recreate_octant_data();
}

bool GridMap::area_is_exterior_portal(int p_area) const {

	ERR_FAIL_COND_V(!area_map.has(p_area), false);

	const Area *a = area_map[p_area];
	return a->exterior_portal;
}

void GridMap::area_set_portal_disable_distance(int p_area, float p_distance) {

	ERR_FAIL_COND(!area_map.has(p_area));

	Area *a = area_map[p_area];
	a->portal_disable_distance = p_distance;
}

float GridMap::area_get_portal_disable_distance(int p_area) const {

	ERR_FAIL_COND_V(!area_map.has(p_area), 0);

	const Area *a = area_map[p_area];
	return a->portal_disable_distance;
}

void GridMap::area_set_portal_disable_color(int p_area, Color p_color) {

	ERR_FAIL_COND(!area_map.has(p_area));

	Area *a = area_map[p_area];
	a->portal_disable_color = p_color;
}

Color GridMap::area_get_portal_disable_color(int p_area) const {

	ERR_FAIL_COND_V(!area_map.has(p_area), Color());

	const Area *a = area_map[p_area];
	return a->portal_disable_color;
}

void GridMap::get_area_list(List<int> *p_areas) const {

	for (const Map<int, Area *>::Element *E = area_map.front(); E; E = E->next()) {

		p_areas->push_back(E->key());
	}
}

GridMap::Area::Portal::~Portal() {

	if (instance.is_valid())
		VisualServer::get_singleton()->free(instance);
}

GridMap::Area::Area() {

	base_portal = VisualServer::get_singleton()->portal_create();
	Vector<Point2> points;
	points.push_back(Point2(0, 1));
	points.push_back(Point2(1, 1));
	points.push_back(Point2(1, 0));
	points.push_back(Point2(0, 0));
	VisualServer::get_singleton()->portal_set_shape(base_portal, points);
}

GridMap::Area::~Area() {

	if (instance.is_valid())
		VisualServer::get_singleton()->free(instance);
	VisualServer::get_singleton()->free(base_portal);
}

void GridMap::erase_area(int p_area) {

	ERR_FAIL_COND(!area_map.has(p_area));

	Area *a = area_map[p_area];
	memdelete(a);
	area_map.erase(p_area);
	_recreate_octant_data();
}

int GridMap::get_unused_area_id() const {

	if (area_map.empty())
		return 1;
	else
		return area_map.back()->key() + 1;
}

void GridMap::set_cell_scale(float p_scale) {

	cell_scale = p_scale;
	_queue_dirty_map();
}

float GridMap::get_cell_scale() const {

	return cell_scale;
}

Array GridMap::get_meshes() {

	if (theme.is_null())
		return Array();

	Vector3 ofs(cell_size * 0.5 * int(center_x), cell_size * 0.5 * int(center_y), cell_size * 0.5 * int(center_z));
	Array meshes;

	for (Map<IndexKey, Cell>::Element *E = cell_map.front(); E; E = E->next()) {

		int id = E->get().item;
		if (!theme->has_item(id))
			continue;
		Ref<Mesh> mesh = theme->get_item_mesh(id);
		if (mesh.is_null())
			continue;

		IndexKey ik = E->key();

		Vector3 cellpos = Vector3(ik.x, ik.y, ik.z);

		Transform xform;

		xform.basis.set_orthogonal_index(E->get().rot);

		xform.set_origin(cellpos * cell_size + ofs);
		xform.basis.scale(Vector3(cell_scale, cell_scale, cell_scale));

		meshes.push_back(xform);
		meshes.push_back(mesh);
	}

	return meshes;
}

GridMap::GridMap() {

	cell_size = 2;
	octant_size = 4;
	awaiting_update = false;
	_in_tree = false;
	center_x = true;
	center_y = true;
	center_z = true;

	clip = false;
	clip_floor = 0;
	clip_axis = Vector3::AXIS_Z;
	clip_above = true;
	cell_scale = 1.0;

	navigation = NULL;
	set_notify_transform(true);
}

GridMap::~GridMap() {

	if (!theme.is_null())
		theme->unregister_owner(this);

	clear();
}
