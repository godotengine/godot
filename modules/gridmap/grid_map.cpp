/*************************************************************************/
/*  grid_map.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
		if (p_value.get_type() == Variant::INT || p_value.get_type() == Variant::REAL) {
			//compatibility
			float cs = p_value;
			set_cell_size(Vector3(cs, cs, cs));
		} else {
			set_cell_size(p_value);
		}
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
	} else if (name == "baked_meshes") {

		clear_baked_meshes();

		Array meshes = p_value;

		for (int i = 0; i < meshes.size(); i++) {
			BakedMesh bm;
			bm.mesh = meshes[i];
			ERR_CONTINUE(!bm.mesh.is_valid());
			bm.instance = VS::get_singleton()->instance_create();
			VS::get_singleton()->get_singleton()->instance_set_base(bm.instance, bm.mesh->get_rid());
			VS::get_singleton()->instance_attach_object_instance_id(bm.instance, get_instance_id());
			if (is_inside_tree()) {
				VS::get_singleton()->instance_set_scenario(bm.instance, get_world()->get_scenario());
				VS::get_singleton()->instance_set_transform(bm.instance, get_global_transform());
			}
			baked_meshes.push_back(bm);
		}

		_recreate_octant_data();

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
	} else if (name == "baked_meshes") {

		Array ret;
		ret.resize(baked_meshes.size());
		for (int i = 0; i < baked_meshes.size(); i++) {
			ret.push_back(baked_meshes[i].mesh);
		}
		r_ret = ret;

	} else
		return false;

	return true;
}

void GridMap::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back(PropertyInfo(Variant::OBJECT, "theme", PROPERTY_HINT_RESOURCE_TYPE, "MeshLibrary"));
	p_list->push_back(PropertyInfo(Variant::NIL, "Cell", PROPERTY_HINT_NONE, "cell_", PROPERTY_USAGE_GROUP));
	p_list->push_back(PropertyInfo(Variant::VECTOR3, "cell_size"));
	p_list->push_back(PropertyInfo(Variant::INT, "cell_octant_size", PROPERTY_HINT_RANGE, "1,1024,1"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "cell_center_x"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "cell_center_y"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "cell_center_z"));
	p_list->push_back(PropertyInfo(Variant::REAL, "cell_scale"));
	if (baked_meshes.size()) {
		p_list->push_back(PropertyInfo(Variant::ARRAY, "baked_meshes", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	}

	p_list->push_back(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
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

void GridMap::set_cell_size(const Vector3 &p_size) {

	ERR_FAIL_COND(p_size.x < 0.001 || p_size.y < 0.001 || p_size.z < 0.001);
	cell_size = p_size;
	_recreate_octant_data();
}
Vector3 GridMap::get_cell_size() const {

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

void GridMap::set_cell_item(int p_x, int p_y, int p_z, int p_item, int p_rot) {

	if (baked_meshes.size() && !recreating_octants) {
		//if you set a cell item, baked meshes go good bye
		clear_baked_meshes();
		_recreate_octant_data();
	}

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

	if (p_item < 0) {
		//erase
		if (cell_map.has(key)) {
			OctantKey octantkey = ok;

			ERR_FAIL_COND(!octant_map.has(octantkey));
			Octant &g = *octant_map[octantkey];
			g.cells.erase(key);
			g.dirty = true;
			cell_map.erase(key);
			_queue_octants_dirty();
		}
		return;
	}

	OctantKey octantkey = ok;

	if (!octant_map.has(octantkey)) {
		//create octant because it does not exist
		Octant *g = memnew(Octant);
		g->dirty = true;
		g->static_body = PhysicsServer::get_singleton()->body_create(PhysicsServer::BODY_MODE_STATIC);
		PhysicsServer::get_singleton()->body_attach_object_instance_id(g->static_body, get_instance_id());
		SceneTree *st = SceneTree::get_singleton();

		if (st && st->is_debugging_collisions_hint()) {

			g->collision_debug = VisualServer::get_singleton()->mesh_create();
			g->collision_debug_instance = VisualServer::get_singleton()->instance_create();
			VisualServer::get_singleton()->instance_set_base(g->collision_debug_instance, g->collision_debug);
		}

		octant_map[octantkey] = g;

		if (is_inside_world()) {
			_octant_enter_world(octantkey);
			_octant_transform(octantkey);
		}
	}

	Octant &g = *octant_map[octantkey];
	g.cells.insert(key);
	g.dirty = true;
	_queue_octants_dirty();

	Cell c;
	c.item = p_item;
	c.rot = p_rot;

	cell_map[key] = c;
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

Vector3 GridMap::world_to_map(const Vector3 &p_world_pos) const {
	Vector3 map_pos = p_world_pos / cell_size;
	map_pos.x = floor(map_pos.x);
	map_pos.y = floor(map_pos.y);
	map_pos.z = floor(map_pos.z);
	return map_pos;
}

Vector3 GridMap::map_to_world(int p_x, int p_y, int p_z) const {
	Vector3 offset = _get_offset();
	Vector3 world_pos(
			p_x * cell_size.x + offset.x,
			p_y * cell_size.y + offset.y,
			p_z * cell_size.z + offset.z);
	return world_pos;
}

void GridMap::_octant_transform(const OctantKey &p_key) {

	ERR_FAIL_COND(!octant_map.has(p_key));
	Octant &g = *octant_map[p_key];
	PhysicsServer::get_singleton()->body_set_state(g.static_body, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());

	if (g.collision_debug_instance.is_valid()) {
		VS::get_singleton()->instance_set_transform(g.collision_debug_instance, get_global_transform());
	}

	for (int i = 0; i < g.multimesh_instances.size(); i++) {
		VS::get_singleton()->instance_set_transform(g.multimesh_instances[i].instance, get_global_transform());
	}
}

bool GridMap::_octant_update(const OctantKey &p_key) {
	ERR_FAIL_COND_V(!octant_map.has(p_key), false);
	Octant &g = *octant_map[p_key];
	if (!g.dirty)
		return false;

	//erase body shapes
	PhysicsServer::get_singleton()->body_clear_shapes(g.static_body);

	//erase body shapes debug
	if (g.collision_debug.is_valid()) {

		VS::get_singleton()->mesh_clear(g.collision_debug);
	}

	//erase navigation
	if (navigation) {
		for (Map<IndexKey, Octant::NavMesh>::Element *E = g.navmesh_ids.front(); E; E = E->next()) {
			navigation->navmesh_remove(E->get().id);
		}
		g.navmesh_ids.clear();
	}

	//erase multimeshes

	for (int i = 0; i < g.multimesh_instances.size(); i++) {

		VS::get_singleton()->free(g.multimesh_instances[i].instance);
		VS::get_singleton()->free(g.multimesh_instances[i].multimesh);
	}
	g.multimesh_instances.clear();

	if (g.cells.size() == 0) {
		//octant no longer needed
		_octant_clean_up(p_key);
		return true;
	}

	PoolVector<Vector3> col_debug;

	/*
	 * foreach item in this octant,
	 * set item's multimesh's instance count to number of cells which have this item
	 * and set said multimesh bounding box to one containing all cells which have this item
	 */

	Map<int, List<Pair<Transform, IndexKey> > > multimesh_items;

	for (Set<IndexKey>::Element *E = g.cells.front(); E; E = E->next()) {

		ERR_CONTINUE(!cell_map.has(E->get()));
		const Cell &c = cell_map[E->get()];

		if (!theme.is_valid() || !theme->has_item(c.item))
			continue;

		//print_line("OCTANT, CELLS: "+itos(ii.cells.size()));

		Vector3 cellpos = Vector3(E->get().x, E->get().y, E->get().z);
		Vector3 ofs = _get_offset();

		Transform xform;

		if (clip && ((clip_above && cellpos[clip_axis] > clip_floor) || (!clip_above && cellpos[clip_axis] < clip_floor))) {

		} else {
		}

		xform.basis.set_orthogonal_index(c.rot);
		xform.set_origin(cellpos * cell_size + ofs);
		xform.basis.scale(Vector3(cell_scale, cell_scale, cell_scale));
		if (baked_meshes.size() == 0) {
			if (theme->get_item_mesh(c.item).is_valid()) {
				if (!multimesh_items.has(c.item)) {
					multimesh_items[c.item] = List<Pair<Transform, IndexKey> >();
				}

				Pair<Transform, IndexKey> p;
				p.first = xform;
				p.second = E->get();
				multimesh_items[c.item].push_back(p);
			}
		}

		Vector<MeshLibrary::ShapeData> shapes = theme->get_item_shapes(c.item);
		// add the item's shape at given xform to octant's static_body
		for (int i = 0; i < shapes.size(); i++) {
			// add the item's shape
			if (!shapes[i].shape.is_valid())
				continue;
			PhysicsServer::get_singleton()->body_add_shape(g.static_body, shapes[i].shape->get_rid(), xform * shapes[i].local_transform);
			if (g.collision_debug.is_valid()) {
				shapes[i].shape->add_vertices_to_array(col_debug, xform * shapes[i].local_transform);
			}

			//print_line("PHIS x: "+xform);
		}

		// add the item's navmesh at given xform to GridMap's Navigation ancestor
		Ref<NavigationMesh> navmesh = theme->get_item_navmesh(c.item);
		if (navmesh.is_valid()) {
			Octant::NavMesh nm;
			nm.xform = xform;

			if (navigation) {
				nm.id = navigation->navmesh_add(navmesh, xform, this);
			} else {
				nm.id = -1;
			}
			g.navmesh_ids[E->get()] = nm;
		}
	}

	//update multimeshes, only if not baked
	if (baked_meshes.size() == 0) {

		for (Map<int, List<Pair<Transform, IndexKey> > >::Element *E = multimesh_items.front(); E; E = E->next()) {
			Octant::MultimeshInstance mmi;

			RID mm = VS::get_singleton()->multimesh_create();
			VS::get_singleton()->multimesh_allocate(mm, E->get().size(), VS::MULTIMESH_TRANSFORM_3D, VS::MULTIMESH_COLOR_NONE);
			VS::get_singleton()->multimesh_set_mesh(mm, theme->get_item_mesh(E->key())->get_rid());

			int idx = 0;
			for (List<Pair<Transform, IndexKey> >::Element *F = E->get().front(); F; F = F->next()) {
				VS::get_singleton()->multimesh_instance_set_transform(mm, idx, F->get().first);
#ifdef TOOLS_ENABLED

				Octant::MultimeshInstance::Item it;
				it.index = idx;
				it.transform = F->get().first;
				it.key = F->get().second;
				mmi.items.push_back(it);
#endif

				idx++;
			}

			RID instance = VS::get_singleton()->instance_create();
			VS::get_singleton()->instance_set_base(instance, mm);

			if (is_inside_tree()) {
				VS::get_singleton()->instance_set_scenario(instance, get_world()->get_scenario());
				VS::get_singleton()->instance_set_transform(instance, get_global_transform());
			}

			mmi.multimesh = mm;
			mmi.instance = instance;

			g.multimesh_instances.push_back(mmi);
		}
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

	return false;
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
	}

	for (int i = 0; i < g.multimesh_instances.size(); i++) {
		VS::get_singleton()->instance_set_scenario(g.multimesh_instances[i].instance, get_world()->get_scenario());
		VS::get_singleton()->instance_set_transform(g.multimesh_instances[i].instance, get_global_transform());
	}

	if (navigation && theme.is_valid()) {
		for (Map<IndexKey, Octant::NavMesh>::Element *F = g.navmesh_ids.front(); F; F = F->next()) {

			if (cell_map.has(F->key()) && F->get().id < 0) {
				Ref<NavigationMesh> nm = theme->get_item_navmesh(cell_map[F->key()].item);
				if (nm.is_valid()) {
					F->get().id = navigation->navmesh_add(nm, F->get().xform, this);
				}
			}
		}
	}
}

void GridMap::_octant_exit_world(const OctantKey &p_key) {

	ERR_FAIL_COND(!octant_map.has(p_key));
	Octant &g = *octant_map[p_key];
	PhysicsServer::get_singleton()->body_set_state(g.static_body, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());
	PhysicsServer::get_singleton()->body_set_space(g.static_body, RID());

	if (g.collision_debug_instance.is_valid()) {

		VS::get_singleton()->instance_set_scenario(g.collision_debug_instance, RID());
	}

	for (int i = 0; i < g.multimesh_instances.size(); i++) {
		VS::get_singleton()->instance_set_scenario(g.multimesh_instances[i].instance, RID());
	}

	if (navigation) {
		for (Map<IndexKey, Octant::NavMesh>::Element *F = g.navmesh_ids.front(); F; F = F->next()) {

			if (F->get().id >= 0) {
				navigation->navmesh_remove(F->get().id);
				F->get().id = -1;
			}
		}
	}
}

void GridMap::_octant_clean_up(const OctantKey &p_key) {

	ERR_FAIL_COND(!octant_map.has(p_key));
	Octant &g = *octant_map[p_key];

	if (g.collision_debug.is_valid())
		VS::get_singleton()->free(g.collision_debug);
	if (g.collision_debug_instance.is_valid())
		VS::get_singleton()->free(g.collision_debug_instance);

	PhysicsServer::get_singleton()->free(g.static_body);

	//erase navigation
	if (navigation) {
		for (Map<IndexKey, Octant::NavMesh>::Element *E = g.navmesh_ids.front(); E; E = E->next()) {
			navigation->navmesh_remove(E->get().id);
		}
		g.navmesh_ids.clear();
	}

	//erase multimeshes

	for (int i = 0; i < g.multimesh_instances.size(); i++) {

		VS::get_singleton()->free(g.multimesh_instances[i].instance);
		VS::get_singleton()->free(g.multimesh_instances[i].multimesh);
	}
	g.multimesh_instances.clear();
}

void GridMap::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_WORLD: {

			Spatial *c = this;
			while (c) {
				navigation = Object::cast_to<Navigation>(c);
				if (navigation) {
					break;
				}

				c = Object::cast_to<Spatial>(c->get_parent());
			}

			last_transform = get_global_transform();

			for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
				_octant_enter_world(E->key());
			}

			for (int i = 0; i < baked_meshes.size(); i++) {
				VS::get_singleton()->instance_set_scenario(baked_meshes[i].instance, get_world()->get_scenario());
				VS::get_singleton()->instance_set_transform(baked_meshes[i].instance, get_global_transform());
			}

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

			for (int i = 0; i < baked_meshes.size(); i++) {
				VS::get_singleton()->instance_set_transform(baked_meshes[i].instance, get_global_transform());
			}

		} break;
		case NOTIFICATION_EXIT_WORLD: {

			for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
				_octant_exit_world(E->key());
			}

			navigation = NULL;

			//_queue_octants_dirty(MAP_DIRTY_INSTANCES|MAP_DIRTY_TRANSFORMS);
			//_update_octants_callback();
			//_update_area_instances();
			for (int i = 0; i < baked_meshes.size(); i++) {
				VS::get_singleton()->instance_set_scenario(baked_meshes[i].instance, RID());
			}

		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_visibility();
		} break;
	}
}

void GridMap::_update_visibility() {
	if (!is_inside_tree())
		return;

	_change_notify("visible");

	for (Map<OctantKey, Octant *>::Element *e = octant_map.front(); e; e = e->next()) {
		Octant *octant = e->value();
		for (int i = 0; i < octant->multimesh_instances.size(); i++) {
			Octant::MultimeshInstance &mi = octant->multimesh_instances[i];
			VS::get_singleton()->instance_set_visible(mi.instance, is_visible());
		}
	}
}

void GridMap::_queue_octants_dirty() {

	if (awaiting_update)
		return;

	MessageQueue::get_singleton()->push_call(this, "_update_octants_callback");
	awaiting_update = true;
}

void GridMap::_recreate_octant_data() {

	recreating_octants = true;
	Map<IndexKey, Cell> cell_copy = cell_map;
	_clear_internal();
	for (Map<IndexKey, Cell>::Element *E = cell_copy.front(); E; E = E->next()) {

		set_cell_item(E->key().x, E->key().y, E->key().z, E->get().item, E->get().rot);
	}
	recreating_octants = false;
}

void GridMap::_clear_internal() {

	for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
		if (is_inside_world())
			_octant_exit_world(E->key());

		_octant_clean_up(E->key());
		memdelete(E->get());
	}

	octant_map.clear();
	cell_map.clear();
}

void GridMap::clear() {

	_clear_internal();
	clear_baked_meshes();
}

void GridMap::resource_changed(const RES &p_res) {

	_recreate_octant_data();
}

void GridMap::_update_octants_callback() {

	if (!awaiting_update)
		return;

	List<OctantKey> to_delete;
	for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {

		if (_octant_update(E->key())) {
			to_delete.push_back(E->key());
		}
	}

	while (to_delete.front()) {
		octant_map.erase(to_delete.front()->get());
		to_delete.pop_back();
	}

	_update_visibility();
	awaiting_update = false;
}

void GridMap::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_theme", "theme"), &GridMap::set_theme);
	ClassDB::bind_method(D_METHOD("get_theme"), &GridMap::get_theme);

	ClassDB::bind_method(D_METHOD("set_cell_size", "size"), &GridMap::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &GridMap::get_cell_size);

	ClassDB::bind_method(D_METHOD("set_octant_size", "size"), &GridMap::set_octant_size);
	ClassDB::bind_method(D_METHOD("get_octant_size"), &GridMap::get_octant_size);

	ClassDB::bind_method(D_METHOD("set_cell_item", "x", "y", "z", "item", "orientation"), &GridMap::set_cell_item, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_cell_item", "x", "y", "z"), &GridMap::get_cell_item);
	ClassDB::bind_method(D_METHOD("get_cell_item_orientation", "x", "y", "z"), &GridMap::get_cell_item_orientation);

	ClassDB::bind_method(D_METHOD("world_to_map", "pos"), &GridMap::world_to_map);
	ClassDB::bind_method(D_METHOD("map_to_world", "x", "y", "z"), &GridMap::map_to_world);

	//ClassDB::bind_method(D_METHOD("_recreate_octants"),&GridMap::_recreate_octants);
	ClassDB::bind_method(D_METHOD("_update_octants_callback"), &GridMap::_update_octants_callback);
	ClassDB::bind_method(D_METHOD("resource_changed", "resource"), &GridMap::resource_changed);

	ClassDB::bind_method(D_METHOD("set_center_x", "enable"), &GridMap::set_center_x);
	ClassDB::bind_method(D_METHOD("get_center_x"), &GridMap::get_center_x);
	ClassDB::bind_method(D_METHOD("set_center_y", "enable"), &GridMap::set_center_y);
	ClassDB::bind_method(D_METHOD("get_center_y"), &GridMap::get_center_y);
	ClassDB::bind_method(D_METHOD("set_center_z", "enable"), &GridMap::set_center_z);
	ClassDB::bind_method(D_METHOD("get_center_z"), &GridMap::get_center_z);

	ClassDB::bind_method(D_METHOD("set_clip", "enabled", "clipabove", "floor", "axis"), &GridMap::set_clip, DEFVAL(true), DEFVAL(0), DEFVAL(Vector3::AXIS_X));

	ClassDB::bind_method(D_METHOD("clear"), &GridMap::clear);

	ClassDB::bind_method(D_METHOD("get_used_cells"), &GridMap::get_used_cells);

	ClassDB::bind_method(D_METHOD("get_meshes"), &GridMap::get_meshes);
	ClassDB::bind_method(D_METHOD("get_bake_meshes"), &GridMap::get_bake_meshes);
	ClassDB::bind_method(D_METHOD("get_bake_mesh_instance", "idx"), &GridMap::get_bake_mesh_instance);

	ClassDB::bind_method(D_METHOD("clear_baked_meshes"), &GridMap::clear_baked_meshes);
	ClassDB::bind_method(D_METHOD("make_baked_meshes", "gen_lightmap_uv", "lightmap_uv_texel_size"), &GridMap::make_baked_meshes, DEFVAL(false), DEFVAL(0.1));

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
	_update_octants_callback();
}

void GridMap::set_cell_scale(float p_scale) {

	cell_scale = p_scale;
	_recreate_octant_data();
}

float GridMap::get_cell_scale() const {

	return cell_scale;
}

Array GridMap::get_used_cells() const {

	Array a;
	a.resize(cell_map.size());
	int i = 0;
	for (Map<IndexKey, Cell>::Element *E = cell_map.front(); E; E = E->next()) {
		Vector3 p(E->key().x, E->key().y, E->key().z);
		a[i++] = p;
	}

	return a;
}

Array GridMap::get_meshes() {

	if (theme.is_null())
		return Array();

	Vector3 ofs = _get_offset();
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

Vector3 GridMap::_get_offset() const {
	return Vector3(
			cell_size.x * 0.5 * int(center_x),
			cell_size.y * 0.5 * int(center_y),
			cell_size.z * 0.5 * int(center_z));
}

void GridMap::clear_baked_meshes() {

	for (int i = 0; i < baked_meshes.size(); i++) {
		VS::get_singleton()->free(baked_meshes[i].instance);
	}
	baked_meshes.clear();

	_recreate_octant_data();
}

void GridMap::make_baked_meshes(bool p_gen_lightmap_uv, float p_lightmap_uv_texel_size) {

	if (!theme.is_valid())
		return;

	//generate
	Map<OctantKey, Map<Ref<Material>, Ref<SurfaceTool> > > surface_map;

	for (Map<IndexKey, Cell>::Element *E = cell_map.front(); E; E = E->next()) {

		IndexKey key = E->key();

		int item = E->get().item;
		if (!theme->has_item(item))
			continue;

		Ref<Mesh> mesh = theme->get_item_mesh(item);
		if (!mesh.is_valid())
			continue;

		Vector3 cellpos = Vector3(key.x, key.y, key.z);
		Vector3 ofs = _get_offset();

		Transform xform;

		xform.basis.set_orthogonal_index(E->get().rot);
		xform.set_origin(cellpos * cell_size + ofs);
		xform.basis.scale(Vector3(cell_scale, cell_scale, cell_scale));

		OctantKey ok;
		ok.x = key.x / octant_size;
		ok.y = key.y / octant_size;
		ok.z = key.z / octant_size;

		if (!surface_map.has(ok)) {
			surface_map[ok] = Map<Ref<Material>, Ref<SurfaceTool> >();
		}

		Map<Ref<Material>, Ref<SurfaceTool> > &mat_map = surface_map[ok];

		for (int i = 0; i < mesh->get_surface_count(); i++) {

			if (mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES)
				continue;

			Ref<Material> surf_mat = mesh->surface_get_material(i);
			if (!mat_map.has(surf_mat)) {
				Ref<SurfaceTool> st;
				st.instance();
				st->begin(Mesh::PRIMITIVE_TRIANGLES);
				st->set_material(surf_mat);
				mat_map[surf_mat] = st;
			}

			mat_map[surf_mat]->append_from(mesh, i, xform);
		}
	}

	int ofs = 0;

	for (Map<OctantKey, Map<Ref<Material>, Ref<SurfaceTool> > >::Element *E = surface_map.front(); E; E = E->next()) {

		print_line("generating mesh " + itos(ofs++) + "/" + itos(surface_map.size()));
		Ref<ArrayMesh> mesh;
		mesh.instance();
		for (Map<Ref<Material>, Ref<SurfaceTool> >::Element *F = E->get().front(); F; F = F->next()) {
			F->get()->commit(mesh);
		}

		BakedMesh bm;
		bm.mesh = mesh;
		bm.instance = VS::get_singleton()->instance_create();
		VS::get_singleton()->get_singleton()->instance_set_base(bm.instance, bm.mesh->get_rid());
		VS::get_singleton()->instance_attach_object_instance_id(bm.instance, get_instance_id());
		if (is_inside_tree()) {
			VS::get_singleton()->instance_set_scenario(bm.instance, get_world()->get_scenario());
			VS::get_singleton()->instance_set_transform(bm.instance, get_global_transform());
		}

		if (p_gen_lightmap_uv) {
			mesh->lightmap_unwrap(get_global_transform(), p_lightmap_uv_texel_size);
		}
		baked_meshes.push_back(bm);
	}

	_recreate_octant_data();
}

Array GridMap::get_bake_meshes() {

	if (!baked_meshes.size()) {
		make_baked_meshes(true);
	}

	Array arr;
	for (int i = 0; i < baked_meshes.size(); i++) {
		arr.push_back(baked_meshes[i].mesh);
		arr.push_back(Transform());
	}

	return arr;
}

RID GridMap::get_bake_mesh_instance(int p_idx) {

	ERR_FAIL_INDEX_V(p_idx, baked_meshes.size(), RID());
	return baked_meshes[p_idx].instance;
}

GridMap::GridMap() {

	cell_size = Vector3(2, 2, 2);
	octant_size = 8;
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
	recreating_octants = false;
}

GridMap::~GridMap() {

	if (!theme.is_null())
		theme->unregister_owner(this);

	clear();
}
