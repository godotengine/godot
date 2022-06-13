/*************************************************************************/
/*  grid_map.cpp                                                         */
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

#include "grid_map.h"

#include "core/io/marshalls.h"
#include "core/message_queue.h"
#include "scene/3d/light.h"
#include "scene/resources/mesh_library.h"
#include "scene/resources/physics_material.h"
#include "scene/resources/primitive_meshes.h"
#include "scene/resources/surface_tool.h"
#include "scene/scene_string_names.h"
#include "servers/navigation_server.h"
#include "servers/visual_server.h"

bool GridMap::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;

	if (name == "data") {
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
			bm.instance = RID_PRIME(VS::get_singleton()->instance_create());
			VS::get_singleton()->instance_set_portal_mode(bm.instance, VisualServer::InstancePortalMode::INSTANCE_PORTAL_MODE_GLOBAL);
			VS::get_singleton()->get_singleton()->instance_set_base(bm.instance, bm.mesh->get_rid());
			VS::get_singleton()->instance_attach_object_instance_id(bm.instance, get_instance_id());
			if (is_inside_tree()) {
				VS::get_singleton()->instance_set_scenario(bm.instance, get_world()->get_scenario());
				VS::get_singleton()->instance_set_transform(bm.instance, get_global_transform());
			}
			baked_meshes.push_back(bm);
		}

		_recreate_octant_data();

	} else {
		return false;
	}

	return true;
}

bool GridMap::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;

	if (name == "data") {
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
			ret[i] = baked_meshes[i].mesh;
		}
		r_ret = ret;

	} else {
		return false;
	}

	return true;
}

void GridMap::_get_property_list(List<PropertyInfo> *p_list) const {
	if (baked_meshes.size()) {
		p_list->push_back(PropertyInfo(Variant::ARRAY, "baked_meshes", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	}

	p_list->push_back(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
}

void GridMap::set_collision_layer(uint32_t p_layer) {
	collision_layer = p_layer;
	_reset_physic_bodies_collision_filters();
}

uint32_t GridMap::get_collision_layer() const {
	return collision_layer;
}

void GridMap::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
	_reset_physic_bodies_collision_filters();
}

uint32_t GridMap::get_collision_mask() const {
	return collision_mask;
}

void GridMap::set_collision_mask_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision mask bit must be between 0 and 31 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << p_bit;
	} else {
		mask &= ~(1 << p_bit);
	}
	set_collision_mask(mask);
}

bool GridMap::get_collision_mask_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision mask bit must be between 0 and 31 inclusive.");
	return get_collision_mask() & (1 << p_bit);
}

void GridMap::set_collision_layer_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision layer bit must be between 0 and 31 inclusive.");
	uint32_t layer = get_collision_layer();
	if (p_value) {
		layer |= 1 << p_bit;
	} else {
		layer &= ~(1 << p_bit);
	}
	set_collision_layer(layer);
}

bool GridMap::get_collision_layer_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision layer bit must be between 0 and 31 inclusive.");
	return get_collision_layer() & (1 << p_bit);
}

void GridMap::set_physics_material(Ref<PhysicsMaterial> p_material) {
	physics_material = p_material;
	_recreate_octant_data();
}

Ref<PhysicsMaterial> GridMap::get_physics_material() const {
	return physics_material;
}

Array GridMap::get_collision_shapes() const {
	Array shapes;
	for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
		Octant *g = E->get();
		RID body = g->static_body;
		Transform body_xform = PhysicsServer::get_singleton()->body_get_state(body, PhysicsServer::BODY_STATE_TRANSFORM);
		int nshapes = PhysicsServer::get_singleton()->body_get_shape_count(body);
		for (int i = 0; i < nshapes; i++) {
			RID shape = PhysicsServer::get_singleton()->body_get_shape(body, i);
			Transform xform = PhysicsServer::get_singleton()->body_get_shape_transform(body, i);
			shapes.push_back(body_xform * xform);
			shapes.push_back(shape);
		}
	}

	return shapes;
}

void GridMap::set_bake_navigation(bool p_bake_navigation) {
	bake_navigation = p_bake_navigation;
	_recreate_octant_data();
}

bool GridMap::is_baking_navigation() {
	return bake_navigation;
}

void GridMap::set_navigation_layers(uint32_t p_navigation_layers) {
	navigation_layers = p_navigation_layers;
	_recreate_octant_data();
}

uint32_t GridMap::get_navigation_layers() {
	return navigation_layers;
}

void GridMap::set_mesh_library(const Ref<MeshLibrary> &p_mesh_library) {
	if (!mesh_library.is_null()) {
		mesh_library->unregister_owner(this);
	}
	mesh_library = p_mesh_library;
	if (!mesh_library.is_null()) {
		mesh_library->register_owner(this);
	}

	_recreate_octant_data();
	_change_notify("mesh_library");
}

Ref<MeshLibrary> GridMap::get_mesh_library() const {
	return mesh_library;
}

void GridMap::set_use_in_baked_light(bool p_use_baked_light) {
	use_in_baked_light = p_use_baked_light;
}

bool GridMap::get_use_in_baked_light() const {
	return use_in_baked_light;
}

void GridMap::set_cell_size(const Vector3 &p_size) {
	ERR_FAIL_COND(p_size.x < 0.001 || p_size.y < 0.001 || p_size.z < 0.001);
	cell_size = p_size;
	_recreate_octant_data();
	emit_signal("cell_size_changed", cell_size);
}
Vector3 GridMap::get_cell_size() const {
	return cell_size;
}

void GridMap::set_octant_size(int p_size) {
	ERR_FAIL_COND(p_size == 0);
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
		g->static_body = RID_PRIME(PhysicsServer::get_singleton()->body_create(PhysicsServer::BODY_MODE_STATIC));
		PhysicsServer::get_singleton()->body_attach_object_instance_id(g->static_body, get_instance_id());
		PhysicsServer::get_singleton()->body_set_collision_layer(g->static_body, collision_layer);
		PhysicsServer::get_singleton()->body_set_collision_mask(g->static_body, collision_mask);
		if (physics_material.is_valid()) {
			PhysicsServer::get_singleton()->body_set_param(g->static_body, PhysicsServer::BODY_PARAM_FRICTION, physics_material->get_friction());
			PhysicsServer::get_singleton()->body_set_param(g->static_body, PhysicsServer::BODY_PARAM_BOUNCE, physics_material->get_bounce());
		}
		SceneTree *st = SceneTree::get_singleton();

		if (st && st->is_debugging_collisions_hint()) {
			g->collision_debug = RID_PRIME(VisualServer::get_singleton()->mesh_create());
			g->collision_debug_instance = RID_PRIME(VisualServer::get_singleton()->instance_create());
			VS::get_singleton()->instance_set_portal_mode(g->collision_debug_instance, VisualServer::InstancePortalMode::INSTANCE_PORTAL_MODE_GLOBAL);
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

	if (!cell_map.has(key)) {
		return INVALID_CELL_ITEM;
	}
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

	if (!cell_map.has(key)) {
		return -1;
	}
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

	// update transform for NavigationServer regions and navigation debugmesh instances
	if (bake_navigation) {
		for (Map<IndexKey, Octant::NavMesh>::Element *E = g.navmesh_ids.front(); E; E = E->next()) {
			if (E->get().region.is_valid()) {
				NavigationServer::get_singleton()->region_set_transform(E->get().region, get_global_transform() * E->get().xform);
			}
			if (E->get().navmesh_debug_instance.is_valid()) {
				VS::get_singleton()->instance_set_transform(E->get().navmesh_debug_instance, get_global_transform() * E->get().xform);
			}
		}
	}

	for (int i = 0; i < g.multimesh_instances.size(); i++) {
		VS::get_singleton()->instance_set_transform(g.multimesh_instances[i].instance, get_global_transform());
	}
}

bool GridMap::_octant_update(const OctantKey &p_key) {
	ERR_FAIL_COND_V(!octant_map.has(p_key), false);
	Octant &g = *octant_map[p_key];
	if (!g.dirty) {
		return false;
	}

	//erase body shapes
	PhysicsServer::get_singleton()->body_clear_shapes(g.static_body);

	//erase body shapes debug
	if (g.collision_debug.is_valid()) {
		VS::get_singleton()->mesh_clear(g.collision_debug);
	}

	//erase navigation
	for (Map<IndexKey, Octant::NavMesh>::Element *E = g.navmesh_ids.front(); E; E = E->next()) {
		NavigationServer::get_singleton()->free(E->get().region);
		if (E->get().navmesh_debug_instance.is_valid()) {
			VS::get_singleton()->free(E->get().navmesh_debug_instance);
		}
	}
	g.navmesh_ids.clear();

	//erase multimeshes

	for (int i = 0; i < g.multimesh_instances.size(); i++) {
		if (g.multimesh_instances[i].instance.is_valid()) {
			VS::get_singleton()->free(g.multimesh_instances[i].instance);
		}
		if (g.multimesh_instances[i].multimesh.is_valid()) {
			VS::get_singleton()->free(g.multimesh_instances[i].multimesh);
		}
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

	Map<int, List<Pair<Transform, IndexKey>>> multimesh_items;

	for (Set<IndexKey>::Element *E = g.cells.front(); E; E = E->next()) {
		ERR_CONTINUE(!cell_map.has(E->get()));
		const Cell &c = cell_map[E->get()];

		if (!mesh_library.is_valid() || !mesh_library->has_item(c.item)) {
			continue;
		}

		Vector3 cellpos = Vector3(E->get().x, E->get().y, E->get().z);
		Vector3 ofs = _get_offset();

		Transform xform;

		xform.basis.set_orthogonal_index(c.rot);
		xform.set_origin(cellpos * cell_size + ofs);
		xform.basis.scale(Vector3(cell_scale, cell_scale, cell_scale));
		if (baked_meshes.size() == 0) {
			if (mesh_library->get_item_mesh(c.item).is_valid()) {
				if (!multimesh_items.has(c.item)) {
					multimesh_items[c.item] = List<Pair<Transform, IndexKey>>();
				}

				Pair<Transform, IndexKey> p;
				p.first = xform * mesh_library->get_item_mesh_transform(c.item);
				p.second = E->get();
				multimesh_items[c.item].push_back(p);
			}
		}

		Vector<MeshLibrary::ShapeData> shapes = mesh_library->get_item_shapes(c.item);
		// add the item's shape at given xform to octant's static_body
		for (int i = 0; i < shapes.size(); i++) {
			// add the item's shape
			if (!shapes[i].shape.is_valid()) {
				continue;
			}
			PhysicsServer::get_singleton()->body_add_shape(g.static_body, shapes[i].shape->get_rid(), xform * shapes[i].local_transform);
			if (g.collision_debug.is_valid()) {
				shapes.write[i].shape->add_vertices_to_array(col_debug, xform * shapes[i].local_transform);
			}
		}

		// add the item's navmesh at given xform to GridMap's Navigation ancestor
		Ref<NavigationMesh> navmesh = mesh_library->get_item_navmesh(c.item);
		if (navmesh.is_valid()) {
			Octant::NavMesh nm;
			nm.xform = xform * mesh_library->get_item_navmesh_transform(c.item);

			if (bake_navigation) {
				RID region = NavigationServer::get_singleton()->region_create();
				NavigationServer::get_singleton()->region_set_navigation_layers(region, navigation_layers);
				NavigationServer::get_singleton()->region_set_navmesh(region, navmesh);
				NavigationServer::get_singleton()->region_set_transform(region, get_global_transform() * nm.xform);
				if (navigation) {
					NavigationServer::get_singleton()->region_set_map(region, navigation->get_rid());
				} else {
					NavigationServer::get_singleton()->region_set_map(region, get_world()->get_navigation_map());
				}
				nm.region = region;

				// add navigation debugmesh visual instances if debug is enabled
				SceneTree *st = SceneTree::get_singleton();
				if (st && st->is_debugging_navigation_hint()) {
					if (!nm.navmesh_debug_instance.is_valid()) {
						RID navmesh_debug_rid = navmesh->get_debug_mesh()->get_rid();
						nm.navmesh_debug_instance = VS::get_singleton()->instance_create();
						VS::get_singleton()->instance_set_base(nm.navmesh_debug_instance, navmesh_debug_rid);
						VS::get_singleton()->mesh_surface_set_material(navmesh_debug_rid, 0, st->get_debug_navigation_material()->get_rid());
					}
					if (is_inside_tree()) {
						VS::get_singleton()->instance_set_scenario(nm.navmesh_debug_instance, get_world()->get_scenario());
						VS::get_singleton()->instance_set_transform(nm.navmesh_debug_instance, get_global_transform() * nm.xform);
					}
				}
			}
			g.navmesh_ids[E->get()] = nm;
		}
	}

	//update multimeshes, only if not baked
	if (baked_meshes.size() == 0) {
		for (Map<int, List<Pair<Transform, IndexKey>>>::Element *E = multimesh_items.front(); E; E = E->next()) {
			Octant::MultimeshInstance mmi;

			RID mm = RID_PRIME(VS::get_singleton()->multimesh_create());
			VS::get_singleton()->multimesh_allocate(mm, E->get().size(), VS::MULTIMESH_TRANSFORM_3D, VS::MULTIMESH_COLOR_NONE);
			VS::get_singleton()->multimesh_set_mesh(mm, mesh_library->get_item_mesh(E->key())->get_rid());

			int idx = 0;
			for (List<Pair<Transform, IndexKey>>::Element *F = E->get().front(); F; F = F->next()) {
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

			RID instance = RID_PRIME(VS::get_singleton()->instance_create());
			VS::get_singleton()->instance_set_base(instance, mm);
			VS::get_singleton()->instance_set_portal_mode(instance, VisualServer::InstancePortalMode::INSTANCE_PORTAL_MODE_GLOBAL);

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

void GridMap::_reset_physic_bodies_collision_filters() {
	for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
		PhysicsServer::get_singleton()->body_set_collision_layer(E->get()->static_body, collision_layer);
		PhysicsServer::get_singleton()->body_set_collision_mask(E->get()->static_body, collision_mask);
	}
}

void GridMap::_octant_enter_world(const OctantKey &p_key) {
	ERR_FAIL_COND(!octant_map.has(p_key));
	Octant &g = *octant_map[p_key];
	PhysicsServer::get_singleton()->body_set_state(g.static_body, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());
	PhysicsServer::get_singleton()->body_set_space(g.static_body, get_world()->get_space());

	if (g.collision_debug_instance.is_valid()) {
		VS::get_singleton()->instance_set_scenario(g.collision_debug_instance, get_world()->get_scenario());
		VS::get_singleton()->instance_set_transform(g.collision_debug_instance, get_global_transform());
	}

	for (int i = 0; i < g.multimesh_instances.size(); i++) {
		VS::get_singleton()->instance_set_scenario(g.multimesh_instances[i].instance, get_world()->get_scenario());
		VS::get_singleton()->instance_set_transform(g.multimesh_instances[i].instance, get_global_transform());
	}

	if (bake_navigation && mesh_library.is_valid()) {
		for (Map<IndexKey, Octant::NavMesh>::Element *E = g.navmesh_ids.front(); E; E = E->next()) {
			if (cell_map.has(E->key()) && E->get().region.is_valid() == false) {
				Ref<NavigationMesh> nm = mesh_library->get_item_navmesh(cell_map[E->key()].item);
				if (nm.is_valid()) {
					RID region = NavigationServer::get_singleton()->region_create();
					NavigationServer::get_singleton()->region_set_navigation_layers(region, navigation_layers);
					NavigationServer::get_singleton()->region_set_navmesh(region, nm);
					NavigationServer::get_singleton()->region_set_transform(region, get_global_transform() * E->get().xform);
					if (navigation) {
						NavigationServer::get_singleton()->region_set_map(region, navigation->get_rid());
					} else {
						NavigationServer::get_singleton()->region_set_map(region, get_world()->get_navigation_map());
					}
					E->get().region = region;
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

	for (Map<IndexKey, Octant::NavMesh>::Element *E = g.navmesh_ids.front(); E; E = E->next()) {
		if (E->get().region.is_valid()) {
			NavigationServer::get_singleton()->free(E->get().region);
			E->get().region = RID();
		}
		if (E->get().navmesh_debug_instance.is_valid()) {
			VS::get_singleton()->free(E->get().navmesh_debug_instance);
			E->get().navmesh_debug_instance = RID();
		}
	}
}

void GridMap::_octant_clean_up(const OctantKey &p_key) {
	ERR_FAIL_COND(!octant_map.has(p_key));
	Octant &g = *octant_map[p_key];

	if (g.collision_debug.is_valid()) {
		VS::get_singleton()->free(g.collision_debug);
		g.collision_debug = RID();
	}

	if (g.collision_debug_instance.is_valid()) {
		VS::get_singleton()->free(g.collision_debug_instance);
		g.collision_debug_instance = RID();
	}

	if (g.static_body.is_valid()) {
		PhysicsServer::get_singleton()->free(g.static_body);
		g.static_body = RID();
	}

	// Erase navigation
	for (Map<IndexKey, Octant::NavMesh>::Element *E = g.navmesh_ids.front(); E; E = E->next()) {
		if (E->get().region.is_valid()) {
			NavigationServer::get_singleton()->free(E->get().region);
		}
		if (E->get().navmesh_debug_instance.is_valid()) {
			VS::get_singleton()->free(E->get().navmesh_debug_instance);
		}
	}
	g.navmesh_ids.clear();

	//erase multimeshes

	for (int i = 0; i < g.multimesh_instances.size(); i++) {
		if (g.multimesh_instances[i].instance.is_valid()) {
			VS::get_singleton()->free(g.multimesh_instances[i].instance);
		}
		if (g.multimesh_instances[i].multimesh.is_valid()) {
			VS::get_singleton()->free(g.multimesh_instances[i].multimesh);
		}
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
			if (new_xform == last_transform) {
				break;
			}
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

			navigation = nullptr;

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
	if (!is_inside_tree()) {
		return;
	}

	_change_notify("visible");

	for (Map<OctantKey, Octant *>::Element *e = octant_map.front(); e; e = e->next()) {
		Octant *octant = e->value();
		for (int i = 0; i < octant->multimesh_instances.size(); i++) {
			const Octant::MultimeshInstance &mi = octant->multimesh_instances[i];
			VS::get_singleton()->instance_set_visible(mi.instance, is_visible_in_tree());
		}
	}

	for (int i = 0; i < baked_meshes.size(); i++) {
		VS::get_singleton()->instance_set_visible(baked_meshes[i].instance, is_visible_in_tree());
	}
}

void GridMap::_queue_octants_dirty() {
	if (awaiting_update) {
		return;
	}

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
		if (is_inside_world()) {
			_octant_exit_world(E->key());
		}

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
	if (!awaiting_update) {
		return;
	}

	List<OctantKey> to_delete;
	for (Map<OctantKey, Octant *>::Element *E = octant_map.front(); E; E = E->next()) {
		if (_octant_update(E->key())) {
			to_delete.push_back(E->key());
		}
	}

	while (to_delete.front()) {
		memdelete(octant_map[to_delete.front()->get()]);
		octant_map.erase(to_delete.front()->get());
		to_delete.pop_front();
	}

	_update_visibility();
	awaiting_update = false;
}

void GridMap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_collision_layer", "layer"), &GridMap::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &GridMap::get_collision_layer);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &GridMap::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &GridMap::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_bit", "bit", "value"), &GridMap::set_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("get_collision_mask_bit", "bit"), &GridMap::get_collision_mask_bit);

	ClassDB::bind_method(D_METHOD("set_collision_layer_bit", "bit", "value"), &GridMap::set_collision_layer_bit);
	ClassDB::bind_method(D_METHOD("get_collision_layer_bit", "bit"), &GridMap::get_collision_layer_bit);

	ClassDB::bind_method(D_METHOD("set_physics_material", "material"), &GridMap::set_physics_material);
	ClassDB::bind_method(D_METHOD("get_physics_material"), &GridMap::get_physics_material);

	ClassDB::bind_method(D_METHOD("set_bake_navigation", "bake_navigation"), &GridMap::set_bake_navigation);
	ClassDB::bind_method(D_METHOD("is_baking_navigation"), &GridMap::is_baking_navigation);

	ClassDB::bind_method(D_METHOD("set_navigation_layers", "navigation_layers"), &GridMap::set_navigation_layers);
	ClassDB::bind_method(D_METHOD("get_navigation_layers"), &GridMap::get_navigation_layers);

	ClassDB::bind_method(D_METHOD("set_mesh_library", "mesh_library"), &GridMap::set_mesh_library);
	ClassDB::bind_method(D_METHOD("get_mesh_library"), &GridMap::get_mesh_library);

	ClassDB::bind_method(D_METHOD("set_cell_size", "size"), &GridMap::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &GridMap::get_cell_size);

	ClassDB::bind_method(D_METHOD("set_cell_scale", "scale"), &GridMap::set_cell_scale);
	ClassDB::bind_method(D_METHOD("get_cell_scale"), &GridMap::get_cell_scale);

	ClassDB::bind_method(D_METHOD("set_octant_size", "size"), &GridMap::set_octant_size);
	ClassDB::bind_method(D_METHOD("get_octant_size"), &GridMap::get_octant_size);

	ClassDB::bind_method(D_METHOD("set_cell_item", "x", "y", "z", "item", "orientation"), &GridMap::set_cell_item, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_cell_item", "x", "y", "z"), &GridMap::get_cell_item);
	ClassDB::bind_method(D_METHOD("get_cell_item_orientation", "x", "y", "z"), &GridMap::get_cell_item_orientation);

	ClassDB::bind_method(D_METHOD("world_to_map", "pos"), &GridMap::world_to_map);
	ClassDB::bind_method(D_METHOD("map_to_world", "x", "y", "z"), &GridMap::map_to_world);

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
	ClassDB::bind_method(D_METHOD("get_used_cells_by_item", "item"), &GridMap::get_used_cells_by_item);

	ClassDB::bind_method(D_METHOD("get_meshes"), &GridMap::get_meshes);
	ClassDB::bind_method(D_METHOD("get_bake_meshes"), &GridMap::get_bake_meshes);
	ClassDB::bind_method(D_METHOD("get_bake_mesh_instance", "idx"), &GridMap::get_bake_mesh_instance);

	ClassDB::bind_method(D_METHOD("clear_baked_meshes"), &GridMap::clear_baked_meshes);
	ClassDB::bind_method(D_METHOD("make_baked_meshes", "gen_lightmap_uv", "lightmap_uv_texel_size"), &GridMap::make_baked_meshes, DEFVAL(false), DEFVAL(0.1));

	ClassDB::bind_method(D_METHOD("set_use_in_baked_light", "use_in_baked_light"), &GridMap::set_use_in_baked_light);
	ClassDB::bind_method(D_METHOD("get_use_in_baked_light"), &GridMap::get_use_in_baked_light);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh_library", PROPERTY_HINT_RESOURCE_TYPE, "MeshLibrary"), "set_mesh_library", "get_mesh_library");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material", "get_physics_material");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_in_baked_light"), "set_use_in_baked_light", "get_use_in_baked_light");
	ADD_GROUP("Cell", "cell_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "cell_size"), "set_cell_size", "get_cell_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cell_octant_size", PROPERTY_HINT_RANGE, "1,1024,1"), "set_octant_size", "get_octant_size");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cell_center_x"), "set_center_x", "get_center_x");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cell_center_y"), "set_center_y", "get_center_y");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cell_center_z"), "set_center_z", "get_center_z");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "cell_scale"), "set_cell_scale", "get_cell_scale");
	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bake_navigation"), "set_bake_navigation", "is_baking_navigation");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_layers", PROPERTY_HINT_LAYERS_3D_NAVIGATION), "set_navigation_layers", "get_navigation_layers");

	BIND_CONSTANT(INVALID_CELL_ITEM);

	ADD_SIGNAL(MethodInfo("cell_size_changed", PropertyInfo(Variant::VECTOR3, "cell_size")));
}

void GridMap::set_clip(bool p_enabled, bool p_clip_above, int p_floor, Vector3::Axis p_axis) {
	if (!p_enabled && !clip) {
		return;
	}
	if (clip && p_enabled && clip_floor == p_floor && p_clip_above == clip_above && p_axis == clip_axis) {
		return;
	}

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

Array GridMap::get_used_cells_by_item(int p_item) const {
	Array a;
	for (Map<IndexKey, Cell>::Element *E = cell_map.front(); E; E = E->next()) {
		if (E->value().item == p_item) {
			Vector3 p(E->key().x, E->key().y, E->key().z);
			a.push_back(p);
		}
	}

	return a;
}

Array GridMap::get_meshes() const {
	if (mesh_library.is_null()) {
		return Array();
	}

	Vector3 ofs = _get_offset();
	Array meshes;

	for (Map<IndexKey, Cell>::Element *E = cell_map.front(); E; E = E->next()) {
		int id = E->get().item;
		if (!mesh_library->has_item(id)) {
			continue;
		}
		Ref<Mesh> mesh = mesh_library->get_item_mesh(id);
		if (mesh.is_null()) {
			continue;
		}

		IndexKey ik = E->key();

		Vector3 cellpos = Vector3(ik.x, ik.y, ik.z);

		Transform xform;

		xform.basis.set_orthogonal_index(E->get().rot);

		xform.set_origin(cellpos * cell_size + ofs);
		xform.basis.scale(Vector3(cell_scale, cell_scale, cell_scale));

		meshes.push_back(xform * mesh_library->get_item_mesh_transform(id));
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
		if (baked_meshes[i].instance.is_valid()) {
			VS::get_singleton()->free(baked_meshes[i].instance);
		}
	}
	baked_meshes.clear();

	_recreate_octant_data();
}

void GridMap::make_baked_meshes(bool p_gen_lightmap_uv, float p_lightmap_uv_texel_size) {
	if (!mesh_library.is_valid()) {
		return;
	}

	//generate
	Map<OctantKey, Map<Ref<Material>, Ref<SurfaceTool>>> surface_map;

	for (Map<IndexKey, Cell>::Element *E = cell_map.front(); E; E = E->next()) {
		IndexKey key = E->key();

		int item = E->get().item;
		if (!mesh_library->has_item(item)) {
			continue;
		}

		Ref<Mesh> mesh = mesh_library->get_item_mesh(item);
		if (!mesh.is_valid()) {
			continue;
		}

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
			surface_map[ok] = Map<Ref<Material>, Ref<SurfaceTool>>();
		}

		Map<Ref<Material>, Ref<SurfaceTool>> &mat_map = surface_map[ok];

		for (int i = 0; i < mesh->get_surface_count(); i++) {
			if (mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
				continue;
			}

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

	for (Map<OctantKey, Map<Ref<Material>, Ref<SurfaceTool>>>::Element *E = surface_map.front(); E; E = E->next()) {
		Ref<ArrayMesh> mesh;
		mesh.instance();
		for (Map<Ref<Material>, Ref<SurfaceTool>>::Element *F = E->get().front(); F; F = F->next()) {
			F->get()->commit(mesh);
		}

		BakedMesh bm;
		bm.mesh = mesh;
		bm.instance = RID_PRIME(VS::get_singleton()->instance_create());
		VS::get_singleton()->instance_set_portal_mode(bm.instance, VisualServer::InstancePortalMode::INSTANCE_PORTAL_MODE_GLOBAL);
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
	if (!use_in_baked_light) {
		return Array();
	}

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
	collision_layer = 1;
	collision_mask = 1;

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

	navigation = nullptr;
	set_notify_transform(true);
	recreating_octants = false;

	use_in_baked_light = false;
}

GridMap::~GridMap() {
	if (!mesh_library.is_null()) {
		mesh_library->unregister_owner(this);
	}

	clear();
}
