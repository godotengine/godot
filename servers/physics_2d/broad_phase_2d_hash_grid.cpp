/*************************************************************************/
/*  broad_phase_2d_hash_grid.cpp                                         */
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

#include "broad_phase_2d_hash_grid.h"
#include "collision_object_2d_sw.h"
#include "core/project_settings.h"

#define LARGE_ELEMENT_FI 1.01239812

void BroadPhase2DHashGrid::_pair_attempt(Element *p_elem, Element *p_with) {
	if (p_elem->owner == p_with->owner) {
		return;
	}
	if (!_test_collision_mask(p_elem->collision_mask, p_elem->collision_layer, p_with->collision_mask, p_with->collision_layer)) {
		return;
	}
	Map<Element *, PairData *>::Element *E = p_elem->paired.find(p_with);

	ERR_FAIL_COND(p_elem->_static && p_with->_static);

	if (!E) {
		PairData *pd = memnew(PairData);
		p_elem->paired[p_with] = pd;
		p_with->paired[p_elem] = pd;
	} else {
		E->get()->rc++;
	}
}

void BroadPhase2DHashGrid::_unpair_attempt(Element *p_elem, Element *p_with) {
	if (p_elem->owner == p_with->owner) {
		return;
	}
	if (!_test_collision_mask(p_elem->collision_mask, p_elem->collision_layer, p_with->collision_mask, p_with->collision_layer)) {
		return;
	}
	Map<Element *, PairData *>::Element *E = p_elem->paired.find(p_with);

	ERR_FAIL_COND(!E); //this should really be paired..

	E->get()->rc--;

	if (E->get()->rc == 0) {
		if (E->get()->colliding) {
			//uncollide
			if (unpair_callback) {
				unpair_callback(p_elem->owner, p_elem->subindex, p_with->owner, p_with->subindex, E->get()->ud, unpair_userdata);
			}
		}

		memdelete(E->get());
		p_elem->paired.erase(E);
		p_with->paired.erase(p_elem);
	}
}

void BroadPhase2DHashGrid::_check_motion(Element *p_elem) {
	for (Map<Element *, PairData *>::Element *E = p_elem->paired.front(); E; E = E->next()) {
		bool physical_collision = p_elem->aabb.intersects(E->key()->aabb);
		bool logical_collision = p_elem->owner->test_collision_mask(E->key()->owner);

		if (physical_collision && logical_collision) {
			if (!E->get()->colliding && pair_callback) {
				E->get()->ud = pair_callback(p_elem->owner, p_elem->subindex, E->key()->owner, E->key()->subindex, nullptr, pair_userdata);
			}
			E->get()->colliding = true;
		} else { // No collision
			if (E->get()->colliding && unpair_callback) {
				unpair_callback(p_elem->owner, p_elem->subindex, E->key()->owner, E->key()->subindex, E->get()->ud, unpair_userdata);
				E->get()->ud = nullptr;
			}
			E->get()->colliding = false;
		}
	}
}

void BroadPhase2DHashGrid::_enter_grid(Element *p_elem, const Rect2 &p_rect, bool p_static, bool p_force_enter) {
	Vector2 sz = (p_rect.size / cell_size * LARGE_ELEMENT_FI); //use magic number to avoid floating point issues
	if (sz.width * sz.height > large_object_min_surface) {
		//large object, do not use grid, must check against all elements
		for (Map<ID, Element>::Element *E = element_map.front(); E; E = E->next()) {
			if (E->key() == p_elem->self) {
				continue; // do not pair against itself
			}
			if (E->get()._static && p_static) {
				continue;
			}
			_pair_attempt(p_elem, &E->get());
		}

		large_elements[p_elem].inc();
		return;
	}

	Point2i from = (p_rect.position / cell_size).floor();
	Point2i to = ((p_rect.position + p_rect.size) / cell_size).floor();

	for (int i = from.x; i <= to.x; i++) {
		for (int j = from.y; j <= to.y; j++) {
			PosKey pk;
			pk.x = i;
			pk.y = j;

			uint32_t idx = pk.hash() % hash_table_size;
			PosBin *pb = hash_table[idx];

			while (pb) {
				if (pb->key == pk) {
					break;
				}

				pb = pb->next;
			}

			bool entered = p_force_enter;

			if (!pb) {
				//does not exist, create!
				pb = memnew(PosBin);
				pb->key = pk;
				pb->next = hash_table[idx];
				hash_table[idx] = pb;
			}

			if (p_static) {
				if (pb->static_object_set[p_elem].inc() == 1) {
					entered = true;
				}
			} else {
				if (pb->object_set[p_elem].inc() == 1) {
					entered = true;
				}
			}

			if (entered) {
				for (Map<Element *, RC>::Element *E = pb->object_set.front(); E; E = E->next()) {
					_pair_attempt(p_elem, E->key());
				}

				if (!p_static) {
					for (Map<Element *, RC>::Element *E = pb->static_object_set.front(); E; E = E->next()) {
						_pair_attempt(p_elem, E->key());
					}
				}
			}
		}
	}

	//pair separatedly with large elements

	for (Map<Element *, RC>::Element *E = large_elements.front(); E; E = E->next()) {
		if (E->key() == p_elem) {
			continue; // do not pair against itself
		}
		if (E->key()->_static && p_static) {
			continue;
		}
		_pair_attempt(E->key(), p_elem);
	}
}

void BroadPhase2DHashGrid::_exit_grid(Element *p_elem, const Rect2 &p_rect, bool p_static, bool p_force_exit) {
	Vector2 sz = (p_rect.size / cell_size * LARGE_ELEMENT_FI);
	if (sz.width * sz.height > large_object_min_surface) {
		//unpair all elements, instead of checking all, just check what is already paired, so we at least save from checking static vs static
		Map<Element *, PairData *>::Element *E = p_elem->paired.front();
		while (E) {
			Map<Element *, PairData *>::Element *next = E->next();
			_unpair_attempt(p_elem, E->key());
			E = next;
		}

		if (large_elements[p_elem].dec() == 0) {
			large_elements.erase(p_elem);
		}
		return;
	}

	Point2i from = (p_rect.position / cell_size).floor();
	Point2i to = ((p_rect.position + p_rect.size) / cell_size).floor();

	for (int i = from.x; i <= to.x; i++) {
		for (int j = from.y; j <= to.y; j++) {
			PosKey pk;
			pk.x = i;
			pk.y = j;

			uint32_t idx = pk.hash() % hash_table_size;
			PosBin *pb = hash_table[idx];

			while (pb) {
				if (pb->key == pk) {
					break;
				}

				pb = pb->next;
			}

			ERR_CONTINUE(!pb); //should exist!!

			bool exited = p_force_exit;

			if (p_static) {
				if (pb->static_object_set[p_elem].dec() == 0) {
					pb->static_object_set.erase(p_elem);
					exited = true;
				}
			} else {
				if (pb->object_set[p_elem].dec() == 0) {
					pb->object_set.erase(p_elem);
					exited = true;
				}
			}

			if (exited) {
				for (Map<Element *, RC>::Element *E = pb->object_set.front(); E; E = E->next()) {
					_unpair_attempt(p_elem, E->key());
				}

				if (!p_static) {
					for (Map<Element *, RC>::Element *E = pb->static_object_set.front(); E; E = E->next()) {
						_unpair_attempt(p_elem, E->key());
					}
				}
			}

			if (pb->object_set.empty() && pb->static_object_set.empty()) {
				if (hash_table[idx] == pb) {
					hash_table[idx] = pb->next;
				} else {
					PosBin *px = hash_table[idx];

					while (px) {
						if (px->next == pb) {
							px->next = pb->next;
							break;
						}

						px = px->next;
					}

					ERR_CONTINUE(!px);
				}

				memdelete(pb);
			}
		}
	}

	for (Map<Element *, RC>::Element *E = large_elements.front(); E; E = E->next()) {
		if (E->key() == p_elem) {
			continue; // do not pair against itself
		}
		if (E->key()->_static && p_static) {
			continue;
		}
		//unpair from large elements
		_unpair_attempt(p_elem, E->key());
	}
}

BroadPhase2DHashGrid::ID BroadPhase2DHashGrid::create(CollisionObject2DSW *p_object, int p_subindex, const Rect2 &p_aabb, bool p_static) {
	current++;

	Element e;
	e.owner = p_object;
	e._static = false;
	e.collision_mask = p_object->get_collision_mask();
	e.collision_layer = p_object->get_collision_layer();
	e.subindex = p_subindex;
	e.self = current;
	e.pass = 0;

	element_map[current] = e;
	return current;
}

void BroadPhase2DHashGrid::move(ID p_id, const Rect2 &p_aabb) {
	Map<ID, Element>::Element *E = element_map.find(p_id);
	ERR_FAIL_COND(!E);

	Element &e = E->get();
	bool layer_changed = e.collision_mask != e.owner->get_collision_mask() || e.collision_layer != e.owner->get_collision_layer();

	if (p_aabb != e.aabb || layer_changed) {
		uint32_t old_mask = e.collision_mask;
		uint32_t old_layer = e.collision_layer;
		if (p_aabb != Rect2()) {
			e.collision_mask = e.owner->get_collision_mask();
			e.collision_layer = e.owner->get_collision_layer();

			_enter_grid(&e, p_aabb, e._static, layer_changed);
		}

		if (e.aabb != Rect2()) {
			// Need _exit_grid to remove from cells based on the old layer values.
			e.collision_mask = old_mask;
			e.collision_layer = old_layer;

			_exit_grid(&e, e.aabb, e._static, layer_changed);

			e.collision_mask = e.owner->get_collision_mask();
			e.collision_layer = e.owner->get_collision_layer();
		}

		e.aabb = p_aabb;
	}

	_check_motion(&e);
}

void BroadPhase2DHashGrid::recheck_pairs(ID p_id) {
	Map<ID, Element>::Element *E = element_map.find(p_id);
	ERR_FAIL_COND(!E);

	Element &e = E->get();
	move(p_id, e.aabb);
}

void BroadPhase2DHashGrid::set_static(ID p_id, bool p_static) {
	Map<ID, Element>::Element *E = element_map.find(p_id);
	ERR_FAIL_COND(!E);

	Element &e = E->get();

	if (e._static == p_static) {
		return;
	}

	if (e.aabb != Rect2()) {
		_exit_grid(&e, e.aabb, e._static, false);
	}

	e._static = p_static;

	if (e.aabb != Rect2()) {
		_enter_grid(&e, e.aabb, e._static, false);
		_check_motion(&e);
	}
}
void BroadPhase2DHashGrid::remove(ID p_id) {
	Map<ID, Element>::Element *E = element_map.find(p_id);
	ERR_FAIL_COND(!E);

	Element &e = E->get();

	if (e.aabb != Rect2()) {
		_exit_grid(&e, e.aabb, e._static, false);
	}

	element_map.erase(p_id);
}

CollisionObject2DSW *BroadPhase2DHashGrid::get_object(ID p_id) const {
	const Map<ID, Element>::Element *E = element_map.find(p_id);
	ERR_FAIL_COND_V(!E, nullptr);
	return E->get().owner;
}
bool BroadPhase2DHashGrid::is_static(ID p_id) const {
	const Map<ID, Element>::Element *E = element_map.find(p_id);
	ERR_FAIL_COND_V(!E, false);
	return E->get()._static;
}
int BroadPhase2DHashGrid::get_subindex(ID p_id) const {
	const Map<ID, Element>::Element *E = element_map.find(p_id);
	ERR_FAIL_COND_V(!E, -1);
	return E->get().subindex;
}

template <bool use_aabb, bool use_segment>
void BroadPhase2DHashGrid::_cull(const Point2i p_cell, const Rect2 &p_aabb, const Point2 &p_from, const Point2 &p_to, CollisionObject2DSW **p_results, int p_max_results, int *p_result_indices, int &index) {
	PosKey pk;
	pk.x = p_cell.x;
	pk.y = p_cell.y;

	uint32_t idx = pk.hash() % hash_table_size;
	PosBin *pb = hash_table[idx];

	while (pb) {
		if (pb->key == pk) {
			break;
		}

		pb = pb->next;
	}

	if (!pb) {
		return;
	}

	for (Map<Element *, RC>::Element *E = pb->object_set.front(); E; E = E->next()) {
		if (index >= p_max_results) {
			break;
		}
		if (E->key()->pass == pass) {
			continue;
		}

		E->key()->pass = pass;

		if (use_aabb && !p_aabb.intersects(E->key()->aabb)) {
			continue;
		}

		if (use_segment && !E->key()->aabb.intersects_segment(p_from, p_to)) {
			continue;
		}

		p_results[index] = E->key()->owner;
		p_result_indices[index] = E->key()->subindex;
		index++;
	}

	for (Map<Element *, RC>::Element *E = pb->static_object_set.front(); E; E = E->next()) {
		if (index >= p_max_results) {
			break;
		}
		if (E->key()->pass == pass) {
			continue;
		}

		if (use_aabb && !p_aabb.intersects(E->key()->aabb)) {
			continue;
		}

		if (use_segment && !E->key()->aabb.intersects_segment(p_from, p_to)) {
			continue;
		}

		E->key()->pass = pass;
		p_results[index] = E->key()->owner;
		p_result_indices[index] = E->key()->subindex;
		index++;
	}
}

int BroadPhase2DHashGrid::cull_segment(const Vector2 &p_from, const Vector2 &p_to, CollisionObject2DSW **p_results, int p_max_results, int *p_result_indices) {
	pass++;

	Vector2 dir = (p_to - p_from);
	if (dir == Vector2()) {
		return 0;
	}
	//avoid divisions by zero
	dir.normalize();
	if (dir.x == 0.0) {
		dir.x = 0.000001;
	}
	if (dir.y == 0.0) {
		dir.y = 0.000001;
	}
	Vector2 delta = dir.abs();

	delta.x = cell_size / delta.x;
	delta.y = cell_size / delta.y;

	Point2i pos = (p_from / cell_size).floor();
	Point2i end = (p_to / cell_size).floor();

	Point2i step = Vector2(SGN(dir.x), SGN(dir.y));

	Vector2 max;

	if (dir.x < 0) {
		max.x = (Math::floor((double)pos.x) * cell_size - p_from.x) / dir.x;
	} else {
		max.x = (Math::floor((double)pos.x + 1) * cell_size - p_from.x) / dir.x;
	}

	if (dir.y < 0) {
		max.y = (Math::floor((double)pos.y) * cell_size - p_from.y) / dir.y;
	} else {
		max.y = (Math::floor((double)pos.y + 1) * cell_size - p_from.y) / dir.y;
	}

	int cullcount = 0;
	_cull<false, true>(pos, Rect2(), p_from, p_to, p_results, p_max_results, p_result_indices, cullcount);

	bool reached_x = false;
	bool reached_y = false;

	while (true) {
		if (max.x < max.y) {
			max.x += delta.x;
			pos.x += step.x;
		} else {
			max.y += delta.y;
			pos.y += step.y;
		}

		if (step.x > 0) {
			if (pos.x >= end.x) {
				reached_x = true;
			}
		} else if (pos.x <= end.x) {
			reached_x = true;
		}

		if (step.y > 0) {
			if (pos.y >= end.y) {
				reached_y = true;
			}
		} else if (pos.y <= end.y) {
			reached_y = true;
		}

		_cull<false, true>(pos, Rect2(), p_from, p_to, p_results, p_max_results, p_result_indices, cullcount);

		if (reached_x && reached_y) {
			break;
		}
	}

	for (Map<Element *, RC>::Element *E = large_elements.front(); E; E = E->next()) {
		if (cullcount >= p_max_results) {
			break;
		}
		if (E->key()->pass == pass) {
			continue;
		}

		E->key()->pass = pass;

		/*
		if (use_aabb && !p_aabb.intersects(E->key()->aabb))
			continue;
		*/

		if (!E->key()->aabb.intersects_segment(p_from, p_to)) {
			continue;
		}

		p_results[cullcount] = E->key()->owner;
		p_result_indices[cullcount] = E->key()->subindex;
		cullcount++;
	}

	return cullcount;
}

int BroadPhase2DHashGrid::cull_aabb(const Rect2 &p_aabb, CollisionObject2DSW **p_results, int p_max_results, int *p_result_indices) {
	pass++;

	Point2i from = (p_aabb.position / cell_size).floor();
	Point2i to = ((p_aabb.position + p_aabb.size) / cell_size).floor();
	int cullcount = 0;

	for (int i = from.x; i <= to.x; i++) {
		for (int j = from.y; j <= to.y; j++) {
			_cull<true, false>(Point2i(i, j), p_aabb, Point2(), Point2(), p_results, p_max_results, p_result_indices, cullcount);
		}
	}

	for (Map<Element *, RC>::Element *E = large_elements.front(); E; E = E->next()) {
		if (cullcount >= p_max_results) {
			break;
		}
		if (E->key()->pass == pass) {
			continue;
		}

		E->key()->pass = pass;

		if (!p_aabb.intersects(E->key()->aabb)) {
			continue;
		}

		/*
		if (!E->key()->aabb.intersects_segment(p_from,p_to))
			continue;
		*/

		p_results[cullcount] = E->key()->owner;
		p_result_indices[cullcount] = E->key()->subindex;
		cullcount++;
	}
	return cullcount;
}

void BroadPhase2DHashGrid::set_pair_callback(PairCallback p_pair_callback, void *p_userdata) {
	pair_callback = p_pair_callback;
	pair_userdata = p_userdata;
}
void BroadPhase2DHashGrid::set_unpair_callback(UnpairCallback p_unpair_callback, void *p_userdata) {
	unpair_callback = p_unpair_callback;
	unpair_userdata = p_userdata;
}

void BroadPhase2DHashGrid::update() {
}

BroadPhase2DSW *BroadPhase2DHashGrid::_create() {
	return memnew(BroadPhase2DHashGrid);
}

BroadPhase2DHashGrid::BroadPhase2DHashGrid() {
	hash_table_size = GLOBAL_GET("physics/2d/bp_hash_table_size");
	ProjectSettings::get_singleton()->set_custom_property_info("physics/2d/bp_hash_table_size", PropertyInfo(Variant::INT, "physics/2d/bp_hash_table_size", PROPERTY_HINT_RANGE, "0,8192,1,or_greater"));
	hash_table_size = Math::larger_prime(hash_table_size);
	hash_table = memnew_arr(PosBin *, hash_table_size);

	cell_size = GLOBAL_GET("physics/2d/cell_size");
	ProjectSettings::get_singleton()->set_custom_property_info("physics/2d/cell_size", PropertyInfo(Variant::INT, "physics/2d/cell_size", PROPERTY_HINT_RANGE, "0,512,1,or_greater"));

	large_object_min_surface = GLOBAL_GET("physics/2d/large_object_surface_threshold_in_cells");
	ProjectSettings::get_singleton()->set_custom_property_info("physics/2d/large_object_surface_threshold_in_cells", PropertyInfo(Variant::INT, "physics/2d/large_object_surface_threshold_in_cells", PROPERTY_HINT_RANGE, "0,1024,1,or_greater"));

	for (uint32_t i = 0; i < hash_table_size; i++) {
		hash_table[i] = nullptr;
	}
	pass = 1;

	current = 0;
}

BroadPhase2DHashGrid::~BroadPhase2DHashGrid() {
	for (uint32_t i = 0; i < hash_table_size; i++) {
		while (hash_table[i]) {
			PosBin *pb = hash_table[i];
			hash_table[i] = pb->next;
			memdelete(pb);
		}
	}

	memdelete_arr(hash_table);
}

/* 3D version of voxel traversal:

public IEnumerable<Point3D> GetCellsOnRay(Ray ray, int maxDepth)
{
    // Implementation is based on:
    // "A Fast Voxel Traversal Algorithm for Ray Tracing"
    // John Amanatides, Andrew Woo
    // http://www.cse.yorku.ca/~amana/research/grid.pdf
    // https://web.archive.org/web/20100616193049/http://www.devmaster.net/articles/raytracing_series/A%20faster%20voxel%20traversal%20algorithm%20for%20ray%20tracing.pdf

    // NOTES:
    // * This code assumes that the ray's position and direction are in 'cell coordinates', which means
    //   that one unit equals one cell in all directions.
    // * When the ray doesn't start within the voxel grid, calculate the first position at which the
    //   ray could enter the grid. If it never enters the grid, there is nothing more to do here.
    // * Also, it is important to test when the ray exits the voxel grid when the grid isn't infinite.
    // * The Point3D structure is a simple structure having three integer fields (X, Y and Z).

    // The cell in which the ray starts.
    Point3D start = GetCellAt(ray.Position);        // Rounds the position's X, Y and Z down to the nearest integer values.
    int x = start.X;
    int y = start.Y;
    int z = start.Z;

    // Determine which way we go.
    int stepX = Math.Sign(ray.Direction.X);
    int stepY = Math.Sign(ray.Direction.Y);
    int stepZ = Math.Sign(ray.Direction.Z);

    // Calculate cell boundaries. When the step (i.e. direction sign) is positive,
    // the next boundary is AFTER our current position, meaning that we have to add 1.
    // Otherwise, it is BEFORE our current position, in which case we add nothing.
    Point3D cellBoundary = new Point3D(
	x + (stepX > 0 ? 1 : 0),
	y + (stepY > 0 ? 1 : 0),
	z + (stepZ > 0 ? 1 : 0));

    // NOTE: For the following calculations, the result will be Single.PositiveInfinity
    // when ray.Direction.X, Y or Z equals zero, which is OK. However, when the left-hand
    // value of the division also equals zero, the result is Single.NaN, which is not OK.

    // Determine how far we can travel along the ray before we hit a voxel boundary.
    Vector3 tMax = new Vector3(
	(cellBoundary.X - ray.Position.X) / ray.Direction.X,    // Boundary is a plane on the YZ axis.
	(cellBoundary.Y - ray.Position.Y) / ray.Direction.Y,    // Boundary is a plane on the XZ axis.
	(cellBoundary.Z - ray.Position.Z) / ray.Direction.Z);    // Boundary is a plane on the XY axis.
    if (Single.IsNaN(tMax.X)) tMax.X = Single.PositiveInfinity;
    if (Single.IsNaN(tMax.Y)) tMax.Y = Single.PositiveInfinity;
    if (Single.IsNaN(tMax.Z)) tMax.Z = Single.PositiveInfinity;

    // Determine how far we must travel along the ray before we have crossed a gridcell.
    Vector3 tDelta = new Vector3(
	stepX / ray.Direction.X,                    // Crossing the width of a cell.
	stepY / ray.Direction.Y,                    // Crossing the height of a cell.
	stepZ / ray.Direction.Z);                    // Crossing the depth of a cell.
    if (Single.IsNaN(tDelta.X)) tDelta.X = Single.PositiveInfinity;
    if (Single.IsNaN(tDelta.Y)) tDelta.Y = Single.PositiveInfinity;
    if (Single.IsNaN(tDelta.Z)) tDelta.Z = Single.PositiveInfinity;

    // For each step, determine which distance to the next voxel boundary is lowest (i.e.
    // which voxel boundary is nearest) and walk that way.
    for (int i = 0; i < maxDepth; i++)
    {
	// Return it.
	yield return new Point3D(x, y, z);

	// Do the next step.
	if (tMax.X < tMax.Y && tMax.X < tMax.Z)
	{
	    // tMax.X is the lowest, an YZ cell boundary plane is nearest.
	    x += stepX;
	    tMax.X += tDelta.X;
	}
	else if (tMax.Y < tMax.Z)
	{
	    // tMax.Y is the lowest, an XZ cell boundary plane is nearest.
	    y += stepY;
	    tMax.Y += tDelta.Y;
	}
	else
	{
	    // tMax.Z is the lowest, an XY cell boundary plane is nearest.
	    z += stepZ;
	    tMax.Z += tDelta.Z;
	}
    }

    */
