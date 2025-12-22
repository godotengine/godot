/**************************************************************************/
/*  nav_mesh_generator_3d.cpp                                             */
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

#include "nav_mesh_generator_3d.h"

#include "core/config/project_settings.h"
#include "core/math/math_defs.h"
#include "core/os/thread.h"
#include "core/string/print_string.h"
#include "scene/3d/node_3d.h"
#include "scene/resources/3d/navigation_mesh_source_geometry_data_3d.h"
#include "scene/resources/navigation_mesh.h"

#include <Recast.h>
#include <cfloat>

NavMeshGenerator3D *NavMeshGenerator3D::singleton = nullptr;
Mutex NavMeshGenerator3D::baking_navmesh_mutex;
Mutex NavMeshGenerator3D::generator_task_mutex;
RWLock NavMeshGenerator3D::generator_parsers_rwlock;
bool NavMeshGenerator3D::use_threads = true;
bool NavMeshGenerator3D::baking_use_multiple_threads = true;
bool NavMeshGenerator3D::baking_use_high_priority_threads = true;
HashMap<Ref<NavigationMesh>, NavMeshGenerator3D::NavMeshGeneratorTask3D *> NavMeshGenerator3D::baking_navmeshes;
HashMap<WorkerThreadPool::TaskID, NavMeshGenerator3D::NavMeshGeneratorTask3D *> NavMeshGenerator3D::generator_tasks;
LocalVector<NavMeshGeometryParser3D *> NavMeshGenerator3D::generator_parsers;

static const char *_navmesh_bake_state_msgs[(size_t)NavMeshGenerator3D::NavMeshBakeState::BAKE_STATE_MAX] = {
	"",
	"Setting up configuration...",
	"Calculating grid size...",
	"Creating heightfield...",
	"Marking walkable triangles...",
	"Constructing compact heightfield...", // step 5
	"Eroding walkable area...",
	"Sample partitioning...",
	"Creating contours...",
	"Creating polymesh...",
	"Converting to native navigation mesh...", // step 10
	"Baking cleanup...",
	"Baking finished.",
};

namespace {

static constexpr float NAVMESH_TILE_SIZE_DEFAULT_METERS = 30.0f;
static constexpr float NAVMESH_TILE_SIZE_MIN_METERS = 5.0f;
static constexpr float NAVMESH_TILE_SIZE_MAX_METERS = 200.0f;
static constexpr float NAVMESH_TILE_MIN_BORDER_METERS = 1.0f;

struct TileCoreClip3D {
	bool enabled = false;
	Vector3 min;
	Vector3 max;
};

struct TileDefinition3D {
	AABB bake_bounds;
	TileCoreClip3D clip;
};

struct NavMeshBuildAccumulator3D {
	HashMap<Vector3, int> vertex_lookup;
	Vector<Vector3> vertices;
	Vector<Vector<int>> polygons;

	int get_or_add_vertex(const Vector3 &p_vertex) {
		int *existing_index = vertex_lookup.getptr(p_vertex);
		if (existing_index) {
			return *existing_index;
		}
		const int new_index = vertices.size();
		vertex_lookup[p_vertex] = new_index;
		vertices.push_back(p_vertex);
		return new_index;
	}
};

static bool navmesh_polygon_in_core(const Vector<Vector3> &p_tile_vertices, const Vector<int> &p_polygon, const TileCoreClip3D &p_clip) {
	if (!p_clip.enabled) {
		return true;
	}
	if (p_polygon.is_empty()) {
		return false;
	}

	bool vertex_inside = false;
	float poly_min_x = FLT_MAX;
	float poly_max_x = -FLT_MAX;
	float poly_min_z = FLT_MAX;
	float poly_max_z = -FLT_MAX;

	for (int i = 0; i < p_polygon.size(); i++) {
		const int vertex_index = p_polygon[i];
		if (vertex_index < 0 || vertex_index >= p_tile_vertices.size()) {
			return false;
		}
		const Vector3 &vertex = p_tile_vertices[vertex_index];
		if (vertex.x >= p_clip.min.x - CMP_EPSILON && vertex.x <= p_clip.max.x + CMP_EPSILON &&
				vertex.z >= p_clip.min.z - CMP_EPSILON && vertex.z <= p_clip.max.z + CMP_EPSILON) {
			vertex_inside = true;
		}
		poly_min_x = MIN(poly_min_x, vertex.x);
		poly_max_x = MAX(poly_max_x, vertex.x);
		poly_min_z = MIN(poly_min_z, vertex.z);
		poly_max_z = MAX(poly_max_z, vertex.z);
	}

	if (vertex_inside) {
		return true;
	}

	return poly_max_x >= p_clip.min.x - CMP_EPSILON && poly_min_x <= p_clip.max.x + CMP_EPSILON &&
			poly_max_z >= p_clip.min.z - CMP_EPSILON && poly_min_z <= p_clip.max.z + CMP_EPSILON;
}

static void clamp_tile_vertices_to_clip(Vector<Vector3> &p_vertices, const TileCoreClip3D &p_clip, float p_cell_size) {
	if (!p_clip.enabled || p_vertices.is_empty()) {
		return;
	}
	const float tolerance = MAX(p_cell_size * 0.5f, 0.05f);
	for (int i = 0; i < p_vertices.size(); i++) {
		Vector3 vertex = p_vertices[i];
		if (vertex.x < p_clip.min.x && vertex.x > p_clip.min.x - tolerance) {
			vertex.x = p_clip.min.x;
		} else if (vertex.x > p_clip.max.x && vertex.x < p_clip.max.x + tolerance) {
			vertex.x = p_clip.max.x;
		}

		if (vertex.z < p_clip.min.z && vertex.z > p_clip.min.z - tolerance) {
			vertex.z = p_clip.min.z;
		} else if (vertex.z > p_clip.max.z && vertex.z < p_clip.max.z + tolerance) {
			vertex.z = p_clip.max.z;
		}
		p_vertices.write[i] = vertex;
	}
}

static void append_tile_to_accumulator(Vector<Vector3> &p_tile_vertices, const Vector<Vector<int>> &p_tile_polygons, const TileCoreClip3D &p_clip, float p_cell_size, NavMeshBuildAccumulator3D &r_accumulator) {
	if (p_tile_vertices.is_empty()) {
		return;
	}

	clamp_tile_vertices_to_clip(p_tile_vertices, p_clip, p_cell_size);

	Vector<int> remap;
	remap.resize(p_tile_vertices.size());
	for (int i = 0; i < p_tile_vertices.size(); i++) {
		remap.write[i] = r_accumulator.get_or_add_vertex(p_tile_vertices[i]);
	}

	for (int polygon_index = 0; polygon_index < p_tile_polygons.size(); polygon_index++) {
		const Vector<int> &tile_polygon = p_tile_polygons[polygon_index];
		if (tile_polygon.size() < 3) {
			continue;
		}
		if (!navmesh_polygon_in_core(p_tile_vertices, tile_polygon, p_clip)) {
			continue;
		}

		bool polygon_valid = true;
		Vector<int> remapped_polygon = tile_polygon;
		for (int corner = 0; corner < tile_polygon.size(); corner++) {
			const int tile_vertex_index = tile_polygon[corner];
			if (tile_vertex_index < 0 || tile_vertex_index >= remap.size()) {
				polygon_valid = false;
				break;
			}
			remapped_polygon.write[corner] = remap[tile_vertex_index];
		}
		if (!polygon_valid) {
			continue;
		}
		r_accumulator.polygons.push_back(remapped_polygon);
	}
}

static void compute_tile_definitions(const AABB &p_baking_bounds, float p_cell_size, int p_width_cells, int p_height_cells, int p_walkable_radius, int p_minimum_overlap_voxels, bool p_enable_tiling, float p_tile_size_meters, LocalVector<TileDefinition3D> &r_tiles) {
	r_tiles.clear();

	if (!p_enable_tiling) {
		TileDefinition3D tile;
		tile.bake_bounds = p_baking_bounds;
		tile.clip.enabled = false;
		r_tiles.push_back(tile);
		return;
	}

	const float tile_core_size = MAX(p_tile_size_meters, p_cell_size);
	const int tile_core_span_x = MAX(1, (int)Math::ceil(tile_core_size / p_cell_size));
	const int tile_core_span_z = MAX(1, (int)Math::ceil(tile_core_size / p_cell_size));

	const int desired_overlap = MAX(MAX(p_walkable_radius + 2, 4), p_minimum_overlap_voxels);
	const int overlap_x = MIN(desired_overlap, MAX(1, tile_core_span_x / 2));
	const int overlap_z = MIN(desired_overlap, MAX(1, tile_core_span_z / 2));

	const int tile_step_x = MAX(1, tile_core_span_x);
	const int tile_step_z = MAX(1, tile_core_span_z);

	for (int core_min_z = 0; core_min_z < p_height_cells; core_min_z += tile_step_z) {
		const int core_max_z = MIN(p_height_cells, core_min_z + tile_core_span_z);
		const int tile_min_z = MAX(0, core_min_z - overlap_z);
		const int tile_max_z = MIN(p_height_cells, core_max_z + overlap_z);

		for (int core_min_x = 0; core_min_x < p_width_cells; core_min_x += tile_step_x) {
			const int core_max_x = MIN(p_width_cells, core_min_x + tile_core_span_x);
			const int tile_min_x = MAX(0, core_min_x - overlap_x);
			const int tile_max_x = MIN(p_width_cells, core_max_x + overlap_x);

			if (tile_max_x <= tile_min_x || tile_max_z <= tile_min_z) {
				continue;
			}

			TileDefinition3D tile;
			tile.bake_bounds.position = Vector3(
					p_baking_bounds.position.x + tile_min_x * p_cell_size,
					p_baking_bounds.position.y,
					p_baking_bounds.position.z + tile_min_z * p_cell_size);
			tile.bake_bounds.size = Vector3(
					(tile_max_x - tile_min_x) * p_cell_size,
					p_baking_bounds.size.y,
					(tile_max_z - tile_min_z) * p_cell_size);

			tile.clip.enabled = true;
			tile.clip.min = Vector3(
					p_baking_bounds.position.x + core_min_x * p_cell_size,
					p_baking_bounds.position.y,
					p_baking_bounds.position.z + core_min_z * p_cell_size);
			tile.clip.max = Vector3(
					p_baking_bounds.position.x + core_max_x * p_cell_size,
					p_baking_bounds.position.y + p_baking_bounds.size.y,
					p_baking_bounds.position.z + core_max_z * p_cell_size);

			r_tiles.push_back(tile);
		}
	}
}

} // namespace

NavMeshGenerator3D *NavMeshGenerator3D::get_singleton() {
	return singleton;
}

NavMeshGenerator3D::NavMeshGenerator3D() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;

	baking_use_multiple_threads = GLOBAL_GET("navigation/baking/thread_model/baking_use_multiple_threads");
	baking_use_high_priority_threads = GLOBAL_GET("navigation/baking/thread_model/baking_use_high_priority_threads");

	// Using threads might cause problems on certain exports or with the Editor on certain devices.
	// This is the main switch to turn threaded navmesh baking off should the need arise.
	use_threads = baking_use_multiple_threads;
}

NavMeshGenerator3D::~NavMeshGenerator3D() {
	cleanup();
}

void NavMeshGenerator3D::sync() {
	if (generator_tasks.is_empty()) {
		return;
	}

	MutexLock baking_navmesh_lock(baking_navmesh_mutex);
	{
		MutexLock generator_task_lock(generator_task_mutex);

		LocalVector<WorkerThreadPool::TaskID> finished_task_ids;

		for (KeyValue<WorkerThreadPool::TaskID, NavMeshGeneratorTask3D *> &E : generator_tasks) {
			if (WorkerThreadPool::get_singleton()->is_task_completed(E.key)) {
				WorkerThreadPool::get_singleton()->wait_for_task_completion(E.key);
				finished_task_ids.push_back(E.key);

				NavMeshGeneratorTask3D *generator_task = E.value;
				DEV_ASSERT(generator_task->status == NavMeshGeneratorTask3D::TaskStatus::BAKING_FINISHED);

				baking_navmeshes.erase(generator_task->navigation_mesh);
				if (generator_task->callback.is_valid()) {
					generator_emit_callback(generator_task->callback);
				}
				generator_task->navigation_mesh->emit_changed();
				memdelete(generator_task);
			}
		}

		for (WorkerThreadPool::TaskID finished_task_id : finished_task_ids) {
			generator_tasks.erase(finished_task_id);
		}
	}
}

void NavMeshGenerator3D::cleanup() {
	MutexLock baking_navmesh_lock(baking_navmesh_mutex);
	{
		MutexLock generator_task_lock(generator_task_mutex);

		baking_navmeshes.clear();

		for (KeyValue<WorkerThreadPool::TaskID, NavMeshGeneratorTask3D *> &E : generator_tasks) {
			WorkerThreadPool::get_singleton()->wait_for_task_completion(E.key);
			NavMeshGeneratorTask3D *generator_task = E.value;
			memdelete(generator_task);
		}
		generator_tasks.clear();

		generator_parsers_rwlock.write_lock();
		generator_parsers.clear();
		generator_parsers_rwlock.write_unlock();
	}
}

void NavMeshGenerator3D::finish() {
	cleanup();
}

void NavMeshGenerator3D::parse_source_geometry_data(Ref<NavigationMesh> p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, Node *p_root_node, const Callable &p_callback) {
	ERR_FAIL_COND(!Thread::is_main_thread());
	ERR_FAIL_COND(p_navigation_mesh.is_null());
	ERR_FAIL_NULL(p_root_node);
	ERR_FAIL_COND(!p_root_node->is_inside_tree());
	ERR_FAIL_COND(p_source_geometry_data.is_null());

	generator_parse_source_geometry_data(p_navigation_mesh, p_source_geometry_data, p_root_node);

	if (p_callback.is_valid()) {
		generator_emit_callback(p_callback);
	}
}

void NavMeshGenerator3D::bake_from_source_geometry_data(Ref<NavigationMesh> p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, const Callable &p_callback) {
	ERR_FAIL_COND(p_navigation_mesh.is_null());
	ERR_FAIL_COND(p_source_geometry_data.is_null());

	if (!p_source_geometry_data->has_data()) {
		p_navigation_mesh->clear();
		if (p_callback.is_valid()) {
			generator_emit_callback(p_callback);
		}
		p_navigation_mesh->emit_changed();
		return;
	}

	if (is_baking(p_navigation_mesh)) {
		ERR_FAIL_MSG("NavigationMesh is already baking. Wait for current bake to finish.");
	}
	baking_navmesh_mutex.lock();
	NavMeshGeneratorTask3D generator_task;
	baking_navmeshes.insert(p_navigation_mesh, &generator_task);
	baking_navmesh_mutex.unlock();

	generator_task.navigation_mesh = p_navigation_mesh;
	generator_task.source_geometry_data = p_source_geometry_data;
	generator_task.status = NavMeshGeneratorTask3D::TaskStatus::BAKING_STARTED;

	generator_bake_from_source_geometry_data(&generator_task);

	baking_navmesh_mutex.lock();
	baking_navmeshes.erase(p_navigation_mesh);
	baking_navmesh_mutex.unlock();

	if (p_callback.is_valid()) {
		generator_emit_callback(p_callback);
	}

	p_navigation_mesh->emit_changed();
}

void NavMeshGenerator3D::bake_from_source_geometry_data_async(Ref<NavigationMesh> p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, const Callable &p_callback) {
	ERR_FAIL_COND(p_navigation_mesh.is_null());
	ERR_FAIL_COND(p_source_geometry_data.is_null());

	if (!p_source_geometry_data->has_data()) {
		p_navigation_mesh->clear();
		if (p_callback.is_valid()) {
			generator_emit_callback(p_callback);
		}
		p_navigation_mesh->emit_changed();
		return;
	}

	if (!use_threads) {
		bake_from_source_geometry_data(p_navigation_mesh, p_source_geometry_data, p_callback);
		return;
	}

	if (is_baking(p_navigation_mesh)) {
		ERR_FAIL_MSG("NavigationMesh is already baking. Wait for current bake to finish.");
		return;
	}
	baking_navmesh_mutex.lock();
	NavMeshGeneratorTask3D *generator_task = memnew(NavMeshGeneratorTask3D);
	baking_navmeshes.insert(p_navigation_mesh, generator_task);
	baking_navmesh_mutex.unlock();

	generator_task->navigation_mesh = p_navigation_mesh;
	generator_task->source_geometry_data = p_source_geometry_data;
	generator_task->callback = p_callback;
	generator_task->status = NavMeshGeneratorTask3D::TaskStatus::BAKING_STARTED;
	generator_task->thread_task_id = WorkerThreadPool::get_singleton()->add_native_task(&NavMeshGenerator3D::generator_thread_bake, generator_task, NavMeshGenerator3D::baking_use_high_priority_threads, SNAME("NavMeshGeneratorBake3D"));
	MutexLock generator_task_lock(generator_task_mutex);
	generator_tasks.insert(generator_task->thread_task_id, generator_task);
}

bool NavMeshGenerator3D::is_baking(Ref<NavigationMesh> p_navigation_mesh) {
	MutexLock baking_navmesh_lock(baking_navmesh_mutex);
	return baking_navmeshes.has(p_navigation_mesh);
}

String NavMeshGenerator3D::get_baking_state_msg(Ref<NavigationMesh> p_navigation_mesh) {
	String bake_state_msg;
	MutexLock baking_navmesh_lock(baking_navmesh_mutex);
	if (baking_navmeshes.has(p_navigation_mesh)) {
		bake_state_msg = _navmesh_bake_state_msgs[baking_navmeshes[p_navigation_mesh]->bake_state];
	} else {
		bake_state_msg = _navmesh_bake_state_msgs[NavMeshBakeState::BAKE_STATE_NONE];
	}
	return bake_state_msg;
}

void NavMeshGenerator3D::generator_thread_bake(void *p_arg) {
	NavMeshGeneratorTask3D *generator_task = static_cast<NavMeshGeneratorTask3D *>(p_arg);

	generator_bake_from_source_geometry_data(generator_task);

	generator_task->status = NavMeshGeneratorTask3D::TaskStatus::BAKING_FINISHED;
}

void NavMeshGenerator3D::generator_parse_geometry_node(const Ref<NavigationMesh> &p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, Node *p_node, bool p_recurse_children) {
	generator_parsers_rwlock.read_lock();
	for (const NavMeshGeometryParser3D *parser : generator_parsers) {
		if (!parser->callback.is_valid()) {
			continue;
		}
		parser->callback.call(p_navigation_mesh, p_source_geometry_data, p_node);
	}
	generator_parsers_rwlock.read_unlock();

	if (p_recurse_children) {
		for (int i = 0; i < p_node->get_child_count(); i++) {
			generator_parse_geometry_node(p_navigation_mesh, p_source_geometry_data, p_node->get_child(i), p_recurse_children);
		}
	}
}

void NavMeshGenerator3D::set_generator_parsers(LocalVector<NavMeshGeometryParser3D *> p_parsers) {
	RWLockWrite write_lock(generator_parsers_rwlock);
	generator_parsers = p_parsers;
}

void NavMeshGenerator3D::generator_parse_source_geometry_data(const Ref<NavigationMesh> &p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, Node *p_root_node) {
	Vector<Node *> parse_nodes;

	if (p_navigation_mesh->get_source_geometry_mode() == NavigationMesh::SOURCE_GEOMETRY_ROOT_NODE_CHILDREN) {
		parse_nodes.push_back(p_root_node);
	} else {
		parse_nodes = p_root_node->get_tree()->get_nodes_in_group(p_navigation_mesh->get_source_group_name());
	}

	Transform3D root_node_transform = Transform3D();
	if (Object::cast_to<Node3D>(p_root_node)) {
		root_node_transform = Object::cast_to<Node3D>(p_root_node)->get_global_transform().affine_inverse();
	}

	p_source_geometry_data->clear();
	p_source_geometry_data->root_node_transform = root_node_transform;

	bool recurse_children = p_navigation_mesh->get_source_geometry_mode() != NavigationMesh::SOURCE_GEOMETRY_GROUPS_EXPLICIT;

	for (Node *parse_node : parse_nodes) {
		generator_parse_geometry_node(p_navigation_mesh, p_source_geometry_data, parse_node, recurse_children);
	}
}

void NavMeshGenerator3D::generator_bake_from_source_geometry_data(NavMeshGeneratorTask3D *p_generator_task) {
	Ref<NavigationMesh> p_navigation_mesh = p_generator_task->navigation_mesh;
	const Ref<NavigationMeshSourceGeometryData3D> &p_source_geometry_data = p_generator_task->source_geometry_data;

	if (p_navigation_mesh.is_null() || p_source_geometry_data.is_null()) {
		return;
	}

	Vector<float> source_geometry_vertices;
	Vector<int> source_geometry_indices;
	Vector<NavigationMeshSourceGeometryData3D::ProjectedObstruction> projected_obstructions;

	p_source_geometry_data->get_data(
			source_geometry_vertices,
			source_geometry_indices,
			projected_obstructions);

	if (source_geometry_vertices.size() < 3 || source_geometry_indices.size() < 3) {
		return;
	}

	const float *verts = source_geometry_vertices.ptr();
	const int nverts = source_geometry_vertices.size() / 3;

	float raw_bmin[3];
	float raw_bmax[3];
	rcCalcBounds(verts, nverts, raw_bmin, raw_bmax);
	AABB baking_bounds(Vector3(raw_bmin[0], raw_bmin[1], raw_bmin[2]), Vector3(raw_bmax[0] - raw_bmin[0], raw_bmax[1] - raw_bmin[1], raw_bmax[2] - raw_bmin[2]));
	AABB baking_aabb = p_navigation_mesh->get_filter_baking_aabb();
	if (baking_aabb.has_volume()) {
		const Vector3 baking_aabb_offset = p_navigation_mesh->get_filter_baking_aabb_offset();
		baking_bounds.position = baking_aabb.position + baking_aabb_offset;
		baking_bounds.size = baking_aabb.size;
	} else {
	}

	rcConfig bounds_cfg;
	memset(&bounds_cfg, 0, sizeof(bounds_cfg));
	bounds_cfg.cs = p_navigation_mesh->get_cell_size();
	bounds_cfg.ch = p_navigation_mesh->get_cell_height();
	bounds_cfg.bmin[0] = baking_bounds.position.x;
	bounds_cfg.bmin[1] = baking_bounds.position.y;
	bounds_cfg.bmin[2] = baking_bounds.position.z;
	bounds_cfg.bmax[0] = baking_bounds.position.x + baking_bounds.size.x;
	bounds_cfg.bmax[1] = baking_bounds.position.y + baking_bounds.size.y;
	bounds_cfg.bmax[2] = baking_bounds.position.z + baking_bounds.size.z;

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_CALC_GRID_SIZE; // step #2
	rcCalcGridSize(bounds_cfg.bmin, bounds_cfg.bmax, bounds_cfg.cs, &bounds_cfg.width, &bounds_cfg.height);

	if (bounds_cfg.width <= 0 || bounds_cfg.height <= 0) {
		ERR_FAIL_MSG("Baking interrupted.\nNavigationMesh baking bounds are too small for the current configuration.");
	}

	const int walkable_radius = (int)Math::ceil(p_navigation_mesh->get_agent_radius() / bounds_cfg.cs);
	const bool use_tile_baking = p_navigation_mesh->is_tile_baking_enabled();
	float requested_tile_size = p_navigation_mesh->get_tile_baking_size();
	if (!use_tile_baking || requested_tile_size <= 0.0f) {
		requested_tile_size = NAVMESH_TILE_SIZE_DEFAULT_METERS;
	}
	requested_tile_size = CLAMP(requested_tile_size, NAVMESH_TILE_SIZE_MIN_METERS, NAVMESH_TILE_SIZE_MAX_METERS);

	int minimum_overlap_voxels = walkable_radius + 2;
	if (use_tile_baking && bounds_cfg.cs > 0.0f) {
		float effective_border_size = p_navigation_mesh->get_border_size();
		const float recommended_border = (walkable_radius + 3.0f) * bounds_cfg.cs;
		effective_border_size = MAX(effective_border_size, recommended_border);
		effective_border_size = MAX(effective_border_size, NAVMESH_TILE_MIN_BORDER_METERS);
		const int border_voxels = (int)Math::ceil(effective_border_size / bounds_cfg.cs);
		minimum_overlap_voxels = MAX(minimum_overlap_voxels, border_voxels);
	}
	minimum_overlap_voxels = MAX(minimum_overlap_voxels, 4);

	LocalVector<TileDefinition3D> tiles;
	compute_tile_definitions(baking_bounds, bounds_cfg.cs, bounds_cfg.width, bounds_cfg.height, walkable_radius, minimum_overlap_voxels, use_tile_baking, requested_tile_size, tiles);

	const int tile_count = tiles.size();
	if (tile_count == 0) {
		WARN_PRINT("NavigationMesh baking aborted because no tiles were generated.");
		return;
	}

	NavMeshBuildAccumulator3D build_accumulator;
	for (int tile_index = 0; tile_index < tile_count; tile_index++) {
		Vector<Vector3> tile_vertices;
		Vector<Vector<int>> tile_polygons;
		const Error tile_err = generator_bake_single_tile(
				p_generator_task,
				source_geometry_vertices,
				source_geometry_indices,
				projected_obstructions,
				tiles[tile_index].bake_bounds,
				tile_vertices,
				tile_polygons,
				tile_index,
				tile_count);

		if (tile_err != OK) {
			return;
		}

		append_tile_to_accumulator(tile_vertices, tile_polygons, tiles[tile_index].clip, p_navigation_mesh->get_cell_size(), build_accumulator);
	}

	p_navigation_mesh->set_data(build_accumulator.vertices, build_accumulator.polygons);

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_BAKE_FINISHED; // step #12
}

Error NavMeshGenerator3D::generator_bake_single_tile(NavMeshGeneratorTask3D *p_generator_task, const Vector<float> &p_source_geometry_vertices, const Vector<int> &p_source_geometry_indices, const Vector<NavigationMeshSourceGeometryData3D::ProjectedObstruction> &p_projected_obstructions, const AABB &p_bake_bounds, Vector<Vector3> &r_nav_vertices, Vector<Vector<int>> &r_nav_polygons, int p_tile_index, int p_tile_count) {
	ERR_FAIL_COND_V(p_generator_task == nullptr, ERR_INVALID_PARAMETER);
	Ref<NavigationMesh> navigation_mesh = p_generator_task->navigation_mesh;
	ERR_FAIL_COND_V(navigation_mesh.is_null(), ERR_INVALID_PARAMETER);

	Error err = OK;

	rcHeightfield *hf = nullptr;
	rcCompactHeightfield *chf = nullptr;
	rcContourSet *cset = nullptr;
	rcPolyMesh *poly_mesh = nullptr;
	rcPolyMeshDetail *detail_mesh = nullptr;
	rcContext ctx;

	Vector<Vector3> nav_vertices;
	Vector<Vector<int>> nav_polygons;

	HashMap<Vector3, int> recast_vertex_to_native_index;
	LocalVector<int> recast_index_to_native_index;

	const float *verts = p_source_geometry_vertices.ptr();
	const int *tris = p_source_geometry_indices.ptr();
	const int nverts = p_source_geometry_vertices.size() / 3;
	const int ntris = p_source_geometry_indices.size() / 3;

	rcConfig cfg;
	memset(&cfg, 0, sizeof(cfg));

	cfg.cs = navigation_mesh->get_cell_size();
	cfg.ch = navigation_mesh->get_cell_height();
	float effective_border_size = navigation_mesh->get_border_size();
	if (navigation_mesh->is_tile_baking_enabled()) {
		// Ensure tiles have enough overlap after Recast erodes by the agent radius.
		const float cell_size = navigation_mesh->get_cell_size();
		if (cell_size > 0.0f) {
			const float walkable_radius_voxels = Math::ceil(navigation_mesh->get_agent_radius() / cell_size);
			const float recommended_border = (walkable_radius_voxels + 3.0f) * cell_size;
			effective_border_size = MAX(effective_border_size, recommended_border);
		}
		effective_border_size = MAX(effective_border_size, NAVMESH_TILE_MIN_BORDER_METERS);
	}
	if (effective_border_size > 0.0f) {
		cfg.borderSize = (int)Math::ceil(effective_border_size / cfg.cs);
	}
	cfg.walkableSlopeAngle = navigation_mesh->get_agent_max_slope();
	cfg.walkableHeight = (int)Math::ceil(navigation_mesh->get_agent_height() / cfg.ch);
	cfg.walkableClimb = (int)Math::floor(navigation_mesh->get_agent_max_climb() / cfg.ch);
	cfg.walkableRadius = (int)Math::ceil(navigation_mesh->get_agent_radius() / cfg.cs);
	cfg.maxEdgeLen = (int)(navigation_mesh->get_edge_max_length() / navigation_mesh->get_cell_size());
	cfg.maxSimplificationError = navigation_mesh->get_edge_max_error();
	cfg.minRegionArea = (int)(navigation_mesh->get_region_min_size() * navigation_mesh->get_region_min_size());
	cfg.mergeRegionArea = (int)(navigation_mesh->get_region_merge_size() * navigation_mesh->get_region_merge_size());
	cfg.maxVertsPerPoly = (int)navigation_mesh->get_vertices_per_polygon();
	cfg.detailSampleDist = MAX(navigation_mesh->get_cell_size() * navigation_mesh->get_detail_sample_distance(), 0.1f);
	cfg.detailSampleMaxError = navigation_mesh->get_cell_height() * navigation_mesh->get_detail_sample_max_error();

	if (navigation_mesh->get_border_size() > 0.0 && !Math::is_zero_approx(Math::fmod(navigation_mesh->get_border_size(), navigation_mesh->get_cell_size()))) {
		WARN_PRINT("Property border_size is ceiled to cell_size voxel units and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.walkableHeight * cfg.ch, navigation_mesh->get_agent_height())) {
		WARN_PRINT("Property agent_height is ceiled to cell_height voxel units and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.walkableClimb * cfg.ch, navigation_mesh->get_agent_max_climb())) {
		WARN_PRINT("Property agent_max_climb is floored to cell_height voxel units and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.walkableRadius * cfg.cs, navigation_mesh->get_agent_radius())) {
		WARN_PRINT("Property agent_radius is ceiled to cell_size voxel units and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.maxEdgeLen * cfg.cs, navigation_mesh->get_edge_max_length())) {
		WARN_PRINT("Property edge_max_length is rounded to cell_size voxel units and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.minRegionArea, navigation_mesh->get_region_min_size() * navigation_mesh->get_region_min_size())) {
		WARN_PRINT("Property region_min_size is converted to int and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.mergeRegionArea, navigation_mesh->get_region_merge_size() * navigation_mesh->get_region_merge_size())) {
		WARN_PRINT("Property region_merge_size is converted to int and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.maxVertsPerPoly, navigation_mesh->get_vertices_per_polygon())) {
		WARN_PRINT("Property vertices_per_polygon is converted to int and loses precision.");
	}
	if (navigation_mesh->get_cell_size() * navigation_mesh->get_detail_sample_distance() < 0.1f) {
		WARN_PRINT("Property detail_sample_distance is clamped to 0.1 world units as the resulting value from multiplying with cell_size is too low.");
	}

	cfg.bmin[0] = p_bake_bounds.position.x;
	cfg.bmin[1] = p_bake_bounds.position.y;
	cfg.bmin[2] = p_bake_bounds.position.z;
	cfg.bmax[0] = p_bake_bounds.position.x + p_bake_bounds.size.x;
	cfg.bmax[1] = p_bake_bounds.position.y + p_bake_bounds.size.y;
	cfg.bmax[2] = p_bake_bounds.position.z + p_bake_bounds.size.z;

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_CONFIGURATION; // step #1

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_CALC_GRID_SIZE; // step #2
	rcCalcGridSize(cfg.bmin, cfg.bmax, cfg.cs, &cfg.width, &cfg.height);

	if ((cfg.width * cfg.height) > 30000000 && GLOBAL_GET("navigation/baking/use_crash_prevention_checks")) {
		ERR_PRINT("NavigationMesh baking tile exceeds safety threshold. Increase cell_size / cell_height or disable crash prevention checks to continue.");
		err = ERR_CANT_CREATE;
		goto tile_cleanup;
	}

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_CREATE_HEIGHTFIELD; // step #3
	hf = rcAllocHeightfield();
	if (!hf) {
		err = ERR_OUT_OF_MEMORY;
		goto tile_cleanup;
	}
	if (!rcCreateHeightfield(&ctx, *hf, cfg.width, cfg.height, cfg.bmin, cfg.bmax, cfg.cs, cfg.ch)) {
		err = ERR_CANT_CREATE;
		goto tile_cleanup;
	}

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_MARK_WALKABLE_TRIANGLES; // step #4
	{
		Vector<unsigned char> tri_areas;
		tri_areas.resize(ntris);

		if (tri_areas.is_empty()) {
			err = ERR_CANT_CREATE;
			goto tile_cleanup;
		}

		memset(tri_areas.ptrw(), 0, ntris * sizeof(unsigned char));
		rcMarkWalkableTriangles(&ctx, cfg.walkableSlopeAngle, verts, nverts, tris, ntris, tri_areas.ptrw());

		if (!rcRasterizeTriangles(&ctx, verts, nverts, tris, tri_areas.ptr(), ntris, *hf, cfg.walkableClimb)) {
			err = ERR_CANT_CREATE;
			goto tile_cleanup;
		}
	}

	if (navigation_mesh->get_filter_low_hanging_obstacles()) {
		rcFilterLowHangingWalkableObstacles(&ctx, cfg.walkableClimb, *hf);
	}
	if (navigation_mesh->get_filter_ledge_spans()) {
		rcFilterLedgeSpans(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf);
	}
	if (navigation_mesh->get_filter_walkable_low_height_spans()) {
		rcFilterWalkableLowHeightSpans(&ctx, cfg.walkableHeight, *hf);
	}

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_CONSTRUCT_COMPACT_HEIGHTFIELD; // step #5

	chf = rcAllocCompactHeightfield();
	if (!chf) {
		err = ERR_OUT_OF_MEMORY;
		goto tile_cleanup;
	}
	if (!rcBuildCompactHeightfield(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf, *chf)) {
		err = ERR_CANT_CREATE;
		goto tile_cleanup;
	}

	rcFreeHeightField(hf);
	hf = nullptr;

	if (!p_projected_obstructions.is_empty()) {
		for (const NavigationMeshSourceGeometryData3D::ProjectedObstruction &projected_obstruction : p_projected_obstructions) {
			if (projected_obstruction.carve) {
				continue;
			}
			if (projected_obstruction.vertices.is_empty() || projected_obstruction.vertices.size() % 3 != 0) {
				continue;
			}

			const float *projected_obstruction_verts = projected_obstruction.vertices.ptr();
			const int projected_obstruction_nverts = projected_obstruction.vertices.size() / 3;

			rcMarkConvexPolyArea(&ctx, projected_obstruction_verts, projected_obstruction_nverts, projected_obstruction.elevation, projected_obstruction.elevation + projected_obstruction.height, RC_NULL_AREA, *chf);
		}
	}

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_ERODE_WALKABLE_AREA; // step #6

	if (!rcErodeWalkableArea(&ctx, cfg.walkableRadius, *chf)) {
		err = ERR_CANT_CREATE;
		goto tile_cleanup;
	}

	if (!p_projected_obstructions.is_empty()) {
		for (const NavigationMeshSourceGeometryData3D::ProjectedObstruction &projected_obstruction : p_projected_obstructions) {
			if (!projected_obstruction.carve) {
				continue;
			}
			if (projected_obstruction.vertices.is_empty() || projected_obstruction.vertices.size() % 3 != 0) {
				continue;
			}

			const float *projected_obstruction_verts = projected_obstruction.vertices.ptr();
			const int projected_obstruction_nverts = projected_obstruction.vertices.size() / 3;

			rcMarkConvexPolyArea(&ctx, projected_obstruction_verts, projected_obstruction_nverts, projected_obstruction.elevation, projected_obstruction.elevation + projected_obstruction.height, RC_NULL_AREA, *chf);
		}
	}

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_SAMPLE_PARTITIONING; // step #7

	if (navigation_mesh->get_sample_partition_type() == NavigationMesh::SAMPLE_PARTITION_WATERSHED) {
		if (!rcBuildDistanceField(&ctx, *chf)) {
			err = ERR_CANT_CREATE;
			goto tile_cleanup;
		}
		if (!rcBuildRegions(&ctx, *chf, cfg.borderSize, cfg.minRegionArea, cfg.mergeRegionArea)) {
			err = ERR_CANT_CREATE;
			goto tile_cleanup;
		}
	} else if (navigation_mesh->get_sample_partition_type() == NavigationMesh::SAMPLE_PARTITION_MONOTONE) {
		if (!rcBuildRegionsMonotone(&ctx, *chf, cfg.borderSize, cfg.minRegionArea, cfg.mergeRegionArea)) {
			err = ERR_CANT_CREATE;
			goto tile_cleanup;
		}
	} else {
		if (!rcBuildLayerRegions(&ctx, *chf, cfg.borderSize, cfg.minRegionArea)) {
			err = ERR_CANT_CREATE;
			goto tile_cleanup;
		}
	}

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_CREATING_CONTOURS; // step #8

	cset = rcAllocContourSet();
	if (!cset) {
		err = ERR_OUT_OF_MEMORY;
		goto tile_cleanup;
	}
	if (!rcBuildContours(&ctx, *chf, cfg.maxSimplificationError, cfg.maxEdgeLen, *cset)) {
		err = ERR_CANT_CREATE;
		goto tile_cleanup;
	}

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_CREATING_POLYMESH; // step #9

	poly_mesh = rcAllocPolyMesh();
	if (!poly_mesh) {
		err = ERR_OUT_OF_MEMORY;
		goto tile_cleanup;
	}
	if (!rcBuildPolyMesh(&ctx, *cset, cfg.maxVertsPerPoly, *poly_mesh)) {
		err = ERR_CANT_CREATE;
		goto tile_cleanup;
	}

	detail_mesh = rcAllocPolyMeshDetail();
	if (!detail_mesh) {
		err = ERR_OUT_OF_MEMORY;
		goto tile_cleanup;
	}
	if (!rcBuildPolyMeshDetail(&ctx, *poly_mesh, *chf, cfg.detailSampleDist, cfg.detailSampleMaxError, *detail_mesh)) {
		err = ERR_CANT_CREATE;
		goto tile_cleanup;
	}

	rcFreeCompactHeightfield(chf);
	chf = nullptr;
	rcFreeContourSet(cset);
	cset = nullptr;

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_CONVERTING_NATIVE_NAVMESH; // step #10

	recast_index_to_native_index.resize(detail_mesh->nverts);

	for (int i = 0; i < detail_mesh->nverts; i++) {
		const float *v = &detail_mesh->verts[i * 3];
		const Vector3 vertex = Vector3(v[0], v[1], v[2]);
		int *existing_index_ptr = recast_vertex_to_native_index.getptr(vertex);
		if (!existing_index_ptr) {
			int new_index = recast_vertex_to_native_index.size();
			recast_index_to_native_index[i] = new_index;
			recast_vertex_to_native_index[vertex] = new_index;
			nav_vertices.push_back(vertex);
		} else {
			recast_index_to_native_index[i] = *existing_index_ptr;
		}
	}

	for (int i = 0; i < detail_mesh->nmeshes; i++) {
		const unsigned int *detail_mesh_m = &detail_mesh->meshes[i * 4];
		const unsigned int detail_mesh_bverts = detail_mesh_m[0];
		const unsigned int detail_mesh_m_btris = detail_mesh_m[2];
		const unsigned int detail_mesh_ntris = detail_mesh_m[3];
		const unsigned char *detail_mesh_tris = &detail_mesh->tris[detail_mesh_m_btris * 4];
		for (unsigned int j = 0; j < detail_mesh_ntris; j++) {
			Vector<int> nav_indices;
			nav_indices.resize(3);
			const int index1 = ((int)(detail_mesh_bverts + detail_mesh_tris[j * 4 + 0]));
			const int index2 = ((int)(detail_mesh_bverts + detail_mesh_tris[j * 4 + 2]));
			const int index3 = ((int)(detail_mesh_bverts + detail_mesh_tris[j * 4 + 1]));

			nav_indices.write[0] = recast_index_to_native_index[index1];
			nav_indices.write[1] = recast_index_to_native_index[index2];
			nav_indices.write[2] = recast_index_to_native_index[index3];

			nav_polygons.push_back(nav_indices);
		}
	}

	r_nav_vertices = nav_vertices;
	r_nav_polygons = nav_polygons;

	p_generator_task->bake_state = NavMeshBakeState::BAKE_STATE_BAKE_CLEANUP; // step #11

tile_cleanup:
	if (poly_mesh) {
		rcFreePolyMesh(poly_mesh);
		poly_mesh = nullptr;
	}
	if (detail_mesh) {
		rcFreePolyMeshDetail(detail_mesh);
		detail_mesh = nullptr;
	}
	if (chf) {
		rcFreeCompactHeightfield(chf);
		chf = nullptr;
	}
	if (cset) {
		rcFreeContourSet(cset);
		cset = nullptr;
	}
	if (hf) {
		rcFreeHeightField(hf);
		hf = nullptr;
	}

	return err;
}

bool NavMeshGenerator3D::generator_emit_callback(const Callable &p_callback) {
	ERR_FAIL_COND_V(!p_callback.is_valid(), false);

	Callable::CallError ce;
	Variant result;
	p_callback.callp(nullptr, 0, result, ce);

	return ce.error == Callable::CallError::CALL_OK;
}
