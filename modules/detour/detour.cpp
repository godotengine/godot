#include "detour.h"
#include "modules/csg/csg_shape.h"
#include "obstacle.h"
#include "scene/3d/mesh_instance.h"
#include <DetourNavMesh.h>
#include <DetourNavMeshBuilder.h>
#include <DetourNavMeshQuery.h>
#include <DetourTileCache.h>
#include <DetourTileCacheBuilder.h>
#include <Recast.h>

inline unsigned int nextPow2(unsigned int v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

inline unsigned int ilog2(unsigned int v) {
	unsigned int r;
	unsigned int shift;
	r = (v > 0xffff) << 4;
	v >>= r;
	shift = (v > 0xff) << 3;
	v >>= shift;
	r |= shift;
	shift = (v > 0xf) << 2;
	v >>= shift;
	r |= shift;
	shift = (v > 0x3) << 1;
	v >>= shift;
	r |= shift;
	r |= (v >> 1);
	return r;
}

void DetourNavigationMeshInstance::build() {
	if (geometries.size() == 0)
		return;
	if (!mesh.is_valid())
		return;
	print_line("Building");
	for (int i = 0; i < geometries.size(); i++)
		if (geometries[i].is_valid()) {
			AABB convbox = geometries[i]->get_aabb();
			convbox = xforms[i].xform(convbox);
			mesh->bounding_box.merge_with(convbox);
		}
	print_line("mesh bb: " + String(mesh->bounding_box));
	mesh->bounding_box.position -= mesh->padding;
	mesh->bounding_box.size += mesh->padding * 2.0;
	int gridH = 0, gridW = 0;
	float tile_edge_length = mesh->get_tile_edge_length();
	Vector3 bmin = mesh->bounding_box.position;
	Vector3 bmax = mesh->bounding_box.position + mesh->bounding_box.size;
	rcCalcGridSize(&bmin.coord[0], &bmax.coord[0], mesh->cell_size, &gridW,
			&gridH);
	mesh->set_num_tiles(gridW, gridH);
	print_line(String() + "tiles x: " + itos(mesh->get_num_tiles_x()) +
			   " tiles z: " + itos(mesh->get_num_tiles_z()));
	unsigned int tile_bits = (unsigned int)ilog2(
			nextPow2(mesh->get_num_tiles_x() * mesh->get_num_tiles_z()));
	if (tile_bits > 14)
		tile_bits = 14;
	unsigned int poly_bits = 22 - tile_bits;
	unsigned int max_tiles = 1u << tile_bits;
	unsigned int max_polys = 1 << poly_bits;
	dtNavMeshParams params;
	rcVcopy(params.orig, &bmin.coord[0]);
	params.tileWidth = tile_edge_length;
	params.tileHeight = tile_edge_length;
	params.maxTiles = max_tiles;
	params.maxPolys = max_polys;
	if (!mesh->alloc())
		return;
	if (!mesh->init(&params))
		return;
#ifdef TILE_CACHE
	dtTileCacheParams tile_cache_params;
	memset(&tile_cache_params, 0, sizeof(tile_cache_params));
	rcVcopy(tile_cache_params.orig, &bmin.coord[0]);
	tile_cache_params.ch = mesh->cell_height;
	tile_cache_params.cs = mesh->cell_size;
	tile_cache_params.width = mesh->tile_size;
	tile_cache_params.height = mesh->tile_size;
	tile_cache_params.maxSimplificationError = mesh->edge_max_error;
	tile_cache_params.maxTiles =
			mesh->get_num_tiles_x() * mesh->get_num_tiles_z() * mesh->max_layers;
	tile_cache_params.maxObstacles = mesh->max_obstacles;
	tile_cache_params.walkableClimb = mesh->agent_max_climb;
	tile_cache_params.walkableHeight = mesh->agent_height;
	tile_cache_params.walkableRadius = mesh->agent_radius;
	if (!mesh->alloc_tile_cache())
		return;
	if (!mesh->init_tile_cache(&tile_cache_params))
		return;
#endif
	Transform xform = get_global_transform();
	unsigned int result = mesh->build_tiles(xform, geometries, xforms, 0, 0,
			mesh->get_num_tiles_x() - 1,
			mesh->get_num_tiles_z() - 1);
	print_line(String() + "built tiles: " + itos(result));
	print_line("mesh final bb: " + String(mesh->bounding_box));
#ifdef TILE_CACHE
	for (int i = 0; i < obstacles.size(); i++) {
		DetourNavigationObstacle *obstacle = obstacles[i];
		/* TODO: Fix transforms */
		unsigned int id =
				mesh->add_obstacle(obstacle->get_global_transform().origin,
						obstacle->get_radius(), obstacle->get_height());
		obstacle->id = id;
	}
#else
	if (debug_view && mesh.is_valid()) {
		print_line("rebuilding debug navmesh");
		mesh->clear_debug_mesh();
		Object::cast_to<MeshInstance>(debug_view)->set_mesh(mesh->get_debug_mesh());
	}
#endif
}
/* More complicated queries follow */

DetourNavigationMeshInstance::DetourNavigationMeshInstance() :
		Spatial(),
		mesh(0),
		debug_view(0) {}

void DetourNavigation::_bind_methods() {}
void DetourNavigationArea::_bind_methods() {}
void DetourNavigationOffmeshConnection::_bind_methods() {}
void DetourNavigationMeshInstance::collect_geometries(bool recursive) {
	if (!mesh.is_valid()) {
		print_line("No valid navmesh set, please set valid navmesh resource");
		return;
	}
	List<Node *> groupNodes;
	Set<Node *> processedNodes;
	List<Node *> node_queue;
	geometries.clear();
	get_tree()->get_nodes_in_group(mesh->get_group(), &groupNodes);
	for (const List<Node *>::Element *E = groupNodes.front(); E; E = E->next()) {
		Node *groupNode = E->get();
		node_queue.push_back(groupNode);
	}
	print_line(String() + "node_queue size: " + itos(node_queue.size()));
	while (node_queue.size() > 0) {
		Node *groupNode = node_queue.front()->get();
		node_queue.pop_front();
		if (Object::cast_to<MeshInstance>(groupNode)) {
			MeshInstance *mi = Object::cast_to<MeshInstance>(groupNode);
			Ref<Mesh> mesh = mi->get_mesh();
			Transform xform = mi->get_global_transform();
			if (mesh.is_valid())
				add_mesh(mesh, xform);
		} else if (Object::cast_to<CSGShape>(groupNode)) {
			CSGShape *shape = Object::cast_to<CSGShape>(groupNode);
			Ref<ArrayMesh> mesh(memnew(ArrayMesh));
			Array arrays;
			arrays.resize(Mesh::ARRAY_MAX);
			PoolVector<Vector3> faces = shape->get_brush_faces();
			arrays[ArrayMesh::ARRAY_VERTEX] = faces;
			mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);

			Transform xform = shape->get_global_transform();
			if (mesh.is_valid())
				add_mesh(mesh, xform);
#ifdef TILE_CACHE
		} else if (Object::cast_to<DetourNavigationObstacle>(groupNode)) {
			DetourNavigationObstacle *obstacle =
					Object::cast_to<DetourNavigationObstacle>(groupNode);
			obstacles.push_back(obstacle);
#endif
		} else if (Object::cast_to<DetourNavigationOffmeshConnection>(groupNode)) {
			DetourNavigationOffmeshConnection *offcon =
					Object::cast_to<DetourNavigationOffmeshConnection>(groupNode);
			Transform xform = offcon->get_global_transform();
			Transform base = get_global_transform().inverse();
			Vector3 start = (base * xform).xform(Vector3());
			Vector3 end = (base * xform).xform(offcon->end);
			mesh->add_offmesh_connection(start, end, offcon->radius, offcon->flags,
					offcon->area, offcon->bidirectional);
		}
		if (recursive)
			for (int i = 0; i < groupNode->get_child_count(); i++)
				node_queue.push_back(groupNode->get_child(i));
	}
	print_line(String() + "geometries size: " + itos(geometries.size()));
}
void DetourNavigationMeshInstance::add_mesh(const Ref<Mesh> &mesh,
		const Transform &xform) {
	geometries.push_back(mesh);
	xforms.push_back(xform);
}
void DetourNavigationMeshInstance::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (get_tree()->is_debugging_navigation_hint()) {
				MeshInstance *dm = memnew(MeshInstance);
				if (mesh.is_valid())
					dm->set_mesh(mesh->get_debug_mesh());
				dm->set_material_override(get_tree()->get_debug_navigation_material());
				add_child(dm);
				debug_view = dm;
			}
#ifdef TILE_CACHE
			set_process(true);
#endif
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (debug_view) {
				debug_view->queue_delete();
				debug_view = NULL;
			}
#ifdef TILE_CACHE
			set_process(false);
#endif
		} break;
#ifdef TILE_CACHE
		case NOTIFICATION_PROCESS: {
			float delta = get_process_delta_time();
			if (mesh.is_valid()) {
				dtTileCache *tile_cache = mesh->get_tile_cache();
				if (tile_cache) {
					tile_cache->update(delta, mesh->get_navmesh());
					if (debug_view)
						Object::cast_to<MeshInstance>(debug_view)
								->set_mesh(mesh->get_debug_mesh());
				}
			}
		} break;
#endif
	}
}
void DetourNavigationMeshInstance::set_navmesh(
		const Ref<DetourNavigationMesh> &mesh) {
	if (this->mesh != mesh) {
		this->mesh = mesh;
		if (debug_view && this->mesh.is_valid())
			Object::cast_to<MeshInstance>(debug_view)
					->set_mesh(this->mesh->get_debug_mesh());
	}
}
void DetourNavigationMeshInstance::_bind_methods() {
	/* Navmesh */
	ClassDB::bind_method(D_METHOD("build"), &DetourNavigationMeshInstance::build);
	ClassDB::bind_method(D_METHOD("collect_geometries", "recursive"),
			&DetourNavigationMeshInstance::collect_geometries);
	ClassDB::bind_method(D_METHOD("set_navmesh", "navmesh"),
			&DetourNavigationMeshInstance::set_navmesh);
	ClassDB::bind_method(D_METHOD("get_navmesh"),
			&DetourNavigationMeshInstance::get_navmesh);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navmesh",
						 PROPERTY_HINT_RESOURCE_TYPE,
						 "DetourNavigationMesh"),
			"set_navmesh", "get_navmesh");
}
#undef SETGET
