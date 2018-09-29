#include "detour-navmesh.h"
#include <DetourNavMesh.h>
#include <DetourNavMeshBuilder.h>
#include <DetourTileCache.h>
#include <DetourTileCacheBuilder.h>
#include <Recast.h>
#include <fastlz.h>

static const int DEFAULT_TILE_SIZE = 64;
static const float DEFAULT_CELL_SIZE = 0.3f;
static const float DEFAULT_CELL_HEIGHT = 0.2f;
static const float DEFAULT_AGENT_HEIGHT = 2.0f;
static const float DEFAULT_AGENT_RADIUS = 0.6f;
static const float DEFAULT_AGENT_MAX_CLIMB = 0.9f;
static const float DEFAULT_AGENT_MAX_SLOPE = 45.0f;
static const float DEFAULT_REGION_MIN_SIZE = 8.0f;
static const float DEFAULT_REGION_MERGE_SIZE = 20.0f;
static const float DEFAULT_EDGE_MAX_LENGTH = 12.0f;
static const float DEFAULT_EDGE_MAX_ERROR = 1.3f;
static const float DEFAULT_DETAIL_SAMPLE_DISTANCE = 6.0f;
static const float DEFAULT_DETAIL_SAMPLE_MAX_ERROR = 1.0f;

#ifdef TILE_CACHE
static const int DEFAULT_MAX_OBSTACLES = 1024;
static const int DEFAULT_MAX_LAYERS = 16;
#endif

#ifdef TILE_CACHE
struct FastLZCompressor : public dtTileCacheCompressor {
	virtual int maxCompressedSize(const int bufferSize) {
		return (int)(bufferSize * 1.05f);
	}

	virtual dtStatus compress(const unsigned char *buffer, const int bufferSize,
			unsigned char *compressed,
			const int /*maxCompressedSize*/,
			int *compressedSize) {
		*compressedSize =
				fastlz_compress((const void *)buffer, bufferSize, compressed);
		return DT_SUCCESS;
	}

	virtual dtStatus decompress(const unsigned char *compressed,
			const int compressedSize, unsigned char *buffer,
			const int maxBufferSize, int *bufferSize) {
		*bufferSize =
				fastlz_decompress(compressed, compressedSize, buffer, maxBufferSize);
		return *bufferSize < 0 ? DT_FAILURE : DT_SUCCESS;
	}
};
struct LinearAllocator : public dtTileCacheAlloc {
	unsigned char *buffer;
	size_t capacity;
	size_t top;
	size_t high;

	LinearAllocator(const size_t cap) :
			buffer(0),
			capacity(0),
			top(0),
			high(0) {
		resize(cap);
	}

	~LinearAllocator() { dtFree(buffer); }

	void resize(const size_t cap) {
		if (buffer)
			dtFree(buffer);
		buffer = (unsigned char *)dtAlloc(cap, DT_ALLOC_PERM);
		capacity = cap;
	}

	virtual void reset() {
		high = MAX(high, top);
		top = 0;
	}

	virtual void *alloc(const size_t size) {
		if (!buffer)
			return 0;
		if (top + size > capacity)
			return 0;
		unsigned char *mem = &buffer[top];
		top += size;
		return mem;
	}

	virtual void free(void * /*ptr*/) {
		// Empty
	}
};

struct NavMeshProcess : public dtTileCacheMeshProcess {
	DetourNavigationMesh *nav;
	inline explicit NavMeshProcess(DetourNavigationMesh *mesh) :
			nav(mesh) {}
	virtual void process(struct dtNavMeshCreateParams *params,
			unsigned char *polyAreas, unsigned short *polyFlags) {
		/* Add proper flags and offmesh connections here */
		for (int i = 0; i < params->polyCount; i++) {
			if (polyAreas[i] != RC_NULL_AREA)
				polyFlags[i] = RC_WALKABLE_AREA;
		}
		params->offMeshConCount = nav->offmesh_radii.size();
		if (params->offMeshConCount > 0) {
			params->offMeshConVerts =
					reinterpret_cast<const float *>(&nav->offmesh_vertices[0]);
			params->offMeshConRad = &nav->offmesh_radii[0];
			params->offMeshConFlags = &nav->offmesh_flags[0];
			params->offMeshConAreas = &nav->offmesh_areas[0];
			params->offMeshConDir = &nav->offmesh_dir[0];
			print_line("added offmesh connection");
		} else {
			print_line("NO offmesh connection");
			params->offMeshConVerts = NULL;
			params->offMeshConRad = NULL;
			params->offMeshConFlags = NULL;
			params->offMeshConAreas = NULL;
			params->offMeshConDir = NULL;
		}
		nav->clear_debug_mesh();
		nav->get_debug_mesh();
	}
};

#endif

DetourNavigationMesh::DetourNavigationMesh() :
		Resource(),
		navmesh(NULL),
#ifdef TILE_CACHE
		tile_cache(0),
		tile_cache_alloc(new LinearAllocator(64000)),
		tile_cache_compressor(new FastLZCompressor),
		mesh_process(new NavMeshProcess(this)),
#endif
		group(""),
		initialized(false),
#ifdef TILE_CACHE
		max_obstacles(DEFAULT_MAX_OBSTACLES),
		max_layers(DEFAULT_MAX_LAYERS),
#endif
		tile_size(DEFAULT_TILE_SIZE),
		cell_size(DEFAULT_CELL_SIZE),
		cell_height(DEFAULT_CELL_HEIGHT),
		agent_height(DEFAULT_AGENT_HEIGHT),
		agent_radius(DEFAULT_AGENT_RADIUS),
		agent_max_climb(DEFAULT_AGENT_MAX_CLIMB),
		agent_max_slope(DEFAULT_AGENT_MAX_SLOPE),
		region_min_size(DEFAULT_REGION_MIN_SIZE),
		region_merge_size(DEFAULT_REGION_MERGE_SIZE),
		edge_max_length(DEFAULT_EDGE_MAX_LENGTH),
		edge_max_error(DEFAULT_EDGE_MAX_ERROR),
		detail_sample_distance(DEFAULT_DETAIL_SAMPLE_DISTANCE),
		detail_sample_max_error(DEFAULT_DETAIL_SAMPLE_MAX_ERROR),
		bounding_box(AABB()),
		padding(Vector3(1.0f, 1.0f, 1.0f)) {
}

bool DetourNavigationMesh::alloc_tile_cache() {
	tile_cache = dtAllocTileCache();
	if (!tile_cache) {
		ERR_PRINT("Could not allocate tile cache");
		release_navmesh();
		return false;
	}
	return true;
}

void DetourNavigationMesh::release_navmesh() {
	dtFreeNavMesh((dtNavMesh *)navmesh);
	navmesh = NULL;
	num_tiles_x = 0;
	num_tiles_z = 0;
	bounding_box = AABB();
}

void DetourNavigationMesh::set_group(const String &group) {
	this->group = group;
}

unsigned int DetourNavigationMesh::add_obstacle(const Vector3 &pos,
		real_t radius, real_t height) {
	/* Need to test how this works and why this needed at all */
	/* TODO implement navmesh changes queue */
	//	while (tile_cache->isObstacleQueueFull())
	//		tile_cache->update(1, navMesh_);
	dtObstacleRef ref = 0;
	if (dtStatusFailed(
				tile_cache->addObstacle(&pos.coord[0], radius, height, &ref))) {
		ERR_PRINT("can't add obstacle");
		return 0;
	}
	return (unsigned int)ref;
}
void DetourNavigationMesh::remove_obstacle(unsigned int id) {
	/* Need to test how this works and why this needed at all */
	/* TODO implement navmesh changes queue */
	//	while (tile_cache->isObstacleQueueFull())
	//		tile_cache->update(1, navMesh_);
	if (dtStatusFailed(tile_cache->removeObstacle(id)))
		ERR_PRINT("failed to remove obstacle");
}

bool DetourNavigationMesh::alloc() {
	navmesh = dtAllocNavMesh();
	return navmesh ? true : false;
}
bool DetourNavigationMesh::init(dtNavMeshParams *params) {
	if (dtStatusFailed((navmesh)->init(params))) {
		release_navmesh();
		return false;
	}
	initialized = true;
	return true;
}
#ifdef TILE_CACHE
bool DetourNavigationMesh::init_tile_cache(dtTileCacheParams *params) {
	if (dtStatusFailed(tile_cache->init(params, tile_cache_alloc,
				tile_cache_compressor, mesh_process))) {
		ERR_PRINT("Could not initialize tile cache");
		release_navmesh();
		return false;
	}
	return true;
}
#endif
Ref<ArrayMesh> DetourNavigationMesh::get_debug_mesh() {
	if (debug_mesh.is_valid())
		return debug_mesh;
	if (!navmesh)
		return debug_mesh;
	print_line("building debug navmesh");
	List<Vector3> lines;
	const dtNavMesh *navm = navmesh;
	for (int i = 0; i < navm->getMaxTiles(); i++) {
		const dtMeshTile *tile = navm->getTile(i);
		if (!tile || !tile->header)
			continue;
		for (int j = 0; j < tile->header->polyCount; j++) {
			dtPoly *poly = tile->polys + j;
			if (poly->getType() != DT_POLYTYPE_OFFMESH_CONNECTION) {
				for (int k = 0; k < poly->vertCount; k++) {
					lines.push_back(*reinterpret_cast<const Vector3 *>(
							&tile->verts[poly->verts[k] * 3]));
					lines.push_back(*reinterpret_cast<const Vector3 *>(
							&tile->verts[poly->verts[(k + 1) % poly->vertCount] * 3]));
				}
			} else if (poly->getType() == DT_POLYTYPE_OFFMESH_CONNECTION) {
				const dtOffMeshConnection *con =
						&tile->offMeshCons[j - tile->header->offMeshBase];
				const float *va = &tile->verts[poly->verts[0] * 3];
				const float *vb = &tile->verts[poly->verts[1] * 3];
#if 0
				/* TODO: implement offmesh debug proper */
				bool startSet = false;
				bool endSet = false;
				for (unsigned int k = poly->firstLink; k != DT_NULL_LINK;
						k = tile->links[k].next) {
					if (tile->links[k].edge == 0)
						startSet = true;
					if (tile->links[k].edge == 1)
						endSet = true;
				}
#endif
				Vector3 p0 = *reinterpret_cast<const Vector3 *>(va);
				Vector3 p1 = *reinterpret_cast<const Vector3 *>(&con[0]);
				Vector3 p2 = *reinterpret_cast<const Vector3 *>(&con[3]);
				Vector3 p3 = *reinterpret_cast<const Vector3 *>(vb);
				lines.push_back(p0);
				lines.push_back(p1);
				lines.push_back(p1);
				lines.push_back(p2);
				lines.push_back(p2);
				lines.push_back(p3);
				print_line("debug offmesh connection");
			}
		}
	}
	print_line("debug mesh lines: " + itos(lines.size()));

	PoolVector<Vector3> varr;
	varr.resize(lines.size());
	PoolVector<Vector3>::Write w = varr.write();
	int idx = 0;
	for (List<Vector3>::Element *E = lines.front(); E; E = E->next()) {
		w[idx++] = E->get();
	}

	debug_mesh = Ref<ArrayMesh>(memnew(ArrayMesh));

	Array arr;
	arr.resize(Mesh::ARRAY_MAX);
	arr[Mesh::ARRAY_VERTEX] = varr;

	debug_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, arr);

	return debug_mesh;
}

void DetourNavigationMesh::set_data(const Dictionary &p_value) {
	dtNavMeshParams params;
	Vector3 orig = p_value["orig"];
	rcVcopy(params.orig, &orig.coord[0]);
	params.tileWidth = p_value["tile_edge_length"];
	params.tileHeight = p_value["tile_edge_length"];
	params.maxTiles = p_value["max_tiles"];
	params.maxPolys = p_value["max_polys"];
	if (navmesh) {
		if (initialized)
			dtFreeNavMesh(navmesh);
		else
			dtFree(navmesh);
		navmesh = NULL;
	}
	if (!alloc())
		return;
	if (!init(&params))
		return;
}

Dictionary DetourNavigationMesh::get_data() {
	Dictionary t;
#if 0
	t["initialized"] = initialized;
	if (!initialized) {
		return t;
	}
        const dtNavMeshParams *params = navmesh->getParams();
	Vector3 orig;
	rcVcopy(&orig.coord[0], params->orig);
	t["orig"] = orig;
	t["tile_edge_length"] = params->tileWidth;
	t["max_tiles"] = params->maxTiles;
	t["max_polys"] = params->maxPolys;
	PoolVector<uint8_t> data;
	PoolVector<uint8_t>::Write data_w = data.write();
	const dtNavMesh *nm = navmesh;
	int pos = 0;
	for (int z = 0; z < num_tiles_z; z++)
		for (int x = 0; x < num_tiles_x; x++) {
			const dtMeshTile* tile = nm->getTileAt(x, z, 0);
			if (!tile)
				continue;
			if (pos >= data.size())
				data.resize(data.size() + sizeof(int) * 2 + sizeof(unsigned int) * 2 + tile->dataSize);
			memcpy(&data_w[pos], &x, sizeof(x));
			pos += sizeof(x);
			memcpy(&data_w[pos], &z, sizeof(x));
			pos += sizeof(z);
			uint32_t tile_ref = (uint32_t)nm->getTileRef(tile);
			memcpy(&data_w[pos], &tile_ref, sizeof(tile_ref));
			pos += sizeof(tile_ref);
			uint32_t data_size = (uint32_t)tile->dataSize;
			memcpy(&data_w[pos], &data_size, sizeof(data_size));
			pos += sizeof(data_size);
			memcpy(&data_w[pos], tile->data, data_size);
			pos += data_size;
		}
	print_line("submitted: " + itos(data.size()));
	t["data"] = data;
#endif
	return t;
}

#define SETGET(v, t)                                                           \
	ClassDB::bind_method(D_METHOD("set_" #v, #v),                              \
			&DetourNavigationMesh::set_##v);                                   \
	ClassDB::bind_method(D_METHOD("get_" #v), &DetourNavigationMesh::get_##v); \
	ADD_PROPERTY(PropertyInfo(Variant::t, #v), "set_" #v, "get_" #v)

void DetourNavigationMesh::_bind_methods() {
	SETGET(cell_size, REAL);
	SETGET(cell_height, REAL);
	SETGET(agent_height, REAL);
	SETGET(agent_radius, REAL);
	SETGET(agent_max_climb, REAL);
	SETGET(agent_max_slope, REAL);
	SETGET(region_min_size, REAL);
	SETGET(region_merge_size, REAL);
	SETGET(edge_max_length, REAL);
	SETGET(edge_max_error, REAL);
	SETGET(detail_sample_distance, REAL);
	SETGET(detail_sample_max_error, REAL);
	SETGET(group, STRING);
	SETGET(padding, VECTOR3);
	SETGET(tile_size, INT);
	BIND_ENUM_CONSTANT(PARTITION_WATERSHED);
	BIND_ENUM_CONSTANT(PARTITION_MONOTONE);
	ClassDB::bind_method(D_METHOD("set_partition_type", "type"),
			&DetourNavigationMesh::set_partition_type);
	ClassDB::bind_method(D_METHOD("get_partition_type"),
			&DetourNavigationMesh::get_partition_type);
	ClassDB::bind_method(D_METHOD("set_data", "data"),
			&DetourNavigationMesh::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &DetourNavigationMesh::get_data);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "partition_type", PROPERTY_HINT_ENUM,
						 "watershed,monotone"),
			"set_partition_type", "get_partition_type");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "",
						 PROPERTY_USAGE_STORAGE),
			"set_data", "get_data");
}
#undef SETGET

void DetourNavigationMesh::get_tile_bounding_box(int x, int z, Vector3 &bmin,
		Vector3 &bmax) {
	const float tile_edge_length = (float)tile_size * cell_size;
	bmin = bounding_box.position +
		   Vector3(tile_edge_length * (float)x, 0, tile_edge_length * (float)z);
	bmax =
			bmin + Vector3(tile_edge_length, bounding_box.size.y, tile_edge_length);
	// print_line("tile bounding box: " +itos(x) + " " + itos(z) + ": " +
	// String(bmin) + "/" + String(bmax));
	// print_line("mesh bounding box: " + String(mesh->bounding_box));
}
bool DetourNavigationMesh::build_tile(const Transform &xform,
		const Vector<Ref<Mesh> > &geometries,
		const Vector<Transform> &xforms, int x,
		int z) {
	Vector3 bmin, bmax;
	get_tile_bounding_box(x, z, bmin, bmax);
	dtNavMesh *nav = get_navmesh();
#ifdef TILE_CACHE
	dtTileCache *tile_cache = get_tile_cache();
	tile_cache->removeTile(nav->getTileRefAt(x, z, 0), NULL, NULL);
#else
	nav->removeTile(nav->getTileRefAt(x, z, 0), NULL, NULL);
#endif
	rcConfig cfg;
	cfg.cs = cell_size;
	cfg.ch = cell_height;
	cfg.walkableSlopeAngle = agent_max_slope;
	cfg.walkableHeight = (int)ceil(agent_height / cfg.ch);
	cfg.walkableClimb = (int)floor(agent_max_climb / cfg.ch);
	cfg.walkableRadius = (int)ceil(agent_radius / cfg.cs);
	cfg.maxEdgeLen = (int)(edge_max_length / cfg.cs);
	cfg.maxSimplificationError = edge_max_error;
	cfg.minRegionArea = (int)sqrtf(region_min_size);
	cfg.mergeRegionArea = (int)sqrtf(region_merge_size);
	cfg.maxVertsPerPoly = 6;
	cfg.tileSize = tile_size;
	cfg.borderSize = cfg.walkableRadius + 3;
	cfg.width = cfg.tileSize + cfg.borderSize * 2;
	cfg.height = cfg.tileSize + cfg.borderSize * 2;
	cfg.detailSampleDist =
			detail_sample_distance < 0.9f ? 0.0f : cell_size * detail_sample_distance;
	cfg.detailSampleMaxError = cell_height * detail_sample_max_error;
	rcVcopy(cfg.bmin, &bmin.coord[0]);
	rcVcopy(cfg.bmax, &bmax.coord[0]);
	cfg.bmin[0] -= cfg.borderSize * cfg.cs;
	cfg.bmin[2] -= cfg.borderSize * cfg.cs;
	cfg.bmax[0] += cfg.borderSize * cfg.cs;
	cfg.bmax[2] += cfg.borderSize * cfg.cs;

	AABB expbox(bmin, bmax - bmin);
	expbox.position.x -= cfg.borderSize * cfg.cs;
	expbox.position.z -= cfg.borderSize * cfg.cs;
	expbox.size.x += 2.0 * cfg.borderSize * cfg.cs;
	expbox.size.z += 2.0 * cfg.borderSize * cfg.cs;
	Vector<float> points;
	Vector<int> indices;
	Transform base = xform.inverse();
	for (int idx = 0; idx < geometries.size(); idx++) {
		if (!geometries[idx].is_valid())
			continue;
		AABB mesh_aabb = geometries[idx]->get_aabb();
		Transform xform = base * xforms[idx];
		mesh_aabb = xform.xform(mesh_aabb);
		if (!mesh_aabb.intersects_inclusive(expbox) &&
				!expbox.encloses(mesh_aabb)) {
			continue;
		}
		// Add offmesh
		// Add NavArea
		// Add PhysicsBodies?
		Ref<Mesh> mdata = geometries[idx];
		// FIXME
		add_meshdata(mdata, xform, points, indices);
	}
// print_line(String() + "points: " + itos(points.size()) + " indices: " +
// itos(indices.size()) + " tile_size: " + itos(mesh->tile_size));
#if 0
	print_line("mesh points:");
	for (int k = 0; k < points.size(); k += 3)
		print_line("point: " + itos(k) + ": " + rtos(points[k]) + ", " + rtos(points[k + 1]) + ", " + rtos(points[k + 2]));
#endif
	if (points.size() == 0 || indices.size() == 0)
		/* Nothing to do */
		return true;
	rcHeightfield *heightfield = rcAllocHeightfield();
	if (!heightfield) {
		ERR_PRINT("Failed to allocate height field");
		return false;
	}
	rcContext *ctx = new rcContext(true);
	if (!rcCreateHeightfield(ctx, *heightfield, cfg.width, cfg.height, cfg.bmin,
				cfg.bmax, cfg.cs, cfg.ch)) {
		ERR_PRINT("Failed to create height field");
		return false;
	}
	int ntris = indices.size() / 3;
	Vector<unsigned char> tri_areas;
	tri_areas.resize(ntris);
	memset(&tri_areas.write[0], 0, ntris);
	rcMarkWalkableTriangles(ctx, cfg.walkableSlopeAngle, &points[0],
			points.size() / 3, &indices[0], ntris,
			&tri_areas.write[0]);
	rcRasterizeTriangles(ctx, &points[0], points.size() / 3, &indices[0],
			&tri_areas[0], ntris, *heightfield, cfg.walkableClimb);
	rcFilterLowHangingWalkableObstacles(ctx, cfg.walkableClimb, *heightfield);

	rcFilterLedgeSpans(ctx, cfg.walkableHeight, cfg.walkableClimb, *heightfield);
	rcFilterWalkableLowHeightSpans(ctx, cfg.walkableHeight, *heightfield);

	rcCompactHeightfield *compact_heightfield = rcAllocCompactHeightfield();
	if (!compact_heightfield) {
		ERR_PRINT("Failed to allocate compact height field");
		return false;
	}
	if (!rcBuildCompactHeightfield(ctx, cfg.walkableHeight, cfg.walkableClimb,
				*heightfield, *compact_heightfield)) {
		ERR_PRINT("Could not build compact height field");
		return false;
	}
	if (!rcErodeWalkableArea(ctx, cfg.walkableRadius, *compact_heightfield)) {
		ERR_PRINT("Could not erode walkable area");
		return false;
	}

// TODO: Implement area storage in navmesh data and build from that data
// populate at collect_geometry stage
// areas should indicate if walkable
#if 0
	for (unsigned int i = 0; i < nav_areas.size(); i++) {
		Vector3 amin = nav_areas[i].bounds.position;
		Vector3 amax = amin + nav_areas[i].bounds.size;
		int id = nav_areas[i].id;
		rcMarkBoxArea(ctx, &amin.coord[0], &amax.coord[0],
			id, *compact_heightfield);
	}
#endif
	if (partition_type == DetourNavigationMesh::PARTITION_WATERSHED) {
		if (!rcBuildDistanceField(ctx, *compact_heightfield))
			return false;
		if (!rcBuildRegions(ctx, *compact_heightfield, cfg.borderSize,
					cfg.minRegionArea, cfg.mergeRegionArea))
			return false;
	} else if (!rcBuildRegionsMonotone(ctx, *compact_heightfield, cfg.borderSize,
					   cfg.minRegionArea, cfg.mergeRegionArea))
		return false;
#ifdef TILE_CACHE
	rcHeightfieldLayerSet *heightfield_layer_set = rcAllocHeightfieldLayerSet();
	if (!heightfield_layer_set) {
		ERR_PRINT("Could not allocate height field layer set");
		return false;
	}
	if (!rcBuildHeightfieldLayers(ctx, *compact_heightfield, cfg.borderSize,
				cfg.walkableHeight, *heightfield_layer_set)) {
		ERR_PRINT("Could not build heightfield layers");
		return false;
	}
	for (int i = 0; i < heightfield_layer_set->nlayers; i++) {
		dtTileCacheLayerHeader header;
		header.magic = DT_TILECACHE_MAGIC;
		header.version = DT_TILECACHE_VERSION;
		header.tx = x;
		header.ty = z;
		header.tlayer = i;
		rcHeightfieldLayer *layer = &heightfield_layer_set->layers[i];
		rcVcopy(header.bmin, layer->bmin);
		rcVcopy(header.bmax, layer->bmax);
		header.width = (unsigned char)layer->width;
		header.height = (unsigned char)layer->height;
		header.minx = (unsigned char)layer->minx;
		header.maxx = (unsigned char)layer->maxx;
		header.miny = (unsigned char)layer->miny;
		header.maxy = (unsigned char)layer->maxy;
		header.hmin = (unsigned short)layer->hmin;
		header.hmax = (unsigned short)layer->hmax;
		unsigned char *tile_data;
		int tile_data_size;
		if (dtStatusFailed(dtBuildTileCacheLayer(
					get_tile_cache_compressor(), &header, layer->heights, layer->areas,
					layer->cons, &tile_data, &tile_data_size))) {
			ERR_PRINT("Failed to build tile cache layers");
			return false;
		}
		dtCompressedTileRef tileRef;
		int status = tile_cache->addTile(tile_data, tile_data_size,
				DT_COMPRESSEDTILE_FREE_DATA, &tileRef);
		if (dtStatusFailed((dtStatus)status)) {
			dtFree(tile_data);
			tile_data = NULL;
		}
		tile_cache->buildNavMeshTilesAt(x, z, nav);
	}
#else
	rcContourSet *contour_set = rcAllocContourSet();
	if (!contour_set)
		return false;
	print_line("allocated contour set");
	if (!rcBuildContours(ctx, *compact_heightfield, cfg.maxSimplificationError,
				cfg.maxEdgeLen, *contour_set))
		return false;
	print_line("created contour set");
	rcPolyMesh *poly_mesh = rcAllocPolyMesh();
	if (!poly_mesh)
		return false;
	print_line("allocated polymesh");
	if (!rcBuildPolyMesh(ctx, *contour_set, cfg.maxVertsPerPoly, *poly_mesh))
		return false;
	print_line("created polymesh");
	rcPolyMeshDetail *poly_mesh_detail = rcAllocPolyMeshDetail();
	if (!poly_mesh_detail)
		return false;
	print_line("allocated polymesh detail");
	if (!rcBuildPolyMeshDetail(ctx, *poly_mesh, *compact_heightfield,
				cfg.detailSampleDist, cfg.detailSampleMaxError,
				*poly_mesh_detail))
		return false;
	print_line("created polymesh detail");
	/* Assign area flags TODO: use nav area assignment here */
	for (int i = 0; i < poly_mesh->npolys; i++) {
		if (poly_mesh->areas[i] != RC_NULL_AREA)
			poly_mesh->flags[i] = 0x1;
	}
	print_line("created area flags");
	unsigned char *nav_data = NULL;
	int nav_data_size = 0;
	dtNavMeshCreateParams params;
	memset(&params, 0, sizeof params);
	params.verts = poly_mesh->verts;
	params.vertCount = poly_mesh->nverts;
	params.polys = poly_mesh->polys;
	params.polyAreas = poly_mesh->areas;
	params.polyFlags = poly_mesh->flags;
	params.polyCount = poly_mesh->npolys;
	params.nvp = poly_mesh->nvp;
	params.detailMeshes = poly_mesh_detail->meshes;
	params.detailVerts = poly_mesh_detail->verts;
	params.detailVertsCount = poly_mesh_detail->nverts;
	params.detailTris = poly_mesh_detail->tris;
	params.detailTriCount = poly_mesh_detail->ntris;
	params.walkableHeight = mesh->agent_height;
	params.walkableRadius = mesh->agent_radius;
	params.walkableClimb = mesh->agent_max_climb;
	params.tileX = x;
	params.tileY = z;
	rcVcopy(params.bmin, poly_mesh->bmin);
	rcVcopy(params.bmax, poly_mesh->bmax);
	params.cs = cfg.cs;
	params.ch = cfg.ch;
	params.buildBvTree = true;
#if 0
	// building offmesh conections
    if (build.offMeshRadii_.Size())
    {
        params.offMeshConCount = build.offMeshRadii_.Size();
        params.offMeshConVerts = &build.offMeshVertices_[0].x_;
        params.offMeshConRad = &build.offMeshRadii_[0];
        params.offMeshConFlags = &build.offMeshFlags_[0];
        params.offMeshConAreas = &build.offMeshAreas_[0];
        params.offMeshConDir = &build.offMeshDir_[0];
    }
#endif
	print_line("setup offmesh connections");

	if (!dtCreateNavMeshData(&params, &nav_data, &nav_data_size))
		return false;
	if (dtStatusFailed(mesh->get_navmesh()->addTile(
				nav_data, nav_data_size, DT_TILE_FREE_DATA, 0, NULL))) {
		dtFree(nav_data);
		return false;
	}
	print_line("created navmesh data");
#endif
	return true;
}

void DetourNavigationMesh::add_meshdata(const Ref<Mesh> &p_mesh,
		const Transform &p_xform,
		Vector<float> &p_verticies,
		Vector<int> &p_indices) {
	int current_vertex_count = 0;

	for (int i = 0; i < p_mesh->get_surface_count(); i++) {
		current_vertex_count = p_verticies.size() / 3;

		if (p_mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES)
			continue;

		int index_count = 0;
		if (p_mesh->surface_get_format(i) & Mesh::ARRAY_FORMAT_INDEX) {
			index_count = p_mesh->surface_get_array_index_len(i);
		} else {
			index_count = p_mesh->surface_get_array_len(i);
		}

		ERR_CONTINUE((index_count == 0 || (index_count % 3) != 0));

		int face_count = index_count / 3;

		Array a = p_mesh->surface_get_arrays(i);

		PoolVector<Vector3> mesh_vertices = a[Mesh::ARRAY_VERTEX];
		PoolVector<Vector3>::Read vr = mesh_vertices.read();

		if (p_mesh->surface_get_format(i) & Mesh::ARRAY_FORMAT_INDEX) {

			PoolVector<int> mesh_indices = a[Mesh::ARRAY_INDEX];
			PoolVector<int>::Read ir = mesh_indices.read();

			for (int i = 0; i < mesh_vertices.size(); i++) {
				Vector3 p_vec3 = p_xform.xform(vr[i]);
				p_verticies.push_back(p_vec3.x);
				p_verticies.push_back(p_vec3.y);
				p_verticies.push_back(p_vec3.z);
			}

			for (int i = 0; i < face_count; i++) {
				// CCW
				p_indices.push_back(current_vertex_count + (ir[i * 3 + 0]));
				p_indices.push_back(current_vertex_count + (ir[i * 3 + 2]));
				p_indices.push_back(current_vertex_count + (ir[i * 3 + 1]));
			}
		} else {
			face_count = mesh_vertices.size() / 3;
			for (int i = 0; i < face_count; i++) {
				Vector3 p_vec3 = p_xform.xform(vr[i * 3 + 0]);
				p_verticies.push_back(p_vec3.x);
				p_verticies.push_back(p_vec3.y);
				p_verticies.push_back(p_vec3.z);
				p_vec3 = p_xform.xform(vr[i * 3 + 2]);
				p_verticies.push_back(p_vec3.x);
				p_verticies.push_back(p_vec3.y);
				p_verticies.push_back(p_vec3.z);
				p_vec3 = p_xform.xform(vr[i * 3 + 1]);
				p_verticies.push_back(p_vec3.x);
				p_verticies.push_back(p_vec3.y);
				p_verticies.push_back(p_vec3.z);

				p_indices.push_back(current_vertex_count + (i * 3 + 0));
				p_indices.push_back(current_vertex_count + (i * 3 + 1));
				p_indices.push_back(current_vertex_count + (i * 3 + 2));
			}
		}
	}
}

void DetourNavigationMesh::remove_tile(int x, int z) {
#ifdef TILE_CACHE
	dtTileCache *tile_cache = get_tile_cache();
	tile_cache->removeTile(navmesh->getTileRefAt(x, z, 0), NULL, NULL);
#else
	nav->removeTile(navmesh->getTileRefAt(x, z, 0), NULL, NULL);
#endif
}

unsigned int DetourNavigationMesh::build_tiles(
		const Transform &xform, const Vector<Ref<Mesh> > &geometries,
		const Vector<Transform> &xforms, int x1, int z1, int x2, int z2) {
	unsigned ret = 0;
	for (int z = z1; z <= z2; z++) {
		for (int x = x1; x <= x2; x++) {
			if (build_tile(xform, geometries, xforms, x, z))
				ret++;
		}
	}
#ifdef TILE_CACHE
	get_tile_cache()->update(0, get_navmesh());
#endif
	return ret;
}
