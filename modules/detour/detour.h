/*************************************************************************/
/*  detour.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef DETOUR_H
#define DETOUR_H
#include "core/resource.h"
#include "scene/3d/spatial.h"
#include "scene/resources/mesh.h"

#include "detour-navmesh.h"
class DetourNavigation : public Spatial {
	GDCLASS(DetourNavigation, Spatial);
	DetourNavigation() :
			Spatial() {
	}
	static void _bind_methods();
};
class DetourNavigationMeshInstance;
class DetourNavigationOffmeshConnection : public Spatial {
	GDCLASS(DetourNavigationOffmeshConnection, Spatial);
	friend class DetourNavigationMeshInstance;
	Vector3 end;
	float radius;
	unsigned short flags;
	unsigned char area;
	bool bidirectional;
	static void _bind_methods();

public:
	Vector3 endpoint;
	DetourNavigationOffmeshConnection() :
			Spatial(),
			end(Vector3(0, 0, 10.0f)),
			radius(5.0f),
			flags(1),
			area(1),
			bidirectional(true) {
	}
};

class DetourNavigationArea : public Spatial {
	GDCLASS(DetourNavigationArea, Spatial);
	static void _bind_methods();

public:
	AABB bounds;
	int id;
	unsigned int flags;
};

class dtNavMesh;
struct dtNavMeshParams;
class dtNavMeshQuery;
class dtQueryFilter;
#ifdef TILE_CACHE
class dtTileCache;
struct dtTileCacheAlloc;
struct dtTileCacheCompressor;
struct dtTileCacheMeshProcess;
struct dtTileCacheLayer;
struct dtTileCacheContourSet;
struct dtTileCachePolyMesh;
struct dtTileCacheParams;
struct NavMeshProcess;
#endif
#ifdef TILE_CACHE
class DetourNavigationObstacle;
#endif
class DetourNavigationMeshInstance : public Spatial {
	class DetourNavigationQueryData;
	GDCLASS(DetourNavigationMeshInstance, Spatial);
	Ref<DetourNavigationMesh> mesh;
	static void _bind_methods();
	void _notification(int p_what);
	Node *debug_view;
#ifdef TILE_CACHE
	Vector<DetourNavigationObstacle *> obstacles;
#endif
protected:
	static float random();

public:
	void set_navmesh(const Ref<DetourNavigationMesh> &mesh);
	Ref<DetourNavigationMesh> get_navmesh() {
		return mesh;
	}
	DetourNavigationMeshInstance();
	void build();
	void add_mesh(const Ref<Mesh> &mesh, const Transform &transform);
	void set_group(const String &group) {
		mesh->set_group(group);
	}
	const String &get_group() const {
		return mesh->get_group();
	}
	Vector<Ref<Mesh> > geometries;
	Vector<Transform> xforms;
	Vector<DetourNavigationArea> nav_areas;
	void collect_geometries(bool recursive);
};
#endif
