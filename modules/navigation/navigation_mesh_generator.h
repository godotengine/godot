/*************************************************************************/
/*  navigation_mesh_generator.h                                          */
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

#ifndef NAVIGATION_MESH_GENERATOR_H
#define NAVIGATION_MESH_GENERATOR_H

#ifndef _3D_DISABLED

#include "scene/3d/navigation_mesh_instance.h"

#include <Recast.h>

#ifdef TOOLS_ENABLED
struct EditorProgress;
#endif

class NavigationMeshGenerator : public Object {
	GDCLASS(NavigationMeshGenerator, Object);

	static NavigationMeshGenerator *singleton;

protected:
	static void _bind_methods();

	static void _add_vertex(const Vector3 &p_vec3, Vector<float> &p_vertices);
	static void _add_mesh(const Ref<Mesh> &p_mesh, const Transform &p_xform, Vector<float> &p_vertices, Vector<int> &p_indices);
	static void _add_faces(const PoolVector3Array &p_faces, const Transform &p_xform, Vector<float> &p_vertices, Vector<int> &p_indices);
	static void _parse_geometry(const Transform &p_navmesh_xform, Node *p_node, Vector<float> &p_vertices, Vector<int> &p_indices, int p_generate_from, uint32_t p_collision_mask, bool p_recurse_children);

	static void _convert_detail_mesh_to_native_navigation_mesh(const rcPolyMeshDetail *p_detail_mesh, Ref<NavigationMesh> p_nav_mesh);
	static void _build_recast_navigation_mesh(
			Ref<NavigationMesh> p_nav_mesh,
#ifdef TOOLS_ENABLED
			EditorProgress *ep,
#endif
			rcHeightfield *hf,
			rcCompactHeightfield *chf,
			rcContourSet *cset,
			rcPolyMesh *poly_mesh,
			rcPolyMeshDetail *detail_mesh,
			Vector<float> &vertices,
			Vector<int> &indices);

public:
	static NavigationMeshGenerator *get_singleton();

	NavigationMeshGenerator();
	~NavigationMeshGenerator();

	void bake(Ref<NavigationMesh> p_nav_mesh, Node *p_node);
	void clear(Ref<NavigationMesh> p_nav_mesh);
};

#endif

#endif // NAVIGATION_MESH_GENERATOR_H
