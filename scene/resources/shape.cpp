/*************************************************************************/
/*  shape.cpp                                                            */
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
#include "shape.h"

#include "os/os.h"
#include "scene/main/scene_main_loop.h"
#include "scene/resources/mesh.h"
#include "servers/physics_server.h"

void Shape::add_vertices_to_array(PoolVector<Vector3> &array, const Transform &p_xform) {

	Vector<Vector3> toadd = _gen_debug_mesh_lines();

	if (toadd.size()) {

		int base = array.size();
		array.resize(base + toadd.size());
		PoolVector<Vector3>::Write w = array.write();
		for (int i = 0; i < toadd.size(); i++) {
			w[i + base] = p_xform.xform(toadd[i]);
		}
	}
}

Ref<Mesh> Shape::get_debug_mesh() {

	if (debug_mesh_cache.is_valid())
		return debug_mesh_cache;

	Vector<Vector3> lines = _gen_debug_mesh_lines();

	debug_mesh_cache = Ref<Mesh>(memnew(Mesh));

	if (!lines.empty()) {
		//make mesh
		PoolVector<Vector3> array;
		array.resize(lines.size());
		{

			PoolVector<Vector3>::Write w = array.write();
			for (int i = 0; i < lines.size(); i++) {
				w[i] = lines[i];
			}
		}

		Array arr;
		arr.resize(Mesh::ARRAY_MAX);
		arr[Mesh::ARRAY_VERTEX] = array;

		SceneTree *st = OS::get_singleton()->get_main_loop()->cast_to<SceneTree>();

		debug_mesh_cache->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, arr);

		if (st) {
			debug_mesh_cache->surface_set_material(0, st->get_debug_collision_material());
		}
	}

	return debug_mesh_cache;
}

Shape::Shape() {

	ERR_PRINT("Constructor must not be called!");
}

Shape::Shape(RID p_shape) {

	shape = p_shape;
}

Shape::~Shape() {

	PhysicsServer::get_singleton()->free(shape);
}
