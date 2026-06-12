/**************************************************************************/
/*  navigation_region_3d_gizmo_plugin.cpp                                 */
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

#include "navigation_region_3d_gizmo_plugin.h"

#include "core/math/random_pcg.h"
#include "scene/3d/label_3d.h"
#include "scene/3d/navigation/navigation_region_3d.h"
#include "servers/navigation_3d/navigation_server_3d.h"

NavigationRegion3DGizmoPlugin::NavigationRegion3DGizmoPlugin() {
	create_material("face_material", NavigationServer3D::get_singleton()->get_debug_navigation_geometry_face_color(), false, false, true);
	create_material("face_material_disabled", NavigationServer3D::get_singleton()->get_debug_navigation_geometry_face_disabled_color(), false, false, true);
	create_material("edge_material", NavigationServer3D::get_singleton()->get_debug_navigation_geometry_edge_color());
	create_material("edge_material_disabled", NavigationServer3D::get_singleton()->get_debug_navigation_geometry_edge_disabled_color());

	Color baking_aabb_material_color = Color(0.8, 0.5, 0.7);
	baking_aabb_material_color.a = 0.1;
	create_material("baking_aabb_material", baking_aabb_material_color);
}

bool NavigationRegion3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<NavigationRegion3D>(p_spatial) != nullptr;
}

String NavigationRegion3DGizmoPlugin::get_gizmo_name() const {
	return "NavigationRegion3D";
}

int NavigationRegion3DGizmoPlugin::get_priority() const {
	return -1;
}

void NavigationRegion3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	NavigationRegion3D *navigationregion = Object::cast_to<NavigationRegion3D>(p_gizmo->get_node_3d());

	p_gizmo->clear();
	Ref<NavigationMesh> navigationmesh = navigationregion->get_navigation_mesh();
	if (navigationmesh.is_null()) {
		Node *debug_holder = navigationregion->find_child("_debug_holder", false, false);
		if (debug_holder != nullptr) {
			navigationregion->remove_child(debug_holder);
			debug_holder->queue_free();
		}
		return;
	}

	AABB baking_aabb = navigationmesh->get_filter_baking_aabb();
	if (baking_aabb.has_volume()) {
		Vector3 baking_aabb_offset = navigationmesh->get_filter_baking_aabb_offset();

		if (p_gizmo->is_selected()) {
			Ref<Material> material = get_material("baking_aabb_material", p_gizmo);
			p_gizmo->add_solid_box(material, baking_aabb.get_size(), baking_aabb.get_center() + baking_aabb_offset);
		}
	}

	Vector<Vector3> vertices = navigationmesh->get_vertices();
	const Vector3 *vr = vertices.ptr();
	List<Face3> faces;
	for (int i = 0; i < navigationmesh->get_polygon_count(); i++) {
		Vector<int> p = navigationmesh->get_polygon(i);

		for (int j = 2; j < p.size(); j++) {
			Face3 f;
			f.vertex[0] = vr[p[0]];
			f.vertex[1] = vr[p[j - 1]];
			f.vertex[2] = vr[p[j]];

			faces.push_back(f);
		}
	}

	if (faces.is_empty()) {
		Node *debug_holder = navigationregion->find_child("_debug_holder", false, false);
		if (debug_holder != nullptr) {
			navigationregion->remove_child(debug_holder);
			debug_holder->queue_free();
		}
		return;
	}

	HashMap<_EdgeKey, bool, _EdgeKey> edge_map;
	Vector<Vector3> tmeshfaces;
	tmeshfaces.resize(faces.size() * 3);

	{
		Vector3 *tw = tmeshfaces.ptrw();
		int tidx = 0;

		for (const Face3 &f : faces) {
			for (int j = 0; j < 3; j++) {
				tw[tidx++] = f.vertex[j];
				_EdgeKey ek;
				ek.from = f.vertex[j].snappedf(CMP_EPSILON);
				ek.to = f.vertex[(j + 1) % 3].snappedf(CMP_EPSILON);
				if (ek.from < ek.to) {
					SWAP(ek.from, ek.to);
				}

				HashMap<_EdgeKey, bool, _EdgeKey>::Iterator F = edge_map.find(ek);

				if (F) {
					F->value = false;

				} else {
					edge_map[ek] = true;
				}
			}
		}
	}
	Vector<Vector3> lines;

	for (const KeyValue<_EdgeKey, bool> &E : edge_map) {
		if (E.value) {
			lines.push_back(E.key.from);
			lines.push_back(E.key.to);
		}
	}

	Ref<TriangleMesh> tmesh = memnew(TriangleMesh);
	tmesh->create(tmeshfaces);

	p_gizmo->add_collision_triangles(tmesh);
	p_gizmo->add_collision_segments(lines);

	Ref<ArrayMesh> debug_mesh = Ref<ArrayMesh>(memnew(ArrayMesh));
	int polygon_count = navigationmesh->get_polygon_count();

	// build geometry face surface
	Vector<Vector3> face_vertex_array;
	face_vertex_array.resize(polygon_count * 3);

	for (int i = 0; i < polygon_count; i++) {
		Vector<int> polygon = navigationmesh->get_polygon(i);

		face_vertex_array.push_back(vertices[polygon[0]]);
		face_vertex_array.push_back(vertices[polygon[1]]);
		face_vertex_array.push_back(vertices[polygon[2]]);
	}

	Array face_mesh_array;
	face_mesh_array.resize(Mesh::ARRAY_MAX);
	face_mesh_array[Mesh::ARRAY_VERTEX] = face_vertex_array;

	bool enabled_geometry_face_random_color = NavigationServer3D::get_singleton()->get_debug_navigation_enable_geometry_face_random_color();
	bool has_polygon_meta = navigationmesh->get_polygon_meta_count() == polygon_count;

	if (enabled_geometry_face_random_color || has_polygon_meta) {
		// Face coloring.
		Color debug_navigation_geometry_face_color = NavigationServer3D::get_singleton()->get_debug_navigation_geometry_face_color();
		Color debug_navigation_geometry_face_area_color = NavigationServer3D::get_singleton()->get_debug_navigation_geometry_face_area_color();
		Color hole_color = Color(0, 0, 0, 0);
		Color polygon_color = debug_navigation_geometry_face_color;

		RandomPCG rand;
		Vector<Color> face_color_array;
		face_color_array.resize(polygon_count * 3);
		uint32_t default_navigation_layers = navigationregion->get_navigation_layers();

		for (int i = 0; i < polygon_count; i++) {
			if (has_polygon_meta && navigationmesh->get_polygon_meta(i) != default_navigation_layers) {
				if (navigationmesh->get_polygon_meta(i) == 0) {
					polygon_color = hole_color; // Temporary hole.
				} else {
					// Color faces that were generated because of area meshes differently using vertex colors.
					polygon_color = debug_navigation_geometry_face_area_color;
				}
			} else if (enabled_geometry_face_random_color) {
				// If enabled add vertex colors to colorize all other faces differently.
				polygon_color.set_hsv(debug_navigation_geometry_face_color.get_h() + rand.random(-1.0, 1.0) * 0.1, debug_navigation_geometry_face_color.get_s(), debug_navigation_geometry_face_color.get_v() + rand.random(-1.0, 1.0) * 0.2);
				polygon_color.a = debug_navigation_geometry_face_color.a;
			} else {
				polygon_color = debug_navigation_geometry_face_color;
			}

			face_color_array.push_back(polygon_color);
			face_color_array.push_back(polygon_color);
			face_color_array.push_back(polygon_color);
		}

		face_mesh_array[Mesh::ARRAY_COLOR] = face_color_array;
	}

	debug_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, face_mesh_array);
	p_gizmo->add_mesh(debug_mesh, navigationregion->is_enabled() ? get_material("face_material", p_gizmo) : get_material("face_material_disabled", p_gizmo));

	// if enabled build geometry edge line surface
	bool enabled_edge_lines = NavigationServer3D::get_singleton()->get_debug_navigation_enable_edge_lines();
	if (enabled_edge_lines) {
		Vector<Vector3> line_vertex_array;
		line_vertex_array.resize(polygon_count * 6);

		for (int i = 0; i < polygon_count; i++) {
			Vector<int> polygon = navigationmesh->get_polygon(i);

			line_vertex_array.push_back(vertices[polygon[0]]);
			line_vertex_array.push_back(vertices[polygon[1]]);
			line_vertex_array.push_back(vertices[polygon[1]]);
			line_vertex_array.push_back(vertices[polygon[2]]);
			line_vertex_array.push_back(vertices[polygon[2]]);
			line_vertex_array.push_back(vertices[polygon[0]]);
		}

		p_gizmo->add_lines(line_vertex_array, navigationregion->is_enabled() ? get_material("edge_material", p_gizmo) : get_material("edge_material_disabled", p_gizmo));
	}

	if (get_state() == EditorNode3DGizmoPlugin::HIDDEN || !p_gizmo->is_selected()) {
		Node *debug_holder = navigationregion->find_child("_debug_holder", false, false);
		if (debug_holder != nullptr) {
			navigationregion->remove_child(debug_holder);
			debug_holder->queue_free();
		}
	} else {
		Array debug_data = navigationmesh->_get_debug_data();

		if (debug_data.size() > 0) {
			Node3D *debug_holder = Object::cast_to<Node3D>(navigationregion->find_child("_debug_holder", false, false));
			if (debug_holder == nullptr) {
				debug_holder = memnew(Node3D);
				debug_holder->set_name("_debug_holder");
				navigationregion->add_child(debug_holder, false, Node::INTERNAL_MODE_BACK);
				// NOTE: not setting any owner to prevent these debug nodes from being saved.
			} else {
				TypedArray<Node> nodes = debug_holder->get_children(true);
				if (nodes.size() > 0) {
					for (Variant &v : nodes) {
						Node *node = Object::cast_to<Node>(v);
						if (node) {
							node->queue_free();
							debug_holder->remove_child(node);
						}
					}
				}
			}

			Array bake_ids = navigationmesh->get_area_bake_ids();
			bool use_bake_id = bake_ids.size() == debug_data.size();

			for (int i = 0; i < debug_data.size(); i++) {
				Vector3 pos = debug_data[i];

				Label3D *area_index_label = memnew(Label3D);
				String text;
				if (use_bake_id) {
					text = bake_ids[i];
					if (!text.is_empty()) {
						text = vformat("%s (%d)", text, i);
					}
				}
				if (text.is_empty()) {
					text = vformat("%d", i);
				}
				area_index_label->set_text(text);
				area_index_label->set_draw_flag(Label3D::DrawFlags::FLAG_DISABLE_DEPTH_TEST, true);
				area_index_label->set_font_size(200);
				area_index_label->set_outline_size(40);
				area_index_label->set_position(pos);
				area_index_label->rotate_x(-(Math::PI / 2)); // -90 degrees.
				debug_holder->add_child(area_index_label);
				area_index_label->set_owner(debug_holder);
			}
		}
	}
}
