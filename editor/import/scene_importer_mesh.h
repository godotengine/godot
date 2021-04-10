/*************************************************************************/
/*  scene_importer_mesh.h                                                */
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

#ifndef EDITOR_SCENE_IMPORTER_MESH_H
#define EDITOR_SCENE_IMPORTER_MESH_H

#include "core/io/resource.h"
#include "scene/resources/concave_polygon_shape_3d.h"
#include "scene/resources/convex_polygon_shape_3d.h"
#include "scene/resources/mesh.h"
#include "scene/resources/navigation_mesh.h"
// The following classes are used by importers instead of ArrayMesh and MeshInstance3D
// so the data is not registered (hence, quality loss), importing happens faster and
// its easier to modify before saving

class EditorSceneImporterMesh : public Resource {
	GDCLASS(EditorSceneImporterMesh, Resource)

	struct Surface {
		Mesh::PrimitiveType primitive;
		Array arrays;
		struct BlendShape {
			Array arrays;
		};
		Vector<BlendShape> blend_shape_data;
		struct LOD {
			Vector<int> indices;
			float distance;
		};
		Vector<LOD> lods;
		Ref<Material> material;
		String name;
	};
	Vector<Surface> surfaces;
	Vector<String> blend_shapes;
	Mesh::BlendShapeMode blend_shape_mode = Mesh::BLEND_SHAPE_MODE_NORMALIZED;

	Ref<ArrayMesh> mesh;

	Ref<EditorSceneImporterMesh> shadow_mesh;

	Size2i lightmap_size_hint;
	Basis compute_rotation_matrix_from_ortho_6d(Vector3 p_x_raw, Vector3 y_raw);

protected:
	void _set_data(const Dictionary &p_data);
	Dictionary _get_data() const;

	static void _bind_methods();

public:
	void add_blend_shape(const String &p_name);
	int get_blend_shape_count() const;
	String get_blend_shape_name(int p_blend_shape) const;

	void add_surface(Mesh::PrimitiveType p_primitive, const Array &p_arrays, const Array &p_blend_shapes = Array(), const Dictionary &p_lods = Dictionary(), const Ref<Material> &p_material = Ref<Material>(), const String &p_name = String());
	int get_surface_count() const;

	void set_blend_shape_mode(Mesh::BlendShapeMode p_blend_shape_mode);
	Mesh::BlendShapeMode get_blend_shape_mode() const;

	Mesh::PrimitiveType get_surface_primitive_type(int p_surface);
	String get_surface_name(int p_surface) const;
	Array get_surface_arrays(int p_surface) const;
	Array get_surface_blend_shape_arrays(int p_surface, int p_blend_shape) const;
	int get_surface_lod_count(int p_surface) const;
	Vector<int> get_surface_lod_indices(int p_surface, int p_lod) const;
	float get_surface_lod_size(int p_surface, int p_lod) const;
	Ref<Material> get_surface_material(int p_surface) const;

	void set_surface_material(int p_surface, const Ref<Material> &p_material);

	void generate_lods();

	void create_shadow_mesh();
	Ref<EditorSceneImporterMesh> get_shadow_mesh() const;

	Vector<Face3> get_faces() const;
	Vector<Ref<Shape3D>> convex_decompose() const;
	Ref<Shape3D> create_trimesh_shape() const;
	Ref<NavigationMesh> create_navigation_mesh();
	Error lightmap_unwrap_cached(const Transform &p_base_transform, float p_texel_size, const Vector<uint8_t> &p_src_cache, Vector<uint8_t> &r_dst_cache);

	void set_lightmap_size_hint(const Size2i &p_size);
	Size2i get_lightmap_size_hint() const;

	bool has_mesh() const;
	Ref<ArrayMesh> get_mesh(const Ref<Mesh> &p_base = Ref<Mesh>());
	void clear();
};
#endif // EDITOR_SCENE_IMPORTER_MESH_H
