/**************************************************************************/
/*  merging_tool.h                                                        */
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

#ifndef MERGING_TOOL_H
#define MERGING_TOOL_H

#include "core/local_vector.h"
#include "core/vector.h"
#include "scene/resources/mesh.h"

class MeshInstance;
class GeometryInstance;
class CSGShape;
class SurfaceTool;
struct CSGBrush;

#ifdef DEV_ENABLED
// Only enable this for development testing.
// #define GODOT_MERGING_VERBOSE
#endif

// NOTE : These merging and joining functions DO NOT move children, or delete source nodes.
// That is the responsibility of the caller.

class MergingTool {
public:
	////////////////////////////////////////////////////////////////////////////////////
	// WRAPPED versions are accessible via script via MeshInstance.
	// These have to cope with Variants as lists of MeshInstances is not easy from script.
	static bool wrapped_merge_meshes(MeshInstance &r_dest_mi, Vector<Variant> p_list, bool p_use_global_space, bool p_check_compatibility, bool p_shadows_only, Mesh::StorageMode p_storage_mode);
	static bool wrapped_split_by_surface(const MeshInstance &p_source_mi, Vector<Variant> p_destination_mesh_instances, Mesh::StorageMode p_storage_mode);

	////////////////////////////////////////////////////////////////////////////////////

	// Are two mesh instances mergeable with each other?
	static bool is_mergeable_with(const MeshInstance &p_mi, const MeshInstance &p_other, bool p_check_surface_material_match);
	static bool is_shadow_mergeable_with(const MeshInstance &p_mi, const MeshInstance &p_other);

	// Merges all mesh details.
	static bool merge_meshes(MeshInstance &r_dest_mi, Vector<MeshInstance *> p_list, bool p_use_global_space, bool p_check_compatibility);

	// Join all surfaces into one ubermesh.
	static bool join_meshes(MeshInstance &r_dest_mi, Vector<MeshInstance *> p_list);

	// Adds a surface from one mesh to another.
	static bool join_mesh_surface(const MeshInstance &p_source_mi, uint32_t p_source_surface_id, MeshInstance &r_dest_mi);

	// Only concerned with data necessary for shadow proxy - opaque tris, no normals / tangents / uvs etc.
	static bool merge_shadow_meshes(MeshInstance &r_dest_mi, Vector<MeshInstance *> p_list, bool p_use_global_space, bool p_check_compatibility);

	// For splitting a MeshInstance with multiple surfaces to a MeshInstance per surface.
	static bool split_surface_to_mesh_instance(const MeshInstance &p_source_mi, int p_surface_id, MeshInstance &r_mi);

	// Convert a CSG surface to MeshInstance.
	static bool split_csg_surface_to_mesh_instance(const CSGShape &p_shape, MeshInstance &r_mi, const Ref<ArrayMesh> &p_array_mesh, CSGBrush *p_brush, int p_surface);

	// Remove degenerate triangles.
	static bool clean_mesh_instance(MeshInstance &p_mi);

	static void split_mesh_instance_by_locality(MeshInstance &r_mi, const AABB &p_bound, uint32_t p_splits_horz, uint32_t p_splits_vert, uint32_t p_min_split_poly_count);

	// For debugging purposes.
	static void debug_mesh_instance(const MeshInstance &p_mi);
#ifdef DEV_ENABLED
	static void debug_branch(Node *p_node, const char *p_title = nullptr, int p_depth = 0);
#endif
#ifdef TOOLS_ENABLED
	static void append_editor_description(Node *p_node, String p_string, Node *p_node_named = nullptr);
#endif

	// Helper functions (used from MergeGroup).
	static void _set_owner_logged(Node *p_node, Node *p_owner);
	static void _reparent(Node *p_branch, Node *p_new_parent, Node *p_new_owner);
	static void _invalidate_owner_recursive(Node *p_node, Node *p_old_owner, Node *p_new_owner);
	static bool _node_has_valid_children(Node *p_node);
	static void _mesh_set_storage_mode(Mesh *p_mesh, Mesh::StorageMode p_mode);

private:
	static void _reparent_subscene_send_new_owner(Node *p_node, Node *p_new_owner);

	static void _copy_mesh_instance_settings(const MeshInstance &p_source, MeshInstance &r_dest, bool p_copy_transform, bool p_copy_materials);
	static bool _is_mergeable_with_common(const MeshInstance &p_mi, const MeshInstance &p_other);
	static bool _is_shadow_mergeable(const MeshInstance &p_mi);
	static bool _is_material_opaque(const Ref<Material> &p_mat);
	static bool _ensure_indices_valid(LocalVector<int> &r_indices, const PoolVector<Vector3> &p_verts);
	static bool _check_for_valid_indices(const LocalVector<int> &p_inds, const PoolVector<Vector3> &p_verts, LocalVector<int> *r_inds);
	static bool _triangle_is_degenerate(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, real_t p_epsilon);
	static int _clean_mesh_surface(const String &p_source_name, const Transform &p_xform, Ref<Mesh> &p_rmesh, int p_surface_id, Ref<ArrayMesh> r_dest_mesh);
	static void _copy_geometry_instance_settings(const GeometryInstance &p_source, MeshInstance &r_dest, bool p_copy_transform);
	static void _set_rmesh_material(MeshInstance &r_mi, Ref<Mesh> r_rmesh, int p_surface_id, Ref<Material> p_material);

	static void _split_mesh_instance_by_locality(const SurfaceTool &p_st_main, const MeshInstance &p_source_mi, const LocalVector<uint32_t> &p_tri_ids, uint32_t p_local_id, uint32_t p_surface_id, uint32_t p_x, uint32_t p_y, uint32_t p_z);

	static void _merge_log(String p_string, int p_priority = 1);
};

#endif // MERGING_TOOL_H
