/**************************************************************************/
/*  merging_tool.cpp                                                      */
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

#include "merging_tool.h"

#include "core/engine.h"
#include "core/os/os.h"
#include "scene/3d/mesh_instance.h"
#include "scene/resources/surface_tool.h"

#include "modules/modules_enabled.gen.h" // For csg.
#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif

bool MergingTool::wrapped_split_by_surface(const MeshInstance &p_source_mi, Vector<Variant> p_destination_mesh_instances, Mesh::StorageMode p_storage_mode) {
	ERR_FAIL_COND_V_MSG(!p_source_mi.is_inside_tree(), false, "Source MeshInstance must be inside the SceneTree.");
	ERR_FAIL_COND_V_MSG(!p_source_mi.get_mesh().is_valid(), false, "Source MeshInstance must have a valid mesh to split.");

	// For simplicity we are requiring that the destination MeshInstances have the same parent
	// as the source. This means we can use identical transforms.
	Node *parent = p_source_mi.get_parent();
	ERR_FAIL_NULL_V_MSG(parent, false, "Source MeshInstance must have a parent node.");

	// Bound function only support variants, so we need to convert to a list of MeshInstances.
	Vector<MeshInstance *> mis;

	for (int n = 0; n < p_destination_mesh_instances.size(); n++) {
		MeshInstance *mi = Object::cast_to<MeshInstance>(p_destination_mesh_instances[n]);

		ERR_FAIL_NULL_V_MSG(mi, false, "Can only be split to MeshInstances.");
		ERR_FAIL_COND_V_MSG(mi == &p_source_mi, false, "Source MeshInstance cannot be a destination.");
		ERR_FAIL_COND_V_MSG(mi->get_parent() != parent, false, "Destination MeshInstances must be siblings of the source MeshInstance.");
		mis.push_back(mi);
	}

	ERR_FAIL_COND_V_MSG(mis.size() != p_source_mi.get_mesh()->get_surface_count(), false, "Number of source surfaces and number of destination MeshInstances must match.");

	// Go through each surface, and fill the relevant mesh instance.
	const Mesh *source_mesh = p_source_mi.get_mesh().ptr();
	DEV_ASSERT(source_mesh);

	ERR_FAIL_COND_V_MSG(source_mesh->get_surface_count() <= 1, false, "Source MeshInstance must contain multiple surfaces.");

	for (int s = 0; s < source_mesh->get_surface_count(); s++) {
		MeshInstance &dest_mi = *mis[s];
		if (split_surface_to_mesh_instance(p_source_mi, s, dest_mi)) {
			// Change storage mode if required.
#ifdef TOOLS_ENABLED
			Ref<Mesh> rmesh = dest_mi.get_mesh();
			if (rmesh.is_valid()) {
				_mesh_set_storage_mode(rmesh.ptr(), p_storage_mode);
			}
#endif
		}
	}

	return true;
}

bool MergingTool::wrapped_merge_meshes(MeshInstance &r_dest_mi, Vector<Variant> p_list, bool p_use_global_space, bool p_check_compatibility, bool p_shadows_only, Mesh::StorageMode p_storage_mode) {
	// Bound function only support variants, so we need to convert to a list of MeshInstances.
	Vector<MeshInstance *> mis;

	for (int n = 0; n < p_list.size(); n++) {
		MeshInstance *mi = Object::cast_to<MeshInstance>(p_list[n]);
		if (mi) {
			ERR_FAIL_COND_V_MSG(mi == &r_dest_mi, false, "Destination MeshInstance cannot be a source.");
			mis.push_back(mi);
		} else {
			ERR_PRINT("Only MeshInstances can be merged.");
		}
	}

	ERR_FAIL_COND_V_MSG(!mis.size(), false, "Array contains no MeshInstances");

	bool result;
	if (p_shadows_only) {
		result = merge_shadow_meshes(r_dest_mi, mis, p_use_global_space, p_check_compatibility);
	} else {
		result = merge_meshes(r_dest_mi, mis, p_use_global_space, p_check_compatibility);
	}

	// Change storage mode if required.
	if (result) {
#ifdef TOOLS_ENABLED
		Ref<Mesh> rmesh = r_dest_mi.get_mesh();
		if (rmesh.is_valid()) {
			_mesh_set_storage_mode(rmesh.ptr(), p_storage_mode);
		}
#endif
	}

	return result;
}

bool MergingTool::_is_material_opaque(const Ref<Material> &p_mat) {
	if (p_mat.is_null()) {
		return true;
	}

	Ref<SpatialMaterial> material = p_mat;
	if (material.is_null()) {
		// Shaders not yet supported.
		return false;
	}

	if (material->get_feature(SpatialMaterial::FEATURE_TRANSPARENT)) {
		return false;
	}

	// Not sure if this can only occur with FEATURE_TRANSPARENT?
	if (material->get_flag(SpatialMaterial::FLAG_USE_ALPHA_SCISSOR)) {
		return false;
	}

	// Only supporting default cull mode for now.
	if (material->get_cull_mode() != SpatialMaterial::CULL_BACK) {
		return false;
	}

	return true;
}

bool MergingTool::_is_shadow_mergeable(const MeshInstance &p_mi) {
	if (p_mi.get_cast_shadows_setting() == GeometryInstance::ShadowCastingSetting::SHADOW_CASTING_SETTING_OFF) {
		return false;
	}

	if (!_is_material_opaque(p_mi.get_material_overlay())) {
		return false;
	}

	if (!_is_material_opaque(p_mi.get_material_override())) {
		return false;
	}

	int num_surfaces = p_mi.get_mesh()->get_surface_count();
	for (int n = 0; n < num_surfaces; n++) {
		if (!_is_material_opaque(p_mi.get_active_material(n))) {
			return false;
		}
	}

	return true;
}

bool MergingTool::is_shadow_mergeable_with(const MeshInstance &p_mi, const MeshInstance &p_other) {
	// Various settings that must match.
	if (!_is_mergeable_with_common(p_mi, p_other)) {
		return false;
	}

	if (!_is_shadow_mergeable(p_mi) || !_is_shadow_mergeable(p_other)) {
		return false;
	}

	return true;
}

bool MergingTool::_is_mergeable_with_common(const MeshInstance &p_mi, const MeshInstance &p_other) {
	if (!p_mi.get_mesh().is_valid() || !p_other.get_mesh().is_valid()) {
		return false;
	}
	if (!p_mi.is_merging_allowed() || !p_other.is_merging_allowed()) {
		return false;
	}

	if (p_mi.get_cast_shadows_setting() != p_other.get_cast_shadows_setting()) {
		return false;
	}
	if (p_mi.is_visible() != p_other.is_visible()) {
		return false;
	}
	if (p_mi.is_visible_in_tree() != p_other.is_visible_in_tree()) {
		return false;
	}
	if (p_mi.get_layer_mask() != p_other.get_layer_mask()) {
		return false;
	}

	if (p_mi.get_portal_mode() != p_other.get_portal_mode()) {
		return false;
	}
	if (p_mi.get_include_in_bound() != p_other.get_include_in_bound()) {
		return false;
	}
	if (p_mi.get_portal_autoplace_priority() != p_other.get_portal_autoplace_priority()) {
		return false;
	}
	if (p_mi.get_extra_cull_margin() != p_other.get_extra_cull_margin()) {
		return false;
	}

	return true;
}

bool MergingTool::is_mergeable_with(const MeshInstance &p_mi, const MeshInstance &p_other, bool p_check_surface_material_match) {
	if (!_is_mergeable_with_common(p_mi, p_other)) {
		return false;
	}

	// Various settings that must match.
	if (p_mi.is_visible() != p_other.is_visible()) {
		return false;
	}
	if (p_mi.get_material_overlay() != p_other.get_material_overlay()) {
		return false;
	}
	if (p_mi.get_material_override() != p_other.get_material_override()) {
		return false;
	}
	if (p_mi.get_flag(GeometryInstance::FLAG_USE_BAKED_LIGHT) != p_other.get_flag(GeometryInstance::FLAG_USE_BAKED_LIGHT)) {
		return false;
	}
	if (p_mi.get_generate_lightmap() != p_other.get_generate_lightmap()) {
		return false;
	}
	if (p_mi.get_lightmap_scale() != p_other.get_lightmap_scale()) {
		return false;
	}

	if (p_check_surface_material_match) {
		Ref<Mesh> rmesh_a = p_mi.get_mesh();
		Ref<Mesh> rmesh_b = p_other.get_mesh();

		int num_surfaces = rmesh_a->get_surface_count();
		if (num_surfaces != rmesh_b->get_surface_count()) {
			return false;
		}

		for (int n = 0; n < num_surfaces; n++) {
			// Materials must match.
			if (p_mi.get_active_material(n) != p_other.get_active_material(n)) {
				return false;
			}

			// Formats must match.
			uint32_t format_a = rmesh_a->surface_get_format(n);
			uint32_t format_b = rmesh_b->surface_get_format(n);

			if (format_a != format_b) {
				return false;
			}
		}
	}

	// NOTE : These three commented out sections below are more conservative
	// checks for whether to allow mesh merging. I am not absolutely sure a priori
	// how conservative we need to be, so we can further enable this if testing
	// shows they are required.

	//	if (get_surface_material_count() != p_other.get_surface_material_count()) {
	//		return false;
	//	}

	//	for (int n = 0; n < get_surface_material_count(); n++) {
	//		if (get_surface_material(n) != p_other.get_surface_material(n)) {
	//			return false;
	//		}
	//	}

	// test only allow identical meshes
	//	if (get_mesh() != p_other.get_mesh()) {
	//		return false;
	//	}

	return true;
}

void MergingTool::split_mesh_instance_by_locality(MeshInstance &r_mi, const AABB &p_bound, uint32_t p_splits_horz, uint32_t p_splits_vert, uint32_t p_min_split_poly_count) {
	Ref<Mesh> rmesh = r_mi.get_mesh();
	if (!rmesh.is_valid()) {
		return;
	}

	// Need a parent to attach results to.
	if (!r_mi.get_parent()) {
		return;
	}

	Vector3 cell_size = p_bound.size;
	cell_size.x /= p_splits_horz;
	cell_size.y /= p_splits_vert;
	cell_size.z /= p_splits_horz;

	DEV_ASSERT(p_splits_horz);
	DEV_ASSERT(p_splits_vert);
	int splits_horz_minus_one = p_splits_horz - 1;
	int splits_vert_minus_one = p_splits_vert - 1;

	// This is to prevent a warning as error in release builds, as this is only used
	// for DEV_ASSERT
#ifdef DEV_ENABLED
	uint32_t total_zones = p_splits_horz * p_splits_horz * p_splits_vert;
#endif

	AABB aabb;
	aabb.size = cell_size;

	ERR_FAIL_COND(!r_mi.is_inside_tree());
	Transform xform = r_mi.get_global_transform();

	SurfaceTool st_main;
	for (int s = 0; s < rmesh->get_surface_count(); s++) {
		st_main.create_from(rmesh, s);

		uint32_t tri_count = st_main.get_num_draw_vertices() / 3;

		// Bug .. we want to keep this surface in this case! and not delete the whole mesh instance?
		// at the moment this ASSUMES there is only one surface.
		if (tri_count < p_min_split_poly_count) {
			continue;
		}

		// Input for bounds routine should be deindexed.
		st_main.deindex();

		// Assign each triangle to a split zone.
		uint32_t num_tris = st_main.vertex_array.size() / 3;

		Vector3 v[3];
		const SurfaceTool::Vertex *input = st_main.vertex_array.ptr();

		LocalVector<uint32_t> tri_ids;
		tri_ids.resize(num_tris);

		for (uint32_t t = 0; t < num_tris; t++) {
			// Split in world space.
			Vector3 center;

			for (int c = 0; c < 3; c++) {
				v[c] = input->vertex;
				input++;
				v[c] = xform.xform(v[c]);
				center += v[c];
			}
			center /= 3;

			// Get relative to bound.
			center -= p_bound.position;

			// Find the x y z .
			center /= cell_size;
			int x = center.x;
			int y = center.y;
			int z = center.z;
			x = CLAMP(x, 0, splits_horz_minus_one);
			y = CLAMP(y, 0, splits_vert_minus_one);
			z = CLAMP(z, 0, splits_horz_minus_one);

			uint32_t id = (x + (z * p_splits_horz) + (y * p_splits_horz * p_splits_vert));
			tri_ids[t] = id;
			DEV_ASSERT(id < total_zones);
		}

		for (uint32_t x = 0; x < p_splits_horz; x++) {
			for (uint32_t y = 0; y < p_splits_vert; y++) {
				for (uint32_t z = 0; z < p_splits_horz; z++) {
					uint32_t id = (x + (z * p_splits_horz) + (y * p_splits_horz * p_splits_vert));
					_split_mesh_instance_by_locality(st_main, r_mi, tri_ids, id, s, x, y, z);
				}
			}
		}

	} // for s
}

void MergingTool::_split_mesh_instance_by_locality(const SurfaceTool &p_st_main, const MeshInstance &p_source_mi, const LocalVector<uint32_t> &p_tri_ids, uint32_t p_local_id, uint32_t p_surface_id, uint32_t p_x, uint32_t p_y, uint32_t p_z) {
	SurfaceTool st;
	int num_inds = st.create_from_subset(p_st_main, p_tri_ids, p_local_id);

	// This could be quite common, bounds with no triangles within.
	if (!num_inds) {
		return;
	}

	Node *parent = p_source_mi.get_parent();
	DEV_ASSERT(parent);

	// Create a mesh instance to hold this "zone".
	MeshInstance *sib = memnew(MeshInstance);
	parent->add_child(sib);
	sib->set_owner(p_source_mi.get_owner());

	String new_name = String(p_source_mi.get_name());
	if (p_surface_id) {
		new_name += " _surf_" + itos(p_surface_id);
	}
	new_name += " split (" + itos(p_x) + "," + itos(p_y) + "," + itos(p_z) + ")";

	sib->set_name(new_name);

#ifdef TOOLS_ENABLED
#if 0
	_merge_log("_split_mesh_instance_by_locality " + itos(num_inds) + " inds : " + new_name);
#endif
#endif

	Ref<ArrayMesh> am;
	am.instance();

	Array arr = st.commit_to_arrays();
	am->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arr, Array(), Mesh::ARRAY_COMPRESS_DEFAULT);

	// Set all the surfaces on the mesh.
	sib->set_mesh(am);

	_copy_mesh_instance_settings(p_source_mi, *sib, true, true);
}

bool MergingTool::clean_mesh_instance(MeshInstance &p_mi) {
	Ref<Mesh> rmesh = p_mi.get_mesh();
	ERR_FAIL_COND_V(!rmesh.is_valid(), false);
	ERR_FAIL_COND_V(!p_mi.is_inside_tree(), false);

	Transform tr = p_mi.get_global_transform();
	String name = p_mi.get_name();

	bool data_changed = false;

	Ref<ArrayMesh> am;
	am.instance();

	int inds_removed = 0;

	for (int s = 0; s < rmesh->get_surface_count(); s++) {
		inds_removed += _clean_mesh_surface(name, tr, rmesh, s, am);
	}

	if (inds_removed) {
		_merge_log("cleaning MeshInstance \"" + p_mi.get_name() + "\" removed " + itos(inds_removed) + " indices.", 2);
		p_mi.set_mesh(am);
	}

	return data_changed;
}

int MergingTool::_clean_mesh_surface(const String &p_source_name, const Transform &p_xform, Ref<Mesh> &p_rmesh, int p_surface_id, Ref<ArrayMesh> r_dest_mesh) {
	Array arrays = p_rmesh->surface_get_arrays(p_surface_id);
	LocalVector<Vector3> verts = PoolVector<Vector3>(arrays[VS::ARRAY_VERTEX]);
	if (!verts.size()) {
		// Early out if there are no vertices, no point in doing anything else.
		return 0;
	}
	LocalVector<int> indices = PoolVector<int>(arrays[VS::ARRAY_INDEX]);

	// Transform verts to world space.
	for (uint32_t n = 0; n < verts.size(); n++) {
		verts[n] = p_xform.xform(verts[n]);
	}

	// Special case, if no indices, create some.
	unsigned int num_indices_before = indices.size();
	if (!_ensure_indices_valid(indices, verts)) {
#ifdef TOOLS_ENABLED
		_merge_log("\tignoring INVALID TRIANGLES (duplicate indices or zero area triangle) detected in " + p_source_name + ", num inds before / after " + itos(num_indices_before) + " / " + itos(indices.size()));
#endif

		// Save the modified index array.
		arrays[VS::ARRAY_INDEX] = PoolVector<int>(indices);

		// Note we aren't removing the unused verts here, to save hassle, but hopefully there won't be too many.
		r_dest_mesh->add_surface_from_arrays(p_rmesh->surface_get_primitive_type(p_surface_id), arrays);
		r_dest_mesh->surface_set_material(p_surface_id, p_rmesh->surface_get_material(p_surface_id));

		// Returns true if data changed.
		if (indices.size() >= num_indices_before) {
			ERR_PRINT_ONCE("Indices after cleaning is higher than before.");
			return 1;
		}
		return num_indices_before - indices.size();
	}
	// Still add the surface, as a later one may be modified.
	r_dest_mesh->add_surface_from_arrays(p_rmesh->surface_get_primitive_type(p_surface_id), arrays);
	r_dest_mesh->surface_set_material(p_surface_id, p_rmesh->surface_get_material(p_surface_id));

	return 0;
}

bool MergingTool::_ensure_indices_valid(LocalVector<int> &r_indices, const PoolVector<Vector3> &p_verts) {
	// No indices? create some.
	if (!r_indices.size()) {
#ifdef TOOLS_ENABLED
		_merge_log("\t\t\t\tindices are blank, creating...");
#endif

		// Indices are blank!! Let's create some, assuming the mesh is using triangles.
		r_indices.resize(p_verts.size());

		// This is assuming each triangle vertex is unique.
		for (unsigned int n = 0; n < r_indices.size(); n++) {
			r_indices[n] = n;
		}
	}

	if (!_check_for_valid_indices(r_indices, p_verts, nullptr)) {
		LocalVector<int> new_inds;
		_check_for_valid_indices(r_indices, p_verts, &new_inds);

		// Copy the new indices.
		r_indices = new_inds;

		return false;
	}

	return true;
}

// Check for invalid tris, or make a list of the valid triangles, depending on whether r_inds is set.
bool MergingTool::_check_for_valid_indices(const LocalVector<int> &p_inds, const PoolVector<Vector3> &p_verts, LocalVector<int> *r_inds) {
	int nTris = p_inds.size();
	nTris /= 3;
	int indCount = 0;

	for (int t = 0; t < nTris; t++) {
		int i0 = p_inds[indCount++];
		int i1 = p_inds[indCount++];
		int i2 = p_inds[indCount++];

		bool ok = true;

		// If the indices are the same, the triangle is invalid.
		if (i0 == i1) {
			ok = false;
		}
		if (i1 == i2) {
			ok = false;
		}
		if (i0 == i2) {
			ok = false;
		}

		// Check positions.
		if (ok) {
			// Vertex positions.
			const Vector3 &p0 = p_verts[i0];
			const Vector3 &p1 = p_verts[i1];
			const Vector3 &p2 = p_verts[i2];

			// If the area is zero, the triangle is invalid (and will crash xatlas if we use it).
			if (_triangle_is_degenerate(p0, p1, p2, 0.00001)) {
#ifdef TOOLS_ENABLED
				_merge_log("\t\tdetected zero area triangle, ignoring");
#endif
				ok = false;
			}
		}

		if (ok) {
			// If the triangle is ok, we will output it if we are outputting.
			if (r_inds) {
				r_inds->push_back(i0);
				r_inds->push_back(i1);
				r_inds->push_back(i2);
			}
		} else {
			// If triangle not ok, return failed check if we are not outputting.
			if (!r_inds) {
				return false;
			}
		}
	}

	return true;
}

bool MergingTool::_triangle_is_degenerate(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, real_t p_epsilon) {
	// Not interested in the actual area, but numerical stability.
	Vector3 edge1 = p_b - p_a;
	Vector3 edge2 = p_c - p_a;

	// For numerical stability keep these values reasonably high.
	edge1 *= 1024.0;
	edge2 *= 1024.0;

	Vector3 vec = edge1.cross(edge2);
	real_t sl = vec.length_squared();

	if (sl <= p_epsilon) {
		return true;
	}

	return false;
}

// If p_check_compatibility is set to false you MUST have performed a prior check using
// is_shadow_mergeable_with, otherwise you could get mismatching surface formats leading to graphical errors etc.
bool MergingTool::merge_shadow_meshes(MeshInstance &r_dest_mi, Vector<MeshInstance *> p_list, bool p_use_global_space, bool p_check_compatibility) {
	ERR_FAIL_COND_V(p_list.size() < 1, false);

	// Use the first mesh instance to get common data like number of surfaces.
	const MeshInstance *first = p_list[0];

	// Mesh compatibility checking. This is relatively expensive, so if done already (e.g. in Room system)
	// this step can be avoided.
	LocalVector<bool> compat_list;
	if (p_check_compatibility) {
		compat_list.resize(p_list.size());

		for (int n = 0; n < p_list.size(); n++) {
			compat_list[n] = false;
		}

		compat_list[0] = true;

		for (uint32_t n = 1; n < compat_list.size(); n++) {
			compat_list[n] = is_shadow_mergeable_with(*first, *p_list[n]);

			if (compat_list[n] == false) {
				WARN_PRINT("MeshInstance " + p_list[n]->get_name() + " is incompatible for shadow merging with " + first->get_name() + ", ignoring.");
			}
		}
	}

	Ref<ArrayMesh> am;
	am.instance();

	// If we want a local space result, we need the world space transform of this MeshInstance
	// available to back transform verts from world space.
	Transform dest_tr_inv;
	if (!p_use_global_space) {
		if (r_dest_mi.is_inside_tree()) {
			dest_tr_inv = r_dest_mi.get_global_transform();
			dest_tr_inv.affine_invert();
		} else {
			WARN_PRINT("MeshInstance must be inside tree to merge using local space, falling back to global space.");
		}
	}

	SurfaceTool surface_tool;
	for (int n = 0; n < p_list.size(); n++) {
		// Ignore if the mesh is incompatible.
		if (p_check_compatibility && (!compat_list[n])) {
			continue;
		}
		MeshInstance *source_mi = p_list[n];
		Ref<Mesh> rmesh = source_mi->get_mesh();

		Transform adjustment_xform = dest_tr_inv * source_mi->get_global_transform();
		for (int s = 0; s < rmesh->get_surface_count(); s++) {
			surface_tool.append_from(rmesh, s, adjustment_xform);

#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				_merge_log("merging from \"" + source_mi->get_name() + "\" surf " + itos(s) + " to \"" + r_dest_mi.get_name() + "\"");
				MergingTool::append_editor_description(&r_dest_mi, "merging from", source_mi);
			}
#endif
		}
	} // for n through source meshes

	// We are only interested in position data for shadow proxy meshes, and indices if present.
	surface_tool._mask_format_flags(Mesh::ARRAY_FORMAT_VERTEX | Mesh::ARRAY_FORMAT_INDEX);

	Array arr = surface_tool.commit_to_arrays();
	am->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arr, Array(), Mesh::ARRAY_COMPRESS_DEFAULT);

	// Set all the surfaces on the mesh.
	r_dest_mi.set_mesh(am);

	_copy_geometry_instance_settings(*first, r_dest_mi, false);
	r_dest_mi.set_cast_shadows_setting(GeometryInstance::ShadowCastingSetting::SHADOW_CASTING_SETTING_SHADOWS_ONLY);

	// Don't want these set, they get set by the _copy_geometry_instance_settings call.
	r_dest_mi.set_material_overlay(Ref<Material>());
	r_dest_mi.set_material_override(Ref<Material>());

	return true;
}

void MergingTool::_mesh_set_storage_mode(Mesh *p_mesh, Mesh::StorageMode p_mode) {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_NULL(p_mesh);
		p_mesh->set_storage_mode(p_mode);
	}
#endif
}

bool MergingTool::split_surface_to_mesh_instance(const MeshInstance &p_source_mi, int p_surface_id, MeshInstance &r_mi) {
	SurfaceTool surface_tool;
	Ref<Mesh> rmesh = p_source_mi.get_mesh();
	if (!rmesh.is_valid()) {
		return false;
	}

	// Hard coded to local space for now.
	surface_tool.append_from(rmesh, p_surface_id, Transform());

	Ref<ArrayMesh> am;
	am.instance();
	_mesh_set_storage_mode(am.ptr(), Mesh::STORAGE_MODE_CPU);

	Array arr = surface_tool.commit_to_arrays();
	am->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arr, Array(), Mesh::ARRAY_COMPRESS_DEFAULT);

	r_mi.set_mesh(am);

	// Set the material on the new mesh instance.
	_set_rmesh_material(r_mi, r_mi.get_mesh(), 0, p_source_mi.get_active_material(p_surface_id));

	// Set some properties to match the source mesh.
	// As they are guaranteed siblings, the transform can be identical.
	_copy_geometry_instance_settings(p_source_mi, r_mi, true);

	return true;
}

#ifdef TOOLS_ENABLED
void MergingTool::append_editor_description(Node *p_node, String p_string, Node *p_node_named) {
	ERR_FAIL_NULL(p_node);
	String existing = p_node->get_editor_description();
	if (existing.size() > 512) {
		// limit the max length of the description to prevent things getting ridiculous
		return;
	}
	String add;
	if (existing.size()) {
		add += "\n";
	}
	add += p_string;
	if (p_node_named) {
		add += " \"" + p_node_named->get_name() + "\"";
	}
	p_node->set_editor_description(existing + add);
}
#endif

#ifdef DEV_ENABLED
void MergingTool::debug_branch(Node *p_node, const char *p_title, int p_depth) {
	if (OS::get_singleton()->is_stdout_verbose()) {
		if (p_title) {
			_merge_log(p_title);
		}

		if (p_node->is_queued_for_deletion()) {
			return;
		}

		String s;
		for (int n = 0; n < p_depth; n++) {
			s += "\t";
		}
		s += "\"" + p_node->get_name() + "\"\t";

		String filename = p_node->get_filename();
		if (filename.size()) {
			s += "[filename " + p_node->get_filename() + "] ";
		}

		s += "owner (";
		if (p_node->get_owner()) {
			s += p_node->get_owner()->get_name();
		} else {
			s += "NULL";
		}
		s += ")";
		_merge_log(s);

		for (int n = 0; n < p_node->get_child_count(); n++) {
			debug_branch(p_node->get_child(n), nullptr, p_depth + 1);
		}
	} // if verbose output
}
#endif

void MergingTool::debug_mesh_instance(const MeshInstance &p_mi) {
#ifdef DEV_ENABLED
	_merge_log("debug " + p_mi.get_name());
	Ref<Mesh> rmesh = p_mi.get_mesh();
	if (!rmesh.is_valid()) {
		_merge_log("\tinvalid mesh");
		return;
	}
	for (int s = 0; s < rmesh->get_surface_count(); s++) {
		_merge_log("\tsurf " + itos(s) + " inds " + itos(rmesh->surface_get_array_index_len(s)) + " verts " + itos(rmesh->surface_get_array_len(s)));
	}
#endif
}

bool MergingTool::join_mesh_surface(const MeshInstance &p_source_mi, uint32_t p_source_surface_id, MeshInstance &r_dest_mi) {
	Ref<Mesh> r_sourcemesh = p_source_mi.get_mesh();
	ERR_FAIL_COND_V(!r_sourcemesh.is_valid(), false);
	ERR_FAIL_COND_V((int)p_source_surface_id >= r_sourcemesh->get_surface_count(), false);

	// Note this can be NULL if the destination mesh instance contains no meshes yet.
	// We should deal with this case.
	Ref<ArrayMesh> ra_destmesh = r_dest_mi.get_mesh();
	if (!ra_destmesh.is_valid()) {
		ra_destmesh.instance();
		_mesh_set_storage_mode(ra_destmesh.ptr(), Mesh::STORAGE_MODE_CPU);
	}

	// Relative xform ..
	Transform relative_xform = r_dest_mi.get_global_transform().inverse() * p_source_mi.get_global_transform();

	SurfaceTool surface_tool;
	surface_tool.append_from(r_sourcemesh, p_source_surface_id, relative_xform);

	int new_surface_id = 0;
	if (ra_destmesh.is_valid()) {
		new_surface_id = ra_destmesh->get_surface_count();
		r_dest_mi.set_mesh(surface_tool.commit(ra_destmesh));
	} else {
		r_dest_mi.set_mesh(surface_tool.commit());
	}

	Ref<Mesh> new_rmesh = r_dest_mi.get_mesh();

	// If no surface has been added.
	ERR_FAIL_COND_V(new_rmesh->get_surface_count() <= new_surface_id, false);

	// Deal with materials.
	_set_rmesh_material(r_dest_mi, new_rmesh, new_surface_id, p_source_mi.get_active_material(p_source_surface_id));

	return true;
}

// No compat checking, no renaming.
bool MergingTool::join_meshes(MeshInstance &r_dest_mi, Vector<MeshInstance *> p_list) {
	if (p_list.size() < 1) {
		// Should not happen but just in case...
		return false;
	}

	// For future use of compatibility check.
	LocalVector<MeshInstance *> list = p_list;

	MeshInstance *first = list[0];

	// First copy the properties of the first meshinstance.
	_copy_mesh_instance_settings(*first, r_dest_mi, false, false);

	for (unsigned int n = 0; n < list.size(); n++) {
		MeshInstance *mi = list[n];
		Ref<Mesh> rmesh = mi->get_mesh();

		for (int s = 0; s < rmesh->get_surface_count(); s++) {
			if (MergingTool::join_mesh_surface(*mi, s, r_dest_mi)) {
#ifdef DEV_ENABLED
				_merge_log("joining \"" + mi->get_name() + "\" to \"" + r_dest_mi.get_name() + "\"");
#endif
#ifdef TOOLS_ENABLED
				if (Engine::get_singleton()->is_editor_hint()) {
					MergingTool::append_editor_description(&r_dest_mi, "joined to", mi);
				}
#endif
			} else {
#ifdef DEV_ENABLED
				_merge_log("failed to join \"" + mi->get_name() + "\" to \"" + r_dest_mi.get_name() + "\"");
#endif
			}
		}
	}

	return true;
}

// If p_check_compatibility is set to false you MUST have performed a prior check using
// is_mergeable_with, otherwise you could get mismatching surface formats leading to graphical errors etc.
bool MergingTool::merge_meshes(MeshInstance &r_dest_mi, Vector<MeshInstance *> p_list, bool p_use_global_space, bool p_check_compatibility) {
	ERR_FAIL_COND_V(p_list.size() < 1, false);

	// Use the first mesh instance to get common data like number of surfaces.
	const MeshInstance *first = p_list[0];
	Ref<Mesh> rmesh_first = first->get_mesh();
	if (!rmesh_first.is_valid()) {
		return false;
	}
	int surface_count = rmesh_first->get_surface_count();
	if (surface_count <= 0) {
#ifdef TOOLS_ENABLED
		_merge_log("merge_meshes : " + first->get_name() + " contains no surfaces, ignoring.");
#endif
		return false;
	}

	// Mesh compatibility checking. This is relatively expensive, so if done already (e.g. in Room system)
	// this step can be avoided.
	LocalVector<bool> compat_list;
	if (p_check_compatibility) {
		compat_list.resize(p_list.size());

		for (int n = 0; n < p_list.size(); n++) {
			compat_list[n] = false;
		}

		compat_list[0] = true;

		for (uint32_t n = 1; n < compat_list.size(); n++) {
			compat_list[n] = is_mergeable_with(*first, *p_list[n], true);

			if (compat_list[n] == false) {
				WARN_PRINT("MeshInstance " + p_list[n]->get_name() + " is incompatible for merging with " + first->get_name() + ", ignoring.");
			}
		}
	}

	Ref<ArrayMesh> am;
	am.instance();
	_mesh_set_storage_mode(am.ptr(), Mesh::STORAGE_MODE_CPU);

	// If we want a local space result, we need the world space transform of this MeshInstance
	// available to back transform verts from world space.
	Transform dest_tr_inv;
	if (!p_use_global_space) {
		if (r_dest_mi.is_inside_tree()) {
			dest_tr_inv = r_dest_mi.get_global_transform();
			dest_tr_inv.affine_invert();
		} else {
			WARN_PRINT("MeshInstance must be inside tree to merge using local space, falling back to global space.");
		}
	}

	for (int s = 0; s < surface_count; s++) {
		SurfaceTool surface_tool;

		for (int n = 0; n < p_list.size(); n++) {
			// Ignore if the mesh is incompatible.
			if (p_check_compatibility && (!compat_list[n])) {
				continue;
			}

			Ref<Mesh> rmesh = p_list[n]->get_mesh();

			Transform adjustment_xform = dest_tr_inv * p_list[n]->get_global_transform();
			surface_tool.append_from(rmesh, s, adjustment_xform);

#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				_merge_log("merging from \"" + p_list[n]->get_name() + "\" surf " + itos(s) + " to \"" + r_dest_mi.get_name() + "\"");
				MergingTool::append_editor_description(&r_dest_mi, "merging from", p_list[n]);
			}
#endif
		}

		Array arr = surface_tool.commit_to_arrays();
		am->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arr, Array(), Mesh::ARRAY_COMPRESS_DEFAULT);
	} // for s

	// Set all the surfaces on the mesh.
	r_dest_mi.set_mesh(am);

	// Set some properties to match the merged meshes.
	_copy_mesh_instance_settings(*first, r_dest_mi, false, true);

	return true;
}

bool MergingTool::split_csg_surface_to_mesh_instance(const CSGShape &p_shape, MeshInstance &r_mi, const Ref<ArrayMesh> &p_array_mesh, CSGBrush *p_brush, int p_surface) {
#ifdef MODULE_CSG_ENABLED

	SurfaceTool surface_tool;
	Ref<ArrayMesh> am;
	am.instance();

	Ref<Mesh> rmesh = p_array_mesh;

	// We are matching the local transforms of the source and destination, as they are always
	// siblings for now.
	surface_tool.append_from(rmesh, p_surface, Transform());

	Array arr = surface_tool.commit_to_arrays();

	am->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arr, Array(), Mesh::ARRAY_COMPRESS_DEFAULT);
	r_mi.set_mesh(am);

	// Set the material on the new mesh instance.
	if (p_surface < p_brush->materials.size()) {
		_set_rmesh_material(r_mi, r_mi.get_mesh(), 0, p_brush->materials[p_surface]);
	}

	// As they are guaranteed siblings, the transform can be identical.
	_copy_geometry_instance_settings(p_shape, r_mi, true);

	return true;
#else
	return false;
#endif
}

void MergingTool::_copy_mesh_instance_settings(const MeshInstance &p_source, MeshInstance &r_dest, bool p_copy_transform, bool p_copy_materials) {
	_copy_geometry_instance_settings(p_source, r_dest, p_copy_transform);

	if (p_copy_materials) {
		// Set merged materials.
		Ref<Mesh> rmesh = p_source.get_mesh();
		if (rmesh.is_valid()) {
			for (int n = 0; n < rmesh->get_surface_count(); n++) {
				_set_rmesh_material(r_dest, r_dest.get_mesh(), n, p_source.get_active_material(n));
			}
		}
	}
}

void MergingTool::_set_rmesh_material(MeshInstance &r_mi, Ref<Mesh> r_rmesh, int p_surface_id, Ref<Material> p_material) {
	// Here we can either set the material on the rmesh, or on the mesh instance.
	// Setting it directly in the rmesh seems more desired by users, but perhaps this could be
	// switchable?
	r_rmesh->surface_set_material(p_surface_id, p_material);
	// r_mi.set_surface_material(p_surface_id, p_material);
}

void MergingTool::_copy_geometry_instance_settings(const GeometryInstance &p_source, MeshInstance &r_dest, bool p_copy_transform) {
	// Set some properties to match the source mesh.
	r_dest.set_material_overlay(p_source.get_material_overlay());
	r_dest.set_material_override(p_source.get_material_override());
	r_dest.set_cast_shadows_setting(p_source.get_cast_shadows_setting());
	r_dest.set_flag(GeometryInstance::FLAG_USE_BAKED_LIGHT, p_source.get_flag(GeometryInstance::FLAG_USE_BAKED_LIGHT));

	r_dest.set_portal_mode(p_source.get_portal_mode());
	r_dest.set_include_in_bound(p_source.get_include_in_bound());
	r_dest.set_portal_autoplace_priority(p_source.get_portal_autoplace_priority());
	r_dest.set_extra_cull_margin(p_source.get_extra_cull_margin());

	// As they are guaranteed siblings, the transform can be identical.
	if (p_copy_transform) {
		r_dest.set_transform(p_source.get_transform());
	}

	// Preserve visibility.
	// If they are siblings, they can share the visible flag, if not, we need to take into account visibility in tree.
	if (p_source.get_parent() == r_dest.get_parent()) {
		r_dest.set_visible(p_source.is_visible());
	} else {
		r_dest.set_visible(p_source.is_visible_in_tree());
	}
}

void MergingTool::_merge_log(String p_string, int p_priority) {
#ifdef TOOLS_ENABLED
	switch (p_priority) {
		case 0: {
			print_verbose(p_string);
		} break;
		case 2: {
			print_line(p_string);
		} break;
		default: {
#ifdef DEV_ENABLED
			print_line(p_string);
#else
			print_verbose(p_string);
#endif
		} break;
	}
#endif
}

void MergingTool::_set_owner_logged(Node *p_node, Node *p_owner) {
	DEV_ASSERT(p_node != p_owner);

#ifdef DEV_ENABLED
#if 0
	// Check whether the Node::set_owner() routine will allow this .. the owner must be in the tree above
	// for the call to work.
	bool valid = false;
	Node *probe = p_node->get_parent();
	while (probe) {
		if (probe == p_owner) {
			valid = true;
			break;
		}
		probe = probe->get_parent();
	}

	DEV_ASSERT(valid);
#endif
#endif

	if (p_node->get_owner() == p_owner) {
		return;
	}

#ifdef DEV_ENABLED
#if 0
	String string = "\tchanging owner of \"" + p_node->get_name() + "\" from ";
	if (p_node->get_owner()) {
		string += p_node->get_owner()->get_name();
	} else {
		string += "NULL";
	}
	string += " to ";
	if (p_owner) {
		string += p_owner->get_name();
	} else {
		string += "NULL";
	}

	_merge_log(string);
#endif
#endif

	p_node->set_owner(p_owner);
	DEV_ASSERT(p_node->get_owner());
}

bool MergingTool::_node_has_valid_children(Node *p_node) {
	for (int n = 0; n < p_node->get_child_count(); n++) {
		if (!p_node->get_child(n)->is_queued_for_deletion()) {
			return true;
		}
	}
	return false;
}

void MergingTool::_invalidate_owner_recursive(Node *p_node, Node *p_old_owner, Node *p_new_owner) {
	if (p_node->get_owner() == p_old_owner) {
		_set_owner_logged(p_node, p_new_owner);
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		_invalidate_owner_recursive(p_node->get_child(n), p_old_owner, p_new_owner);
	}
}

void MergingTool::_reparent(Node *p_branch, Node *p_new_parent, Node *p_new_owner) {
#ifdef GODOT_MERGING_VERBOSE
	if (p_branch->get_parent()) {
		_merge_log("reparenting child " + p_branch->get_name() + " from parent " + p_branch->get_parent()->get_name() + " to parent " + p_new_parent->get_name());
	} else {
		_merge_log("reparenting child " + p_branch->get_name() + " from parent NULL to parent " + p_new_parent->get_name());
	}
#endif

	// noop
	if (p_branch->get_parent() == p_new_parent) {
		return;
	}

	// Detach (if attached).
	if (p_branch->get_parent()) {
		p_branch->get_parent()->remove_child(p_branch);
	}

	// Must be added to the scene BEFORE setting the new owner
	// otherwise the set_owner() calls will fail to find the new owner.
	p_new_parent->add_child(p_branch);

	_reparent_subscene_send_new_owner(p_branch, p_new_owner);
}

void MergingTool::_reparent_subscene_send_new_owner(Node *p_node, Node *p_new_owner) {
	bool owner_found = false;

	// Is the current owner in the subscene? If so keep it, else change.
	Node *current_owner = p_node->get_owner();
	if (current_owner) {
		Node *probe = p_node;
		while (probe) {
			if (probe == current_owner) {
				// Owner already exists in the subscene, no need to change.
				owner_found = true;
				break;
			}
			probe = probe->get_parent();
		}
	} // if there was a current owner

	if (!owner_found) {
		_set_owner_logged(p_node, p_new_owner);
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		_reparent_subscene_send_new_owner(p_node->get_child(n), p_new_owner);
	}
}
