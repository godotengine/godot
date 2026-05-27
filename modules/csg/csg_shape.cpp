/**************************************************************************/
/*  csg_shape.cpp                                                         */
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

#include "csg_shape.h"

#include "core/config/engine.h"
#include "core/math/geometry_2d.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/3d/navigation_mesh_source_geometry_data_3d.h"
#include "scene/resources/navigation_mesh.h"
#include "servers/rendering/rendering_server.h"

#ifdef DEV_ENABLED
#include "core/io/json.h"
#endif // DEV_ENABLED

#ifndef NAVIGATION_3D_DISABLED
#include "servers/navigation_3d/navigation_server_3d.h"
#endif // NAVIGATION_3D_DISABLED

#include <manifold/manifold.h>

#include <cfloat> // FLT_EPSILON

#ifndef NAVIGATION_3D_DISABLED
Callable CSGShape3D::_navmesh_source_geometry_parsing_callback;
RID CSGShape3D::_navmesh_source_geometry_parser;

void CSGShape3D::navmesh_parse_init() {
	ERR_FAIL_NULL(NavigationServer3D::get_singleton());
	if (!_navmesh_source_geometry_parser.is_valid()) {
		_navmesh_source_geometry_parsing_callback = callable_mp_static(&CSGShape3D::navmesh_parse_source_geometry);
		_navmesh_source_geometry_parser = NavigationServer3D::get_singleton()->source_geometry_parser_create();
		NavigationServer3D::get_singleton()->source_geometry_parser_set_callback(_navmesh_source_geometry_parser, _navmesh_source_geometry_parsing_callback);
	}
}

void CSGShape3D::navmesh_parse_source_geometry(const Ref<NavigationMesh> &p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, Node *p_node) {
	CSGShape3D *csgshape3d = Object::cast_to<CSGShape3D>(p_node);

	if (csgshape3d == nullptr) {
		return;
	}

	NavigationMesh::ParsedGeometryType parsed_geometry_type = p_navigation_mesh->get_parsed_geometry_type();

#ifndef PHYSICS_3D_DISABLED
	bool nav_collision = (parsed_geometry_type == NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS && csgshape3d->is_using_collision() && (csgshape3d->get_collision_layer() & p_navigation_mesh->get_collision_mask()));
#else
	bool nav_collision = false;
#endif // PHYSICS_3D_DISABLED
	if (parsed_geometry_type == NavigationMesh::PARSED_GEOMETRY_MESH_INSTANCES || nav_collision || parsed_geometry_type == NavigationMesh::PARSED_GEOMETRY_BOTH) {
		Array meshes = csgshape3d->get_meshes();
		if (!meshes.is_empty()) {
			Ref<Mesh> mesh = meshes[1];
			if (mesh.is_valid()) {
				p_source_geometry_data->add_mesh(mesh, csgshape3d->get_global_transform());
			}
		}
	}
}
#endif // NAVIGATION_3D_DISABLED

#ifndef PHYSICS_3D_DISABLED
void CSGShape3D::set_use_collision(bool p_enable) {
	if (use_collision == p_enable) {
		return;
	}

	use_collision = p_enable;

	if (!is_inside_tree() || !is_root_shape()) {
		return;
	}

	if (use_collision) {
		root_collision_shape.instantiate();
		root_collision_instance = PhysicsServer3D::get_singleton()->body_create();
		PhysicsServer3D::get_singleton()->body_set_mode(root_collision_instance, PhysicsServer3D::BODY_MODE_STATIC);
		PhysicsServer3D::get_singleton()->body_set_state(root_collision_instance, PhysicsServer3D::BODY_STATE_TRANSFORM, get_global_transform());
		PhysicsServer3D::get_singleton()->body_add_shape(root_collision_instance, root_collision_shape->get_rid());
		PhysicsServer3D::get_singleton()->body_set_space(root_collision_instance, get_world_3d()->get_space());
		PhysicsServer3D::get_singleton()->body_attach_object_instance_id(root_collision_instance, get_instance_id());
		set_collision_layer(collision_layer);
		set_collision_mask(collision_mask);
		set_collision_priority(collision_priority);
		_make_painted(); //force update
	} else {
		PhysicsServer3D::get_singleton()->free_rid(root_collision_instance);
		root_collision_instance = RID();
		root_collision_shape.unref();
	}
	notify_property_list_changed();
}

bool CSGShape3D::is_using_collision() const {
	return use_collision;
}

void CSGShape3D::set_collision_layer(uint32_t p_layer) {
	collision_layer = p_layer;
	if (root_collision_instance.is_valid()) {
		PhysicsServer3D::get_singleton()->body_set_collision_layer(root_collision_instance, p_layer);
	}
}

uint32_t CSGShape3D::get_collision_layer() const {
	return collision_layer;
}

void CSGShape3D::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
	if (root_collision_instance.is_valid()) {
		PhysicsServer3D::get_singleton()->body_set_collision_mask(root_collision_instance, p_mask);
	}
}

uint32_t CSGShape3D::get_collision_mask() const {
	return collision_mask;
}

void CSGShape3D::set_collision_layer_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Collision layer number must be between 1 and 32 inclusive.");
	uint32_t layer = get_collision_layer();
	if (p_value) {
		layer |= 1 << (p_layer_number - 1);
	} else {
		layer &= ~(1 << (p_layer_number - 1));
	}
	set_collision_layer(layer);
}

bool CSGShape3D::get_collision_layer_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_collision_layer() & (1 << (p_layer_number - 1));
}

void CSGShape3D::set_collision_mask_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Collision layer number must be between 1 and 32 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << (p_layer_number - 1);
	} else {
		mask &= ~(1 << (p_layer_number - 1));
	}
	set_collision_mask(mask);
}

bool CSGShape3D::get_collision_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_collision_mask() & (1 << (p_layer_number - 1));
}

RID CSGShape3D::_get_root_collision_instance() const {
	if (root_collision_instance.is_valid()) {
		return root_collision_instance;
	} else if (parent_shape) {
		return parent_shape->_get_root_collision_instance();
	}

	return RID();
}

void CSGShape3D::set_collision_priority(real_t p_priority) {
	collision_priority = p_priority;
	if (root_collision_instance.is_valid()) {
		PhysicsServer3D::get_singleton()->body_set_collision_priority(root_collision_instance, p_priority);
	}
}

real_t CSGShape3D::get_collision_priority() const {
	return collision_priority;
}

void CSGShape3D::set_autosmooth(bool p_smooth) {
	autosmooth = p_smooth;
	_make_painted();
	notify_property_list_changed();
}

bool CSGShape3D::is_autosmooth() const {
	return autosmooth;
}

void CSGShape3D::set_smoothing_angle(const float p_angle) {
	smoothing_angle = p_angle;
	_make_painted();
}

float CSGShape3D::get_smoothing_angle() const {
	return smoothing_angle;
}

#endif // PHYSICS_3D_DISABLED

bool CSGShape3D::is_root_shape() const {
	return !parent_shape;
}

#ifndef DISABLE_DEPRECATED
void CSGShape3D::set_snap(float p_snap) {
	if (snap == p_snap) {
		return;
	}

	snap = p_snap;
	_make_painted();
}

float CSGShape3D::get_snap() const {
	return snap;
}
#endif // DISABLE_DEPRECATED

void CSGShape3D::_make_dirty(bool p_parent_removing) {
#ifndef PHYSICS_3D_DISABLED
	if ((p_parent_removing || is_root_shape()) && !dirty) {
		callable_mp(this, &CSGShape3D::update_shape).call_deferred(); // Must be deferred; otherwise, is_root_shape() will use the previous parent.
	}
#endif // PHYSICS_3D_DISABLED

	if (!is_root_shape()) {
		parent_shape->_make_painted();
	}
#ifndef PHYSICS_3D_DISABLED
	else if (!dirty) {
		callable_mp(this, &CSGShape3D::update_shape).call_deferred();
	}
#endif // PHYSICS_3D_DISABLED
	children_modified = true;
	painted = false;
	dirty = true;
	notify_property_list_changed();
}

void CSGShape3D::_make_painted(bool p_paint, bool p_parent_removing) {
	// Update combined_brush.
	children_modified = true;

	painted = painted || p_paint;
	if (!painted) {
		_make_dirty(p_parent_removing);
		return;
	} else if (!brush) {
		_make_dirty(p_parent_removing);
		return;
	}

	if (p_parent_removing) {
		callable_mp(this, &CSGShape3D::update_shape).call_deferred();
	} else if (!is_root_shape()) {
		// The !is_root_shape() means the node has a parent so it has been added already.
		update_configuration_warnings();
		notify_property_list_changed();
		// We force a rebuild of the root shape only.
		parent_shape->_make_painted();
	} else {
		// Properties are set before add_child, so the node is root shape because it doesn't have a parent.
		callable_mp(this, &CSGShape3D::update_shape).call_deferred();
	}
}

enum ManifoldProperty {
	MANIFOLD_PROPERTY_POSITION_X = 0,
	MANIFOLD_PROPERTY_POSITION_Y,
	MANIFOLD_PROPERTY_POSITION_Z,
	MANIFOLD_PROPERTY_INVERT,
	MANIFOLD_PROPERTY_SMOOTH_GROUP,
	MANIFOLD_PROPERTY_UV_X_0,
	MANIFOLD_PROPERTY_UV_Y_0,
	MANIFOLD_PROPERTY_MAX
};

static void _unpack_manifold(
		const manifold::Manifold &p_manifold,
		const HashMap<int32_t, Ref<Material>> &p_mesh_materials,
		CSGBrush *r_mesh_merge) {
	manifold::MeshGL64 mesh = p_manifold.GetMeshGL64();

	constexpr int32_t order[3] = { 0, 2, 1 };

	for (size_t run_i = 0; run_i < mesh.runIndex.size() - 1; run_i++) {
		uint32_t original_id = -1;
		if (run_i < mesh.runOriginalID.size()) {
			original_id = mesh.runOriginalID[run_i];
		}

		Ref<Material> material;
		if (p_mesh_materials.has(original_id)) {
			material = p_mesh_materials[original_id];
		}
		// Find or reserve a material ID in the brush.
		int32_t material_id = r_mesh_merge->materials.find(material);
		if (material_id == -1) {
			material_id = r_mesh_merge->materials.size();
			r_mesh_merge->materials.push_back(material);
		}

		size_t begin = mesh.runIndex[run_i];
		size_t end = mesh.runIndex[run_i + 1];
		for (size_t vert_i = begin; vert_i < end; vert_i += 3) {
			CSGBrush::Face face;
			face.material = material_id;
			int32_t first_property_index = mesh.triVerts[vert_i + order[0]];
			face.smooth = mesh.vertProperties[first_property_index * mesh.numProp + MANIFOLD_PROPERTY_SMOOTH_GROUP] > 0.5f;
			face.invert = mesh.vertProperties[first_property_index * mesh.numProp + MANIFOLD_PROPERTY_INVERT] > 0.5f;

			for (int32_t tri_order_i = 0; tri_order_i < 3; tri_order_i++) {
				int32_t property_i = mesh.triVerts[vert_i + order[tri_order_i]];
				ERR_FAIL_COND_MSG(property_i * mesh.numProp >= mesh.vertProperties.size(), "Invalid index into vertex properties");
				face.vertices[tri_order_i] = Vector3(
						mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_POSITION_X],
						mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_POSITION_Y],
						mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_POSITION_Z]);
				face.uvs[tri_order_i] = Vector2(
						mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_UV_X_0],
						mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_UV_Y_0]);
			}
			r_mesh_merge->faces.push_back(face);
		}
	}

	r_mesh_merge->_regen_face_aabbs();
}

#ifdef DEV_ENABLED
static String _export_meshgl_as_json(const manifold::MeshGL64 &p_mesh) {
	Dictionary mesh_dict;
	mesh_dict["numProp"] = p_mesh.numProp;

	Array vert_properties;
	for (const double &val : p_mesh.vertProperties) {
		vert_properties.append(val);
	}
	mesh_dict["vertProperties"] = vert_properties;

	Array tri_verts;
	for (const uint64_t &val : p_mesh.triVerts) {
		tri_verts.append(val);
	}
	mesh_dict["triVerts"] = tri_verts;

	Array merge_from_vert;
	for (const uint64_t &val : p_mesh.mergeFromVert) {
		merge_from_vert.append(val);
	}
	mesh_dict["mergeFromVert"] = merge_from_vert;

	Array merge_to_vert;
	for (const uint64_t &val : p_mesh.mergeToVert) {
		merge_to_vert.append(val);
	}
	mesh_dict["mergeToVert"] = merge_to_vert;

	Array run_index;
	for (const uint64_t &val : p_mesh.runIndex) {
		run_index.append(val);
	}
	mesh_dict["runIndex"] = run_index;

	Array run_original_id;
	for (const uint32_t &val : p_mesh.runOriginalID) {
		run_original_id.append(val);
	}
	mesh_dict["runOriginalID"] = run_original_id;

	Array run_transform;
	for (const double &val : p_mesh.runTransform) {
		run_transform.append(val);
	}
	mesh_dict["runTransform"] = run_transform;

	Array face_id;
	for (const uint64_t &val : p_mesh.faceID) {
		face_id.append(val);
	}
	mesh_dict["faceID"] = face_id;

	Array halfedge_tangent;
	for (const double &val : p_mesh.halfedgeTangent) {
		halfedge_tangent.append(val);
	}
	mesh_dict["halfedgeTangent"] = halfedge_tangent;

	mesh_dict["tolerance"] = p_mesh.tolerance;

	String json_string = JSON::stringify(mesh_dict);
	return json_string;
}
#endif // DEV_ENABLED

static void _pack_manifold(
		const CSGBrush *const p_mesh_merge,
		manifold::Manifold &r_manifold,
		HashMap<int32_t, Ref<Material>> &p_mesh_materials,
		CSGShape3D *p_csg_shape) {
	ERR_FAIL_NULL_MSG(p_mesh_merge, "p_mesh_merge is null");
	ERR_FAIL_NULL_MSG(p_csg_shape, "p_shape is null");
	HashMap<uint32_t, Vector<CSGBrush::Face>> faces_by_material;
	for (int face_i = 0; face_i < p_mesh_merge->faces.size(); face_i++) {
		const CSGBrush::Face &face = p_mesh_merge->faces[face_i];
		faces_by_material[face.material].push_back(face);
	}

	manifold::MeshGL64 mesh;
	mesh.numProp = MANIFOLD_PROPERTY_MAX;
	mesh.runOriginalID.reserve(faces_by_material.size());
	mesh.runIndex.reserve(faces_by_material.size() + 1);
	mesh.vertProperties.reserve(p_mesh_merge->faces.size() * 3 * MANIFOLD_PROPERTY_MAX);

	// Make a run of triangles for each material.
	for (const KeyValue<uint32_t, Vector<CSGBrush::Face>> &E : faces_by_material) {
		const uint32_t material_id = E.key;
		const Vector<CSGBrush::Face> &faces = E.value;
		mesh.runIndex.push_back(mesh.triVerts.size());

		// Associate the material with an ID.
		uint32_t reserved_id = r_manifold.ReserveIDs(1);
		mesh.runOriginalID.push_back(reserved_id);
		Ref<Material> material;
		if (material_id < p_mesh_merge->materials.size()) {
			material = p_mesh_merge->materials[material_id];
		}

		p_mesh_materials.insert(reserved_id, material);
		for (const CSGBrush::Face &face : faces) {
			for (int32_t tri_order_i = 0; tri_order_i < 3; tri_order_i++) {
				constexpr int32_t order[3] = { 0, 2, 1 };
				int i = order[tri_order_i];

				mesh.triVerts.push_back(mesh.vertProperties.size() / MANIFOLD_PROPERTY_MAX);

				size_t begin = mesh.vertProperties.size();
				mesh.vertProperties.resize(mesh.vertProperties.size() + MANIFOLD_PROPERTY_MAX);
				// Add the vertex properties.
				// Use CSGBrush constants rather than push_back for clarity.
				double *vert = &mesh.vertProperties[begin];
				vert[MANIFOLD_PROPERTY_POSITION_X] = face.vertices[i].x;
				vert[MANIFOLD_PROPERTY_POSITION_Y] = face.vertices[i].y;
				vert[MANIFOLD_PROPERTY_POSITION_Z] = face.vertices[i].z;
				vert[MANIFOLD_PROPERTY_UV_X_0] = face.uvs[i].x;
				vert[MANIFOLD_PROPERTY_UV_Y_0] = face.uvs[i].y;
				vert[MANIFOLD_PROPERTY_SMOOTH_GROUP] = face.smooth ? 1.0f : 0.0f;
				vert[MANIFOLD_PROPERTY_INVERT] = face.invert ? 1.0f : 0.0f;
			}
		}
	}
	// runIndex needs an explicit end value.
	mesh.runIndex.push_back(mesh.triVerts.size());
	mesh.tolerance = 2 * FLT_EPSILON;
	ERR_FAIL_COND_MSG(mesh.vertProperties.size() % mesh.numProp != 0, "Invalid vertex properties size.");
	mesh.Merge();
#ifdef DEV_ENABLED
	print_verbose(_export_meshgl_as_json(mesh));
#endif // DEV_ENABLED
	r_manifold = manifold::Manifold(mesh);
}

struct ManifoldOperation {
	manifold::Manifold manifold;
	manifold::OpType operation;
	static manifold::OpType convert_csg_op(CSGShape3D::Operation op) {
		switch (op) {
			case CSGShape3D::OPERATION_SUBTRACTION:
				return manifold::OpType::Subtract;
			case CSGShape3D::OPERATION_INTERSECTION:
				return manifold::OpType::Intersect;
			default:
				return manifold::OpType::Add;
		}
	}
	ManifoldOperation() :
			operation(manifold::OpType::Add) {}
	ManifoldOperation(const manifold::Manifold &m, manifold::OpType op) :
			manifold(m), operation(op) {}
};

CSGBrush *CSGShape3D::_get_combined_brush() {
	int child_count = get_child_count();
	if (!dirty && !children_modified) {
		if (child_count < 1) {
			// Single node should not have combined_brush.
			if (combined_brush) {
				memdelete(combined_brush);
				combined_brush = nullptr;
			}
			if (brush) {
				return brush;
			}
		}
		if (combined_brush) {
			return combined_brush;
		}
	}
	// If dirty or doesn't have combined_brush, perform boolean operations and store combined_brush.
	if (combined_brush) {
		memdelete(combined_brush);
	}

	// This will generate a CSGBrush if it's not assigned.
	CSGBrush *orig = _get_brush();

	if (child_count < 1) {
		// Dirty but single node, we can skip the rest of the code.
		combined_brush = nullptr;
		children_modified = false;
		return brush;
	}

	CSGBrush *n = memnew(CSGBrush);
	n->copy_from(*orig, Transform3D());
	HashMap<int32_t, Ref<Material>> mesh_materials;
	manifold::Manifold root_manifold;
	_pack_manifold(n, root_manifold, mesh_materials, this);
	manifold::OpType current_op = ManifoldOperation::convert_csg_op(get_operation());
	std::vector<manifold::Manifold> manifolds;
	manifolds.push_back(root_manifold);

	bool has_children = false;
	for (int i = 0; i < child_count; i++) {
		CSGShape3D *child = Object::cast_to<CSGShape3D>(get_child(i));
		if (!child || !child->is_visible()) {
			continue;
		}
		CSGBrush *child_brush = child->_get_combined_brush();
		if (!child_brush) {
			continue;
		}
		CSGBrush transformed_brush;
		transformed_brush.copy_from(*child_brush, child->get_transform());
		manifold::Manifold child_manifold;
		_pack_manifold(&transformed_brush, child_manifold, mesh_materials, child);
		manifold::OpType child_operation = ManifoldOperation::convert_csg_op(child->get_operation());
		if (child_operation != current_op) {
			manifold::Manifold result = manifold::Manifold::BatchBoolean(manifolds, current_op);
			manifolds.clear();
			manifolds.push_back(result);
			current_op = child_operation;
		}
		manifolds.push_back(child_manifold);
		// Node has at least one CSGShape3D child.
		has_children = true;
	}
	if (!manifolds.empty()) {
		manifold::Manifold manifold_result = manifold::Manifold::BatchBoolean(manifolds, current_op);
		if (n) {
			memdelete(n);
		}
		n = memnew(CSGBrush);
		_unpack_manifold(manifold_result, mesh_materials, n);
		if (n == nullptr) {
			csg_push_warning(NON_MANIFOLD);
		} else if (n->faces.is_empty()) {
			csg_push_warning(NON_MANIFOLD);
		}
	}
	if (has_children) {
		_expand_aabb(n);
	} else {
		// Node has children but those children are not CSGShape3D.
		_expand_aabb(orig);
	}
	combined_brush = n;
	dirty = false;
	children_modified = false;
	update_configuration_warnings();
	return combined_brush;
}

CSGBrush *CSGShape3D::_get_brush() {
	if (!dirty) {
		return brush;
	}
	if (brush) {
		memdelete(brush);
	}
	brush = nullptr;
	CSGBrush *n = _build_brush();
	if (get_child_count() < 1 || !combined_brush) {
		// This is a hack and needs a better solution to the problem of non CSGShape3D children.
		_expand_aabb(n);
	}
	brush = n;
	dirty = false;
	update_configuration_warnings();
	return brush;
}

void CSGShape3D::_expand_aabb(CSGBrush *p_brush) {
	AABB aabb;
	if (p_brush && !p_brush->faces.is_empty()) {
		aabb.position = p_brush->faces[0].vertices[0];
		for (const CSGBrush::Face &face : p_brush->faces) {
			for (int i = 0; i < 3; ++i) {
				aabb.expand_to(face.vertices[i]);
			}
		}
	}
	node_aabb = aabb;
}

int CSGShape3D::mikktGetNumFaces(const SMikkTSpaceContext *pContext) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	return surface.vertices.size() / 3;
}

int CSGShape3D::mikktGetNumVerticesOfFace(const SMikkTSpaceContext *pContext, const int iFace) {
	// always 3
	return 3;
}

void CSGShape3D::mikktGetPosition(const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace, const int iVert) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	Vector3 v = surface.verticesw[iFace * 3 + iVert];
	fvPosOut[0] = v.x;
	fvPosOut[1] = v.y;
	fvPosOut[2] = v.z;
}

void CSGShape3D::mikktGetNormal(const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace, const int iVert) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	Vector3 n = surface.normalsw[iFace * 3 + iVert];
	fvNormOut[0] = n.x;
	fvNormOut[1] = n.y;
	fvNormOut[2] = n.z;
}

void CSGShape3D::mikktGetTexCoord(const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace, const int iVert) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	Vector2 t = surface.uvsw[iFace * 3 + iVert];
	fvTexcOut[0] = t.x;
	fvTexcOut[1] = t.y;
}

void CSGShape3D::mikktSetTSpaceDefault(const SMikkTSpaceContext *pContext, const float fvTangent[], const float fvBiTangent[], const float fMagS, const float fMagT,
		const tbool bIsOrientationPreserving, const int iFace, const int iVert) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	int i = iFace * 3 + iVert;
	Vector3 normal = surface.normalsw[i];
	Vector3 tangent = Vector3(fvTangent[0], fvTangent[1], fvTangent[2]);
	Vector3 bitangent = Vector3(-fvBiTangent[0], -fvBiTangent[1], -fvBiTangent[2]); // for some reason these are reversed, something with the coordinate system in Godot
	float d = bitangent.dot(normal.cross(tangent));

	i *= 4;
	surface.tansw[i++] = tangent.x;
	surface.tansw[i++] = tangent.y;
	surface.tansw[i++] = tangent.z;
	surface.tansw[i++] = d < 0 ? -1 : 1;
}

void CSGShape3D::update_shape() {
	if (!is_root_shape()) {
		return;
	}

	set_base(RID());
	root_mesh.unref(); //byebye root mesh

	CSGBrush *n = _get_combined_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");

	Vector<int> face_count;
	face_count.resize(n->materials.size() + 1);

	Vector<Vector3> smooth_faces;
	LocalVector<Vector3> smooth_vertex;
	smooth_faces.resize(n->faces.size());
	smooth_vertex.resize(n->faces.size() * 3);

	for (int i = 0; i < face_count.size(); i++) {
		face_count.write[i] = 0;
	}

	for (int i = 0; i < n->faces.size(); i++) {
		int mat = n->faces[i].material;
		ERR_CONTINUE(mat < -1 || mat >= face_count.size());
		int idx = mat == -1 ? face_count.size() - 1 : mat;

		Plane p(n->faces[i].vertices[0], n->faces[i].vertices[1], n->faces[i].vertices[2]);

		smooth_faces.write[i] = p.normal;
		// Not sure if resize populates the LocalVector.
		smooth_vertex[i * 3 + 0] = Vector3(p.normal);
		smooth_vertex[i * 3 + 1] = Vector3(p.normal);
		smooth_vertex[i * 3 + 2] = Vector3(p.normal);
		// We could use a AHashMap Vector3, int to store the number of connections of each vertex position and end the loop earlier. But I'm not sure if the performance gains outweigh the cost.
		face_count.write[idx]++;
	}

	if (autosmooth) {
		// We could add a `use_groups` property later to only apply autosmooth on smooth faces or respect smoothing groups in some way.
		if (smoothing_angle > 0.1) {
			float smooth_angle_rad = Math::cos(Math::deg_to_rad(smoothing_angle));
			for (int i = 0; i < smooth_faces.size(); i++) {
				for (int k = 0; k < 3; k++) {
					int curr_vert = i * 3 + k;
					Vector3 vert_a = n->faces[i].vertices[k];
					for (int j = i + 1; j < smooth_faces.size(); j++) {
						// Compare the angles of faces instead of vertices.
						if (smooth_faces[i].dot(smooth_faces[j]) > smooth_angle_rad) {
							for (int h = 0; h < 3; h++) {
								Vector3 vert_b = n->faces[j].vertices[h];
								if (vert_a == vert_b) {
									int curr_j = j * 3 + h;
									smooth_vertex[curr_vert] += smooth_faces[j];
									smooth_vertex[curr_j] += smooth_faces[i];
									break;
								}
							}
						}
					}
					smooth_vertex[curr_vert].normalize();
				}
			}
		}
	} else {
		for (int i = 0; i < smooth_faces.size(); i++) {
			bool face_is_smooth = n->faces[i].smooth;
			if (face_is_smooth) {
				for (int k = 0; k < 3; k++) {
					Vector3 vert_a = n->faces[i].vertices[k];
					int curr_vert = i * 3 + k;
					// Skip the other vertices of the face as they will never occupy the same position.
					for (int j = i + 1; j < smooth_faces.size(); j++) {
						// Preparing for when and if we replace Vector of bool for Vector of int smoothing groups. for now, face_is_smooth is always true.
						if (face_is_smooth == n->faces[j].smooth) {
							for (int h = 0; h < 3; h++) {
								Vector3 vert_b = n->faces[j].vertices[h];
								if (vert_a == vert_b) {
									int curr_j = j * 3 + h;
									smooth_vertex[curr_vert] += smooth_faces[j];
									smooth_vertex[curr_j] += smooth_faces[i];
									// Skip the other 2 vertices as only one vertex of each face can connect with one vertex of other face.
									break;
								}
							}
						}
					}
					smooth_vertex[curr_vert].normalize();
				}
			}
		}
	}

	Vector<ShapeUpdateSurface> surfaces;

	surfaces.resize(face_count.size());

	//create arrays
	for (int i = 0; i < surfaces.size(); i++) {
		surfaces.write[i].vertices.resize(face_count[i] * 3);
		surfaces.write[i].normals.resize(face_count[i] * 3);
		surfaces.write[i].uvs.resize(face_count[i] * 3);
		if (calculate_tangents) {
			surfaces.write[i].tans.resize(face_count[i] * 3 * 4);
		}
		surfaces.write[i].last_added = 0;

		if (i != surfaces.size() - 1) {
			surfaces.write[i].material = n->materials[i];
		}

		surfaces.write[i].verticesw = surfaces.write[i].vertices.ptrw();
		surfaces.write[i].normalsw = surfaces.write[i].normals.ptrw();
		surfaces.write[i].uvsw = surfaces.write[i].uvs.ptrw();
		if (calculate_tangents) {
			surfaces.write[i].tansw = surfaces.write[i].tans.ptrw();
		}
	}

	//fill arrays
	{
		for (int i = 0; i < n->faces.size(); i++) {
			int order[3] = { 0, 1, 2 };

			if (n->faces[i].invert) {
				SWAP(order[1], order[2]);
			}

			int mat = n->faces[i].material;
			ERR_CONTINUE(mat < -1 || mat >= face_count.size());
			int idx = mat == -1 ? face_count.size() - 1 : mat;

			int last = surfaces[idx].last_added;

			int face_pos_i = i * 3;

			for (int j = 0; j < 3; j++) {
				Vector3 v = n->faces[i].vertices[j];

				Vector3 normal = smooth_vertex[face_pos_i + j];

				if (n->faces[i].invert) {
					normal = -normal;
				}

				int k = last + order[j];
				surfaces[idx].verticesw[k] = v;
				surfaces[idx].uvsw[k] = n->faces[i].uvs[j];
				surfaces[idx].normalsw[k] = normal;

				if (calculate_tangents) {
					// zero out our tangents for now
					k *= 4;
					surfaces[idx].tansw[k++] = 0.0;
					surfaces[idx].tansw[k++] = 0.0;
					surfaces[idx].tansw[k++] = 0.0;
					surfaces[idx].tansw[k++] = 0.0;
				}
			}

			surfaces.write[idx].last_added += 3;
		}
	}

	root_mesh.instantiate();
	//create surfaces

	for (int i = 0; i < surfaces.size(); i++) {
		// calculate tangents for this surface
		bool have_tangents = calculate_tangents;
		if (have_tangents) {
			SMikkTSpaceInterface mkif;
			mkif.m_getNormal = mikktGetNormal;
			mkif.m_getNumFaces = mikktGetNumFaces;
			mkif.m_getNumVerticesOfFace = mikktGetNumVerticesOfFace;
			mkif.m_getPosition = mikktGetPosition;
			mkif.m_getTexCoord = mikktGetTexCoord;
			mkif.m_setTSpace = mikktSetTSpaceDefault;
			mkif.m_setTSpaceBasic = nullptr;

			SMikkTSpaceContext msc;
			msc.m_pInterface = &mkif;
			msc.m_pUserData = &surfaces.write[i];
			have_tangents = genTangSpaceDefault(&msc);
		}

		if (surfaces[i].last_added == 0) {
			continue;
		}

		// and convert to surface array
		Array array;
		array.resize(Mesh::ARRAY_MAX);

		array[Mesh::ARRAY_VERTEX] = surfaces[i].vertices;
		array[Mesh::ARRAY_NORMAL] = surfaces[i].normals;
		array[Mesh::ARRAY_TEX_UV] = surfaces[i].uvs;
		if (have_tangents) {
			array[Mesh::ARRAY_TANGENT] = surfaces[i].tans;
		}

		int idx = root_mesh->get_surface_count();
		root_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, array);
		root_mesh->surface_set_material(idx, surfaces[i].material);
	}

	set_base(root_mesh->get_rid());

	update_gizmos();

#ifndef PHYSICS_3D_DISABLED
	_update_collision_faces();
#endif // PHYSICS_3D_DISABLED
}

Ref<ArrayMesh> CSGShape3D::bake_static_mesh() {
	Ref<ArrayMesh> baked_mesh;
	if (is_root_shape() && root_mesh.is_valid()) {
		baked_mesh = root_mesh;
	}
	return baked_mesh;
}

#ifndef PHYSICS_3D_DISABLED
Vector<Vector3> CSGShape3D::_get_brush_collision_faces() {
	Vector<Vector3> collision_faces;
	CSGBrush *n = _get_combined_brush();
	ERR_FAIL_NULL_V_MSG(n, collision_faces, "Cannot get CSGBrush.");
	collision_faces.resize(n->faces.size() * 3);
	Vector3 *collision_faces_ptrw = collision_faces.ptrw();

	for (int i = 0; i < n->faces.size(); i++) {
		int order[3] = { 0, 1, 2 };

		if (n->faces[i].invert) {
			SWAP(order[1], order[2]);
		}

		collision_faces_ptrw[i * 3 + 0] = n->faces[i].vertices[order[0]];
		collision_faces_ptrw[i * 3 + 1] = n->faces[i].vertices[order[1]];
		collision_faces_ptrw[i * 3 + 2] = n->faces[i].vertices[order[2]];
	}

	return collision_faces;
}

void CSGShape3D::_update_collision_faces() {
	if (use_collision && is_root_shape() && root_collision_shape.is_valid()) {
		root_collision_shape->set_faces(_get_brush_collision_faces());

		if (_is_debug_collision_shape_visible()) {
			_update_debug_collision_shape();
		}
	}
}

Ref<ConcavePolygonShape3D> CSGShape3D::bake_collision_shape() {
	Ref<ConcavePolygonShape3D> baked_collision_shape;
	if (is_root_shape() && root_collision_shape.is_valid()) {
		baked_collision_shape.instantiate();
		baked_collision_shape->set_faces(root_collision_shape->get_faces());
	} else if (is_root_shape()) {
		baked_collision_shape.instantiate();
		baked_collision_shape->set_faces(_get_brush_collision_faces());
	}
	return baked_collision_shape;
}

bool CSGShape3D::_is_debug_collision_shape_visible() {
	return !Engine::get_singleton()->is_editor_hint() && is_inside_tree() && get_tree()->is_debugging_collisions_hint();
}

void CSGShape3D::_update_debug_collision_shape() {
	if (!use_collision || !is_root_shape() || root_collision_shape.is_null() || !_is_debug_collision_shape_visible()) {
		return;
	}

	ERR_FAIL_NULL(RenderingServer::get_singleton());

	if (root_collision_debug_instance.is_null()) {
		root_collision_debug_instance = RS::get_singleton()->instance_create();
	}

	Ref<Mesh> debug_mesh = root_collision_shape->get_debug_mesh();
	RS::get_singleton()->instance_set_scenario(root_collision_debug_instance, get_world_3d()->get_scenario());
	RS::get_singleton()->instance_set_base(root_collision_debug_instance, debug_mesh->get_rid());
	RS::get_singleton()->instance_set_transform(root_collision_debug_instance, get_global_transform());
}

void CSGShape3D::_clear_debug_collision_shape() {
	if (root_collision_debug_instance.is_valid()) {
		RS::get_singleton()->free_rid(root_collision_debug_instance);
		root_collision_debug_instance = RID();
	}
}

void CSGShape3D::_on_transform_changed() {
	if (root_collision_debug_instance.is_valid() && !debug_shape_old_transform.is_equal_approx(get_global_transform())) {
		debug_shape_old_transform = get_global_transform();
		RS::get_singleton()->instance_set_transform(root_collision_debug_instance, debug_shape_old_transform);
	}
}
#endif // PHYSICS_3D_DISABLED

AABB CSGShape3D::get_aabb() const {
	return node_aabb;
}

Vector<Vector3> CSGShape3D::get_brush_faces() {
	ERR_FAIL_COND_V(!is_inside_tree(), Vector<Vector3>());
	// This should fix the unit test. But should brush faces return the combined brush like it did before, or just the current brush? This is an important question for when a face editor is implemented.
	CSGBrush *b = _get_combined_brush();
	if (!b) {
		return Vector<Vector3>();
	}

	Vector<Vector3> faces;
	int fc = b->faces.size();
	faces.resize(fc * 3);
	{
		Vector3 *w = faces.ptrw();
		for (int i = 0; i < fc; i++) {
			w[i * 3 + 0] = b->faces[i].vertices[0];
			w[i * 3 + 1] = b->faces[i].vertices[1];
			w[i * 3 + 2] = b->faces[i].vertices[2];
		}
	}

	return faces;
}

void CSGShape3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PARENTED: {
			Node *parentn = get_parent();
			if (parentn) {
				parent_shape = Object::cast_to<CSGShape3D>(parentn);
				if (parent_shape) {
					set_base(RID());
					root_mesh.unref();
					_make_painted();
				}
			}
			last_visible = is_visible();
		} break;

		case NOTIFICATION_UNPARENTED: {
			if (!is_root_shape()) {
				// Update this node and its previous parent only if it's currently being removed from another CSG shape
				_make_painted(false, true); // Must be forced since is_root_shape() uses the previous parent
			}
			parent_shape = nullptr;
		} break;

		case NOTIFICATION_CHILD_ORDER_CHANGED: {
			_make_painted();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_root_shape() && last_visible != is_visible()) {
				// Update this node's parent only if its own visibility has changed, not the visibility of parent nodes
				parent_shape->_make_painted();
			}
			last_visible = is_visible();
		} break;

		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			if (!is_root_shape()) {
				// Update this node's parent only if its own transformation has changed, not the transformation of parent nodes
				parent_shape->_make_painted();
			}
		} break;

#ifndef PHYSICS_3D_DISABLED
		case NOTIFICATION_ENTER_TREE: {
			if (use_collision && is_root_shape()) {
				root_collision_shape.instantiate();
				root_collision_instance = PhysicsServer3D::get_singleton()->body_create();
				PhysicsServer3D::get_singleton()->body_set_mode(root_collision_instance, PhysicsServer3D::BODY_MODE_STATIC);
				PhysicsServer3D::get_singleton()->body_set_state(root_collision_instance, PhysicsServer3D::BODY_STATE_TRANSFORM, get_global_transform());
				PhysicsServer3D::get_singleton()->body_add_shape(root_collision_instance, root_collision_shape->get_rid());
				PhysicsServer3D::get_singleton()->body_set_space(root_collision_instance, get_world_3d()->get_space());
				PhysicsServer3D::get_singleton()->body_attach_object_instance_id(root_collision_instance, get_instance_id());
				set_collision_layer(collision_layer);
				set_collision_mask(collision_mask);
				set_collision_priority(collision_priority);
				debug_shape_old_transform = get_global_transform();
				_make_painted();
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (use_collision && is_root_shape() && root_collision_instance.is_valid()) {
				PhysicsServer3D::get_singleton()->free_rid(root_collision_instance);
				root_collision_instance = RID();
				root_collision_shape.unref();
				_clear_debug_collision_shape();
			}
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (use_collision && is_root_shape() && root_collision_instance.is_valid()) {
				PhysicsServer3D::get_singleton()->body_set_state(root_collision_instance, PhysicsServer3D::BODY_STATE_TRANSFORM, get_global_transform());
			}
			_on_transform_changed();
		} break;
#endif // PHYSICS_3D_DISABLED
	}
}

void CSGShape3D::set_operation(Operation p_operation) {
	operation = p_operation;
	_make_painted();
	update_gizmos();
}

CSGShape3D::Operation CSGShape3D::get_operation() const {
	return operation;
}

void CSGShape3D::set_calculate_tangents(bool p_calculate_tangents) {
	calculate_tangents = p_calculate_tangents;
	_make_painted();
}

bool CSGShape3D::is_calculating_tangents() const {
	return calculate_tangents;
}

void CSGShape3D::_validate_property(PropertyInfo &p_property) const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	if (p_property.name == "smoothing_angle") {
		if (!autosmooth || (is_inside_tree() && !is_root_shape())) {
			p_property.usage = PROPERTY_USAGE_NO_EDITOR;
		}
	}

	if (p_property.name == "autosmooth") {
		if (is_inside_tree() && !is_root_shape()) {
			p_property.usage = PROPERTY_USAGE_NO_EDITOR;
		}
	}

	if (p_property.name == "combine_csg_children") {
		if (is_inside_tree() && is_root_shape()) {
			p_property.usage = PROPERTY_USAGE_NO_EDITOR;
		}
	}

	bool is_collision_prefixed = p_property.name.begins_with("collision_");
	if ((is_collision_prefixed || p_property.name.begins_with("use_collision")) && is_inside_tree() && !is_root_shape()) {
		//hide collision if not root
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	} else if (is_collision_prefixed && !bool(get("use_collision"))) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

void CSGShape3D::csg_push_warning(CSGWarning p_warning) {
	brush_warning = p_warning;
}

void CSGShape3D::rebuild_brush() {
	if (is_inside_tree()) {
		_make_dirty();
	}
}

void CSGShape3D::brush_modified() {
	// The method set_csg_brush() doesn't call _make_painted() and that causes issues with undo. This can be called at that point.
	_make_painted();
}

Dictionary CSGShape3D::get_csg_brush() {
	Dictionary p_brush_data;
	p_brush_data["painted"] = painted;

	if (!brush) {
		return p_brush_data;
	}

	if (is_root_shape()) {
		return p_brush_data;
	}

	if (!painted) {
		// Only save painted brushes.
		return p_brush_data;
	}

	CSGBrush *n = _get_brush();
	if (n == nullptr) {
		return p_brush_data;
	}

	if (n->faces.size() <= 0) {
		return p_brush_data;
	}

	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<uint8_t> smooths;
	Vector<uint8_t> mat_id;
	Array materials;

	for (int i = 0; i < n->faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			vertices.push_back(n->faces[i].vertices[j]);
			uvs.push_back(n->faces[i].uvs[j]);
		}

		smooths.push_back(n->faces[i].smooth ? 1 : 0);
		mat_id.push_back(n->faces[i].material);
	}

	p_brush_data["vertices"] = vertices;
	p_brush_data["uvs"] = uvs;
	p_brush_data["smooth"] = smooths;
	// We will not save combined shapes.
	p_brush_data["inverted"] = n->faces[0].invert;
	p_brush_data["material_id"] = mat_id;
	p_brush_data["ngons"] = n->ngons;

	for (int i = 0; i < n->materials.size(); i++) {
		materials.push_back(n->materials[i]);
	}

	p_brush_data["materials"] = materials;

	return p_brush_data;
}

void CSGShape3D::set_csg_brush(const Dictionary &p_brush_data) {
	ERR_FAIL_COND_MSG(!p_brush_data.has("painted"), "Node doesn't have a valid _csg_brush Dictionary.");

	if (!p_brush_data["painted"]) {
		// Brush has not been modified or is root shape. Rebuild brush.
		return;
	}

	ERR_FAIL_COND(!p_brush_data.has("material_id"));
	ERR_FAIL_COND(!p_brush_data.has("vertices"));
	ERR_FAIL_COND(!p_brush_data.has("uvs"));
	ERR_FAIL_COND(!p_brush_data.has("smooth"));
	ERR_FAIL_COND(!p_brush_data.has("inverted"));
	ERR_FAIL_COND(!p_brush_data.has("materials"));

	painted = p_brush_data["painted"];

	Vector<uint8_t> mat_id = p_brush_data["material_id"];

	int face_count = mat_id.size();
	if (face_count < 4) {
		// The smallest shape with volume is a 3 sided pyramid.
		csg_push_warning(CSG_SET_NON_MANIFOLD);
		update_configuration_warnings();
		ERR_FAIL_MSG("Brush is not manifold!");
	}

	Vector<Vector3> faces = p_brush_data["vertices"];
	ERR_FAIL_COND_MSG((faces.size() / 3) != face_count, "The number of elements in 'vertices' should be 3 times the number of elements in 'material_id'.");

	Vector<Vector2> uvs = p_brush_data["uvs"];
	ERR_FAIL_COND_MSG(faces.size() != uvs.size(), "The number of elements in 'vertices' and 'uvs' should be the same.");

	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		bool invrt = p_brush_data["inverted"];
		Vector<uint8_t> smooth_i = p_brush_data["smooth"];
		ERR_FAIL_COND_MSG(smooth_i.size() != face_count, "The number of elements in 'smooth' and 'vertices' should be the same.");

		Array mats = p_brush_data["materials"];

		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		for (int i = 0; i < face_count; i++) {
			smoothw[i] = smooth_i[i] > 0;
			int i_mat = mat_id[i];
			if (i_mat < mats.size()) {
				Ref<Material> t_mat = mats[i_mat];
				if (t_mat.is_valid()) {
					materialsw[i] = t_mat;
				} else {
					materialsw[i] = nullptr;
				}
			}
			invertw[i] = invrt;
		}
	}

	CSGBrush *n = memnew(CSGBrush);
	n->build_from_faces(faces, uvs, smooth, materials, invert);

	if (p_brush_data.has("ngons")) {
		Vector<int> t_ngons = p_brush_data["ngons"];
		if (t_ngons.size() != face_count) {
			WARN_PRINT("The number of elements in 'ngons' and 'material_id' should be the same. Invalid ngon data!");
		} else {
			n->add_ngons(t_ngons);
		}
	} else {
		WARN_PRINT("Resource doesn't have ngon data.");
	}

	if (brush) {
		memdelete(brush);
		brush = nullptr;
	}

	{
		// Test if CSGBrush is manifold.
		HashMap<int32_t, Ref<Material>> mesh_materials;
		manifold::Manifold t_manifold;
		_pack_manifold(n, t_manifold, mesh_materials, this);
		if (t_manifold.Status() != manifold::Manifold::Error::NoError) {
			// We don't save a non-manifold brush so the warning triggers.
			memdelete(n);
			csg_push_warning(CSG_SET_NON_MANIFOLD);
			update_configuration_warnings();
			ERR_FAIL_MSG("Brush is not manifold!");
		}
		csg_push_warning(NO_WARNING);
	}

	_expand_aabb(n);
	brush = n;
	dirty = false;
	update_configuration_warnings();
	// Recalculate parent.
	_make_painted();
}

void CSGShape3D::set_uv_offsets(const Vector<int> &p_faces, const Vector2 &p_prev_offset, const Vector2 &p_offset) {
	// Use the offset obtained from `get_uv_offsets` when selecting faces.
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");

	if (n->faces.is_empty()) {
		return;
	}

	for (int i = 0; i < p_faces.size(); i++) {
		int p = p_faces[i];
		ERR_FAIL_INDEX(p, n->faces.size());
		for (int j = 0; j < 3; j++) {
			n->faces.write[p].uvs[j] = n->faces[p].uvs[j] + p_offset - p_prev_offset;
		}
	}
	_make_painted(true);
}

Vector2 CSGShape3D::get_uv_offsets(int p_face) {
	if (!brush) {
		return Vector2(0.0, 0.0);
	}

	CSGBrush *n = _get_brush();
	if (n == nullptr) {
		return Vector2(0.0, 0.0);
	}

	if (n->faces.is_empty()) {
		return Vector2(0.0, 0.0);
	}

	if (p_face > n->faces.size() || p_face < 0) {
		return Vector2(0.0, 0.0);
	}

	Vector2 smallest = n->faces[p_face].uvs[0];

	for (int j = 1; j < 3; j++) {
		smallest = smallest.min(n->faces[p_face].uvs[j]);
	}
	return smallest;
}

void CSGShape3D::set_uv_scale(const Vector<int> &p_faces, const Vector2 &p_prev_scale, const Vector2 &p_scale) {
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");

	if (n->faces.is_empty()) {
		return;
	}

	if (p_scale.x == 0.0 || p_scale.y == 0.0) {
		ERR_PRINT("Scale should not be zero!");
		notify_property_list_changed();
		return;
	}

	if (p_prev_scale.x == 0.0 || p_prev_scale.y == 0.0) {
		// We fix this mess.
		ERR_PRINT("Scale is zero, rebuilding shape.");
		_make_dirty();
		return;
	}

	Vector2 offset = get_uv_offsets(p_faces[0]);
	Vector2 n_scale = p_scale / p_prev_scale;

	for (int i = 0; i < p_faces.size(); i++) {
		int p = p_faces[i];
		ERR_FAIL_INDEX(p, n->faces.size());
		for (int j = 0; j < 3; j++) {
			n->faces.write[p].uvs[j] = (n->faces[p].uvs[j] * n_scale) + offset;
		}
	}
	_make_painted(true);
}

Vector2 CSGShape3D::get_uv_scale(int p_face) {
	if (!brush) {
		return Vector2(1.0, 1.0);
	}

	CSGBrush *n = _get_brush();
	if (n == nullptr) {
		return Vector2(1.0, 1.0);
	}

	if (n->faces.is_empty()) {
		return Vector2(1.0, 1.0);
	}

	Vector2 offset = get_uv_offsets(p_face);
	if (p_face > n->faces.size() || p_face < 0) {
		return Vector2(1.0, 1.0);
	}

	Vector2 largest = (n->faces[p_face].uvs[0] - offset).abs();

	for (int j = 1; j < 3; j++) {
		largest = largest.max(n->faces[p_face].uvs[j] - offset);
	}
	return largest;
}

void CSGShape3D::rotate_uv(const Vector<int> &p_faces, float p_angle) {
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");

	if (n->faces.is_empty()) {
		return;
	}

	Transform2D p_rotation = Transform2D(p_angle, Vector2(0.0, 0.0));
	for (int i = 0; i < p_faces.size(); i++) {
		int p = p_faces[i];
		ERR_FAIL_INDEX(p, n->faces.size());
		for (int j = 0; j < 3; j++) {
			n->faces.write[p].uvs[j] = p_rotation.xform(n->faces[p].uvs[j]);
		}
	}
	_make_painted(true);
}

void CSGShape3D::flip_x(const Vector<int> &p_faces) {
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");

	if (n->faces.is_empty()) {
		return;
	}

	float mirror_x = -1.0;
	for (int i = 0; i < p_faces.size(); i++) {
		int p = p_faces[i];
		ERR_FAIL_INDEX(p, n->faces.size());
		for (int j = 0; j < 3; j++) {
			Vector2 curr = n->faces[p].uvs[j];
			curr.x *= mirror_x;
			n->faces.write[p].uvs[j] = curr;
		}
	}
	_make_painted(true);
}

void CSGShape3D::flip_y(const Vector<int> &p_faces) {
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");

	if (n->faces.is_empty()) {
		return;
	}

	float mirror_y = -1.0;
	for (int i = 0; i < p_faces.size(); i++) {
		int p = p_faces[i];
		ERR_FAIL_INDEX(p, n->faces.size());
		for (int j = 0; j < 3; j++) {
			Vector2 curr = n->faces[p].uvs[j];
			curr.y *= mirror_y;
			n->faces.write[p].uvs[j] = curr;
		}
	}
	_make_painted(true);
}

void CSGShape3D::calculate_cube_map(const Vector<int> &p_faces, const Vector3 &uv_scale, bool p_use_global) {
	// Simple cube map unwrapping. Calculates the normal of each face and aligns them with one of three axes.
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");
	Vector3 p_scale = uv_scale;
	if (p_scale.x == 0.0) {
		// Continue working.
		p_scale.x = 1.0;
	}
	if (p_scale.y == 0.0) {
		p_scale.y = 1.0;
	}
	if (p_scale.z == 0.0) {
		p_scale.z = 1.0;
	}

	if (n->faces.is_empty()) {
		return;
	}

	// 3D goes bot to top while 2D goes top to bottom.
	Transform2D p_rotation = Transform2D(Math::deg_to_rad(180.0), Vector2(0.0, 0.0));
	Transform3D trans = get_global_transform();

	for (int i = 0; i < p_faces.size(); i++) {
		int p = p_faces[i];
		ERR_FAIL_INDEX(p, n->faces.size());

		Vector3 v1 = n->faces[p].vertices[0];
		Vector3 v2 = n->faces[p].vertices[1];
		Vector3 v3 = n->faces[p].vertices[2];

		if (p_use_global) {
			// This includes rotation, position and scale. More parameters could be added to handle rotation and position separately.
			v1 = trans.xform(v1);
			v2 = trans.xform(v2);
			v3 = trans.xform(v3);
		}

		Plane ang(v1, v2, v3);
		Vector3 nor = ang.normal.abs();

		if (nor.x > nor.y && nor.x > nor.z) {
			// Direction X, ZY.
			float mir_x = ang.normal.x < 0.0 ? -p_scale.z : p_scale.z;
			n->faces.write[p].uvs[0] = p_rotation.xform(Vector2(mir_x * v1.z, v1.y * p_scale.y));
			n->faces.write[p].uvs[1] = p_rotation.xform(Vector2(mir_x * v2.z, v2.y * p_scale.y));
			n->faces.write[p].uvs[2] = p_rotation.xform(Vector2(mir_x * v3.z, v3.y * p_scale.y));
		} else if (nor.y > nor.z) {
			// Direction Y, XZ.
			float mir_x = ang.normal.y < 0.0 ? -p_scale.x : p_scale.x;
			n->faces.write[p].uvs[0] = Vector2(mir_x * v1.x, v1.z * p_scale.z);
			n->faces.write[p].uvs[1] = Vector2(mir_x * v2.x, v2.z * p_scale.z);
			n->faces.write[p].uvs[2] = Vector2(mir_x * v3.x, v3.z * p_scale.z);
		} else {
			// Direction Z, XY.
			float mir_x = ang.normal.z < 0.0 ? p_scale.x : -p_scale.x;
			n->faces.write[p].uvs[0] = p_rotation.xform(Vector2(mir_x * v1.x, v1.y * p_scale.y));
			n->faces.write[p].uvs[1] = p_rotation.xform(Vector2(mir_x * v2.x, v2.y * p_scale.y));
			n->faces.write[p].uvs[2] = p_rotation.xform(Vector2(mir_x * v3.x, v3.y * p_scale.y));
		}
	}
	_make_painted(true);
}

void CSGShape3D::calculate_cylinder_map(const Vector<int> &p_faces, const Vector3 &uv_scale, bool p_use_global) {
	// Simple cylinder unwrapping.
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");
	Vector3 p_scale = uv_scale;
	if (p_scale.x == 0.0) {
		// Continue working.
		p_scale.x = p_scale.z == 0.0 ? 1.0 : p_scale.z;
	}
	if (p_scale.y == 0.0) {
		p_scale.y = 1.0;
	}
	if (p_scale.z == 0.0) {
		p_scale.z = p_scale.x == 0.0 ? 1.0 : p_scale.x;
	}

	if (n->faces.is_empty()) {
		return;
	}

	float flatten = 1.0 / Math::PI;
	// 3D goes bot to top while 2D goes top to bottom.
	Transform2D p_rotation = Transform2D(Math::deg_to_rad(180.0), Vector2(0.0, 0.0));
	Transform3D trans = get_global_transform();
	// Cylinder map is generated around the center, using global position breaks it, so we reset position.
	trans.set_origin(Vector3(0, 0, 0));

	for (int i = 0; i < p_faces.size(); i++) {
		int p = p_faces[i];
		ERR_FAIL_INDEX(p, n->faces.size());

		Vector3 v1 = n->faces[p].vertices[0];
		Vector3 v2 = n->faces[p].vertices[1];
		Vector3 v3 = n->faces[p].vertices[2];

		if (p_use_global) {
			// Not sure how useful this will be for cylinder.
			v1 = trans.xform(v1);
			v2 = trans.xform(v2);
			v3 = trans.xform(v3);
		}

		Plane ang(v1, v2, v3);
		if (Math::abs(ang.normal.y) > 0.6) {
			// Top and bottom faces.
			float mir_x = ang.normal.x < 0.0 ? -p_scale.x : p_scale.x;
			n->faces.write[p].uvs[0] = Vector2(mir_x * v1.x, v1.z * p_scale.z);
			n->faces.write[p].uvs[1] = Vector2(mir_x * v2.x, v2.z * p_scale.z);
			n->faces.write[p].uvs[2] = Vector2(mir_x * v3.x, v3.z * p_scale.z);
		} else {
			// This calculates two mirrored sides and then mirrors one back.
			float inv = ang.normal.z < 0.0 ? -1.0 : 1.0;
			// MAX is needed because under one specific circumstance the resulting value is -0.0 and that returns the angle -PI, causing the UV to stretch on the last face.
			float t_ang_1 = Vector2(v1.x, MAX(0.0, v1.z * inv)).normalized().angle() * flatten * inv;
			float t_ang_2 = Vector2(v2.x, MAX(0.0, v2.z * inv)).normalized().angle() * flatten * inv;
			float t_ang_3 = Vector2(v3.x, MAX(0.0, v3.z * inv)).normalized().angle() * flatten * inv;
			// Correctly calculating `uv_scale` makes the code too complicated. Using `x` only.
			n->faces.write[p].uvs[0] = p_rotation.xform(Vector2(t_ang_1 * p_scale.x, v1.y * p_scale.y));
			n->faces.write[p].uvs[1] = p_rotation.xform(Vector2(t_ang_2 * p_scale.x, v2.y * p_scale.y));
			n->faces.write[p].uvs[2] = p_rotation.xform(Vector2(t_ang_3 * p_scale.x, v3.y * p_scale.y));
		}
	}
	_make_painted(true);
}

void CSGShape3D::set_csg_face_smooth_group(const Vector<int> &p_faces, int p_smooth) {
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");

	if (n->faces.is_empty()) {
		return;
	}

	for (int i = 0; i < p_faces.size(); i++) {
		int p = p_faces[i];
		ERR_FAIL_INDEX(p, n->faces.size());
		for (int j = 0; j < 3; j++) {
			n->faces.write[p].smooth = p_smooth > 0;
		}
	}
	_make_painted(true);
}

int CSGShape3D::get_csg_face_smooth_group(int p_face) {
	if (!brush) {
		return 0;
	}

	CSGBrush *n = _get_brush();
	if (n == nullptr) {
		return 0;
	}

	if (n->faces.is_empty()) {
		return 0;
	}

	if (p_face > n->faces.size() || p_face < 0) {
		return 0;
	}

	return n->faces[p_face].smooth ? 1 : 0;
}

void CSGShape3D::set_face_material(const Vector<int> &p_faces, const Ref<Material> &p_material) {
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");

	if (n->faces.is_empty()) {
		return;
	}

	int find_mat = -2;
	if (p_material.is_valid()) {
		for (int j = 0; j < n->materials.size(); j++) {
			if (p_material == n->materials[j]) {
				find_mat = j;
				break;
			}
		}
		if (find_mat == -2) {
			find_mat = n->materials.size();
			n->materials.push_back(p_material);
		}
	} else {
		find_mat = -1;
	}

	for (int i = 0; i < p_faces.size(); i++) {
		int p = p_faces[i];
		ERR_FAIL_INDEX(p, n->faces.size());
		n->faces.write[p].material = find_mat;
	}

	{
		Vector<Ref<Material>> nmats;
		Vector<int> minds;
		minds.resize(n->faces.size());
		int k = 0;
		for (int i = 0; i < n->materials.size(); i++) {
			bool is_material_used = false;
			for (int j = 0; j < n->faces.size(); j++) {
				if (n->faces[j].material == i) {
					minds.write[j] = k;
					is_material_used = true;
				}
			}
			if (is_material_used) {
				nmats.push_back(n->materials[i]);
				k++;
			}
		}

		for (int i = 0; i < n->faces.size(); i++) {
			if (n->faces[i].material == -1) {
				n->faces.write[i].material = -1;
			} else {
				n->faces.write[i].material = minds[i];
			}
		}

		n->materials.resize(nmats.size());
		for (int i = 0; i < nmats.size(); i++) {
			n->materials.write[i] = nmats[i];
		}
	}
	_make_painted(true);
}

Ref<Material> CSGShape3D::get_face_material(int p_face) {
	// Using only one face as only one material will be returned.
	CSGBrush *n = _get_brush();
	if (n == nullptr) {
		return nullptr;
	}

	if (n->faces.is_empty()) {
		return nullptr;
	}

	if (p_face > n->faces.size() || p_face < 0) {
		return nullptr;
	}

	int mat = n->faces[p_face].material;
	if (mat < 0) {
		// mat can be -1
		return nullptr;
	}
	return n->materials[mat];
}

bool CSGShape3D::resize_brush(const Vector3 &p_prev_size, const Vector3 &p_size) {
	if (!is_inside_tree()) {
		return true;
	}

	CSGBrush *n = _get_brush();
	if (n == nullptr) {
		return true;
	}

	if (n->faces.is_empty()) {
		return false;
	}

	if (p_prev_size.x == 0.0 || p_prev_size.y == 0.0 || p_prev_size.z == 0.0) {
		ERR_PRINT("size can should be non-zero!");
		_make_dirty();
		return false;
	}

	if (p_size.x == 0.0 || p_size.y == 0.0 || p_size.z == 0.0) {
		ERR_FAIL_V_MSG(false, "Size should be non-zero.");
	}

	Vector3 n_size = p_size / p_prev_size;
	AABB aabb;
	aabb.position = n->faces[0].vertices[0];
	for (int i = 0; i < n->faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			Vector3 curr = n->faces[i].vertices[j];
			curr *= n_size;
			aabb.expand_to(curr);
			n->faces.write[i].vertices[j] = curr;
		}
	}
	if (get_child_count() < 1 || !combined_brush) {
		// This is a hack and needs a better solution to the problem of non CSGShape3D children.
		node_aabb = aabb;
	}

	return true;
}

void CSGShape3D::resize_brush_rework() {
	// Resizes the brush without lossing changes. Used for CSGTorus3D.
	if (!is_inside_tree()) {
		return;
	}

	CSGBrush *n = _get_brush();
	if (n == nullptr) {
		return;
	}

	if (n->faces.is_empty()) {
		return;
	}

	CSGBrush *b = _build_brush();

	if (b->faces.size() != n->faces.size()) {
		memdelete(b);
		csg_push_warning(NON_MANIFOLD);
		return;
	}

	AABB aabb;
	if (b) {
		if (!b->faces.is_empty()) {
			aabb.position = b->faces[0].vertices[0];
			for (const CSGBrush::Face &face : b->faces) {
				for (int i = 0; i < 3; ++i) {
					aabb.expand_to(face.vertices[i]);
				}
			}
			for (int i = 0; i < n->faces.size(); i++) {
				for (int j = 0; j < 3; j++) {
					n->faces.write[i].vertices[j] = b->faces[i].vertices[j];
				}
			}
			if (get_child_count() < 1 || !combined_brush) {
				// This is a hack and needs a better solution to the problem of non CSGShape3D children.
				node_aabb = aabb;
			}
		}
		memdelete(b);
	}
}

void CSGShape3D::set_csg_invert(bool p_inv_val) {
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");

	if (n->faces.is_empty()) {
		return;
	}

	for (int i = 0; i < n->faces.size(); i++) {
		n->faces.write[i].invert = p_inv_val;
	}
	_make_painted();
}

void CSGShape3D::set_csg_flat(bool p_mode) {
	// This changes all faces so it can't be used with CSGCylinder3D.
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");

	if (n->faces.is_empty()) {
		return;
	}

	for (int i = 0; i < n->faces.size(); i++) {
		n->faces.write[i].smooth = p_mode;
	}
}

Vector<Vector3> CSGShape3D::get_vertices() {
	// Use this for making a vertex editor.
	CSGBrush *n = _get_brush();

	Vector<Vector3> vertices;

	if (n == nullptr) {
		return vertices;
	}

	for (int i = 0; i < n->faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			Vector3 v = n->faces[i].vertices[j];
			bool found_v = false;
			for (int k = 0; k < vertices.size(); k++) {
				if (vertices[k].is_equal_approx(v)) {
					found_v = true;
					break;
				}
			}
			if (!found_v) {
				vertices.push_back(v);
			}
		}
	}

	return vertices;
}

void CSGShape3D::set_vertex_position(const Vector3 &p_curr_pos, const Vector3 &p_pos) {
	CSGBrush *n = _get_brush();
	ERR_FAIL_NULL_MSG(n, "Cannot get CSGBrush.");

	if (n->faces.is_empty()) {
		WARN_PRINT("CSGBrush is empty.");
		return;
	}

	for (int i = 0; i < n->faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			if (n->faces[i].vertices[j].is_equal_approx(p_curr_pos)) {
				n->faces.write[i].vertices[j] = p_pos;
			}
		}
	}
	_make_painted(true);
}

Vector<Vector3> CSGShape3D::get_selected_faces(const Vector<int> &p_faces) {
	// Use this to obtain a vector of selected faces to display as a mesh in editor.
	CSGBrush *n = _get_brush();

	Vector<Vector3> ret;

	if (n == nullptr) {
		return ret;
	}

	if (n->faces.is_empty()) {
		return ret;
	}

	for (int i = 0; i < p_faces.size(); i++) {
		int p = p_faces[i];
		ERR_CONTINUE_MSG(p > n->faces.size() || p < 0, "Invalid face in p_faces.");
		for (int j = 0; j < 3; j++) {
			ret.push_back(n->faces[p].vertices[j]);
		}
	}
	return ret;
}

int CSGShape3D::get_csg_num_faces() {
	CSGBrush *n = _get_brush();

	if (n == nullptr) {
		return 0;
	}

	return n->faces.size();
}

Vector<int> CSGShape3D::get_all_csg_faces() {
	Vector<int> all_faces;
	all_faces.resize(get_csg_num_faces());
	for (int i = 0; i < all_faces.size(); i++) {
		all_faces.write[i] = i;
	}
	return all_faces;
}

Vector<int> CSGShape3D::get_faces_from_ngon(int p_ngon) {
	// Returns the faces (tris) forming the requested ngon. Use in combination with get_selected_faces().
	CSGBrush *n = _get_brush();
	Vector<int> ret;

	if (n == nullptr) {
		return ret;
	}

	if (n->faces.is_empty()) {
		return ret;
	}

	ERR_FAIL_COND_V_MSG(n->ngons.is_empty() || n->num_ngons < 1, ret, "CSGBrush has no ngons!");

	ret = n->get_ngon_faces(p_ngon);
	return ret;
}

TypedArray<Vector<Vector3>> CSGShape3D::get_csg_ngon_colliders() {
	// Returns an array of Vector<Vector3> to use to create a TriangleMesh for each collider of ngon to be used in gizmos.
	TypedArray<Vector<Vector3>> ret;
	CSGBrush *n = _get_brush();

	if (n == nullptr) {
		return ret;
	}

	ERR_FAIL_COND_V_MSG(n->ngons.is_empty() || n->num_ngons < 1, ret, "CSGBrush has no ngons");
	ERR_FAIL_COND_V_MSG(n->faces.is_empty(), ret, "CSGBrush has no faces!");
	ERR_FAIL_COND_V_MSG(n->num_ngons < 0, ret, "num_ngons is less than 0.");

	for (int i = 0; i < n->num_ngons; i++) {
		Vector<int> curr_ngon = n->get_ngon_faces(i);
		if (!curr_ngon.is_empty()) {
			Vector<Vector3> curr_faces;
			for (int j = 0; j < curr_ngon.size(); j++) {
				curr_faces.push_back(n->faces[curr_ngon[j]].vertices[0]);
				curr_faces.push_back(n->faces[curr_ngon[j]].vertices[1]);
				curr_faces.push_back(n->faces[curr_ngon[j]].vertices[2]);
			}
			ret.push_back(curr_faces);
		}
	}

	// Flip faces so they can be selected from the inside.
	if (get_operation() == OPERATION_SUBTRACTION) {
		ret.reverse();
	}

	return ret;
}

bool CSGShape3D::is_painted() const {
	return painted;
}

Array CSGShape3D::get_meshes() const {
	if (root_mesh.is_valid()) {
		Array arr;
		arr.resize(2);
		arr[0] = Transform3D();
		arr[1] = root_mesh;
		return arr;
	}

	return Array();
}

PackedStringArray CSGShape3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node::get_configuration_warnings();
	// We can make the errors more specific. Also, CSGCombiner3D will no longer raise a warning.
	if (brush_warning == NON_MANIFOLD) {
		warnings.push_back(RTR("This CSGShape3D has an empty shape.\nCSGShape3D empty shapes typically occur because the mesh is not manifold.\nA manifold mesh forms a solid object without gaps, holes, or loose edges.\nEach edge must be a member of exactly two faces."));
	} else if (brush_warning == CSG_SET_NON_MANIFOLD) {
		warnings.push_back(RTR("This CSGShape3D has an empty shape because `set_csg_brush` was called on it with a non-manifold mesh.\nA manifold mesh forms a solid object without gaps, holes, or loose edges.\nEach edge must be a member of exactly two faces.\nYou can press the `Rebuild` button on the top bar to reconstruct this node's shape using its node properties.\nThis node can not be saved with a non-manifold shape and will be automatically rebuilt the next time the scene loads."));
	} else if (brush_warning == CSG_MESH_NOT_ASSIGNED) {
		warnings.push_back(RTR("This CSGMesh3D has an empty shape because its `mesh` has not been set or is not valid."));
	} else if (brush_warning == CSG_MESH_NON_MANIFOLD) {
		warnings.push_back(RTR("This CSGMesh3D has an empty shape because its `mesh` is not manifold.\nA manifold mesh forms a solid object without gaps, holes, or loose edges.\nEach edge must be a member of exactly two faces."));
	}
	return warnings;
}

Ref<TriangleMesh> CSGShape3D::generate_triangle_mesh() const {
	if (root_mesh.is_valid()) {
		return root_mesh->generate_triangle_mesh();
	}
	return Ref<TriangleMesh>();
}

void CSGShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_root_shape"), &CSGShape3D::is_root_shape);

	ClassDB::bind_method(D_METHOD("set_operation", "operation"), &CSGShape3D::set_operation);
	ClassDB::bind_method(D_METHOD("get_operation"), &CSGShape3D::get_operation);

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("_update_shape"), &CSGShape3D::update_shape);
	ClassDB::bind_method(D_METHOD("set_snap", "snap"), &CSGShape3D::set_snap);
	ClassDB::bind_method(D_METHOD("get_snap"), &CSGShape3D::get_snap);
#endif // DISABLE_DEPRECATED

#ifndef PHYSICS_3D_DISABLED
	ClassDB::bind_method(D_METHOD("set_use_collision", "operation"), &CSGShape3D::set_use_collision);
	ClassDB::bind_method(D_METHOD("is_using_collision"), &CSGShape3D::is_using_collision);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "layer"), &CSGShape3D::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &CSGShape3D::get_collision_layer);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &CSGShape3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &CSGShape3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_value", "layer_number", "value"), &CSGShape3D::set_collision_mask_value);
	ClassDB::bind_method(D_METHOD("get_collision_mask_value", "layer_number"), &CSGShape3D::get_collision_mask_value);

	ClassDB::bind_method(D_METHOD("_get_root_collision_instance"), &CSGShape3D::_get_root_collision_instance);

	ClassDB::bind_method(D_METHOD("set_collision_layer_value", "layer_number", "value"), &CSGShape3D::set_collision_layer_value);
	ClassDB::bind_method(D_METHOD("get_collision_layer_value", "layer_number"), &CSGShape3D::get_collision_layer_value);

	ClassDB::bind_method(D_METHOD("set_collision_priority", "priority"), &CSGShape3D::set_collision_priority);
	ClassDB::bind_method(D_METHOD("get_collision_priority"), &CSGShape3D::get_collision_priority);

	ClassDB::bind_method(D_METHOD("bake_collision_shape"), &CSGShape3D::bake_collision_shape);
#endif // PHYSICS_3D_DISABLED

	ClassDB::bind_method(D_METHOD("set_calculate_tangents", "enabled"), &CSGShape3D::set_calculate_tangents);
	ClassDB::bind_method(D_METHOD("is_calculating_tangents"), &CSGShape3D::is_calculating_tangents);

	ClassDB::bind_method(D_METHOD("rebuild_brush"), &CSGShape3D::rebuild_brush);
	ClassDB::bind_method(D_METHOD("brush_modified"), &CSGShape3D::brush_modified);
	ClassDB::bind_method(D_METHOD("set_csg_brush", "csg_brush"), &CSGShape3D::set_csg_brush);
	ClassDB::bind_method(D_METHOD("get_csg_brush"), &CSGShape3D::get_csg_brush);
	ClassDB::bind_method(D_METHOD("set_uv_offsets", "faces", "prev_offset", "offset"), &CSGShape3D::set_uv_offsets);
	ClassDB::bind_method(D_METHOD("get_uv_offsets", "face"), &CSGShape3D::get_uv_offsets);
	ClassDB::bind_method(D_METHOD("set_uv_scale", "faces", "prev_scale", "scale"), &CSGShape3D::set_uv_scale);
	ClassDB::bind_method(D_METHOD("get_uv_scale", "face"), &CSGShape3D::get_uv_scale);
	ClassDB::bind_method(D_METHOD("rotate_uv", "faces", "angle"), &CSGShape3D::rotate_uv);
	ClassDB::bind_method(D_METHOD("flip_x", "faces"), &CSGShape3D::flip_x);
	ClassDB::bind_method(D_METHOD("flip_y", "faces"), &CSGShape3D::flip_y);
	ClassDB::bind_method(D_METHOD("calculate_cube_map", "faces", "uv_scale", "use_global"), &CSGShape3D::calculate_cube_map);
	ClassDB::bind_method(D_METHOD("calculate_cylinder_map", "faces", "uv_scale", "use_global"), &CSGShape3D::calculate_cylinder_map);
	ClassDB::bind_method(D_METHOD("set_csg_face_smooth_group", "faces", "smooth"), &CSGShape3D::set_csg_face_smooth_group);
	ClassDB::bind_method(D_METHOD("get_csg_face_smooth_group", "face"), &CSGShape3D::get_csg_face_smooth_group);
	ClassDB::bind_method(D_METHOD("set_face_material", "faces", "material"), &CSGShape3D::set_face_material);
	ClassDB::bind_method(D_METHOD("get_face_material", "face"), &CSGShape3D::get_face_material);
	ClassDB::bind_method(D_METHOD("resize_brush", "prev_size", "size"), &CSGShape3D::resize_brush);
	ClassDB::bind_method(D_METHOD("get_vertices"), &CSGShape3D::get_vertices);
	ClassDB::bind_method(D_METHOD("set_vertex_position", "from", "to"), &CSGShape3D::set_vertex_position);
	ClassDB::bind_method(D_METHOD("get_selected_faces", "faces"), &CSGShape3D::get_selected_faces);
	ClassDB::bind_method(D_METHOD("get_csg_num_faces"), &CSGShape3D::get_csg_num_faces);
	ClassDB::bind_method(D_METHOD("get_all_csg_faces"), &CSGShape3D::get_all_csg_faces);
	ClassDB::bind_method(D_METHOD("get_faces_from_ngon", "ngon"), &CSGShape3D::get_faces_from_ngon);
	ClassDB::bind_method(D_METHOD("get_csg_ngon_colliders"), &CSGShape3D::get_csg_ngon_colliders);

	ClassDB::bind_method(D_METHOD("get_meshes"), &CSGShape3D::get_meshes);

	ClassDB::bind_method(D_METHOD("bake_static_mesh"), &CSGShape3D::bake_static_mesh);

	ClassDB::bind_method(D_METHOD("set_autosmooth", "autosmooth"), &CSGShape3D::set_autosmooth);
	ClassDB::bind_method(D_METHOD("is_autosmooth"), &CSGShape3D::is_autosmooth);

	ClassDB::bind_method(D_METHOD("set_smoothing_angle", "smoothing_angle"), &CSGShape3D::set_smoothing_angle);
	ClassDB::bind_method(D_METHOD("get_smoothing_angle"), &CSGShape3D::get_smoothing_angle);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autosmooth"), "set_autosmooth", "is_autosmooth");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "smoothing_angle", PROPERTY_HINT_RANGE, "0,180,0.1,degrees"), "set_smoothing_angle", "get_smoothing_angle");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_csg_brush", PROPERTY_HINT_NO_NODEPATH, "", PROPERTY_USAGE_NO_EDITOR), "set_csg_brush", "get_csg_brush");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operation", PROPERTY_HINT_ENUM, "Union,Intersection,Subtraction"), "set_operation", "get_operation");
#ifndef DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "snap", PROPERTY_HINT_RANGE, "0.000001,1,0.000001,suffix:m", PROPERTY_USAGE_NONE), "set_snap", "get_snap");
#endif // DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "calculate_tangents"), "set_calculate_tangents", "is_calculating_tangents");

#ifndef PHYSICS_3D_DISABLED
	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_collision"), "set_use_collision", "is_using_collision");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_priority"), "set_collision_priority", "get_collision_priority");
#endif // PHYSICS_3D_DISABLED

	BIND_ENUM_CONSTANT(OPERATION_UNION);
	BIND_ENUM_CONSTANT(OPERATION_INTERSECTION);
	BIND_ENUM_CONSTANT(OPERATION_SUBTRACTION);
}

CSGShape3D::CSGShape3D() {
	set_notify_local_transform(true);
}

CSGShape3D::~CSGShape3D() {
	if (brush) {
		memdelete(brush);
		brush = nullptr;
	}
	if (combined_brush) {
		memdelete(combined_brush);
		combined_brush = nullptr;
	}
}

//////////////////////////////////

CSGBrush *CSGCombiner3D::_build_brush() {
	csg_push_warning(NO_WARNING);
	return memnew(CSGBrush); //does not build anything
}

CSGCombiner3D::CSGCombiner3D() {
}

/////////////////////

CSGBrush *CSGPrimitive3D::_create_brush_from_arrays(const Vector<Vector3> &p_vertices, const Vector<Vector2> &p_uv, const Vector<bool> &p_smooth, const Vector<Ref<Material>> &p_materials, const Vector<int> &p_ngons) {
	CSGBrush *new_brush = memnew(CSGBrush);

	Vector<bool> invert;
	invert.resize(p_vertices.size() / 3);
	{
		int ic = invert.size();
		bool *w = invert.ptrw();
		for (int i = 0; i < ic; i++) {
			w[i] = flip_faces;
		}
	}
	new_brush->build_from_faces(p_vertices, p_uv, p_smooth, p_materials, invert);
	new_brush->add_ngons(p_ngons);

	// Test if CSGBrush is manifold.
	HashMap<int32_t, Ref<Material>> mesh_materials;
	manifold::Manifold t_manifold;
	_pack_manifold(new_brush, t_manifold, mesh_materials, this);
	if (t_manifold.Status() != manifold::Manifold::Error::NoError) {
		// We don't save a non-manifold brush so the warning triggers.
		memdelete(new_brush);
		csg_push_warning(CSG_MESH_NON_MANIFOLD);
		update_configuration_warnings();
		ERR_FAIL_V_MSG(nullptr, "CSGMesh3D brush is not manifold!");
	} else {
		csg_push_warning(NO_WARNING);
	}

	return new_brush;
}

void CSGPrimitive3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_flip_faces", "flip_faces"), &CSGPrimitive3D::set_flip_faces);
	ClassDB::bind_method(D_METHOD("get_flip_faces"), &CSGPrimitive3D::get_flip_faces);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_faces"), "set_flip_faces", "get_flip_faces");
}

void CSGPrimitive3D::set_flip_faces(bool p_invert) {
	if (flip_faces == p_invert) {
		return;
	}

	flip_faces = p_invert;

	if (!is_root_shape() && is_inside_tree()) {
		set_csg_invert(flip_faces);
	}

	_make_painted();
}

bool CSGPrimitive3D::get_flip_faces() {
	return flip_faces;
}

CSGPrimitive3D::CSGPrimitive3D() {
	flip_faces = false;
}

/////////////////////

CSGBrush *CSGMesh3D::_build_brush() {
	if (mesh.is_null()) {
		csg_push_warning(CSG_MESH_NOT_ASSIGNED);
		return memnew(CSGBrush);
	}

	Vector<Vector3> vertices;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<Vector2> uvs;
	Ref<Material> base_material = get_material();
	Vector<int> ngons;

	int ngon_counter = 0;

	for (int i = 0; i < mesh->get_surface_count(); i++) {
		if (mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
			continue;
		}

		Array arrays = mesh->surface_get_arrays(i);

		if (arrays.is_empty()) {
			_make_dirty();
			ERR_FAIL_COND_V(arrays.is_empty(), memnew(CSGBrush));
		}

		Vector<Vector3> avertices = arrays[Mesh::ARRAY_VERTEX];
		if (avertices.is_empty()) {
			continue;
		}

		const Vector3 *vr = avertices.ptr();

		Vector<Vector3> anormals = arrays[Mesh::ARRAY_NORMAL];
		const Vector3 *nr = nullptr;
		if (anormals.size()) {
			nr = anormals.ptr();
		}

		Vector<Vector2> auvs = arrays[Mesh::ARRAY_TEX_UV];
		const Vector2 *uvr = nullptr;
		if (auvs.size()) {
			uvr = auvs.ptr();
		}

		Ref<Material> mat;
		if (base_material.is_valid()) {
			mat = base_material;
		} else {
			mat = mesh->surface_get_material(i);
		}

		Vector<int> aindices = arrays[Mesh::ARRAY_INDEX];
		if (aindices.size()) {
			int as = vertices.size();
			int is = aindices.size();

			vertices.resize(as + is);
			smooth.resize((as + is) / 3);
			materials.resize((as + is) / 3);
			uvs.resize(as + is);
			ngons.resize((as + is) / 3);

			Vector3 *vw = vertices.ptrw();
			bool *sw = smooth.ptrw();
			Vector2 *uvw = uvs.ptrw();
			Ref<Material> *mw = materials.ptrw();
			int *ngonw = ngons.ptrw();

			const int *ir = aindices.ptr();

			Vector3 prev_tri_normal = Vector3(0, 0, 0);

			for (int j = 0; j < is; j += 3) {
				Vector3 vertex[3];
				Vector3 normal[3];
				Vector2 uv[3];

				for (int k = 0; k < 3; k++) {
					int idx = ir[j + k];
					vertex[k] = vr[idx];
					if (nr) {
						normal[k] = nr[idx];
					}
					if (uvr) {
						uv[k] = uvr[idx];
					}
				}

				bool flat = normal[0].is_equal_approx(normal[1]) && normal[0].is_equal_approx(normal[2]);

				if (flat) {
					if (!prev_tri_normal.is_equal_approx(normal[0])) {
						ngon_counter++;
					}
					prev_tri_normal = normal[0];
				} else {
					prev_tri_normal = Vector3(0, 0, 0);
					ngon_counter++;
				}
				ngonw[(as + j) / 3] = ngon_counter;

				vw[as + j + 0] = vertex[0];
				vw[as + j + 1] = vertex[1];
				vw[as + j + 2] = vertex[2];

				uvw[as + j + 0] = uv[0];
				uvw[as + j + 1] = uv[1];
				uvw[as + j + 2] = uv[2];

				sw[(as + j) / 3] = !flat;
				mw[(as + j) / 3] = mat;
			}
		} else {
			int as = vertices.size();
			int is = avertices.size();

			vertices.resize(as + is);
			smooth.resize((as + is) / 3);
			uvs.resize(as + is);
			materials.resize((as + is) / 3);
			ngons.resize((as + is) / 3);

			Vector3 *vw = vertices.ptrw();
			bool *sw = smooth.ptrw();
			Vector2 *uvw = uvs.ptrw();
			Ref<Material> *mw = materials.ptrw();
			int *ngonw = ngons.ptrw();

			Vector3 prev_tri_normal = Vector3(0, 0, 0);

			for (int j = 0; j < is; j += 3) {
				Vector3 vertex[3];
				Vector3 normal[3];
				Vector2 uv[3];

				for (int k = 0; k < 3; k++) {
					vertex[k] = vr[j + k];
					if (nr) {
						normal[k] = nr[j + k];
					}
					if (uvr) {
						uv[k] = uvr[j + k];
					}
				}

				bool flat = normal[0].is_equal_approx(normal[1]) && normal[0].is_equal_approx(normal[2]);

				if (flat) {
					if (!prev_tri_normal.is_equal_approx(normal[0])) {
						ngon_counter++;
					}
					prev_tri_normal = normal[0];
				} else {
					prev_tri_normal = Vector3(0, 0, 0);
					ngon_counter++;
				}
				ngonw[(as + j) / 3] = ngon_counter;

				vw[as + j + 0] = vertex[0];
				vw[as + j + 1] = vertex[1];
				vw[as + j + 2] = vertex[2];

				uvw[as + j + 0] = uv[0];
				uvw[as + j + 1] = uv[1];
				uvw[as + j + 2] = uv[2];

				sw[(as + j) / 3] = !flat;
				mw[(as + j) / 3] = mat;
			}
		}
	}

	if (vertices.is_empty()) {
		csg_push_warning(CSG_MESH_NON_MANIFOLD);
		return memnew(CSGBrush);
	}

	csg_push_warning(NO_WARNING);
	return _create_brush_from_arrays(vertices, uvs, smooth, materials, ngons);
}

void CSGMesh3D::_mesh_changed() {
	_make_dirty();

	callable_mp((Node3D *)this, &Node3D::update_gizmos).call_deferred();
}

void CSGMesh3D::set_material(const Ref<Material> &p_material) {
	if (material == p_material) {
		return;
	}
	material = p_material;
	if (is_inside_tree()) {
		set_face_material(get_all_csg_faces(), material);
	}
}

Ref<Material> CSGMesh3D::get_material() const {
	return material;
}

void CSGMesh3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &CSGMesh3D::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &CSGMesh3D::get_mesh);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGMesh3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGMesh3D::get_material);

	// Hide PrimitiveMeshes that are always non-manifold and therefore can't be used as CSG meshes.
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh,-PlaneMesh,-PointMesh,-QuadMesh,-RibbonTrailMesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "BaseMaterial3D,ShaderMaterial"), "set_material", "get_material");
}

void CSGMesh3D::set_mesh(const Ref<Mesh> &p_mesh) {
	if (mesh == p_mesh) {
		return;
	}
	if (mesh.is_valid()) {
		mesh->disconnect_changed(callable_mp(this, &CSGMesh3D::_mesh_changed));
	}
	mesh = p_mesh;

	if (mesh.is_valid()) {
		mesh->connect_changed(callable_mp(this, &CSGMesh3D::_mesh_changed));
	}

	_mesh_changed();
}

Ref<Mesh> CSGMesh3D::get_mesh() {
	return mesh;
}

////////////////////////////////

CSGBrush *CSGSphere3D::_build_brush() {
	// set our bounding box

	CSGBrush *new_brush = memnew(CSGBrush);

	int face_count = rings * radial_segments * 2 - radial_segments * 2;

	bool invert_val = get_flip_faces();
	Ref<Material> base_material = get_material();

	Vector<Vector3> faces;
	Vector<Vector2> uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	Vector<int> ngons;
	ngons.resize(face_count);

	faces.resize(face_count * 3);
	uvs.resize(face_count * 3);

	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *uvsw = uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();
		int *ngonw = ngons.ptrw();

		int ngon_counter = 0;

		// We want to follow an order that's convenient for UVs.
		// For latitude step we start at the top and move down like in an image.
		const double latitude_step = -Math::PI / rings;
		const double longitude_step = Math::TAU / radial_segments;
		int face = 0;
		for (int i = 0; i < rings; i++) {
			double cos0 = 0;
			double sin0 = 1;
			if (i > 0) {
				double latitude0 = latitude_step * i + Math::TAU / 4;
				cos0 = Math::cos(latitude0);
				sin0 = Math::sin(latitude0);
			}
			double v0 = double(i) / rings;

			double cos1 = 0;
			double sin1 = -1;
			if (i < rings - 1) {
				double latitude1 = latitude_step * (i + 1) + Math::TAU / 4;
				cos1 = Math::cos(latitude1);
				sin1 = Math::sin(latitude1);
			}
			double v1 = double(i + 1) / rings;

			for (int j = 0; j < radial_segments; j++) {
				double longitude0 = longitude_step * j;
				// We give sin to X and cos to Z on purpose.
				// This allows UVs to be CCW on +X so it maps to images well.
				double x0 = Math::sin(longitude0);
				double z0 = Math::cos(longitude0);
				double u0 = double(j) / radial_segments;

				double longitude1 = longitude_step * (j + 1);
				if (j == radial_segments - 1) {
					longitude1 = 0;
				}

				double x1 = Math::sin(longitude1);
				double z1 = Math::cos(longitude1);
				double u1 = double(j + 1) / radial_segments;

				Vector3 v[4] = {
					Vector3(x0 * cos0, sin0, z0 * cos0) * radius,
					Vector3(x1 * cos0, sin0, z1 * cos0) * radius,
					Vector3(x1 * cos1, sin1, z1 * cos1) * radius,
					Vector3(x0 * cos1, sin1, z0 * cos1) * radius,
				};

				Vector2 u[4] = {
					Vector2(u0, v0),
					Vector2(u1, v0),
					Vector2(u1, v1),
					Vector2(u0, v1),
				};

				// Draw the first face, but skip this at the north pole (i == 0).
				if (i > 0) {
					facesw[face * 3 + 0] = v[0];
					facesw[face * 3 + 1] = v[1];
					facesw[face * 3 + 2] = v[2];

					uvsw[face * 3 + 0] = u[0];
					uvsw[face * 3 + 1] = u[1];
					uvsw[face * 3 + 2] = u[2];

					smoothw[face] = smooth_faces;
					invertw[face] = invert_val;
					materialsw[face] = base_material;

					ngonw[face] = ngon_counter;

					face++;
				}

				// Draw the second face, but skip this at the south pole (i == rings - 1).
				if (i < rings - 1) {
					facesw[face * 3 + 0] = v[2];
					facesw[face * 3 + 1] = v[3];
					facesw[face * 3 + 2] = v[0];

					uvsw[face * 3 + 0] = u[2];
					uvsw[face * 3 + 1] = u[3];
					uvsw[face * 3 + 2] = u[0];

					smoothw[face] = smooth_faces;
					invertw[face] = invert_val;
					materialsw[face] = base_material;

					ngonw[face] = ngon_counter;

					face++;
				}
				ngon_counter++;
			}
		}

		if (face != face_count) {
			ERR_PRINT("Face mismatch bug! fix code");
		}
	}

	new_brush->build_from_faces(faces, uvs, smooth, materials, invert);
	new_brush->add_ngons(ngons);
	csg_push_warning(NO_WARNING);

	return new_brush;
}

void CSGSphere3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CSGSphere3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CSGSphere3D::get_radius);

	ClassDB::bind_method(D_METHOD("set_radial_segments", "radial_segments"), &CSGSphere3D::set_radial_segments);
	ClassDB::bind_method(D_METHOD("get_radial_segments"), &CSGSphere3D::get_radial_segments);
	ClassDB::bind_method(D_METHOD("set_rings", "rings"), &CSGSphere3D::set_rings);
	ClassDB::bind_method(D_METHOD("get_rings"), &CSGSphere3D::get_rings);

	ClassDB::bind_method(D_METHOD("set_smooth_faces", "smooth_faces"), &CSGSphere3D::set_smooth_faces);
	ClassDB::bind_method(D_METHOD("get_smooth_faces"), &CSGSphere3D::get_smooth_faces);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGSphere3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGSphere3D::get_material);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "1,100,1"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "1,100,1"), "set_rings", "get_rings");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "BaseMaterial3D,ShaderMaterial"), "set_material", "get_material");
}

void CSGSphere3D::set_radius(const float p_radius) {
	ERR_FAIL_COND(p_radius <= 0);
	if (!is_painted()) {
		radius = p_radius;
	} else if (resize_brush(Vector3(radius, radius, radius), Vector3(p_radius, p_radius, p_radius))) {
		radius = p_radius;
	}
	_make_painted();
	update_gizmos();
}

float CSGSphere3D::get_radius() const {
	return radius;
}

void CSGSphere3D::set_radial_segments(const int p_radial_segments) {
	radial_segments = p_radial_segments > 4 ? p_radial_segments : 4;
	rebuild_brush();
	update_gizmos();
}

int CSGSphere3D::get_radial_segments() const {
	return radial_segments;
}

void CSGSphere3D::set_rings(const int p_rings) {
	rings = p_rings > 1 ? p_rings : 1;
	rebuild_brush();
	update_gizmos();
}

int CSGSphere3D::get_rings() const {
	return rings;
}

void CSGSphere3D::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	if (is_inside_tree()) {
		set_csg_flat(smooth_faces);
	}
	_make_painted();
}

bool CSGSphere3D::get_smooth_faces() const {
	return smooth_faces;
}

void CSGSphere3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	if (is_inside_tree()) {
		set_face_material(get_all_csg_faces(), material);
	}
}

Ref<Material> CSGSphere3D::get_material() const {
	return material;
}

CSGSphere3D::CSGSphere3D() {
	// defaults
	radius = 0.5;
	radial_segments = 12;
	rings = 6;
	smooth_faces = true;
}

///////////////

CSGBrush *CSGBox3D::_build_brush() {
	// set our bounding box

	CSGBrush *new_brush = memnew(CSGBrush);

	int face_count = 12; //it's a cube..

	bool invert_val = get_flip_faces();
	Ref<Material> base_material = get_material();

	Vector<Vector3> faces;
	Vector<Vector2> uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	Vector<int> ngons;
	ngons.resize(face_count);

	faces.resize(face_count * 3);
	uvs.resize(face_count * 3);

	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *uvsw = uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();
		int *ngonw = ngons.ptrw();

		int face = 0;

		Vector3 vertex_mul = size / 2;

		{
			int ngon_counter = 0;
			for (int i = 0; i < 6; i++) {
				Vector3 face_points[4];
				float uv_points[8] = { 0, 0, 0, 1, 1, 1, 1, 0 };

				for (int j = 0; j < 4; j++) {
					float v[3];
					v[0] = 1.0;
					v[1] = 1 - 2 * ((j >> 1) & 1);
					v[2] = v[1] * (1 - 2 * (j & 1));

					for (int k = 0; k < 3; k++) {
						if (i < 3) {
							face_points[j][(i + k) % 3] = v[k];
						} else {
							face_points[3 - j][(i + k) % 3] = -v[k];
						}
					}
				}

				Vector2 u[4];
				for (int j = 0; j < 4; j++) {
					u[j] = Vector2(uv_points[j * 2 + 0], uv_points[j * 2 + 1]);
				}

				//face 1
				facesw[face * 3 + 0] = face_points[0] * vertex_mul;
				facesw[face * 3 + 1] = face_points[1] * vertex_mul;
				facesw[face * 3 + 2] = face_points[2] * vertex_mul;

				uvsw[face * 3 + 0] = u[0];
				uvsw[face * 3 + 1] = u[1];
				uvsw[face * 3 + 2] = u[2];

				smoothw[face] = false;
				invertw[face] = invert_val;
				materialsw[face] = base_material;
				ngonw[face] = ngon_counter;

				face++;
				//face 2
				facesw[face * 3 + 0] = face_points[2] * vertex_mul;
				facesw[face * 3 + 1] = face_points[3] * vertex_mul;
				facesw[face * 3 + 2] = face_points[0] * vertex_mul;

				uvsw[face * 3 + 0] = u[2];
				uvsw[face * 3 + 1] = u[3];
				uvsw[face * 3 + 2] = u[0];

				smoothw[face] = false;
				invertw[face] = invert_val;
				materialsw[face] = base_material;
				ngonw[face] = ngon_counter;
				ngon_counter++;

				face++;
			}
		}

		if (face != face_count) {
			ERR_PRINT("Face mismatch bug! fix code");
		}
	}

	new_brush->build_from_faces(faces, uvs, smooth, materials, invert);
	new_brush->add_ngons(ngons);
	csg_push_warning(NO_WARNING);

	return new_brush;
}

void CSGBox3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &CSGBox3D::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &CSGBox3D::get_size);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGBox3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGBox3D::get_material);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size", PROPERTY_HINT_NONE, "suffix:m"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "BaseMaterial3D,ShaderMaterial"), "set_material", "get_material");
}

void CSGBox3D::set_size(const Vector3 &p_size) {
	if (!is_painted()) {
		size = p_size;
	} else if (resize_brush(size / 2, p_size / 2)) {
		size = p_size;
	}
	_make_painted();
	update_gizmos();
}

Vector3 CSGBox3D::get_size() const {
	return size;
}

#ifndef DISABLE_DEPRECATED
// Kept for compatibility from 3.x to 4.0.
bool CSGBox3D::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "width") {
		size.x = p_value;
		_make_painted();
		update_gizmos();
		return true;
	} else if (p_name == "height") {
		size.y = p_value;
		_make_painted();
		update_gizmos();
		return true;
	} else if (p_name == "depth") {
		size.z = p_value;
		_make_painted();
		update_gizmos();
		return true;
	} else {
		return false;
	}
}
#endif

void CSGBox3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	if (is_inside_tree()) {
		set_face_material(get_all_csg_faces(), material);
	}
	update_gizmos();
}

Ref<Material> CSGBox3D::get_material() const {
	return material;
}

///////////////

CSGBrush *CSGCylinder3D::_build_brush() {
	// set our bounding box

	CSGBrush *new_brush = memnew(CSGBrush);

	int face_count = sides * (cone ? 1 : 2) + sides + (cone ? 0 : sides);

	bool invert_val = get_flip_faces();
	Ref<Material> base_material = get_material();

	Vector<Vector3> faces;
	Vector<Vector2> uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	Vector<int> ngons;
	ngons.resize(face_count);

	faces.resize(face_count * 3);
	uvs.resize(face_count * 3);

	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *uvsw = uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		int *ngonw = ngons.ptrw();

		int face = 0;

		Vector3 vertex_mul(radius, height * 0.5, radius);

		{
			int ngon_counter = cone ? 1 : 2;
			for (int i = 0; i < sides; i++) {
				float inc = float(i) / sides;
				float inc_n = float((i + 1)) / sides;
				if (i == sides - 1) {
					inc_n = 0;
				}

				float ang = inc * Math::TAU;
				float ang_n = inc_n * Math::TAU;

				Vector3 face_base(Math::cos(ang), 0, Math::sin(ang));
				Vector3 face_base_n(Math::cos(ang_n), 0, Math::sin(ang_n));

				Vector3 face_points[4] = {
					face_base + Vector3(0, -1, 0),
					face_base_n + Vector3(0, -1, 0),
					face_base_n * (cone ? 0.0 : 1.0) + Vector3(0, 1, 0),
					face_base * (cone ? 0.0 : 1.0) + Vector3(0, 1, 0),
				};

				Vector2 u[4] = {
					Vector2(inc, 0),
					Vector2(inc_n, 0),
					Vector2(inc_n, 1),
					Vector2(inc, 1),
				};

				//side face 1
				facesw[face * 3 + 0] = face_points[0] * vertex_mul;
				facesw[face * 3 + 1] = face_points[1] * vertex_mul;
				facesw[face * 3 + 2] = face_points[2] * vertex_mul;

				uvsw[face * 3 + 0] = u[0];
				uvsw[face * 3 + 1] = u[1];
				uvsw[face * 3 + 2] = u[2];

				smoothw[face] = smooth_faces;
				invertw[face] = invert_val;
				materialsw[face] = base_material;

				ngonw[face] = ngon_counter;

				face++;

				if (!cone) {
					//side face 2
					facesw[face * 3 + 0] = face_points[2] * vertex_mul;
					facesw[face * 3 + 1] = face_points[3] * vertex_mul;
					facesw[face * 3 + 2] = face_points[0] * vertex_mul;

					uvsw[face * 3 + 0] = u[2];
					uvsw[face * 3 + 1] = u[3];
					uvsw[face * 3 + 2] = u[0];

					smoothw[face] = smooth_faces;
					invertw[face] = invert_val;
					materialsw[face] = base_material;

					ngonw[face] = ngon_counter;

					face++;
				}

				ngon_counter++;

				//bottom face 1
				facesw[face * 3 + 0] = face_points[1] * vertex_mul;
				facesw[face * 3 + 1] = face_points[0] * vertex_mul;
				facesw[face * 3 + 2] = Vector3(0, -1, 0) * vertex_mul;

				uvsw[face * 3 + 0] = Vector2(face_points[1].x, face_points[1].y) * 0.5 + Vector2(0.5, 0.5);
				uvsw[face * 3 + 1] = Vector2(face_points[0].x, face_points[0].y) * 0.5 + Vector2(0.5, 0.5);
				uvsw[face * 3 + 2] = Vector2(0.5, 0.5);

				smoothw[face] = false;
				invertw[face] = invert_val;
				materialsw[face] = base_material;
				ngonw[face] = 0;
				face++;

				if (!cone) {
					//top face 1
					facesw[face * 3 + 0] = face_points[3] * vertex_mul;
					facesw[face * 3 + 1] = face_points[2] * vertex_mul;
					facesw[face * 3 + 2] = Vector3(0, 1, 0) * vertex_mul;

					uvsw[face * 3 + 0] = Vector2(face_points[1].x, face_points[1].y) * 0.5 + Vector2(0.5, 0.5);
					uvsw[face * 3 + 1] = Vector2(face_points[0].x, face_points[0].y) * 0.5 + Vector2(0.5, 0.5);
					uvsw[face * 3 + 2] = Vector2(0.5, 0.5);

					smoothw[face] = false;
					invertw[face] = invert_val;
					materialsw[face] = base_material;
					ngonw[face] = 1;
					face++;
				}
			}
		}

		if (face != face_count) {
			ERR_PRINT("Face mismatch bug! fix code");
		}
	}

	new_brush->build_from_faces(faces, uvs, smooth, materials, invert);
	new_brush->add_ngons(ngons);
	csg_push_warning(NO_WARNING);

	return new_brush;
}

void CSGCylinder3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CSGCylinder3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CSGCylinder3D::get_radius);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGCylinder3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGCylinder3D::get_height);

	ClassDB::bind_method(D_METHOD("set_sides", "sides"), &CSGCylinder3D::set_sides);
	ClassDB::bind_method(D_METHOD("get_sides"), &CSGCylinder3D::get_sides);

	ClassDB::bind_method(D_METHOD("set_cone", "cone"), &CSGCylinder3D::set_cone);
	ClassDB::bind_method(D_METHOD("is_cone"), &CSGCylinder3D::is_cone);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGCylinder3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGCylinder3D::get_material);

	ClassDB::bind_method(D_METHOD("set_smooth_faces", "smooth_faces"), &CSGCylinder3D::set_smooth_faces);
	ClassDB::bind_method(D_METHOD("get_smooth_faces"), &CSGCylinder3D::get_smooth_faces);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,1000.0,0.001,or_greater,exp,suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,1000.0,0.001,or_greater,exp,suffix:m"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_sides", "get_sides");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cone"), "set_cone", "is_cone");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "BaseMaterial3D,ShaderMaterial"), "set_material", "get_material");
}

void CSGCylinder3D::set_radius(const float p_radius) {
	if (!is_painted()) {
		radius = p_radius;
	} else if (resize_brush(Vector3(radius, 1.0, radius), Vector3(p_radius, 1.0, p_radius))) {
		radius = p_radius;
	}
	_make_painted();
	update_gizmos();
}

float CSGCylinder3D::get_radius() const {
	return radius;
}

void CSGCylinder3D::set_height(const float p_height) {
	if (!is_painted()) {
		height = p_height;
	} else if (resize_brush(Vector3(1.0, height * 0.5, 1.0), Vector3(1.0, p_height * 0.5, 1.0))) {
		height = p_height;
	}
	_make_painted();
	update_gizmos();
}

float CSGCylinder3D::get_height() const {
	return height;
}

void CSGCylinder3D::set_sides(const int p_sides) {
	ERR_FAIL_COND(p_sides < 3);
	sides = p_sides;
	if (is_inside_tree()) {
		rebuild_brush();
	}
	update_gizmos();
}

int CSGCylinder3D::get_sides() const {
	return sides;
}

void CSGCylinder3D::set_cone(const bool p_cone) {
	cone = p_cone;
	rebuild_brush();
	update_gizmos();
}

bool CSGCylinder3D::is_cone() const {
	return cone;
}

void CSGCylinder3D::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	_make_painted();
}

bool CSGCylinder3D::get_smooth_faces() const {
	return smooth_faces;
}

void CSGCylinder3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	if (is_inside_tree()) {
		set_face_material(get_all_csg_faces(), material);
	}
}

Ref<Material> CSGCylinder3D::get_material() const {
	return material;
}

CSGCylinder3D::CSGCylinder3D() {
	// defaults
	radius = 0.5;
	height = 2.0;
	sides = 8;
	cone = false;
	smooth_faces = true;
}

///////////////

CSGBrush *CSGTorus3D::_build_brush() {
	// set our bounding box

	float min_radius = inner_radius;
	float max_radius = outer_radius;

	if (min_radius == max_radius) {
		return memnew(CSGBrush); //sorry, can't
	}

	if (min_radius > max_radius) {
		SWAP(min_radius, max_radius);
	}

	float radius = (max_radius - min_radius) * 0.5;

	CSGBrush *new_brush = memnew(CSGBrush);

	int face_count = ring_sides * sides * 2;

	bool invert_val = get_flip_faces();
	Ref<Material> base_material = get_material();

	Vector<Vector3> faces;
	Vector<Vector2> uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	Vector<int> ngons;
	ngons.resize(face_count);

	faces.resize(face_count * 3);
	uvs.resize(face_count * 3);

	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *uvsw = uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		int *ngonw = ngons.ptrw();

		int face = 0;

		{
			int ngon_counter = 0;
			for (int i = 0; i < sides; i++) {
				float inci = float(i) / sides;
				float inci_n = float((i + 1)) / sides;
				if (i == sides - 1) {
					inci_n = 0;
				}

				float angi = inci * Math::TAU;
				float angi_n = inci_n * Math::TAU;

				Vector3 normali = Vector3(Math::cos(angi), 0, Math::sin(angi));
				Vector3 normali_n = Vector3(Math::cos(angi_n), 0, Math::sin(angi_n));

				for (int j = 0; j < ring_sides; j++) {
					float incj = float(j) / ring_sides;
					float incj_n = float((j + 1)) / ring_sides;
					if (j == ring_sides - 1) {
						incj_n = 0;
					}

					float angj = incj * Math::TAU;
					float angj_n = incj_n * Math::TAU;

					Vector2 normalj = Vector2(Math::cos(angj), Math::sin(angj)) * radius + Vector2(min_radius + radius, 0);
					Vector2 normalj_n = Vector2(Math::cos(angj_n), Math::sin(angj_n)) * radius + Vector2(min_radius + radius, 0);

					Vector3 face_points[4] = {
						Vector3(normali.x * normalj.x, normalj.y, normali.z * normalj.x),
						Vector3(normali.x * normalj_n.x, normalj_n.y, normali.z * normalj_n.x),
						Vector3(normali_n.x * normalj_n.x, normalj_n.y, normali_n.z * normalj_n.x),
						Vector3(normali_n.x * normalj.x, normalj.y, normali_n.z * normalj.x)
					};

					Vector2 u[4] = {
						Vector2(inci, incj),
						Vector2(inci, incj_n),
						Vector2(inci_n, incj_n),
						Vector2(inci_n, incj),
					};

					// face 1
					facesw[face * 3 + 0] = face_points[0];
					facesw[face * 3 + 1] = face_points[2];
					facesw[face * 3 + 2] = face_points[1];

					uvsw[face * 3 + 0] = u[0];
					uvsw[face * 3 + 1] = u[2];
					uvsw[face * 3 + 2] = u[1];

					smoothw[face] = smooth_faces;
					invertw[face] = invert_val;
					materialsw[face] = base_material;

					ngonw[face] = ngon_counter;

					face++;

					//face 2
					facesw[face * 3 + 0] = face_points[3];
					facesw[face * 3 + 1] = face_points[2];
					facesw[face * 3 + 2] = face_points[0];

					uvsw[face * 3 + 0] = u[3];
					uvsw[face * 3 + 1] = u[2];
					uvsw[face * 3 + 2] = u[0];

					smoothw[face] = smooth_faces;
					invertw[face] = invert_val;
					materialsw[face] = base_material;

					ngonw[face] = ngon_counter;

					face++;
					ngon_counter++;
				}
			}
		}

		if (face != face_count) {
			ERR_PRINT("Face mismatch bug! fix code");
		}
	}

	new_brush->build_from_faces(faces, uvs, smooth, materials, invert);
	new_brush->add_ngons(ngons);
	csg_push_warning(NO_WARNING);

	return new_brush;
}

void CSGTorus3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_inner_radius", "radius"), &CSGTorus3D::set_inner_radius);
	ClassDB::bind_method(D_METHOD("get_inner_radius"), &CSGTorus3D::get_inner_radius);

	ClassDB::bind_method(D_METHOD("set_outer_radius", "radius"), &CSGTorus3D::set_outer_radius);
	ClassDB::bind_method(D_METHOD("get_outer_radius"), &CSGTorus3D::get_outer_radius);

	ClassDB::bind_method(D_METHOD("set_sides", "sides"), &CSGTorus3D::set_sides);
	ClassDB::bind_method(D_METHOD("get_sides"), &CSGTorus3D::get_sides);

	ClassDB::bind_method(D_METHOD("set_ring_sides", "sides"), &CSGTorus3D::set_ring_sides);
	ClassDB::bind_method(D_METHOD("get_ring_sides"), &CSGTorus3D::get_ring_sides);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGTorus3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGTorus3D::get_material);

	ClassDB::bind_method(D_METHOD("set_smooth_faces", "smooth_faces"), &CSGTorus3D::set_smooth_faces);
	ClassDB::bind_method(D_METHOD("get_smooth_faces"), &CSGTorus3D::get_smooth_faces);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inner_radius", PROPERTY_HINT_RANGE, "0.001,1000.0,0.001,or_greater,exp,suffix:m"), "set_inner_radius", "get_inner_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "outer_radius", PROPERTY_HINT_RANGE, "0.001,1000.0,0.001,or_greater,exp,suffix:m"), "set_outer_radius", "get_outer_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_sides", "get_sides");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ring_sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_ring_sides", "get_ring_sides");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "BaseMaterial3D,ShaderMaterial"), "set_material", "get_material");
}

void CSGTorus3D::set_inner_radius(const float p_inner_radius) {
	inner_radius = p_inner_radius;
	if (is_painted()) {
		resize_brush_rework();
	}
	_make_painted();
	update_gizmos();
}

float CSGTorus3D::get_inner_radius() const {
	return inner_radius;
}

void CSGTorus3D::set_outer_radius(const float p_outer_radius) {
	outer_radius = p_outer_radius;
	if (is_painted()) {
		resize_brush_rework();
	}
	_make_painted();
	update_gizmos();
}

float CSGTorus3D::get_outer_radius() const {
	return outer_radius;
}

void CSGTorus3D::set_sides(const int p_sides) {
	ERR_FAIL_COND(p_sides < 3);
	sides = p_sides;
	rebuild_brush();
	update_gizmos();
}

int CSGTorus3D::get_sides() const {
	return sides;
}

void CSGTorus3D::set_ring_sides(const int p_ring_sides) {
	ERR_FAIL_COND(p_ring_sides < 3);
	ring_sides = p_ring_sides;
	rebuild_brush();
	update_gizmos();
}

int CSGTorus3D::get_ring_sides() const {
	return ring_sides;
}

void CSGTorus3D::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	if (is_inside_tree()) {
		set_csg_flat(smooth_faces);
	}
	_make_painted();
}

bool CSGTorus3D::get_smooth_faces() const {
	return smooth_faces;
}

void CSGTorus3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	if (is_inside_tree()) {
		set_face_material(get_all_csg_faces(), material);
	}
}

Ref<Material> CSGTorus3D::get_material() const {
	return material;
}

CSGTorus3D::CSGTorus3D() {
	// defaults
	inner_radius = 0.5;
	outer_radius = 1.0;
	sides = 8;
	ring_sides = 6;
	smooth_faces = true;
}

///////////////

CSGBrush *CSGPolygon3D::_build_brush() {
	CSGBrush *new_brush = memnew(CSGBrush);

	if (polygon.size() < 3) {
		return new_brush;
	}

	// Triangulate polygon shape.
	Vector<Point2> shape_polygon = polygon;
	if (Triangulate::get_area(shape_polygon) > 0) {
		shape_polygon.reverse();
	}
	int shape_sides = shape_polygon.size();
	Vector<int> shape_faces = Geometry2D::triangulate_polygon(shape_polygon);
	ERR_FAIL_COND_V_MSG(shape_faces.size() < 3, new_brush, "Failed to triangulate CSGPolygon. Make sure the polygon doesn't have any intersecting edges.");

	// Get polygon enclosing Rect2.
	Rect2 shape_rect(shape_polygon[0], Vector2());
	for (int i = 1; i < shape_sides; i++) {
		shape_rect.expand_to(shape_polygon[i]);
	}

	// If MODE_PATH, check if curve has changed.
	Ref<Curve3D> curve;
	if (mode == MODE_PATH) {
		Path3D *current_path = Object::cast_to<Path3D>(get_node_or_null(path_node));
		if (path != current_path) {
			if (path) {
				path->disconnect(SceneStringName(tree_exited), callable_mp(this, &CSGPolygon3D::_path_exited));
				path->disconnect("curve_changed", callable_mp(this, &CSGPolygon3D::_path_changed));
				path->set_update_callback(Callable());
			}
			path = current_path;
			if (path) {
				path->connect(SceneStringName(tree_exited), callable_mp(this, &CSGPolygon3D::_path_exited));
				path->connect("curve_changed", callable_mp(this, &CSGPolygon3D::_path_changed));
				path->set_update_callback(callable_mp(this, &CSGPolygon3D::_path_changed));
			}
		}

		if (!path) {
			return new_brush;
		}

		curve = path->get_curve();
		if (curve.is_null() || curve->get_point_count() < 2) {
			return new_brush;
		}
	}

	// Calculate the number extrusions, ends and faces.
	int extrusions = 0;
	int extrusion_face_count = shape_sides * 2;
	int end_count = 0;
	int shape_face_count = shape_faces.size() / 3;
	real_t curve_length = 1.0;
	switch (mode) {
		case MODE_DEPTH:
			extrusions = 1;
			end_count = 2;
			break;
		case MODE_SPIN:
			extrusions = spin_sides;
			if (spin_degrees < 360) {
				end_count = 2;
			}
			break;
		case MODE_PATH: {
			curve_length = curve->get_baked_length();
			if (path_interval_type == PATH_INTERVAL_DISTANCE) {
				extrusions = MAX(1, Math::ceil(curve_length / path_interval)) + 1;
			} else {
				extrusions = Math::ceil(1.0 * curve->get_point_count() / path_interval);
			}
			if (!path_joined) {
				end_count = 2;
				extrusions -= 1;
			}
		} break;
	}
	int face_count = extrusions * extrusion_face_count + end_count * shape_face_count;

	// Initialize variables used to create the mesh.
	Ref<Material> base_material = get_material();

	Vector<Vector3> faces;
	Vector<Vector2> uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	Vector<int> ngons;
	ngons.resize(face_count);

	faces.resize(face_count * 3);
	uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);
	int faces_removed = 0;

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *uvsw = uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		int *ngonw = ngons.ptrw();

		int ngon_counter = 0;

		int face = 0;
		Transform3D base_xform;
		Transform3D current_xform;
		Transform3D previous_xform;
		Transform3D previous_previous_xform;
		double u_step = 1.0 / extrusions;
		if (path_u_distance > 0.0) {
			u_step *= curve_length / path_u_distance;
		}
		double v_step = 1.0 / shape_sides;
		double spin_step = Math::deg_to_rad(spin_degrees / spin_sides);
		double extrusion_step = 1.0 / extrusions;
		if (mode == MODE_PATH) {
			if (path_joined) {
				extrusion_step = 1.0 / (extrusions - 1);
			}
			extrusion_step *= curve_length;
		}

		if (mode == MODE_PATH) {
			if (!path_local && path->is_inside_tree()) {
				base_xform = path->get_global_transform();
			}

			Vector3 current_point;
			Vector3 current_up = Vector3(0, 1, 0);
			Vector3 direction;

			switch (path_rotation) {
				case PATH_ROTATION_POLYGON:
					current_point = curve->sample_baked(0);
					direction = Vector3(0, 0, -1);
					break;
				case PATH_ROTATION_PATH:
				case PATH_ROTATION_PATH_FOLLOW:
					if (!path_rotation_accurate) {
						current_point = curve->sample_baked(0);
						Vector3 next_point = curve->sample_baked(extrusion_step);
						direction = next_point - current_point;

						if (path_joined) {
							Vector3 last_point = curve->sample_baked(curve->get_baked_length());
							direction = next_point - last_point;
						}
					} else {
						Transform3D current_sample_xform = curve->sample_baked_with_rotation(0);
						current_point = current_sample_xform.get_origin();
						direction = current_sample_xform.get_basis().xform(Vector3(0, 0, -1));
					}

					if (path_rotation == PATH_ROTATION_PATH_FOLLOW) {
						current_up = curve->sample_baked_up_vector(0, true);
					}
					break;
			}

			Transform3D facing = Transform3D().looking_at(direction, current_up);
			current_xform = base_xform.translated_local(current_point) * facing;
		}

		// Create the mesh.
		if (end_count > 0) {
			// Add front end face.
			for (int face_idx = 0; face_idx < shape_face_count; face_idx++) {
				for (int face_vertex_idx = 0; face_vertex_idx < 3; face_vertex_idx++) {
					// We need to reverse the rotation of the shape face vertices.
					int index = shape_faces[face_idx * 3 + 2 - face_vertex_idx];
					Point2 p = shape_polygon[index];
					Point2 uv = (p - shape_rect.position) / shape_rect.size;

					// Use the left side of the bottom half of the y-inverted texture.
					uv.x = uv.x / 2;
					uv.y = 1 - (uv.y / 2);

					facesw[face * 3 + face_vertex_idx] = current_xform.xform(Vector3(p.x, p.y, 0));
					uvsw[face * 3 + face_vertex_idx] = uv;
				}

				smoothw[face] = false;
				materialsw[face] = base_material;
				invertw[face] = flip_faces;
				ngonw[face] = ngon_counter;
				face++;
			}
			ngon_counter++;
		}

		real_t angle_simplify_dot = Math::cos(Math::deg_to_rad(path_simplify_angle));
		Vector3 previous_simplify_dir = Vector3(0, 0, 0);
		int faces_combined = 0;

		// Add extrusion faces.
		for (int x0 = 0; x0 < extrusions; x0++) {
			previous_previous_xform = previous_xform;
			previous_xform = current_xform;

			switch (mode) {
				case MODE_DEPTH: {
					current_xform.translate_local(Vector3(0, 0, -depth));
				} break;
				case MODE_SPIN: {
					if (end_count == 0 && x0 == extrusions - 1) {
						current_xform = base_xform;
					} else {
						current_xform.rotate(Vector3(0, 1, 0), spin_step);
					}
				} break;
				case MODE_PATH: {
					double previous_offset = x0 * extrusion_step;
					double current_offset = (x0 + 1) * extrusion_step;
					if (path_joined && x0 == extrusions - 1) {
						current_offset = 0;
					}

					Vector3 previous_point = curve->sample_baked(previous_offset);
					Transform3D current_sample_xform = curve->sample_baked_with_rotation(current_offset);
					Vector3 current_point = current_sample_xform.get_origin();
					Vector3 current_up = Vector3(0, 1, 0);
					Vector3 current_extrusion_dir = (current_point - previous_point).normalized();
					Vector3 direction;

					// If the angles are similar, remove the previous face and replace it with this one.
					if (path_simplify_angle > 0.0 && x0 > 0 && previous_simplify_dir.dot(current_extrusion_dir) > angle_simplify_dot) {
						faces_combined += 1;
						previous_xform = previous_previous_xform;
						face -= extrusion_face_count;
						faces_removed += extrusion_face_count;
					} else {
						faces_combined = 0;
						previous_simplify_dir = current_extrusion_dir;
					}

					switch (path_rotation) {
						case PATH_ROTATION_POLYGON:
							direction = Vector3(0, 0, -1);
							break;
						case PATH_ROTATION_PATH:
						case PATH_ROTATION_PATH_FOLLOW:
							if (!path_rotation_accurate) {
								double next_offset = (x0 + 2) * extrusion_step;
								if (x0 == extrusions - 1) {
									next_offset = path_joined ? extrusion_step : current_offset;
								}
								Vector3 next_point = curve->sample_baked(next_offset);
								direction = next_point - previous_point;
							} else {
								direction = current_sample_xform.get_basis().xform(Vector3(0, 0, -1));
							}

							if (path_rotation == PATH_ROTATION_PATH_FOLLOW) {
								current_up = curve->sample_baked_up_vector(current_offset, true);
							}
							break;
					}

					Transform3D facing = Transform3D().looking_at(direction, current_up);
					current_xform = base_xform.translated_local(current_point) * facing;
				} break;
			}

			double u0 = (x0 - faces_combined) * u_step;
			double u1 = ((x0 + 1) * u_step);
			if (mode == MODE_PATH && !path_continuous_u) {
				u0 = 0.0;
				u1 = 1.0;
			}

			for (int y0 = 0; y0 < shape_sides; y0++) {
				int y1 = (y0 + 1) % shape_sides;
				// Use the top half of the texture.
				double v0 = (y0 * v_step) / 2;
				double v1 = ((y0 + 1) * v_step) / 2;

				Vector3 v[4] = {
					previous_xform.xform(Vector3(shape_polygon[y0].x, shape_polygon[y0].y, 0)),
					current_xform.xform(Vector3(shape_polygon[y0].x, shape_polygon[y0].y, 0)),
					current_xform.xform(Vector3(shape_polygon[y1].x, shape_polygon[y1].y, 0)),
					previous_xform.xform(Vector3(shape_polygon[y1].x, shape_polygon[y1].y, 0)),
				};

				Vector2 u[4] = {
					Vector2(u0, v0),
					Vector2(u1, v0),
					Vector2(u1, v1),
					Vector2(u0, v1),
				};

				// Face 1
				facesw[face * 3 + 0] = v[0];
				facesw[face * 3 + 1] = v[1];
				facesw[face * 3 + 2] = v[2];

				uvsw[face * 3 + 0] = u[0];
				uvsw[face * 3 + 1] = u[1];
				uvsw[face * 3 + 2] = u[2];

				smoothw[face] = smooth_faces;
				invertw[face] = flip_faces;
				materialsw[face] = base_material;
				ngonw[face] = ngon_counter;

				face++;

				// Face 2
				facesw[face * 3 + 0] = v[2];
				facesw[face * 3 + 1] = v[3];
				facesw[face * 3 + 2] = v[0];

				uvsw[face * 3 + 0] = u[2];
				uvsw[face * 3 + 1] = u[3];
				uvsw[face * 3 + 2] = u[0];

				smoothw[face] = smooth_faces;
				invertw[face] = flip_faces;
				materialsw[face] = base_material;
				ngonw[face] = ngon_counter;
				ngon_counter++;

				face++;
			}
		}

		if (end_count > 1) {
			// Add back end face.
			for (int face_idx = 0; face_idx < shape_face_count; face_idx++) {
				for (int face_vertex_idx = 0; face_vertex_idx < 3; face_vertex_idx++) {
					int index = shape_faces[face_idx * 3 + face_vertex_idx];
					Point2 p = shape_polygon[index];
					Point2 uv = (p - shape_rect.position) / shape_rect.size;

					// Use the x-inverted ride side of the bottom half of the y-inverted texture.
					uv.x = 1 - uv.x / 2;
					uv.y = 1 - (uv.y / 2);

					facesw[face * 3 + face_vertex_idx] = current_xform.xform(Vector3(p.x, p.y, 0));
					uvsw[face * 3 + face_vertex_idx] = uv;
				}

				smoothw[face] = false;
				materialsw[face] = base_material;
				invertw[face] = flip_faces;
				ngonw[face] = ngon_counter;
				face++;
			}
		}

		face_count -= faces_removed;
		ERR_FAIL_COND_V_MSG(face != face_count, new_brush, "Bug: Failed to create the CSGPolygon mesh correctly.");
	}

	if (faces_removed > 0) {
		faces.resize(face_count * 3);
		uvs.resize(face_count * 3);
		smooth.resize(face_count);
		materials.resize(face_count);
		invert.resize(face_count);
		ngons.resize(face_count); // What does this do?
	}

	new_brush->build_from_faces(faces, uvs, smooth, materials, invert);
	new_brush->add_ngons(ngons);
	csg_push_warning(NO_WARNING);

	return new_brush;
}

void CSGPolygon3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_EXIT_TREE) {
		if (path) {
			path->disconnect(SceneStringName(tree_exited), callable_mp(this, &CSGPolygon3D::_path_exited));
			path->disconnect("curve_changed", callable_mp(this, &CSGPolygon3D::_path_changed));
			path = nullptr;
		}
	}
}

void CSGPolygon3D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name.begins_with("spin") && mode != MODE_SPIN) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
	if (p_property.name.begins_with("path") && mode != MODE_PATH) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
	if (p_property.name == "depth" && mode != MODE_DEPTH) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void CSGPolygon3D::_path_changed() {
	_make_painted();
	update_gizmos();
}

void CSGPolygon3D::_path_exited() {
	path = nullptr;
}

void CSGPolygon3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_polygon", "polygon"), &CSGPolygon3D::set_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon"), &CSGPolygon3D::get_polygon);

	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &CSGPolygon3D::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &CSGPolygon3D::get_mode);

	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &CSGPolygon3D::set_depth);
	ClassDB::bind_method(D_METHOD("get_depth"), &CSGPolygon3D::get_depth);

	ClassDB::bind_method(D_METHOD("set_spin_degrees", "degrees"), &CSGPolygon3D::set_spin_degrees);
	ClassDB::bind_method(D_METHOD("get_spin_degrees"), &CSGPolygon3D::get_spin_degrees);

	ClassDB::bind_method(D_METHOD("set_spin_sides", "spin_sides"), &CSGPolygon3D::set_spin_sides);
	ClassDB::bind_method(D_METHOD("get_spin_sides"), &CSGPolygon3D::get_spin_sides);

	ClassDB::bind_method(D_METHOD("set_path_node", "path"), &CSGPolygon3D::set_path_node);
	ClassDB::bind_method(D_METHOD("get_path_node"), &CSGPolygon3D::get_path_node);

	ClassDB::bind_method(D_METHOD("set_path_interval_type", "interval_type"), &CSGPolygon3D::set_path_interval_type);
	ClassDB::bind_method(D_METHOD("get_path_interval_type"), &CSGPolygon3D::get_path_interval_type);

	ClassDB::bind_method(D_METHOD("set_path_interval", "interval"), &CSGPolygon3D::set_path_interval);
	ClassDB::bind_method(D_METHOD("get_path_interval"), &CSGPolygon3D::get_path_interval);

	ClassDB::bind_method(D_METHOD("set_path_simplify_angle", "degrees"), &CSGPolygon3D::set_path_simplify_angle);
	ClassDB::bind_method(D_METHOD("get_path_simplify_angle"), &CSGPolygon3D::get_path_simplify_angle);

	ClassDB::bind_method(D_METHOD("set_path_rotation", "path_rotation"), &CSGPolygon3D::set_path_rotation);
	ClassDB::bind_method(D_METHOD("get_path_rotation"), &CSGPolygon3D::get_path_rotation);

	ClassDB::bind_method(D_METHOD("set_path_rotation_accurate", "enable"), &CSGPolygon3D::set_path_rotation_accurate);
	ClassDB::bind_method(D_METHOD("get_path_rotation_accurate"), &CSGPolygon3D::get_path_rotation_accurate);

	ClassDB::bind_method(D_METHOD("set_path_local", "enable"), &CSGPolygon3D::set_path_local);
	ClassDB::bind_method(D_METHOD("is_path_local"), &CSGPolygon3D::is_path_local);

	ClassDB::bind_method(D_METHOD("set_path_continuous_u", "enable"), &CSGPolygon3D::set_path_continuous_u);
	ClassDB::bind_method(D_METHOD("is_path_continuous_u"), &CSGPolygon3D::is_path_continuous_u);

	ClassDB::bind_method(D_METHOD("set_path_u_distance", "distance"), &CSGPolygon3D::set_path_u_distance);
	ClassDB::bind_method(D_METHOD("get_path_u_distance"), &CSGPolygon3D::get_path_u_distance);

	ClassDB::bind_method(D_METHOD("set_path_joined", "enable"), &CSGPolygon3D::set_path_joined);
	ClassDB::bind_method(D_METHOD("is_path_joined"), &CSGPolygon3D::is_path_joined);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGPolygon3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGPolygon3D::get_material);

	ClassDB::bind_method(D_METHOD("set_smooth_faces", "smooth_faces"), &CSGPolygon3D::set_smooth_faces);
	ClassDB::bind_method(D_METHOD("get_smooth_faces"), &CSGPolygon3D::get_smooth_faces);

	ClassDB::bind_method(D_METHOD("_is_editable_3d_polygon"), &CSGPolygon3D::_is_editable_3d_polygon);
	ClassDB::bind_method(D_METHOD("_has_editable_3d_polygon_no_depth"), &CSGPolygon3D::_has_editable_3d_polygon_no_depth);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "polygon"), "set_polygon", "get_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Depth,Spin,Path"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "depth", PROPERTY_HINT_RANGE, "0.01,100.0,0.01,or_greater,exp,suffix:m"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "spin_degrees", PROPERTY_HINT_RANGE, "1,360,0.1"), "set_spin_degrees", "get_spin_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spin_sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_spin_sides", "get_spin_sides");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "path_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Path3D"), "set_path_node", "get_path_node");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_interval_type", PROPERTY_HINT_ENUM, "Distance,Subdivide"), "set_path_interval_type", "get_path_interval_type");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_interval", PROPERTY_HINT_RANGE, "0.01,1.0,0.01,exp,or_greater"), "set_path_interval", "get_path_interval");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_simplify_angle", PROPERTY_HINT_RANGE, "0.0,180.0,0.1"), "set_path_simplify_angle", "get_path_simplify_angle");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_rotation", PROPERTY_HINT_ENUM, "Polygon,Path,PathFollow"), "set_path_rotation", "get_path_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_rotation_accurate"), "set_path_rotation_accurate", "get_path_rotation_accurate");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_local"), "set_path_local", "is_path_local");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_continuous_u"), "set_path_continuous_u", "is_path_continuous_u");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_u_distance", PROPERTY_HINT_RANGE, "0.0,10.0,0.01,or_greater,suffix:m"), "set_path_u_distance", "get_path_u_distance");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_joined"), "set_path_joined", "is_path_joined");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "BaseMaterial3D,ShaderMaterial"), "set_material", "get_material");

	BIND_ENUM_CONSTANT(MODE_DEPTH);
	BIND_ENUM_CONSTANT(MODE_SPIN);
	BIND_ENUM_CONSTANT(MODE_PATH);

	BIND_ENUM_CONSTANT(PATH_ROTATION_POLYGON);
	BIND_ENUM_CONSTANT(PATH_ROTATION_PATH);
	BIND_ENUM_CONSTANT(PATH_ROTATION_PATH_FOLLOW);

	BIND_ENUM_CONSTANT(PATH_INTERVAL_DISTANCE);
	BIND_ENUM_CONSTANT(PATH_INTERVAL_SUBDIVIDE);
}

void CSGPolygon3D::set_polygon(const Vector<Vector2> &p_polygon) {
	polygon = p_polygon;
	rebuild_brush();
	update_gizmos();
}

Vector<Vector2> CSGPolygon3D::get_polygon() const {
	return polygon;
}

void CSGPolygon3D::set_mode(Mode p_mode) {
	mode = p_mode;
	rebuild_brush();
	update_gizmos();
	notify_property_list_changed();
}

CSGPolygon3D::Mode CSGPolygon3D::get_mode() const {
	return mode;
}

void CSGPolygon3D::set_depth(const float p_depth) {
	ERR_FAIL_COND(p_depth < 0.001);
	depth = p_depth;
	if (is_painted()) {
		resize_brush_rework();
	}
	_make_painted();
	update_gizmos();
}

float CSGPolygon3D::get_depth() const {
	return depth;
}

void CSGPolygon3D::set_path_continuous_u(bool p_enable) {
	path_continuous_u = p_enable;
	rebuild_brush();
}

bool CSGPolygon3D::is_path_continuous_u() const {
	return path_continuous_u;
}

void CSGPolygon3D::set_path_u_distance(real_t p_path_u_distance) {
	path_u_distance = p_path_u_distance;
	rebuild_brush();
	update_gizmos();
}

real_t CSGPolygon3D::get_path_u_distance() const {
	return path_u_distance;
}

void CSGPolygon3D::set_spin_degrees(const float p_spin_degrees) {
	ERR_FAIL_COND(p_spin_degrees < 0.01 || p_spin_degrees > 360);
	spin_degrees = p_spin_degrees;
	rebuild_brush();
	update_gizmos();
}

float CSGPolygon3D::get_spin_degrees() const {
	return spin_degrees;
}

void CSGPolygon3D::set_spin_sides(int p_spin_sides) {
	ERR_FAIL_COND(p_spin_sides < 3);
	spin_sides = p_spin_sides;
	rebuild_brush();
	update_gizmos();
}

int CSGPolygon3D::get_spin_sides() const {
	return spin_sides;
}

void CSGPolygon3D::set_path_node(const NodePath &p_path) {
	path_node = p_path;
	rebuild_brush();
	update_gizmos();
}

NodePath CSGPolygon3D::get_path_node() const {
	return path_node;
}

void CSGPolygon3D::set_path_interval_type(PathIntervalType p_interval_type) {
	path_interval_type = p_interval_type;
	rebuild_brush();
	update_gizmos();
}

CSGPolygon3D::PathIntervalType CSGPolygon3D::get_path_interval_type() const {
	return path_interval_type;
}

void CSGPolygon3D::set_path_interval(float p_interval) {
	path_interval = p_interval;
	rebuild_brush();
	update_gizmos();
}

float CSGPolygon3D::get_path_interval() const {
	return path_interval;
}

void CSGPolygon3D::set_path_simplify_angle(float p_angle) {
	path_simplify_angle = p_angle;
	rebuild_brush();
	update_gizmos();
}

float CSGPolygon3D::get_path_simplify_angle() const {
	return path_simplify_angle;
}

void CSGPolygon3D::set_path_rotation(PathRotation p_rotation) {
	path_rotation = p_rotation;
	rebuild_brush();
	update_gizmos();
}

CSGPolygon3D::PathRotation CSGPolygon3D::get_path_rotation() const {
	return path_rotation;
}

void CSGPolygon3D::set_path_rotation_accurate(bool p_enabled) {
	path_rotation_accurate = p_enabled;
	rebuild_brush();
	update_gizmos();
}

bool CSGPolygon3D::get_path_rotation_accurate() const {
	return path_rotation_accurate;
}

void CSGPolygon3D::set_path_local(bool p_enable) {
	path_local = p_enable;
	if (is_painted()) {
		resize_brush_rework();
	}
	_make_painted();
	update_gizmos();
}

bool CSGPolygon3D::is_path_local() const {
	return path_local;
}

void CSGPolygon3D::set_path_joined(bool p_enable) {
	path_joined = p_enable;
	rebuild_brush();
	update_gizmos();
}

bool CSGPolygon3D::is_path_joined() const {
	return path_joined;
}

void CSGPolygon3D::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	_make_painted();
}

bool CSGPolygon3D::get_smooth_faces() const {
	return smooth_faces;
}

void CSGPolygon3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	if (is_inside_tree()) {
		set_face_material(get_all_csg_faces(), material);
	}
}

Ref<Material> CSGPolygon3D::get_material() const {
	return material;
}

bool CSGPolygon3D::_is_editable_3d_polygon() const {
	return true;
}

bool CSGPolygon3D::_has_editable_3d_polygon_no_depth() const {
	return true;
}

CSGPolygon3D::CSGPolygon3D() {
	// defaults
	mode = MODE_DEPTH;
	polygon.push_back(Vector2(0, 0));
	polygon.push_back(Vector2(0, 1));
	polygon.push_back(Vector2(1, 1));
	polygon.push_back(Vector2(1, 0));
	depth = 1.0;
	spin_degrees = 360;
	spin_sides = 8;
	smooth_faces = false;
	path_interval_type = PATH_INTERVAL_DISTANCE;
	path_interval = 1.0;
	path_simplify_angle = 0.0;
	path_rotation = PATH_ROTATION_PATH_FOLLOW;
	path_rotation_accurate = false;
	path_local = false;
	path_continuous_u = true;
	path_u_distance = 1.0;
	path_joined = false;
	path = nullptr;
}
