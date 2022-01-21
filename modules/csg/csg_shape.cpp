/*************************************************************************/
/*  csg_shape.cpp                                                        */
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

#include "csg_shape.h"

void CSGShape::set_use_collision(bool p_enable) {
	if (use_collision == p_enable) {
		return;
	}

	use_collision = p_enable;

	if (!is_inside_tree() || !is_root_shape()) {
		return;
	}

	if (use_collision) {
		root_collision_shape.instance();
		root_collision_instance = RID_PRIME(PhysicsServer::get_singleton()->body_create(PhysicsServer::BODY_MODE_STATIC));
		PhysicsServer::get_singleton()->body_set_state(root_collision_instance, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());
		PhysicsServer::get_singleton()->body_add_shape(root_collision_instance, root_collision_shape->get_rid());
		PhysicsServer::get_singleton()->body_set_space(root_collision_instance, get_world()->get_space());
		PhysicsServer::get_singleton()->body_attach_object_instance_id(root_collision_instance, get_instance_id());
		set_collision_layer(collision_layer);
		set_collision_mask(collision_mask);
		_make_dirty(); //force update
	} else {
		PhysicsServer::get_singleton()->free(root_collision_instance);
		root_collision_instance = RID();
		root_collision_shape.unref();
	}
	_change_notify();
}

bool CSGShape::is_using_collision() const {
	return use_collision;
}

void CSGShape::set_collision_layer(uint32_t p_layer) {
	collision_layer = p_layer;
	if (root_collision_instance.is_valid()) {
		PhysicsServer::get_singleton()->body_set_collision_layer(root_collision_instance, p_layer);
	}
}

uint32_t CSGShape::get_collision_layer() const {
	return collision_layer;
}

void CSGShape::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
	if (root_collision_instance.is_valid()) {
		PhysicsServer::get_singleton()->body_set_collision_mask(root_collision_instance, p_mask);
	}
}

uint32_t CSGShape::get_collision_mask() const {
	return collision_mask;
}

void CSGShape::set_collision_mask_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision mask bit must be between 0 and 31 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << p_bit;
	} else {
		mask &= ~(1 << p_bit);
	}
	set_collision_mask(mask);
}

bool CSGShape::get_collision_mask_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision mask bit must be between 0 and 31 inclusive.");
	return get_collision_mask() & (1 << p_bit);
}

void CSGShape::set_collision_layer_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision layer bit must be between 0 and 31 inclusive.");
	uint32_t layer = get_collision_layer();
	if (p_value) {
		layer |= 1 << p_bit;
	} else {
		layer &= ~(1 << p_bit);
	}
	set_collision_layer(layer);
}

bool CSGShape::get_collision_layer_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision layer bit must be between 0 and 31 inclusive.");
	return get_collision_layer() & (1 << p_bit);
}

bool CSGShape::is_root_shape() const {
	return !parent;
}

void CSGShape::set_snap(float p_snap) {
	snap = p_snap;
}

float CSGShape::get_snap() const {
	return snap;
}

void CSGShape::_make_dirty() {
	if (!is_inside_tree()) {
		return;
	}

	if (parent) {
		parent->_make_dirty();
	} else if (!dirty) {
		call_deferred("_update_shape");
	}

	dirty = true;
}

CSGBrush *CSGShape::_get_brush() {
	if (dirty) {
		if (brush) {
			memdelete(brush);
		}
		brush = nullptr;

		CSGBrush *n = _build_brush();

		for (int i = 0; i < get_child_count(); i++) {
			CSGShape *child = Object::cast_to<CSGShape>(get_child(i));
			if (!child) {
				continue;
			}
			if (!child->is_visible_in_tree()) {
				continue;
			}

			CSGBrush *n2 = child->_get_brush();
			if (!n2) {
				continue;
			}
			if (!n) {
				n = memnew(CSGBrush);

				n->copy_from(*n2, child->get_transform());

			} else {
				CSGBrush *nn = memnew(CSGBrush);
				CSGBrush *nn2 = memnew(CSGBrush);
				nn2->copy_from(*n2, child->get_transform());

				CSGBrushOperation bop;

				switch (child->get_operation()) {
					case CSGShape::OPERATION_UNION:
						bop.merge_brushes(CSGBrushOperation::OPERATION_UNION, *n, *nn2, *nn, snap);
						break;
					case CSGShape::OPERATION_INTERSECTION:
						bop.merge_brushes(CSGBrushOperation::OPERATION_INTERSECTION, *n, *nn2, *nn, snap);
						break;
					case CSGShape::OPERATION_SUBTRACTION:
						bop.merge_brushes(CSGBrushOperation::OPERATION_SUBTRACTION, *n, *nn2, *nn, snap);
						break;
				}
				memdelete(n);
				memdelete(nn2);
				n = nn;
			}
		}

		if (n) {
			AABB aabb;
			for (int i = 0; i < n->faces.size(); i++) {
				for (int j = 0; j < 3; j++) {
					if (i == 0 && j == 0) {
						aabb.position = n->faces[i].vertices[j];
					} else {
						aabb.expand_to(n->faces[i].vertices[j]);
					}
				}
			}
			node_aabb = aabb;
		} else {
			node_aabb = AABB();
		}

		brush = n;

		dirty = false;
	}

	return brush;
}

int CSGShape::mikktGetNumFaces(const SMikkTSpaceContext *pContext) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	return surface.vertices.size() / 3;
}

int CSGShape::mikktGetNumVerticesOfFace(const SMikkTSpaceContext *pContext, const int iFace) {
	// always 3
	return 3;
}

void CSGShape::mikktGetPosition(const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace, const int iVert) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	Vector3 v = surface.verticesw[iFace * 3 + iVert];
	fvPosOut[0] = v.x;
	fvPosOut[1] = v.y;
	fvPosOut[2] = v.z;
}

void CSGShape::mikktGetNormal(const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace, const int iVert) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	Vector3 n = surface.normalsw[iFace * 3 + iVert];
	fvNormOut[0] = n.x;
	fvNormOut[1] = n.y;
	fvNormOut[2] = n.z;
}

void CSGShape::mikktGetTexCoord(const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace, const int iVert) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	Vector2 t = surface.uvsw[iFace * 3 + iVert];
	fvTexcOut[0] = t.x;
	fvTexcOut[1] = t.y;
}

void CSGShape::mikktSetTSpaceDefault(const SMikkTSpaceContext *pContext, const float fvTangent[], const float fvBiTangent[], const float fMagS, const float fMagT,
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

void CSGShape::_update_shape() {
	if (parent || !is_inside_tree()) {
		return;
	}

	set_base(RID());
	root_mesh.unref(); //byebye root mesh

	CSGBrush *n = _get_brush();
	ERR_FAIL_COND_MSG(!n, "Cannot get CSGBrush.");

	OAHashMap<Vector3, Vector3> vec_map;

	Vector<int> face_count;
	face_count.resize(n->materials.size() + 1);
	for (int i = 0; i < face_count.size(); i++) {
		face_count.write[i] = 0;
	}

	for (int i = 0; i < n->faces.size(); i++) {
		int mat = n->faces[i].material;
		ERR_CONTINUE(mat < -1 || mat >= face_count.size());
		int idx = mat == -1 ? face_count.size() - 1 : mat;

		Plane p(n->faces[i].vertices[0], n->faces[i].vertices[1], n->faces[i].vertices[2]);

		for (int j = 0; j < 3; j++) {
			Vector3 v = n->faces[i].vertices[j];
			Vector3 add;
			if (vec_map.lookup(v, add)) {
				add += p.normal;
			} else {
				add = p.normal;
			}
			vec_map.set(v, add);
		}

		face_count.write[idx]++;
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

		surfaces.write[i].verticesw = surfaces.write[i].vertices.write();
		surfaces.write[i].normalsw = surfaces.write[i].normals.write();
		surfaces.write[i].uvsw = surfaces.write[i].uvs.write();
		if (calculate_tangents) {
			surfaces.write[i].tansw = surfaces.write[i].tans.write();
		}
	}

	// Update collision faces.
	if (root_collision_shape.is_valid()) {
		PoolVector<Vector3> physics_faces;
		physics_faces.resize(n->faces.size() * 3);
		PoolVector<Vector3>::Write physicsw = physics_faces.write();

		for (int i = 0; i < n->faces.size(); i++) {
			int order[3] = { 0, 1, 2 };

			if (n->faces[i].invert) {
				SWAP(order[1], order[2]);
			}

			physicsw[i * 3 + 0] = n->faces[i].vertices[order[0]];
			physicsw[i * 3 + 1] = n->faces[i].vertices[order[1]];
			physicsw[i * 3 + 2] = n->faces[i].vertices[order[2]];
		}

		root_collision_shape->set_faces(physics_faces);
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

			Plane p(n->faces[i].vertices[0], n->faces[i].vertices[1], n->faces[i].vertices[2]);

			for (int j = 0; j < 3; j++) {
				Vector3 v = n->faces[i].vertices[j];

				Vector3 normal = p.normal;

				if (n->faces[i].smooth && vec_map.lookup(v, normal)) {
					normal.normalize();
				}

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

	root_mesh.instance();
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

		// unset write access
		surfaces.write[i].verticesw.release();
		surfaces.write[i].normalsw.release();
		surfaces.write[i].uvsw.release();
		surfaces.write[i].tansw.release();

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
}
AABB CSGShape::get_aabb() const {
	return node_aabb;
}

PoolVector<Vector3> CSGShape::get_brush_faces() {
	ERR_FAIL_COND_V(!is_inside_tree(), PoolVector<Vector3>());
	CSGBrush *b = _get_brush();
	if (!b) {
		return PoolVector<Vector3>();
	}

	PoolVector<Vector3> faces;
	int fc = b->faces.size();
	faces.resize(fc * 3);
	{
		PoolVector<Vector3>::Write w = faces.write();
		for (int i = 0; i < fc; i++) {
			w[i * 3 + 0] = b->faces[i].vertices[0];
			w[i * 3 + 1] = b->faces[i].vertices[1];
			w[i * 3 + 2] = b->faces[i].vertices[2];
		}
	}

	return faces;
}

PoolVector<Face3> CSGShape::get_faces(uint32_t p_usage_flags) const {
	return PoolVector<Face3>();
}

void CSGShape::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		Node *parentn = get_parent();
		if (parentn) {
			parent = Object::cast_to<CSGShape>(parentn);
			if (parent) {
				set_base(RID());
				root_mesh.unref();
			}
		}

		if (use_collision && is_root_shape()) {
			root_collision_shape.instance();
			root_collision_instance = RID_PRIME(PhysicsServer::get_singleton()->body_create(PhysicsServer::BODY_MODE_STATIC));
			PhysicsServer::get_singleton()->body_set_state(root_collision_instance, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());
			PhysicsServer::get_singleton()->body_add_shape(root_collision_instance, root_collision_shape->get_rid());
			PhysicsServer::get_singleton()->body_set_space(root_collision_instance, get_world()->get_space());
			PhysicsServer::get_singleton()->body_attach_object_instance_id(root_collision_instance, get_instance_id());
			set_collision_layer(collision_layer);
			set_collision_mask(collision_mask);
		}

		_make_dirty();
	}

	if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {
		if (use_collision && is_root_shape() && root_collision_instance.is_valid()) {
			PhysicsServer::get_singleton()->body_set_state(root_collision_instance, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());
		}
	}

	if (p_what == NOTIFICATION_LOCAL_TRANSFORM_CHANGED) {
		if (parent) {
			parent->_make_dirty();
		}
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (parent) {
			parent->_make_dirty();
		}
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
		if (parent) {
			parent->_make_dirty();
		}
		parent = nullptr;

		if (use_collision && is_root_shape() && root_collision_instance.is_valid()) {
			PhysicsServer::get_singleton()->free(root_collision_instance);
			root_collision_instance = RID();
			root_collision_shape.unref();
		}
		_make_dirty();
	}
}

void CSGShape::set_operation(Operation p_operation) {
	operation = p_operation;
	_make_dirty();
	update_gizmo();
}

CSGShape::Operation CSGShape::get_operation() const {
	return operation;
}

void CSGShape::set_calculate_tangents(bool p_calculate_tangents) {
	calculate_tangents = p_calculate_tangents;
	_make_dirty();
}

bool CSGShape::is_calculating_tangents() const {
	return calculate_tangents;
}

void CSGShape::_validate_property(PropertyInfo &property) const {
	bool is_collision_prefixed = property.name.begins_with("collision_");
	if ((is_collision_prefixed || property.name.begins_with("use_collision")) && is_inside_tree() && !is_root_shape()) {
		//hide collision if not root
		property.usage = PROPERTY_USAGE_NOEDITOR;
	} else if (is_collision_prefixed && !bool(get("use_collision"))) {
		property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
	}
}

// Calling _make_dirty() normally calls a deferred _update_shape.
// This is problematic if we need to read the geometry immediately.
// This function provides a means to make sure the shape is updated
// immediately. It should only be used where necessary to prevent
// updating CSGs multiple times per frame. Use _make_dirty in preference.
void CSGShape::force_update_shape() {
	if (dirty) {
		_update_shape();
	}
}

Array CSGShape::get_meshes() const {
	if (root_mesh.is_valid()) {
		Array arr;
		arr.resize(2);
		arr[0] = Transform();
		arr[1] = root_mesh;
		return arr;
	}

	return Array();
}
void CSGShape::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_shape"), &CSGShape::_update_shape);
	ClassDB::bind_method(D_METHOD("is_root_shape"), &CSGShape::is_root_shape);

	ClassDB::bind_method(D_METHOD("set_operation", "operation"), &CSGShape::set_operation);
	ClassDB::bind_method(D_METHOD("get_operation"), &CSGShape::get_operation);

	ClassDB::bind_method(D_METHOD("set_snap", "snap"), &CSGShape::set_snap);
	ClassDB::bind_method(D_METHOD("get_snap"), &CSGShape::get_snap);

	ClassDB::bind_method(D_METHOD("set_use_collision", "operation"), &CSGShape::set_use_collision);
	ClassDB::bind_method(D_METHOD("is_using_collision"), &CSGShape::is_using_collision);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "layer"), &CSGShape::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &CSGShape::get_collision_layer);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &CSGShape::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &CSGShape::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_bit", "bit", "value"), &CSGShape::set_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("get_collision_mask_bit", "bit"), &CSGShape::get_collision_mask_bit);

	ClassDB::bind_method(D_METHOD("set_collision_layer_bit", "bit", "value"), &CSGShape::set_collision_layer_bit);
	ClassDB::bind_method(D_METHOD("get_collision_layer_bit", "bit"), &CSGShape::get_collision_layer_bit);

	ClassDB::bind_method(D_METHOD("set_calculate_tangents", "enabled"), &CSGShape::set_calculate_tangents);
	ClassDB::bind_method(D_METHOD("is_calculating_tangents"), &CSGShape::is_calculating_tangents);

	ClassDB::bind_method(D_METHOD("get_meshes"), &CSGShape::get_meshes);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operation", PROPERTY_HINT_ENUM, "Union,Intersection,Subtraction"), "set_operation", "get_operation");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "snap", PROPERTY_HINT_RANGE, "0.0001,1,0.001"), "set_snap", "get_snap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "calculate_tangents"), "set_calculate_tangents", "is_calculating_tangents");

	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_collision"), "set_use_collision", "is_using_collision");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");

	BIND_ENUM_CONSTANT(OPERATION_UNION);
	BIND_ENUM_CONSTANT(OPERATION_INTERSECTION);
	BIND_ENUM_CONSTANT(OPERATION_SUBTRACTION);
}

CSGShape::CSGShape() {
	operation = OPERATION_UNION;
	parent = nullptr;
	brush = nullptr;
	dirty = false;
	snap = 0.001;
	use_collision = false;
	collision_layer = 1;
	collision_mask = 1;
	calculate_tangents = true;
	set_notify_local_transform(true);
}

CSGShape::~CSGShape() {
	if (brush) {
		memdelete(brush);
		brush = nullptr;
	}
}
//////////////////////////////////

CSGBrush *CSGCombiner::_build_brush() {
	return memnew(CSGBrush); //does not build anything
}

CSGCombiner::CSGCombiner() {
}

/////////////////////

CSGBrush *CSGPrimitive::_create_brush_from_arrays(const PoolVector<Vector3> &p_vertices, const PoolVector<Vector2> &p_uv, const PoolVector<bool> &p_smooth, const PoolVector<Ref<Material>> &p_materials) {
	CSGBrush *brush = memnew(CSGBrush);

	PoolVector<bool> invert;
	invert.resize(p_vertices.size() / 3);
	{
		int ic = invert.size();
		PoolVector<bool>::Write w = invert.write();
		for (int i = 0; i < ic; i++) {
			w[i] = invert_faces;
		}
	}
	brush->build_from_faces(p_vertices, p_uv, p_smooth, p_materials, invert);

	return brush;
}

void CSGPrimitive::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_invert_faces", "invert_faces"), &CSGPrimitive::set_invert_faces);
	ClassDB::bind_method(D_METHOD("is_inverting_faces"), &CSGPrimitive::is_inverting_faces);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert_faces"), "set_invert_faces", "is_inverting_faces");
}

void CSGPrimitive::set_invert_faces(bool p_invert) {
	if (invert_faces == p_invert) {
		return;
	}

	invert_faces = p_invert;

	_make_dirty();
}

bool CSGPrimitive::is_inverting_faces() {
	return invert_faces;
}

CSGPrimitive::CSGPrimitive() {
	invert_faces = false;
}

/////////////////////

CSGBrush *CSGMesh::_build_brush() {
	if (!mesh.is_valid()) {
		return memnew(CSGBrush);
	}

	PoolVector<Vector3> vertices;
	PoolVector<bool> smooth;
	PoolVector<Ref<Material>> materials;
	PoolVector<Vector2> uvs;
	Ref<Material> material = get_material();

	for (int i = 0; i < mesh->get_surface_count(); i++) {
		if (mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
			continue;
		}

		Array arrays = mesh->surface_get_arrays(i);

		if (arrays.size() == 0) {
			_make_dirty();
			ERR_FAIL_COND_V(arrays.size() == 0, memnew(CSGBrush));
		}

		PoolVector<Vector3> avertices = arrays[Mesh::ARRAY_VERTEX];
		if (avertices.size() == 0) {
			continue;
		}

		PoolVector<Vector3>::Read vr = avertices.read();

		PoolVector<Vector3> anormals = arrays[Mesh::ARRAY_NORMAL];
		PoolVector<Vector3>::Read nr;
		bool nr_used = false;
		if (anormals.size()) {
			nr = anormals.read();
			nr_used = true;
		}

		PoolVector<Vector2> auvs = arrays[Mesh::ARRAY_TEX_UV];
		PoolVector<Vector2>::Read uvr;
		bool uvr_used = false;
		if (auvs.size()) {
			uvr = auvs.read();
			uvr_used = true;
		}

		Ref<Material> mat;
		if (material.is_valid()) {
			mat = material;
		} else {
			mat = mesh->surface_get_material(i);
		}

		PoolVector<int> aindices = arrays[Mesh::ARRAY_INDEX];
		if (aindices.size()) {
			int as = vertices.size();
			int is = aindices.size();

			vertices.resize(as + is);
			smooth.resize((as + is) / 3);
			materials.resize((as + is) / 3);
			uvs.resize(as + is);

			PoolVector<Vector3>::Write vw = vertices.write();
			PoolVector<bool>::Write sw = smooth.write();
			PoolVector<Vector2>::Write uvw = uvs.write();
			PoolVector<Ref<Material>>::Write mw = materials.write();

			PoolVector<int>::Read ir = aindices.read();

			for (int j = 0; j < is; j += 3) {
				Vector3 vertex[3];
				Vector3 normal[3];
				Vector2 uv[3];

				for (int k = 0; k < 3; k++) {
					int idx = ir[j + k];
					vertex[k] = vr[idx];
					if (nr_used) {
						normal[k] = nr[idx];
					}
					if (uvr_used) {
						uv[k] = uvr[idx];
					}
				}

				bool flat = normal[0].distance_to(normal[1]) < CMP_EPSILON && normal[0].distance_to(normal[2]) < CMP_EPSILON;

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

			PoolVector<Vector3>::Write vw = vertices.write();
			PoolVector<bool>::Write sw = smooth.write();
			PoolVector<Vector2>::Write uvw = uvs.write();
			PoolVector<Ref<Material>>::Write mw = materials.write();

			for (int j = 0; j < is; j += 3) {
				Vector3 vertex[3];
				Vector3 normal[3];
				Vector2 uv[3];

				for (int k = 0; k < 3; k++) {
					vertex[k] = vr[j + k];
					if (nr_used) {
						normal[k] = nr[j + k];
					}
					if (uvr_used) {
						uv[k] = uvr[j + k];
					}
				}

				bool flat = normal[0].distance_to(normal[1]) < CMP_EPSILON && normal[0].distance_to(normal[2]) < CMP_EPSILON;

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

	if (vertices.size() == 0) {
		return memnew(CSGBrush);
	}

	return _create_brush_from_arrays(vertices, uvs, smooth, materials);
}

void CSGMesh::_mesh_changed() {
	_make_dirty();
	update_gizmo();
}

void CSGMesh::set_material(const Ref<Material> &p_material) {
	if (material == p_material) {
		return;
	}
	material = p_material;
	_make_dirty();
}

Ref<Material> CSGMesh::get_material() const {
	return material;
}

void CSGMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &CSGMesh::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &CSGMesh::get_mesh);

	ClassDB::bind_method(D_METHOD("_mesh_changed"), &CSGMesh::_mesh_changed);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGMesh::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGMesh::get_material);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "SpatialMaterial,ShaderMaterial"), "set_material", "get_material");
}

void CSGMesh::set_mesh(const Ref<Mesh> &p_mesh) {
	if (mesh == p_mesh) {
		return;
	}
	if (mesh.is_valid()) {
		mesh->disconnect("changed", this, "_mesh_changed");
	}
	mesh = p_mesh;

	if (mesh.is_valid()) {
		mesh->connect("changed", this, "_mesh_changed");
	}

	_mesh_changed();
}

Ref<Mesh> CSGMesh::get_mesh() {
	return mesh;
}

////////////////////////////////

CSGBrush *CSGSphere::_build_brush() {
	// set our bounding box

	CSGBrush *brush = memnew(CSGBrush);

	int face_count = rings * radial_segments * 2 - radial_segments * 2;

	bool invert_val = is_inverting_faces();
	Ref<Material> material = get_material();

	PoolVector<Vector3> faces;
	PoolVector<Vector2> uvs;
	PoolVector<bool> smooth;
	PoolVector<Ref<Material>> materials;
	PoolVector<bool> invert;

	faces.resize(face_count * 3);
	uvs.resize(face_count * 3);

	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		PoolVector<Vector3>::Write facesw = faces.write();
		PoolVector<Vector2>::Write uvsw = uvs.write();
		PoolVector<bool>::Write smoothw = smooth.write();
		PoolVector<Ref<Material>>::Write materialsw = materials.write();
		PoolVector<bool>::Write invertw = invert.write();

		// We want to follow an order that's convenient for UVs.
		// For latitude step we start at the top and move down like in an image.
		const double latitude_step = -Math_PI / rings;
		const double longitude_step = Math_TAU / radial_segments;
		int face = 0;
		for (int i = 0; i < rings; i++) {
			double latitude0 = latitude_step * i + Math_TAU / 4;
			double cos0 = Math::cos(latitude0);
			double sin0 = Math::sin(latitude0);
			double v0 = double(i) / rings;

			double latitude1 = latitude_step * (i + 1) + Math_TAU / 4;
			double cos1 = Math::cos(latitude1);
			double sin1 = Math::sin(latitude1);
			double v1 = double(i + 1) / rings;

			for (int j = 0; j < radial_segments; j++) {
				double longitude0 = longitude_step * j;
				// We give sin to X and cos to Z on purpose.
				// This allows UVs to be CCW on +X so it maps to images well.
				double x0 = Math::sin(longitude0);
				double z0 = Math::cos(longitude0);
				double u0 = double(j) / radial_segments;

				double longitude1 = longitude_step * (j + 1);
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
					materialsw[face] = material;

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
					materialsw[face] = material;

					face++;
				}
			}
		}

		if (face != face_count) {
			ERR_PRINT("Face mismatch bug! fix code");
		}
	}

	brush->build_from_faces(faces, uvs, smooth, materials, invert);

	return brush;
}

void CSGSphere::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CSGSphere::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CSGSphere::get_radius);

	ClassDB::bind_method(D_METHOD("set_radial_segments", "radial_segments"), &CSGSphere::set_radial_segments);
	ClassDB::bind_method(D_METHOD("get_radial_segments"), &CSGSphere::get_radial_segments);
	ClassDB::bind_method(D_METHOD("set_rings", "rings"), &CSGSphere::set_rings);
	ClassDB::bind_method(D_METHOD("get_rings"), &CSGSphere::get_rings);

	ClassDB::bind_method(D_METHOD("set_smooth_faces", "smooth_faces"), &CSGSphere::set_smooth_faces);
	ClassDB::bind_method(D_METHOD("get_smooth_faces"), &CSGSphere::get_smooth_faces);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGSphere::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGSphere::get_material);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "1,100,1"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "1,100,1"), "set_rings", "get_rings");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "SpatialMaterial,ShaderMaterial"), "set_material", "get_material");
}

void CSGSphere::set_radius(const float p_radius) {
	ERR_FAIL_COND(p_radius <= 0);
	radius = p_radius;
	_make_dirty();
	update_gizmo();
	_change_notify("radius");
}

float CSGSphere::get_radius() const {
	return radius;
}

void CSGSphere::set_radial_segments(const int p_radial_segments) {
	radial_segments = p_radial_segments > 4 ? p_radial_segments : 4;
	_make_dirty();
	update_gizmo();
}

int CSGSphere::get_radial_segments() const {
	return radial_segments;
}

void CSGSphere::set_rings(const int p_rings) {
	rings = p_rings > 1 ? p_rings : 1;
	_make_dirty();
	update_gizmo();
}

int CSGSphere::get_rings() const {
	return rings;
}

void CSGSphere::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	_make_dirty();
}

bool CSGSphere::get_smooth_faces() const {
	return smooth_faces;
}

void CSGSphere::set_material(const Ref<Material> &p_material) {
	material = p_material;
	_make_dirty();
}

Ref<Material> CSGSphere::get_material() const {
	return material;
}

CSGSphere::CSGSphere() {
	// defaults
	radius = 1.0;
	radial_segments = 12;
	rings = 6;
	smooth_faces = true;
}

///////////////

CSGBrush *CSGBox::_build_brush() {
	// set our bounding box

	CSGBrush *brush = memnew(CSGBrush);

	int face_count = 12; //it's a cube..

	bool invert_val = is_inverting_faces();
	Ref<Material> material = get_material();

	PoolVector<Vector3> faces;
	PoolVector<Vector2> uvs;
	PoolVector<bool> smooth;
	PoolVector<Ref<Material>> materials;
	PoolVector<bool> invert;

	faces.resize(face_count * 3);
	uvs.resize(face_count * 3);

	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		PoolVector<Vector3>::Write facesw = faces.write();
		PoolVector<Vector2>::Write uvsw = uvs.write();
		PoolVector<bool>::Write smoothw = smooth.write();
		PoolVector<Ref<Material>>::Write materialsw = materials.write();
		PoolVector<bool>::Write invertw = invert.write();

		int face = 0;

		Vector3 vertex_mul(width * 0.5, height * 0.5, depth * 0.5);

		{
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
				materialsw[face] = material;

				face++;
				//face 1
				facesw[face * 3 + 0] = face_points[2] * vertex_mul;
				facesw[face * 3 + 1] = face_points[3] * vertex_mul;
				facesw[face * 3 + 2] = face_points[0] * vertex_mul;

				uvsw[face * 3 + 0] = u[2];
				uvsw[face * 3 + 1] = u[3];
				uvsw[face * 3 + 2] = u[0];

				smoothw[face] = false;
				invertw[face] = invert_val;
				materialsw[face] = material;

				face++;
			}
		}

		if (face != face_count) {
			ERR_PRINT("Face mismatch bug! fix code");
		}
	}

	brush->build_from_faces(faces, uvs, smooth, materials, invert);

	return brush;
}

void CSGBox::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_width", "width"), &CSGBox::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &CSGBox::get_width);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGBox::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGBox::get_height);

	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &CSGBox::set_depth);
	ClassDB::bind_method(D_METHOD("get_depth"), &CSGBox::get_depth);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGBox::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGBox::get_material);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "width", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "depth", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "SpatialMaterial,ShaderMaterial"), "set_material", "get_material");
}

void CSGBox::set_width(const float p_width) {
	width = p_width;
	_make_dirty();
	update_gizmo();
	_change_notify("width");
}

float CSGBox::get_width() const {
	return width;
}

void CSGBox::set_height(const float p_height) {
	height = p_height;
	_make_dirty();
	update_gizmo();
	_change_notify("height");
}

float CSGBox::get_height() const {
	return height;
}

void CSGBox::set_depth(const float p_depth) {
	depth = p_depth;
	_make_dirty();
	update_gizmo();
	_change_notify("depth");
}

float CSGBox::get_depth() const {
	return depth;
}

void CSGBox::set_material(const Ref<Material> &p_material) {
	material = p_material;
	_make_dirty();
	update_gizmo();
}

Ref<Material> CSGBox::get_material() const {
	return material;
}

CSGBox::CSGBox() {
	// defaults
	width = 2.0;
	height = 2.0;
	depth = 2.0;
}

///////////////

CSGBrush *CSGCylinder::_build_brush() {
	// set our bounding box

	CSGBrush *brush = memnew(CSGBrush);

	int face_count = sides * (cone ? 1 : 2) + sides + (cone ? 0 : sides);

	bool invert_val = is_inverting_faces();
	Ref<Material> material = get_material();

	PoolVector<Vector3> faces;
	PoolVector<Vector2> uvs;
	PoolVector<bool> smooth;
	PoolVector<Ref<Material>> materials;
	PoolVector<bool> invert;

	faces.resize(face_count * 3);
	uvs.resize(face_count * 3);

	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		PoolVector<Vector3>::Write facesw = faces.write();
		PoolVector<Vector2>::Write uvsw = uvs.write();
		PoolVector<bool>::Write smoothw = smooth.write();
		PoolVector<Ref<Material>>::Write materialsw = materials.write();
		PoolVector<bool>::Write invertw = invert.write();

		int face = 0;

		Vector3 vertex_mul(radius, height * 0.5, radius);

		{
			for (int i = 0; i < sides; i++) {
				float inc = float(i) / sides;
				float inc_n = float((i + 1)) / sides;

				float ang = inc * Math_PI * 2.0;
				float ang_n = inc_n * Math_PI * 2.0;

				Vector3 base(Math::cos(ang), 0, Math::sin(ang));
				Vector3 base_n(Math::cos(ang_n), 0, Math::sin(ang_n));

				Vector3 face_points[4] = {
					base + Vector3(0, -1, 0),
					base_n + Vector3(0, -1, 0),
					base_n * (cone ? 0.0 : 1.0) + Vector3(0, 1, 0),
					base * (cone ? 0.0 : 1.0) + Vector3(0, 1, 0),
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
				materialsw[face] = material;

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
					materialsw[face] = material;
					face++;
				}

				//bottom face 1
				facesw[face * 3 + 0] = face_points[1] * vertex_mul;
				facesw[face * 3 + 1] = face_points[0] * vertex_mul;
				facesw[face * 3 + 2] = Vector3(0, -1, 0) * vertex_mul;

				uvsw[face * 3 + 0] = Vector2(face_points[1].x, face_points[1].y) * 0.5 + Vector2(0.5, 0.5);
				uvsw[face * 3 + 1] = Vector2(face_points[0].x, face_points[0].y) * 0.5 + Vector2(0.5, 0.5);
				uvsw[face * 3 + 2] = Vector2(0.5, 0.5);

				smoothw[face] = false;
				invertw[face] = invert_val;
				materialsw[face] = material;
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
					materialsw[face] = material;
					face++;
				}
			}
		}

		if (face != face_count) {
			ERR_PRINT("Face mismatch bug! fix code");
		}
	}

	brush->build_from_faces(faces, uvs, smooth, materials, invert);

	return brush;
}

void CSGCylinder::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CSGCylinder::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CSGCylinder::get_radius);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGCylinder::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGCylinder::get_height);

	ClassDB::bind_method(D_METHOD("set_sides", "sides"), &CSGCylinder::set_sides);
	ClassDB::bind_method(D_METHOD("get_sides"), &CSGCylinder::get_sides);

	ClassDB::bind_method(D_METHOD("set_cone", "cone"), &CSGCylinder::set_cone);
	ClassDB::bind_method(D_METHOD("is_cone"), &CSGCylinder::is_cone);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGCylinder::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGCylinder::get_material);

	ClassDB::bind_method(D_METHOD("set_smooth_faces", "smooth_faces"), &CSGCylinder::set_smooth_faces);
	ClassDB::bind_method(D_METHOD("get_smooth_faces"), &CSGCylinder::get_smooth_faces);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_sides", "get_sides");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cone"), "set_cone", "is_cone");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "SpatialMaterial,ShaderMaterial"), "set_material", "get_material");
}

void CSGCylinder::set_radius(const float p_radius) {
	radius = p_radius;
	_make_dirty();
	update_gizmo();
	_change_notify("radius");
}

float CSGCylinder::get_radius() const {
	return radius;
}

void CSGCylinder::set_height(const float p_height) {
	height = p_height;
	_make_dirty();
	update_gizmo();
	_change_notify("height");
}

float CSGCylinder::get_height() const {
	return height;
}

void CSGCylinder::set_sides(const int p_sides) {
	ERR_FAIL_COND(p_sides < 3);
	sides = p_sides;
	_make_dirty();
	update_gizmo();
}

int CSGCylinder::get_sides() const {
	return sides;
}

void CSGCylinder::set_cone(const bool p_cone) {
	cone = p_cone;
	_make_dirty();
	update_gizmo();
}

bool CSGCylinder::is_cone() const {
	return cone;
}

void CSGCylinder::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	_make_dirty();
}

bool CSGCylinder::get_smooth_faces() const {
	return smooth_faces;
}

void CSGCylinder::set_material(const Ref<Material> &p_material) {
	material = p_material;
	_make_dirty();
}

Ref<Material> CSGCylinder::get_material() const {
	return material;
}

CSGCylinder::CSGCylinder() {
	// defaults
	radius = 1.0;
	height = 1.0;
	sides = 8;
	cone = false;
	smooth_faces = true;
}

///////////////

CSGBrush *CSGTorus::_build_brush() {
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

	CSGBrush *brush = memnew(CSGBrush);

	int face_count = ring_sides * sides * 2;

	bool invert_val = is_inverting_faces();
	Ref<Material> material = get_material();

	PoolVector<Vector3> faces;
	PoolVector<Vector2> uvs;
	PoolVector<bool> smooth;
	PoolVector<Ref<Material>> materials;
	PoolVector<bool> invert;

	faces.resize(face_count * 3);
	uvs.resize(face_count * 3);

	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		PoolVector<Vector3>::Write facesw = faces.write();
		PoolVector<Vector2>::Write uvsw = uvs.write();
		PoolVector<bool>::Write smoothw = smooth.write();
		PoolVector<Ref<Material>>::Write materialsw = materials.write();
		PoolVector<bool>::Write invertw = invert.write();

		int face = 0;

		{
			for (int i = 0; i < sides; i++) {
				float inci = float(i) / sides;
				float inci_n = float((i + 1)) / sides;

				float angi = inci * Math_PI * 2.0;
				float angi_n = inci_n * Math_PI * 2.0;

				Vector3 normali = Vector3(Math::cos(angi), 0, Math::sin(angi));
				Vector3 normali_n = Vector3(Math::cos(angi_n), 0, Math::sin(angi_n));

				for (int j = 0; j < ring_sides; j++) {
					float incj = float(j) / ring_sides;
					float incj_n = float((j + 1)) / ring_sides;

					float angj = incj * Math_PI * 2.0;
					float angj_n = incj_n * Math_PI * 2.0;

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
					materialsw[face] = material;

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
					materialsw[face] = material;
					face++;
				}
			}
		}

		if (face != face_count) {
			ERR_PRINT("Face mismatch bug! fix code");
		}
	}

	brush->build_from_faces(faces, uvs, smooth, materials, invert);

	return brush;
}

void CSGTorus::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_inner_radius", "radius"), &CSGTorus::set_inner_radius);
	ClassDB::bind_method(D_METHOD("get_inner_radius"), &CSGTorus::get_inner_radius);

	ClassDB::bind_method(D_METHOD("set_outer_radius", "radius"), &CSGTorus::set_outer_radius);
	ClassDB::bind_method(D_METHOD("get_outer_radius"), &CSGTorus::get_outer_radius);

	ClassDB::bind_method(D_METHOD("set_sides", "sides"), &CSGTorus::set_sides);
	ClassDB::bind_method(D_METHOD("get_sides"), &CSGTorus::get_sides);

	ClassDB::bind_method(D_METHOD("set_ring_sides", "sides"), &CSGTorus::set_ring_sides);
	ClassDB::bind_method(D_METHOD("get_ring_sides"), &CSGTorus::get_ring_sides);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGTorus::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGTorus::get_material);

	ClassDB::bind_method(D_METHOD("set_smooth_faces", "smooth_faces"), &CSGTorus::set_smooth_faces);
	ClassDB::bind_method(D_METHOD("get_smooth_faces"), &CSGTorus::get_smooth_faces);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "inner_radius", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_inner_radius", "get_inner_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "outer_radius", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_outer_radius", "get_outer_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_sides", "get_sides");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ring_sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_ring_sides", "get_ring_sides");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "SpatialMaterial,ShaderMaterial"), "set_material", "get_material");
}

void CSGTorus::set_inner_radius(const float p_inner_radius) {
	inner_radius = p_inner_radius;
	_make_dirty();
	update_gizmo();
	_change_notify("inner_radius");
}

float CSGTorus::get_inner_radius() const {
	return inner_radius;
}

void CSGTorus::set_outer_radius(const float p_outer_radius) {
	outer_radius = p_outer_radius;
	_make_dirty();
	update_gizmo();
	_change_notify("outer_radius");
}

float CSGTorus::get_outer_radius() const {
	return outer_radius;
}

void CSGTorus::set_sides(const int p_sides) {
	ERR_FAIL_COND(p_sides < 3);
	sides = p_sides;
	_make_dirty();
	update_gizmo();
}

int CSGTorus::get_sides() const {
	return sides;
}

void CSGTorus::set_ring_sides(const int p_ring_sides) {
	ERR_FAIL_COND(p_ring_sides < 3);
	ring_sides = p_ring_sides;
	_make_dirty();
	update_gizmo();
}

int CSGTorus::get_ring_sides() const {
	return ring_sides;
}

void CSGTorus::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	_make_dirty();
}

bool CSGTorus::get_smooth_faces() const {
	return smooth_faces;
}

void CSGTorus::set_material(const Ref<Material> &p_material) {
	material = p_material;
	_make_dirty();
}

Ref<Material> CSGTorus::get_material() const {
	return material;
}

CSGTorus::CSGTorus() {
	// defaults
	inner_radius = 2.0;
	outer_radius = 3.0;
	sides = 8;
	ring_sides = 6;
	smooth_faces = true;
}

///////////////

CSGBrush *CSGPolygon::_build_brush() {
	CSGBrush *brush = memnew(CSGBrush);

	if (polygon.size() < 3) {
		return brush;
	}

	// Triangulate polygon shape.
	Vector<Point2> shape_polygon = polygon;
	if (Triangulate::get_area(shape_polygon) > 0) {
		shape_polygon.invert();
	}
	int shape_sides = shape_polygon.size();
	Vector<int> shape_faces = Geometry::triangulate_polygon(shape_polygon);
	ERR_FAIL_COND_V_MSG(shape_faces.size() < 3, brush, "Failed to triangulate CSGPolygon. Make sure the polygon doesn't have any intersecting edges.");

	// Get polygon enclosing Rect2.
	Rect2 shape_rect(shape_polygon[0], Vector2());
	for (int i = 1; i < shape_sides; i++) {
		shape_rect.expand_to(shape_polygon[i]);
	}

	// If MODE_PATH, check if curve has changed.
	Ref<Curve3D> curve;
	if (mode == MODE_PATH) {
		Path *current_path = Object::cast_to<Path>(get_node_or_null(path_node));
		if (path != current_path) {
			if (path) {
				path->disconnect("tree_exited", this, "_path_exited");
				path->disconnect("curve_changed", this, "_path_changed");
			}
			path = current_path;
			if (path) {
				path->connect("tree_exited", this, "_path_exited");
				path->connect("curve_changed", this, "_path_changed");
			}
		}

		if (!path) {
			return brush;
		}

		curve = path->get_curve();
		if (curve.is_null() || curve->get_point_count() < 2) {
			return brush;
		}
	}

	// Calculate the number of extrusions, ends and faces.
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
	Ref<Material> material = get_material();

	PoolVector<Vector3> faces;
	PoolVector<Vector2> uvs;
	PoolVector<bool> smooth;
	PoolVector<Ref<Material>> materials;
	PoolVector<bool> invert;

	faces.resize(face_count * 3);
	uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);
	int faces_removed = 0;

	{
		PoolVector<Vector3>::Write facesw = faces.write();
		PoolVector<Vector2>::Write uvsw = uvs.write();
		PoolVector<bool>::Write smoothw = smooth.write();
		PoolVector<Ref<Material>>::Write materialsw = materials.write();
		PoolVector<bool>::Write invertw = invert.write();

		int face = 0;
		Transform base_xform;
		Transform current_xform;
		Transform previous_xform;
		Transform previous_previous_xform;
		double u_step = 1.0 / extrusions;
		if (path_u_distance > 0.0) {
			u_step *= curve_length / path_u_distance;
		}
		double v_step = 1.0 / shape_sides;
		double spin_step = Math::deg2rad(spin_degrees / spin_sides);
		double extrusion_step = 1.0 / extrusions;
		if (mode == MODE_PATH) {
			if (path_joined) {
				extrusion_step = 1.0 / (extrusions - 1);
			}
			extrusion_step *= curve_length;
		}

		if (mode == MODE_PATH) {
			if (!path_local) {
				base_xform = path->get_global_transform();
			}

			Vector3 current_point = curve->interpolate_baked(0);
			Vector3 next_point = curve->interpolate_baked(extrusion_step);
			Vector3 current_up = Vector3(0, 1, 0);
			Vector3 direction = next_point - current_point;

			if (path_joined) {
				Vector3 last_point = curve->interpolate_baked(curve->get_baked_length());
				direction = next_point - last_point;
			}

			switch (path_rotation) {
				case PATH_ROTATION_POLYGON:
					direction = Vector3(0, 0, -1);
					break;
				case PATH_ROTATION_PATH:
					break;
				case PATH_ROTATION_PATH_FOLLOW:
					current_up = curve->interpolate_baked_up_vector(0);
					break;
			}

			Transform facing = Transform().looking_at(direction, current_up);
			current_xform = base_xform.translated(current_point) * facing;
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
				materialsw[face] = material;
				invertw[face] = invert_faces;
				face++;
			}
		}

		real_t angle_simplify_dot = Math::cos(Math::deg2rad(path_simplify_angle));
		Vector3 previous_simplify_dir = Vector3(0, 0, 0);
		int faces_combined = 0;

		// Add extrusion faces.
		for (int x0 = 0; x0 < extrusions; x0++) {
			previous_previous_xform = previous_xform;
			previous_xform = current_xform;

			switch (mode) {
				case MODE_DEPTH: {
					current_xform.translate(Vector3(0, 0, -depth));
				} break;
				case MODE_SPIN: {
					current_xform.rotate(Vector3(0, 1, 0), spin_step);
				} break;
				case MODE_PATH: {
					double previous_offset = x0 * extrusion_step;
					double current_offset = (x0 + 1) * extrusion_step;
					double next_offset = (x0 + 2) * extrusion_step;
					if (x0 == extrusions - 1) {
						if (path_joined) {
							current_offset = 0;
							next_offset = extrusion_step;
						} else {
							next_offset = current_offset;
						}
					}

					Vector3 previous_point = curve->interpolate_baked(previous_offset);
					Vector3 current_point = curve->interpolate_baked(current_offset);
					Vector3 next_point = curve->interpolate_baked(next_offset);
					Vector3 current_up = Vector3(0, 1, 0);
					Vector3 direction = next_point - previous_point;
					Vector3 current_dir = (current_point - previous_point).normalized();

					// If the angles are similar, remove the previous face and replace it with this one.
					if (path_simplify_angle > 0.0 && x0 > 0 && previous_simplify_dir.dot(current_dir) > angle_simplify_dot) {
						faces_combined += 1;
						previous_xform = previous_previous_xform;
						face -= extrusion_face_count;
						faces_removed += extrusion_face_count;
					} else {
						faces_combined = 0;
						previous_simplify_dir = current_dir;
					}

					switch (path_rotation) {
						case PATH_ROTATION_POLYGON:
							direction = Vector3(0, 0, -1);
							break;
						case PATH_ROTATION_PATH:
							break;
						case PATH_ROTATION_PATH_FOLLOW:
							current_up = curve->interpolate_baked_up_vector(current_offset);
							break;
					}

					Transform facing = Transform().looking_at(direction, current_up);
					current_xform = base_xform.translated(current_point) * facing;
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
				invertw[face] = invert_faces;
				materialsw[face] = material;

				face++;

				// Face 2
				facesw[face * 3 + 0] = v[2];
				facesw[face * 3 + 1] = v[3];
				facesw[face * 3 + 2] = v[0];

				uvsw[face * 3 + 0] = u[2];
				uvsw[face * 3 + 1] = u[3];
				uvsw[face * 3 + 2] = u[0];

				smoothw[face] = smooth_faces;
				invertw[face] = invert_faces;
				materialsw[face] = material;

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
				materialsw[face] = material;
				invertw[face] = invert_faces;
				face++;
			}
		}

		face_count -= faces_removed;
		ERR_FAIL_COND_V_MSG(face != face_count, brush, "Bug: Failed to create the CSGPolygon mesh correctly.");
	}

	if (faces_removed > 0) {
		faces.resize(face_count * 3);
		uvs.resize(face_count * 3);
		smooth.resize(face_count);
		materials.resize(face_count);
		invert.resize(face_count);
	}

	brush->build_from_faces(faces, uvs, smooth, materials, invert);

	return brush;
}

void CSGPolygon::_notification(int p_what) {
	if (p_what == NOTIFICATION_EXIT_TREE) {
		if (path) {
			path->disconnect("tree_exited", this, "_path_exited");
			path->disconnect("curve_changed", this, "_path_changed");
			path = nullptr;
		}
	}
}

void CSGPolygon::_validate_property(PropertyInfo &property) const {
	if (property.name.begins_with("spin") && mode != MODE_SPIN) {
		property.usage = 0;
	}
	if (property.name.begins_with("path") && mode != MODE_PATH) {
		property.usage = 0;
	}
	if (property.name == "depth" && mode != MODE_DEPTH) {
		property.usage = 0;
	}

	CSGShape::_validate_property(property);
}

void CSGPolygon::_path_changed() {
	_make_dirty();
	update_gizmo();
}

void CSGPolygon::_path_exited() {
	path = nullptr;
}

void CSGPolygon::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_polygon", "polygon"), &CSGPolygon::set_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon"), &CSGPolygon::get_polygon);

	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &CSGPolygon::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &CSGPolygon::get_mode);

	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &CSGPolygon::set_depth);
	ClassDB::bind_method(D_METHOD("get_depth"), &CSGPolygon::get_depth);

	ClassDB::bind_method(D_METHOD("set_spin_degrees", "degrees"), &CSGPolygon::set_spin_degrees);
	ClassDB::bind_method(D_METHOD("get_spin_degrees"), &CSGPolygon::get_spin_degrees);

	ClassDB::bind_method(D_METHOD("set_spin_sides", "spin_sides"), &CSGPolygon::set_spin_sides);
	ClassDB::bind_method(D_METHOD("get_spin_sides"), &CSGPolygon::get_spin_sides);

	ClassDB::bind_method(D_METHOD("set_path_node", "path"), &CSGPolygon::set_path_node);
	ClassDB::bind_method(D_METHOD("get_path_node"), &CSGPolygon::get_path_node);

	ClassDB::bind_method(D_METHOD("set_path_interval_type", "interval_type"), &CSGPolygon::set_path_interval_type);
	ClassDB::bind_method(D_METHOD("get_path_interval_type"), &CSGPolygon::get_path_interval_type);

	ClassDB::bind_method(D_METHOD("set_path_interval", "path_interval"), &CSGPolygon::set_path_interval);
	ClassDB::bind_method(D_METHOD("get_path_interval"), &CSGPolygon::get_path_interval);

	ClassDB::bind_method(D_METHOD("set_path_simplify_angle", "degrees"), &CSGPolygon::set_path_simplify_angle);
	ClassDB::bind_method(D_METHOD("get_path_simplify_angle"), &CSGPolygon::get_path_simplify_angle);

	ClassDB::bind_method(D_METHOD("set_path_rotation", "path_rotation"), &CSGPolygon::set_path_rotation);
	ClassDB::bind_method(D_METHOD("get_path_rotation"), &CSGPolygon::get_path_rotation);

	ClassDB::bind_method(D_METHOD("set_path_local", "enable"), &CSGPolygon::set_path_local);
	ClassDB::bind_method(D_METHOD("is_path_local"), &CSGPolygon::is_path_local);

	ClassDB::bind_method(D_METHOD("set_path_continuous_u", "enable"), &CSGPolygon::set_path_continuous_u);
	ClassDB::bind_method(D_METHOD("is_path_continuous_u"), &CSGPolygon::is_path_continuous_u);

	ClassDB::bind_method(D_METHOD("set_path_u_distance", "distance"), &CSGPolygon::set_path_u_distance);
	ClassDB::bind_method(D_METHOD("get_path_u_distance"), &CSGPolygon::get_path_u_distance);

	ClassDB::bind_method(D_METHOD("set_path_joined", "enable"), &CSGPolygon::set_path_joined);
	ClassDB::bind_method(D_METHOD("is_path_joined"), &CSGPolygon::is_path_joined);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGPolygon::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGPolygon::get_material);

	ClassDB::bind_method(D_METHOD("set_smooth_faces", "smooth_faces"), &CSGPolygon::set_smooth_faces);
	ClassDB::bind_method(D_METHOD("get_smooth_faces"), &CSGPolygon::get_smooth_faces);

	ClassDB::bind_method(D_METHOD("_is_editable_3d_polygon"), &CSGPolygon::_is_editable_3d_polygon);
	ClassDB::bind_method(D_METHOD("_has_editable_3d_polygon_no_depth"), &CSGPolygon::_has_editable_3d_polygon_no_depth);

	ClassDB::bind_method(D_METHOD("_path_exited"), &CSGPolygon::_path_exited);
	ClassDB::bind_method(D_METHOD("_path_changed"), &CSGPolygon::_path_changed);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "polygon"), "set_polygon", "get_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Depth,Spin,Path"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "depth", PROPERTY_HINT_EXP_RANGE, "0.01,100.0,0.01,or_greater"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "spin_degrees", PROPERTY_HINT_RANGE, "1,360,0.1"), "set_spin_degrees", "get_spin_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spin_sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_spin_sides", "get_spin_sides");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "path_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Path"), "set_path_node", "get_path_node");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_interval_type", PROPERTY_HINT_ENUM, "Distance,Subdivide"), "set_path_interval_type", "get_path_interval_type");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "path_interval", PROPERTY_HINT_RANGE, "0.01,1.0,0.01,exp,or_greater"), "set_path_interval", "get_path_interval");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "path_simplify_angle", PROPERTY_HINT_EXP_RANGE, "0.0,180.0,0.1,or_greater"), "set_path_simplify_angle", "get_path_simplify_angle");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_rotation", PROPERTY_HINT_ENUM, "Polygon,Path,PathFollow"), "set_path_rotation", "get_path_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_local"), "set_path_local", "is_path_local");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_continuous_u"), "set_path_continuous_u", "is_path_continuous_u");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "path_u_distance", PROPERTY_HINT_RANGE, "0.0,10.0,0.01,or_greater"), "set_path_u_distance", "get_path_u_distance");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_joined"), "set_path_joined", "is_path_joined");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "SpatialMaterial,ShaderMaterial"), "set_material", "get_material");

	BIND_ENUM_CONSTANT(MODE_DEPTH);
	BIND_ENUM_CONSTANT(MODE_SPIN);
	BIND_ENUM_CONSTANT(MODE_PATH);

	BIND_ENUM_CONSTANT(PATH_ROTATION_POLYGON);
	BIND_ENUM_CONSTANT(PATH_ROTATION_PATH);
	BIND_ENUM_CONSTANT(PATH_ROTATION_PATH_FOLLOW);

	BIND_ENUM_CONSTANT(PATH_INTERVAL_DISTANCE);
	BIND_ENUM_CONSTANT(PATH_INTERVAL_SUBDIVIDE);
}

void CSGPolygon::set_polygon(const Vector<Vector2> &p_polygon) {
	polygon = p_polygon;
	_make_dirty();
	update_gizmo();
}

Vector<Vector2> CSGPolygon::get_polygon() const {
	return polygon;
}

void CSGPolygon::set_mode(Mode p_mode) {
	mode = p_mode;
	_make_dirty();
	update_gizmo();
	_change_notify();
}

CSGPolygon::Mode CSGPolygon::get_mode() const {
	return mode;
}

void CSGPolygon::set_depth(const float p_depth) {
	ERR_FAIL_COND(p_depth < 0.001);
	depth = p_depth;
	_make_dirty();
	update_gizmo();
}

float CSGPolygon::get_depth() const {
	return depth;
}

void CSGPolygon::set_path_continuous_u(bool p_enable) {
	path_continuous_u = p_enable;
	_make_dirty();
}

bool CSGPolygon::is_path_continuous_u() const {
	return path_continuous_u;
}

void CSGPolygon::set_path_u_distance(real_t p_path_u_distance) {
	path_u_distance = p_path_u_distance;
	_make_dirty();
	update_gizmo();
}

real_t CSGPolygon::get_path_u_distance() const {
	return path_u_distance;
}

void CSGPolygon::set_spin_degrees(const float p_spin_degrees) {
	ERR_FAIL_COND(p_spin_degrees < 0.01 || p_spin_degrees > 360);
	spin_degrees = p_spin_degrees;
	_make_dirty();
	update_gizmo();
}

float CSGPolygon::get_spin_degrees() const {
	return spin_degrees;
}

void CSGPolygon::set_spin_sides(int p_spin_sides) {
	ERR_FAIL_COND(p_spin_sides < 3);
	spin_sides = p_spin_sides;
	_make_dirty();
	update_gizmo();
}

int CSGPolygon::get_spin_sides() const {
	return spin_sides;
}

void CSGPolygon::set_path_node(const NodePath &p_path) {
	path_node = p_path;
	_make_dirty();
	update_gizmo();
}

NodePath CSGPolygon::get_path_node() const {
	return path_node;
}

void CSGPolygon::set_path_interval_type(PathIntervalType p_interval_type) {
	path_interval_type = p_interval_type;
	_make_dirty();
	update_gizmo();
}

CSGPolygon::PathIntervalType CSGPolygon::get_path_interval_type() const {
	return path_interval_type;
}

void CSGPolygon::set_path_interval(float p_interval) {
	path_interval = p_interval;
	_make_dirty();
	update_gizmo();
}

float CSGPolygon::get_path_interval() const {
	return path_interval;
}

void CSGPolygon::set_path_simplify_angle(float angle) {
	path_simplify_angle = angle;
	_make_dirty();
	update_gizmo();
}

float CSGPolygon::get_path_simplify_angle() const {
	return path_simplify_angle;
}

void CSGPolygon::set_path_rotation(PathRotation p_rotation) {
	path_rotation = p_rotation;
	_make_dirty();
	update_gizmo();
}

CSGPolygon::PathRotation CSGPolygon::get_path_rotation() const {
	return path_rotation;
}

void CSGPolygon::set_path_local(bool p_enable) {
	path_local = p_enable;
	_make_dirty();
	update_gizmo();
}

bool CSGPolygon::is_path_local() const {
	return path_local;
}

void CSGPolygon::set_path_joined(bool p_enable) {
	path_joined = p_enable;
	_make_dirty();
	update_gizmo();
}

bool CSGPolygon::is_path_joined() const {
	return path_joined;
}

void CSGPolygon::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	_make_dirty();
}

bool CSGPolygon::get_smooth_faces() const {
	return smooth_faces;
}

void CSGPolygon::set_material(const Ref<Material> &p_material) {
	material = p_material;
	_make_dirty();
}

Ref<Material> CSGPolygon::get_material() const {
	return material;
}

bool CSGPolygon::_is_editable_3d_polygon() const {
	return true;
}

bool CSGPolygon::_has_editable_3d_polygon_no_depth() const {
	return true;
}

CSGPolygon::CSGPolygon() {
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
	path_local = false;
	path_continuous_u = true;
	path_u_distance = 1.0;
	path_joined = false;
	path = nullptr;
}
