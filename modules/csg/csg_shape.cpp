/*************************************************************************/
/*  csg_shape.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "core/math/geometry_2d.h"
#include "scene/3d/path_3d.h"

void CSGShape3D::set_use_collision(bool p_enable) {
	if (use_collision == p_enable) {
		return;
	}

	use_collision = p_enable;

	if (!is_inside_tree() || !is_root_shape()) {
		return;
	}

	if (use_collision) {
		root_collision_shape.instance();
		root_collision_instance = PhysicsServer3D::get_singleton()->body_create(PhysicsServer3D::BODY_MODE_STATIC);
		PhysicsServer3D::get_singleton()->body_set_state(root_collision_instance, PhysicsServer3D::BODY_STATE_TRANSFORM, get_global_transform());
		PhysicsServer3D::get_singleton()->body_add_shape(root_collision_instance, root_collision_shape->get_rid());
		PhysicsServer3D::get_singleton()->body_set_space(root_collision_instance, get_world_3d()->get_space());
		PhysicsServer3D::get_singleton()->body_attach_object_instance_id(root_collision_instance, get_instance_id());
		set_collision_layer(collision_layer);
		set_collision_mask(collision_mask);
		_make_dirty(); //force update
	} else {
		PhysicsServer3D::get_singleton()->free(root_collision_instance);
		root_collision_instance = RID();
		root_collision_shape.unref();
	}
	_change_notify();
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

void CSGShape3D::set_collision_mask_bit(int p_bit, bool p_value) {
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << p_bit;
	} else {
		mask &= ~(1 << p_bit);
	}
	set_collision_mask(mask);
}

bool CSGShape3D::get_collision_mask_bit(int p_bit) const {
	return get_collision_mask() & (1 << p_bit);
}

void CSGShape3D::set_collision_layer_bit(int p_bit, bool p_value) {
	uint32_t mask = get_collision_layer();
	if (p_value) {
		mask |= 1 << p_bit;
	} else {
		mask &= ~(1 << p_bit);
	}
	set_collision_layer(mask);
}

bool CSGShape3D::get_collision_layer_bit(int p_bit) const {
	return get_collision_layer() & (1 << p_bit);
}

bool CSGShape3D::is_root_shape() const {
	return !parent;
}

void CSGShape3D::set_snap(float p_snap) {
	snap = p_snap;
}

float CSGShape3D::get_snap() const {
	return snap;
}

void CSGShape3D::_make_dirty() {
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

CSGBrush *CSGShape3D::_get_brush() {
	if (dirty) {
		if (brush) {
			memdelete(brush);
		}
		brush = nullptr;

		CSGBrush *n = _build_brush();

		for (int i = 0; i < get_child_count(); i++) {
			CSGShape3D *child = Object::cast_to<CSGShape3D>(get_child(i));
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
					case CSGShape3D::OPERATION_UNION:
						bop.merge_brushes(CSGBrushOperation::OPERATION_UNION, *n, *nn2, *nn, snap);
						break;
					case CSGShape3D::OPERATION_INTERSECTION:
						bop.merge_brushes(CSGBrushOperation::OPERATION_INTERSECTION, *n, *nn2, *nn, snap);
						break;
					case CSGShape3D::OPERATION_SUBTRACTION:
						bop.merge_brushes(CSGBrushOperation::OPERATION_SUBSTRACTION, *n, *nn2, *nn, snap);
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

void CSGShape3D::_update_shape() {
	if (parent) {
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

		surfaces.write[i].verticesw = surfaces.write[i].vertices.ptrw();
		surfaces.write[i].normalsw = surfaces.write[i].normals.ptrw();
		surfaces.write[i].uvsw = surfaces.write[i].uvs.ptrw();
		if (calculate_tangents) {
			surfaces.write[i].tansw = surfaces.write[i].tans.ptrw();
		}
	}

	// Update collision faces.
	if (root_collision_shape.is_valid()) {
		Vector<Vector3> physics_faces;
		physics_faces.resize(n->faces.size() * 3);
		Vector3 *physicsw = physics_faces.ptrw();

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

AABB CSGShape3D::get_aabb() const {
	return node_aabb;
}

Vector<Vector3> CSGShape3D::get_brush_faces() {
	ERR_FAIL_COND_V(!is_inside_tree(), Vector<Vector3>());
	CSGBrush *b = _get_brush();
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

Vector<Face3> CSGShape3D::get_faces(uint32_t p_usage_flags) const {
	return Vector<Face3>();
}

void CSGShape3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		Node *parentn = get_parent();
		if (parentn) {
			parent = Object::cast_to<CSGShape3D>(parentn);
			if (parent) {
				set_base(RID());
				root_mesh.unref();
			}
		}

		if (use_collision && is_root_shape()) {
			root_collision_shape.instance();
			root_collision_instance = PhysicsServer3D::get_singleton()->body_create(PhysicsServer3D::BODY_MODE_STATIC);
			PhysicsServer3D::get_singleton()->body_set_state(root_collision_instance, PhysicsServer3D::BODY_STATE_TRANSFORM, get_global_transform());
			PhysicsServer3D::get_singleton()->body_add_shape(root_collision_instance, root_collision_shape->get_rid());
			PhysicsServer3D::get_singleton()->body_set_space(root_collision_instance, get_world_3d()->get_space());
			PhysicsServer3D::get_singleton()->body_attach_object_instance_id(root_collision_instance, get_instance_id());
			set_collision_layer(collision_layer);
			set_collision_mask(collision_mask);
		}

		_make_dirty();
	}

	if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {
		if (use_collision && is_root_shape() && root_collision_instance.is_valid()) {
			PhysicsServer3D::get_singleton()->body_set_state(root_collision_instance, PhysicsServer3D::BODY_STATE_TRANSFORM, get_global_transform());
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
			PhysicsServer3D::get_singleton()->free(root_collision_instance);
			root_collision_instance = RID();
			root_collision_shape.unref();
		}
		_make_dirty();
	}
}

void CSGShape3D::set_operation(Operation p_operation) {
	operation = p_operation;
	_make_dirty();
	update_gizmo();
}

CSGShape3D::Operation CSGShape3D::get_operation() const {
	return operation;
}

void CSGShape3D::set_calculate_tangents(bool p_calculate_tangents) {
	calculate_tangents = p_calculate_tangents;
	_make_dirty();
}

bool CSGShape3D::is_calculating_tangents() const {
	return calculate_tangents;
}

void CSGShape3D::_validate_property(PropertyInfo &property) const {
	bool is_collision_prefixed = property.name.begins_with("collision_");
	if ((is_collision_prefixed || property.name.begins_with("use_collision")) && is_inside_tree() && !is_root_shape()) {
		//hide collision if not root
		property.usage = PROPERTY_USAGE_NOEDITOR;
	} else if (is_collision_prefixed && !bool(get("use_collision"))) {
		property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
	}
}

Array CSGShape3D::get_meshes() const {
	if (root_mesh.is_valid()) {
		Array arr;
		arr.resize(2);
		arr[0] = Transform();
		arr[1] = root_mesh;
		return arr;
	}

	return Array();
}

void CSGShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_shape"), &CSGShape3D::_update_shape);
	ClassDB::bind_method(D_METHOD("is_root_shape"), &CSGShape3D::is_root_shape);

	ClassDB::bind_method(D_METHOD("set_operation", "operation"), &CSGShape3D::set_operation);
	ClassDB::bind_method(D_METHOD("get_operation"), &CSGShape3D::get_operation);

	ClassDB::bind_method(D_METHOD("set_snap", "snap"), &CSGShape3D::set_snap);
	ClassDB::bind_method(D_METHOD("get_snap"), &CSGShape3D::get_snap);

	ClassDB::bind_method(D_METHOD("set_use_collision", "operation"), &CSGShape3D::set_use_collision);
	ClassDB::bind_method(D_METHOD("is_using_collision"), &CSGShape3D::is_using_collision);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "layer"), &CSGShape3D::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &CSGShape3D::get_collision_layer);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &CSGShape3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &CSGShape3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_bit", "bit", "value"), &CSGShape3D::set_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("get_collision_mask_bit", "bit"), &CSGShape3D::get_collision_mask_bit);

	ClassDB::bind_method(D_METHOD("set_collision_layer_bit", "bit", "value"), &CSGShape3D::set_collision_layer_bit);
	ClassDB::bind_method(D_METHOD("get_collision_layer_bit", "bit"), &CSGShape3D::get_collision_layer_bit);

	ClassDB::bind_method(D_METHOD("set_calculate_tangents", "enabled"), &CSGShape3D::set_calculate_tangents);
	ClassDB::bind_method(D_METHOD("is_calculating_tangents"), &CSGShape3D::is_calculating_tangents);

	ClassDB::bind_method(D_METHOD("get_meshes"), &CSGShape3D::get_meshes);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operation", PROPERTY_HINT_ENUM, "Union,Intersection,Subtraction"), "set_operation", "get_operation");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "snap", PROPERTY_HINT_RANGE, "0.0001,1,0.001"), "set_snap", "get_snap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "calculate_tangents"), "set_calculate_tangents", "is_calculating_tangents");

	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_collision"), "set_use_collision", "is_using_collision");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");

	BIND_ENUM_CONSTANT(OPERATION_UNION);
	BIND_ENUM_CONSTANT(OPERATION_INTERSECTION);
	BIND_ENUM_CONSTANT(OPERATION_SUBTRACTION);
}

CSGShape3D::CSGShape3D() {
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

CSGShape3D::~CSGShape3D() {
	if (brush) {
		memdelete(brush);
		brush = nullptr;
	}
}

//////////////////////////////////

CSGBrush *CSGCombiner3D::_build_brush() {
	return memnew(CSGBrush); //does not build anything
}

CSGCombiner3D::CSGCombiner3D() {
}

/////////////////////

CSGBrush *CSGPrimitive3D::_create_brush_from_arrays(const Vector<Vector3> &p_vertices, const Vector<Vector2> &p_uv, const Vector<bool> &p_smooth, const Vector<Ref<Material>> &p_materials) {
	CSGBrush *brush = memnew(CSGBrush);

	Vector<bool> invert;
	invert.resize(p_vertices.size() / 3);
	{
		int ic = invert.size();
		bool *w = invert.ptrw();
		for (int i = 0; i < ic; i++) {
			w[i] = invert_faces;
		}
	}
	brush->build_from_faces(p_vertices, p_uv, p_smooth, p_materials, invert);

	return brush;
}

void CSGPrimitive3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_invert_faces", "invert_faces"), &CSGPrimitive3D::set_invert_faces);
	ClassDB::bind_method(D_METHOD("is_inverting_faces"), &CSGPrimitive3D::is_inverting_faces);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert_faces"), "set_invert_faces", "is_inverting_faces");
}

void CSGPrimitive3D::set_invert_faces(bool p_invert) {
	if (invert_faces == p_invert) {
		return;
	}

	invert_faces = p_invert;

	_make_dirty();
}

bool CSGPrimitive3D::is_inverting_faces() {
	return invert_faces;
}

CSGPrimitive3D::CSGPrimitive3D() {
	invert_faces = false;
}

/////////////////////

CSGBrush *CSGMesh3D::_build_brush() {
	if (!mesh.is_valid()) {
		return nullptr;
	}

	Vector<Vector3> vertices;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<Vector2> uvs;
	Ref<Material> material = get_material();

	for (int i = 0; i < mesh->get_surface_count(); i++) {
		if (mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
			continue;
		}

		Array arrays = mesh->surface_get_arrays(i);

		if (arrays.size() == 0) {
			_make_dirty();
			ERR_FAIL_COND_V(arrays.size() == 0, nullptr);
		}

		Vector<Vector3> avertices = arrays[Mesh::ARRAY_VERTEX];
		if (avertices.size() == 0) {
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
		if (material.is_valid()) {
			mat = material;
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

			Vector3 *vw = vertices.ptrw();
			bool *sw = smooth.ptrw();
			Vector2 *uvw = uvs.ptrw();
			Ref<Material> *mw = materials.ptrw();

			const int *ir = aindices.ptr();

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

			Vector3 *vw = vertices.ptrw();
			bool *sw = smooth.ptrw();
			Vector2 *uvw = uvs.ptrw();
			Ref<Material> *mw = materials.ptrw();

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
		return nullptr;
	}

	return _create_brush_from_arrays(vertices, uvs, smooth, materials);
}

void CSGMesh3D::_mesh_changed() {
	_make_dirty();
	update_gizmo();
}

void CSGMesh3D::set_material(const Ref<Material> &p_material) {
	if (material == p_material) {
		return;
	}
	material = p_material;
	_make_dirty();
}

Ref<Material> CSGMesh3D::get_material() const {
	return material;
}

void CSGMesh3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &CSGMesh3D::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &CSGMesh3D::get_mesh);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGMesh3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGMesh3D::get_material);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "StandardMaterial3D,ShaderMaterial"), "set_material", "get_material");
}

void CSGMesh3D::set_mesh(const Ref<Mesh> &p_mesh) {
	if (mesh == p_mesh) {
		return;
	}
	if (mesh.is_valid()) {
		mesh->disconnect("changed", callable_mp(this, &CSGMesh3D::_mesh_changed));
	}
	mesh = p_mesh;

	if (mesh.is_valid()) {
		mesh->connect("changed", callable_mp(this, &CSGMesh3D::_mesh_changed));
	}

	_make_dirty();
}

Ref<Mesh> CSGMesh3D::get_mesh() {
	return mesh;
}

////////////////////////////////

CSGBrush *CSGSphere3D::_build_brush() {
	// set our bounding box

	CSGBrush *brush = memnew(CSGBrush);

	int face_count = rings * radial_segments * 2 - radial_segments * 2;

	bool invert_val = is_inverting_faces();
	Ref<Material> material = get_material();

	Vector<Vector3> faces;
	Vector<Vector2> uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

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

		int face = 0;

		for (int i = 1; i <= rings; i++) {
			double lat0 = Math_PI * (-0.5 + (double)(i - 1) / rings);
			double z0 = Math::sin(lat0);
			double zr0 = Math::cos(lat0);
			double u0 = double(i - 1) / rings;

			double lat1 = Math_PI * (-0.5 + (double)i / rings);
			double z1 = Math::sin(lat1);
			double zr1 = Math::cos(lat1);
			double u1 = double(i) / rings;

			for (int j = radial_segments; j >= 1; j--) {
				double lng0 = 2 * Math_PI * (double)(j - 1) / radial_segments;
				double x0 = Math::cos(lng0);
				double y0 = Math::sin(lng0);
				double v0 = double(i - 1) / radial_segments;

				double lng1 = 2 * Math_PI * (double)(j) / radial_segments;
				double x1 = Math::cos(lng1);
				double y1 = Math::sin(lng1);
				double v1 = double(i) / radial_segments;

				Vector3 v[4] = {
					Vector3(x1 * zr0, z0, y1 * zr0) * radius,
					Vector3(x1 * zr1, z1, y1 * zr1) * radius,
					Vector3(x0 * zr1, z1, y0 * zr1) * radius,
					Vector3(x0 * zr0, z0, y0 * zr0) * radius
				};

				Vector2 u[4] = {
					Vector2(v1, u0),
					Vector2(v1, u1),
					Vector2(v0, u1),
					Vector2(v0, u0),

				};

				if (i < rings) {
					//face 1
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

				if (i > 1) {
					//face 2
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

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "1,100,1"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "1,100,1"), "set_rings", "get_rings");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "StandardMaterial3D,ShaderMaterial"), "set_material", "get_material");
}

void CSGSphere3D::set_radius(const float p_radius) {
	ERR_FAIL_COND(p_radius <= 0);
	radius = p_radius;
	_make_dirty();
	update_gizmo();
	_change_notify("radius");
}

float CSGSphere3D::get_radius() const {
	return radius;
}

void CSGSphere3D::set_radial_segments(const int p_radial_segments) {
	radial_segments = p_radial_segments > 4 ? p_radial_segments : 4;
	_make_dirty();
	update_gizmo();
}

int CSGSphere3D::get_radial_segments() const {
	return radial_segments;
}

void CSGSphere3D::set_rings(const int p_rings) {
	rings = p_rings > 1 ? p_rings : 1;
	_make_dirty();
	update_gizmo();
}

int CSGSphere3D::get_rings() const {
	return rings;
}

void CSGSphere3D::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	_make_dirty();
}

bool CSGSphere3D::get_smooth_faces() const {
	return smooth_faces;
}

void CSGSphere3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	_make_dirty();
}

Ref<Material> CSGSphere3D::get_material() const {
	return material;
}

CSGSphere3D::CSGSphere3D() {
	// defaults
	radius = 1.0;
	radial_segments = 12;
	rings = 6;
	smooth_faces = true;
}

///////////////

CSGBrush *CSGBox3D::_build_brush() {
	// set our bounding box

	CSGBrush *brush = memnew(CSGBrush);

	int face_count = 12; //it's a cube..

	bool invert_val = is_inverting_faces();
	Ref<Material> material = get_material();

	Vector<Vector3> faces;
	Vector<Vector2> uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

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

void CSGBox3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_width", "width"), &CSGBox3D::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &CSGBox3D::get_width);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGBox3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGBox3D::get_height);

	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &CSGBox3D::set_depth);
	ClassDB::bind_method(D_METHOD("get_depth"), &CSGBox3D::get_depth);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGBox3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGBox3D::get_material);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "depth", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "StandardMaterial3D,ShaderMaterial"), "set_material", "get_material");
}

void CSGBox3D::set_width(const float p_width) {
	width = p_width;
	_make_dirty();
	update_gizmo();
	_change_notify("width");
}

float CSGBox3D::get_width() const {
	return width;
}

void CSGBox3D::set_height(const float p_height) {
	height = p_height;
	_make_dirty();
	update_gizmo();
	_change_notify("height");
}

float CSGBox3D::get_height() const {
	return height;
}

void CSGBox3D::set_depth(const float p_depth) {
	depth = p_depth;
	_make_dirty();
	update_gizmo();
	_change_notify("depth");
}

float CSGBox3D::get_depth() const {
	return depth;
}

void CSGBox3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	_make_dirty();
	update_gizmo();
}

Ref<Material> CSGBox3D::get_material() const {
	return material;
}

CSGBox3D::CSGBox3D() {
	// defaults
	width = 2.0;
	height = 2.0;
	depth = 2.0;
}

///////////////

CSGBrush *CSGCylinder3D::_build_brush() {
	// set our bounding box

	CSGBrush *brush = memnew(CSGBrush);

	int face_count = sides * (cone ? 1 : 2) + sides + (cone ? 0 : sides);

	bool invert_val = is_inverting_faces();
	Ref<Material> material = get_material();

	Vector<Vector3> faces;
	Vector<Vector2> uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

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

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_sides", "get_sides");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cone"), "set_cone", "is_cone");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "StandardMaterial3D,ShaderMaterial"), "set_material", "get_material");
}

void CSGCylinder3D::set_radius(const float p_radius) {
	radius = p_radius;
	_make_dirty();
	update_gizmo();
	_change_notify("radius");
}

float CSGCylinder3D::get_radius() const {
	return radius;
}

void CSGCylinder3D::set_height(const float p_height) {
	height = p_height;
	_make_dirty();
	update_gizmo();
	_change_notify("height");
}

float CSGCylinder3D::get_height() const {
	return height;
}

void CSGCylinder3D::set_sides(const int p_sides) {
	ERR_FAIL_COND(p_sides < 3);
	sides = p_sides;
	_make_dirty();
	update_gizmo();
}

int CSGCylinder3D::get_sides() const {
	return sides;
}

void CSGCylinder3D::set_cone(const bool p_cone) {
	cone = p_cone;
	_make_dirty();
	update_gizmo();
}

bool CSGCylinder3D::is_cone() const {
	return cone;
}

void CSGCylinder3D::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	_make_dirty();
}

bool CSGCylinder3D::get_smooth_faces() const {
	return smooth_faces;
}

void CSGCylinder3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	_make_dirty();
}

Ref<Material> CSGCylinder3D::get_material() const {
	return material;
}

CSGCylinder3D::CSGCylinder3D() {
	// defaults
	radius = 1.0;
	height = 1.0;
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
		return nullptr; //sorry, can't
	}

	if (min_radius > max_radius) {
		SWAP(min_radius, max_radius);
	}

	float radius = (max_radius - min_radius) * 0.5;

	CSGBrush *brush = memnew(CSGBrush);

	int face_count = ring_sides * sides * 2;

	bool invert_val = is_inverting_faces();
	Ref<Material> material = get_material();

	Vector<Vector3> faces;
	Vector<Vector2> uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

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

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inner_radius", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_inner_radius", "get_inner_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "outer_radius", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_outer_radius", "get_outer_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_sides", "get_sides");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ring_sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_ring_sides", "get_ring_sides");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "StandardMaterial3D,ShaderMaterial"), "set_material", "get_material");
}

void CSGTorus3D::set_inner_radius(const float p_inner_radius) {
	inner_radius = p_inner_radius;
	_make_dirty();
	update_gizmo();
	_change_notify("inner_radius");
}

float CSGTorus3D::get_inner_radius() const {
	return inner_radius;
}

void CSGTorus3D::set_outer_radius(const float p_outer_radius) {
	outer_radius = p_outer_radius;
	_make_dirty();
	update_gizmo();
	_change_notify("outer_radius");
}

float CSGTorus3D::get_outer_radius() const {
	return outer_radius;
}

void CSGTorus3D::set_sides(const int p_sides) {
	ERR_FAIL_COND(p_sides < 3);
	sides = p_sides;
	_make_dirty();
	update_gizmo();
}

int CSGTorus3D::get_sides() const {
	return sides;
}

void CSGTorus3D::set_ring_sides(const int p_ring_sides) {
	ERR_FAIL_COND(p_ring_sides < 3);
	ring_sides = p_ring_sides;
	_make_dirty();
	update_gizmo();
}

int CSGTorus3D::get_ring_sides() const {
	return ring_sides;
}

void CSGTorus3D::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	_make_dirty();
}

bool CSGTorus3D::get_smooth_faces() const {
	return smooth_faces;
}

void CSGTorus3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	_make_dirty();
}

Ref<Material> CSGTorus3D::get_material() const {
	return material;
}

CSGTorus3D::CSGTorus3D() {
	// defaults
	inner_radius = 2.0;
	outer_radius = 3.0;
	sides = 8;
	ring_sides = 6;
	smooth_faces = true;
}

///////////////

CSGBrush *CSGPolygon3D::_build_brush() {
	// set our bounding box

	if (polygon.size() < 3) {
		return nullptr;
	}

	Vector<Point2> final_polygon = polygon;

	if (Triangulate::get_area(final_polygon) > 0) {
		final_polygon.invert();
	}

	Vector<int> triangles = Geometry2D::triangulate_polygon(final_polygon);

	if (triangles.size() < 3) {
		return nullptr;
	}

	Path3D *path = nullptr;
	Ref<Curve3D> curve;

	// get bounds for our polygon
	Vector2 final_polygon_min;
	Vector2 final_polygon_max;
	for (int i = 0; i < final_polygon.size(); i++) {
		Vector2 p = final_polygon[i];
		if (i == 0) {
			final_polygon_min = p;
			final_polygon_max = final_polygon_min;
		} else {
			if (p.x < final_polygon_min.x) {
				final_polygon_min.x = p.x;
			}
			if (p.y < final_polygon_min.y) {
				final_polygon_min.y = p.y;
			}

			if (p.x > final_polygon_max.x) {
				final_polygon_max.x = p.x;
			}
			if (p.y > final_polygon_max.y) {
				final_polygon_max.y = p.y;
			}
		}
	}
	Vector2 final_polygon_size = final_polygon_max - final_polygon_min;

	if (mode == MODE_PATH) {
		if (!has_node(path_node)) {
			return nullptr;
		}
		Node *n = get_node(path_node);
		if (!n) {
			return nullptr;
		}
		path = Object::cast_to<Path3D>(n);
		if (!path) {
			return nullptr;
		}

		if (path != path_cache) {
			if (path_cache) {
				path_cache->disconnect("tree_exited", callable_mp(this, &CSGPolygon3D::_path_exited));
				path_cache->disconnect("curve_changed", callable_mp(this, &CSGPolygon3D::_path_changed));
				path_cache = nullptr;
			}

			path_cache = path;

			path_cache->connect("tree_exited", callable_mp(this, &CSGPolygon3D::_path_exited));
			path_cache->connect("curve_changed", callable_mp(this, &CSGPolygon3D::_path_changed));
			path_cache = nullptr;
		}
		curve = path->get_curve();
		if (curve.is_null()) {
			return nullptr;
		}
		if (curve->get_baked_length() <= 0) {
			return nullptr;
		}
	}
	CSGBrush *brush = memnew(CSGBrush);

	int face_count = 0;

	switch (mode) {
		case MODE_DEPTH:
			face_count = triangles.size() * 2 / 3 + (final_polygon.size()) * 2;
			break;
		case MODE_SPIN:
			face_count = (spin_degrees < 360 ? triangles.size() * 2 / 3 : 0) + (final_polygon.size()) * 2 * spin_sides;
			break;
		case MODE_PATH: {
			float bl = curve->get_baked_length();
			int splits = MAX(2, Math::ceil(bl / path_interval));
			if (path_joined) {
				face_count = splits * final_polygon.size() * 2;
			} else {
				face_count = triangles.size() * 2 / 3 + splits * final_polygon.size() * 2;
			}
		} break;
	}

	bool invert_val = is_inverting_faces();
	Ref<Material> material = get_material();

	Vector<Vector3> faces;
	Vector<Vector2> uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	faces.resize(face_count * 3);
	uvs.resize(face_count * 3);

	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	AABB aabb; //must be computed
	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *uvsw = uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		int face = 0;

		switch (mode) {
			case MODE_DEPTH: {
				//add triangles, front and back
				for (int i = 0; i < 2; i++) {
					for (int j = 0; j < triangles.size(); j += 3) {
						for (int k = 0; k < 3; k++) {
							int src[3] = { 0, i == 0 ? 1 : 2, i == 0 ? 2 : 1 };
							Vector2 p = final_polygon[triangles[j + src[k]]];
							Vector3 v = Vector3(p.x, p.y, 0);
							if (i == 0) {
								v.z -= depth;
							}
							facesw[face * 3 + k] = v;
							uvsw[face * 3 + k] = (p - final_polygon_min) / final_polygon_size;
							if (i == 0) {
								uvsw[face * 3 + k].x = 1.0 - uvsw[face * 3 + k].x; /* flip x */
							}
						}

						smoothw[face] = false;
						materialsw[face] = material;
						invertw[face] = invert_val;
						face++;
					}
				}

				//add triangles for depth
				for (int i = 0; i < final_polygon.size(); i++) {
					int i_n = (i + 1) % final_polygon.size();

					Vector3 v[4] = {
						Vector3(final_polygon[i].x, final_polygon[i].y, -depth),
						Vector3(final_polygon[i_n].x, final_polygon[i_n].y, -depth),
						Vector3(final_polygon[i_n].x, final_polygon[i_n].y, 0),
						Vector3(final_polygon[i].x, final_polygon[i].y, 0),
					};

					Vector2 u[4] = {
						Vector2(0, 0),
						Vector2(0, 1),
						Vector2(1, 1),
						Vector2(1, 0)
					};

					// face 1
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

					// face 2
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

			} break;
			case MODE_SPIN: {
				for (int i = 0; i < spin_sides; i++) {
					float inci = float(i) / spin_sides;
					float inci_n = float((i + 1)) / spin_sides;

					float angi = -(inci * spin_degrees / 360.0) * Math_PI * 2.0;
					float angi_n = -(inci_n * spin_degrees / 360.0) * Math_PI * 2.0;

					Vector3 normali = Vector3(Math::cos(angi), 0, Math::sin(angi));
					Vector3 normali_n = Vector3(Math::cos(angi_n), 0, Math::sin(angi_n));

					//add triangles for depth
					for (int j = 0; j < final_polygon.size(); j++) {
						int j_n = (j + 1) % final_polygon.size();

						Vector3 v[4] = {
							Vector3(normali.x * final_polygon[j].x, final_polygon[j].y, normali.z * final_polygon[j].x),
							Vector3(normali.x * final_polygon[j_n].x, final_polygon[j_n].y, normali.z * final_polygon[j_n].x),
							Vector3(normali_n.x * final_polygon[j_n].x, final_polygon[j_n].y, normali_n.z * final_polygon[j_n].x),
							Vector3(normali_n.x * final_polygon[j].x, final_polygon[j].y, normali_n.z * final_polygon[j].x),
						};

						Vector2 u[4] = {
							Vector2(0, 0),
							Vector2(0, 1),
							Vector2(1, 1),
							Vector2(1, 0)
						};

						// face 1
						facesw[face * 3 + 0] = v[0];
						facesw[face * 3 + 1] = v[2];
						facesw[face * 3 + 2] = v[1];

						uvsw[face * 3 + 0] = u[0];
						uvsw[face * 3 + 1] = u[2];
						uvsw[face * 3 + 2] = u[1];

						smoothw[face] = smooth_faces;
						invertw[face] = invert_val;
						materialsw[face] = material;

						face++;

						// face 2
						facesw[face * 3 + 0] = v[2];
						facesw[face * 3 + 1] = v[0];
						facesw[face * 3 + 2] = v[3];

						uvsw[face * 3 + 0] = u[2];
						uvsw[face * 3 + 1] = u[0];
						uvsw[face * 3 + 2] = u[3];

						smoothw[face] = smooth_faces;
						invertw[face] = invert_val;
						materialsw[face] = material;

						face++;
					}

					if (i == 0 && spin_degrees < 360) {
						for (int j = 0; j < triangles.size(); j += 3) {
							for (int k = 0; k < 3; k++) {
								int src[3] = { 0, 2, 1 };
								Vector2 p = final_polygon[triangles[j + src[k]]];
								Vector3 v = Vector3(p.x, p.y, 0);
								facesw[face * 3 + k] = v;
								uvsw[face * 3 + k] = (p - final_polygon_min) / final_polygon_size;
							}

							smoothw[face] = false;
							materialsw[face] = material;
							invertw[face] = invert_val;
							face++;
						}
					}

					if (i == spin_sides - 1 && spin_degrees < 360) {
						for (int j = 0; j < triangles.size(); j += 3) {
							for (int k = 0; k < 3; k++) {
								int src[3] = { 0, 1, 2 };
								Vector2 p = final_polygon[triangles[j + src[k]]];
								Vector3 v = Vector3(normali_n.x * p.x, p.y, normali_n.z * p.x);
								facesw[face * 3 + k] = v;
								uvsw[face * 3 + k] = (p - final_polygon_min) / final_polygon_size;
								uvsw[face * 3 + k].x = 1.0 - uvsw[face * 3 + k].x; /* flip x */
							}

							smoothw[face] = false;
							materialsw[face] = material;
							invertw[face] = invert_val;
							face++;
						}
					}
				}
			} break;
			case MODE_PATH: {
				float bl = curve->get_baked_length();
				int splits = MAX(2, Math::ceil(bl / path_interval));
				float u1 = 0.0;
				float u2 = path_continuous_u ? 0.0 : 1.0;

				Transform path_to_this;
				if (!path_local) {
					// center on paths origin
					path_to_this = get_global_transform().affine_inverse() * path->get_global_transform();
				}

				Transform prev_xf;

				Vector3 lookat_dir;

				if (path_rotation == PATH_ROTATION_POLYGON) {
					lookat_dir = (path->get_global_transform().affine_inverse() * get_global_transform()).xform(Vector3(0, 0, -1));
				} else {
					Vector3 p1, p2;
					p1 = curve->interpolate_baked(0);
					p2 = curve->interpolate_baked(0.1);
					lookat_dir = (p2 - p1).normalized();
				}

				for (int i = 0; i <= splits; i++) {
					float ofs = i * path_interval;
					if (ofs > bl) {
						ofs = bl;
					}
					if (i == splits && path_joined) {
						ofs = 0.0;
					}

					Transform xf;
					xf.origin = curve->interpolate_baked(ofs);

					Vector3 local_dir;

					if (path_rotation == PATH_ROTATION_PATH_FOLLOW && ofs > 0) {
						//before end
						Vector3 p1 = curve->interpolate_baked(ofs - 0.1);
						Vector3 p2 = curve->interpolate_baked(ofs);
						local_dir = (p2 - p1).normalized();

					} else {
						local_dir = lookat_dir;
					}

					xf = xf.looking_at(xf.origin + local_dir, Vector3(0, 1, 0));
					Basis rot(Vector3(0, 0, 1), curve->interpolate_baked_tilt(ofs));

					xf = xf * rot; //post mult

					xf = path_to_this * xf;

					if (i > 0) {
						if (path_continuous_u) {
							u1 = u2;
							u2 += (prev_xf.origin - xf.origin).length();
						};

						//put triangles where they belong
						//add triangles for depth
						for (int j = 0; j < final_polygon.size(); j++) {
							int j_n = (j + 1) % final_polygon.size();

							Vector3 v[4] = {
								prev_xf.xform(Vector3(final_polygon[j].x, final_polygon[j].y, 0)),
								prev_xf.xform(Vector3(final_polygon[j_n].x, final_polygon[j_n].y, 0)),
								xf.xform(Vector3(final_polygon[j_n].x, final_polygon[j_n].y, 0)),
								xf.xform(Vector3(final_polygon[j].x, final_polygon[j].y, 0)),
							};

							Vector2 u[4] = {
								Vector2(u1, 1),
								Vector2(u1, 0),
								Vector2(u2, 0),
								Vector2(u2, 1)
							};

							// face 1
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

							// face 2
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

					if (i == 0 && !path_joined) {
						for (int j = 0; j < triangles.size(); j += 3) {
							for (int k = 0; k < 3; k++) {
								int src[3] = { 0, 1, 2 };
								Vector2 p = final_polygon[triangles[j + src[k]]];
								Vector3 v = Vector3(p.x, p.y, 0);
								facesw[face * 3 + k] = xf.xform(v);
								uvsw[face * 3 + k] = (p - final_polygon_min) / final_polygon_size;
							}

							smoothw[face] = false;
							materialsw[face] = material;
							invertw[face] = invert_val;
							face++;
						}
					}

					if (i == splits && !path_joined) {
						for (int j = 0; j < triangles.size(); j += 3) {
							for (int k = 0; k < 3; k++) {
								int src[3] = { 0, 2, 1 };
								Vector2 p = final_polygon[triangles[j + src[k]]];
								Vector3 v = Vector3(p.x, p.y, 0);
								facesw[face * 3 + k] = xf.xform(v);
								uvsw[face * 3 + k] = (p - final_polygon_min) / final_polygon_size;
								uvsw[face * 3 + k].x = 1.0 - uvsw[face * 3 + k].x; /* flip x */
							}

							smoothw[face] = false;
							materialsw[face] = material;
							invertw[face] = invert_val;
							face++;
						}
					}

					prev_xf = xf;
				}

			} break;
		}

		if (face != face_count) {
			ERR_PRINT("Face mismatch bug! fix code");
		}
		for (int i = 0; i < face_count * 3; i++) {
			if (i == 0) {
				aabb.position = facesw[i];
			} else {
				aabb.expand_to(facesw[i]);
			}

			// invert UVs on the Y-axis OpenGL = upside down
			uvsw[i].y = 1.0 - uvsw[i].y;
		}
	}

	brush->build_from_faces(faces, uvs, smooth, materials, invert);

	return brush;
}

void CSGPolygon3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_EXIT_TREE) {
		if (path_cache) {
			path_cache->disconnect("tree_exited", callable_mp(this, &CSGPolygon3D::_path_exited));
			path_cache->disconnect("curve_changed", callable_mp(this, &CSGPolygon3D::_path_changed));
			path_cache = nullptr;
		}
	}
}

void CSGPolygon3D::_validate_property(PropertyInfo &property) const {
	if (property.name.begins_with("spin") && mode != MODE_SPIN) {
		property.usage = 0;
	}
	if (property.name.begins_with("path") && mode != MODE_PATH) {
		property.usage = 0;
	}
	if (property.name == "depth" && mode != MODE_DEPTH) {
		property.usage = 0;
	}

	CSGShape3D::_validate_property(property);
}

void CSGPolygon3D::_path_changed() {
	_make_dirty();
	update_gizmo();
}

void CSGPolygon3D::_path_exited() {
	path_cache = nullptr;
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

	ClassDB::bind_method(D_METHOD("set_path_interval", "distance"), &CSGPolygon3D::set_path_interval);
	ClassDB::bind_method(D_METHOD("get_path_interval"), &CSGPolygon3D::get_path_interval);

	ClassDB::bind_method(D_METHOD("set_path_rotation", "mode"), &CSGPolygon3D::set_path_rotation);
	ClassDB::bind_method(D_METHOD("get_path_rotation"), &CSGPolygon3D::get_path_rotation);

	ClassDB::bind_method(D_METHOD("set_path_local", "enable"), &CSGPolygon3D::set_path_local);
	ClassDB::bind_method(D_METHOD("is_path_local"), &CSGPolygon3D::is_path_local);

	ClassDB::bind_method(D_METHOD("set_path_continuous_u", "enable"), &CSGPolygon3D::set_path_continuous_u);
	ClassDB::bind_method(D_METHOD("is_path_continuous_u"), &CSGPolygon3D::is_path_continuous_u);

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
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "depth", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "spin_degrees", PROPERTY_HINT_RANGE, "1,360,0.1"), "set_spin_degrees", "get_spin_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spin_sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_spin_sides", "get_spin_sides");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "path_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Path"), "set_path_node", "get_path_node");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_interval", PROPERTY_HINT_EXP_RANGE, "0.001,1000.0,0.001,or_greater"), "set_path_interval", "get_path_interval");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_rotation", PROPERTY_HINT_ENUM, "Polygon,Path,PathFollow"), "set_path_rotation", "get_path_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_local"), "set_path_local", "is_path_local");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_continuous_u"), "set_path_continuous_u", "is_path_continuous_u");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_joined"), "set_path_joined", "is_path_joined");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "StandardMaterial3D,ShaderMaterial"), "set_material", "get_material");

	BIND_ENUM_CONSTANT(MODE_DEPTH);
	BIND_ENUM_CONSTANT(MODE_SPIN);
	BIND_ENUM_CONSTANT(MODE_PATH);

	BIND_ENUM_CONSTANT(PATH_ROTATION_POLYGON);
	BIND_ENUM_CONSTANT(PATH_ROTATION_PATH);
	BIND_ENUM_CONSTANT(PATH_ROTATION_PATH_FOLLOW);
}

void CSGPolygon3D::set_polygon(const Vector<Vector2> &p_polygon) {
	polygon = p_polygon;
	_make_dirty();
	update_gizmo();
}

Vector<Vector2> CSGPolygon3D::get_polygon() const {
	return polygon;
}

void CSGPolygon3D::set_mode(Mode p_mode) {
	mode = p_mode;
	_make_dirty();
	update_gizmo();
	_change_notify();
}

CSGPolygon3D::Mode CSGPolygon3D::get_mode() const {
	return mode;
}

void CSGPolygon3D::set_depth(const float p_depth) {
	ERR_FAIL_COND(p_depth < 0.001);
	depth = p_depth;
	_make_dirty();
	update_gizmo();
}

float CSGPolygon3D::get_depth() const {
	return depth;
}

void CSGPolygon3D::set_path_continuous_u(bool p_enable) {
	path_continuous_u = p_enable;
	_make_dirty();
}

bool CSGPolygon3D::is_path_continuous_u() const {
	return path_continuous_u;
}

void CSGPolygon3D::set_spin_degrees(const float p_spin_degrees) {
	ERR_FAIL_COND(p_spin_degrees < 0.01 || p_spin_degrees > 360);
	spin_degrees = p_spin_degrees;
	_make_dirty();
	update_gizmo();
}

float CSGPolygon3D::get_spin_degrees() const {
	return spin_degrees;
}

void CSGPolygon3D::set_spin_sides(const int p_spin_sides) {
	ERR_FAIL_COND(p_spin_sides < 3);
	spin_sides = p_spin_sides;
	_make_dirty();
	update_gizmo();
}

int CSGPolygon3D::get_spin_sides() const {
	return spin_sides;
}

void CSGPolygon3D::set_path_node(const NodePath &p_path) {
	path_node = p_path;
	_make_dirty();
	update_gizmo();
}

NodePath CSGPolygon3D::get_path_node() const {
	return path_node;
}

void CSGPolygon3D::set_path_interval(float p_interval) {
	ERR_FAIL_COND_MSG(p_interval < 0.001, "Path interval cannot be smaller than 0.001.");
	path_interval = p_interval;
	_make_dirty();
	update_gizmo();
}

float CSGPolygon3D::get_path_interval() const {
	return path_interval;
}

void CSGPolygon3D::set_path_rotation(PathRotation p_rotation) {
	path_rotation = p_rotation;
	_make_dirty();
	update_gizmo();
}

CSGPolygon3D::PathRotation CSGPolygon3D::get_path_rotation() const {
	return path_rotation;
}

void CSGPolygon3D::set_path_local(bool p_enable) {
	path_local = p_enable;
	_make_dirty();
	update_gizmo();
}

bool CSGPolygon3D::is_path_local() const {
	return path_local;
}

void CSGPolygon3D::set_path_joined(bool p_enable) {
	path_joined = p_enable;
	_make_dirty();
	update_gizmo();
}

bool CSGPolygon3D::is_path_joined() const {
	return path_joined;
}

void CSGPolygon3D::set_smooth_faces(const bool p_smooth_faces) {
	smooth_faces = p_smooth_faces;
	_make_dirty();
}

bool CSGPolygon3D::get_smooth_faces() const {
	return smooth_faces;
}

void CSGPolygon3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	_make_dirty();
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
	path_interval = 1;
	path_rotation = PATH_ROTATION_PATH;
	path_local = false;
	path_continuous_u = false;
	path_joined = false;
	path_cache = nullptr;
}
