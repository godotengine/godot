/*************************************************************************/
/*  gpu_particles_collision_3d.cpp                                       */
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

#include "gpu_particles_collision_3d.h"

#include "mesh_instance_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/main/viewport.h"

void GPUParticlesCollision3D::set_cull_mask(uint32_t p_cull_mask) {
	cull_mask = p_cull_mask;
	RS::get_singleton()->particles_collision_set_cull_mask(collision, p_cull_mask);
}

uint32_t GPUParticlesCollision3D::get_cull_mask() const {
	return cull_mask;
}

void GPUParticlesCollision3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &GPUParticlesCollision3D::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &GPUParticlesCollision3D::get_cull_mask);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");
}

GPUParticlesCollision3D::GPUParticlesCollision3D(RS::ParticlesCollisionType p_type) {
	collision = RS::get_singleton()->particles_collision_create();
	RS::get_singleton()->particles_collision_set_collision_type(collision, p_type);
	set_base(collision);
}

GPUParticlesCollision3D::~GPUParticlesCollision3D() {
	RS::get_singleton()->free(collision);
}

/////////////////////////////////

void GPUParticlesCollisionSphere3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &GPUParticlesCollisionSphere3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &GPUParticlesCollisionSphere3D::get_radius);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.01,1024,0.01,or_greater"), "set_radius", "get_radius");
}

void GPUParticlesCollisionSphere3D::set_radius(real_t p_radius) {
	radius = p_radius;
	RS::get_singleton()->particles_collision_set_sphere_radius(_get_collision(), radius);
	update_gizmos();
}

real_t GPUParticlesCollisionSphere3D::get_radius() const {
	return radius;
}

AABB GPUParticlesCollisionSphere3D::get_aabb() const {
	return AABB(Vector3(-radius, -radius, -radius), Vector3(radius * 2, radius * 2, radius * 2));
}

GPUParticlesCollisionSphere3D::GPUParticlesCollisionSphere3D() :
		GPUParticlesCollision3D(RS::PARTICLES_COLLISION_TYPE_SPHERE_COLLIDE) {
}

GPUParticlesCollisionSphere3D::~GPUParticlesCollisionSphere3D() {
}

///////////////////////////

void GPUParticlesCollisionBox3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &GPUParticlesCollisionBox3D::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &GPUParticlesCollisionBox3D::get_extents);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents", PROPERTY_HINT_RANGE, "0.01,1024,0.01,or_greater"), "set_extents", "get_extents");
}

void GPUParticlesCollisionBox3D::set_extents(const Vector3 &p_extents) {
	extents = p_extents;
	RS::get_singleton()->particles_collision_set_box_extents(_get_collision(), extents);
	update_gizmos();
}

Vector3 GPUParticlesCollisionBox3D::get_extents() const {
	return extents;
}

AABB GPUParticlesCollisionBox3D::get_aabb() const {
	return AABB(-extents, extents * 2);
}

GPUParticlesCollisionBox3D::GPUParticlesCollisionBox3D() :
		GPUParticlesCollision3D(RS::PARTICLES_COLLISION_TYPE_BOX_COLLIDE) {
}

GPUParticlesCollisionBox3D::~GPUParticlesCollisionBox3D() {
}

///////////////////////////////
///////////////////////////

void GPUParticlesCollisionSDF3D::_find_meshes(const AABB &p_aabb, Node *p_at_node, List<PlotMesh> &plot_meshes) {
	MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(p_at_node);
	if (mi && mi->is_visible_in_tree()) {
		Ref<Mesh> mesh = mi->get_mesh();
		if (mesh.is_valid()) {
			AABB aabb = mesh->get_aabb();

			Transform3D xf = get_global_transform().affine_inverse() * mi->get_global_transform();

			if (p_aabb.intersects(xf.xform(aabb))) {
				PlotMesh pm;
				pm.local_xform = xf;
				pm.mesh = mesh;
				plot_meshes.push_back(pm);
			}
		}
	}

	Node3D *s = Object::cast_to<Node3D>(p_at_node);
	if (s) {
		if (s->is_visible_in_tree()) {
			Array meshes = p_at_node->call("get_meshes");
			for (int i = 0; i < meshes.size(); i += 2) {
				Transform3D mxf = meshes[i];
				Ref<Mesh> mesh = meshes[i + 1];
				if (!mesh.is_valid()) {
					continue;
				}

				AABB aabb = mesh->get_aabb();

				Transform3D xf = get_global_transform().affine_inverse() * (s->get_global_transform() * mxf);

				if (p_aabb.intersects(xf.xform(aabb))) {
					PlotMesh pm;
					pm.local_xform = xf;
					pm.mesh = mesh;
					plot_meshes.push_back(pm);
				}
			}
		}
	}

	for (int i = 0; i < p_at_node->get_child_count(); i++) {
		Node *child = p_at_node->get_child(i);
		_find_meshes(p_aabb, child, plot_meshes);
	}
}

uint32_t GPUParticlesCollisionSDF3D::_create_bvh(LocalVector<BVH> &bvh_tree, FacePos *p_faces, uint32_t p_face_count, const Face3 *p_triangles, float p_thickness) {
	if (p_face_count == 1) {
		return BVH::LEAF_BIT | p_faces[0].index;
	}

	uint32_t index = bvh_tree.size();
	{
		BVH bvh;

		for (uint32_t i = 0; i < p_face_count; i++) {
			const Face3 &f = p_triangles[p_faces[i].index];
			AABB aabb(f.vertex[0], Vector3());
			aabb.expand_to(f.vertex[1]);
			aabb.expand_to(f.vertex[2]);
			if (p_thickness > 0.0) {
				Vector3 normal = p_triangles[p_faces[i].index].get_plane().normal;
				aabb.expand_to(f.vertex[0] - normal * p_thickness);
				aabb.expand_to(f.vertex[1] - normal * p_thickness);
				aabb.expand_to(f.vertex[2] - normal * p_thickness);
			}
			if (i == 0) {
				bvh.bounds = aabb;
			} else {
				bvh.bounds.merge_with(aabb);
			}
		}
		bvh_tree.push_back(bvh);
	}

	uint32_t middle = p_face_count / 2;

	SortArray<FacePos, FaceSort> s;
	s.compare.axis = bvh_tree[index].bounds.get_longest_axis_index();
	s.sort(p_faces, p_face_count);

	uint32_t left = _create_bvh(bvh_tree, p_faces, middle, p_triangles, p_thickness);
	uint32_t right = _create_bvh(bvh_tree, p_faces + middle, p_face_count - middle, p_triangles, p_thickness);

	bvh_tree[index].children[0] = left;
	bvh_tree[index].children[1] = right;

	return index;
}

static _FORCE_INLINE_ real_t Vector3_dot2(const Vector3 &p_vec3) {
	return p_vec3.dot(p_vec3);
}

void GPUParticlesCollisionSDF3D::_find_closest_distance(const Vector3 &p_pos, const BVH *bvh, uint32_t p_bvh_cell, const Face3 *triangles, float thickness, float &closest_distance) {
	if (p_bvh_cell & BVH::LEAF_BIT) {
		p_bvh_cell &= BVH::LEAF_MASK; //remove bit

		Vector3 point = p_pos;
		Plane p = triangles[p_bvh_cell].get_plane();
		float d = p.distance_to(point);
		float inside_d = 1e20;
		if (d < 0 && d > -thickness) {
			//inside planes, do this in 2D

			Vector3 x_axis = (triangles[p_bvh_cell].vertex[0] - triangles[p_bvh_cell].vertex[1]).normalized();
			Vector3 y_axis = p.normal.cross(x_axis).normalized();

			Vector2 points[3];
			for (int i = 0; i < 3; i++) {
				points[i] = Vector2(x_axis.dot(triangles[p_bvh_cell].vertex[i]), y_axis.dot(triangles[p_bvh_cell].vertex[i]));
			}

			Vector2 p2d = Vector2(x_axis.dot(point), y_axis.dot(point));

			{
				// https://www.shadertoy.com/view/XsXSz4

				Vector2 e0 = points[1] - points[0];
				Vector2 e1 = points[2] - points[1];
				Vector2 e2 = points[0] - points[2];

				Vector2 v0 = p2d - points[0];
				Vector2 v1 = p2d - points[1];
				Vector2 v2 = p2d - points[2];

				Vector2 pq0 = v0 - e0 * CLAMP(v0.dot(e0) / e0.dot(e0), 0.0, 1.0);
				Vector2 pq1 = v1 - e1 * CLAMP(v1.dot(e1) / e1.dot(e1), 0.0, 1.0);
				Vector2 pq2 = v2 - e2 * CLAMP(v2.dot(e2) / e2.dot(e2), 0.0, 1.0);

				float s = SIGN(e0.x * e2.y - e0.y * e2.x);
				Vector2 d2 = Vector2(pq0.dot(pq0), s * (v0.x * e0.y - v0.y * e0.x)).min(Vector2(pq1.dot(pq1), s * (v1.x * e1.y - v1.y * e1.x))).min(Vector2(pq2.dot(pq2), s * (v2.x * e2.y - v2.y * e2.x)));

				inside_d = -Math::sqrt(d2.x) * SIGN(d2.y);
			}

			//make sure distance to planes is not shorter if inside
			if (inside_d < 0) {
				inside_d = MAX(inside_d, d);
				inside_d = MAX(inside_d, -(thickness + d));
			}

			closest_distance = MIN(closest_distance, inside_d);
		} else {
			if (d < 0) {
				point -= p.normal * thickness; //flatten
			}

			// https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
			Vector3 a = triangles[p_bvh_cell].vertex[0];
			Vector3 b = triangles[p_bvh_cell].vertex[1];
			Vector3 c = triangles[p_bvh_cell].vertex[2];

			Vector3 ba = b - a;
			Vector3 pa = point - a;
			Vector3 cb = c - b;
			Vector3 pb = point - b;
			Vector3 ac = a - c;
			Vector3 pc = point - c;
			Vector3 nor = ba.cross(ac);

			inside_d = Math::sqrt(
					(SIGN(ba.cross(nor).dot(pa)) + SIGN(cb.cross(nor).dot(pb)) + SIGN(ac.cross(nor).dot(pc)) < 2.0)
							? MIN(MIN(
										  Vector3_dot2(ba * CLAMP(ba.dot(pa) / Vector3_dot2(ba), 0.0, 1.0) - pa),
										  Vector3_dot2(cb * CLAMP(cb.dot(pb) / Vector3_dot2(cb), 0.0, 1.0) - pb)),
									  Vector3_dot2(ac * CLAMP(ac.dot(pc) / Vector3_dot2(ac), 0.0, 1.0) - pc))
							: nor.dot(pa) * nor.dot(pa) / Vector3_dot2(nor));

			closest_distance = MIN(closest_distance, inside_d);
		}

	} else {
		bool pass = true;
		if (!bvh[p_bvh_cell].bounds.has_point(p_pos)) {
			//outside, find closest point
			Vector3 he = bvh[p_bvh_cell].bounds.size * 0.5;
			Vector3 center = bvh[p_bvh_cell].bounds.position + he;

			Vector3 rel = (p_pos - center).abs();
			Vector3 closest(MIN(rel.x, he.x), MIN(rel.y, he.y), MIN(rel.z, he.z));
			float d = rel.distance_to(closest);

			if (d >= closest_distance) {
				pass = false; //already closer than this aabb, discard
			}
		}

		if (pass) {
			_find_closest_distance(p_pos, bvh, bvh[p_bvh_cell].children[0], triangles, thickness, closest_distance);
			_find_closest_distance(p_pos, bvh, bvh[p_bvh_cell].children[1], triangles, thickness, closest_distance);
		}
	}
}

void GPUParticlesCollisionSDF3D::_compute_sdf_z(uint32_t p_z, ComputeSDFParams *params) {
	int32_t z_ofs = p_z * params->size.y * params->size.x;
	for (int32_t y = 0; y < params->size.y; y++) {
		int32_t y_ofs = z_ofs + y * params->size.x;
		for (int32_t x = 0; x < params->size.x; x++) {
			int32_t x_ofs = y_ofs + x;
			float &cell = params->cells[x_ofs];

			Vector3 pos = params->cell_offset + Vector3(x, y, p_z) * params->cell_size;

			cell = 1e20;

			_find_closest_distance(pos, params->bvh, 0, params->triangles, params->thickness, cell);
		}
	}
}

void GPUParticlesCollisionSDF3D::_compute_sdf(ComputeSDFParams *params) {
	ThreadWorkPool work_pool;
	work_pool.init();
	work_pool.begin_work(params->size.z, this, &GPUParticlesCollisionSDF3D::_compute_sdf_z, params);
	while (!work_pool.is_done_dispatching()) {
		OS::get_singleton()->delay_usec(10000);
		bake_step_function(work_pool.get_work_index() * 100 / params->size.z, "Baking SDF");
	}
	work_pool.end_work();
	work_pool.finish();
}

Vector3i GPUParticlesCollisionSDF3D::get_estimated_cell_size() const {
	static const int subdivs[RESOLUTION_MAX] = { 16, 32, 64, 128, 256, 512 };
	int subdiv = subdivs[get_resolution()];

	AABB aabb(-extents, extents * 2);

	float cell_size = aabb.get_longest_axis_size() / float(subdiv);

	Vector3i sdf_size = Vector3i(aabb.size / cell_size);
	sdf_size.x = MAX(1, sdf_size.x);
	sdf_size.y = MAX(1, sdf_size.y);
	sdf_size.z = MAX(1, sdf_size.z);
	return sdf_size;
}

Ref<Image> GPUParticlesCollisionSDF3D::bake() {
	static const int subdivs[RESOLUTION_MAX] = { 16, 32, 64, 128, 256, 512 };
	int subdiv = subdivs[get_resolution()];

	AABB aabb(-extents, extents * 2);

	float cell_size = aabb.get_longest_axis_size() / float(subdiv);

	Vector3i sdf_size = Vector3i(aabb.size / cell_size);
	sdf_size.x = MAX(1, sdf_size.x);
	sdf_size.y = MAX(1, sdf_size.y);
	sdf_size.z = MAX(1, sdf_size.z);

	if (bake_begin_function) {
		bake_begin_function(100);
	}

	aabb.size = Vector3(sdf_size) * cell_size;

	List<PlotMesh> plot_meshes;
	_find_meshes(aabb, get_parent(), plot_meshes);

	LocalVector<Face3> faces;

	if (bake_step_function) {
		bake_step_function(0, "Finding Meshes");
	}

	for (const PlotMesh &pm : plot_meshes) {
		for (int i = 0; i < pm.mesh->get_surface_count(); i++) {
			if (pm.mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
				continue; //only triangles
			}

			Array a = pm.mesh->surface_get_arrays(i);

			Vector<Vector3> vertices = a[Mesh::ARRAY_VERTEX];
			const Vector3 *vr = vertices.ptr();
			Vector<int> index = a[Mesh::ARRAY_INDEX];

			if (index.size()) {
				int facecount = index.size() / 3;
				const int *ir = index.ptr();

				for (int j = 0; j < facecount; j++) {
					Face3 face;

					for (int k = 0; k < 3; k++) {
						face.vertex[k] = pm.local_xform.xform(vr[ir[j * 3 + k]]);
					}

					//test against original bounds
					if (!Geometry3D::triangle_box_overlap(aabb.get_center(), aabb.size * 0.5, face.vertex)) {
						continue;
					}

					faces.push_back(face);
				}

			} else {
				int facecount = vertices.size() / 3;

				for (int j = 0; j < facecount; j++) {
					Face3 face;

					for (int k = 0; k < 3; k++) {
						face.vertex[k] = pm.local_xform.xform(vr[j * 3 + k]);
					}

					//test against original bounds
					if (!Geometry3D::triangle_box_overlap(aabb.get_center(), aabb.size * 0.5, face.vertex)) {
						continue;
					}

					faces.push_back(face);
				}
			}
		}
	}

	//compute bvh

	ERR_FAIL_COND_V(faces.size() <= 1, Ref<Image>());

	LocalVector<FacePos> face_pos;

	face_pos.resize(faces.size());

	float th = cell_size * thickness;

	for (uint32_t i = 0; i < faces.size(); i++) {
		face_pos[i].index = i;
		face_pos[i].center = (faces[i].vertex[0] + faces[i].vertex[1] + faces[i].vertex[2]) / 2;
		if (th > 0.0) {
			face_pos[i].center -= faces[i].get_plane().normal * th * 0.5;
		}
	}

	if (bake_step_function) {
		bake_step_function(0, "Creating BVH");
	}

	LocalVector<BVH> bvh;

	_create_bvh(bvh, face_pos.ptr(), face_pos.size(), faces.ptr(), th);

	Vector<uint8_t> data;
	data.resize(sdf_size.z * sdf_size.y * sdf_size.x * (int)sizeof(float));

	if (bake_step_function) {
		bake_step_function(0, "Baking SDF");
	}

	ComputeSDFParams params;
	params.cells = (float *)data.ptrw();
	params.size = sdf_size;
	params.cell_size = cell_size;
	params.cell_offset = aabb.position + Vector3(cell_size * 0.5, cell_size * 0.5, cell_size * 0.5);
	params.bvh = bvh.ptr();
	params.triangles = faces.ptr();
	params.thickness = th;
	_compute_sdf(&params);

	Ref<Image> ret;
	ret.instantiate();
	ret->create(sdf_size.x, sdf_size.y * sdf_size.z, false, Image::FORMAT_RF, data);
	ret->convert(Image::FORMAT_RH); //convert to half, save space
	ret->set_meta("depth", sdf_size.z); //hack, make sure to add to the docs of this function

	if (bake_end_function) {
		bake_end_function();
	}

	return ret;
}

void GPUParticlesCollisionSDF3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &GPUParticlesCollisionSDF3D::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &GPUParticlesCollisionSDF3D::get_extents);

	ClassDB::bind_method(D_METHOD("set_resolution", "resolution"), &GPUParticlesCollisionSDF3D::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution"), &GPUParticlesCollisionSDF3D::get_resolution);

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &GPUParticlesCollisionSDF3D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &GPUParticlesCollisionSDF3D::get_texture);

	ClassDB::bind_method(D_METHOD("set_thickness", "thickness"), &GPUParticlesCollisionSDF3D::set_thickness);
	ClassDB::bind_method(D_METHOD("get_thickness"), &GPUParticlesCollisionSDF3D::get_thickness);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents", PROPERTY_HINT_RANGE, "0.01,1024,0.01,or_greater"), "set_extents", "get_extents");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "resolution", PROPERTY_HINT_ENUM, "16,32,64,128,256,512"), "set_resolution", "get_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "thickness", PROPERTY_HINT_RANGE, "0.0,2.0,0.01"), "set_thickness", "get_thickness");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture3D"), "set_texture", "get_texture");

	BIND_ENUM_CONSTANT(RESOLUTION_16);
	BIND_ENUM_CONSTANT(RESOLUTION_32);
	BIND_ENUM_CONSTANT(RESOLUTION_64);
	BIND_ENUM_CONSTANT(RESOLUTION_128);
	BIND_ENUM_CONSTANT(RESOLUTION_256);
	BIND_ENUM_CONSTANT(RESOLUTION_512);
	BIND_ENUM_CONSTANT(RESOLUTION_MAX);
}

void GPUParticlesCollisionSDF3D::set_thickness(float p_thickness) {
	thickness = p_thickness;
}

float GPUParticlesCollisionSDF3D::get_thickness() const {
	return thickness;
}

void GPUParticlesCollisionSDF3D::set_extents(const Vector3 &p_extents) {
	extents = p_extents;
	RS::get_singleton()->particles_collision_set_box_extents(_get_collision(), extents);
	update_gizmos();
}

Vector3 GPUParticlesCollisionSDF3D::get_extents() const {
	return extents;
}

void GPUParticlesCollisionSDF3D::set_resolution(Resolution p_resolution) {
	resolution = p_resolution;
	update_gizmos();
}

GPUParticlesCollisionSDF3D::Resolution GPUParticlesCollisionSDF3D::get_resolution() const {
	return resolution;
}

void GPUParticlesCollisionSDF3D::set_texture(const Ref<Texture3D> &p_texture) {
	texture = p_texture;
	RID tex = texture.is_valid() ? texture->get_rid() : RID();
	RS::get_singleton()->particles_collision_set_field_texture(_get_collision(), tex);
}

Ref<Texture3D> GPUParticlesCollisionSDF3D::get_texture() const {
	return texture;
}

AABB GPUParticlesCollisionSDF3D::get_aabb() const {
	return AABB(-extents, extents * 2);
}

GPUParticlesCollisionSDF3D::BakeBeginFunc GPUParticlesCollisionSDF3D::bake_begin_function = nullptr;
GPUParticlesCollisionSDF3D::BakeStepFunc GPUParticlesCollisionSDF3D::bake_step_function = nullptr;
GPUParticlesCollisionSDF3D::BakeEndFunc GPUParticlesCollisionSDF3D::bake_end_function = nullptr;

GPUParticlesCollisionSDF3D::GPUParticlesCollisionSDF3D() :
		GPUParticlesCollision3D(RS::PARTICLES_COLLISION_TYPE_SDF_COLLIDE) {
}

GPUParticlesCollisionSDF3D::~GPUParticlesCollisionSDF3D() {
}

////////////////////////////
////////////////////////////

void GPUParticlesCollisionHeightField3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_INTERNAL_PROCESS) {
		if (update_mode == UPDATE_MODE_ALWAYS) {
			RS::get_singleton()->particles_collision_height_field_update(_get_collision());
		}

		if (follow_camera_mode && get_viewport()) {
			Camera3D *cam = get_viewport()->get_camera_3d();
			if (cam) {
				Transform3D xform = get_global_transform();
				Vector3 x_axis = xform.basis.get_axis(Vector3::AXIS_X).normalized();
				Vector3 z_axis = xform.basis.get_axis(Vector3::AXIS_Z).normalized();
				float x_len = xform.basis.get_scale().x;
				float z_len = xform.basis.get_scale().z;

				Vector3 cam_pos = cam->get_global_transform().origin;
				Transform3D new_xform = xform;

				while (x_axis.dot(cam_pos - new_xform.origin) > x_len) {
					new_xform.origin += x_axis * x_len;
				}
				while (x_axis.dot(cam_pos - new_xform.origin) < -x_len) {
					new_xform.origin -= x_axis * x_len;
				}

				while (z_axis.dot(cam_pos - new_xform.origin) > z_len) {
					new_xform.origin += z_axis * z_len;
				}
				while (z_axis.dot(cam_pos - new_xform.origin) < -z_len) {
					new_xform.origin -= z_axis * z_len;
				}

				if (new_xform != xform) {
					set_global_transform(new_xform);
					RS::get_singleton()->particles_collision_height_field_update(_get_collision());
				}
			}
		}
	}

	if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {
		RS::get_singleton()->particles_collision_height_field_update(_get_collision());
	}
}

void GPUParticlesCollisionHeightField3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &GPUParticlesCollisionHeightField3D::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &GPUParticlesCollisionHeightField3D::get_extents);

	ClassDB::bind_method(D_METHOD("set_resolution", "resolution"), &GPUParticlesCollisionHeightField3D::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution"), &GPUParticlesCollisionHeightField3D::get_resolution);

	ClassDB::bind_method(D_METHOD("set_update_mode", "update_mode"), &GPUParticlesCollisionHeightField3D::set_update_mode);
	ClassDB::bind_method(D_METHOD("get_update_mode"), &GPUParticlesCollisionHeightField3D::get_update_mode);

	ClassDB::bind_method(D_METHOD("set_follow_camera_mode", "enabled"), &GPUParticlesCollisionHeightField3D::set_follow_camera_mode);
	ClassDB::bind_method(D_METHOD("is_follow_camera_mode_enabled"), &GPUParticlesCollisionHeightField3D::is_follow_camera_mode_enabled);

	ClassDB::bind_method(D_METHOD("set_follow_camera_push_ratio", "ratio"), &GPUParticlesCollisionHeightField3D::set_follow_camera_push_ratio);
	ClassDB::bind_method(D_METHOD("get_follow_camera_push_ratio"), &GPUParticlesCollisionHeightField3D::get_follow_camera_push_ratio);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents", PROPERTY_HINT_RANGE, "0.01,1024,0.01,or_greater"), "set_extents", "get_extents");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "resolution", PROPERTY_HINT_ENUM, "256,512,1024,2048,4096,8192"), "set_resolution", "get_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "update_mode", PROPERTY_HINT_ENUM, "WhenMoved,Always"), "set_update_mode", "get_update_mode");
	ADD_GROUP("Follow Camera", "follow_camera_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "follow_camera_enabled"), "set_follow_camera_mode", "is_follow_camera_mode_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "follow_camera_push_ratio", PROPERTY_HINT_RANGE, "0.01,1,0.01"), "set_follow_camera_push_ratio", "get_follow_camera_push_ratio");

	BIND_ENUM_CONSTANT(RESOLUTION_256);
	BIND_ENUM_CONSTANT(RESOLUTION_512);
	BIND_ENUM_CONSTANT(RESOLUTION_1024);
	BIND_ENUM_CONSTANT(RESOLUTION_2048);
	BIND_ENUM_CONSTANT(RESOLUTION_4096);
	BIND_ENUM_CONSTANT(RESOLUTION_8192);
	BIND_ENUM_CONSTANT(RESOLUTION_MAX);

	BIND_ENUM_CONSTANT(UPDATE_MODE_WHEN_MOVED);
	BIND_ENUM_CONSTANT(UPDATE_MODE_ALWAYS);
}

void GPUParticlesCollisionHeightField3D::set_follow_camera_push_ratio(float p_follow_camera_push_ratio) {
	follow_camera_push_ratio = p_follow_camera_push_ratio;
}

float GPUParticlesCollisionHeightField3D::get_follow_camera_push_ratio() const {
	return follow_camera_push_ratio;
}

void GPUParticlesCollisionHeightField3D::set_extents(const Vector3 &p_extents) {
	extents = p_extents;
	RS::get_singleton()->particles_collision_set_box_extents(_get_collision(), extents);
	update_gizmos();
	RS::get_singleton()->particles_collision_height_field_update(_get_collision());
}

Vector3 GPUParticlesCollisionHeightField3D::get_extents() const {
	return extents;
}

void GPUParticlesCollisionHeightField3D::set_resolution(Resolution p_resolution) {
	resolution = p_resolution;
	RS::get_singleton()->particles_collision_set_height_field_resolution(_get_collision(), RS::ParticlesCollisionHeightfieldResolution(resolution));
	update_gizmos();
	RS::get_singleton()->particles_collision_height_field_update(_get_collision());
}

GPUParticlesCollisionHeightField3D::Resolution GPUParticlesCollisionHeightField3D::get_resolution() const {
	return resolution;
}

void GPUParticlesCollisionHeightField3D::set_update_mode(UpdateMode p_update_mode) {
	update_mode = p_update_mode;
	set_process_internal(follow_camera_mode || update_mode == UPDATE_MODE_ALWAYS);
}

GPUParticlesCollisionHeightField3D::UpdateMode GPUParticlesCollisionHeightField3D::get_update_mode() const {
	return update_mode;
}

void GPUParticlesCollisionHeightField3D::set_follow_camera_mode(bool p_enabled) {
	follow_camera_mode = p_enabled;
	set_process_internal(follow_camera_mode || update_mode == UPDATE_MODE_ALWAYS);
}

bool GPUParticlesCollisionHeightField3D::is_follow_camera_mode_enabled() const {
	return follow_camera_mode;
}

AABB GPUParticlesCollisionHeightField3D::get_aabb() const {
	return AABB(-extents, extents * 2);
}

GPUParticlesCollisionHeightField3D::GPUParticlesCollisionHeightField3D() :
		GPUParticlesCollision3D(RS::PARTICLES_COLLISION_TYPE_HEIGHTFIELD_COLLIDE) {
}

GPUParticlesCollisionHeightField3D::~GPUParticlesCollisionHeightField3D() {
}

////////////////////////////
////////////////////////////

void GPUParticlesAttractor3D::set_cull_mask(uint32_t p_cull_mask) {
	cull_mask = p_cull_mask;
	RS::get_singleton()->particles_collision_set_cull_mask(collision, p_cull_mask);
}

uint32_t GPUParticlesAttractor3D::get_cull_mask() const {
	return cull_mask;
}

void GPUParticlesAttractor3D::set_strength(real_t p_strength) {
	strength = p_strength;
	RS::get_singleton()->particles_collision_set_attractor_strength(collision, p_strength);
}

real_t GPUParticlesAttractor3D::get_strength() const {
	return strength;
}

void GPUParticlesAttractor3D::set_attenuation(real_t p_attenuation) {
	attenuation = p_attenuation;
	RS::get_singleton()->particles_collision_set_attractor_attenuation(collision, p_attenuation);
}

real_t GPUParticlesAttractor3D::get_attenuation() const {
	return attenuation;
}

void GPUParticlesAttractor3D::set_directionality(real_t p_directionality) {
	directionality = p_directionality;
	RS::get_singleton()->particles_collision_set_attractor_directionality(collision, p_directionality);
	update_gizmos();
}

real_t GPUParticlesAttractor3D::get_directionality() const {
	return directionality;
}

void GPUParticlesAttractor3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &GPUParticlesAttractor3D::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &GPUParticlesAttractor3D::get_cull_mask);

	ClassDB::bind_method(D_METHOD("set_strength", "strength"), &GPUParticlesAttractor3D::set_strength);
	ClassDB::bind_method(D_METHOD("get_strength"), &GPUParticlesAttractor3D::get_strength);

	ClassDB::bind_method(D_METHOD("set_attenuation", "attenuation"), &GPUParticlesAttractor3D::set_attenuation);
	ClassDB::bind_method(D_METHOD("get_attenuation"), &GPUParticlesAttractor3D::get_attenuation);

	ClassDB::bind_method(D_METHOD("set_directionality", "amount"), &GPUParticlesAttractor3D::set_directionality);
	ClassDB::bind_method(D_METHOD("get_directionality"), &GPUParticlesAttractor3D::get_directionality);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "strength", PROPERTY_HINT_RANGE, "-128,128,0.01,or_greater,or_lesser"), "set_strength", "get_strength");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attenuation", PROPERTY_HINT_EXP_EASING, "0,8,0.01"), "set_attenuation", "get_attenuation");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "directionality", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_directionality", "get_directionality");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");
}

GPUParticlesAttractor3D::GPUParticlesAttractor3D(RS::ParticlesCollisionType p_type) {
	collision = RS::get_singleton()->particles_collision_create();
	RS::get_singleton()->particles_collision_set_collision_type(collision, p_type);
	set_base(collision);
}
GPUParticlesAttractor3D::~GPUParticlesAttractor3D() {
	RS::get_singleton()->free(collision);
}

/////////////////////////////////

void GPUParticlesAttractorSphere3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &GPUParticlesAttractorSphere3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &GPUParticlesAttractorSphere3D::get_radius);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.01,1024,0.01,or_greater"), "set_radius", "get_radius");
}

void GPUParticlesAttractorSphere3D::set_radius(real_t p_radius) {
	radius = p_radius;
	RS::get_singleton()->particles_collision_set_sphere_radius(_get_collision(), radius);
	update_gizmos();
}

real_t GPUParticlesAttractorSphere3D::get_radius() const {
	return radius;
}

AABB GPUParticlesAttractorSphere3D::get_aabb() const {
	return AABB(Vector3(-radius, -radius, -radius), Vector3(radius * 2, radius * 2, radius * 2));
}

GPUParticlesAttractorSphere3D::GPUParticlesAttractorSphere3D() :
		GPUParticlesAttractor3D(RS::PARTICLES_COLLISION_TYPE_SPHERE_ATTRACT) {
}

GPUParticlesAttractorSphere3D::~GPUParticlesAttractorSphere3D() {
}

///////////////////////////

void GPUParticlesAttractorBox3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &GPUParticlesAttractorBox3D::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &GPUParticlesAttractorBox3D::get_extents);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents", PROPERTY_HINT_RANGE, "0.01,1024,0.01,or_greater"), "set_extents", "get_extents");
}

void GPUParticlesAttractorBox3D::set_extents(const Vector3 &p_extents) {
	extents = p_extents;
	RS::get_singleton()->particles_collision_set_box_extents(_get_collision(), extents);
	update_gizmos();
}

Vector3 GPUParticlesAttractorBox3D::get_extents() const {
	return extents;
}

AABB GPUParticlesAttractorBox3D::get_aabb() const {
	return AABB(-extents, extents * 2);
}

GPUParticlesAttractorBox3D::GPUParticlesAttractorBox3D() :
		GPUParticlesAttractor3D(RS::PARTICLES_COLLISION_TYPE_BOX_ATTRACT) {
}

GPUParticlesAttractorBox3D::~GPUParticlesAttractorBox3D() {
}

///////////////////////////

void GPUParticlesAttractorVectorField3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &GPUParticlesAttractorVectorField3D::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &GPUParticlesAttractorVectorField3D::get_extents);

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &GPUParticlesAttractorVectorField3D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &GPUParticlesAttractorVectorField3D::get_texture);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents", PROPERTY_HINT_RANGE, "0.01,1024,0.01,or_greater"), "set_extents", "get_extents");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture3D"), "set_texture", "get_texture");
}

void GPUParticlesAttractorVectorField3D::set_extents(const Vector3 &p_extents) {
	extents = p_extents;
	RS::get_singleton()->particles_collision_set_box_extents(_get_collision(), extents);
	update_gizmos();
}

Vector3 GPUParticlesAttractorVectorField3D::get_extents() const {
	return extents;
}

void GPUParticlesAttractorVectorField3D::set_texture(const Ref<Texture3D> &p_texture) {
	texture = p_texture;
	RID tex = texture.is_valid() ? texture->get_rid() : RID();
	RS::get_singleton()->particles_collision_set_field_texture(_get_collision(), tex);
}

Ref<Texture3D> GPUParticlesAttractorVectorField3D::get_texture() const {
	return texture;
}

AABB GPUParticlesAttractorVectorField3D::get_aabb() const {
	return AABB(-extents, extents * 2);
}

GPUParticlesAttractorVectorField3D::GPUParticlesAttractorVectorField3D() :
		GPUParticlesAttractor3D(RS::PARTICLES_COLLISION_TYPE_VECTOR_FIELD_ATTRACT) {
}

GPUParticlesAttractorVectorField3D::~GPUParticlesAttractorVectorField3D() {
}
