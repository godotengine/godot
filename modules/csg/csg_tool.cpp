/*************************************************************************/
/*  csg_tool.cpp                                                         */
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

#include "csg_tool.h"

#include "scene/3d/path_3d.h"

// CSGPrimitive3D
void CSGPrimitiveShape3D::_create_brush_from_arrays(const Vector<Vector3> &p_vertices, const Vector<Vector2> &p_uv, const Vector<bool> &p_smooth, const Vector<Ref<Material>> &p_materials) {
	brush = CSGBrush();

	Vector<bool> invert;
	invert.resize(p_vertices.size() / 3);
	{
		int ic = invert.size();
		bool *w = invert.ptrw();
		for (int i = 0; i < ic; i++) {
			w[i] = invert_faces;
		}
	}
	brush.build_from_faces(p_vertices, p_uv, p_smooth, p_materials, invert);
}

CSGBrush CSGPrimitiveShape3D::get_brush() {
	_update_brush();
	return brush;
}

void CSGPrimitiveShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_invert_faces", "invert_faces"), &CSGPrimitiveShape3D::set_invert_faces);
	ClassDB::bind_method(D_METHOD("is_inverting_faces"), &CSGPrimitiveShape3D::is_inverting_faces);
	ClassDB::bind_method(D_METHOD("set_smooth_faces", "smooth_faces"), &CSGPrimitiveShape3D::set_smooth_faces);
	ClassDB::bind_method(D_METHOD("get_smooth_faces"), &CSGPrimitiveShape3D::get_smooth_faces);
	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGPrimitiveShape3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGPrimitiveShape3D::get_material);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert_faces"), "set_invert_faces", "is_inverting_faces");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_faces"), "set_smooth_faces", "get_smooth_faces");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "StandardMaterial3D,ShaderMaterial"), "set_material", "get_material");
}

bool CSGPrimitiveShape3D::is_inverting_faces() const {
	return invert_faces;
}

void CSGPrimitiveShape3D::set_invert_faces(bool p_invert) {
	if (invert_faces == p_invert) {
		return;
	}
	dirty = true;
	emit_changed();
	invert_faces = p_invert;
}

bool CSGPrimitiveShape3D::get_smooth_faces() const {
	return smooth_faces;
}

void CSGPrimitiveShape3D::set_smooth_faces(bool p_smooth_faces) {
	if (smooth_faces == p_smooth_faces) {
		return;
	}
	dirty = true;
	emit_changed();
	smooth_faces = p_smooth_faces;
}

Ref<Material> CSGPrimitiveShape3D::get_material() const {
	return material;
}

void CSGPrimitiveShape3D::set_material(const Ref<Material> &p_material) {
	if (material == p_material) {
		return;
	}
	dirty = true;
	emit_changed();
	material = p_material;
}

// CSGMesh3D
void CSGMeshShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &CSGMeshShape3D::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &CSGMeshShape3D::get_mesh);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
}

Ref<Mesh> CSGMeshShape3D::get_mesh() const {
	return mesh;
}

void CSGMeshShape3D::set_mesh(const Ref<Mesh> &p_mesh) {
	if (mesh == p_mesh) {
		return;
	}
	dirty = true;
	emit_changed();
	mesh = p_mesh;
}

void CSGMeshShape3D::_update_brush() {
	if (dirty) {
		if (!mesh.is_valid()) {
			return;
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
				dirty = true;
				ERR_FAIL_COND(arrays.size() == 0);
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

					bool flat = normal[0].is_equal_approx(normal[1]) && normal[0].is_equal_approx(normal[2]);

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

					bool flat = normal[0].is_equal_approx(normal[1]) && normal[0].is_equal_approx(normal[2]);

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
			return;
		}

		_create_brush_from_arrays(vertices, uvs, smooth, materials);

		dirty = false;
	}
}

// CSGBox3D
void CSGBoxShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &CSGBoxShape3D::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &CSGBoxShape3D::get_size);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size"), "set_size", "get_size");
}

Vector3 CSGBoxShape3D::get_size() const {
	return size;
}

void CSGBoxShape3D::set_size(const Vector3 &p_size) {
	if (size == p_size) {
		return;
	}
	dirty = true;
	emit_changed();
	size = p_size;
}

void CSGBoxShape3D::_update_brush() {
	if (dirty) {
		brush = CSGBrush();

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

			Vector3 vertex_mul = size / 2;

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
					//face 2
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

		brush.build_from_faces(faces, uvs, smooth, materials, invert);
		dirty = false;
	}
}

// CSGCylinder3D
void CSGCylinderShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CSGCylinderShape3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CSGCylinderShape3D::get_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGCylinderShape3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGCylinderShape3D::get_height);

	ClassDB::bind_method(D_METHOD("set_sides", "sides"), &CSGCylinderShape3D::set_sides);
	ClassDB::bind_method(D_METHOD("get_sides"), &CSGCylinderShape3D::get_sides);
	ClassDB::bind_method(D_METHOD("set_cone", "cone"), &CSGCylinderShape3D::set_cone);
	ClassDB::bind_method(D_METHOD("is_cone"), &CSGCylinderShape3D::is_cone);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,1000.0,0.001,or_greater,exp"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,1000.0,0.001,or_greater,exp"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_sides", "get_sides");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cone"), "set_cone", "is_cone");
}

float CSGCylinderShape3D::get_radius() const {
	return radius;
}

void CSGCylinderShape3D::set_radius(float p_radius) {
	ERR_FAIL_COND(p_radius <= 0);
	if (radius == p_radius) {
		return;
	}
	dirty = true;
	emit_changed();
	radius = p_radius;
}

float CSGCylinderShape3D::get_height() const {
	return height;
}

void CSGCylinderShape3D::set_height(float p_height) {
	if (height == p_height) {
		return;
	}
	dirty = true;
	emit_changed();
	height = p_height;
}

int CSGCylinderShape3D::get_sides() const {
	return sides;
}

void CSGCylinderShape3D::set_sides(int p_sides) {
	ERR_FAIL_COND(p_sides < 3);
	if (sides == p_sides) {
		return;
	}
	dirty = true;
	emit_changed();
	sides = p_sides;
}

bool CSGCylinderShape3D::is_cone() const {
	return cone;
}

void CSGCylinderShape3D::set_cone(bool p_cone) {
	if (cone == p_cone) {
		return;
	}
	dirty = true;
	emit_changed();
	cone = p_cone;
}

void CSGCylinderShape3D::_update_brush() {
	if (dirty) {
		brush = CSGBrush();

		// set our bounding box

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

					float ang = inc * Math_TAU;
					float ang_n = inc_n * Math_TAU;

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

					smoothw[face] = get_smooth_faces();
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

						smoothw[face] = get_smooth_faces();
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

		brush.build_from_faces(faces, uvs, smooth, materials, invert);
		dirty = false;
	}
}

// CSGSphere3D
void CSGSphereShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CSGSphereShape3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CSGSphereShape3D::get_radius);
	ClassDB::bind_method(D_METHOD("set_radial_segments", "radial_segments"), &CSGSphereShape3D::set_radial_segments);
	ClassDB::bind_method(D_METHOD("get_radial_segments"), &CSGSphereShape3D::get_radial_segments);
	ClassDB::bind_method(D_METHOD("set_rings", "rings"), &CSGSphereShape3D::set_rings);
	ClassDB::bind_method(D_METHOD("get_rings"), &CSGSphereShape3D::get_rings);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "1,100,1"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "1,100,1"), "set_rings", "get_rings");
}

float CSGSphereShape3D::get_radius() const {
	return radius;
}

void CSGSphereShape3D::set_radius(float p_radius) {
	if (radius == p_radius) {
		return;
	}
	dirty = true;
	emit_changed();
	radius = p_radius;
}

int CSGSphereShape3D::get_radial_segments() const {
	return radial_segments;
}

void CSGSphereShape3D::set_radial_segments(int p_radial_segments) {
	p_radial_segments = p_radial_segments > 4 ? p_radial_segments : 4;
	if (radial_segments == p_radial_segments) {
		return;
	}

	dirty = true;
	emit_changed();
	radial_segments = p_radial_segments;
}

int CSGSphereShape3D::get_rings() const {
	return rings;
}

void CSGSphereShape3D::set_rings(int p_rings) {
	p_rings = p_rings > 1 ? p_rings : 1;
	if (rings == p_rings) {
		return;
	}
	dirty = true;
	emit_changed();
	rings = p_rings;
}

void CSGSphereShape3D::_update_brush() {
	// set our bounding box
	if (dirty) {
		brush = CSGBrush();

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
			const double lat_step = Math_TAU / rings;
			const double lon_step = Math_TAU / radial_segments;

			for (int i = 1; i <= rings; i++) {
				double lat0 = lat_step * (i - 1) - Math_TAU / 4;
				double z0 = Math::sin(lat0);
				double zr0 = Math::cos(lat0);
				double u0 = double(i - 1) / rings;

				double lat1 = lat_step * i - Math_TAU / 4;
				double z1 = Math::sin(lat1);
				double zr1 = Math::cos(lat1);
				double u1 = double(i) / rings;

				for (int j = radial_segments; j >= 1; j--) {
					double lng0 = lon_step * (j - 1);
					double x0 = Math::cos(lng0);
					double y0 = Math::sin(lng0);
					double v0 = double(i - 1) / radial_segments;

					double lng1 = lon_step * j;
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

						smoothw[face] = get_smooth_faces();
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

						smoothw[face] = get_smooth_faces();
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

		brush.build_from_faces(faces, uvs, smooth, materials, invert);
	}
}

// CSGTorus3D
void CSGTorusShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_inner_radius", "radius"), &CSGTorusShape3D::set_inner_radius);
	ClassDB::bind_method(D_METHOD("get_inner_radius"), &CSGTorusShape3D::get_inner_radius);
	ClassDB::bind_method(D_METHOD("set_outer_radius", "radius"), &CSGTorusShape3D::set_outer_radius);
	ClassDB::bind_method(D_METHOD("get_outer_radius"), &CSGTorusShape3D::get_outer_radius);

	ClassDB::bind_method(D_METHOD("set_sides", "sides"), &CSGTorusShape3D::set_sides);
	ClassDB::bind_method(D_METHOD("get_sides"), &CSGTorusShape3D::get_sides);
	ClassDB::bind_method(D_METHOD("set_ring_sides", "sides"), &CSGTorusShape3D::set_ring_sides);
	ClassDB::bind_method(D_METHOD("get_ring_sides"), &CSGTorusShape3D::get_ring_sides);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inner_radius", PROPERTY_HINT_RANGE, "0.001,1000.0,0.001,or_greater,exp"), "set_inner_radius", "get_inner_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "outer_radius", PROPERTY_HINT_RANGE, "0.001,1000.0,0.001,or_greater,exp"), "set_outer_radius", "get_outer_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_sides", "get_sides");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ring_sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_ring_sides", "get_ring_sides");
}

float CSGTorusShape3D::get_inner_radius() const {
	return inner_radius;
}

void CSGTorusShape3D::set_inner_radius(float p_inner_radius) {
	if (inner_radius == p_inner_radius) {
		return;
	}
	dirty = true;
	emit_changed();
	inner_radius = p_inner_radius;
}

float CSGTorusShape3D::get_outer_radius() const {
	return outer_radius;
}

void CSGTorusShape3D::set_outer_radius(float p_outer_radius) {
	if (outer_radius == p_outer_radius) {
		return;
	}

	dirty = true;
	emit_changed();
	outer_radius = p_outer_radius;
}

int CSGTorusShape3D::get_sides() const {
	return sides;
}

void CSGTorusShape3D::set_sides(int p_sides) {
	ERR_FAIL_COND(p_sides < 3);
	if (sides == p_sides) {
		return;
	}
	dirty = true;
	emit_changed();
	sides = p_sides;
}

int CSGTorusShape3D::get_ring_sides() const {
	return ring_sides;
}

void CSGTorusShape3D::set_ring_sides(int p_ring_sides) {
	ERR_FAIL_COND(p_ring_sides < 3);
	if (ring_sides == p_ring_sides) {
		return;
	}
	dirty = true;
	emit_changed();
	ring_sides = p_ring_sides;
}

void CSGTorusShape3D::_update_brush() {
	if (dirty) {
		brush = CSGBrush();

		// set our bounding box

		float min_radius = inner_radius;
		float max_radius = outer_radius;

		ERR_FAIL_COND_MSG(min_radius == max_radius, "Condition \"inner_radius == outer_radius\" is true");

		if (min_radius > max_radius) {
			SWAP(min_radius, max_radius);
		}

		float radius = (max_radius - min_radius) * 0.5;

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

					float angi = inci * Math_TAU;
					float angi_n = inci_n * Math_TAU;

					Vector3 normali = Vector3(Math::cos(angi), 0, Math::sin(angi));
					Vector3 normali_n = Vector3(Math::cos(angi_n), 0, Math::sin(angi_n));

					for (int j = 0; j < ring_sides; j++) {
						float incj = float(j) / ring_sides;
						float incj_n = float((j + 1)) / ring_sides;

						float angj = incj * Math_TAU;
						float angj_n = incj_n * Math_TAU;

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

						smoothw[face] = get_smooth_faces();
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

						smoothw[face] = get_smooth_faces();
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

		brush.build_from_faces(faces, uvs, smooth, materials, invert);
		dirty = false;
	}
}

// CSGPolygonShape3D
void CSGPolygonShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_polygon", "polygon"), &CSGPolygonShape3D::set_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon"), &CSGPolygonShape3D::get_polygon);

	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &CSGPolygonShape3D::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &CSGPolygonShape3D::get_mode);

	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &CSGPolygonShape3D::set_depth);
	ClassDB::bind_method(D_METHOD("get_depth"), &CSGPolygonShape3D::get_depth);

	ClassDB::bind_method(D_METHOD("set_spin_degrees", "degrees"), &CSGPolygonShape3D::set_spin_degrees);
	ClassDB::bind_method(D_METHOD("get_spin_degrees"), &CSGPolygonShape3D::get_spin_degrees);

	ClassDB::bind_method(D_METHOD("set_spin_sides", "spin_sides"), &CSGPolygonShape3D::set_spin_sides);
	ClassDB::bind_method(D_METHOD("get_spin_sides"), &CSGPolygonShape3D::get_spin_sides);

	ClassDB::bind_method(D_METHOD("set_path_curve", "curve"), &CSGPolygonShape3D::set_path_curve);
	ClassDB::bind_method(D_METHOD("get_path_curve"), &CSGPolygonShape3D::get_path_curve);

	ClassDB::bind_method(D_METHOD("set_path_transform", "xform"), &CSGPolygonShape3D::set_path_transform);
	ClassDB::bind_method(D_METHOD("get_path_transform"), &CSGPolygonShape3D::get_path_transform);

	ClassDB::bind_method(D_METHOD("set_path_interval", "distance"), &CSGPolygonShape3D::set_path_interval);
	ClassDB::bind_method(D_METHOD("get_path_interval"), &CSGPolygonShape3D::get_path_interval);

	ClassDB::bind_method(D_METHOD("set_path_rotation", "mode"), &CSGPolygonShape3D::set_path_rotation);
	ClassDB::bind_method(D_METHOD("get_path_rotation"), &CSGPolygonShape3D::get_path_rotation);

	ClassDB::bind_method(D_METHOD("set_path_continuous_u", "enable"), &CSGPolygonShape3D::set_path_continuous_u);
	ClassDB::bind_method(D_METHOD("is_path_continuous_u"), &CSGPolygonShape3D::is_path_continuous_u);

	ClassDB::bind_method(D_METHOD("set_path_joined", "enable"), &CSGPolygonShape3D::set_path_joined);
	ClassDB::bind_method(D_METHOD("is_path_joined"), &CSGPolygonShape3D::is_path_joined);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "polygon"), "set_polygon", "get_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Depth,Spin,Path"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "depth", PROPERTY_HINT_RANGE, "0.01,100.0,0.01,or_greater,exp"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "spin_degrees", PROPERTY_HINT_RANGE, "1,360,0.1"), "set_spin_degrees", "get_spin_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spin_sides", PROPERTY_HINT_RANGE, "3,64,1"), "set_spin_sides", "get_spin_sides");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "path_curve", PROPERTY_HINT_RESOURCE_TYPE), "set_path_curve", "get_path_curve");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "path_transform"), "set_path_transform", "get_path_transform");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_interval", PROPERTY_HINT_RANGE, "0.1,1.0,0.05,exp"), "set_path_interval", "get_path_interval");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_rotation", PROPERTY_HINT_ENUM, "Polygon,Path,PathFollow"), "set_path_rotation", "get_path_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_continuous_u"), "set_path_continuous_u", "is_path_continuous_u");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "path_joined"), "set_path_joined", "is_path_joined");

	BIND_ENUM_CONSTANT(MODE_DEPTH);
	BIND_ENUM_CONSTANT(MODE_SPIN);
	BIND_ENUM_CONSTANT(MODE_PATH);

	BIND_ENUM_CONSTANT(PATH_ROTATION_POLYGON);
	BIND_ENUM_CONSTANT(PATH_ROTATION_PATH);
	BIND_ENUM_CONSTANT(PATH_ROTATION_PATH_FOLLOW);
}

void CSGPolygonShape3D::_validate_property(PropertyInfo &property) const {
	if (property.name.begins_with("spin") && mode != CSGPolygonShape3D::MODE_SPIN) {
		property.usage = PROPERTY_USAGE_NONE;
	}
	if (property.name.begins_with("path") && mode != CSGPolygonShape3D::MODE_PATH) {
		property.usage = PROPERTY_USAGE_NONE;
	}
	if (property.name == "depth" && mode != CSGPolygonShape3D::MODE_DEPTH) {
		property.usage = PROPERTY_USAGE_NONE;
	}

	CSGPrimitiveShape3D::_validate_property(property);
}

void CSGPolygonShape3D::set_polygon(const Vector<Vector2> &p_polygon) {
	dirty = true;
	emit_changed();
	polygon = p_polygon;
}

Vector<Vector2> CSGPolygonShape3D::get_polygon() const {
	return polygon;
}

void CSGPolygonShape3D::set_mode(Mode p_mode) {
	dirty = true;
	emit_changed();
	mode = p_mode;
	notify_property_list_changed();
}

CSGPolygonShape3D::Mode CSGPolygonShape3D::get_mode() const {
	return mode;
}

void CSGPolygonShape3D::set_depth(const float p_depth) {
	ERR_FAIL_COND(p_depth < 0.001);
	dirty = true;
	emit_changed();
	depth = p_depth;
}

float CSGPolygonShape3D::get_depth() const {
	return depth;
}

void CSGPolygonShape3D::set_path_curve(const Ref<Curve3D> &p_curve) {
	dirty = true;
	emit_changed();
	path_curve = p_curve;
}

Ref<Curve3D> CSGPolygonShape3D::get_path_curve() const {
	return path_curve;
}

void CSGPolygonShape3D::set_path_transform(Transform3D p_xform) {
	dirty = true;
	emit_changed();
	path_transform = p_xform;
}

Transform3D CSGPolygonShape3D::get_path_transform() const {
	return path_transform;
}

void CSGPolygonShape3D::set_path_continuous_u(bool p_enable) {
	dirty = true;
	emit_changed();
	path_continuous_u = p_enable;
}

bool CSGPolygonShape3D::is_path_continuous_u() const {
	return path_continuous_u;
}

void CSGPolygonShape3D::set_spin_degrees(const float p_spin_degrees) {
	ERR_FAIL_COND(p_spin_degrees < 0.01 || p_spin_degrees > 360);
	dirty = true;
	emit_changed();
	spin_degrees = p_spin_degrees;
}

float CSGPolygonShape3D::get_spin_degrees() const {
	return spin_degrees;
}

void CSGPolygonShape3D::set_spin_sides(const int p_spin_sides) {
	ERR_FAIL_COND(p_spin_sides < 3);
	dirty = true;
	emit_changed();
	spin_sides = p_spin_sides;
}

int CSGPolygonShape3D::get_spin_sides() const {
	return spin_sides;
}

void CSGPolygonShape3D::set_path_interval(float p_interval) {
	ERR_FAIL_COND_MSG(p_interval < 0.001, "Path interval cannot be smaller than 0.001.");
	dirty = true;
	emit_changed();
	path_interval = p_interval;
}

float CSGPolygonShape3D::get_path_interval() const {
	return path_interval;
}

void CSGPolygonShape3D::set_path_rotation(PathRotation p_rotation) {
	dirty = true;
	emit_changed();
	path_rotation = p_rotation;
}

CSGPolygonShape3D::PathRotation CSGPolygonShape3D::get_path_rotation() const {
	return path_rotation;
}

void CSGPolygonShape3D::set_path_joined(bool p_enable) {
	dirty = true;
	emit_changed();
	path_joined = p_enable;
}

bool CSGPolygonShape3D::is_path_joined() const {
	return path_joined;
}

void CSGPolygonShape3D::_update_brush() {
	if (dirty) {
		brush = CSGBrush();

		if (polygon.size() < 3) {
			dirty = false;
			return;
		}

		// Triangulate polygon shape.
		Vector<Point2> shape_polygon = polygon;
		if (Triangulate::get_area(shape_polygon) > 0) {
			shape_polygon.reverse();
		}
		int shape_sides = shape_polygon.size();
		Vector<int> shape_faces = Geometry2D::triangulate_polygon(shape_polygon);
		ERR_FAIL_COND_MSG(shape_faces.size() < 3, "Failed to triangulate CSGPolygon");

		// Get polygon enclosing Rect2.
		Rect2 shape_rect(shape_polygon[0], Vector2());
		for (int i = 1; i < shape_sides; i++) {
			shape_rect.expand_to(shape_polygon[i]);
		}

		// If MODE_PATH, check if curve has changed.
		if (mode == MODE_PATH) {
			ERR_FAIL_COND(path_curve.is_null());
		}

		// Calculate the number extrusions, ends and faces.
		int extrusions = 0;
		int extrusion_face_count = shape_sides * 2;
		int end_count = 0;
		int shape_face_count = shape_faces.size() / 3;
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
				extrusions = Math::ceil(1.0 * path_curve->get_point_count() / path_interval);
				if (!path_joined) {
					end_count = 2;
					extrusions -= 1;
				}
			} break;
		}
		int face_count = extrusions * extrusion_face_count + end_count * shape_face_count;

		// Intialize variables used to create the mesh.
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

		Vector3 *facesw = faces.ptrw();
		Vector2 *uvsw = uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		int face = 0;
		Transform3D current_xform;
		Transform3D previous_xform;
		double u_step = 1.0 / extrusions;
		double v_step = 1.0 / shape_sides;
		double spin_step = Math::deg2rad(spin_degrees / spin_sides);
		double extrusion_step = 1.0 / extrusions;
		if (mode == MODE_PATH) {
			if (path_joined) {
				extrusion_step = 1.0 / (extrusions - 1);
			}
			extrusion_step *= path_curve->get_baked_length();
		}

		if (mode == MODE_PATH) {
			Vector3 current_point = path_curve->interpolate_baked(0);
			Vector3 next_point = path_curve->interpolate_baked(extrusion_step);
			Vector3 current_up = Vector3(0, 1, 0);
			Vector3 direction = next_point - current_point;

			if (path_joined) {
				Vector3 last_point = path_curve->interpolate_baked(path_curve->get_baked_length());
				direction = next_point - last_point;
			}

			switch (path_rotation) {
				case PATH_ROTATION_POLYGON:
					direction = Vector3(0, 0, -1);
					break;
				case PATH_ROTATION_PATH:
					break;
				case PATH_ROTATION_PATH_FOLLOW:
					current_up = path_curve->interpolate_baked_up_vector(0);
					break;
			}

			Transform3D facing = Transform3D().looking_at(direction, current_up);
			current_xform = path_transform.translated(current_point) * facing;
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
				invertw[face] = is_inverting_faces();
				face++;
			}
		}

		// Add extrusion faces.
		for (int x0 = 0; x0 < extrusions; x0++) {
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

					Vector3 previous_point = path_curve->interpolate_baked(previous_offset);
					Vector3 current_point = path_curve->interpolate_baked(current_offset);
					Vector3 next_point = path_curve->interpolate_baked(next_offset);
					Vector3 current_up = Vector3(0, 1, 0);
					Vector3 direction = next_point - previous_point;

					switch (path_rotation) {
						case PATH_ROTATION_POLYGON:
							direction = Vector3(0, 0, -1);
							break;
						case PATH_ROTATION_PATH:
							break;
						case PATH_ROTATION_PATH_FOLLOW:
							current_up = path_curve->interpolate_baked_up_vector(current_offset);
							break;
					}

					Transform3D facing = Transform3D().looking_at(direction, current_up);
					current_xform = path_transform.translated(current_point) * facing;
				} break;
			}

			double u0 = x0 * u_step;
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

				smoothw[face] = get_smooth_faces();
				invertw[face] = is_inverting_faces();
				materialsw[face] = material;

				face++;

				// Face 2
				facesw[face * 3 + 0] = v[2];
				facesw[face * 3 + 1] = v[3];
				facesw[face * 3 + 2] = v[0];

				uvsw[face * 3 + 0] = u[2];
				uvsw[face * 3 + 1] = u[3];
				uvsw[face * 3 + 2] = u[0];

				smoothw[face] = get_smooth_faces();
				invertw[face] = is_inverting_faces();
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
				invertw[face] = is_inverting_faces();
				face++;
			}
		}

		ERR_FAIL_COND_MSG(face != face_count, "Bug: Failed to create the CSGPolygon mesh correctly.");

		brush.build_from_faces(faces, uvs, smooth, materials, invert);
	}
}

// CSGTool

int CSGTool::mikktGetNumFaces(const SMikkTSpaceContext *pContext) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	return surface.vertices.size() / 3;
}

int CSGTool::mikktGetNumVerticesOfFace(const SMikkTSpaceContext *pContext, const int iFace) {
	// always 3
	return 3;
}

void CSGTool::mikktGetPosition(const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace, const int iVert) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	Vector3 v = surface.verticesw[iFace * 3 + iVert];
	fvPosOut[0] = v.x;
	fvPosOut[1] = v.y;
	fvPosOut[2] = v.z;
}

void CSGTool::mikktGetNormal(const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace, const int iVert) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	Vector3 n = surface.normalsw[iFace * 3 + iVert];
	fvNormOut[0] = n.x;
	fvNormOut[1] = n.y;
	fvNormOut[2] = n.z;
}

void CSGTool::mikktGetTexCoord(const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace, const int iVert) {
	ShapeUpdateSurface &surface = *((ShapeUpdateSurface *)pContext->m_pUserData);

	Vector2 t = surface.uvsw[iFace * 3 + iVert];
	fvTexcOut[0] = t.x;
	fvTexcOut[1] = t.y;
}

void CSGTool::mikktSetTSpaceDefault(const SMikkTSpaceContext *pContext, const float fvTangent[], const float fvBiTangent[], const float fMagS, const float fMagT,
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

void CSGTool::add_brush(CSGBrush &p_brush, Operation p_operation, float p_vertex_snap) {
	if (brush.faces.is_empty()) {
		brush = p_brush;
	} else {
		CSGBrush newbrush;
		CSGBrushOperation brush_operation;
		brush_operation.merge_brushes(static_cast<CSGBrushOperation::Operation>(p_operation), brush, p_brush, newbrush, p_vertex_snap);
		brush = newbrush;
	}
}

Vector<Vector3> CSGTool::get_brush_faces() const {
	Vector<Vector3> faces;
	int fc = brush.faces.size();
	faces.resize(fc * 3);
	{
		Vector3 *w = faces.ptrw();
		for (int i = 0; i < fc; i++) {
			w[i * 3 + 0] = brush.faces[i].vertices[0];
			w[i * 3 + 1] = brush.faces[i].vertices[1];
			w[i * 3 + 2] = brush.faces[i].vertices[2];
		}
	}

	return faces;
}

void CSGTool::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_primitive", "primitive", "operation", "xform", "vertex_snap"), &CSGTool::add_primitive, DEFVAL(OPERATION_UNION), DEFVAL(Transform3D()), DEFVAL(0.001));
	ClassDB::bind_method(D_METHOD("create_trimesh_shape"), &CSGTool::create_trimesh_shape);
	ClassDB::bind_method(D_METHOD("commit", "existing", "generate_tangents"), &CSGTool::commit, DEFVAL(Variant()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_aabb"), &CSGTool::get_aabb);

	BIND_ENUM_CONSTANT(OPERATION_UNION);
	BIND_ENUM_CONSTANT(OPERATION_INTERSECTION);
	BIND_ENUM_CONSTANT(OPERATION_SUBTRACTION);
}

void CSGTool::add_primitive(Ref<CSGPrimitiveShape3D> p_primitive, Operation p_operation, Transform3D p_xform, float p_vertex_snap) {
	CSGBrush newbrush;
	newbrush.copy_from(p_primitive->get_brush(), p_xform);
	add_brush(newbrush, p_operation, p_vertex_snap);
}

Ref<ConcavePolygonShape3D> CSGTool::create_trimesh_shape() const {
	Ref<ConcavePolygonShape3D> shape;
	shape.instantiate();

	Vector<Vector3> physics_faces;
	physics_faces.resize(brush.faces.size() * 3);
	Vector3 *physicsw = physics_faces.ptrw();

	for (int i = 0; i < brush.faces.size(); i++) {
		int order[3] = { 0, 1, 2 };

		if (brush.faces[i].invert) {
			SWAP(order[1], order[2]);
		}

		physicsw[i * 3 + 0] = brush.faces[i].vertices[order[0]];
		physicsw[i * 3 + 1] = brush.faces[i].vertices[order[1]];
		physicsw[i * 3 + 2] = brush.faces[i].vertices[order[2]];
	}
	return shape;
}

Ref<ArrayMesh> CSGTool::commit(const Ref<ArrayMesh> &p_existing, bool p_generate_tangents) {
	Ref<ArrayMesh> mesh;
	if (p_existing.is_valid()) {
		mesh = p_existing;
	} else {
		mesh.instantiate();
	}

	OAHashMap<Vector3, Vector3> vec_map;

	Vector<int> face_count;
	face_count.resize(brush.materials.size() + 1);
	for (int i = 0; i < face_count.size(); i++) {
		face_count.write[i] = 0;
	}

	for (int i = 0; i < brush.faces.size(); i++) {
		int mat = brush.faces[i].material;
		ERR_CONTINUE(mat < -1 || mat >= face_count.size());
		int idx = mat == -1 ? face_count.size() - 1 : mat;

		Plane p(brush.faces[i].vertices[0], brush.faces[i].vertices[1], brush.faces[i].vertices[2]);

		for (int j = 0; j < 3; j++) {
			Vector3 v = brush.faces[i].vertices[j];
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
		if (p_generate_tangents) {
			surfaces.write[i].tans.resize(face_count[i] * 3 * 4);
		}
		surfaces.write[i].last_added = 0;

		if (i != surfaces.size() - 1) {
			surfaces.write[i].material = brush.materials[i];
		}

		surfaces.write[i].verticesw = surfaces.write[i].vertices.ptrw();
		surfaces.write[i].normalsw = surfaces.write[i].normals.ptrw();
		surfaces.write[i].uvsw = surfaces.write[i].uvs.ptrw();
		if (p_generate_tangents) {
			surfaces.write[i].tansw = surfaces.write[i].tans.ptrw();
		}
	}

	//fill arrays
	{
		for (int i = 0; i < brush.faces.size(); i++) {
			int order[3] = { 0, 1, 2 };

			if (brush.faces[i].invert) {
				SWAP(order[1], order[2]);
			}

			int mat = brush.faces[i].material;
			ERR_CONTINUE(mat < -1 || mat >= face_count.size());
			int idx = mat == -1 ? face_count.size() - 1 : mat;

			int last = surfaces[idx].last_added;

			Plane p(brush.faces[i].vertices[0], brush.faces[i].vertices[1], brush.faces[i].vertices[2]);

			for (int j = 0; j < 3; j++) {
				Vector3 v = brush.faces[i].vertices[j];

				Vector3 normal = p.normal;

				if (brush.faces[i].smooth && vec_map.lookup(v, normal)) {
					normal.normalize();
				}

				if (brush.faces[i].invert) {
					normal = -normal;
				}

				int k = last + order[j];
				surfaces[idx].verticesw[k] = v;
				surfaces[idx].uvsw[k] = brush.faces[i].uvs[j];
				surfaces[idx].normalsw[k] = normal;

				if (p_generate_tangents) {
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

	//create surfaces

	for (int i = 0; i < surfaces.size(); i++) {
		// generate tangents for this surface
		bool have_tangents = p_generate_tangents;
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

		int idx = mesh->get_surface_count();
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, array);
		mesh->surface_set_material(idx, surfaces[i].material);
	}
	return mesh;
}

AABB CSGTool::get_aabb() const {
	AABB aabb;
	for (int i = 0; i < brush.faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			if (i == 0 && j == 0) {
				aabb.position = brush.faces[i].vertices[j];
			} else {
				aabb.expand_to(brush.faces[i].vertices[j]);
			}
		}
	}
	return aabb;
}
