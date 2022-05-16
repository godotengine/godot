/*************************************************************************/
/*  voxel_light_baker.cpp                                                */
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

#include "voxel_light_baker.h"

#include "core/os/os.h"
#include "core/os/threaded_array_processor.h"

#include <stdlib.h>

#define FINDMINMAX(x0, x1, x2, min, max) \
	min = max = x0;                      \
	if (x1 < min)                        \
		min = x1;                        \
	if (x1 > max)                        \
		max = x1;                        \
	if (x2 < min)                        \
		min = x2;                        \
	if (x2 > max)                        \
		max = x2;

static bool planeBoxOverlap(Vector3 normal, float d, Vector3 maxbox) {
	int q;
	Vector3 vmin, vmax;
	for (q = 0; q <= 2; q++) {
		if (normal[q] > 0.0f) {
			vmin[q] = -maxbox[q];
			vmax[q] = maxbox[q];
		} else {
			vmin[q] = maxbox[q];
			vmax[q] = -maxbox[q];
		}
	}
	if (normal.dot(vmin) + d > 0.0f) {
		return false;
	}
	if (normal.dot(vmax) + d >= 0.0f) {
		return true;
	}

	return false;
}

/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)                 \
	p0 = a * v0.y - b * v0.z;                      \
	p2 = a * v2.y - b * v2.z;                      \
	if (p0 < p2) {                                 \
		min = p0;                                  \
		max = p2;                                  \
	} else {                                       \
		min = p2;                                  \
		max = p0;                                  \
	}                                              \
	rad = fa * boxhalfsize.y + fb * boxhalfsize.z; \
	if (min > rad || max < -rad)                   \
		return false;

#define AXISTEST_X2(a, b, fa, fb)                  \
	p0 = a * v0.y - b * v0.z;                      \
	p1 = a * v1.y - b * v1.z;                      \
	if (p0 < p1) {                                 \
		min = p0;                                  \
		max = p1;                                  \
	} else {                                       \
		min = p1;                                  \
		max = p0;                                  \
	}                                              \
	rad = fa * boxhalfsize.y + fb * boxhalfsize.z; \
	if (min > rad || max < -rad)                   \
		return false;

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)                 \
	p0 = -a * v0.x + b * v0.z;                     \
	p2 = -a * v2.x + b * v2.z;                     \
	if (p0 < p2) {                                 \
		min = p0;                                  \
		max = p2;                                  \
	} else {                                       \
		min = p2;                                  \
		max = p0;                                  \
	}                                              \
	rad = fa * boxhalfsize.x + fb * boxhalfsize.z; \
	if (min > rad || max < -rad)                   \
		return false;

#define AXISTEST_Y1(a, b, fa, fb)                  \
	p0 = -a * v0.x + b * v0.z;                     \
	p1 = -a * v1.x + b * v1.z;                     \
	if (p0 < p1) {                                 \
		min = p0;                                  \
		max = p1;                                  \
	} else {                                       \
		min = p1;                                  \
		max = p0;                                  \
	}                                              \
	rad = fa * boxhalfsize.x + fb * boxhalfsize.z; \
	if (min > rad || max < -rad)                   \
		return false;

/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)                 \
	p1 = a * v1.x - b * v1.y;                      \
	p2 = a * v2.x - b * v2.y;                      \
	if (p2 < p1) {                                 \
		min = p2;                                  \
		max = p1;                                  \
	} else {                                       \
		min = p1;                                  \
		max = p2;                                  \
	}                                              \
	rad = fa * boxhalfsize.x + fb * boxhalfsize.y; \
	if (min > rad || max < -rad)                   \
		return false;

#define AXISTEST_Z0(a, b, fa, fb)                  \
	p0 = a * v0.x - b * v0.y;                      \
	p1 = a * v1.x - b * v1.y;                      \
	if (p0 < p1) {                                 \
		min = p0;                                  \
		max = p1;                                  \
	} else {                                       \
		min = p1;                                  \
		max = p0;                                  \
	}                                              \
	rad = fa * boxhalfsize.x + fb * boxhalfsize.y; \
	if (min > rad || max < -rad)                   \
		return false;

static bool fast_tri_box_overlap(const Vector3 &boxcenter, const Vector3 boxhalfsize, const Vector3 *triverts) {
	/*    use separating axis theorem to test overlap between triangle and box */
	/*    need to test for overlap in these directions: */
	/*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */
	/*       we do not even need to test these) */
	/*    2) normal of the triangle */
	/*    3) crossproduct(edge from tri, {x,y,z}-directin) */
	/*       this gives 3x3=9 more tests */
	Vector3 v0, v1, v2;
	float min, max, d, p0, p1, p2, rad, fex, fey, fez;
	Vector3 normal, e0, e1, e2;

	/* This is the fastest branch on Sun */
	/* move everything so that the boxcenter is in (0,0,0) */

	v0 = triverts[0] - boxcenter;
	v1 = triverts[1] - boxcenter;
	v2 = triverts[2] - boxcenter;

	/* compute triangle edges */
	e0 = v1 - v0; /* tri edge 0 */
	e1 = v2 - v1; /* tri edge 1 */
	e2 = v0 - v2; /* tri edge 2 */

	/* Bullet 3:  */
	/*  test the 9 tests first (this was faster) */
	fex = Math::abs(e0.x);
	fey = Math::abs(e0.y);
	fez = Math::abs(e0.z);
	AXISTEST_X01(e0.z, e0.y, fez, fey);
	AXISTEST_Y02(e0.z, e0.x, fez, fex);
	AXISTEST_Z12(e0.y, e0.x, fey, fex);

	fex = Math::abs(e1.x);
	fey = Math::abs(e1.y);
	fez = Math::abs(e1.z);
	AXISTEST_X01(e1.z, e1.y, fez, fey);
	AXISTEST_Y02(e1.z, e1.x, fez, fex);
	AXISTEST_Z0(e1.y, e1.x, fey, fex);

	fex = Math::abs(e2.x);
	fey = Math::abs(e2.y);
	fez = Math::abs(e2.z);
	AXISTEST_X2(e2.z, e2.y, fez, fey);
	AXISTEST_Y1(e2.z, e2.x, fez, fex);
	AXISTEST_Z12(e2.y, e2.x, fey, fex);

	/* Bullet 1: */
	/*  first test overlap in the {x,y,z}-directions */
	/*  find min, max of the triangle each direction, and test for overlap in */
	/*  that direction -- this is equivalent to testing a minimal AABB around */
	/*  the triangle against the AABB */

	/* test in X-direction */
	FINDMINMAX(v0.x, v1.x, v2.x, min, max);
	if (min > boxhalfsize.x || max < -boxhalfsize.x) {
		return false;
	}

	/* test in Y-direction */
	FINDMINMAX(v0.y, v1.y, v2.y, min, max);
	if (min > boxhalfsize.y || max < -boxhalfsize.y) {
		return false;
	}

	/* test in Z-direction */
	FINDMINMAX(v0.z, v1.z, v2.z, min, max);
	if (min > boxhalfsize.z || max < -boxhalfsize.z) {
		return false;
	}

	/* Bullet 2: */
	/*  test if the box intersects the plane of the triangle */
	/*  compute plane equation of triangle: normal*x+d=0 */
	normal = e0.cross(e1);
	d = -normal.dot(v0); /* plane eq: normal.x+d=0 */
	return planeBoxOverlap(normal, d, boxhalfsize); /* if true, box and triangle overlaps */
}

static _FORCE_INLINE_ void get_uv_and_normal(const Vector3 &p_pos, const Vector3 *p_vtx, const Vector2 *p_uv, const Vector3 *p_normal, Vector2 &r_uv, Vector3 &r_normal) {
	if (p_pos.distance_squared_to(p_vtx[0]) < CMP_EPSILON2) {
		r_uv = p_uv[0];
		r_normal = p_normal[0];
		return;
	}
	if (p_pos.distance_squared_to(p_vtx[1]) < CMP_EPSILON2) {
		r_uv = p_uv[1];
		r_normal = p_normal[1];
		return;
	}
	if (p_pos.distance_squared_to(p_vtx[2]) < CMP_EPSILON2) {
		r_uv = p_uv[2];
		r_normal = p_normal[2];
		return;
	}

	Vector3 v0 = p_vtx[1] - p_vtx[0];
	Vector3 v1 = p_vtx[2] - p_vtx[0];
	Vector3 v2 = p_pos - p_vtx[0];

	float d00 = v0.dot(v0);
	float d01 = v0.dot(v1);
	float d11 = v1.dot(v1);
	float d20 = v2.dot(v0);
	float d21 = v2.dot(v1);
	float denom = (d00 * d11 - d01 * d01);
	if (denom == 0) {
		r_uv = p_uv[0];
		r_normal = p_normal[0];
		return;
	}
	float v = (d11 * d20 - d01 * d21) / denom;
	float w = (d00 * d21 - d01 * d20) / denom;
	float u = 1.0f - v - w;

	r_uv = p_uv[0] * u + p_uv[1] * v + p_uv[2] * w;
	r_normal = (p_normal[0] * u + p_normal[1] * v + p_normal[2] * w).normalized();
}

void VoxelLightBaker::_plot_face(int p_idx, int p_level, int p_x, int p_y, int p_z, const Vector3 *p_vtx, const Vector3 *p_normal, const Vector2 *p_uv, const MaterialCache &p_material, const AABB &p_aabb) {
	if (p_level == cell_subdiv - 1) {
		//plot the face by guessing its albedo and emission value

		//find best axis to map to, for scanning values
		int closest_axis = 0;
		float closest_dot = 0;

		Plane plane = Plane(p_vtx[0], p_vtx[1], p_vtx[2]);
		Vector3 normal = plane.normal;

		for (int i = 0; i < 3; i++) {
			Vector3 axis;
			axis[i] = 1.0;
			float dot = ABS(normal.dot(axis));
			if (i == 0 || dot > closest_dot) {
				closest_axis = i;
				closest_dot = dot;
			}
		}

		Vector3 axis;
		axis[closest_axis] = 1.0;
		Vector3 t1;
		t1[(closest_axis + 1) % 3] = 1.0;
		Vector3 t2;
		t2[(closest_axis + 2) % 3] = 1.0;

		t1 *= p_aabb.size[(closest_axis + 1) % 3] / float(color_scan_cell_width);
		t2 *= p_aabb.size[(closest_axis + 2) % 3] / float(color_scan_cell_width);

		Color albedo_accum;
		Color emission_accum;
		Vector3 normal_accum;

		float alpha = 0.0;

		//map to a grid average in the best axis for this face
		for (int i = 0; i < color_scan_cell_width; i++) {
			Vector3 ofs_i = float(i) * t1;

			for (int j = 0; j < color_scan_cell_width; j++) {
				Vector3 ofs_j = float(j) * t2;

				Vector3 from = p_aabb.position + ofs_i + ofs_j;
				Vector3 to = from + t1 + t2 + axis * p_aabb.size[closest_axis];
				Vector3 half = (to - from) * 0.5;

				//is in this cell?
				if (!fast_tri_box_overlap(from + half, half, p_vtx)) {
					continue; //face does not span this cell
				}

				//go from -size to +size*2 to avoid skipping collisions
				Vector3 ray_from = from + (t1 + t2) * 0.5 - axis * p_aabb.size[closest_axis];
				Vector3 ray_to = ray_from + axis * p_aabb.size[closest_axis] * 2;

				if (normal.dot(ray_from - ray_to) < 0) {
					SWAP(ray_from, ray_to);
				}

				Vector3 intersection;

				if (!plane.intersects_segment(ray_from, ray_to, &intersection)) {
					if (ABS(plane.distance_to(ray_from)) < ABS(plane.distance_to(ray_to))) {
						intersection = plane.project(ray_from);
					} else {
						intersection = plane.project(ray_to);
					}
				}

				intersection = Face3(p_vtx[0], p_vtx[1], p_vtx[2]).get_closest_point_to(intersection);

				Vector2 uv;
				Vector3 lnormal;
				get_uv_and_normal(intersection, p_vtx, p_uv, p_normal, uv, lnormal);
				if (lnormal == Vector3()) { //just in case normal as nor provided
					lnormal = normal;
				}

				int uv_x = CLAMP(int(Math::fposmod(uv.x, 1.0f) * bake_texture_size), 0, bake_texture_size - 1);
				int uv_y = CLAMP(int(Math::fposmod(uv.y, 1.0f) * bake_texture_size), 0, bake_texture_size - 1);

				int ofs = uv_y * bake_texture_size + uv_x;
				albedo_accum.r += p_material.albedo[ofs].r;
				albedo_accum.g += p_material.albedo[ofs].g;
				albedo_accum.b += p_material.albedo[ofs].b;
				albedo_accum.a += p_material.albedo[ofs].a;

				emission_accum.r += p_material.emission[ofs].r;
				emission_accum.g += p_material.emission[ofs].g;
				emission_accum.b += p_material.emission[ofs].b;

				normal_accum += lnormal;

				alpha += 1.0;
			}
		}

		if (alpha == 0) {
			//could not in any way get texture information.. so use closest point to center

			Face3 f(p_vtx[0], p_vtx[1], p_vtx[2]);
			Vector3 inters = f.get_closest_point_to(p_aabb.position + p_aabb.size * 0.5);

			Vector3 lnormal;
			Vector2 uv;
			get_uv_and_normal(inters, p_vtx, p_uv, p_normal, uv, normal);
			if (lnormal == Vector3()) { //just in case normal as nor provided
				lnormal = normal;
			}

			int uv_x = CLAMP(Math::fposmod(uv.x, 1.0f) * bake_texture_size, 0, bake_texture_size - 1);
			int uv_y = CLAMP(Math::fposmod(uv.y, 1.0f) * bake_texture_size, 0, bake_texture_size - 1);

			int ofs = uv_y * bake_texture_size + uv_x;

			alpha = 1.0 / (color_scan_cell_width * color_scan_cell_width);

			albedo_accum.r = p_material.albedo[ofs].r * alpha;
			albedo_accum.g = p_material.albedo[ofs].g * alpha;
			albedo_accum.b = p_material.albedo[ofs].b * alpha;
			albedo_accum.a = p_material.albedo[ofs].a * alpha;

			emission_accum.r = p_material.emission[ofs].r * alpha;
			emission_accum.g = p_material.emission[ofs].g * alpha;
			emission_accum.b = p_material.emission[ofs].b * alpha;

			normal_accum = lnormal * alpha;

		} else {
			float accdiv = 1.0 / (color_scan_cell_width * color_scan_cell_width);
			alpha *= accdiv;

			albedo_accum.r *= accdiv;
			albedo_accum.g *= accdiv;
			albedo_accum.b *= accdiv;
			albedo_accum.a *= accdiv;

			emission_accum.r *= accdiv;
			emission_accum.g *= accdiv;
			emission_accum.b *= accdiv;

			normal_accum *= accdiv;
		}

		//put this temporarily here, corrected in a later step
		bake_cells.write[p_idx].albedo[0] += albedo_accum.r;
		bake_cells.write[p_idx].albedo[1] += albedo_accum.g;
		bake_cells.write[p_idx].albedo[2] += albedo_accum.b;
		bake_cells.write[p_idx].emission[0] += emission_accum.r;
		bake_cells.write[p_idx].emission[1] += emission_accum.g;
		bake_cells.write[p_idx].emission[2] += emission_accum.b;
		bake_cells.write[p_idx].normal[0] += normal_accum.x;
		bake_cells.write[p_idx].normal[1] += normal_accum.y;
		bake_cells.write[p_idx].normal[2] += normal_accum.z;
		bake_cells.write[p_idx].alpha += alpha;

	} else {
		//go down

		int half = (1 << (cell_subdiv - 1)) >> (p_level + 1);
		for (int i = 0; i < 8; i++) {
			AABB aabb = p_aabb;
			aabb.size *= 0.5;

			int nx = p_x;
			int ny = p_y;
			int nz = p_z;

			if (i & 1) {
				aabb.position.x += aabb.size.x;
				nx += half;
			}
			if (i & 2) {
				aabb.position.y += aabb.size.y;
				ny += half;
			}
			if (i & 4) {
				aabb.position.z += aabb.size.z;
				nz += half;
			}
			//make sure to not plot beyond limits
			if (nx < 0 || nx >= axis_cell_size[0] || ny < 0 || ny >= axis_cell_size[1] || nz < 0 || nz >= axis_cell_size[2]) {
				continue;
			}

			{
				AABB test_aabb = aabb;
				//test_aabb.grow_by(test_aabb.get_longest_axis_size()*0.05); //grow a bit to avoid numerical error in real-time
				Vector3 qsize = test_aabb.size * 0.5; //quarter size, for fast aabb test

				if (!fast_tri_box_overlap(test_aabb.position + qsize, qsize, p_vtx)) {
					//if (!Face3(p_vtx[0],p_vtx[1],p_vtx[2]).intersects_aabb2(aabb)) {
					//does not fit in child, go on
					continue;
				}
			}

			if (bake_cells[p_idx].children[i] == CHILD_EMPTY) {
				//sub cell must be created

				uint32_t child_idx = bake_cells.size();
				bake_cells.write[p_idx].children[i] = child_idx;
				bake_cells.resize(bake_cells.size() + 1);
				bake_cells.write[child_idx].level = p_level + 1;
			}

			_plot_face(bake_cells[p_idx].children[i], p_level + 1, nx, ny, nz, p_vtx, p_normal, p_uv, p_material, aabb);
		}
	}
}

Vector<Color> VoxelLightBaker::_get_bake_texture(Ref<Image> p_image, const Color &p_color_mul, const Color &p_color_add) {
	Vector<Color> ret;

	if (p_image.is_null() || p_image->empty()) {
		ret.resize(bake_texture_size * bake_texture_size);
		for (int i = 0; i < bake_texture_size * bake_texture_size; i++) {
			ret.write[i] = p_color_add;
		}

		return ret;
	}
	p_image = p_image->duplicate();

	if (p_image->is_compressed()) {
		p_image->decompress();
	}
	p_image->convert(Image::FORMAT_RGBA8);
	p_image->resize(bake_texture_size, bake_texture_size, Image::INTERPOLATE_CUBIC);

	PoolVector<uint8_t>::Read r = p_image->get_data().read();
	ret.resize(bake_texture_size * bake_texture_size);

	for (int i = 0; i < bake_texture_size * bake_texture_size; i++) {
		Color c;
		c.r = (r[i * 4 + 0] / 255.0) * p_color_mul.r + p_color_add.r;
		c.g = (r[i * 4 + 1] / 255.0) * p_color_mul.g + p_color_add.g;
		c.b = (r[i * 4 + 2] / 255.0) * p_color_mul.b + p_color_add.b;

		c.a = r[i * 4 + 3] / 255.0;

		ret.write[i] = c;
	}

	return ret;
}

VoxelLightBaker::MaterialCache VoxelLightBaker::_get_material_cache(Ref<Material> p_material) {
	//this way of obtaining materials is inaccurate and also does not support some compressed formats very well
	Ref<SpatialMaterial> mat = p_material;

	Ref<Material> material = mat; //hack for now

	if (material_cache.has(material)) {
		return material_cache[material];
	}

	MaterialCache mc;

	Ref<Image> empty;

	if (mat.is_valid()) {
		Ref<Texture> albedo_tex = mat->get_texture(SpatialMaterial::TEXTURE_ALBEDO);

		Ref<Image> img_albedo;
		if (albedo_tex.is_valid()) {
			img_albedo = albedo_tex->get_data();
			mc.albedo = _get_bake_texture(img_albedo, mat->get_albedo(), Color(0, 0, 0)); // albedo texture, color is multiplicative
		} else {
			mc.albedo = _get_bake_texture(img_albedo, Color(1, 1, 1), mat->get_albedo()); // no albedo texture, color is additive
		}

		if (mat->get_feature(SpatialMaterial::FEATURE_EMISSION)) {
			Ref<Texture> emission_tex = mat->get_texture(SpatialMaterial::TEXTURE_EMISSION);

			Color emission_col = mat->get_emission();
			float emission_energy = mat->get_emission_energy();

			Ref<Image> img_emission;

			if (emission_tex.is_valid()) {
				img_emission = emission_tex->get_data();
			}

			if (mat->get_emission_operator() == SpatialMaterial::EMISSION_OP_ADD) {
				mc.emission = _get_bake_texture(img_emission, Color(1, 1, 1) * emission_energy, emission_col * emission_energy);
			} else {
				mc.emission = _get_bake_texture(img_emission, emission_col * emission_energy, Color(0, 0, 0));
			}
		} else {
			mc.emission = _get_bake_texture(empty, Color(0, 0, 0), Color(0, 0, 0));
		}

	} else {
		mc.albedo = _get_bake_texture(empty, Color(0, 0, 0), Color(1, 1, 1));
		mc.emission = _get_bake_texture(empty, Color(0, 0, 0), Color(0, 0, 0));
	}

	material_cache[p_material] = mc;
	return mc;
}

void VoxelLightBaker::plot_mesh(const Transform &p_xform, Ref<Mesh> &p_mesh, const Vector<Ref<Material>> &p_materials, const Ref<Material> &p_override_material) {
	for (int i = 0; i < p_mesh->get_surface_count(); i++) {
		if (p_mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
			continue; //only triangles
		}

		Ref<Material> src_material;

		if (p_override_material.is_valid()) {
			src_material = p_override_material;
		} else if (i < p_materials.size() && p_materials[i].is_valid()) {
			src_material = p_materials[i];
		} else {
			src_material = p_mesh->surface_get_material(i);
		}
		MaterialCache material = _get_material_cache(src_material);

		Array a = p_mesh->surface_get_arrays(i);

		PoolVector<Vector3> vertices = a[Mesh::ARRAY_VERTEX];
		PoolVector<Vector3>::Read vr = vertices.read();
		PoolVector<Vector2> uv = a[Mesh::ARRAY_TEX_UV];
		PoolVector<Vector2>::Read uvr;
		PoolVector<Vector3> normals = a[Mesh::ARRAY_NORMAL];
		PoolVector<Vector3>::Read nr;
		PoolVector<int> index = a[Mesh::ARRAY_INDEX];

		bool read_uv = false;
		bool read_normals = false;

		if (uv.size()) {
			uvr = uv.read();
			read_uv = true;
		}

		if (normals.size()) {
			read_normals = true;
			nr = normals.read();
		}

		if (index.size()) {
			int facecount = index.size() / 3;
			PoolVector<int>::Read ir = index.read();

			for (int j = 0; j < facecount; j++) {
				Vector3 vtxs[3];
				Vector2 uvs[3];
				Vector3 normal[3];

				for (int k = 0; k < 3; k++) {
					vtxs[k] = p_xform.xform(vr[ir[j * 3 + k]]);
				}

				if (read_uv) {
					for (int k = 0; k < 3; k++) {
						uvs[k] = uvr[ir[j * 3 + k]];
					}
				}

				if (read_normals) {
					for (int k = 0; k < 3; k++) {
						normal[k] = nr[ir[j * 3 + k]];
					}
				}

				//test against original bounds
				if (!fast_tri_box_overlap(original_bounds.position + original_bounds.size * 0.5, original_bounds.size * 0.5, vtxs)) {
					continue;
				}
				//plot
				_plot_face(0, 0, 0, 0, 0, vtxs, normal, uvs, material, po2_bounds);
			}

		} else {
			int facecount = vertices.size() / 3;

			for (int j = 0; j < facecount; j++) {
				Vector3 vtxs[3];
				Vector2 uvs[3];
				Vector3 normal[3];

				for (int k = 0; k < 3; k++) {
					vtxs[k] = p_xform.xform(vr[j * 3 + k]);
				}

				if (read_uv) {
					for (int k = 0; k < 3; k++) {
						uvs[k] = uvr[j * 3 + k];
					}
				}

				if (read_normals) {
					for (int k = 0; k < 3; k++) {
						normal[k] = nr[j * 3 + k];
					}
				}

				//test against original bounds
				if (!fast_tri_box_overlap(original_bounds.position + original_bounds.size * 0.5, original_bounds.size * 0.5, vtxs)) {
					continue;
				}
				//plot face
				_plot_face(0, 0, 0, 0, 0, vtxs, normal, uvs, material, po2_bounds);
			}
		}
	}

	max_original_cells = bake_cells.size();
}

void VoxelLightBaker::_init_light_plot(int p_idx, int p_level, int p_x, int p_y, int p_z, uint32_t p_parent) {
	bake_light.write[p_idx].x = p_x;
	bake_light.write[p_idx].y = p_y;
	bake_light.write[p_idx].z = p_z;

	if (p_level == cell_subdiv - 1) {
		bake_light.write[p_idx].next_leaf = first_leaf;
		first_leaf = p_idx;
	} else {
		//go down
		int half = (1 << (cell_subdiv - 1)) >> (p_level + 1);
		for (int i = 0; i < 8; i++) {
			uint32_t child = bake_cells[p_idx].children[i];

			if (child == CHILD_EMPTY) {
				continue;
			}

			int nx = p_x;
			int ny = p_y;
			int nz = p_z;

			if (i & 1) {
				nx += half;
			}
			if (i & 2) {
				ny += half;
			}
			if (i & 4) {
				nz += half;
			}

			_init_light_plot(child, p_level + 1, nx, ny, nz, p_idx);
		}
	}
}

void VoxelLightBaker::begin_bake_light(BakeQuality p_quality, float p_propagation) {
	_check_init_light();
	propagation = p_propagation;
	bake_quality = p_quality;
}

void VoxelLightBaker::_check_init_light() {
	if (bake_light.size() == 0) {
		direct_lights_baked = false;
		leaf_voxel_count = 0;
		_fixup_plot(0, 0); //pre fixup, so normal, albedo, emission, etc. work for lighting.
		bake_light.resize(bake_cells.size());
		//memset(bake_light.ptrw(), 0, bake_light.size() * sizeof(Light));
		first_leaf = -1;
		_init_light_plot(0, 0, 0, 0, 0, CHILD_EMPTY);
	}
}

static float _get_normal_advance(const Vector3 &p_normal) {
	Vector3 normal = p_normal;
	Vector3 unorm = normal.abs();

	if ((unorm.x >= unorm.y) && (unorm.x >= unorm.z)) {
		// x code
		unorm = normal.x > 0.0 ? Vector3(1.0, 0.0, 0.0) : Vector3(-1.0, 0.0, 0.0);
	} else if ((unorm.y > unorm.x) && (unorm.y >= unorm.z)) {
		// y code
		unorm = normal.y > 0.0 ? Vector3(0.0, 1.0, 0.0) : Vector3(0.0, -1.0, 0.0);
	} else if ((unorm.z > unorm.x) && (unorm.z > unorm.y)) {
		// z code
		unorm = normal.z > 0.0 ? Vector3(0.0, 0.0, 1.0) : Vector3(0.0, 0.0, -1.0);
	} else {
		// oh-no we messed up code
		// has to be
		unorm = Vector3(1.0, 0.0, 0.0);
	}

	return 1.0 / normal.dot(unorm);
}

static const Vector3 aniso_normal[6] = {
	Vector3(-1, 0, 0),
	Vector3(1, 0, 0),
	Vector3(0, -1, 0),
	Vector3(0, 1, 0),
	Vector3(0, 0, -1),
	Vector3(0, 0, 1)
};

uint32_t VoxelLightBaker::_find_cell_at_pos(const Cell *cells, int x, int y, int z) {
	uint32_t cell = 0;

	int ofs_x = 0;
	int ofs_y = 0;
	int ofs_z = 0;
	int size = 1 << (cell_subdiv - 1);
	int half = size / 2;

	if (x < 0 || x >= size) {
		return -1;
	}
	if (y < 0 || y >= size) {
		return -1;
	}
	if (z < 0 || z >= size) {
		return -1;
	}

	for (int i = 0; i < cell_subdiv - 1; i++) {
		const Cell *bc = &cells[cell];

		int child = 0;
		if (x >= ofs_x + half) {
			child |= 1;
			ofs_x += half;
		}
		if (y >= ofs_y + half) {
			child |= 2;
			ofs_y += half;
		}
		if (z >= ofs_z + half) {
			child |= 4;
			ofs_z += half;
		}

		cell = bc->children[child];
		if (cell == CHILD_EMPTY) {
			return CHILD_EMPTY;
		}

		half >>= 1;
	}

	return cell;
}
void VoxelLightBaker::plot_light_directional(const Vector3 &p_direction, const Color &p_color, float p_energy, float p_indirect_energy, bool p_direct) {
	_check_init_light();

	float max_len = Vector3(axis_cell_size[0], axis_cell_size[1], axis_cell_size[2]).length() * 1.1;

	if (p_direct) {
		direct_lights_baked = true;
	}

	Vector3 light_axis = p_direction;
	Plane clip[3];
	int clip_planes = 0;

	Light *light_data = bake_light.ptrw();
	const Cell *cells = bake_cells.ptr();

	for (int i = 0; i < 3; i++) {
		if (Math::is_zero_approx(light_axis[i])) {
			continue;
		}
		clip[clip_planes].normal[i] = 1.0;

		if (light_axis[i] < 0) {
			clip[clip_planes].d = axis_cell_size[i] + 1;
		} else {
			clip[clip_planes].d -= 1.0;
		}

		clip_planes++;
	}

	float distance_adv = _get_normal_advance(light_axis);

	Vector3 light_energy = Vector3(p_color.r, p_color.g, p_color.b) * p_energy * p_indirect_energy;

	int idx = first_leaf;
	while (idx >= 0) {
		Light *light = &light_data[idx];

		Vector3 to(light->x + 0.5, light->y + 0.5, light->z + 0.5);
		to += -light_axis.sign() * 0.47; //make it more likely to receive a ray

		Vector3 from = to - max_len * light_axis;

		for (int j = 0; j < clip_planes; j++) {
			clip[j].intersects_segment(from, to, &from);
		}

		float distance = (to - from).length();
		distance += distance_adv - Math::fmod(distance, distance_adv); //make it reach the center of the box always
		from = to - light_axis * distance;

		uint32_t result = 0xFFFFFFFF;

		while (distance > -distance_adv) { //use this to avoid precision errors

			result = _find_cell_at_pos(cells, int(floor(from.x)), int(floor(from.y)), int(floor(from.z)));
			if (result != 0xFFFFFFFF) {
				break;
			}

			from += light_axis * distance_adv;
			distance -= distance_adv;
		}

		if (result == (uint32_t)idx) {
			//cell hit itself! hooray!

			Vector3 normal(cells[idx].normal[0], cells[idx].normal[1], cells[idx].normal[2]);
			if (normal == Vector3()) {
				for (int i = 0; i < 6; i++) {
					light->accum[i][0] += light_energy.x * cells[idx].albedo[0];
					light->accum[i][1] += light_energy.y * cells[idx].albedo[1];
					light->accum[i][2] += light_energy.z * cells[idx].albedo[2];
				}

			} else {
				for (int i = 0; i < 6; i++) {
					float s = MAX(0.0, aniso_normal[i].dot(-normal));
					light->accum[i][0] += light_energy.x * cells[idx].albedo[0] * s;
					light->accum[i][1] += light_energy.y * cells[idx].albedo[1] * s;
					light->accum[i][2] += light_energy.z * cells[idx].albedo[2] * s;
				}
			}

			if (p_direct) {
				for (int i = 0; i < 6; i++) {
					float s = MAX(0.0, aniso_normal[i].dot(-light_axis)); //light depending on normal for direct
					light->direct_accum[i][0] += light_energy.x * s;
					light->direct_accum[i][1] += light_energy.y * s;
					light->direct_accum[i][2] += light_energy.z * s;
				}
			}
		}

		idx = light_data[idx].next_leaf;
	}
}

void VoxelLightBaker::plot_light_omni(const Vector3 &p_pos, const Color &p_color, float p_energy, float p_indirect_energy, float p_radius, float p_attenutation, bool p_direct) {
	_check_init_light();

	if (p_direct) {
		direct_lights_baked = true;
	}

	Plane clip[3];
	int clip_planes = 0;

	// uint64_t us = OS::get_singleton()->get_ticks_usec();

	Vector3 light_pos = to_cell_space.xform(p_pos) + Vector3(0.5, 0.5, 0.5);
	//Vector3 spot_axis = -light_cache.transform.basis.get_axis(2).normalized();

	float local_radius = to_cell_space.basis.xform(Vector3(0, 0, 1)).length() * p_radius;

	Light *light_data = bake_light.ptrw();
	const Cell *cells = bake_cells.ptr();
	Vector3 light_energy = Vector3(p_color.r, p_color.g, p_color.b) * p_energy * p_indirect_energy;

	int idx = first_leaf;
	while (idx >= 0) {
		Light *light = &light_data[idx];

		Vector3 to(light->x + 0.5, light->y + 0.5, light->z + 0.5);
		to += (light_pos - to).sign() * 0.47; //make it more likely to receive a ray

		Vector3 light_axis = (to - light_pos).normalized();
		float distance_adv = _get_normal_advance(light_axis);

		Vector3 normal(cells[idx].normal[0], cells[idx].normal[1], cells[idx].normal[2]);

		if (normal != Vector3() && normal.dot(-light_axis) < 0.001) {
			idx = light_data[idx].next_leaf;
			continue;
		}

		float att = 1.0;
		{
			float d = light_pos.distance_to(to);
			if (d + distance_adv > local_radius) {
				idx = light_data[idx].next_leaf;
				continue; // too far away
			}

			float dt = CLAMP((d + distance_adv) / local_radius, 0, 1);
			att *= powf(1.0 - dt, p_attenutation);
		}

		clip_planes = 0;

		for (int c = 0; c < 3; c++) {
			if (Math::is_zero_approx(light_axis[c])) {
				continue;
			}
			clip[clip_planes].normal[c] = 1.0;

			if (light_axis[c] < 0) {
				clip[clip_planes].d = (1 << (cell_subdiv - 1)) + 1;
			} else {
				clip[clip_planes].d -= 1.0;
			}

			clip_planes++;
		}

		Vector3 from = light_pos;

		for (int j = 0; j < clip_planes; j++) {
			clip[j].intersects_segment(from, to, &from);
		}

		float distance = (to - from).length();

		distance -= Math::fmod(distance, distance_adv); //make it reach the center of the box always, but this tame make it closer
		from = to - light_axis * distance;
		to += (light_pos - to).sign() * 0.47; //make it more likely to receive a ray

		uint32_t result = 0xFFFFFFFF;

		while (distance > -distance_adv) { //use this to avoid precision errors

			result = _find_cell_at_pos(cells, int(floor(from.x)), int(floor(from.y)), int(floor(from.z)));
			if (result != 0xFFFFFFFF) {
				break;
			}

			from += light_axis * distance_adv;
			distance -= distance_adv;
		}

		if (result == (uint32_t)idx) {
			//cell hit itself! hooray!

			if (normal == Vector3()) {
				for (int i = 0; i < 6; i++) {
					light->accum[i][0] += light_energy.x * cells[idx].albedo[0] * att;
					light->accum[i][1] += light_energy.y * cells[idx].albedo[1] * att;
					light->accum[i][2] += light_energy.z * cells[idx].albedo[2] * att;
				}

			} else {
				for (int i = 0; i < 6; i++) {
					float s = MAX(0.0, aniso_normal[i].dot(-normal));
					light->accum[i][0] += light_energy.x * cells[idx].albedo[0] * s * att;
					light->accum[i][1] += light_energy.y * cells[idx].albedo[1] * s * att;
					light->accum[i][2] += light_energy.z * cells[idx].albedo[2] * s * att;
				}
			}

			if (p_direct) {
				for (int i = 0; i < 6; i++) {
					float s = MAX(0.0, aniso_normal[i].dot(-light_axis)); //light depending on normal for direct
					light->direct_accum[i][0] += light_energy.x * s * att;
					light->direct_accum[i][1] += light_energy.y * s * att;
					light->direct_accum[i][2] += light_energy.z * s * att;
				}
			}
		}

		idx = light_data[idx].next_leaf;
	}
}

void VoxelLightBaker::plot_light_spot(const Vector3 &p_pos, const Vector3 &p_axis, const Color &p_color, float p_energy, float p_indirect_energy, float p_radius, float p_attenutation, float p_spot_angle, float p_spot_attenuation, bool p_direct) {
	_check_init_light();

	if (p_direct) {
		direct_lights_baked = true;
	}

	Plane clip[3];
	int clip_planes = 0;

	// uint64_t us = OS::get_singleton()->get_ticks_usec();

	Vector3 light_pos = to_cell_space.xform(p_pos) + Vector3(0.5, 0.5, 0.5);
	Vector3 spot_axis = to_cell_space.basis.xform(p_axis).normalized();

	float local_radius = to_cell_space.basis.xform(Vector3(0, 0, 1)).length() * p_radius;

	Light *light_data = bake_light.ptrw();
	const Cell *cells = bake_cells.ptr();
	Vector3 light_energy = Vector3(p_color.r, p_color.g, p_color.b) * p_energy * p_indirect_energy;

	int idx = first_leaf;
	while (idx >= 0) {
		Light *light = &light_data[idx];

		Vector3 to(light->x + 0.5, light->y + 0.5, light->z + 0.5);

		Vector3 light_axis = (to - light_pos).normalized();
		float distance_adv = _get_normal_advance(light_axis);

		Vector3 normal(cells[idx].normal[0], cells[idx].normal[1], cells[idx].normal[2]);

		if (normal != Vector3() && normal.dot(-light_axis) < 0.001) {
			idx = light_data[idx].next_leaf;
			continue;
		}

		float angle = Math::rad2deg(Math::acos(light_axis.dot(-spot_axis)));
		if (angle > p_spot_angle) {
			idx = light_data[idx].next_leaf;
			continue; // too far away
		}

		float att = Math::pow(1.0f - angle / p_spot_angle, p_spot_attenuation);

		{
			float d = light_pos.distance_to(to);
			if (d + distance_adv > local_radius) {
				idx = light_data[idx].next_leaf;
				continue; // too far away
			}

			float dt = CLAMP((d + distance_adv) / local_radius, 0, 1);
			att *= powf(1.0 - dt, p_attenutation);
		}

		clip_planes = 0;

		for (int c = 0; c < 3; c++) {
			if (Math::is_zero_approx(light_axis[c])) {
				continue;
			}
			clip[clip_planes].normal[c] = 1.0;

			if (light_axis[c] < 0) {
				clip[clip_planes].d = (1 << (cell_subdiv - 1)) + 1;
			} else {
				clip[clip_planes].d -= 1.0;
			}

			clip_planes++;
		}

		Vector3 from = light_pos;

		for (int j = 0; j < clip_planes; j++) {
			clip[j].intersects_segment(from, to, &from);
		}

		float distance = (to - from).length();

		distance -= Math::fmod(distance, distance_adv); //make it reach the center of the box always, but this tame make it closer
		from = to - light_axis * distance;

		uint32_t result = 0xFFFFFFFF;

		while (distance > -distance_adv) { //use this to avoid precision errors

			result = _find_cell_at_pos(cells, int(floor(from.x)), int(floor(from.y)), int(floor(from.z)));
			if (result != 0xFFFFFFFF) {
				break;
			}

			from += light_axis * distance_adv;
			distance -= distance_adv;
		}

		if (result == (uint32_t)idx) {
			//cell hit itself! hooray!

			if (normal == Vector3()) {
				for (int i = 0; i < 6; i++) {
					light->accum[i][0] += light_energy.x * cells[idx].albedo[0] * att;
					light->accum[i][1] += light_energy.y * cells[idx].albedo[1] * att;
					light->accum[i][2] += light_energy.z * cells[idx].albedo[2] * att;
				}

			} else {
				for (int i = 0; i < 6; i++) {
					float s = MAX(0.0, aniso_normal[i].dot(-normal));
					light->accum[i][0] += light_energy.x * cells[idx].albedo[0] * s * att;
					light->accum[i][1] += light_energy.y * cells[idx].albedo[1] * s * att;
					light->accum[i][2] += light_energy.z * cells[idx].albedo[2] * s * att;
				}
			}

			if (p_direct) {
				for (int i = 0; i < 6; i++) {
					float s = MAX(0.0, aniso_normal[i].dot(-light_axis)); //light depending on normal for direct
					light->direct_accum[i][0] += light_energy.x * s * att;
					light->direct_accum[i][1] += light_energy.y * s * att;
					light->direct_accum[i][2] += light_energy.z * s * att;
				}
			}
		}

		idx = light_data[idx].next_leaf;
	}
}

void VoxelLightBaker::_fixup_plot(int p_idx, int p_level) {
	if (p_level == cell_subdiv - 1) {
		leaf_voxel_count++;
		float alpha = bake_cells[p_idx].alpha;

		bake_cells.write[p_idx].albedo[0] /= alpha;
		bake_cells.write[p_idx].albedo[1] /= alpha;
		bake_cells.write[p_idx].albedo[2] /= alpha;

		//transfer emission to light
		bake_cells.write[p_idx].emission[0] /= alpha;
		bake_cells.write[p_idx].emission[1] /= alpha;
		bake_cells.write[p_idx].emission[2] /= alpha;

		bake_cells.write[p_idx].normal[0] /= alpha;
		bake_cells.write[p_idx].normal[1] /= alpha;
		bake_cells.write[p_idx].normal[2] /= alpha;

		Vector3 n(bake_cells[p_idx].normal[0], bake_cells[p_idx].normal[1], bake_cells[p_idx].normal[2]);
		if (n.length() < 0.01) {
			//too much fight over normal, zero it
			bake_cells.write[p_idx].normal[0] = 0;
			bake_cells.write[p_idx].normal[1] = 0;
			bake_cells.write[p_idx].normal[2] = 0;
		} else {
			n.normalize();
			bake_cells.write[p_idx].normal[0] = n.x;
			bake_cells.write[p_idx].normal[1] = n.y;
			bake_cells.write[p_idx].normal[2] = n.z;
		}

		bake_cells.write[p_idx].alpha = 1.0;

		/*if (bake_light.size()) {
			for(int i=0;i<6;i++) {

			}
		}*/

	} else {
		//go down

		bake_cells.write[p_idx].emission[0] = 0;
		bake_cells.write[p_idx].emission[1] = 0;
		bake_cells.write[p_idx].emission[2] = 0;
		bake_cells.write[p_idx].normal[0] = 0;
		bake_cells.write[p_idx].normal[1] = 0;
		bake_cells.write[p_idx].normal[2] = 0;
		bake_cells.write[p_idx].albedo[0] = 0;
		bake_cells.write[p_idx].albedo[1] = 0;
		bake_cells.write[p_idx].albedo[2] = 0;
		if (bake_light.size()) {
			for (int j = 0; j < 6; j++) {
				bake_light.write[p_idx].accum[j][0] = 0;
				bake_light.write[p_idx].accum[j][1] = 0;
				bake_light.write[p_idx].accum[j][2] = 0;
			}
		}

		float alpha_average = 0;
		int children_found = 0;

		for (int i = 0; i < 8; i++) {
			uint32_t child = bake_cells[p_idx].children[i];

			if (child == CHILD_EMPTY) {
				continue;
			}

			_fixup_plot(child, p_level + 1);
			alpha_average += bake_cells[child].alpha;

			if (bake_light.size() > 0) {
				for (int j = 0; j < 6; j++) {
					bake_light.write[p_idx].accum[j][0] += bake_light[child].accum[j][0];
					bake_light.write[p_idx].accum[j][1] += bake_light[child].accum[j][1];
					bake_light.write[p_idx].accum[j][2] += bake_light[child].accum[j][2];
				}
				bake_cells.write[p_idx].emission[0] += bake_cells[child].emission[0];
				bake_cells.write[p_idx].emission[1] += bake_cells[child].emission[1];
				bake_cells.write[p_idx].emission[2] += bake_cells[child].emission[2];
			}

			children_found++;
		}

		bake_cells.write[p_idx].alpha = alpha_average / 8.0;
		if (bake_light.size() && children_found) {
			float divisor = Math::lerp(8, children_found, propagation);
			for (int j = 0; j < 6; j++) {
				bake_light.write[p_idx].accum[j][0] /= divisor;
				bake_light.write[p_idx].accum[j][1] /= divisor;
				bake_light.write[p_idx].accum[j][2] /= divisor;
			}
			bake_cells.write[p_idx].emission[0] /= divisor;
			bake_cells.write[p_idx].emission[1] /= divisor;
			bake_cells.write[p_idx].emission[2] /= divisor;
		}
	}
}

void VoxelLightBaker::begin_bake(int p_subdiv, const AABB &p_bounds) {
	original_bounds = p_bounds;
	cell_subdiv = p_subdiv;
	bake_cells.resize(1);
	material_cache.clear();

	//find out the actual real bounds, power of 2, which gets the highest subdivision
	po2_bounds = p_bounds;
	int longest_axis = po2_bounds.get_longest_axis_index();
	axis_cell_size[longest_axis] = (1 << (cell_subdiv - 1));
	leaf_voxel_count = 0;

	for (int i = 0; i < 3; i++) {
		if (i == longest_axis) {
			continue;
		}

		axis_cell_size[i] = axis_cell_size[longest_axis];
		float axis_size = po2_bounds.size[longest_axis];

		//shrink until fit subdiv
		while (axis_size / 2.0 >= po2_bounds.size[i]) {
			axis_size /= 2.0;
			axis_cell_size[i] >>= 1;
		}

		po2_bounds.size[i] = po2_bounds.size[longest_axis];
	}

	Transform to_bounds;
	to_bounds.basis.scale(Vector3(po2_bounds.size[longest_axis], po2_bounds.size[longest_axis], po2_bounds.size[longest_axis]));
	to_bounds.origin = po2_bounds.position;

	Transform to_grid;
	to_grid.basis.scale(Vector3(axis_cell_size[longest_axis], axis_cell_size[longest_axis], axis_cell_size[longest_axis]));

	to_cell_space = to_grid * to_bounds.affine_inverse();

	cell_size = po2_bounds.size[longest_axis] / axis_cell_size[longest_axis];
}

void VoxelLightBaker::end_bake() {
	_fixup_plot(0, 0);
}

//create the data for visual server

PoolVector<int> VoxelLightBaker::create_gi_probe_data() {
	PoolVector<int> data;

	data.resize(16 + (8 + 1 + 1 + 1 + 1) * bake_cells.size()); //4 for header, rest for rest.

	{
		PoolVector<int>::Write w = data.write();

		uint32_t *w32 = (uint32_t *)w.ptr();

		w32[0] = 0; //version
		w32[1] = cell_subdiv; //subdiv
		w32[2] = axis_cell_size[0];
		w32[3] = axis_cell_size[1];
		w32[4] = axis_cell_size[2];
		w32[5] = bake_cells.size();
		w32[6] = leaf_voxel_count;

		int ofs = 16;

		for (int i = 0; i < bake_cells.size(); i++) {
			for (int j = 0; j < 8; j++) {
				w32[ofs++] = bake_cells[i].children[j];
			}

			{ //albedo
				uint32_t rgba = uint32_t(CLAMP(bake_cells[i].albedo[0] * 255.0, 0, 255)) << 16;
				rgba |= uint32_t(CLAMP(bake_cells[i].albedo[1] * 255.0, 0, 255)) << 8;
				rgba |= uint32_t(CLAMP(bake_cells[i].albedo[2] * 255.0, 0, 255)) << 0;

				w32[ofs++] = rgba;
			}
			{ //emission

				Vector3 e(bake_cells[i].emission[0], bake_cells[i].emission[1], bake_cells[i].emission[2]);
				float l = e.length();
				if (l > 0) {
					e.normalize();
					l = CLAMP(l / 8.0, 0, 1.0);
				}

				uint32_t em = uint32_t(CLAMP(e[0] * 255, 0, 255)) << 24;
				em |= uint32_t(CLAMP(e[1] * 255, 0, 255)) << 16;
				em |= uint32_t(CLAMP(e[2] * 255, 0, 255)) << 8;
				em |= uint32_t(CLAMP(l * 255, 0, 255));

				w32[ofs++] = em;
			}

			//w32[ofs++]=bake_cells[i].used_sides;
			{ //normal

				Vector3 n(bake_cells[i].normal[0], bake_cells[i].normal[1], bake_cells[i].normal[2]);
				n = n * Vector3(0.5, 0.5, 0.5) + Vector3(0.5, 0.5, 0.5);
				uint32_t norm = 0;

				norm |= uint32_t(CLAMP(n.x * 255.0, 0, 255)) << 16;
				norm |= uint32_t(CLAMP(n.y * 255.0, 0, 255)) << 8;
				norm |= uint32_t(CLAMP(n.z * 255.0, 0, 255)) << 0;

				w32[ofs++] = norm;
			}

			{
				uint16_t alpha = MIN(uint32_t(bake_cells[i].alpha * 65535.0), 65535);
				uint16_t level = bake_cells[i].level;

				w32[ofs++] = (uint32_t(level) << 16) | uint32_t(alpha);
			}
		}
	}

	return data;
}

void VoxelLightBaker::_debug_mesh(int p_idx, int p_level, const AABB &p_aabb, Ref<MultiMesh> &p_multimesh, int &idx, DebugMode p_mode) {
	if (p_level == cell_subdiv - 1) {
		Vector3 center = p_aabb.position + p_aabb.size * 0.5;
		Transform xform;
		xform.origin = center;
		xform.basis.scale(p_aabb.size * 0.5);
		p_multimesh->set_instance_transform(idx, xform);
		Color col;
		if (p_mode == DEBUG_ALBEDO) {
			col = Color(bake_cells[p_idx].albedo[0], bake_cells[p_idx].albedo[1], bake_cells[p_idx].albedo[2]);
		} else if (p_mode == DEBUG_LIGHT) {
			for (int i = 0; i < 6; i++) {
				col.r += bake_light[p_idx].accum[i][0];
				col.g += bake_light[p_idx].accum[i][1];
				col.b += bake_light[p_idx].accum[i][2];
				col.r += bake_light[p_idx].direct_accum[i][0];
				col.g += bake_light[p_idx].direct_accum[i][1];
				col.b += bake_light[p_idx].direct_accum[i][2];
			}
		}
		//Color col = Color(bake_cells[p_idx].emission[0], bake_cells[p_idx].emission[1], bake_cells[p_idx].emission[2]);
		p_multimesh->set_instance_color(idx, col);

		idx++;

	} else {
		for (int i = 0; i < 8; i++) {
			uint32_t child = bake_cells[p_idx].children[i];

			if (child == CHILD_EMPTY || child >= (uint32_t)max_original_cells) {
				continue;
			}

			AABB aabb = p_aabb;
			aabb.size *= 0.5;

			if (i & 1) {
				aabb.position.x += aabb.size.x;
			}
			if (i & 2) {
				aabb.position.y += aabb.size.y;
			}
			if (i & 4) {
				aabb.position.z += aabb.size.z;
			}

			_debug_mesh(bake_cells[p_idx].children[i], p_level + 1, aabb, p_multimesh, idx, p_mode);
		}
	}
}

Ref<MultiMesh> VoxelLightBaker::create_debug_multimesh(DebugMode p_mode) {
	Ref<MultiMesh> mm;

	ERR_FAIL_COND_V(p_mode == DEBUG_LIGHT && bake_light.size() == 0, mm);
	mm.instance();

	mm->set_transform_format(MultiMesh::TRANSFORM_3D);
	mm->set_color_format(MultiMesh::COLOR_8BIT);
	mm->set_instance_count(leaf_voxel_count);

	Ref<ArrayMesh> mesh;
	mesh.instance();

	{
		Array arr;
		arr.resize(Mesh::ARRAY_MAX);

		PoolVector<Vector3> vertices;
		PoolVector<Color> colors;
#define ADD_VTX(m_idx)                      \
	;                                       \
	vertices.push_back(face_points[m_idx]); \
	colors.push_back(Color(1, 1, 1, 1));

		for (int i = 0; i < 6; i++) {
			Vector3 face_points[4];

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

			//tri 1
			ADD_VTX(0);
			ADD_VTX(1);
			ADD_VTX(2);
			//tri 2
			ADD_VTX(2);
			ADD_VTX(3);
			ADD_VTX(0);
		}

		arr[Mesh::ARRAY_VERTEX] = vertices;
		arr[Mesh::ARRAY_COLOR] = colors;
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arr);
	}

	{
		Ref<SpatialMaterial> fsm;
		fsm.instance();
		fsm->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
		fsm->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		fsm->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
		fsm->set_albedo(Color(1, 1, 1, 1));

		mesh->surface_set_material(0, fsm);
	}

	mm->set_mesh(mesh);

	int idx = 0;
	_debug_mesh(0, 0, po2_bounds, mm, idx, p_mode);

	return mm;
}

struct VoxelLightBakerOctree {
	enum {
		CHILD_EMPTY = 0xFFFFFFFF
	};

	uint16_t light[6][3]; //anisotropic light
	float alpha;
	uint32_t children[8];
};

PoolVector<uint8_t> VoxelLightBaker::create_capture_octree(int p_subdiv) {
	p_subdiv = MIN(p_subdiv, cell_subdiv); // use the smaller one

	Vector<uint32_t> remap;
	int bc = bake_cells.size();
	remap.resize(bc);
	Vector<uint32_t> demap;

	int new_size = 0;
	for (int i = 0; i < bc; i++) {
		uint32_t c = CHILD_EMPTY;
		if (bake_cells[i].level < p_subdiv) {
			c = new_size;
			new_size++;
			demap.push_back(i);
		}
		remap.write[i] = c;
	}

	Vector<VoxelLightBakerOctree> octree;
	octree.resize(new_size);

	for (int i = 0; i < new_size; i++) {
		octree.write[i].alpha = bake_cells[demap[i]].alpha;
		for (int j = 0; j < 6; j++) {
			for (int k = 0; k < 3; k++) {
				float l = bake_light[demap[i]].accum[j][k]; //add anisotropic light
				l += bake_cells[demap[i]].emission[k]; //add emission
				octree.write[i].light[j][k] = CLAMP(l * 1024, 0, 65535); //give two more bits to octree
			}
		}

		for (int j = 0; j < 8; j++) {
			uint32_t child = bake_cells[demap[i]].children[j];
			octree.write[i].children[j] = child == CHILD_EMPTY ? CHILD_EMPTY : remap[child];
		}
	}

	PoolVector<uint8_t> ret;
	int ret_bytes = octree.size() * sizeof(VoxelLightBakerOctree);
	ret.resize(ret_bytes);
	{
		PoolVector<uint8_t>::Write w = ret.write();
		memcpy(w.ptr(), octree.ptr(), ret_bytes);
	}

	return ret;
}

float VoxelLightBaker::get_cell_size() const {
	return cell_size;
}

Transform VoxelLightBaker::get_to_cell_space_xform() const {
	return to_cell_space;
}
VoxelLightBaker::VoxelLightBaker() {
	color_scan_cell_width = 4;
	bake_texture_size = 128;
	propagation = 0.85;
}
