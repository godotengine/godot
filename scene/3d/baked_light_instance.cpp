/*************************************************************************/
/*  baked_light_instance.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "baked_light_instance.h"
#include "light.h"
#include "math.h"
#include "mesh_instance.h"
#include "scene/scene_string_names.h"

#define FINDMINMAX(x0, x1, x2, min, max) \
	min = max = x0;                      \
	if (x1 < min) min = x1;              \
	if (x1 > max) max = x1;              \
	if (x2 < min) min = x2;              \
	if (x2 > max) max = x2;

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
	if (normal.dot(vmin) + d > 0.0f) return false;
	if (normal.dot(vmax) + d >= 0.0f) return true;

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
	if (min > rad || max < -rad) return false;

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
	if (min > rad || max < -rad) return false;

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
	if (min > rad || max < -rad) return false;

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
	if (min > rad || max < -rad) return false;

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
	if (min > rad || max < -rad) return false;

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
	if (min > rad || max < -rad) return false;

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
	if (min > boxhalfsize.x || max < -boxhalfsize.x) return false;

	/* test in Y-direction */
	FINDMINMAX(v0.y, v1.y, v2.y, min, max);
	if (min > boxhalfsize.y || max < -boxhalfsize.y) return false;

	/* test in Z-direction */
	FINDMINMAX(v0.z, v1.z, v2.z, min, max);
	if (min > boxhalfsize.z || max < -boxhalfsize.z) return false;

	/* Bullet 2: */
	/*  test if the box intersects the plane of the triangle */
	/*  compute plane equation of triangle: normal*x+d=0 */
	normal = e0.cross(e1);
	d = -normal.dot(v0); /* plane eq: normal.x+d=0 */
	if (!planeBoxOverlap(normal, d, boxhalfsize)) return false;

	return true; /* box and triangle overlaps */
}

Vector<Color> BakedLight::_get_bake_texture(Image &p_image, const Color &p_color) {

	Vector<Color> ret;

	if (p_image.empty()) {

		ret.resize(bake_texture_size * bake_texture_size);
		for (int i = 0; i < bake_texture_size * bake_texture_size; i++) {
			ret[i] = p_color;
		}

		return ret;
	}

	p_image.convert(Image::FORMAT_RGBA8);
	p_image.resize(bake_texture_size, bake_texture_size, Image::INTERPOLATE_CUBIC);

	PoolVector<uint8_t>::Read r = p_image.get_data().read();
	ret.resize(bake_texture_size * bake_texture_size);

	for (int i = 0; i < bake_texture_size * bake_texture_size; i++) {
		Color c;
		c.r = r[i * 4 + 0] / 255.0;
		c.g = r[i * 4 + 1] / 255.0;
		c.b = r[i * 4 + 2] / 255.0;
		c.a = r[i * 4 + 3] / 255.0;
		ret[i] = c;
	}

	return ret;
}

BakedLight::MaterialCache BakedLight::_get_material_cache(Ref<Material> p_material) {

	//this way of obtaining materials is inaccurate and also does not support some compressed formats very well
	Ref<SpatialMaterial> mat = p_material;

	Ref<Material> material = mat; //hack for now

	if (material_cache.has(material)) {
		return material_cache[material];
	}

	MaterialCache mc;

	if (mat.is_valid()) {

		Ref<ImageTexture> albedo_tex = mat->get_texture(SpatialMaterial::TEXTURE_ALBEDO);

		Image img_albedo;
		if (albedo_tex.is_valid()) {

			img_albedo = albedo_tex->get_data();
		}

		mc.albedo = _get_bake_texture(img_albedo, mat->get_albedo());

		Ref<ImageTexture> emission_tex = mat->get_texture(SpatialMaterial::TEXTURE_EMISSION);

		Color emission_col = mat->get_emission();
		emission_col.r *= mat->get_emission_energy();
		emission_col.g *= mat->get_emission_energy();
		emission_col.b *= mat->get_emission_energy();

		Image img_emission;

		if (emission_tex.is_valid()) {

			img_emission = emission_tex->get_data();
		}

		mc.emission = _get_bake_texture(img_emission, emission_col);

	} else {
		Image empty;

		mc.albedo = _get_bake_texture(empty, Color(0.7, 0.7, 0.7));
		mc.emission = _get_bake_texture(empty, Color(0, 0, 0));
	}

	material_cache[p_material] = mc;
	return mc;
}

static _FORCE_INLINE_ Vector2 get_uv(const Vector3 &p_pos, const Vector3 *p_vtx, const Vector2 *p_uv) {

	if (p_pos.distance_squared_to(p_vtx[0]) < CMP_EPSILON2)
		return p_uv[0];
	if (p_pos.distance_squared_to(p_vtx[1]) < CMP_EPSILON2)
		return p_uv[1];
	if (p_pos.distance_squared_to(p_vtx[2]) < CMP_EPSILON2)
		return p_uv[2];

	Vector3 v0 = p_vtx[1] - p_vtx[0];
	Vector3 v1 = p_vtx[2] - p_vtx[0];
	Vector3 v2 = p_pos - p_vtx[0];

	float d00 = v0.dot(v0);
	float d01 = v0.dot(v1);
	float d11 = v1.dot(v1);
	float d20 = v2.dot(v0);
	float d21 = v2.dot(v1);
	float denom = (d00 * d11 - d01 * d01);
	if (denom == 0)
		return p_uv[0];
	float v = (d11 * d20 - d01 * d21) / denom;
	float w = (d00 * d21 - d01 * d20) / denom;
	float u = 1.0f - v - w;

	return p_uv[0] * u + p_uv[1] * v + p_uv[2] * w;
}

void BakedLight::_plot_face(int p_idx, int p_level, const Vector3 *p_vtx, const Vector2 *p_uv, const MaterialCache &p_material, const Rect3 &p_aabb) {

	if (p_level == cell_subdiv - 1) {
		//plot the face by guessing it's albedo and emission value

		//find best axis to map to, for scanning values
		int closest_axis;
		float closest_dot;

		Vector3 normal = Plane(p_vtx[0], p_vtx[1], p_vtx[2]).normal;

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
		float alpha = 0.0;

		//map to a grid average in the best axis for this face
		for (int i = 0; i < color_scan_cell_width; i++) {

			Vector3 ofs_i = float(i) * t1;

			for (int j = 0; j < color_scan_cell_width; j++) {

				Vector3 ofs_j = float(j) * t2;

				Vector3 from = p_aabb.pos + ofs_i + ofs_j;
				Vector3 to = from + t1 + t2 + axis * p_aabb.size[closest_axis];
				Vector3 half = (to - from) * 0.5;

				//is in this cell?
				if (!fast_tri_box_overlap(from + half, half, p_vtx)) {
					continue; //face does not span this cell
				}

				//go from -size to +size*2 to avoid skipping collisions
				Vector3 ray_from = from + (t1 + t2) * 0.5 - axis * p_aabb.size[closest_axis];
				Vector3 ray_to = ray_from + axis * p_aabb.size[closest_axis] * 2;

				Vector3 intersection;

				if (!Geometry::ray_intersects_triangle(ray_from, ray_to, p_vtx[0], p_vtx[1], p_vtx[2], &intersection)) {
					//no intersect? look in edges

					float closest_dist = 1e20;
					for (int j = 0; j < 3; j++) {
						Vector3 c;
						Vector3 inters;
						Geometry::get_closest_points_between_segments(p_vtx[j], p_vtx[(j + 1) % 3], ray_from, ray_to, inters, c);
						float d = c.distance_to(intersection);
						if (j == 0 || d < closest_dist) {
							closest_dist = d;
							intersection = inters;
						}
					}
				}

				Vector2 uv = get_uv(intersection, p_vtx, p_uv);

				int uv_x = CLAMP(Math::fposmod(uv.x, 1.0f) * bake_texture_size, 0, bake_texture_size - 1);
				int uv_y = CLAMP(Math::fposmod(uv.y, 1.0f) * bake_texture_size, 0, bake_texture_size - 1);

				int ofs = uv_y * bake_texture_size + uv_x;
				albedo_accum.r += p_material.albedo[ofs].r;
				albedo_accum.g += p_material.albedo[ofs].g;
				albedo_accum.b += p_material.albedo[ofs].b;
				albedo_accum.a += p_material.albedo[ofs].a;

				emission_accum.r += p_material.emission[ofs].r;
				emission_accum.g += p_material.emission[ofs].g;
				emission_accum.b += p_material.emission[ofs].b;
				alpha += 1.0;
			}
		}

		if (alpha == 0) {
			//could not in any way get texture information.. so use closest point to center

			Face3 f(p_vtx[0], p_vtx[1], p_vtx[2]);
			Vector3 inters = f.get_closest_point_to(p_aabb.pos + p_aabb.size * 0.5);

			Vector2 uv = get_uv(inters, p_vtx, p_uv);

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

			zero_alphas++;
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
		}

		//put this temporarily here, corrected in a later step
		bake_cells_write[p_idx].albedo[0] += albedo_accum.r;
		bake_cells_write[p_idx].albedo[1] += albedo_accum.g;
		bake_cells_write[p_idx].albedo[2] += albedo_accum.b;
		bake_cells_write[p_idx].light[0] += emission_accum.r;
		bake_cells_write[p_idx].light[1] += emission_accum.g;
		bake_cells_write[p_idx].light[2] += emission_accum.b;
		bake_cells_write[p_idx].alpha += alpha;

		static const Vector3 side_normals[6] = {
			Vector3(-1, 0, 0),
			Vector3(1, 0, 0),
			Vector3(0, -1, 0),
			Vector3(0, 1, 0),
			Vector3(0, 0, -1),
			Vector3(0, 0, 1),
		};

		for (int i = 0; i < 6; i++) {
			if (normal.dot(side_normals[i]) > CMP_EPSILON) {
				bake_cells_write[p_idx].used_sides |= (1 << i);
			}
		}

	} else {
		//go down
		for (int i = 0; i < 8; i++) {

			Rect3 aabb = p_aabb;
			aabb.size *= 0.5;

			if (i & 1)
				aabb.pos.x += aabb.size.x;
			if (i & 2)
				aabb.pos.y += aabb.size.y;
			if (i & 4)
				aabb.pos.z += aabb.size.z;

			{
				Rect3 test_aabb = aabb;
				//test_aabb.grow_by(test_aabb.get_longest_axis_size()*0.05); //grow a bit to avoid numerical error in real-time
				Vector3 qsize = test_aabb.size * 0.5; //quarter size, for fast aabb test

				if (!fast_tri_box_overlap(test_aabb.pos + qsize, qsize, p_vtx)) {
					//if (!Face3(p_vtx[0],p_vtx[1],p_vtx[2]).intersects_aabb2(aabb)) {
					//does not fit in child, go on
					continue;
				}
			}

			if (bake_cells_write[p_idx].childs[i] == CHILD_EMPTY) {
				//sub cell must be created

				if (bake_cells_used == (1 << bake_cells_alloc)) {
					//exhausted cells, creating more space
					bake_cells_alloc++;
					bake_cells_write = PoolVector<BakeCell>::Write();
					bake_cells.resize(1 << bake_cells_alloc);
					bake_cells_write = bake_cells.write();
				}

				bake_cells_write[p_idx].childs[i] = bake_cells_used;
				bake_cells_level_used[p_level + 1]++;
				bake_cells_used++;
			}

			_plot_face(bake_cells_write[p_idx].childs[i], p_level + 1, p_vtx, p_uv, p_material, aabb);
		}
	}
}

void BakedLight::_fixup_plot(int p_idx, int p_level, int p_x, int p_y, int p_z) {

	if (p_level == cell_subdiv - 1) {

		float alpha = bake_cells_write[p_idx].alpha;

		bake_cells_write[p_idx].albedo[0] /= alpha;
		bake_cells_write[p_idx].albedo[1] /= alpha;
		bake_cells_write[p_idx].albedo[2] /= alpha;

		//transfer emission to light
		bake_cells_write[p_idx].light[0] /= alpha;
		bake_cells_write[p_idx].light[1] /= alpha;
		bake_cells_write[p_idx].light[2] /= alpha;

		bake_cells_write[p_idx].alpha = 1.0;

		//remove neighbours from used sides

		for (int n = 0; n < 6; n++) {

			int ofs[3] = { 0, 0, 0 };

			ofs[n / 2] = (n & 1) ? 1 : -1;

			//convert to x,y,z on this level
			int x = p_x;
			int y = p_y;
			int z = p_z;

			x += ofs[0];
			y += ofs[1];
			z += ofs[2];

			int ofs_x = 0;
			int ofs_y = 0;
			int ofs_z = 0;
			int size = 1 << p_level;
			int half = size / 2;

			if (x < 0 || x >= size || y < 0 || y >= size || z < 0 || z >= size) {
				//neighbour is out, can't use it
				bake_cells_write[p_idx].used_sides &= ~(1 << uint32_t(n));
				continue;
			}

			uint32_t neighbour = 0;

			for (int i = 0; i < cell_subdiv - 1; i++) {

				BakeCell *bc = &bake_cells_write[neighbour];

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

				neighbour = bc->childs[child];
				if (neighbour == CHILD_EMPTY) {
					break;
				}

				half >>= 1;
			}

			if (neighbour != CHILD_EMPTY) {
				bake_cells_write[p_idx].used_sides &= ~(1 << uint32_t(n));
			}
		}
	} else {

		//go down

		float alpha_average = 0;
		int half = cells_per_axis >> (p_level + 1);
		for (int i = 0; i < 8; i++) {

			uint32_t child = bake_cells_write[p_idx].childs[i];

			if (child == CHILD_EMPTY)
				continue;

			int nx = p_x;
			int ny = p_y;
			int nz = p_z;

			if (i & 1)
				nx += half;
			if (i & 2)
				ny += half;
			if (i & 4)
				nz += half;

			_fixup_plot(child, p_level + 1, nx, ny, nz);
			alpha_average += bake_cells_write[child].alpha;
		}

		bake_cells_write[p_idx].alpha = alpha_average / 8.0;
		bake_cells_write[p_idx].light[0] = 0;
		bake_cells_write[p_idx].light[1] = 0;
		bake_cells_write[p_idx].light[2] = 0;
		bake_cells_write[p_idx].albedo[0] = 0;
		bake_cells_write[p_idx].albedo[1] = 0;
		bake_cells_write[p_idx].albedo[2] = 0;
	}

	//clean up light
	bake_cells_write[p_idx].light_pass = 0;
	//find neighbours
}

void BakedLight::_bake_add_mesh(const Transform &p_xform, Ref<Mesh> &p_mesh) {

	for (int i = 0; i < p_mesh->get_surface_count(); i++) {

		if (p_mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES)
			continue; //only triangles

		MaterialCache material = _get_material_cache(p_mesh->surface_get_material(i));

		Array a = p_mesh->surface_get_arrays(i);

		PoolVector<Vector3> vertices = a[Mesh::ARRAY_VERTEX];
		PoolVector<Vector3>::Read vr = vertices.read();
		PoolVector<Vector2> uv = a[Mesh::ARRAY_TEX_UV];
		PoolVector<Vector2>::Read uvr;
		PoolVector<int> index = a[Mesh::ARRAY_INDEX];

		bool read_uv = false;

		if (uv.size()) {

			uvr = uv.read();
			read_uv = true;
		}

		if (index.size()) {

			int facecount = index.size() / 3;
			PoolVector<int>::Read ir = index.read();

			for (int j = 0; j < facecount; j++) {

				Vector3 vtxs[3];
				Vector2 uvs[3];

				for (int k = 0; k < 3; k++) {
					vtxs[k] = p_xform.xform(vr[ir[j * 3 + k]]);
				}

				if (read_uv) {
					for (int k = 0; k < 3; k++) {
						uvs[k] = uvr[ir[j * 3 + k]];
					}
				}

				//plot face
				_plot_face(0, 0, vtxs, uvs, material, bounds);
			}

		} else {

			int facecount = vertices.size() / 3;

			for (int j = 0; j < facecount; j++) {

				Vector3 vtxs[3];
				Vector2 uvs[3];

				for (int k = 0; k < 3; k++) {
					vtxs[k] = p_xform.xform(vr[j * 3 + k]);
				}

				if (read_uv) {
					for (int k = 0; k < 3; k++) {
						uvs[k] = uvr[j * 3 + k];
					}
				}

				//plot face
				_plot_face(0, 0, vtxs, uvs, material, bounds);
			}
		}
	}
}

void BakedLight::_bake_add_to_aabb(const Transform &p_xform, Ref<Mesh> &p_mesh, bool &first) {

	for (int i = 0; i < p_mesh->get_surface_count(); i++) {

		if (p_mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES)
			continue; //only triangles

		Array a = p_mesh->surface_get_arrays(i);
		PoolVector<Vector3> vertices = a[Mesh::ARRAY_VERTEX];
		int vc = vertices.size();
		PoolVector<Vector3>::Read vr = vertices.read();

		if (first) {
			bounds.pos = p_xform.xform(vr[0]);
			first = false;
		}

		for (int j = 0; j < vc; j++) {
			bounds.expand_to(p_xform.xform(vr[j]));
		}
	}
}

void BakedLight::bake() {

	bake_cells_alloc = 16;
	bake_cells.resize(1 << bake_cells_alloc);
	bake_cells_used = 1;
	cells_per_axis = (1 << (cell_subdiv - 1));
	zero_alphas = 0;

	bool aabb_first = true;
	print_line("Generating AABB");

	bake_cells_level_used.resize(cell_subdiv);
	for (int i = 0; i < cell_subdiv; i++) {
		bake_cells_level_used[i] = 0;
	}

	int count = 0;
	for (Set<GeometryInstance *>::Element *E = geometries.front(); E; E = E->next()) {

		print_line("aabb geom " + itos(count) + "/" + itos(geometries.size()));

		GeometryInstance *geom = E->get();

		if (geom->cast_to<MeshInstance>()) {

			MeshInstance *mesh_instance = geom->cast_to<MeshInstance>();
			Ref<Mesh> mesh = mesh_instance->get_mesh();
			if (mesh.is_valid()) {

				_bake_add_to_aabb(geom->get_relative_transform(this), mesh, aabb_first);
			}
		}
		count++;
	}

	print_line("AABB: " + bounds);
	ERR_FAIL_COND(aabb_first);

	bake_cells_write = bake_cells.write();
	count = 0;

	for (Set<GeometryInstance *>::Element *E = geometries.front(); E; E = E->next()) {

		GeometryInstance *geom = E->get();
		print_line("plot geom " + itos(count) + "/" + itos(geometries.size()));

		if (geom->cast_to<MeshInstance>()) {

			MeshInstance *mesh_instance = geom->cast_to<MeshInstance>();
			Ref<Mesh> mesh = mesh_instance->get_mesh();
			if (mesh.is_valid()) {

				_bake_add_mesh(geom->get_relative_transform(this), mesh);
			}
		}

		count++;
	}

	_fixup_plot(0, 0, 0, 0, 0);

	bake_cells_write = PoolVector<BakeCell>::Write();

	bake_cells.resize(bake_cells_used);

	print_line("total bake cells used: " + itos(bake_cells_used));
	for (int i = 0; i < cell_subdiv; i++) {
		print_line("level " + itos(i) + ": " + itos(bake_cells_level_used[i]));
	}
	print_line("zero alphas: " + itos(zero_alphas));
}

void BakedLight::_bake_directional(int p_idx, int p_level, int p_x, int p_y, int p_z, const Vector3 &p_dir, const Color &p_color, int p_sign) {

	if (p_level == cell_subdiv - 1) {

		Vector3 end;
		end.x = float(p_x + 0.5) / cells_per_axis;
		end.y = float(p_y + 0.5) / cells_per_axis;
		end.z = float(p_z + 0.5) / cells_per_axis;

		end = bounds.pos + bounds.size * end;

		float max_ray_len = (bounds.size).length() * 1.2;

		Vector3 begin = end + max_ray_len * -p_dir;

		//clip begin

		for (int i = 0; i < 3; i++) {

			if (ABS(p_dir[i]) < CMP_EPSILON) {
				continue; // parallel to axis, don't clip
			}

			Plane p;
			p.normal[i] = 1.0;
			p.d = bounds.pos[i];
			if (p_dir[i] < 0) {
				p.d += bounds.size[i];
			}

			Vector3 inters;
			if (p.intersects_segment(end, begin, &inters)) {
				begin = inters;
			}
		}

		int idx = _plot_ray(begin, end);

		if (idx >= 0 && light_pass != bake_cells_write[idx].light_pass) {
			//hit something, add or remove light to it

			Color albedo = Color(bake_cells_write[idx].albedo[0], bake_cells_write[idx].albedo[1], bake_cells_write[idx].albedo[2]);
			bake_cells_write[idx].light[0] += albedo.r * p_color.r * p_sign;
			bake_cells_write[idx].light[1] += albedo.g * p_color.g * p_sign;
			bake_cells_write[idx].light[2] += albedo.b * p_color.b * p_sign;
			bake_cells_write[idx].light_pass = light_pass;
		}

	} else {

		int half = cells_per_axis >> (p_level + 1);

		//go down
		for (int i = 0; i < 8; i++) {

			uint32_t child = bake_cells_write[p_idx].childs[i];

			if (child == CHILD_EMPTY)
				continue;

			int nx = p_x;
			int ny = p_y;
			int nz = p_z;

			if (i & 1)
				nx += half;
			if (i & 2)
				ny += half;
			if (i & 4)
				nz += half;

			_bake_directional(child, p_level + 1, nx, ny, nz, p_dir, p_color, p_sign);
		}
	}
}

void BakedLight::_bake_light(Light *p_light) {

	if (p_light->cast_to<DirectionalLight>()) {

		DirectionalLight *dl = p_light->cast_to<DirectionalLight>();

		Transform rel_xf = dl->get_relative_transform(this);

		Vector3 light_dir = -rel_xf.basis.get_axis(2);

		Color color = dl->get_color();
		float nrg = dl->get_param(Light::PARAM_ENERGY);
		color.r *= nrg;
		color.g *= nrg;
		color.b *= nrg;

		light_pass++;
		_bake_directional(0, 0, 0, 0, 0, light_dir, color, 1);
	}
}

void BakedLight::_upscale_light(int p_idx, int p_level) {

	//go down

	float light_accum[3] = { 0, 0, 0 };
	float alpha_accum = 0;

	bool check_children = p_level < (cell_subdiv - 2);

	for (int i = 0; i < 8; i++) {

		uint32_t child = bake_cells_write[p_idx].childs[i];

		if (child == CHILD_EMPTY)
			continue;

		if (check_children) {
			_upscale_light(child, p_level + 1);
		}

		light_accum[0] += bake_cells_write[child].light[0];
		light_accum[1] += bake_cells_write[child].light[1];
		light_accum[2] += bake_cells_write[child].light[2];
		alpha_accum += bake_cells_write[child].alpha;
	}

	bake_cells_write[p_idx].light[0] = light_accum[0] / 8.0;
	bake_cells_write[p_idx].light[1] = light_accum[1] / 8.0;
	bake_cells_write[p_idx].light[2] = light_accum[2] / 8.0;
	bake_cells_write[p_idx].alpha = alpha_accum / 8.0;
}

void BakedLight::bake_lights() {

	ERR_FAIL_COND(bake_cells.size() == 0);

	bake_cells_write = bake_cells.write();

	for (Set<Light *>::Element *E = lights.front(); E; E = E->next()) {

		_bake_light(E->get());
	}

	_upscale_light(0, 0);

	bake_cells_write = PoolVector<BakeCell>::Write();
}

Color BakedLight::_cone_trace(const Vector3 &p_from, const Vector3 &p_dir, float p_half_angle) {

	Color color(0, 0, 0, 0);
	float tha = Math::tan(p_half_angle); //tan half angle
	Vector3 from = (p_from - bounds.pos) / bounds.size; //convert to 0..1
	from /= cells_per_axis; //convert to voxels of size 1
	Vector3 dir = (p_dir / bounds.size).normalized();

	float max_dist = Vector3(cells_per_axis, cells_per_axis, cells_per_axis).length();

	float dist = 1.0;
	// self occlusion in flat surfaces

	float alpha = 0;

	while (dist < max_dist && alpha < 0.95) {

#if 0
		// smallest sample diameter possible is the voxel size
		float diameter = MAX(1.0, 2.0 * tha * dist);
		float lod = log2(diameter);

		Vector3 sample_pos = from + dist * dir;


		Color samples_base[2][8]={{Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0)},
		{Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0),Color(0,0,0,0)}};

		float levelf = Math::fposmod(lod,1.0);
		float fx = Math::fposmod(sample_pos.x,1.0);
		float fy = Math::fposmod(sample_pos.y,1.0);
		float fz = Math::fposmod(sample_pos.z,1.0);

		for(int l=0;l<2;l++){

			int bx = Math::floor(sample_pos.x);
			int by = Math::floor(sample_pos.y);
			int bz = Math::floor(sample_pos.z);

			int lodn=int(Math::floor(lod))-l;

			bx>>=lodn;
			by>>=lodn;
			bz>>=lodn;

			int limit = MAX(0,cell_subdiv-lodn-1);

			for(int c=0;c<8;c++) {

				int x = bx;
				int y = by;
				int z = bz;

				if (c&1) {
					x+=1;
				}
				if (c&2) {
					y+=1;
				}
				if (c&4) {
					z+=1;
				}

				int ofs_x=0;
				int ofs_y=0;
				int ofs_z=0;
				int size = cells_per_axis>>lodn;
				int half=size/2;

				bool outside=x<0 || x>=size || y<0 || y>=size || z<0 || z>=size;

				if (outside)
					continue;


				uint32_t cell=0;

				for(int i=0;i<limit;i++) {

					BakeCell *bc = &bake_cells_write[cell];

					int child = 0;
					if (x >= ofs_x + half) {
						child|=1;
						ofs_x+=half;
					}
					if (y >= ofs_y + half) {
						child|=2;
						ofs_y+=half;
					}
					if (z >= ofs_z + half) {
						child|=4;
						ofs_z+=half;
					}

					cell = bc->childs[child];
					if (cell==CHILD_EMPTY)
						break;

					half>>=1;
				}

				if (cell!=CHILD_EMPTY) {

					samples_base[l][c].r=bake_cells_write[cell].light[0];
					samples_base[l][c].g=bake_cells_write[cell].light[1];
					samples_base[l][c].b=bake_cells_write[cell].light[2];
					samples_base[l][c].a=bake_cells_write[cell].alpha;
				}

			}


		}

		Color m0x0 = samples_base[0][0].linear_interpolate(samples_base[0][1],fx);
		Color m0x1 = samples_base[0][2].linear_interpolate(samples_base[0][3],fx);
		Color m0y0 = m0x0.linear_interpolate(m0x1,fy);
		m0x0 = samples_base[0][4].linear_interpolate(samples_base[0][5],fx);
		m0x1 = samples_base[0][6].linear_interpolate(samples_base[0][7],fx);
		Color m0y1 = m0x0.linear_interpolate(m0x1,fy);
		Color m0z = m0y0.linear_interpolate(m0y1,fz);

		Color m1x0 = samples_base[1][0].linear_interpolate(samples_base[1][1],fx);
		Color m1x1 = samples_base[1][2].linear_interpolate(samples_base[1][3],fx);
		Color m1y0 = m1x0.linear_interpolate(m1x1,fy);
		m1x0 = samples_base[1][4].linear_interpolate(samples_base[1][5],fx);
		m1x1 = samples_base[1][6].linear_interpolate(samples_base[1][7],fx);
		Color m1y1 = m1x0.linear_interpolate(m1x1,fy);
		Color m1z = m1y0.linear_interpolate(m1y1,fz);

		Color m = m0z.linear_interpolate(m1z,levelf);
#else
		float diameter = 1.0;
		Vector3 sample_pos = from + dist * dir;

		Color m(0, 0, 0, 0);
		{
			int x = Math::floor(sample_pos.x);
			int y = Math::floor(sample_pos.y);
			int z = Math::floor(sample_pos.z);

			int ofs_x = 0;
			int ofs_y = 0;
			int ofs_z = 0;
			int size = cells_per_axis;
			int half = size / 2;

			bool outside = x < 0 || x >= size || y < 0 || y >= size || z < 0 || z >= size;

			if (!outside) {

				uint32_t cell = 0;

				for (int i = 0; i < cell_subdiv - 1; i++) {

					BakeCell *bc = &bake_cells_write[cell];

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

					cell = bc->childs[child];
					if (cell == CHILD_EMPTY)
						break;

					half >>= 1;
				}

				if (cell != CHILD_EMPTY) {

					m.r = bake_cells_write[cell].light[0];
					m.g = bake_cells_write[cell].light[1];
					m.b = bake_cells_write[cell].light[2];
					m.a = bake_cells_write[cell].alpha;
				}
			}
		}

#endif
		// front-to-back compositing
		float a = (1.0 - alpha);
		color.r += a * m.r;
		color.g += a * m.g;
		color.b += a * m.b;
		alpha += a * m.a;
		//occlusion += a * voxelColor.a;
		//occlusion += (a * voxelColor.a) / (1.0 + 0.03 * diameter);
		dist += diameter * 0.5; // smoother
		//dist += diameter; // faster but misses more voxels
	}

	return color;
}

void BakedLight::_bake_radiance(int p_idx, int p_level, int p_x, int p_y, int p_z) {

	if (p_level == cell_subdiv - 1) {

		const int NUM_CONES = 6;
		Vector3 cone_directions[6] = {
			Vector3(1, 0, 0),
			Vector3(0.5, 0.866025, 0),
			Vector3(0.5, 0.267617, 0.823639),
			Vector3(0.5, -0.700629, 0.509037),
			Vector3(0.5, -0.700629, -0.509037),
			Vector3(0.5, 0.267617, -0.823639)
		};
		float coneWeights[6] = { 0.25, 0.15, 0.15, 0.15, 0.15, 0.15 };

		Vector3 pos = (Vector3(p_x, p_y, p_z) / float(cells_per_axis)) * bounds.size + bounds.pos;
		Vector3 voxel_size = bounds.size / float(cells_per_axis);
		pos += voxel_size * 0.5;

		Color accum;

		bake_cells_write[p_idx].light[0] = 0;
		bake_cells_write[p_idx].light[1] = 0;
		bake_cells_write[p_idx].light[2] = 0;

		int freepix = 0;
		for (int i = 0; i < 6; i++) {

			if (!(bake_cells_write[p_idx].used_sides & (1 << i)))
				continue;

			if ((i & 1) == 0)
				bake_cells_write[p_idx].light[i / 2] = 1.0;
			freepix++;
			continue;

			int ofs = i / 2;

			Vector3 dir;
			if ((i & 1) == 0)
				dir[ofs] = 1.0;
			else
				dir[ofs] = -1.0;

			for (int j = 0; j < 1; j++) {

				Vector3 cone_dir;
				cone_dir.x = cone_directions[j][(ofs + 0) % 3];
				cone_dir.y = cone_directions[j][(ofs + 1) % 3];
				cone_dir.z = cone_directions[j][(ofs + 2) % 3];

				cone_dir[ofs] *= dir[ofs];

				Color res = _cone_trace(pos + dir * voxel_size, cone_dir, Math::deg2rad(29.9849));
				accum.r += res.r; //*coneWeights[j];
				accum.g += res.g; //*coneWeights[j];
				accum.b += res.b; //*coneWeights[j];
			}
		}
#if 0
		if (freepix==0) {
			bake_cells_write[p_idx].light[0]=0;
			bake_cells_write[p_idx].light[1]=0;
			bake_cells_write[p_idx].light[2]=0;
		}

		if (freepix==1) {
			bake_cells_write[p_idx].light[0]=1;
			bake_cells_write[p_idx].light[1]=0;
			bake_cells_write[p_idx].light[2]=0;
		}

		if (freepix==2) {
			bake_cells_write[p_idx].light[0]=0;
			bake_cells_write[p_idx].light[1]=1;
			bake_cells_write[p_idx].light[2]=0;
		}

		if (freepix==3) {
			bake_cells_write[p_idx].light[0]=1;
			bake_cells_write[p_idx].light[1]=1;
			bake_cells_write[p_idx].light[2]=0;
		}

		if (freepix==4) {
			bake_cells_write[p_idx].light[0]=0;
			bake_cells_write[p_idx].light[1]=0;
			bake_cells_write[p_idx].light[2]=1;
		}

		if (freepix==5) {
			bake_cells_write[p_idx].light[0]=1;
			bake_cells_write[p_idx].light[1]=0;
			bake_cells_write[p_idx].light[2]=1;
		}

		if (freepix==6) {
			bake_cells_write[p_idx].light[0]=0;
			bake_cells_write[p_idx].light[0]=1;
			bake_cells_write[p_idx].light[0]=1;
		}
#endif
		//bake_cells_write[p_idx].radiance[0]=accum.r;
		//bake_cells_write[p_idx].radiance[1]=accum.g;
		//bake_cells_write[p_idx].radiance[2]=accum.b;

	} else {

		int half = cells_per_axis >> (p_level + 1);

		//go down
		for (int i = 0; i < 8; i++) {

			uint32_t child = bake_cells_write[p_idx].childs[i];

			if (child == CHILD_EMPTY)
				continue;

			int nx = p_x;
			int ny = p_y;
			int nz = p_z;

			if (i & 1)
				nx += half;
			if (i & 2)
				ny += half;
			if (i & 4)
				nz += half;

			_bake_radiance(child, p_level + 1, nx, ny, nz);
		}
	}
}

void BakedLight::bake_radiance() {

	ERR_FAIL_COND(bake_cells.size() == 0);

	bake_cells_write = bake_cells.write();

	_bake_radiance(0, 0, 0, 0, 0);

	bake_cells_write = PoolVector<BakeCell>::Write();
}
int BakedLight::_find_cell(int x, int y, int z) {

	uint32_t cell = 0;

	int ofs_x = 0;
	int ofs_y = 0;
	int ofs_z = 0;
	int size = cells_per_axis;
	int half = size / 2;

	if (x < 0 || x >= size)
		return -1;
	if (y < 0 || y >= size)
		return -1;
	if (z < 0 || z >= size)
		return -1;

	for (int i = 0; i < cell_subdiv - 1; i++) {

		BakeCell *bc = &bake_cells_write[cell];

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

		cell = bc->childs[child];
		if (cell == CHILD_EMPTY)
			return -1;

		half >>= 1;
	}

	return cell;
}

int BakedLight::_plot_ray(const Vector3 &p_from, const Vector3 &p_to) {

	Vector3 from = (p_from - bounds.pos) / bounds.size;
	Vector3 to = (p_to - bounds.pos) / bounds.size;

	int x1 = Math::floor(from.x * cells_per_axis);
	int y1 = Math::floor(from.y * cells_per_axis);
	int z1 = Math::floor(from.z * cells_per_axis);

	int x2 = Math::floor(to.x * cells_per_axis);
	int y2 = Math::floor(to.y * cells_per_axis);
	int z2 = Math::floor(to.z * cells_per_axis);

	int i, dx, dy, dz, l, m, n, x_inc, y_inc, z_inc, err_1, err_2, dx2, dy2, dz2;
	int point[3];

	point[0] = x1;
	point[1] = y1;
	point[2] = z1;
	dx = x2 - x1;
	dy = y2 - y1;
	dz = z2 - z1;
	x_inc = (dx < 0) ? -1 : 1;
	l = ABS(dx);
	y_inc = (dy < 0) ? -1 : 1;
	m = ABS(dy);
	z_inc = (dz < 0) ? -1 : 1;
	n = ABS(dz);
	dx2 = l << 1;
	dy2 = m << 1;
	dz2 = n << 1;

	if ((l >= m) && (l >= n)) {
		err_1 = dy2 - l;
		err_2 = dz2 - l;
		for (i = 0; i < l; i++) {
			int cell = _find_cell(point[0], point[1], point[2]);
			if (cell >= 0)
				return cell;

			if (err_1 > 0) {
				point[1] += y_inc;
				err_1 -= dx2;
			}
			if (err_2 > 0) {
				point[2] += z_inc;
				err_2 -= dx2;
			}
			err_1 += dy2;
			err_2 += dz2;
			point[0] += x_inc;
		}
	} else if ((m >= l) && (m >= n)) {
		err_1 = dx2 - m;
		err_2 = dz2 - m;
		for (i = 0; i < m; i++) {
			int cell = _find_cell(point[0], point[1], point[2]);
			if (cell >= 0)
				return cell;
			if (err_1 > 0) {
				point[0] += x_inc;
				err_1 -= dy2;
			}
			if (err_2 > 0) {
				point[2] += z_inc;
				err_2 -= dy2;
			}
			err_1 += dx2;
			err_2 += dz2;
			point[1] += y_inc;
		}
	} else {
		err_1 = dy2 - n;
		err_2 = dx2 - n;
		for (i = 0; i < n; i++) {
			int cell = _find_cell(point[0], point[1], point[2]);
			if (cell >= 0)
				return cell;

			if (err_1 > 0) {
				point[1] += y_inc;
				err_1 -= dz2;
			}
			if (err_2 > 0) {
				point[0] += x_inc;
				err_2 -= dz2;
			}
			err_1 += dy2;
			err_2 += dx2;
			point[2] += z_inc;
		}
	}
	return _find_cell(point[0], point[1], point[2]);
}

void BakedLight::set_cell_subdiv(int p_subdiv) {

	cell_subdiv = p_subdiv;

	//VS::get_singleton()->baked_light_set_subdivision(baked_light,p_subdiv);
}

int BakedLight::get_cell_subdiv() const {

	return cell_subdiv;
}

Rect3 BakedLight::get_aabb() const {

	return Rect3(Vector3(0, 0, 0), Vector3(1, 1, 1));
}
PoolVector<Face3> BakedLight::get_faces(uint32_t p_usage_flags) const {

	return PoolVector<Face3>();
}

String BakedLight::get_configuration_warning() const {
	return String();
}

void BakedLight::_debug_mesh(int p_idx, int p_level, const Rect3 &p_aabb, DebugMode p_mode, Ref<MultiMesh> &p_multimesh, int &idx) {

	if (p_level == cell_subdiv - 1) {

		Vector3 center = p_aabb.pos + p_aabb.size * 0.5;
		Transform xform;
		xform.origin = center;
		xform.basis.scale(p_aabb.size * 0.5);
		p_multimesh->set_instance_transform(idx, xform);
		Color col;
		switch (p_mode) {
			case DEBUG_ALBEDO: {
				col = Color(bake_cells_write[p_idx].albedo[0], bake_cells_write[p_idx].albedo[1], bake_cells_write[p_idx].albedo[2]);
			} break;
			case DEBUG_LIGHT: {
				col = Color(bake_cells_write[p_idx].light[0], bake_cells_write[p_idx].light[1], bake_cells_write[p_idx].light[2]);
				Color colr = Color(bake_cells_write[p_idx].radiance[0], bake_cells_write[p_idx].radiance[1], bake_cells_write[p_idx].radiance[2]);
				col.r += colr.r;
				col.g += colr.g;
				col.b += colr.b;
			} break;
		}
		p_multimesh->set_instance_color(idx, col);

		idx++;

	} else {

		for (int i = 0; i < 8; i++) {

			if (bake_cells_write[p_idx].childs[i] == CHILD_EMPTY)
				continue;

			Rect3 aabb = p_aabb;
			aabb.size *= 0.5;

			if (i & 1)
				aabb.pos.x += aabb.size.x;
			if (i & 2)
				aabb.pos.y += aabb.size.y;
			if (i & 4)
				aabb.pos.z += aabb.size.z;

			_debug_mesh(bake_cells_write[p_idx].childs[i], p_level + 1, aabb, p_mode, p_multimesh, idx);
		}
	}
}

void BakedLight::create_debug_mesh(DebugMode p_mode) {

	Ref<MultiMesh> mm;
	mm.instance();

	mm->set_transform_format(MultiMesh::TRANSFORM_3D);
	mm->set_color_format(MultiMesh::COLOR_8BIT);
	mm->set_instance_count(bake_cells_level_used[cell_subdiv - 1]);

	Ref<Mesh> mesh;
	mesh.instance();

	{
		Array arr;
		arr.resize(Mesh::ARRAY_MAX);

		PoolVector<Vector3> vertices;
		PoolVector<Color> colors;

		int vtx_idx = 0;
#define ADD_VTX(m_idx)                      \
	;                                       \
	vertices.push_back(face_points[m_idx]); \
	colors.push_back(Color(1, 1, 1, 1));    \
	vtx_idx++;

		for (int i = 0; i < 6; i++) {

			Vector3 face_points[4];

			for (int j = 0; j < 4; j++) {

				float v[3];
				v[0] = 1.0;
				v[1] = 1 - 2 * ((j >> 1) & 1);
				v[2] = v[1] * (1 - 2 * (j & 1));

				for (int k = 0; k < 3; k++) {

					if (i < 3)
						face_points[j][(i + k) % 3] = v[k] * (i >= 3 ? -1 : 1);
					else
						face_points[3 - j][(i + k) % 3] = v[k] * (i >= 3 ? -1 : 1);
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

	bake_cells_write = bake_cells.write();

	int idx = 0;
	_debug_mesh(0, 0, bounds, p_mode, mm, idx);

	print_line("written: " + itos(idx) + " total: " + itos(bake_cells_level_used[cell_subdiv - 1]));

	MultiMeshInstance *mmi = memnew(MultiMeshInstance);
	mmi->set_multimesh(mm);
	add_child(mmi);
#ifdef TOOLS_ENABLED
	if (get_tree()->get_edited_scene_root() == this) {
		mmi->set_owner(this);
	} else {
		mmi->set_owner(get_owner());
	}
#else
	mmi->set_owner(get_owner());
#endif
}

void BakedLight::_debug_mesh_albedo() {
	create_debug_mesh(DEBUG_ALBEDO);
}

void BakedLight::_debug_mesh_light() {
	create_debug_mesh(DEBUG_LIGHT);
}

void BakedLight::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_cell_subdiv", "steps"), &BakedLight::set_cell_subdiv);
	ClassDB::bind_method(D_METHOD("get_cell_subdiv"), &BakedLight::get_cell_subdiv);

	ClassDB::bind_method(D_METHOD("bake"), &BakedLight::bake);
	ClassDB::set_method_flags(get_class_static(), _scs_create("bake"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ClassDB::bind_method(D_METHOD("bake_lights"), &BakedLight::bake_lights);
	ClassDB::set_method_flags(get_class_static(), _scs_create("bake_lights"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ClassDB::bind_method(D_METHOD("bake_radiance"), &BakedLight::bake_radiance);
	ClassDB::set_method_flags(get_class_static(), _scs_create("bake_radiance"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ClassDB::bind_method(D_METHOD("debug_mesh_albedo"), &BakedLight::_debug_mesh_albedo);
	ClassDB::set_method_flags(get_class_static(), _scs_create("debug_mesh_albedo"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ClassDB::bind_method(D_METHOD("debug_mesh_light"), &BakedLight::_debug_mesh_light);
	ClassDB::set_method_flags(get_class_static(), _scs_create("debug_mesh_light"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "cell_subdiv"), "set_cell_subdiv", "get_cell_subdiv");
	ADD_SIGNAL(MethodInfo("baked_light_changed"));
}

BakedLight::BakedLight() {

	//baked_light=VisualServer::get_singleton()->baked_light_create();
	VS::get_singleton()->instance_set_base(get_instance(), baked_light);

	cell_subdiv = 8;
	bake_texture_size = 128;
	color_scan_cell_width = 8;
	light_pass = 0;
}

BakedLight::~BakedLight() {

	VS::get_singleton()->free(baked_light);
}

/////////////////////////

#if 0
void BakedLightSampler::set_param(Param p_param,float p_value) {
	ERR_FAIL_INDEX(p_param,PARAM_MAX);
	params[p_param]=p_value;
	VS::get_singleton()->baked_light_sampler_set_param(base,VS::BakedLightSamplerParam(p_param),p_value);
}

float BakedLightSampler::get_param(Param p_param) const{

	ERR_FAIL_INDEX_V(p_param,PARAM_MAX,0);
	return params[p_param];

}

void BakedLightSampler::set_resolution(int p_resolution){

    ERR_FAIL_COND(p_resolution<4 || p_resolution>32);
	resolution=p_resolution;
	VS::get_singleton()->baked_light_sampler_set_resolution(base,resolution);
}
int BakedLightSampler::get_resolution() const {

	return resolution;
}

AABB BakedLightSampler::get_aabb() const {

	float r = get_param(PARAM_RADIUS);
	return AABB( Vector3(-r,-r,-r),Vector3(r*2,r*2,r*2));
}
DVector<Face3> BakedLightSampler::get_faces(uint32_t p_usage_flags) const {
	return DVector<Face3>();
}

void BakedLightSampler::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_param","param","value"),&BakedLightSampler::set_param);
	ClassDB::bind_method(D_METHOD("get_param","param"),&BakedLightSampler::get_param);

	ClassDB::bind_method(D_METHOD("set_resolution","resolution"),&BakedLightSampler::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution"),&BakedLightSampler::get_resolution);


	BIND_CONSTANT( PARAM_RADIUS );
	BIND_CONSTANT( PARAM_STRENGTH );
	BIND_CONSTANT( PARAM_ATTENUATION );
	BIND_CONSTANT( PARAM_DETAIL_RATIO );
	BIND_CONSTANT( PARAM_MAX );

	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"params/radius",PROPERTY_HINT_RANGE,"0.01,1024,0.01"),"set_param","get_param",PARAM_RADIUS);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"params/strength",PROPERTY_HINT_RANGE,"0.01,16,0.01"),"set_param","get_param",PARAM_STRENGTH);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"params/attenuation",PROPERTY_HINT_EXP_EASING),"set_param","get_param",PARAM_ATTENUATION);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"params/detail_ratio",PROPERTY_HINT_RANGE,"0.01,1.0,0.01"),"set_param","get_param",PARAM_DETAIL_RATIO);
	//ADD_PROPERTYI( PropertyInfo(Variant::REAL,"params/detail_ratio",PROPERTY_HINT_RANGE,"0,20,1"),"set_param","get_param",PARAM_DETAIL_RATIO);
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"params/resolution",PROPERTY_HINT_RANGE,"4,32,1"),"set_resolution","get_resolution");

}

BakedLightSampler::BakedLightSampler() {

	base = VS::get_singleton()->baked_light_sampler_create();
	set_base(base);

	params[PARAM_RADIUS]=1.0;
	params[PARAM_STRENGTH]=1.0;
	params[PARAM_ATTENUATION]=1.0;
	params[PARAM_DETAIL_RATIO]=0.1;
	resolution=16;


}

BakedLightSampler::~BakedLightSampler(){

	VS::get_singleton()->free(base);
}
#endif
