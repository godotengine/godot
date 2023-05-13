/**************************************************************************/
/*  renderer_canvas_render.cpp                                            */
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

#include "renderer_canvas_render.h"
#include "servers/rendering/rendering_server_globals.h"

#include "core/math/geometry_2d.h"

const Rect2 &RendererCanvasRender::Item::get_rect() const {
	if (custom_rect || (!rect_dirty && !update_when_visible && skeleton == RID())) {
		return rect;
	}

	//must update rect

	if (commands == nullptr) {
		rect = Rect2();
		rect_dirty = false;
		return rect;
	}

	Transform2D xf;
	bool found_xform = false;
	bool first = true;

	const Item::Command *c = commands;

	while (c) {
		Rect2 r;

		switch (c->type) {
			case Item::Command::TYPE_RECT: {
				const Item::CommandRect *crect = static_cast<const Item::CommandRect *>(c);
				r = crect->rect;

			} break;
			case Item::Command::TYPE_NINEPATCH: {
				const Item::CommandNinePatch *style = static_cast<const Item::CommandNinePatch *>(c);
				r = style->rect;
			} break;

			case Item::Command::TYPE_POLYGON: {
				const Item::CommandPolygon *polygon = static_cast<const Item::CommandPolygon *>(c);
				r = polygon->polygon.rect_cache;
			} break;
			case Item::Command::TYPE_PRIMITIVE: {
				const Item::CommandPrimitive *primitive = static_cast<const Item::CommandPrimitive *>(c);
				for (uint32_t j = 0; j < primitive->point_count; j++) {
					if (j == 0) {
						r.position = primitive->points[0];
					} else {
						r.expand_to(primitive->points[j]);
					}
				}
			} break;
			case Item::Command::TYPE_MESH: {
				const Item::CommandMesh *mesh = static_cast<const Item::CommandMesh *>(c);
				AABB aabb = RSG::mesh_storage->mesh_get_aabb(mesh->mesh, skeleton);

				r = Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);

			} break;
			case Item::Command::TYPE_MULTIMESH: {
				const Item::CommandMultiMesh *multimesh = static_cast<const Item::CommandMultiMesh *>(c);
				AABB aabb = RSG::mesh_storage->multimesh_get_aabb(multimesh->multimesh);

				r = Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);

			} break;
			case Item::Command::TYPE_PARTICLES: {
				const Item::CommandParticles *particles_cmd = static_cast<const Item::CommandParticles *>(c);
				if (particles_cmd->particles.is_valid()) {
					AABB aabb = RSG::particles_storage->particles_get_aabb(particles_cmd->particles);
					r = Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);
				}

			} break;
			case Item::Command::TYPE_FILLED_CURVE: {
				const Item::CommandFilledCurve *filled_curve = static_cast<const Item::CommandFilledCurve *>(c);
				r = filled_curve->bounds;
			} break;
			case Item::Command::TYPE_TRANSFORM: {
				const Item::CommandTransform *transform = static_cast<const Item::CommandTransform *>(c);
				xf = transform->xform;
				found_xform = true;
				[[fallthrough]];
			}
			default: {
				c = c->next;
				continue;
			}
		}

		if (found_xform) {
			r = xf.xform(r);
		}

		if (first) {
			rect = r;
			first = false;
		} else {
			rect = rect.merge(r);
		}
		c = c->next;
	}

	rect_dirty = false;
	return rect;
}

RendererCanvasRender::Item::CommandMesh::~CommandMesh() {
	if (mesh_instance.is_valid()) {
		RSG::mesh_storage->mesh_instance_free(mesh_instance);
	}
}

RendererCanvasRender::Item::CommandFilledCurve::~CommandFilledCurve() {
	if (mesh_cache.is_valid()) {
		RS::get_singleton()->free(mesh_cache);
	}
}

void RendererCanvasRender::Item::CommandFilledCurve::generate_mesh(Transform2D transform, Rect2 rect) {
	if (mesh_cache.is_valid() && transform_cache.is_equal_approx(transform) && rect_cache.is_equal_approx(rect)) {
		return;
	}
	transform_cache = transform;
	rect_cache = rect;
	if (!mesh_cache.is_valid()) {
		mesh_cache = RS::get_singleton()->mesh_create();
	} else {
		RS::get_singleton()->mesh_clear(mesh_cache);
	}
	DEV_ASSERT(points.size() == types.size());
	Vector<Vector<Vector2>> tessellated;
	for (int i = 0; i < points.size(); i++) {
		tessellated.push_back(Geometry2D::tessellate_curve_in_rect(points[i], types[i], transform, rect, use_order5));
	}
	Vector<Vector2> mesh_points;
	Vector<int32_t> mesh_triangles;
	Geometry2D::triangulate_polygons(tessellated, winding_even_odd, mesh_points, mesh_triangles);
	if (mesh_points.size() <= 0 || mesh_triangles.size() <= 0) {
		return;
	}
	Array arrs;
	arrs.resize(RS::ARRAY_MAX);
	arrs[RS::ARRAY_VERTEX] = mesh_points;
	arrs[RS::ARRAY_INDEX] = mesh_triangles;
	arrs[RS::ARRAY_TEX_UV] = transform_uv.xform(mesh_points);
	RS::get_singleton()->mesh_add_surface_from_arrays(mesh_cache, RS::PRIMITIVE_TRIANGLES, arrs);
}
