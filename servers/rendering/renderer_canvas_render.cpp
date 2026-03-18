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

RendererCanvasRender *RendererCanvasRender::singleton = nullptr;

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
