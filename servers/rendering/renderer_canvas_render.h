/*************************************************************************/
/*  renderer_canvas_render.h                                             */
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

#ifndef RENDERINGSERVERCANVASRENDER_H
#define RENDERINGSERVERCANVASRENDER_H

#include "servers/rendering/renderer_storage.h"

class RendererCanvasRender {
public:
	static RendererCanvasRender *singleton;

	enum CanvasRectFlags {
		CANVAS_RECT_REGION = 1,
		CANVAS_RECT_TILE = 2,
		CANVAS_RECT_FLIP_H = 4,
		CANVAS_RECT_FLIP_V = 8,
		CANVAS_RECT_TRANSPOSE = 16,
		CANVAS_RECT_CLIP_UV = 32,
		CANVAS_RECT_IS_GROUP = 64,
	};

	struct Light {
		bool enabled = true;
		Color color = Color(1, 1, 1);
		Transform2D xform;
		float height = 0;
		float energy = 1.0;
		float scale = 1.0;
		int z_min = -1024;
		int z_max = 1024;
		int layer_min = 0;
		int layer_max = 0;
		int item_mask = 1;
		int item_shadow_mask = 1;
		float directional_distance = 10000.0;
		RS::CanvasLightMode mode = RS::CANVAS_LIGHT_MODE_POINT;
		RS::CanvasLightBlendMode blend_mode = RS::CANVAS_LIGHT_BLEND_MODE_ADD;
		RID texture;
		Vector2 texture_offset;
		RID canvas;
		bool use_shadow = false;
		int shadow_buffer_size = 2048;
		RS::CanvasLightShadowFilter shadow_filter = RS::CANVAS_LIGHT_FILTER_NONE;
		Color shadow_color = Color(0, 0, 0, 0);
		float shadow_smooth = 0.0;

		//void *texture_cache; // implementation dependent
		Rect2 rect_cache;
		Transform2D xform_cache;
		float radius_cache = 0.0; //used for shadow far plane
		//CameraMatrix shadow_matrix_cache;

		Transform2D light_shader_xform;
		//Vector2 light_shader_pos;

		Light *shadows_next_ptr = nullptr;
		Light *filter_next_ptr = nullptr;
		Light *next_ptr = nullptr;
		Light *directional_next_ptr = nullptr;

		RID light_internal;
		uint64_t version = 0;

		int32_t render_index_cache = -1;
	};

	//easier wrap to avoid mistakes

	struct Item;

	typedef uint64_t PolygonID;
	virtual PolygonID request_polygon(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs = Vector<Point2>(), const Vector<int> &p_bones = Vector<int>(), const Vector<float> &p_weights = Vector<float>()) = 0;
	virtual void free_polygon(PolygonID p_polygon) = 0;

	//also easier to wrap to avoid mistakes
	struct Polygon {
		PolygonID polygon_id = 0;
		Rect2 rect_cache;

		_FORCE_INLINE_ void create(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs = Vector<Point2>(), const Vector<int> &p_bones = Vector<int>(), const Vector<float> &p_weights = Vector<float>()) {
			ERR_FAIL_COND(polygon_id != 0);
			{
				uint32_t pc = p_points.size();
				const Vector2 *v2 = p_points.ptr();
				rect_cache.position = *v2;
				for (uint32_t i = 1; i < pc; i++) {
					rect_cache.expand_to(v2[i]);
				}
			}
			polygon_id = singleton->request_polygon(p_indices, p_points, p_colors, p_uvs, p_bones, p_weights);
		}

		_FORCE_INLINE_ Polygon() {}
		_FORCE_INLINE_ ~Polygon() {
			if (polygon_id) {
				singleton->free_polygon(polygon_id);
			}
		}
	};

	//item

	struct Item {
		//commands are allocated in blocks of 4k to improve performance
		//and cache coherence.
		//blocks always grow but never shrink.

		struct CommandBlock {
			enum {
				MAX_SIZE = 4096
			};
			uint32_t usage = 0;
			uint8_t *memory = nullptr;
		};

		struct Command {
			enum Type {
				TYPE_RECT,
				TYPE_NINEPATCH,
				TYPE_POLYGON,
				TYPE_PRIMITIVE,
				TYPE_MESH,
				TYPE_MULTIMESH,
				TYPE_PARTICLES,
				TYPE_TRANSFORM,
				TYPE_CLIP_IGNORE,
			};

			Command *next = nullptr;
			Type type = TYPE_RECT;
			virtual ~Command() {}
		};

		struct CommandRect : public Command {
			Rect2 rect;
			Color modulate;
			Rect2 source;
			uint8_t flags = 0;

			RID texture;

			CommandRect() {
				type = TYPE_RECT;
			}
		};

		struct CommandNinePatch : public Command {
			Rect2 rect;
			Rect2 source;
			float margin[4] = {};
			bool draw_center = true;
			Color color;
			RS::NinePatchAxisMode axis_x = RS::NinePatchAxisMode::NINE_PATCH_STRETCH;
			RS::NinePatchAxisMode axis_y = RS::NinePatchAxisMode::NINE_PATCH_STRETCH;

			RID texture;

			CommandNinePatch() {
				type = TYPE_NINEPATCH;
			}
		};

		struct CommandPolygon : public Command {
			RS::PrimitiveType primitive = RS::PrimitiveType::PRIMITIVE_MAX;
			Polygon polygon;

			RID texture;

			CommandPolygon() {
				type = TYPE_POLYGON;
			}
		};

		struct CommandPrimitive : public Command {
			uint32_t point_count = 0;
			Vector2 points[4];
			Vector2 uvs[4];
			Color colors[4];

			RID texture;

			CommandPrimitive() {
				type = TYPE_PRIMITIVE;
			}
		};

		struct CommandMesh : public Command {
			RID mesh;
			Transform2D transform;
			Color modulate;

			RID texture;

			CommandMesh() { type = TYPE_MESH; }
		};

		struct CommandMultiMesh : public Command {
			RID multimesh;

			RID texture;

			CommandMultiMesh() { type = TYPE_MULTIMESH; }
		};

		struct CommandParticles : public Command {
			RID particles;

			RID texture;

			CommandParticles() { type = TYPE_PARTICLES; }
		};

		struct CommandTransform : public Command {
			Transform2D xform;
			CommandTransform() { type = TYPE_TRANSFORM; }
		};

		struct CommandClipIgnore : public Command {
			bool ignore = false;
			CommandClipIgnore() {
				type = TYPE_CLIP_IGNORE;
			}
		};

		struct ViewportRender {
			RenderingServer *owner = nullptr;
			void *udata = nullptr;
			Rect2 rect;
		};

		Transform2D xform;
		bool clip = false;
		bool visible = true;
		bool behind = false;
		bool update_when_visible = false;

		struct CanvasGroup {
			RS::CanvasGroupMode mode = RS::CanvasGroupMode::CANVAS_GROUP_MODE_DISABLED;
			bool fit_empty = false;
			float fit_margin = 0.0;
			bool blur_mipmaps = false;
			float clear_margin = 0.0;
		};

		CanvasGroup *canvas_group = nullptr;
		int light_mask = 1;
		int z_final = 0;

		mutable bool custom_rect = false;
		mutable bool rect_dirty = true;
		mutable Rect2 rect;
		RID material;
		RID skeleton;

		Item *next = nullptr;

		struct CopyBackBuffer {
			Rect2 rect;
			Rect2 screen_rect;
			bool full = false;
		};
		CopyBackBuffer *copy_back_buffer = nullptr;

		Color final_modulate = Color(1, 1, 1, 1);
		Transform2D final_transform;
		Rect2 final_clip_rect;
		Item *final_clip_owner = nullptr;
		Item *material_owner = nullptr;
		Item *canvas_group_owner = nullptr;
		ViewportRender *vp_render = nullptr;
		bool distance_field = false;
		bool light_masked = false;

		Rect2 global_rect_cache;

		const Rect2 &get_rect() const {
			if (custom_rect || (!rect_dirty && !update_when_visible)) {
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
						AABB aabb = RendererStorage::base_singleton->mesh_get_aabb(mesh->mesh, RID());

						r = Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);

					} break;
					case Item::Command::TYPE_MULTIMESH: {
						const Item::CommandMultiMesh *multimesh = static_cast<const Item::CommandMultiMesh *>(c);
						AABB aabb = RendererStorage::base_singleton->multimesh_get_aabb(multimesh->multimesh);

						r = Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);

					} break;
					case Item::Command::TYPE_PARTICLES: {
						const Item::CommandParticles *particles_cmd = static_cast<const Item::CommandParticles *>(c);
						if (particles_cmd->particles.is_valid()) {
							AABB aabb = RendererStorage::base_singleton->particles_get_aabb(particles_cmd->particles);
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
					found_xform = false;
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

		Command *commands = nullptr;
		Command *last_command = nullptr;
		Vector<CommandBlock> blocks;
		uint32_t current_block = 0;

		template <class T>
		T *alloc_command() {
			T *command;
			if (commands == nullptr) {
				// As the most common use case of canvas items is to
				// use only one command, the first is done with it's
				// own allocation. The rest of them use blocks.
				command = memnew(T);
				command->next = nullptr;
				commands = command;
				last_command = command;
			} else {
				//Subsequent commands go into a block.

				while (true) {
					if (unlikely(current_block == (uint32_t)blocks.size())) {
						// If we need more blocks, we allocate them
						// (they won't be freed until this CanvasItem is
						// deleted, though).
						CommandBlock cb;
						cb.memory = (uint8_t *)memalloc(CommandBlock::MAX_SIZE);
						cb.usage = 0;
						blocks.push_back(cb);
					}

					CommandBlock *c = &blocks.write[current_block];
					size_t space_left = CommandBlock::MAX_SIZE - c->usage;
					if (space_left < sizeof(T)) {
						current_block++;
						continue;
					}

					//allocate block and add to the linked list
					void *memory = c->memory + c->usage;
					command = memnew_placement(memory, T);
					command->next = nullptr;
					last_command->next = command;
					last_command = command;
					c->usage += sizeof(T);
					break;
				}
			}

			rect_dirty = true;
			return command;
		}

		void clear() {
			// The first one is always allocated on heap
			// the rest go in the blocks
			Command *c = commands;
			while (c) {
				Command *n = c->next;
				if (c == commands) {
					memdelete(commands);
					commands = nullptr;
				} else {
					c->~Command();
				}
				c = n;
			}
			{
				uint32_t cbc = MIN((current_block + 1), (uint32_t)blocks.size());
				CommandBlock *blockptr = blocks.ptrw();
				for (uint32_t i = 0; i < cbc; i++) {
					blockptr[i].usage = 0;
				}
			}

			last_command = nullptr;
			commands = nullptr;
			current_block = 0;
			clip = false;
			rect_dirty = true;
			final_clip_owner = nullptr;
			material_owner = nullptr;
			light_masked = false;
		}

		RS::CanvasItemTextureFilter texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT;
		RS::CanvasItemTextureRepeat texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT;

		Item() {}
		virtual ~Item() {
			clear();
			for (int i = 0; i < blocks.size(); i++) {
				memfree(blocks[i].memory);
			}
			if (copy_back_buffer) {
				memdelete(copy_back_buffer);
			}
		}
	};

	virtual void canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, Light *p_directional_list, const Transform2D &p_canvas_transform, RS::CanvasItemTextureFilter p_default_filter, RS::CanvasItemTextureRepeat p_default_repeat, bool p_snap_2d_vertices_to_pixel, bool &r_sdf_used) = 0;
	virtual void canvas_debug_viewport_shadows(Light *p_lights_with_shadow) = 0;

	struct LightOccluderInstance {
		bool enabled = true;
		RID canvas;
		RID polygon;
		RID occluder;
		Rect2 aabb_cache;
		Transform2D xform;
		Transform2D xform_cache;
		int light_mask = 1;
		bool sdf_collision = false;
		RS::CanvasOccluderPolygonCullMode cull_cache = RS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED;

		LightOccluderInstance *next = nullptr;
	};

	virtual RID light_create() = 0;
	virtual void light_set_texture(RID p_rid, RID p_texture) = 0;
	virtual void light_set_use_shadow(RID p_rid, bool p_enable) = 0;
	virtual void light_update_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders) = 0;
	virtual void light_update_directional_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_cull_distance, const Rect2 &p_clip_rect, LightOccluderInstance *p_occluders) = 0;

	virtual void render_sdf(RID p_render_target, LightOccluderInstance *p_occluders) = 0;

	virtual RID occluder_polygon_create() = 0;
	virtual void occluder_polygon_set_shape(RID p_occluder, const Vector<Vector2> &p_points, bool p_closed) = 0;
	virtual void occluder_polygon_set_cull_mode(RID p_occluder, RS::CanvasOccluderPolygonCullMode p_mode) = 0;
	virtual void set_shadow_texture_size(int p_size) = 0;

	virtual void draw_window_margins(int *p_margins, RID *p_margin_textures) = 0;

	virtual bool free(RID p_rid) = 0;
	virtual void update() = 0;

	RendererCanvasRender() { singleton = this; }
	virtual ~RendererCanvasRender() {}
};

#endif // RENDERINGSERVERCANVASRENDER_H
