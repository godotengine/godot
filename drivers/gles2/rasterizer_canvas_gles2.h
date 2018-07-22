/*************************************************************************/
/*  rasterizer_canvas_gles2.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef RASTERIZERCANVASGLES2_H
#define RASTERIZERCANVASGLES2_H

#include "rasterizer_storage_gles2.h"
#include "servers/visual/rasterizer.h"

#include "shaders/canvas.glsl.gen.h"

// #include "shaders/canvas_shadow.glsl.gen.h"

class RasterizerSceneGLES2;

class RasterizerCanvasGLES2 : public RasterizerCanvas {
public:
	struct Uniforms {
		Transform projection_matrix;

		Transform2D modelview_matrix;
		Transform2D extra_matrix;

		Color final_modulate;

		float time;
	};

	struct Data {

		GLuint canvas_quad_vertices;
		GLuint polygon_buffer;
		GLuint polygon_index_buffer;

		uint32_t polygon_buffer_size;
		uint32_t polygon_index_buffer_size;

		GLuint ninepatch_vertices;
		GLuint ninepatch_elements;

		uint8_t *polygon_mem_buffer;
		uint32_t polygon_mem_offset;

		uint8_t *polygon_index_mem_buffer;
		uint32_t polygon_index_mem_offset;

	} data;

	struct State {
		Uniforms uniforms;
		bool canvas_texscreen_used;
		CanvasShaderGLES2 canvas_shader;
		// CanvasShadowShaderGLES3 canvas_shadow_shader;

		bool using_texture_rect;
		bool using_ninepatch;

		RID current_tex;
		RID current_normal;
		RasterizerStorageGLES2::Texture *current_tex_ptr;

		Transform vp;

		Item *current_clip;
		RasterizerStorageGLES2::Shader *shader_cache;
		bool rebind_shader;
		Size2 rt_size;
		RID canvas_last_material;
		int last_blend_mode;
		bool reclip;
		Color p_modulate;

	} state;

	struct RenderItem {
		enum Type {
			TYPE_PASSTHROUGH,
			TYPE_CHANGEITEM,
			TYPE_POLYGON,
			TYPE_GENERIC,
			TYPE_GUI_PRIMITIVE,
			TYPE_BEGINLIST,
			TYPE_ENDLIST,
			TYPE_NINEPATCH,
			TYPE_RECTS
		};

		struct CommandBase {
			Type type;
		};

		struct BeginListCommand : public CommandBase {
			static const Type _type = TYPE_BEGINLIST;
			Color p_modulate;
		};

		struct EndListCommand : public CommandBase {
			static const Type _type = TYPE_ENDLIST;
		};

		struct PassThroughCommand : public CommandBase {
			static const Type _type = TYPE_PASSTHROUGH;
			RasterizerCanvas::Item::Command *command;
		};

		struct ChangeItemCommand : public CommandBase {
			static const Type _type = TYPE_CHANGEITEM;
			RasterizerCanvas::Item *item;
		};

		struct NinepatchCommand : public CommandBase {
			static const Type _type = TYPE_NINEPATCH;
			RasterizerCanvas::Item::CommandNinePatch *command;

			uint32_t vertices_offset;
		};

		struct RectsCommand : public CommandBase {
			static const Type _type = TYPE_RECTS;

			uint32_t vertices_offset;
			uint32_t count;

			bool use_texture;
			RID texture;

			bool untile;
		};

		struct PolygonCommand : public CommandBase {
			static const Type _type = TYPE_POLYGON;
			RasterizerCanvas::Item::Command *command;

			uint32_t count;

			Color color;

			uint32_t vertices_offset;

			bool use_colors;
			uint32_t colors_offset;

			bool use_uvs;
			uint32_t uvs_offset;

			uint32_t indices_offset;
		};

		struct GenericCommand : public CommandBase {
			static const Type _type = TYPE_GENERIC;
			GLuint primitive;

			uint32_t count;

			Color color;

			uint32_t vertices_offset;

			bool use_colors;
			uint32_t colors_offset;

			bool use_uvs;
			uint32_t uvs_offset;
		};

		struct GuiPrimitiveCommand : public CommandBase {
			static const Type _type = TYPE_GUI_PRIMITIVE;
			RasterizerCanvas::Item::Command *command;
			uint32_t count;

			uint32_t offset;

			bool use_colors;
			uint32_t color_offset;

			bool use_uvs;
			uint32_t uvs_offset;

			uint32_t stride;
		};
	};

	struct RenderCommands {

		struct Item {
			uint32_t size;
		};

		struct Ptr {

			Ptr(RenderCommands *owner) {
				this->owner = owner;
				offset = 0;

				read_current();
			}

			_FORCE_INLINE_ bool next() {
				if (curr == NULL)
					return false;

				read_current();
				return curr;
			}

			_FORCE_INLINE_ RenderItem::CommandBase *current() {
				return curr;
			}

		private:
			uint32_t offset;
			RenderItem::CommandBase *curr;

			RenderCommands *owner;

			_FORCE_INLINE_ void read_current() {
				if (offset >= owner->offset) {
					curr = NULL;
					return;
				}

				Item *item = reinterpret_cast<Item *>(owner->buffer + offset);
				offset += item->size;

				curr = reinterpret_cast<RenderItem::CommandBase *>(item + 1);
			}
		};

		RenderCommands() {
			buffer = NULL;
			size = 0;
			clear();
		}

		~RenderCommands() {
			memfree(buffer);
		}

		_FORCE_INLINE_ Ptr ptr() {
			return Ptr(this);
		}

		_FORCE_INLINE_ bool empty() {
			return offset == 0;
		}

		template <typename T>
		_FORCE_INLINE_ T *allocate() {
			const uint32_t item_size = sizeof(Item) + sizeof(T);

			if (size - offset < item_size)
				return NULL;

			uint8_t *ptr = buffer + offset;

			Item *item = reinterpret_cast<Item *>(ptr);
			item->size = item_size;

			T *result = reinterpret_cast<T *>(ptr + sizeof(Item));
			result->type = T::_type;

			offset += item_size;

			return result;
		}

		_FORCE_INLINE_ void clear() {
			offset = 0;
		}

		void init(const uint32_t size) {
			ERR_FAIL_COND(buffer); // do not allow re-initialization

			clear();

			buffer = reinterpret_cast<uint8_t *>(memrealloc(buffer, size));
			this->size = size;
		}

	private:
		uint8_t *buffer;
		uint32_t size;
		uint32_t offset;
	} render_commands;

	typedef void Texture;

	RasterizerSceneGLES2 *scene_render;

	RasterizerStorageGLES2 *storage;

	virtual RID light_internal_create();
	virtual void light_internal_update(RID p_rid, Light *p_light);
	virtual void light_internal_free(RID p_rid);

	void _set_uniforms();

	virtual void canvas_begin();
	virtual void canvas_end();

	_FORCE_INLINE_ void _set_texture_rect_mode(bool p_enable, bool p_ninepatch = false);

	_FORCE_INLINE_ bool _prepare_gui_primitive(RasterizerCanvas::Item::Command *command, int p_points, const Vector2 *p_vertices, const Color *p_colors, const Vector2 *p_uvs);
	_FORCE_INLINE_ bool _prepare_polygon(RasterizerCanvas::Item::Command *cmd, const int *p_indices, int p_index_count, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor);
	_FORCE_INLINE_ bool _prepare_generic(GLuint p_primitive, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor);
	_FORCE_INLINE_ bool _prepare_ninepatch(RasterizerCanvas::Item::CommandNinePatch *command);

	_FORCE_INLINE_ void _canvas_item_process(Item *p_item);

	_FORCE_INLINE_ void _flush_gui_primitive(const RenderItem::GuiPrimitiveCommand *cmd);
	_FORCE_INLINE_ void _flush_polygon(const RenderItem::PolygonCommand *cmd);
	_FORCE_INLINE_ void _flush_generic(const RenderItem::GenericCommand *cmd);
	_FORCE_INLINE_ void _flush_ninepatch(const RenderItem::NinepatchCommand *cmd);
	_FORCE_INLINE_ void _flush_rects(const RenderItem::RectsCommand *cmd);

	template <typename T>
	_FORCE_INLINE_ T *push_command() {
		T *cmd = render_commands.allocate<T>();
		if (!cmd) {
			_flush();

			cmd = render_commands.allocate<T>();
			ERR_FAIL_COND_V(!cmd, NULL);
		}

		return cmd;
	}

	void _flush();

	_FORCE_INLINE_ void _canvas_render_command(Item::Command *command, Item *current_clip, bool &reclip);

	_FORCE_INLINE_ void _copy_texscreen(const Rect2 &p_rect);

	virtual void canvas_render_items(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	virtual void canvas_debug_viewport_shadows(Light *p_lights_with_shadow);

	virtual void canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache);

	virtual void reset_canvas();

	RasterizerStorageGLES2::Texture *_bind_canvas_texture(const RID &p_texture, const RID &p_normal_map);

	void _bind_quad_buffer();
	void draw_generic_textured_rect(const Rect2 &p_rect, const Rect2 &p_src);

	void initialize();
	void finalize();

	virtual void draw_window_margins(int *black_margin, RID *black_image);

	RasterizerCanvasGLES2();
};

#endif // RASTERIZERCANVASGLES2_H
