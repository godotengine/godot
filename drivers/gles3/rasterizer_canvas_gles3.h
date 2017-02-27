#ifndef RASTERIZERCANVASGLES3_H
#define RASTERIZERCANVASGLES3_H

#include "servers/visual/rasterizer.h"
#include "rasterizer_storage_gles3.h"
#include "shaders/canvas_shadow.glsl.h"


class RasterizerCanvasGLES3 : public RasterizerCanvas {
public:

	struct CanvasItemUBO {

		float projection_matrix[16];
		float time[4];

	};

	struct Data {

		GLuint canvas_quad_vertices;
		GLuint canvas_quad_array;

		GLuint primitive_quad_buffer;
		GLuint primitive_quad_buffer_arrays[4];

	} data;

	struct State {
		CanvasItemUBO canvas_item_ubo_data;
		GLuint canvas_item_ubo;
		bool canvas_texscreen_used;
		CanvasShaderGLES3 canvas_shader;
		CanvasShadowShaderGLES3 canvas_shadow_shader;

		bool using_texture_rect;


		RID current_tex;
		RasterizerStorageGLES3::Texture *current_tex_ptr;

		Transform vp;

		Color canvas_item_modulate;
		Transform2D extra_matrix;
		Transform2D final_transform;

	} state;

	RasterizerStorageGLES3 *storage;

	struct LightInternal : public RID_Data {

		struct UBOData {

			float light_matrix[16];
			float local_matrix[16];
			float shadow_matrix[16];
			float color[4];
			float shadow_color[4];
			float light_pos[2];
			float shadowpixel_size;
			float shadow_gradient;
			float light_height;
			float light_outside_alpha;
			float shadow_distance_mult;
		} ubo_data;

		GLuint ubo;
	};

	RID_Owner<LightInternal> light_internal_owner;

	virtual RID light_internal_create();
	virtual void light_internal_update(RID p_rid, Light* p_light);
	virtual void light_internal_free(RID p_rid);


	virtual void canvas_begin();
	virtual void canvas_end();

	_FORCE_INLINE_ void _set_texture_rect_mode(bool p_enable);
	_FORCE_INLINE_ RasterizerStorageGLES3::Texture* _bind_canvas_texture(const RID& p_texture);

	_FORCE_INLINE_ void _draw_gui_primitive(int p_points, const Vector2 *p_vertices, const Color* p_colors, const Vector2 *p_uvs);
	_FORCE_INLINE_ void _draw_polygon(int p_vertex_count, const int* p_indices, const Vector2* p_vertices, const Vector2* p_uvs, const Color* p_colors,const RID& p_texture,bool p_singlecolor);
	_FORCE_INLINE_ void _canvas_item_render_commands(Item *p_item,Item *current_clip,bool &reclip);


	virtual void canvas_render_items(Item *p_item_list,int p_z,const Color& p_modulate,Light *p_light);
	virtual void canvas_debug_viewport_shadows(Light* p_lights_with_shadow);

	virtual void canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D& p_light_xform, int p_light_mask,float p_near, float p_far, LightOccluderInstance* p_occluders, CameraMatrix *p_xform_cache);


	virtual void reset_canvas();

	void draw_generic_textured_rect(const Rect2& p_rect, const Rect2& p_src);


	void initialize();
	void finalize();

	RasterizerCanvasGLES3();
};

#endif // RASTERIZERCANVASGLES3_H
