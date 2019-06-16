#ifndef RASTERIZER_CANVAS_RD_H
#define RASTERIZER_CANVAS_RD_H

#include "servers/visual/rasterizer/rasterizer.h"
#include "servers/visual/rasterizer/rasterizer_storage_rd.h"
#include "servers/visual/rasterizer/shaders/canvas.glsl.gen.h"
#include "servers/visual/rendering_device.h"

class RasterizerCanvasRD : public RasterizerCanvas {

	RasterizerStorageRD *storage;

	enum ShaderVariant {
		SHADER_VARIANT_QUAD,
		SHADER_VARIANT_NINEPATCH,
		SHADER_VARIANT_VERTICES,
		SHADER_VARIANT_POINTS,
		SHADER_VARIANT_MAX
	};

	enum RenderTargetFormat {
		RENDER_TARGET_FORMAT_8_BIT_INT,
		RENDER_TARGET_FORMAT_16_BIT_FLOAT,
		RENDER_TARGET_FORMAT_MAX
	};

	enum {
		FLAGS_INSTANCING_STRIDE_MASK = 0xF,
		FLAGS_INSTANCING_ENABLED = (1 << 4),
		FLAGS_INSTANCING_HAS_COLORS = (1 << 5),
		FLAGS_INSTANCING_COLOR_8BIT = (1 << 6),
		FLAGS_INSTANCING_HAS_CUSTOM_DATA = (1 << 7),
		FLAGS_INSTANCING_CUSTOM_DATA_8_BIT = (1 << 8),

		FLAGS_CLIP_RECT_UV = (1 << 9),
		FLAGS_TRANSPOSE_RECT = (1 << 10),
		FLAGS_NINEPACH_DRAW_CENTER = (1 << 12),
		FLAGS_USING_PARTICLES = (1 << 13),
		FLAGS_USE_PIXEL_SNAP = (1 << 14),

		FLAGS_USE_SKELETON = (1 << 16)
	};

	/****************/
	/**** SHADER ****/
	/****************/

	enum PipelineVariant {
		PIPELINE_VARIANT_QUAD,
		PIPELINE_VARIANT_NINEPATCH,
		PIPELINE_VARIANT_TRIANGLES,
		PIPELINE_VARIANT_LINES,
		PIPELINE_VARIANT_POINTS,
		PIPELINE_VARIANT_TRIANGLES_COMPRESSED,
		PIPELINE_VARIANT_LINES_COMPRESSED,
		PIPELINE_VARIANT_POINTS_COMPRESSED,
		PIPELINE_VARIANT_MAX
	};
	struct PipelineVariants {
		RID variants[RENDER_TARGET_FORMAT_MAX][PIPELINE_VARIANT_MAX];
	};

	struct {
		CanvasShaderRD canvas_shader;
		RD::FramebufferFormatID framebuffer_formats[RENDER_TARGET_FORMAT_MAX];
		RID default_version;
		RID default_version_rd_shader;
		RID quad_index_array;
		PipelineVariants pipeline_variants;

		// default_skeleton uniform set
		RID default_material_skeleton_uniform;
		RID default_material_uniform_set;

	} shader;

	/**************************/
	/**** TEXTURE BINDINGS ****/
	/**************************/

	// bindings used to render commands,
	// cached for performance.

	struct TextureBindingKey {
		RID texture;
		RID normalmap;
		RID specular;
		RID multimesh;
		VS::CanvasItemTextureFilter texture_filter;
		VS::CanvasItemTextureRepeat texture_repeat;
		bool operator==(const TextureBindingKey &p_key) const {
			return texture == p_key.texture && normalmap == p_key.normalmap && specular == p_key.specular && multimesh == p_key.specular && texture_filter == p_key.texture_filter && texture_repeat == p_key.texture_repeat;
		}
	};

	struct TextureBindingKeyHasher {
		static _FORCE_INLINE_ uint32_t hash(const TextureBindingKey &p_key) {
			uint32_t hash = hash_djb2_one_64(p_key.texture.get_id());
			hash = hash_djb2_one_64(p_key.normalmap.get_id(), hash);
			hash = hash_djb2_one_64(p_key.specular.get_id(), hash);
			hash = hash_djb2_one_64(p_key.multimesh.get_id(), hash);
			hash = hash_djb2_one_32(uint32_t(p_key.texture_filter) << 16 | uint32_t(p_key.texture_repeat), hash);
			return hash;
		}
	};

	struct TextureBinding {
		TextureBindingID id;
		TextureBindingKey key;
		SelfList<TextureBinding> to_dispose;
		uint32_t reference_count;
		RID uniform_set;
		TextureBinding() :
				to_dispose(this) {
			reference_count = 0;
		}
	};

	struct {
		SelfList<TextureBinding>::List to_dispose_list;

		TextureBindingID id_generator;
		HashMap<TextureBindingKey, TextureBindingID, TextureBindingKeyHasher> texture_key_bindings;
		HashMap<TextureBindingID, TextureBinding *> texture_bindings;

		TextureBindingID default_empty;
	} bindings;

	void _dispose_bindings();
	struct {
		RID white_texture;
		RID black_texture;
		RID normal_texture;
		RID aniso_texture;

		RID default_multimesh_tb;

	} default_textures;

	struct {
		RID samplers[VS::CANVAS_ITEM_TEXTURE_FILTER_MAX][VS::CANVAS_ITEM_TEXTURE_REPEAT_MAX];
		VS::CanvasItemTextureFilter default_filter;
		VS::CanvasItemTextureRepeat default_repeat;
	} default_samplers;

	/*******************/
	/**** MATERIALS ****/
	/*******************/

	/***************/
	/**** STATE ****/
	/***************/

	//state that does not vary across rendering all items

	struct State {

		//state buffer
		struct Buffer {
			float canvas_transform[16];
			float screen_transform[16];
			//uint32_t light_count;
			//uint32_t pad[3];
		};
		RID canvas_state_buffer;
		//light buffer
		RID canvas_state_light_buffer;

		//uniform set for all the above
		RID canvas_state_uniform_set;
	} state;

	struct PushConstant {
		float world[8];
		float modulation[4];
		float ninepatch_margins[4];
		float dst_rect[4];
		float src_rect[4];
		uint32_t flags;
		uint32_t ninepatch_repeat;
		float color_texture_pixel_size[2];
		uint32_t specular_shininess;
		uint32_t pad[3];
	};

	struct SkeletonUniform {
		float skeleton_transform[16];
		float skeleton_inverse[16];
	};

	enum {
		MAX_RENDER_ITEMS = 256 * 1024
	};

	Item *items[MAX_RENDER_ITEMS];

	void _render_item(RenderingDevice::DrawListID p_draw_list, const Item *p_item, RenderTargetFormat p_render_target_format, const Color &p_modulate, const Transform2D &p_canvas_transform_inverse);
	void _render_items(RID p_to_render_target, bool p_clear, const Color &p_clear_color, int p_item_count, const Color &p_modulate, const Transform2D &p_transform);

	void _update_transform_2d_to_mat2x4(const Transform2D &p_transform, float *p_mat2x4);

	void _update_transform_2d_to_mat4(const Transform2D &p_transform, float *p_mat4);
	void _update_transform_to_mat4(const Transform &p_transform, float *p_mat4);

	void _update_canvas_state_uniform_set();

public:
	TextureBindingID request_texture_binding(RID p_texture, RID p_normalmap, RID p_specular, VS::CanvasItemTextureFilter p_filter, VS::CanvasItemTextureRepeat p_repeat, RID p_multimesh);
	void free_texture_binding(TextureBindingID p_binding);

	RID light_internal_create() { return RID(); }
	void light_internal_update(RID p_rid, Light *p_light) {}
	void light_internal_free(RID p_rid) {}

	void canvas_render_items(RID p_to_render_target, bool p_clear, const Color &p_clear_color, Item *p_item_list, const Color &p_modulate, Light *p_light_list, const Transform2D &p_canvas_transform);

	void canvas_debug_viewport_shadows(Light *p_lights_with_shadow){};

	void canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache) {}

	void reset_canvas() {}

	void draw_window_margins(int *p_margins, RID *p_margin_textures) {}

	void update();
	RasterizerCanvasRD(RasterizerStorageRD *p_storage);
	~RasterizerCanvasRD() {}
};

#endif // RASTERIZER_CANVAS_RD_H
