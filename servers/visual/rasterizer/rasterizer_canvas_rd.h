#ifndef RASTERIZER_CANVAS_RD_H
#define RASTERIZER_CANVAS_RD_H

#include "servers/visual/rasterizer/rasterizer.h"
#include "servers/visual/rasterizer/rasterizer_storage_rd.h"
#include "servers/visual/rasterizer/render_pipeline_vertex_format_cache_rd.h"
#include "servers/visual/rasterizer/shaders/canvas.glsl.gen.h"
#include "servers/visual/rendering_device.h"

class RasterizerCanvasRD : public RasterizerCanvas {

	RasterizerStorageRD *storage;

	enum ShaderVariant {
		SHADER_VARIANT_QUAD,
		SHADER_VARIANT_NINEPATCH,
		SHADER_VARIANT_PRIMITIVE,
		SHADER_VARIANT_PRIMITIVE_POINTS,
		SHADER_VARIANT_ATTRIBUTES,
		SHADER_VARIANT_ATTRIBUTES_POINTS,
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

		FLAGS_USE_SKELETON = (1 << 15),
		FLAGS_NINEPATCH_H_MODE_SHIFT = 16,
		FLAGS_NINEPATCH_V_MODE_SHIFT = 18
	};

	/****************/
	/**** SHADER ****/
	/****************/

	enum PipelineVariant {
		PIPELINE_VARIANT_QUAD,
		PIPELINE_VARIANT_NINEPATCH,
		PIPELINE_VARIANT_PRIMITIVE_TRIANGLES,
		PIPELINE_VARIANT_PRIMITIVE_LINES,
		PIPELINE_VARIANT_PRIMITIVE_POINTS,
		PIPELINE_VARIANT_ATTRIBUTE_TRIANGLES,
		PIPELINE_VARIANT_ATTRIBUTE_LINES,
		PIPELINE_VARIANT_ATTRIBUTE_POINTS,
		PIPELINE_VARIANT_MAX
	};
	struct PipelineVariants {
		RenderPipelineVertexFormatCacheRD variants[RENDER_TARGET_FORMAT_MAX][PIPELINE_VARIANT_MAX];
	};

	struct {
		CanvasShaderRD canvas_shader;
		RD::FramebufferFormatID framebuffer_formats[RENDER_TARGET_FORMAT_MAX];
		RID default_version;
		RID default_version_rd_shader;
		RID quad_index_buffer;
		RID quad_index_array;
		PipelineVariants pipeline_variants;

		// default_skeleton uniform set
		RID default_skeleton_uniform;
		RID default_skeleton_uniform_set;

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

	RID _create_texture_binding(RID p_texture, RID p_normalmap, RID p_specular, VisualServer::CanvasItemTextureFilter p_filter, VisualServer::CanvasItemTextureRepeat p_repeat, RID p_multimesh);
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

	/******************/
	/**** POLYGONS ****/
	/******************/

	struct PolygonBuffers {
		RD::VertexFormatID vertex_format_id;
		RID vertex_buffer;
		RID vertex_array;
		RID index_buffer;
		RID indices;
	};

	struct {
		HashMap<PolygonID, PolygonBuffers> polygons;
		PolygonID last_id;
	} polygon_buffers;

	/********************/
	/**** PRIMITIVES ****/
	/********************/

	struct {
		RID index_array[4];
	} primitive_arrays;

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
		float world[6];
		uint32_t flags;
		uint32_t specular_shininess;
		union {
			//rect
			struct {
				float modulation[4];
				float ninepatch_margins[4];
				float dst_rect[4];
				float src_rect[4];
				float color_texture_pixel_size[2];
				uint32_t pad[6];
			};
			//primitive
			struct {
				float points[8]; // vec2 points[4]
				uint32_t colors[8]; // colors encoded as half
				float uvs[8]; // vec2 points[4]
			};
		};
	};

	struct SkeletonUniform {
		float skeleton_transform[16];
		float skeleton_inverse[16];
	};

	enum {
		MAX_RENDER_ITEMS = 256 * 1024
	};

	Item *items[MAX_RENDER_ITEMS];

	Size2i _bind_texture_binding(TextureBindingID p_binding, RenderingDevice::DrawListID p_draw_list);
	void _render_item(RenderingDevice::DrawListID p_draw_list, const Item *p_item, RenderTargetFormat p_render_target_format, RenderingDevice::TextureSamples p_samples, const Color &p_modulate, const Transform2D &p_canvas_transform_inverse, Item *&current_clip);
	void _render_items(RID p_to_render_target, int p_item_count, const Color &p_modulate, const Transform2D &p_transform);

	void _update_transform_2d_to_mat2x4(const Transform2D &p_transform, float *p_mat2x4);
	void _update_transform_2d_to_mat2x3(const Transform2D &p_transform, float *p_mat2x3);

	void _update_transform_2d_to_mat4(const Transform2D &p_transform, float *p_mat4);
	void _update_transform_to_mat4(const Transform &p_transform, float *p_mat4);

	void _update_canvas_state_uniform_set();

public:
	TextureBindingID request_texture_binding(RID p_texture, RID p_normalmap, RID p_specular, VS::CanvasItemTextureFilter p_filter, VS::CanvasItemTextureRepeat p_repeat, RID p_multimesh);
	void free_texture_binding(TextureBindingID p_binding);

	PolygonID request_polygon(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs = Vector<Point2>(), const Vector<int> &p_bones = Vector<int>(), const Vector<float> &p_weights = Vector<float>());
	void free_polygon(PolygonID p_polygon);

	RID light_internal_create() { return RID(); }
	void light_internal_update(RID p_rid, Light *p_light) {}
	void light_internal_free(RID p_rid) {}

	void canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, const Transform2D &p_canvas_transform);

	void canvas_debug_viewport_shadows(Light *p_lights_with_shadow){};

	void canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache) {}

	void reset_canvas() {}

	void draw_window_margins(int *p_margins, RID *p_margin_textures) {}

	void update();
	RasterizerCanvasRD(RasterizerStorageRD *p_storage);
	~RasterizerCanvasRD();
};

#endif // RASTERIZER_CANVAS_RD_H
