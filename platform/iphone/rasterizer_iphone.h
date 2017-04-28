/*************************************************************************/
/*  rasterizer_iphone.h                                                  */
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
#ifdef IPHONE_ENABLED

#ifndef RASTERIZER_IPHONE_H
#define RASTERIZER_IPHONE_H

#include "servers/visual/rasterizer.h"

#include "camera_matrix.h"
#include "image.h"
#include "list.h"
#include "map.h"
#include "rid.h"
#include "servers/visual_server.h"
#include "sort.h"
#include <ES1/gl.h>

/**
        @author Juan Linietsky <reduzio@gmail.com>
*/
class RasterizerIPhone : public Rasterizer {

	enum {
		SKINNED_BUFFER_SIZE = 1024 * 128, // 10k vertices
		MAX_LIGHTS = 8,
	};

	uint8_t skinned_buffer[SKINNED_BUFFER_SIZE];

	struct Texture {

		uint32_t flags;
		int width, height;
		Image::Format format;

		GLenum target;
		GLenum gl_format_cache;
		int gl_components_cache;
		bool has_alpha;
		bool format_has_alpha;

		bool active;
		GLuint tex_id;
		bool mipmap_dirty;

		Texture() {

			flags = width = height = 0;
			tex_id = 0;
			format = Image::FORMAT_L8;
			gl_components_cache = 0;
			format_has_alpha = false;
			has_alpha = false;
			active = false;
			mipmap_dirty = true;
		}

		~Texture() {

			if (tex_id != 0) {

				glDeleteTextures(1, &tex_id);
			}
		}
	};

	mutable RID_Owner<Texture> texture_owner;

	struct Material {

		bool flags[VS::MATERIAL_FLAG_MAX];
		Variant parameters[VisualServer::FIXED_MATERIAL_PARAM_MAX];
		RID textures[VisualServer::FIXED_MATERIAL_PARAM_MAX];

		Transform uv_transform;
		VS::SpatialMaterialTexCoordMode texcoord_mode[VisualServer::FIXED_MATERIAL_PARAM_MAX];

		VS::MaterialBlendMode detail_blend_mode;

		VS::SpatialMaterialTexGenMode texgen_mode;

		Material() {

			flags[VS::MATERIAL_FLAG_VISIBLE] = true;
			flags[VS::MATERIAL_FLAG_DOUBLE_SIDED] = false;
			flags[VS::MATERIAL_FLAG_INVERT_FACES] = false;
			flags[VS::MATERIAL_FLAG_UNSHADED] = false;
			flags[VS::MATERIAL_FLAG_WIREFRAME] = false;

			parameters[VS::FIXED_MATERIAL_PARAM_DIFFUSE] = Color(0.8, 0.8, 0.8);
			parameters[VS::FIXED_MATERIAL_PARAM_SPECULAR_EXP] = 12;

			for (int i = 0; i < VisualServer::FIXED_MATERIAL_PARAM_MAX; i++) {
				texcoord_mode[i] = VS::FIXED_MATERIAL_TEXCOORD_UV;
			};
			detail_blend_mode = VS::MATERIAL_BLEND_MODE_MIX;
			texgen_mode = VS::FIXED_MATERIAL_TEXGEN_SPHERE;
		}
	};
	mutable RID_Owner<Material> material_owner;

	struct Geometry {

		enum Type {
			GEOMETRY_INVALID,
			GEOMETRY_SURFACE,
			GEOMETRY_POLY,
			GEOMETRY_PARTICLES,
			GEOMETRY_BEAM,
			GEOMETRY_DETAILER,
		};

		Type type;
		RID material;
		bool has_alpha;
		bool material_owned;

		Vector3 scale;
		Vector3 uv_scale;

		Geometry()
			: scale(1, 1, 1) {
			has_alpha = false;
			material_owned = false;
		}
		virtual ~Geometry(){};
	};

	struct GeometryOwner {

		virtual ~GeometryOwner() {}
	};

	struct Surface : public Geometry {

		struct ArrayData {

			uint32_t ofs, size;
			bool configured;
			int components;
			ArrayData() {
				ofs = 0;
				size = 0;
				configured = false;
			}
		};

		ArrayData array[VS::ARRAY_MAX];
		// support for vertex array objects
		GLuint array_object_id;
		// support for vertex buffer object
		GLuint vertex_id; // 0 means, unconfigured
		GLuint index_id; // 0 means, unconfigured
		// no support for the above, array in localmem.
		uint8_t *array_local;
		uint8_t *index_array_local;

		AABB aabb;

		int array_len;
		int index_array_len;

		VS::PrimitiveType primitive;

		uint32_t format;

		int stride;

		bool active;

		Point2 uv_min;
		Point2 uv_max;

		bool has_alpha_cache;

		Surface() {

			array_len = 0;
			type = GEOMETRY_SURFACE;
			primitive = VS::PRIMITIVE_POINTS;
			index_array_len = VS::NO_INDEX_ARRAY;
			format = 0;
			stride = 0;

			array_local = index_array_local = 0;
			vertex_id = index_id = 0;

			active = false;
		}

		~Surface() {
		}
	};

	struct Mesh {

		bool active;
		Vector<Surface *> surfaces;

		mutable uint64_t last_pass;
		Mesh() {
			last_pass = 0;
			active = false;
		}
	};
	mutable RID_Owner<Mesh> mesh_owner;

	struct Poly : public Geometry {

		struct Primitive {

			Vector<Vector3> vertices;
			Vector<Vector3> normals;
			Vector<Vector3> uvs;
			Vector<Color> colors;
		};

		AABB aabb;
		List<Primitive> primitives;
		Poly() {

			type = GEOMETRY_POLY;
		}
	};

	mutable RID_Owner<Poly> poly_owner;

	struct Skeleton {

		Vector<Transform> bones;
	};

	mutable RID_Owner<Skeleton> skeleton_owner;

	struct Light {

		VS::LightType type;
		float vars[VS::LIGHT_PARAM_MAX];
		Color colors[3];
		bool shadow_enabled;
		RID projector;
		bool volumetric_enabled;
		Color volumetric_color;

		Light() {

			vars[VS::LIGHT_PARAM_SPOT_ATTENUATION] = 1;
			vars[VS::LIGHT_PARAM_SPOT_ANGLE] = 45;
			vars[VS::LIGHT_PARAM_ATTENUATION] = 1.0;
			vars[VS::LIGHT_PARAM_ENERGY] = 1.0;
			vars[VS::LIGHT_PARAM_RADIUS] = 1.0;
			colors[VS::LIGHT_COLOR_AMBIENT] = Color(0, 0, 0);
			colors[VS::LIGHT_COLOR_DIFFUSE] = Color(1, 1, 1);
			colors[VS::LIGHT_COLOR_SPECULAR] = Color(1, 1, 1);
			shadow_enabled = false;
			volumetric_enabled = false;
		}
	};

	struct ShadowBuffer;

	struct LightInstance {

		struct SplitInfo {

			CameraMatrix camera;
			Transform transform;
			float near;
			float far;
		};

		RID light;
		Light *base;
		uint64_t last_pass;
		Transform transform;

		CameraMatrix projection;
		Vector<SplitInfo> splits;

		Vector3 light_vector;
		Vector3 spot_vector;
		float linear_att;

		uint64_t hash_aux;
	};
	mutable RID_Owner<Light> light_owner;
	mutable RID_Owner<LightInstance> light_instance_owner;

	LightInstance *light_instances[MAX_LIGHTS];
	int light_instance_count;

	struct RenderList {

		enum {
			MAX_ELEMENTS = 4096,
			MAX_LIGHTS = 4
		};

		struct Element {

			float depth;
			const Skeleton *skeleton;
			Transform transform;
			LightInstance *lights[MAX_LIGHTS];
			int light_count;
			const Geometry *geometry;
			const Material *material;
			uint64_t light_hash;
			GeometryOwner *owner;
			const ParamOverrideMap *material_overrides;
		};

		Element _elements[MAX_ELEMENTS];
		Element *elements[MAX_ELEMENTS];
		int element_count;

		void clear() {

			element_count = 0;
		}

		struct SortZ {

			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {

				return A->depth > B->depth;
			}
		};

		void sort_z() {

			SortArray<Element *, SortZ> sorter;
			sorter.sort(elements, element_count);
		}

		struct SortSkel {

			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {

				if (A->geometry < B->geometry)
					return true;
				else if (A->geometry > B->geometry)
					return false;
				else
					return (!A->skeleton && B->skeleton);
			}
		};

		void sort_skel() {

			SortArray<Element *, SortSkel> sorter;
			sorter.sort(elements, element_count);
		}

		struct SortMat {

			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {

				if (A->geometry == B->geometry) {

					if (A->material == B->material) {

						return (A->material_overrides < B->material_overrides);
					} else {

						return (A->material < B->material);
					}
				} else {

					return (A->geometry < B->geometry);
				}
			}
		};

		void sort_mat() {

			SortArray<Element *, SortMat> sorter;
			sorter.sort(elements, element_count);
		}

		struct SortMatLight {

			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {

				if (A->geometry == B->geometry) {

					if (A->material == B->material) {

						if (A->light_hash == B->light_hash)
							return (A->material_overrides < B->material_overrides);
						else
							return A->light_hash < B->light_hash;
					} else {

						return (A->material < B->material);
					}
				} else {

					return (A->geometry < B->geometry);
				}
			}
		};

		void sort_mat_light() {

			SortArray<Element *, SortMatLight> sorter;
			sorter.sort(elements, element_count);
		}

		struct LISort {

			_FORCE_INLINE_ bool operator()(const LightInstance *A, const LightInstance *B) const {

				return (A->hash_aux < B->hash_aux);
			}
		};

		_FORCE_INLINE_ void add_element(const Geometry *p_geometry, const Material *p_material, const Transform &p_transform, LightInstance **p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides, const Skeleton *p_skeleton, float p_depth, GeometryOwner *p_owner = NULL) {

			ERR_FAIL_COND(element_count >= MAX_ELEMENTS);
			Element *e = elements[element_count++];

			e->geometry = p_geometry;
			e->material = p_material;
			e->transform = p_transform;
			e->skeleton = p_skeleton;
			e->light_hash = 0;
			e->light_count = p_light_count;
			e->owner = p_owner;
			e->material_overrides = p_material_overrides;

			if (e->light_count > 0) {

				SortArray<LightInstance *, LISort> light_sort;
				light_sort.sort(p_light_instances, p_light_count);
				//@TODO OPTIOMIZE

				for (int i = 0; i < p_light_count; i++) {

					e->lights[i] = p_light_instances[i];

					if (i == 0)
						e->light_hash = hash_djb2_one_64(make_uint64_t(e->lights[i]));
					else
						e->light_hash = hash_djb2_one_64(make_uint64_t(e->lights[i]), e->light_hash);
				}
			}
		}

		RenderList() {

			for (int i = 0; i < MAX_ELEMENTS; i++)
				elements[i] = &_elements[i]; // assign elements
		}
	};

	RenderList opaque_render_list;
	RenderList alpha_render_list;

	RID default_material;

	struct FX {

		bool bgcolor_active;
		Color bgcolor;

		bool skybox_active;
		RID skybox_cubemap;

		bool antialias_active;
		float antialias_tolerance;

		bool glow_active;
		int glow_passes;
		float glow_attenuation;
		float glow_bloom;

		bool ssao_active;
		float ssao_attenuation;
		float ssao_radius;
		float ssao_max_distance;
		float ssao_range_max;
		float ssao_range_min;
		bool ssao_only;

		bool fog_active;
		float fog_distance;
		float fog_attenuation;
		Color fog_color_near;
		Color fog_color_far;
		bool fog_bg;

		bool toon_active;
		float toon_treshold;
		float toon_soft;

		bool edge_active;
		Color edge_color;
		float edge_size;

		FX();
	};
	mutable RID_Owner<FX> fx_owner;

	FX *scene_fx;
	CameraMatrix camera_projection;
	Transform camera_transform;
	Transform camera_transform_inverse;
	float camera_z_near;
	float camera_z_far;
	Size2 camera_vp_size;

	Plane camera_plane;

	void _add_geometry(const Geometry *p_geometry, const Transform &p_world, uint32_t p_vertex_format, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides, const Skeleton *p_skeleton, GeometryOwner *p_owner);
	void _render_list_forward(RenderList *p_render_list);

	void _setup_light(LightInstance *p_instance, int p_idx);
	void _setup_lights(LightInstance **p_lights, int p_light_count);
	void _setup_material(const Geometry *p_geometry, const Material *p_material);

	void _setup_geometry(const Geometry *p_geometry, const Material *p_material);
	void _render(const Geometry *p_geometry, const Material *p_material, const Skeleton *p_skeleton);

	/*********/
	/* FRAME */
	/*********/

	Size2 window_size;
	VS::ViewportRect viewport;
	Transform canvas_transform;
	double last_time;
	double time_delta;
	uint64_t frame;

public:
	/* TEXTURE API */

	virtual RID texture_create();
	virtual void texture_allocate(RID p_texture, int p_width, int p_height, Image::Format p_format, uint32_t p_flags = VS::TEXTURE_FLAGS_DEFAULT);
	virtual void texture_blit_rect(RID p_texture, int p_x, int p_y, const Image &p_image, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT);
	virtual Image texture_get_rect(RID p_texture, int p_x, int p_y, int p_width, int p_height, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT) const;
	virtual void texture_set_flags(RID p_texture, uint32_t p_flags);
	virtual uint32_t texture_get_flags(RID p_texture) const;
	virtual Image::Format texture_get_format(RID p_texture) const;
	virtual uint32_t texture_get_width(RID p_texture) const;
	virtual uint32_t texture_get_height(RID p_texture) const;
	virtual bool texture_has_alpha(RID p_texture) const;

	/* SHADER API */

	virtual RID shader_create();

	virtual void shader_node_add(RID p_shader, VS::ShaderNodeType p_type, int p_id);
	virtual void shader_node_remove(RID p_shader, int p_id);
	virtual void shader_node_change_type(RID p_shader, int p_id, VS::ShaderNodeType p_type);
	virtual void shader_node_set_param(RID p_shader, int p_id, const Variant &p_value);

	virtual void shader_get_node_list(RID p_shader, List<int> *p_node_list) const;
	virtual VS::ShaderNodeType shader_node_get_type(RID p_shader, int p_id) const;
	virtual Variant shader_node_get_param(RID p_shader, int p_id) const;

	virtual void shader_connect(RID p_shader, int p_src_id, int p_src_slot, int p_dst_id, int p_dst_slot);
	virtual bool shader_is_connected(RID p_shader, int p_src_id, int p_src_slot, int p_dst_id, int p_dst_slot) const;
	virtual void shader_disconnect(RID p_shader, int p_src_id, int p_src_slot, int p_dst_id, int p_dst_slot);

	virtual void shader_get_connections(RID p_shader, List<VS::ShaderConnection> *p_connections) const;

	virtual void shader_clear(RID p_shader);

	/* COMMON MATERIAL API */

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value);
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const;
	virtual void material_get_param_list(RID p_material, List<String> *p_param_list) const;

	virtual void material_set_flag(RID p_material, VS::MaterialFlag p_flag, bool p_enabled);
	virtual bool material_get_flag(RID p_material, VS::MaterialFlag p_flag) const;

	virtual void material_set_blend_mode(RID p_material, VS::MaterialBlendMode p_mode);
	virtual VS::MaterialBlendMode material_get_blend_mode(RID p_material) const;

	virtual void material_set_line_width(RID p_material, float p_line_width);
	virtual float material_get_line_width(RID p_material) const;

	/* FIXED MATERIAL */

	virtual RID material_create();

	virtual void fixed_material_set_parameter(RID p_material, VS::SpatialMaterialParam p_parameter, const Variant &p_value);
	virtual Variant fixed_material_get_parameter(RID p_material, VS::SpatialMaterialParam p_parameter) const;

	virtual void fixed_material_set_texture(RID p_material, VS::SpatialMaterialParam p_parameter, RID p_texture);
	virtual RID fixed_material_get_texture(RID p_material, VS::SpatialMaterialParam p_parameter) const;

	virtual void fixed_material_set_detail_blend_mode(RID p_material, VS::MaterialBlendMode p_mode);
	virtual VS::MaterialBlendMode fixed_material_get_detail_blend_mode(RID p_material) const;

	virtual void fixed_material_set_texgen_mode(RID p_material, VS::SpatialMaterialTexGenMode p_mode);
	virtual VS::SpatialMaterialTexGenMode fixed_material_get_texgen_mode(RID p_material) const;

	virtual void fixed_material_set_texcoord_mode(RID p_material, VS::SpatialMaterialParam p_parameter, VS::SpatialMaterialTexCoordMode p_mode);
	virtual VS::SpatialMaterialTexCoordMode fixed_material_get_texcoord_mode(RID p_material, VS::SpatialMaterialParam p_parameter) const;

	virtual void fixed_material_set_uv_transform(RID p_material, const Transform &p_transform);
	virtual Transform fixed_material_get_uv_transform(RID p_material) const;

	/* SHADER MATERIAL */

	virtual RID shader_material_create() const;

	virtual void shader_material_set_vertex_shader(RID p_material, RID p_shader, bool p_owned = false);
	virtual RID shader_material_get_vertex_shader(RID p_material) const;

	virtual void shader_material_set_fragment_shader(RID p_material, RID p_shader, bool p_owned = false);
	virtual RID shader_material_get_fragment_shader(RID p_material) const;

	/* MESH API */

	virtual RID mesh_create();

	virtual void mesh_add_surface(RID p_mesh, VS::PrimitiveType p_primitive, uint32_t p_format, int p_array_len, int p_index_array_len = VS::NO_INDEX_ARRAY);

	virtual Error mesh_surface_set_array(RID p_mesh, int p_surface, VS::ArrayType p_type, const Variant &p_array);
	virtual Variant mesh_surface_get_array(RID p_mesh, int p_surface, VS::ArrayType p_type) const;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material, bool p_owned = false);
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const;

	virtual int mesh_surface_get_array_len(RID p_mesh, int p_surface) const;
	virtual int mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const;
	virtual uint32_t mesh_surface_get_format(RID p_mesh, int p_surface) const;
	virtual VS::PrimitiveType mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const;

	virtual void mesh_erase_surface(RID p_mesh, int p_index);
	virtual int mesh_get_surface_count(RID p_mesh) const;

	virtual AABB mesh_get_aabb(RID p_mesh) const;

	/* MULTIMESH API */

	virtual RID multimesh_create();

	virtual void multimesh_set_instance_count(RID p_multimesh, int p_count);
	virtual int multimesh_get_instance_count(RID p_multimesh) const;

	virtual void multimesh_set_mesh(RID p_multimesh, RID p_mesh);
	virtual void multimesh_set_aabb(RID p_multimesh, const AABB &p_aabb);
	virtual void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform);
	virtual void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color);

	virtual RID multimesh_get_mesh(RID p_multimesh) const;
	virtual AABB multimesh_get_aabb(RID p_multimesh) const;

	virtual Transform multimesh_instance_get_transform(RID p_multimesh, int p_index) const;
	virtual Color multimesh_instance_get_color(RID p_multimesh, int p_index) const;

	/* POLY API */

	virtual RID poly_create();
	virtual void poly_set_material(RID p_poly, RID p_material, bool p_owned = false);
	virtual void poly_add_primitive(RID p_poly, const Vector<Vector3> &p_points, const Vector<Vector3> &p_normals, const Vector<Color> &p_colors, const Vector<Vector3> &p_uvs);
	virtual void poly_clear(RID p_poly);

	virtual AABB poly_get_aabb(RID p_poly) const;

	/* PARTICLES API */

	virtual RID particles_create();

	virtual void particles_set_amount(RID p_particles, int p_amount);
	virtual int particles_get_amount(RID p_particles) const;

	virtual void particles_set_emitting(RID p_particles, bool p_emitting);
	virtual bool particles_is_emitting(RID p_particles) const;

	virtual void particles_set_visibility_aabb(RID p_particles, const AABB &p_visibility);
	virtual AABB particles_get_visibility_aabb(RID p_particles) const;

	virtual void particles_set_emission_half_extents(RID p_particles, const Vector3 &p_half_extents);
	virtual Vector3 particles_get_emission_half_extents(RID p_particles) const;

	virtual void particles_set_gravity_normal(RID p_particles, const Vector3 &p_normal);
	virtual Vector3 particles_get_gravity_normal(RID p_particles) const;

	virtual void particles_set_variable(RID p_particles, VS::ParticleVariable p_variable, float p_value);
	virtual float particles_get_variable(RID p_particles, VS::ParticleVariable p_variable) const;

	virtual void particles_set_randomness(RID p_particles, VS::ParticleVariable p_variable, float p_randomness);
	virtual float particles_get_randomness(RID p_particles, VS::ParticleVariable p_variable) const;

	virtual void particles_set_color_phase_pos(RID p_particles, int p_phase, float p_pos);
	virtual float particles_get_color_phase_pos(RID p_particles, int p_phase) const;

	virtual void particles_set_color_phases(RID p_particles, int p_phases);
	virtual int particles_get_color_phases(RID p_particles) const;

	virtual void particles_set_color_phase_color(RID p_particles, int p_phase, const Color &p_color);
	virtual Color particles_get_color_phase_color(RID p_particles, int p_phase) const;

	virtual void particles_set_attractors(RID p_particles, int p_attractors);
	virtual int particles_get_attractors(RID p_particles) const;

	virtual void particles_set_attractor_pos(RID p_particles, int p_attractor, const Vector3 &p_pos);
	virtual Vector3 particles_get_attractor_pos(RID p_particles, int p_attractor) const;

	virtual void particles_set_attractor_strength(RID p_particles, int p_attractor, float p_force);
	virtual float particles_get_attractor_strength(RID p_particles, int p_attractor) const;

	virtual void particles_set_material(RID p_particles, RID p_material, bool p_owned = false);
	virtual RID particles_get_material(RID p_particles) const;

	virtual AABB particles_get_aabb(RID p_particles) const;
	/* BEAM API */

	virtual RID beam_create();

	virtual void beam_set_point_count(RID p_beam, int p_count);
	virtual int beam_get_point_count(RID p_beam) const;
	virtual void beam_clear(RID p_beam);

	virtual void beam_set_point(RID p_beam, int p_point, Vector3 &p_pos);
	virtual Vector3 beam_get_point(RID p_beam, int p_point) const;

	virtual void beam_set_primitive(RID p_beam, VS::BeamPrimitive p_primitive);
	virtual VS::BeamPrimitive beam_get_primitive(RID p_beam) const;

	virtual void beam_set_material(RID p_beam, RID p_material);
	virtual RID beam_get_material(RID p_beam) const;

	virtual AABB beam_get_aabb(RID p_particles) const;
	/* SKELETON API */

	virtual RID skeleton_create();
	virtual void skeleton_resize(RID p_skeleton, int p_bones);
	virtual int skeleton_get_bone_count(RID p_skeleton) const;
	virtual void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform);
	virtual Transform skeleton_bone_get_transform(RID p_skeleton, int p_bone);

	/* LIGHT API */

	virtual RID light_create(VS::LightType p_type);
	virtual VS::LightType light_get_type(RID p_light) const;

	virtual void light_set_color(RID p_light, VS::LightColor p_type, const Color &p_color);
	virtual Color light_get_color(RID p_light, VS::LightColor p_type) const;

	virtual void light_set_shadow(RID p_light, bool p_enabled);
	virtual bool light_has_shadow(RID p_light) const;

	virtual void light_set_volumetric(RID p_light, bool p_enabled);
	virtual bool light_is_volumetric(RID p_light) const;

	virtual void light_set_projector(RID p_light, RID p_texture);
	virtual RID light_get_projector(RID p_light) const;

	virtual void light_set_var(RID p_light, VS::LightParam p_var, float p_value);
	virtual float light_get_var(RID p_light, VS::LightParam p_var) const;

	virtual AABB light_get_aabb(RID p_poly) const;

	virtual RID light_instance_create(RID p_light);
	virtual void light_instance_set_transform(RID p_light_instance, const Transform &p_transform);

	virtual void light_instance_set_active_hint(RID p_light_instance);
	virtual bool light_instance_has_shadow(RID p_light_instance) const;
	virtual bool light_instance_assign_shadow(RID p_light_instance);
	virtual ShadowType light_instance_get_shadow_type(RID p_light_instance) const;
	virtual int light_instance_get_shadow_passes(RID p_light_instance) const;
	virtual void light_instance_set_pssm_split_info(RID p_light_instance, int p_split, float p_near, float p_far, const CameraMatrix &p_camera, const Transform &p_transform);

	/* PARTICLES INSTANCE */

	virtual RID particles_instance_create(RID p_particles);
	virtual void particles_instance_set_transform(RID p_particles_instance, const Transform &p_transform);

	/* RENDER API */
	/* all calls (inside begin/end shadow) are always warranted to be in the following order: */

	virtual void begin_frame();

	virtual void set_viewport(const VS::ViewportRect &p_viewport);

	virtual void begin_scene(RID p_fx = RID(), VS::ScenarioDebugMode p_debug = VS::SCENARIO_DEBUG_DISABLED);
	virtual void begin_shadow_map(RID p_light_instance, int p_shadow_pass);

	virtual void set_camera(const Transform &p_world, const CameraMatrix &p_projection);

	virtual void add_light(RID p_light_instance); ///< all "add_light" calls happen before add_geometry calls

	typedef Map<StringName, Variant> ParamOverrideMap;

	virtual void add_mesh(RID p_mesh, const Transform *p_world, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides = NULL, RID p_skeleton = RID());
	virtual void add_multimesh(RID p_multimesh, const Transform *p_world, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides = NULL);
	virtual void add_poly(RID p_poly, const Transform *p_world, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides = NULL);
	virtual void add_beam(RID p_beam, const Transform *p_world, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides = NULL);
	virtual void add_particles(RID p_particle_instance, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides = NULL);

	virtual void end_scene();
	virtual void end_shadow_map();

	virtual void end_frame();

	/* CANVAS API */

	virtual void canvas_begin();
	virtual void canvas_set_transparency(float p_transparency);
	virtual void canvas_set_rect(const Rect2 &p_rect, bool p_clip);
	virtual void canvas_draw_line(const Point2 &p_from, const Point2 &p_to, const Color &p_color, float p_width);
	virtual void canvas_draw_rect(const Rect2 &p_rect, bool p_region, const Rect2 &p_source, bool p_tile, RID p_texture, const Color &p_modulate);
	virtual void canvas_draw_style_box(const Rect2 &p_rect, RID p_texture, const float *p_margins, bool p_draw_center = true);
	virtual void canvas_draw_primitive(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, RID p_texture);

	/* FX */

	virtual RID fx_create();
	virtual void fx_get_effects(RID p_fx, List<String> *p_effects) const;
	virtual void fx_set_active(RID p_fx, const String &p_effect, bool p_active);
	virtual bool fx_is_active(RID p_fx, const String &p_effect) const;
	virtual void fx_get_effect_params(RID p_fx, const String &p_effect, List<PropertyInfo> *p_params) const;
	virtual Variant fx_get_effect_param(RID p_fx, const String &p_effect, const String &p_param) const;
	virtual void fx_set_effect_param(RID p_fx, const String &p_effect, const String &p_param, const Variant &p_pvalue);

	/*MISC*/

	virtual bool is_texture(const RID &p_rid) const;
	virtual bool is_material(const RID &p_rid) const;
	virtual bool is_mesh(const RID &p_rid) const;
	virtual bool is_multimesh(const RID &p_rid) const;
	virtual bool is_poly(const RID &p_rid) const;
	virtual bool is_particles(const RID &p_beam) const;
	virtual bool is_beam(const RID &p_beam) const;

	virtual bool is_light(const RID &p_rid) const;
	virtual bool is_light_instance(const RID &p_rid) const;
	virtual bool is_particles_instance(const RID &p_rid) const;
	virtual bool is_skeleton(const RID &p_rid) const;
	virtual bool is_fx(const RID &p_rid) const;
	virtual bool is_shader(const RID &p_rid) const;

	virtual void free(const RID &p_rid) const;

	virtual void init();
	virtual void finish();

	virtual int get_render_info(VS::RenderInfo p_info);

	RasterizerIPhone();
	virtual ~RasterizerIPhone();
};

#endif
#endif
