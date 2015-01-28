/*************************************************************************/
/*  visual_server.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef VISUAL_SERVER_H
#define VISUAL_SERVER_H


#include "rid.h"
#include "variant.h"
#include "math_2d.h"
#include "bsp_tree.h"
#include "geometry.h"
#include "object.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class VisualServer : public Object {

	OBJ_TYPE( VisualServer, Object );

	static VisualServer *singleton;

	int mm_policy;

	DVector<String> _shader_get_param_list(RID p_shader) const;
	void _camera_set_orthogonal(RID p_camera,float p_size,float p_z_near,float p_z_far);
	void _viewport_set_rect(RID p_viewport,const Rect2& p_rect);
	Rect2 _viewport_get_rect(RID p_viewport) const;
	void _canvas_item_add_style_box(RID p_item, const Rect2& p_rect, RID p_texture,const Vector<float>& p_margins, const Color& p_modulate=Color(1,1,1));
protected:	
	RID _make_test_cube();
	RID test_texture;
	RID white_texture;
	RID material_2d[16];
	
	static VisualServer* (*create_func)();
	static void _bind_methods();	
public:
	
	static VisualServer *get_singleton();
	static VisualServer *create();

	enum MipMapPolicy {

		MIPMAPS_ENABLED,
		MIPMAPS_ENABLED_FOR_PO2,
		MIPMAPS_DISABLED
	};


	virtual void set_mipmap_policy(MipMapPolicy p_policy);
	virtual MipMapPolicy get_mipmap_policy() const;


	
	enum {
	
		NO_INDEX_ARRAY=-1,
		CUSTOM_ARRAY_SIZE=8,
		ARRAY_WEIGHTS_SIZE=4,
		MAX_PARTICLE_COLOR_PHASES=4,
		MAX_PARTICLE_ATTRACTORS=4,
		CANVAS_ITEM_Z_MIN=-4096,
		CANVAS_ITEM_Z_MAX=4096,



		MAX_CURSORS = 8,
	};
	
	/* TEXTURE API */

	enum TextureFlags {
		TEXTURE_FLAG_MIPMAPS=1, /// Enable automatic mipmap generation - when available
		TEXTURE_FLAG_REPEAT=2, /// Repeat texture (Tiling), otherwise Clamping
		TEXTURE_FLAG_FILTER=4, /// Create texure with linear (or available) filter
		TEXTURE_FLAG_ANISOTROPIC_FILTER=8,
		TEXTURE_FLAG_CONVERT_TO_LINEAR=16,
		TEXTURE_FLAG_CUBEMAP=2048,
		TEXTURE_FLAG_VIDEO_SURFACE=4096,
		TEXTURE_FLAGS_DEFAULT=TEXTURE_FLAG_REPEAT|TEXTURE_FLAG_MIPMAPS|TEXTURE_FLAG_FILTER
	};	
	
	enum CubeMapSide {
	
		CUBEMAP_LEFT,
		CUBEMAP_RIGHT,
		CUBEMAP_BOTTOM,
		CUBEMAP_TOP,
		CUBEMAP_FRONT,	
		CUBEMAP_BACK
	};


	virtual RID texture_create()=0;
	RID texture_create_from_image(const Image& p_image,uint32_t p_flags=TEXTURE_FLAGS_DEFAULT); // helper		
	virtual void texture_allocate(RID p_texture,int p_width, int p_height,Image::Format p_format,uint32_t p_flags=TEXTURE_FLAGS_DEFAULT)=0;
	virtual void texture_set_data(RID p_texture,const Image& p_image,CubeMapSide p_cube_side=CUBEMAP_LEFT)=0;
	virtual Image texture_get_data(RID p_texture,CubeMapSide p_cube_side=CUBEMAP_LEFT) const=0;
	virtual void texture_set_flags(RID p_texture,uint32_t p_flags) =0;
	virtual uint32_t texture_get_flags(RID p_texture) const=0;
	virtual Image::Format texture_get_format(RID p_texture) const=0;
	virtual uint32_t texture_get_width(RID p_texture) const=0;
	virtual uint32_t texture_get_height(RID p_texture) const=0;
	virtual void texture_set_size_override(RID p_texture,int p_width, int p_height)=0;
	virtual bool texture_can_stream(RID p_texture) const=0;
	virtual void texture_set_reload_hook(RID p_texture,ObjectID p_owner,const StringName& p_function) const=0;



	/* SHADER API */

	enum ShaderMode {

		SHADER_MATERIAL,
		SHADER_CANVAS_ITEM,
		SHADER_POST_PROCESS,
	};


	virtual RID shader_create(ShaderMode p_mode=SHADER_MATERIAL)=0;

	virtual void shader_set_mode(RID p_shader,ShaderMode p_mode)=0;
	virtual ShaderMode shader_get_mode(RID p_shader) const=0;

	virtual void shader_set_code(RID p_shader, const String& p_vertex, const String& p_fragment,const String& p_light, int p_vertex_ofs=0,int p_fragment_ofs=0,int p_light_ofs=0)=0;
	virtual String shader_get_fragment_code(RID p_shader) const=0;
	virtual String shader_get_vertex_code(RID p_shader) const=0;
	virtual String shader_get_light_code(RID p_shader) const=0;
	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const=0;

	virtual void shader_set_default_texture_param(RID p_shader, const StringName& p_name, RID p_texture)=0;
	virtual RID shader_get_default_texture_param(RID p_shader, const StringName& p_name) const=0;


	/* COMMON MATERIAL API */

	virtual RID material_create()=0;

	virtual void material_set_shader(RID p_shader_material, RID p_shader)=0;
	virtual RID material_get_shader(RID p_shader_material) const=0;

	virtual void material_set_param(RID p_material, const StringName& p_param, const Variant& p_value)=0;
	virtual Variant material_get_param(RID p_material, const StringName& p_param) const=0;

	enum MaterialFlag {
		MATERIAL_FLAG_VISIBLE,
		MATERIAL_FLAG_DOUBLE_SIDED,
		MATERIAL_FLAG_INVERT_FACES, ///< Invert front/back of the object
		MATERIAL_FLAG_UNSHADED,
		MATERIAL_FLAG_ONTOP,
		MATERIAL_FLAG_LIGHTMAP_ON_UV2,
		MATERIAL_FLAG_COLOR_ARRAY_SRGB,
		MATERIAL_FLAG_MAX,
	};

	virtual void material_set_flag(RID p_material, MaterialFlag p_flag,bool p_enabled)=0;
	virtual bool material_get_flag(RID p_material,MaterialFlag p_flag) const=0;

	enum MaterialDepthDrawMode {
		MATERIAL_DEPTH_DRAW_ALWAYS,
		MATERIAL_DEPTH_DRAW_OPAQUE_ONLY,
		MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA,
		MATERIAL_DEPTH_DRAW_NEVER
	};

	virtual void material_set_depth_draw_mode(RID p_material, MaterialDepthDrawMode p_mode)=0;
	virtual MaterialDepthDrawMode material_get_depth_draw_mode(RID p_material) const=0;

	enum MaterialBlendMode {
		MATERIAL_BLEND_MODE_MIX, //default
		MATERIAL_BLEND_MODE_ADD,
		MATERIAL_BLEND_MODE_SUB,
		MATERIAL_BLEND_MODE_MUL,
		MATERIAL_BLEND_MODE_PREMULT_ALPHA
	};


	virtual void material_set_blend_mode(RID p_material,MaterialBlendMode p_mode)=0;
	virtual MaterialBlendMode material_get_blend_mode(RID p_material) const=0;

	virtual void material_set_line_width(RID p_material,float p_line_width)=0;
	virtual float material_get_line_width(RID p_material) const=0;


	//fixed material api

	virtual RID fixed_material_create()=0;

	enum FixedMaterialParam {

		FIXED_MATERIAL_PARAM_DIFFUSE,
		FIXED_MATERIAL_PARAM_DETAIL,
		FIXED_MATERIAL_PARAM_SPECULAR,
		FIXED_MATERIAL_PARAM_EMISSION,
		FIXED_MATERIAL_PARAM_SPECULAR_EXP,
		FIXED_MATERIAL_PARAM_GLOW,
		FIXED_MATERIAL_PARAM_NORMAL,
		FIXED_MATERIAL_PARAM_SHADE_PARAM,		
		FIXED_MATERIAL_PARAM_MAX
	};

	enum FixedMaterialTexCoordMode {

		FIXED_MATERIAL_TEXCOORD_UV,
		FIXED_MATERIAL_TEXCOORD_UV_TRANSFORM,
		FIXED_MATERIAL_TEXCOORD_UV2,
		FIXED_MATERIAL_TEXCOORD_SPHERE
	};

	enum FixedMaterialFlags {

		FIXED_MATERIAL_FLAG_USE_ALPHA,
		FIXED_MATERIAL_FLAG_USE_COLOR_ARRAY,
		FIXED_MATERIAL_FLAG_USE_POINT_SIZE,
		FIXED_MATERIAL_FLAG_DISCARD_ALPHA,
		FIXED_MATERIAL_FLAG_USE_XY_NORMALMAP,
		FIXED_MATERIAL_FLAG_MAX,
	};


	virtual void fixed_material_set_flag(RID p_material, FixedMaterialFlags p_flag, bool p_enabled)=0;
	virtual bool fixed_material_get_flag(RID p_material, FixedMaterialFlags p_flag) const=0;

	virtual void fixed_material_set_param(RID p_material, FixedMaterialParam p_parameter, const Variant& p_value)=0;
	virtual Variant fixed_material_get_param(RID p_material,FixedMaterialParam p_parameter) const=0;

	virtual void fixed_material_set_texture(RID p_material,FixedMaterialParam p_parameter, RID p_texture)=0;
	virtual RID fixed_material_get_texture(RID p_material,FixedMaterialParam p_parameter) const=0;


	enum FixedMaterialLightShader {

		FIXED_MATERIAL_LIGHT_SHADER_LAMBERT,
		FIXED_MATERIAL_LIGHT_SHADER_WRAP,
		FIXED_MATERIAL_LIGHT_SHADER_VELVET,
		FIXED_MATERIAL_LIGHT_SHADER_TOON,

	};


	virtual void fixed_material_set_light_shader(RID p_material,FixedMaterialLightShader p_shader)=0;
	virtual FixedMaterialLightShader fixed_material_get_light_shader(RID p_material) const=0;

	virtual void fixed_material_set_texcoord_mode(RID p_material,FixedMaterialParam p_parameter, FixedMaterialTexCoordMode p_mode)=0;
	virtual FixedMaterialTexCoordMode fixed_material_get_texcoord_mode(RID p_material,FixedMaterialParam p_parameter) const=0;

	virtual void fixed_material_set_uv_transform(RID p_material,const Transform& p_transform)=0;
	virtual Transform fixed_material_get_uv_transform(RID p_material) const=0;

	virtual void fixed_material_set_point_size(RID p_material,float p_size)=0;
	virtual float fixed_material_get_point_size(RID p_material) const=0;

	/* MESH API */

	enum ArrayType {

		ARRAY_VERTEX=0,
		ARRAY_NORMAL=1,
		ARRAY_TANGENT=2,
		ARRAY_COLOR=3,		
		ARRAY_TEX_UV=4,		
		ARRAY_TEX_UV2=5,
		ARRAY_BONES=6,
		ARRAY_WEIGHTS=7,		
		ARRAY_INDEX=8,
		ARRAY_MAX=9
	};	
	
	enum ArrayFormat {
		/* ARRAY FORMAT FLAGS */
		ARRAY_FORMAT_VERTEX=1<<ARRAY_VERTEX, // mandatory
		ARRAY_FORMAT_NORMAL=1<<ARRAY_NORMAL,
		ARRAY_FORMAT_TANGENT=1<<ARRAY_TANGENT,
		ARRAY_FORMAT_COLOR=1<<ARRAY_COLOR,
		ARRAY_FORMAT_TEX_UV=1<<ARRAY_TEX_UV,
		ARRAY_FORMAT_TEX_UV2=1<<ARRAY_TEX_UV2,
		ARRAY_FORMAT_BONES=1<<ARRAY_BONES,
		ARRAY_FORMAT_WEIGHTS=1<<ARRAY_WEIGHTS,
		ARRAY_FORMAT_INDEX=1<<ARRAY_INDEX,
	};	
		
	enum PrimitiveType {
		PRIMITIVE_POINTS=0,
		PRIMITIVE_LINES=1,
		PRIMITIVE_LINE_STRIP=2,
		PRIMITIVE_LINE_LOOP=3,
		PRIMITIVE_TRIANGLES=4,
		PRIMITIVE_TRIANGLE_STRIP=5,
		PRIMITIVE_TRIANGLE_FAN=6,
		PRIMITIVE_MAX=7,
	};	

	virtual RID mesh_create()=0;
	
	virtual void mesh_add_surface(RID p_mesh,PrimitiveType p_primitive,const Array& p_arrays,const Array& p_blend_shapes=Array(),bool p_alpha_sort=false)=0;
	virtual Array mesh_get_surface_arrays(RID p_mesh,int p_surface) const=0;
	virtual Array mesh_get_surface_morph_arrays(RID p_mesh,int p_surface) const=0;


	virtual void mesh_add_custom_surface(RID p_mesh,const Variant& p_dat)=0; //this is used by each platform in a different way
	virtual void mesh_set_morph_target_count(RID p_mesh,int p_amount)=0;
	virtual int mesh_get_morph_target_count(RID p_mesh) const=0;

	enum MorphTargetMode {
		MORPH_MODE_NORMALIZED,
		MORPH_MODE_RELATIVE,
	};

	virtual void mesh_set_morph_target_mode(RID p_mesh,MorphTargetMode p_mode)=0;
	virtual MorphTargetMode mesh_get_morph_target_mode(RID p_mesh) const=0;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material,bool p_owned=false)=0;
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const=0;

	virtual int mesh_surface_get_array_len(RID p_mesh, int p_surface) const=0;
	virtual int mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const=0;
	virtual uint32_t mesh_surface_get_format(RID p_mesh, int p_surface) const=0;
	virtual PrimitiveType mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const=0;
	
	virtual void mesh_remove_surface(RID p_mesh,int p_index)=0;
	virtual int mesh_get_surface_count(RID p_mesh) const=0;

	virtual void mesh_set_custom_aabb(RID p_mesh,const AABB& p_aabb)=0;
	virtual AABB mesh_get_custom_aabb(RID p_mesh) const=0;

	/* MULTIMESH API */

	virtual RID multimesh_create()=0;

	virtual void multimesh_set_instance_count(RID p_multimesh,int p_count)=0;
	virtual int multimesh_get_instance_count(RID p_multimesh) const=0;

	virtual void multimesh_set_mesh(RID p_multimesh,RID p_mesh)=0;
	virtual void multimesh_set_aabb(RID p_multimesh,const AABB& p_aabb)=0;
	virtual void multimesh_instance_set_transform(RID p_multimesh,int p_index,const Transform& p_transform)=0;
	virtual void multimesh_instance_set_color(RID p_multimesh,int p_index,const Color& p_color)=0;

	virtual RID multimesh_get_mesh(RID p_multimesh) const=0;
	virtual AABB multimesh_get_aabb(RID p_multimesh,const AABB& p_aabb) const=0;;

	virtual Transform multimesh_instance_get_transform(RID p_multimesh,int p_index) const=0;
	virtual Color multimesh_instance_get_color(RID p_multimesh,int p_index) const=0;

	virtual void multimesh_set_visible_instances(RID p_multimesh,int p_visible)=0;
	virtual int multimesh_get_visible_instances(RID p_multimesh) const=0;

	/* IMMEDIATE API */

	virtual RID immediate_create()=0;
	virtual void immediate_begin(RID p_immediate,PrimitiveType p_rimitive,RID p_texture=RID())=0;
	virtual void immediate_vertex(RID p_immediate,const Vector3& p_vertex)=0;
	virtual void immediate_normal(RID p_immediate,const Vector3& p_normal)=0;
	virtual void immediate_tangent(RID p_immediate,const Plane& p_tangent)=0;
	virtual void immediate_color(RID p_immediate,const Color& p_color)=0;
	virtual void immediate_uv(RID p_immediate,const Vector2& tex_uv)=0;
	virtual void immediate_uv2(RID p_immediate,const Vector2& tex_uv)=0;
	virtual void immediate_end(RID p_immediate)=0;
	virtual void immediate_clear(RID p_immediate)=0;
	virtual void immediate_set_material(RID p_immediate,RID p_material)=0;
	virtual RID immediate_get_material(RID p_immediate) const=0;


	/* PARTICLES API */
		
	virtual RID particles_create()=0;
	
	enum ParticleVariable {
		PARTICLE_LIFETIME,
		PARTICLE_SPREAD,
		PARTICLE_GRAVITY,
		PARTICLE_LINEAR_VELOCITY,
		PARTICLE_ANGULAR_VELOCITY,
		PARTICLE_LINEAR_ACCELERATION,
		PARTICLE_RADIAL_ACCELERATION,
		PARTICLE_TANGENTIAL_ACCELERATION,
		PARTICLE_DAMPING,
		PARTICLE_INITIAL_SIZE,
		PARTICLE_FINAL_SIZE,
		PARTICLE_INITIAL_ANGLE,
		PARTICLE_HEIGHT,
		PARTICLE_HEIGHT_SPEED_SCALE,
		PARTICLE_VAR_MAX
	};
		
	virtual void particles_set_amount(RID p_particles, int p_amount)=0;
	virtual int particles_get_amount(RID p_particles) const=0;
		
	virtual void particles_set_emitting(RID p_particles, bool p_emitting)=0;
	virtual bool particles_is_emitting(RID p_particles) const=0;
		
	virtual void particles_set_visibility_aabb(RID p_particles, const AABB& p_visibility)=0;
	virtual AABB particles_get_visibility_aabb(RID p_particles) const=0;
		
	virtual void particles_set_emission_half_extents(RID p_particles, const Vector3& p_half_extents)=0;
	virtual Vector3 particles_get_emission_half_extents(RID p_particles) const=0;

	virtual void particles_set_emission_base_velocity(RID p_particles, const Vector3& p_base_velocity)=0;
	virtual Vector3 particles_get_emission_base_velocity(RID p_particles) const=0;

	virtual void particles_set_emission_points(RID p_particles, const DVector<Vector3>& p_points)=0;
	virtual DVector<Vector3> particles_get_emission_points(RID p_particles) const=0;
		
	virtual void particles_set_gravity_normal(RID p_particles, const Vector3& p_normal)=0;
	virtual Vector3 particles_get_gravity_normal(RID p_particles) const=0;
		
	virtual void particles_set_variable(RID p_particles, ParticleVariable p_variable,float p_value)=0;
	virtual float particles_get_variable(RID p_particles, ParticleVariable p_variable) const=0;
	
	virtual void particles_set_randomness(RID p_particles, ParticleVariable p_variable,float p_randomness)=0;
	virtual float particles_get_randomness(RID p_particles, ParticleVariable p_variable) const=0;

	virtual void particles_set_color_phases(RID p_particles, int p_phases)=0;
	virtual int particles_get_color_phases(RID p_particles) const=0;

	virtual void particles_set_color_phase_pos(RID p_particles, int p_phase, float p_pos)=0;
	virtual float particles_get_color_phase_pos(RID p_particles, int p_phase) const=0;
	
	virtual void particles_set_color_phase_color(RID p_particles, int p_phase, const Color& p_color)=0;
	virtual Color particles_get_color_phase_color(RID p_particles, int p_phase) const=0;

	virtual void particles_set_attractors(RID p_particles, int p_attractors)=0;
	virtual int particles_get_attractors(RID p_particles) const=0;

	virtual void particles_set_attractor_pos(RID p_particles, int p_attractor, const Vector3& p_pos)=0;
	virtual Vector3 particles_get_attractor_pos(RID p_particles,int p_attractor) const=0;

	virtual void particles_set_attractor_strength(RID p_particles, int p_attractor, float p_force)=0;
	virtual float particles_get_attractor_strength(RID p_particles,int p_attractor) const=0;

	virtual void particles_set_material(RID p_particles, RID p_material,bool p_owned=false)=0;
	virtual RID particles_get_material(RID p_particles) const=0;
	
	virtual void particles_set_height_from_velocity(RID p_particles, bool p_enable)=0;
	virtual bool particles_has_height_from_velocity(RID p_particles) const=0;

	virtual void particles_set_use_local_coordinates(RID p_particles, bool p_enable)=0;
	virtual bool particles_is_using_local_coordinates(RID p_particles) const=0;

	/* Light API */
	
	enum LightType {
		LIGHT_DIRECTIONAL,
		LIGHT_OMNI,
		LIGHT_SPOT
	};

	enum LightColor {		
		LIGHT_COLOR_DIFFUSE,
		LIGHT_COLOR_SPECULAR
	};
	
	enum LightParam {
		
		LIGHT_PARAM_SPOT_ATTENUATION,
		LIGHT_PARAM_SPOT_ANGLE,
		LIGHT_PARAM_RADIUS,
		LIGHT_PARAM_ENERGY,
		LIGHT_PARAM_ATTENUATION,
		LIGHT_PARAM_SHADOW_DARKENING,
		LIGHT_PARAM_SHADOW_Z_OFFSET,
		LIGHT_PARAM_SHADOW_Z_SLOPE_SCALE,
		LIGHT_PARAM_SHADOW_ESM_MULTIPLIER,
		LIGHT_PARAM_SHADOW_BLUR_PASSES,
		LIGHT_PARAM_MAX
	};

	virtual RID light_create(LightType p_type)=0;
	virtual LightType light_get_type(RID p_light) const=0;

	virtual void light_set_color(RID p_light,LightColor p_type, const Color& p_color)=0;
	virtual Color light_get_color(RID p_light,LightColor p_type) const=0;	

	virtual void light_set_shadow(RID p_light,bool p_enabled)=0;
	virtual bool light_has_shadow(RID p_light) const=0;	
	
	virtual void light_set_volumetric(RID p_light,bool p_enabled)=0;
	virtual bool light_is_volumetric(RID p_light) const=0;	
	
	virtual void light_set_projector(RID p_light,RID p_texture)=0;
	virtual RID light_get_projector(RID p_light) const=0;
	
	virtual void light_set_param(RID p_light, LightParam p_var, float p_value)=0;
	virtual float light_get_param(RID p_light, LightParam p_var) const=0;
	
	enum LightOp {

		LIGHT_OPERATOR_ADD,
		LIGHT_OPERATOR_SUB
	};

	virtual void light_set_operator(RID p_light,LightOp p_op)=0;
	virtual LightOp light_get_operator(RID p_light) const=0;

	// omni light
	enum LightOmniShadowMode {
		LIGHT_OMNI_SHADOW_DEFAULT,
		LIGHT_OMNI_SHADOW_DUAL_PARABOLOID,
		LIGHT_OMNI_SHADOW_CUBEMAP
	};

	virtual void light_omni_set_shadow_mode(RID p_light,LightOmniShadowMode p_mode)=0;
	virtual LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) const=0;

	// directional light
	enum LightDirectionalShadowMode {
		LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL,
		LIGHT_DIRECTIONAL_SHADOW_PERSPECTIVE,
		LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS,
		LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS
	};

	virtual void light_directional_set_shadow_mode(RID p_light,LightDirectionalShadowMode p_mode)=0;
	virtual LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) const=0;

	enum LightDirectionalShadowParam {

		LIGHT_DIRECTIONAL_SHADOW_PARAM_MAX_DISTANCE,
		LIGHT_DIRECTIONAL_SHADOW_PARAM_PSSM_SPLIT_WEIGHT,
		LIGHT_DIRECTIONAL_SHADOW_PARAM_PSSM_ZOFFSET_SCALE,
	};

	virtual void light_directional_set_shadow_param(RID p_light,LightDirectionalShadowParam p_param, float p_value)=0;
	virtual float light_directional_get_shadow_param(RID p_light,LightDirectionalShadowParam p_param) const=0;

	//@TODO fallof model and all that stuff
	
	/* SKELETON API */
	
	virtual RID skeleton_create()=0;
	virtual void skeleton_resize(RID p_skeleton,int p_bones)=0;
	virtual int skeleton_get_bone_count(RID p_skeleton) const=0;
	virtual void skeleton_bone_set_transform(RID p_skeleton,int p_bone, const Transform& p_transform)=0;
	virtual Transform skeleton_bone_get_transform(RID p_skeleton,int p_bone)=0;
	
	/* ROOM API */

	virtual RID room_create()=0;
	virtual void room_set_bounds(RID p_room, const BSP_Tree& p_bounds)=0;
	virtual BSP_Tree room_get_bounds(RID p_room) const=0;
			
	/* PORTAL API */

	// portals are only (x/y) points, forming a convex shape, which its clockwise
	// order points outside. (z is 0);
	
	virtual RID portal_create()=0;
	virtual void portal_set_shape(RID p_portal, const Vector<Point2>& p_shape)=0;
	virtual Vector<Point2> portal_get_shape(RID p_portal) const=0;
	virtual void portal_set_enabled(RID p_portal, bool p_enabled)=0;
	virtual bool portal_is_enabled(RID p_portal) const=0;
	virtual void portal_set_disable_distance(RID p_portal, float p_distance)=0;
	virtual float portal_get_disable_distance(RID p_portal) const=0;
	virtual void portal_set_disabled_color(RID p_portal, const Color& p_color)=0;
	virtual Color portal_get_disabled_color(RID p_portal) const=0;
	virtual void portal_set_connect_range(RID p_portal, float p_range) =0;
	virtual float portal_get_connect_range(RID p_portal) const =0;


	/* BAKED LIGHT API */

	virtual RID baked_light_create()=0;
	enum BakedLightMode {
		BAKED_LIGHT_OCTREE,
		BAKED_LIGHT_LIGHTMAPS
	};

	virtual void baked_light_set_mode(RID p_baked_light,BakedLightMode p_mode)=0;
	virtual BakedLightMode baked_light_get_mode(RID p_baked_light) const=0;

	virtual void baked_light_set_octree(RID p_baked_light,const DVector<uint8_t> p_octree)=0;
	virtual DVector<uint8_t> baked_light_get_octree(RID p_baked_light) const=0;

	virtual void baked_light_set_light(RID p_baked_light,const DVector<uint8_t> p_light)=0;
	virtual DVector<uint8_t> baked_light_get_light(RID p_baked_light) const=0;

	virtual void baked_light_set_sampler_octree(RID p_baked_light,const DVector<int> &p_sampler)=0;
	virtual DVector<int> baked_light_get_sampler_octree(RID p_baked_light) const=0;

	virtual void baked_light_set_lightmap_multiplier(RID p_baked_light,float p_multiplier)=0;
	virtual float baked_light_get_lightmap_multiplier(RID p_baked_light) const=0;

	virtual void baked_light_add_lightmap(RID p_baked_light,const RID p_texture,int p_id)=0;
	virtual void baked_light_clear_lightmaps(RID p_baked_light)=0;

	/* BAKED LIGHT SAMPLER */

	virtual RID baked_light_sampler_create()=0;

	enum BakedLightSamplerParam {
		BAKED_LIGHT_SAMPLER_RADIUS,
		BAKED_LIGHT_SAMPLER_STRENGTH,
		BAKED_LIGHT_SAMPLER_ATTENUATION,
		BAKED_LIGHT_SAMPLER_DETAIL_RATIO,
		BAKED_LIGHT_SAMPLER_MAX
	};

	virtual void baked_light_sampler_set_param(RID p_baked_light_sampler,BakedLightSamplerParam p_param,float p_value)=0;
	virtual float baked_light_sampler_get_param(RID p_baked_light_sampler,BakedLightSamplerParam p_param) const=0;

	virtual void baked_light_sampler_set_resolution(RID p_baked_light_sampler,int p_resolution)=0;
	virtual int baked_light_sampler_get_resolution(RID p_baked_light_sampler) const=0;

	/* CAMERA API */
	
	virtual RID camera_create()=0;
	virtual void camera_set_perspective(RID p_camera,float p_fovy_degrees, float p_z_near, float p_z_far)=0;
	virtual void camera_set_orthogonal(RID p_camera,float p_size, float p_z_near, float p_z_far)=0;
	virtual void camera_set_transform(RID p_camera,const Transform& p_transform)=0;	

	virtual void camera_set_visible_layers(RID p_camera,uint32_t p_layers)=0;
	virtual uint32_t camera_get_visible_layers(RID p_camera) const=0;

	virtual void camera_set_environment(RID p_camera,RID p_env)=0;
	virtual RID camera_get_environment(RID p_camera) const=0;

	virtual void camera_set_use_vertical_aspect(RID p_camera,bool p_enable)=0;
	virtual bool camera_is_using_vertical_aspect(RID p_camera,bool p_enable) const=0;

/*
	virtual void camera_add_layer(RID p_camera);
	virtual void camera_layer_move_up(RID p_camera,int p_layer);
	virtual void camera_layer_move_down(RID p_camera,int p_layer);
	virtual void camera_layer_set_mask(RID p_camera,int p_layer,int p_mask);
	virtual int camera_layer_get_mask(RID p_camera,int p_layer) const;

	enum CameraLayerFlag {

		FLAG_CLEAR_DEPTH,
		FLAG_CLEAR_COLOR,
		FLAG_IGNORE_FOG,
	};
	virtual void camera_layer_set_flag(RID p_camera,int p_layer,bool p_enable);
	virtual bool camera_layer_get_flag(RID p_camera,int p_layer) const;

*/


	/* VIEWPORT API */

	virtual RID viewport_create()=0;

	virtual void viewport_attach_to_screen(RID p_viewport,int p_screen=0)=0;
	virtual void viewport_detach(RID p_viewport)=0;
	virtual void viewport_set_render_target_to_screen_rect(RID p_viewport,const Rect2& p_rect)=0;

	enum RenderTargetUpdateMode {
		RENDER_TARGET_UPDATE_DISABLED,
		RENDER_TARGET_UPDATE_ONCE, //then goes to disabled
		RENDER_TARGET_UPDATE_WHEN_VISIBLE, // default
		RENDER_TARGET_UPDATE_ALWAYS
	};


	virtual void viewport_set_as_render_target(RID p_viewport,bool p_enable)=0;
	virtual void viewport_set_render_target_update_mode(RID p_viewport,RenderTargetUpdateMode p_mode)=0;
	virtual RenderTargetUpdateMode viewport_get_render_target_update_mode(RID p_viewport) const=0;
	virtual RID viewport_get_render_target_texture(RID p_viewport) const=0;
	virtual void viewport_set_render_target_vflip(RID p_viewport,bool p_enable)=0;
	virtual bool viewport_get_render_target_vflip(RID p_viewport) const=0;

	virtual void viewport_queue_screen_capture(RID p_viewport)=0;
	virtual Image viewport_get_screen_capture(RID p_viewport) const=0;



	struct ViewportRect {
	
		int x,y,width,height;
		ViewportRect() { x=y=width=height=0; }
	};
	
	virtual void viewport_set_rect(RID p_viewport,const ViewportRect& p_rect)=0;
	virtual ViewportRect viewport_get_rect(RID p_viewport) const=0;
	
	virtual void viewport_set_hide_scenario(RID p_viewport,bool p_hide)=0;
	virtual void viewport_set_hide_canvas(RID p_viewport,bool p_hide)=0;

	virtual void viewport_attach_camera(RID p_viewport,RID p_camera)=0;
	virtual void viewport_set_scenario(RID p_viewport,RID p_scenario)=0;
	virtual RID viewport_get_attached_camera(RID  p_viewport) const=0;
	virtual RID viewport_get_scenario(RID  p_viewport) const=0;
	virtual void viewport_attach_canvas(RID p_viewport,RID p_canvas)=0;
	virtual void viewport_remove_canvas(RID p_viewport,RID p_canvas)=0;
	virtual void viewport_set_canvas_transform(RID p_viewport,RID p_canvas,const Matrix32& p_offset)=0;
	virtual Matrix32 viewport_get_canvas_transform(RID p_viewport,RID p_canvas) const=0;
	virtual void viewport_set_transparent_background(RID p_viewport,bool p_enabled)=0;
	virtual bool viewport_has_transparent_background(RID p_viewport) const=0;


	virtual void viewport_set_global_canvas_transform(RID p_viewport,const Matrix32& p_transform)=0;
	virtual Matrix32 viewport_get_global_canvas_transform(RID p_viewport) const=0;
	virtual void viewport_set_canvas_layer(RID p_viewport,RID p_canvas,int p_layer)=0;



	/* ENVIRONMENT API */

	virtual RID environment_create()=0;

	enum EnvironmentBG {

		ENV_BG_KEEP,
		ENV_BG_DEFAULT_COLOR,
		ENV_BG_COLOR,
		ENV_BG_TEXTURE,
		ENV_BG_CUBEMAP,
		ENV_BG_TEXTURE_RGBE,
		ENV_BG_CUBEMAP_RGBE,
		ENV_BG_MAX
	};

	virtual void environment_set_background(RID p_env,EnvironmentBG p_bg)=0;
	virtual EnvironmentBG environment_get_background(RID p_env) const=0;

	enum EnvironmentBGParam {

		ENV_BG_PARAM_COLOR,
		ENV_BG_PARAM_TEXTURE,
		ENV_BG_PARAM_CUBEMAP,
		ENV_BG_PARAM_ENERGY,
		ENV_BG_PARAM_SCALE,
		ENV_BG_PARAM_GLOW,
		ENV_BG_PARAM_MAX
	};


	virtual void environment_set_background_param(RID p_env,EnvironmentBGParam p_param, const Variant& p_value)=0;
	virtual Variant environment_get_background_param(RID p_env,EnvironmentBGParam p_param) const=0;

	enum EnvironmentFx {
		ENV_FX_AMBIENT_LIGHT,
		ENV_FX_FXAA,
		ENV_FX_GLOW,
		ENV_FX_DOF_BLUR,
		ENV_FX_HDR,
		ENV_FX_FOG,
		ENV_FX_BCS,
		ENV_FX_SRGB,
		ENV_FX_MAX
	};



	virtual void environment_set_enable_fx(RID p_env,EnvironmentFx p_effect,bool p_enabled)=0;
	virtual bool environment_is_fx_enabled(RID p_env,EnvironmentFx p_mode) const=0;

	enum EnvironmentFxBlurBlendMode {
		ENV_FX_BLUR_BLEND_MODE_ADDITIVE,
		ENV_FX_BLUR_BLEND_MODE_SCREEN,
		ENV_FX_BLUR_BLEND_MODE_SOFTLIGHT,
	};

	enum EnvironmentFxHDRToneMapper {
		ENV_FX_HDR_TONE_MAPPER_LINEAR,
		ENV_FX_HDR_TONE_MAPPER_LOG,
		ENV_FX_HDR_TONE_MAPPER_REINHARDT,
		ENV_FX_HDR_TONE_MAPPER_REINHARDT_AUTOWHITE,
	};

	enum EnvironmentFxParam {
		ENV_FX_PARAM_AMBIENT_LIGHT_COLOR,
		ENV_FX_PARAM_AMBIENT_LIGHT_ENERGY,
		ENV_FX_PARAM_GLOW_BLUR_PASSES,
		ENV_FX_PARAM_GLOW_BLUR_SCALE,
		ENV_FX_PARAM_GLOW_BLUR_STRENGTH,
		ENV_FX_PARAM_GLOW_BLUR_BLEND_MODE,
		ENV_FX_PARAM_GLOW_BLOOM,
		ENV_FX_PARAM_GLOW_BLOOM_TRESHOLD,
		ENV_FX_PARAM_DOF_BLUR_PASSES,
		ENV_FX_PARAM_DOF_BLUR_BEGIN,
		ENV_FX_PARAM_DOF_BLUR_RANGE,
		ENV_FX_PARAM_HDR_TONEMAPPER,
		ENV_FX_PARAM_HDR_EXPOSURE,
		ENV_FX_PARAM_HDR_WHITE,
		ENV_FX_PARAM_HDR_GLOW_TRESHOLD,
		ENV_FX_PARAM_HDR_GLOW_SCALE,
		ENV_FX_PARAM_HDR_MIN_LUMINANCE,
		ENV_FX_PARAM_HDR_MAX_LUMINANCE,
		ENV_FX_PARAM_HDR_EXPOSURE_ADJUST_SPEED,
		ENV_FX_PARAM_FOG_BEGIN,
		ENV_FX_PARAM_FOG_BEGIN_COLOR,
		ENV_FX_PARAM_FOG_END_COLOR,
		ENV_FX_PARAM_FOG_ATTENUATION,
		ENV_FX_PARAM_FOG_BG,
		ENV_FX_PARAM_BCS_BRIGHTNESS,
		ENV_FX_PARAM_BCS_CONTRAST,
		ENV_FX_PARAM_BCS_SATURATION,
		ENV_FX_PARAM_MAX
	};

	virtual void environment_fx_set_param(RID p_env,EnvironmentFxParam p_effect,const Variant& p_param)=0;
	virtual Variant environment_fx_get_param(RID p_env,EnvironmentFxParam p_effect) const=0;


	/* SCENARIO API */




	virtual RID scenario_create()=0;	

	enum ScenarioDebugMode {
		SCENARIO_DEBUG_DISABLED,
		SCENARIO_DEBUG_WIREFRAME,
		SCENARIO_DEBUG_OVERDRAW,
		SCENARIO_DEBUG_SHADELESS,

	};


	virtual void scenario_set_debug(RID p_scenario,ScenarioDebugMode p_debug_mode)=0;
	virtual void scenario_set_environment(RID p_scenario, RID p_environment)=0;
	virtual RID scenario_get_environment(RID p_scenario, RID p_environment) const=0;
	virtual void scenario_set_fallback_environment(RID p_scenario, RID p_environment)=0;


	/* INSTANCING API */
	
	enum InstanceType {
	
		INSTANCE_NONE,
		INSTANCE_MESH,
		INSTANCE_MULTIMESH,
		INSTANCE_IMMEDIATE,
		INSTANCE_PARTICLES,
		INSTANCE_LIGHT,
		INSTANCE_ROOM,
		INSTANCE_PORTAL,
		INSTANCE_BAKED_LIGHT,
		INSTANCE_BAKED_LIGHT_SAMPLER,

		INSTANCE_GEOMETRY_MASK=(1<<INSTANCE_MESH)|(1<<INSTANCE_MULTIMESH)|(1<<INSTANCE_IMMEDIATE)|(1<<INSTANCE_PARTICLES)
	};
	


	virtual RID instance_create2(RID p_base, RID p_scenario);

//	virtual RID instance_create(RID p_base,RID p_scenario)=0; // from can be mesh, light,  area and portal so far.
	virtual RID instance_create()=0; // from can be mesh, light, poly, area and portal so far.

	virtual void instance_set_base(RID p_instance, RID p_base)=0; // from can be mesh, light, poly, area and portal so far.
	virtual RID instance_get_base(RID p_instance) const=0;

	virtual void instance_set_scenario(RID p_instance, RID p_scenario)=0; // from can be mesh, light, poly, area and portal so far.
	virtual RID instance_get_scenario(RID p_instance) const=0;

	virtual void instance_set_layer_mask(RID p_instance, uint32_t p_mask)=0;
	virtual uint32_t instance_get_layer_mask(RID p_instance) const=0;

	virtual AABB instance_get_base_aabb(RID p_instance) const=0;

	virtual void instance_set_transform(RID p_instance, const Transform& p_transform)=0;
	virtual Transform instance_get_transform(RID p_instance) const=0;
	

	virtual void instance_attach_object_instance_ID(RID p_instance,uint32_t p_ID)=0;
	virtual uint32_t instance_get_object_instance_ID(RID p_instance) const=0;

	virtual void instance_set_morph_target_weight(RID p_instance,int p_shape, float p_weight)=0;
	virtual float instance_get_morph_target_weight(RID p_instance,int p_shape) const=0;
	
	virtual void instance_attach_skeleton(RID p_instance,RID p_skeleton)=0;
	virtual RID instance_get_skeleton(RID p_instance) const=0;
	
	virtual void instance_set_exterior( RID p_instance, bool p_enabled )=0;
	virtual bool instance_is_exterior( RID p_instance) const=0;

	virtual void instance_set_room( RID p_instance, RID p_room )=0;
	virtual RID instance_get_room( RID p_instance ) const =0;

	virtual void instance_set_extra_visibility_margin( RID p_instance, real_t p_margin )=0;
	virtual real_t instance_get_extra_visibility_margin( RID p_instance ) const =0;

	// don't use these in a game!
	virtual Vector<RID> instances_cull_aabb(const AABB& p_aabb, RID p_scenario=RID()) const=0;
	virtual Vector<RID> instances_cull_ray(const Vector3& p_from, const Vector3& p_to, RID p_scenario=RID()) const=0;
	virtual Vector<RID> instances_cull_convex(const Vector<Plane>& p_convex, RID p_scenario=RID()) const=0;

	enum InstanceFlags {
		INSTANCE_FLAG_VISIBLE,
		INSTANCE_FLAG_BILLBOARD,
		INSTANCE_FLAG_BILLBOARD_FIX_Y,
		INSTANCE_FLAG_CAST_SHADOW,
		INSTANCE_FLAG_RECEIVE_SHADOWS,
		INSTANCE_FLAG_DEPH_SCALE,
		INSTANCE_FLAG_VISIBLE_IN_ALL_ROOMS,
		INSTANCE_FLAG_USE_BAKED_LIGHT,
		INSTANCE_FLAG_MAX
	};

	virtual void instance_geometry_set_flag(RID p_instance,InstanceFlags p_flags,bool p_enabled)=0;
	virtual bool instance_geometry_get_flag(RID p_instance,InstanceFlags p_flags) const=0;

	virtual void instance_geometry_set_material_override(RID p_instance, RID p_material)=0;
	virtual RID instance_geometry_get_material_override(RID p_instance) const=0;

	virtual void instance_geometry_set_draw_range(RID p_instance,float p_min,float p_max)=0;
	virtual float instance_geometry_get_draw_range_max(RID p_instance) const=0;
	virtual float instance_geometry_get_draw_range_min(RID p_instance) const=0;

	virtual void instance_geometry_set_baked_light(RID p_instance,RID p_baked_light)=0;
	virtual RID instance_geometry_get_baked_light(RID p_instance) const=0;

	virtual void instance_geometry_set_baked_light_sampler(RID p_instance,RID p_baked_light_sampler)=0;
	virtual RID instance_geometry_get_baked_light_sampler(RID p_instance) const=0;

	virtual void instance_geometry_set_baked_light_texture_index(RID p_instance,int p_tex_id)=0;
	virtual int instance_geometry_get_baked_light_texture_index(RID p_instance) const=0;


	virtual void instance_light_set_enabled(RID p_instance,bool p_enabled)=0;
	virtual bool instance_light_is_enabled(RID p_instance) const=0;

	/* CANVAS (2D) */

	virtual RID canvas_create()=0;
	virtual void canvas_set_item_mirroring(RID p_canvas,RID p_item,const Point2& p_mirroring)=0;
	virtual Point2 canvas_get_item_mirroring(RID p_canvas,RID p_item) const=0;


	virtual RID canvas_item_create()=0;
	virtual void canvas_item_set_parent(RID p_item,RID p_parent)=0;
	virtual RID canvas_item_get_parent(RID p_canvas_item) const=0;

	virtual void canvas_item_set_visible(RID p_item,bool p_visible)=0;
	virtual bool canvas_item_is_visible(RID p_item) const=0;

	virtual void canvas_item_set_blend_mode(RID p_canvas_item,MaterialBlendMode p_blend)=0;

	virtual void canvas_item_attach_viewport(RID p_item, RID p_viewport)=0;

	//virtual void canvas_item_set_rect(RID p_item, const Rect2& p_rect)=0;

	virtual void canvas_item_set_transform(RID p_item, const Matrix32& p_transform)=0;
	virtual void canvas_item_set_clip(RID p_item, bool p_clip)=0;
	virtual void canvas_item_set_custom_rect(RID p_item, bool p_custom_rect,const Rect2& p_rect=Rect2())=0;
	virtual void canvas_item_set_opacity(RID p_item, float p_opacity)=0;
	virtual float canvas_item_get_opacity(RID p_item, float p_opacity) const=0;

	virtual void canvas_item_set_self_opacity(RID p_item, float p_self_opacity)=0;
	virtual float canvas_item_get_self_opacity(RID p_item, float p_self_opacity) const=0;

	virtual void canvas_item_set_on_top(RID p_item, bool p_on_top)=0;
	virtual bool canvas_item_is_on_top(RID p_item) const=0;

	virtual void canvas_item_add_line(RID p_item, const Point2& p_from, const Point2& p_to,const Color& p_color,float p_width=1.0)=0;
	virtual void canvas_item_add_rect(RID p_item, const Rect2& p_rect, const Color& p_color)=0;
	virtual void canvas_item_add_circle(RID p_item, const Point2& p_pos, float p_radius,const Color& p_color)=0;
	virtual void canvas_item_add_texture_rect(RID p_item, const Rect2& p_rect, RID p_texture,bool p_tile=false,const Color& p_modulate=Color(1,1,1))=0;
	virtual void canvas_item_add_texture_rect_region(RID p_item, const Rect2& p_rect, RID p_texture,const Rect2& p_src_rect,const Color& p_modulate=Color(1,1,1))=0;
	virtual void canvas_item_add_style_box(RID p_item, const Rect2& p_rect, RID p_texture,const Vector2& p_topleft, const Vector2& p_bottomright, bool p_draw_center=true,const Color& p_modulate=Color(1,1,1))=0;
	virtual void canvas_item_add_primitive(RID p_item, const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs, RID p_texture,float p_width=1.0)=0;
	virtual void canvas_item_add_polygon(RID p_item, const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs=Vector<Point2>(), RID p_texture=RID())=0;
	virtual void canvas_item_add_triangle_array(RID p_item, const Vector<int>& p_indices, const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs=Vector<Point2>(), RID p_texture=RID(), int p_count=-1)=0;
	virtual void canvas_item_add_triangle_array_ptr(RID p_item, int p_count, const int* p_indices, const Point2* p_points, const Color* p_colors,const Point2* p_uvs=NULL, RID p_texture=RID())=0;
	virtual void canvas_item_add_set_transform(RID p_item,const Matrix32& p_transform)=0;
	virtual void canvas_item_add_set_blend_mode(RID p_item, MaterialBlendMode p_blend)=0;
	virtual void canvas_item_add_clip_ignore(RID p_item, bool p_ignore)=0;
	virtual void canvas_item_set_sort_children_by_y(RID p_item, bool p_enable)=0;
	virtual void canvas_item_set_z(RID p_item, int p_z)=0;
	virtual void canvas_item_set_z_as_relative_to_parent(RID p_item, bool p_enable)=0;

	virtual void canvas_item_clear(RID p_item)=0;
	virtual void canvas_item_raise(RID p_item)=0;

	virtual void canvas_item_set_shader(RID p_item, RID p_shader)=0;
	virtual RID canvas_item_get_shader(RID p_item) const=0;

	virtual void canvas_item_set_use_parent_shader(RID p_item, bool p_enable)=0;

	virtual void canvas_item_set_shader_param(RID p_canvas_item, const StringName& p_param, const Variant& p_value)=0;
	virtual Variant canvas_item_get_shader_param(RID p_canvas_item, const StringName& p_param) const=0;

	virtual RID canvas_light_create()=0;
	virtual void canvas_light_attach_to_canvas(RID p_light,RID p_canvas)=0;
	virtual void canvas_light_set_enabled(RID p_light, bool p_enabled)=0;
	virtual void canvas_light_set_transform(RID p_light, const Matrix32& p_transform)=0;
	virtual void canvas_light_set_texture(RID p_light, RID p_texture)=0;
	virtual void canvas_light_set_texture_offset(RID p_light, const Vector2& p_offset)=0;
	virtual void canvas_light_set_color(RID p_light, const Color& p_color)=0;
	virtual void canvas_light_set_height(RID p_light, float p_height)=0;
	virtual void canvas_light_set_z_range(RID p_light, int p_min_z,int p_max_z)=0;
	virtual void canvas_light_set_item_mask(RID p_light, int p_mask)=0;

	enum CanvasLightBlendMode {
		CANVAS_LIGHT_BLEND_ADD,
		CANVAS_LIGHT_BLEND_SUB,
		CANVAS_LIGHT_BLEND_MULTIPLY,
		CANVAS_LIGHT_BLEND_DODGE,
		CANVAS_LIGHT_BLEND_BURN,
		CANVAS_LIGHT_BLEND_LIGHTEN,
		CANVAS_LIGHT_BLEND_DARKEN,
		CANVAS_LIGHT_BLEND_OVERLAY,
		CANVAS_LIGHT_BLEND_SCREEN,
	};
	virtual void canvas_light_set_blend_mode(RID p_light, CanvasLightBlendMode p_blend_mode)=0;
	virtual void canvas_light_set_shadow_enabled(RID p_light, bool p_enabled)=0;
	virtual void canvas_light_set_shadow_buffer_size(RID p_light, int p_size)=0;
	virtual void canvas_light_set_shadow_filter(RID p_light, int p_size)=0;


	virtual RID canvas_light_occluder_create()=0;
	virtual void canvas_light_occluder_attach_to_canvas(RID p_occluder,RID p_canvas)=0;
	virtual void canvas_light_occluder_set_enabled(RID p_occluder,bool p_enabled)=0;
	virtual void canvas_light_occluder_set_shape(RID p_occluder,const DVector<Vector2>& p_shape)=0;

	/* CURSOR */
	virtual void cursor_set_rotation(float p_rotation, int p_cursor = 0)=0; // radians
	virtual void cursor_set_texture(RID p_texture, const Point2 &p_center_offset = Point2(0, 0), int p_cursor=0)=0;
	virtual void cursor_set_visible(bool p_visible, int p_cursor = 0)=0;
	virtual void cursor_set_pos(const Point2& p_pos, int p_cursor = 0)=0;

	/* BLACK BARS */


	virtual void black_bars_set_margins(int p_left, int p_top, int p_right, int p_bottom)=0;
	virtual void black_bars_set_images(RID p_left, RID p_top, RID p_right, RID p_bottom)=0;


	/* FREE */

	virtual void free( RID p_rid )=0; ///< free RIDs associated with the visual server

	/* CUSTOM SHADING */

	virtual void custom_shade_model_set_shader(int p_model, RID p_shader)=0;
	virtual RID custom_shade_model_get_shader(int p_model) const=0;
	virtual void custom_shade_model_set_name(int p_model, const String& p_name)=0;
	virtual String custom_shade_model_get_name(int p_model) const=0;
	virtual void custom_shade_model_set_param_info(int p_model, const List<PropertyInfo>& p_info)=0;
	virtual void custom_shade_model_get_param_info(int p_model, List<PropertyInfo>* p_info) const=0;

	/* EVENT QUEUING */

	virtual void draw()=0;
	virtual void flush()=0;
	virtual bool has_changed() const=0;
	virtual void init()=0;
	virtual void finish()=0;

	/* STATUS INFORMATION */

	enum RenderInfo {

		INFO_OBJECTS_IN_FRAME,
		INFO_VERTICES_IN_FRAME,
		INFO_MATERIAL_CHANGES_IN_FRAME,
		INFO_SHADER_CHANGES_IN_FRAME,
		INFO_SURFACE_CHANGES_IN_FRAME,
		INFO_DRAW_CALLS_IN_FRAME,
		INFO_USAGE_VIDEO_MEM_TOTAL,
		INFO_VIDEO_MEM_USED,
		INFO_TEXTURE_MEM_USED,
		INFO_VERTEX_MEM_USED,
	};

	virtual int get_render_info(RenderInfo p_info)=0;


	/* Materials for 2D on 3D */


	RID material_2d_get(bool p_shaded, bool p_transparent, bool p_cut_alpha,bool p_opaque_prepass);
	

	/* TESTING */
	
	virtual RID get_test_cube()=0;

	virtual RID get_test_texture();
	virtual RID get_white_texture();

	virtual RID make_sphere_mesh(int p_lats,int p_lons,float p_radius);

	virtual void mesh_add_surface_from_mesh_data( RID p_mesh, const Geometry::MeshData& p_mesh_data);
	virtual void mesh_add_surface_from_planes( RID p_mesh, const DVector<Plane>& p_planes);

	virtual void set_boot_image(const Image& p_image, const Color& p_color)=0;
	virtual void set_default_clear_color(const Color& p_color)=0;

	enum Features {
		FEATURE_SHADERS,
		FEATURE_MULTITHREADED,
		FEATURE_NEEDS_RELOAD_HOOK,
	};

	virtual bool has_feature(Features p_feature) const=0;

	VisualServer();	
	virtual ~VisualServer();

};

// make variant understand the enums

VARIANT_ENUM_CAST( VisualServer::CubeMapSide );
VARIANT_ENUM_CAST( VisualServer::TextureFlags );
VARIANT_ENUM_CAST( VisualServer::ShaderMode );
VARIANT_ENUM_CAST( VisualServer::MaterialFlag );
VARIANT_ENUM_CAST( VisualServer::MaterialBlendMode );
VARIANT_ENUM_CAST( VisualServer::ParticleVariable );
VARIANT_ENUM_CAST( VisualServer::ArrayType );
VARIANT_ENUM_CAST( VisualServer::ArrayFormat );
VARIANT_ENUM_CAST( VisualServer::PrimitiveType );
VARIANT_ENUM_CAST( VisualServer::LightType );
VARIANT_ENUM_CAST( VisualServer::LightColor );
VARIANT_ENUM_CAST( VisualServer::LightParam );
VARIANT_ENUM_CAST( VisualServer::ScenarioDebugMode );
VARIANT_ENUM_CAST( VisualServer::InstanceType );
VARIANT_ENUM_CAST( VisualServer::RenderInfo );
VARIANT_ENUM_CAST( VisualServer::MipMapPolicy );
//typedef VisualServer VS; // makes it easier to use
#define VS VisualServer

#endif
