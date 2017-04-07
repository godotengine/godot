/*************************************************************************/
/*  visual_server_raster.cpp                                             */
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
#include "visual_server_raster.h"
#include "default_mouse_cursor.xpm"
#include "global_config.h"
#include "io/marshalls.h"
#include "os/os.h"
#include "sort.h"
#include "visual_server_canvas.h"
#include "visual_server_global.h"
#include "visual_server_scene.h"

// careful, these may run in different threads than the visual server

int VisualServerRaster::changes = 0;

/* CURSOR */
void VisualServerRaster::cursor_set_rotation(float p_rotation, int p_cursor) {
}
void VisualServerRaster::cursor_set_texture(RID p_texture, const Point2 &p_center_offset, int p_cursor, const Rect2 &p_region) {
}
void VisualServerRaster::cursor_set_visible(bool p_visible, int p_cursor) {
}
void VisualServerRaster::cursor_set_pos(const Point2 &p_pos, int p_cursor) {
}

/* BLACK BARS */

void VisualServerRaster::black_bars_set_margins(int p_left, int p_top, int p_right, int p_bottom) {
}
void VisualServerRaster::black_bars_set_images(RID p_left, RID p_top, RID p_right, RID p_bottom) {
}

/* FREE */

void VisualServerRaster::free(RID p_rid) {

	if (VSG::storage->free(p_rid))
		return;
	if (VSG::canvas->free(p_rid))
		return;
	if (VSG::viewport->free(p_rid))
		return;
	if (VSG::scene->free(p_rid))
		return;
}

/* EVENT QUEUING */

void VisualServerRaster::draw() {

	/*
	if (changes)
		print_line("changes: "+itos(changes));
	*/

	changes = 0;

	VSG::rasterizer->begin_frame();

	VSG::scene->update_dirty_instances(); //update scene stuff

	VSG::viewport->draw_viewports();
	VSG::scene->render_probes();
	//_draw_cursors_and_margins();
	VSG::rasterizer->end_frame();
	//draw_extra_frame=VS:rasterizer->needs_to_draw_next_frame();
}
void VisualServerRaster::sync() {
}
bool VisualServerRaster::has_changed() const {

	return changes > 0;
}
void VisualServerRaster::init() {

	VSG::rasterizer->initialize();
}
void VisualServerRaster::finish() {

	if (test_cube.is_valid()) {
		free(test_cube);
	}

	VSG::rasterizer->finalize();
}

/* STATUS INFORMATION */

int VisualServerRaster::get_render_info(RenderInfo p_info) {

	return 0;
}

/* TESTING */

void VisualServerRaster::set_boot_image(const Image &p_image, const Color &p_color, bool p_scale) {
}
void VisualServerRaster::set_default_clear_color(const Color &p_color) {
}

bool VisualServerRaster::has_feature(Features p_feature) const {

	return false;
}

RID VisualServerRaster::get_test_cube() {
	if (!test_cube.is_valid()) {
		test_cube = _make_test_cube();
	}
	return test_cube;
}

bool VisualServerRaster::has_os_feature(const String &p_feature) const {

	return VSG::storage->has_os_feature(p_feature);
}

VisualServerRaster::VisualServerRaster() {

	VSG::canvas = memnew(VisualServerCanvas);
	VSG::viewport = memnew(VisualServerViewport);
	VSG::scene = memnew(VisualServerScene);
	VSG::rasterizer = Rasterizer::create();
	VSG::storage = VSG::rasterizer->get_storage();
	VSG::canvas_render = VSG::rasterizer->get_canvas();
	VSG::scene_render = VSG::rasterizer->get_scene();
}

VisualServerRaster::~VisualServerRaster() {

	memdelete(VSG::canvas);
	memdelete(VSG::viewport);
	memdelete(VSG::rasterizer);
}

#if 0

BalloonAllocator<> *VisualServerRaster::OctreeAllocator::allocator=NULL;

#define VS_CHANGED \
	changes++;\

	//print_line(__FUNCTION__);




RID VisualServerRaster::texture_create() {

	return rasterizer->texture_create();
}


void VisualServerRaster::texture_allocate(RID p_texture, int p_width, int p_height,Image::Format p_format,uint32_t p_flags) {

	rasterizer->texture_allocate(p_texture,p_width,p_height,p_format,p_flags);
}

void VisualServerRaster::texture_set_flags(RID p_texture,uint32_t p_flags)  {

	VS_CHANGED;
	rasterizer->texture_set_flags(p_texture,p_flags);
}


void VisualServerRaster::texture_set_data(RID p_texture,const Image& p_image,CubeMapSide p_cube_side) {

	VS_CHANGED;
	rasterizer->texture_set_data(p_texture,p_image,p_cube_side);


}

Image VisualServerRaster::texture_get_data(RID p_texture,CubeMapSide p_cube_side) const {


	return rasterizer->texture_get_data(p_texture,p_cube_side);
}


uint32_t VisualServerRaster::texture_get_flags(RID p_texture) const {

	return rasterizer->texture_get_flags(p_texture);

}
Image::Format VisualServerRaster::texture_get_format(RID p_texture) const {

	return rasterizer->texture_get_format(p_texture);

}
uint32_t VisualServerRaster::texture_get_width(RID p_texture) const {

	return rasterizer->texture_get_width(p_texture);
}

uint32_t VisualServerRaster::texture_get_height(RID p_texture) const {

	return rasterizer->texture_get_height(p_texture);
}

void VisualServerRaster::texture_set_size_override(RID p_texture,int p_width, int p_height) {

	rasterizer->texture_set_size_override(p_texture,p_width,p_height);
}

bool VisualServerRaster::texture_can_stream(RID p_texture) const {

	return false;
}

void VisualServerRaster::texture_set_reload_hook(RID p_texture,ObjectID p_owner,const StringName& p_function) const {

	rasterizer->texture_set_reload_hook(p_texture,p_owner,p_function);
}

void VisualServerRaster::texture_set_path(RID p_texture,const String& p_path) {

	rasterizer->texture_set_path(p_texture,p_path);
}

String VisualServerRaster::texture_get_path(RID p_texture) const{

	return rasterizer->texture_get_path(p_texture);
}

void VisualServerRaster::texture_debug_usage(List<TextureInfo> *r_info){

	rasterizer->texture_debug_usage(r_info);
}

void VisualServerRaster::texture_set_shrink_all_x2_on_set_data(bool p_enable) {

	rasterizer->texture_set_shrink_all_x2_on_set_data(p_enable);
}

/* SHADER API */

RID VisualServerRaster::shader_create(ShaderMode p_mode) {

	return rasterizer->shader_create(p_mode);
}

void VisualServerRaster::shader_set_mode(RID p_shader,ShaderMode p_mode){

	VS_CHANGED;
	rasterizer->shader_set_mode(p_shader,p_mode);
}
VisualServer::ShaderMode VisualServerRaster::shader_get_mode(RID p_shader) const{

	return rasterizer->shader_get_mode(p_shader);
}


void VisualServerRaster::shader_set_code(RID p_shader, const String& p_vertex, const String& p_fragment,const String& p_light,int p_vertex_ofs,int p_fragment_ofs,int p_light_ofs) {

	VS_CHANGED;
	rasterizer->shader_set_code(p_shader,p_vertex,p_fragment,p_light,p_vertex_ofs,p_fragment_ofs,p_light_ofs);
}

String VisualServerRaster::shader_get_vertex_code(RID p_shader) const{

	return rasterizer->shader_get_vertex_code(p_shader);
}

String VisualServerRaster::shader_get_fragment_code(RID p_shader) const{

	return rasterizer->shader_get_fragment_code(p_shader);
}

String VisualServerRaster::shader_get_light_code(RID p_shader) const{

	return rasterizer->shader_get_light_code(p_shader);
}

void VisualServerRaster::shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const {

	return rasterizer->shader_get_param_list(p_shader,p_param_list);
}


void VisualServerRaster::shader_set_default_texture_param(RID p_shader, const StringName& p_name, RID p_texture) {

	rasterizer->shader_set_default_texture_param(p_shader,p_name,p_texture);
}

RID VisualServerRaster::shader_get_default_texture_param(RID p_shader, const StringName& p_name) const{

	return rasterizer->shader_get_default_texture_param(p_shader,p_name);
}


/* Material */

RID VisualServerRaster::material_create() {

	return rasterizer->material_create();
}

void VisualServerRaster::material_set_shader(RID p_material, RID p_shader) {

	VS_CHANGED;
	rasterizer->material_set_shader(p_material, p_shader );
}

RID VisualServerRaster::material_get_shader(RID p_material) const {

	return rasterizer->material_get_shader(p_material);
}

void VisualServerRaster::material_set_param(RID p_material, const StringName& p_param, const Variant& p_value) {

	VS_CHANGED;
	rasterizer->material_set_param(p_material, p_param,p_value );
}

Variant VisualServerRaster::material_get_param(RID p_material, const StringName& p_param) const {

	return rasterizer->material_get_param(p_material,p_param);
}

void VisualServerRaster::material_set_flag(RID p_material, MaterialFlag p_flag,bool p_enabled) {
	VS_CHANGED;
	rasterizer->material_set_flag(p_material,p_flag,p_enabled);
}

void VisualServerRaster::material_set_depth_draw_mode(RID p_material, MaterialDepthDrawMode p_mode) {

	VS_CHANGED;
	rasterizer->material_set_depth_draw_mode(p_material,p_mode);

}

VS::MaterialDepthDrawMode VisualServerRaster::material_get_depth_draw_mode(RID p_material) const {

	return rasterizer->material_get_depth_draw_mode(p_material);
}


bool VisualServerRaster::material_get_flag(RID p_material,MaterialFlag p_flag) const {

	return rasterizer->material_get_flag(p_material,p_flag);
}

void VisualServerRaster::material_set_blend_mode(RID p_material,MaterialBlendMode p_mode) {
	VS_CHANGED;
	rasterizer->material_set_blend_mode(p_material,p_mode);
}
VS::MaterialBlendMode VisualServerRaster::material_get_blend_mode(RID p_material) const {

	return rasterizer->material_get_blend_mode(p_material);
}

void VisualServerRaster::material_set_line_width(RID p_material,float p_line_width) {
	VS_CHANGED;
	rasterizer->material_set_line_width(p_material,p_line_width);
}
float VisualServerRaster::material_get_line_width(RID p_material) const {

	return rasterizer->material_get_line_width(p_material);
}

/* FIXED MATERIAL */

RID VisualServerRaster::fixed_material_create() {

	return rasterizer->fixed_material_create();
}

void VisualServerRaster::fixed_material_set_flag(RID p_material, SpatialMaterialFlags p_flag, bool p_enabled) {

	rasterizer->fixed_material_set_flag(p_material,p_flag,p_enabled);
}

bool VisualServerRaster::fixed_material_get_flag(RID p_material, SpatialMaterialFlags p_flag) const {

	return rasterizer->fixed_material_get_flag(p_material,p_flag);
}

void VisualServerRaster::fixed_material_set_param(RID p_material, SpatialMaterialParam p_parameter, const Variant& p_value) {
	VS_CHANGED;
	rasterizer->fixed_material_set_parameter(p_material,p_parameter,p_value);
}

Variant VisualServerRaster::fixed_material_get_param(RID p_material,SpatialMaterialParam p_parameter) const {

	return rasterizer->fixed_material_get_parameter(p_material,p_parameter);
}


void VisualServerRaster::fixed_material_set_texture(RID p_material,SpatialMaterialParam p_parameter, RID p_texture) {
	VS_CHANGED;
	rasterizer->fixed_material_set_texture(p_material,p_parameter,p_texture);
}

RID VisualServerRaster::fixed_material_get_texture(RID p_material,SpatialMaterialParam p_parameter) const {

	return rasterizer->fixed_material_get_texture(p_material,p_parameter);
}




void VisualServerRaster::fixed_material_set_texcoord_mode(RID p_material,SpatialMaterialParam p_parameter, SpatialMaterialTexCoordMode p_mode) {
	VS_CHANGED;
	rasterizer->fixed_material_set_texcoord_mode(p_material,p_parameter,p_mode);
}

VS::SpatialMaterialTexCoordMode VisualServerRaster::fixed_material_get_texcoord_mode(RID p_material,SpatialMaterialParam p_parameter) const {

	return rasterizer->fixed_material_get_texcoord_mode(p_material,p_parameter);
}

void VisualServerRaster::fixed_material_set_point_size(RID p_material,float p_size) {
	VS_CHANGED

	rasterizer->fixed_material_set_point_size(p_material,p_size);
}

float VisualServerRaster::fixed_material_get_point_size(RID p_material) const{

	return rasterizer->fixed_material_get_point_size(p_material);
}


void VisualServerRaster::fixed_material_set_uv_transform(RID p_material,const Transform& p_transform) {
	VS_CHANGED;
	rasterizer->fixed_material_set_uv_transform(p_material,p_transform);
}

Transform VisualServerRaster::fixed_material_get_uv_transform(RID p_material) const {

	return rasterizer->fixed_material_get_uv_transform(p_material);
}

void VisualServerRaster::fixed_material_set_light_shader(RID p_material,SpatialMaterialLightShader p_shader) {

	VS_CHANGED;
	rasterizer->fixed_material_set_light_shader(p_material,p_shader);

}

VisualServerRaster::SpatialMaterialLightShader VisualServerRaster::fixed_material_get_light_shader(RID p_material) const{

	return rasterizer->fixed_material_get_light_shader(p_material);
}



/* MESH API */

RID VisualServerRaster::mesh_create() {

	return rasterizer->mesh_create();
}

void VisualServerRaster::mesh_set_morph_target_count(RID p_mesh,int p_amount) {

	rasterizer->mesh_set_morph_target_count(p_mesh,p_amount);
	int amount = rasterizer->mesh_get_morph_target_count(p_mesh);


	Map< RID, Set<RID> >::Element * E = instance_dependency_map.find( p_mesh );

	if (!E)
		return;


	Set<RID>::Element *I = E->get().front();

	while(I) {

		Instance *ins = instance_owner.get( I->get() );
		ins->data.morph_values.resize(amount);
		I = I->next();
	}
}

int VisualServerRaster::mesh_get_morph_target_count(RID p_mesh) const {

	return rasterizer->mesh_get_morph_target_count(p_mesh);
}

void VisualServerRaster::mesh_set_morph_target_mode(RID p_mesh,MorphTargetMode p_mode) {

	rasterizer->mesh_set_morph_target_mode(p_mesh,p_mode);
}

VisualServer::MorphTargetMode VisualServerRaster::mesh_get_morph_target_mode(RID p_mesh) const{

	return rasterizer->mesh_get_morph_target_mode(p_mesh);
}

void VisualServerRaster::mesh_add_custom_surface(RID p_mesh,const Variant& p_dat) {


}


void VisualServerRaster::mesh_add_surface(RID p_mesh,PrimitiveType p_primitive,const Array& p_arrays,const Array& p_blend_shapes,bool p_alpha_sort) {

	VS_CHANGED;
	_dependency_queue_update(p_mesh,true,true);
	rasterizer->mesh_add_surface(p_mesh,p_primitive,p_arrays,p_blend_shapes,p_alpha_sort);

}

Array VisualServerRaster::mesh_get_surface_arrays(RID p_mesh,int p_surface) const {

	return rasterizer->mesh_get_surface_arrays(p_mesh,p_surface);
}
Array VisualServerRaster::mesh_get_surface_morph_arrays(RID p_mesh,int p_surface) const {

	return rasterizer->mesh_get_surface_morph_arrays(p_mesh,p_surface);
}




void VisualServerRaster::mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material,bool p_owned){
	VS_CHANGED;
	rasterizer->mesh_surface_set_material(p_mesh,p_surface,p_material,p_owned);
}

RID VisualServerRaster::mesh_surface_get_material(RID p_mesh,int p_surface) const {

	return rasterizer->mesh_surface_get_material(p_mesh,p_surface);

}


int VisualServerRaster::mesh_surface_get_array_len(RID p_mesh, int p_surface) const{

	return rasterizer->mesh_surface_get_array_len(p_mesh,p_surface);
}

int VisualServerRaster::mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const{


	return rasterizer->mesh_surface_get_array_index_len(p_mesh,p_surface);
}

uint32_t VisualServerRaster::mesh_surface_get_format(RID p_mesh, int p_surface) const{

	return rasterizer->mesh_surface_get_format(p_mesh,p_surface);
}

VisualServer::PrimitiveType VisualServerRaster::mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const{

	return rasterizer->mesh_surface_get_primitive_type(p_mesh,p_surface);
}


void VisualServerRaster::mesh_remove_surface(RID p_mesh,int p_surface){

	rasterizer->mesh_remove_surface(p_mesh,p_surface);
	_dependency_queue_update(p_mesh,true,true);
}

int VisualServerRaster::mesh_get_surface_count(RID p_mesh) const{

	return rasterizer->mesh_get_surface_count(p_mesh);

}


void VisualServerRaster::mesh_set_custom_aabb(RID p_mesh,const AABB& p_aabb) {

	VS_CHANGED;
	_dependency_queue_update(p_mesh,true);
	rasterizer->mesh_set_custom_aabb(p_mesh,p_aabb);

}

AABB VisualServerRaster::mesh_get_custom_aabb(RID p_mesh) const {

	return rasterizer->mesh_get_custom_aabb(p_mesh);
}

void VisualServerRaster::mesh_clear(RID p_mesh) {

	ERR_FAIL_COND(!rasterizer->is_mesh(p_mesh));
	while(rasterizer->mesh_get_surface_count(p_mesh)) {
		rasterizer->mesh_remove_surface(p_mesh,0);
	}

	_dependency_queue_update(p_mesh,true,true);
}


/* MULTIMESH */

RID VisualServerRaster::multimesh_create() {

	return rasterizer->multimesh_create();
}

void VisualServerRaster::multimesh_set_instance_count(RID p_multimesh,int p_count) {
	VS_CHANGED;
	rasterizer->multimesh_set_instance_count(p_multimesh,p_count);
}

int VisualServerRaster::multimesh_get_instance_count(RID p_multimesh) const {

	return rasterizer->multimesh_get_instance_count(p_multimesh);
}

void VisualServerRaster::multimesh_set_mesh(RID p_multimesh,RID p_mesh) {
	VS_CHANGED;
	rasterizer->multimesh_set_mesh(p_multimesh,p_mesh);

}
void VisualServerRaster::multimesh_set_aabb(RID p_multimesh,const AABB& p_aabb) {
	VS_CHANGED;
	rasterizer->multimesh_set_aabb(p_multimesh,p_aabb);
	_dependency_queue_update(p_multimesh,true);

}

void VisualServerRaster::multimesh_instance_set_transform(RID p_multimesh,int p_index,const Transform& p_transform) {
	VS_CHANGED;
	rasterizer->multimesh_instance_set_transform(p_multimesh,p_index,p_transform);

}
void VisualServerRaster::multimesh_instance_set_color(RID p_multimesh,int p_index,const Color& p_color) {
	VS_CHANGED;
	rasterizer->multimesh_instance_set_color(p_multimesh,p_index,p_color);

}
RID VisualServerRaster::multimesh_get_mesh(RID p_multimesh) const {

	return rasterizer->multimesh_get_mesh(p_multimesh);
}
AABB VisualServerRaster::multimesh_get_aabb(RID p_multimesh,const AABB& p_aabb) const {


	return rasterizer->multimesh_get_aabb(p_multimesh);
}

Transform VisualServerRaster::multimesh_instance_get_transform(RID p_multimesh,int p_index) const {

	return rasterizer->multimesh_instance_get_transform(p_multimesh,p_index);
}
Color VisualServerRaster::multimesh_instance_get_color(RID p_multimesh,int p_index) const {

	return rasterizer->multimesh_instance_get_color(p_multimesh,p_index);
}

void VisualServerRaster::multimesh_set_visible_instances(RID p_multimesh,int p_visible) {

	rasterizer->multimesh_set_visible_instances(p_multimesh,p_visible);

}

int VisualServerRaster::multimesh_get_visible_instances(RID p_multimesh) const {

	return rasterizer->multimesh_get_visible_instances(p_multimesh);
}


/* IMMEDIATE API */


RID VisualServerRaster::immediate_create() {

	return rasterizer->immediate_create();
}

void VisualServerRaster::immediate_begin(RID p_immediate,PrimitiveType p_primitive,RID p_texture){

	rasterizer->immediate_begin(p_immediate,p_primitive,p_texture);
}
void VisualServerRaster::immediate_vertex(RID p_immediate,const Vector3& p_vertex){

	rasterizer->immediate_vertex(p_immediate,p_vertex);

}
void VisualServerRaster::immediate_normal(RID p_immediate,const Vector3& p_normal){

	rasterizer->immediate_normal(p_immediate,p_normal);

}
void VisualServerRaster::immediate_tangent(RID p_immediate,const Plane& p_tangent){

	rasterizer->immediate_tangent(p_immediate,p_tangent);

}
void VisualServerRaster::immediate_color(RID p_immediate,const Color& p_color){

	rasterizer->immediate_color(p_immediate,p_color);

}
void VisualServerRaster::immediate_uv(RID p_immediate,const Vector2& p_uv){

	rasterizer->immediate_uv(p_immediate,p_uv);

}
void VisualServerRaster::immediate_uv2(RID p_immediate,const Vector2& p_uv2){

	rasterizer->immediate_uv2(p_immediate,p_uv2);

}
void VisualServerRaster::immediate_end(RID p_immediate){

	VS_CHANGED;
	_dependency_queue_update(p_immediate,true);
	rasterizer->immediate_end(p_immediate);

}
void VisualServerRaster::immediate_clear(RID p_immediate){

	VS_CHANGED;
	_dependency_queue_update(p_immediate,true);
	rasterizer->immediate_clear(p_immediate);

}

void VisualServerRaster::immediate_set_material(RID p_immediate,RID p_material) {

	rasterizer->immediate_set_material(p_immediate,p_material);
}

RID VisualServerRaster::immediate_get_material(RID p_immediate) const {

	return rasterizer->immediate_get_material(p_immediate);
}


/* PARTICLES API */

RID VisualServerRaster::particles_create() {

	return rasterizer->particles_create();
}

void VisualServerRaster::particles_set_amount(RID p_particles, int p_amount) {
	VS_CHANGED;
	rasterizer->particles_set_amount(p_particles,p_amount);
}

int VisualServerRaster::particles_get_amount(RID p_particles) const {

	return rasterizer->particles_get_amount(p_particles);
}

void VisualServerRaster::particles_set_emitting(RID p_particles, bool p_emitting) {
	VS_CHANGED;
	rasterizer->particles_set_emitting(p_particles,p_emitting);
}
bool VisualServerRaster::particles_is_emitting(RID p_particles) const {

	return rasterizer->particles_is_emitting(p_particles);
}

void VisualServerRaster::particles_set_visibility_aabb(RID p_particles, const AABB& p_visibility) {
	VS_CHANGED;
	rasterizer->particles_set_visibility_aabb(p_particles, p_visibility);
}
AABB VisualServerRaster::particles_get_visibility_aabb(RID p_particles) const {

	return rasterizer->particles_get_visibility_aabb(p_particles);
}

void VisualServerRaster::particles_set_emission_half_extents(RID p_particles, const Vector3& p_half_extents) {
	VS_CHANGED;
	rasterizer->particles_set_emission_half_extents(p_particles,p_half_extents);
}
Vector3 VisualServerRaster::particles_get_emission_half_extents(RID p_particles) const {

	return rasterizer->particles_get_emission_half_extents(p_particles);
}

void VisualServerRaster::particles_set_emission_base_velocity(RID p_particles, const Vector3& p_base_velocity) {
	VS_CHANGED;
	rasterizer->particles_set_emission_base_velocity(p_particles,p_base_velocity);
}
Vector3 VisualServerRaster::particles_get_emission_base_velocity(RID p_particles) const {

	return rasterizer->particles_get_emission_base_velocity(p_particles);
}

void VisualServerRaster::particles_set_emission_points(RID p_particles, const PoolVector<Vector3>& p_points) {

	VS_CHANGED;
	rasterizer->particles_set_emission_points(p_particles,p_points);
}

PoolVector<Vector3> VisualServerRaster::particles_get_emission_points(RID p_particles) const {

	return rasterizer->particles_get_emission_points(p_particles);
}

void VisualServerRaster::particles_set_gravity_normal(RID p_particles, const Vector3& p_normal) {
	VS_CHANGED;
	rasterizer->particles_set_gravity_normal(p_particles,p_normal);
}
Vector3 VisualServerRaster::particles_get_gravity_normal(RID p_particles) const {

	return rasterizer->particles_get_gravity_normal(p_particles);
}

void VisualServerRaster::particles_set_variable(RID p_particles, ParticleVariable p_variable,float p_value) {
	VS_CHANGED;
	rasterizer->particles_set_variable(p_particles,p_variable,p_value);
}
float VisualServerRaster::particles_get_variable(RID p_particles, ParticleVariable p_variable) const {

	return rasterizer->particles_get_variable(p_particles,p_variable);
}

void VisualServerRaster::particles_set_randomness(RID p_particles, ParticleVariable p_variable,float p_randomness) {
	VS_CHANGED;
	rasterizer->particles_set_randomness(p_particles,p_variable,p_randomness);
}
float VisualServerRaster::particles_get_randomness(RID p_particles, ParticleVariable p_variable) const {

	return rasterizer->particles_get_randomness(p_particles,p_variable);
}

void VisualServerRaster::particles_set_color_phases(RID p_particles, int p_phases) {
	VS_CHANGED;
	rasterizer->particles_set_color_phases(p_particles,p_phases);
}
int VisualServerRaster::particles_get_color_phases(RID p_particles) const {

	return rasterizer->particles_get_color_phases(p_particles);
}

void VisualServerRaster::particles_set_color_phase_pos(RID p_particles, int p_phase, float p_pos) {
	VS_CHANGED;
	rasterizer->particles_set_color_phase_pos(p_particles,p_phase,p_pos);
}
float VisualServerRaster::particles_get_color_phase_pos(RID p_particles, int p_phase) const {

	return rasterizer->particles_get_color_phase_pos(p_particles,p_phase);
}

void VisualServerRaster::particles_set_attractors(RID p_particles, int p_attractors) {
	VS_CHANGED;
	rasterizer->particles_set_attractors(p_particles,p_attractors);
}
int VisualServerRaster::particles_get_attractors(RID p_particles) const {

	return rasterizer->particles_get_attractors(p_particles);
}

void VisualServerRaster::particles_set_attractor_pos(RID p_particles, int p_attractor, const Vector3& p_pos) {
	VS_CHANGED;
	rasterizer->particles_set_attractor_pos(p_particles,p_attractor,p_pos);
}
Vector3 VisualServerRaster::particles_get_attractor_pos(RID p_particles,int p_attractor) const {

	return rasterizer->particles_get_attractor_pos(p_particles,p_attractor);
}

void VisualServerRaster::particles_set_attractor_strength(RID p_particles, int p_attractor, float p_force) {
	VS_CHANGED;
	rasterizer->particles_set_attractor_strength(p_particles,p_attractor,p_force);
}

float VisualServerRaster::particles_get_attractor_strength(RID p_particles,int p_attractor) const {

	return rasterizer->particles_get_attractor_strength(p_particles,p_attractor);
}

void VisualServerRaster::particles_set_color_phase_color(RID p_particles, int p_phase, const Color& p_color) {
	VS_CHANGED;
	rasterizer->particles_set_color_phase_color(p_particles,p_phase,p_color);
}
Color VisualServerRaster::particles_get_color_phase_color(RID p_particles, int p_phase) const {

	return rasterizer->particles_get_color_phase_color(p_particles,p_phase);
}

void VisualServerRaster::particles_set_material(RID p_particles, RID p_material,bool p_owned) {
	VS_CHANGED;
	rasterizer->particles_set_material(p_particles,p_material,p_owned);
}
RID VisualServerRaster::particles_get_material(RID p_particles) const {

	return rasterizer->particles_get_material(p_particles);
}


void VisualServerRaster::particles_set_height_from_velocity(RID p_particles, bool p_enable) {

	VS_CHANGED;
	rasterizer->particles_set_height_from_velocity(p_particles,p_enable);

}

bool VisualServerRaster::particles_has_height_from_velocity(RID p_particles) const {

	return rasterizer->particles_has_height_from_velocity(p_particles);
}

void VisualServerRaster::particles_set_use_local_coordinates(RID p_particles, bool p_enable) {

	rasterizer->particles_set_use_local_coordinates(p_particles,p_enable);
}

bool VisualServerRaster::particles_is_using_local_coordinates(RID p_particles) const {

	return rasterizer->particles_is_using_local_coordinates(p_particles);
}


/* Light API */

RID VisualServerRaster::light_create(LightType p_type) {

	return rasterizer->light_create(p_type);
}
VisualServer::LightType VisualServerRaster::light_get_type(RID p_light) const {

	return rasterizer->light_get_type(p_light);
}

void VisualServerRaster::light_set_color(RID p_light,LightColor p_type, const Color& p_color) {
	VS_CHANGED;
	rasterizer->light_set_color(p_light,p_type,p_color);

}
Color VisualServerRaster::light_get_color(RID p_light,LightColor p_type) const {

	return rasterizer->light_get_color(p_light,p_type);

}


void VisualServerRaster::light_set_shadow(RID p_light,bool p_enabled) {
	VS_CHANGED;
	rasterizer->light_set_shadow(p_light,p_enabled);
}

bool VisualServerRaster::light_has_shadow(RID p_light) const {

	return rasterizer->light_has_shadow(p_light);
}



void VisualServerRaster::light_set_volumetric(RID p_light,bool p_enabled) {
	VS_CHANGED;
	rasterizer->light_set_volumetric(p_light,p_enabled);
}

bool VisualServerRaster::light_is_volumetric(RID p_light) const {

	return rasterizer->light_is_volumetric(p_light);
}

void VisualServerRaster::light_set_projector(RID p_light,RID p_texture) {
	VS_CHANGED;
	rasterizer->light_set_projector(p_light,p_texture);
}

RID VisualServerRaster::light_get_projector(RID p_light) const {

	return rasterizer->light_get_projector(p_light);
}

void VisualServerRaster::light_set_param(RID p_light, LightParam p_var, float p_value) {
	VS_CHANGED;
	rasterizer->light_set_var(p_light,p_var,p_value);
	_dependency_queue_update(p_light,true);

}

float VisualServerRaster::light_get_param(RID p_light, LightParam p_var) const {


	return rasterizer->light_get_var(p_light,p_var);
}

void VisualServerRaster::light_set_operator(RID p_light,LightOp p_op) {
	VS_CHANGED;
	rasterizer->light_set_operator(p_light,p_op);
}

VisualServerRaster::LightOp VisualServerRaster::light_get_operator(RID p_light) const {

	return rasterizer->light_get_operator(p_light);
}

void VisualServerRaster::light_omni_set_shadow_mode(RID p_light,LightOmniShadowMode p_mode) {
	VS_CHANGED;
	rasterizer->light_omni_set_shadow_mode(p_light,p_mode);
}

VisualServerRaster::LightOmniShadowMode VisualServerRaster::light_omni_get_shadow_mode(RID p_light) const {

	return rasterizer->light_omni_get_shadow_mode(p_light);
}

void VisualServerRaster::light_directional_set_shadow_mode(RID p_light,LightDirectionalShadowMode p_mode){
	VS_CHANGED;
	rasterizer->light_directional_set_shadow_mode(p_light,p_mode);
}

VS::LightDirectionalShadowMode VisualServerRaster::light_directional_get_shadow_mode(RID p_light) const{

	return rasterizer->light_directional_get_shadow_mode(p_light);
}

void VisualServerRaster::light_directional_set_shadow_param(RID p_light,LightDirectionalShadowParam p_param, float p_value) {
	VS_CHANGED;
	rasterizer->light_directional_set_shadow_param(p_light,p_param,p_value);
}

float VisualServerRaster::light_directional_get_shadow_param(RID p_light,LightDirectionalShadowParam p_param) const {

	return rasterizer->light_directional_get_shadow_param(p_light,p_param);
}


RID VisualServerRaster::skeleton_create() {

	return rasterizer->skeleton_create();
}

void VisualServerRaster::skeleton_resize(RID p_skeleton,int p_bones) {
	VS_CHANGED;
	rasterizer->skeleton_resize(p_skeleton,p_bones);
}

int VisualServerRaster::skeleton_get_bone_count(RID p_skeleton) const {

	return rasterizer->skeleton_get_bone_count(p_skeleton);
}

void VisualServerRaster::skeleton_bone_set_transform(RID p_skeleton,int p_bone, const Transform& p_transform) {
	VS_CHANGED;
	rasterizer->skeleton_bone_set_transform(p_skeleton,p_bone,p_transform);

	Map< RID, Set<Instance*> >::Element *E=skeleton_dependency_map.find(p_skeleton);

	if (E) {
		//detach skeletons
		for (Set<Instance*>::Element *F=E->get().front();F;F=F->next()) {

			_instance_queue_update( F->get() , true);
		}
	}
}

Transform VisualServerRaster::skeleton_bone_get_transform(RID p_skeleton,int p_bone) {


	return rasterizer->skeleton_bone_get_transform(p_skeleton,p_bone);

}


/* VISIBILITY API */

/* ROOM API */

RID VisualServerRaster::room_create() {

	Room *room = memnew( Room );
	ERR_FAIL_COND_V(!room,RID());
	return room_owner.make_rid( room );

}

void VisualServerRaster::room_set_bounds(RID p_room, const BSP_Tree& p_bounds) {
	VS_CHANGED;
	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND(!room);
	room->bounds=p_bounds;
	_dependency_queue_update(p_room,true);

}

BSP_Tree VisualServerRaster::room_get_bounds(RID p_room) const {

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND_V(!room, BSP_Tree());
	return room->bounds;

}

/* PORTAL API */

RID VisualServerRaster::portal_create() {
	VS_CHANGED;
	Portal *portal = memnew( Portal );
	ERR_FAIL_COND_V(!portal,RID());
	return portal_owner.make_rid( portal );


}


void VisualServerRaster::portal_set_shape(RID p_portal, const Vector<Point2>& p_shape) {
	VS_CHANGED;
	Portal *portal = portal_owner.get(p_portal);
	ERR_FAIL_COND(!portal);
	portal->shape=p_shape;

	portal->bounds=Rect2();

	for(int i=0;i<p_shape.size();i++) {

		if (i==0)
			portal->bounds.pos=p_shape[i];
		else
			portal->bounds.expand_to(p_shape[i]);
	}

	_dependency_queue_update(p_portal,true);
}


Vector<Point2> VisualServerRaster::portal_get_shape(RID p_portal) const {

	Portal *portal = portal_owner.get(p_portal);
	ERR_FAIL_COND_V(!portal, Vector<Point2>());
	return portal->shape;


}

void VisualServerRaster::portal_set_enabled(RID p_portal, bool p_enabled) {
	VS_CHANGED;
	Portal *portal = portal_owner.get(p_portal);
	ERR_FAIL_COND(!portal);
	portal->enabled=p_enabled;

}

bool VisualServerRaster::portal_is_enabled(RID p_portal) const {

	Portal *portal = portal_owner.get(p_portal);
	ERR_FAIL_COND_V(!portal, false);
	return portal->enabled;

}
void VisualServerRaster::portal_set_disable_distance(RID p_portal, float p_distance) {
	VS_CHANGED;
	Portal *portal = portal_owner.get(p_portal);
	ERR_FAIL_COND(!portal);
	portal->disable_distance=p_distance;

}
float VisualServerRaster::portal_get_disable_distance(RID p_portal) const {

	Portal *portal = portal_owner.get(p_portal);
	ERR_FAIL_COND_V(!portal, -1);
	return portal->disable_distance;

}
void VisualServerRaster::portal_set_disabled_color(RID p_portal, const Color& p_color) {
	VS_CHANGED;
	Portal *portal = portal_owner.get(p_portal);
	ERR_FAIL_COND(!portal);
	portal->disable_color=p_color;

}
Color VisualServerRaster::portal_get_disabled_color(RID p_portal) const {

	Portal *portal = portal_owner.get(p_portal);
	ERR_FAIL_COND_V(!portal, Color());
	return portal->disable_color;
}

void VisualServerRaster::portal_set_connect_range(RID p_portal, float p_range) {
	VS_CHANGED;
	Portal *portal = portal_owner.get(p_portal);
	ERR_FAIL_COND(!portal);
	portal->connect_range=p_range;
	_dependency_queue_update(p_portal,true);
}

float VisualServerRaster::portal_get_connect_range(RID p_portal) const {

	Portal *portal = portal_owner.get(p_portal);
	ERR_FAIL_COND_V(!portal,0);
	return portal->connect_range;
}


RID VisualServerRaster::baked_light_create() {

	BakedLight *baked_light = memnew( BakedLight );
	ERR_FAIL_COND_V(!baked_light,RID());
	baked_light->data.mode=BAKED_LIGHT_OCTREE;

	baked_light->data.octree_lattice_size=0;
	baked_light->data.octree_lattice_divide=0;
	baked_light->data.octree_steps=1;
	baked_light->data.lightmap_multiplier=1.0;
	baked_light->data.realtime_color_enabled=false;
	baked_light->data.realtime_color=Color(1.0, 1.0, 1.0);
	baked_light->data.realtime_energy = 1.0;

	return baked_light_owner.make_rid( baked_light );

}

void VisualServerRaster::baked_light_set_mode(RID p_baked_light,BakedLightMode p_mode){

	VS_CHANGED;
	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND(!baked_light);
	baked_light->data.mode=p_mode;
	baked_light->data.color_multiplier=1.0;
	_dependency_queue_update(p_baked_light,true);


}

VisualServer::BakedLightMode VisualServerRaster::baked_light_get_mode(RID p_baked_light) const{

	const BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND_V(!baked_light,BAKED_LIGHT_OCTREE);
	return baked_light->data.mode;

}

void VisualServerRaster::baked_light_set_lightmap_multiplier(RID p_baked_light,float p_multiplier) {

	VS_CHANGED;
	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND(!baked_light);

	baked_light->data.lightmap_multiplier=p_multiplier;

}

float VisualServerRaster::baked_light_get_lightmap_multiplier(RID p_baked_light) const{

	const BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND_V(!baked_light,0);

	return baked_light->data.lightmap_multiplier;

}


void VisualServerRaster::baked_light_set_octree(RID p_baked_light,const PoolVector<uint8_t> p_octree){

	VS_CHANGED;
	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND(!baked_light);

	if (p_octree.size()==0) {
		if (baked_light->data.octree_texture.is_valid())
			rasterizer->free(baked_light->data.octree_texture);
		baked_light->data.octree_texture;
		baked_light->octree_aabb=AABB();
		baked_light->octree_tex_size=Size2();
	} else {

		int tex_w;
		int tex_h;
		int light_tex_w;
		int light_tex_h;
		bool is16;
		bool has_light_tex=false;
		{

			PoolVector<uint8_t>::Read r=p_octree.read();
			tex_w = decode_uint32(&r[0]);
			tex_h = decode_uint32(&r[4]);
			print_line("TEX W: "+itos(tex_w)+" TEX H:"+itos(tex_h)+" LEN: "+itos(p_octree.size()));
			is16=decode_uint32(&r[8]);
			baked_light->data.octree_lattice_size=decode_float(&r[12]);
			baked_light->data.octree_lattice_divide=tex_w/4.0;
			print_line("LATTICE SIZE: "+rtos(baked_light->data.octree_lattice_size));
			print_line("LATTICE DIVIDE: "+rtos(baked_light->data.octree_lattice_divide));
			baked_light->data.octree_steps=decode_uint32(&r[16]);
			baked_light->data.octree_tex_pixel_size.x=1.0/tex_w;
			baked_light->data.octree_tex_pixel_size.y=1.0/tex_h;

			baked_light->data.texture_multiplier=decode_uint32(&r[20]);
			light_tex_w=decode_uint16(&r[24]);
			light_tex_h=decode_uint16(&r[26]);
			print_line("ltexw "+itos(light_tex_w));
			print_line("ltexh "+itos(light_tex_h));

			if (light_tex_w>0 && light_tex_h>0) {
				baked_light->data.light_tex_pixel_size.x=1.0/light_tex_w;
				baked_light->data.light_tex_pixel_size.y=1.0/light_tex_h;
				has_light_tex=true;
			} else {
				baked_light->data.light_tex_pixel_size=baked_light->data.octree_tex_pixel_size;

			}



			baked_light->octree_aabb.pos.x=decode_float(&r[32]);
			baked_light->octree_aabb.pos.y=decode_float(&r[36]);
			baked_light->octree_aabb.pos.z=decode_float(&r[40]);
			baked_light->octree_aabb.size.x=decode_float(&r[44]);
			baked_light->octree_aabb.size.y=decode_float(&r[48]);
			baked_light->octree_aabb.size.z=decode_float(&r[52]);


		}

		if (baked_light->data.octree_texture.is_valid()) {
			if (tex_w!=baked_light->octree_tex_size.x || tex_h!=baked_light->octree_tex_size.y) {

				rasterizer->free(baked_light->data.octree_texture);
				baked_light->data.octree_texture;
				baked_light->octree_tex_size.x=0;
				baked_light->octree_tex_size.y=0;
			}
		}

		if (baked_light->data.light_texture.is_valid()) {
			if (!has_light_tex || light_tex_w!=baked_light->light_tex_size.x || light_tex_h!=baked_light->light_tex_size.y) {
				rasterizer->free(baked_light->data.light_texture);
				baked_light->data.light_texture;
				baked_light->light_tex_size.x=0;
				baked_light->light_tex_size.y=0;
			}
		}

		if (!baked_light->data.octree_texture.is_valid()) {
			baked_light->data.octree_texture=rasterizer->texture_create();
			rasterizer->texture_allocate(baked_light->data.octree_texture,tex_w,tex_h,Image::FORMAT_RGBA8,TEXTURE_FLAG_FILTER);
			baked_light->octree_tex_size.x=tex_w;
			baked_light->octree_tex_size.y=tex_h;
		}

		if (!baked_light->data.light_texture.is_valid() && has_light_tex) {
			baked_light->data.light_texture=rasterizer->texture_create();
			rasterizer->texture_allocate(baked_light->data.light_texture,light_tex_w,light_tex_h,Image::FORMAT_RGBA8,TEXTURE_FLAG_FILTER);
			baked_light->light_tex_size.x=light_tex_w;
			baked_light->light_tex_size.y=light_tex_h;

		}

		Image img(tex_w,tex_h,0,Image::FORMAT_RGBA8,p_octree);
		rasterizer->texture_set_data(baked_light->data.octree_texture,img);

	}


	_dependency_queue_update(p_baked_light,true);

}


PoolVector<uint8_t> VisualServerRaster::baked_light_get_octree(RID p_baked_light) const{


	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND_V(!baked_light,PoolVector<uint8_t>());

	if (rasterizer->is_texture(baked_light->data.octree_texture)) {

		Image img = rasterizer->texture_get_data(baked_light->data.octree_texture);
		return img.get_data();
	} else {
		return PoolVector<uint8_t>();
	}
}

void VisualServerRaster::baked_light_set_light(RID p_baked_light,const PoolVector<uint8_t> p_light) {

	VS_CHANGED;
	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND(!baked_light);
	ERR_FAIL_COND(p_light.size()==0);

	int tex_w=baked_light->light_tex_size.x;
	int tex_h=baked_light->light_tex_size.y;

	ERR_FAIL_COND(tex_w==0 && tex_h==0);
	ERR_FAIL_COND(!baked_light->data.light_texture.is_valid());



	print_line("w: "+itos(tex_w)+" h: "+itos(tex_h)+" lightsize: "+itos(p_light.size()));

	Image img(tex_w,tex_h,0,Image::FORMAT_RGBA8,p_light);
	rasterizer->texture_set_data(baked_light->data.light_texture,img);



}

PoolVector<uint8_t> VisualServerRaster::baked_light_get_light(RID p_baked_light) const{

	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND_V(!baked_light,PoolVector<uint8_t>());

	if (rasterizer->is_texture(baked_light->data.light_texture)) {

		Image img = rasterizer->texture_get_data(baked_light->data.light_texture);
		return img.get_data();
	} else {
		return PoolVector<uint8_t>();
	}
}



void VisualServerRaster::baked_light_set_sampler_octree(RID p_baked_light, const PoolVector<int> &p_sampler) {

	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND(!baked_light);

	baked_light->sampler=p_sampler;



}

PoolVector<int> VisualServerRaster::baked_light_get_sampler_octree(RID p_baked_light) const {

	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND_V(!baked_light,PoolVector<int>());

	return baked_light->sampler;

}


void VisualServerRaster::baked_light_add_lightmap(RID p_baked_light,const RID p_texture,int p_id){

	VS_CHANGED;
	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND(!baked_light);
	baked_light->data.lightmaps.insert(p_id,p_texture);

}
void VisualServerRaster::baked_light_clear_lightmaps(RID p_baked_light){

	VS_CHANGED;
	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND(!baked_light);
	baked_light->data.lightmaps.clear();

}

void VisualServerRaster::baked_light_set_realtime_color_enabled(RID p_baked_light, const bool p_enabled) {

	VS_CHANGED;
	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND(!baked_light);

	baked_light->data.realtime_color_enabled = p_enabled;

}

bool VisualServerRaster::baked_light_get_realtime_color_enabled(RID p_baked_light) const{

	const BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND_V(!baked_light, false);

	return baked_light->data.realtime_color_enabled;

}

void VisualServerRaster::baked_light_set_realtime_color(RID p_baked_light, const Color& p_color) {

	VS_CHANGED;
	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND(!baked_light);

	baked_light->data.realtime_color = p_color;

}

Color VisualServerRaster::baked_light_get_realtime_color(RID p_baked_light) const{

	const BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND_V(!baked_light, Color(1.0, 1.0, 1.0));

	return baked_light->data.realtime_color;

}

void VisualServerRaster::baked_light_set_realtime_energy(RID p_baked_light, const float p_energy) {

	VS_CHANGED;
	BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND(!baked_light);

	baked_light->data.realtime_energy = p_energy;

}

float VisualServerRaster::baked_light_get_realtime_energy(RID p_baked_light) const{

	const BakedLight *baked_light = baked_light_owner.get(p_baked_light);
	ERR_FAIL_COND_V(!baked_light, 1.0f);

	return baked_light->data.realtime_energy;

}


/* BAKED LIGHT SAMPLER */

RID VisualServerRaster::baked_light_sampler_create() {

	BakedLightSampler * blsamp = memnew( BakedLightSampler );
	RID rid = baked_light_sampler_owner.make_rid(blsamp);
	_update_baked_light_sampler_dp_cache(blsamp);
	return rid;
}

void VisualServerRaster::baked_light_sampler_set_param(RID p_baked_light_sampler,BakedLightSamplerParam p_param,float p_value){

	VS_CHANGED;
	BakedLightSampler * blsamp = baked_light_sampler_owner.get(p_baked_light_sampler);
	ERR_FAIL_COND(!blsamp);
	ERR_FAIL_INDEX(p_param,BAKED_LIGHT_SAMPLER_MAX);
	blsamp->params[p_param]=p_value;
	_dependency_queue_update(p_baked_light_sampler,true);
}

float VisualServerRaster::baked_light_sampler_get_param(RID p_baked_light_sampler,BakedLightSamplerParam p_param) const{


	BakedLightSampler * blsamp = baked_light_sampler_owner.get(p_baked_light_sampler);
	ERR_FAIL_COND_V(!blsamp,0);
	ERR_FAIL_INDEX_V(p_param,BAKED_LIGHT_SAMPLER_MAX,0);
	return blsamp->params[p_param];
}

void VisualServerRaster::_update_baked_light_sampler_dp_cache(BakedLightSampler * blsamp) {

	int res = blsamp->resolution;
	blsamp->dp_cache.resize(res*res*2);
	Vector3 *dp_normals=blsamp->dp_cache.ptr();

	for(int p=0;p<2;p++) {
		float sign = p==0?1:-1;
		int ofs = res*res*p;
		for(int i=0;i<res;i++) {
			for(int j=0;j<res;j++) {

				Vector2 v(
					(i/float(res))*2.0-1.0,
					(j/float(res))*2.0-1.0
				);

				float l=v.length();
				if (l>1.0) {
					v/=l;
					l=1.0; //clamp to avoid imaginary
				}
				v*=(2*l)/(l*l+1); //inverse of the dual paraboloid function
				Vector3 n = Vector3(v.x,v.y,sign*sqrtf(MAX(1 - v.dot(v),0))); //reconstruction of z
				n.y*=sign;
				dp_normals[j*res+i+ofs]=n;
			}
		}
	}

}

void VisualServerRaster::baked_light_sampler_set_resolution(RID p_baked_light_sampler,int p_resolution){

    ERR_FAIL_COND(p_resolution<4 || p_resolution>64);
	VS_CHANGED;
	BakedLightSampler * blsamp = baked_light_sampler_owner.get(p_baked_light_sampler);
	ERR_FAIL_COND(!blsamp);
	blsamp->resolution=p_resolution;
	_update_baked_light_sampler_dp_cache(blsamp);

}
int VisualServerRaster::baked_light_sampler_get_resolution(RID p_baked_light_sampler) const{

	BakedLightSampler * blsamp = baked_light_sampler_owner.get(p_baked_light_sampler);
	ERR_FAIL_COND_V(!blsamp,0);
	return blsamp->resolution;
}

/* CAMERA API */

RID VisualServerRaster::camera_create() {

	Camera * camera = memnew( Camera );
	return camera_owner.make_rid( camera );

}

void VisualServerRaster::camera_set_perspective(RID p_camera,float p_fovy_degrees, float p_z_near, float p_z_far) {
	VS_CHANGED
	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);
	camera->type=Camera::PERSPECTIVE;
	camera->fov=p_fovy_degrees;
	camera->znear=p_z_near;
	camera->zfar=p_z_far;

}

void VisualServerRaster::camera_set_orthogonal(RID p_camera,float p_size, float p_z_near, float p_z_far) {
	VS_CHANGED;
	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);
	camera->type=Camera::ORTHOGONAL;
	camera->size=p_size;
	camera->znear=p_z_near;
	camera->zfar=p_z_far;
}

void VisualServerRaster::camera_set_transform(RID p_camera,const Transform& p_transform) {
	VS_CHANGED;
	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);
	camera->transform=p_transform.orthonormalized();


}

void VisualServerRaster::camera_set_visible_layers(RID p_camera,uint32_t p_layers) {

	VS_CHANGED;
	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);

	camera->visible_layers=p_layers;

}

uint32_t VisualServerRaster::camera_get_visible_layers(RID p_camera) const{

	const Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND_V(!camera,0);

	return camera->visible_layers;
}

void VisualServerRaster::camera_set_environment(RID p_camera,RID p_env) {

	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);
	camera->env=p_env;

}

RID VisualServerRaster::camera_get_environment(RID p_camera) const {

	const Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND_V(!camera,RID());
	return camera->env;

}

void VisualServerRaster::camera_set_use_vertical_aspect(RID p_camera,bool p_enable) {

	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);
	camera->vaspect=p_enable;

}
bool VisualServerRaster::camera_is_using_vertical_aspect(RID p_camera,bool p_enable) const{

	const Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND_V(!camera,false);
	return camera->vaspect;

}


/* VIEWPORT API */


RID VisualServerRaster::viewport_create() {

	Viewport *viewport = memnew( Viewport );
	RID rid = viewport_owner.make_rid( viewport );
	ERR_FAIL_COND_V( !rid.is_valid(), rid );

	viewport->self=rid;
	viewport->hide_scenario=false;
	viewport->hide_canvas=false;
	viewport->viewport_data=rasterizer->viewport_data_create();

	return rid;
}

void VisualServerRaster::viewport_attach_to_screen(RID p_viewport,int p_screen) {

	VS_CHANGED;
	Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);


	screen_viewports[p_viewport]=p_screen;
}

void VisualServerRaster::viewport_detach(RID p_viewport) {

	VS_CHANGED;
	Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	ERR_FAIL_COND(!screen_viewports.has(p_viewport));
	screen_viewports.erase(p_viewport);

}

void VisualServerRaster::viewport_set_as_render_target(RID p_viewport,bool p_enable) {

	VS_CHANGED;
	Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	if (viewport->render_target.is_valid()==p_enable)
		return;
	if (!p_enable) {

		rasterizer->free(viewport->render_target);
		viewport->render_target;
		viewport->render_target_texture;
		if (viewport->update_list.in_list())
			viewport_update_list.remove(&viewport->update_list);

	} else {

		viewport->render_target=rasterizer->render_target_create();
		rasterizer->render_target_set_size(viewport->render_target,viewport->rect.width,viewport->rect.height);
		viewport->render_target_texture=rasterizer->render_target_get_texture(viewport->render_target);
		if (viewport->render_target_update_mode!=RENDER_TARGET_UPDATE_DISABLED)
			viewport_update_list.add(&viewport->update_list);
	}

}

void VisualServerRaster::viewport_set_render_target_update_mode(RID p_viewport,RenderTargetUpdateMode p_mode){

	VS_CHANGED;
	Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	if (viewport->render_target.is_valid() && viewport->update_list.in_list())
		viewport_update_list.remove(&viewport->update_list);

	viewport->render_target_update_mode=p_mode;

	if (viewport->render_target.is_valid() &&viewport->render_target_update_mode!=RENDER_TARGET_UPDATE_DISABLED)
		viewport_update_list.add(&viewport->update_list);

}
VisualServer::RenderTargetUpdateMode VisualServerRaster::viewport_get_render_target_update_mode(RID p_viewport) const{

	const Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND_V(!viewport,RENDER_TARGET_UPDATE_DISABLED);

	return viewport->render_target_update_mode;
}
RID VisualServerRaster::viewport_get_render_target_texture(RID p_viewport) const{

	Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND_V(!viewport,RID());

	return viewport->render_target_texture;

}

void VisualServerRaster::viewport_set_render_target_vflip(RID p_viewport,bool p_enable) {

	Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	viewport->render_target_vflip=p_enable;

}

void VisualServerRaster::viewport_set_render_target_clear_on_new_frame(RID p_viewport,bool p_enable) {

	Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	viewport->render_target_clear_on_new_frame=p_enable;

}

void VisualServerRaster::viewport_set_render_target_to_screen_rect(RID p_viewport,const Rect2& p_rect) {

	Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	viewport->rt_to_screen_rect=p_rect;

}

bool VisualServerRaster::viewport_get_render_target_vflip(RID p_viewport) const{

	const Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND_V(!viewport,false);

	return viewport->render_target_vflip;

}

bool VisualServerRaster::viewport_get_render_target_clear_on_new_frame(RID p_viewport) const{

	const Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND_V(!viewport,false);

	return viewport->render_target_clear_on_new_frame;

}

void VisualServerRaster::viewport_render_target_clear(RID p_viewport) {

	Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	viewport->render_target_clear=true;

}

void VisualServerRaster::viewport_queue_screen_capture(RID p_viewport) {

	VS_CHANGED;
	Viewport *viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);
	viewport->queue_capture=true;

}

Image VisualServerRaster::viewport_get_screen_capture(RID p_viewport) const {

	Viewport *viewport = (Viewport*)viewport_owner.get( p_viewport );
	ERR_FAIL_COND_V(!viewport,Image());

	Image ret = viewport->capture;
	viewport->capture=Image();
	return ret;
}

void VisualServerRaster::viewport_set_rect(RID p_viewport,const ViewportRect& p_rect) {
	VS_CHANGED;
	Viewport *viewport=NULL;

	viewport = viewport_owner.get( p_viewport );

	ERR_FAIL_COND(!viewport);

	viewport->rect=p_rect;
	if (viewport->render_target.is_valid()) {
		rasterizer->render_target_set_size(viewport->render_target,viewport->rect.width,viewport->rect.height);
	}
}


VisualServer::ViewportRect VisualServerRaster::viewport_get_rect(RID p_viewport) const {

	const Viewport *viewport=NULL;

	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND_V(!viewport, ViewportRect());

	return viewport->rect;
}

void VisualServerRaster::viewport_set_hide_scenario(RID p_viewport,bool p_hide) {

	VS_CHANGED;

	Viewport *viewport=NULL;

	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	viewport->hide_scenario=p_hide;


}

void VisualServerRaster::viewport_set_hide_canvas(RID p_viewport,bool p_hide) {

	VS_CHANGED;

	Viewport *viewport=NULL;

	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	viewport->hide_canvas=p_hide;


}

void VisualServerRaster::viewport_set_disable_environment(RID p_viewport,bool p_disable) {

	VS_CHANGED;

	Viewport *viewport=NULL;
	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);
	viewport->disable_environment=p_disable;

}

void VisualServerRaster::viewport_attach_camera(RID p_viewport,RID p_camera) {
	VS_CHANGED;

	Viewport *viewport=NULL;
	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);




	if (p_camera.is_valid()) {

		ERR_FAIL_COND(!camera_owner.owns(p_camera));
		// a camera
		viewport->camera=p_camera;
	} else {
		viewport->camera;
	}

}

void VisualServerRaster::viewport_set_scenario(RID p_viewport,RID p_scenario) {

	VS_CHANGED;

	Viewport *viewport=NULL;
	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	if (p_scenario.is_valid()) {

		ERR_FAIL_COND(!scenario_owner.owns(p_scenario));
		// a camera
		viewport->scenario=p_scenario;
	} else {
		viewport->scenario;
	}

}

RID VisualServerRaster::viewport_get_attached_camera(RID p_viewport) const {

	const Viewport *viewport=NULL;

	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND_V(!viewport, RID());

	return viewport->camera;
}

void VisualServerRaster::viewport_attach_canvas(RID p_viewport,RID p_canvas) {
	VS_CHANGED;
	Viewport *viewport=NULL;

	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	Canvas *canvas = canvas_owner.get( p_canvas );
	ERR_FAIL_COND(!canvas);

	ERR_EXPLAIN("Canvas already attached.");
	ERR_FAIL_COND(viewport->canvas_map.has(p_canvas));


	Viewport::CanvasData cd;
	cd.canvas=canvas;
	cd.layer=0;

	viewport->canvas_map[p_canvas]=cd;
	canvas->viewports.insert(p_viewport);

}


void VisualServerRaster::viewport_set_canvas_transform(RID p_viewport,RID p_canvas,const Matrix32& p_transform) {

	VS_CHANGED;
	Viewport *viewport=NULL;
	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	Map<RID,Viewport::CanvasData>::Element *E=viewport->canvas_map.find(p_canvas);
	if (!E) {
		ERR_EXPLAIN("Viewport does not contain the canvas");
		ERR_FAIL_COND(!E);
	}

	E->get().transform=p_transform;

}

Matrix32 VisualServerRaster::viewport_get_canvas_transform(RID p_viewport,RID p_canvas) const {

	Viewport *viewport=NULL;
	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND_V(!viewport,Matrix32());

	Map<RID,Viewport::CanvasData>::Element *E=viewport->canvas_map.find(p_canvas);
	if (!E) {
		ERR_EXPLAIN("Viewport does not contain the canvas");
		ERR_FAIL_COND_V(!E,Matrix32());
	}


	return E->get().transform;
}


void VisualServerRaster::viewport_set_global_canvas_transform(RID p_viewport,const Matrix32& p_transform) {

	VS_CHANGED
	Viewport *viewport=NULL;
	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	viewport->global_transform=p_transform;

}

Matrix32 VisualServerRaster::viewport_get_global_canvas_transform(RID p_viewport) const{

	Viewport *viewport=NULL;
	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND_V(!viewport,Matrix32());
	return viewport->global_transform;
}

void VisualServerRaster::viewport_remove_canvas(RID p_viewport,RID p_canvas) {

	VS_CHANGED;
	Viewport *viewport=NULL;

	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	Canvas *canvas = canvas_owner.get( p_canvas );
	ERR_FAIL_COND(!canvas);


	Map<RID,Viewport::CanvasData>::Element *E=viewport->canvas_map.find(p_canvas);
	if (!E) {
		ERR_EXPLAIN("Viewport does not contain the canvas");
		ERR_FAIL_COND(!E);
	}


	canvas->viewports.erase(p_viewport);
	viewport->canvas_map.erase(E);

}


void VisualServerRaster::viewport_set_canvas_layer(RID p_viewport,RID p_canvas,int p_layer) {

	VS_CHANGED;
	Viewport *viewport=NULL;

	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	Map<RID,Viewport::CanvasData>::Element *E=viewport->canvas_map.find(p_canvas);
	if (!E) {
		ERR_EXPLAIN("Viewport does not contain the canvas");
		ERR_FAIL_COND(!E);
	}

	E->get().layer=p_layer;

}

void VisualServerRaster::viewport_set_transparent_background(RID p_viewport,bool p_enabled) {

	VS_CHANGED;
	Viewport *viewport=viewport_owner.get( p_viewport );
	ERR_FAIL_COND(!viewport);

	viewport->transparent_bg=p_enabled;
}

bool VisualServerRaster::viewport_has_transparent_background(RID p_viewport) const {

	Viewport *viewport=viewport_owner.get( p_viewport );
	ERR_FAIL_COND_V(!viewport, false);

	return viewport->transparent_bg;
}


RID VisualServerRaster::viewport_get_scenario(RID  p_viewport) const {

	const Viewport *viewport=NULL;

	viewport = viewport_owner.get( p_viewport );
	ERR_FAIL_COND_V(!viewport, RID());

	return viewport->scenario;
}


RID VisualServerRaster::environment_create() {

	return rasterizer->environment_create();
}

void VisualServerRaster::environment_set_background(RID p_env,EnvironmentBG p_bg){

	rasterizer->environment_set_background(p_env,p_bg);
}
VisualServer::EnvironmentBG VisualServerRaster::environment_get_background(RID p_env) const{

	return rasterizer->environment_get_background(p_env);
}

void VisualServerRaster::environment_set_background_param(RID p_env,EnvironmentBGParam p_param, const Variant& p_value){


	rasterizer->environment_set_background_param(p_env,p_param,p_value);
}
Variant VisualServerRaster::environment_get_background_param(RID p_env,EnvironmentBGParam p_param) const{

	return rasterizer->environment_get_background_param(p_env,p_param);
}

void VisualServerRaster::environment_set_enable_fx(RID p_env,EnvironmentFx p_effect,bool p_enabled){

	rasterizer->environment_set_enable_fx(p_env,p_effect,p_enabled);
}
bool VisualServerRaster::environment_is_fx_enabled(RID p_env,EnvironmentFx p_effect) const{

	return rasterizer->environment_is_fx_enabled(p_env,p_effect);
}


void VisualServerRaster::environment_fx_set_param(RID p_env,EnvironmentFxParam p_param,const Variant& p_value){

	rasterizer->environment_fx_set_param(p_env,p_param,p_value);
}
Variant VisualServerRaster::environment_fx_get_param(RID p_env,EnvironmentFxParam p_param) const {

	return environment_fx_get_param(p_env,p_param);
}



/* SCENARIO API */

void VisualServerRaster::_dependency_queue_update(RID p_rid,bool p_update_aabb,bool p_update_materials) {

	Map< RID, Set<RID> >::Element * E = instance_dependency_map.find( p_rid );

	if (!E)
		return;


	Set<RID>::Element *I = E->get().front();

	while(I) {

		Instance *ins = instance_owner.get( I->get() );
		_instance_queue_update( ins , p_update_aabb, p_update_materials );

		I = I->next();
	}

}

void VisualServerRaster::_instance_queue_update(Instance *p_instance,bool p_update_aabb,bool p_update_materials) {

	if (p_update_aabb)
		p_instance->update_aabb=true;
	if (p_update_materials)
		p_instance->update_materials=true;

	if (p_instance->update)
		return;
	p_instance->update_next=instance_update_list;
	instance_update_list=p_instance;
	p_instance->update=true;

}

RID VisualServerRaster::scenario_create() {

	Scenario *scenario = memnew( Scenario );
	ERR_FAIL_COND_V(!scenario,RID());
	RID scenario_rid = scenario_owner.make_rid( scenario );
	scenario->self=scenario_rid;
	scenario->octree.set_pair_callback(instance_pair,this);
	scenario->octree.set_unpair_callback(instance_unpair,this);

	return scenario_rid;
}

void VisualServerRaster::scenario_set_debug(RID p_scenario,ScenarioDebugMode p_debug_mode) {
	VS_CHANGED;

	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND(!scenario);
	scenario->debug=p_debug_mode;
}

void VisualServerRaster::scenario_set_environment(RID p_scenario, RID p_environment) {

	VS_CHANGED;

	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND(!scenario);
	scenario->environment=p_environment;

}

void VisualServerRaster::scenario_set_fallback_environment(RID p_scenario, RID p_environment) {

	VS_CHANGED;

	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND(!scenario);
	scenario->fallback_environment=p_environment;


}

RID VisualServerRaster::scenario_get_environment(RID p_scenario, RID p_environment) const{

	const Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND_V(!scenario,RID());
	return scenario->environment;

}


/* INSTANCING API */


RID VisualServerRaster::instance_create() {

	Instance *instance = memnew( Instance );
	ERR_FAIL_COND_V(!instance,RID());

	RID instance_rid = instance_owner.make_rid(instance);
	instance->self=instance_rid;
	instance->base_type=INSTANCE_NONE;
	instance->scenario=NULL;

	return instance_rid;
}

void VisualServerRaster::instance_set_base(RID p_instance, RID p_base) {

	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	if (instance->base_type!=INSTANCE_NONE) {
		//free anything related to that base

		Map< RID, Set<RID> >::Element * E = instance_dependency_map.find( instance->base_rid );

		if (E) {
			// wtf, no E?
			E->get().erase( instance->self );

		} else {

			ERR_PRINT("no base E? Bug?");
		}

		if ( instance->room ) {

			instance_set_room(p_instance,RID());
			/*
			if((1<<instance->base_type)&INSTANCE_GEOMETRY_MASK)
				instance->room->room_info->owned_geometry_instances.erase(instance->RE);
			else if (instance->base_type==INSTANCE_PORTAL) {
				print_line("freeing portal, is it there? "+itos(instance->room->room_info->owned_portal_instances.(instance->RE)));
				instance->room->room_info->owned_portal_instances.erase(instance->RE);
			} else if (instance->base_type==INSTANCE_ROOM)
				instance->room->room_info->owned_room_instances.erase(instance->RE);
			else if (instance->base_type==INSTANCE_LIGHT)
				instance->room->room_info->owned_light_instances.erase(instance->RE);

			instance->RE=NULL;*/
		}





		if (instance->light_info) {

			if (instance->scenario && instance->light_info->D)
				instance->scenario->directional_lights.erase( instance->light_info->D );
			rasterizer->free(instance->light_info->instance);
			memdelete(instance->light_info);
			instance->light_info=NULL;
		}


		if (instance->portal_info) {

			_portal_disconnect(instance,true);
			memdelete(instance->portal_info);
			instance->portal_info=NULL;

		}

		if (instance->baked_light_info) {

			while(instance->baked_light_info->owned_instances.size()) {

				Instance *owned=instance->baked_light_info->owned_instances.front()->get();
				owned->baked_light=NULL;
				owned->data.baked_light=NULL;
				owned->data.baked_light_octree_xform=NULL;
				owned->BLE=NULL;
				instance->baked_light_info->owned_instances.pop_front();
			}

			memdelete(instance->baked_light_info);
			instance->baked_light_info=NULL;

		}

		if (instance->scenario && instance->octree_id) {
			instance->scenario->octree.erase( instance->octree_id );
			instance->octree_id=0;
		}


		if (instance->room_info) {

			for(List<Instance*>::Element *E=instance->room_info->owned_geometry_instances.front();E;E=E->next()) {

				Instance *owned = E->get();
				owned->room=NULL;
				owned->RE=NULL;
			}
			for(List<Instance*>::Element *E=instance->room_info->owned_portal_instances.front();E;E=E->next()) {

				_portal_disconnect(E->get(),true);
				Instance *owned = E->get();
				owned->room=NULL;
				owned->RE=NULL;
			}

			for(List<Instance*>::Element *E=instance->room_info->owned_room_instances.front();E;E=E->next()) {

				Instance *owned = E->get();
				owned->room=NULL;
				owned->RE=NULL;
			}

			if (instance->room_info->disconnected_child_portals.size()) {
				ERR_PRINT("BUG: Disconnected portals remain!");
			}
			memdelete(instance->room_info);
			instance->room_info=NULL;

		}

		if (instance->particles_info) {

			rasterizer->free( instance->particles_info->instance );
			memdelete(instance->particles_info);
			instance->particles_info=NULL;

		}

		if (instance->baked_light_sampler_info) {

			while (instance->baked_light_sampler_info->owned_instances.size()) {

				instance_geometry_set_baked_light_sampler(instance->baked_light_sampler_info->owned_instances.front()->get()->self,RID());
			}

			if (instance->baked_light_sampler_info->sampled_light.is_valid()) {
				rasterizer->free(instance->baked_light_sampler_info->sampled_light);
			}
			memdelete( instance->baked_light_sampler_info );
			instance->baked_light_sampler_info=NULL;
		}

		instance->data.morph_values.clear();
		instance->data.materials.clear();

	}


	instance->base_type=INSTANCE_NONE;
	instance->base_rid;


	if (p_base.is_valid()) {

		if (rasterizer->is_mesh(p_base)) {
			instance->base_type=INSTANCE_MESH;
			instance->data.morph_values.resize( rasterizer->mesh_get_morph_target_count(p_base));
			instance->data.materials.resize( rasterizer->mesh_get_surface_count(p_base));
		} else if (rasterizer->is_multimesh(p_base)) {
			instance->base_type=INSTANCE_MULTIMESH;
		} else if (rasterizer->is_immediate(p_base)) {
			instance->base_type=INSTANCE_IMMEDIATE;
		} else if (rasterizer->is_particles(p_base)) {
			instance->base_type=INSTANCE_PARTICLES;
			instance->particles_info=memnew( Instance::ParticlesInfo );
			instance->particles_info->instance = rasterizer->particles_instance_create( p_base );
		} else if (rasterizer->is_light(p_base)) {

			instance->base_type=INSTANCE_LIGHT;
			instance->light_info = memnew( Instance::LightInfo );
			instance->light_info->instance = rasterizer->light_instance_create(p_base);
			if (instance->scenario && rasterizer->light_get_type(p_base)==LIGHT_DIRECTIONAL) {

				instance->light_info->D = instance->scenario->directional_lights.push_back(instance->self);
			}

		} else if (room_owner.owns(p_base)) {
			instance->base_type=INSTANCE_ROOM;
			instance->room_info  = memnew( Instance::RoomInfo );
			instance->room_info->room=room_owner.get(p_base);
		} else if (portal_owner.owns(p_base)) {

			instance->base_type=INSTANCE_PORTAL;
			instance->portal_info = memnew(Instance::PortalInfo);
			instance->portal_info->portal=portal_owner.get(p_base);
		} else if (baked_light_owner.owns(p_base)) {

			instance->base_type=INSTANCE_BAKED_LIGHT;
			instance->baked_light_info=memnew(Instance::BakedLightInfo);
			instance->baked_light_info->baked_light=baked_light_owner.get(p_base);

			//instance->portal_info = memnew(Instance::PortalInfo);
			//instance->portal_info->portal=portal_owner.get(p_base);
		} else if (baked_light_sampler_owner.owns(p_base)) {


			instance->base_type=INSTANCE_BAKED_LIGHT_SAMPLER;
			instance->baked_light_sampler_info=memnew( Instance::BakedLightSamplerInfo);
			instance->baked_light_sampler_info->sampler=baked_light_sampler_owner.get(p_base);

			//instance->portal_info = memnew(Instance::PortalInfo);
			//instance->portal_info->portal=portal_owner.get(p_base);

		} else {
			ERR_EXPLAIN("Invalid base RID for instance!")
			ERR_FAIL();
		}

		instance_dependency_map[ p_base ].insert( instance->self );

		instance->base_rid=p_base;

		if (instance->scenario)
			_instance_queue_update(instance,true);
	}

}

RID VisualServerRaster::instance_get_base(RID p_instance) const {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, RID() );
	return instance->base_rid;

}

void VisualServerRaster::instance_set_scenario(RID p_instance, RID p_scenario) {

	VS_CHANGED;

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	if (instance->scenario) {

		Map< RID, Set<RID> >::Element *E = instance_dependency_map.find( instance->scenario->self );

		if (E) {
			// wtf, no E?
			E->get().erase( instance->self );

		} else {

			ERR_PRINT("no scenario E? Bug?");
		}

		if (instance->light_info) {

			if (instance->light_info->D)
				instance->scenario->directional_lights.erase( instance->light_info->D );
		}

		if (instance->portal_info) {

			_portal_disconnect(instance,true);
		}

		if (instance->octree_id) {
			instance->scenario->octree.erase( instance->octree_id );
			instance->octree_id=0;
		}

		instance->scenario=NULL;
	}


	if (p_scenario.is_valid()) {
		Scenario *scenario = scenario_owner.get( p_scenario );
		ERR_FAIL_COND(!scenario);

		instance->scenario=scenario;

		instance_dependency_map[ p_scenario ].insert( instance->self );
		instance->scenario=scenario;

		if (instance->base_type==INSTANCE_LIGHT && rasterizer->light_get_type(instance->base_rid)==LIGHT_DIRECTIONAL) {

			instance->light_info->D = instance->scenario->directional_lights.push_back(instance->self);
		}

		_instance_queue_update(instance,true);
	}

}
RID VisualServerRaster::instance_get_scenario(RID p_instance) const {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, RID() );
	if (instance->scenario)
		return instance->scenario->self;
	else
		return RID();


}


void VisualServerRaster::instance_set_layer_mask(RID p_instance, uint32_t p_mask) {

	VS_CHANGED;

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	instance->layer_mask=p_mask;

}

uint32_t VisualServerRaster::instance_get_layer_mask(RID p_instance) const{

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, 0 );

	return instance->layer_mask;
}


AABB VisualServerRaster::instance_get_base_aabb(RID p_instance) const {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, AABB() );
	return instance->aabb;

}

void VisualServerRaster::instance_attach_object_instance_ID(RID p_instance,uint32_t p_ID) {
	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	instance->object_ID=p_ID;
}
uint32_t VisualServerRaster::instance_get_object_instance_ID(RID p_instance) const {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, 0 );
	return instance->object_ID;

}

void VisualServerRaster::instance_attach_skeleton(RID p_instance,RID p_skeleton) {
	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	if (instance->data.skeleton.is_valid()) {
		skeleton_dependency_map[instance->data.skeleton].erase(instance);
	}

	instance->data.skeleton=p_skeleton;

	if (instance->data.skeleton.is_valid()) {
		skeleton_dependency_map[instance->data.skeleton].insert(instance);
	}

}

RID VisualServerRaster::instance_get_skeleton(RID p_instance) const {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, RID() );
	return instance->data.skeleton;

}

void VisualServerRaster::instance_set_morph_target_weight(RID p_instance,int p_shape, float p_weight) {

	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );
	ERR_FAIL_INDEX( p_shape, instance->data.morph_values.size() );
	instance->data.morph_values[p_shape]=p_weight;
}

float VisualServerRaster::instance_get_morph_target_weight(RID p_instance,int p_shape) const {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, 0 );
	ERR_FAIL_INDEX_V( p_shape, instance->data.morph_values.size(), 0 );
	return instance->data.morph_values[p_shape];
}

void VisualServerRaster::instance_set_surface_material(RID p_instance,int p_surface, RID p_material) {

	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance);
	ERR_FAIL_INDEX( p_surface, instance->data.materials.size() );
	instance->data.materials[p_surface]=p_material;
}


void VisualServerRaster::instance_set_transform(RID p_instance, const Transform& p_transform) {
	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	if (p_transform==instance->data.transform) // must improve somehow
		return;

	instance->data.transform=p_transform;
	if (instance->base_type==INSTANCE_LIGHT)
		instance->data.transform.orthonormalize();
	_instance_queue_update(instance);

}

Transform VisualServerRaster::instance_get_transform(RID p_instance) const {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, Transform() );

	return instance->data.transform;

}

void VisualServerRaster::instance_set_exterior( RID p_instance, bool p_enabled ) {
	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	ERR_EXPLAIN("Portals can't be assigned to be exterior");

	ERR_FAIL_COND( instance->base_type == INSTANCE_PORTAL );
	if (instance->exterior==p_enabled)
		return;
	instance->exterior=p_enabled;
	_instance_queue_update( instance );


}

bool VisualServerRaster::instance_is_exterior( RID p_instance) const {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, false );

	return instance->exterior;
}


void VisualServerRaster::instance_set_room( RID p_instance, RID p_room ) {
	VS_CHANGED;

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	if (instance->room && instance->RE) {

		//instance already havs a room, remove it from there

		if ( (1<<instance->base_type) & INSTANCE_GEOMETRY_MASK ) {

			instance->room->room_info->owned_geometry_instances.erase(instance->RE);

			if (!p_room.is_valid() && instance->octree_id) {
				//remove from the octree, so it's re-added with different flags
				instance->scenario->octree.erase( instance->octree_id );
				instance->octree_id=0;
				_instance_queue_update( instance,true );
			}


		} else if ( instance->base_type==INSTANCE_ROOM ) {

			instance->room->room_info->owned_room_instances.erase(instance->RE);

			for(List<Instance*>::Element *E=instance->room_info->owned_portal_instances.front();E;E=E->next()) {
				_portal_disconnect(E->get());
				_instance_queue_update( E->get(),false );
			}

		} else if ( instance->base_type==INSTANCE_PORTAL ) {

			_portal_disconnect(instance,true);
			bool ss = instance->room->room_info->owned_portal_instances.erase(instance->RE);

		} else if ( instance->base_type==INSTANCE_LIGHT ) {

			instance->room->room_info->owned_light_instances.erase(instance->RE);
		} else {

			ERR_FAIL();

		}

		instance->RE=NULL;
		instance->room=NULL;


	} else {

		if (p_room.is_valid() && instance->octree_id) {
			//remove from the octree, so it's re-added with different flags
			instance->scenario->octree.erase( instance->octree_id );
			instance->octree_id=0;
			_instance_queue_update( instance,true );
		}

	}

	if (!p_room.is_valid())
		return; // just clearning the room

	Instance *room = instance_owner.get( p_room );

	ERR_FAIL_COND( !room );
	ERR_FAIL_COND( room->base_type!=INSTANCE_ROOM );


	if (instance->base_type==INSTANCE_ROOM) {

		//perform cycle test

		Instance *parent = instance;

		while(parent) {

			ERR_EXPLAIN("Cycle in room assignment");
			ERR_FAIL_COND( parent == room );
			parent=parent->room;
		}
	}

	if ( (1<<instance->base_type) & INSTANCE_GEOMETRY_MASK ) {

		instance->RE = room->room_info->owned_geometry_instances.push_back(instance);
	} else if ( instance->base_type==INSTANCE_ROOM ) {

		instance->RE = room->room_info->owned_room_instances.push_back(instance);
		for(List<Instance*>::Element *E=instance->room_info->owned_portal_instances.front();E;E=E->next())
			_instance_queue_update( E->get(),false );


	} else if ( instance->base_type==INSTANCE_PORTAL ) {

		instance->RE = room->room_info->owned_portal_instances.push_back(instance);
	} else if ( instance->base_type==INSTANCE_LIGHT ) {

		instance->RE = room->room_info->owned_light_instances.push_back(instance);
	} else {

		ERR_FAIL();

	}

	instance->room=room;
}

RID VisualServerRaster::instance_get_room( RID p_instance ) const  {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, RID() );

	if (instance->room)
		return instance->room->self;
	else
		return RID();
}

void VisualServerRaster::instance_set_extra_visibility_margin( RID p_instance, real_t p_margin ) {

	VS_CHANGED;

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	instance->extra_margin=p_margin;
}
real_t VisualServerRaster::instance_get_extra_visibility_margin( RID p_instance ) const{


	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, 0 );

	return instance->extra_margin;
}


Vector<RID> VisualServerRaster::instances_cull_aabb(const AABB& p_aabb, RID p_scenario) const {


	Vector<RID> instances;
	Scenario *scenario=scenario_owner.get(p_scenario);
	ERR_FAIL_COND_V(!scenario,instances);

	const_cast<VisualServerRaster*>(this)->_update_instances(); // check dirty instances before culling

	int culled=0;
	Instance *cull[1024];
	culled=scenario->octree.cull_AABB(p_aabb,cull,1024);

	for (int i=0;i<culled;i++) {

		Instance *instance=cull[i];
		ERR_CONTINUE(!instance);
		instances.push_back(instance->self);
	}

	return instances;
}
Vector<RID> VisualServerRaster::instances_cull_ray(const Vector3& p_from, const Vector3& p_to, RID p_scenario) const{

	Vector<RID> instances;
	Scenario *scenario=scenario_owner.get(p_scenario);
	ERR_FAIL_COND_V(!scenario,instances);
	const_cast<VisualServerRaster*>(this)->_update_instances(); // check dirty instances before culling

	int culled=0;
	Instance *cull[1024];
	culled=scenario->octree.cull_segment(p_from,p_to*10000,cull,1024);


	for (int i=0;i<culled;i++) {

		Instance *instance=cull[i];
		ERR_CONTINUE(!instance);
		instances.push_back(instance->self);
	}

	return instances;

}
Vector<RID> VisualServerRaster::instances_cull_convex(const Vector<Plane>& p_convex,  RID p_scenario) const{

	Vector<RID> instances;
	Scenario *scenario=scenario_owner.get(p_scenario);
	ERR_FAIL_COND_V(!scenario,instances);
	const_cast<VisualServerRaster*>(this)->_update_instances(); // check dirty instances before culling

	int culled=0;
	Instance *cull[1024];


	culled=scenario->octree.cull_convex(p_convex,cull,1024);

	for (int i=0;i<culled;i++) {

		Instance *instance=cull[i];
		ERR_CONTINUE(!instance);
		instances.push_back(instance->self);
	}

	return instances;

}

void VisualServerRaster::instance_geometry_set_flag(RID p_instance,InstanceFlags p_flags,bool p_enabled) {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );
	//ERR_FAIL_COND( ! ( (1<<instance->base_type) & INSTANCE_GEOMETRY_MASK) );

	switch(p_flags) {

		case INSTANCE_FLAG_VISIBLE: {

			instance->visible=p_enabled;

		} break;
		case INSTANCE_FLAG_BILLBOARD: {

			instance->data.billboard=p_enabled;

		} break;
		case INSTANCE_FLAG_BILLBOARD_FIX_Y: {

			instance->data.billboard_y=p_enabled;

		} break;
		case INSTANCE_FLAG_CAST_SHADOW: {
			if (p_enabled == true) {
				instance->data.cast_shadows = SHADOW_CASTING_SETTING_ON;
			}
			else {
				instance->data.cast_shadows = SHADOW_CASTING_SETTING_OFF;
			}

		} break;
		case INSTANCE_FLAG_RECEIVE_SHADOWS: {

			instance->data.receive_shadows=p_enabled;

		} break;
		case INSTANCE_FLAG_DEPH_SCALE: {

			instance->data.depth_scale=p_enabled;

		} break;
		case INSTANCE_FLAG_VISIBLE_IN_ALL_ROOMS: {

			instance->visible_in_all_rooms=p_enabled;

		} break;

	}

}

bool VisualServerRaster::instance_geometry_get_flag(RID p_instance,InstanceFlags p_flags) const{

	const Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, false );
	//ERR_FAIL_COND_V( ! ( (1<<instance->base_type) & INSTANCE_GEOMETRY_MASK), false );

	switch(p_flags) {

		case INSTANCE_FLAG_VISIBLE: {

			return instance->visible;

		} break;
		case INSTANCE_FLAG_BILLBOARD: {

			return instance->data.billboard;

		} break;
		case INSTANCE_FLAG_BILLBOARD_FIX_Y: {

			return instance->data.billboard_y;

		} break;
		case INSTANCE_FLAG_CAST_SHADOW: {
			if(instance->data.cast_shadows == SHADOW_CASTING_SETTING_OFF) {
				return false;
			}
			else {
				return true;
			}

		} break;
		case INSTANCE_FLAG_RECEIVE_SHADOWS: {

			return instance->data.receive_shadows;

		} break;
		case INSTANCE_FLAG_DEPH_SCALE: {

			return instance->data.depth_scale;

		} break;
		case INSTANCE_FLAG_VISIBLE_IN_ALL_ROOMS: {

			return instance->visible_in_all_rooms;

		} break;

	}

	return false;
}

void VisualServerRaster::instance_geometry_set_cast_shadows_setting(RID p_instance, VS::ShadowCastingSetting p_shadow_casting_setting) {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	instance->data.cast_shadows = p_shadow_casting_setting;
}

VS::ShadowCastingSetting VisualServerRaster::instance_geometry_get_cast_shadows_setting(RID p_instance) const{

	const Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, SHADOW_CASTING_SETTING_OFF );

	return instance->data.cast_shadows;
}


void VisualServerRaster::instance_geometry_set_material_override(RID p_instance, RID p_material) {

	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );
	instance->data.material_override=p_material;

}

RID VisualServerRaster::instance_geometry_get_material_override(RID p_instance) const{

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance, RID() );
	return instance->data.material_override;

}

void VisualServerRaster::instance_geometry_set_draw_range(RID p_instance,float p_min,float p_max){

	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	instance->draw_range_begin=p_min;
	instance->draw_range_end=p_max;

}

float VisualServerRaster::instance_geometry_get_draw_range_min(RID p_instance) const{

	const Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance,0 );

	return instance->draw_range_begin;


}

float VisualServerRaster::instance_geometry_get_draw_range_max(RID p_instance) const{

	const Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance,0 );

	return instance->draw_range_end;


}


void VisualServerRaster::instance_geometry_set_baked_light(RID p_instance,RID p_baked_light) {

	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );


	if (instance->baked_light) {


		instance->baked_light->baked_light_info->owned_instances.erase(instance->BLE);
		instance->BLE=NULL;
		instance->baked_light=NULL;
		instance->data.baked_light=NULL;
		instance->data.baked_light_octree_xform=NULL;

	}

	if (!p_baked_light.is_valid())
		return;
	Instance *bl_instance = instance_owner.get( p_baked_light );
	ERR_FAIL_COND( !bl_instance );
	ERR_FAIL_COND( bl_instance->base_type!=INSTANCE_BAKED_LIGHT );

	instance->baked_light=bl_instance;
	instance->BLE=bl_instance->baked_light_info->owned_instances.push_back(instance);
	instance->data.baked_light=&bl_instance->baked_light_info->baked_light->data;
	instance->data.baked_light_octree_xform=&bl_instance->baked_light_info->affine_inverse;

}

RID VisualServerRaster::instance_geometry_get_baked_light(RID p_instance) const{

	const Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance,RID() );
	if (instance->baked_light)
		return instance->baked_light->self;
	return RID();

}


void VisualServerRaster::instance_geometry_set_baked_light_sampler(RID p_instance,RID p_baked_light_sampler) {

	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	if (instance->sampled_light) {
		instance->sampled_light->baked_light_sampler_info->owned_instances.erase(instance);
		instance->data.sampled_light;
	}

	if(p_baked_light_sampler.is_valid()) {
		Instance *sampler_instance = instance_owner.get( p_baked_light_sampler );
		ERR_FAIL_COND( !sampler_instance );
		ERR_FAIL_COND( sampler_instance->base_type!=INSTANCE_BAKED_LIGHT_SAMPLER );
		instance->sampled_light=sampler_instance;
		instance->sampled_light->baked_light_sampler_info->owned_instances.insert(instance);
	} else {
		instance->sampled_light=NULL;
	}

	instance->data.sampled_light;

}

RID VisualServerRaster::instance_geometry_get_baked_light_sampler(RID p_instance) const {

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance,RID() );

	if (instance->sampled_light)
		return instance->sampled_light->self;
	else
		return RID();

}


void VisualServerRaster::instance_geometry_set_baked_light_texture_index(RID p_instance,int p_tex_id){

	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	instance->data.baked_lightmap_id=p_tex_id;


}
int VisualServerRaster::instance_geometry_get_baked_light_texture_index(RID p_instance) const{

	const Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance,0 );

	return instance->data.baked_lightmap_id;

}


void VisualServerRaster::_update_instance(Instance *p_instance) {

	p_instance->version++;

	if (p_instance->base_type == INSTANCE_LIGHT) {

		rasterizer->light_instance_set_transform( p_instance->light_info->instance, p_instance->data.transform );

	}


	if (p_instance->aabb.has_no_surface())
		return;


	if (p_instance->base_type == INSTANCE_PARTICLES) {

		rasterizer->particles_instance_set_transform( p_instance->particles_info->instance, p_instance->data.transform );
	}


	if ((1<<p_instance->base_type)&INSTANCE_GEOMETRY_MASK) {

		//make sure lights are updated
		InstanceSet::Element *E=p_instance->lights.front();
		while(E) {

			E->get()->version++;
			E=E->next();
		}

	} else if (p_instance->base_type == INSTANCE_ROOM) {

		p_instance->room_info->affine_inverse=p_instance->data.transform.affine_inverse();
	} else if (p_instance->base_type == INSTANCE_BAKED_LIGHT) {

		Transform scale;
		scale.basis.scale(p_instance->baked_light_info->baked_light->octree_aabb.size);
		scale.origin=p_instance->baked_light_info->baked_light->octree_aabb.pos;
		//print_line("scale: "+scale);
		p_instance->baked_light_info->affine_inverse=(p_instance->data.transform*scale).affine_inverse();
	}



	p_instance->data.mirror = p_instance->data.transform.basis.determinant() < 0.0;

	AABB new_aabb;

	if (p_instance->base_type==INSTANCE_PORTAL) {

		//portals need to be transformed in a special way, so they don't become too wide if they have scale..
		Transform portal_xform = p_instance->data.transform;
		portal_xform.basis.set_axis(2,portal_xform.basis.get_axis(2).normalized());

		p_instance->portal_info->plane_cache=Plane( p_instance->data.transform.origin, portal_xform.basis.get_axis(2));
		int point_count=p_instance->portal_info->portal->shape.size();
		p_instance->portal_info->transformed_point_cache.resize(point_count);

		AABB portal_aabb;

		for(int i=0;i<point_count;i++) {

			Point2 src = p_instance->portal_info->portal->shape[i];
			Vector3 point = portal_xform.xform(Vector3(src.x,src.y,0));
			p_instance->portal_info->transformed_point_cache[i]=point;
			if (i==0)
				portal_aabb.pos=point;
			else
				portal_aabb.expand_to(point);
		}

		portal_aabb.grow_by(p_instance->portal_info->portal->connect_range);

		new_aabb = portal_aabb;

	} else {

		new_aabb = p_instance->data.transform.xform(p_instance->aabb);
	}


	for(InstanceSet::Element *E=p_instance->lights.front();E;E=E->next()) {
		Instance *light = E->get();
		light->version++;
	}


	p_instance->transformed_aabb=new_aabb;

	if (!p_instance->scenario) {


		return;
	}



	if (p_instance->octree_id==0) {

		uint32_t base_type = 1<<p_instance->base_type;
		uint32_t pairable_mask=0;
		bool pairable=false;

		if (p_instance->base_type == INSTANCE_LIGHT) {

			pairable_mask=p_instance->light_info->enabled?INSTANCE_GEOMETRY_MASK:0;
			pairable=true;
		}

		if (p_instance->base_type == INSTANCE_PORTAL) {

			pairable_mask=(1<<INSTANCE_PORTAL);
			pairable=true;
		}

		if (p_instance->base_type == INSTANCE_BAKED_LIGHT_SAMPLER) {

			pairable_mask=(1<<INSTANCE_BAKED_LIGHT);
			pairable=true;
		}


		if (!p_instance->room && (1<<p_instance->base_type)&INSTANCE_GEOMETRY_MASK) {

			base_type|=INSTANCE_ROOMLESS_MASK;
		}

		if (p_instance->base_type == INSTANCE_ROOM) {

			pairable_mask=INSTANCE_ROOMLESS_MASK;
			pairable=true;
		}


		// not inside octree
		p_instance->octree_id = p_instance->scenario->octree.create(p_instance,new_aabb,0,pairable,base_type,pairable_mask);

	} else {

		/*
		if (new_aabb==p_instance->data.transformed_aabb)
			return;
		*/

		p_instance->scenario->octree.move(p_instance->octree_id,new_aabb);
	}

	if (p_instance->base_type==INSTANCE_PORTAL) {

		_portal_attempt_connect(p_instance);
	}

	if (!p_instance->room && (1<<p_instance->base_type)&INSTANCE_GEOMETRY_MASK) {

		_instance_validate_autorooms(p_instance);
	}

	if (p_instance->base_type == INSTANCE_ROOM) {

		for(Set<Instance*>::Element *E=p_instance->room_info->owned_autoroom_geometry.front();E;E=E->next())
			_instance_validate_autorooms(E->get());
	}


}

void VisualServerRaster::_update_instance_aabb(Instance *p_instance) {

	AABB new_aabb;

	ERR_FAIL_COND(p_instance->base_type!=INSTANCE_NONE && !p_instance->base_rid.is_valid());

	switch(p_instance->base_type) {
		case VisualServer::INSTANCE_NONE: {

			// do nothing
		} break;
		case VisualServer::INSTANCE_MESH: {

			new_aabb = rasterizer->mesh_get_aabb(p_instance->base_rid,p_instance->data.skeleton);

		} break;
		case VisualServer::INSTANCE_MULTIMESH: {

			new_aabb = rasterizer->multimesh_get_aabb(p_instance->base_rid);

		} break;
		case VisualServer::INSTANCE_IMMEDIATE: {

			new_aabb = rasterizer->immediate_get_aabb(p_instance->base_rid);


		} break;
		case VisualServer::INSTANCE_PARTICLES: {

			new_aabb = rasterizer->particles_get_aabb(p_instance->base_rid);


		} break;
		case VisualServer::INSTANCE_LIGHT: {

			new_aabb = rasterizer->light_get_aabb(p_instance->base_rid);

		} break;
		case VisualServer::INSTANCE_ROOM: {

			Room *room = room_owner.get( p_instance->base_rid );
			ERR_FAIL_COND(!room);
			new_aabb=room->bounds.get_aabb();

		} break;
		case VisualServer::INSTANCE_PORTAL: {

			Portal *portal = portal_owner.get( p_instance->base_rid );
			ERR_FAIL_COND(!portal);
			for (int i=0;i<portal->shape.size();i++) {

				Vector3 point( portal->shape[i].x, portal->shape[i].y, 0 );
				if (i==0) {

					new_aabb.pos=point;
					new_aabb.size.z=0.01; // make it not flat for octree
				} else {

					new_aabb.expand_to(point);
				}
			}

		} break;
		case VisualServer::INSTANCE_BAKED_LIGHT: {

			BakedLight *baked_light = baked_light_owner.get( p_instance->base_rid );
			ERR_FAIL_COND(!baked_light);
			new_aabb=baked_light->octree_aabb;

		} break;
		case VisualServer::INSTANCE_BAKED_LIGHT_SAMPLER: {

			BakedLightSampler *baked_light_sampler = baked_light_sampler_owner.get( p_instance->base_rid );
			ERR_FAIL_COND(!baked_light_sampler);
			float radius = baked_light_sampler->params[VS::BAKED_LIGHT_SAMPLER_RADIUS];

			new_aabb=AABB(Vector3(-radius,-radius,-radius),Vector3(radius*2,radius*2,radius*2));

		} break;

		default: {}
	}

	if (p_instance->extra_margin)
		new_aabb.grow_by(p_instance->extra_margin);

	p_instance->aabb=new_aabb;

}

void VisualServerRaster::_update_instances() {

	while(instance_update_list) {

		Instance *instance=instance_update_list;

		instance_update_list=instance_update_list->update_next;

		if (instance->update_aabb)
			_update_instance_aabb(instance);

		if (instance->update_materials) {
			if (instance->base_type==INSTANCE_MESH) {
				instance->data.materials.resize(rasterizer->mesh_get_surface_count(instance->base_rid));
			}
		}

		_update_instance(instance);

		instance->update=false;
		instance->update_aabb=false;
		instance->update_materials=false;
		instance->update_next=0;
	}
}

void VisualServerRaster::instance_light_set_enabled(RID p_instance,bool p_enabled) {

	VS_CHANGED;
	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );
	ERR_FAIL_COND( instance->base_type!=INSTANCE_LIGHT );

	if (p_enabled==instance->light_info->enabled)
		return;

	instance->light_info->enabled=p_enabled;
	if (light_get_type(instance->base_rid)!=VS::LIGHT_DIRECTIONAL && instance->octree_id && instance->scenario)
		instance->scenario->octree.set_pairable(instance->octree_id,p_enabled,1<<INSTANCE_LIGHT,p_enabled?INSTANCE_GEOMETRY_MASK:0);

	//_instance_queue_update( instance , true );

}

bool VisualServerRaster::instance_light_is_enabled(RID p_instance) const {

	const Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND_V( !instance,false );
	ERR_FAIL_COND_V( instance->base_type!=INSTANCE_LIGHT,false );

	return instance->light_info->enabled;
}

/****** CANVAS *********/
RID VisualServerRaster::canvas_create() {

	Canvas * canvas = memnew( Canvas );
	ERR_FAIL_COND_V(!canvas,RID());
	RID rid = canvas_owner.make_rid( canvas );

	return rid;
}


void VisualServerRaster::canvas_set_item_mirroring(RID p_canvas,RID p_item,const Point2& p_mirroring) {

	Canvas * canvas = canvas_owner.get(p_canvas);
	ERR_FAIL_COND(!canvas);
	CanvasItem *canvas_item = canvas_item_owner.get(p_item);
	ERR_FAIL_COND(!canvas_item);

	int idx = canvas->find_item(canvas_item);
	ERR_FAIL_COND(idx==-1);
	canvas->child_items[idx].mirror=p_mirroring;

}

Point2 VisualServerRaster::canvas_get_item_mirroring(RID p_canvas,RID p_item) const {

	Canvas * canvas = canvas_owner.get(p_canvas);
	ERR_FAIL_COND_V(!canvas,Point2());
	CanvasItem *canvas_item = memnew( CanvasItem );
	ERR_FAIL_COND_V(!canvas_item,Point2());

	int idx = canvas->find_item(canvas_item);
	ERR_FAIL_COND_V(idx==-1,Point2());
	return canvas->child_items[idx].mirror;
}

void VisualServerRaster::canvas_set_modulate(RID p_canvas,const Color& p_color) {

	Canvas * canvas = canvas_owner.get(p_canvas);
	ERR_FAIL_COND(!canvas);
	canvas->modulate=p_color;
}



RID VisualServerRaster::canvas_item_create() {

	CanvasItem *canvas_item = memnew( CanvasItem );
	ERR_FAIL_COND_V(!canvas_item,RID());

	return canvas_item_owner.make_rid( canvas_item );
}

void VisualServerRaster::canvas_item_set_parent(RID p_item,RID p_parent) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	if (canvas_item->parent.is_valid()) {

		if (canvas_owner.owns(canvas_item->parent)) {

			Canvas *canvas = canvas_owner.get(canvas_item->parent);
			canvas->erase_item(canvas_item);
		} else if (canvas_item_owner.owns(canvas_item->parent)) {

			CanvasItem *item_owner = canvas_item_owner.get(canvas_item->parent);
			item_owner->child_items.erase(canvas_item);
		}

		canvas_item->parent;
	}


	if (p_parent.is_valid()) {
		if (canvas_owner.owns(p_parent)) {

			Canvas *canvas = canvas_owner.get(p_parent);
			Canvas::ChildItem ci;
			ci.item=canvas_item;
			canvas->child_items.push_back(ci);
		} else if (canvas_item_owner.owns(p_parent)) {

			CanvasItem *item_owner = canvas_item_owner.get(p_parent);
			item_owner->child_items.push_back(canvas_item);

		} else {

			ERR_EXPLAIN("Invalid parent");
			ERR_FAIL();
		}


	}

	canvas_item->parent=p_parent;


}

RID VisualServerRaster::canvas_item_get_parent(RID p_canvas_item) const {

	CanvasItem *canvas_item = canvas_item_owner.get( p_canvas_item );
	ERR_FAIL_COND_V(!canvas_item,RID());

	return canvas_item->parent;
}

void VisualServerRaster::canvas_item_set_visible(RID p_item,bool p_visible) {

	VS_CHANGED;

	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	canvas_item->visible=p_visible;
}


bool VisualServerRaster::canvas_item_is_visible(RID p_item) const {

	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND_V(!canvas_item,RID());

	return canvas_item->visible;

}

void VisualServerRaster::canvas_item_set_light_mask(RID p_canvas_item,int p_mask) {

	VS_CHANGED;

	CanvasItem *canvas_item = canvas_item_owner.get( p_canvas_item );
	ERR_FAIL_COND(!canvas_item);

	if (canvas_item->light_mask==p_mask)
		return;
	VS_CHANGED;

	canvas_item->light_mask=p_mask;

}


void VisualServerRaster::canvas_item_set_blend_mode(RID p_canvas_item,MaterialBlendMode p_blend) {

	VS_CHANGED;

	CanvasItem *canvas_item = canvas_item_owner.get( p_canvas_item );
	ERR_FAIL_COND(!canvas_item);

	if (canvas_item->blend_mode==p_blend)
		return;
	VS_CHANGED;

	canvas_item->blend_mode=p_blend;

}

void VisualServerRaster::canvas_item_attach_viewport(RID p_canvas_item, RID p_viewport) {

	CanvasItem *canvas_item = canvas_item_owner.get( p_canvas_item );
	ERR_FAIL_COND(!canvas_item);

	VS_CHANGED;

	canvas_item->viewport=p_viewport;

}


/*
void VisualServerRaster::canvas_item_set_rect(RID p_item, const Rect2& p_rect) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	canvas_item->rect=p_rect;
}*/

void VisualServerRaster::canvas_item_set_clip(RID p_item, bool p_clip) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	canvas_item->clip=p_clip;
}

void VisualServerRaster::canvas_item_set_distance_field_mode(RID p_item, bool p_distance_field) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	canvas_item->distance_field=p_distance_field;
}


void VisualServerRaster::canvas_item_set_transform(RID p_item, const Matrix32& p_transform) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	canvas_item->xform=p_transform;

}


void VisualServerRaster::canvas_item_set_custom_rect(RID p_item, bool p_custom_rect,const Rect2& p_rect) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	canvas_item->custom_rect=p_custom_rect;
	if (p_custom_rect)
		canvas_item->rect=p_rect;

}

void VisualServerRaster::canvas_item_set_opacity(RID p_item, float p_opacity) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);
	canvas_item->opacity=p_opacity;

}
float VisualServerRaster::canvas_item_get_opacity(RID p_item, float p_opacity) const {

	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND_V(!canvas_item,-1);
	return canvas_item->opacity;

}

void VisualServerRaster::canvas_item_set_on_top(RID p_item, bool p_on_top) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);
	canvas_item->ontop=p_on_top;

}

bool VisualServerRaster::canvas_item_is_on_top(RID p_item) const{
	const CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND_V(!canvas_item,false);
	return canvas_item->ontop;

}


void VisualServerRaster::canvas_item_set_self_opacity(RID p_item, float p_self_opacity) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);
	canvas_item->self_opacity=p_self_opacity;

}
float VisualServerRaster::canvas_item_get_self_opacity(RID p_item, float p_self_opacity) const {

	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND_V(!canvas_item,-1);
	return canvas_item->self_opacity;

}


void VisualServerRaster::canvas_item_add_line(RID p_item, const Point2& p_from, const Point2& p_to,const Color& p_color,float p_width,bool p_antialiased) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	CanvasItem::CommandLine * line = memnew( CanvasItem::CommandLine );
	ERR_FAIL_COND(!line);
	line->color=p_color;
	line->from=p_from;
	line->to=p_to;
	line->width=p_width;
	line->antialiased=p_antialiased;
	canvas_item->rect_dirty=true;


	canvas_item->commands.push_back(line);
}

void VisualServerRaster::canvas_item_add_rect(RID p_item, const Rect2& p_rect, const Color& p_color) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	CanvasItem::CommandRect * rect = memnew( CanvasItem::CommandRect );
	ERR_FAIL_COND(!rect);
	rect->modulate=p_color;
	rect->rect=p_rect;
	canvas_item->rect_dirty=true;

	canvas_item->commands.push_back(rect);
}

void VisualServerRaster::canvas_item_add_circle(RID p_item, const Point2& p_pos, float p_radius,const Color& p_color) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	CanvasItem::CommandCircle * circle = memnew( CanvasItem::CommandCircle );
	ERR_FAIL_COND(!circle);
	circle->color=p_color;
	circle->pos=p_pos;
	circle->radius=p_radius;

	canvas_item->commands.push_back(circle);

}

void VisualServerRaster::canvas_item_add_texture_rect(RID p_item, const Rect2& p_rect, RID p_texture,bool p_tile,const Color& p_modulate,bool p_transpose) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	CanvasItem::CommandRect * rect = memnew( CanvasItem::CommandRect );
	ERR_FAIL_COND(!rect);
	rect->modulate=p_modulate;
	rect->rect=p_rect;
	rect->flags=0;
	if (p_tile) {
		rect->flags|=Rasterizer::CANVAS_RECT_TILE;
		rect->flags|=Rasterizer::CANVAS_RECT_REGION;
		rect->source=Rect2(0,0,p_rect.size.width,p_rect.size.height);
	}

	if (p_rect.size.x<0) {

		rect->flags|=Rasterizer::CANVAS_RECT_FLIP_H;
		rect->rect.size.x = -rect->rect.size.x;
	}
	if (p_rect.size.y<0) {

		rect->flags|=Rasterizer::CANVAS_RECT_FLIP_V;
		rect->rect.size.y = -rect->rect.size.y;
	}
	if (p_transpose) {
		rect->flags|=Rasterizer::CANVAS_RECT_TRANSPOSE;
		SWAP(rect->rect.size.x, rect->rect.size.y);
	}
	rect->texture=p_texture;
	canvas_item->rect_dirty=true;
	canvas_item->commands.push_back(rect);
}

void VisualServerRaster::canvas_item_add_texture_rect_region(RID p_item, const Rect2& p_rect, RID p_texture,const Rect2& p_src_rect,const Color& p_modulate,bool p_transpose)  {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	CanvasItem::CommandRect * rect = memnew( CanvasItem::CommandRect );
	ERR_FAIL_COND(!rect);
	rect->modulate=p_modulate;
	rect->rect=p_rect;
	rect->texture=p_texture;
	rect->source=p_src_rect;
	rect->flags=Rasterizer::CANVAS_RECT_REGION;

	if (p_rect.size.x<0) {

		rect->flags|=Rasterizer::CANVAS_RECT_FLIP_H;
		rect->rect.size.x = -rect->rect.size.x;
	}
	if (p_rect.size.y<0) {

		rect->flags|=Rasterizer::CANVAS_RECT_FLIP_V;
		rect->rect.size.y = -rect->rect.size.y;
	}
	if (p_transpose) {
		rect->flags|=Rasterizer::CANVAS_RECT_TRANSPOSE;
		SWAP(rect->rect.size.x, rect->rect.size.y);
	}

	canvas_item->rect_dirty=true;

	canvas_item->commands.push_back(rect);

}

void VisualServerRaster::canvas_item_add_style_box(RID p_item, const Rect2& p_rect, const Rect2& p_source, RID p_texture, const Vector2& p_topleft, const Vector2& p_bottomright, bool p_draw_center,const Color& p_modulate) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	CanvasItem::CommandStyle * style = memnew( CanvasItem::CommandStyle );
	ERR_FAIL_COND(!style);
	style->texture=p_texture;
	style->rect=p_rect;
	style->source=p_source;
	style->draw_center=p_draw_center;
	style->color=p_modulate;
	style->margin[MARGIN_LEFT]=p_topleft.x;
	style->margin[MARGIN_TOP]=p_topleft.y;
	style->margin[MARGIN_RIGHT]=p_bottomright.x;
	style->margin[MARGIN_BOTTOM]=p_bottomright.y;
	canvas_item->rect_dirty=true;

	canvas_item->commands.push_back(style);
}
void VisualServerRaster::canvas_item_add_primitive(RID p_item,const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs, RID p_texture,float p_width) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	CanvasItem::CommandPrimitive * prim = memnew( CanvasItem::CommandPrimitive );
	ERR_FAIL_COND(!prim);
	prim->texture=p_texture;
	prim->points=p_points;
	prim->uvs=p_uvs;
	prim->colors=p_colors;
	prim->width=p_width;
	canvas_item->rect_dirty=true;

	canvas_item->commands.push_back(prim);
}

void VisualServerRaster::canvas_item_add_polygon(RID p_item, const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs, RID p_texture) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);
#ifdef DEBUG_ENABLED
	int pointcount = p_points.size();
	ERR_FAIL_COND(pointcount<3);
	int color_size=p_colors.size();
	int uv_size=p_uvs.size();
	ERR_FAIL_COND(color_size!=0 && color_size!=1 && color_size!=pointcount);
	ERR_FAIL_COND(uv_size!=0 && (uv_size!=pointcount || !p_texture.is_valid()));
#endif
	Vector<int> indices = Geometry::triangulate_polygon(p_points);

	if (indices.empty()) {

		ERR_EXPLAIN("Bad Polygon!");
		ERR_FAIL_V();
	}

	CanvasItem::CommandPolygon * polygon = memnew( CanvasItem::CommandPolygon );
	ERR_FAIL_COND(!polygon);
	polygon->texture=p_texture;
	polygon->points=p_points;
	polygon->uvs=p_uvs;
	polygon->colors=p_colors;
	polygon->indices=indices;
	polygon->count=indices.size();
	canvas_item->rect_dirty=true;

	canvas_item->commands.push_back(polygon);

}

void VisualServerRaster::canvas_item_add_triangle_array_ptr(RID p_item, int p_count, const int* p_indices, const Point2* p_points, const Color* p_colors,const Point2* p_uvs, RID p_texture) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	ERR_FAIL_COND(p_count <= 0);

	ERR_FAIL_COND(p_points == NULL);

	CanvasItem::CommandPolygonPtr * polygon = memnew( CanvasItem::CommandPolygonPtr );
	ERR_FAIL_COND(!polygon);
	polygon->texture=p_texture;
	polygon->points=p_points;
	polygon->uvs=p_uvs;
	polygon->colors=p_colors;
	polygon->indices=p_indices;
	polygon->count = p_count * 3;
	canvas_item->rect_dirty=true;

	canvas_item->commands.push_back(polygon);
};

void VisualServerRaster::canvas_item_add_triangle_array(RID p_item, const Vector<int>& p_indices, const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs, RID p_texture, int p_count) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	int ps = p_points.size();
	ERR_FAIL_COND(!p_colors.empty() && p_colors.size()!=ps && p_colors.size()!=1);
	ERR_FAIL_COND(!p_uvs.empty() && p_uvs.size()!=ps);

	Vector<int> indices = p_indices;

	int count = p_count * 3;

	if (indices.empty()) {

		ERR_FAIL_COND( ps % 3 != 0 );
		if (p_count == -1)
			count = ps;
	} else {

		ERR_FAIL_COND( indices.size() % 3 != 0 );
		if (p_count == -1)
			count = indices.size();
	}

	CanvasItem::CommandPolygon * polygon = memnew( CanvasItem::CommandPolygon );
	ERR_FAIL_COND(!polygon);
	polygon->texture=p_texture;
	polygon->points=p_points;
	polygon->uvs=p_uvs;
	polygon->colors=p_colors;
	polygon->indices=indices;
	polygon->count = count;
	canvas_item->rect_dirty=true;

	canvas_item->commands.push_back(polygon);
}


void VisualServerRaster::canvas_item_add_set_transform(RID p_item,const Matrix32& p_transform) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	CanvasItem::CommandTransform * tr = memnew( CanvasItem::CommandTransform );
	ERR_FAIL_COND(!tr);
	tr->xform=p_transform;

	canvas_item->commands.push_back(tr);

}

void VisualServerRaster::canvas_item_add_set_blend_mode(RID p_item, MaterialBlendMode p_blend) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	CanvasItem::CommandBlendMode * bm = memnew( CanvasItem::CommandBlendMode );
	ERR_FAIL_COND(!bm);
	bm->blend_mode = p_blend;

	canvas_item->commands.push_back(bm);
};

void VisualServerRaster::canvas_item_set_z(RID p_item, int p_z) {

	ERR_FAIL_COND(p_z<CANVAS_ITEM_Z_MIN || p_z>CANVAS_ITEM_Z_MAX);
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);
	canvas_item->z=p_z;

}

void VisualServerRaster::canvas_item_set_z_as_relative_to_parent(RID p_item, bool p_enable) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);
	canvas_item->z_relative=p_enable;

}

void VisualServerRaster::canvas_item_set_copy_to_backbuffer(RID p_item, bool p_enable, const Rect2& p_rect) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);
	if (bool(canvas_item->copy_back_buffer!=NULL) !=p_enable) {
		if (p_enable) {
			canvas_item->copy_back_buffer = memnew( Rasterizer::CanvasItem::CopyBackBuffer );
		} else {
			memdelete(canvas_item->copy_back_buffer);
			canvas_item->copy_back_buffer=NULL;
		}
	}

	if (p_enable) {
		canvas_item->copy_back_buffer->rect=p_rect;
		canvas_item->copy_back_buffer->full=p_rect==Rect2();
	}

}

void VisualServerRaster::canvas_item_set_use_parent_material(RID p_item, bool p_enable) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);
	canvas_item->use_parent_material=p_enable;

}

void VisualServerRaster::canvas_item_set_material(RID p_item, RID p_material) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	if (canvas_item->material)
		canvas_item->material->owners.erase(canvas_item);

	canvas_item->material=NULL;

	if (canvas_item_material_owner.owns(p_material)) {
		canvas_item->material=canvas_item_material_owner.get(p_material);
		canvas_item->material->owners.insert(canvas_item);
	}
}

void VisualServerRaster::canvas_item_set_sort_children_by_y(RID p_item, bool p_enable) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);
	canvas_item->sort_y=p_enable;
}


void VisualServerRaster::canvas_item_add_clip_ignore(RID p_item, bool p_ignore) {

	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	CanvasItem::CommandClipIgnore * ci = memnew( CanvasItem::CommandClipIgnore);
	ERR_FAIL_COND(!ci);
	ci->ignore=p_ignore;

	canvas_item->commands.push_back(ci);

}

void VisualServerRaster::canvas_item_clear(RID p_item) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);


	canvas_item->clear();

}

void VisualServerRaster::canvas_item_raise(RID p_item) {
	VS_CHANGED;
	CanvasItem *canvas_item = canvas_item_owner.get( p_item );
	ERR_FAIL_COND(!canvas_item);

	if (canvas_item->parent.is_valid()) {

		if (canvas_owner.owns(canvas_item->parent)) {

			Canvas *canvas = canvas_owner.get(canvas_item->parent);
			int idx = canvas->find_item(canvas_item);
			ERR_FAIL_COND(idx<0);
			Canvas::ChildItem ci = canvas->child_items[idx];
			canvas->child_items.remove(idx);
			canvas->child_items.push_back(ci);

		} else if (canvas_item_owner.owns(canvas_item->parent)) {

			CanvasItem *item_owner = canvas_item_owner.get(canvas_item->parent);
			int idx = item_owner->child_items.find(canvas_item);
			ERR_FAIL_COND(idx<0);
			item_owner->child_items.remove(idx);
			item_owner->child_items.push_back(canvas_item);

		}
	}

}

/***** CANVAS LIGHT *******/

RID VisualServerRaster::canvas_light_create() {

	Rasterizer::CanvasLight *clight = memnew( Rasterizer::CanvasLight );
	return canvas_light_owner.make_rid(clight);
}

void VisualServerRaster::canvas_light_attach_to_canvas(RID p_light,RID p_canvas){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	if (clight->canvas.is_valid()) {

		Canvas *canvas = canvas_owner.get(clight->canvas);
		canvas->lights.erase(clight);
	}

	if (!canvas_owner.owns(p_canvas))
		p_canvas;
	clight->canvas=p_canvas;

	if (clight->canvas.is_valid()) {

		Canvas *canvas = canvas_owner.get(clight->canvas);
		canvas->lights.insert(clight);
	}



}
void VisualServerRaster::canvas_light_set_enabled(RID p_light, bool p_enabled){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->enabled=p_enabled;

}
void VisualServerRaster::canvas_light_set_transform(RID p_light, const Matrix32& p_transform){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->xform=p_transform;

}
void VisualServerRaster::canvas_light_set_scale(RID p_light, float p_scale) {

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->scale=p_scale;

}


void VisualServerRaster::canvas_light_set_texture(RID p_light, RID p_texture){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->texture=p_texture;

}
void VisualServerRaster::canvas_light_set_texture_offset(RID p_light, const Vector2& p_offset){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->texture_offset=p_offset;

}
void VisualServerRaster::canvas_light_set_color(RID p_light, const Color& p_color){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->color=p_color;


}
void VisualServerRaster::canvas_light_set_height(RID p_light, float p_height){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->height=p_height;

}

void VisualServerRaster::canvas_light_set_energy(RID p_light, float p_energy){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->energy=p_energy;

}

void VisualServerRaster::canvas_light_set_z_range(RID p_light, int p_min_z,int p_max_z){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->z_min=p_min_z;
	clight->z_max=p_max_z;

}

void VisualServerRaster::canvas_light_set_layer_range(RID p_light, int p_min_layer,int p_max_layer) {

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->layer_min=p_min_layer;
	clight->layer_max=p_max_layer;

}

void VisualServerRaster::canvas_light_set_item_mask(RID p_light, int p_mask){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->item_mask=p_mask;

}

void VisualServerRaster::canvas_light_set_item_shadow_mask(RID p_light, int p_mask){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->item_shadow_mask=p_mask;

}


void VisualServerRaster::canvas_light_set_mode(RID p_light, CanvasLightMode p_mode) {


	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->mode=p_mode;

}
void VisualServerRaster::canvas_light_set_shadow_enabled(RID p_light, bool p_enabled){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	if (clight->shadow_buffer.is_valid()==p_enabled)
		return;
	if (p_enabled) {
		clight->shadow_buffer=rasterizer->canvas_light_shadow_buffer_create(clight->shadow_buffer_size);
	} else {
		rasterizer->free(clight->shadow_buffer);
		clight->shadow_buffer;

	}

}

void VisualServerRaster::canvas_light_set_shadow_buffer_size(RID p_light, int p_size){

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	ERR_FAIL_COND(p_size<32 || p_size>16384);

	clight->shadow_buffer_size=nearest_power_of_2(p_size);


	if (clight->shadow_buffer.is_valid()) {
		rasterizer->free(clight->shadow_buffer);
		clight->shadow_buffer=rasterizer->canvas_light_shadow_buffer_create(clight->shadow_buffer_size);
	}

}

void VisualServerRaster::canvas_light_set_shadow_esm_multiplier(RID p_light, float p_multiplier) {

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->shadow_esm_mult=p_multiplier;

}

void VisualServerRaster::canvas_light_set_shadow_color(RID p_light, const Color& p_color) {

	Rasterizer::CanvasLight *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);
	clight->shadow_color=p_color;

}


/****** CANVAS LIGHT OCCLUDER ******/

RID VisualServerRaster::canvas_light_occluder_create() {

	Rasterizer::CanvasLightOccluderInstance *occluder = memnew( Rasterizer::CanvasLightOccluderInstance );

	return canvas_light_occluder_owner.make_rid( occluder );

}

void VisualServerRaster::canvas_light_occluder_attach_to_canvas(RID p_occluder,RID p_canvas) {

	Rasterizer::CanvasLightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!occluder);

	if (occluder->canvas.is_valid()) {

		Canvas *canvas = canvas_owner.get(occluder->canvas);
		canvas->occluders.erase(occluder);
	}

	if (!canvas_owner.owns(p_canvas))
		p_canvas;

	occluder->canvas=p_canvas;

	if (occluder->canvas.is_valid()) {

		Canvas *canvas = canvas_owner.get(occluder->canvas);
		canvas->occluders.insert(occluder);
	}
}

void VisualServerRaster::canvas_light_occluder_set_enabled(RID p_occluder,bool p_enabled){

	Rasterizer::CanvasLightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!occluder);

	occluder->enabled=p_enabled;

}

void VisualServerRaster::canvas_light_occluder_set_polygon(RID p_occluder,RID p_polygon) {

	Rasterizer::CanvasLightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!occluder);

	if (occluder->polygon.is_valid()) {
		CanvasLightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get(p_polygon);
		if (occluder_poly) {
			occluder_poly->owners.erase(occluder);
		}
	}

	occluder->polygon=p_polygon;
	occluder->polygon_buffer;

	if (occluder->polygon.is_valid()) {
		CanvasLightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get(p_polygon);
		if (!occluder_poly)
			occluder->polygon;
		ERR_FAIL_COND(!occluder_poly);
		occluder_poly->owners.insert(occluder);
		occluder->polygon_buffer=occluder_poly->occluder;
		occluder->aabb_cache=occluder_poly->aabb;
		occluder->cull_cache=occluder_poly->cull_mode;
	}

}




void VisualServerRaster::canvas_light_occluder_set_transform(RID p_occluder,const Matrix32& p_xform) {

	Rasterizer::CanvasLightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!occluder);

	occluder->xform=p_xform;

}

void VisualServerRaster::canvas_light_occluder_set_light_mask(RID p_occluder,int p_mask) {

	Rasterizer::CanvasLightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!occluder);

	occluder->light_mask=p_mask;

}


RID VisualServerRaster::canvas_occluder_polygon_create() {

	CanvasLightOccluderPolygon * occluder_poly = memnew( CanvasLightOccluderPolygon );
	occluder_poly->occluder=rasterizer->canvas_light_occluder_create();
	return canvas_light_occluder_polygon_owner.make_rid(occluder_poly);

}

void VisualServerRaster::canvas_occluder_polygon_set_shape(RID p_occluder_polygon, const PoolVector<Vector2>& p_shape, bool p_close){

	if (p_shape.size()<3) {
		canvas_occluder_polygon_set_shape_as_lines(p_occluder_polygon,p_shape);
		return;
	}

	PoolVector<Vector2> lines;
	int lc = p_shape.size()*2;

	lines.resize(lc-(p_close?0:2));
	{
		PoolVector<Vector2>::Write w = lines.write();
		PoolVector<Vector2>::Read r = p_shape.read();

		int max=lc/2;
		if (!p_close) {
			max--;
		}
		for(int i=0;i<max;i++) {

			Vector2 a = r[i];
			Vector2 b = r[(i+1)%(lc/2)];
			w[i*2+0]=a;
			w[i*2+1]=b;
		}

	}

	canvas_occluder_polygon_set_shape_as_lines(p_occluder_polygon,lines);
}

void VisualServerRaster::canvas_occluder_polygon_set_shape_as_lines(RID p_occluder_polygon,const PoolVector<Vector2>& p_shape) {

	CanvasLightOccluderPolygon * occluder_poly = canvas_light_occluder_polygon_owner.get(p_occluder_polygon);
	ERR_FAIL_COND(!occluder_poly);
	ERR_FAIL_COND(p_shape.size()&1);

	int lc = p_shape.size();
	occluder_poly->aabb=Rect2();
	{
		PoolVector<Vector2>::Read r = p_shape.read();
		for(int i=0;i<lc;i++) {
			if (i==0)
				occluder_poly->aabb.pos=r[i];
			else
				occluder_poly->aabb.expand_to(r[i]);
		}
	}

	rasterizer->canvas_light_occluder_set_polylines(occluder_poly->occluder,p_shape);
	for( Set<Rasterizer::CanvasLightOccluderInstance*>::Element *E=occluder_poly->owners.front();E;E=E->next()) {
		E->get()->aabb_cache=occluder_poly->aabb;
	}
}

void VisualServerRaster::canvas_occluder_polygon_set_cull_mode(RID p_occluder_polygon,CanvasOccluderPolygonCullMode p_mode) {

	CanvasLightOccluderPolygon * occluder_poly = canvas_light_occluder_polygon_owner.get(p_occluder_polygon);
	ERR_FAIL_COND(!occluder_poly);
	occluder_poly->cull_mode=p_mode;
	for( Set<Rasterizer::CanvasLightOccluderInstance*>::Element *E=occluder_poly->owners.front();E;E=E->next()) {
		E->get()->cull_cache=p_mode;
	}

}

RID VisualServerRaster::canvas_item_material_create() {

	Rasterizer::ShaderMaterial *material = memnew( Rasterizer::ShaderMaterial );
	return canvas_item_material_owner.make_rid(material);

}

void VisualServerRaster::canvas_item_material_set_shader(RID p_material, RID p_shader){

	VS_CHANGED;
	Rasterizer::ShaderMaterial *material = canvas_item_material_owner.get( p_material );
	ERR_FAIL_COND(!material);
	material->shader=p_shader;

}
void VisualServerRaster::canvas_item_material_set_shader_param(RID p_material, const StringName& p_param, const Variant& p_value){

	VS_CHANGED;
	Rasterizer::ShaderMaterial *material = canvas_item_material_owner.get( p_material );
	ERR_FAIL_COND(!material);
	if (p_value.get_type()==Variant::NIL)
		material->shader_param.erase(p_param);
	else
		material->shader_param[p_param]=p_value;


}
Variant VisualServerRaster::canvas_item_material_get_shader_param(RID p_material, const StringName& p_param) const{
	Rasterizer::ShaderMaterial *material = canvas_item_material_owner.get( p_material );
	ERR_FAIL_COND_V(!material,Variant());
	if (!material->shader_param.has(p_param)) {
		ERR_FAIL_COND_V(!material->shader.is_valid(),Variant());
		return rasterizer->shader_get_default_param(material->shader,p_param);
	}

	return material->shader_param[p_param];
}

void VisualServerRaster::canvas_item_material_set_shading_mode(RID p_material, CanvasItemShadingMode p_mode) {

	VS_CHANGED;
	Rasterizer::ShaderMaterial *material = canvas_item_material_owner.get( p_material );
	ERR_FAIL_COND(!material);
	material->shading_mode=p_mode;

}


/******** CANVAS *********/


void VisualServerRaster::cursor_set_rotation(float p_rotation, int p_cursor) {
	VS_CHANGED;
	ERR_FAIL_INDEX(p_cursor, MAX_CURSORS);

	cursors[p_cursor].rot = p_rotation;
};

void VisualServerRaster::cursor_set_texture(RID p_texture, const Point2 &p_center_offset, int p_cursor, const Rect2 &p_region) {
	VS_CHANGED;
	ERR_FAIL_INDEX(p_cursor, MAX_CURSORS);

	cursors[p_cursor].texture = p_texture;
	cursors[p_cursor].center = p_center_offset;
	cursors[p_cursor].region = p_region;
};

void VisualServerRaster::cursor_set_visible(bool p_visible, int p_cursor) {
	VS_CHANGED;
	ERR_FAIL_INDEX(p_cursor, MAX_CURSORS);

	cursors[p_cursor].visible = p_visible;
};

void VisualServerRaster::cursor_set_pos(const Point2& p_pos, int p_cursor) {

	ERR_FAIL_INDEX(p_cursor, MAX_CURSORS);
	if (cursors[p_cursor].pos==p_pos)
		return;
	VS_CHANGED;
	cursors[p_cursor].pos = p_pos;
};


void VisualServerRaster::black_bars_set_margins(int p_left, int p_top, int p_right, int p_bottom) {

	black_margin[MARGIN_LEFT]=p_left;
	black_margin[MARGIN_TOP]=p_top;
	black_margin[MARGIN_RIGHT]=p_right;
	black_margin[MARGIN_BOTTOM]=p_bottom;
}

void VisualServerRaster::black_bars_set_images(RID p_left, RID p_top, RID p_right, RID p_bottom) {

	black_image[MARGIN_LEFT]=p_left;
	black_image[MARGIN_TOP]=p_top;
	black_image[MARGIN_RIGHT]=p_right;
	black_image[MARGIN_BOTTOM]=p_bottom;
}

void VisualServerRaster::_free_attached_instances(RID p_rid,bool p_free_scenario) {

	Map< RID, Set<RID> >::Element * E = instance_dependency_map.find( p_rid );

	if (E) {
		// has instances
		while( E->get().size() ) {
			// erase all attached instances
			if (p_free_scenario)
				instance_set_scenario( E->get().front()->get(), RID() );
			else
				instance_set_base( E->get().front()->get(), RID() );

		}
	}

	instance_dependency_map.erase(p_rid);

}

void VisualServerRaster::custom_shade_model_set_shader(int p_model, RID p_shader) {

	VS_CHANGED;
	//rasterizer->custom_shade_model_set_shader(p_model,p_shader);
}

RID VisualServerRaster::custom_shade_model_get_shader(int p_model) const {

	//return rasterizer->custom_shade_model_get_shader(p_model);
	return RID();

}
void VisualServerRaster::custom_shade_model_set_name(int p_model, const String& p_name) {

	//rasterizer->custom_shade_model_set_name(p_model,p_name);

}
String VisualServerRaster::custom_shade_model_get_name(int p_model) const {

	//return rasterizer->custom_shade_model_get_name(p_model);
	return "";
}
void VisualServerRaster::custom_shade_model_set_param_info(int p_model, const List<PropertyInfo>& p_info) {

	VS_CHANGED;
	//rasterizer->custom_shade_model_set_param_info(p_model,p_info);
}
void VisualServerRaster::custom_shade_model_get_param_info(int p_model, List<PropertyInfo>* p_info) const {

	//rasterizer->custom_shade_model_get_param_info(p_model,p_info);
}

void VisualServerRaster::free( RID p_rid ) {

	VS_CHANGED;

	if (rasterizer->is_texture(p_rid) || rasterizer->is_material(p_rid) ||  rasterizer->is_shader(p_rid) || rasterizer->is_environment(p_rid)) {

		rasterizer->free(p_rid);
	} else if (rasterizer->is_skeleton(p_rid)) {

		Map< RID, Set<Instance*> >::Element *E=skeleton_dependency_map.find(p_rid);

		if (E) {
			//detach skeletons
			for (Set<Instance*>::Element *F=E->get().front();F;F=F->next()) {

				F->get()->data.skeleton;
			}
			skeleton_dependency_map.erase(E);
		}

		rasterizer->free(p_rid);
	} else if (rasterizer->is_mesh(p_rid) || rasterizer->is_multimesh(p_rid) || rasterizer->is_light(p_rid) || rasterizer->is_particles(p_rid) || rasterizer->is_immediate(p_rid)) {
		//delete the resource

		_free_attached_instances(p_rid);
		rasterizer->free(p_rid);
	} else if (room_owner.owns(p_rid)) {

		_free_attached_instances(p_rid);
		Room *room = room_owner.get(p_rid);
		ERR_FAIL_COND(!room);
		room_owner.free(p_rid);
		memdelete(room);


	} else if (portal_owner.owns(p_rid)) {

		_free_attached_instances(p_rid);

		Portal *portal = portal_owner.get(p_rid);
		ERR_FAIL_COND(!portal);
		portal_owner.free(p_rid);
		memdelete(portal);

	} else if (baked_light_owner.owns(p_rid)) {

		_free_attached_instances(p_rid);

		BakedLight *baked_light = baked_light_owner.get(p_rid);
		ERR_FAIL_COND(!baked_light);
		if (baked_light->data.octree_texture.is_valid())
			rasterizer->free(baked_light->data.octree_texture);
		baked_light_owner.free(p_rid);
		memdelete(baked_light);

	} else if (baked_light_sampler_owner.owns(p_rid)) {

		_free_attached_instances(p_rid);

		BakedLightSampler *baked_light_sampler = baked_light_sampler_owner.get(p_rid);
		ERR_FAIL_COND(!baked_light_sampler);
		/*
		if (baked_light->data.octree_texture.is_valid())
			rasterizer->free(baked_light->data.octree_texture);
		*/
		baked_light_sampler_owner.free(p_rid);
		memdelete(baked_light_sampler);

	} else if (camera_owner.owns(p_rid)) {
		// delete te camera

		Camera *camera = camera_owner.get(p_rid);
		ERR_FAIL_COND(!camera);

		camera_owner.free( p_rid );
		memdelete(camera);

	} else if (viewport_owner.owns(p_rid)) {
		// delete the viewport

		Viewport *viewport = viewport_owner.get( p_rid );
		ERR_FAIL_COND(!viewport);

		//Viewport *parent=NULL;

		rasterizer->free(viewport->viewport_data);
		if (viewport->render_target.is_valid()) {

			rasterizer->free(viewport->render_target);
		}

		if (viewport->update_list.in_list())
			viewport_update_list.remove(&viewport->update_list);
		if (screen_viewports.has(p_rid))
			screen_viewports.erase(p_rid);

		while(viewport->canvas_map.size()) {

			Canvas *c = viewport->canvas_map.front()->get().canvas;
			c->viewports.erase(p_rid);

			viewport->canvas_map.erase(viewport->canvas_map.front());
		}


		viewport_owner.free(p_rid);
		memdelete(viewport);

	} else if (instance_owner.owns(p_rid)) {
		// delete the instance

		_update_instances(); // be sure

		Instance *instance = instance_owner.get(p_rid);
		ERR_FAIL_COND(!instance);

		instance_set_room(p_rid,RID());
		instance_set_scenario(p_rid,RID());
		instance_geometry_set_baked_light(p_rid,RID());
		instance_geometry_set_baked_light_sampler(p_rid,RID());
		instance_set_base(p_rid,RID());

		if (instance->data.skeleton.is_valid())
			instance_attach_skeleton(p_rid,RID());

		instance_owner.free(p_rid);
		memdelete(instance);

	} else if (canvas_owner.owns(p_rid)) {

		Canvas *canvas = canvas_owner.get(p_rid);
		ERR_FAIL_COND(!canvas);

		while(canvas->viewports.size()) {

			Viewport *vp = viewport_owner.get(canvas->viewports.front()->get());
			ERR_FAIL_COND(!vp);

			Map<RID,Viewport::CanvasData>::Element *E=vp->canvas_map.find(p_rid);
			ERR_FAIL_COND(!E);
			vp->canvas_map.erase(p_rid);

			canvas->viewports.erase( canvas->viewports.front() );
		}

		for (int i=0;i<canvas->child_items.size();i++) {

			canvas->child_items[i].item->parent;
		}

		for (Set<Rasterizer::CanvasLight*>::Element *E=canvas->lights.front();E;E=E->next()) {

			E->get()->canvas;
		}

		for (Set<Rasterizer::CanvasLightOccluderInstance*>::Element *E=canvas->occluders.front();E;E=E->next()) {

			E->get()->canvas;
		}

		canvas_owner.free( p_rid );

		memdelete( canvas );

	} else if (canvas_item_owner.owns(p_rid)) {

		CanvasItem *canvas_item = canvas_item_owner.get(p_rid);
		ERR_FAIL_COND(!canvas_item);

		if (canvas_item->parent.is_valid()) {

			if (canvas_owner.owns(canvas_item->parent)) {

				Canvas *canvas = canvas_owner.get(canvas_item->parent);
				canvas->erase_item(canvas_item);
			} else if (canvas_item_owner.owns(canvas_item->parent)) {

				CanvasItem *item_owner = canvas_item_owner.get(canvas_item->parent);
				item_owner->child_items.erase(canvas_item);

			}
		}

		for (int i=0;i<canvas_item->child_items.size();i++) {

			canvas_item->child_items[i]->parent;
		}

		if (canvas_item->material) {
			canvas_item->material->owners.erase(canvas_item);
		}

		canvas_item_owner.free( p_rid );

		memdelete( canvas_item );

	} else if (canvas_item_material_owner.owns(p_rid)) {

		Rasterizer::ShaderMaterial *material = canvas_item_material_owner.get(p_rid);
		ERR_FAIL_COND(!material);
		for(Set<Rasterizer::CanvasItem*>::Element *E=material->owners.front();E;E=E->next()) {

			E->get()->material=NULL;
		}

		canvas_item_material_owner.free(p_rid);
		memdelete(material);

	} else if (canvas_light_owner.owns(p_rid)) {

		Rasterizer::CanvasLight *canvas_light = canvas_light_owner.get(p_rid);
		ERR_FAIL_COND(!canvas_light);

		if (canvas_light->canvas.is_valid()) {
			Canvas* canvas = canvas_owner.get(canvas_light->canvas);
			if (canvas)
				canvas->lights.erase(canvas_light);
		}

		if (canvas_light->shadow_buffer.is_valid())
			rasterizer->free(canvas_light->shadow_buffer);

		canvas_light_owner.free( p_rid );
		memdelete( canvas_light );

	} else if (canvas_light_occluder_owner.owns(p_rid)) {

		Rasterizer::CanvasLightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_rid);
		ERR_FAIL_COND(!occluder);

		if (occluder->polygon.is_valid()) {

			CanvasLightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get(occluder->polygon);
			if (occluder_poly) {
				occluder_poly->owners.erase(occluder);
			}

		}

		if (occluder->canvas.is_valid() && canvas_owner.owns(occluder->canvas)) {

			Canvas *canvas = canvas_owner.get(occluder->canvas);
			canvas->occluders.erase(occluder);

		}

		canvas_light_occluder_owner.free( p_rid );
		memdelete(occluder);

	} else if (canvas_light_occluder_polygon_owner.owns(p_rid)) {

		CanvasLightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get(p_rid);
		ERR_FAIL_COND(!occluder_poly);
		rasterizer->free(occluder_poly->occluder);

		while(occluder_poly->owners.size()) {

			occluder_poly->owners.front()->get()->polygon;
			occluder_poly->owners.erase( occluder_poly->owners.front() );
		}

		canvas_light_occluder_polygon_owner.free( p_rid );
		memdelete(occluder_poly);

	} else if (scenario_owner.owns(p_rid)) {

		Scenario *scenario=scenario_owner.get(p_rid);
		ERR_FAIL_COND(!scenario);

		_update_instances(); // be sure
		_free_attached_instances(p_rid,true);

		//rasterizer->free( scenario->environment );
		scenario_owner.free(p_rid);
		memdelete(scenario);

	} else {

		ERR_FAIL();
	}

}



void VisualServerRaster::_instance_draw(Instance *p_instance) {

	if (p_instance->light_cache_dirty) {
		int l=0;
		//add positional lights
		InstanceSet::Element *LE=p_instance->lights.front();
		p_instance->data.light_instances.resize(p_instance->lights.size());
		while(LE) {

			p_instance->data.light_instances[l++]=LE->get()->light_info->instance;
			LE=LE->next();
		}
		p_instance->light_cache_dirty=false;
	}


	switch(p_instance->base_type) {

		case INSTANCE_MESH: {
			rasterizer->add_mesh(p_instance->base_rid, &p_instance->data);
		} break;
		case INSTANCE_MULTIMESH: {
			rasterizer->add_multimesh(p_instance->base_rid, &p_instance->data);
		} break;
		case INSTANCE_IMMEDIATE: {
			rasterizer->add_immediate(p_instance->base_rid, &p_instance->data);
		} break;
		case INSTANCE_PARTICLES: {
			rasterizer->add_particles(p_instance->particles_info->instance, &p_instance->data);
		} break;
		default: {};
	}
}


Vector<Vector3> VisualServerRaster::_camera_generate_endpoints(Instance *p_light,Camera *p_camera,float p_range_min, float p_range_max) {

	// setup a camera matrix for that range!
	CameraMatrix camera_matrix;

	switch(p_camera->type) {

		case Camera::ORTHOGONAL: {

			camera_matrix.set_orthogonal(p_camera->size,viewport_rect.width / (float)viewport_rect.height,p_range_min,p_range_max,p_camera->vaspect);
		} break;
		case Camera::PERSPECTIVE: {

			camera_matrix.set_perspective(
				p_camera->fov,
				viewport_rect.width / (float)viewport_rect.height,
				p_range_min,
				p_range_max,
				p_camera->vaspect
			);

		} break;
	}

	//obtain the frustum endpoints

	Vector<Vector3> endpoints;
	endpoints.resize(8);
	bool res = camera_matrix.get_endpoints(p_camera->transform,&endpoints[0]);
	ERR_FAIL_COND_V(!res,Vector<Vector3>());

	return endpoints;
}

Vector<Plane> VisualServerRaster::_camera_generate_orthogonal_planes(Instance *p_light,Camera *p_camera,float p_range_min, float p_range_max) {

	Vector<Vector3> endpoints=_camera_generate_endpoints(p_light,p_camera,p_range_min,p_range_max); // frustum plane endpoints
	ERR_FAIL_COND_V(endpoints.empty(),Vector<Plane>());

	// obtain the light frustm ranges (given endpoints)

	Vector3 x_vec=p_light->data.transform.basis.get_axis( Vector3::AXIS_X ).normalized();
	Vector3 y_vec=p_light->data.transform.basis.get_axis( Vector3::AXIS_Y ).normalized();
	Vector3 z_vec=p_light->data.transform.basis.get_axis( Vector3::AXIS_Z ).normalized();

	float x_min,x_max;
	float y_min,y_max;
	float z_min,z_max;

	for(int j=0;j<8;j++) {

		float d_x=x_vec.dot(endpoints[j]);
		float d_y=y_vec.dot(endpoints[j]);
		float d_z=z_vec.dot(endpoints[j]);

		if (j==0 || d_x<x_min)
			x_min=d_x;
		if (j==0 || d_x>x_max)
			x_max=d_x;

		if (j==0 || d_y<y_min)
			y_min=d_y;
		if (j==0 || d_y>y_max)
			y_max=d_y;

		if (j==0 || d_z<z_min)
			z_min=d_z;
		if (j==0 || d_z>z_max)
			z_max=d_z;


	}
	//now that we now all ranges, we can proceed to make the light frustum planes, for culling octree

	Vector<Plane> light_frustum_planes;
	light_frustum_planes.resize(6);

	//right/left
	light_frustum_planes[0]=Plane( x_vec, x_max );
	light_frustum_planes[1]=Plane( -x_vec, -x_min );
	//top/bottom
	light_frustum_planes[2]=Plane( y_vec, y_max );
	light_frustum_planes[3]=Plane( -y_vec, -y_min );
	//near/far
	light_frustum_planes[4]=Plane( z_vec, z_max+1e6 );
	light_frustum_planes[5]=Plane( -z_vec, -z_min ); // z_min is ok, since casters further than far-light plane are not needed

	//TODO@ add more actual frustum planes to minimize get

	return light_frustum_planes;

}
void VisualServerRaster::_light_instance_update_pssm_shadow(Instance *p_light,Scenario *p_scenario,Camera *p_camera,const CullRange& p_cull_range) {

	int splits = rasterizer->light_instance_get_shadow_passes( p_light->light_info->instance );

	float split_weight=rasterizer->light_directional_get_shadow_param(p_light->base_rid,LIGHT_DIRECTIONAL_SHADOW_PARAM_PSSM_SPLIT_WEIGHT);


	float distances[5];
	float texsize=rasterizer->light_instance_get_shadow_size( p_light->light_info->instance );

	//float cull_min=p_cull_range.min;
	//float cull_max=p_cull_range.max;


	bool overlap = 	rasterizer->light_instance_get_pssm_shadow_overlap(p_light->light_info->instance);

	float cull_min=p_camera->znear;
	float cull_max=p_camera->zfar;
	float max_dist = rasterizer->light_directional_get_shadow_param(p_light->base_rid,VS::LIGHT_DIRECTIONAL_SHADOW_PARAM_MAX_DISTANCE);
	if (max_dist>0.0)
		cull_max=MIN(cull_max,max_dist);

	for(int i = 0; i < splits; i++) {
		float idm = i / (float)splits;
		float lg = cull_min * Math::pow(cull_max/cull_min, idm);
		float uniform = cull_min + (cull_max - cull_min) * idm;
		distances[i] = lg * split_weight + uniform * (1.0 - split_weight);

	}

	distances[0]=cull_min;
	distances[splits]=cull_max;

	for (int i=0;i<splits;i++) {

		// setup a camera matrix for that range!
		CameraMatrix camera_matrix;

		switch(p_camera->type) {

			case Camera::ORTHOGONAL: {

				camera_matrix.set_orthogonal(
					p_camera->size,
					viewport_rect.width / (float)viewport_rect.height,
					distances[(i==0 || !overlap )?i:i-1],
					distances[i+1],
					p_camera->vaspect

				);
			} break;
			case Camera::PERSPECTIVE: {


				camera_matrix.set_perspective(
					p_camera->fov,
					viewport_rect.width / (float)viewport_rect.height,
					distances[(i==0 || !overlap )?i:i-1],
					distances[i+1],
					p_camera->vaspect

				);

			} break;
		}

		//obtain the frustum endpoints

		Vector3 endpoints[8]; // frustum plane endpoints
		bool res = camera_matrix.get_endpoints(p_camera->transform,endpoints);
		ERR_CONTINUE(!res);

		// obtain the light frustm ranges (given endpoints)

		Vector3 x_vec=p_light->data.transform.basis.get_axis( Vector3::AXIS_X ).normalized();
		Vector3 y_vec=p_light->data.transform.basis.get_axis( Vector3::AXIS_Y ).normalized();
		Vector3 z_vec=p_light->data.transform.basis.get_axis( Vector3::AXIS_Z ).normalized();
		//z_vec points agsint the camera, like in default opengl

		float x_min,x_max;
		float y_min,y_max;
		float z_min,z_max;

		float x_min_cam,x_max_cam;
		float y_min_cam,y_max_cam;
		float z_min_cam,z_max_cam;


		//used for culling
		for(int j=0;j<8;j++) {

			float d_x=x_vec.dot(endpoints[j]);
			float d_y=y_vec.dot(endpoints[j]);
			float d_z=z_vec.dot(endpoints[j]);

			if (j==0 || d_x<x_min)
				x_min=d_x;
			if (j==0 || d_x>x_max)
				x_max=d_x;

			if (j==0 || d_y<y_min)
				y_min=d_y;
			if (j==0 || d_y>y_max)
				y_max=d_y;

			if (j==0 || d_z<z_min)
				z_min=d_z;
			if (j==0 || d_z>z_max)
				z_max=d_z;


		}





		{
			//camera viewport stuff
			//this trick here is what stabilizes the shadow (make potential jaggies to not move)
			//at the cost of some wasted resolution. Still the quality increase is very well worth it


			Vector3 center;

			for(int j=0;j<8;j++) {

				center+=endpoints[j];
			}
			center/=8.0;

			//center=x_vec*(x_max-x_min)*0.5 + y_vec*(y_max-y_min)*0.5 + z_vec*(z_max-z_min)*0.5;

			float radius=0;

			for(int j=0;j<8;j++) {

				float d = center.distance_to(endpoints[j]);
				if (d>radius)
					radius=d;
			}


			radius *= texsize/(texsize-2.0); //add a texel by each side, so stepified texture will always fit

			x_max_cam=x_vec.dot(center)+radius;
			x_min_cam=x_vec.dot(center)-radius;
			y_max_cam=y_vec.dot(center)+radius;
			y_min_cam=y_vec.dot(center)-radius;
			z_max_cam=z_vec.dot(center)+radius;
			z_min_cam=z_vec.dot(center)-radius;

			float unit = radius*2.0/texsize;

			x_max_cam=Math::stepify(x_max_cam,unit);
			x_min_cam=Math::stepify(x_min_cam,unit);
			y_max_cam=Math::stepify(y_max_cam,unit);
			y_min_cam=Math::stepify(y_min_cam,unit);

		}

		//now that we now all ranges, we can proceed to make the light frustum planes, for culling octree

		Vector<Plane> light_frustum_planes;
		light_frustum_planes.resize(6);

		//right/left
		light_frustum_planes[0]=Plane( x_vec, x_max );
		light_frustum_planes[1]=Plane( -x_vec, -x_min );
		//top/bottom
		light_frustum_planes[2]=Plane( y_vec, y_max );
		light_frustum_planes[3]=Plane( -y_vec, -y_min );
		//near/far
		light_frustum_planes[4]=Plane( z_vec, z_max+1e6 );
		light_frustum_planes[5]=Plane( -z_vec, -z_min ); // z_min is ok, since casters further than far-light plane are not needed

		int caster_cull_count = p_scenario->octree.cull_convex(light_frustum_planes,instance_shadow_cull_result,MAX_INSTANCE_CULL,INSTANCE_GEOMETRY_MASK);

		// a pre pass will need to be needed to determine the actual z-near to be used
		for(int j=0;j<caster_cull_count;j++) {

			float min,max;
			Instance *ins=instance_shadow_cull_result[j];
			if (!ins->visible || ins->data.cast_shadows == VS::SHADOW_CASTING_SETTING_OFF)
				continue;
			ins->transformed_aabb.project_range_in_plane(Plane(z_vec,0),min,max);

			if (max>z_max)
				z_max=max;
		}

		{
			CameraMatrix ortho_camera;
			real_t half_x = (x_max_cam-x_min_cam) * 0.5;
			real_t half_y = (y_max_cam-y_min_cam) * 0.5;


			ortho_camera.set_orthogonal( -half_x, half_x,-half_y,half_y, 0, (z_max-z_min_cam) );

			Transform ortho_transform;
			ortho_transform.basis=p_light->data.transform.basis;
			ortho_transform.origin=x_vec*(x_min_cam+half_x)+y_vec*(y_min_cam+half_y)+z_vec*z_max;

			rasterizer->light_instance_set_shadow_transform(p_light->light_info->instance, i, ortho_camera, ortho_transform,distances[i],distances[i+1] );
		}

		rasterizer->begin_shadow_map( p_light->light_info->instance, i );

		for (int j=0;j<caster_cull_count;j++) {

			Instance *instance = instance_shadow_cull_result[j];
			if (!instance->visible || instance->data.cast_shadows==VS::SHADOW_CASTING_SETTING_OFF)
				continue;
			_instance_draw(instance);
		}

		rasterizer->end_shadow_map();


	}


}


CameraMatrix _lispm_look( const Vector3 pos, const Vector3 dir, const Vector3 up) {

	Vector3 dirN;
	Vector3 upN;
	Vector3 lftN;

	lftN=dir.cross(up);
	lftN.normalize();

	upN=lftN.cross(dir);
	upN.normalize();
	dirN=dir.normalized();

	CameraMatrix cmout;
	float *output=&cmout.matrix[0][0];
	output[ 0] = lftN[0];
	output[ 1] = upN[0];
	output[ 2] = -dirN[0];
	output[ 3] = 0.0;

	output[ 4] = lftN[1];
	output[ 5] = upN[1];
	output[ 6] = -dirN[1];
	output[ 7] = 0.0;

	output[ 8] = lftN[2];
	output[ 9] = upN[2];
	output[10] = -dirN[2];
	output[11] = 0.0;

	output[12] = -lftN.dot(pos);
	output[13] = -upN.dot(pos);
	output[14] = dirN.dot(pos);
	output[15] = 1.0;

	return cmout;
}

#if 1

void VisualServerRaster::_light_instance_update_lispsm_shadow(Instance *p_light,Scenario *p_scenario,Camera *p_camera,const CullRange& p_cull_range) {

	Vector3 light_vec = -p_light->data.transform.basis.get_axis(2);
	Vector3 view_vec = -p_camera->transform.basis.get_axis(2);

	float near_dist=1;

	Vector<Plane> light_frustum_planes = _camera_generate_orthogonal_planes(p_light,p_camera,p_cull_range.min,p_cull_range.max);
	int caster_count = p_scenario->octree.cull_convex(light_frustum_planes,instance_shadow_cull_result,MAX_INSTANCE_CULL,INSTANCE_GEOMETRY_MASK);

	// this could be faster by just getting supports from the AABBs..
	// but, safer to do as the original implementation explains for now..

	Vector<Vector3> caster_pointcloud;
	caster_pointcloud.resize(caster_count*8);
	int caster_pointcloud_size=0;

	{

		//fill pointcloud
		Vector3* caster_pointcloud_ptr=&caster_pointcloud[0];

		for(int i=0;i<caster_count;i++) {

			Instance *ins = instance_shadow_cull_result[i];
			if (!ins->visible || ins->data.cast_shadows == VS::SHADOW_CASTING_SETTING_OFF)
				continue;

			for(int j=0;j<8;j++) {

				Vector3 v = ins->aabb.get_endpoint(j);
				v = ins->data.transform.xform(v);
				caster_pointcloud_ptr[caster_pointcloud_size+j]=v;
			}

			caster_pointcloud_size+=8;

		}
	}

	// now generate a pointcloud that contains the maximum bound (camera extruded by light)

	Vector<Vector3> camera_pointcloud = _camera_generate_endpoints(p_light,p_camera,p_cull_range.min,p_cull_range.max);
	int cpcsize=camera_pointcloud.size();
	camera_pointcloud.resize( cpcsize*2 );

	for(int i=0;i<cpcsize;i++) {

		camera_pointcloud[i+cpcsize]=camera_pointcloud[i]-light_vec*1000;
	}



	// Vector<Vector3> frustum_points=_camera_generate_endpoints(p_light,p_camera,p_cull_range.min,p_cull_range.max);


	//  compute the "light-space" basis, using the algorithm described in the paper
	//  note:  since bodyB is defined in eye space, all of these vectors should also be defined in eye space


	Vector3 eye = p_camera->transform.origin;
	Vector3 up = light_vec.cross(view_vec).cross(light_vec).normalized();


	CameraMatrix light_space_basis = _lispm_look(eye,light_vec,up);

	AABB light_space_aabb;


	{ //create an optimal AABB from both the camera pointcloud and the objects pointcloud
		AABB light_space_pointcloud_aabb;
		AABB light_space_camera_aabb;
		//xform pointcloud
		const Vector3* caster_pointcloud_ptr=&caster_pointcloud[0];

		for(int i=0;i<caster_pointcloud_size;i++) {

			Vector3 p = light_space_basis.xform(caster_pointcloud_ptr[i]);
			if (i==0) {
				light_space_pointcloud_aabb.pos=p;
			} else {
				light_space_pointcloud_aabb.expand_to(p);
			}
		}
		for(int i=0;i<camera_pointcloud.size();i++) {

			Vector3 p = light_space_basis.xform(camera_pointcloud[i]);
			if (i==0) {
				light_space_camera_aabb.pos=p;
			} else {
				light_space_camera_aabb.expand_to(p);
			}
		}

		light_space_aabb=light_space_pointcloud_aabb.intersection(light_space_camera_aabb);
	}

	float lvdp = light_vec.dot(view_vec);

	float sin_gamma = Math::sqrt(1.0-lvdp*lvdp);
	//use the formulas of the paper to get n (and f)
	float factor = 1.0/sin_gamma;
	float z_n = factor*near_dist; //often 1
	float d = Math::abs(light_space_aabb.size.y); //perspective transform depth //light space y extents
	float z_f = z_n + d*sin_gamma;
	float n = (z_n+Math::sqrt(z_f*z_n))/sin_gamma;
	float f = n+d;

	Vector3 pos = eye - up*(n-near_dist);

	CameraMatrix light_space_basis2 = _lispm_look(pos,light_vec,up);
	//Transform light_space_basis2;
	//light_space_basis2.set_look_at(pos,light_vec-pos,up);
	//light_space_basis2.affine_invert();

	//one possibility for a simple perspective transformation matrix
	//with the two parameters n(near) and f(far) in y direction

	CameraMatrix lisp_matrix;
	lisp_matrix.matrix[1][1]=(f+n)/(f-n);
	lisp_matrix.matrix[3][1]=-2*f*n/(f-n);
	lisp_matrix.matrix[1][3]=1;
	lisp_matrix.matrix[3][3]=0;

	CameraMatrix projection = lisp_matrix * light_space_basis2;
	//CameraMatrix projection = light_space_basis2 * lisp_matrix;


	AABB proj_space_aabb;

	{

		AABB proj_space_pointcloud_aabb;
		AABB proj_space_camera_aabb;
		//xform pointcloud
		Vector3* caster_pointcloud_ptr=&caster_pointcloud[0];
		for(int i=0;i<caster_pointcloud_size;i++) {

			Vector3 p = projection.xform(caster_pointcloud_ptr[i]);
			if (i==0) {
				proj_space_pointcloud_aabb.pos=p;
			} else {
				proj_space_pointcloud_aabb.expand_to(p);
			}
		}

		for(int i=0;i<camera_pointcloud.size();i++) {

			Vector3 p = projection.xform(camera_pointcloud[i]);
			if (i==0) {
				proj_space_camera_aabb.pos=p;
			} else {
				proj_space_camera_aabb.expand_to(p);
			}
		}

		//proj_space_aabb=proj_space_pointcloud_aabb.intersection_with(proj_space_camera_aabb);
		proj_space_aabb=proj_space_pointcloud_aabb;
	}

	projection.scale_translate_to_fit(proj_space_aabb);
	projection=projection * lisp_matrix;

	CameraMatrix scale;
	scale.make_scale(Vector3(1.0,1.0,-1.0)); // transform to left handed

	projection=scale * projection;

	rasterizer->light_instance_set_shadow_transform(p_light->light_info->instance,0, projection , light_space_basis2.inverse() );

	rasterizer->begin_shadow_map( p_light->light_info->instance,  0 );

	for(int i=0;i<caster_count;i++) {

		Instance *instance = instance_shadow_cull_result[i];

		if (!instance->visible || instance->data.cast_shadows == VS::SHADOW_CASTING_SETTING_OFF)
			continue;
		_instance_draw(instance);
	}

	rasterizer->end_shadow_map();


}

#else


void VisualServerRaster::_light_instance_update_lispsm_shadow(Instance *p_light,Scenario *p_scenario,Camera *p_camera,const CullRange& p_cull_range) {

	/* STEP 1: GENERATE LIGHT TRANSFORM */



	Vector3 light_vec = -p_light->data.transform.basis.get_axis(2);
	Vector3 view_vec = -p_camera->transform.basis.get_axis(2);
	float viewdot = Math::absf(light_vec.dot(view_vec));

	Vector3 up = light_vec.cross(view_vec).cross(light_vec).normalized();

	Transform light_transform;
	light_transform.set_look_at(Vector3(),light_vec,up);


	/* STEP 2: GENERATE WORDLSPACE PLANES AND VECTORS*/
	float range_min=0.01; //p_cull_range.min
	float range_max=20;//p_cull_range.max;

	Vector<Vector3> camera_endpoints=_camera_generate_endpoints(p_light,p_camera,range_min,range_max); // frustum plane endpoints
	ERR_FAIL_COND(camera_endpoints.empty());

	// obtain the light frustm ranges (given endpoints)


	Vector3 light_x_vec=light_transform.basis.get_axis( Vector3::AXIS_X ).normalized();
	Vector3 light_y_vec=light_transform.basis.get_axis( Vector3::AXIS_Y ).normalized();
	Vector3 light_z_vec=light_transform.basis.get_axis( Vector3::AXIS_Z ).normalized();

	Vector3 light_axis_max;
	Vector3 light_axis_min;

	for(int j=0;j<8;j++) {

		float d_x=light_x_vec.dot(camera_endpoints[j]);
		float d_y=light_y_vec.dot(camera_endpoints[j]);
		float d_z=light_z_vec.dot(camera_endpoints[j]);

		if (j==0 || d_x<light_axis_min.x)
			light_axis_min.x=d_x;
		if (j==0 || d_x>light_axis_max.x)
			light_axis_max.x=d_x;

		if (j==0 || d_y<light_axis_min.y)
			light_axis_min.y=d_y;
		if (j==0 || d_y>light_axis_max.y)
			light_axis_max.y=d_y;

		if (j==0 || d_z<light_axis_min.z)
			light_axis_min.z=d_z;
		if (j==0 || d_z>light_axis_max.z)
			light_axis_max.z=d_z;


	}

	//now that we now all ranges, we can proceed to make the light frustum planes, for culling octree

	Vector<Plane> light_cull_planes;
	light_cull_planes.resize(6);


	//right/left
	light_cull_planes[0]=Plane( light_x_vec, light_axis_max.x );
	light_cull_planes[1]=Plane( -light_x_vec, -light_axis_min.x );
	//top/bottom
	light_cull_planes[2]=Plane( light_y_vec, light_axis_max.y );
	light_cull_planes[3]=Plane( -light_y_vec, -light_axis_min.y );
	//near/far
	light_cull_planes[4]=Plane( light_z_vec, light_axis_max.z+1e6 );
	light_cull_planes[5]=Plane( -light_z_vec, -light_axis_min.z ); // z_min is ok, since casters further than far-light plane are not needed


	/* STEP 3: CULL CASTERS */

	int caster_count = p_scenario->octree.cull_convex(light_cull_planes,instance_shadow_cull_result,MAX_INSTANCE_CULL,INSTANCE_GEOMETRY_MASK);

	/* STEP 4: ADJUST FAR Z PLANE */

	float caster_max_z=1e-1;
	for(int i=0;i<caster_count;i++) {

		Instance *ins=instance_shadow_cull_result[i];
		if (!ins->visible || ins->cast_shadows==VS::SHADOW_CASTING_SETTING_OFF)
			continue;

		//@TODO optimize using support mapping
		for(int j=0;j<8;j++) {

			Vector3 v=ins->data.transform.xform(ins->aabb.get_endpoint(j));
			float d = light_z_vec.dot(v);
			if (d>caster_max_z)
				caster_max_z=d;

		}

	}

	float expand = caster_max_z-light_axis_max.z;
	if (expand<0)
		expand=0;
	light_axis_max.z=MAX(caster_max_z,light_axis_max.z);

	/* STEP 5: CREATE ORTHOGONAL PROJECTION */

	CameraMatrix light_projection;

	real_t half_x = (light_axis_max.x-light_axis_min.x) * 0.5;
	real_t half_y = (light_axis_max.y-light_axis_min.y) * 0.5;
	light_projection.set_orthogonal( -half_x, half_x,half_y, -half_y, 0, (light_axis_max.z-light_axis_min.z) );
	light_transform.origin=light_x_vec*(light_axis_min.x+half_x)+light_y_vec*(light_axis_min.y+half_y)+light_z_vec*light_axis_max.z;


	if (/*false &&*/ viewdot<0.96) {

		float lvdp = light_vec.dot(view_vec);

		float near_dist=1.0;
		float sin_gamma = Math::sqrt(1.0-lvdp*lvdp);
		//use the formulas of the paper to get n (and f)
		float factor = 1.0/sin_gamma;
		float z_n = factor*near_dist; //often 1
		float d = Math::abs(light_axis_max.y-light_axis_min.y); //perspective transform depth //light space y extents
		float z_f = z_n + d*sin_gamma;
		float n = (z_n+Math::sqrt(z_f*z_n))/sin_gamma;
		float f = n+d;

		CameraMatrix lisp_matrix;
		lisp_matrix.matrix[1][1]=(f+n)/(f-n);
		lisp_matrix.matrix[3][1]=-2*f*n/(f-n);
		lisp_matrix.matrix[1][3]=1;
		lisp_matrix.matrix[3][3]=0;

		Vector3 pos = p_camera->transform.origin - up*(n-near_dist);

		CameraMatrix world2light = _lispm_look(pos,light_vec,up);
		CameraMatrix projection = lisp_matrix * world2light;

		AABB projection_bounds;
		for(int i=0;i<camera_endpoints.size();i++) {

			Vector3 p=camera_endpoints[i];
			if (i==0)
				projection_bounds.pos=projection.xform(p);
			else
				projection_bounds.expand_to(projection.xform(p));

			projection_bounds.expand_to(projection.xform(p+light_vec*-expand));
		}

		CameraMatrix scaletrans;
		scaletrans.scale_translate_to_fit(projection_bounds);
		projection=scaletrans * lisp_matrix;

		CameraMatrix scale;
		scale.make_scale(Vector3(1.0,1.0,-1.0)); // transform to left handed

		projection=scale * projection;


		rasterizer->light_instance_set_shadow_transform(p_light->light_info->instance,0, projection, world2light.inverse(), viewdot);

	} else {
		//orthogonal
		rasterizer->light_instance_set_shadow_transform(p_light->light_info->instance,0, light_projection , light_transform, viewdot);
	}

	rasterizer->begin_shadow_map( p_light->light_info->instance,  0 );

	for(int i=0;i<caster_count;i++) {

		Instance *instance = instance_shadow_cull_result[i];

		if (!instance->visible || instance->cast_shadows==VS::SHADOW_CASTING_SETTING_OFF)
			continue;
		_instance_draw(instance);
	}

	rasterizer->end_shadow_map();

}

#endif


void VisualServerRaster::_light_instance_update_shadow(Instance *p_light,Scenario *p_scenario,Camera *p_camera,const CullRange& p_cull_range) {



	if (!rasterizer->shadow_allocate_near( p_light->light_info->instance ))
		return; // shadow could not be updated


	/* VisualServerRaster supports for many shadow techniques, using the one the rasterizer requests */

	Rasterizer::ShadowType shadow_type = rasterizer->light_instance_get_shadow_type(p_light->light_info->instance);

	switch(shadow_type) {

		case Rasterizer::SHADOW_SIMPLE: {
			/* SPOT SHADOW */


			rasterizer->begin_shadow_map( p_light->light_info->instance, 0 );

			//using this one ensures that raster deferred will have it

			float far = rasterizer->light_get_var( p_light->base_rid, VS::LIGHT_PARAM_RADIUS);

			float angle = rasterizer->light_get_var( p_light->base_rid, VS::LIGHT_PARAM_SPOT_ANGLE );

			CameraMatrix cm;
			cm.set_perspective( angle*2.0, 1.0, 0.001, far );

			Vector<Plane> planes = cm.get_projection_planes(p_light->data.transform);
			int cull_count = p_scenario->octree.cull_convex(planes,instance_shadow_cull_result,MAX_INSTANCE_CULL,INSTANCE_GEOMETRY_MASK);


			for (int i=0;i<cull_count;i++) {

				Instance *instance = instance_shadow_cull_result[i];
				if (!instance->visible || instance->data.cast_shadows == VS::SHADOW_CASTING_SETTING_OFF)
					continue;
				_instance_draw(instance);
			}

			rasterizer->end_shadow_map();

		} break;
		case Rasterizer::SHADOW_DUAL_PARABOLOID: {

			/* OMNI SHADOW */

			int passes = rasterizer->light_instance_get_shadow_passes( p_light->light_info->instance );

			if (passes==2) {

				for(int i=0;i<2;i++) {

					rasterizer->begin_shadow_map( p_light->light_info->instance, i );


					//using this one ensures that raster deferred will have it

					float radius = rasterizer->light_get_var( p_light->base_rid, VS::LIGHT_PARAM_RADIUS);

					float z =i==0?-1:1;
					Vector<Plane> planes;
					planes.resize(5);
					planes[0]=p_light->data.transform.xform(Plane(Vector3(0,0,z),radius));
					planes[1]=p_light->data.transform.xform(Plane(Vector3(1,0,z).normalized(),radius));
					planes[2]=p_light->data.transform.xform(Plane(Vector3(-1,0,z).normalized(),radius));
					planes[3]=p_light->data.transform.xform(Plane(Vector3(0,1,z).normalized(),radius));
					planes[4]=p_light->data.transform.xform(Plane(Vector3(0,-1,z).normalized(),radius));


					int cull_count = p_scenario->octree.cull_convex(planes,instance_shadow_cull_result,MAX_INSTANCE_CULL,INSTANCE_GEOMETRY_MASK);


					for (int j=0;j<cull_count;j++) {

						Instance *instance = instance_shadow_cull_result[j];
						if (!instance->visible || instance->data.cast_shadows == VS::SHADOW_CASTING_SETTING_OFF)
							continue;

						_instance_draw(instance);
					}

					rasterizer->end_shadow_map();
				}
			} else if (passes==1) {
				//one go



			}

		} break;
		case Rasterizer::SHADOW_CUBE: {

			// todo
		} break;
		case Rasterizer::SHADOW_ORTHOGONAL: {

			_light_instance_update_pssm_shadow(p_light,p_scenario,p_camera,p_cull_range);
		} break;
		case Rasterizer::SHADOW_PSSM: {

			_light_instance_update_pssm_shadow(p_light,p_scenario,p_camera,p_cull_range);
		} break;
		case Rasterizer::SHADOW_PSM: {

			_light_instance_update_lispsm_shadow(p_light,p_scenario,p_camera,p_cull_range);
		 // todo
		} break;
		default: {}
	}

}

void VisualServerRaster::_portal_disconnect(Instance *p_portal,bool p_cleanup) {

	if (p_portal->portal_info->connected) {

		//disconnect first
		p_portal->portal_info->connected->portal_info->connected=NULL;
		p_portal->portal_info->connected=NULL;

	}

	if (p_portal->room && p_portal->room->room) {

		if (p_cleanup) {

			p_portal->room->room->room_info->disconnected_child_portals.erase(p_portal);
			//p_portal->room->room->room_info->disconnected_child_portals.erase(p_portal);
		} else {
			p_portal->room->room->room_info->disconnected_child_portals.insert(p_portal);
		}
	}

}

void VisualServerRaster::_instance_validate_autorooms(Instance *p_geometry) {

	if (p_geometry->auto_rooms.size()==0)
		return;

	p_geometry->valid_auto_rooms.clear();

	int point_count = aabb_random_points.size();
	const Vector3 * src_points = &aabb_random_points[0];

	for(Set<Instance*>::Element *E=p_geometry->valid_auto_rooms.front();E;E=E->next()) {

		Instance *room = E->get();
		Vector3 *dst_points=&transformed_aabb_random_points[0];

		//generate points
		for(int i=0;i<point_count;i++) {

			dst_points[i] = room->room_info->affine_inverse.xform(p_geometry->data.transform.xform((src_points[i]*p_geometry->transformed_aabb.size)+p_geometry->transformed_aabb.pos));
		}

		int pass = room->room_info->room->bounds.get_points_inside(dst_points,point_count);

		float ratio = pass;
		if( point_count != 0 ) {
			ratio /= (float)point_count;
		}

		if (ratio>0.5) // should make some constant
			p_geometry->valid_auto_rooms.insert(room);
	}
}

void VisualServerRaster::_portal_attempt_connect(Instance *p_portal) {


	_portal_disconnect(p_portal);

	Vector3 A_norm = p_portal->data.transform.basis.get_axis(Vector3::AXIS_Z).normalized();
	Plane A_plane( p_portal->data.transform.origin, A_norm );
	float A_surface = p_portal->portal_info->portal->bounds.get_area();
	if (A_surface==0)
		return; //wtf

	Instance *found=NULL;
	Transform affine_inverse = p_portal->data.transform.affine_inverse();

	for(Set<Instance*>::Element *E=p_portal->portal_info->candidate_set.front();E;E=E->next()) {

		Instance *B = E->get();

		if (B->portal_info->connected)
			continue; // in use

		Vector3 B_norm = B->data.transform.basis.get_axis(Vector3::AXIS_Z).normalized();

		// check that they are in front of another
		float dot = A_norm.dot(-B_norm);

		if (dot<0.707) // 45 degrees, TODO unharcode this
			continue;

		// check the max distance to the other portal

		bool valid=true;

		Rect2 local_bounds;

		for(int i=0;i<B->portal_info->portal->shape.size();i++) {

			Point2 point2 = B->portal_info->portal->shape[i];

			Vector3 point = B->data.transform.xform( Vector3( point2.x, point2.y, 0 ) );

			float dist = Math::abs(A_plane.distance_to(point));

			if (
				dist>p_portal->portal_info->portal->connect_range ||
				dist>B->portal_info->portal->connect_range ) {
					valid=false;
					break;
				}


			Vector3 point_local = affine_inverse.xform(A_plane.project(point));
			point2 = Point2(point_local.x,point_local.y);

			if (i==0)
				local_bounds.pos=point2;
			else
				local_bounds.expand_to(point2);


		}

		if (!valid)
			continue;

		float B_surface = B->portal_info->portal->bounds.get_area();
		if (B_surface==0)
			continue; //wtf

		float clip_area = p_portal->portal_info->portal->bounds.clip(local_bounds).get_area();


		//check that most of the area is shared

		if ( (clip_area/A_surface) < 0.5 || (clip_area/B_surface) < 0.5) // TODO change for something else
			continue;

		found=B;
		break;



	}


	if (!found) {

		if (p_portal->room && p_portal->room->room) {

			p_portal->room->room->room_info->disconnected_child_portals.insert(p_portal);
		}

		return;
	}

	p_portal->portal_info->connected=found;
	found->portal_info->connected=p_portal;


}

void* VisualServerRaster::instance_pair(void *p_self, OctreeElementID, Instance *p_A,int, OctreeElementID, Instance *p_B,int) {

	VisualServerRaster *self = (VisualServerRaster*)p_self;
	Instance *A = p_A;
	Instance *B = p_B;

	if (A->base_type==INSTANCE_PORTAL) {

		ERR_FAIL_COND_V( B->base_type!=INSTANCE_PORTAL,NULL );

		A->portal_info->candidate_set.insert(B);
		B->portal_info->candidate_set.insert(A);

		self->_portal_attempt_connect(A);
		//attempt to conncet portal A (will go through B anyway)
		//this is a little hackish, but works fine in practice

	} else if (A->base_type==INSTANCE_BAKED_LIGHT || B->base_type==INSTANCE_BAKED_LIGHT) {

		if (B->base_type==INSTANCE_BAKED_LIGHT) {
			SWAP(A,B);
		}

		ERR_FAIL_COND_V(B->base_type!=INSTANCE_BAKED_LIGHT_SAMPLER,NULL);
		B->baked_light_sampler_info->baked_lights.insert(A);

	} else if (A->base_type==INSTANCE_ROOM || B->base_type==INSTANCE_ROOM) {

		if (B->base_type==INSTANCE_ROOM)
			SWAP(A,B);

		ERR_FAIL_COND_V(! ((1<<B->base_type)&INSTANCE_GEOMETRY_MASK ),NULL);

		B->auto_rooms.insert(A);
		A->room_info->owned_autoroom_geometry.insert(B);

		self->_instance_validate_autorooms(B);


	} else {

		if (B->base_type==INSTANCE_LIGHT) {

			SWAP(A,B);
		} else if (A->base_type!=INSTANCE_LIGHT) {
			return NULL;
		}


		A->light_info->affected.insert(B);
		B->lights.insert(A);
		B->light_cache_dirty=true;
	}

	return NULL;

}
void VisualServerRaster::instance_unpair(void *p_self, OctreeElementID, Instance *p_A,int, OctreeElementID, Instance *p_B,int,void*) {

	VisualServerRaster *self = (VisualServerRaster*)p_self;
	Instance *A = p_A;
	Instance *B = p_B;

	if (A->base_type==INSTANCE_PORTAL) {

		ERR_FAIL_COND( B->base_type!=INSTANCE_PORTAL );


		A->portal_info->candidate_set.erase(B);
		B->portal_info->candidate_set.erase(A);

		//after disconnecting them, see if they can connect again
		self->_portal_attempt_connect(A);
		self->_portal_attempt_connect(B);

	} else if (A->base_type==INSTANCE_BAKED_LIGHT || B->base_type==INSTANCE_BAKED_LIGHT) {

		if (B->base_type==INSTANCE_BAKED_LIGHT) {
			SWAP(A,B);
		}

		ERR_FAIL_COND(B->base_type!=INSTANCE_BAKED_LIGHT_SAMPLER);
		B->baked_light_sampler_info->baked_lights.erase(A);

	} else if (A->base_type==INSTANCE_ROOM || B->base_type==INSTANCE_ROOM) {

		if (B->base_type==INSTANCE_ROOM)
			SWAP(A,B);

		ERR_FAIL_COND(! ((1<<B->base_type)&INSTANCE_GEOMETRY_MASK ));

		B->auto_rooms.erase(A);
		B->valid_auto_rooms.erase(A);
		A->room_info->owned_autoroom_geometry.erase(B);

	}else {


		if (B->base_type==INSTANCE_LIGHT) {

			SWAP(A,B);
		} else if (A->base_type!=INSTANCE_LIGHT) {
			return;
		}


		A->light_info->affected.erase(B);
		B->lights.erase(A);
		B->light_cache_dirty=true;
	}
}

bool VisualServerRaster::_test_portal_cull(Camera *p_camera, Instance *p_from_portal, Instance *p_to_portal) {


	int src_point_count=p_from_portal->portal_info->transformed_point_cache.size();
	int dst_point_count=p_to_portal->portal_info->transformed_point_cache.size();

	if (src_point_count<2 || dst_point_count<2)
		return false;

	const Vector3 *src_points=&p_from_portal->portal_info->transformed_point_cache[0];
	const Vector3 *dst_points=&p_to_portal->portal_info->transformed_point_cache[0];

	bool outside=false;

	bool clockwise = !p_from_portal->portal_info->plane_cache.is_point_over(p_camera->transform.origin);

	for(int i=0;i<src_point_count;i++) {

		const Vector3& point_prev = src_points[i?(i-1):(src_point_count-1)];
		const Vector3& point = src_points[i];

		Plane p = clockwise?Plane(p_camera->transform.origin,point,point_prev):Plane(p_camera->transform.origin,point_prev,point);

		bool all_over=true;

		for(int j=0;j<dst_point_count;j++) {

			if (!p.is_point_over(dst_points[j])) {

				all_over=false;
				break;
			}

		}

		if (all_over) {
			outside=true;
			break;
		}

	}

	return !outside;

}

void VisualServerRaster::_cull_portal(Camera *p_camera, Instance *p_portal,Instance *p_from_portal) {

	ERR_FAIL_COND(!p_portal->scenario); //scenario outside

	Instance *portal = p_portal;

	if (!portal->room) {

		return; //portals need all to belong to a room, it may be unconfigured yet
	} else if (portal->last_render_pass!=render_pass) {

		return; //invalid portal, ignore
	} else if (portal->portal_info->last_visited_pass==render_pass) {

		return; //portal already visited
	} else if (portal==p_from_portal) {

		return; // came from this portal, don't even bother testing
	}

	/* TEST DISABLE DISTANCE */

	float disable_distance = p_portal->portal_info->portal->disable_distance;
	if (disable_distance) {
		//has disable distance..
		float distance = p_camera->transform.origin.distance_to(portal->data.transform.origin);
		if (disable_distance < distance) {

			return;
		}
	}

	/* TEST PORTAL NOT FACING OPTIMIZATION */


	if (p_portal->portal_info->connected) {
		//connected portal means, it must face against the camera to be seen
		if (p_portal->portal_info->plane_cache.is_point_over(p_camera->transform.origin)) { //portal facing against camera (exterior)

			return;
		}
	} else {
		//disconencted portals (go from room to parent room or exterior) must face towards the canera
		if (!p_portal->portal_info->plane_cache.is_point_over(p_camera->transform.origin)) { //portal facing against camera (exterior)

			return;
		}
	}

	if (p_from_portal && !_test_portal_cull(p_camera, p_from_portal, portal)) {
		return; // portal not visible (culled)
	}

	portal->portal_info->last_visited_pass=render_pass;

	if (portal->portal_info->connected) {

		//interior<->interior portal
		Instance *to_room = portal->portal_info->connected->room;
		if (!to_room) {
			return; //wtf.. oh well, connected to a roomless (invalid) portal
		}

		_cull_room(p_camera, to_room, portal->portal_info->connected);

	} else {
		//to exterior/to parent roomportal

		Instance *parent_room = portal->room->room;

		_cull_room(p_camera, parent_room, portal);
	}

}

void VisualServerRaster::_cull_room(Camera *p_camera, Instance *p_room,Instance *p_from_portal) {

	if (p_room==NULL) {
		//exterior
		exterior_visited=true;

		for(int i=0;i<exterior_portal_cull_count;i++) {

			_cull_portal(p_camera, exterior_portal_cull_result[i],p_from_portal);
		}

	} else {

		ERR_FAIL_COND(!p_room->scenario);

		if (p_room->last_render_pass!=render_pass)
			return; //this room is invalid

		//interior
		//first of all, validate the room
		p_room->room_info->last_visited_pass=render_pass;
		//see about going around portals
		if (!p_room->room_info->room->occlude_exterior)
			exterior_visited=true;

		for(List<Instance*>::Element * E=p_room->room_info->owned_portal_instances.front();E;E=E->next()) {

			_cull_portal(p_camera, E->get(),p_from_portal);

		}

		for(Set<Instance*>::Element * E=p_room->room_info->disconnected_child_portals.front();E;E=E->next()) {

			_cull_portal(p_camera, E->get(),p_from_portal);

		}


	}

}

void VisualServerRaster::_process_sampled_light(const Transform& p_camera,Instance *p_sampled_light,bool p_linear_colorspace) {


	BakedLightSampler *sampler_opts = p_sampled_light->baked_light_sampler_info->sampler;
	int res = sampler_opts->resolution;
	int dp_size = res*res*2;
	Color * dp_map = (Color*)alloca( sizeof(Color)*dp_size); //allocate the dual parabolloid colors
	Vector3 * dp_normals = (Vector3*)alloca( sizeof(Vector3)*dp_size); //allocate the dual parabolloid normals
	const Vector3 * dp_src_normals = p_sampled_light->baked_light_sampler_info->sampler->dp_cache.ptr();


	if (!p_sampled_light->baked_light_sampler_info->sampled_light.is_valid() || p_sampled_light->baked_light_sampler_info->resolution!=sampler_opts->resolution) {
		if (p_sampled_light->baked_light_sampler_info->sampled_light.is_valid()) {
			rasterizer->free(p_sampled_light->baked_light_sampler_info->sampled_light);
		}

		p_sampled_light->baked_light_sampler_info->resolution=sampler_opts->resolution;
		p_sampled_light->baked_light_sampler_info->sampled_light=rasterizer->sampled_light_dp_create(sampler_opts->resolution,sampler_opts->resolution*2);


	}


	zeromem(dp_map,sizeof(Color)*dp_size);
	bool valid=false;
	int samples=0;


	for(Set<Instance*>::Element *E=p_sampled_light->baked_light_sampler_info->baked_lights.front();E;E=E->next()) {

		Instance *bl = E->get();
		if (bl->baked_light_info->baked_light->sampler.size()==0)
			continue; //not usable


		Matrix3 norm_xform = bl->baked_light_info->affine_inverse.basis;//.inverse();
		for(int i=0;i<dp_size;i++) {
			dp_normals[i]=norm_xform.xform(dp_src_normals[i]).normalized();
		}

		//normals in place


		//sample octree

		float r = sampler_opts->params[VS::BAKED_LIGHT_SAMPLER_RADIUS];
		float att = sampler_opts->params[VS::BAKED_LIGHT_SAMPLER_ATTENUATION];
		float str = sampler_opts->params[VS::BAKED_LIGHT_SAMPLER_STRENGTH];
		Vector3 s = p_sampled_light->data.transform.basis.get_scale();

		r*=MAX(MAX(s.x,s.y),s.z);
		AABB sample_aabb= bl->data.transform.affine_inverse().xform(AABB(Vector3(-r,-r,-r)+p_sampled_light->data.transform.origin,Vector3(r*2,r*2,r*2)));
		//ok got octree local AABB

		PoolVector<int>::Read rp = bl->baked_light_info->baked_light->sampler.read();
		const int *rptr = rp.ptr();

		int first = rptr[1];
		int depth = rptr[2];
		bool islinear = rptr[3]&1;
		depth+=1;

		AABB aabb;
		aabb.pos.x=decode_float((const uint8_t*)&rptr[4]);
		aabb.pos.y=decode_float((const uint8_t*)&rptr[5]);
		aabb.pos.z=decode_float((const uint8_t*)&rptr[6]);
		aabb.size.x=decode_float((const uint8_t*)&rptr[7]);
		aabb.size.y=decode_float((const uint8_t*)&rptr[8]);
		aabb.size.z=decode_float((const uint8_t*)&rptr[9]);

		uint32_t *stack=(uint32_t*)alloca(depth*sizeof(uint32_t));
		int *stack_ptr=(int*)alloca(depth*sizeof(int));
		AABB *aabb_stack=(AABB*)alloca(depth*sizeof(AABB));

		stack[0]=0;
		stack_ptr[0]=first;
		aabb_stack[0]=aabb;
		Vector3 center = sample_aabb.pos + sample_aabb.size * 0.5;


		int stack_pos=0;
		Color max_col;

		//int reso = sampler_opts->params[VS::BAKED_LIGHT_SAMPLER_DETAIL_RATIO];

		int lalimit = sample_aabb.get_longest_axis_index();
		float limit = sampler_opts->params[VS::BAKED_LIGHT_SAMPLER_DETAIL_RATIO]*sample_aabb.size[lalimit];


		while(true) {


			bool leaf = (rptr[ stack_ptr[stack_pos] ]>>16)==0;

			if (aabb_stack[stack_pos].size[lalimit]<limit) {
				leaf=true;
			}


			if (leaf) {

				Vector3 from = aabb_stack[stack_pos].pos + aabb_stack[stack_pos].size * 0.5;
				Vector3 norm = (from-center).normalized();


				Color col;
				col.r = ((rptr[ stack_ptr[stack_pos] ]&0xFFFF)/256.0);
				col.g = ((rptr[ stack_ptr[stack_pos]+1 ]>>16)/256.0);
				col.b = ((rptr[ stack_ptr[stack_pos]+1 ]&0xFFFF)/256.0);


				max_col.r = MAX(max_col.r,col.r);
				max_col.g = MAX(max_col.g,col.g);
				max_col.b = MAX(max_col.b,col.b);

				if (!islinear && p_linear_colorspace) {
					col=col.to_linear();
				}

				float distance;

				if (aabb_stack[stack_pos].has_point(center)) {
					distance=0;
				} else {

					Vector3 support = aabb_stack[stack_pos].get_support(norm);
					distance = Math::absf(norm.dot(support)-norm.dot(center));

				}

				if (distance>r)
					distance=r;

				float mult = Math::pow(1.0-distance/r,att)*str;
				if (mult>0) {
					col.r*=mult;
					col.g*=mult;
					col.b*=mult;



					for(int i=0;i<dp_size;i++) {
						float mult2 = norm.dot(dp_normals[i]);
						if (mult2<0)
							mult2=0;
						Color col2(col.r*mult2,col.g*mult2,col.b*mult2,1.0);
						dp_map[i].r=MAX(dp_map[i].r,col2.r);
						dp_map[i].g=MAX(dp_map[i].g,col2.g);
						dp_map[i].b=MAX(dp_map[i].b,col2.b);
					}

				}

				samples++;
				//nothing is valid unless you hit a leaf
				valid=true;
				stack_pos--;
			} else if ((stack[stack_pos]&0xFF)<8) {

				int i = stack[stack_pos]&0xFF;
				int base = (stack[stack_pos]>>8);

				if (!((rptr[ stack_ptr[stack_pos] ]>>16)&(1<<i))) {
					//no bit, no test
					stack[stack_pos]=(base<<8)+(i+1);
					continue;
				}

				stack[stack_pos]=((base+1)<<8)+(i+1);

				AABB child_aabb = aabb_stack[stack_pos];
				child_aabb.size*=0.5;
				if (i&1)
					child_aabb.pos.x+=child_aabb.size.x;
				if (i&2)
					child_aabb.pos.y+=child_aabb.size.y;
				if (i&4)
					child_aabb.pos.z+=child_aabb.size.z;

				if (!child_aabb.intersects(sample_aabb)) {
					continue;
				}

				if (child_aabb.encloses(sample_aabb)) {
					stack[stack_pos]=(base<<8)|8; //don't test the rest
				}

				stack_pos++;
				ERR_FAIL_COND(stack_pos>=depth);

				stack[stack_pos]=0;
				stack_ptr[stack_pos]=rptr[ stack_ptr[stack_pos-1]+2+base ];
				aabb_stack[stack_pos]=child_aabb;
			} else {
				stack_pos--;
				if (stack_pos<0)
					break;
			}
		}


	}

	//print_line("samples "+itos(samples) );

	if (valid) {

		for(int i=0;i<res;i++) {
			//average seams to avoid aliasing
			{
				//top
				int ofs1 = i;
				int ofs2 = dp_size-res+i;
				Color avg(
					(dp_map[ofs1].r+dp_map[ofs2].r)*0.5,
					(dp_map[ofs1].g+dp_map[ofs2].g)*0.5,
					(dp_map[ofs1].b+dp_map[ofs2].b)*0.5,
					1.0
				);
				dp_map[ofs1]=avg;
				dp_map[ofs2]=avg;
			}
			{
				//bottom
				int ofs1 = res*res-res+i;
				int ofs2 = res*res+i;
				Color avg(
					(dp_map[ofs1].r+dp_map[ofs2].r)*0.5,
					(dp_map[ofs1].g+dp_map[ofs2].g)*0.5,
					(dp_map[ofs1].b+dp_map[ofs2].b)*0.5,
					1.0
				);
				dp_map[ofs1]=avg;
				dp_map[ofs2]=avg;
			}
			{
				//left
				int ofs1 = i*res;
				int ofs2 = res*res+(res-i-1)*res;
				Color avg(
					(dp_map[ofs1].r+dp_map[ofs2].r)*0.5,
					(dp_map[ofs1].g+dp_map[ofs2].g)*0.5,
					(dp_map[ofs1].b+dp_map[ofs2].b)*0.5,
					1.0
				);
				dp_map[ofs1]=avg;
				dp_map[ofs2]=avg;
			}
			{
				//right
				int ofs1 = i*res+(res-1);
				int ofs2 = res*res+(res-i-1)*res+(res-1);
				Color avg(
					(dp_map[ofs1].r+dp_map[ofs2].r)*0.5,
					(dp_map[ofs1].g+dp_map[ofs2].g)*0.5,
					(dp_map[ofs1].b+dp_map[ofs2].b)*0.5,
					1.0
				);
				dp_map[ofs1]=avg;
				dp_map[ofs2]=avg;
			}

		}

		rasterizer->sampled_light_dp_update(p_sampled_light->baked_light_sampler_info->sampled_light,dp_map,1.0);
		for(Set<Instance*>::Element *F=p_sampled_light->baked_light_sampler_info->owned_instances.front();F;F=F->next()) {

			F->get()->data.sampled_light=p_sampled_light->baked_light_sampler_info->sampled_light;
		}


	} else {

		for(Set<Instance*>::Element *F=p_sampled_light->baked_light_sampler_info->owned_instances.front();F;F=F->next()) {

			F->get()->data.sampled_light; //do not use because nothing close
		}
	}




/*
		highp vec3 vtx = vertex_interp;
		vtx.z*=dual_paraboloid.y; //side to affect
		vtx.z+=0.01;
		dp_clip=vtx.z;
		highp float len=length( vtx );
		vtx=normalize(vtx);
		vtx.xy/=1.0+vtx.z;
		vtx.z = len*dual_paraboloid.x; // it's a reciprocal(len - z_near) / (z_far - z_near);
		vtx+=normalize(vtx)*0.025;
		vtx.z = vtx.z * 2.0 - 1.0; // fit to clipspace
		vertex_interp=vtx;
*/




}


void VisualServerRaster::_render_no_camera(Viewport *p_viewport,Camera *p_camera, Scenario *p_scenario) {
	RID environment;
	if (p_scenario->environment.is_valid())
		environment=p_scenario->environment;
	else
		environment=p_scenario->fallback_environment;

	rasterizer->set_camera(Transform(),CameraMatrix(),false);
	rasterizer->begin_scene(p_viewport->viewport_data,environment,p_scenario->debug);
	rasterizer->set_viewport(viewport_rect);
	rasterizer->end_scene();
}


void VisualServerRaster::_render_camera(Viewport *p_viewport,Camera *p_camera, Scenario *p_scenario) {


	render_pass++;
	uint32_t camera_layer_mask=p_camera->visible_layers;

	/* STEP 1 - SETUP CAMERA */
	CameraMatrix camera_matrix;
	bool ortho=false;

	switch(p_camera->type) {
		case Camera::ORTHOGONAL: {

			camera_matrix.set_orthogonal(
				p_camera->size,
				viewport_rect.width / (float)viewport_rect.height,
				p_camera->znear,
				p_camera->zfar,
				p_camera->vaspect

			);
			ortho=true;
		} break;
		case Camera::PERSPECTIVE: {

			camera_matrix.set_perspective(
				p_camera->fov,
				viewport_rect.width / (float)viewport_rect.height,
				p_camera->znear,
				p_camera->zfar,
				p_camera->vaspect

			);
			ortho=false;

		} break;
	}


	rasterizer->set_camera(p_camera->transform, camera_matrix,ortho);

	Vector<Plane> planes = camera_matrix.get_projection_planes(p_camera->transform);

	CullRange cull_range; // cull range is used for PSSM, and having an idea of the rendering depth
	cull_range.nearp=Plane(p_camera->transform.origin,-p_camera->transform.basis.get_axis(2).normalized());
	cull_range.z_near=camera_matrix.get_z_near();
	cull_range.z_far=camera_matrix.get_z_far();
	cull_range.min=cull_range.z_far;
	cull_range.max=cull_range.z_near;

	/* STEP 2 - CULL */
	int cull_count = p_scenario->octree.cull_convex(planes,instance_cull_result,MAX_INSTANCE_CULL);
	light_cull_count=0;
	light_samplers_culled=0;

/*	print_line("OT: "+rtos( (OS::get_singleton()->get_ticks_usec()-t)/1000.0));
	print_line("OTO: "+itos(p_scenario->octree.get_octant_count()));
	//print_line("OTE: "+itos(p_scenario->octree.get_elem_count()));
	print_line("OTP: "+itos(p_scenario->octree.get_pair_count()));
*/

	/* STEP 3 - PROCESS PORTALS, VALIDATE ROOMS */


	// compute portals

	exterior_visited=false;
	exterior_portal_cull_count=0;

	if (room_cull_enabled) {
		for(int i=0;i<cull_count;i++) {

			Instance *ins = instance_cull_result[i];
			ins->last_render_pass=render_pass;

			if (ins->base_type!=INSTANCE_PORTAL)
				continue;

			if (ins->room)
				continue;

			ERR_CONTINUE(exterior_portal_cull_count>=MAX_EXTERIOR_PORTALS);
			exterior_portal_cull_result[exterior_portal_cull_count++]=ins;

		}

		room_cull_count = p_scenario->octree.cull_point(p_camera->transform.origin,room_cull_result,MAX_ROOM_CULL,NULL,(1<<INSTANCE_ROOM)|(1<<INSTANCE_PORTAL));


		Set<Instance*> current_rooms;
		Set<Instance*> portal_rooms;
		//add to set
		for(int i=0;i<room_cull_count;i++) {

			if (room_cull_result[i]->base_type==INSTANCE_ROOM) {
				current_rooms.insert(room_cull_result[i]);
			}
			if (room_cull_result[i]->base_type==INSTANCE_PORTAL) {
				//assume inside that room if also inside the portal..
				if (room_cull_result[i]->room) {
					portal_rooms.insert(room_cull_result[i]->room);
				}

				SWAP(room_cull_result[i],room_cull_result[room_cull_count-1]);
				room_cull_count--;
				i--;
			}
		}

		//remove from set if it has a parent room or BSP doesn't contain
		for(int i=0;i<room_cull_count;i++) {
			Instance *r = room_cull_result[i];

			//check inside BSP
			Vector3 room_local_point = r->room_info->affine_inverse.xform( p_camera->transform.origin );

			if (!portal_rooms.has(r) && !r->room_info->room->bounds.point_is_inside(room_local_point)) {

				current_rooms.erase(r);
				continue;
			}

			//check parent
			while (r->room) {// has parent room

				current_rooms.erase(r);
				r=r->room;
			}

		}

		if (current_rooms.size()) {
			//camera is inside a room
			// go through rooms
			for(Set<Instance*>::Element *E=current_rooms.front();E;E=E->next()) {
				_cull_room(p_camera,E->get());
			}

		} else {
			//start from exterior
			_cull_room(p_camera,NULL);

		}
	}

	/* STEP 4 - REMOVE FURTHER CULLED OBJECTS, ADD LIGHTS */

	for(int i=0;i<cull_count;i++) {

		Instance *ins = instance_cull_result[i];

		bool keep=false;


		if ((camera_layer_mask&ins->layer_mask)==0) {

			//failure
		} else if (ins->base_type==INSTANCE_LIGHT) {

			if (light_cull_count<MAX_LIGHTS_CULLED) {
				light_cull_result[light_cull_count++]=ins;
				//rasterizer->light_instance_set_active_hint(ins->light_info->instance);
				{
					//compute distance to camera using aabb support
					Vector3 n = ins->data.transform.basis.xform_inv(cull_range.nearp.normal).normalized();
					Vector3 s = ins->data.transform.xform(ins->aabb.get_support(n));
					ins->light_info->dtc=cull_range.nearp.distance_to(s);
				}
			}

		} else if ((1<<ins->base_type)&INSTANCE_GEOMETRY_MASK && ins->visible && ins->data.cast_shadows!=VS::SHADOW_CASTING_SETTING_SHADOWS_ONLY) {


			bool discarded=false;

			if (ins->draw_range_end>0) {

				float d = cull_range.nearp.distance_to(ins->data.transform.origin);
				if (d<0)
					d=0;
				discarded=(d<ins->draw_range_begin || d>=ins->draw_range_end);


			}

			if (!discarded) {

				// test if this geometry should be visible

				if (room_cull_enabled) {


					if (ins->visible_in_all_rooms) {
						keep=true;
					} else if (ins->room) {

						if (ins->room->room_info->last_visited_pass==render_pass)
							keep=true;
					} else if (ins->auto_rooms.size()) {


						for(Set<Instance*>::Element *E=ins->auto_rooms.front();E;E=E->next()) {

							if (E->get()->room_info->last_visited_pass==render_pass) {
								keep=true;
								break;
							}
						}
					} else if(exterior_visited)
						keep=true;
				} else {

					keep=true;
				}


			}


			if (keep) {
				// update cull range
				float min,max;
				ins->transformed_aabb.project_range_in_plane(cull_range.nearp,min,max);

				if (min<cull_range.min)
					cull_range.min=min;
				if (max>cull_range.max)
					cull_range.max=max;

				if (ins->sampled_light && ins->sampled_light->baked_light_sampler_info->last_pass!=render_pass) {
					if (light_samplers_culled<MAX_LIGHT_SAMPLERS) {
						light_sampler_cull_result[light_samplers_culled++]=ins->sampled_light;
						ins->sampled_light->baked_light_sampler_info->last_pass=render_pass;
					}
				}
			}

		}

		if (!keep) {
			// remove, no reason to keep
			cull_count--;
			SWAP( instance_cull_result[i], instance_cull_result[ cull_count ] );
			i--;
			ins->last_render_pass=0; // make invalid
		} else {

			ins->last_render_pass=render_pass;
		}
	}

	if (cull_range.max > cull_range.z_far )
		cull_range.max=cull_range.z_far;
	if (cull_range.min < cull_range.z_near )
		cull_range.min=cull_range.z_near;

	/* STEP 5 - PROCESS LIGHTS */

	rasterizer->shadow_clear_near(); //clear near shadows, will be recreated

	// directional lights
	{
		List<RID>::Element *E=p_scenario->directional_lights.front();


		while(E) {

			Instance  *light = E->get().is_valid()?instance_owner.get(E->get()):NULL;

			if (light && light->light_info->enabled && rasterizer->light_has_shadow(light->base_rid)) {
				//rasterizer->light_instance_set_active_hint(light->light_info->instance);
				_light_instance_update_shadow(light,p_scenario,p_camera,cull_range);
			}

			E=E->next();
		}
	}


	//discard lights not affecting anything (useful for deferred rendering, shadowmaps, etc)

	for (int i=0;i<light_cull_count;i++) {

		Instance *ins = light_cull_result[i];

		if (light_discard_enabled) {

			//see if the light should be pre discarded because no one is seeing it
			//this test may seem expensive, but in reality, it shouldn't be
			//because of early out condition. It will only go through everything
			//if it's being discarded.

			bool valid=false;
			InstanceSet::Element *E =ins->light_info->affected.front();
			while(E) {

				if (E->get()->last_render_pass==render_pass) {

					valid=true; // early out.
					break;
				}
				E=E->next();
			}
			if (!valid) {

				light_cull_count--;
				SWAP( light_cull_result[i], light_cull_result[ light_cull_count ] );
				i--;

			}
		}

	}

	{ //this should eventually change to
		//assign shadows by distance to camera
		SortArray<Instance*,_InstanceLightsort> sorter;
		sorter.sort(light_cull_result,light_cull_count);
		for (int i=0;i<light_cull_count;i++) {

			Instance *ins = light_cull_result[i];

			if (!rasterizer->light_has_shadow(ins->base_rid) || !shadows_enabled)
				continue;

			/* for far shadows?
			if (ins->version == ins->light_info->last_version && rasterizer->light_instance_has_far_shadow(ins->light_info->instance))
				continue; // didn't change
			*/

			_light_instance_update_shadow(ins,p_scenario,p_camera,cull_range);
			ins->light_info->last_version=ins->version;
		}
	}

	/* ENVIRONMENT */

	RID environment;
	if (p_camera->env.is_valid()) //camera has more environment priority
		environment=p_camera->env;
	else if (p_scenario->environment.is_valid())
		environment=p_scenario->environment;
	else
		environment=p_scenario->fallback_environment;


	/* STEP 6 - SAMPLE BAKED LIGHT */

	bool islinear =false;
	if (environment.is_valid()) {
		islinear = rasterizer->environment_is_fx_enabled(environment,VS::ENV_FX_SRGB);
	}

	for(int i=0;i<light_samplers_culled;i++) {

		_process_sampled_light(p_camera->transform,light_sampler_cull_result[i],islinear);
	}

	/* STEP 7 - PROCESS GEOMETRY AND DRAW SCENE*/


	rasterizer->begin_scene(p_viewport->viewport_data,environment,p_scenario->debug);
	rasterizer->set_viewport(viewport_rect);

	// add lights

	{
		List<RID>::Element *E=p_scenario->directional_lights.front();


		for(;E;E=E->next()) {
			Instance  *light = E->get().is_valid()?instance_owner.get(E->get()):NULL;

			ERR_CONTINUE(!light);
			if (!light->light_info->enabled)
				continue;

			rasterizer->add_light(light->light_info->instance);
			light->light_info->last_add_pass=render_pass;
		}

		for (int i=0;i<light_cull_count;i++) {

			Instance *ins = light_cull_result[i];
			rasterizer->add_light(ins->light_info->instance);
			ins->light_info->last_add_pass=render_pass;
		}
	}
		// add geometry

	for(int i=0;i<cull_count;i++) {

		Instance *ins = instance_cull_result[i];

		ERR_CONTINUE(!((1<<ins->base_type)&INSTANCE_GEOMETRY_MASK));

		_instance_draw(ins);
	}

	rasterizer->end_scene();
}


void VisualServerRaster::_render_canvas_item_tree(CanvasItem *p_canvas_item, const Matrix32& p_transform, const Rect2& p_clip_rect, const Color& p_modulate, Rasterizer::CanvasLight *p_lights) {


	static const int z_range = CANVAS_ITEM_Z_MAX-CANVAS_ITEM_Z_MIN+1;
	Rasterizer::CanvasItem *z_list[z_range];
	Rasterizer::CanvasItem *z_last_list[z_range];

	for(int i=0;i<z_range;i++) {
		z_list[i]=NULL;
		z_last_list[i]=NULL;
	}


	_render_canvas_item(p_canvas_item,p_transform,p_clip_rect,1.0,0,z_list,z_last_list,NULL,NULL);

	for(int i=0;i<z_range;i++) {
		if (!z_list[i])
			continue;
		rasterizer->canvas_render_items(z_list[i],CANVAS_ITEM_Z_MIN+i,p_modulate,p_lights);
	}

}


void VisualServerRaster::_render_canvas_item_viewport(VisualServer* p_self,void *p_vp,const Rect2& p_rect) {

	VisualServerRaster *self=(VisualServerRaster*)(p_self);
	Viewport *vp=(Viewport*)p_vp;
	self->_draw_viewport(vp,p_rect.pos.x,p_rect.pos.y,p_rect.size.x,p_rect.size.y);
	self->rasterizer->canvas_begin();

}


void VisualServerRaster::_render_canvas_item(CanvasItem *p_canvas_item,const Matrix32& p_transform,const Rect2& p_clip_rect, float p_opacity,int p_z,Rasterizer::CanvasItem **z_list,Rasterizer::CanvasItem **z_last_list,CanvasItem *p_canvas_clip,CanvasItem *p_material_owner) {

	CanvasItem *ci = p_canvas_item;

	if (!ci->visible)
		return;

	if (p_opacity<0.007)
		return;


	Rect2 rect = ci->get_rect();
	Matrix32 xform = p_transform * ci->xform;
	Rect2 global_rect = xform.xform(rect);
	global_rect.pos+=p_clip_rect.pos;


	if (global_rect.intersects(p_clip_rect) && ci->viewport.is_valid() && viewport_owner.owns(ci->viewport)) {

		Viewport *vp = viewport_owner.get(ci->viewport);

		Point2i from = xform.get_origin() + Point2(viewport_rect.x,viewport_rect.y);
		Point2i size = rect.size;
		size.x *= xform[0].length();
		size.y *= xform[1].length();

		ci->vp_render = memnew( Rasterizer::CanvasItem::ViewportRender );
		ci->vp_render->owner=this;
		ci->vp_render->udata=vp;
		ci->vp_render->rect=Rect2(from.x,
					  from.y,
					  size.x,
					  size.y);
/*
		_draw_viewport(vp,
				from.x,
				from.y,
				size.x,
				size.y);
*/
		//rasterizer->canvas_begin();
	} else {
		ci->vp_render=NULL;
	}

	if (ci->use_parent_material && p_material_owner)
		ci->material_owner=p_material_owner;
	else {
		p_material_owner=ci;
		ci->material_owner=NULL;
	}


	float opacity = ci->opacity * p_opacity;


	int child_item_count=ci->child_items.size();
	CanvasItem **child_items=(CanvasItem**)alloca(child_item_count*sizeof(CanvasItem*));
	copymem(child_items,ci->child_items.ptr(),child_item_count*sizeof(CanvasItem*));

	if (ci->clip) {
		if (p_canvas_clip != NULL) {
			ci->final_clip_rect=p_canvas_clip->final_clip_rect.clip(global_rect);
		} else {
			ci->final_clip_rect=global_rect;
		}
		ci->final_clip_owner=ci;

	} else {
		ci->final_clip_owner=p_canvas_clip;
	}

	if (ci->sort_y) {

		SortArray<CanvasItem*,CanvasItemPtrSort> sorter;
		sorter.sort(child_items,child_item_count);
	}

	if (ci->z_relative)
		p_z=CLAMP(p_z+ci->z,CANVAS_ITEM_Z_MIN,CANVAS_ITEM_Z_MAX);
	else
		p_z=ci->z;

	for(int i=0;i<child_item_count;i++) {

		if (child_items[i]->ontop)
			continue;
		_render_canvas_item(child_items[i],xform,p_clip_rect,opacity,p_z,z_list,z_last_list,(CanvasItem*)ci->final_clip_owner,p_material_owner);
	}

	if (ci->copy_back_buffer) {

		ci->copy_back_buffer->screen_rect = xform.xform(ci->copy_back_buffer->rect).clip(p_clip_rect);
	}

	if ((!ci->commands.empty() && p_clip_rect.intersects(global_rect)) || ci->vp_render || ci->copy_back_buffer) {
		//something to draw?
		ci->final_transform=xform;
		ci->final_opacity=opacity * ci->self_opacity;
		ci->global_rect_cache=global_rect;
		ci->global_rect_cache.pos-=p_clip_rect.pos;
		ci->light_masked=false;

		int zidx = p_z-CANVAS_ITEM_Z_MIN;

		if (z_last_list[zidx]) {
			z_last_list[zidx]->next=ci;
			z_last_list[zidx]=ci;

		} else {
			z_list[zidx]=ci;
			z_last_list[zidx]=ci;
		}



		ci->next=NULL;

	}

	for(int i=0;i<child_item_count;i++) {

		if (!child_items[i]->ontop)
			continue;
		_render_canvas_item(child_items[i],xform,p_clip_rect,opacity,p_z,z_list,z_last_list,(CanvasItem*)ci->final_clip_owner,p_material_owner);
	}

}

void VisualServerRaster::_light_mask_canvas_items(int p_z,Rasterizer::CanvasItem *p_canvas_item,Rasterizer::CanvasLight *p_masked_lights) {

	if (!p_masked_lights)
		return;

	Rasterizer::CanvasItem *ci=p_canvas_item;

	while(ci) {

		Rasterizer::CanvasLight *light=p_masked_lights;
		while(light) {

			if (ci->light_mask&light->item_mask && p_z>=light->z_min && p_z<=light->z_max && ci->global_rect_cache.intersects_transformed(light->xform_cache,light->rect_cache)) {
				ci->light_masked=true;
			}

			light=light->mask_next_ptr;
		}

		ci=ci->next;
	}




}

void VisualServerRaster::_render_canvas(Canvas *p_canvas,const Matrix32 &p_transform,Rasterizer::CanvasLight *p_lights,Rasterizer::CanvasLight *p_masked_lights) {

	rasterizer->canvas_begin();

	int l = p_canvas->child_items.size();
	Canvas::ChildItem *ci=p_canvas->child_items.ptr();

	bool has_mirror=false;
	for(int i=0;i<l;i++) {
		if (ci[i].mirror.x || ci[i].mirror.y) {
			has_mirror=true;
			break;
		}
	}

	Rect2 clip_rect(viewport_rect.x,viewport_rect.y,viewport_rect.width,viewport_rect.height);
	if (!has_mirror) {

		static const int z_range = CANVAS_ITEM_Z_MAX-CANVAS_ITEM_Z_MIN+1;
		Rasterizer::CanvasItem *z_list[z_range];
		Rasterizer::CanvasItem *z_last_list[z_range];

		for(int i=0;i<z_range;i++) {
			z_list[i]=NULL;
			z_last_list[i]=NULL;
		}
		for(int i=0;i<l;i++) {
			_render_canvas_item(ci[i].item,p_transform,clip_rect,1.0,0,z_list,z_last_list,NULL,NULL);
		}

		for(int i=0;i<z_range;i++) {
			if (!z_list[i])
				continue;

			if (p_masked_lights) {
				_light_mask_canvas_items(CANVAS_ITEM_Z_MIN+i,z_list[i],p_masked_lights);
			}

			rasterizer->canvas_render_items(z_list[i],CANVAS_ITEM_Z_MIN+i,p_canvas->modulate,p_lights);
		}
	} else {

		for(int i=0;i<l;i++) {

			Canvas::ChildItem& ci=p_canvas->child_items[i];
			_render_canvas_item_tree(ci.item,p_transform,clip_rect,p_canvas->modulate,p_lights);

			//mirroring (useful for scrolling backgrounds)
			if (ci.mirror.x!=0) {

				Matrix32 xform2 = p_transform * Matrix32(0,Vector2(ci.mirror.x,0));
				_render_canvas_item_tree(ci.item,xform2,clip_rect,p_canvas->modulate,p_lights);
			}
			if (ci.mirror.y!=0) {

				Matrix32 xform2 = p_transform * Matrix32(0,Vector2(0,ci.mirror.y));
				_render_canvas_item_tree(ci.item,xform2,clip_rect,p_canvas->modulate,p_lights);
			}
			if (ci.mirror.y!=0 && ci.mirror.x!=0) {

				Matrix32 xform2 = p_transform * Matrix32(0,ci.mirror);
				_render_canvas_item_tree(ci.item,xform2,clip_rect,p_canvas->modulate,p_lights);
			}

		}
	}

}


void VisualServerRaster::_draw_viewport_camera(Viewport *p_viewport,bool p_ignore_camera) {


	Camera *camera=NULL;
	if (camera_owner.owns( p_viewport->camera ))
		camera=camera_owner.get( p_viewport->camera );
	Scenario *scenario = scenario_owner.get( p_viewport->scenario );

	_update_instances(); // check dirty instances before rendering

	if (p_ignore_camera)
		_render_no_camera(p_viewport, camera,scenario );
	else
		_render_camera(p_viewport, camera,scenario );

}

void VisualServerRaster::_draw_viewport(Viewport *p_viewport,int p_ofs_x, int p_ofs_y,int p_parent_w,int p_parent_h) {

	ViewportRect desired_rect=p_viewport->rect;
	ViewportRect old_rect = viewport_rect;
	//bool vpchanged=false;
	// convert default expanding viewports to actual size
	//if (desired_rect.x==0 && desired_rect.y==0 && desired_rect.width==0 && desired_rect.height==0) {
	if (p_parent_w != 0 && p_parent_h != 0) {

		desired_rect.width=p_parent_w;
		desired_rect.height=p_parent_h;
	}

	ERR_FAIL_COND(desired_rect.width<=0 || desired_rect.height<=0);

	desired_rect.x+=p_ofs_x;
	desired_rect.y+=p_ofs_y;


	// if the viewport is different than the actual one, change it

	if ( p_viewport->render_target.is_valid() || viewport_rect.x != desired_rect.x ||
		viewport_rect.y != desired_rect.y ||
		viewport_rect.width != desired_rect.width ||
		viewport_rect.height != desired_rect.height ) {


		viewport_rect=desired_rect;
		rasterizer->set_viewport(viewport_rect);

	}

	/* Camera should always be BEFORE any other 3D */

	bool scenario_draw_canvas_bg=false;
	int scenario_canvas_max_layer=0;

	if (!p_viewport->hide_canvas && !p_viewport->disable_environment && scenario_owner.owns(p_viewport->scenario)) {

		Scenario *scenario=scenario_owner.get(p_viewport->scenario);
		if (scenario->environment.is_valid()) {
			if (rasterizer->is_environment(scenario->environment)) {
				scenario_draw_canvas_bg=rasterizer->environment_get_background(scenario->environment)==VS::ENV_BG_CANVAS;
				scenario_canvas_max_layer=rasterizer->environment_get_background_param(scenario->environment,VS::ENV_BG_PARAM_CANVAS_MAX_LAYER);
			}
		}
	}

	bool can_draw_3d=!p_viewport->hide_scenario && camera_owner.owns(p_viewport->camera) && scenario_owner.owns(p_viewport->scenario);


	if (scenario_draw_canvas_bg) {

		rasterizer->begin_canvas_bg();
	}

	if (!scenario_draw_canvas_bg && can_draw_3d) {

		_draw_viewport_camera(p_viewport,false);

	} else if (true /*|| !p_viewport->canvas_list.empty()*/){

		//clear the viewport black because of no camera? i seriously should..
		if (p_viewport->render_target_clear_on_new_frame || p_viewport->render_target_clear) {
			if (p_viewport->transparent_bg) {
				rasterizer->clear_viewport(Color(0,0,0,0));
			}
			else {
				Color cc=clear_color;
				if (scenario_draw_canvas_bg)
					cc.a=0;
				rasterizer->clear_viewport(cc);
			}
			p_viewport->render_target_clear=false;
		}
	}

	if (!p_viewport->hide_canvas) {
		int i=0;

		Map<Viewport::CanvasKey,Viewport::CanvasData*> canvas_map;

		Rect2 clip_rect(0,0,viewport_rect.width,viewport_rect.height);
		Rasterizer::CanvasLight *lights=NULL;
		Rasterizer::CanvasLight *lights_with_shadow=NULL;
		Rasterizer::CanvasLight *lights_with_mask=NULL;
		Rect2 shadow_rect;

		int light_count=0;

		for (Map<RID,Viewport::CanvasData>::Element *E=p_viewport->canvas_map.front();E;E=E->next()) {

			Matrix32 xf = p_viewport->global_transform * E->get().transform;

			//find lights in canvas


			for(Set<Rasterizer::CanvasLight*>::Element *F=E->get().canvas->lights.front();F;F=F->next()) {


				Rasterizer::CanvasLight* cl=F->get();
				if (cl->enabled && cl->texture.is_valid()) {
					//not super efficient..
					Size2 tsize(rasterizer->texture_get_width(cl->texture),rasterizer->texture_get_height(cl->texture));
					tsize*=cl->scale;

					Vector2 offset=tsize/2.0;
					cl->rect_cache=Rect2(-offset+cl->texture_offset,tsize);
					cl->xform_cache=xf * cl->xform;


					if (clip_rect.intersects_transformed(cl->xform_cache,cl->rect_cache)) {

						cl->filter_next_ptr=lights;
						lights=cl;
						cl->texture_cache=NULL;
						Matrix32 scale;
						scale.scale(cl->rect_cache.size);
						scale.elements[2]=cl->rect_cache.pos;
						cl->light_shader_xform = (cl->xform_cache * scale).affine_inverse();
						cl->light_shader_pos=cl->xform_cache[2];
						if (cl->shadow_buffer.is_valid()) {

							cl->shadows_next_ptr=lights_with_shadow;
							if (lights_with_shadow==NULL) {
								shadow_rect = cl->xform_cache.xform(cl->rect_cache);
							} else {
								shadow_rect=shadow_rect.merge( cl->xform_cache.xform(cl->rect_cache) );
							}
							lights_with_shadow=cl;
							cl->radius_cache=cl->rect_cache.size.length();

						}
						if (cl->mode==CANVAS_LIGHT_MODE_MASK) {
							cl->mask_next_ptr=lights_with_mask;
							lights_with_mask=cl;
						}

						light_count++;
					}

				}
			}

			//print_line("lights: "+itos(light_count));
			canvas_map[ Viewport::CanvasKey( E->key(), E->get().layer) ]=&E->get();

		}

		if (lights_with_shadow) {
			//update shadows if any

			Rasterizer::CanvasLightOccluderInstance * occluders=NULL;

			//make list of occluders
			for (Map<RID,Viewport::CanvasData>::Element *E=p_viewport->canvas_map.front();E;E=E->next()) {

				Matrix32 xf = p_viewport->global_transform * E->get().transform;

				for(Set<Rasterizer::CanvasLightOccluderInstance*>::Element *F=E->get().canvas->occluders.front();F;F=F->next()) {

					if (!F->get()->enabled)
						continue;
					F->get()->xform_cache = xf * F->get()->xform;
					if (shadow_rect.intersects_transformed(F->get()->xform_cache,F->get()->aabb_cache)) {

						F->get()->next=occluders;
						occluders=F->get();

					}
				}
			}
			//update the light shadowmaps with them
			Rasterizer::CanvasLight *light=lights_with_shadow;
			while(light) {

				rasterizer->canvas_light_shadow_buffer_update(light->shadow_buffer,light->xform_cache.affine_inverse(),light->item_mask,light->radius_cache/1000.0,light->radius_cache*1.1,occluders,&light->shadow_matrix_cache);
				light=light->shadows_next_ptr;
			}

			rasterizer->set_viewport(viewport_rect); //must reset viewport afterwards
		}




		if (scenario_draw_canvas_bg && canvas_map.front() && canvas_map.front()->key().layer>scenario_canvas_max_layer) {

			_draw_viewport_camera(p_viewport,!can_draw_3d);
			scenario_draw_canvas_bg=false;

		}

		for (Map<Viewport::CanvasKey,Viewport::CanvasData*>::Element *E=canvas_map.front();E;E=E->next()) {


			//print_line("canvas "+itos(i)+" size: "+itos(I->get()->canvas->child_items.size()));
			//print_line("GT "+p_viewport->global_transform+". CT: "+E->get()->transform);
			Matrix32 xform = p_viewport->global_transform * E->get()->transform;

			Rasterizer::CanvasLight *canvas_lights=NULL;

			Rasterizer::CanvasLight *ptr=lights;
			while(ptr) {
				if (E->get()->layer>=ptr->layer_min && E->get()->layer<=ptr->layer_max) {
					ptr->next_ptr=canvas_lights;
					canvas_lights=ptr;
				}
				ptr=ptr->filter_next_ptr;
			}

			_render_canvas( E->get()->canvas,xform,canvas_lights,lights_with_mask );
			i++;

			if (scenario_draw_canvas_bg && E->key().layer>=scenario_canvas_max_layer) {
				_draw_viewport_camera(p_viewport,!can_draw_3d);
				scenario_draw_canvas_bg=false;
			}


		}

		if (scenario_draw_canvas_bg) {
			_draw_viewport_camera(p_viewport,!can_draw_3d);
			scenario_draw_canvas_bg=false;
		}


		//rasterizer->canvas_debug_viewport_shadows(lights_with_shadow);
	}

	//capture

	if (p_viewport->queue_capture) {

		rasterizer->capture_viewport(&p_viewport->capture);
		p_viewport->queue_capture = false;
	}

	//restore
	if ( viewport_rect.x != old_rect.x ||
		viewport_rect.y != old_rect.y ||
		viewport_rect.width != old_rect.width ||
		viewport_rect.height != old_rect.height ) {

		viewport_rect=old_rect;

		rasterizer->set_viewport(viewport_rect);
	}


}

void VisualServerRaster::_draw_viewports() {

	//draw viewports for render targets

	List<Viewport*> to_blit;
	List<Viewport*> to_disable;
	for(SelfList<Viewport> *E=viewport_update_list.first();E;E=E->next()) {

		Viewport *vp = E->self();
		ERR_CONTINUE(!vp);
		if (
			vp->render_target_update_mode==RENDER_TARGET_UPDATE_WHEN_VISIBLE &&
			!vp->rendered_in_prev_frame &&
			!vp->queue_capture
		    ) {

			continue;
		}

		if (vp->rt_to_screen_rect!=Rect2())
			to_blit.push_back(vp);

		rasterizer->set_render_target(vp->render_target,vp->transparent_bg,vp->render_target_vflip);
		_draw_viewport(vp,0,0,vp->rect.width,vp->rect.height);

		if ( (vp->queue_capture && vp->render_target_update_mode==RENDER_TARGET_UPDATE_DISABLED) || vp->render_target_update_mode==RENDER_TARGET_UPDATE_ONCE) {
			//was only enabled for capture
			to_disable.push_back(vp);
			vp->render_target_update_mode=RENDER_TARGET_UPDATE_DISABLED;
		}

	}

	rasterizer->set_render_target(RID());

	while(to_disable.size()) {
		//disable again because it was only for capture
		viewport_update_list.remove(&to_disable.front()->get()->update_list);
		to_disable.pop_front();
	}


	//draw RTs directly to screen when requested

	for (List<Viewport*>::Element *E=to_blit.front();E;E=E->next()) {

		int window_w = OS::get_singleton()->get_video_mode().width;
		int window_h = OS::get_singleton()->get_video_mode().height;

		ViewportRect desired_rect;
		desired_rect.x = desired_rect.y = 0;
		desired_rect.width = window_w;
		desired_rect.height = window_h;

		if ( viewport_rect.x != desired_rect.x ||
			viewport_rect.y != desired_rect.y ||
			viewport_rect.width != desired_rect.width ||
			viewport_rect.height != desired_rect.height ) {

			viewport_rect=desired_rect;

			rasterizer->set_viewport(viewport_rect);
		}

		rasterizer->canvas_begin();
		rasterizer->canvas_disable_blending();
		rasterizer->canvas_begin_rect(Matrix32());
		rasterizer->canvas_draw_rect(E->get()->rt_to_screen_rect,0,Rect2(Point2(),E->get()->rt_to_screen_rect.size),E->get()->render_target_texture,Color(1,1,1));

	}



	//draw viewports attached to screen

	for(Map<RID,int>::Element *E=screen_viewports.front();E;E=E->next()) {

		Viewport *vp = viewport_owner.get(E->key());
		ERR_CONTINUE(!vp);

		int window_w = OS::get_singleton()->get_video_mode(E->get()).width;
		int window_h = OS::get_singleton()->get_video_mode(E->get()).height;

		Rect2 r(0,0,vp->rect.width,vp->rect.height);
		if (r.size.width==0)
			r.size.width=window_w;
		if (r.size.height==0)
			r.size.height=window_h;


		_draw_viewport(vp,r.pos.x,r.pos.y,r.size.width,r.size.height);
	}


	//check when a viewport associated to a render target was drawn

	for(SelfList<Viewport> *E=viewport_update_list.first();E;E=E->next()) {

		Viewport *vp = E->self();
		ERR_CONTINUE(!vp);
		if (vp->render_target_update_mode!=RENDER_TARGET_UPDATE_WHEN_VISIBLE)
			continue;
		vp->rendered_in_prev_frame=rasterizer->render_target_renedered_in_frame(vp->render_target);
	}

}



void VisualServerRaster::_draw_cursors_and_margins() {

	int window_w = OS::get_singleton()->get_video_mode().width;
	int window_h = OS::get_singleton()->get_video_mode().height;

	ViewportRect desired_rect;
	desired_rect.x = desired_rect.y = 0;
	desired_rect.width = window_w;
	desired_rect.height = window_h;

	if ( viewport_rect.x != desired_rect.x ||
		viewport_rect.y != desired_rect.y ||
		viewport_rect.width != desired_rect.width ||
		viewport_rect.height != desired_rect.height ) {

		viewport_rect=desired_rect;

		rasterizer->set_viewport(viewport_rect);
	}

	rasterizer->canvas_begin();
	rasterizer->canvas_begin_rect(Matrix32());

	for (int i=0; i<MAX_CURSORS; i++) {

		if (!cursors[i].visible) {

			continue;
		};

		RID tex = cursors[i].texture?cursors[i].texture:default_cursor_texture;
		ERR_CONTINUE( !tex );
		if (cursors[i].region.has_no_area()) {
			Point2 size(texture_get_width(tex), texture_get_height(tex));
			rasterizer->canvas_draw_rect(Rect2(cursors[i].pos-cursors[i].center, size), 0, Rect2(), tex, Color(1, 1, 1, 1));
		} else {
			Point2 size = cursors[i].region.size;
			rasterizer->canvas_draw_rect(Rect2(cursors[i].pos-cursors[i].center, size), Rasterizer::CANVAS_RECT_REGION, cursors[i].region, tex, Color(1, 1, 1, 1));
		}
	};



	if (black_image[MARGIN_LEFT].is_valid()) {
		Size2 sz(rasterizer->texture_get_width(black_image[MARGIN_LEFT]),rasterizer->texture_get_height(black_image[MARGIN_LEFT]));
		rasterizer->canvas_draw_rect(Rect2(0,0,black_margin[MARGIN_LEFT],window_h),0,Rect2(0,0,sz.x,sz.y),black_image[MARGIN_LEFT],Color(1,1,1));
	} else if (black_margin[MARGIN_LEFT])
		rasterizer->canvas_draw_rect(Rect2(0,0,black_margin[MARGIN_LEFT],window_h),0,Rect2(0,0,1,1),RID(),Color(0,0,0));

	if (black_image[MARGIN_RIGHT].is_valid()) {
		Size2 sz(rasterizer->texture_get_width(black_image[MARGIN_RIGHT]),rasterizer->texture_get_height(black_image[MARGIN_RIGHT]));
		rasterizer->canvas_draw_rect(Rect2(window_w-black_margin[MARGIN_RIGHT],0,black_margin[MARGIN_RIGHT],window_h),0,Rect2(0,0,sz.x,sz.y),black_image[MARGIN_RIGHT],Color(1,1,1));
	} else if (black_margin[MARGIN_RIGHT])
		rasterizer->canvas_draw_rect(Rect2(window_w-black_margin[MARGIN_RIGHT],0,black_margin[MARGIN_RIGHT],window_h),0,Rect2(0,0,1,1),RID(),Color(0,0,0));

	if (black_image[MARGIN_TOP].is_valid()) {
		Size2 sz(rasterizer->texture_get_width(black_image[MARGIN_TOP]),rasterizer->texture_get_height(black_image[MARGIN_TOP]));
		rasterizer->canvas_draw_rect(Rect2(0,0,window_w,black_margin[MARGIN_TOP]),0,Rect2(0,0,sz.x,sz.y),black_image[MARGIN_TOP],Color(1,1,1));

	} else if (black_margin[MARGIN_TOP]) {
		rasterizer->canvas_draw_rect(Rect2(0,0,window_w,black_margin[MARGIN_TOP]),0,Rect2(0,0,1,1),RID(),Color(0,0,0));
	}

	if (black_image[MARGIN_BOTTOM].is_valid()) {

		Size2 sz(rasterizer->texture_get_width(black_image[MARGIN_BOTTOM]),rasterizer->texture_get_height(black_image[MARGIN_BOTTOM]));
		rasterizer->canvas_draw_rect(Rect2(0,window_h-black_margin[MARGIN_BOTTOM],window_w,black_margin[MARGIN_BOTTOM]),0,Rect2(0,0,sz.x,sz.y),black_image[MARGIN_BOTTOM],Color(1,1,1));
	} else if (black_margin[MARGIN_BOTTOM]) {
		rasterizer->canvas_draw_rect(Rect2(0,window_h-black_margin[MARGIN_BOTTOM],window_w,black_margin[MARGIN_BOTTOM]),0,Rect2(0,0,1,1),RID(),Color(0,0,0));
	}

	rasterizer->canvas_end_rect();
};

void VisualServerRaster::sync() {
	//do none
}

void VisualServerRaster::draw() {
	/*
	if (changes)
		print_line("changes: "+itos(changes));
	*/
	changes=0;
	shadows_enabled=GLOBAL_DEF("render/shadows_enabled",true);
	room_cull_enabled = GLOBAL_DEF("render/room_cull_enabled",true);
	light_discard_enabled = GLOBAL_DEF("render/light_discard_enabled",true);
	rasterizer->begin_frame();
	_draw_viewports();
	_draw_cursors_and_margins();
	rasterizer->end_frame();
	draw_extra_frame=rasterizer->needs_to_draw_next_frame();
}

bool VisualServerRaster::has_changed() const {

	return changes>0 || draw_extra_frame;
}

int VisualServerRaster::get_render_info(RenderInfo p_info) {

	return rasterizer->get_render_info(p_info);
}

bool VisualServerRaster::has_feature(Features p_feature) const {

	return rasterizer->has_feature(p_feature); // lies for now
}

void VisualServerRaster::set_default_clear_color(const Color& p_color) {

	clear_color=p_color;
}

void VisualServerRaster::set_boot_image(const Image& p_image, const Color& p_color,bool p_scale) {

	if (p_image.empty())
		return;

	rasterizer->restore_framebuffer();

	rasterizer->begin_frame();

	int window_w = OS::get_singleton()->get_video_mode(0).width;
	int window_h = OS::get_singleton()->get_video_mode(0).height;
	ViewportRect vr;
	vr.x=0;
	vr.y=0;
	vr.width=OS::get_singleton()->get_video_mode(0).width;
	vr.height=OS::get_singleton()->get_video_mode(0).height;
	rasterizer->set_viewport(vr);
	rasterizer->clear_viewport(p_color);
	rasterizer->canvas_begin();
	RID texture = texture_create();
	texture_allocate(texture,p_image.get_width(),p_image.get_height(),p_image.get_format(),TEXTURE_FLAG_FILTER);
	texture_set_data(texture,p_image);
	rasterizer->canvas_begin_rect(Matrix32());
	Rect2 imgrect(0,0,p_image.get_width(),p_image.get_height());
	Rect2 screenrect;
	if (p_scale) {

		if (window_w > window_h) {
			//scale horizontally
			screenrect.size.y = window_h;
			screenrect.size.x = imgrect.size.x * window_h / imgrect.size.y;
			screenrect.pos.x = (window_w - screenrect.size.x)/2;

		} else {
			//scale vertically
			screenrect.size.x = window_w;
			screenrect.size.y = imgrect.size.y * window_w / imgrect.size.x;
			screenrect.pos.y = (window_h - screenrect.size.y)/2;
		}
	} else {

		screenrect=imgrect;
		screenrect.pos+=((Size2(vr.width,vr.height)-screenrect.size)/2.0).floor();

	}
	rasterizer->canvas_draw_rect(screenrect,0,imgrect,texture,Color(1,1,1,1));
	rasterizer->canvas_end_rect();

	rasterizer->end_frame();
	rasterizer->flush_frame();

	free(texture); // free since it's only one frame that stays there

}

void VisualServerRaster::init() {

	rasterizer->init();

	shadows_enabled=GLOBAL_DEF("render/shadows_enabled",true);
	//default_scenario = scenario_create();
	//default_viewport = viewport_create();
	for(int i=0;i<4;i++)
		black_margin[i]=0;

	Image img;
	img.create(default_mouse_cursor_xpm);
	//img.convert(Image::FORMAT_RGB8);
	default_cursor_texture = texture_create_from_image(img, 0);

	aabb_random_points.resize( GLOBAL_DEF("render/aabb_random_points",16) );
	for(int i=0;i<aabb_random_points.size();i++)
		aabb_random_points[i]=Vector3(Math::random(0,1),Math::random(0,1),Math::random(0,1));
	transformed_aabb_random_points.resize(aabb_random_points.size());
	changes=0;
}

void VisualServerRaster::_clean_up_owner(RID_OwnerBase *p_owner,String p_type) {

	List<RID> rids;
	p_owner->get_owned_list(&rids);

	int lost=0;
	for(List<RID>::Element *I=rids.front();I;I=I->next()) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			lost++;
		}
		free(I->get());
	}

	if (lost)
		print_line("VisualServerRaster: WARNING: Lost "+itos(lost)+" RIDs of type "+p_type);

}

void VisualServerRaster::finish() {


	free(default_cursor_texture);
	if (test_cube.is_valid())
		free(test_cube);

	_free_internal_rids();

	_clean_up_owner( &room_owner,"Room" );
	_clean_up_owner( &portal_owner,"Portal" );

	_clean_up_owner( &camera_owner,"Camera" );
	_clean_up_owner( &viewport_owner,"Viewport" );

	_clean_up_owner( &scenario_owner,"Scenario" );
	_clean_up_owner( &instance_owner,"Instance" );

	_clean_up_owner( &canvas_owner,"Canvas" );
	_clean_up_owner( &canvas_item_owner,"CanvasItem" );

	rasterizer->finish();
	octree_allocator.clear();

	if (instance_dependency_map.size()) {
		print_line("Base resources missing amount: "+itos(instance_dependency_map.size()));
	}
	ERR_FAIL_COND( instance_dependency_map.size() );
}


RID VisualServerRaster::get_test_cube()  {

	if (test_cube.is_valid())
		return test_cube;

	test_cube=_make_test_cube();
	return test_cube;

}



VisualServerRaster::VisualServerRaster(Rasterizer *p_rasterizer) {

	rasterizer=p_rasterizer;
	rasterizer->draw_viewport_func=_render_canvas_item_viewport;
	instance_update_list=NULL;
	render_pass=0;
	clear_color=Color(0.3,0.3,0.3,1.0);
	OctreeAllocator::allocator=&octree_allocator;
	draw_extra_frame=false;

}


VisualServerRaster::~VisualServerRaster()
{
}
#endif
