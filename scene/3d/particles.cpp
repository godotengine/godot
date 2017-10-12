/*************************************************************************/
/*  particles.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "particles.h"
#include "scene/resources/surface_tool.h"
#include "servers/visual_server.h"

Rect3 Particles::get_aabb() const {

	return Rect3();
}
PoolVector<Face3> Particles::get_faces(uint32_t p_usage_flags) const {

	return PoolVector<Face3>();
}

void Particles::set_emitting(bool p_emitting) {

	emitting = p_emitting;
	VS::get_singleton()->particles_set_emitting(particles, emitting);
}

void Particles::set_amount(int p_amount) {

	ERR_FAIL_COND(p_amount < 1);
	amount = p_amount;
	VS::get_singleton()->particles_set_amount(particles, amount);
}
void Particles::set_lifetime(float p_lifetime) {

	ERR_FAIL_COND(p_lifetime <= 0);
	lifetime = p_lifetime;
	VS::get_singleton()->particles_set_lifetime(particles, lifetime);
}

void Particles::set_one_shot(bool p_one_shot) {

	one_shot = p_one_shot;
	VS::get_singleton()->particles_set_one_shot(particles, one_shot);
	if (!one_shot && emitting)
		VisualServer::get_singleton()->particles_restart(particles);
}

void Particles::set_pre_process_time(float p_time) {

	pre_process_time = p_time;
	VS::get_singleton()->particles_set_pre_process_time(particles, pre_process_time);
}
void Particles::set_explosiveness_ratio(float p_ratio) {

	explosiveness_ratio = p_ratio;
	VS::get_singleton()->particles_set_explosiveness_ratio(particles, explosiveness_ratio);
}
void Particles::set_randomness_ratio(float p_ratio) {

	randomness_ratio = p_ratio;
	VS::get_singleton()->particles_set_randomness_ratio(particles, randomness_ratio);
}
void Particles::set_visibility_aabb(const Rect3 &p_aabb) {

	visibility_aabb = p_aabb;
	VS::get_singleton()->particles_set_custom_aabb(particles, visibility_aabb);
	update_gizmo();
	_change_notify("visibility_aabb");
}
void Particles::set_use_local_coordinates(bool p_enable) {

	local_coords = p_enable;
	VS::get_singleton()->particles_set_use_local_coordinates(particles, local_coords);
}
void Particles::set_process_material(const Ref<Material> &p_material) {

	process_material = p_material;
	RID material_rid;
	if (process_material.is_valid())
		material_rid = process_material->get_rid();
	VS::get_singleton()->particles_set_process_material(particles, material_rid);

	update_configuration_warning();
}

void Particles::set_speed_scale(float p_scale) {

	speed_scale = p_scale;
	VS::get_singleton()->particles_set_speed_scale(particles, p_scale);
}

bool Particles::is_emitting() const {

	return emitting;
}
int Particles::get_amount() const {

	return amount;
}
float Particles::get_lifetime() const {

	return lifetime;
}
bool Particles::get_one_shot() const {

	return one_shot;
}

float Particles::get_pre_process_time() const {

	return pre_process_time;
}
float Particles::get_explosiveness_ratio() const {

	return explosiveness_ratio;
}
float Particles::get_randomness_ratio() const {

	return randomness_ratio;
}
Rect3 Particles::get_visibility_aabb() const {

	return visibility_aabb;
}
bool Particles::get_use_local_coordinates() const {

	return local_coords;
}
Ref<Material> Particles::get_process_material() const {

	return process_material;
}

float Particles::get_speed_scale() const {

	return speed_scale;
}

void Particles::set_draw_order(DrawOrder p_order) {

	draw_order = p_order;
	VS::get_singleton()->particles_set_draw_order(particles, VS::ParticlesDrawOrder(p_order));
}

Particles::DrawOrder Particles::get_draw_order() const {

	return draw_order;
}

void Particles::set_draw_passes(int p_count) {

	ERR_FAIL_COND(p_count < 1);
	draw_passes.resize(p_count);
	VS::get_singleton()->particles_set_draw_passes(particles, p_count);
	_change_notify();
}
int Particles::get_draw_passes() const {

	return draw_passes.size();
}

void Particles::set_draw_pass_mesh(int p_pass, const Ref<Mesh> &p_mesh) {

	ERR_FAIL_INDEX(p_pass, draw_passes.size());

	draw_passes[p_pass] = p_mesh;

	RID mesh_rid;
	if (p_mesh.is_valid())
		mesh_rid = p_mesh->get_rid();

	VS::get_singleton()->particles_set_draw_pass_mesh(particles, p_pass, mesh_rid);

	update_configuration_warning();
}

Ref<Mesh> Particles::get_draw_pass_mesh(int p_pass) const {

	ERR_FAIL_INDEX_V(p_pass, draw_passes.size(), Ref<Mesh>());

	return draw_passes[p_pass];
}

void Particles::set_fixed_fps(int p_count) {
	fixed_fps = p_count;
	VS::get_singleton()->particles_set_fixed_fps(particles, p_count);
}

int Particles::get_fixed_fps() const {
	return fixed_fps;
}

void Particles::set_fractional_delta(bool p_enable) {
	fractional_delta = p_enable;
	VS::get_singleton()->particles_set_fractional_delta(particles, p_enable);
}

bool Particles::get_fractional_delta() const {
	return fractional_delta;
}

String Particles::get_configuration_warning() const {

	String warnings;

	bool meshes_found = false;

	for (int i = 0; i < draw_passes.size(); i++) {
		if (draw_passes[i].is_valid()) {
			meshes_found = true;
			break;
		}
	}

	if (!meshes_found) {
		warnings += "- " + TTR("Nothing is visible because meshes have not been assigned to draw passes.");
	}

	if (process_material.is_null()) {
		if (warnings != String())
			warnings += "\n";
		warnings += "- " + TTR("A material to process the particles is not assigned, so no behavior is imprinted.");
	}

	return warnings;
}

void Particles::restart() {

	VisualServer::get_singleton()->particles_restart(particles);
}

Rect3 Particles::capture_aabb() const {

	return VS::get_singleton()->particles_get_current_aabb(particles);
}

void Particles::_validate_property(PropertyInfo &property) const {

	if (property.name.begins_with("draw_pass_")) {
		int index = property.name.get_slicec('_', 2).to_int() - 1;
		if (index >= draw_passes.size()) {
			property.usage = 0;
			return;
		}
	}
}

void Particles::_notification(int p_what) {

	if (p_what == NOTIFICATION_PAUSED || p_what == NOTIFICATION_UNPAUSED) {
		if (can_process()) {
			VS::get_singleton()->particles_set_speed_scale(particles, speed_scale);
		} else {

			VS::get_singleton()->particles_set_speed_scale(particles, 0);
		}
	}
}

void Particles::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_emitting", "emitting"), &Particles::set_emitting);
	ClassDB::bind_method(D_METHOD("set_amount", "amount"), &Particles::set_amount);
	ClassDB::bind_method(D_METHOD("set_lifetime", "secs"), &Particles::set_lifetime);
	ClassDB::bind_method(D_METHOD("set_one_shot", "enable"), &Particles::set_one_shot);
	ClassDB::bind_method(D_METHOD("set_pre_process_time", "secs"), &Particles::set_pre_process_time);
	ClassDB::bind_method(D_METHOD("set_explosiveness_ratio", "ratio"), &Particles::set_explosiveness_ratio);
	ClassDB::bind_method(D_METHOD("set_randomness_ratio", "ratio"), &Particles::set_randomness_ratio);
	ClassDB::bind_method(D_METHOD("set_visibility_aabb", "aabb"), &Particles::set_visibility_aabb);
	ClassDB::bind_method(D_METHOD("set_use_local_coordinates", "enable"), &Particles::set_use_local_coordinates);
	ClassDB::bind_method(D_METHOD("set_fixed_fps", "fps"), &Particles::set_fixed_fps);
	ClassDB::bind_method(D_METHOD("set_fractional_delta", "enable"), &Particles::set_fractional_delta);
	ClassDB::bind_method(D_METHOD("set_process_material", "material"), &Particles::set_process_material);
	ClassDB::bind_method(D_METHOD("set_speed_scale", "scale"), &Particles::set_speed_scale);

	ClassDB::bind_method(D_METHOD("is_emitting"), &Particles::is_emitting);
	ClassDB::bind_method(D_METHOD("get_amount"), &Particles::get_amount);
	ClassDB::bind_method(D_METHOD("get_lifetime"), &Particles::get_lifetime);
	ClassDB::bind_method(D_METHOD("get_one_shot"), &Particles::get_one_shot);
	ClassDB::bind_method(D_METHOD("get_pre_process_time"), &Particles::get_pre_process_time);
	ClassDB::bind_method(D_METHOD("get_explosiveness_ratio"), &Particles::get_explosiveness_ratio);
	ClassDB::bind_method(D_METHOD("get_randomness_ratio"), &Particles::get_randomness_ratio);
	ClassDB::bind_method(D_METHOD("get_visibility_aabb"), &Particles::get_visibility_aabb);
	ClassDB::bind_method(D_METHOD("get_use_local_coordinates"), &Particles::get_use_local_coordinates);
	ClassDB::bind_method(D_METHOD("get_fixed_fps"), &Particles::get_fixed_fps);
	ClassDB::bind_method(D_METHOD("get_fractional_delta"), &Particles::get_fractional_delta);
	ClassDB::bind_method(D_METHOD("get_process_material"), &Particles::get_process_material);
	ClassDB::bind_method(D_METHOD("get_speed_scale"), &Particles::get_speed_scale);

	ClassDB::bind_method(D_METHOD("set_draw_order", "order"), &Particles::set_draw_order);

	ClassDB::bind_method(D_METHOD("get_draw_order"), &Particles::get_draw_order);

	ClassDB::bind_method(D_METHOD("set_draw_passes", "passes"), &Particles::set_draw_passes);
	ClassDB::bind_method(D_METHOD("set_draw_pass_mesh", "pass", "mesh"), &Particles::set_draw_pass_mesh);

	ClassDB::bind_method(D_METHOD("get_draw_passes"), &Particles::get_draw_passes);
	ClassDB::bind_method(D_METHOD("get_draw_pass_mesh", "pass"), &Particles::get_draw_pass_mesh);

	ClassDB::bind_method(D_METHOD("restart"), &Particles::restart);
	ClassDB::bind_method(D_METHOD("capture_aabb"), &Particles::capture_aabb);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "emitting"), "set_emitting", "is_emitting");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "amount", PROPERTY_HINT_RANGE, "1,100000,1"), "set_amount", "get_amount");
	ADD_GROUP("Time", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lifetime", PROPERTY_HINT_RANGE, "0.01,600.0,0.01"), "set_lifetime", "get_lifetime");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "one_shot"), "set_one_shot", "get_one_shot");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "preprocess", PROPERTY_HINT_RANGE, "0.00,600.0,0.01"), "set_pre_process_time", "get_pre_process_time");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "speed_scale", PROPERTY_HINT_RANGE, "0.01,64,0.01"), "set_speed_scale", "get_speed_scale");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "explosiveness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_explosiveness_ratio", "get_explosiveness_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "randomness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_randomness_ratio", "get_randomness_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fixed_fps", PROPERTY_HINT_RANGE, "0,1000,1"), "set_fixed_fps", "get_fixed_fps");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fract_delta"), "set_fractional_delta", "get_fractional_delta");
	ADD_GROUP("Drawing", "");
	ADD_PROPERTY(PropertyInfo(Variant::RECT3, "visibility_aabb"), "set_visibility_aabb", "get_visibility_aabb");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "local_coords"), "set_use_local_coordinates", "get_use_local_coordinates");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "draw_order", PROPERTY_HINT_ENUM, "Index,Lifetime,View Depth"), "set_draw_order", "get_draw_order");
	ADD_GROUP("Process Material", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "process_material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,ParticlesMaterial"), "set_process_material", "get_process_material");
	ADD_GROUP("Draw Passes", "draw_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "draw_passes", PROPERTY_HINT_RANGE, "0," + itos(MAX_DRAW_PASSES) + ",1"), "set_draw_passes", "get_draw_passes");
	for (int i = 0; i < MAX_DRAW_PASSES; i++) {

		ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "draw_pass_" + itos(i + 1), PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_draw_pass_mesh", "get_draw_pass_mesh", i);
	}

	BIND_ENUM_CONSTANT(DRAW_ORDER_INDEX);
	BIND_ENUM_CONSTANT(DRAW_ORDER_LIFETIME);
	BIND_ENUM_CONSTANT(DRAW_ORDER_VIEW_DEPTH);

	BIND_CONSTANT(MAX_DRAW_PASSES);
}

Particles::Particles() {

	particles = VS::get_singleton()->particles_create();
	set_base(particles);
	set_emitting(true);
	set_one_shot(false);
	set_amount(8);
	set_lifetime(1);
	set_fixed_fps(0);
	set_fractional_delta(true);
	set_pre_process_time(0);
	set_explosiveness_ratio(0);
	set_randomness_ratio(0);
	set_visibility_aabb(Rect3(Vector3(-4, -4, -4), Vector3(8, 8, 8)));
	set_use_local_coordinates(true);
	set_draw_passes(1);
	set_draw_order(DRAW_ORDER_INDEX);
	set_speed_scale(1);
}

Particles::~Particles() {

	VS::get_singleton()->free(particles);
}

//////////////////////////////////////

Mutex *ParticlesMaterial::material_mutex = NULL;
SelfList<ParticlesMaterial>::List ParticlesMaterial::dirty_materials;
Map<ParticlesMaterial::MaterialKey, ParticlesMaterial::ShaderData> ParticlesMaterial::shader_map;
ParticlesMaterial::ShaderNames *ParticlesMaterial::shader_names = NULL;

void ParticlesMaterial::init_shaders() {

#ifndef NO_THREADS
	material_mutex = Mutex::create();
#endif

	shader_names = memnew(ShaderNames);

	shader_names->spread = "spread";
	shader_names->flatness = "flatness";
	shader_names->initial_linear_velocity = "initial_linear_velocity";
	shader_names->initial_angle = "initial_angle";
	shader_names->angular_velocity = "angular_velocity";
	shader_names->orbit_velocity = "orbit_velocity";
	shader_names->linear_accel = "linear_accel";
	shader_names->radial_accel = "radial_accel";
	shader_names->tangent_accel = "tangent_accel";
	shader_names->damping = "damping";
	shader_names->scale = "scale";
	shader_names->hue_variation = "hue_variation";
	shader_names->anim_speed = "anim_speed";
	shader_names->anim_offset = "anim_offset";

	shader_names->initial_linear_velocity_random = "initial_linear_velocity_random";
	shader_names->initial_angle_random = "initial_angle_random";
	shader_names->angular_velocity_random = "angular_velocity_random";
	shader_names->orbit_velocity_random = "orbit_velocity_random";
	shader_names->linear_accel_random = "linear_accel_random";
	shader_names->radial_accel_random = "radial_accel_random";
	shader_names->tangent_accel_random = "tangent_accel_random";
	shader_names->damping_random = "damping_random";
	shader_names->scale_random = "scale_random";
	shader_names->hue_variation_random = "hue_variation_random";
	shader_names->anim_speed_random = "anim_speed_random";
	shader_names->anim_offset_random = "anim_offset_random";

	shader_names->angle_texture = "angle_texture";
	shader_names->angular_velocity_texture = "angular_velocity_texture";
	shader_names->orbit_velocity_texture = "orbit_velocity_texture";
	shader_names->linear_accel_texture = "linear_accel_texture";
	shader_names->radial_accel_texture = "radial_accel_texture";
	shader_names->tangent_accel_texture = "tangent_accel_texture";
	shader_names->damping_texture = "damping_texture";
	shader_names->scale_texture = "scale_texture";
	shader_names->hue_variation_texture = "hue_variation_texture";
	shader_names->anim_speed_texture = "anim_speed_texture";
	shader_names->anim_offset_texture = "anim_offset_texture";

	shader_names->color = "color_value";
	shader_names->color_ramp = "color_ramp";

	shader_names->emission_sphere_radius = "emission_sphere_radius";
	shader_names->emission_box_extents = "emission_box_extents";
	shader_names->emission_texture_point_count = "emission_texture_point_count";
	shader_names->emission_texture_points = "emission_texture_points";
	shader_names->emission_texture_normal = "emission_texture_normal";
	shader_names->emission_texture_color = "emission_texture_color";

	shader_names->trail_divisor = "trail_divisor";
	shader_names->trail_size_modifier = "trail_size_modifier";
	shader_names->trail_color_modifier = "trail_color_modifier";

	shader_names->gravity = "gravity";
}

void ParticlesMaterial::finish_shaders() {

#ifndef NO_THREADS
	memdelete(material_mutex);
#endif

	memdelete(shader_names);
}

void ParticlesMaterial::_update_shader() {

	dirty_materials.remove(&element);

	MaterialKey mk = _compute_key();
	if (mk.key == current_key.key)
		return; //no update required in the end

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			VS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}
	}

	current_key = mk;

	if (shader_map.has(mk)) {

		VS::get_singleton()->material_set_shader(_get_material(), shader_map[mk].shader);
		shader_map[mk].users++;
		return;
	}

	//must create a shader!

	String code = "shader_type particles;\n";

	code += "uniform float spread;\n";
	code += "uniform float flatness;\n";
	code += "uniform float initial_linear_velocity;\n";
	code += "uniform float initial_angle;\n";
	code += "uniform float angular_velocity;\n";
	code += "uniform float orbit_velocity;\n";
	code += "uniform float linear_accel;\n";
	code += "uniform float radial_accel;\n";
	code += "uniform float tangent_accel;\n";
	code += "uniform float damping;\n";
	code += "uniform float scale;\n";
	code += "uniform float hue_variation;\n";
	code += "uniform float anim_speed;\n";
	code += "uniform float anim_offset;\n";

	code += "uniform float initial_linear_velocity_random;\n";
	code += "uniform float initial_angle_random;\n";
	code += "uniform float angular_velocity_random;\n";
	code += "uniform float orbit_velocity_random;\n";
	code += "uniform float linear_accel_random;\n";
	code += "uniform float radial_accel_random;\n";
	code += "uniform float tangent_accel_random;\n";
	code += "uniform float damping_random;\n";
	code += "uniform float scale_random;\n";
	code += "uniform float hue_variation_random;\n";
	code += "uniform float anim_speed_random;\n";
	code += "uniform float anim_offset_random;\n";

	switch (emission_shape) {
		case EMISSION_SHAPE_POINT: {
			//do none
		} break;
		case EMISSION_SHAPE_SPHERE: {
			code += "uniform float emission_sphere_radius;\n";
		} break;
		case EMISSION_SHAPE_BOX: {
			code += "uniform vec3 emission_box_extents;\n";
		} break;
		case EMISSION_SHAPE_DIRECTED_POINTS: {
			code += "uniform sampler2D emission_texture_normal : hint_black;\n";
		} //fallthrough
		case EMISSION_SHAPE_POINTS: {
			code += "uniform sampler2D emission_texture_points : hint_black;\n";
			code += "uniform int emission_texture_point_count;\n";
			if (emission_color_texture.is_valid()) {
				code += "uniform sampler2D emission_texture_color : hint_white;\n";
			}
		} break;
	}

	code += "uniform vec4 color_value : hint_color;\n";

	code += "uniform int trail_divisor;\n";

	code += "uniform vec3 gravity;\n";

	if (color_ramp.is_valid())
		code += "uniform sampler2D color_ramp;\n";

	if (tex_parameters[PARAM_INITIAL_LINEAR_VELOCITY].is_valid())
		code += "uniform sampler2D linear_velocity_texture;\n";
	if (tex_parameters[PARAM_ORBIT_VELOCITY].is_valid())
		code += "uniform sampler2D orbit_velocity_texture;\n";
	if (tex_parameters[PARAM_ANGULAR_VELOCITY].is_valid())
		code += "uniform sampler2D angular_velocity_texture;\n";
	if (tex_parameters[PARAM_LINEAR_ACCEL].is_valid())
		code += "uniform sampler2D linear_accel_texture;\n";
	if (tex_parameters[PARAM_RADIAL_ACCEL].is_valid())
		code += "uniform sampler2D radial_accel_texture;\n";
	if (tex_parameters[PARAM_TANGENTIAL_ACCEL].is_valid())
		code += "uniform sampler2D tangent_accel_texture;\n";
	if (tex_parameters[PARAM_DAMPING].is_valid())
		code += "uniform sampler2D damping_texture;\n";
	if (tex_parameters[PARAM_ANGLE].is_valid())
		code += "uniform sampler2D angle_texture;\n";
	if (tex_parameters[PARAM_SCALE].is_valid())
		code += "uniform sampler2D scale_texture;\n";
	if (tex_parameters[PARAM_HUE_VARIATION].is_valid())
		code += "uniform sampler2D hue_variation_texture;\n";
	if (tex_parameters[PARAM_ANIM_SPEED].is_valid())
		code += "uniform sampler2D anim_speed_texture;\n";
	if (tex_parameters[PARAM_ANIM_OFFSET].is_valid())
		code += "uniform sampler2D anim_offset_texture;\n";

	if (trail_size_modifier.is_valid()) {
		code += "uniform sampler2D trail_size_modifier;\n";
	}

	if (trail_color_modifier.is_valid()) {
		code += "uniform sampler2D trail_color_modifier;\n";
	}

	//need a random function
	code += "\n\n";
	code += "float rand_from_seed(inout uint seed) {\n";
	code += "	int k;\n";
	code += "	int s = int(seed);\n";
	code += "	if (s == 0)\n";
	code += "	s = 305420679;\n";
	code += "	k = s / 127773;\n";
	code += "	s = 16807 * (s - k * 127773) - 2836 * k;\n";
	code += "	if (s < 0)\n";
	code += "		s += 2147483647;\n";
	code += "	seed = uint(s);\n";
	code += "	return float(seed % uint(65536))/65535.0;\n";
	code += "}\n";
	code += "\n";

	//improve seed quality
	code += "uint hash(uint x) {\n";
	code += "	x = ((x >> uint(16)) ^ x) * uint(73244475);\n";
	code += "	x = ((x >> uint(16)) ^ x) * uint(73244475);\n";
	code += "	x = (x >> uint(16)) ^ x;\n";
	code += "	return x;\n";
	code += "}\n";
	code += "\n";

	code += "void vertex() {\n";
	code += "	uint base_number = NUMBER/uint(trail_divisor);\n";
	code += "	uint alt_seed = hash(base_number+uint(1)+RANDOM_SEED);\n";
	code += "	float angle_rand = rand_from_seed(alt_seed);\n";
	code += "	float scale_rand = rand_from_seed(alt_seed);\n";
	code += "	float hue_rot_rand = rand_from_seed(alt_seed);\n";
	code += "	float anim_offset_rand = rand_from_seed(alt_seed);\n";
	code += "\n";

	if (emission_shape >= EMISSION_SHAPE_POINTS) {
		code += "	int point = min(emission_texture_point_count-1,int(rand_from_seed(alt_seed) * float(emission_texture_point_count)));\n";
		code += "	ivec2 emission_tex_size = textureSize( emission_texture_points, 0 );\n";
		code += "	ivec2 emission_tex_ofs = ivec2( point % emission_tex_size.x, point / emission_tex_size.x );\n";
	}
	code += "	if (RESTART) {\n";

	if (tex_parameters[PARAM_INITIAL_LINEAR_VELOCITY].is_valid())
		code += "		float tex_linear_velocity = textureLod(linear_velocity_texture,vec2(0.0,0.0),0.0).r;\n";
	else
		code += "		float tex_linear_velocity = 0.0;\n";

	if (tex_parameters[PARAM_ANGLE].is_valid())
		code += "		float tex_angle = textureLod(angle_texture,vec2(0.0,0.0),0.0).r;\n";
	else
		code += "		float tex_angle = 0.0;\n";

	if (tex_parameters[PARAM_ANIM_OFFSET].is_valid())
		code += "		float tex_anim_offset = textureLod(anim_offset_texture,vec2(0.0,0.0),0.0).r;\n";
	else
		code += "		float tex_anim_offset = 0.0;\n";

	if (flags[FLAG_DISABLE_Z]) {

		code += "		float angle1 = (rand_from_seed(alt_seed)*2.0-1.0)*spread/180.0*3.1416;\n";
		code += "		vec3 rot = vec3( cos(angle1), sin(angle1),0.0 );\n";
		code += "		VELOCITY = (rot*initial_linear_velocity+rot*initial_linear_velocity_random*rand_from_seed(alt_seed));\n";

	} else {
		//initiate velocity spread in 3D
		code += "		float angle1 = rand_from_seed(alt_seed)*spread*3.1416;\n";
		code += "		float angle2 = rand_from_seed(alt_seed)*20.0*3.1416; // make it more random like\n";
		code += "		vec3 rot_xz = vec3( sin(angle1), 0.0, cos(angle1) );\n";
		code += "		vec3 rot = vec3( cos(angle2)*rot_xz.x,sin(angle2)*rot_xz.x, rot_xz.z);\n";
		code += "		VELOCITY = (rot*initial_linear_velocity+rot*initial_linear_velocity_random*rand_from_seed(alt_seed));\n";
	}

	code += "		float base_angle = (initial_angle+tex_angle)*mix(1.0,angle_rand,initial_angle_random);\n";
	code += "		CUSTOM.x = base_angle*3.1416/180.0;\n"; //angle
	code += "		CUSTOM.y = 0.0;\n"; //phase
	code += "		CUSTOM.z = (anim_offset+tex_anim_offset)*mix(1.0,anim_offset_rand,anim_offset_random);\n"; //animation offset (0-1)
	switch (emission_shape) {
		case EMISSION_SHAPE_POINT: {
			//do none
		} break;
		case EMISSION_SHAPE_SPHERE: {
			code += "		TRANSFORM[3].xyz = normalize(vec3(rand_from_seed(alt_seed) * 2.0 - 1.0, rand_from_seed(alt_seed) * 2.0-1.0, rand_from_seed(alt_seed) * 2.0-1.0 ))*emission_sphere_radius;\n";
		} break;
		case EMISSION_SHAPE_BOX: {
			code += "		TRANSFORM[3].xyz = vec3(rand_from_seed(alt_seed) * 2.0 - 1.0, rand_from_seed(alt_seed) * 2.0-1.0, rand_from_seed(alt_seed) * 2.0-1.0)*emission_box_extents;\n";
		} break;
		case EMISSION_SHAPE_POINTS:
		case EMISSION_SHAPE_DIRECTED_POINTS: {
			code += "		TRANSFORM[3].xyz = texelFetch(emission_texture_points, emission_tex_ofs,0).xyz;\n";

			if (emission_shape == EMISSION_SHAPE_DIRECTED_POINTS) {
				if (flags[FLAG_DISABLE_Z]) {

					code += "		mat2 rotm;";
					code += "		rotm[0] = texelFetch(emission_texture_normal, emission_tex_ofs,0).xy;\n";
					code += "		rotm[1] = rotm[0].yx * vec2(1.0,-1.0);\n";
					code += "		VELOCITY.xy = rotm * VELOCITY.xy;\n";
				} else {
					code += "		vec3 normal = texelFetch(emission_texture_normal, emission_tex_ofs,0).xyz;\n";
					code += "		vec3 v0 = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0, 1.0, 0.0);\n";
					code += "		vec3 tangent = normalize(cross(v0, normal));\n";
					code += "		vec3 bitangent = normalize(cross(tangent, normal));\n";
					code += "		VELOCITY = mat3(tangent,bitangent,normal) * VELOCITY;\n";
				}
			}
		} break;
	}
	code += "		VELOCITY = (EMISSION_TRANSFORM * vec4(VELOCITY,0.0)).xyz;\n";
	code += "		TRANSFORM = EMISSION_TRANSFORM * TRANSFORM;\n";
	if (flags[FLAG_DISABLE_Z]) {
		code += "		VELOCITY.z = 0.0;\n";
		code += "		TRANSFORM[3].z = 0.0;\n";
	}

	code += "	} else {\n";

	code += "		CUSTOM.y += DELTA/LIFETIME;\n";
	if (tex_parameters[PARAM_INITIAL_LINEAR_VELOCITY].is_valid())
		code += "		float tex_linear_velocity = textureLod(linear_velocity_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "		float tex_linear_velocity = 0.0;\n";

	if (tex_parameters[PARAM_ORBIT_VELOCITY].is_valid())
		code += "		float tex_orbit_velocity = textureLod(orbit_velocity_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "		float tex_orbit_velocity = 0.0;\n";

	if (tex_parameters[PARAM_ANGULAR_VELOCITY].is_valid())
		code += "		float tex_angular_velocity = textureLod(angular_velocity_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "		float tex_angular_velocity = 0.0;\n";

	if (tex_parameters[PARAM_LINEAR_ACCEL].is_valid())
		code += "		float tex_linear_accel = textureLod(linear_accel_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "		float tex_linear_accel = 0.0;\n";

	if (tex_parameters[PARAM_RADIAL_ACCEL].is_valid())
		code += "		float tex_radial_accel = textureLod(radial_accel_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "		float tex_radial_accel = 0.0;\n";

	if (tex_parameters[PARAM_TANGENTIAL_ACCEL].is_valid())
		code += "		float tex_tangent_accel = textureLod(tangent_accel_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "		float tex_tangent_accel = 0.0;\n";

	if (tex_parameters[PARAM_DAMPING].is_valid())
		code += "		float tex_damping = textureLod(damping_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "		float tex_damping = 0.0;\n";

	if (tex_parameters[PARAM_ANGLE].is_valid())
		code += "		float tex_angle = textureLod(angle_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "		float tex_angle = 0.0;\n";

	if (tex_parameters[PARAM_ANIM_SPEED].is_valid())
		code += "		float tex_anim_speed = textureLod(anim_speed_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "		float tex_anim_speed = 0.0;\n";

	if (tex_parameters[PARAM_ANIM_OFFSET].is_valid())
		code += "		float tex_anim_offset = textureLod(anim_offset_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "		float tex_anim_offset = 0.0;\n";

	code += "		vec3 force = gravity; \n";
	code += "		vec3 pos = TRANSFORM[3].xyz; \n";
	if (flags[FLAG_DISABLE_Z]) {
		code += "		pos.z = 0.0; \n";
	}
	code += "		//apply linear acceleration\n";
	code += "		force += length(VELOCITY) > 0.0 ? normalize(VELOCITY) * (linear_accel+tex_linear_accel)*mix(1.0,rand_from_seed(alt_seed),linear_accel_random) : vec3(0.0);\n";
	code += "		//apply radial acceleration\n";
	code += "		vec3 org = vec3(0.0);\n";
	code += "		vec3 diff = pos-org;\n";
	code += "		force += length(diff) > 0.0 ? normalize(diff) * (radial_accel+tex_radial_accel)*mix(1.0,rand_from_seed(alt_seed),radial_accel_random) : vec3(0.0);\n";
	code += "		//apply tangential acceleration;\n";
	if (flags[FLAG_DISABLE_Z]) {
		code += "		force += length(diff.yx) > 0.0 ? vec3(normalize(diff.yx * vec2(-1.0,1.0)),0.0) * ((tangent_accel+tex_tangent_accel)*mix(1.0,rand_from_seed(alt_seed),radial_accel_random)) : vec3(0.0);\n";

	} else {
		code += "		vec3 crossDiff = cross(normalize(diff),normalize(gravity));\n";
		code += "		force += length(crossDiff) > 0.0 ? normalize(crossDiff) * ((tangent_accel+tex_tangent_accel)*mix(1.0,rand_from_seed(alt_seed),radial_accel_random)) : vec3(0.0);\n";
	}
	code += "		//apply attractor forces\n";
	code += "		VELOCITY += force * DELTA;\n";
	if (tex_parameters[PARAM_INITIAL_LINEAR_VELOCITY].is_valid()) {
		code += "		VELOCITY = normalize(VELOCITY)*tex_linear_velocity;\n";
	}
	code += "		if (damping + tex_damping > 0.0) {\n";
	code += "		\n";
	code += "			float v = length(VELOCITY);\n";
	code += "			float damp = (damping+tex_damping)*mix(1.0,rand_from_seed(alt_seed),damping_random);\n";
	code += "			v -= damp * DELTA;\n";
	code += "			if (v < 0.0) {\n";
	code += "				VELOCITY = vec3(0.0);\n";
	code += "			} else {\n";
	code += "				VELOCITY = normalize(VELOCITY) * v;\n";
	code += "			}\n";
	code += "		}\n";
	code += "		float base_angle = (initial_angle+tex_angle)*mix(1.0,angle_rand,initial_angle_random);\n";
	code += "		base_angle += CUSTOM.y*LIFETIME*(angular_velocity+tex_angular_velocity)*mix(1.0,rand_from_seed(alt_seed)*2.0-1.0,angular_velocity_random);\n";
	code += "		CUSTOM.x = base_angle*3.1416/180.0;\n"; //angle
	code += "		CUSTOM.z = (anim_offset+tex_anim_offset)*mix(1.0,anim_offset_rand,anim_offset_random)+CUSTOM.y*(anim_speed+tex_anim_speed)*mix(1.0,rand_from_seed(alt_seed),anim_speed_random);\n"; //angle
	if (flags[FLAG_ANIM_LOOP]) {
		code += "		CUSTOM.z = mod(CUSTOM.z,1.0);\n"; //loop

	} else {
		code += "		CUSTOM.z = clamp(CUSTOM.z,0.0,1.0);\n"; //0 to 1 only
	}
	code += "	}\n";
	//apply color
	//apply hue rotation
	if (tex_parameters[PARAM_SCALE].is_valid())
		code += "	float tex_scale = textureLod(scale_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "	float tex_scale = 1.0;\n";

	if (tex_parameters[PARAM_HUE_VARIATION].is_valid())
		code += "	float tex_hue_variation = textureLod(hue_variation_texture,vec2(CUSTOM.y,0.0),0.0).r;\n";
	else
		code += "	float tex_hue_variation = 0.0;\n";

	code += "	float hue_rot_angle = (hue_variation+tex_hue_variation)*3.1416*2.0*mix(1.0,hue_rot_rand*2.0-1.0,hue_variation_random);\n";
	code += "	float hue_rot_c = cos(hue_rot_angle);\n";
	code += "	float hue_rot_s = sin(hue_rot_angle);\n";
	code += "	mat4 hue_rot_mat = mat4( vec4(0.299,  0.587,  0.114, 0.0),\n";
	code += "			vec4(0.299,  0.587,  0.114, 0.0),\n";
	code += "			vec4(0.299,  0.587,  0.114, 0.0),\n";
	code += "			vec4(0.000,  0.000,  0.000, 1.0)) +\n";
	code += "		mat4( vec4(0.701, -0.587, -0.114, 0.0),\n";
	code += "			vec4(-0.299,  0.413, -0.114, 0.0),\n";
	code += "			vec4(-0.300, -0.588,  0.886, 0.0),\n";
	code += "			vec4(0.000,  0.000,  0.000, 0.0)) * hue_rot_c +\n";
	code += "		mat4( vec4(0.168,  0.330, -0.497, 0.0),\n";
	code += "			vec4(-0.328,  0.035,  0.292, 0.0),\n";
	code += "			vec4(1.250, -1.050, -0.203, 0.0),\n";
	code += "			vec4(0.000,  0.000,  0.000, 0.0)) * hue_rot_s;\n";
	if (color_ramp.is_valid()) {
		code += "	COLOR = textureLod(color_ramp,vec2(CUSTOM.y,0.0),0.0) * hue_rot_mat;\n";
	} else {
		code += "	COLOR = color_value * hue_rot_mat;\n";
	}
	if (emission_color_texture.is_valid() && emission_shape >= EMISSION_SHAPE_POINTS) {
		code += "	COLOR*= texelFetch(emission_texture_color,emission_tex_ofs,0);\n";
	}
	if (trail_color_modifier.is_valid()) {
		code += "	if (trail_divisor > 1) { COLOR *= textureLod(trail_color_modifier,vec2(float(int(NUMBER)%trail_divisor)/float(trail_divisor-1),0.0),0.0); }\n";
	}
	code += "\n";

	if (flags[FLAG_DISABLE_Z]) {

		code += "	TRANSFORM[0] = vec4(cos(CUSTOM.x),-sin(CUSTOM.x),0.0,0.0);\n";
		code += "	TRANSFORM[1] = vec4(sin(CUSTOM.x),cos(CUSTOM.x),0.0,0.0);\n";
		code += "	TRANSFORM[2] = vec4(0.0,0.0,1.0,0.0);\n";

	} else {
		//orient particle Y towards velocity
		if (flags[FLAG_ALIGN_Y_TO_VELOCITY]) {
			code += "	if (length(VELOCITY) > 0.0) { TRANSFORM[1].xyz = normalize(VELOCITY); } else { TRANSFORM[1].xyz = normalize(TRANSFORM[1].xyz); }\n";
			code += "	if (TRANSFORM[1].xyz == normalize(TRANSFORM[0].xyz)) {\n";
			code += "		TRANSFORM[0].xyz = normalize(cross(normalize(TRANSFORM[1].xyz),normalize(TRANSFORM[2].xyz)));\n";
			code += "		TRANSFORM[2].xyz = normalize(cross(normalize(TRANSFORM[0].xyz),normalize(TRANSFORM[1].xyz)));\n";
			code += "	} else {\n";
			code += "		TRANSFORM[2].xyz = normalize(cross(normalize(TRANSFORM[0].xyz),normalize(TRANSFORM[1].xyz)));\n";
			code += "		TRANSFORM[0].xyz = normalize(cross(normalize(TRANSFORM[1].xyz),normalize(TRANSFORM[2].xyz)));\n";
			code += "	}\n";
		} else {
			code += "	TRANSFORM[0].xyz = normalize(TRANSFORM[0].xyz);\n";
			code += "	TRANSFORM[1].xyz = normalize(TRANSFORM[1].xyz);\n";
			code += "	TRANSFORM[2].xyz = normalize(TRANSFORM[2].xyz);\n";
		}
		//turn particle by rotation in Y
		if (flags[FLAG_ROTATE_Y]) {
			code += "	TRANSFORM = TRANSFORM * mat4( vec4(cos(CUSTOM.x),0.0,-sin(CUSTOM.x),0.0), vec4(0.0,1.0,0.0,0.0),vec4(sin(CUSTOM.x),0.0,cos(CUSTOM.x),0.0),vec4(0.0,0.0,0.0,1.0));\n";
		}
	}
	//scale by scale
	code += "	float base_scale = mix(scale*tex_scale,1.0,scale_random*scale_rand);\n";
	if (trail_size_modifier.is_valid()) {
		code += "	if (trail_divisor > 1) { base_scale *= textureLod(trail_size_modifier,vec2(float(int(NUMBER)%trail_divisor)/float(trail_divisor-1),0.0),0.0).r; } \n";
	}

	code += "	TRANSFORM[0].xyz *= base_scale;\n";
	code += "	TRANSFORM[1].xyz *= base_scale;\n";
	code += "	TRANSFORM[2].xyz *= base_scale;\n";
	if (flags[FLAG_DISABLE_Z]) {
		code += "	VELOCITY.z = 0.0;\n";
		code += "	TRANSFORM[3].z = 0.0;\n";
	}
	code += "}\n";
	code += "\n";

	ShaderData shader_data;
	shader_data.shader = VS::get_singleton()->shader_create();
	shader_data.users = 1;

	VS::get_singleton()->shader_set_code(shader_data.shader, code);

	shader_map[mk] = shader_data;

	VS::get_singleton()->material_set_shader(_get_material(), shader_data.shader);
}

void ParticlesMaterial::flush_changes() {

	if (material_mutex)
		material_mutex->lock();

	while (dirty_materials.first()) {

		dirty_materials.first()->self()->_update_shader();
	}

	if (material_mutex)
		material_mutex->unlock();
}

void ParticlesMaterial::_queue_shader_change() {

	if (material_mutex)
		material_mutex->lock();

	if (!element.in_list()) {
		dirty_materials.add(&element);
	}

	if (material_mutex)
		material_mutex->unlock();
}

bool ParticlesMaterial::_is_shader_dirty() const {

	bool dirty = false;

	if (material_mutex)
		material_mutex->lock();

	dirty = element.in_list();

	if (material_mutex)
		material_mutex->unlock();

	return dirty;
}

void ParticlesMaterial::set_spread(float p_spread) {

	spread = p_spread;
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->spread, p_spread);
}

float ParticlesMaterial::get_spread() const {

	return spread;
}

void ParticlesMaterial::set_flatness(float p_flatness) {

	flatness = p_flatness;
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->flatness, p_flatness);
}
float ParticlesMaterial::get_flatness() const {

	return flatness;
}

void ParticlesMaterial::set_param(Parameter p_param, float p_value) {

	ERR_FAIL_INDEX(p_param, PARAM_MAX);

	parameters[p_param] = p_value;

	switch (p_param) {
		case PARAM_INITIAL_LINEAR_VELOCITY: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->initial_linear_velocity, p_value);
		} break;
		case PARAM_ANGULAR_VELOCITY: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->angular_velocity, p_value);
		} break;
		case PARAM_ORBIT_VELOCITY: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->orbit_velocity, p_value);
		} break;
		case PARAM_LINEAR_ACCEL: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->linear_accel, p_value);
		} break;
		case PARAM_RADIAL_ACCEL: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->radial_accel, p_value);
		} break;
		case PARAM_TANGENTIAL_ACCEL: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->tangent_accel, p_value);
		} break;
		case PARAM_DAMPING: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->damping, p_value);
		} break;
		case PARAM_ANGLE: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->initial_angle, p_value);
		} break;
		case PARAM_SCALE: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->scale, p_value);
		} break;
		case PARAM_HUE_VARIATION: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->hue_variation, p_value);
		} break;
		case PARAM_ANIM_SPEED: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->anim_speed, p_value);
		} break;
		case PARAM_ANIM_OFFSET: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->anim_offset, p_value);
		} break;
		case PARAM_MAX: {
		};
	}
}
float ParticlesMaterial::get_param(Parameter p_param) const {

	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);

	return parameters[p_param];
}

void ParticlesMaterial::set_param_randomness(Parameter p_param, float p_value) {

	ERR_FAIL_INDEX(p_param, PARAM_MAX);

	randomness[p_param] = p_value;

	switch (p_param) {
		case PARAM_INITIAL_LINEAR_VELOCITY: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->initial_linear_velocity_random, p_value);
		} break;
		case PARAM_ANGULAR_VELOCITY: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->angular_velocity_random, p_value);
		} break;
		case PARAM_ORBIT_VELOCITY: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->orbit_velocity_random, p_value);
		} break;
		case PARAM_LINEAR_ACCEL: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->linear_accel_random, p_value);
		} break;
		case PARAM_RADIAL_ACCEL: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->radial_accel_random, p_value);
		} break;
		case PARAM_TANGENTIAL_ACCEL: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->tangent_accel_random, p_value);
		} break;
		case PARAM_DAMPING: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->damping_random, p_value);
		} break;
		case PARAM_ANGLE: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->initial_angle_random, p_value);
		} break;
		case PARAM_SCALE: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->scale_random, p_value);
		} break;
		case PARAM_HUE_VARIATION: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->hue_variation_random, p_value);
		} break;
		case PARAM_ANIM_SPEED: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->anim_speed_random, p_value);
		} break;
		case PARAM_ANIM_OFFSET: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->anim_offset_random, p_value);
		} break;
		case PARAM_MAX: {
		};
	}
}
float ParticlesMaterial::get_param_randomness(Parameter p_param) const {

	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);

	return randomness[p_param];
}

static void _adjust_curve_range(const Ref<Texture> &p_texture, float p_min, float p_max) {

	Ref<CurveTexture> curve_tex = p_texture;
	if (!curve_tex.is_valid())
		return;

	curve_tex->ensure_default_setup(p_min, p_max);
}

void ParticlesMaterial::set_param_texture(Parameter p_param, const Ref<Texture> &p_texture) {

	ERR_FAIL_INDEX(p_param, PARAM_MAX);

	tex_parameters[p_param] = p_texture;

	switch (p_param) {
		case PARAM_INITIAL_LINEAR_VELOCITY: {
			//do none for this one
		} break;
		case PARAM_ANGULAR_VELOCITY: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->angular_velocity_texture, p_texture);
			_adjust_curve_range(p_texture, -360, 360);
		} break;
		case PARAM_ORBIT_VELOCITY: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->orbit_velocity_texture, p_texture);
			_adjust_curve_range(p_texture, -500, 500);
		} break;
		case PARAM_LINEAR_ACCEL: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->linear_accel_texture, p_texture);
			_adjust_curve_range(p_texture, -200, 200);
		} break;
		case PARAM_RADIAL_ACCEL: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->radial_accel_texture, p_texture);
			_adjust_curve_range(p_texture, -200, 200);
		} break;
		case PARAM_TANGENTIAL_ACCEL: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->tangent_accel_texture, p_texture);
			_adjust_curve_range(p_texture, -200, 200);
		} break;
		case PARAM_DAMPING: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->damping_texture, p_texture);
			_adjust_curve_range(p_texture, 0, 100);
		} break;
		case PARAM_ANGLE: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->angle_texture, p_texture);
			_adjust_curve_range(p_texture, -360, 360);
		} break;
		case PARAM_SCALE: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->scale_texture, p_texture);

			Ref<CurveTexture> curve_tex = p_texture;
			if (curve_tex.is_valid()) {
				curve_tex->ensure_default_setup();
			}

		} break;
		case PARAM_HUE_VARIATION: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->hue_variation_texture, p_texture);
			_adjust_curve_range(p_texture, -1, 1);
		} break;
		case PARAM_ANIM_SPEED: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->anim_speed_texture, p_texture);
			_adjust_curve_range(p_texture, 0, 200);
		} break;
		case PARAM_ANIM_OFFSET: {
			VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->anim_offset_texture, p_texture);
		} break;
		case PARAM_MAX: {
		};
	}

	_queue_shader_change();
}
Ref<Texture> ParticlesMaterial::get_param_texture(Parameter p_param) const {

	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, Ref<Texture>());

	return tex_parameters[p_param];
}

void ParticlesMaterial::set_color(const Color &p_color) {

	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->color, p_color);
	color = p_color;
}

Color ParticlesMaterial::get_color() const {

	return color;
}

void ParticlesMaterial::set_color_ramp(const Ref<Texture> &p_texture) {

	color_ramp = p_texture;
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->color_ramp, p_texture);
	_queue_shader_change();
	_change_notify();
}

Ref<Texture> ParticlesMaterial::get_color_ramp() const {

	return color_ramp;
}

void ParticlesMaterial::set_flag(Flags p_flag, bool p_enable) {
	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	flags[p_flag] = p_enable;
	_queue_shader_change();
}

bool ParticlesMaterial::get_flag(Flags p_flag) const {
	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags[p_flag];
}

void ParticlesMaterial::set_emission_shape(EmissionShape p_shape) {

	emission_shape = p_shape;
	_change_notify();
	_queue_shader_change();
}

void ParticlesMaterial::set_emission_sphere_radius(float p_radius) {

	emission_sphere_radius = p_radius;
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->emission_sphere_radius, p_radius);
}

void ParticlesMaterial::set_emission_box_extents(Vector3 p_extents) {

	emission_box_extents = p_extents;
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->emission_box_extents, p_extents);
}

void ParticlesMaterial::set_emission_point_texture(const Ref<Texture> &p_points) {

	emission_point_texture = p_points;
	RID texture;
	if (p_points.is_valid())
		texture = p_points->get_rid();
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->emission_texture_points, texture);
}

void ParticlesMaterial::set_emission_normal_texture(const Ref<Texture> &p_normals) {

	emission_normal_texture = p_normals;
	RID texture;
	if (p_normals.is_valid())
		texture = p_normals->get_rid();
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->emission_texture_normal, texture);
}

void ParticlesMaterial::set_emission_color_texture(const Ref<Texture> &p_colors) {

	emission_color_texture = p_colors;
	RID texture;
	if (p_colors.is_valid())
		texture = p_colors->get_rid();
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->emission_texture_color, texture);
	_queue_shader_change();
}

void ParticlesMaterial::set_emission_point_count(int p_count) {

	emission_point_count = p_count;
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->emission_texture_point_count, p_count);
}

ParticlesMaterial::EmissionShape ParticlesMaterial::get_emission_shape() const {

	return emission_shape;
}

float ParticlesMaterial::get_emission_sphere_radius() const {

	return emission_sphere_radius;
}
Vector3 ParticlesMaterial::get_emission_box_extents() const {

	return emission_box_extents;
}
Ref<Texture> ParticlesMaterial::get_emission_point_texture() const {

	return emission_point_texture;
}
Ref<Texture> ParticlesMaterial::get_emission_normal_texture() const {

	return emission_normal_texture;
}

Ref<Texture> ParticlesMaterial::get_emission_color_texture() const {

	return emission_color_texture;
}

int ParticlesMaterial::get_emission_point_count() const {

	return emission_point_count;
}

void ParticlesMaterial::set_trail_divisor(int p_divisor) {

	trail_divisor = p_divisor;
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->trail_divisor, p_divisor);
}

int ParticlesMaterial::get_trail_divisor() const {

	return trail_divisor;
}

void ParticlesMaterial::set_trail_size_modifier(const Ref<CurveTexture> &p_trail_size_modifier) {

	trail_size_modifier = p_trail_size_modifier;

	Ref<CurveTexture> curve = trail_size_modifier;
	if (curve.is_valid()) {
		curve->ensure_default_setup();
	}

	RID texture;
	if (p_trail_size_modifier.is_valid())
		texture = p_trail_size_modifier->get_rid();
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->trail_size_modifier, texture);
	_queue_shader_change();
}

Ref<CurveTexture> ParticlesMaterial::get_trail_size_modifier() const {

	return trail_size_modifier;
}

void ParticlesMaterial::set_trail_color_modifier(const Ref<GradientTexture> &p_trail_color_modifier) {

	trail_color_modifier = p_trail_color_modifier;
	RID texture;
	if (p_trail_color_modifier.is_valid())
		texture = p_trail_color_modifier->get_rid();
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->trail_color_modifier, texture);
	_queue_shader_change();
}

Ref<GradientTexture> ParticlesMaterial::get_trail_color_modifier() const {

	return trail_color_modifier;
}

void ParticlesMaterial::set_gravity(const Vector3 &p_gravity) {

	gravity = p_gravity;
	Vector3 gset = gravity;
	if (gset == Vector3()) {
		gset = Vector3(0, -0.000001, 0); //as gravity is used as upvector in some calculations
	}
	VisualServer::get_singleton()->material_set_param(_get_material(), shader_names->gravity, gset);
}

Vector3 ParticlesMaterial::get_gravity() const {

	return gravity;
}

RID ParticlesMaterial::get_shader_rid() const {

	ERR_FAIL_COND_V(!shader_map.has(current_key), RID());
	return shader_map[current_key].shader;
}

void ParticlesMaterial::_validate_property(PropertyInfo &property) const {

	if (property.name == "color" && color_ramp.is_valid()) {
		property.usage = 0;
	}

	if (property.name == "emission_sphere_radius" && emission_shape != EMISSION_SHAPE_SPHERE) {
		property.usage = 0;
	}

	if (property.name == "emission_box_extents" && emission_shape != EMISSION_SHAPE_BOX) {
		property.usage = 0;
	}

	if ((property.name == "emission_point_texture" || property.name == "emission_color_texture") && (emission_shape < EMISSION_SHAPE_POINTS)) {
		property.usage = 0;
	}

	if (property.name == "emission_normal_texture" && emission_shape != EMISSION_SHAPE_DIRECTED_POINTS) {
		property.usage = 0;
	}

	if (property.name == "emission_point_count" && (emission_shape != EMISSION_SHAPE_POINTS && emission_shape != EMISSION_SHAPE_DIRECTED_POINTS)) {
		property.usage = 0;
	}
}

void ParticlesMaterial::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_spread", "degrees"), &ParticlesMaterial::set_spread);
	ClassDB::bind_method(D_METHOD("get_spread"), &ParticlesMaterial::get_spread);

	ClassDB::bind_method(D_METHOD("set_flatness", "amount"), &ParticlesMaterial::set_flatness);
	ClassDB::bind_method(D_METHOD("get_flatness"), &ParticlesMaterial::get_flatness);

	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &ParticlesMaterial::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &ParticlesMaterial::get_param);

	ClassDB::bind_method(D_METHOD("set_param_randomness", "param", "randomness"), &ParticlesMaterial::set_param_randomness);
	ClassDB::bind_method(D_METHOD("get_param_randomness", "param"), &ParticlesMaterial::get_param_randomness);

	ClassDB::bind_method(D_METHOD("set_param_texture", "param", "texture"), &ParticlesMaterial::set_param_texture);
	ClassDB::bind_method(D_METHOD("get_param_texture", "param"), &ParticlesMaterial::get_param_texture);

	ClassDB::bind_method(D_METHOD("set_color", "color"), &ParticlesMaterial::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &ParticlesMaterial::get_color);

	ClassDB::bind_method(D_METHOD("set_color_ramp", "ramp"), &ParticlesMaterial::set_color_ramp);
	ClassDB::bind_method(D_METHOD("get_color_ramp"), &ParticlesMaterial::get_color_ramp);

	ClassDB::bind_method(D_METHOD("set_flag", "flag", "enable"), &ParticlesMaterial::set_flag);
	ClassDB::bind_method(D_METHOD("get_flag", "flag"), &ParticlesMaterial::get_flag);

	ClassDB::bind_method(D_METHOD("set_emission_shape", "shape"), &ParticlesMaterial::set_emission_shape);
	ClassDB::bind_method(D_METHOD("get_emission_shape"), &ParticlesMaterial::get_emission_shape);

	ClassDB::bind_method(D_METHOD("set_emission_sphere_radius", "radius"), &ParticlesMaterial::set_emission_sphere_radius);
	ClassDB::bind_method(D_METHOD("get_emission_sphere_radius"), &ParticlesMaterial::get_emission_sphere_radius);

	ClassDB::bind_method(D_METHOD("set_emission_box_extents", "extents"), &ParticlesMaterial::set_emission_box_extents);
	ClassDB::bind_method(D_METHOD("get_emission_box_extents"), &ParticlesMaterial::get_emission_box_extents);

	ClassDB::bind_method(D_METHOD("set_emission_point_texture", "texture"), &ParticlesMaterial::set_emission_point_texture);
	ClassDB::bind_method(D_METHOD("get_emission_point_texture"), &ParticlesMaterial::get_emission_point_texture);

	ClassDB::bind_method(D_METHOD("set_emission_normal_texture", "texture"), &ParticlesMaterial::set_emission_normal_texture);
	ClassDB::bind_method(D_METHOD("get_emission_normal_texture"), &ParticlesMaterial::get_emission_normal_texture);

	ClassDB::bind_method(D_METHOD("set_emission_color_texture", "texture"), &ParticlesMaterial::set_emission_color_texture);
	ClassDB::bind_method(D_METHOD("get_emission_color_texture"), &ParticlesMaterial::get_emission_color_texture);

	ClassDB::bind_method(D_METHOD("set_emission_point_count", "point_count"), &ParticlesMaterial::set_emission_point_count);
	ClassDB::bind_method(D_METHOD("get_emission_point_count"), &ParticlesMaterial::get_emission_point_count);

	ClassDB::bind_method(D_METHOD("set_trail_divisor", "divisor"), &ParticlesMaterial::set_trail_divisor);
	ClassDB::bind_method(D_METHOD("get_trail_divisor"), &ParticlesMaterial::get_trail_divisor);

	ClassDB::bind_method(D_METHOD("set_trail_size_modifier", "texture"), &ParticlesMaterial::set_trail_size_modifier);
	ClassDB::bind_method(D_METHOD("get_trail_size_modifier"), &ParticlesMaterial::get_trail_size_modifier);

	ClassDB::bind_method(D_METHOD("set_trail_color_modifier", "texture"), &ParticlesMaterial::set_trail_color_modifier);
	ClassDB::bind_method(D_METHOD("get_trail_color_modifier"), &ParticlesMaterial::get_trail_color_modifier);

	ClassDB::bind_method(D_METHOD("get_gravity"), &ParticlesMaterial::get_gravity);
	ClassDB::bind_method(D_METHOD("set_gravity", "accel_vec"), &ParticlesMaterial::set_gravity);

	ADD_GROUP("Trail", "trail_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "trail_divisor", PROPERTY_HINT_RANGE, "1,1000000,1"), "set_trail_divisor", "get_trail_divisor");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "trail_size_modifier", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_trail_size_modifier", "get_trail_size_modifier");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "trail_color_modifier", PROPERTY_HINT_RESOURCE_TYPE, "GradientTexture"), "set_trail_color_modifier", "get_trail_color_modifier");
	ADD_GROUP("Emission Shape", "emission_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "emission_shape", PROPERTY_HINT_ENUM, "Point,Sphere,Box,Points,Directed Points"), "set_emission_shape", "get_emission_shape");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "emission_sphere_radius", PROPERTY_HINT_RANGE, "0.01,128,0.01"), "set_emission_sphere_radius", "get_emission_sphere_radius");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "emission_box_extents"), "set_emission_box_extents", "get_emission_box_extents");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "emission_point_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_emission_point_texture", "get_emission_point_texture");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "emission_normal_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_emission_normal_texture", "get_emission_normal_texture");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "emission_color_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_emission_color_texture", "get_emission_color_texture");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "emission_point_count", PROPERTY_HINT_RANGE, "0,1000000,1"), "set_emission_point_count", "get_emission_point_count");
	ADD_GROUP("Flags", "flag_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flag_align_y"), "set_flag", "get_flag", FLAG_ALIGN_Y_TO_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flag_rotate_y"), "set_flag", "get_flag", FLAG_ROTATE_Y);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flag_disable_z"), "set_flag", "get_flag", FLAG_DISABLE_Z);
	ADD_GROUP("Spread", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "spread", PROPERTY_HINT_RANGE, "0,180,0.01"), "set_spread", "get_spread");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "flatness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_flatness", "get_flatness");
	ADD_GROUP("Gravity", "");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
	ADD_GROUP("Initial Velocity", "initial_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "initial_velocity", PROPERTY_HINT_RANGE, "0,1000,0.01"), "set_param", "get_param", PARAM_INITIAL_LINEAR_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "initial_velocity_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_INITIAL_LINEAR_VELOCITY);
	ADD_GROUP("Angular Velocity", "angular_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_velocity", PROPERTY_HINT_RANGE, "-360,360,0.01"), "set_param", "get_param", PARAM_ANGULAR_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_velocity_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_ANGULAR_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "angular_velocity_curve", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_param_texture", "get_param_texture", PARAM_ANGULAR_VELOCITY);
	ADD_GROUP("Orbit Velocity", "orbit_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "orbit_velocity", PROPERTY_HINT_RANGE, "-1000,1000,0.01"), "set_param", "get_param", PARAM_ORBIT_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "orbit_velocity_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_ORBIT_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "orbit_velocity_curve", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_param_texture", "get_param_texture", PARAM_ORBIT_VELOCITY);
	ADD_GROUP("Linear Accel", "linear_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_accel", PROPERTY_HINT_RANGE, "-100,100,0.01"), "set_param", "get_param", PARAM_LINEAR_ACCEL);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_accel_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_LINEAR_ACCEL);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "linear_accel_curve", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_param_texture", "get_param_texture", PARAM_LINEAR_ACCEL);
	ADD_GROUP("Radial Accel", "radial_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "radial_accel", PROPERTY_HINT_RANGE, "-100,100,0.01"), "set_param", "get_param", PARAM_RADIAL_ACCEL);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "radial_accel_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_RADIAL_ACCEL);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "radial_accel_curve", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_param_texture", "get_param_texture", PARAM_RADIAL_ACCEL);
	ADD_GROUP("Tangential Accel", "tangential_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "tangential_accel", PROPERTY_HINT_RANGE, "-100,100,0.01"), "set_param", "get_param", PARAM_TANGENTIAL_ACCEL);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "tangential_accel_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_TANGENTIAL_ACCEL);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "tangential_accel_curve", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_param_texture", "get_param_texture", PARAM_TANGENTIAL_ACCEL);
	ADD_GROUP("Damping", "");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "damping", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_param", "get_param", PARAM_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "damping_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "damping_curve", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_param_texture", "get_param_texture", PARAM_DAMPING);
	ADD_GROUP("Angle", "");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angle", PROPERTY_HINT_RANGE, "-720,720,0.1"), "set_param", "get_param", PARAM_ANGLE);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angle_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_ANGLE);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "angle_curve", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_param_texture", "get_param_texture", PARAM_ANGLE);
	ADD_GROUP("Scale", "");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "scale", PROPERTY_HINT_RANGE, "0,1000,0.01"), "set_param", "get_param", PARAM_SCALE);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "scale_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_SCALE);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "scale_curve", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_param_texture", "get_param_texture", PARAM_SCALE);
	ADD_GROUP("Color", "");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "color_ramp", PROPERTY_HINT_RESOURCE_TYPE, "GradientTexture"), "set_color_ramp", "get_color_ramp");

	ADD_GROUP("Hue Variation", "hue_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "hue_variation", PROPERTY_HINT_RANGE, "-1,1,0.1"), "set_param", "get_param", PARAM_HUE_VARIATION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "hue_variation_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_HUE_VARIATION);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "hue_variation_curve", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_param_texture", "get_param_texture", PARAM_HUE_VARIATION);
	ADD_GROUP("Animation", "anim_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "anim_speed", PROPERTY_HINT_RANGE, "0,128,0.01"), "set_param", "get_param", PARAM_ANIM_SPEED);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "anim_speed_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_ANIM_SPEED);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "anim_speed_curve", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_param_texture", "get_param_texture", PARAM_ANIM_SPEED);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "anim_offset", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param", "get_param", PARAM_ANIM_OFFSET);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "anim_offset_random", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param_randomness", "get_param_randomness", PARAM_ANIM_OFFSET);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "anim_offset_curve", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_param_texture", "get_param_texture", PARAM_ANIM_OFFSET);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "anim_loop"), "set_flag", "get_flag", FLAG_ANIM_LOOP);

	BIND_ENUM_CONSTANT(PARAM_INITIAL_LINEAR_VELOCITY);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_VELOCITY);
	BIND_ENUM_CONSTANT(PARAM_ORBIT_VELOCITY);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_ACCEL);
	BIND_ENUM_CONSTANT(PARAM_RADIAL_ACCEL);
	BIND_ENUM_CONSTANT(PARAM_TANGENTIAL_ACCEL);
	BIND_ENUM_CONSTANT(PARAM_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_ANGLE);
	BIND_ENUM_CONSTANT(PARAM_SCALE);
	BIND_ENUM_CONSTANT(PARAM_HUE_VARIATION);
	BIND_ENUM_CONSTANT(PARAM_ANIM_SPEED);
	BIND_ENUM_CONSTANT(PARAM_ANIM_OFFSET);
	BIND_ENUM_CONSTANT(PARAM_MAX);

	BIND_ENUM_CONSTANT(FLAG_ALIGN_Y_TO_VELOCITY);
	BIND_ENUM_CONSTANT(FLAG_ROTATE_Y);
	BIND_ENUM_CONSTANT(FLAG_MAX);

	BIND_ENUM_CONSTANT(EMISSION_SHAPE_POINT);
	BIND_ENUM_CONSTANT(EMISSION_SHAPE_SPHERE);
	BIND_ENUM_CONSTANT(EMISSION_SHAPE_BOX);
	BIND_ENUM_CONSTANT(EMISSION_SHAPE_POINTS);
	BIND_ENUM_CONSTANT(EMISSION_SHAPE_DIRECTED_POINTS);
}

ParticlesMaterial::ParticlesMaterial()
	: element(this) {

	set_spread(45);
	set_flatness(0);
	set_param(PARAM_INITIAL_LINEAR_VELOCITY, 1);
	set_param(PARAM_ORBIT_VELOCITY, 0);
	set_param(PARAM_LINEAR_ACCEL, 0);
	set_param(PARAM_RADIAL_ACCEL, 0);
	set_param(PARAM_TANGENTIAL_ACCEL, 0);
	set_param(PARAM_DAMPING, 0);
	set_param(PARAM_ANGLE, 0);
	set_param(PARAM_SCALE, 1);
	set_param(PARAM_HUE_VARIATION, 0);
	set_param(PARAM_ANIM_SPEED, 0);
	set_param(PARAM_ANIM_OFFSET, 0);
	set_emission_shape(EMISSION_SHAPE_POINT);
	set_emission_sphere_radius(1);
	set_emission_box_extents(Vector3(1, 1, 1));
	set_trail_divisor(1);
	set_gravity(Vector3(0, -9.8, 0));
	emission_point_count = 1;

	for (int i = 0; i < PARAM_MAX; i++) {
		set_param_randomness(Parameter(i), 0);
	}

	for (int i = 0; i < FLAG_MAX; i++) {
		flags[i] = false;
	}

	set_color(Color(1, 1, 1, 1));

	current_key.key = 0;
	current_key.invalid_key = 1;

	_queue_shader_change();
}

ParticlesMaterial::~ParticlesMaterial() {
}
