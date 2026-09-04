/**************************************************************************/
/*  rendering_server.compat.inc                                           */
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

#ifndef DISABLE_DEPRECATED

#include "rendering_server.h"

#include "core/object/class_db.h"

void RenderingServer::_multimesh_allocate_data_bind_compat_99455(RID p_multimesh, int p_instances, RSE::MultimeshTransformFormat p_transform_format, bool p_use_colors, bool p_use_custom_data) {
	multimesh_allocate_data(p_multimesh, p_instances, p_transform_format, p_use_colors, p_use_custom_data, false);
}

void RenderingServer::_environment_set_fog_bind_compat_84792(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective, float p_sky_affect) {
	environment_set_fog(p_env, p_enable, p_light_color, p_light_energy, p_sun_scatter, p_density, p_height, p_height_density, p_aerial_perspective, p_sky_affect, RSE::EnvironmentFogMode::ENV_FOG_MODE_EXPONENTIAL);
}

void RenderingServer::_canvas_item_add_multiline_bind_compat_84523(RID p_item, const Vector<Point2> &p_points, const Vector<Color> &p_colors, float p_width) {
	canvas_item_add_multiline(p_item, p_points, p_colors, p_width, false);
}

void RenderingServer::_canvas_item_add_rect_bind_compat_84523(RID p_item, const Rect2 &p_rect, const Color &p_color) {
	canvas_item_add_rect(p_item, p_rect, p_color, false);
}

void RenderingServer::_canvas_item_add_circle_bind_compat_84523(RID p_item, const Point2 &p_pos, float p_radius, const Color &p_color) {
	canvas_item_add_circle(p_item, p_pos, p_radius, p_color, false);
}

void RenderingServer::_instance_set_interpolated_bind_compat_104269(RID p_instance, bool p_interpolated) {
	WARN_PRINT_ONCE("instance_set_interpolated() is deprecated.");
}

void RenderingServer::_instance_reset_physics_interpolation_bind_compat_104269(RID p_instance) {
	WARN_PRINT_ONCE("instance_reset_physics_interpolation() is deprecated.");
}

void RenderingServer::_viewport_set_size_bind_compat_115799(RID p_viewport, int p_width, int p_height) {
	viewport_set_size(p_viewport, p_width, p_height, 1);
}

void RenderingServer::_particles_request_process_time_bind_compat_109142(RID p_particles, real_t p_request_process_time) {
	particles_request_process_time(p_particles, p_request_process_time, 0.0);
}

void RenderingServer::light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) {
	light_set_max_hddagi_cascade(p_light, p_cascade);
}

void RenderingServer::sdfgi_reset() {
	hddagi_reset();
}

void RenderingServer::environment_set_sdfgi(RID p_env, bool p_enable, int p_cascades, float p_min_cell_size, RSE::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) {
	ERR_FAIL_INDEX(int(p_y_scale), int(RSE::ENV_SDFGI_Y_SCALE_100_PERCENT) + 1);
	static constexpr RSE::EnvironmentHDDAGICascadeFormat cascade_formats[3] = {
		RSE::ENV_HDDAGI_CASCADE_FORMAT_16x16x16_50_PERCENT_HEIGHT,
		RSE::ENV_HDDAGI_CASCADE_FORMAT_16x16x16_75_PERCENT_HEIGHT,
		RSE::ENV_HDDAGI_CASCADE_FORMAT_16x16x16,
	};
	environment_set_hddagi(p_env, p_enable, p_cascades, cascade_formats[p_y_scale], p_min_cell_size, p_use_occlusion, p_bounce_feedback, p_read_sky, p_energy, p_normal_bias, p_probe_bias, p_probe_bias, p_probe_bias, true, true);
}

void RenderingServer::environment_set_sdfgi_ray_count(RSE::EnvironmentSDFGIRayCount p_ray_count) {
	ERR_FAIL_INDEX(int(p_ray_count), int(RSE::ENV_SDFGI_RAY_COUNT_MAX));
	WARN_PRINT_ONCE("environment_set_sdfgi_ray_count() is deprecated and has no effect with HDDAGI.");
}

void RenderingServer::environment_set_sdfgi_frames_to_converge(RSE::EnvironmentSDFGIFramesToConverge p_frames) {
	ERR_FAIL_INDEX(int(p_frames), int(RSE::ENV_SDFGI_CONVERGE_MAX));
	static constexpr RSE::EnvironmentHDDAGIFramesToConverge convergence_mapping[RSE::ENV_SDFGI_CONVERGE_MAX] = {
		RSE::ENV_HDDAGI_CONVERGE_IN_6_FRAMES,
		RSE::ENV_HDDAGI_CONVERGE_IN_12_FRAMES,
		RSE::ENV_HDDAGI_CONVERGE_IN_18_FRAMES,
		RSE::ENV_HDDAGI_CONVERGE_IN_24_FRAMES,
		RSE::ENV_HDDAGI_CONVERGE_IN_24_FRAMES,
		RSE::ENV_HDDAGI_CONVERGE_IN_32_FRAMES,
	};
	environment_set_hddagi_frames_to_converge(convergence_mapping[p_frames]);
}

void RenderingServer::environment_set_sdfgi_frames_to_update_light(RSE::EnvironmentSDFGIFramesToUpdateLight p_update) {
	ERR_FAIL_INDEX(int(p_update), int(RSE::ENV_SDFGI_UPDATE_LIGHT_MAX));
	environment_set_hddagi_frames_to_update_light(RSE::EnvironmentHDDAGIFramesToUpdateLight(p_update));
}

void RenderingServer::sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) {
	hddagi_set_debug_probe_select(p_position, p_dir);
}

void RenderingServer::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("multimesh_allocate_data", "multimesh", "instances", "transform_format", "color_format", "custom_data_format"), &RenderingServer::_multimesh_allocate_data_bind_compat_99455, DEFVAL(false), DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("environment_set_fog", "env", "enable", "light_color", "light_energy", "sun_scatter", "density", "height", "height_density", "aerial_perspective", "sky_affect"), &RenderingServer::_environment_set_fog_bind_compat_84792);
	ClassDB::bind_compatibility_method(D_METHOD("canvas_item_add_multiline", "item", "points", "colors", "width"), &RenderingServer::_canvas_item_add_multiline_bind_compat_84523, DEFVAL(-1.0));
	ClassDB::bind_compatibility_method(D_METHOD("canvas_item_add_rect", "item", "rect", "color"), &RenderingServer::_canvas_item_add_rect_bind_compat_84523);
	ClassDB::bind_compatibility_method(D_METHOD("canvas_item_add_circle", "item", "pos", "radius", "color"), &RenderingServer::_canvas_item_add_circle_bind_compat_84523);
	ClassDB::bind_compatibility_method(D_METHOD("instance_set_interpolated", "instance", "interpolated"), &RenderingServer::_instance_set_interpolated_bind_compat_104269);
	ClassDB::bind_compatibility_method(D_METHOD("instance_reset_physics_interpolation", "instance"), &RenderingServer::_instance_reset_physics_interpolation_bind_compat_104269);
	ClassDB::bind_compatibility_method(D_METHOD("viewport_set_size", "viewport", "width", "height"), &RenderingServer::_viewport_set_size_bind_compat_115799);
	ClassDB::bind_compatibility_method(D_METHOD("particles_request_process_time", "particles", "process_time"), &RenderingServer::_particles_request_process_time_bind_compat_109142);

	ClassDB::bind_method(D_METHOD("light_set_max_sdfgi_cascade", "light", "cascade"), &RenderingServer::light_set_max_sdfgi_cascade);
	ClassDB::bind_method(D_METHOD("environment_set_sdfgi", "env", "enable", "cascades", "min_cell_size", "y_scale", "use_occlusion", "bounce_feedback", "read_sky", "energy", "normal_bias", "probe_bias"), &RenderingServer::environment_set_sdfgi);
	ClassDB::bind_method(D_METHOD("environment_set_sdfgi_ray_count", "ray_count"), &RenderingServer::environment_set_sdfgi_ray_count);
	ClassDB::bind_method(D_METHOD("environment_set_sdfgi_frames_to_converge", "frames"), &RenderingServer::environment_set_sdfgi_frames_to_converge);
	ClassDB::bind_method(D_METHOD("environment_set_sdfgi_frames_to_update_light", "frames"), &RenderingServer::environment_set_sdfgi_frames_to_update_light);

	BIND_ENUM_CONSTANT(RSE::VIEWPORT_DEBUG_DRAW_SDFGI);
	BIND_ENUM_CONSTANT(RSE::VIEWPORT_DEBUG_DRAW_SDFGI_PROBES);

	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_Y_SCALE_50_PERCENT);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_Y_SCALE_75_PERCENT);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_Y_SCALE_100_PERCENT);

	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_RAY_COUNT_4);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_RAY_COUNT_8);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_RAY_COUNT_16);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_RAY_COUNT_32);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_RAY_COUNT_64);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_RAY_COUNT_96);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_RAY_COUNT_128);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_RAY_COUNT_MAX);

	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_CONVERGE_IN_5_FRAMES);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_CONVERGE_IN_10_FRAMES);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_CONVERGE_IN_15_FRAMES);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_CONVERGE_IN_20_FRAMES);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_CONVERGE_IN_25_FRAMES);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_CONVERGE_IN_30_FRAMES);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_CONVERGE_MAX);

	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_UPDATE_LIGHT_IN_1_FRAME);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_UPDATE_LIGHT_IN_2_FRAMES);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_UPDATE_LIGHT_IN_4_FRAMES);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_UPDATE_LIGHT_IN_8_FRAMES);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_UPDATE_LIGHT_IN_16_FRAMES);
	BIND_ENUM_CONSTANT(RSE::ENV_SDFGI_UPDATE_LIGHT_MAX);
}

#endif // DISABLE_DEPRECATED
