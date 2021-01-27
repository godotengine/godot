/*************************************************************************/
/*  godot_xr.h                                                           */
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

#ifndef GODOT_NATIVEXR_H
#define GODOT_NATIVEXR_H

#include <gdnative/gdnative.h>

#ifdef __cplusplus
extern "C" {
#endif

// For future versions of the API we should only add new functions at the end of the structure and use the
// version info to detect whether a call is available

// Use these to populate version in your plugin
#define GODOTVR_API_MAJOR 1
#define GODOTVR_API_MINOR 1

typedef struct {
	godot_gdnative_api_version version; /* version of our API */
	void *(*constructor)(godot_object *);
	void (*destructor)(void *);
	godot_string (*get_name)(const void *);
	godot_int (*get_capabilities)(const void *);
	godot_bool (*get_anchor_detection_is_enabled)(const void *);
	void (*set_anchor_detection_is_enabled)(void *, godot_bool);
	godot_bool (*is_stereo)(const void *);
	godot_bool (*is_initialized)(const void *);
	godot_bool (*initialize)(void *);
	void (*uninitialize)(void *);
	godot_vector2 (*get_render_targetsize)(const void *);
	godot_transform (*get_transform_for_eye)(void *, godot_int, godot_transform *);
	void (*fill_projection_for_eye)(void *, godot_float *, godot_int, godot_float, godot_float, godot_float);
	void (*commit_for_eye)(void *, godot_int, godot_rid *, godot_rect2 *);
	void (*process)(void *);
	godot_int (*get_external_texture_for_eye)(void *, godot_int);
	void (*notification)(void *, godot_int);
	godot_int (*get_camera_feed_id)(void *);
} godot_xr_interface_gdnative;

void GDAPI godot_xr_register_interface(const godot_xr_interface_gdnative *p_interface);

// helper functions to access XRServer data
godot_float GDAPI godot_xr_get_worldscale();
godot_transform GDAPI godot_xr_get_reference_frame();

// helper functions for rendering
void GDAPI godot_xr_blit(godot_int p_eye, godot_rid *p_render_target, godot_rect2 *p_rect);
godot_int GDAPI godot_xr_get_texid(godot_rid *p_render_target);

// helper functions for updating XR controllers
godot_int GDAPI godot_xr_add_controller(char *p_device_name, godot_int p_hand, godot_bool p_tracks_orientation, godot_bool p_tracks_position);
void GDAPI godot_xr_remove_controller(godot_int p_controller_id);
void GDAPI godot_xr_set_controller_transform(godot_int p_controller_id, godot_transform *p_transform, godot_bool p_tracks_orientation, godot_bool p_tracks_position);
void GDAPI godot_xr_set_controller_button(godot_int p_controller_id, godot_int p_button, godot_bool p_is_pressed);
void GDAPI godot_xr_set_controller_axis(godot_int p_controller_id, godot_int p_axis, godot_float p_value, godot_bool p_can_be_negative);
godot_float GDAPI godot_xr_get_controller_rumble(godot_int p_controller_id);

#ifdef __cplusplus
}
#endif

#endif /* !GODOT_NATIVEXR_H */
