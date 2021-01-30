/*************************************************************************/
/*  godot_js.h                                                           */
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

#ifndef GODOT_JS_H
#define GODOT_JS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stddef.h"

// Config
extern void godot_js_config_locale_get(char *p_ptr, int p_ptr_max);
extern void godot_js_config_canvas_id_get(char *p_ptr, int p_ptr_max);

// OS
extern void godot_js_os_finish_async(void (*p_callback)());
extern void godot_js_os_request_quit_cb(void (*p_callback)());
extern int godot_js_os_fs_is_persistent();
extern void godot_js_os_fs_sync(void (*p_callback)());
extern int godot_js_os_execute(const char *p_json);
extern void godot_js_os_shell_open(const char *p_uri);
extern int godot_js_os_hw_concurrency_get();

// Display
extern int godot_js_display_screen_dpi_get();
extern double godot_js_display_pixel_ratio_get();
extern void godot_js_display_alert(const char *p_text);
extern int godot_js_display_touchscreen_is_available();
extern int godot_js_display_is_swap_ok_cancel();

// Display canvas
extern void godot_js_display_canvas_focus();
extern int godot_js_display_canvas_is_focused();

// Display window
extern void godot_js_display_desired_size_set(int p_width, int p_height);
extern int godot_js_display_size_update();
extern void godot_js_display_window_size_get(int32_t *p_x, int32_t *p_y);
extern void godot_js_display_screen_size_get(int32_t *p_x, int32_t *p_y);
extern int godot_js_display_fullscreen_request();
extern int godot_js_display_fullscreen_exit();
extern void godot_js_display_compute_position(int p_x, int p_y, int32_t *r_x, int32_t *r_y);
extern void godot_js_display_window_title_set(const char *p_text);
extern void godot_js_display_window_icon_set(const uint8_t *p_ptr, int p_len);

// Display clipboard
extern int godot_js_display_clipboard_set(const char *p_text);
extern int godot_js_display_clipboard_get(void (*p_callback)(const char *p_text));

// Display cursor
extern void godot_js_display_cursor_set_shape(const char *p_cursor);
extern int godot_js_display_cursor_is_hidden();
extern void godot_js_display_cursor_set_custom_shape(const char *p_shape, const uint8_t *p_ptr, int p_len, int p_hotspot_x, int p_hotspot_y);
extern void godot_js_display_cursor_set_visible(int p_visible);

// Display gamepad
extern char *godot_js_display_gamepad_cb(void (*p_on_change)(int p_index, int p_connected, const char *p_id, const char *p_guid));
extern int godot_js_display_gamepad_sample();
extern int godot_js_display_gamepad_sample_count();
extern int godot_js_display_gamepad_sample_get(int p_idx, float r_btns[16], int32_t *r_btns_num, float r_axes[10], int32_t *r_axes_num, int32_t *r_standard);

// Display listeners
extern void godot_js_display_notification_cb(void (*p_callback)(int p_notification), int p_enter, int p_exit, int p_in, int p_out);
extern void godot_js_display_paste_cb(void (*p_callback)(const char *p_text));
extern void godot_js_display_drop_files_cb(void (*p_callback)(char **p_filev, int p_filec));
extern void godot_js_display_setup_canvas(int p_width, int p_height, int p_fullscreen);
#ifdef __cplusplus
}
#endif

#endif /* GODOT_JS_H */
