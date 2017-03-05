/*************************************************************************/
/*  godot_c.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef GODOT_C_H
#define GODOT_C_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(GDAPI_BUILT_IN) || !defined(WINDOWS_ENABLED)
#define GDAPI
#elif defined(GDAPI_EXPORTS)
#define GDAPI __declspec(dllexport)
#else
#define GDAPI __declspec(dllimport)
#endif

#define GODOT_API_VERSION 1

typedef int godot_bool;

#define GODOT_FALSE 0
#define GODOT_TRUE 1

////// Image

#define GODOT_IMAGE_FORMAT_GRAYSCALE 0
#define GODOT_IMAGE_FORMAT_INTENSITY 1
#define GODOT_IMAGE_FORMAT_GRAYSCALE_ALPHA 2
#define GODOT_IMAGE_FORMAT_RGB 3
#define GODOT_IMAGE_FORMAT_RGBA 4
#define GODOT_IMAGE_FORMAT_INDEXED 5
#define GODOT_IMAGE_FORMAT_INDEXED_ALPHA 6
#define GODOT_IMAGE_FORMAT_YUV_422 7
#define GODOT_IMAGE_FORMAT_YUV_444 8
#define GODOT_IMAGE_FORMAT_BC1 9
#define GODOT_IMAGE_FORMAT_BC2 10
#define GODOT_IMAGE_FORMAT_BC3 11
#define GODOT_IMAGE_FORMAT_BC4 12
#define GODOT_IMAGE_FORMAT_BC5 13
#define GODOT_IMAGE_FORMAT_PVRTC2 14
#define GODOT_IMAGE_FORMAT_PVRTC2_ALPHA 15
#define GODOT_IMAGE_FORMAT_PVRTC4 16
#define GODOT_IMAGE_FORMAT_PVRTC4_ALPHA 17
#define GODOT_IMAGE_FORMAT_ETC 18
#define GODOT_IMAGE_FORMAT_ATC 19
#define GODOT_IMAGE_FORMAT_ATC_ALPHA_EXPLICIT 20
#define GODOT_IMAGE_FORMAT_ATC_ALPHA_INTERPOLATED 21

typedef void *godot_image;

godot_image GDAPI godot_image_create_empty();
godot_image GDAPI godot_image_create(int p_width, int p_height, int p_format, int p_use_mipmaps);
godot_image GDAPI godot_image_create_with_data(int p_width, int p_height, int p_format, int p_use_mipmaps, unsigned char *p_buffer);
int GDAPI godot_image_get_width(godot_image p_image);
int GDAPI godot_image_get_height(godot_image p_image);
int GDAPI godot_image_get_format(godot_image p_image);
int GDAPI godot_image_get_mipmap_count(godot_image p_image);
godot_image GDAPI godot_image_copy(godot_image p_image);
void GDAPI godot_image_free(godot_image p_image);

////// RID

typedef void *godot_rid;

godot_rid GDAPI godot_rid_create();
godot_rid GDAPI godot_rid_copy(godot_rid p_rid);
void GDAPI godot_rid_free(godot_rid p_rid);

////// Variant (forward declared)

typedef void *godot_variant;

////// Dictionary

typedef void *godot_dictionary;

godot_dictionary GDAPI godot_dictionary_create();
void GDAPI godot_dictionary_has(godot_dictionary p_dictionary, godot_variant p_key);
godot_variant GDAPI godot_dictionary_get(godot_dictionary p_dictionary, godot_variant p_key);
void GDAPI godot_dictionary_insert(godot_dictionary p_dictionary, godot_variant p_key, godot_variant p_value);
void GDAPI godot_dictionary_remove(godot_dictionary p_dictionary, godot_variant p_key);
void GDAPI godot_dictionary_clear(godot_dictionary p_dictionary);
int GDAPI godot_dictionary_get_size(godot_dictionary p_dictionary);
void GDAPI godot_dictionary_get_keys(godot_dictionary p_dictionary, godot_variant *p_keys);
godot_dictionary GDAPI godot_dictionary_copy(godot_dictionary p_dictionary);
void GDAPI godot_dictionary_free(godot_dictionary p_dictionary);

////// Array

typedef void *godot_array;

godot_array GDAPI godot_array_create();
godot_variant GDAPI godot_array_get(godot_array p_array, int p_index);
void GDAPI godot_array_set(godot_array p_array, int p_index, godot_variant p_value);
void GDAPI godot_array_resize(godot_array p_array, int p_size);
void GDAPI godot_array_insert(godot_array p_array, int p_position, godot_variant p_value);
void GDAPI godot_array_remove(godot_array p_array, int p_position);
void GDAPI godot_array_clear(godot_array p_array);
int GDAPI godot_array_get_size(godot_array p_array);
int GDAPI godot_array_find(godot_array p_array, godot_variant p_value, int p_from_pos = -1);
godot_array GDAPI godot_array_copy(godot_array p_array);
void GDAPI godot_array_free(godot_array p_array);

////// InputEvent

#define INPUT_EVENT_BUTTON_LEFT 1
#define INPUT_EVENT_BUTTON_RIGHT 2
#define INPUT_EVENT_BUTTON_MIDDLE 3
#define INPUT_EVENT_BUTTON_WHEEL_UP 4
#define INPUT_EVENT_BUTTON_WHEEL_DOWN 5
#define INPUT_EVENT_BUTTON_WHEEL_LEFT 6
#define INPUT_EVENT_BUTTON_WHEEL_RIGHT 7
#define INPUT_EVENT_BUTTON_MASK_LEFT (1 << (INPUT_EVENT_BUTTON_LEFT - 1))
#define INPUT_EVENT_BUTTON_MASK_RIGHT (1 << (INPUT_EVENT_BUTTON_RIGHT - 1))
#define INPUT_EVENT_BUTTON_MASK_MIDDLE (1 << (INPUT_EVENT_BUTTON_MIDDLE - 1))

#define INPUT_EVENT_TYPE_NONE 0
#define INPUT_EVENT_TYPE_KEY 1
#define INPUT_EVENT_TYPE_MOUSE_MOTION 2
#define INPUT_EVENT_TYPE_MOUSE_BUTTON 3
#define INPUT_EVENT_TYPE_JOYPAD_MOTION 4
#define INPUT_EVENT_TYPE_JOYPAD_BUTTON 5
#define INPUT_EVENT_TYPE_SCREEN_TOUCH 6
#define INPUT_EVENT_TYPE_SCREEN_DRAG 7
#define INPUT_EVENT_TYPE_ACTION 8

typedef void *godot_input_event;

godot_input_event GDAPI godot_input_event_create();
godot_input_event GDAPI godot_input_event_copy(godot_input_event p_input_event);
void GDAPI godot_input_event_free(godot_input_event p_input_event);

int GDAPI godot_input_event_get_type(godot_input_event p_event);
int GDAPI godot_input_event_get_device(godot_input_event p_event);

godot_bool GDAPI godot_input_event_mod_has_alt(godot_input_event p_event);
godot_bool GDAPI godot_input_event_mod_has_ctrl(godot_input_event p_event);
godot_bool GDAPI godot_input_event_mod_has_command(godot_input_event p_event);
godot_bool GDAPI godot_input_event_mod_has_shift(godot_input_event p_event);
godot_bool GDAPI godot_input_event_mod_has_meta(godot_input_event p_event);

int GDAPI godot_input_event_key_get_scancode(godot_input_event p_event);
int GDAPI godot_input_event_key_get_unicode(godot_input_event p_event);
godot_bool GDAPI godot_input_event_key_is_pressed(godot_input_event p_event);
godot_bool GDAPI godot_input_event_key_is_echo(godot_input_event p_event);

int GDAPI godot_input_event_mouse_get_x(godot_input_event p_event);
int GDAPI godot_input_event_mouse_get_y(godot_input_event p_event);
int GDAPI godot_input_event_mouse_get_global_x(godot_input_event p_event);
int GDAPI godot_input_event_mouse_get_global_y(godot_input_event p_event);
int GDAPI godot_input_event_mouse_get_button_mask(godot_input_event p_event);

int GDAPI godot_input_event_mouse_button_get_button_index(godot_input_event p_event);
godot_bool GDAPI godot_input_event_mouse_button_is_pressed(godot_input_event p_event);
godot_bool GDAPI godot_input_event_mouse_button_is_doubleclick(godot_input_event p_event);

int GDAPI godot_input_event_mouse_motion_get_relative_x(godot_input_event p_event);
int GDAPI godot_input_event_mouse_motion_get_relative_y(godot_input_event p_event);

int GDAPI godot_input_event_mouse_motion_get_speed_x(godot_input_event p_event);
int GDAPI godot_input_event_mouse_motion_get_speed_y(godot_input_event p_event);

int GDAPI godot_input_event_joypad_motion_get_axis(godot_input_event p_event);
float GDAPI godot_input_event_joypad_motion_get_axis_value(godot_input_event p_event);

int GDAPI godot_input_event_joypad_button_get_button_index(godot_input_event p_event);
godot_bool GDAPI godot_input_event_joypad_button_is_pressed(godot_input_event p_event);
float GDAPI godot_input_event_joypad_button_get_pressure(godot_input_event p_event);

int GDAPI godot_input_event_screen_touch_get_index(godot_input_event p_event);
int GDAPI godot_input_event_screen_touch_get_x(godot_input_event p_event);
int GDAPI godot_input_event_screen_touch_get_y(godot_input_event p_event);
int GDAPI godot_input_event_screen_touch_is_pressed(godot_input_event p_event);

int GDAPI godot_input_event_screen_drag_get_index(godot_input_event p_event);
int GDAPI godot_input_event_screen_drag_get_x(godot_input_event p_event);
int GDAPI godot_input_event_screen_drag_get_y(godot_input_event p_event);
int GDAPI godot_input_event_screen_drag_get_relative_x(godot_input_event p_event);
int GDAPI godot_input_event_screen_drag_get_relative_y(godot_input_event p_event);
int GDAPI godot_input_event_screen_drag_get_speed_x(godot_input_event p_event);
int GDAPI godot_input_event_screen_drag_get_speed_y(godot_input_event p_event);

int GDAPI godot_input_event_is_action(godot_input_event p_event, char *p_action);
int GDAPI godot_input_event_is_action_pressed(godot_input_event p_event, char *p_action);

////// ByteArray

typedef void *godot_byte_array;

godot_byte_array GDAPI godot_byte_array_create();
godot_byte_array GDAPI godot_byte_array_copy(godot_byte_array p_byte_array);
void GDAPI godot_byte_array_free(godot_byte_array p_byte_array);

int GDAPI godot_byte_array_get_size(godot_byte_array p_byte_array);
unsigned char GDAPI godot_byte_array_get(godot_byte_array p_byte_array, int p_index);
void GDAPI godot_byte_array_set(godot_byte_array p_byte_array, int p_index, unsigned char p_value);
void GDAPI godot_byte_array_remove(godot_byte_array p_byte_array, int p_index);
void GDAPI godot_byte_array_clear(godot_byte_array p_byte_array);

typedef void *godot_byte_array_lock;

godot_byte_array_lock GDAPI godot_byte_array_get_lock(godot_byte_array p_byte_array);
unsigned char GDAPI *godot_byte_array_lock_get_pointer(godot_byte_array_lock p_byte_array_lock);
void GDAPI godot_byte_array_lock_free(godot_byte_array_lock p_byte_array_lock);

godot_image GDAPI godot_image_create_with_array(int p_width, int p_height, int p_format, int p_use_mipmaps, godot_array p_array);
godot_byte_array GDAPI godot_image_get_data(godot_image p_image);

////// IntArray

typedef void *godot_int_array;

godot_int_array GDAPI godot_int_array_create();
godot_int_array GDAPI godot_int_array_copy(godot_int_array p_int_array);
void GDAPI godot_int_array_free(godot_int_array p_int_array);

int GDAPI godot_int_array_get_size(godot_int_array p_int_array);
int GDAPI godot_int_array_get(godot_int_array p_int_array, int p_index);
void GDAPI godot_int_array_set(godot_int_array p_int_array, int p_index, int p_value);
void GDAPI godot_int_array_remove(godot_int_array p_int_array, int p_index);
void GDAPI godot_int_array_clear(godot_int_array p_int_array);

typedef void *godot_int_array_lock;

godot_int_array_lock GDAPI godot_int_array_get_lock(godot_int_array p_int_array);
int GDAPI *godot_int_array_lock_get_pointer(godot_int_array_lock p_int_array_lock);
void GDAPI godot_int_array_lock_free(godot_int_array_lock p_int_array_lock);

////// RealArray

typedef void *godot_real_array;

godot_real_array GDAPI godot_real_array_create();
godot_real_array GDAPI godot_real_array_copy(godot_real_array p_real_array);
void GDAPI godot_real_array_free(godot_real_array p_real_array);

int GDAPI godot_real_array_get_size(godot_real_array p_real_array);
float GDAPI godot_real_array_get(godot_real_array p_real_array, int p_index);
void GDAPI godot_real_array_set(godot_real_array p_real_array, int p_index, float p_value);
void GDAPI godot_real_array_remove(godot_real_array p_real_array, int p_index);
void GDAPI godot_real_array_clear(godot_real_array p_real_array);

typedef void *godot_real_array_lock;

godot_real_array_lock GDAPI godot_real_array_get_lock(godot_real_array p_real_array);
float GDAPI *godot_real_array_lock_get_pointer(godot_real_array_lock p_real_array_lock);
void GDAPI godot_real_array_lock_free(godot_real_array_lock p_real_array_lock);

////// StringArray

typedef void *godot_string_array;

godot_string_array GDAPI godot_string_array_create();
godot_string_array GDAPI godot_string_array_copy(godot_string_array p_string_array);
void GDAPI godot_string_array_free(godot_string_array p_string_array);

int GDAPI godot_string_array_get_size(godot_string_array p_string_array);
int GDAPI godot_string_array_get(godot_string_array p_string_array, int p_index, unsigned char *p_string, int p_max_len);
void GDAPI godot_string_array_set(godot_string_array p_string_array, int p_index, unsigned char *p_string);
void GDAPI godot_string_array_remove(godot_string_array p_string_array, int p_index);
void GDAPI godot_string_array_clear(godot_string_array p_string_array);

////// Vector2Array

typedef void *godot_vector2_array;

godot_vector2_array GDAPI godot_vector2_array_create();
godot_vector2_array GDAPI godot_vector2_array_copy(godot_vector2_array p_vector2_array);
void GDAPI godot_vector2_array_free(godot_vector2_array p_vector2_array);

int GDAPI godot_vector2_array_get_size(godot_vector2_array p_vector2_array);
int GDAPI godot_vector2_array_get_stride(godot_vector2_array p_vector2_array);
void GDAPI godot_vector2_array_get(godot_vector2_array p_vector2_array, int p_index, float *p_vector2);
void GDAPI godot_vector2_array_set(godot_vector2_array p_vector2_array, int p_index, float *p_vector2);
void GDAPI godot_vector2_array_remove(godot_vector2_array p_vector2_array, int p_index);
void GDAPI godot_vector2_array_clear(godot_vector2_array p_vector2_array);

typedef void *godot_vector2_array_lock;

godot_vector2_array_lock GDAPI godot_vector2_array_get_lock(godot_vector2_array p_vector2_array);
float GDAPI *godot_vector2_array_lock_get_pointer(godot_vector2_array_lock p_vector2_array_lock);
void GDAPI godot_vector2_array_lock_free(godot_vector2_array_lock p_vector2_array_lock);

////// Vector3Array

typedef void *godot_vector3_array;

godot_vector3_array GDAPI godot_vector3_array_create();
godot_vector3_array GDAPI godot_vector3_array_copy(godot_vector3_array p_vector3_array);
void GDAPI godot_vector3_array_free(godot_vector3_array p_vector3_array);

int GDAPI godot_vector3_array_get_size(godot_vector3_array p_vector3_array);
int GDAPI godot_vector3_array_get_stride(godot_vector3_array p_vector3_array);
void GDAPI godot_vector3_array_get(godot_vector3_array p_vector3_array, int p_index, float *p_vector3);
void GDAPI godot_vector3_array_set(godot_vector3_array p_vector3_array, int p_index, float *p_vector3);
void GDAPI godot_vector3_array_remove(godot_vector3_array p_vector3_array, int p_index);
void GDAPI godot_vector3_array_clear(godot_vector3_array p_vector3_array);

typedef void *godot_vector3_array_lock;

godot_vector3_array_lock GDAPI godot_vector3_array_get_lock(godot_vector3_array p_vector3_array);
float GDAPI *godot_vector3_array_lock_get_pointer(godot_vector3_array_lock p_vector3_array_lock);
void GDAPI godot_vector3_array_lock_free(godot_vector3_array_lock p_vector3_array_lock);

////// ColorArray

typedef void *godot_color_array;

godot_color_array GDAPI godot_color_array_create();
godot_color_array GDAPI godot_color_array_copy(godot_color_array p_color_array);
void GDAPI godot_color_array_free(godot_color_array p_color_array);

int GDAPI godot_color_array_get_size(godot_color_array p_color_array);
int GDAPI godot_color_array_get_stride(godot_color_array p_color_array);
void GDAPI godot_color_array_get(godot_color_array p_color_array, int p_index, float *p_color);
void GDAPI godot_color_array_set(godot_color_array p_color_array, int p_index, float *p_color);
void GDAPI godot_color_array_remove(godot_color_array p_color_array, int p_index);
void GDAPI godot_color_array_clear(godot_color_array p_color_array);

typedef void *godot_color_array_lock;

godot_color_array_lock GDAPI godot_color_array_get_lock(godot_color_array p_color_array);
float GDAPI *godot_color_array_lock_get_pointer(godot_color_array_lock p_color_array_lock);
void GDAPI godot_color_array_lock_free(godot_color_array_lock p_color_array_lock);

////// Instance (forward declared)

typedef void *godot_instance;

////// Variant

#define GODOT_VARIANT_NIL 0
#define GODOT_VARIANT_BOOL 1
#define GODOT_VARIANT_INT 2
#define GODOT_VARIANT_REAL 3
#define GODOT_VARIANT_STRING 4
#define GODOT_VARIANT_VECTOR2 5
#define GODOT_VARIANT_RECT2 6
#define GODOT_VARIANT_VECTOR3 7
#define GODOT_VARIANT_MATRIX32 8
#define GODOT_VARIANT_PLANE 9
#define GODOT_VARIANT_QUAT 10
#define GODOT_VARIANT_AABB 11
#define GODOT_VARIANT_MATRIX3 12
#define GODOT_VARIANT_TRANSFORM 13
#define GODOT_VARIANT_COLOR 14
#define GODOT_VARIANT_IMAGE 15
#define GODOT_VARIANT_NODE_PATH 16
#define GODOT_VARIANT_RID 17
#define GODOT_VARIANT_OBJECT 18
#define GODOT_VARIANT_INPUT_EVENT 19
#define GODOT_VARIANT_DICTIONARY 20
#define GODOT_VARIANT_ARRAY 21
#define GODOT_VARIANT_BYTE_ARRAY 22
#define GODOT_VARIANT_INT_ARRAY 23
#define GODOT_VARIANT_REAL_ARRAY 24
#define GODOT_VARIANT_STRING_ARRAY 25
#define GODOT_VARIANT_VECTOR2_ARRAY 26
#define GODOT_VARIANT_VECTOR3_ARRAY 27
#define GODOT_VARIANT_COLOR_ARRAY 28
#define GODOT_VARIANT_MAX 29

godot_variant *godot_variant_new();

int GDAPI godot_variant_get_type(godot_variant p_variant);

void GDAPI godot_variant_set_null(godot_variant p_variant);
void GDAPI godot_variant_set_bool(godot_variant p_variant, godot_bool p_bool);
void GDAPI godot_variant_set_int(godot_variant p_variant, int p_int);
void GDAPI godot_variant_set_float(godot_variant p_variant, int p_float);
void GDAPI godot_variant_set_string(godot_variant p_variant, char *p_string);
void GDAPI godot_variant_set_vector2(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_set_rect2(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_set_vector3(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_set_matrix32(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_set_plane(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_set_aabb(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_set_matrix3(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_set_transform(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_set_color(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_set_image(godot_variant p_variant, godot_image *p_image);
void GDAPI godot_variant_set_node_path(godot_variant p_variant, char *p_path);
void GDAPI godot_variant_set_rid(godot_variant p_variant, char *p_path);
void GDAPI godot_variant_set_instance(godot_variant p_variant, godot_instance p_instance);
void GDAPI godot_variant_set_input_event(godot_variant p_variant, godot_input_event p_instance);
void GDAPI godot_variant_set_dictionary(godot_variant p_variant, godot_dictionary p_dictionary);
void GDAPI godot_variant_set_array(godot_variant p_variant, godot_array p_array);
void GDAPI godot_variant_set_byte_array(godot_variant p_variant, godot_byte_array p_array);
void GDAPI godot_variant_set_int_array(godot_variant p_variant, godot_byte_array p_array);
void GDAPI godot_variant_set_string_array(godot_variant p_variant, godot_string_array p_array);
void GDAPI godot_variant_set_vector2_array(godot_variant p_variant, godot_vector2_array p_array);
void GDAPI godot_variant_set_vector3_array(godot_variant p_variant, godot_vector3_array p_array);
void GDAPI godot_variant_set_color_array(godot_variant p_variant, godot_color_array p_array);

godot_bool GDAPI godot_variant_get_bool(godot_variant p_variant);
int GDAPI godot_variant_get_int(godot_variant p_variant);
float GDAPI godot_variant_get_float(godot_variant p_variant);
int GDAPI godot_variant_get_string(godot_variant p_variant, char *p_string, int p_bufsize);
void GDAPI godot_variant_get_vector2(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_get_rect2(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_get_vector3(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_get_matrix32(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_get_plane(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_get_aabb(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_get_matrix3(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_get_transform(godot_variant p_variant, float *p_elems);
void GDAPI godot_variant_get_color(godot_variant p_variant, float *p_elems);
godot_image GDAPI *godot_variant_get_image(godot_variant p_variant);
int GDAPI godot_variant_get_node_path(godot_variant p_variant, char *p_path, int p_bufsize);
godot_rid GDAPI godot_variant_get_rid(godot_variant p_variant);
godot_instance GDAPI godot_variant_get_instance(godot_variant p_variant);
void GDAPI godot_variant_get_input_event(godot_variant p_variant, godot_input_event);
void GDAPI godot_variant_get_dictionary(godot_variant p_variant, godot_dictionary);
godot_array GDAPI godot_variant_get_array(godot_variant p_variant);
godot_byte_array GDAPI godot_variant_get_byte_array(godot_variant p_variant);
godot_byte_array GDAPI godot_variant_get_int_array(godot_variant p_variant);
godot_string_array GDAPI godot_variant_get_string_array(godot_variant p_variant);
godot_vector2_array GDAPI godot_variant_get_vector2_array(godot_variant p_variant);
godot_vector3_array GDAPI godot_variant_get_vector3_array(godot_variant p_variant);
godot_color_array GDAPI godot_variant_get_color_array(godot_variant p_variant);

void GDAPI godot_variant_delete(godot_variant p_variant);

////// Class
///

char GDAPI **godot_class_get_list(); //get list of classes in array to array of strings, must be freed, use godot_list_free()

int GDAPI godot_class_get_base(char *p_class, char *p_base, int p_max_len);
int GDAPI godot_class_get_name(char *p_class, char *p_base, int p_max_len);

char GDAPI **godot_class_get_method_list(char *p_class); //free with godot_list_free()
int GDAPI godot_class_method_get_argument_count(char *p_class, char *p_method);
int GDAPI godot_class_method_get_argument_type(char *p_class, char *p_method, int p_argument);
godot_variant GDAPI godot_class_method_get_argument_default_value(char *p_class, char *p_method, int p_argument);

char GDAPI **godot_class_get_constant_list(char *p_class); //free with godot_list_free()
int GDAPI godot_class_constant_get_value(char *p_class, char *p_constant);

////// Instance

typedef int godot_call_error;

#define GODOT_CALL_OK
#define GODOT_CALL_ERROR_WRONG_ARGUMENTS
#define GODOT_CALL_ERROR_INVALID_INSTANCE

godot_instance GDAPI godot_instance_new(char *p_class);
int GDAPI godot_instance_get_class(godot_instance p_instance, char *p_class, int p_max_len);

typedef struct {
	char *name;
	int hint;
	char *hint_string;
	int usage;
} godot_property_info;

godot_call_error GDAPI godot_instance_call(godot_instance p_instance, char *p_method, ...);
godot_call_error GDAPI godot_instance_call_ret(godot_instance p_instance, godot_variant r_return, char *p_method, ...);
godot_bool GDAPI godot_instance_set(godot_instance p_instance, char *p_prop, godot_variant p_value);
godot_variant GDAPI godot_instance_get(godot_instance p_instance, char *p_prop);

#define GODOT_PROPERTY_HINT_NONE 0 ///< no hint provided.
#define GODOT_PROPERTY_HINT_RANGE 1 ///< hint_text = "min,max,step,slider; //slider is optional"
#define GODOT_PROPERTY_HINT_EXP_RANGE 2 ///< hint_text = "min,max,step", exponential edit
#define GODOT_PROPERTY_HINT_ENUM 3 ///< hint_text= "val1,val2,val3,etc"
#define GODOT_PROPERTY_HINT_EXP_EASING 4 /// exponential easing funciton (Math::ease)
#define GODOT_PROPERTY_HINT_LENGTH 5 ///< hint_text= "length" (as integer)
#define GODOT_PROPERTY_HINT_SPRITE_FRAME 6
#define GODOT_PROPERTY_HINT_KEY_ACCEL 7 ///< hint_text= "length" (as integer)
#define GODOT_PROPERTY_HINT_FLAGS 8 ///< hint_text= "flag1,flag2,etc" (as bit flags)
#define GODOT_PROPERTY_HINT_ALL_FLAGS 9
#define GODOT_PROPERTY_HINT_FILE 10 ///< a file path must be passed, hint_text (optionally) is a filter "*.png,*.wav,*.doc,"
#define GODOT_PROPERTY_HINT_DIR 11 ///< a directort path must be passed
#define GODOT_PROPERTY_HINT_GLOBAL_FILE 12 ///< a file path must be passed, hint_text (optionally) is a filter "*.png,*.wav,*.doc,"
#define GODOT_PROPERTY_HINT_GLOBAL_DIR 13 ///< a directort path must be passed
#define GODOT_PROPERTY_HINT_RESOURCE_TYPE 14 ///< a resource object type
#define GODOT_PROPERTY_HINT_MULTILINE_TEXT 15 ///< used for string properties that can contain multiple lines
#define GODOT_PROPERTY_HINT_COLOR_NO_ALPHA 16 ///< used for ignoring alpha component when editing a color
#define GODOT_PROPERTY_HINT_IMAGE_COMPRESS_LOSSY 17
#define GODOT_PROPERTY_HINT_IMAGE_COMPRESS_LOSSLESS 18
#define GODOT_PROPERTY_HINT_OBJECT_ID 19

#define GODOT_PROPERTY_USAGE_STORAGE 1
#define GODOT_PROPERTY_USAGE_EDITOR 2
#define GODOT_PROPERTY_USAGE_NETWORK 4
#define GODOT_PROPERTY_USAGE_EDITOR_HELPER 8
#define GODOT_PROPERTY_USAGE_CHECKABLE 16 //used for editing global variables
#define GODOT_PROPERTY_USAGE_CHECKED 32 //used for editing global variables
#define GODOT_PROPERTY_USAGE_INTERNATIONALIZED 64 //hint for internationalized strings
#define GODOT_PROPERTY_USAGE_BUNDLE 128 //used for optimized bundles
#define GODOT_PROPERTY_USAGE_CATEGORY 256
#define GODOT_PROPERTY_USAGE_STORE_IF_NONZERO 512 //only store if nonzero
#define GODOT_PROPERTY_USAGE_STORE_IF_NONONE 1024 //only store if false
#define GODOT_PROPERTY_USAGE_NO_INSTANCE_STATE 2048
#define GODOT_PROPERTY_USAGE_RESTART_IF_CHANGED 4096
#define GODOT_PROPERTY_USAGE_SCRIPT_VARIABLE 8192
#define GODOT_PROPERTY_USAGE_STORE_IF_NULL 16384
#define GODOT_PROPERTY_USAGE_ANIMATE_AS_TRIGGER 32768

#define GODOT_PROPERTY_USAGE_DEFAULT GODOT_PROPERTY_USAGE_STORAGE | GODOT_PROPERTY_USAGE_EDITOR | GODOT_PROPERTY_USAGE_NETWORK
#define GODOT_PROPERTY_USAGE_DEFAULT_INTL GODOT_PROPERTY_USAGE_STORAGE | GODOT_PROPERTY_USAGE_EDITOR | GODOT_PROPERTY_USAGE_NETWORK | GODOT_PROPERTY_USAGE_INTERNATIONALIZED
#define GODOT_PROPERTY_USAGE_NOEDITOR GODOT_PROPERTY_USAGE_STORAGE | GODOT_PROPERTY_USAGE_NETWORK

godot_property_info GDAPI **godot_instance_get_property_list(godot_instance p_instance);
void GDAPI godot_instance_free_property_list(godot_instance p_instance, godot_property_info **p_list);

void GDAPI godot_list_free(char **p_name); //helper to free all the class list

////// Script API

typedef void *(godot_script_instance_func)(godot_instance); //passed an instance, return a pointer to your userdata
typedef void(godot_script_free_func)(godot_instance, void *); //passed an instance, please free your userdata

void GDAPI godot_script_register(char *p_base, char *p_name, godot_script_instance_func p_instance_func, godot_script_free_func p_free_func);
void GDAPI godot_script_unregister(char *p_name);

typedef GDAPI godot_variant(godot_script_func)(godot_instance, void *, godot_variant *, int); //instance,userdata,arguments,argument count. Return something or NULL. Arguments must not be freed.

void GDAPI godot_script_add_function(char *p_name, char *p_function_name, godot_script_func p_func);
void GDAPI godot_script_add_validated_function(char *p_name, char *p_function_name, godot_script_func p_func, int *p_arg_types, int p_arg_count, godot_variant *p_default_args, int p_default_arg_count);

typedef void(godot_set_property_func)(godot_instance, void *, godot_variant); //instance,userdata,value. Value must not be freed.
typedef godot_variant(godot_get_property_func)(godot_instance, void *); //instance,userdata. Return a value or NULL.

void GDAPI godot_script_add_property(char *p_name, char *p_path, godot_set_property_func p_set_func, godot_get_property_func p_get_func);
void GDAPI godot_script_add_listed_property(char *p_name, char *p_path, godot_set_property_func p_set_func, godot_get_property_func p_get_func, int p_type, int p_hint, char *p_hint_string, int p_usage);

////// System Functions

//using these will help Godot track how much memory is in use in debug mode
void GDAPI *godot_alloc(int p_bytes);
void GDAPI *godot_realloc(void *p_ptr, int p_bytes);
void GDAPI godot_free(void *p_ptr);

#ifdef __cplusplus
}
#endif

#endif // GODOT_C_H
