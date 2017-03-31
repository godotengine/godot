/*************************************************************************/
/*  godot_c.cpp                                                          */
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
#include "godot_c.h"

#include "core/image.h"
#include "core/dvector.h"

#include "core/variant.h"
#include "core/dictionary.h"
#include "core/list.h"
#include "core/array.h"
#include "core/ustring.h"

#include "core/os/input_event.h"

#include "core/math/math_2d.h"
#include "core/math/vector3.h"

#include "core/class_db.h"

#include <cstring>

////// Image

godot_image GDAPI godot_image_create_empty() {
  Image *new_image = new Image();
  return (godot_image) new_image;
};

godot_image GDAPI godot_image_create(int p_width, int p_height, int p_format, int p_use_mipmaps) {
  Image *new_image = new Image(p_width, p_height, p_use_mipmaps, (Image::Format) p_format);
  return (godot_image) new_image;
}

godot_image GDAPI godot_image_create_with_data(int p_width, int p_height, int p_format, int p_use_mipmaps, unsigned char *p_buffer) {
  PoolVector<uint8_t> p_data;
  int pixel_size = Image::get_format_pixel_size((Image::Format) p_format);
  p_data.resize(p_width * p_height * pixel_size);
  std::memcpy(p_data.write().ptr(), p_buffer, p_data.size());
  Image* new_image = new Image(p_width, p_height, p_use_mipmaps, (Image::Format) p_format);
  return (godot_image) new_image;
}

int GDAPI godot_image_get_width(godot_image p_image) {
  Image *image = (Image *) p_image;
  return image->get_width();
}

int GDAPI godot_image_get_height(godot_image p_image) {
  Image *image = (Image *) p_image;
  return image->get_height();
}

int GDAPI godot_image_get_format(godot_image p_image) {
  Image *image = (Image *) p_image;
  return (int) image->get_format();
}

int GDAPI godot_image_get_mipmap_count(godot_image p_image) {
  Image *image = (Image *) p_image;
  return (int) image->get_mipmap_count();
}

godot_image GDAPI godot_image_copy(godot_image p_image) {
  Image *image = (Image *) p_image;
  Image *copy = new Image( *image );
  return (godot_image) copy;
}

void GDAPI godot_image_free(godot_image p_image) {
  delete ( (Image *) p_image );
}
 
////// RID
//
//#include "core/rid.h"
//
//godot_rid GDAPI godot_rid_create() {
//  RID *rid = new RID();
//  return (godot_rid) rid;
//}
//
//godot_rid GDAPI godot_rid_copy(godot_rid p_rid) {
//  
//}
//
// void GDAPI godot_rid_free(godot_rid p_rid); 
 
////// Dictionary

godot_dictionary GDAPI godot_dictionary_create() {
  Dictionary *dictionary = new Dictionary();
  return (godot_dictionary) dictionary;
}

godot_bool GDAPI godot_dictionary_has(godot_dictionary p_dictionary, godot_variant p_key) {
  Dictionary *dictionary = (Dictionary *) p_dictionary;
  Variant *key = (Variant *) p_key;
  bool has_key = dictionary->has(*key);
  return has_key ? GODOT_TRUE : GODOT_FALSE;
}

godot_variant GDAPI godot_dictionary_get(godot_dictionary p_dictionary, godot_variant p_key) {
  Dictionary *dictionary = (Dictionary *) p_dictionary;
  Variant *key = (Variant *) p_key;
  return (godot_variant) dictionary->getptr(*key);
}

void GDAPI godot_dictionary_insert(godot_dictionary p_dictionary, godot_variant p_key, godot_variant p_value) {
  Dictionary *dictionary = (Dictionary *) p_dictionary;
  Variant *key = (Variant *) p_key;
  Variant *value = (Variant *) p_value;
  (*dictionary)[*key] = *value;
}

void GDAPI godot_dictionary_remove(godot_dictionary p_dictionary, godot_variant p_key) {
  Dictionary *dictionary = (Dictionary *) p_dictionary;
  Variant *key = (Variant *) p_key;
  dictionary->erase(*key);
}

void GDAPI godot_dictionary_clear(godot_dictionary p_dictionary) {
  Dictionary *dictionary = (Dictionary *) p_dictionary;
  dictionary->clear();
}

int GDAPI godot_dictionary_get_size(godot_dictionary p_dictionary) {
  Dictionary *dictionary = (Dictionary *) p_dictionary;
  return dictionary->size();
}

void GDAPI godot_dictionary_get_keys(godot_dictionary p_dictionary, godot_variant *p_keys) {
  Dictionary *dictionary = (Dictionary *) p_dictionary;
  const Variant *key = NULL;
  for(int i = 0; i < dictionary->size(); i++) {
    key = dictionary->next(key);
    p_keys[i] = (godot_variant) key;
  }
}

godot_dictionary GDAPI godot_dictionary_copy(godot_dictionary p_dictionary) {
  Dictionary *dictionary = (Dictionary *) p_dictionary;
  Dictionary *copy = new Dictionary(*dictionary);
  return (godot_dictionary) copy;
}

void GDAPI godot_dictionary_free(godot_dictionary p_dictionary) {
  Dictionary *dictionary = (Dictionary *) p_dictionary;
  delete dictionary;
}

////// Array

godot_array GDAPI godot_array_create() {
  Array *array = new Array();
  return (godot_array) array;
}

godot_variant GDAPI godot_array_get(godot_array p_array, int p_index) {
  Array *array = (Array *) p_array;
  Variant *value = new Variant(array->get(p_index));
  return (godot_variant) value;
}

void GDAPI godot_array_set(godot_array p_array, int p_index, godot_variant p_value) {
  Array *array = (Array *) p_array;
  Variant *value = (Variant *) p_value;
  array->set(p_index, *value);
}

void GDAPI godot_array_resize(godot_array p_array, int p_size) {
  Array *array = (Array *) p_array;
  array->resize(p_size);
}

void GDAPI godot_array_insert(godot_array p_array, int p_position, godot_variant p_value) {
  Array *array = (Array *) p_array;
  Variant *value = (Variant *) p_value;
  array->insert(p_position, *value);
}

void GDAPI godot_array_remove(godot_array p_array, int p_position) {
  Array *array = (Array *) p_array;
  array->remove(p_position);
}

void GDAPI godot_array_clear(godot_array p_array) {
  Array *array = (Array *) p_array;
  array->clear();
}

int GDAPI godot_array_get_size(godot_array p_array) {
  Array *array = (Array *) p_array;
  return array->size();
}

int GDAPI godot_array_find(godot_array p_array, godot_variant p_value, int p_from_pos) {
  Array *array = (Array *) p_array;
  Variant *value = (Variant *) p_value;
  return array->find(*value, p_from_pos);
}

godot_array GDAPI godot_array_copy(godot_array p_array) {
  Array *array = (Array *) p_array;
  Array *copy = new Array(*array);
  return copy;
}

void GDAPI godot_array_free(godot_array p_array) {
  Array *array = (Array *) p_array;
  delete array;
}

////// InputEvent

godot_input_event GDAPI godot_input_event_create() {
  InputEvent *event = new InputEvent();
  return (godot_input_event) event;
}

godot_input_event GDAPI godot_input_event_copy(godot_input_event p_input_event) {
  InputEvent *event = (InputEvent *) p_input_event;
  InputEvent *copy = new InputEvent(*event);
  return (godot_input_event) copy;
}

void GDAPI godot_input_event_free(godot_input_event p_input_event) {
  InputEvent *event = (InputEvent *) p_input_event;
  delete event;
}
 
int GDAPI godot_input_event_get_type(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  return (int) event->type;
}

int GDAPI godot_input_event_get_device(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  return (int) event->device;
} 

godot_bool GDAPI godot_input_event_mod_has_alt(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  bool has_alt = false;
  if(event->type == InputEvent::Type::KEY) has_alt = event->key.mod.alt;
  else if(event->type == InputEvent::Type::MOUSE_BUTTON) has_alt = event->mouse_button.mod.alt;
  else if(event->type == InputEvent::Type::MOUSE_MOTION) has_alt = event->mouse_motion.mod.alt;
  return has_alt ? GODOT_TRUE : GODOT_FALSE;
}

godot_bool GDAPI godot_input_event_mod_has_ctrl(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  bool has_control = false;
  if(event->type == InputEvent::Type::KEY) has_control = event->key.mod.control;
  else if(event->type == InputEvent::Type::MOUSE_BUTTON) has_control = event->mouse_button.mod.control;
  else if(event->type == InputEvent::Type::MOUSE_MOTION) has_control = event->mouse_motion.mod.control;
  return has_control ? GODOT_TRUE : GODOT_FALSE;
}

godot_bool GDAPI godot_input_event_mod_has_command(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  bool has_command = false;
  if(event->type == InputEvent::Type::KEY) has_command = event->key.mod.command;
  else if(event->type == InputEvent::Type::MOUSE_BUTTON) has_command = event->mouse_button.mod.command;
  else if(event->type == InputEvent::Type::MOUSE_MOTION) has_command = event->mouse_motion.mod.command;
  return has_command ? GODOT_TRUE : GODOT_FALSE;
}

godot_bool GDAPI godot_input_event_mod_has_shift(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  bool has_shift = false;
  if(event->type == InputEvent::Type::KEY) has_shift = event->key.mod.shift;
  else if(event->type == InputEvent::Type::MOUSE_BUTTON) has_shift = event->mouse_button.mod.shift;
  else if(event->type == InputEvent::Type::MOUSE_MOTION) has_shift = event->mouse_motion.mod.shift;
  return has_shift ? GODOT_TRUE : GODOT_FALSE;
}

godot_bool GDAPI godot_input_event_mod_has_meta(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  bool has_meta = false;
  if(event->type == InputEvent::Type::KEY) has_meta = event->key.mod.meta;
  else if(event->type == InputEvent::Type::MOUSE_BUTTON) has_meta = event->mouse_button.mod.meta;
  else if(event->type == InputEvent::Type::MOUSE_MOTION) has_meta = event->mouse_motion.mod.meta;
  return has_meta ? GODOT_TRUE : GODOT_FALSE;
}

int GDAPI godot_input_event_key_get_scancode(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::KEY) return 0;
  return event->key.scancode;
}

int GDAPI godot_input_event_key_get_unicode(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::KEY) return 0;
  return event->key.unicode;
}

godot_bool GDAPI godot_input_event_key_is_pressed(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  return event->is_pressed() ? GODOT_TRUE : GODOT_FALSE;
}

godot_bool GDAPI godot_input_event_key_is_echo(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  return event->is_echo() ? GODOT_TRUE : GODOT_FALSE;
}
 
float GDAPI godot_input_event_mouse_get_x(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_BUTTON) return 0.0f;
  return event->mouse_button.x;
}

float GDAPI godot_input_event_mouse_get_y(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_BUTTON) return 0.0f;
  return event->mouse_button.y;
}

float GDAPI godot_input_event_mouse_get_global_x(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_BUTTON) return 0.0f;
  return event->mouse_button.global_x;
}

float GDAPI godot_input_event_mouse_get_global_y(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_BUTTON) return 0.0f;
  return event->mouse_button.global_y;
}

int GDAPI godot_input_event_mouse_get_button_mask(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_BUTTON) return 0;
  return event->mouse_button.button_mask;
}

int GDAPI godot_input_event_mouse_button_get_button_index(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_BUTTON) return -1;
  return event->mouse_button.button_index;
}

godot_bool GDAPI godot_input_event_mouse_button_is_pressed(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_BUTTON) return GODOT_FALSE;
  return event->mouse_button.pressed ? GODOT_TRUE : GODOT_FALSE;
}

godot_bool GDAPI godot_input_event_mouse_button_is_doubleclick(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_BUTTON) return GODOT_FALSE;
  return event->mouse_button.doubleclick ? GODOT_TRUE : GODOT_FALSE;
}

float GDAPI godot_input_event_mouse_motion_get_relative_x(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_MOTION) return 0.0f;
  return event->mouse_motion.relative_x;
}

float GDAPI godot_input_event_mouse_motion_get_relative_y(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_MOTION) return 0.0f;
  return event->mouse_motion.relative_y;
}

float GDAPI godot_input_event_mouse_motion_get_speed_x(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_MOTION) return 0.0f;
  return event->mouse_motion.speed_x;
}

float GDAPI godot_input_event_mouse_motion_get_speed_y(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::MOUSE_MOTION) return 0.0f;
  return event->mouse_motion.speed_y;
}

int GDAPI godot_input_event_joypad_motion_get_axis(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::JOYPAD_MOTION) return -1;
  return event->joy_motion.axis;
}

float GDAPI godot_input_event_joypad_motion_get_axis_value(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::JOYPAD_MOTION) return 0.0f;
  return event->joy_motion.axis_value;
}
 
int GDAPI godot_input_event_joypad_button_get_button_index(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::JOYPAD_BUTTON) return -1;
  return event->joy_button.button_index;
}

godot_bool GDAPI godot_input_event_joypad_button_is_pressed(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::JOYPAD_BUTTON) return GODOT_FALSE;
  return event->joy_button.pressed;
}

float GDAPI godot_input_event_joypad_button_get_pressure(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::JOYPAD_BUTTON) return 0.0f;
  return event->joy_button.pressure;
}

int GDAPI godot_input_event_screen_touch_get_index(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::SCREEN_TOUCH) return -1;
  return event->screen_touch.index;
}

float GDAPI godot_input_event_screen_touch_get_x(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::SCREEN_TOUCH) return 0.0f;
  return event->screen_touch.x;
}

float GDAPI godot_input_event_screen_touch_get_y(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::SCREEN_TOUCH) return 0.0f;
  return event->screen_touch.y;
}

godot_bool GDAPI godot_input_event_screen_touch_is_pressed(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::SCREEN_TOUCH) return 0;
  return event->screen_touch.pressed;
}

int GDAPI godot_input_event_screen_drag_get_index(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::SCREEN_DRAG) return 0;
  return event->screen_drag.index;
}

float GDAPI godot_input_event_screen_drag_get_x(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::SCREEN_DRAG) return 0.0f;
  return event->screen_drag.x;
}

float GDAPI godot_input_event_screen_drag_get_y(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::SCREEN_DRAG) return 0.0f;
  return event->screen_drag.y;
}

float GDAPI godot_input_event_screen_drag_get_relative_x(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::SCREEN_DRAG) return 0.0f;
  return event->screen_drag.relative_x;
}

float GDAPI godot_input_event_screen_drag_get_relative_y(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::SCREEN_DRAG) return 0.0f;
  return event->screen_drag.relative_y;
}

float GDAPI godot_input_event_screen_drag_get_speed_x(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::SCREEN_DRAG) return 0.0f;
  return event->screen_drag.speed_x;
}

float GDAPI godot_input_event_screen_drag_get_speed_y(godot_input_event p_event) {
  InputEvent *event = (InputEvent *) p_event;
  if(event->type != InputEvent::Type::SCREEN_DRAG) return 0.0f;
  return event->screen_drag.speed_y;
}

godot_bool GDAPI godot_input_event_is_action(godot_input_event p_event, char *p_action) {
  InputEvent *event = (InputEvent *) p_event;
  return event->is_action(String(p_action)) ? GODOT_TRUE : GODOT_FALSE;
}

godot_bool GDAPI godot_input_event_is_action_pressed(godot_input_event p_event, char *p_action) {
  InputEvent *event = (InputEvent *) p_event;
  return event->is_action_pressed(String(p_action)) ? GODOT_TRUE : GODOT_FALSE;
}

////// ByteArray

godot_byte_array GDAPI godot_byte_array_create() {
  PoolByteArray *byte_array = new PoolByteArray();
  return (godot_byte_array) byte_array;
}

godot_byte_array GDAPI godot_byte_array_copy(godot_byte_array p_byte_array) {
  PoolByteArray *byte_array = (PoolByteArray *) p_byte_array;
  PoolByteArray *copy = new PoolByteArray(byte_array->subarray(0,byte_array->size()));
  return (godot_byte_array) copy;
}

void GDAPI godot_byte_array_free(godot_byte_array p_byte_array) {
  PoolByteArray *byte_array = (PoolByteArray *) p_byte_array;
  delete byte_array;
}

int GDAPI godot_byte_array_get_size(godot_byte_array p_byte_array) {
  PoolByteArray *byte_array = (PoolByteArray *) p_byte_array;
  return byte_array->size();
}

unsigned char GDAPI godot_byte_array_get(godot_byte_array p_byte_array, int p_index) {
  PoolByteArray *byte_array = (PoolByteArray *) p_byte_array;
  return byte_array->get(p_index);
}

void GDAPI godot_byte_array_set(godot_byte_array p_byte_array, int p_index, unsigned char p_value) {
  PoolByteArray *byte_array = (PoolByteArray *) p_byte_array;
  byte_array->set(p_index, p_value);
}

void GDAPI godot_byte_array_remove(godot_byte_array p_byte_array, int p_index) {
  PoolByteArray *byte_array = (PoolByteArray *) p_byte_array;
  byte_array->remove(p_index);
}

void GDAPI godot_byte_array_clear(godot_byte_array p_byte_array) {
  PoolByteArray *byte_array = (PoolByteArray *) p_byte_array;
  byte_array->resize(0);
}
 
// godot_byte_array_lock GDAPI godot_byte_array_get_lock(godot_byte_array p_byte_array);
// unsigned char GDAPI *godot_byte_array_lock_get_pointer(godot_byte_array_lock p_byte_array_lock);
// void GDAPI godot_byte_array_lock_free(godot_byte_array_lock p_byte_array_lock);
// 
godot_image GDAPI godot_image_create_with_array(int p_width, int p_height, int p_format, int p_use_mipmaps, godot_array p_array) {
  PoolByteArray *byte_array = (PoolByteArray *) p_array;
  Image *image = new Image(p_width, p_height, (bool) p_use_mipmaps, (Image::Format) p_format, *byte_array);
  return (godot_image) image;
}

godot_byte_array GDAPI godot_image_get_data(godot_image p_image) {
  Image *image = (Image *) p_image;
  PoolByteArray *byte_array = new PoolByteArray(image->get_data().subarray(0,image->get_data().size()));
  return byte_array;
}

////// IntArray

godot_int_array GDAPI godot_int_array_create() {
  PoolIntArray *int_array = new PoolIntArray();
  return (godot_int_array) int_array;
}

godot_int_array GDAPI godot_int_array_copy(godot_int_array p_int_array) {
  PoolIntArray *int_array = (PoolIntArray *) p_int_array;
  PoolIntArray *copy = new PoolIntArray(int_array->subarray(0,int_array->size()));
  return (godot_int_array) copy;
}

void GDAPI godot_int_array_free(godot_int_array p_int_array) {
  PoolIntArray *int_array = (PoolIntArray *) p_int_array;
  delete int_array;
}

int GDAPI godot_int_array_get_size(godot_int_array p_int_array) {
  PoolIntArray *int_array = (PoolIntArray *) p_int_array;
  return int_array->size();
}

int GDAPI godot_int_array_get(godot_int_array p_int_array, int p_index) {
  PoolIntArray *int_array = (PoolIntArray *) p_int_array;
  return int_array->get(p_index);
}

void GDAPI godot_int_array_set(godot_int_array p_int_array, int p_index, int p_value) {
  PoolIntArray *int_array = (PoolIntArray *) p_int_array;
  int_array->set(p_index, p_value);
}

void GDAPI godot_int_array_remove(godot_int_array p_int_array, int p_index) {
  PoolIntArray *int_array = (PoolIntArray *) p_int_array;
  int_array->remove(p_index);
}

void GDAPI godot_int_array_clear(godot_int_array p_int_array) {
  PoolIntArray *int_array = (PoolIntArray *) p_int_array;
  int_array->resize(0);
}

// godot_int_array_lock GDAPI godot_int_array_get_lock(godot_int_array p_int_array);
// int GDAPI *godot_int_array_lock_get_pointer(godot_int_array_lock p_int_array_lock);
// void GDAPI godot_int_array_lock_free(godot_int_array_lock p_int_array_lock);

////// RealArray

godot_real_array GDAPI godot_real_array_create() {
  PoolRealArray *real_array = new PoolRealArray();
  return real_array;
}

godot_real_array GDAPI godot_real_array_copy(godot_real_array p_real_array) {
  PoolRealArray *real_array = (PoolRealArray *) p_real_array;
  PoolRealArray *copy = new PoolRealArray(real_array->subarray(0, real_array->size()));
  return (godot_real_array) copy;
}

void GDAPI godot_real_array_free(godot_real_array p_real_array) {
  PoolRealArray *real_array = (PoolRealArray *) p_real_array;
  delete real_array;
}
 
int GDAPI godot_real_array_get_size(godot_real_array p_real_array) {
  PoolRealArray *real_array = (PoolRealArray *) p_real_array;
  return real_array->size();
}

float GDAPI godot_real_array_get(godot_real_array p_real_array, int p_index) {
  PoolRealArray *real_array = (PoolRealArray *) p_real_array;
  return (float) real_array->get(p_index);
}

void GDAPI godot_real_array_set(godot_real_array p_real_array, int p_index, float p_value) {
  PoolRealArray *real_array = (PoolRealArray *) p_real_array;
  real_array->set(p_index, p_value);
}

void GDAPI godot_real_array_remove(godot_real_array p_real_array, int p_index) {
  PoolRealArray *real_array = (PoolRealArray *) p_real_array;
  real_array->remove(p_index);
}

void GDAPI godot_real_array_clear(godot_real_array p_real_array) {
  PoolRealArray *real_array = (PoolRealArray *) p_real_array;
  real_array->resize(0);
}
 
// godot_real_array_lock GDAPI godot_real_array_get_lock(godot_real_array p_real_array);
// float GDAPI *godot_real_array_lock_get_pointer(godot_real_array_lock p_real_array_lock);
// void GDAPI godot_real_array_lock_free(godot_real_array_lock p_real_array_lock);

////// StringArray

godot_string_array GDAPI godot_string_array_create() {
  PoolStringArray *string_array = new PoolStringArray();
  return (godot_string_array) string_array;
}

godot_string_array GDAPI godot_string_array_copy(godot_string_array p_string_array) {
  PoolStringArray *string_array = (PoolStringArray *) p_string_array;
  PoolStringArray *copy = new PoolStringArray(string_array->subarray(0, string_array->size()));
  return (godot_string_array) copy;
}

void GDAPI godot_string_array_free(godot_string_array p_string_array) {
  PoolStringArray *string_array = (PoolStringArray *) p_string_array;
  delete string_array;
}
 
int GDAPI godot_string_array_get_size(godot_string_array p_string_array) {
  PoolStringArray *string_array = (PoolStringArray *) p_string_array;
  return string_array->size();
}

int GDAPI godot_string_array_get(godot_string_array p_string_array, int p_index, unsigned char *p_string, int p_max_len) {
  PoolStringArray *string_array = (PoolStringArray *) p_string_array;
  String string_value = string_array->get(p_index);
  std::strncpy((char *) p_string, string_value.ascii().get_data(), p_max_len);
  return string_value.length();
}

void GDAPI godot_string_array_set(godot_string_array p_string_array, int p_index, unsigned char *p_string) {
  PoolStringArray *string_array = (PoolStringArray *) p_string_array;
  String string_value((const char *) p_string);
  string_array->set(p_index, string_value);
}

void GDAPI godot_string_array_remove(godot_string_array p_string_array, int p_index) {
  PoolStringArray *string_array = (PoolStringArray *) p_string_array;
  string_array->remove(p_index);
}

void GDAPI godot_string_array_clear(godot_string_array p_string_array) {
  PoolStringArray *string_array = (PoolStringArray *) p_string_array;
  string_array->resize(0);
}

////// Vector2Array

godot_vector2_array GDAPI godot_vector2_array_create() {
  PoolVector2Array *vector2_array = new PoolVector2Array();
  return (godot_vector2_array) vector2_array;
}

godot_vector2_array GDAPI godot_vector2_array_copy(godot_vector2_array p_vector2_array) {
  PoolVector2Array *vector2_array = (PoolVector2Array *) p_vector2_array;
  PoolVector2Array *copy = new PoolVector2Array(vector2_array->subarray(0, vector2_array->size()));
  return (godot_vector2_array) copy;
}

void GDAPI godot_vector2_array_free(godot_vector2_array p_vector2_array) {
  PoolVector2Array *vector2_array = (PoolVector2Array *) p_vector2_array;
  delete vector2_array;
}
 
int GDAPI godot_vector2_array_get_size(godot_vector2_array p_vector2_array) {
  PoolVector2Array *vector2_array = (PoolVector2Array *) p_vector2_array;
  return vector2_array->size();
}

int GDAPI godot_vector2_array_get_stride(godot_vector2_array p_vector2_array) {
  PoolVector2Array *vector2_array = (PoolVector2Array *) p_vector2_array;
  return 2; //?
}

void GDAPI godot_vector2_array_get(godot_vector2_array p_vector2_array, int p_index, float *p_vector2) {
  PoolVector2Array *vector2_array = (PoolVector2Array *) p_vector2_array;
  p_vector2[0] = vector2_array->get(p_index).x;
  p_vector2[1] = vector2_array->get(p_index).y;
}

void GDAPI godot_vector2_array_set(godot_vector2_array p_vector2_array, int p_index, float *p_vector2) {
  PoolVector2Array *vector2_array = (PoolVector2Array *) p_vector2_array;
  Vector2 vector2_value(p_vector2[0], p_vector2[1]);
  vector2_array->set(p_index, vector2_value);
}

void GDAPI godot_vector2_array_remove(godot_vector2_array p_vector2_array, int p_index) {
  PoolVector2Array *vector2_array = (PoolVector2Array *) p_vector2_array;
  vector2_array->remove(p_index);
}

void GDAPI godot_vector2_array_clear(godot_vector2_array p_vector2_array) {
  PoolVector2Array *vector2_array = (PoolVector2Array *) p_vector2_array;
  vector2_array->resize(0);
}

// godot_vector2_array_lock GDAPI godot_vector2_array_get_lock(godot_vector2_array p_vector2_array);
// float GDAPI *godot_vector2_array_lock_get_pointer(godot_vector2_array_lock p_vector2_array_lock);
// void GDAPI godot_vector2_array_lock_free(godot_vector2_array_lock p_vector2_array_lock);

////// Vector3Array

godot_vector3_array GDAPI godot_vector3_array_create() {
  PoolVector3Array *vector3_array = new PoolVector3Array();
  return (godot_vector3_array) vector3_array;
}

godot_vector3_array GDAPI godot_vector3_array_copy(godot_vector3_array p_vector3_array) {
  PoolVector3Array *vector3_array = (PoolVector3Array *) p_vector3_array;
  PoolVector3Array *copy = new PoolVector3Array(vector3_array->subarray(0, vector3_array->size()));
  return (godot_vector3_array) copy;
}

void GDAPI godot_vector3_array_free(godot_vector3_array p_vector3_array) {
  PoolVector3Array *vector3_array = (PoolVector3Array *) p_vector3_array;
  delete vector3_array;
}
 
int GDAPI godot_vector3_array_get_size(godot_vector3_array p_vector3_array) {
  PoolVector3Array *vector3_array = (PoolVector3Array *) p_vector3_array;
  return vector3_array->size();
}

int GDAPI godot_vector3_array_get_stride(godot_vector3_array p_vector3_array) {
  PoolVector3Array *vector3_array = (PoolVector3Array *) p_vector3_array;
  return 3; //?
}

void GDAPI godot_vector3_array_get(godot_vector3_array p_vector3_array, int p_index, float *p_vector3) {
  PoolVector3Array *vector3_array = (PoolVector3Array *) p_vector3_array;
  p_vector3[0] = vector3_array->get(p_index).x;
  p_vector3[1] = vector3_array->get(p_index).y;
  p_vector3[2] = vector3_array->get(p_index).z;
}

void GDAPI godot_vector3_array_set(godot_vector3_array p_vector3_array, int p_index, float *p_vector3) {
  PoolVector3Array *vector3_array = (PoolVector3Array *) p_vector3_array;
  Vector3 vector3_value(p_vector3[0], p_vector3[1], p_vector3[2]);
  vector3_array->set(p_index, vector3_value);
}

void GDAPI godot_vector3_array_remove(godot_vector3_array p_vector3_array, int p_index) {
  PoolVector3Array *vector3_array = (PoolVector3Array *) p_vector3_array;
  vector3_array->remove(p_index);
}

void GDAPI godot_vector3_array_clear(godot_vector3_array p_vector3_array) {
  PoolVector3Array *vector3_array = (PoolVector3Array *) p_vector3_array;
  vector3_array->resize(0);
}

// godot_vector3_array_lock GDAPI godot_vector3_array_get_lock(godot_vector3_array p_vector3_array);
// float GDAPI *godot_vector3_array_lock_get_pointer(godot_vector3_array_lock p_vector3_array_lock);
// void GDAPI godot_vector3_array_lock_free(godot_vector3_array_lock p_vector3_array_lock);

////// ColorArray

godot_color_array GDAPI godot_color_array_create() {
  PoolColorArray *color_array = new PoolColorArray();
  return (godot_color_array) color_array;
}

godot_color_array GDAPI godot_color_array_copy(godot_color_array p_color_array) {
  PoolColorArray *color_array = (PoolColorArray *) p_color_array;
  PoolColorArray *copy = new PoolColorArray(color_array->subarray(0, color_array->size()));
  return (godot_color_array) copy;
}

void GDAPI godot_color_array_free(godot_color_array p_color_array) {
  PoolColorArray *color_array = (PoolColorArray *) p_color_array;
  delete color_array;
}
 
int GDAPI godot_color_array_get_size(godot_color_array p_color_array) {
  PoolColorArray *color_array = (PoolColorArray *) p_color_array;
  return color_array->size();
}

int GDAPI godot_color_array_get_stride(godot_color_array p_color_array) {
  return 4; //?
}

void GDAPI godot_color_array_get(godot_color_array p_color_array, int p_index, float *p_color) {
  PoolColorArray *color_array = (PoolColorArray *) p_color_array;
  p_color[0] = color_array->get(p_index).r;
  p_color[1] = color_array->get(p_index).g;
  p_color[2] = color_array->get(p_index).b;
  p_color[3] = color_array->get(p_index).a;
}

void GDAPI godot_color_array_set(godot_color_array p_color_array, int p_index, float *p_color) {
  PoolColorArray *color_array = (PoolColorArray *) p_color_array;
  Color color_value(p_color[0], p_color[1], p_color[2], p_color[3]);
  color_array->set(p_index, color_value);
}

void GDAPI godot_color_array_remove(godot_color_array p_color_array, int p_index) {
  PoolColorArray *color_array = (PoolColorArray *) p_color_array;
  color_array->remove(p_index);
}

void GDAPI godot_color_array_clear(godot_color_array p_color_array) {
  PoolColorArray *color_array = (PoolColorArray *) p_color_array;
  color_array->resize(0);
}

// godot_color_array_lock GDAPI godot_color_array_get_lock(godot_color_array p_color_array);
// float GDAPI *godot_color_array_lock_get_pointer(godot_color_array_lock p_color_array_lock);
// void GDAPI godot_color_array_lock_free(godot_color_array_lock p_color_array_lock);

////// Variant

godot_variant godot_variant_new() {
  Variant *variant = new Variant();
  return (godot_variant) variant;
}

int GDAPI godot_variant_get_type(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  return (int) variant->get_type();
}

void GDAPI godot_variant_set_null(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant();
}

void GDAPI godot_variant_set_bool(godot_variant p_variant, godot_bool p_bool) {
  Variant *variant = (Variant *) p_variant;
  bool boolean_value = (p_bool == GODOT_TRUE) ? true : false;
  *variant = Variant(boolean_value);
}

void GDAPI godot_variant_set_int(godot_variant p_variant, int p_int) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(p_int);
}

void GDAPI godot_variant_set_float(godot_variant p_variant, int p_float) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(p_float);
}

void GDAPI godot_variant_set_string(godot_variant p_variant, char *p_string) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(p_string);
}

void GDAPI godot_variant_set_vector2(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Vector2 vector2_value(p_elems[0], p_elems[1]);
  *variant = Variant(vector2_value);
}

void GDAPI godot_variant_set_rect2(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Rect2 rect2_value(p_elems[0], p_elems[1], p_elems[2], p_elems[3]);
  *variant = Variant(rect2_value);
}

void GDAPI godot_variant_set_vector3(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Vector3 vector3_value(p_elems[0], p_elems[1], p_elems[2]);
  *variant = Variant(vector3_value);
}

void GDAPI godot_variant_set_transform2d(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Transform2D transform2d_value(p_elems[0], p_elems[1], p_elems[2], p_elems[3], p_elems[4], p_elems[5]);
  *variant = Variant(transform2d_value);
}

void GDAPI godot_variant_set_plane(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Vector3 normal(p_elems[0], p_elems[1], p_elems[2]);
  Plane plane_value(normal, p_elems[3]);
  *variant = Variant(plane_value);
}

void GDAPI godot_variant_set_rect3(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Vector3 position(p_elems[0], p_elems[1], p_elems[2]);
  Vector3 size(p_elems[3], p_elems[4], p_elems[5]);
  Rect3 rect3_value(position, size);
  *variant = Variant(rect3_value);
}

void GDAPI godot_variant_set_basis(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Vector3 axis(p_elems[0], p_elems[1], p_elems[2]);
  Basis basis_value(axis, p_elems[3]);
  *variant = Variant(basis_value);
}

void GDAPI godot_variant_set_transform(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Vector3 axis(p_elems[0], p_elems[1], p_elems[2]);
  Basis basis(axis, p_elems[3]);
  Vector3 origin(p_elems[4], p_elems[5], p_elems[6]);
  Transform transform_value(basis, origin);
  *variant = Variant(transform_value);
}

void GDAPI godot_variant_set_color(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Color color_value(p_elems[0], p_elems[1], p_elems[2], p_elems[3]);
  *variant = Variant(color_value);
}

void GDAPI godot_variant_set_image(godot_variant p_variant, godot_image *p_image) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(*((Image *) p_image));
}

void GDAPI godot_variant_set_node_path(godot_variant p_variant, char *p_path) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(NodePath(p_path));
}

//void GDAPI godot_variant_set_rid(godot_variant p_variant, char *p_path) {
//  Variant *variant = (Variant *) p_variant;
//  *variant = Variant(RID(p_path));
//}

//void GDAPI godot_variant_set_instance(godot_variant p_variant, godot_instance p_instance) {
//  Variant *variant = (Variant *) p_variant;
//  *variant = Variant(Instance(p_path));
//}

void GDAPI godot_variant_set_input_event(godot_variant p_variant, godot_input_event p_event) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(*((InputEvent *) p_event));
}

void GDAPI godot_variant_set_dictionary(godot_variant p_variant, godot_dictionary p_dictionary) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(*((Dictionary *) p_dictionary));
}

void GDAPI godot_variant_set_array(godot_variant p_variant, godot_array p_array) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(*((Array *) p_array));
}

void GDAPI godot_variant_set_byte_array(godot_variant p_variant, godot_byte_array p_array) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(*((PoolByteArray *) p_array));
}

void GDAPI godot_variant_set_int_array(godot_variant p_variant, godot_byte_array p_array) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(*((PoolIntArray *) p_array));
}

void GDAPI godot_variant_set_string_array(godot_variant p_variant, godot_string_array p_array) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(*((PoolStringArray *) p_array));
}

void GDAPI godot_variant_set_vector2_array(godot_variant p_variant, godot_vector2_array p_array) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(*((PoolVector2Array *) p_array));
}

void GDAPI godot_variant_set_vector3_array(godot_variant p_variant, godot_vector3_array p_array) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(*((PoolVector3Array *) p_array));
}

void GDAPI godot_variant_set_color_array(godot_variant p_variant, godot_color_array p_array) {
  Variant *variant = (Variant *) p_variant;
  *variant = Variant(*((PoolColorArray *) p_array));
}


godot_bool GDAPI godot_variant_get_bool(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  bool boolean_value = (bool) *variant;
  return boolean_value ? GODOT_TRUE : GODOT_FALSE;
}

int GDAPI godot_variant_get_int(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  return (int) *variant;
}

float GDAPI godot_variant_get_float(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  return (float) *variant;
}

int GDAPI godot_variant_get_string(godot_variant p_variant, char *p_string, int p_bufsize) {
  Variant *variant = (Variant *) p_variant;
  String string_value = (String) *variant;
  std::strncpy(p_string, string_value.ascii().get_data(), p_bufsize);
  return string_value.length();
}

void GDAPI godot_variant_get_vector2(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Vector2 vector2_value = (Vector2) *variant;
  p_elems[0] = vector2_value.x;
  p_elems[1] = vector2_value.y;
}

void GDAPI godot_variant_get_rect2(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Rect2 rect2_value = (Rect2) *variant;
  p_elems[0] = rect2_value.get_pos().x;
  p_elems[1] = rect2_value.get_pos().y;
  p_elems[2] = rect2_value.get_size().width;
  p_elems[3] = rect2_value.get_size().height;
}

void GDAPI godot_variant_get_vector3(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Vector3 vector3_value = (Vector3) *variant;
  p_elems[0] = vector3_value.x;
  p_elems[1] = vector3_value.y;
  p_elems[2] = vector3_value.z;
}

void GDAPI godot_variant_get_transform2d(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Transform2D transform2d_value = (Transform2D) *variant;
  p_elems[0] = transform2d_value.get_axis(0).x;
  p_elems[1] = transform2d_value.get_axis(0).y;
  p_elems[2] = transform2d_value.get_axis(1).x;
  p_elems[3] = transform2d_value.get_axis(1).y;
  p_elems[4] = transform2d_value.get_axis(2).x;
  p_elems[5] = transform2d_value.get_axis(2).y;
}

void GDAPI godot_variant_get_plane(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Plane plane_value = (Plane) *variant;
  p_elems[0] = plane_value.get_normal().x;
  p_elems[1] = plane_value.get_normal().y;
  p_elems[2] = plane_value.get_normal().z;
  p_elems[3] = plane_value.d;
}

void GDAPI godot_variant_get_rect3(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Rect3 rect3_value = (Rect3) *variant;
  p_elems[0] = rect3_value.get_pos().x;
  p_elems[1] = rect3_value.get_pos().y;
  p_elems[2] = rect3_value.get_pos().z;
  p_elems[3] = rect3_value.get_size().y;
  p_elems[4] = rect3_value.get_size().x;
  p_elems[5] = rect3_value.get_size().z;
}

void GDAPI godot_variant_get_basis(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Basis basis_value = variant->operator Basis();
  Vector3 axis;
  real_t phi;
  basis_value.get_axis_and_angle(axis, phi);
  p_elems[0] = axis.x;
  p_elems[1] = axis.y;
  p_elems[2] = axis.z;
  p_elems[3] = phi;
}

void GDAPI godot_variant_get_transform(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Transform transform_value = variant->operator Transform();
  Vector3 axis;
  real_t phi;
  transform_value.get_basis().get_axis_and_angle(axis, phi);
  p_elems[0] = axis.x;
  p_elems[1] = axis.y;
  p_elems[2] = axis.z;
  p_elems[3] = phi;
  p_elems[4] = transform_value.get_origin().x;
  p_elems[5] = transform_value.get_origin().y;
  p_elems[6] = transform_value.get_origin().z;
}

void GDAPI godot_variant_get_color(godot_variant p_variant, float *p_elems) {
  Variant *variant = (Variant *) p_variant;
  Color color_value = (Color) *variant;
  p_elems[0] = color_value.r;
  p_elems[1] = color_value.g;
  p_elems[2] = color_value.b;
  p_elems[3] = color_value.a;
}

godot_image GDAPI godot_variant_get_image(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  Image *image = new Image(*variant);
  return (godot_image) image;
}

int GDAPI godot_variant_get_node_path(godot_variant p_variant, char *p_path, int p_bufsize) {
  Variant *variant = (Variant *) p_variant;
  String string_value = (String) *variant;
  std::strncpy(p_path, string_value.ascii().get_data(), p_bufsize);
  return string_value.length();
}

//godot_rid GDAPI godot_variant_get_rid(godot_variant p_variant) {
//  Variant *variant = (Variant *) p_variant;
//}

//godot_instance GDAPI godot_variant_get_instance(godot_variant p_variant) {
//  Variant *variant = (Variant *) p_variant;
//}

godot_input_event GDAPI godot_variant_get_input_event(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  InputEvent *event = new InputEvent(*variant);
  return (godot_input_event) event;
}

godot_dictionary GDAPI godot_variant_get_dictionary(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  Dictionary *dictionary = new Dictionary(*variant);
  return (godot_dictionary) dictionary;
}

godot_array GDAPI godot_variant_get_array(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  Array *array = new Array(*variant);
  return (godot_array) array;
}

godot_byte_array GDAPI godot_variant_get_byte_array(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  PoolByteArray *byte_array = new PoolByteArray(*variant);
  return (godot_byte_array) byte_array;
}

godot_int_array GDAPI godot_variant_get_int_array(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  PoolIntArray *int_array = new PoolIntArray(*variant);
  return (godot_int_array) int_array;
}

godot_real_array GDAPI godot_variant_get_real_array(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  PoolRealArray *real_array = new PoolRealArray(*variant);
  return (godot_real_array) real_array;
}

godot_string_array GDAPI godot_variant_get_string_array(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  PoolStringArray *string_array = new PoolStringArray(*variant);
  return (godot_string_array) string_array;
}

godot_vector2_array GDAPI godot_variant_get_vector2_array(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  PoolVector2Array *vector2_array = new PoolVector2Array(*variant);
  return (godot_vector2_array) vector2_array;
}

godot_vector3_array GDAPI godot_variant_get_vector3_array(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  PoolVector3Array *vector3_array = new PoolVector3Array(*variant);
  return (godot_vector3_array) vector3_array;
}

godot_color_array GDAPI godot_variant_get_color_array(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  PoolColorArray *color_array = new PoolColorArray(*variant);
  return (godot_color_array) color_array;
}


void GDAPI godot_variant_delete(godot_variant p_variant) {
  Variant *variant = (Variant *) p_variant;
  delete variant;
}


////// Class

char GDAPI **godot_class_get_list() {
  List<StringName> class_names;
  ClassDB::get_class_list(&class_names);
  char **class_names_list = new char *[class_names.size()+1];
  for(int i = 0; i < class_names.size(); i++) {
    String string_value = (String) class_names[i];
    class_names_list[i] = new char[string_value.length()];
    std::strncpy(class_names_list[i], string_value.ascii().get_data(), string_value.length());
  }
  class_names_list[class_names.size()] = NULL;
  return class_names_list;
}

//int GDAPI godot_class_get_base(char *p_class, char *p_base, int p_max_len);
//int GDAPI godot_class_get_name(char *p_class, char *p_base, int p_max_len);

char GDAPI **godot_class_get_method_list(char *p_class) {
  StringName class_name(p_class);
  List<MethodInfo> method_list;
  ClassDB::get_method_list(class_name, &method_list);
  char **method_names_list = new char *[method_list.size()+1];
  for(int i = 0; i < method_list.size(); i++) {
    String string_value = method_list[i].name;
    method_names_list[i] = new char[string_value.length()];
    std::strncpy(method_names_list[i], string_value.ascii().get_data(), string_value.length());
  }
  method_names_list[method_list.size()] = NULL;
  return method_names_list;
}

int GDAPI godot_class_method_get_argument_count(char *p_class, char *p_method) {
  StringName class_name(p_class);
  StringName method_name(p_method);
  MethodBind *method = ClassDB::get_method(class_name, method_name);
  return method->get_argument_count();
}

int GDAPI godot_class_method_get_argument_type(char *p_class, char *p_method, int p_argument) {
  StringName class_name(p_class);
  StringName method_name(p_method);
  MethodBind *method = ClassDB::get_method(class_name, method_name);
  return method->get_default_arguments()[p_argument].get_type();
}

godot_variant GDAPI godot_class_method_get_argument_default_value(char *p_class, char *p_method, int p_argument) {
  StringName class_name(p_class);
  StringName method_name(p_method);
  MethodBind *method = ClassDB::get_method(class_name, method_name);
  return (godot_variant) &(method->get_default_arguments()[p_argument]);
}

char GDAPI **godot_class_get_constant_list(char *p_class){
  StringName class_name(p_class);
  List<String> constant_list;
  ClassDB::get_integer_constant_list(class_name, &constant_list);
  char **constant_names_list = new char *[constant_list.size()+1];
  for(int i = 0; i < constant_list.size(); i++) {
    String &string_value = constant_list[i];
    constant_names_list[i] = new char[string_value.length()];
    std::strncpy(constant_names_list[i], string_value.ascii().get_data(), string_value.length());
  }
  constant_names_list[constant_list.size()] = NULL;
  return constant_names_list;
}

int GDAPI godot_class_constant_get_value(char *p_class, char *p_constant) {
  StringName class_name(p_class);
  StringName constant_name(p_constant);
  return ClassDB::get_integer_constant(class_name, constant_name);
}

////// Instance

// godot_instance GDAPI godot_instance_new(char *p_class);
// int GDAPI godot_instance_get_class(godot_instance p_instance, char *p_class, int p_max_len);
// 
// godot_call_error GDAPI godot_instance_call(godot_instance p_instance, char *p_method, ...);
// godot_call_error GDAPI godot_instance_call_ret(godot_instance p_instance, godot_variant r_return, char *p_method, ...);
// godot_bool GDAPI godot_instance_set(godot_instance p_instance, char *p_prop, godot_variant p_value);
// godot_variant GDAPI godot_instance_get(godot_instance p_instance, char *p_prop);
// 
// godot_property_info GDAPI **godot_instance_get_property_list(godot_instance p_instance);
// void GDAPI godot_instance_free_property_list(godot_instance p_instance, godot_property_info **p_list);

void GDAPI godot_list_free(char **p_name) {
  char *string_value = p_name[0];
  while(string_value != NULL) {
    delete string_value;
    string_value++;
  }
  delete p_name;
}

////// Script API

// void GDAPI godot_script_register(char *p_base, char *p_name, godot_script_instance_func p_instance_func, godot_script_free_func p_free_func);
// void GDAPI godot_script_unregister(char *p_name);
// 
// void GDAPI godot_script_add_function(char *p_name, char *p_function_name, godot_script_func p_func);
// void GDAPI godot_script_add_validated_function(char *p_name, char *p_function_name, godot_script_func p_func, int *p_arg_types, int p_arg_count, godot_variant *p_default_args, int p_default_arg_count);
// 
// void GDAPI godot_script_add_property(char *p_name, char *p_path, godot_set_property_func p_set_func, godot_get_property_func p_get_func);
// void GDAPI godot_script_add_listed_property(char *p_name, char *p_path, godot_set_property_func p_set_func, godot_get_property_func p_get_func, int p_type, int p_hint, char *p_hint_string, int p_usage);

////// System Functions

//using these will help Godot track how much memory is in use in debug mode
// void GDAPI *godot_alloc(int p_bytes);
// void GDAPI *godot_realloc(void *p_ptr, int p_bytes);
// void GDAPI godot_free(void *p_ptr);
