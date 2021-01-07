/*************************************************************************/
/*  rich_text_effect.cpp                                                 */
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

#include "rich_text_effect.h"

#include "core/object/script_language.h"

void RichTextEffect::_bind_methods() {
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "_process_custom_fx", PropertyInfo(Variant::OBJECT, "char_fx", PROPERTY_HINT_RESOURCE_TYPE, "CharFXTransform")));
}

Variant RichTextEffect::get_bbcode() const {
	Variant r;
	if (get_script_instance()) {
		if (!get_script_instance()->get("bbcode", r)) {
			String path = get_script_instance()->get_script()->get_path();
			r = path.get_file().get_basename();
		}
	}
	return r;
}

bool RichTextEffect::_process_effect_impl(Ref<CharFXTransform> p_cfx) {
	bool return_value = false;
	if (get_script_instance()) {
		Variant v = get_script_instance()->call("_process_custom_fx", p_cfx);
		if (v.get_type() != Variant::BOOL) {
			return_value = false;
		} else {
			return_value = (bool)v;
		}
	}
	return return_value;
}

RichTextEffect::RichTextEffect() {
}

void CharFXTransform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_range"), &CharFXTransform::get_range);
	ClassDB::bind_method(D_METHOD("set_range", "range"), &CharFXTransform::set_range);

	ClassDB::bind_method(D_METHOD("get_elapsed_time"), &CharFXTransform::get_elapsed_time);
	ClassDB::bind_method(D_METHOD("set_elapsed_time", "time"), &CharFXTransform::set_elapsed_time);

	ClassDB::bind_method(D_METHOD("is_visible"), &CharFXTransform::is_visible);
	ClassDB::bind_method(D_METHOD("set_visibility", "visibility"), &CharFXTransform::set_visibility);

	ClassDB::bind_method(D_METHOD("is_outline"), &CharFXTransform::is_outline);
	ClassDB::bind_method(D_METHOD("set_outline", "outline"), &CharFXTransform::set_outline);

	ClassDB::bind_method(D_METHOD("get_offset"), &CharFXTransform::get_offset);
	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &CharFXTransform::set_offset);

	ClassDB::bind_method(D_METHOD("get_color"), &CharFXTransform::get_color);
	ClassDB::bind_method(D_METHOD("set_color", "color"), &CharFXTransform::set_color);

	ClassDB::bind_method(D_METHOD("get_environment"), &CharFXTransform::get_environment);
	ClassDB::bind_method(D_METHOD("set_environment", "environment"), &CharFXTransform::set_environment);

	ClassDB::bind_method(D_METHOD("get_glyph_index"), &CharFXTransform::get_glyph_index);
	ClassDB::bind_method(D_METHOD("set_glyph_index", "glyph_index"), &CharFXTransform::set_glyph_index);

	ClassDB::bind_method(D_METHOD("get_font"), &CharFXTransform::get_font);
	ClassDB::bind_method(D_METHOD("set_font", "font"), &CharFXTransform::set_font);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "range"), "set_range", "get_range");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "elapsed_time"), "set_elapsed_time", "get_elapsed_time");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "visible"), "set_visibility", "is_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "outline"), "set_outline", "is_outline");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "env"), "set_environment", "get_environment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "glyph_index"), "set_glyph_index", "get_glyph_index");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "font"), "set_font", "get_font");
}

CharFXTransform::CharFXTransform() {
}

CharFXTransform::~CharFXTransform() {
	environment.clear();
}
