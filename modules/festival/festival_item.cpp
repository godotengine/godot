/**************************************************************************/
/*  festival_item.cpp                                                     */
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

#include "festival_item.h"

void FestivalItem::set_id(const StringName &p_id) { id = p_id; }
StringName FestivalItem::get_id() const { return id; }

void FestivalItem::set_display_name(const String &p_display_name) { display_name = p_display_name; }
String FestivalItem::get_display_name() const { return display_name; }

void FestivalItem::set_description(const String &p_description) { description = p_description; }
String FestivalItem::get_description() const { return description; }

void FestivalItem::set_tags(const PackedStringArray &p_tags) { tags = p_tags; }
PackedStringArray FestivalItem::get_tags() const { return tags; }

void FestivalItem::set_presentable(bool p_presentable) { presentable = p_presentable; }
bool FestivalItem::is_presentable() const { return presentable; }

void FestivalItem::set_stackable(bool p_stackable) { stackable = p_stackable; }
bool FestivalItem::is_stackable() const { return stackable; }

bool FestivalItem::has_tag(const String &p_tag) const { return tags.has(p_tag); }

void FestivalItem::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_id", "id"), &FestivalItem::set_id);
	ClassDB::bind_method(D_METHOD("get_id"), &FestivalItem::get_id);
	ClassDB::bind_method(D_METHOD("set_display_name", "display_name"), &FestivalItem::set_display_name);
	ClassDB::bind_method(D_METHOD("get_display_name"), &FestivalItem::get_display_name);
	ClassDB::bind_method(D_METHOD("set_description", "description"), &FestivalItem::set_description);
	ClassDB::bind_method(D_METHOD("get_description"), &FestivalItem::get_description);
	ClassDB::bind_method(D_METHOD("set_tags", "tags"), &FestivalItem::set_tags);
	ClassDB::bind_method(D_METHOD("get_tags"), &FestivalItem::get_tags);
	ClassDB::bind_method(D_METHOD("set_presentable", "presentable"), &FestivalItem::set_presentable);
	ClassDB::bind_method(D_METHOD("is_presentable"), &FestivalItem::is_presentable);
	ClassDB::bind_method(D_METHOD("set_stackable", "stackable"), &FestivalItem::set_stackable);
	ClassDB::bind_method(D_METHOD("is_stackable"), &FestivalItem::is_stackable);
	ClassDB::bind_method(D_METHOD("has_tag", "tag"), &FestivalItem::has_tag);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "id"), "set_id", "get_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "display_name"), "set_display_name", "get_display_name");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description", PROPERTY_HINT_MULTILINE_TEXT), "set_description", "get_description");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "tags"), "set_tags", "get_tags");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "presentable"), "set_presentable", "is_presentable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stackable"), "set_stackable", "is_stackable");
}
