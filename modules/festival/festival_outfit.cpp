/**************************************************************************/
/*  festival_outfit.cpp                                                   */
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

#include "festival_outfit.h"

void FestivalOutfit::set_id(const StringName &p_id) { id = p_id; }
StringName FestivalOutfit::get_id() const { return id; }

void FestivalOutfit::set_display_name(const String &p_display_name) { display_name = p_display_name; }
String FestivalOutfit::get_display_name() const { return display_name; }

void FestivalOutfit::set_role(const StringName &p_role) { role = p_role; }
StringName FestivalOutfit::get_role() const { return role; }

void FestivalOutfit::set_authority(int p_authority) { authority = p_authority; }
int FestivalOutfit::get_authority() const { return authority; }

void FestivalOutfit::set_tags(const PackedStringArray &p_tags) { tags = p_tags; }
PackedStringArray FestivalOutfit::get_tags() const { return tags; }

bool FestivalOutfit::has_tag(const String &p_tag) const { return tags.has(p_tag); }

void FestivalOutfit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_id", "id"), &FestivalOutfit::set_id);
	ClassDB::bind_method(D_METHOD("get_id"), &FestivalOutfit::get_id);
	ClassDB::bind_method(D_METHOD("set_display_name", "display_name"), &FestivalOutfit::set_display_name);
	ClassDB::bind_method(D_METHOD("get_display_name"), &FestivalOutfit::get_display_name);
	ClassDB::bind_method(D_METHOD("set_role", "role"), &FestivalOutfit::set_role);
	ClassDB::bind_method(D_METHOD("get_role"), &FestivalOutfit::get_role);
	ClassDB::bind_method(D_METHOD("set_authority", "authority"), &FestivalOutfit::set_authority);
	ClassDB::bind_method(D_METHOD("get_authority"), &FestivalOutfit::get_authority);
	ClassDB::bind_method(D_METHOD("set_tags", "tags"), &FestivalOutfit::set_tags);
	ClassDB::bind_method(D_METHOD("get_tags"), &FestivalOutfit::get_tags);
	ClassDB::bind_method(D_METHOD("has_tag", "tag"), &FestivalOutfit::has_tag);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "id"), "set_id", "get_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "display_name"), "set_display_name", "get_display_name");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "role"), "set_role", "get_role");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "authority"), "set_authority", "get_authority");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "tags"), "set_tags", "get_tags");
}
