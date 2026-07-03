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

#include "core/object/class_db.h"

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

void FestivalItem::set_holder_type(const StringName &p_holder_type) { holder_type = p_holder_type; }
StringName FestivalItem::get_holder_type() const { return holder_type; }

void FestivalItem::set_holder_id(const StringName &p_holder_id) { holder_id = p_holder_id; }
StringName FestivalItem::get_holder_id() const { return holder_id; }

void FestivalItem::set_grants_extra_power(bool p_grants_extra_power) { grants_extra_power = p_grants_extra_power; }
bool FestivalItem::get_grants_extra_power() const { return grants_extra_power; }

void FestivalItem::set_power_description(const String &p_power_description) { power_description = p_power_description; }
String FestivalItem::get_power_description() const { return power_description; }

void FestivalItem::set_power_category(const String &p_power_category) { power_category = p_power_category; }
String FestivalItem::get_power_category() const { return power_category; }

void FestivalItem::set_gives_password(const String &p_gives_password) { gives_password = p_gives_password; }
String FestivalItem::get_gives_password() const { return gives_password; }

void FestivalItem::set_linked_plot_hook_id(const StringName &p_id) { linked_plot_hook_id = p_id; }
StringName FestivalItem::get_linked_plot_hook_id() const { return linked_plot_hook_id; }

void FestivalItem::set_is_progression(bool p_is_progression) { is_progression = p_is_progression; }
bool FestivalItem::get_is_progression() const { return is_progression; }

void FestivalItem::set_stage_availability(const Array &p_stage_availability) { stage_availability = p_stage_availability; }
Array FestivalItem::get_stage_availability() const { return stage_availability; }

void FestivalItem::set_census_data(const Dictionary &p_census_data) { census_data = p_census_data; }
Dictionary FestivalItem::get_census_data() const { return census_data; }

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
	ClassDB::bind_method(D_METHOD("set_holder_type", "holder_type"), &FestivalItem::set_holder_type);
	ClassDB::bind_method(D_METHOD("get_holder_type"), &FestivalItem::get_holder_type);
	ClassDB::bind_method(D_METHOD("set_holder_id", "holder_id"), &FestivalItem::set_holder_id);
	ClassDB::bind_method(D_METHOD("get_holder_id"), &FestivalItem::get_holder_id);
	ClassDB::bind_method(D_METHOD("set_grants_extra_power", "grants_extra_power"), &FestivalItem::set_grants_extra_power);
	ClassDB::bind_method(D_METHOD("get_grants_extra_power"), &FestivalItem::get_grants_extra_power);
	ClassDB::bind_method(D_METHOD("set_power_description", "power_description"), &FestivalItem::set_power_description);
	ClassDB::bind_method(D_METHOD("get_power_description"), &FestivalItem::get_power_description);
	ClassDB::bind_method(D_METHOD("set_power_category", "power_category"), &FestivalItem::set_power_category);
	ClassDB::bind_method(D_METHOD("get_power_category"), &FestivalItem::get_power_category);
	ClassDB::bind_method(D_METHOD("set_gives_password", "gives_password"), &FestivalItem::set_gives_password);
	ClassDB::bind_method(D_METHOD("get_gives_password"), &FestivalItem::get_gives_password);
	ClassDB::bind_method(D_METHOD("set_linked_plot_hook_id", "id"), &FestivalItem::set_linked_plot_hook_id);
	ClassDB::bind_method(D_METHOD("get_linked_plot_hook_id"), &FestivalItem::get_linked_plot_hook_id);
	ClassDB::bind_method(D_METHOD("set_is_progression", "is_progression"), &FestivalItem::set_is_progression);
	ClassDB::bind_method(D_METHOD("get_is_progression"), &FestivalItem::get_is_progression);
	ClassDB::bind_method(D_METHOD("set_stage_availability", "stage_availability"), &FestivalItem::set_stage_availability);
	ClassDB::bind_method(D_METHOD("get_stage_availability"), &FestivalItem::get_stage_availability);
	ClassDB::bind_method(D_METHOD("set_census_data", "census_data"), &FestivalItem::set_census_data);
	ClassDB::bind_method(D_METHOD("get_census_data"), &FestivalItem::get_census_data);
	ClassDB::bind_method(D_METHOD("has_tag", "tag"), &FestivalItem::has_tag);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "id"), "set_id", "get_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "display_name"), "set_display_name", "get_display_name");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description", PROPERTY_HINT_MULTILINE_TEXT), "set_description", "get_description");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "tags"), "set_tags", "get_tags");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "presentable"), "set_presentable", "is_presentable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stackable"), "set_stackable", "is_stackable");

	ADD_GROUP("Placement", "");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "holder_type"), "set_holder_type", "get_holder_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "holder_id"), "set_holder_id", "get_holder_id");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "stage_availability"), "set_stage_availability", "get_stage_availability");

	ADD_GROUP("Powers", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "grants_extra_power"), "set_grants_extra_power", "get_grants_extra_power");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "power_description", PROPERTY_HINT_MULTILINE_TEXT), "set_power_description", "get_power_description");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "power_category"), "set_power_category", "get_power_category");

	ADD_GROUP("Connections", "");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "gives_password"), "set_gives_password", "get_gives_password");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "linked_plot_hook_id"), "set_linked_plot_hook_id", "get_linked_plot_hook_id");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_progression"), "set_is_progression", "get_is_progression");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "census_data"), "set_census_data", "get_census_data");
}
