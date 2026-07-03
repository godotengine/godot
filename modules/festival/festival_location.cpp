/**************************************************************************/
/*  festival_location.cpp                                                 */
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

#include "festival_location.h"

#include "core/object/class_db.h"

void FestivalLocation::set_id(const StringName &p_id) { id = p_id; }
StringName FestivalLocation::get_id() const { return id; }
void FestivalLocation::set_display_name(const String &p_display_name) { display_name = p_display_name; }
String FestivalLocation::get_display_name() const { return display_name; }
void FestivalLocation::set_kind(const StringName &p_kind) { kind = p_kind; }
StringName FestivalLocation::get_kind() const { return kind; }
void FestivalLocation::set_parent_id(const StringName &p_parent_id) { parent_id = p_parent_id; }
StringName FestivalLocation::get_parent_id() const { return parent_id; }
void FestivalLocation::set_description(const String &p_description) { description = p_description; }
String FestivalLocation::get_description() const { return description; }
void FestivalLocation::set_notes(const String &p_notes) { notes = p_notes; }
String FestivalLocation::get_notes() const { return notes; }
void FestivalLocation::set_required_password(const String &p_required_password) { required_password = p_required_password; }
String FestivalLocation::get_required_password() const { return required_password; }
void FestivalLocation::set_is_residence(bool p_is_residence) { is_residence = p_is_residence; }
bool FestivalLocation::get_is_residence() const { return is_residence; }
void FestivalLocation::set_is_progression(bool p_is_progression) { is_progression = p_is_progression; }
bool FestivalLocation::get_is_progression() const { return is_progression; }
void FestivalLocation::set_is_template(bool p_is_template) { is_template = p_is_template; }
bool FestivalLocation::get_is_template() const { return is_template; }
void FestivalLocation::set_resident_npc_ids(const PackedStringArray &p_ids) { resident_npc_ids = p_ids; }
PackedStringArray FestivalLocation::get_resident_npc_ids() const { return resident_npc_ids; }
void FestivalLocation::set_connected_location_ids(const PackedStringArray &p_ids) { connected_location_ids = p_ids; }
PackedStringArray FestivalLocation::get_connected_location_ids() const { return connected_location_ids; }
void FestivalLocation::set_linked_item_ids(const PackedStringArray &p_ids) { linked_item_ids = p_ids; }
PackedStringArray FestivalLocation::get_linked_item_ids() const { return linked_item_ids; }
void FestivalLocation::set_linked_plot_hook_ids(const PackedStringArray &p_ids) { linked_plot_hook_ids = p_ids; }
PackedStringArray FestivalLocation::get_linked_plot_hook_ids() const { return linked_plot_hook_ids; }
void FestivalLocation::set_census_data(const Dictionary &p_census_data) { census_data = p_census_data; }
Dictionary FestivalLocation::get_census_data() const { return census_data; }

void FestivalLocation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_id", "id"), &FestivalLocation::set_id);
	ClassDB::bind_method(D_METHOD("get_id"), &FestivalLocation::get_id);
	ClassDB::bind_method(D_METHOD("set_display_name", "display_name"), &FestivalLocation::set_display_name);
	ClassDB::bind_method(D_METHOD("get_display_name"), &FestivalLocation::get_display_name);
	ClassDB::bind_method(D_METHOD("set_kind", "kind"), &FestivalLocation::set_kind);
	ClassDB::bind_method(D_METHOD("get_kind"), &FestivalLocation::get_kind);
	ClassDB::bind_method(D_METHOD("set_parent_id", "parent_id"), &FestivalLocation::set_parent_id);
	ClassDB::bind_method(D_METHOD("get_parent_id"), &FestivalLocation::get_parent_id);
	ClassDB::bind_method(D_METHOD("set_description", "description"), &FestivalLocation::set_description);
	ClassDB::bind_method(D_METHOD("get_description"), &FestivalLocation::get_description);
	ClassDB::bind_method(D_METHOD("set_notes", "notes"), &FestivalLocation::set_notes);
	ClassDB::bind_method(D_METHOD("get_notes"), &FestivalLocation::get_notes);
	ClassDB::bind_method(D_METHOD("set_required_password", "required_password"), &FestivalLocation::set_required_password);
	ClassDB::bind_method(D_METHOD("get_required_password"), &FestivalLocation::get_required_password);
	ClassDB::bind_method(D_METHOD("set_is_residence", "is_residence"), &FestivalLocation::set_is_residence);
	ClassDB::bind_method(D_METHOD("get_is_residence"), &FestivalLocation::get_is_residence);
	ClassDB::bind_method(D_METHOD("set_is_progression", "is_progression"), &FestivalLocation::set_is_progression);
	ClassDB::bind_method(D_METHOD("get_is_progression"), &FestivalLocation::get_is_progression);
	ClassDB::bind_method(D_METHOD("set_is_template", "is_template"), &FestivalLocation::set_is_template);
	ClassDB::bind_method(D_METHOD("get_is_template"), &FestivalLocation::get_is_template);
	ClassDB::bind_method(D_METHOD("set_resident_npc_ids", "ids"), &FestivalLocation::set_resident_npc_ids);
	ClassDB::bind_method(D_METHOD("get_resident_npc_ids"), &FestivalLocation::get_resident_npc_ids);
	ClassDB::bind_method(D_METHOD("set_connected_location_ids", "ids"), &FestivalLocation::set_connected_location_ids);
	ClassDB::bind_method(D_METHOD("get_connected_location_ids"), &FestivalLocation::get_connected_location_ids);
	ClassDB::bind_method(D_METHOD("set_linked_item_ids", "ids"), &FestivalLocation::set_linked_item_ids);
	ClassDB::bind_method(D_METHOD("get_linked_item_ids"), &FestivalLocation::get_linked_item_ids);
	ClassDB::bind_method(D_METHOD("set_linked_plot_hook_ids", "ids"), &FestivalLocation::set_linked_plot_hook_ids);
	ClassDB::bind_method(D_METHOD("get_linked_plot_hook_ids"), &FestivalLocation::get_linked_plot_hook_ids);
	ClassDB::bind_method(D_METHOD("set_census_data", "census_data"), &FestivalLocation::set_census_data);
	ClassDB::bind_method(D_METHOD("get_census_data"), &FestivalLocation::get_census_data);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "id"), "set_id", "get_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "display_name"), "set_display_name", "get_display_name");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "kind"), "set_kind", "get_kind");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "parent_id"), "set_parent_id", "get_parent_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description", PROPERTY_HINT_MULTILINE_TEXT), "set_description", "get_description");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "notes", PROPERTY_HINT_MULTILINE_TEXT), "set_notes", "get_notes");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "required_password"), "set_required_password", "get_required_password");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_residence"), "set_is_residence", "get_is_residence");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_progression"), "set_is_progression", "get_is_progression");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_template"), "set_is_template", "get_is_template");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "resident_npc_ids"), "set_resident_npc_ids", "get_resident_npc_ids");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "connected_location_ids"), "set_connected_location_ids", "get_connected_location_ids");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "linked_item_ids"), "set_linked_item_ids", "get_linked_item_ids");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "linked_plot_hook_ids"), "set_linked_plot_hook_ids", "get_linked_plot_hook_ids");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "census_data"), "set_census_data", "get_census_data");
}
