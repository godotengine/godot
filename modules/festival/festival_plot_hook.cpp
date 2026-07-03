/**************************************************************************/
/*  festival_plot_hook.cpp                                                */
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

#include "festival_plot_hook.h"

#include "core/object/class_db.h"

void FestivalPlotHook::set_id(const StringName &p_id) { id = p_id; }
StringName FestivalPlotHook::get_id() const { return id; }
void FestivalPlotHook::set_title(const String &p_title) { title = p_title; }
String FestivalPlotHook::get_title() const { return title; }
void FestivalPlotHook::set_description(const String &p_description) { description = p_description; }
String FestivalPlotHook::get_description() const { return description; }
void FestivalPlotHook::set_status(const StringName &p_status) { status = p_status; }
StringName FestivalPlotHook::get_status() const { return status; }
void FestivalPlotHook::set_is_progression(bool p_is_progression) { is_progression = p_is_progression; }
bool FestivalPlotHook::get_is_progression() const { return is_progression; }
void FestivalPlotHook::set_linked_npc_ids(const PackedStringArray &p_ids) { linked_npc_ids = p_ids; }
PackedStringArray FestivalPlotHook::get_linked_npc_ids() const { return linked_npc_ids; }
void FestivalPlotHook::set_linked_location_ids(const PackedStringArray &p_ids) { linked_location_ids = p_ids; }
PackedStringArray FestivalPlotHook::get_linked_location_ids() const { return linked_location_ids; }
void FestivalPlotHook::set_linked_item_ids(const PackedStringArray &p_ids) { linked_item_ids = p_ids; }
PackedStringArray FestivalPlotHook::get_linked_item_ids() const { return linked_item_ids; }
void FestivalPlotHook::set_linked_rumor_ids(const PackedStringArray &p_ids) { linked_rumor_ids = p_ids; }
PackedStringArray FestivalPlotHook::get_linked_rumor_ids() const { return linked_rumor_ids; }
void FestivalPlotHook::set_linked_event_ids(const PackedStringArray &p_ids) { linked_event_ids = p_ids; }
PackedStringArray FestivalPlotHook::get_linked_event_ids() const { return linked_event_ids; }
void FestivalPlotHook::set_linked_outfit_ids(const PackedStringArray &p_ids) { linked_outfit_ids = p_ids; }
PackedStringArray FestivalPlotHook::get_linked_outfit_ids() const { return linked_outfit_ids; }
void FestivalPlotHook::set_linked_npc_secrets(const Array &p_secrets) { linked_npc_secrets = p_secrets; }
Array FestivalPlotHook::get_linked_npc_secrets() const { return linked_npc_secrets; }
void FestivalPlotHook::set_census_data(const Dictionary &p_census_data) { census_data = p_census_data; }
Dictionary FestivalPlotHook::get_census_data() const { return census_data; }

void FestivalPlotHook::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_id", "id"), &FestivalPlotHook::set_id);
	ClassDB::bind_method(D_METHOD("get_id"), &FestivalPlotHook::get_id);
	ClassDB::bind_method(D_METHOD("set_title", "title"), &FestivalPlotHook::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &FestivalPlotHook::get_title);
	ClassDB::bind_method(D_METHOD("set_description", "description"), &FestivalPlotHook::set_description);
	ClassDB::bind_method(D_METHOD("get_description"), &FestivalPlotHook::get_description);
	ClassDB::bind_method(D_METHOD("set_status", "status"), &FestivalPlotHook::set_status);
	ClassDB::bind_method(D_METHOD("get_status"), &FestivalPlotHook::get_status);
	ClassDB::bind_method(D_METHOD("set_is_progression", "is_progression"), &FestivalPlotHook::set_is_progression);
	ClassDB::bind_method(D_METHOD("get_is_progression"), &FestivalPlotHook::get_is_progression);
	ClassDB::bind_method(D_METHOD("set_linked_npc_ids", "ids"), &FestivalPlotHook::set_linked_npc_ids);
	ClassDB::bind_method(D_METHOD("get_linked_npc_ids"), &FestivalPlotHook::get_linked_npc_ids);
	ClassDB::bind_method(D_METHOD("set_linked_location_ids", "ids"), &FestivalPlotHook::set_linked_location_ids);
	ClassDB::bind_method(D_METHOD("get_linked_location_ids"), &FestivalPlotHook::get_linked_location_ids);
	ClassDB::bind_method(D_METHOD("set_linked_item_ids", "ids"), &FestivalPlotHook::set_linked_item_ids);
	ClassDB::bind_method(D_METHOD("get_linked_item_ids"), &FestivalPlotHook::get_linked_item_ids);
	ClassDB::bind_method(D_METHOD("set_linked_rumor_ids", "ids"), &FestivalPlotHook::set_linked_rumor_ids);
	ClassDB::bind_method(D_METHOD("get_linked_rumor_ids"), &FestivalPlotHook::get_linked_rumor_ids);
	ClassDB::bind_method(D_METHOD("set_linked_event_ids", "ids"), &FestivalPlotHook::set_linked_event_ids);
	ClassDB::bind_method(D_METHOD("get_linked_event_ids"), &FestivalPlotHook::get_linked_event_ids);
	ClassDB::bind_method(D_METHOD("set_linked_outfit_ids", "ids"), &FestivalPlotHook::set_linked_outfit_ids);
	ClassDB::bind_method(D_METHOD("get_linked_outfit_ids"), &FestivalPlotHook::get_linked_outfit_ids);
	ClassDB::bind_method(D_METHOD("set_linked_npc_secrets", "secrets"), &FestivalPlotHook::set_linked_npc_secrets);
	ClassDB::bind_method(D_METHOD("get_linked_npc_secrets"), &FestivalPlotHook::get_linked_npc_secrets);
	ClassDB::bind_method(D_METHOD("set_census_data", "census_data"), &FestivalPlotHook::set_census_data);
	ClassDB::bind_method(D_METHOD("get_census_data"), &FestivalPlotHook::get_census_data);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "id"), "set_id", "get_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description", PROPERTY_HINT_MULTILINE_TEXT), "set_description", "get_description");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "status"), "set_status", "get_status");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_progression"), "set_is_progression", "get_is_progression");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "linked_npc_ids"), "set_linked_npc_ids", "get_linked_npc_ids");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "linked_location_ids"), "set_linked_location_ids", "get_linked_location_ids");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "linked_item_ids"), "set_linked_item_ids", "get_linked_item_ids");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "linked_rumor_ids"), "set_linked_rumor_ids", "get_linked_rumor_ids");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "linked_event_ids"), "set_linked_event_ids", "get_linked_event_ids");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "linked_outfit_ids"), "set_linked_outfit_ids", "get_linked_outfit_ids");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "linked_npc_secrets"), "set_linked_npc_secrets", "get_linked_npc_secrets");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "census_data"), "set_census_data", "get_census_data");
}
