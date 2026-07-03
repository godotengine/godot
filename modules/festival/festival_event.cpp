/**************************************************************************/
/*  festival_event.cpp                                                    */
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

#include "festival_event.h"

#include "core/object/class_db.h"

void FestivalEvent::set_id(const StringName &p_id) { id = p_id; }
StringName FestivalEvent::get_id() const { return id; }
void FestivalEvent::set_display_name(const String &p_display_name) { display_name = p_display_name; }
String FestivalEvent::get_display_name() const { return display_name; }
void FestivalEvent::set_description(const String &p_description) { description = p_description; }
String FestivalEvent::get_description() const { return description; }
void FestivalEvent::set_script_text(const String &p_script_text) { script_text = p_script_text; }
String FestivalEvent::get_script_text() const { return script_text; }
void FestivalEvent::set_location_id(const StringName &p_location_id) { location_id = p_location_id; }
StringName FestivalEvent::get_location_id() const { return location_id; }
void FestivalEvent::set_trigger_npc_id(const StringName &p_trigger_npc_id) { trigger_npc_id = p_trigger_npc_id; }
StringName FestivalEvent::get_trigger_npc_id() const { return trigger_npc_id; }
void FestivalEvent::set_trigger_dialogue_id(const StringName &p_trigger_dialogue_id) { trigger_dialogue_id = p_trigger_dialogue_id; }
StringName FestivalEvent::get_trigger_dialogue_id() const { return trigger_dialogue_id; }
void FestivalEvent::set_character_ids(const PackedStringArray &p_ids) { character_ids = p_ids; }
PackedStringArray FestivalEvent::get_character_ids() const { return character_ids; }
void FestivalEvent::set_game_states(const PackedStringArray &p_game_states) { game_states = p_game_states; }
PackedStringArray FestivalEvent::get_game_states() const { return game_states; }
void FestivalEvent::set_is_progression(bool p_is_progression) { is_progression = p_is_progression; }
bool FestivalEvent::get_is_progression() const { return is_progression; }
void FestivalEvent::set_census_data(const Dictionary &p_census_data) { census_data = p_census_data; }
Dictionary FestivalEvent::get_census_data() const { return census_data; }

void FestivalEvent::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_id", "id"), &FestivalEvent::set_id);
	ClassDB::bind_method(D_METHOD("get_id"), &FestivalEvent::get_id);
	ClassDB::bind_method(D_METHOD("set_display_name", "display_name"), &FestivalEvent::set_display_name);
	ClassDB::bind_method(D_METHOD("get_display_name"), &FestivalEvent::get_display_name);
	ClassDB::bind_method(D_METHOD("set_description", "description"), &FestivalEvent::set_description);
	ClassDB::bind_method(D_METHOD("get_description"), &FestivalEvent::get_description);
	ClassDB::bind_method(D_METHOD("set_script_text", "script_text"), &FestivalEvent::set_script_text);
	ClassDB::bind_method(D_METHOD("get_script_text"), &FestivalEvent::get_script_text);
	ClassDB::bind_method(D_METHOD("set_location_id", "location_id"), &FestivalEvent::set_location_id);
	ClassDB::bind_method(D_METHOD("get_location_id"), &FestivalEvent::get_location_id);
	ClassDB::bind_method(D_METHOD("set_trigger_npc_id", "trigger_npc_id"), &FestivalEvent::set_trigger_npc_id);
	ClassDB::bind_method(D_METHOD("get_trigger_npc_id"), &FestivalEvent::get_trigger_npc_id);
	ClassDB::bind_method(D_METHOD("set_trigger_dialogue_id", "trigger_dialogue_id"), &FestivalEvent::set_trigger_dialogue_id);
	ClassDB::bind_method(D_METHOD("get_trigger_dialogue_id"), &FestivalEvent::get_trigger_dialogue_id);
	ClassDB::bind_method(D_METHOD("set_character_ids", "ids"), &FestivalEvent::set_character_ids);
	ClassDB::bind_method(D_METHOD("get_character_ids"), &FestivalEvent::get_character_ids);
	ClassDB::bind_method(D_METHOD("set_game_states", "game_states"), &FestivalEvent::set_game_states);
	ClassDB::bind_method(D_METHOD("get_game_states"), &FestivalEvent::get_game_states);
	ClassDB::bind_method(D_METHOD("set_is_progression", "is_progression"), &FestivalEvent::set_is_progression);
	ClassDB::bind_method(D_METHOD("get_is_progression"), &FestivalEvent::get_is_progression);
	ClassDB::bind_method(D_METHOD("set_census_data", "census_data"), &FestivalEvent::set_census_data);
	ClassDB::bind_method(D_METHOD("get_census_data"), &FestivalEvent::get_census_data);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "id"), "set_id", "get_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "display_name"), "set_display_name", "get_display_name");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description", PROPERTY_HINT_MULTILINE_TEXT), "set_description", "get_description");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "script_text", PROPERTY_HINT_MULTILINE_TEXT), "set_script_text", "get_script_text");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "location_id"), "set_location_id", "get_location_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "trigger_npc_id"), "set_trigger_npc_id", "get_trigger_npc_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "trigger_dialogue_id"), "set_trigger_dialogue_id", "get_trigger_dialogue_id");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "character_ids"), "set_character_ids", "get_character_ids");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "game_states"), "set_game_states", "get_game_states");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_progression"), "set_is_progression", "get_is_progression");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "census_data"), "set_census_data", "get_census_data");
}
