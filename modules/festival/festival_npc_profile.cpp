/**************************************************************************/
/*  festival_npc_profile.cpp                                              */
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

#include "festival_npc_profile.h"

#include "core/object/class_db.h"

void FestivalNPCProfile::set_id(const StringName &p_id) { id = p_id; }
StringName FestivalNPCProfile::get_id() const { return id; }
void FestivalNPCProfile::set_display_name(const String &p_display_name) { display_name = p_display_name; }
String FestivalNPCProfile::get_display_name() const { return display_name; }
void FestivalNPCProfile::set_species(const StringName &p_species) { species = p_species; }
StringName FestivalNPCProfile::get_species() const { return species; }
void FestivalNPCProfile::set_costume(const StringName &p_costume) { costume = p_costume; }
StringName FestivalNPCProfile::get_costume() const { return costume; }
void FestivalNPCProfile::set_surface_personality(const String &p_v) { surface_personality = p_v; }
String FestivalNPCProfile::get_surface_personality() const { return surface_personality; }
void FestivalNPCProfile::set_backstory(const String &p_backstory) { backstory = p_backstory; }
String FestivalNPCProfile::get_backstory() const { return backstory; }
void FestivalNPCProfile::set_workplace(const String &p_workplace) { workplace = p_workplace; }
String FestivalNPCProfile::get_workplace() const { return workplace; }
void FestivalNPCProfile::set_occupation(const String &p_occupation) { occupation = p_occupation; }
String FestivalNPCProfile::get_occupation() const { return occupation; }
void FestivalNPCProfile::set_residence(const String &p_residence) { residence = p_residence; }
String FestivalNPCProfile::get_residence() const { return residence; }
void FestivalNPCProfile::set_notes(const String &p_notes) { notes = p_notes; }
String FestivalNPCProfile::get_notes() const { return notes; }

void FestivalNPCProfile::set_schedule_morning(const Dictionary &p_schedule) { schedule_morning = p_schedule; }
Dictionary FestivalNPCProfile::get_schedule_morning() const { return schedule_morning; }
void FestivalNPCProfile::set_schedule_afternoon(const Dictionary &p_schedule) { schedule_afternoon = p_schedule; }
Dictionary FestivalNPCProfile::get_schedule_afternoon() const { return schedule_afternoon; }
void FestivalNPCProfile::set_schedule_night(const Dictionary &p_schedule) { schedule_night = p_schedule; }
Dictionary FestivalNPCProfile::get_schedule_night() const { return schedule_night; }

void FestivalNPCProfile::set_weather_variant_rain(const Dictionary &p_variant) { weather_variant_rain = p_variant; }
Dictionary FestivalNPCProfile::get_weather_variant_rain() const { return weather_variant_rain; }
void FestivalNPCProfile::set_weather_variant_sun(const Dictionary &p_variant) { weather_variant_sun = p_variant; }
Dictionary FestivalNPCProfile::get_weather_variant_sun() const { return weather_variant_sun; }

void FestivalNPCProfile::set_secret(const StringName &p_secret) { secret = p_secret; }
StringName FestivalNPCProfile::get_secret() const { return secret; }
void FestivalNPCProfile::set_dark_secret(const StringName &p_dark_secret) { dark_secret = p_dark_secret; }
StringName FestivalNPCProfile::get_dark_secret() const { return dark_secret; }
void FestivalNPCProfile::set_rumor(const StringName &p_rumor) { rumor = p_rumor; }
StringName FestivalNPCProfile::get_rumor() const { return rumor; }
void FestivalNPCProfile::set_false_rumor(const StringName &p_false_rumor) { false_rumor = p_false_rumor; }
StringName FestivalNPCProfile::get_false_rumor() const { return false_rumor; }

void FestivalNPCProfile::set_interactions(const Array &p_interactions) { interactions = p_interactions; }
Array FestivalNPCProfile::get_interactions() const { return interactions; }

void FestivalNPCProfile::set_relationships(const Array &p_relationships) { relationships = p_relationships; }
Array FestivalNPCProfile::get_relationships() const { return relationships; }
void FestivalNPCProfile::set_custom_fields(const Array &p_custom_fields) { custom_fields = p_custom_fields; }
Array FestivalNPCProfile::get_custom_fields() const { return custom_fields; }
void FestivalNPCProfile::set_gives_passwords(const PackedStringArray &p_passwords) { gives_passwords = p_passwords; }
PackedStringArray FestivalNPCProfile::get_gives_passwords() const { return gives_passwords; }
void FestivalNPCProfile::set_linked_plot_hook_ids(const PackedStringArray &p_ids) { linked_plot_hook_ids = p_ids; }
PackedStringArray FestivalNPCProfile::get_linked_plot_hook_ids() const { return linked_plot_hook_ids; }
void FestivalNPCProfile::set_census_data(const Dictionary &p_census_data) { census_data = p_census_data; }
Dictionary FestivalNPCProfile::get_census_data() const { return census_data; }

Dictionary FestivalNPCProfile::get_schedule_for_phase(int p_phase) const {
	switch (p_phase) {
		case 0:
			return schedule_morning;
		case 1:
			return schedule_afternoon;
		case 2:
			return schedule_night;
		default:
			return Dictionary();
	}
}

Dictionary FestivalNPCProfile::get_state_for(int p_phase, int p_weather) const {
	Dictionary base = get_schedule_for_phase(p_phase).duplicate();
	const Dictionary &variant = (p_weather == 1) ? weather_variant_rain : weather_variant_sun;
	static const char *phase_keys[3] = { "morning", "afternoon", "night" };
	const Array keys = variant.keys();
	for (int i = 0; i < keys.size(); i++) {
		const Variant &key = keys[i];
		bool phase_scoped = false;
		for (int p = 0; p < 3; p++) {
			if (key == Variant(phase_keys[p]) && variant[key].get_type() == Variant::DICTIONARY) {
				// Phase-scoped sub-variant: merge its contents only while that
				// phase is active (e.g. Secret Census morning-rain locations).
				if (p == p_phase) {
					const Dictionary sub = variant[key];
					const Array sub_keys = sub.keys();
					for (int j = 0; j < sub_keys.size(); j++) {
						base[sub_keys[j]] = sub[sub_keys[j]];
					}
				}
				phase_scoped = true;
				break;
			}
		}
		if (!phase_scoped) {
			base[key] = variant[key];
		}
	}
	return base;
}

PackedStringArray FestivalNPCProfile::get_knowledge_ids() const {
	PackedStringArray out;
	if (secret != StringName()) {
		out.push_back(secret);
	}
	if (dark_secret != StringName()) {
		out.push_back(dark_secret);
	}
	if (rumor != StringName()) {
		out.push_back(rumor);
	}
	if (false_rumor != StringName()) {
		out.push_back(false_rumor);
	}
	return out;
}

void FestivalNPCProfile::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_id", "id"), &FestivalNPCProfile::set_id);
	ClassDB::bind_method(D_METHOD("get_id"), &FestivalNPCProfile::get_id);
	ClassDB::bind_method(D_METHOD("set_display_name", "display_name"), &FestivalNPCProfile::set_display_name);
	ClassDB::bind_method(D_METHOD("get_display_name"), &FestivalNPCProfile::get_display_name);
	ClassDB::bind_method(D_METHOD("set_species", "species"), &FestivalNPCProfile::set_species);
	ClassDB::bind_method(D_METHOD("get_species"), &FestivalNPCProfile::get_species);
	ClassDB::bind_method(D_METHOD("set_costume", "costume"), &FestivalNPCProfile::set_costume);
	ClassDB::bind_method(D_METHOD("get_costume"), &FestivalNPCProfile::get_costume);
	ClassDB::bind_method(D_METHOD("set_surface_personality", "surface_personality"), &FestivalNPCProfile::set_surface_personality);
	ClassDB::bind_method(D_METHOD("get_surface_personality"), &FestivalNPCProfile::get_surface_personality);
	ClassDB::bind_method(D_METHOD("set_backstory", "backstory"), &FestivalNPCProfile::set_backstory);
	ClassDB::bind_method(D_METHOD("get_backstory"), &FestivalNPCProfile::get_backstory);
	ClassDB::bind_method(D_METHOD("set_workplace", "workplace"), &FestivalNPCProfile::set_workplace);
	ClassDB::bind_method(D_METHOD("get_workplace"), &FestivalNPCProfile::get_workplace);
	ClassDB::bind_method(D_METHOD("set_occupation", "occupation"), &FestivalNPCProfile::set_occupation);
	ClassDB::bind_method(D_METHOD("get_occupation"), &FestivalNPCProfile::get_occupation);
	ClassDB::bind_method(D_METHOD("set_residence", "residence"), &FestivalNPCProfile::set_residence);
	ClassDB::bind_method(D_METHOD("get_residence"), &FestivalNPCProfile::get_residence);
	ClassDB::bind_method(D_METHOD("set_notes", "notes"), &FestivalNPCProfile::set_notes);
	ClassDB::bind_method(D_METHOD("get_notes"), &FestivalNPCProfile::get_notes);

	ClassDB::bind_method(D_METHOD("set_schedule_morning", "schedule"), &FestivalNPCProfile::set_schedule_morning);
	ClassDB::bind_method(D_METHOD("get_schedule_morning"), &FestivalNPCProfile::get_schedule_morning);
	ClassDB::bind_method(D_METHOD("set_schedule_afternoon", "schedule"), &FestivalNPCProfile::set_schedule_afternoon);
	ClassDB::bind_method(D_METHOD("get_schedule_afternoon"), &FestivalNPCProfile::get_schedule_afternoon);
	ClassDB::bind_method(D_METHOD("set_schedule_night", "schedule"), &FestivalNPCProfile::set_schedule_night);
	ClassDB::bind_method(D_METHOD("get_schedule_night"), &FestivalNPCProfile::get_schedule_night);

	ClassDB::bind_method(D_METHOD("set_weather_variant_rain", "variant"), &FestivalNPCProfile::set_weather_variant_rain);
	ClassDB::bind_method(D_METHOD("get_weather_variant_rain"), &FestivalNPCProfile::get_weather_variant_rain);
	ClassDB::bind_method(D_METHOD("set_weather_variant_sun", "variant"), &FestivalNPCProfile::set_weather_variant_sun);
	ClassDB::bind_method(D_METHOD("get_weather_variant_sun"), &FestivalNPCProfile::get_weather_variant_sun);

	ClassDB::bind_method(D_METHOD("set_secret", "secret"), &FestivalNPCProfile::set_secret);
	ClassDB::bind_method(D_METHOD("get_secret"), &FestivalNPCProfile::get_secret);
	ClassDB::bind_method(D_METHOD("set_dark_secret", "dark_secret"), &FestivalNPCProfile::set_dark_secret);
	ClassDB::bind_method(D_METHOD("get_dark_secret"), &FestivalNPCProfile::get_dark_secret);
	ClassDB::bind_method(D_METHOD("set_rumor", "rumor"), &FestivalNPCProfile::set_rumor);
	ClassDB::bind_method(D_METHOD("get_rumor"), &FestivalNPCProfile::get_rumor);
	ClassDB::bind_method(D_METHOD("set_false_rumor", "false_rumor"), &FestivalNPCProfile::set_false_rumor);
	ClassDB::bind_method(D_METHOD("get_false_rumor"), &FestivalNPCProfile::get_false_rumor);

	ClassDB::bind_method(D_METHOD("set_interactions", "interactions"), &FestivalNPCProfile::set_interactions);
	ClassDB::bind_method(D_METHOD("get_interactions"), &FestivalNPCProfile::get_interactions);

	ClassDB::bind_method(D_METHOD("set_relationships", "relationships"), &FestivalNPCProfile::set_relationships);
	ClassDB::bind_method(D_METHOD("get_relationships"), &FestivalNPCProfile::get_relationships);
	ClassDB::bind_method(D_METHOD("set_custom_fields", "custom_fields"), &FestivalNPCProfile::set_custom_fields);
	ClassDB::bind_method(D_METHOD("get_custom_fields"), &FestivalNPCProfile::get_custom_fields);
	ClassDB::bind_method(D_METHOD("set_gives_passwords", "passwords"), &FestivalNPCProfile::set_gives_passwords);
	ClassDB::bind_method(D_METHOD("get_gives_passwords"), &FestivalNPCProfile::get_gives_passwords);
	ClassDB::bind_method(D_METHOD("set_linked_plot_hook_ids", "ids"), &FestivalNPCProfile::set_linked_plot_hook_ids);
	ClassDB::bind_method(D_METHOD("get_linked_plot_hook_ids"), &FestivalNPCProfile::get_linked_plot_hook_ids);
	ClassDB::bind_method(D_METHOD("set_census_data", "census_data"), &FestivalNPCProfile::set_census_data);
	ClassDB::bind_method(D_METHOD("get_census_data"), &FestivalNPCProfile::get_census_data);

	ClassDB::bind_method(D_METHOD("get_schedule_for_phase", "phase"), &FestivalNPCProfile::get_schedule_for_phase);
	ClassDB::bind_method(D_METHOD("get_state_for", "phase", "weather"), &FestivalNPCProfile::get_state_for);
	ClassDB::bind_method(D_METHOD("get_knowledge_ids"), &FestivalNPCProfile::get_knowledge_ids);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "id"), "set_id", "get_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "display_name"), "set_display_name", "get_display_name");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "species"), "set_species", "get_species");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "costume"), "set_costume", "get_costume");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "surface_personality", PROPERTY_HINT_MULTILINE_TEXT), "set_surface_personality", "get_surface_personality");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "backstory", PROPERTY_HINT_MULTILINE_TEXT), "set_backstory", "get_backstory");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "workplace"), "set_workplace", "get_workplace");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "occupation"), "set_occupation", "get_occupation");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "residence"), "set_residence", "get_residence");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "notes", PROPERTY_HINT_MULTILINE_TEXT), "set_notes", "get_notes");

	ADD_GROUP("Schedule", "schedule_");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "schedule_morning"), "set_schedule_morning", "get_schedule_morning");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "schedule_afternoon"), "set_schedule_afternoon", "get_schedule_afternoon");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "schedule_night"), "set_schedule_night", "get_schedule_night");

	ADD_GROUP("Weather Variants", "weather_variant_");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "weather_variant_rain"), "set_weather_variant_rain", "get_weather_variant_rain");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "weather_variant_sun"), "set_weather_variant_sun", "get_weather_variant_sun");

	ADD_GROUP("Knowledge", "");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "secret"), "set_secret", "get_secret");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "dark_secret"), "set_dark_secret", "get_dark_secret");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "rumor"), "set_rumor", "get_rumor");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "false_rumor"), "set_false_rumor", "get_false_rumor");

	ADD_GROUP("Interactions", "");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "interactions"), "set_interactions", "get_interactions");

	ADD_GROUP("Connections", "");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "relationships"), "set_relationships", "get_relationships");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "custom_fields"), "set_custom_fields", "get_custom_fields");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "gives_passwords"), "set_gives_passwords", "get_gives_passwords");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "linked_plot_hook_ids"), "set_linked_plot_hook_ids", "get_linked_plot_hook_ids");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "census_data"), "set_census_data", "get_census_data");
}
