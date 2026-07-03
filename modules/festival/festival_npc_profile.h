/**************************************************************************/
/*  festival_npc_profile.h                                                */
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

#pragma once

#include "core/io/resource.h"

// The full authoring schema for one islander. NPCs are layered systems, not
// one-note quest dispensers: each carries schedules per phase, weather variants,
// four kinds of guarded knowledge, and a data-driven list of conditional
// interactions evaluated by the FestivalDirector.
//
// Each entry of `interactions` is a Dictionary understood by the director:
//   {
//     "id": StringName,                 # unique within this NPC
//     "requires_outfit": StringName/Array, # accepted outfit id(s) (optional)
//     "requires_role": StringName/Array,   # accepted perceived role(s) (optional)
//     "requires_items": Array,          # item ids that must be held
//     "requires_presented": Array,      # item ids that must be visibly presented
//     "requires_knowledge": Array,      # knowledge ids Alex must already know
//     "requires_flags": Dictionary,     # world flags that must equal given values
//     "requires_phase": Array,          # allowed FestivalClock.Phase ints (empty = any)
//     "requires_weather": int,          # FestivalWeather.Weather int, or -1 for any
//     "dialogue": String,               # line shown when chosen
//     "grants_knowledge": Array,        # knowledge ids learned (persist across runs)
//     "sets_flags": Dictionary,         # world flags written on apply
//     "gives_items": Array,             # item ids added to inventory
//     "takes_items": Array,             # item ids removed from inventory
//     "advance_phase": bool,            # advance the clock when applied
//     "trigger_milestone": StringName,  # milestone raised when applied
//   }
class FestivalNPCProfile : public Resource {
	GDCLASS(FestivalNPCProfile, Resource);

	StringName id;
	String display_name;
	StringName species;
	StringName costume; // The outfit id this NPC themselves wears.
	String surface_personality;
	String backstory;
	String workplace;
	String occupation;
	String residence;
	String notes;

	Dictionary schedule_morning;
	Dictionary schedule_afternoon;
	Dictionary schedule_night;

	Dictionary weather_variant_rain;
	Dictionary weather_variant_sun;

	StringName secret;
	StringName dark_secret;
	StringName rumor;
	StringName false_rumor;

	Array interactions;

	// Directed NPC->NPC edges: { "targetNpcId": String, "type": String,
	// "reciprocalType": String } (camelCase keys are part of the Secret Census
	// import contract).
	Array relationships;
	// Arbitrary authored fields: { "id", "label", "value" }.
	Array custom_fields;
	PackedStringArray gives_passwords;
	PackedStringArray linked_plot_hook_ids;
	// The raw Secret Census record this profile was imported from (verbatim).
	Dictionary census_data;

protected:
	static void _bind_methods();

public:
	void set_id(const StringName &p_id);
	StringName get_id() const;
	void set_display_name(const String &p_display_name);
	String get_display_name() const;
	void set_species(const StringName &p_species);
	StringName get_species() const;
	void set_costume(const StringName &p_costume);
	StringName get_costume() const;
	void set_surface_personality(const String &p_surface_personality);
	String get_surface_personality() const;
	void set_backstory(const String &p_backstory);
	String get_backstory() const;
	void set_workplace(const String &p_workplace);
	String get_workplace() const;
	void set_occupation(const String &p_occupation);
	String get_occupation() const;
	void set_residence(const String &p_residence);
	String get_residence() const;
	void set_notes(const String &p_notes);
	String get_notes() const;

	void set_schedule_morning(const Dictionary &p_schedule);
	Dictionary get_schedule_morning() const;
	void set_schedule_afternoon(const Dictionary &p_schedule);
	Dictionary get_schedule_afternoon() const;
	void set_schedule_night(const Dictionary &p_schedule);
	Dictionary get_schedule_night() const;

	void set_weather_variant_rain(const Dictionary &p_variant);
	Dictionary get_weather_variant_rain() const;
	void set_weather_variant_sun(const Dictionary &p_variant);
	Dictionary get_weather_variant_sun() const;

	void set_secret(const StringName &p_secret);
	StringName get_secret() const;
	void set_dark_secret(const StringName &p_dark_secret);
	StringName get_dark_secret() const;
	void set_rumor(const StringName &p_rumor);
	StringName get_rumor() const;
	void set_false_rumor(const StringName &p_false_rumor);
	StringName get_false_rumor() const;

	void set_interactions(const Array &p_interactions);
	Array get_interactions() const;

	void set_relationships(const Array &p_relationships);
	Array get_relationships() const;
	void set_custom_fields(const Array &p_custom_fields);
	Array get_custom_fields() const;
	void set_gives_passwords(const PackedStringArray &p_passwords);
	PackedStringArray get_gives_passwords() const;
	void set_linked_plot_hook_ids(const PackedStringArray &p_ids);
	PackedStringArray get_linked_plot_hook_ids() const;
	void set_census_data(const Dictionary &p_census_data);
	Dictionary get_census_data() const;

	// Convenience: the base schedule Dictionary for a FestivalClock.Phase int.
	Dictionary get_schedule_for_phase(int p_phase) const;
	// The schedule for a phase with the matching weather variant merged on top.
	// Variant keys "morning"/"afternoon"/"night" holding Dictionaries are
	// phase-scoped: they merge only when that phase is active. Any other
	// variant key merges for every phase.
	Dictionary get_state_for(int p_phase, int p_weather) const;
	// Knowledge ids of the four guarded facts, skipping empty ones.
	PackedStringArray get_knowledge_ids() const;
};
