/**************************************************************************/
/*  test_festival.h                                                       */
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

#include "../festival_npc_profile.h"
#include "../festival_outfit.h"

#include "tests/test_macros.h"

namespace TestFestival {

TEST_CASE("[Modules][Festival] NPC schedule merges weather variant over the phase state") {
	Ref<FestivalNPCProfile> npc;
	npc.instantiate();

	Dictionary morning;
	morning["location"] = "plaza";
	morning["activity"] = "festival setup";
	npc->set_schedule_morning(morning);

	Dictionary rain;
	rain["location"] = "tavern"; // Rain drives this NPC indoors.
	npc->set_weather_variant_rain(rain);

	// Sun (weather 0): the base morning schedule is untouched.
	const Dictionary sun_state = npc->get_state_for(0, 0);
	CHECK(String(sun_state["location"]) == "plaza");
	CHECK(String(sun_state["activity"]) == "festival setup");

	// Rain (weather 1): only the overridden key changes, the rest is preserved.
	const Dictionary rain_state = npc->get_state_for(0, 1);
	CHECK(String(rain_state["location"]) == "tavern");
	CHECK(String(rain_state["activity"]) == "festival setup");
}

TEST_CASE("[Modules][Festival] Outfit tags and NPC knowledge id collection") {
	Ref<FestivalOutfit> outfit;
	outfit.instantiate();
	PackedStringArray tags;
	tags.push_back("uniform");
	tags.push_back("authority");
	outfit->set_tags(tags);
	CHECK(outfit->has_tag("uniform"));
	CHECK_FALSE(outfit->has_tag("mask"));

	Ref<FestivalNPCProfile> npc;
	npc.instantiate();
	npc->set_secret("secret_a");
	npc->set_rumor("rumor_b");
	// dark_secret and false_rumor are left empty and must be skipped.
	const PackedStringArray ids = npc->get_knowledge_ids();
	CHECK(ids.size() == 2);
	CHECK(ids.has("secret_a"));
	CHECK(ids.has("rumor_b"));
}

} // namespace TestFestival
