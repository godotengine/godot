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

#include "../festival_census_importer.h"
#include "../festival_npc_profile.h"
#include "../festival_outfit.h"
#include "../festival_registry.h"

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

TEST_CASE("[Modules][Festival] Phase-scoped weather variants merge only for their phase") {
	Ref<FestivalNPCProfile> npc;
	npc.instantiate();

	Dictionary morning;
	morning["location"] = "plaza";
	npc->set_schedule_morning(morning);
	Dictionary afternoon;
	afternoon["location"] = "market";
	npc->set_schedule_afternoon(afternoon);

	// Secret Census only has a rain override for the morning; it must not
	// leak into the afternoon schedule.
	Dictionary morning_rain;
	morning_rain["location"] = "tavern";
	Dictionary rain_variant;
	rain_variant["morning"] = morning_rain;
	npc->set_weather_variant_rain(rain_variant);

	CHECK(String(npc->get_state_for(0, 0)["location"]) == "plaza");
	CHECK(String(npc->get_state_for(0, 1)["location"]) == "tavern");
	CHECK(String(npc->get_state_for(1, 1)["location"]) == "market");
}

TEST_CASE("[Modules][Festival] Password knowledge ids are derived per the census contract") {
	CHECK(FestivalCensusImporter::password_knowledge_id("Say Friend & Enter") == "password.say_friend_enter");
	CHECK(FestivalCensusImporter::password_knowledge_id("  swordfish  ") == "password.swordfish");
	CHECK(FestivalCensusImporter::password_knowledge_id("A1-B2") == "password.a1_b2");
}

TEST_CASE("[Modules][Festival] Census importer rejects bad packages") {
	// The engine singletons are created by the module's SCENE-level init.
	FestivalCensusImporter *importer = FestivalCensusImporter::get_singleton();
	REQUIRE(importer != nullptr);

	Dictionary bad_json = importer->import_text("not json at all");
	CHECK_FALSE((bool)bad_json["ok"]);

	Dictionary wrong_format = importer->import_text("{\"format\":\"something-else\",\"formatVersion\":1}");
	CHECK_FALSE((bool)wrong_format["ok"]);

	Dictionary future_version = importer->import_text("{\"format\":\"secret-census-world\",\"formatVersion\":2}");
	CHECK_FALSE((bool)future_version["ok"]);
}

TEST_CASE("[Modules][Festival] Census importer builds the connected world") {
	FestivalRegistry *registry = FestivalRegistry::get_singleton();
	FestivalCensusImporter *importer = FestivalCensusImporter::get_singleton();
	REQUIRE(registry != nullptr);
	REQUIRE(importer != nullptr);
	registry->clear();

	const String package = R"({
		"format": "secret-census-world",
		"formatVersion": 1,
		"exportedAt": 1751500000000,
		"game": { "id": "game1", "name": "Kraed Maas", "description": "", "createdAt": 0 },
		"settings": { "gameStates": ["Morning", "Afternoon", "Night"], "worldRules": "Costumes are real." },
		"npcs": [{
			"id": "mayor",
			"name": "Mayor Sorrel",
			"species": "Badger",
			"personality": "Genial in public.",
			"secret": "Rigged the raffle.",
			"darkSecret": "",
			"rumors": [
				{ "id": "r1", "text": "He naps in the archive.", "isTrue": true, "sourceType": "external", "sourceId": "gossip", "verification": "" },
				{ "id": "r2", "text": "He is secretly wealthy.", "isTrue": false, "sourceType": "external", "sourceId": "gossip", "verification": "" }
			],
			"relationships": [{ "targetNpcId": "clerk", "type": "Employer", "reciprocalType": "Employee" }],
			"customFields": [{ "id": "cf1", "label": "Favorite tea", "value": "Nettle" }],
			"locations": { "morningSun": "plaza", "morningRain": "town_hall", "afternoon": "market", "night": "manor" },
			"dialogue": [{
				"id": "d1",
				"gameState": "Morning",
				"outfitId": "constable",
				"requiredPassword": "Say Friend & Enter",
				"type": "secret",
				"hasSpecialItem": true,
				"specialItemName": "Warrant",
				"text": "Fine, I confess about the raffle.",
				"givesItemId": "ledger",
				"givesPassword": "archive key"
			}],
			"givesPasswords": ["archive key"],
			"linkedPlotHookIds": ["hook1"]
		}],
		"locations": [{
			"id": "plaza",
			"name": "Festival Plaza",
			"type": "overworld",
			"parentId": null,
			"isResidence": false,
			"residentNpcIds": [],
			"connectedLocationIds": ["market"],
			"requiredPassword": "",
			"isTemplate": false
		}],
		"items": [{
			"id": "warrant",
			"name": "Warrant",
			"description": "An official-looking paper.",
			"locationType": "location",
			"locationId": "plaza",
			"grantsExtraPower": true,
			"powerDescription": "Compels honesty.",
			"linkedPlotHookId": "hook1",
			"stageAvailability": [{ "gameState": "Morning", "locationType": "location", "locationId": "plaza", "availableByNormalMeans": true }]
		}],
		"outfits": [{
			"id": "constable",
			"name": "Constable Uniform",
			"description": "Crisp and blue.",
			"grantsExtraPower": false,
			"linkedDialogueIds": ["d1"]
		}],
		"plotHooks": [{
			"id": "hook1",
			"title": "The rigged raffle",
			"status": "active",
			"linkedNpcIds": ["mayor"],
			"linkedItemIds": ["warrant"],
			"linkedNpcSecrets": [{ "npcId": "mayor", "secretType": "secret" }]
		}],
		"events": [{
			"id": "ev1",
			"name": "Raffle draw",
			"script": "The mayor draws the winning ticket.",
			"locationId": "plaza",
			"triggerNpcId": "mayor",
			"triggerDialogueId": "d1",
			"characterIds": ["mayor"],
			"gameStates": ["Afternoon"]
		}]
	})";

	const Dictionary summary = importer->import_text(package);
	CHECK((bool)summary["ok"]);
	CHECK((int)summary["npcs"] == 1);
	CHECK((int)summary["locations"] == 1);
	CHECK((int)summary["items"] == 1);
	CHECK((int)summary["outfits"] == 1);
	CHECK((int)summary["plot_hooks"] == 1);
	CHECK((int)summary["events"] == 1);
	// mayor secret + 2 rumors + 2 passwords ("Say Friend & Enter", "archive key").
	CHECK((int)summary["knowledge"] == 5);

	// Everything landed in the registry, connected by the same ids.
	Ref<FestivalNPCProfile> mayor = registry->get_npc("mayor");
	REQUIRE(mayor.is_valid());
	CHECK(mayor->get_display_name() == "Mayor Sorrel");
	CHECK(mayor->get_secret() == StringName("npc.mayor.secret"));
	CHECK(mayor->get_dark_secret() == StringName());
	CHECK(mayor->get_rumor() == StringName("rumor.r1"));
	CHECK(mayor->get_false_rumor() == StringName("rumor.r2"));
	CHECK(mayor->get_relationships().size() == 1);
	CHECK(mayor->get_custom_fields().size() == 1);
	CHECK(mayor->get_linked_plot_hook_ids().has("hook1"));
	CHECK(String(mayor->get_census_data()["species"]) == "Badger");

	// Schedules: sun morning in the plaza, rain morning scoped to town hall,
	// afternoon untouched by the rain variant.
	CHECK(String(mayor->get_state_for(0, 0)["location"]) == "plaza");
	CHECK(String(mayor->get_state_for(0, 1)["location"]) == "town_hall");
	CHECK(String(mayor->get_state_for(1, 1)["location"]) == "market");

	// The dialogue entry became a director-ready interaction.
	REQUIRE(mayor->get_interactions().size() == 1);
	const Dictionary interaction = mayor->get_interactions()[0];
	CHECK(interaction["id"] == Variant(StringName("d1")));
	CHECK(String(interaction["dialogue"]) == "Fine, I confess about the raffle.");
	CHECK(interaction["requires_outfit"] == Variant(StringName("constable")));
	const Array requires_knowledge = interaction["requires_knowledge"];
	CHECK(requires_knowledge.has("password.say_friend_enter"));
	const Array grants = interaction["grants_knowledge"];
	CHECK(grants.has("npc.mayor.secret"));
	CHECK(grants.has("password.archive_key"));
	const Array requires_items = interaction["requires_items"];
	CHECK(requires_items.has("warrant")); // Resolved from "Warrant" by name.
	const Array gives_items = interaction["gives_items"];
	CHECK(gives_items.has("ledger"));
	const Array phases = interaction["requires_phase"];
	CHECK(phases.has(0)); // "Morning" maps to phase 0.
	CHECK(!((Dictionary)interaction["census"]).is_empty());

	// Generated knowledge is registered and correctly categorized.
	Ref<FestivalKnowledge> rumor2 = registry->get_knowledge("rumor.r2");
	REQUIRE(rumor2.is_valid());
	CHECK(rumor2->get_category() == FestivalKnowledge::CATEGORY_FALSE_RUMOR);
	CHECK_FALSE(rumor2->get_veracity());
	CHECK(rumor2->get_subject() == StringName("mayor"));
	Ref<FestivalKnowledge> password = registry->get_knowledge("password.archive_key");
	REQUIRE(password.is_valid());
	CHECK(password->get_category() == FestivalKnowledge::CATEGORY_PASSWORD);

	// The other entity types kept their fields and connections.
	Ref<FestivalLocation> plaza = registry->get_location("plaza");
	REQUIRE(plaza.is_valid());
	CHECK(plaza->get_kind() == StringName("overworld"));
	CHECK(plaza->get_connected_location_ids().has("market"));
	Ref<FestivalItem> warrant = registry->get_item("warrant");
	REQUIRE(warrant.is_valid());
	CHECK(warrant->get_holder_type() == StringName("location"));
	CHECK(warrant->get_holder_id() == StringName("plaza"));
	CHECK(warrant->get_grants_extra_power());
	CHECK(warrant->get_stage_availability().size() == 1);
	Ref<FestivalOutfit> constable = registry->get_outfit("constable");
	REQUIRE(constable.is_valid());
	CHECK(constable->get_linked_dialogue_ids().has("d1"));
	Ref<FestivalPlotHook> hook = registry->get_plot_hook("hook1");
	REQUIRE(hook.is_valid());
	CHECK(hook->get_status() == StringName("active"));
	CHECK(hook->get_linked_npc_ids().has("mayor"));
	CHECK(hook->get_linked_npc_secrets().size() == 1);
	Ref<FestivalEvent> event = registry->get_event("ev1");
	REQUIRE(event.is_valid());
	CHECK(event->get_trigger_npc_id() == StringName("mayor"));
	CHECK(event->get_game_states().has("Afternoon"));

	// World metadata rides along for the game to read.
	const Dictionary world_info = registry->get_world_info();
	CHECK(String(((Dictionary)world_info["game"])["name"]) == "Kraed Maas");
	CHECK(String(((Dictionary)world_info["settings"])["worldRules"]) == "Costumes are real.");

	registry->clear();
}

TEST_CASE("[Modules][Festival] Census importer accepts sparse packages") {
	FestivalCensusImporter *importer = FestivalCensusImporter::get_singleton();
	REQUIRE(importer != nullptr);

	// Not every field has to be filled out to be exported.
	const Dictionary summary = importer->import_text(
			"{\"format\":\"secret-census-world\",\"formatVersion\":1,"
			"\"npcs\":[{\"id\":\"ghost\"}]}");
	CHECK((bool)summary["ok"]);
	CHECK((int)summary["npcs"] == 1);
	CHECK((int)summary["knowledge"] == 0);

	FestivalRegistry *registry = FestivalRegistry::get_singleton();
	if (registry) {
		registry->clear();
	}
}

} // namespace TestFestival
