/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "festival_census_importer.h"
#include "festival_clock.h"
#include "festival_director.h"
#include "festival_event.h"
#include "festival_item.h"
#include "festival_knowledge.h"
#include "festival_location.h"
#include "festival_notebook.h"
#include "festival_npc.h"
#include "festival_npc_profile.h"
#include "festival_outfit.h"
#include "festival_plot_hook.h"
#include "festival_registry.h"
#include "festival_weather.h"
#include "festival_world.h"

#include "core/config/engine.h"
#include "core/object/class_db.h"

void initialize_festival_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	// Authoring resources.
	GDREGISTER_CLASS(FestivalOutfit);
	GDREGISTER_CLASS(FestivalItem);
	GDREGISTER_CLASS(FestivalKnowledge);
	GDREGISTER_CLASS(FestivalNPCProfile);
	GDREGISTER_CLASS(FestivalLocation);
	GDREGISTER_CLASS(FestivalPlotHook);
	GDREGISTER_CLASS(FestivalEvent);

	// Scene node.
	GDREGISTER_CLASS(FestivalNPC);

	// Runtime singletons.
	GDREGISTER_CLASS(FestivalClock);
	GDREGISTER_CLASS(FestivalWeather);
	GDREGISTER_CLASS(FestivalWorld);
	GDREGISTER_CLASS(FestivalNotebook);
	GDREGISTER_CLASS(FestivalRegistry);
	GDREGISTER_CLASS(FestivalCensusImporter);
	GDREGISTER_CLASS(FestivalDirector);

	Engine *engine = Engine::get_singleton();
	engine->add_singleton(Engine::Singleton("FestivalClock", memnew(FestivalClock)));
	engine->add_singleton(Engine::Singleton("FestivalWeather", memnew(FestivalWeather)));
	engine->add_singleton(Engine::Singleton("FestivalWorld", memnew(FestivalWorld)));
	engine->add_singleton(Engine::Singleton("FestivalNotebook", memnew(FestivalNotebook)));
	engine->add_singleton(Engine::Singleton("FestivalRegistry", memnew(FestivalRegistry)));
	engine->add_singleton(Engine::Singleton("FestivalCensusImporter", memnew(FestivalCensusImporter)));
	// The director is exposed under the short name `Festival`.
	engine->add_singleton(Engine::Singleton("Festival", memnew(FestivalDirector)));
}

void uninitialize_festival_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	if (FestivalDirector::get_singleton()) {
		memdelete(FestivalDirector::get_singleton());
	}
	if (FestivalCensusImporter::get_singleton()) {
		memdelete(FestivalCensusImporter::get_singleton());
	}
	if (FestivalRegistry::get_singleton()) {
		memdelete(FestivalRegistry::get_singleton());
	}
	if (FestivalNotebook::get_singleton()) {
		memdelete(FestivalNotebook::get_singleton());
	}
	if (FestivalWorld::get_singleton()) {
		memdelete(FestivalWorld::get_singleton());
	}
	if (FestivalWeather::get_singleton()) {
		memdelete(FestivalWeather::get_singleton());
	}
	if (FestivalClock::get_singleton()) {
		memdelete(FestivalClock::get_singleton());
	}
}
