/*************************************************************************/
/*  scenario_fx.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "scenario_fx.h"

void WorldEnvironment::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_WORLD) {

		if (environment.is_valid()) {
			if (get_world()->get_environment().is_valid()) {
				WARN_PRINT("World already has an environment (Another WorldEnvironment?), overriding.");
			}
			get_world()->set_environment(environment);
			add_to_group("_world_environment_" + itos(get_world()->get_scenario().get_id()));
		}

	} else if (p_what == NOTIFICATION_EXIT_WORLD) {

		if (environment.is_valid() && get_world()->get_environment() == environment) {
			get_world()->set_environment(Ref<Environment>());
			remove_from_group("_world_environment_" + itos(get_world()->get_scenario().get_id()));
		}
	}
}

void WorldEnvironment::set_environment(const Ref<Environment> &p_environment) {

	if (is_inside_world() && environment.is_valid() && get_world()->get_environment() == environment) {
		get_world()->set_environment(Ref<Environment>());
		remove_from_group("_world_environment_" + itos(get_world()->get_scenario().get_id()));
		//clean up
	}

	environment = p_environment;
	if (is_inside_world() && environment.is_valid()) {
		if (get_world()->get_environment().is_valid()) {
			WARN_PRINT("World already has an environment (Another WorldEnvironment?), overriding.");
		}
		get_world()->set_environment(environment);
		add_to_group("_world_environment_" + itos(get_world()->get_scenario().get_id()));
	}

	update_configuration_warning();
}

Ref<Environment> WorldEnvironment::get_environment() const {

	return environment;
}

String WorldEnvironment::get_configuration_warning() const {

	if (!is_visible_in_tree() || !is_inside_tree() || !environment.is_valid())
		return String();

	List<Node *> nodes;
	get_tree()->get_nodes_in_group("_world_environment_" + itos(get_world()->get_scenario().get_id()), &nodes);

	if (nodes.size() > 1) {
		return TTR("Only one WorldEnvironment is allowed per scene (or set of instanced scenes).");
	}

	return String();
}

void WorldEnvironment::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_environment", "env:Environment"), &WorldEnvironment::set_environment);
	ClassDB::bind_method(D_METHOD("get_environment:Environment"), &WorldEnvironment::get_environment);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "environment", PROPERTY_HINT_RESOURCE_TYPE, "Environment"), "set_environment", "get_environment");
}

WorldEnvironment::WorldEnvironment() {
}
