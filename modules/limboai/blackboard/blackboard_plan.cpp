/**
 * blackboard_plan.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "blackboard_plan.h"

bool BlackboardPlan::_set(const StringName &p_name, const Variant &p_value) {
	String prop_name = p_name;

	// * Editor
	if (var_map.has(prop_name)) {
		var_map[prop_name].set_value(p_value);
		if (base.is_valid() && p_value == base->get_var(prop_name).get_value()) {
			// When user pressed reset property button in inspector...
			var_map[prop_name].reset_value_changed();
		}
		return true;
	}

	// * Storage
	if (prop_name.begins_with("var/")) {
		String var_name = prop_name.get_slicec('/', 1);
		String what = prop_name.get_slicec('/', 2);
		if (!var_map.has(var_name) && what == "name") {
			add_var(var_name, BBVariable());
		}
		if (what == "name") {
			// We don't store variable name with the variable.
		} else if (what == "type") {
			var_map[var_name].set_type((Variant::Type)(int)p_value);
		} else if (what == "value") {
			var_map[var_name].set_value(p_value);
		} else if (what == "hint") {
			var_map[var_name].set_hint((PropertyHint)(int)p_value);
		} else if (what == "hint_string") {
			var_map[var_name].set_hint_string(p_value);
		} else {
			return false;
		}
		return true;
	}

	return false;
}

bool BlackboardPlan::_get(const StringName &p_name, Variant &r_ret) const {
	String prop_name = p_name;

	// * Editor
	if (var_map.has(prop_name)) {
		r_ret = var_map[prop_name].get_value();
		return true;
	}

	// * Storage
	if (!prop_name.begins_with("var/")) {
		return false;
	}

	String var_name = prop_name.get_slicec('/', 1);
	String what = prop_name.get_slicec('/', 2);
	ERR_FAIL_COND_V(!var_map.has(var_name), false);

	if (what == "name") {
		r_ret = var_name;
	} else if (what == "type") {
		r_ret = var_map[var_name].get_type();
	} else if (what == "value") {
		r_ret = var_map[var_name].get_value();
	} else if (what == "hint") {
		r_ret = var_map[var_name].get_hint();
	} else if (what == "hint_string") {
		r_ret = var_map[var_name].get_hint_string();
	}
	return true;
}

void BlackboardPlan::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const Pair<String, BBVariable> &p : var_list) {
		String var_name = p.first;
		BBVariable var = p.second;

		// * Editor
		if (!is_derived() || !var_name.begins_with("_")) {
			p_list->push_back(PropertyInfo(var.get_type(), var_name, var.get_hint(), var.get_hint_string(), PROPERTY_USAGE_EDITOR));
		}

		// * Storage
		p_list->push_back(PropertyInfo(Variant::STRING, "var/" + var_name + "/name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::INT, "var/" + var_name + "/type", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(var.get_type(), "var/" + var_name + "/value", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::INT, "var/" + var_name + "/hint", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::STRING, "var/" + var_name + "/hint_string", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
	}
}

bool BlackboardPlan::_property_can_revert(const StringName &p_name) const {
	return base.is_valid() && base->var_map.has(p_name);
}

bool BlackboardPlan::_property_get_revert(const StringName &p_name, Variant &r_property) const {
	if (base->var_map.has(p_name)) {
		r_property = base->var_map[p_name].get_value();
		return true;
	}
	return false;
}

void BlackboardPlan::set_base_plan(const Ref<BlackboardPlan> &p_base) {
	base = p_base;
	sync_with_base_plan();
}

void BlackboardPlan::add_var(const String &p_name, const BBVariable &p_var) {
	ERR_FAIL_COND(var_map.has(p_name));
	var_map.insert(p_name, p_var);
	var_list.push_back(Pair<String, BBVariable>(p_name, p_var));
	notify_property_list_changed();
	emit_changed();
}

void BlackboardPlan::remove_var(const String &p_name) {
	ERR_FAIL_COND(!var_map.has(p_name));
	var_list.erase(Pair<String, BBVariable>(p_name, var_map[p_name]));
	var_map.erase(p_name);
	notify_property_list_changed();
	emit_changed();
}

BBVariable BlackboardPlan::get_var(const String &p_name) {
	ERR_FAIL_COND_V(!var_map.has(p_name), BBVariable());
	return var_map.get(p_name);
}

Pair<String, BBVariable> BlackboardPlan::get_var_by_index(int p_index) {
	Pair<String, BBVariable> ret;
	ERR_FAIL_INDEX_V(p_index, (int)var_map.size(), ret);
	return var_list[p_index];
}

PackedStringArray BlackboardPlan::list_vars() const {
	PackedStringArray ret;
	for (const Pair<String, BBVariable> &p : var_list) {
		ret.append(p.first);
	}
	return ret;
}

String BlackboardPlan::get_var_name(const BBVariable &p_var) const {
	for (const Pair<String, BBVariable> &p : var_list) {
		if (p.second == p_var) {
			return p.first;
		}
	}
	return String();
}

bool BlackboardPlan::is_valid_var_name(const String &p_name) const {
	if (p_name.begins_with("resource_")) {
		return false;
	}
	return p_name.is_valid_identifier() && !var_map.has(p_name);
}

void BlackboardPlan::rename_var(const String &p_name, const String &p_new_name) {
	if (p_name == p_new_name) {
		return;
	}

	ERR_FAIL_COND(!is_valid_var_name(p_new_name));
	ERR_FAIL_COND(!var_map.has(p_name));

	BBVariable var = var_map[p_name];
	Pair<String, BBVariable> new_entry(p_new_name, var);
	Pair<String, BBVariable> old_entry(p_name, var);
	var_list.find(old_entry)->set(new_entry);

	var_map.erase(p_name);
	var_map.insert(p_new_name, var);

	notify_property_list_changed();
	emit_changed();
}

void BlackboardPlan::move_var(int p_index, int p_new_index) {
	ERR_FAIL_INDEX(p_index, (int)var_map.size());
	ERR_FAIL_INDEX(p_new_index, (int)var_map.size());

	if (p_index == p_new_index) {
		return;
	}

	List<Pair<String, BBVariable>>::Element *E = var_list.front();
	for (int i = 0; i < p_index; i++) {
		E = E->next();
	}
	List<Pair<String, BBVariable>>::Element *E2 = var_list.front();
	for (int i = 0; i < p_new_index; i++) {
		E2 = E2->next();
	}

	var_list.move_before(E, E2);
	if (p_new_index > p_index) {
		var_list.move_before(E2, E);
	}

	notify_property_list_changed();
	emit_changed();
}

void BlackboardPlan::sync_with_base_plan() {
	if (base.is_null()) {
		return;
	}

	bool changed = false;

	// Sync variables with the base plan.
	for (const Pair<String, BBVariable> &p : base->var_list) {
		const String &base_name = p.first;
		const BBVariable &base_var = p.second;

		if (!var_map.has(base_name)) {
			add_var(base_name, base_var.duplicate());
			changed = true;
			continue;
		}

		BBVariable var = var_map[base_name];
		if (!var.is_same_prop_info(base_var)) {
			var.copy_prop_info(base_var);
			changed = true;
		}
		if ((!var.is_value_changed() && var.get_value() != base_var.get_value()) ||
				(var.get_value().get_type() != base_var.get_type())) {
			// Reset value according to base plan.
			var.set_value(base_var.get_value());
			var.reset_value_changed();
			changed = true;
		}
	}

	// Erase variables that do not exist in the base plan.
	for (const Pair<String, BBVariable> &p : var_list) {
		if (!base->has_var(p.first)) {
			remove_var(p.first);
			changed = true;
		}
	}

	if (changed) {
		notify_property_list_changed();
		emit_changed();
	}
}

Ref<Blackboard> BlackboardPlan::create_blackboard() {
	Ref<Blackboard> bb = memnew(Blackboard);
	for (const Pair<String, BBVariable> &p : var_list) {
		bb->add_var(p.first, p.second.duplicate());
	}
	return bb;
}

void BlackboardPlan::populate_blackboard(const Ref<Blackboard> &p_blackboard, bool overwrite) {
	for (const Pair<String, BBVariable> &p : var_list) {
		if (p_blackboard->has_var(p.first)) {
			if (overwrite) {
				p_blackboard->erase_var(p.first);
			} else {
				continue;
			}
		}
		p_blackboard->add_var(p.first, p.second.duplicate());
	}
}

BlackboardPlan::BlackboardPlan() {
}
