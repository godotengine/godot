/*************************************************************************/
/*  range.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "range.h"

String Range::get_configuration_warning() const {
	String warning = Control::get_configuration_warning();

	if (shared->exp_ratio && shared->min <= 0) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("If \"Exp Edit\" is enabled, \"Min Value\" must be greater than 0.");
	}

	return warning;
}

void Range::_value_changed_notify() {
	_value_changed(shared->val);
	emit_signal("value_changed", shared->val);
	update();
	_change_notify("value");
}

void Range::Shared::emit_value_changed() {
	for (Set<Range *>::Element *E = owners.front(); E; E = E->next()) {
		Range *r = E->get();
		if (!r->is_inside_tree()) {
			continue;
		}
		r->_value_changed_notify();
	}
}

void Range::_changed_notify(const char *p_what) {
	emit_signal("changed");
	update();
	_change_notify(p_what);
}

void Range::Shared::emit_changed(const char *p_what) {
	for (Set<Range *>::Element *E = owners.front(); E; E = E->next()) {
		Range *r = E->get();
		if (!r->is_inside_tree()) {
			continue;
		}
		r->_changed_notify(p_what);
	}
}

void Range::set_value(double p_val) {
	if (shared->step > 0) {
		p_val = Math::round(p_val / shared->step) * shared->step;
	}

	if (_rounded_values) {
		p_val = Math::round(p_val);
	}

	if (!shared->allow_greater && p_val > shared->max - shared->page) {
		p_val = shared->max - shared->page;
	}

	if (!shared->allow_lesser && p_val < shared->min) {
		p_val = shared->min;
	}

	if (shared->val == p_val) {
		return;
	}

	shared->val = p_val;

	shared->emit_value_changed();
}
void Range::set_min(double p_min) {
	shared->min = p_min;
	set_value(shared->val);

	shared->emit_changed("min");

	update_configuration_warning();
}
void Range::set_max(double p_max) {
	shared->max = p_max;
	set_value(shared->val);

	shared->emit_changed("max");
}
void Range::set_step(double p_step) {
	shared->step = p_step;
	shared->emit_changed("step");
}
void Range::set_page(double p_page) {
	shared->page = p_page;
	set_value(shared->val);

	shared->emit_changed("page");
}

double Range::get_value() const {
	return shared->val;
}
double Range::get_min() const {
	return shared->min;
}
double Range::get_max() const {
	return shared->max;
}
double Range::get_step() const {
	return shared->step;
}
double Range::get_page() const {
	return shared->page;
}

void Range::set_as_ratio(double p_value) {
	double v;

	if (shared->exp_ratio && get_min() >= 0) {
		double exp_min = get_min() == 0 ? 0.0 : Math::log(get_min()) / Math::log((double)2);
		double exp_max = Math::log(get_max()) / Math::log((double)2);
		v = Math::pow(2, exp_min + (exp_max - exp_min) * p_value);
	} else {
		double percent = (get_max() - get_min()) * p_value;
		if (get_step() > 0) {
			double steps = round(percent / get_step());
			v = steps * get_step() + get_min();
		} else {
			v = percent + get_min();
		}
	}
	v = CLAMP(v, get_min(), get_max());
	set_value(v);
}
double Range::get_as_ratio() const {
	if (Math::is_equal_approx(get_max(), get_min())) {
		// Avoid division by zero.
		return 1.0;
	}

	if (shared->exp_ratio && get_min() >= 0) {
		double exp_min = get_min() == 0 ? 0.0 : Math::log(get_min()) / Math::log((double)2);
		double exp_max = Math::log(get_max()) / Math::log((double)2);
		float value = CLAMP(get_value(), shared->min, shared->max);
		double v = Math::log(value) / Math::log((double)2);

		return CLAMP((v - exp_min) / (exp_max - exp_min), 0, 1);

	} else {
		float value = CLAMP(get_value(), shared->min, shared->max);
		return CLAMP((value - get_min()) / (get_max() - get_min()), 0, 1);
	}
}

void Range::_share(Node *p_range) {
	Range *r = Object::cast_to<Range>(p_range);
	ERR_FAIL_COND(!r);
	share(r);
}

void Range::share(Range *p_range) {
	ERR_FAIL_NULL(p_range);

	p_range->_ref_shared(shared);
	p_range->_changed_notify();
	p_range->_value_changed_notify();
}

void Range::unshare() {
	Shared *nshared = memnew(Shared);
	nshared->min = shared->min;
	nshared->max = shared->max;
	nshared->val = shared->val;
	nshared->step = shared->step;
	nshared->page = shared->page;
	nshared->exp_ratio = shared->exp_ratio;
	nshared->allow_greater = shared->allow_greater;
	nshared->allow_lesser = shared->allow_lesser;
	_unref_shared();
	_ref_shared(nshared);
}

void Range::_ref_shared(Shared *p_shared) {
	if (shared && p_shared == shared) {
		return;
	}

	_unref_shared();
	shared = p_shared;
	shared->owners.insert(this);
}

void Range::_unref_shared() {
	if (shared) {
		shared->owners.erase(this);
		if (shared->owners.size() == 0) {
			memdelete(shared);
			shared = nullptr;
		}
	}
}

void Range::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_value"), &Range::get_value);
	ClassDB::bind_method(D_METHOD("get_min"), &Range::get_min);
	ClassDB::bind_method(D_METHOD("get_max"), &Range::get_max);
	ClassDB::bind_method(D_METHOD("get_step"), &Range::get_step);
	ClassDB::bind_method(D_METHOD("get_page"), &Range::get_page);
	ClassDB::bind_method(D_METHOD("get_as_ratio"), &Range::get_as_ratio);
	ClassDB::bind_method(D_METHOD("set_value", "value"), &Range::set_value);
	ClassDB::bind_method(D_METHOD("set_min", "minimum"), &Range::set_min);
	ClassDB::bind_method(D_METHOD("set_max", "maximum"), &Range::set_max);
	ClassDB::bind_method(D_METHOD("set_step", "step"), &Range::set_step);
	ClassDB::bind_method(D_METHOD("set_page", "pagesize"), &Range::set_page);
	ClassDB::bind_method(D_METHOD("set_as_ratio", "value"), &Range::set_as_ratio);
	ClassDB::bind_method(D_METHOD("set_use_rounded_values", "enabled"), &Range::set_use_rounded_values);
	ClassDB::bind_method(D_METHOD("is_using_rounded_values"), &Range::is_using_rounded_values);
	ClassDB::bind_method(D_METHOD("set_exp_ratio", "enabled"), &Range::set_exp_ratio);
	ClassDB::bind_method(D_METHOD("is_ratio_exp"), &Range::is_ratio_exp);
	ClassDB::bind_method(D_METHOD("set_allow_greater", "allow"), &Range::set_allow_greater);
	ClassDB::bind_method(D_METHOD("is_greater_allowed"), &Range::is_greater_allowed);
	ClassDB::bind_method(D_METHOD("set_allow_lesser", "allow"), &Range::set_allow_lesser);
	ClassDB::bind_method(D_METHOD("is_lesser_allowed"), &Range::is_lesser_allowed);

	ClassDB::bind_method(D_METHOD("share", "with"), &Range::_share);
	ClassDB::bind_method(D_METHOD("unshare"), &Range::unshare);

	ADD_SIGNAL(MethodInfo("value_changed", PropertyInfo(Variant::REAL, "value")));
	ADD_SIGNAL(MethodInfo("changed"));

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "min_value"), "set_min", "get_min");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_value"), "set_max", "get_max");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "step"), "set_step", "get_step");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "page"), "set_page", "get_page");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "value"), "set_value", "get_value");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ratio", PROPERTY_HINT_RANGE, "0,1,0.01", 0), "set_as_ratio", "get_as_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "exp_edit"), "set_exp_ratio", "is_ratio_exp");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rounded"), "set_use_rounded_values", "is_using_rounded_values");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_greater"), "set_allow_greater", "is_greater_allowed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_lesser"), "set_allow_lesser", "is_lesser_allowed");
}

void Range::set_use_rounded_values(bool p_enable) {
	_rounded_values = p_enable;
}

bool Range::is_using_rounded_values() const {
	return _rounded_values;
}

void Range::set_exp_ratio(bool p_enable) {
	shared->exp_ratio = p_enable;

	update_configuration_warning();
}

bool Range::is_ratio_exp() const {
	return shared->exp_ratio;
}

void Range::set_allow_greater(bool p_allow) {
	shared->allow_greater = p_allow;
}

bool Range::is_greater_allowed() const {
	return shared->allow_greater;
}

void Range::set_allow_lesser(bool p_allow) {
	shared->allow_lesser = p_allow;
}

bool Range::is_lesser_allowed() const {
	return shared->allow_lesser;
}

Range::Range() {
	shared = memnew(Shared);
	shared->min = 0;
	shared->max = 100;
	shared->val = 0;
	shared->step = 1;
	shared->page = 0;
	shared->owners.insert(this);
	shared->exp_ratio = false;
	shared->allow_greater = false;
	shared->allow_lesser = false;

	_rounded_values = false;
}

Range::~Range() {
	_unref_shared();
}
