/**************************************************************************/
/*  range.cpp                                                             */
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

#include "range.h"

template <typename T>
PackedStringArray RangeTemplate<T>::get_configuration_warnings() const {
	PackedStringArray warnings = Node::get_configuration_warnings();

	if (shared->exp_ratio && shared->min <= 0) {
		warnings.push_back(RTR("If \"Exp Edit\" is enabled, \"Min Value\" must be greater than 0."));
	}

	return warnings;
}

template <typename T>
void RangeTemplate<T>::_value_changed(T p_value) {
	GDVIRTUAL_CALL(_value_changed, p_value);
}
template <typename T>
void RangeTemplate<T>::_value_changed_notify() {
	_value_changed(shared->val);
	emit_signal(SceneStringName(value_changed), shared->val);
	queue_redraw();
}

template <typename T>
void RangeTemplate<T>::Shared::emit_value_changed() {
	for (RangeTemplate<T> *E : owners) {
		RangeTemplate<T> *r = E;
		if (!r->is_inside_tree()) {
			continue;
		}
		r->_value_changed_notify();
	}
}

template <typename T>
void RangeTemplate<T>::_changed_notify(const char *p_what) {
	// TODO: Include what changed as an argument to the signal.
	emit_signal(CoreStringName(changed));
	queue_redraw();
}

template <typename T>
void RangeTemplate<T>::Shared::emit_changed(const char *p_what) {
	for (RangeTemplate<T> *E : owners) {
		RangeTemplate<T> *r = E;
		if (!r->is_inside_tree()) {
			continue;
		}
		r->_changed_notify(p_what);
	}
}

template <typename T>
void RangeTemplate<T>::Shared::redraw_owners() {
	for (RangeTemplate<T> *E : owners) {
		RangeTemplate<T> *r = E;
		if (!r->is_inside_tree()) {
			continue;
		}
		r->queue_redraw();
	}
}

template <typename T>
void RangeTemplate<T>::set_value(T p_val) {
	T prev_val = shared->val;
	_set_value_no_signal(p_val);

	if (shared->val != prev_val) {
		shared->emit_value_changed();
	}
}

template <>
void RangeTemplate<double>::_set_value_no_signal(double p_val) {
	if (!Math::is_finite(p_val)) {
		return;
	}

	if (shared->step > 0) {
		p_val = Math::round((p_val - shared->min) / shared->step) * shared->step + shared->min;
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
}

template <>
void RangeTemplate<int64_t>::_set_value_no_signal(int64_t p_val) {
	if (shared->step > 1) {
		int64_t to_lower = (p_val - shared->min) % shared->step;
		int64_t to_higher = shared->step - to_lower;
		if (to_lower <= to_higher) {
			p_val -= to_lower;
		} else {
			p_val += to_higher;
		}
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
}

template <typename T>
void RangeTemplate<T>::set_value_no_signal(T p_val) {
	T prev_val = shared->val;
	_set_value_no_signal(p_val);

	if (shared->val != prev_val) {
		shared->redraw_owners();
	}
}

template <typename T>
void RangeTemplate<T>::set_min(T p_min) {
	if (shared->min == p_min) {
		return;
	}

	shared->min = p_min;
	shared->max = MAX(shared->max, shared->min);
	_set_page_no_value(shared->page);
	set_value(shared->val);

	shared->emit_changed("min");

	update_configuration_warnings();
}

template <typename T>
void RangeTemplate<T>::set_max(T p_max) {
	T max_validated = MAX(p_max, shared->min);
	if (shared->max == max_validated) {
		return;
	}

	shared->max = max_validated;
	_set_page_no_value(shared->page);
	set_value(shared->val);

	shared->emit_changed("max");
}

template <typename T>
void RangeTemplate<T>::set_step(T p_step) {
	if (shared->step == p_step) {
		return;
	}

	shared->step = p_step;
	shared->emit_changed("step");
}

template <typename T>
bool RangeTemplate<T>::_set_page_no_value(T p_page) {
	T page_max;
	if (std::is_same_v<T, int64_t> && shared->min < 0 && shared->max > INT64_MAX + shared->min) {
		page_max = shared->max;
	} else {
		page_max = shared->max - shared->min;
	}
	T page_validated = CLAMP(p_page, 0, page_max);

	if (shared->page == page_validated) {
		return false;
	}
	shared->page = page_validated;
	shared->emit_changed("page");
	return true;
}

template <typename T>
void RangeTemplate<T>::set_page(T p_page) {
	if (_set_page_no_value(p_page)) {
		set_value(shared->val);
	}
}

template <typename T>
T RangeTemplate<T>::get_value() const {
	return shared->val;
}

template <typename T>
T RangeTemplate<T>::get_min() const {
	return shared->min;
}

template <typename T>
T RangeTemplate<T>::get_max() const {
	return shared->max;
}

template <typename T>
T RangeTemplate<T>::get_step() const {
	return shared->step;
}

template <typename T>
T RangeTemplate<T>::get_page() const {
	return shared->page;
}

template <typename T>
void RangeTemplate<T>::set_as_ratio(double p_value) {
	T v;

	// When T is int64_t, make sure precision loss from floating point calculations
	// doesn't prevent the min or max value from being set for a ratio of 0 and 1 respectively.
	if (p_value == 1) {
		v = get_max();
	} else if (shared->exp_ratio && get_min() >= 0) {
		// TODO: if get_min() < 0, should we CLAMP rather than fall back to liner?
		// Or should we return zero or NaN?
		double exp_min = get_min() == 0 ? 0.0 : Math::log((double)get_min()) / Math::log((double)2);
		double exp_max = Math::log((double)get_max()) / Math::log((double)2);
		v = Math::pow(2, exp_min + (exp_max - exp_min) * p_value);
	} else if (p_value == 0) {
		v = get_min();
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

template <typename T>
double RangeTemplate<T>::get_as_ratio() const {
	if constexpr (std::is_same_v<T, int64_t>) {
		if (get_value() == get_min()) {
			return 0.0;
		}
		if (get_value() == get_max()) {
			return 1.0;
		}
	} else if (Math::is_equal_approx((double)get_max(), (double)get_min())) {
		// Avoid division by zero.
		return 1.0;
	}

	if (shared->exp_ratio && get_min() >= 0) {
		double exp_min = get_min() == 0 ? 0.0 : Math::log((double)get_min()) / Math::log((double)2);
		double exp_max = Math::log((double)get_max()) / Math::log((double)2);
		float value = CLAMP((double)get_value(), shared->min, shared->max);
		double v = Math::log(value) / Math::log((double)2);

		return CLAMP((v - exp_min) / (exp_max - exp_min), 0, 1);
	} else {
		float value = CLAMP(get_value(), shared->min, shared->max);
		return CLAMP((value - get_min()) / (get_max() - get_min()), 0, 1);
	}
}

template <typename T>
void RangeTemplate<T>::_share(Node *p_range) {
	RangeTemplate<T> *r = Object::cast_to<RangeTemplate<T>>(p_range);
	ERR_FAIL_NULL(r);
	share(r);
}

template <typename T>
void RangeTemplate<T>::share(RangeTemplate<T> *p_range) {
	ERR_FAIL_NULL(p_range);

	p_range->_ref_shared(shared);
	p_range->_changed_notify();
	p_range->_value_changed_notify();
}

template <typename T>
void RangeTemplate<T>::unshare() {
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

template <typename T>
void RangeTemplate<T>::_ref_shared(Shared *p_shared) {
	if (shared && p_shared == shared) {
		return;
	}

	_unref_shared();
	shared = p_shared;
	shared->owners.insert(this);
}

template <typename T>
void RangeTemplate<T>::_unref_shared() {
	if (shared) {
		shared->owners.erase(this);
		if (shared->owners.size() == 0) {
			memdelete(shared);
			shared = nullptr;
		}
	}
}

template <typename T>
void RangeTemplate<T>::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_value"), &RangeTemplate<T>::get_value);
	ClassDB::bind_method(D_METHOD("get_min"), &RangeTemplate<T>::get_min);
	ClassDB::bind_method(D_METHOD("get_max"), &RangeTemplate<T>::get_max);
	ClassDB::bind_method(D_METHOD("get_step"), &RangeTemplate<T>::get_step);
	ClassDB::bind_method(D_METHOD("get_page"), &RangeTemplate<T>::get_page);
	ClassDB::bind_method(D_METHOD("get_as_ratio"), &RangeTemplate<T>::get_as_ratio);
	ClassDB::bind_method(D_METHOD("set_value", "value"), &RangeTemplate<T>::set_value);
	ClassDB::bind_method(D_METHOD("set_value_no_signal", "value"), &RangeTemplate<T>::set_value_no_signal);
	ClassDB::bind_method(D_METHOD("set_min", "minimum"), &RangeTemplate<T>::set_min);
	ClassDB::bind_method(D_METHOD("set_max", "maximum"), &RangeTemplate<T>::set_max);
	ClassDB::bind_method(D_METHOD("set_step", "step"), &RangeTemplate<T>::set_step);
	ClassDB::bind_method(D_METHOD("set_page", "pagesize"), &RangeTemplate<T>::set_page);
	ClassDB::bind_method(D_METHOD("set_as_ratio", "value"), &RangeTemplate<T>::set_as_ratio);
	if (std::is_same<T, double>::value) {
		ClassDB::bind_method(D_METHOD("set_use_rounded_values", "enabled"), &RangeTemplate<T>::set_use_rounded_values);
		ClassDB::bind_method(D_METHOD("is_using_rounded_values"), &RangeTemplate<T>::is_using_rounded_values);
	}
	ClassDB::bind_method(D_METHOD("set_exp_ratio", "enabled"), &RangeTemplate<T>::set_exp_ratio);
	ClassDB::bind_method(D_METHOD("is_ratio_exp"), &RangeTemplate<T>::is_ratio_exp);
	ClassDB::bind_method(D_METHOD("set_allow_greater", "allow"), &RangeTemplate<T>::set_allow_greater);
	ClassDB::bind_method(D_METHOD("is_greater_allowed"), &RangeTemplate<T>::is_greater_allowed);
	ClassDB::bind_method(D_METHOD("set_allow_lesser", "allow"), &RangeTemplate<T>::set_allow_lesser);
	ClassDB::bind_method(D_METHOD("is_lesser_allowed"), &RangeTemplate<T>::is_lesser_allowed);

	ClassDB::bind_method(D_METHOD("share", "with"), &RangeTemplate<T>::_share);
	ClassDB::bind_method(D_METHOD("unshare"), &RangeTemplate<T>::unshare);

	ADD_SIGNAL(MethodInfo("value_changed", PropertyInfo(Variant::FLOAT, "value")));
	ADD_SIGNAL(MethodInfo("changed"));

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_value"), "set_min", "get_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_value"), "set_max", "get_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "step"), "set_step", "get_step");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "page"), "set_page", "get_page");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "value"), "set_value", "get_value");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ratio", PROPERTY_HINT_RANGE, "0,1,0.01", PROPERTY_USAGE_NONE), "set_as_ratio", "get_as_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "exp_edit"), "set_exp_ratio", "is_ratio_exp");
	if (std::is_same<T, double>::value) {
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rounded"), "set_use_rounded_values", "is_using_rounded_values");
	}
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_greater"), "set_allow_greater", "is_greater_allowed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_lesser"), "set_allow_lesser", "is_lesser_allowed");

	GDVIRTUAL_BIND(_value_changed, "new_value");

	ADD_LINKED_PROPERTY("min_value", "value");
	ADD_LINKED_PROPERTY("min_value", "max_value");
	ADD_LINKED_PROPERTY("min_value", "page");
	ADD_LINKED_PROPERTY("max_value", "value");
	ADD_LINKED_PROPERTY("max_value", "page");
}

template <typename T>
void RangeTemplate<T>::set_use_rounded_values(bool p_enable) {
	_rounded_values = p_enable;
	// TODO: If set to true, should we reset the value?
	// TODO: Should we emit the "changed" signal?
}

template <typename T>
bool RangeTemplate<T>::is_using_rounded_values() const {
	return _rounded_values;
}

template <typename T>
void RangeTemplate<T>::set_exp_ratio(bool p_enable) {
	if (shared->exp_ratio == p_enable) {
		return;
	}

	shared->exp_ratio = p_enable;

	update_configuration_warnings();
}

template <typename T>
bool RangeTemplate<T>::is_ratio_exp() const {
	return shared->exp_ratio;
}

template <typename T>
void RangeTemplate<T>::set_allow_greater(bool p_allow) {
	shared->allow_greater = p_allow;
	// TODO: If set to false, should we reset the value?
	// TODO: Should we emit the "changed" signal?
}

template <typename T>
bool RangeTemplate<T>::is_greater_allowed() const {
	return shared->allow_greater;
}

template <typename T>
void RangeTemplate<T>::set_allow_lesser(bool p_allow) {
	shared->allow_lesser = p_allow;
	// TODO: If set to false, should we reset the value?
	// TODO: Should we emit the "changed" signal?
}

template <typename T>
bool RangeTemplate<T>::is_lesser_allowed() const {
	return shared->allow_lesser;
}

template <typename T>
RangeTemplate<T>::RangeTemplate() {
	shared = memnew(Shared);
	shared->owners.insert(this);
}

template <typename T>
RangeTemplate<T>::~RangeTemplate() {
	_unref_shared();
}

template class RangeTemplate<double>;
template class RangeTemplate<int64_t>;
