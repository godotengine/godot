/**************************************************************************/
/*  range.h                                                               */
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

#ifndef RANGE_H
#define RANGE_H

#include "scene/gui/control.h"

// clang-format off
#define RANGE_CLASS_MAP \
	std::is_same_v<T, double> ? "Range" : \
	std::is_same_v<T, int64_t> ? "RangeInt" : \
	"RangeInvalid"
// clang-format on

template <typename T>
class RangeTemplate : public Control {
	GDCLASS_TEMPLATE(RangeTemplate<T>, RANGE_CLASS_MAP, Control, "Control", RangeTemplate)

	struct Shared {
		T val = 0.0;
		T min = 0.0;
		T max = 100.0;
		T step = 1.0;
		T page = 0.0;
		bool exp_ratio = false;
		bool allow_greater = false;
		bool allow_lesser = false;
		HashSet<RangeTemplate<T> *> owners;
		void emit_value_changed();
		void emit_changed(const char *p_what = "");
		void redraw_owners();
	};

	Shared *shared = nullptr;

	void _ref_shared(Shared *p_shared);
	void _unref_shared();

	void _share(Node *p_range);

	void _value_changed_notify();
	void _changed_notify(const char *p_what = "");
	void _set_value_no_signal(T p_val);
	bool _set_page_no_value(T p_page);

protected:
	virtual void _value_changed(T p_value);
	void _notify_shared_value_changed() { shared->emit_value_changed(); };

	static void _bind_methods();

	bool _rounded_values = false;

	GDVIRTUAL1(_value_changed, T);

public:
	void set_value(T p_val);
	void set_value_no_signal(T p_val);
	void set_min(T p_min);
	void set_max(T p_max);
	void set_step(T p_step);
	void set_page(T p_page);
	void set_as_ratio(double p_value);

	T get_value() const;
	T get_min() const;
	T get_max() const;
	T get_step() const;
	T get_page() const;
	double get_as_ratio() const;

	void set_use_rounded_values(bool p_enable);
	bool is_using_rounded_values() const;

	void set_exp_ratio(bool p_enable);
	bool is_ratio_exp() const;

	void set_allow_greater(bool p_allow);
	bool is_greater_allowed() const;

	void set_allow_lesser(bool p_allow);
	bool is_lesser_allowed() const;

	void share(RangeTemplate<T> *p_range);
	void unshare();

	PackedStringArray get_configuration_warnings() const override;

	RangeTemplate<T>();
	~RangeTemplate<T>();
};

using Range = RangeTemplate<double>;
using RangeInt = RangeTemplate<int64_t>;

#endif // RANGE_H
