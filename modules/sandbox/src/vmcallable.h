/**************************************************************************/
/*  vmcallable.h                                                          */
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
#include <godot_cpp/variant/variant.hpp>
using namespace godot;

class Sandbox;
struct GuestVariant;

class RiscvCallable : public CallableCustom {
public:
	uint32_t hash() const override {
		return address;
	}

	String get_as_text() const override {
		return "<RiscvCallable>";
	}

	CompareEqualFunc get_compare_equal_func() const override {
		return [](const CallableCustom *p_a, const CallableCustom *p_b) {
			return p_a == p_b;
		};
	}

	CompareLessFunc get_compare_less_func() const override {
		return [](const CallableCustom *p_a, const CallableCustom *p_b) {
			return p_a < p_b;
		};
	}

	bool is_valid() const override {
		return self != nullptr;
	}

	ObjectID get_object() const override {
		return ObjectID();
	}

	void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, GDExtensionCallError &r_call_error) const override;

	void init(Sandbox *self, gaddr_t address, Array args) {
		this->self = self;
		this->address = address;

		for (int i = 0; i < args.size(); i++) {
			m_varargs[i] = args[i];
			m_varargs_ptrs[i] = &m_varargs[i];
		}
		this->m_varargs_base_count = args.size();
	}

private:
	Sandbox *self = nullptr;
	gaddr_t address = 0x0;

	std::array<Variant, 8> m_varargs;
	mutable std::array<const Variant *, 8> m_varargs_ptrs;
	int m_varargs_base_count = 0;
};
