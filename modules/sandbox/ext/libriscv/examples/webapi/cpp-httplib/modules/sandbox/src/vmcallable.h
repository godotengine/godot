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
