#pragma once
#include <godot_cpp/variant/variant.hpp>
using namespace godot;

class Sandbox;

// This class is used to represent a property in the guest.
class SandboxProperty {
	StringName m_name;
	Variant::Type m_type = Variant::Type::NIL;
	uint64_t m_setter_address = 0;
	uint64_t m_getter_address = 0;
	Variant m_def_val;

public:
	SandboxProperty(const StringName &name, Variant::Type type, uint64_t setter, uint64_t getter, const Variant &def = "") :
			m_name(name), m_type(type), m_setter_address(setter), m_getter_address(getter), m_def_val(def) {}

	// Get the name of the property.
	const StringName &name() const { return m_name; }

	// Get the type of the property.
	Variant::Type type() const { return m_type; }

	// Get the address of the setter function.
	uint64_t setter_address() const { return m_setter_address; }
	// Get the address of the getter function.
	uint64_t getter_address() const { return m_getter_address; }

	// Get the default value of the property.
	const Variant &default_value() const { return m_def_val; }

	// Call the setter function.
	void set(Sandbox &sandbox, const Variant &value);
	// Call the getter function.
	Variant get(const Sandbox &sandbox) const;
};
