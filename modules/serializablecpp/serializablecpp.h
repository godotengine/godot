#ifndef SERIALIZABLECPP_H
#define SERIALIZABLECPP_H

// #ifdef _MSC_VER
// #define _CRT_SECURE_NO_WARNINGS 1
// #pragma warning(disable:4996)
// #pragma warning(disable:4828)
// #endif
#include "core/object.h"
#include "core/resource.h"
#include "core/array.h"
#include "core/list.h"
#include "core/variant.h"
#include "core/ustring.h"
#include "modules/hub/hub.h"
// #include "core/core_string_names.h"



class SerializableCPP : public Resource {
	GDCLASS(SerializableCPP, Resource);
private:
	// Read-only value
	const char* CURRENT_SERIALIZABLE_VER = "0.0.1";
	String serializable_ver;

	// Accesible value
	String module_name;
	Vector<String> property_list;

	// Non-formal-value
	Variant debug_value1;

	// Helper functions

protected:
    static void _bind_methods();

public:
	SerializableCPP();
	~SerializableCPP() = default;

	String get_serializable_version();
	void add_properties(Vector<String> plist);
	bool remove_properties(Vector<String> plist);
	bool copy(const Ref<SerializableCPP>& from);
	Ref<SerializableCPP> serializable_dup(bool dup_property);

	// Setters, getters
	void set_forbidden(Variant value);
	
	void set_module_name(String new_name);
	String get_module_name();

	void set_property_list(Vector<String> plist);
	Vector<String> get_props_list();

	void set_dv1(Variant value);
	Variant get_dv1();
};

#endif