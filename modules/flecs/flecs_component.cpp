#include "flecs_component.h"

HashMap<FlecsComponent::ComponentNames, StringName> FlecsComponent::component_mapping;

void FlecsComponent::_bind_methods() {
	ClassDB::bind_static_method("FlecsComponent", D_METHOD("class_name", "component"), &FlecsComponent::class_name);
}

void FlecsComponent::generate_component_enum() {
	List<StringName> classes;
	ClassDB::get_class_list(&classes);

	int enum_value = 0; // Start after NONE
	for (const StringName &class_name : classes) {
		if (ClassDB::is_parent_class(class_name, "FlecsComponent")) {
			// Convert class name to more readable format
			String base_name = String(class_name).replace("Component", "");
			String enum_name = base_name;

			ClassDB::bind_integer_constant("FlecsComponent", "ComponentNames", enum_name, enum_value);
			component_mapping[static_cast<ComponentNames>(enum_value)] = class_name;
			enum_value++;
		}
	}
}

StringName FlecsComponent::class_name(ComponentNames component) {
	if (component_mapping.has(component)) {
		return component_mapping[component];
	}
	return StringName(); // Return empty StringName if not found
}
