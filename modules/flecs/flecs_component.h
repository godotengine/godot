#ifndef FLECS_COMPONENT_H
#define FLECS_COMPONENT_H

#include "core/object/class_db.h"
#include "core/object/object.h"
#include "flecs_world.h"

class FlecsComponent : public Object {
	GDCLASS(FlecsComponent, Object);

protected:
	flecs::entity entity;
	static void _bind_methods();

public:
	virtual void add_component(flecs::entity p_entity) = 0;
	virtual void remove_component(flecs::entity p_entity) = 0;
	virtual bool has_component(flecs::entity p_entity) const = 0;
	virtual void initialize_component(flecs::entity p_entity) { entity = p_entity; }

	// Add this enum declaration with proper type info
	enum ComponentNames {
	};
	static void generate_component_enum();
	static StringName class_name(ComponentNames component);

private:
	static HashMap<ComponentNames, StringName> component_mapping;
};

VARIANT_ENUM_CAST(FlecsComponent::ComponentNames);

#endif // FLECS_COMPONENT_H