#include "flecs_entity.h"
#include "flecs_component.h"
#include "flecs_world.h"

void FlecsEntity::_bind_methods() {
	ClassDB::bind_method(D_METHOD("destroy"), &FlecsEntity::destroy);
	ClassDB::bind_method(D_METHOD("add_component", "component"), &FlecsEntity::add_component);
	ClassDB::bind_method(D_METHOD("remove_component", "component"), &FlecsEntity::remove_component);
	ClassDB::bind_method(D_METHOD("has_component", "component"), &FlecsEntity::has_component);
	ClassDB::bind_method(D_METHOD("get_component", "component"), &FlecsEntity::get_component);
	ClassDB::bind_method(D_METHOD("is_alive"), &FlecsEntity::is_alive);
}

void FlecsEntity::destroy() {
	if (entity.is_alive()) {
		entity.destruct();
	}
}

Variant FlecsEntity::add_component(const StringName &component_name) {
	const StringName class_name = component_name;
	if (ClassDB::class_exists(class_name)) {
		FlecsComponent *component = Object::cast_to<FlecsComponent>(ClassDB::instantiate(class_name));
		if (component) {
			component->add_component(entity);
			return component;
		}
	}
	return Variant();
}

void FlecsEntity::remove_component(const StringName &component_name) {
	const StringName class_name = component_name;
	if (ClassDB::class_exists(class_name)) {
		FlecsComponent *component = Object::cast_to<FlecsComponent>(ClassDB::instantiate(class_name));
		if (component) {
			component->remove_component(entity);
		}
	}
}

bool FlecsEntity::has_component(const StringName &component_name) const {
	const StringName class_name = component_name;
	if (ClassDB::class_exists(class_name)) {
		FlecsComponent *component = Object::cast_to<FlecsComponent>(ClassDB::instantiate(class_name));
		if (component) {
			return component->has_component(entity);
		}
	}
	return false;
}

Variant FlecsEntity::get_component(const StringName &component_name) const {
	const StringName class_name = component_name;
	if (ClassDB::class_exists(class_name)) {
		FlecsComponent *component = Object::cast_to<FlecsComponent>(ClassDB::instantiate(class_name));
		if (component) {
			if (component->has_component(entity)) {
				component->initialize_component(entity);
				return component;
			}
		}
	}

	return Variant();
}
