#include "flecs_prefab.h"
#include "flecs_entity.h"
#include "flecs_entity_node.h"

void FlecsPrefab::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_parent", "parent"), &FlecsPrefab::set_parent);
	ClassDB::bind_method(D_METHOD("get_parent"), &FlecsPrefab::get_parent);
	ClassDB::bind_method(D_METHOD("set_prefab_name", "prefab_name"), &FlecsPrefab::set_prefab_name);
	ClassDB::bind_method(D_METHOD("get_prefab_name"), &FlecsPrefab::get_prefab_name);
	ClassDB::bind_method(D_METHOD("set_modules", "modules"), &FlecsPrefab::set_modules);
	ClassDB::bind_method(D_METHOD("get_modules"), &FlecsPrefab::get_modules);
	ClassDB::bind_method(D_METHOD("instantiate"), &FlecsPrefab::instantiate);
	ClassDB::bind_method(D_METHOD("instantiate_num", "num"), &FlecsPrefab::instantiate_num);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "parent", PROPERTY_HINT_RESOURCE_TYPE, "FlecsPrefab"), "set_parent", "get_parent");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "prefab_name"), "set_prefab_name", "get_prefab_name");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "modules", PROPERTY_HINT_TYPE_STRING, String::num(Variant::OBJECT) + "/" + String::num(PROPERTY_HINT_RESOURCE_TYPE) + ":FlecsMod"), "set_modules", "get_modules");
}

void FlecsPrefab::set_parent(Ref<FlecsPrefab> p_parent) {
	parent = p_parent;
}

Ref<FlecsPrefab> FlecsPrefab::get_parent() const {
	return parent;
}

void FlecsPrefab::set_prefab_name(String p_name) {
	prefab_name = p_name;
}

String FlecsPrefab::get_prefab_name() const {
	return prefab_name;
}

void FlecsPrefab::set_modules(TypedArray<FlecsMod> p_modules) {
	modules = p_modules;
}

TypedArray<FlecsMod> FlecsPrefab::get_modules() const {
	return modules;
}

flecs::entity FlecsPrefab::instantiate_with_parent() const {
	flecs::world world = FlecsWorld::get_singleton()->get_world();

	if (parent.is_valid()) {
		// First, ensure the entire parent chain is initialized
		flecs::entity top_parent = parent->instantiate_with_parent();

		// Initialize current prefab as a child of its parent
		flecs::entity current_prefab = world.lookup(prefab_name.utf8().get_data());
		if (!current_prefab) {
			current_prefab = world.prefab(prefab_name.utf8().get_data());

			// Set parent relationship
			if (top_parent) {
				current_prefab.is_a(top_parent);
			}

			// Initialize current prefab's modules
			for (int j = 0; j < modules.size(); j++) {
				Ref<FlecsMod> module = modules[j];
				module->initialize(current_prefab, world);
			}
		}

		return current_prefab;
	}

	// If no parent, initialize current prefab
	flecs::entity current_prefab = world.lookup(prefab_name.utf8().get_data());
	if (!current_prefab) {
		current_prefab = world.prefab(prefab_name.utf8().get_data());

		// Initialize current prefab's modules
		for (int j = 0; j < modules.size(); j++) {
			Ref<FlecsMod> module = modules[j];
			module->initialize(current_prefab, world);
		}
	}

	return current_prefab;
}

Ref<FlecsEntity> FlecsPrefab::instantiate() const {
	flecs::world world = FlecsWorld::get_singleton()->get_world();

	// Check if prefab already exists
	flecs::entity prefab = instantiate_with_parent();

	Ref<FlecsEntity> entity = memnew(FlecsEntity);
	entity->set_entity(world.entity().is_a(prefab));

	return entity;
}

TypedArray<FlecsEntity> FlecsPrefab::instantiate_num(int p_num) const {
	TypedArray<FlecsEntity> entities;
	flecs::world world = FlecsWorld::get_singleton()->get_world();

	// Create the prefab once
	flecs::entity prefab = instantiate_with_parent();

	ecs_bulk_desc_t desc = {};
	desc.count = p_num;
	desc.ids[0] = { ecs_pair(EcsIsA, prefab) };

	// Bulk create entities
	const ecs_entity_t *ids = ecs_bulk_init(world, &desc);

	// Create FlecsEntity wrappers
	for (int i = 0; i < p_num; i++) {
		Ref<FlecsEntity> entity = memnew(FlecsEntity);
		entity->set_entity(world.entity(ids[i]));
		entities.append(entity);
	}

	return entities;
}

void FlecsPrefab::initialize_entity_data(FlecsEntityNode *entity) const {

	if(parent.is_valid()) {
		parent->initialize_entity_data(entity);
	}

	flecs::world world = FlecsWorld::get_singleton()->get_world();
	
	for (int i = 0; i < modules.size(); i++) 

	{
		Ref<FlecsMod> module = modules[i];
		module->initialize_entity_data(entity, world);
	}
}
