#include "flecs_prefab.h"
#include "flecs_entity.h"
#include "flecs_entity_node.h"
#include "flecs_mod.h"

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

void FlecsPrefab::_resolve_module_dependencies(TypedArray<FlecsMod> p_modules) {
    // Clear existing resolved modules
    TypedArray<FlecsMod> resolved_modules;
    
    // Create a set to track already added modules
    HashSet<StringName> added_modules;
    
    // Recursive function to add modules with dependencies
    auto add_module = [&](Ref<FlecsMod> module, auto&& self) -> void {
        StringName module_name = module->get_class_name();
        
        // Skip already added modules
        if (added_modules.has(module_name)) return;
        
        // Add dependencies first
        TypedArray<FlecsMod> deps = module->get_required_modules();
        for (int i = 0; i < deps.size(); i++) {
            Ref<FlecsMod> dep = deps[i];
			if (!dep.is_valid()) {
				ERR_FAIL_MSG(vformat("Invalid module dependency at index %d for module %s", i, module_name));
			}
            self(dep, self);
        }
        
        // Add the module itself
        resolved_modules.append(module);
        added_modules.insert(module_name);
    };
    
    // Process all input modules
    for (int i = 0; i < p_modules.size(); i++) {
        Ref<FlecsMod> module = p_modules[i];
		if (!module.is_valid()) {
			resolved_modules.append(Variant());
			continue;
		}
		add_module(module, add_module);
		
    }
    
    // Update the modules array with resolved dependencies
    modules = resolved_modules;
}

void FlecsPrefab::set_modules(TypedArray<FlecsMod> p_modules) {
	 _resolve_module_dependencies(p_modules);
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
	entity->set_prefab(Ref<FlecsPrefab>(this));

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
		entity->set_prefab(Ref<FlecsPrefab>(this));
		post_instantiate(entity);
		entities.append(entity);
	}

	return entities;
}

void FlecsPrefab::post_instantiate(Ref<FlecsEntity> entity) const {

	if(parent.is_valid()) {
		parent->post_instantiate(entity);
	}

	flecs::world world = FlecsWorld::get_singleton()->get_world();

	for (int i = 0; i < modules.size(); i++) {
		Ref<FlecsMod> module = modules[i];
		module->post_instantiate(entity, world);
	}
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
