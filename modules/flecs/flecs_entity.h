#ifndef FLECS_ENTITY_H
#define FLECS_ENTITY_H

#include "core/object/class_db.h"
#include "core/object/object.h"
#include "flecs_prefab.h"
#include "flecs_world.h"

class FlecsComponent;

class FlecsEntity : public RefCounted {
	GDCLASS(FlecsEntity, RefCounted);

private:
	Ref<FlecsPrefab> prefab;
	flecs::entity entity;

protected:
	static void _bind_methods();

public:

	void destroy();
	Variant add_component(const StringName &component_name);
	void remove_component(const StringName &component_name);
	bool has_component(const StringName &component_name) const;

	Variant get_component(const StringName &component_name) const;

	bool is_alive() const { return entity.is_alive(); }

	flecs::entity get_entity() const { return entity; }
	void set_entity(flecs::entity p_entity) { entity = p_entity; }

	void set_prefab(Ref<FlecsPrefab> p_prefab) { prefab = p_prefab; }
	Ref<FlecsPrefab> get_prefab() const { return prefab; }
};

#endif // FLECS_ENTITY_H