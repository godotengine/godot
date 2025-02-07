#ifndef FLECS_PREFAB_H
#define FLECS_PREFAB_H

#include "core/io/resource.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "flecs_world.h"

class FlecsEntity;
class FlecsMod;
class FlecsEntityNode;

class FlecsPrefab : public Resource {
	GDCLASS(FlecsPrefab, Resource);

public:
	Ref<FlecsEntity> instantiate() const;
	TypedArray<FlecsEntity> instantiate_num(int p_num) const;

	void post_instantiate(Ref<FlecsEntity> entity) const;
	void initialize_entity_data(FlecsEntityNode *entity) const;

protected:
	static void _bind_methods();

	void set_parent(Ref<FlecsPrefab> p_parent);
	Ref<FlecsPrefab> get_parent() const;

	void set_prefab_name(String p_name);
	String get_prefab_name() const;

	void set_modules(TypedArray<FlecsMod> p_modules);
	TypedArray<FlecsMod> get_modules() const;

	flecs::entity instantiate_with_parent() const;

private:

	void _resolve_module_dependencies(TypedArray<FlecsMod> p_modules);

private:
	String prefab_name;

	Ref<FlecsPrefab> parent;
	TypedArray<FlecsMod> modules;
};

#endif // FLECS_ENTITY_H