#ifndef FLECS_PACKED_SCENE_MODULE_H
#define FLECS_PACKED_SCENE_MODULE_H

#include "../flecs_mod.h"
#include "../flecs_world.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "../components/flecs_packed_scene_component.h"
#include "../flecs_entity.h"


namespace modules {

struct FlecsPackedSceneModule {
	FlecsPackedSceneModule(flecs::world &world) {

		world.module<FlecsPackedSceneModule>();
		world.component<components::FlecsPackedScene>();
	}
};

} //namespace modules

class FlecsPackedSceneMod : public FlecsMod {
	GDCLASS(FlecsPackedSceneMod, FlecsMod);


protected:
	static void _bind_methods();

public:

	TypedArray<FlecsMod> get_required_modules() const override;

	virtual void initialize(flecs::entity &prefab, flecs::world &world) override;
	virtual void post_instantiate(Ref<FlecsEntity> entity, flecs::world &world) override;
	virtual void initialize_entity_data(FlecsEntityNode *entity, flecs::world &world) override;

	Ref<PackedScene> get_scene() const;
	void set_scene(Ref<PackedScene> p_scene);

	bool is_auto_initialize() const;
	void set_auto_initialize(bool p_auto_initialize);

	bool is_add_to_root() const;
	void set_add_to_root(bool p_add_to_root);

public:

	Ref<PackedScene> packed_scene = nullptr;
	bool auto_initialize = true;
	bool add_to_root = true;


};

#endif