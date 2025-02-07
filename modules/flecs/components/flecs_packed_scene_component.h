#ifndef FLECS_PACKED_SCENE_COMPONENT_H
#define FLECS_PACKED_SCENE_COMPONENT_H




#include "../flecs_component.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "scene/resources/packed_scene.h"

namespace components {

struct FlecsPackedScene {
	Ref<PackedScene> value{nullptr};
};



} // namespace components


class FlecsPackedSceneComponent : public FlecsComponent {
	GDCLASS(FlecsPackedSceneComponent, FlecsComponent);




protected:
	static void _bind_methods();

public:

	Ref<PackedScene> get_scene() const;
	void set_scene(Ref<PackedScene> p_scene);


	virtual void add_component(flecs::entity p_entity) override;
	virtual void remove_component(flecs::entity p_entity) override;

	virtual bool has_component(flecs::entity p_entity) const override;

};

#endif // FLECS_PACKED_SCENE_COMPONENT_H