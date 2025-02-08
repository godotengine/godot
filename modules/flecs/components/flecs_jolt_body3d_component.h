#ifndef FLECS_JOLT_BODY3D_COMPONENT_H
#define FLECS_JOLT_BODY3D_COMPONENT_H

#include "../flecs_component.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "modules/jolt_physics/objects/jolt_body_3d.h"

namespace components {

struct FlecsJoltBody3D {
	JoltBody3D * value{ nullptr };
};

struct JoltBody3DToFlecsSyncTag{};
struct FlecsToJoltBody3DSyncTag{};


}

class FlecsJoltBody3DComponent : public FlecsComponent {
	GDCLASS(FlecsJoltBody3DComponent, FlecsComponent);

protected:
	static void _bind_methods();

public:

	virtual void add_component(flecs::entity p_entity) override;
	virtual void remove_component(flecs::entity p_entity) override;
	virtual bool has_component(flecs::entity p_entity) const override;

	void set_jolt_body(JoltBody3D *p_jolt_body);
	JoltBody3D * get_jolt_body() const;

};

#endif // FLECS_JOLT_BODY3D_COMPONENT_H