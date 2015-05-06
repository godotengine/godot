#ifndef BAKED_LIGHT_INSTANCE_H
#define BAKED_LIGHT_INSTANCE_H

#include "scene/3d/visual_instance.h"
#include "scene/resources/baked_light.h"

class BakedLightBaker;


class BakedLightInstance : public VisualInstance {
	OBJ_TYPE(BakedLightInstance,VisualInstance);

	Ref<BakedLight> baked_light;


protected:

	static void _bind_methods();
public:


	RID get_baked_light_instance() const;

	void set_baked_light(const Ref<BakedLight>& baked_light);
	Ref<BakedLight> get_baked_light() const;

	virtual AABB get_aabb() const;
	virtual DVector<Face3> get_faces(uint32_t p_usage_flags) const;

	BakedLightInstance();
};

#endif // BAKED_LIGHT_H
