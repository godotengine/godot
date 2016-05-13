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



class BakedLightSampler : public VisualInstance {
	OBJ_TYPE(BakedLightSampler,VisualInstance);


public:

	enum Param {
		PARAM_RADIUS=VS::BAKED_LIGHT_SAMPLER_RADIUS,
		PARAM_STRENGTH=VS::BAKED_LIGHT_SAMPLER_STRENGTH,
		PARAM_ATTENUATION=VS::BAKED_LIGHT_SAMPLER_ATTENUATION,
		PARAM_DETAIL_RATIO=VS::BAKED_LIGHT_SAMPLER_DETAIL_RATIO,
		PARAM_MAX=VS::BAKED_LIGHT_SAMPLER_MAX
	};



protected:

	RID base;
	float params[PARAM_MAX];
	int resolution;
	static void _bind_methods();
public:

	virtual AABB get_aabb() const;
	virtual DVector<Face3> get_faces(uint32_t p_usage_flags) const;

	void set_param(Param p_param,float p_value);
	float get_param(Param p_param) const;

	void set_resolution(int p_resolution);
	int get_resolution() const;

	BakedLightSampler();
	~BakedLightSampler();
};

VARIANT_ENUM_CAST( BakedLightSampler::Param );


#endif // BAKED_LIGHT_H
