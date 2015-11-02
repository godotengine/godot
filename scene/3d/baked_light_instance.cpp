#include "baked_light_instance.h"
#include "scene/scene_string_names.h"


RID BakedLightInstance::get_baked_light_instance() const {

	if (baked_light.is_null())
		return RID();
	else
		return get_instance();

}

void BakedLightInstance::set_baked_light(const Ref<BakedLight>& p_baked_light) {

	baked_light=p_baked_light;	

	RID base_rid;

	if (baked_light.is_valid())
		base_rid=baked_light->get_rid();
	else
		base_rid=RID();

	set_base(base_rid);

	if (is_inside_world()) {

		emit_signal(SceneStringNames::get_singleton()->baked_light_changed);

//		for (List<Node*>::Element *E=baked_geometry.front();E;E=E->next()) {
//			VS::get_singleton()->instance_geometry_set_baked_light(E->get()->get_instance(),baked_light.is_valid()?get_instance():RID());
//		}
	}
}

Ref<BakedLight> BakedLightInstance::get_baked_light() const{

	return baked_light;
}

AABB BakedLightInstance::get_aabb() const {

	return AABB(Vector3(0,0,0),Vector3(1,1,1));
}
DVector<Face3> BakedLightInstance::get_faces(uint32_t p_usage_flags) const {

	return DVector<Face3>();
}


void BakedLightInstance::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_baked_light","baked_light"),&BakedLightInstance::set_baked_light);
	ObjectTypeDB::bind_method(_MD("get_baked_light"),&BakedLightInstance::get_baked_light);
	ObjectTypeDB::bind_method(_MD("get_baked_light_instance"),&BakedLightInstance::get_baked_light_instance);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"baked_light",PROPERTY_HINT_RESOURCE_TYPE,"BakedLight"),_SCS("set_baked_light"),_SCS("get_baked_light"));
	ADD_SIGNAL( MethodInfo("baked_light_changed"));
}

BakedLightInstance::BakedLightInstance() {


}
/////////////////////////


void BakedLightSampler::set_param(Param p_param,float p_value) {
	ERR_FAIL_INDEX(p_param,PARAM_MAX);
	params[p_param]=p_value;
	VS::get_singleton()->baked_light_sampler_set_param(base,VS::BakedLightSamplerParam(p_param),p_value);
}

float BakedLightSampler::get_param(Param p_param) const{

	ERR_FAIL_INDEX_V(p_param,PARAM_MAX,0);
	return params[p_param];

}

void BakedLightSampler::set_resolution(int p_resolution){

    ERR_FAIL_COND(p_resolution<4 || p_resolution>32);
	resolution=p_resolution;
	VS::get_singleton()->baked_light_sampler_set_resolution(base,resolution);
}
int BakedLightSampler::get_resolution() const {

	return resolution;
}

AABB BakedLightSampler::get_aabb() const {

	float r = get_param(PARAM_RADIUS);
	return AABB( Vector3(-r,-r,-r),Vector3(r*2,r*2,r*2));
}
DVector<Face3> BakedLightSampler::get_faces(uint32_t p_usage_flags) const {
	return DVector<Face3>();
}

void BakedLightSampler::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_param","param","value"),&BakedLightSampler::set_param);
	ObjectTypeDB::bind_method(_MD("get_param","param"),&BakedLightSampler::get_param);

	ObjectTypeDB::bind_method(_MD("set_resolution","resolution"),&BakedLightSampler::set_resolution);
	ObjectTypeDB::bind_method(_MD("get_resolution"),&BakedLightSampler::get_resolution);


	BIND_CONSTANT( PARAM_RADIUS );
	BIND_CONSTANT( PARAM_STRENGTH );
	BIND_CONSTANT( PARAM_ATTENUATION );
	BIND_CONSTANT( PARAM_DETAIL_RATIO );
	BIND_CONSTANT( PARAM_MAX );

	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"params/radius",PROPERTY_HINT_RANGE,"0.01,1024,0.01"),_SCS("set_param"),_SCS("get_param"),PARAM_RADIUS);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"params/strength",PROPERTY_HINT_RANGE,"0.01,16,0.01"),_SCS("set_param"),_SCS("get_param"),PARAM_STRENGTH);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"params/attenuation",PROPERTY_HINT_EXP_EASING),_SCS("set_param"),_SCS("get_param"),PARAM_ATTENUATION);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"params/detail_ratio",PROPERTY_HINT_RANGE,"0.01,1.0,0.01"),_SCS("set_param"),_SCS("get_param"),PARAM_DETAIL_RATIO);
//	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"params/detail_ratio",PROPERTY_HINT_RANGE,"0,20,1"),_SCS("set_param"),_SCS("get_param"),PARAM_DETAIL_RATIO);
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"params/resolution",PROPERTY_HINT_RANGE,"4,32,1"),_SCS("set_resolution"),_SCS("get_resolution"));

}

BakedLightSampler::BakedLightSampler() {

	base = VS::get_singleton()->baked_light_sampler_create();
	set_base(base);

	params[PARAM_RADIUS]=1.0;
	params[PARAM_STRENGTH]=1.0;
	params[PARAM_ATTENUATION]=1.0;
	params[PARAM_DETAIL_RATIO]=0.1;
	resolution=16;


}

BakedLightSampler::~BakedLightSampler(){

	VS::get_singleton()->free(base);
}
