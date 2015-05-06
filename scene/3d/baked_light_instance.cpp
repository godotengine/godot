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
