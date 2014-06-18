#include "baked_light.h"
#include "servers/visual_server.h"

void BakedLight::set_mode(Mode p_mode) {

	mode=p_mode;
	VS::get_singleton()->baked_light_set_mode(baked_light,(VS::BakedLightMode(p_mode)));
}

BakedLight::Mode BakedLight::get_mode() const{

	return mode;
}

void BakedLight::set_octree(const DVector<uint8_t>& p_octree) {

	VS::get_singleton()->baked_light_set_octree(baked_light,p_octree);
}

DVector<uint8_t> BakedLight::get_octree() const {

	return VS::get_singleton()->baked_light_get_octree(baked_light);
}


void BakedLight::_update_lightmaps() {

	VS::get_singleton()->baked_light_clear_lightmaps(baked_light);
	for(Map<int,Ref<Texture> >::Element *E=lightmaps.front();E;E=E->next()) {

		VS::get_singleton()->baked_light_add_lightmap(baked_light,E->get()->get_rid(),E->key());
	}

}

void BakedLight::add_lightmap(const Ref<Texture> p_texture,int p_id) {

	ERR_FAIL_COND(!p_texture.is_valid());
	ERR_FAIL_COND(p_id<0);
	lightmaps[p_id]=p_texture;
	VS::get_singleton()->baked_light_add_lightmap(baked_light,p_texture->get_rid(),p_id);
}

void BakedLight::erase_lightmap(int p_id) {

	ERR_FAIL_COND(!lightmaps.has(p_id));
	lightmaps.erase(p_id);
	_update_lightmaps();
}

void BakedLight::get_lightmaps(List<int> *r_lightmaps) {

	for(Map<int,Ref<Texture> >::Element *E=lightmaps.front();E;E=E->next()) {

		r_lightmaps->push_back(E->key());
	}

}

Ref<Texture> BakedLight::get_lightmap_texture(int p_id) {

	if (!lightmaps.has(p_id))
		return Ref<Texture>();

	return lightmaps[p_id];


}

void BakedLight::clear_lightmaps() {

	lightmaps.clear();
	_update_lightmaps();
}

RID BakedLight::get_rid() const {

	return baked_light;
}

Array BakedLight::_get_lightmap_data() const {

	Array ret;
	ret.resize(lightmaps.size()*2);

	int idx=0;
	for(Map<int,Ref<Texture> >::Element *E=lightmaps.front();E;E=E->next()) {

		ret[idx++]=E->key();
		ret[idx++]=E->get();
	}

	return ret;

}

void BakedLight::_set_lightmap_data(Array p_array){

	lightmaps.clear();
	for(int i=0;i<p_array.size();i+=2) {

		int id = p_array[i];
		Ref<Texture> tex = p_array[i+1];
		ERR_CONTINUE(id<0);
		ERR_CONTINUE(tex.is_null());
		lightmaps[id]=tex;
	}
	_update_lightmaps();
}


void BakedLight::set_cell_subdivision(int p_subdiv) {

	cell_subdiv=p_subdiv;
}

int BakedLight::get_cell_subdivision() const{

	return cell_subdiv;
}

void BakedLight::set_initial_lattice_subdiv(int p_size){

	lattice_subdiv=p_size;
}
int BakedLight::get_initial_lattice_subdiv() const{

	return lattice_subdiv;
}

void BakedLight::set_plot_size(float p_size){

	plot_size=p_size;
}
float BakedLight::get_plot_size() const{

	return plot_size;
}

void BakedLight::set_bounces(int p_size){

	bounces=p_size;
}
int BakedLight::get_bounces() const{

	return bounces;
}

void BakedLight::set_cell_extra_margin(float p_margin) {
	cell_extra_margin=p_margin;
}

float BakedLight::get_cell_extra_margin() const {

	return cell_extra_margin;
}

void BakedLight::set_edge_damp(float p_margin) {
	edge_damp=p_margin;
}

float BakedLight::get_edge_damp() const {

	return edge_damp;
}


void BakedLight::set_normal_damp(float p_margin) {
	normal_damp=p_margin;
}

float BakedLight::get_normal_damp() const {

	return normal_damp;
}

void BakedLight::set_energy_multiplier(float p_multiplier){

	energy_multiply=p_multiplier;
}
float BakedLight::get_energy_multiplier() const{

	return energy_multiply;
}

void BakedLight::set_gamma_adjust(float p_adjust){

	gamma_adjust=p_adjust;
}
float BakedLight::get_gamma_adjust() const{

	return gamma_adjust;
}

void BakedLight::set_bake_flag(BakeFlags p_flags,bool p_enable){

	flags[p_flags]=p_enable;
}
bool BakedLight::get_bake_flag(BakeFlags p_flags) const{

	return flags[p_flags];
}

void BakedLight::set_format(Format p_format) {

	format=p_format;

}

BakedLight::Format BakedLight::get_format() const{

	return format;
}


void BakedLight::_bind_methods(){


	ObjectTypeDB::bind_method(_MD("set_mode","mode"),&BakedLight::set_mode);
	ObjectTypeDB::bind_method(_MD("get_mode"),&BakedLight::get_mode);

	ObjectTypeDB::bind_method(_MD("set_octree","octree"),&BakedLight::set_octree);
	ObjectTypeDB::bind_method(_MD("get_octree"),&BakedLight::get_octree);

	ObjectTypeDB::bind_method(_MD("add_lightmap","texture:Texture","id"),&BakedLight::add_lightmap);
	ObjectTypeDB::bind_method(_MD("erase_lightmap","id"),&BakedLight::erase_lightmap);
	ObjectTypeDB::bind_method(_MD("clear_lightmaps"),&BakedLight::clear_lightmaps);

	ObjectTypeDB::bind_method(_MD("_set_lightmap_data","lightmap_data"),&BakedLight::_set_lightmap_data);
	ObjectTypeDB::bind_method(_MD("_get_lightmap_data"),&BakedLight::_get_lightmap_data);

	ObjectTypeDB::bind_method(_MD("set_cell_subdivision","cell_subdivision"),&BakedLight::set_cell_subdivision);
	ObjectTypeDB::bind_method(_MD("get_cell_subdivision"),&BakedLight::get_cell_subdivision);

	ObjectTypeDB::bind_method(_MD("set_initial_lattice_subdiv","cell_subdivision"),&BakedLight::set_initial_lattice_subdiv);
	ObjectTypeDB::bind_method(_MD("get_initial_lattice_subdiv","cell_subdivision"),&BakedLight::get_initial_lattice_subdiv);

	ObjectTypeDB::bind_method(_MD("set_plot_size","plot_size"),&BakedLight::set_plot_size);
	ObjectTypeDB::bind_method(_MD("get_plot_size"),&BakedLight::get_plot_size);

	ObjectTypeDB::bind_method(_MD("set_bounces","bounces"),&BakedLight::set_bounces);
	ObjectTypeDB::bind_method(_MD("get_bounces"),&BakedLight::get_bounces);

	ObjectTypeDB::bind_method(_MD("set_cell_extra_margin","cell_extra_margin"),&BakedLight::set_cell_extra_margin);
	ObjectTypeDB::bind_method(_MD("get_cell_extra_margin"),&BakedLight::get_cell_extra_margin);

	ObjectTypeDB::bind_method(_MD("set_edge_damp","edge_damp"),&BakedLight::set_edge_damp);
	ObjectTypeDB::bind_method(_MD("get_edge_damp"),&BakedLight::get_edge_damp);

	ObjectTypeDB::bind_method(_MD("set_normal_damp","normal_damp"),&BakedLight::set_normal_damp);
	ObjectTypeDB::bind_method(_MD("get_normal_damp"),&BakedLight::get_normal_damp);

	ObjectTypeDB::bind_method(_MD("set_format","format"),&BakedLight::set_format);
	ObjectTypeDB::bind_method(_MD("get_format"),&BakedLight::get_format);


	ObjectTypeDB::bind_method(_MD("set_energy_multiplier","energy_multiplier"),&BakedLight::set_energy_multiplier);
	ObjectTypeDB::bind_method(_MD("get_energy_multiplier"),&BakedLight::get_energy_multiplier);

	ObjectTypeDB::bind_method(_MD("set_gamma_adjust","gamma_adjust"),&BakedLight::set_gamma_adjust);
	ObjectTypeDB::bind_method(_MD("get_gamma_adjust"),&BakedLight::get_gamma_adjust);

	ObjectTypeDB::bind_method(_MD("set_bake_flag","flag","enabled"),&BakedLight::set_bake_flag);
	ObjectTypeDB::bind_method(_MD("get_bake_flag","flag"),&BakedLight::get_bake_flag);

	ADD_PROPERTY( PropertyInfo(Variant::INT,"mode/mode",PROPERTY_HINT_ENUM,"Octree,Lightmaps"),_SCS("set_mode"),_SCS("get_mode"));

	ADD_PROPERTY( PropertyInfo(Variant::INT,"baking/format",PROPERTY_HINT_ENUM,"RGB,HDR8,HDR16"),_SCS("set_format"),_SCS("get_format"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"baking/cell_subdiv",PROPERTY_HINT_RANGE,"4,14,1"),_SCS("set_cell_subdivision"),_SCS("get_cell_subdivision"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"baking/lattice_subdiv",PROPERTY_HINT_RANGE,"1,5,1"),_SCS("set_initial_lattice_subdiv"),_SCS("get_initial_lattice_subdiv"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"baking/light_bounces",PROPERTY_HINT_RANGE,"0,3,1"),_SCS("set_bounces"),_SCS("get_bounces"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"baking/plot_size",PROPERTY_HINT_RANGE,"1.0,16.0,0.01"),_SCS("set_plot_size"),_SCS("get_plot_size"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"baking/energy_mult",PROPERTY_HINT_RANGE,"0.01,4096.0,0.01"),_SCS("set_energy_multiplier"),_SCS("get_energy_multiplier"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"baking/gamma_adjust",PROPERTY_HINT_EXP_EASING),_SCS("set_gamma_adjust"),_SCS("get_gamma_adjust"));
	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"baking_flags/diffuse"),_SCS("set_bake_flag"),_SCS("get_bake_flag"),BAKE_DIFFUSE);
	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"baking_flags/specular"),_SCS("set_bake_flag"),_SCS("get_bake_flag"),BAKE_SPECULAR);
	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"baking_flags/translucent"),_SCS("set_bake_flag"),_SCS("get_bake_flag"),BAKE_TRANSLUCENT);
	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"baking_flags/conserve_energy"),_SCS("set_bake_flag"),_SCS("get_bake_flag"),BAKE_CONSERVE_ENERGY);

	ADD_PROPERTY( PropertyInfo(Variant::RAW_ARRAY,"octree",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR),_SCS("set_octree"),_SCS("get_octree"));
	ADD_PROPERTY( PropertyInfo(Variant::ARRAY,"lightmaps",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR),_SCS("_set_lightmap_data"),_SCS("_get_lightmap_data"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"advanced/cell_margin",PROPERTY_HINT_RANGE,"0.01,0.8,0.01"),_SCS("set_cell_extra_margin"),_SCS("get_cell_extra_margin"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"advanced/edge_damp",PROPERTY_HINT_RANGE,"0.0,8.0,0.1"),_SCS("set_edge_damp"),_SCS("get_edge_damp"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"advanced/normal_damp",PROPERTY_HINT_RANGE,"0.0,1.0,0.01"),_SCS("set_normal_damp"),_SCS("get_normal_damp"));

	BIND_CONSTANT(  MODE_OCTREE );
	BIND_CONSTANT(  MODE_LIGHTMAPS );

	BIND_CONSTANT(  BAKE_DIFFUSE );
	BIND_CONSTANT(  BAKE_SPECULAR );
	BIND_CONSTANT(  BAKE_TRANSLUCENT );
	BIND_CONSTANT(  BAKE_CONSERVE_ENERGY );
	BIND_CONSTANT(  BAKE_MAX );


}


BakedLight::BakedLight() {

	cell_subdiv=8;
	lattice_subdiv=4;
	plot_size=2.5;
	bounces=1;
	energy_multiply=1.0;
	gamma_adjust=1.0;
	cell_extra_margin=0.05;
	edge_damp=0.0;
	normal_damp=0.0;
	format=FORMAT_RGB;

	flags[BAKE_DIFFUSE]=true;
	flags[BAKE_SPECULAR]=false;
	flags[BAKE_TRANSLUCENT]=true;
	flags[BAKE_CONSERVE_ENERGY]=false;

	mode=MODE_OCTREE;
	baked_light=VS::get_singleton()->baked_light_create();
}

BakedLight::~BakedLight() {

	VS::get_singleton()->free(baked_light);
}
