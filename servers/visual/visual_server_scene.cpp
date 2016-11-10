#include "visual_server_scene.h"
#include "visual_server_global.h"

/* CAMERA API */


RID VisualServerScene::camera_create() {

	Camera * camera = memnew( Camera );
	return camera_owner.make_rid( camera );

}

void VisualServerScene::camera_set_perspective(RID p_camera,float p_fovy_degrees, float p_z_near, float p_z_far) {

	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);
	camera->type=Camera::PERSPECTIVE;
	camera->fov=p_fovy_degrees;
	camera->znear=p_z_near;
	camera->zfar=p_z_far;

}

void VisualServerScene::camera_set_orthogonal(RID p_camera,float p_size, float p_z_near, float p_z_far) {

	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);
	camera->type=Camera::ORTHOGONAL;
	camera->size=p_size;
	camera->znear=p_z_near;
	camera->zfar=p_z_far;
}

void VisualServerScene::camera_set_transform(RID p_camera,const Transform& p_transform) {

	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);
	camera->transform=p_transform.orthonormalized();


}

void VisualServerScene::camera_set_cull_mask(RID p_camera,uint32_t p_layers) {


	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);

	camera->visible_layers=p_layers;

}

void VisualServerScene::camera_set_environment(RID p_camera,RID p_env) {

	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);
	camera->env=p_env;

}


void VisualServerScene::camera_set_use_vertical_aspect(RID p_camera,bool p_enable) {

	Camera *camera = camera_owner.get( p_camera );
	ERR_FAIL_COND(!camera);
	camera->vaspect=p_enable;

}


/* SCENARIO API */



void* VisualServerScene::_instance_pair(void *p_self, OctreeElementID, Instance *p_A,int, OctreeElementID, Instance *p_B,int) {

//	VisualServerScene *self = (VisualServerScene*)p_self;
	Instance *A = p_A;
	Instance *B = p_B;

	//instance indices are designed so greater always contains lesser
	if (A->base_type > B->base_type) {
		SWAP(A,B); //lesser always first
	}

	if (B->base_type==VS::INSTANCE_LIGHT && (1<<A->base_type)&VS::INSTANCE_GEOMETRY_MASK) {

		InstanceLightData * light = static_cast<InstanceLightData*>(B->base_data);
		InstanceGeometryData * geom = static_cast<InstanceGeometryData*>(A->base_data);


		InstanceLightData::PairInfo pinfo;
		pinfo.geometry=A;
		pinfo.L = geom->lighting.push_back(B);

		List<InstanceLightData::PairInfo>::Element *E = light->geometries.push_back(pinfo);

		if (geom->can_cast_shadows) {

			light->shadow_dirty=true;
		}
		geom->lighting_dirty=true;

		return E; //this element should make freeing faster
	}

#if 0
	if (A->base_type==INSTANCE_PORTAL) {

		ERR_FAIL_COND_V( B->base_type!=INSTANCE_PORTAL,NULL );

		A->portal_info->candidate_set.insert(B);
		B->portal_info->candidate_set.insert(A);

		self->_portal_attempt_connect(A);
		//attempt to conncet portal A (will go through B anyway)
		//this is a little hackish, but works fine in practice

	} else if (A->base_type==INSTANCE_BAKED_LIGHT || B->base_type==INSTANCE_BAKED_LIGHT) {

		if (B->base_type==INSTANCE_BAKED_LIGHT) {
			SWAP(A,B);
		}

		ERR_FAIL_COND_V(B->base_type!=INSTANCE_BAKED_LIGHT_SAMPLER,NULL);
		B->baked_light_sampler_info->baked_lights.insert(A);

	} else if (A->base_type==INSTANCE_ROOM || B->base_type==INSTANCE_ROOM) {

		if (B->base_type==INSTANCE_ROOM)
			SWAP(A,B);

		ERR_FAIL_COND_V(! ((1<<B->base_type)&INSTANCE_GEOMETRY_MASK ),NULL);

		B->auto_rooms.insert(A);
		A->room_info->owned_autoroom_geometry.insert(B);

		self->_instance_validate_autorooms(B);


	} else {

		if (B->base_type==INSTANCE_LIGHT) {

			SWAP(A,B);
		} else if (A->base_type!=INSTANCE_LIGHT) {
			return NULL;
		}


		A->light_info->affected.insert(B);
		B->lights.insert(A);
		B->light_cache_dirty=true;


	}
#endif

	return NULL;

}
void VisualServerScene::_instance_unpair(void *p_self, OctreeElementID, Instance *p_A,int, OctreeElementID, Instance *p_B,int,void* udata) {

//	VisualServerScene *self = (VisualServerScene*)p_self;
	Instance *A = p_A;
	Instance *B = p_B;

	//instance indices are designed so greater always contains lesser
	if (A->base_type > B->base_type) {
		SWAP(A,B); //lesser always first
	}



	if (B->base_type==VS::INSTANCE_LIGHT && (1<<A->base_type)&VS::INSTANCE_GEOMETRY_MASK) {

		InstanceLightData * light = static_cast<InstanceLightData*>(B->base_data);
		InstanceGeometryData * geom = static_cast<InstanceGeometryData*>(A->base_data);

		List<InstanceLightData::PairInfo>::Element *E = reinterpret_cast<List<InstanceLightData::PairInfo>::Element*>(udata);

		geom->lighting.erase(E->get().L);
		light->geometries.erase(E);

		if (geom->can_cast_shadows) {
			light->shadow_dirty=true;
		}
		geom->lighting_dirty=true;


	}
#if 0
	if (A->base_type==INSTANCE_PORTAL) {

		ERR_FAIL_COND( B->base_type!=INSTANCE_PORTAL );


		A->portal_info->candidate_set.erase(B);
		B->portal_info->candidate_set.erase(A);

		//after disconnecting them, see if they can connect again
		self->_portal_attempt_connect(A);
		self->_portal_attempt_connect(B);

	} else if (A->base_type==INSTANCE_BAKED_LIGHT || B->base_type==INSTANCE_BAKED_LIGHT) {

		if (B->base_type==INSTANCE_BAKED_LIGHT) {
			SWAP(A,B);
		}

		ERR_FAIL_COND(B->base_type!=INSTANCE_BAKED_LIGHT_SAMPLER);
		B->baked_light_sampler_info->baked_lights.erase(A);

	} else if (A->base_type==INSTANCE_ROOM || B->base_type==INSTANCE_ROOM) {

		if (B->base_type==INSTANCE_ROOM)
			SWAP(A,B);

		ERR_FAIL_COND(! ((1<<B->base_type)&INSTANCE_GEOMETRY_MASK ));

		B->auto_rooms.erase(A);
		B->valid_auto_rooms.erase(A);
		A->room_info->owned_autoroom_geometry.erase(B);

	}else {



	if (B->base_type==INSTANCE_LIGHT) {

			SWAP(A,B);
		} else if (A->base_type!=INSTANCE_LIGHT) {
			return;
		}


		A->light_info->affected.erase(B);
		B->lights.erase(A);
		B->light_cache_dirty=true;

	}
#endif
}

RID VisualServerScene::scenario_create() {

	Scenario *scenario = memnew( Scenario );
	ERR_FAIL_COND_V(!scenario,RID());
	RID scenario_rid = scenario_owner.make_rid( scenario );
	scenario->self=scenario_rid;

	scenario->octree.set_pair_callback(_instance_pair,this);
	scenario->octree.set_unpair_callback(_instance_unpair,this);

	return scenario_rid;
}

void VisualServerScene::scenario_set_debug(RID p_scenario,VS::ScenarioDebugMode p_debug_mode) {

	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND(!scenario);
	scenario->debug=p_debug_mode;
}

void VisualServerScene::scenario_set_environment(RID p_scenario, RID p_environment) {

	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND(!scenario);
	scenario->environment=p_environment;

}

void VisualServerScene::scenario_set_fallback_environment(RID p_scenario, RID p_environment) {


	Scenario *scenario = scenario_owner.get(p_scenario);
	ERR_FAIL_COND(!scenario);
	scenario->fallback_environment=p_environment;


}



/* INSTANCING API */

void VisualServerScene::_instance_queue_update(Instance *p_instance,bool p_update_aabb,bool p_update_materials) {

	if (p_update_aabb)
		p_instance->update_aabb=true;
	if (p_update_materials)
		p_instance->update_materials=true;

	if (p_instance->update_item.in_list())
		return;

	_instance_update_list.add(&p_instance->update_item);


}

// from can be mesh, light,  area and portal so far.
RID VisualServerScene::instance_create(){

	Instance *instance = memnew( Instance );
	ERR_FAIL_COND_V(!instance,RID());

	RID instance_rid = instance_owner.make_rid(instance);
	instance->self=instance_rid;


	return instance_rid;


}

void VisualServerScene::instance_set_base(RID p_instance, RID p_base){

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	Scenario *scenario = instance->scenario;

	if (instance->base_type!=VS::INSTANCE_NONE) {
		//free anything related to that base

		VSG::storage->instance_remove_dependency(instance->base,instance);

		if (scenario && instance->octree_id) {
			scenario->octree.erase(instance->octree_id); //make dependencies generated by the octree go away
			instance->octree_id=0;
		}

		switch(instance->base_type) {
			case VS::INSTANCE_LIGHT: {

				InstanceLightData *light = static_cast<InstanceLightData*>(instance->base_data);

				if (instance->scenario && light->D) {
					instance->scenario->directional_lights.erase( light->D );
					light->D=NULL;
				}
				VSG::scene_render->free(light->instance);
			} break;
		}

		if (instance->base_data) {
			memdelete( instance->base_data );
			instance->base_data=NULL;
		}

		instance->morph_values.clear();

		for(int i=0;i<instance->materials.size();i++) {
			if (instance->materials[i].is_valid()) {
				VSG::storage->material_remove_instance_owner(instance->materials[i],instance);
			}
		}
		instance->materials.clear();

#if 0
		if (instance->light_info) {

			if (instance->scenario && instance->light_info->D)
				instance->scenario->directional_lights.erase( instance->light_info->D );
			rasterizer->free(instance->light_info->instance);
			memdelete(instance->light_info);
			instance->light_info=NULL;
		}



		if ( instance->room ) {

			instance_set_room(p_instance,RID());
			/*
			if((1<<instance->base_type)&INSTANCE_GEOMETRY_MASK)
				instance->room->room_info->owned_geometry_instances.erase(instance->RE);
			else if (instance->base_type==INSTANCE_PORTAL) {
				print_line("freeing portal, is it there? "+itos(instance->room->room_info->owned_portal_instances.(instance->RE)));
				instance->room->room_info->owned_portal_instances.erase(instance->RE);
			} else if (instance->base_type==INSTANCE_ROOM)
				instance->room->room_info->owned_room_instances.erase(instance->RE);
			else if (instance->base_type==INSTANCE_LIGHT)
				instance->room->room_info->owned_light_instances.erase(instance->RE);

			instance->RE=NULL;*/
		}






		if (instance->portal_info) {

			_portal_disconnect(instance,true);
			memdelete(instance->portal_info);
			instance->portal_info=NULL;

		}

		if (instance->baked_light_info) {

			while(instance->baked_light_info->owned_instances.size()) {

				Instance *owned=instance->baked_light_info->owned_instances.front()->get();
				owned->baked_light=NULL;
				owned->data.baked_light=NULL;
				owned->data.baked_light_octree_xform=NULL;
				owned->BLE=NULL;
				instance->baked_light_info->owned_instances.pop_front();
			}

			memdelete(instance->baked_light_info);
			instance->baked_light_info=NULL;

		}

		if (instance->scenario && instance->octree_id) {
			instance->scenario->octree.erase( instance->octree_id );
			instance->octree_id=0;
		}


		if (instance->room_info) {

			for(List<Instance*>::Element *E=instance->room_info->owned_geometry_instances.front();E;E=E->next()) {

				Instance *owned = E->get();
				owned->room=NULL;
				owned->RE=NULL;
			}
			for(List<Instance*>::Element *E=instance->room_info->owned_portal_instances.front();E;E=E->next()) {

				_portal_disconnect(E->get(),true);
				Instance *owned = E->get();
				owned->room=NULL;
				owned->RE=NULL;
			}

			for(List<Instance*>::Element *E=instance->room_info->owned_room_instances.front();E;E=E->next()) {

				Instance *owned = E->get();
				owned->room=NULL;
				owned->RE=NULL;
			}

			if (instance->room_info->disconnected_child_portals.size()) {
				ERR_PRINT("BUG: Disconnected portals remain!");
			}
			memdelete(instance->room_info);
			instance->room_info=NULL;

		}

		if (instance->particles_info) {

			rasterizer->free( instance->particles_info->instance );
			memdelete(instance->particles_info);
			instance->particles_info=NULL;

		}

		if (instance->baked_light_sampler_info) {

			while (instance->baked_light_sampler_info->owned_instances.size()) {

				instance_geometry_set_baked_light_sampler(instance->baked_light_sampler_info->owned_instances.front()->get()->self,RID());
			}

			if (instance->baked_light_sampler_info->sampled_light.is_valid()) {
				rasterizer->free(instance->baked_light_sampler_info->sampled_light);
			}
			memdelete( instance->baked_light_sampler_info );
			instance->baked_light_sampler_info=NULL;
		}
#endif

	}


	instance->base_type=VS::INSTANCE_NONE;
	instance->base=RID();


	if (p_base.is_valid()) {

		instance->base_type=VSG::storage->get_base_type(p_base);
		ERR_FAIL_COND(instance->base_type==VS::INSTANCE_NONE);

		switch(instance->base_type) {
			case VS::INSTANCE_LIGHT: {

				InstanceLightData *light = memnew( InstanceLightData );

				if (scenario && VSG::storage->light_get_type(p_base)==VS::LIGHT_DIRECTIONAL) {
					light->D = scenario->directional_lights.push_back(instance);
				}

				light->instance = VSG::scene_render->light_instance_create(p_base);

				instance->base_data=light;
			} break;
			case VS::INSTANCE_MESH: {

				InstanceGeometryData *geom = memnew( InstanceGeometryData );
				instance->base_data=geom;
			} break;

		}

		VSG::storage->instance_add_dependency(p_base,instance);

		instance->base=p_base;

		if (scenario)
			_instance_queue_update(instance,true,true);


#if 0
		if (rasterizer->is_mesh(p_base)) {
			instance->base_type=INSTANCE_MESH;
			instance->data.morph_values.resize( rasterizer->mesh_get_morph_target_count(p_base));
			instance->data.materials.resize( rasterizer->mesh_get_surface_count(p_base));
		} else if (rasterizer->is_multimesh(p_base)) {
			instance->base_type=INSTANCE_MULTIMESH;
		} else if (rasterizer->is_immediate(p_base)) {
			instance->base_type=INSTANCE_IMMEDIATE;
		} else if (rasterizer->is_particles(p_base)) {
			instance->base_type=INSTANCE_PARTICLES;
			instance->particles_info=memnew( Instance::ParticlesInfo );
			instance->particles_info->instance = rasterizer->particles_instance_create( p_base );
		} else if (rasterizer->is_light(p_base)) {

			instance->base_type=INSTANCE_LIGHT;
			instance->light_info = memnew( Instance::LightInfo );
			instance->light_info->instance = rasterizer->light_instance_create(p_base);
			if (instance->scenario && rasterizer->light_get_type(p_base)==LIGHT_DIRECTIONAL) {

				instance->light_info->D = instance->scenario->directional_lights.push_back(instance->self);
			}

		} else if (room_owner.owns(p_base)) {
			instance->base_type=INSTANCE_ROOM;
			instance->room_info  = memnew( Instance::RoomInfo );
			instance->room_info->room=room_owner.get(p_base);
		} else if (portal_owner.owns(p_base)) {

			instance->base_type=INSTANCE_PORTAL;
			instance->portal_info = memnew(Instance::PortalInfo);
			instance->portal_info->portal=portal_owner.get(p_base);
		} else if (baked_light_owner.owns(p_base)) {

			instance->base_type=INSTANCE_BAKED_LIGHT;
			instance->baked_light_info=memnew(Instance::BakedLightInfo);
			instance->baked_light_info->baked_light=baked_light_owner.get(p_base);

			//instance->portal_info = memnew(Instance::PortalInfo);
			//instance->portal_info->portal=portal_owner.get(p_base);
		} else if (baked_light_sampler_owner.owns(p_base)) {


			instance->base_type=INSTANCE_BAKED_LIGHT_SAMPLER;
			instance->baked_light_sampler_info=memnew( Instance::BakedLightSamplerInfo);
			instance->baked_light_sampler_info->sampler=baked_light_sampler_owner.get(p_base);

			//instance->portal_info = memnew(Instance::PortalInfo);
			//instance->portal_info->portal=portal_owner.get(p_base);

		} else {
			ERR_EXPLAIN("Invalid base RID for instance!")
			ERR_FAIL();
		}

		instance_dependency_map[ p_base ].insert( instance->self );
#endif


	}
}
void VisualServerScene::instance_set_scenario(RID p_instance, RID p_scenario){

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	if (instance->scenario) {

		instance->scenario->instances.remove( &instance->scenario_item );

		if (instance->octree_id) {
			instance->scenario->octree.erase(instance->octree_id); //make dependencies generated by the octree go away
			instance->octree_id=0;
		}


		switch(instance->base_type) {

			case VS::INSTANCE_LIGHT: {


				InstanceLightData *light = static_cast<InstanceLightData*>(instance->base_data);

				if (light->D) {
					instance->scenario->directional_lights.erase( light->D );
					light->D=NULL;
				}
			} break;
		}

		instance->scenario=NULL;
	}


	if (p_scenario.is_valid()) {

		Scenario *scenario = scenario_owner.get( p_scenario );
		ERR_FAIL_COND(!scenario);

		instance->scenario=scenario;

		scenario->instances.add( &instance->scenario_item );


		switch(instance->base_type) {

			case VS::INSTANCE_LIGHT: {


				InstanceLightData *light = static_cast<InstanceLightData*>(instance->base_data);

				if (VSG::storage->light_get_type(instance->base)==VS::LIGHT_DIRECTIONAL) {
					light->D = scenario->directional_lights.push_back(instance);
				}
			} break;
		}

		_instance_queue_update(instance,true,true);
	}
}
void VisualServerScene::instance_set_layer_mask(RID p_instance, uint32_t p_mask){


	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	instance->layer_mask=p_mask;
}
void VisualServerScene::instance_set_transform(RID p_instance, const Transform& p_transform){

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	if (instance->transform==p_transform)
		return; //must be checked to avoid worst evil

	instance->transform=p_transform;
	_instance_queue_update(instance,true);
}
void VisualServerScene::instance_attach_object_instance_ID(RID p_instance,ObjectID p_ID){

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	instance->object_ID=p_ID;

}
void VisualServerScene::instance_set_morph_target_weight(RID p_instance,int p_shape, float p_weight){

}
void VisualServerScene::instance_set_surface_material(RID p_instance,int p_surface, RID p_material){

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	_update_dirty_instance(instance);

	ERR_FAIL_INDEX(p_surface,instance->materials.size());

	if (instance->materials[p_surface].is_valid()) {
		VSG::storage->material_remove_instance_owner(instance->materials[p_surface],instance);
	}
	instance->materials[p_surface]=p_material;
	instance->base_material_changed();

	if (instance->materials[p_surface].is_valid()) {
		VSG::storage->material_add_instance_owner(instance->materials[p_surface],instance);
	}


}

void VisualServerScene::instance_attach_skeleton(RID p_instance,RID p_skeleton){

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	instance->skeleton=p_skeleton;

	_instance_queue_update(instance,true);
}

void VisualServerScene::instance_set_exterior( RID p_instance, bool p_enabled ){

}
void VisualServerScene::instance_set_room( RID p_instance, RID p_room ){

}

void VisualServerScene::instance_set_extra_visibility_margin( RID p_instance, real_t p_margin ){

}

Vector<ObjectID> VisualServerScene::instances_cull_aabb(const AABB& p_aabb, RID p_scenario) const {


	Vector<ObjectID> instances;
	Scenario *scenario=scenario_owner.get(p_scenario);
	ERR_FAIL_COND_V(!scenario,instances);

	const_cast<VisualServerScene*>(this)->update_dirty_instances(); // check dirty instances before culling

	int culled=0;
	Instance *cull[1024];
	culled=scenario->octree.cull_AABB(p_aabb,cull,1024);

	for (int i=0;i<culled;i++) {

		Instance *instance=cull[i];
		ERR_CONTINUE(!instance);
		if (instance->object_ID==0)
			continue;

		instances.push_back(instance->object_ID);
	}

	return instances;
}
Vector<ObjectID> VisualServerScene::instances_cull_ray(const Vector3& p_from, const Vector3& p_to, RID p_scenario) const{

	Vector<ObjectID> instances;
	Scenario *scenario=scenario_owner.get(p_scenario);
	ERR_FAIL_COND_V(!scenario,instances);
	const_cast<VisualServerScene*>(this)->update_dirty_instances(); // check dirty instances before culling

	int culled=0;
	Instance *cull[1024];
	culled=scenario->octree.cull_segment(p_from,p_to*10000,cull,1024);


	for (int i=0;i<culled;i++) {
		Instance *instance=cull[i];
		ERR_CONTINUE(!instance);
		if (instance->object_ID==0)
			continue;

		instances.push_back(instance->object_ID);
	}

	return instances;

}
Vector<ObjectID> VisualServerScene::instances_cull_convex(const Vector<Plane>& p_convex,  RID p_scenario) const{

	Vector<ObjectID> instances;
	Scenario *scenario=scenario_owner.get(p_scenario);
	ERR_FAIL_COND_V(!scenario,instances);
	const_cast<VisualServerScene*>(this)->update_dirty_instances(); // check dirty instances before culling

	int culled=0;
	Instance *cull[1024];


	culled=scenario->octree.cull_convex(p_convex,cull,1024);

	for (int i=0;i<culled;i++) {

		Instance *instance=cull[i];
		ERR_CONTINUE(!instance);
		if (instance->object_ID==0)
			continue;

		instances.push_back(instance->object_ID);
	}

	return instances;

}

void VisualServerScene::instance_geometry_set_flag(RID p_instance,VS::InstanceFlags p_flags,bool p_enabled){

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	switch(p_flags) {

		case VS::INSTANCE_FLAG_VISIBLE: {

			instance->visible=p_enabled;

		} break;
		case VS::INSTANCE_FLAG_BILLBOARD: {

			instance->billboard=p_enabled;

		} break;
		case VS::INSTANCE_FLAG_BILLBOARD_FIX_Y: {

			instance->billboard_y=p_enabled;

		} break;
		case VS::INSTANCE_FLAG_CAST_SHADOW: {
			if (p_enabled == true) {
				instance->cast_shadows = VS::SHADOW_CASTING_SETTING_ON;
			}
			else {
				instance->cast_shadows = VS::SHADOW_CASTING_SETTING_OFF;
			}

			instance->base_material_changed(); // to actually compute if shadows are visible or not

		} break;
		case VS::INSTANCE_FLAG_DEPH_SCALE: {

			instance->depth_scale=p_enabled;

		} break;
		case VS::INSTANCE_FLAG_VISIBLE_IN_ALL_ROOMS: {

			instance->visible_in_all_rooms=p_enabled;

		} break;

	}
}
void VisualServerScene::instance_geometry_set_cast_shadows_setting(RID p_instance, VS::ShadowCastingSetting p_shadow_casting_setting) {

}
void VisualServerScene::instance_geometry_set_material_override(RID p_instance, RID p_material){

	Instance *instance = instance_owner.get( p_instance );
	ERR_FAIL_COND( !instance );

	if (instance->material_override.is_valid()) {
		VSG::storage->material_remove_instance_owner(instance->material_override,instance);
	}
	instance->material_override=p_material;
	instance->base_material_changed();

	if (instance->material_override.is_valid()) {
		VSG::storage->material_add_instance_owner(instance->material_override,instance);
	}

}


void VisualServerScene::instance_geometry_set_draw_range(RID p_instance,float p_min,float p_max,float p_min_margin,float p_max_margin){

}
void VisualServerScene::instance_geometry_set_as_instance_lod(RID p_instance,RID p_as_lod_of_instance){

}


void VisualServerScene::_update_instance(Instance *p_instance) {

	p_instance->version++;

	if (p_instance->base_type == VS::INSTANCE_LIGHT) {

		InstanceLightData *light = static_cast<InstanceLightData*>(p_instance->base_data);

		VSG::scene_render->light_instance_set_transform( light->instance, p_instance->transform );
		light->shadow_dirty=true;

	}


	if (p_instance->aabb.has_no_surface())
		return;

#if 0
	if (p_instance->base_type == VS::INSTANCE_PARTICLES) {

		rasterizer->particles_instance_set_transform( p_instance->particles_info->instance, p_instance->data.transform );
	}

#endif
	if ((1<<p_instance->base_type)&VS::INSTANCE_GEOMETRY_MASK) {

		InstanceGeometryData *geom = static_cast<InstanceGeometryData*>(p_instance->base_data);
		//make sure lights are updated if it casts shadow

		if (geom->can_cast_shadows) {
			for (List<Instance*>::Element *E=geom->lighting.front();E;E=E->next()) {
				InstanceLightData *light = static_cast<InstanceLightData*>(E->get()->base_data);
				light->shadow_dirty=true;
			}
		}

	}
#if 0
	else if (p_instance->base_type == INSTANCE_ROOM) {

		p_instance->room_info->affine_inverse=p_instance->data.transform.affine_inverse();
	} else if (p_instance->base_type == INSTANCE_BAKED_LIGHT) {

		Transform scale;
		scale.basis.scale(p_instance->baked_light_info->baked_light->octree_aabb.size);
		scale.origin=p_instance->baked_light_info->baked_light->octree_aabb.pos;
		//print_line("scale: "+scale);
		p_instance->baked_light_info->affine_inverse=(p_instance->data.transform*scale).affine_inverse();
	}


#endif

	p_instance->mirror = p_instance->transform.basis.determinant() < 0.0;

	AABB new_aabb;
#if 0
	if (p_instance->base_type==INSTANCE_PORTAL) {

		//portals need to be transformed in a special way, so they don't become too wide if they have scale..
		Transform portal_xform = p_instance->data.transform;
		portal_xform.basis.set_axis(2,portal_xform.basis.get_axis(2).normalized());

		p_instance->portal_info->plane_cache=Plane( p_instance->data.transform.origin, portal_xform.basis.get_axis(2));
		int point_count=p_instance->portal_info->portal->shape.size();
		p_instance->portal_info->transformed_point_cache.resize(point_count);

		AABB portal_aabb;

		for(int i=0;i<point_count;i++) {

			Point2 src = p_instance->portal_info->portal->shape[i];
			Vector3 point = portal_xform.xform(Vector3(src.x,src.y,0));
			p_instance->portal_info->transformed_point_cache[i]=point;
			if (i==0)
				portal_aabb.pos=point;
			else
				portal_aabb.expand_to(point);
		}

		portal_aabb.grow_by(p_instance->portal_info->portal->connect_range);

		new_aabb = portal_aabb;

	} else {
#endif
		new_aabb = p_instance->transform.xform(p_instance->aabb);
#if 0
	}
#endif


	p_instance->transformed_aabb=new_aabb;

	if (!p_instance->scenario) {

		return;
	}



	if (p_instance->octree_id==0) {

		uint32_t base_type = 1<<p_instance->base_type;
		uint32_t pairable_mask=0;
		bool pairable=false;

		if (p_instance->base_type == VS::INSTANCE_LIGHT) {

			pairable_mask=p_instance->visible?VS::INSTANCE_GEOMETRY_MASK:0;
			pairable=true;
		}
#if 0

		if (p_instance->base_type == VS::INSTANCE_PORTAL) {

			pairable_mask=(1<<INSTANCE_PORTAL);
			pairable=true;
		}

		if (p_instance->base_type == VS::INSTANCE_BAKED_LIGHT_SAMPLER) {

			pairable_mask=(1<<INSTANCE_BAKED_LIGHT);
			pairable=true;
		}


		if (!p_instance->room && (1<<p_instance->base_type)&VS::INSTANCE_GEOMETRY_MASK) {

			base_type|=VS::INSTANCE_ROOMLESS_MASK;
		}

		if (p_instance->base_type == VS::INSTANCE_ROOM) {

			pairable_mask=INSTANCE_ROOMLESS_MASK;
			pairable=true;
		}
#endif

		// not inside octree
		p_instance->octree_id = p_instance->scenario->octree.create(p_instance,new_aabb,0,pairable,base_type,pairable_mask);

	} else {

	//	if (new_aabb==p_instance->data.transformed_aabb)
	//		return;

		p_instance->scenario->octree.move(p_instance->octree_id,new_aabb);
	}
#if 0
	if (p_instance->base_type==INSTANCE_PORTAL) {

		_portal_attempt_connect(p_instance);
	}

	if (!p_instance->room && (1<<p_instance->base_type)&INSTANCE_GEOMETRY_MASK) {

		_instance_validate_autorooms(p_instance);
	}

	if (p_instance->base_type == INSTANCE_ROOM) {

		for(Set<Instance*>::Element *E=p_instance->room_info->owned_autoroom_geometry.front();E;E=E->next())
			_instance_validate_autorooms(E->get());
	}
#endif

}

void VisualServerScene::_update_instance_aabb(Instance *p_instance) {

	AABB new_aabb;

	ERR_FAIL_COND(p_instance->base_type!=VS::INSTANCE_NONE && !p_instance->base.is_valid());

	switch(p_instance->base_type) {
		case VisualServer::INSTANCE_NONE: {

			// do nothing
		} break;
		case VisualServer::INSTANCE_MESH: {

			new_aabb = VSG::storage->mesh_get_aabb(p_instance->base,p_instance->skeleton);

		} break;
#if 0
		case VisualServer::INSTANCE_MULTIMESH: {

			new_aabb = rasterizer->multimesh_get_aabb(p_instance->base);

		} break;
		case VisualServer::INSTANCE_IMMEDIATE: {

			new_aabb = rasterizer->immediate_get_aabb(p_instance->base);


		} break;
		case VisualServer::INSTANCE_PARTICLES: {

			new_aabb = rasterizer->particles_get_aabb(p_instance->base);


		} break;
#endif
		case VisualServer::INSTANCE_LIGHT: {

			new_aabb = VSG::storage->light_get_aabb(p_instance->base);

		} break;
#if 0
		case VisualServer::INSTANCE_ROOM: {

			Room *room = room_owner.get( p_instance->base );
			ERR_FAIL_COND(!room);
			new_aabb=room->bounds.get_aabb();

		} break;
		case VisualServer::INSTANCE_PORTAL: {

			Portal *portal = portal_owner.get( p_instance->base );
			ERR_FAIL_COND(!portal);
			for (int i=0;i<portal->shape.size();i++) {

				Vector3 point( portal->shape[i].x, portal->shape[i].y, 0 );
				if (i==0) {

					new_aabb.pos=point;
					new_aabb.size.z=0.01; // make it not flat for octree
				} else {

					new_aabb.expand_to(point);
				}
			}

		} break;
		case VisualServer::INSTANCE_BAKED_LIGHT: {

			BakedLight *baked_light = baked_light_owner.get( p_instance->base );
			ERR_FAIL_COND(!baked_light);
			new_aabb=baked_light->octree_aabb;

		} break;
		case VisualServer::INSTANCE_BAKED_LIGHT_SAMPLER: {

			BakedLightSampler *baked_light_sampler = baked_light_sampler_owner.get( p_instance->base );
			ERR_FAIL_COND(!baked_light_sampler);
			float radius = baked_light_sampler->params[VS::BAKED_LIGHT_SAMPLER_RADIUS];

			new_aabb=AABB(Vector3(-radius,-radius,-radius),Vector3(radius*2,radius*2,radius*2));

		} break;
#endif
		default: {}
	}

	if (p_instance->extra_margin)
		new_aabb.grow_by(p_instance->extra_margin);

	p_instance->aabb=new_aabb;

}





void VisualServerScene::_light_instance_update_shadow(Instance *p_instance,Camera* p_camera,RID p_shadow_atlas,Scenario* p_scenario,Size2 p_viewport_rect) {


	InstanceLightData * light = static_cast<InstanceLightData*>(p_instance->base_data);

	switch(VSG::storage->light_get_type(p_instance->base)) {

		case VS::LIGHT_DIRECTIONAL: {

			float max_distance = p_camera->zfar;
			float shadow_max = VSG::storage->light_get_param(p_instance->base,VS::LIGHT_PARAM_SHADOW_MAX_DISTANCE);
			if (shadow_max>0) {
				max_distance=MIN(shadow_max,max_distance);
			}
			max_distance=MAX(max_distance,p_camera->znear+0.001);

			float range = max_distance-p_camera->znear;

			int splits=0;
			switch(VSG::storage->light_directional_get_shadow_mode(p_instance->base)) {
				case VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL: splits=1; break;
				case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS: splits=2; break;
				case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS: splits=4; break;
			}

			float distances[5];

			distances[0]=p_camera->znear;
			for(int i=0;i<splits;i++) {
				distances[i+1]=p_camera->znear+VSG::storage->light_get_param(p_instance->base,VS::LightParam(VS::LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET+i))*range;
			};

			distances[splits]=max_distance;

			float texture_size=VSG::scene_render->get_directional_light_shadow_size(light->instance);

			bool overlap = false;//rasterizer->light_instance_get_pssm_shadow_overlap(p_light->light_info->instance);

			for (int i=0;i<splits;i++) {

				// setup a camera matrix for that range!
				CameraMatrix camera_matrix;

				switch(p_camera->type) {

					case Camera::ORTHOGONAL: {

						camera_matrix.set_orthogonal(
							p_camera->size,
							p_viewport_rect.width / p_viewport_rect.height,
							distances[(i==0 || !overlap )?i:i-1],
							distances[i+1],
							p_camera->vaspect

						);
					} break;
					case Camera::PERSPECTIVE: {


						camera_matrix.set_perspective(
							p_camera->fov,
							p_viewport_rect.width / (float)p_viewport_rect.height,
							distances[(i==0 || !overlap )?i:i-1],
							distances[i+1],
							p_camera->vaspect

						);

					} break;
				}

				//obtain the frustum endpoints

				Vector3 endpoints[8]; // frustum plane endpoints
				bool res = camera_matrix.get_endpoints(p_camera->transform,endpoints);
				ERR_CONTINUE(!res);

				// obtain the light frustm ranges (given endpoints)

				Vector3 x_vec=p_instance->transform.basis.get_axis( Vector3::AXIS_X ).normalized();
				Vector3 y_vec=p_instance->transform.basis.get_axis( Vector3::AXIS_Y ).normalized();
				Vector3 z_vec=p_instance->transform.basis.get_axis( Vector3::AXIS_Z ).normalized();
				//z_vec points agsint the camera, like in default opengl

				float x_min,x_max;
				float y_min,y_max;
				float z_min,z_max;

				float x_min_cam,x_max_cam;
				float y_min_cam,y_max_cam;
				float z_min_cam,z_max_cam;


				//used for culling
				for(int j=0;j<8;j++) {

					float d_x=x_vec.dot(endpoints[j]);
					float d_y=y_vec.dot(endpoints[j]);
					float d_z=z_vec.dot(endpoints[j]);

					if (j==0 || d_x<x_min)
						x_min=d_x;
					if (j==0 || d_x>x_max)
						x_max=d_x;

					if (j==0 || d_y<y_min)
						y_min=d_y;
					if (j==0 || d_y>y_max)
						y_max=d_y;

					if (j==0 || d_z<z_min)
						z_min=d_z;
					if (j==0 || d_z>z_max)
						z_max=d_z;


				}





				{
					//camera viewport stuff
					//this trick here is what stabilizes the shadow (make potential jaggies to not move)
					//at the cost of some wasted resolution. Still the quality increase is very well worth it


					Vector3 center;

					for(int j=0;j<8;j++) {

						center+=endpoints[j];
					}
					center/=8.0;

					//center=x_vec*(x_max-x_min)*0.5 + y_vec*(y_max-y_min)*0.5 + z_vec*(z_max-z_min)*0.5;

					float radius=0;

					for(int j=0;j<8;j++) {

						float d = center.distance_to(endpoints[j]);
						if (d>radius)
							radius=d;
					}


					radius *= texture_size/(texture_size-2.0); //add a texel by each side, so stepified texture will always fit

					x_max_cam=x_vec.dot(center)+radius;
					x_min_cam=x_vec.dot(center)-radius;
					y_max_cam=y_vec.dot(center)+radius;
					y_min_cam=y_vec.dot(center)-radius;
					z_max_cam=z_vec.dot(center)+radius;
					z_min_cam=z_vec.dot(center)-radius;

					float unit = radius*2.0/texture_size;

					x_max_cam=Math::stepify(x_max_cam,unit);
					x_min_cam=Math::stepify(x_min_cam,unit);
					y_max_cam=Math::stepify(y_max_cam,unit);
					y_min_cam=Math::stepify(y_min_cam,unit);

				}

				//now that we now all ranges, we can proceed to make the light frustum planes, for culling octree

				Vector<Plane> light_frustum_planes;
				light_frustum_planes.resize(6);

				//right/left
				light_frustum_planes[0]=Plane( x_vec, x_max );
				light_frustum_planes[1]=Plane( -x_vec, -x_min );
				//top/bottom
				light_frustum_planes[2]=Plane( y_vec, y_max );
				light_frustum_planes[3]=Plane( -y_vec, -y_min );
				//near/far
				light_frustum_planes[4]=Plane( z_vec, z_max+1e6 );
				light_frustum_planes[5]=Plane( -z_vec, -z_min ); // z_min is ok, since casters further than far-light plane are not needed

				int cull_count = p_scenario->octree.cull_convex(light_frustum_planes,instance_shadow_cull_result,MAX_INSTANCE_CULL,VS::INSTANCE_GEOMETRY_MASK);

				// a pre pass will need to be needed to determine the actual z-near to be used


				for (int j=0;j<cull_count;j++) {

					float min,max;
					Instance *instance = instance_shadow_cull_result[j];
					if (!instance->visible || !((1<<instance->base_type)&VS::INSTANCE_GEOMETRY_MASK) || !static_cast<InstanceGeometryData*>(instance->base_data)->can_cast_shadows) {
						cull_count--;
						SWAP(instance_shadow_cull_result[j],instance_shadow_cull_result[cull_count]);
						j--;

					}

					instance->transformed_aabb.project_range_in_plane(Plane(z_vec,0),min,max);
					if (max>z_max)
						z_max=max;
				}

				{
					CameraMatrix ortho_camera;
					real_t half_x = (x_max_cam-x_min_cam) * 0.5;
					real_t half_y = (y_max_cam-y_min_cam) * 0.5;


					ortho_camera.set_orthogonal( -half_x, half_x,-half_y,half_y, 0, (z_max-z_min_cam) );

					Transform ortho_transform;
					ortho_transform.basis=p_instance->transform.basis;
					ortho_transform.origin=x_vec*(x_min_cam+half_x)+y_vec*(y_min_cam+half_y)+z_vec*z_max;

					VSG::scene_render->light_instance_set_shadow_transform(light->instance,ortho_camera,ortho_transform,0,distances[i+1],i);
				}



				VSG::scene_render->render_shadow(light->instance,p_shadow_atlas,i,(RasterizerScene::InstanceBase**)instance_shadow_cull_result,cull_count);

			}

		} break;
		case VS::LIGHT_OMNI: {

			VS::LightOmniShadowMode shadow_mode = VSG::storage->light_omni_get_shadow_mode(p_instance->base);

			switch(shadow_mode) {
				case VS::LIGHT_OMNI_SHADOW_DUAL_PARABOLOID: {

					for(int i=0;i<2;i++) {

						//using this one ensures that raster deferred will have it

						float radius = VSG::storage->light_get_param( p_instance->base, VS::LIGHT_PARAM_RANGE);

						float z =i==0?-1:1;
						Vector<Plane> planes;
						planes.resize(5);
						planes[0]=p_instance->transform.xform(Plane(Vector3(0,0,z),radius));
						planes[1]=p_instance->transform.xform(Plane(Vector3(1,0,z).normalized(),radius));
						planes[2]=p_instance->transform.xform(Plane(Vector3(-1,0,z).normalized(),radius));
						planes[3]=p_instance->transform.xform(Plane(Vector3(0,1,z).normalized(),radius));
						planes[4]=p_instance->transform.xform(Plane(Vector3(0,-1,z).normalized(),radius));


						int cull_count = p_scenario->octree.cull_convex(planes,instance_shadow_cull_result,MAX_INSTANCE_CULL,VS::INSTANCE_GEOMETRY_MASK);

						for (int j=0;j<cull_count;j++) {

							Instance *instance = instance_shadow_cull_result[j];
							if (!instance->visible || !((1<<instance->base_type)&VS::INSTANCE_GEOMETRY_MASK) || !static_cast<InstanceGeometryData*>(instance->base_data)->can_cast_shadows) {
								cull_count--;
								SWAP(instance_shadow_cull_result[j],instance_shadow_cull_result[cull_count]);
								j--;

							}
						}

						VSG::scene_render->light_instance_set_shadow_transform(light->instance,CameraMatrix(),p_instance->transform,radius,0,i);
						VSG::scene_render->render_shadow(light->instance,p_shadow_atlas,i,(RasterizerScene::InstanceBase**)instance_shadow_cull_result,cull_count);
					}
				} break;
				case VS::LIGHT_OMNI_SHADOW_CUBE: {

					float radius = VSG::storage->light_get_param( p_instance->base, VS::LIGHT_PARAM_RANGE);
					CameraMatrix cm;
					cm.set_perspective(90,1,0.01,radius);

					for(int i=0;i<6;i++) {

						//using this one ensures that raster deferred will have it



						static const Vector3 view_normals[6]={
							Vector3(-1, 0, 0),
							Vector3(+1, 0, 0),
							Vector3( 0,-1, 0),
							Vector3( 0,+1, 0),
							Vector3( 0, 0,-1),
							Vector3( 0, 0,+1)
						};
						static const Vector3 view_up[6]={
							Vector3( 0,-1, 0),
							Vector3( 0,-1, 0),
							Vector3( 0, 0,-1),
							Vector3( 0, 0,+1),
							Vector3( 0,-1, 0),
							Vector3( 0,-1, 0)
						};

						Transform xform = p_instance->transform * Transform().looking_at(view_normals[i],view_up[i]);


						Vector<Plane> planes = cm.get_projection_planes(xform);

						int cull_count = p_scenario->octree.cull_convex(planes,instance_shadow_cull_result,MAX_INSTANCE_CULL,VS::INSTANCE_GEOMETRY_MASK);

						for (int j=0;j<cull_count;j++) {

							Instance *instance = instance_shadow_cull_result[j];
							if (!instance->visible || !((1<<instance->base_type)&VS::INSTANCE_GEOMETRY_MASK) || !static_cast<InstanceGeometryData*>(instance->base_data)->can_cast_shadows) {
								cull_count--;
								SWAP(instance_shadow_cull_result[j],instance_shadow_cull_result[cull_count]);
								j--;

							}
						}

						VSG::scene_render->light_instance_set_shadow_transform(light->instance,cm,xform,radius,0,i);
						VSG::scene_render->render_shadow(light->instance,p_shadow_atlas,i,(RasterizerScene::InstanceBase**)instance_shadow_cull_result,cull_count);
					}

					//restore the regular DP matrix
					VSG::scene_render->light_instance_set_shadow_transform(light->instance,CameraMatrix(),p_instance->transform,radius,0,0);

				} break;
			}


		} break;
		case VS::LIGHT_SPOT: {


			float radius = VSG::storage->light_get_param( p_instance->base, VS::LIGHT_PARAM_RANGE);
			float angle = VSG::storage->light_get_param( p_instance->base, VS::LIGHT_PARAM_SPOT_ANGLE);

			CameraMatrix cm;
			cm.set_perspective( 90, 1.0, 0.01, radius );
			print_line("perspective: "+cm);

			Vector<Plane> planes = cm.get_projection_planes(p_instance->transform);
			int cull_count = p_scenario->octree.cull_convex(planes,instance_shadow_cull_result,MAX_INSTANCE_CULL,VS::INSTANCE_GEOMETRY_MASK);

			for (int j=0;j<cull_count;j++) {

				Instance *instance = instance_shadow_cull_result[j];
				if (!instance->visible || !((1<<instance->base_type)&VS::INSTANCE_GEOMETRY_MASK) || !static_cast<InstanceGeometryData*>(instance->base_data)->can_cast_shadows) {
					cull_count--;
					SWAP(instance_shadow_cull_result[j],instance_shadow_cull_result[cull_count]);
					j--;

				}
			}


			print_line("MOMONGO");
			VSG::scene_render->light_instance_set_shadow_transform(light->instance,cm,p_instance->transform,radius,0,0);
			VSG::scene_render->render_shadow(light->instance,p_shadow_atlas,0,(RasterizerScene::InstanceBase**)instance_shadow_cull_result,cull_count);

		} break;
	}

}





void VisualServerScene::render_camera(RID p_camera, RID p_scenario,Size2 p_viewport_size,RID p_shadow_atlas) {


	Camera *camera = camera_owner.getornull(p_camera);
	ERR_FAIL_COND(!camera);

	Scenario *scenario = scenario_owner.getornull(p_scenario);

	render_pass++;
	uint32_t camera_layer_mask=camera->visible_layers;

	VSG::scene_render->set_scene_pass(render_pass);


	/* STEP 1 - SETUP CAMERA */
	CameraMatrix camera_matrix;
	Transform camera_inverse_xform = camera->transform.affine_inverse();
	bool ortho=false;


	switch(camera->type) {
		case Camera::ORTHOGONAL: {

			camera_matrix.set_orthogonal(
				camera->size,
				p_viewport_size.width / (float)p_viewport_size.height,
				camera->znear,
				camera->zfar,
				camera->vaspect

			);
			ortho=true;
		} break;
		case Camera::PERSPECTIVE: {

			camera_matrix.set_perspective(
				camera->fov,
				p_viewport_size.width / (float)p_viewport_size.height,
				camera->znear,
				camera->zfar,
				camera->vaspect

			);
			ortho=false;

		} break;
	}


//	rasterizer->set_camera(camera->transform, camera_matrix,ortho);

	Vector<Plane> planes = camera_matrix.get_projection_planes(camera->transform);

	Plane near_plane(camera->transform.origin,-camera->transform.basis.get_axis(2).normalized());

	/* STEP 2 - CULL */
	int cull_count = scenario->octree.cull_convex(planes,instance_cull_result,MAX_INSTANCE_CULL);
	light_cull_count=0;
//	light_samplers_culled=0;

/*	print_line("OT: "+rtos( (OS::get_singleton()->get_ticks_usec()-t)/1000.0));
	print_line("OTO: "+itos(p_scenario->octree.get_octant_count()));
//	print_line("OTE: "+itos(p_scenario->octree.get_elem_count()));
	print_line("OTP: "+itos(p_scenario->octree.get_pair_count()));
*/

	/* STEP 3 - PROCESS PORTALS, VALIDATE ROOMS */


	// compute portals
#if 0
	exterior_visited=false;
	exterior_portal_cull_count=0;

	if (room_cull_enabled) {
		for(int i=0;i<cull_count;i++) {

			Instance *ins = instance_cull_result[i];
			ins->last_render_pass=render_pass;

			if (ins->base_type!=INSTANCE_PORTAL)
				continue;

			if (ins->room)
				continue;

			ERR_CONTINUE(exterior_portal_cull_count>=MAX_EXTERIOR_PORTALS);
			exterior_portal_cull_result[exterior_portal_cull_count++]=ins;

		}

		room_cull_count = p_scenario->octree.cull_point(camera->transform.origin,room_cull_result,MAX_ROOM_CULL,NULL,(1<<INSTANCE_ROOM)|(1<<INSTANCE_PORTAL));


		Set<Instance*> current_rooms;
		Set<Instance*> portal_rooms;
		//add to set
		for(int i=0;i<room_cull_count;i++) {

			if (room_cull_result[i]->base_type==INSTANCE_ROOM) {
				current_rooms.insert(room_cull_result[i]);
			}
			if (room_cull_result[i]->base_type==INSTANCE_PORTAL) {
				//assume inside that room if also inside the portal..
				if (room_cull_result[i]->room) {
					portal_rooms.insert(room_cull_result[i]->room);
				}

				SWAP(room_cull_result[i],room_cull_result[room_cull_count-1]);
				room_cull_count--;
				i--;
			}
		}

		//remove from set if it has a parent room or BSP doesn't contain
		for(int i=0;i<room_cull_count;i++) {
			Instance *r = room_cull_result[i];

			//check inside BSP
			Vector3 room_local_point = r->room_info->affine_inverse.xform( camera->transform.origin );

			if (!portal_rooms.has(r) && !r->room_info->room->bounds.point_is_inside(room_local_point)) {

				current_rooms.erase(r);
				continue;
			}

			//check parent
			while (r->room) {// has parent room

				current_rooms.erase(r);
				r=r->room;
			}

		}

		if (current_rooms.size()) {
			//camera is inside a room
			// go through rooms
			for(Set<Instance*>::Element *E=current_rooms.front();E;E=E->next()) {
				_cull_room(camera,E->get());
			}

		} else {
			//start from exterior
			_cull_room(camera,NULL);

		}
	}

#endif
	/* STEP 4 - REMOVE FURTHER CULLED OBJECTS, ADD LIGHTS */

	for(int i=0;i<cull_count;i++) {

		Instance *ins = instance_cull_result[i];

		bool keep=false;


		if ((camera_layer_mask&ins->layer_mask)==0) {

			//failure
		} else if (ins->base_type==VS::INSTANCE_LIGHT && ins->visible) {


			if (ins->visible && light_cull_count<MAX_LIGHTS_CULLED) {

				InstanceLightData * light = static_cast<InstanceLightData*>(ins->base_data);

				if (!light->geometries.empty()) {
					//do not add this light if no geometry is affected by it..
					light_cull_result[light_cull_count]=ins;
					light_instance_cull_result[light_cull_count]=light->instance;
					if (p_shadow_atlas.is_valid() && VSG::storage->light_has_shadow(ins->base)) {
						VSG::scene_render->light_instance_mark_visible(light->instance); //mark it visible for shadow allocation later
					}

					light_cull_count++;
				}


			}

		} else if ((1<<ins->base_type)&VS::INSTANCE_GEOMETRY_MASK && ins->visible && ins->cast_shadows!=VS::SHADOW_CASTING_SETTING_SHADOWS_ONLY) {

			keep=true;
#if 0
			bool discarded=false;

			if (ins->draw_range_end>0) {

				float d = cull_range.nearp.distance_to(ins->data.transform.origin);
				if (d<0)
					d=0;
				discarded=(d<ins->draw_range_begin || d>=ins->draw_range_end);


			}

			if (!discarded) {

				// test if this geometry should be visible

				if (room_cull_enabled) {


					if (ins->visible_in_all_rooms) {
						keep=true;
					} else if (ins->room) {

						if (ins->room->room_info->last_visited_pass==render_pass)
							keep=true;
					} else if (ins->auto_rooms.size()) {


						for(Set<Instance*>::Element *E=ins->auto_rooms.front();E;E=E->next()) {

							if (E->get()->room_info->last_visited_pass==render_pass) {
								keep=true;
								break;
							}
						}
					} else if(exterior_visited)
						keep=true;
				} else {

					keep=true;
				}


			}


			if (keep) {
				// update cull range
				float min,max;
				ins->transformed_aabb.project_range_in_plane(cull_range.nearp,min,max);

				if (min<cull_range.min)
					cull_range.min=min;
				if (max>cull_range.max)
					cull_range.max=max;

				if (ins->sampled_light && ins->sampled_light->baked_light_sampler_info->last_pass!=render_pass) {
					if (light_samplers_culled<MAX_LIGHT_SAMPLERS) {
						light_sampler_cull_result[light_samplers_culled++]=ins->sampled_light;
						ins->sampled_light->baked_light_sampler_info->last_pass=render_pass;
					}
				}
			}
#endif


			InstanceGeometryData * geom = static_cast<InstanceGeometryData*>(ins->base_data);

			if (geom->lighting_dirty) {
				int l=0;
				//only called when lights AABB enter/exit this geometry
				ins->light_instances.resize(geom->lighting.size());

				for (List<Instance*>::Element *E=geom->lighting.front();E;E=E->next()) {

					InstanceLightData * light = static_cast<InstanceLightData*>(E->get()->base_data);

					ins->light_instances[l++]=light->instance;
				}

				geom->lighting_dirty=false;
			}

			ins->depth = near_plane.distance_to(ins->transform.origin);

		}

		if (!keep) {
			// remove, no reason to keep
			cull_count--;
			SWAP( instance_cull_result[i], instance_cull_result[ cull_count ] );
			i--;
			ins->last_render_pass=0; // make invalid
		} else {

			ins->last_render_pass=render_pass;
		}
	}

	/* STEP 5 - PROCESS LIGHTS */

	RID *directional_light_ptr=&light_instance_cull_result[light_cull_count];
	int directional_light_count=0;

	// directional lights
	{

		Instance** lights_with_shadow = (Instance**)alloca(sizeof(Instance*)*light_cull_count);
		int directional_shadow_count=0;

		for (List<Instance*>::Element *E=scenario->directional_lights.front();E;E=E->next()) {

			if (light_cull_count+directional_light_count>=MAX_LIGHTS_CULLED) {
				break;
			}

			if (!E->get()->visible)
				continue;

			InstanceLightData * light = static_cast<InstanceLightData*>(E->get()->base_data);


			//check shadow..


			if (light && VSG::storage->light_has_shadow(E->get()->base)) {
				lights_with_shadow[directional_shadow_count++]=E->get();

			}

			//add to list

			directional_light_ptr[directional_light_count++]=light->instance;
		}

		VSG::scene_render->set_directional_shadow_count(directional_shadow_count);

		for(int i=0;i<directional_shadow_count;i++) {

			   _light_instance_update_shadow(lights_with_shadow[i],camera,p_shadow_atlas,scenario,p_viewport_size);

		}
	}


	{ //setup shadow maps

		//SortArray<Instance*,_InstanceLightsort> sorter;
		//sorter.sort(light_cull_result,light_cull_count);
		for (int i=0;i<light_cull_count;i++) {

			Instance *ins = light_cull_result[i];

			if (!p_shadow_atlas.is_valid() || !VSG::storage->light_has_shadow(ins->base))
				continue;

			InstanceLightData * light = static_cast<InstanceLightData*>(ins->base_data);

			float coverage;

			{	//compute coverage


				Transform cam_xf = camera->transform;
				float zn = camera_matrix.get_z_near();
				Plane p (cam_xf.origin + cam_xf.basis.get_axis(2) * -zn, -cam_xf.basis.get_axis(2) ); //camera near plane

				float vp_w,vp_h; //near plane size in screen coordinates
				camera_matrix.get_viewport_size(vp_w,vp_h);


				switch(VSG::storage->light_get_type(ins->base)) {

					case VS::LIGHT_OMNI: {

						float radius = VSG::storage->light_get_param(ins->base,VS::LIGHT_PARAM_RANGE);

						//get two points parallel to near plane
						Vector3 points[2]={
							ins->transform.origin,
							ins->transform.origin+cam_xf.basis.get_axis(0)*radius
						};

						if (!ortho) {
							//if using perspetive, map them to near plane
							for(int j=0;j<2;j++) {
								if (p.distance_to(points[j]) < 0 )	{
									points[j].z=-zn; //small hack to keep size constant when hitting the screen

								}

								p.intersects_segment(cam_xf.origin,points[j],&points[j]); //map to plane
							}


						}

						float screen_diameter = points[0].distance_to(points[1])*2;
						coverage = screen_diameter / (vp_w+vp_h);
					} break;
					case VS::LIGHT_SPOT: {

						float radius = VSG::storage->light_get_param(ins->base,VS::LIGHT_PARAM_RANGE);
						float angle = VSG::storage->light_get_param(ins->base,VS::LIGHT_PARAM_SPOT_ANGLE);


						float w = radius*Math::sin(Math::deg2rad(angle));
						float d = radius*Math::cos(Math::deg2rad(angle));


						Vector3 base = ins->transform.origin-ins->transform.basis.get_axis(2).normalized()*d;

						Vector3 points[2]={
							base,
							base+cam_xf.basis.get_axis(0)*w
						};

						if (!ortho) {
							//if using perspetive, map them to near plane
							for(int j=0;j<2;j++) {
								if (p.distance_to(points[j]) < 0 )	{
									points[j].z=-zn; //small hack to keep size constant when hitting the screen

								}

								p.intersects_segment(cam_xf.origin,points[j],&points[j]); //map to plane
							}


						}

						float screen_diameter = points[0].distance_to(points[1])*2;
						coverage = screen_diameter / (vp_w+vp_h);


					} break;
					default: {
						ERR_PRINT("Invalid Light Type");
					}
				}

			}


			if (light->shadow_dirty) {
				light->last_version++;
				light->shadow_dirty=false;
			}



			bool redraw = VSG::scene_render->shadow_atlas_update_light(p_shadow_atlas,light->instance,coverage,light->last_version);

			if (redraw) {
				//must redraw!
				_light_instance_update_shadow(ins,camera,p_shadow_atlas,scenario,p_viewport_size);
			}

		}
	}

	/* ENVIRONMENT */

	RID environment;
	if (camera->env.is_valid()) //camera has more environment priority
		environment=camera->env;
	else if (scenario->environment.is_valid())
		environment=scenario->environment;
	else
		environment=scenario->fallback_environment;

#if 0
	/* STEP 6 - SAMPLE BAKED LIGHT */

	bool islinear =false;
	if (environment.is_valid()) {
		islinear = rasterizer->environment_is_fx_enabled(environment,VS::ENV_FX_SRGB);
	}

	for(int i=0;i<light_samplers_culled;i++) {

		_process_sampled_light(camera->transform,light_sampler_cull_result[i],islinear);
	}
#endif
	/* STEP 7 - PROCESS GEOMETRY AND DRAW SCENE*/

#if 0
	// add lights

	{
		List<RID>::Element *E=p_scenario->directional_lights.front();


		for(;E;E=E->next()) {
			Instance  *light = E->get().is_valid()?instance_owner.get(E->get()):NULL;

			ERR_CONTINUE(!light);
			if (!light->light_info->enabled)
				continue;

			rasterizer->add_light(light->light_info->instance);
			light->light_info->last_add_pass=render_pass;
		}

		for (int i=0;i<light_cull_count;i++) {

			Instance *ins = light_cull_result[i];
			rasterizer->add_light(ins->light_info->instance);
			ins->light_info->last_add_pass=render_pass;
		}
	}
		// add geometry
#endif



	VSG::scene_render->render_scene(camera->transform, camera_matrix,ortho,(RasterizerScene::InstanceBase**)instance_cull_result,cull_count,light_instance_cull_result,light_cull_count+directional_light_count,environment,p_shadow_atlas);


}



void VisualServerScene::_update_dirty_instance(Instance *p_instance) {


	if (p_instance->update_aabb)
		_update_instance_aabb(p_instance);

	if (p_instance->update_materials) {

		if (p_instance->base_type==VS::INSTANCE_MESH) {
			//remove materials no longer used and un-own them

			int new_mat_count = VSG::storage->mesh_get_surface_count(p_instance->base);
			for(int i=p_instance->materials.size()-1;i>=new_mat_count;i--) {
				if (p_instance->materials[i].is_valid()) {
					VSG::storage->material_remove_instance_owner(p_instance->materials[i],p_instance);
				}
			}
			p_instance->materials.resize(new_mat_count);
		}

		if ((1<<p_instance->base_type)&VS::INSTANCE_GEOMETRY_MASK) {

			InstanceGeometryData *geom = static_cast<InstanceGeometryData*>(p_instance->base_data);

			bool can_cast_shadows=true;

			if (p_instance->cast_shadows==VS::SHADOW_CASTING_SETTING_OFF) {
				can_cast_shadows=false;
			} else if (p_instance->material_override.is_valid()) {
				can_cast_shadows=VSG::storage->material_casts_shadows(p_instance->material_override);
			} else {

				RID mesh;

				if (p_instance->base_type==VS::INSTANCE_MESH) {
					mesh=p_instance->base;
				} else if (p_instance->base_type==VS::INSTANCE_MULTIMESH) {

				}

				if (mesh.is_valid()) {

					bool cast_shadows=false;

					for(int i=0;i<p_instance->materials.size();i++) {


						RID mat = p_instance->materials[i].is_valid()?p_instance->materials[i]:VSG::storage->mesh_surface_get_material(mesh,i);

						if (!mat.is_valid()) {
							cast_shadows=true;
							break;
						}

						if (VSG::storage->material_casts_shadows(mat)) {
							cast_shadows=true;
							break;
						}
					}

					if (!cast_shadows) {
						can_cast_shadows=false;
					}
				}

			}

			if (can_cast_shadows!=geom->can_cast_shadows) {
				//ability to cast shadows change, let lights now
				for (List<Instance*>::Element *E=geom->lighting.front();E;E=E->next()) {
					InstanceLightData *light = static_cast<InstanceLightData*>(E->get()->base_data);
					light->shadow_dirty=true;
				}

				geom->can_cast_shadows=can_cast_shadows;
			}
		}

	}

	_update_instance(p_instance);

	p_instance->update_aabb=false;
	p_instance->update_materials=false;

	_instance_update_list.remove( &p_instance->update_item );
}


void VisualServerScene::update_dirty_instances() {

	while(_instance_update_list.first()) {

		_update_dirty_instance( _instance_update_list.first()->self() );
	}
}

bool VisualServerScene::free(RID p_rid) {

	if (camera_owner.owns(p_rid)) {

		Camera *camera = camera_owner.get( p_rid );

		camera_owner.free(p_rid);
		memdelete(camera);

	} else if (scenario_owner.owns(p_rid)) {

		Scenario *scenario = scenario_owner.get( p_rid );

		while(scenario->instances.first()) {
			instance_set_scenario(scenario->instances.first()->self()->self,RID());
		}

		scenario_owner.free(p_rid);
		memdelete(scenario);

	} else if (instance_owner.owns(p_rid)) {
		// delete the instance

		update_dirty_instances();

		Instance *instance = instance_owner.get(p_rid);

		instance_set_room(p_rid,RID());
		instance_set_scenario(p_rid,RID());
		instance_set_base(p_rid,RID());
		instance_geometry_set_material_override(p_rid,RID());

		if (instance->skeleton.is_valid())
			instance_attach_skeleton(p_rid,RID());

		instance_owner.free(p_rid);
		memdelete(instance);
	} else {
		return false;
	}


	return true;
}

VisualServerScene *VisualServerScene::singleton=NULL;

VisualServerScene::VisualServerScene() {


	render_pass=1;
	singleton=this;

}
