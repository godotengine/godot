/*************************************************************************/
/*  state_machine.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "state_machine.h"
#include "servers/visual_server.h"
#include "scene/resources/surface_tool.h"
#include "message_queue.h"
#include "scene/3d/light.h"
#include "io/marshalls.h"


bool StateMachine::_set(const StringName& p_name, const Variant& p_value) {

	String name=p_name;

	return true;
	/*

	if (name=="theme/theme") {

		set_theme(p_value);
	} else if (name=="cell/size") {
		set_cell_size(p_value);
	} else if (name=="cell/octant_size") {
		set_octant_size(p_value);
	} else if (name=="cell/center_x") {
		set_center_x(p_value);
	} else if (name=="cell/center_y") {
		set_center_y(p_value);
	} else if (name=="cell/center_z") {
		set_center_z(p_value);
	} else if (name=="cell/scale") {
		set_cell_scale(p_value);
	} else if (name=="theme/bake") {
		set_bake(p_value);

	} else if (name=="data") {

		Dictionary d = p_value;

		Dictionary baked;
		if (d.has("baked"))
			baked=d["baked"];
		if (d.has("cells")) {

			DVector<int> cells = d["cells"];
			int amount=cells.size();
			DVector<int>::Read r = cells.read();
			ERR_FAIL_COND_V(amount%3,false); // not even
			cell_map.clear();;
			for(int i=0;i<amount/3;i++) {

				IndexKey ik;
				ik.key=decode_uint64((const uint8_t*)&r[i*3]);
				Cell cell;
				cell.cell=decode_uint32((const uint8_t*)&r[i*3+2]);
				cell_map[ik]=cell;

			}
		}
		baked_lock=baked.size()!=0;
		_recreate_octant_data();
		baked_lock=false;
		if (!baked.empty()) {
			List<Variant> kl;
			baked.get_key_list(&kl);
			for (List<Variant>::Element *E=kl.front();E;E=E->next()) {

				Plane ikv = E->get();
				Ref<Mesh> b=baked[ikv];
				ERR_CONTINUE(!b.is_valid());
				OctantKey ok;
				ok.x=ikv.normal.x;
				ok.y=ikv.normal.y;
				ok.z=ikv.normal.z;
				ok.area=ikv.d;

				ERR_CONTINUE(!octant_map.has(ok));

				Octant &g = *octant_map[ok];

				g.baked=b;
				g.bake_instance=VS::get_singleton()->instance_create();;
				VS::get_singleton()->instance_set_base(g.bake_instance,g.baked->get_rid());
			}
		}


	} else if (name.begins_with("areas/")) {
		int which = name.get_slice("/",1).to_int();
		String what=name.get_slice("/",2);
		if (what=="bounds") {
			ERR_FAIL_COND_V(area_map.has(which),false);
			create_area(which,p_value);
			return true;
		}

		ERR_FAIL_COND_V(!area_map.has(which),false);

		if (what=="name")
			area_set_name(which,p_value);
		else if (what=="disable_distance")
			area_set_portal_disable_distance(which,p_value);
		else if (what=="exterior_portal")
			area_set_portal_disable_color(which,p_value);
		else
			return false;
	} else
		return false;

	return true;
	*/

}

bool StateMachine::_get(const StringName& p_name,Variant &r_ret) const {

	String name=p_name;

    /*
	if (name=="theme/theme") {
		r_ret= get_theme();
	} else if (name=="cell/size") {
		r_ret= get_cell_size();
	} else if (name=="cell/octant_size") {
		r_ret= get_octant_size();
	} else if (name=="cell/center_x") {
		r_ret= get_center_x();
	} else if (name=="cell/center_y") {
		r_ret= get_center_y();
	} else if (name=="cell/center_z") {
		r_ret= get_center_z();
	} else if (name=="cell/scale") {
		r_ret= cell_scale;
	} else if (name=="theme/bake") {
		r_ret= bake;
	} else if (name=="data") {

		Dictionary d;

		DVector<int> cells;
		cells.resize(cell_map.size()*3);
		{
			DVector<int>::Write w = cells.write();
			int i=0;
			for (Map<IndexKey,Cell>::Element *E=cell_map.front();E;E=E->next(),i++) {

				encode_uint64(E->key().key,(uint8_t*)&w[i*3]);
				encode_uint32(E->get().cell,(uint8_t*)&w[i*3+2]);
			}
		}

		d["cells"]=cells;

		Dictionary baked;
		for(Map<OctantKey,Octant*>::Element *E=octant_map.front();E;E=E->next()) {

			Octant &g=*E->get();

			if (g.baked.is_valid()) {

				baked[Plane(E->key().x,E->key().y,E->key().z,E->key().area)]=g.baked;
			}


		}

		if (baked.size()) {
			d["baked"]=baked;
		}

		r_ret= d;
	} else if (name.begins_with("areas/")) {
		int which = name.get_slice("/",1).to_int();
		String what=name.get_slice("/",2);
		if (what=="bounds")
			r_ret= area_get_bounds(which);
		else if (what=="name")
			r_ret= area_get_name(which);
		else if (what=="disable_distance")
			r_ret= area_get_portal_disable_distance(which);
		else if (what=="exterior_portal")
			r_ret= area_is_exterior_portal(which);
		else
			return false;
	} else
		return false;
*/
	return true;

}

void StateMachine::_get_property_list( List<PropertyInfo> *p_list) const {
/*
	p_list->push_back( PropertyInfo( Variant::OBJECT, "theme/theme", PROPERTY_HINT_RESOURCE_TYPE, "MeshLibrary"));
	p_list->push_back( PropertyInfo( Variant::BOOL, "theme/bake"));
	p_list->push_back( PropertyInfo( Variant::REAL, "cell/size",PROPERTY_HINT_RANGE,"0.01,16384,0.01") );
	p_list->push_back( PropertyInfo( Variant::INT, "cell/octant_size",PROPERTY_HINT_RANGE,"1,1024,1") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "cell/center_x") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "cell/center_y") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "cell/center_z") );
	p_list->push_back( PropertyInfo( Variant::REAL, "cell/scale") );

	p_list->push_back( PropertyInfo( Variant::DICTIONARY, "data", PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE) );

	for(const Map<int,Area*>::Element *E=area_map.front();E;E=E->next()) {

		String base="areas/"+itos(E->key())+"/";
		p_list->push_back( PropertyInfo( Variant::_AABB, base+"bounds", PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE) );
		p_list->push_back( PropertyInfo( Variant::STRING, base+"name", PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE) );
		p_list->push_back( PropertyInfo( Variant::REAL, base+"disable_distance", PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE) );
		p_list->push_back( PropertyInfo( Variant::COLOR, base+"disable_color", PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE) );
		p_list->push_back( PropertyInfo( Variant::BOOL, base+"exterior_portal", PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE) );
    }*/
}

void StateMachine::_notification(int p_what) {


	switch(p_what) {
		/*
		case NOTIFICATION_ENTER_WORLD: {

			_update_area_instances();

			for(Map<OctantKey,Octant*>::Element *E=octant_map.front();E;E=E->next()) {
//				IndexKey ik;
//				ik.key = E->key().indexkey;
				_octant_enter_world(E->key());
				_octant_update(E->key());
			}

			awaiting_update=false;

            last_transform=get_global_transform();
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			Transform new_xform = get_global_transform();
			if (new_xform==last_transform)
				break;
			//update run
			for(Map<OctantKey,Octant*>::Element *E=octant_map.front();E;E=E->next()) {
				_octant_transform(E->key());
			}

            last_transform=new_xform;

		} break;
		case NOTIFICATION_EXIT_WORLD: {

			for(Map<OctantKey,Octant*>::Element *E=octant_map.front();E;E=E->next()) {
				_octant_exit_world(E->key());
			}

			//_queue_dirty_map(MAP_DIRTY_INSTANCES|MAP_DIRTY_TRANSFORMS);
			//_update_dirty_map_callback();
			//_update_area_instances();

		} break;
*/	}
}




void StateMachine::resource_changed(const RES& p_res) {

	//_recreate_octant_data();
}

/*
void StateMachine::_update_dirty_map_callback() {

	if (!awaiting_update)
		return;

	for(Map<OctantKey,Octant*>::Element *E=octant_map.front();E;E=E->next()) {
		_octant_update(E->key());
	}


	awaiting_update=false;

}*/


void StateMachine::_bind_methods() {
/*
	ObjectTypeDB::bind_method(_MD("set_theme","theme:MeshLibrary"),&StateMachine::set_theme);
	ObjectTypeDB::bind_method(_MD("get_theme:MeshLibrary"),&StateMachine::get_theme);

	ObjectTypeDB::bind_method(_MD("set_bake","enable"),&StateMachine::set_bake);
	ObjectTypeDB::bind_method(_MD("is_baking_enabled"),&StateMachine::is_baking_enabled);

	ObjectTypeDB::bind_method(_MD("set_cell_size","size"),&StateMachine::set_cell_size);
	ObjectTypeDB::bind_method(_MD("get_cell_size"),&StateMachine::get_cell_size);

	ObjectTypeDB::bind_method(_MD("set_octant_size","size"),&StateMachine::set_octant_size);
	ObjectTypeDB::bind_method(_MD("get_octant_size"),&StateMachine::get_octant_size);

	ObjectTypeDB::bind_method(_MD("set_cell_item","x","y","z","item","orientation"),&StateMachine::set_cell_item,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("get_cell_item","x","y","z"),&StateMachine::get_cell_item);
	ObjectTypeDB::bind_method(_MD("get_cell_item_orientation","x","y","z"),&StateMachine::get_cell_item_orientation);

//	ObjectTypeDB::bind_method(_MD("_recreate_octants"),&StateMachine::_recreate_octants);
	ObjectTypeDB::bind_method(_MD("_update_dirty_map_callback"),&StateMachine::_update_dirty_map_callback);
	ObjectTypeDB::bind_method(_MD("resource_changed"),&StateMachine::resource_changed);

	ObjectTypeDB::bind_method(_MD("set_center_x","enable"),&StateMachine::set_center_x);
	ObjectTypeDB::bind_method(_MD("get_center_x"),&StateMachine::get_center_x);
	ObjectTypeDB::bind_method(_MD("set_center_y","enable"),&StateMachine::set_center_y);
	ObjectTypeDB::bind_method(_MD("get_center_y"),&StateMachine::get_center_y);
	ObjectTypeDB::bind_method(_MD("set_center_z","enable"),&StateMachine::set_center_z);
	ObjectTypeDB::bind_method(_MD("get_center_z"),&StateMachine::get_center_z);

	ObjectTypeDB::bind_method(_MD("set_clip","enabled","clipabove","floor","axis"),&StateMachine::set_clip,DEFVAL(true),DEFVAL(0),DEFVAL(Vector3::AXIS_X));

	ObjectTypeDB::bind_method(_MD("crate_area","id","area"),&StateMachine::create_area);
	ObjectTypeDB::bind_method(_MD("area_get_bounds","area","bounds"),&StateMachine::area_get_bounds);
	ObjectTypeDB::bind_method(_MD("area_set_exterior_portal","area","enable"),&StateMachine::area_set_exterior_portal);
	ObjectTypeDB::bind_method(_MD("area_set_name","area","name"),&StateMachine::area_set_name);
	ObjectTypeDB::bind_method(_MD("area_get_name","area"),&StateMachine::area_get_name);
	ObjectTypeDB::bind_method(_MD("area_is_exterior_portal","area"),&StateMachine::area_is_exterior_portal);
	ObjectTypeDB::bind_method(_MD("area_set_portal_disable_distance","area","distance"),&StateMachine::area_set_portal_disable_distance);
	ObjectTypeDB::bind_method(_MD("area_get_portal_disable_distance","area"),&StateMachine::area_get_portal_disable_distance);
	ObjectTypeDB::bind_method(_MD("area_set_portal_disable_color","area","color"),&StateMachine::area_set_portal_disable_color);
	ObjectTypeDB::bind_method(_MD("area_get_portal_disable_color","area"),&StateMachine::area_get_portal_disable_color);
	ObjectTypeDB::bind_method(_MD("erase_area","area"),&StateMachine::erase_area);
	ObjectTypeDB::bind_method(_MD("get_unused_area_id","area"),&StateMachine::get_unused_area_id);
	ObjectTypeDB::bind_method(_MD("bake_geometry"),&StateMachine::bake_geometry);

	ObjectTypeDB::set_method_flags("StateMachine","bake_geometry",METHOD_FLAGS_DEFAULT|METHOD_FLAG_EDITOR);

	ObjectTypeDB::bind_method(_MD("clear"),&StateMachine::clear);

	BIND_CONSTANT( INVALID_CELL_ITEM );
*/
}






StateMachine::StateMachine() {

    /*cell_size=2;
	octant_size=4;
	awaiting_update=false;
	_in_tree=false;
	center_x=true;
	center_y=true;
	center_z=true;

	clip=false;
	clip_floor=0;
	clip_axis=Vector3::AXIS_Z;
	clip_above=true;
	baked_lock=false;
	bake=false;
	cell_scale=1.0;
*/



}


StateMachine::~StateMachine() {

    //clear();

}
