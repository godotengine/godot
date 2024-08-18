#ifndef MGRASSCHUNK
#define MGRASSCHUNK

#include "servers/rendering_server.h"

struct MGrassChunk // Rendering server multi mesh data
{
    RID multimesh;
    RID instance;
    MGrassChunk* next=nullptr;
    Vector3 world_pos;
    int count=0;
    int total_count=0;
    int lod;
    int region_id;
    MPixelRegion pixel_region;
    bool is_relax=false;
    bool is_part_of_scene=false;
    bool is_out_of_range=false;

    MGrassChunk(const MPixelRegion& _pixel_region,Vector3 _world_pos, int _lod,int _region_id){
        pixel_region = _pixel_region;
        lod = _lod;
        region_id = _region_id;
        world_pos = _world_pos;
    }
    MGrassChunk(){}
    ~MGrassChunk(){
        if(count!=0){
            RenderingServer::get_singleton()->free(multimesh);
            RenderingServer::get_singleton()->free(instance);
        }
        if(next!=nullptr){
            memdelete(next);
        }
    }
    void relax(){
        if(count!=0 && !is_relax){
            RenderingServer::get_singleton()->instance_set_visible(instance,false);
            is_relax = true;
        }
        if(next!=nullptr){
            next->relax();
        }
    }
    void unrelax(){
        if(count!=0 && is_relax){
            RenderingServer::get_singleton()->instance_set_visible(instance,true);
            is_relax = false;
        }
        if(next!=nullptr){
            next->unrelax();
        }
    }
    void set_shadow_setting(RenderingServer::ShadowCastingSetting setting){
        if(count!=0){
            RenderingServer::get_singleton()->instance_geometry_set_cast_shadows_setting(instance,setting);
        }
        if(next!=nullptr){
            next->set_shadow_setting(setting);
        }
    }
    
    void set_gi_mode(GeometryInstance3D::GIMode p_mode){
        if(count!=0){
            switch (p_mode) {
                case GeometryInstance3D::GI_MODE_DISABLED: {
                    RenderingServer::get_singleton()->instance_geometry_set_flag(instance, RenderingServer::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
                    RenderingServer::get_singleton()->instance_geometry_set_flag(instance, RenderingServer::INSTANCE_FLAG_USE_DYNAMIC_GI, false);
                } break;
                case GeometryInstance3D::GI_MODE_STATIC: {
                    RenderingServer::get_singleton()->instance_geometry_set_flag(instance, RenderingServer::INSTANCE_FLAG_USE_BAKED_LIGHT, true);
                    RenderingServer::get_singleton()->instance_geometry_set_flag(instance, RenderingServer::INSTANCE_FLAG_USE_DYNAMIC_GI, false);

                } break;
                case GeometryInstance3D::GI_MODE_DYNAMIC: {
                    RenderingServer::get_singleton()->instance_geometry_set_flag(instance, RenderingServer::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
                    RenderingServer::get_singleton()->instance_geometry_set_flag(instance, RenderingServer::INSTANCE_FLAG_USE_DYNAMIC_GI, true);
                } break;
            }
        }
        if(next!=nullptr){
            next->set_gi_mode(p_mode);
        }
    }

    void clear_tree(){
        if(next!=nullptr){
            memdelete(next);
            next=nullptr;
        }
    }
    void set_buffer(int _count,RID scenario, RID mesh_rid, RID material ,const PackedFloat32Array& data,bool has_color_data=false, bool has_custom_data=false){
        //UtilityFunctions::print("Buffer count ",_count, " c ", count);
        if(_count!=0 && count == 0){ //creating
            multimesh = RenderingServer::get_singleton()->multimesh_create();
            RenderingServer::get_singleton()->multimesh_set_mesh(multimesh, mesh_rid);
            instance = RenderingServer::get_singleton()->instance_create();
            RenderingServer::get_singleton()->instance_set_visible(instance,false);
            is_relax = true;
            RenderingServer::get_singleton()->instance_set_base(instance, multimesh);
            RenderingServer::get_singleton()->instance_geometry_set_material_override(instance,material);
            RenderingServer::get_singleton()->instance_set_scenario(instance,scenario);
        } else if(_count==0 && count!=0){ //destroying
            RenderingServer::get_singleton()->free(multimesh);
            RenderingServer::get_singleton()->free(instance);
            instance = RID();
            multimesh = RID();
            count = 0;
            return;
        } else if(_count==0 && count==0){
            return;
        } else if(count==_count){ // Can be Mesh or material update
            RenderingServer::get_singleton()->multimesh_set_mesh(multimesh, mesh_rid);
            RenderingServer::get_singleton()->instance_geometry_set_material_override(instance,material);
        }
        count = _count;
        RenderingServer::get_singleton()->multimesh_allocate_data(multimesh, _count, RenderingServer::MULTIMESH_TRANSFORM_3D, has_color_data, has_custom_data);
        RenderingServer::get_singleton()->multimesh_set_buffer(multimesh, data);
    }
};


#endif