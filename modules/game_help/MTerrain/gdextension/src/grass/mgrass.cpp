#include "mgrass.h"
#include "../mgrid.h"
#include "../mterrain.h"
#include "core/config/project_settings.h"

#include "core/io/resource_saver.h"

#define CHUNK_INFO grid->grid_update_info[grid_index]
#define PS PhysicsServer3D::get_singleton()


float MGrass::time_rollover_secs = 3600;
Vector<MGrass*> MGrass::all_grass_nodes;


void MGrass::_bind_methods() {
    ADD_SIGNAL(MethodInfo("grass_is_ready"));
    ClassDB::bind_method(D_METHOD("is_init"), &MGrass::is_init);
    ClassDB::bind_method(D_METHOD("has_sublayer"), &MGrass::has_sublayer);
    ClassDB::bind_method(D_METHOD("merge_sublayer"), &MGrass::merge_sublayer);
    ClassDB::bind_method(D_METHOD("make_grass_dirty_by_pixel","x","y"), &MGrass::make_grass_dirty_by_pixel);
    ClassDB::bind_method(D_METHOD("set_grass_by_pixel","x","y","val"), &MGrass::set_grass_by_pixel);
    ClassDB::bind_method(D_METHOD("get_grass_by_pixel","x","y"), &MGrass::get_grass_by_pixel);
    ClassDB::bind_method(D_METHOD("update_dirty_chunks"), &MGrass::update_dirty_chunks_gd);
    ClassDB::bind_method(D_METHOD("recreate_all_grass"), &MGrass::_recreate_all_grass);
    ClassDB::bind_method(D_METHOD("draw_grass","brush_pos","radius","add"), &MGrass::draw_grass);
    ClassDB::bind_method(D_METHOD("get_count"), &MGrass::get_count);
    ClassDB::bind_method(D_METHOD("get_width"), &MGrass::get_width);
    ClassDB::bind_method(D_METHOD("get_height"), &MGrass::get_height);
    ClassDB::bind_method(D_METHOD("grass_px_to_grid_px","x","y"), &MGrass::grass_px_to_grid_px);
    ClassDB::bind_method(D_METHOD("get_closest_pixel","world_pos"), &MGrass::get_closest_pixel);

    ClassDB::bind_method(D_METHOD("set_active","input"), &MGrass::set_active);
    ClassDB::bind_method(D_METHOD("get_active"), &MGrass::get_active);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"active"),"set_active","get_active");
    ClassDB::bind_method(D_METHOD("set_grass_data","input"), &MGrass::set_grass_data);
    ClassDB::bind_method(D_METHOD("get_grass_data"), &MGrass::get_grass_data);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"grass_data",PROPERTY_HINT_RESOURCE_TYPE,"MGrassData"),"set_grass_data","get_grass_data");
    ClassDB::bind_method(D_METHOD("set_cell_creation_time_data_limit","input"), &MGrass::set_cell_creation_time_data_limit);
    ClassDB::bind_method(D_METHOD("get_cell_creation_time_data_limit"), &MGrass::get_cell_creation_time_data_limit);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"cell_creation_time_data_limit"),"set_cell_creation_time_data_limit","get_cell_creation_time_data_limit");
    ClassDB::bind_method(D_METHOD("set_grass_count_limit","input"), &MGrass::set_grass_count_limit);
    ClassDB::bind_method(D_METHOD("get_grass_count_limit"), &MGrass::get_grass_count_limit);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"grass_count_limit"),"set_grass_count_limit","get_grass_count_limit");
    ClassDB::bind_method(D_METHOD("set_min_grass_cutoff","input"), &MGrass::set_min_grass_cutoff);
    ClassDB::bind_method(D_METHOD("get_min_grass_cutoff"), &MGrass::get_min_grass_cutoff);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "min_grass_cutoff"),"set_min_grass_cutoff","get_min_grass_cutoff");
    ClassDB::bind_method(D_METHOD("set_collision_radius","input"), &MGrass::set_collision_radius);
    ClassDB::bind_method(D_METHOD("get_collision_radius"), &MGrass::get_collision_radius);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"collision_radius"), "set_collision_radius","get_collision_radius");
    ClassDB::bind_method(D_METHOD("set_shape_offset","input"), &MGrass::set_shape_offset);
    ClassDB::bind_method(D_METHOD("get_shape_offset"), &MGrass::get_shape_offset);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,"shape_offset"), "set_shape_offset","get_shape_offset");
    ClassDB::bind_method(D_METHOD("set_shape","input"), &MGrass::set_shape);
    ClassDB::bind_method(D_METHOD("get_shape"), &MGrass::get_shape);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"shape",PROPERTY_HINT_RESOURCE_TYPE,"Shape3D"),"set_shape","get_shape");

    ClassDB::bind_method(D_METHOD("set_physics_material","input"), &MGrass::set_physics_material);
    ClassDB::bind_method(D_METHOD("get_physics_material"), &MGrass::get_physics_material);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"physics_material",PROPERTY_HINT_RESOURCE_TYPE,"PhysicsMaterial"),"set_physics_material","get_physics_material");

    ClassDB::bind_method(D_METHOD("get_collision_layer"), &MGrass::get_collision_layer);
    ClassDB::bind_method(D_METHOD("set_collision_layer","input"), &MGrass::set_collision_layer);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"collision_layer",PROPERTY_HINT_LAYERS_3D_PHYSICS),"set_collision_layer","get_collision_layer");

    ClassDB::bind_method(D_METHOD("set_collision_mask","input"), &MGrass::set_collision_mask);
    ClassDB::bind_method(D_METHOD("get_collision_mask"), &MGrass::get_collision_mask);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"collision_mask",PROPERTY_HINT_LAYERS_3D_PHYSICS),"set_collision_mask","get_collision_mask");


    ClassDB::bind_method(D_METHOD("set_active_shape_resize","input"), &MGrass::set_active_shape_resize);
    ClassDB::bind_method(D_METHOD("get_active_shape_resize"), &MGrass::get_active_shape_resize);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"active_shape_resize"),"set_active_shape_resize","get_active_shape_resize");
    ClassDB::bind_method(D_METHOD("set_nav_obstacle_radius","input"), &MGrass::set_nav_obstacle_radius);
    ClassDB::bind_method(D_METHOD("get_nav_obstacle_radius"), &MGrass::get_nav_obstacle_radius);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"nav_obstacle_radius"),"set_nav_obstacle_radius","get_nav_obstacle_radius");
    ClassDB::bind_method(D_METHOD("set_lod_settings","input"), &MGrass::set_lod_settings);
    ClassDB::bind_method(D_METHOD("get_lod_settings"), &MGrass::get_lod_settings);
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY,"lod_settings",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE), "set_lod_settings","get_lod_settings");
    ClassDB::bind_method(D_METHOD("set_meshes","input"), &MGrass::set_meshes);
    ClassDB::bind_method(D_METHOD("get_meshes"), &MGrass::get_meshes);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"meshes",PROPERTY_HINT_RESOURCE_TYPE,"MMeshLod"),"set_meshes","get_meshes");
    ClassDB::bind_method(D_METHOD("set_materials"), &MGrass::set_materials);
    ClassDB::bind_method(D_METHOD("get_materials"), &MGrass::get_materials);
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY,"materials",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE),"set_materials","get_materials");

    ClassDB::bind_method(D_METHOD("save_grass_data"), &MGrass::save_grass_data);

    ClassDB::bind_method(D_METHOD("check_undo"), &MGrass::check_undo);
    ClassDB::bind_method(D_METHOD("undo"), &MGrass::undo);
    ClassDB::bind_method(D_METHOD("_lod_setting_changed"), &MGrass::_lod_setting_changed);

    ClassDB::bind_method(D_METHOD("_update_visibilty"), &MGrass::_update_visibilty);

    ClassDB::bind_static_method("MGrass",D_METHOD("get_all_grass_nodes"),&MGrass::get_all_grass_nodes);
}

TypedArray<MGrass> MGrass::get_all_grass_nodes(){
    TypedArray<MGrass> out;
    for(MGrass* g : all_grass_nodes){
        if(g->is_inside_tree()){
            out.push_back(g);
        }
    }
    return out;
}

MGrass::MGrass(){
    dirty_points_id = memnew(VSet<int>);
    connect("tree_exited",Callable(this,"_update_visibilty"));
    connect("tree_entered",Callable(this,"_update_visibilty"));
    all_grass_nodes.push_back(this);
}
MGrass::~MGrass(){
    clear_grass();
    memdelete(dirty_points_id);
    for(int i=0; i < all_grass_nodes.size();i++){
        if(this==all_grass_nodes[i]){
            all_grass_nodes.remove_at(i);
            break;
        }
    }
}

void MGrass::init_grass(MGrid* _grid) {
    ERR_FAIL_COND_EDMSG(!grass_data.is_valid(),"Grass \""+get_name()+"\" Data is invalid, Please set grass data and save that with .res ext");
    if(!active){
        return;
    }
    time_rollover_secs = ProjectSettings::get_singleton()->get("rendering/limits/time/time_rollover_secs");
    grid = _grid;
    scenario = grid->get_scenario();
    space = grid->space;
    region_grid_width = grid->get_region_grid_size().x;
    grass_region_pixel_width = (uint32_t)round((float)grid->region_size_meter/grass_data->density);
    grass_region_pixel_size = grass_region_pixel_width*grass_region_pixel_width;
    base_grid_size_in_pixel = (uint32_t)round((double)grass_region_pixel_width/(double)grid->region_size);
    width = grass_region_pixel_width*grid->get_region_grid_size().x;
    height = grass_region_pixel_width*grid->get_region_grid_size().z;
    grass_pixel_region.left=0;
    grass_pixel_region.top=0;
    grass_pixel_region.right = width - 1;
    grass_pixel_region.bottom = height - 1;
    grass_bound_limit.left = grass_pixel_region.left;
    grass_bound_limit.right = grass_pixel_region.right;
    grass_bound_limit.top = grass_pixel_region.top;
    grass_bound_limit.bottom = grass_pixel_region.bottom;
    int64_t data_size = ((grass_region_pixel_size*grid->get_regions_count() - 1)/8) + 1;
    if(grass_data->data.size()==0){
        // grass data is empty so we create grass data here
        grass_data->data.resize(data_size);
    } else {
        // Data already created so we check if the data size is correct
        ERR_FAIL_COND_EDMSG(grass_data->data.size()!=data_size,"Grass \""+get_name()+"\" data not match, Some Terrain setting and grass density should not change after grass data creation, change back setting or create a new grass data");
    }
    for(int i=0;i<materials.size();i++){
        Ref<Material> m = materials[i];
        if(m.is_valid()){
            material_rids.push_back(m->get_rid());
        } else {
            material_rids.push_back(RID());
        }
    }
    // Rand num Generation
    default_lod_setting = ResourceLoader::load("res://addons/m_terrain/default_lod_setting.res");
    for(int i=0;i<lod_settings.size();i++){
        Ref<MGrassLodSetting> s = lod_settings[i];
        if(s.is_valid()){
            settings.push_back(s);
        } else {
            settings.push_back(default_lod_setting);
        }
    }
    for(int i=0;i<settings.size();i++){
        if(!settings[i]->is_connected("lod_setting_changed",Callable(this,"_lod_setting_changed")))
        {
            settings[i]->connect("lod_setting_changed",Callable(this,"_lod_setting_changed"));
        }
        int lod_scale = pow(2,i);
        if(settings[i]->grid_lod >=0){
            lod_scale = pow(2,settings[i]->grid_lod);
        }
        float cdensity = grass_data->density*lod_scale;
        rand_buffer_pull.push_back(settings[i]->generate_random_number(cdensity,100));
        set_lod_setting_image_index(settings[i]);
    }
    // Done
    is_grass_init = true;
    update_grass();
    apply_update_grass();
    //Creating Main Physic Body
    //Setting the shape data
    if(shape.is_valid()){
        main_physics_body = PhysicsServer3D::get_singleton()->body_create();
        PS->body_set_mode(main_physics_body,PhysicsServer3D::BODY_MODE_STATIC);
        PS->body_set_space(main_physics_body,space);
        shape_type = PS->shape_get_type(shape->get_rid());
        shape_data = PS->shape_get_data(shape->get_rid());
        PS->body_set_collision_layer(main_physics_body,collision_layer);
        PS->body_set_collision_mask(main_physics_body,collision_mask);
        if(physics_material.is_valid()){
            float friction = physics_material->is_rough() ? - physics_material->get_friction() : physics_material->get_friction();
            float bounce = physics_material->is_absorbent() ? - physics_material->get_bounce() : physics_material->get_bounce();
            PS->body_set_param(main_physics_body,PhysicsServer3D::BODY_PARAM_BOUNCE,bounce);
            PS->body_set_param(main_physics_body,PhysicsServer3D::BODY_PARAM_FRICTION,friction);
        }
    }
    emit_signal("grass_is_ready");
}

void MGrass::clear_grass(){
    if(!is_grass_init){
        return;
    }
    std::lock_guard<std::mutex> lock(update_mutex);
    for(HashMap<int64_t,MGrassChunk*>::Iterator it = grid_to_grass.begin();it!=grid_to_grass.end();++it){
        memdelete(it->value);
    }
    settings.clear();
    rand_buffer_pull.clear();
    grid_to_grass.clear();
    is_grass_init = false;
    final_count = 0;
    if(main_physics_body.is_valid()){
        remove_all_physics();
        PS->body_clear_shapes(main_physics_body);
        PS->free(main_physics_body);
        main_physics_body = RID();
    }
    shapes_ids.clear();
    for(HashMap<int,RID>::Iterator it=shapes_rids.begin();it!=shapes_rids.end();++it){
        PS->free(it->value);
    }
    shapes_rids.clear();
    to_be_visible.clear();
    cell_creation.clear();
    cell_creation_order.clear();
}

void MGrass::update_dirty_chunks_gd(){
    update_dirty_chunks(true);
}

void MGrass::update_dirty_chunks(bool update_lock){
    ERR_FAIL_COND(!is_grass_init);
    update_shader_time();
    if(update_lock){
        std::lock_guard<std::mutex> lock(update_mutex);
    }
    if(update_id!=grid->get_update_id()){
        return; // will be called after grid update will finish
    }
    VSet<int64_t> dirty_instances;
    for(int i=0;i<dirty_points_id->size();i++){
        int64_t din = grid->get_point_instance_id_by_point_id((*dirty_points_id)[i]);
        if(din!=0){
            dirty_instances.insert(din);
        }
    }
    for(int i=0;i<dirty_instances.size();i++){
        if(!grid_to_grass.has(dirty_instances[i])){
            WARN_PRINT("Dirty instance is not found "+itos(dirty_instances[i]));
            continue;
        }
        MGrassChunk* g = grid_to_grass[dirty_instances[i]];
        create_grass_chunk(-1,g);
    }
    //Clearing cell creation if reach the limit
    int cell_creation_data_remove_amount = cell_creation_order.size() - cell_creation_time_data_limit;
    if(cell_creation_data_remove_amount > 0){
        for(int i=0; i < cell_creation_data_remove_amount; i++){
            cell_creation.erase(cell_creation_order[i]);
        }
        cell_creation_order = cell_creation_order.slice(cell_creation_data_remove_amount, cell_creation_order.size());
    }
    memdelete(dirty_points_id);
    dirty_points_id = memnew(VSet<int>);
    cull_out_of_bound();
}

void MGrass::update_grass(){
    int new_chunk_count = grid->grid_update_info.size();
    std::lock_guard<std::mutex> lock(update_mutex);
    if(!is_grass_init){
        return;
    }
    update_id = grid->get_update_id();
    for(int i=0;i<new_chunk_count;i++){
        create_grass_chunk(i);
    }
    if(dirty_points_id->size()!=0){
        // Now in grid there is a valid point to instance
        update_dirty_chunks(false);
    }
    cull_out_of_bound();
}

void MGrass::cull_out_of_bound(){
    int count_pointer = 0;
    for(int i=0;i<grid->instances_distance.size();i++){
        MGrassChunk* g = grid_to_grass.get(grid->instances_distance[i].id);
        if(count_pointer<grass_count_limit){
            count_pointer += g->total_count;
            if(g->is_part_of_scene){
                g->unrelax();
            }else{
                to_be_visible.push_back(g);
            }
        } else {
            g->relax();
        }
    }
    final_count = count_pointer;
}

void MGrass::create_grass_chunk(int grid_index,MGrassChunk* grass_chunk){
    MGrassChunk* g;
    MPixelRegion px;
    if(grass_chunk==nullptr){
        px.left = (uint32_t)round(((double)grass_region_pixel_width)*CHUNK_INFO.region_offset_ratio.x);
        px.top = (uint32_t)round(((double)grass_region_pixel_width)*CHUNK_INFO.region_offset_ratio.y);
        int size_scale = pow(2,CHUNK_INFO.chunk_size);
        px.right = px.left + base_grid_size_in_pixel*size_scale - 1;
        px.bottom = px.top + base_grid_size_in_pixel*size_scale - 1;
        // We keep the chunk information for grass only in root grass chunk
        g = memnew(MGrassChunk(px,CHUNK_INFO.region_world_pos,CHUNK_INFO.lod,CHUNK_INFO.region_id));
        grid_to_grass.insert(CHUNK_INFO.terrain_instance.get_id(),g);
    } else {
        g = grass_chunk;
        // We clear tree to create everything again from start
        g->clear_tree();
        px = grass_chunk->pixel_region;
    }
    int lod = g->lod;
    int lod_scale;
    const float* rand_buffer =(float*)rand_buffer_pull[lod].ptr();
    if(settings[lod]->grid_lod >=0 && settings[lod]->grid_lod < lod_count){
        lod_scale = pow(2,settings[lod]->grid_lod);
    } else {
        lod_scale = pow(2,lod);
    }
    int grass_region_pixel_width_lod = grass_region_pixel_width/lod_scale;

    uint32_t divide_amount= (uint32_t)settings[lod]->multimesh_subdivisions;
    Vector<MPixelRegion> pixel_regions = px.devide(divide_amount);
    int cell_instance_count = settings[lod]->cell_instance_count;
    uint32_t buffer_strid_float = settings[lod]->get_buffer_strid_float();
    uint32_t buffer_strid_byte = settings[lod]->get_buffer_strid_byte();
    int rand_buffer_size = rand_buffer_pull[lod].size()/buffer_strid_float;
    ////////////////////////////////////////
    /// Color data and Custom data
    ////////////////////////////////////////
    bool process_color_data = settings[lod]->process_color_data();
    bool process_custom_data = settings[lod]->process_custom_data();
    int color_img_id = settings[lod]->color_img_index;
    int custom_img_id = settings[lod]->custom_img_index;
    
    

    const uint8_t* ptr = grass_data->data.ptr() + g->region_id*grass_region_pixel_size/8;

    MGrassChunk* root_g=g;
    MGrassChunk* last_g=g;
    uint32_t total_count=0;
    for(int k=0;k<pixel_regions.size();k++){
        px = pixel_regions[k];
        if(k!=0){
            g = memnew(MGrassChunk());
            last_g->next = g;
            last_g = g;
        }
        uint32_t count=0;
        uint32_t index;
        uint32_t x=px.left;
        uint32_t y=px.top;
        uint32_t i=0;
        uint32_t j=1;
        PackedFloat32Array buffer;
        while (true)
        {
            while (true){
                x = px.left + i*lod_scale;
                if(x>px.right){
                    break;
                }
                i++;
                uint32_t offs = (y*grass_region_pixel_width + x);
                uint32_t ibyte = offs/8;
                uint32_t ibit = offs%8;
                uint32_t cell_id = g->region_id*grass_region_pixel_size + offs;
                if( (ptr[ibyte] & (1 << ibit)) != 0){
                    for(int r=0;r<cell_instance_count;r++){
                        index=count*buffer_strid_float;
                        int rand_index = y*grass_region_pixel_width_lod + x*cell_instance_count + r;
                        const float* ptr = rand_buffer + (rand_index%rand_buffer_size)*buffer_strid_float;
                        buffer.resize(buffer.size()+buffer_strid_float);
                        float* ptrw = (float*)buffer.ptrw();
                        ptrw += index;
                        memcpy(ptrw,ptr,buffer_strid_byte);
                        Vector3 pos;
                        pos.x = root_g->world_pos.x + x*grass_data->density + ptrw[3];
                        pos.z = root_g->world_pos.z + y*grass_data->density + ptrw[11];
                        ptrw[7] += grid->get_height(pos);
                        if(std::isnan(ptrw[7])){
                            buffer.resize(buffer.size()-buffer_strid_float);
                            continue;
                        }
                        ptrw[3] = pos.x;
                        ptrw[11] = pos.z;
                        count++;
                        ///Adding Custom data
                        //Order is importnat
                        uint8_t dindex = 12;
                        if(process_color_data){
                            Color color_img;
                            if(color_img_id!=-1){
                                Vector2i px_pos =  grid->get_closest_pixel(pos);
                                color_img = grid->get_pixel(px_pos.x,px_pos.y,color_img_id);
                            }
                            //Red
                            if(settings[lod]->color_r == MGrassLodSetting::CUSTOM::IMAGE){
                                ptrw[dindex] = color_img.r;
                            } else if(settings[lod]->color_r == MGrassLodSetting::CUSTOM::CREATION_TIME){
                                ptrw[dindex] = get_shader_time(cell_id);
                            }
                            dindex++;
                            //Green
                            if(settings[lod]->color_g == MGrassLodSetting::CUSTOM::IMAGE){
                                ptrw[dindex] = color_img.g;
                            } else if(settings[lod]->color_g == MGrassLodSetting::CUSTOM::CREATION_TIME){
                                ptrw[dindex] = get_shader_time(cell_id);
                            }
                            dindex++;
                            //Blue
                            if(settings[lod]->color_b == MGrassLodSetting::CUSTOM::IMAGE){
                                ptrw[dindex] = color_img.b;
                            } else if(settings[lod]->color_b == MGrassLodSetting::CUSTOM::CREATION_TIME){
                                ptrw[dindex] = get_shader_time(cell_id);
                            }
                            dindex++;
                            //Alpha
                            if(settings[lod]->color_a == MGrassLodSetting::CUSTOM::IMAGE){
                                ptrw[dindex] = color_img.a;
                            } else if(settings[lod]->color_a == MGrassLodSetting::CUSTOM::CREATION_TIME){
                                ptrw[dindex] = get_shader_time(cell_id);
                            }
                            dindex++;
                            //Next
                        }
                        if(process_custom_data){
                            Color custom_img;
                            if(custom_img_id!=-1){
                                Vector2i px_pos =  grid->get_closest_pixel(pos);
                                custom_img = grid->get_pixel(px_pos.x,px_pos.y,custom_img_id);
                            }
                            //Red
                            if(settings[lod]->custom_r == MGrassLodSetting::CUSTOM::IMAGE){
                                ptrw[dindex] = custom_img.r;
                            } else if(settings[lod]->custom_r == MGrassLodSetting::CUSTOM::CREATION_TIME){
                                ptrw[dindex] = get_shader_time(cell_id);
                            }
                            dindex++;
                            //Green
                            if(settings[lod]->custom_g == MGrassLodSetting::CUSTOM::IMAGE){
                                ptrw[dindex] = custom_img.g;
                            } else if(settings[lod]->custom_g == MGrassLodSetting::CUSTOM::CREATION_TIME){
                                ptrw[dindex] = get_shader_time(cell_id);
                            }
                            dindex++;
                            //Blue
                            if(settings[lod]->custom_b == MGrassLodSetting::CUSTOM::IMAGE){
                                ptrw[dindex] = custom_img.b;
                            } else if(settings[lod]->custom_b == MGrassLodSetting::CUSTOM::CREATION_TIME){
                                ptrw[dindex] = get_shader_time(cell_id);
                            }
                            dindex++;
                            //Alpha
                            if(settings[lod]->custom_a == MGrassLodSetting::CUSTOM::IMAGE){
                                ptrw[dindex] = custom_img.a;
                            } else if(settings[lod]->custom_a == MGrassLodSetting::CUSTOM::CREATION_TIME){
                                ptrw[dindex] = get_shader_time(cell_id);
                            }
                        }
                    }
                }
            }
            i= 0;
            y= px.top + j*lod_scale;
            if(y>px.bottom){
                break;
            }
            j++;
        }
        // Discard grass chunk in case there is no mesh RID or count is less than min_grass_cutoff
        if(meshes.is_null() || meshes->get_mesh_rid(lod) == RID() || count < min_grass_cutoff){
            g->set_buffer(0,RID(),RID(),RID(),PackedFloat32Array());
            continue;
        }
        g->set_buffer(count,scenario,meshes->get_mesh_rid(lod),material_rids[lod],buffer,settings[lod]->active_color_data,settings[lod]->active_custom_data);
        //UtilityFunctions::print("buffer ",buffer);
        total_count += count;
    }
    root_g->set_shadow_setting(settings[lod]->shadow_setting);
    root_g->set_gi_mode(settings[lod]->gi_mode);
    root_g->total_count = total_count;
    if(grass_chunk!=nullptr){ // This means updating not creating
        root_g->unrelax();
    }
}



void MGrass::apply_update_grass(){
    std::lock_guard<std::mutex> lock(update_mutex);
    if(!is_grass_init){
        return;
    }
    for(int i=0;i<to_be_visible.size();i++){
        if(!to_be_visible[i]->is_out_of_range){
            if(is_visible()){
                to_be_visible[i]->unrelax();
            }
            to_be_visible[i]->is_part_of_scene = true;
        }
    }
    for(int i=0;i<grid->remove_instance_list.size();i++){
        if(grid_to_grass.has(grid->remove_instance_list[i].get_id())){
            MGrassChunk* g = grid_to_grass.get(grid->remove_instance_list[i].get_id());
            memdelete(g);
            grid_to_grass.erase(grid->remove_instance_list[i].get_id());
        } else {
            //UtilityFunctions::print("Grid to grass size ",grid_to_grass.size());
            WARN_PRINT("Instance not found for removing "+ itos(grid->remove_instance_list[i].get_id()));
        }
    }
    to_be_visible.clear();
}

void MGrass::recalculate_grass_config(int max_lod){
    lod_count = max_lod + 1;
    if(materials.size()!=lod_count){
        materials.resize(lod_count);
    }
    if(lod_settings.size()!=lod_count){
        lod_settings.resize(lod_count);
    }
    notify_property_list_changed();
}

void MGrass::make_grass_dirty_by_pixel(uint32_t px, uint32_t py){
    if(!is_grass_init) return;
    if(px>=width) return;
    if(py>=height) return;
    Vector2 flat_pos(float(px)*grass_data->density,float(py)*grass_data->density);
    int point_id = grid->get_point_id_by_non_offs_ws(flat_pos);
    dirty_points_id->insert(point_id);
}

void MGrass::set_grass_by_pixel(uint32_t px, uint32_t py, bool p_value){
    if(!is_grass_init) return;
    if(px>=width) return;
    if(py>=height) return;
    //ERR_FAIL_INDEX(px, width);
    //ERR_FAIL_INDEX(py, height);
    uint32_t rx = px/grass_region_pixel_width;
    uint32_t ry = py/grass_region_pixel_width;
    uint32_t rid = ry*region_grid_width + rx;
    uint32_t x = px%grass_region_pixel_width;
    uint32_t y = py%grass_region_pixel_width;
    uint32_t offs = rid*grass_region_pixel_size + y*grass_region_pixel_width + x;
    uint32_t ibyte = offs/8;
    uint32_t ibit = offs%8;

    uint8_t b = grass_data->data[ibyte];

    //if(p_value == ((b & (1 << ibit)) != 0) ){
    //    return;
    //}
    Vector2 flat_pos(float(px)*grass_data->density,float(py)*grass_data->density);
    int point_id = grid->get_point_id_by_non_offs_ws(flat_pos);
    dirty_points_id->insert(point_id);
    if(!cell_creation.has(offs)){
        cell_creation.insert(offs,std::numeric_limits<float>::quiet_NaN());
        cell_creation_order.push_back(offs);
    }
    if(p_value){
        b |= (1 << ibit);
    } else {
        b &= ~(1 << ibit);
    }
    grass_data->data.set(ibyte,b);
    if(grass_data->backup_exist()){
        grass_data->backup_data.set(ibyte,b);
    }
}

void MGrass::set_grass_sublayer_by_pixel(uint32_t px, uint32_t py, bool p_value){
    if(!is_grass_init) return;
    ERR_FAIL_COND(!grass_data->backup_exist());
    if(px>=width) return;
    if(py>=height) return;
    //ERR_FAIL_INDEX(px, width);
    //ERR_FAIL_INDEX(py, height);
    uint32_t rx = px/grass_region_pixel_width;
    uint32_t ry = py/grass_region_pixel_width;
    uint32_t rid = ry*region_grid_width + rx;
    uint32_t x = px%grass_region_pixel_width;
    uint32_t y = py%grass_region_pixel_width;
    uint32_t offs = rid*grass_region_pixel_size + y*grass_region_pixel_width + x;
    uint32_t ibyte = offs/8;
    uint32_t ibit = offs%8;
    uint8_t b = grass_data->data[ibyte];

    if(p_value == ((b & (1 << ibit)) != 0) ){
        return;
    }
    Vector2 flat_pos(float(px)*grass_data->density,float(py)*grass_data->density);
    int point_id = grid->get_point_id_by_non_offs_ws(flat_pos);
    dirty_points_id->insert(point_id);
    if(!cell_creation.has(offs)){
        cell_creation.insert(offs,std::numeric_limits<float>::quiet_NaN());
        cell_creation_order.push_back(offs);
    }
    if(p_value){
        b |= (1 << ibit);
    } else {
        b &= ~(1 << ibit);
    }
    grass_data->data.set(ibyte,b);
}

bool MGrass::is_init(){
    return is_grass_init;
}

bool MGrass::has_sublayer(){
    if(grass_data.is_null()){
        return false;
    }
    return grass_data->backup_exist();
}

void MGrass::merge_sublayer(){
    if(grass_data.is_null()){
        return;
    }
    grass_data->backup_merge();
}

void MGrass::create_sublayer(){
    ERR_FAIL_COND(grass_data.is_null());
    grass_data->backup_create();
}

void MGrass::clear_grass_sublayer_by_pixel(uint32_t px, uint32_t py){
    if(!is_grass_init) return;
    ERR_FAIL_COND(!grass_data->backup_exist());
    if(px>=width) return;
    if(py>=height) return;
    //ERR_FAIL_INDEX(px, width);
    //ERR_FAIL_INDEX(py, height);
    uint32_t rx = px/grass_region_pixel_width;
    uint32_t ry = py/grass_region_pixel_width;
    uint32_t rid = ry*region_grid_width + rx;
    uint32_t x = px%grass_region_pixel_width;
    uint32_t y = py%grass_region_pixel_width;
    uint32_t offs = rid*grass_region_pixel_size + y*grass_region_pixel_width + x;
    uint32_t ibyte = offs/8;
    uint32_t ibit = offs%8;

    uint8_t byte_backup = grass_data->backup_data[ibyte];
    uint8_t byte = grass_data->data[ibyte];
    if(byte_backup==byte){
        return;
    }
    bool backup_val = (byte_backup & (1 << ibit)) != 0;
    
    Vector2 flat_pos(float(px)*grass_data->density,float(py)*grass_data->density);
    int point_id = grid->get_point_id_by_non_offs_ws(flat_pos);
    dirty_points_id->insert(point_id);

    if(backup_val){
        byte |= (1 << ibit);
    } else {
        byte &= ~(1 << ibit);
    }
    grass_data->data.set(ibyte,byte);
}

bool MGrass::get_grass_by_pixel(uint32_t px, uint32_t py) {
    if(!is_grass_init) return false;
    ERR_FAIL_INDEX_V(px, width,false);
    ERR_FAIL_INDEX_V(py, height,false);
    uint32_t rx = px/grass_region_pixel_width;
    uint32_t ry = py/grass_region_pixel_width;
    uint32_t rid = ry*region_grid_width + rx;
    uint32_t x = px%grass_region_pixel_width;
    uint32_t y = py%grass_region_pixel_width;
    uint32_t offs = rid*grass_region_pixel_size + y*grass_region_pixel_width + x;
    uint32_t ibyte = offs/8;
    uint32_t ibit = offs%8;
    return (grass_data->data[ibyte] & (1 << ibit)) != 0;
}

Vector2i MGrass::get_closest_pixel(Vector3 pos){
    if(!is_grass_init) return Vector2i();
    if(!grid) return Vector2i();
    pos -= grid->offset;
    pos = pos / grass_data->density;
    return Vector2i(round(pos.x),round(pos.z));
}

Vector3 MGrass::get_pixel_world_pos(uint32_t px, uint32_t py){
    Vector3 out(0,0,0);
    ERR_FAIL_COND_V(!grass_data.is_valid(),out);
    out.x = grid->offset.x + ((float)px)*grass_data->density;
    out.z = grid->offset.z + ((float)py)*grass_data->density;
    return out;
}

Vector2i MGrass::grass_px_to_grid_px(uint32_t px, uint32_t py){
    Vector2 v;
    ERR_FAIL_COND_V(!grass_data.is_valid(),Vector2i(v));
    v = Vector2(px,py)*grass_data->density;
    v = v/grid->get_h_scale();
    return Vector2i(round(v.x),round(v.y));
}

// At least for now it is not safe to put this function inside a thread
// because set_grass_by_pixel is chaning dirty_points_id
// And I don't think that we need to do that because it is not a huge process
void MGrass::draw_grass(Vector3 brush_pos,real_t radius,bool add){
    ERR_FAIL_COND(!is_grass_init);
    //ERR_FAIL_COND(update_id!=grid->get_update_id());
    Vector2i px_pos = get_closest_pixel(brush_pos);
    if(px_pos.x<0 || px_pos.y<0 || px_pos.x>width || px_pos.y>height){
        return;
    }
    uint32_t brush_px_radius = (uint32_t)(radius/grass_data->density);
    uint32_t brush_px_pos_x = px_pos.x;
    uint32_t brush_px_pos_y = px_pos.y;
    // Setting left right top bottom
    MPixelRegion px;
    px.left = (brush_px_pos_x>brush_px_radius) ? brush_px_pos_x - brush_px_radius : 0;
    px.right = brush_px_pos_x + brush_px_radius;
    px.right = px.right > grass_pixel_region.right ? grass_pixel_region.right : px.right;
    px.top = (brush_px_pos_y>brush_px_radius) ? brush_px_pos_y - brush_px_radius : 0;
    px.bottom = brush_px_pos_y + brush_px_radius;
    px.bottom = (px.bottom>grass_pixel_region.bottom) ? grass_pixel_region.bottom : px.bottom;
    //VariantUtilityFunctions::_print("brush pos ", brush_pos);
    //VariantUtilityFunctions::_print("draw R ",brush_px_radius);
    //VariantUtilityFunctions::_print("L ",itos(px.left)," R ",itos(px.right)," T ",itos(px.top), " B ",itos(px.bottom));
    // LOD Scale
    //int lod_scale = pow(2,lod);
    // LOOP
    uint32_t x=px.left;
    uint32_t y=px.top;
    uint32_t i=0;
    uint32_t j=1;
    for(uint32_t y = px.top; y<=px.bottom;y++){
        for(uint32_t x = px.left; x<=px.right;x++){
            uint32_t dif_x = UABS_DIFF(x,brush_px_pos_x);
            uint32_t dif_y = UABS_DIFF(y,brush_px_pos_y);
            uint32_t dis = sqrt(dif_x*dif_x + dif_y*dif_y);
            Vector2i grid_px = grass_px_to_grid_px(x,y);
            if(dis<brush_px_radius && grid->get_brush_mask_value_bool(grid_px.x,grid_px.y))
                set_grass_by_pixel(x,y,add);
        }
    }
    update_dirty_chunks();
}

void MGrass::clear_grass_sublayer_aabb(AABB aabb){
    ERR_FAIL_COND(!is_grass_init);
    if(!grass_data->backup_exist()){
        return;
    }
    Vector2i start = get_closest_pixel(aabb.position);
    Vector2i end = get_closest_pixel(aabb.position + aabb.size);
    start.x = start.x < 0 ? 0 : start.x;
    start.y = start.y < 0 ? 0 : start.y;
    end.x = end.x < width ? end.x : width - 1;
    end.y = end.y < height ? end.y : height - 1;

    for(int j=start.y; j <= end.y; j++){
        for(int i=start.x; i <= end.x ; i++){
            clear_grass_sublayer_by_pixel(i,j);
        }
    }
    update_dirty_chunks();
}

void MGrass::set_lod_setting_image_index(Ref<MGrassLodSetting> lod_setting){
    if(lod_setting->has_color_img()){
        lod_setting->color_img_index = grid->get_terrain_material()->get_texture_id(lod_setting->color_img);
    }
    if(lod_setting->has_custom_img()){
        lod_setting->custom_img_index = grid->get_terrain_material()->get_texture_id(lod_setting->custom_img);
    }
}

void MGrass::set_active(bool input){
    active = input;
}
bool MGrass::get_active(){
    return active;
}
void MGrass::set_grass_data(Ref<MGrassData> d){
    grass_data = d;
}

Ref<MGrassData> MGrass::get_grass_data(){
    return grass_data;
}

void MGrass::set_cell_creation_time_data_limit(int input){
    cell_creation_time_data_limit = input;
}

int MGrass::get_cell_creation_time_data_limit(){
    return cell_creation_time_data_limit;
}

void MGrass::set_grass_count_limit(int input){
    grass_count_limit = input;
}
int MGrass::get_grass_count_limit(){
    return grass_count_limit;
}

void MGrass::set_min_grass_cutoff(int input){
    ERR_FAIL_COND(input<0);
    min_grass_cutoff = input;
}

int MGrass::get_min_grass_cutoff(){
    return min_grass_cutoff;
}

void MGrass::set_lod_settings(Array input){
    lod_settings = input;
}
Array MGrass::get_lod_settings(){
    return lod_settings;
}

void MGrass::set_meshes(Variant input){ // For comtibilty with older MTerrain version
    if(meshes.is_valid()){
        meshes->disconnect("meshes_changed",Callable(this,"recreate_all_grass"));
    }
    if(input.get_type()==Variant::ARRAY){
        Array arr = input;
        if(arr.size()==0){
            meshes = Ref<MMeshLod>();
        }
        TypedArray<Mesh> arr_mesh;
        for(int i=0; i < arr.size(); i++){
            Ref<Mesh> _m = arr[i];
            arr_mesh.push_back(_m);
        }
        if(meshes.is_null()){
            meshes.instantiate();
        }
        meshes->set_meshes(arr_mesh);
        return;
    } else {
        meshes = input;
    }
    if(meshes.is_valid()){
        meshes->connect("meshes_changed",Callable(this,"recreate_all_grass"));
    }
    recreate_all_grass();
}
Ref<MMeshLod> MGrass::get_meshes(){
    return meshes;
}

void MGrass::set_materials(Array input){
    materials = input;
}

Array MGrass::get_materials(){
    return materials;
}

uint32_t MGrass::get_width(){
    return width;
}
uint32_t MGrass::get_height(){
    return height;
}

int64_t MGrass::get_count(){
    return final_count;
}

void MGrass::set_collision_radius(float input){
    collision_radius=input;
}

float MGrass::get_collision_radius(){
    return collision_radius;
}

void MGrass::set_shape_offset(Vector3 input){
    shape_offset = input;
}

Vector3 MGrass::get_shape_offset(){
    return shape_offset;
}

void MGrass::set_shape(Ref<Shape3D> input){
    shape = input;
}
Ref<Shape3D> MGrass::get_shape(){
    return shape;
}

int MGrass::get_collision_layer(){
    return collision_layer;
}
void MGrass::set_collision_layer(int input){
    collision_layer = input;
}
int MGrass::get_collision_mask(){
    return collision_mask;
}
void MGrass::set_collision_mask(int input){
    collision_mask = input;
}
Ref<PhysicsMaterial> MGrass::get_physics_material(){
    return physics_material;
}
void MGrass::set_physics_material(Ref<PhysicsMaterial> input){
    physics_material = input;
}

void MGrass::set_active_shape_resize(bool input){
    active_shape_resize = input;
}

bool MGrass::get_active_shape_resize(){
    return active_shape_resize;
}

void MGrass::set_nav_obstacle_radius(float input){
    ERR_FAIL_COND(input<0.05);
    nav_obstacle_radius = input;
}
float MGrass::get_nav_obstacle_radius(){
    return nav_obstacle_radius;
}
/*
Vector3 pos;
pos.x = root_g->world_pos.x + x*grass_data->density + ptrw[3];
pos.z = root_g->world_pos.z + y*grass_data->density + ptrw[11];
ptrw[3] = pos.x;
ptrw[7] += grid->get_height(pos);
ptrw[11] = pos.z;
uint32_t rx = x/grass_region_pixel_width;
uint32_t ry = y/grass_region_pixel_width;
*/
void MGrass::update_physics(Vector3 cam_pos){
    ERR_FAIL_COND(!main_physics_body.is_valid());
    if(!shape.is_valid()){
        return;
    }
    ERR_FAIL_COND(!is_grass_init);
    int cell_instance_count = settings[0]->cell_instance_count;
    uint32_t buffer_strid_float = settings[0]->get_buffer_strid_float();
    uint32_t buffer_strid_byte = settings[0]->get_buffer_strid_byte();
    cam_pos -= grid->offset;
    cam_pos = cam_pos / grass_data->density;
    int px_x = round(cam_pos.x);
    int px_y = round(cam_pos.z);
    int col_r = round(collision_radius/grass_data->density);
    if(px_x < - col_r || px_y < - col_r){
        remove_all_physics();
        return;
    }
    physics_search_bound = MBound(MGridPos(px_x,0,px_y));
    physics_search_bound.grow(grass_bound_limit,col_r,col_r);
    if(!(physics_search_bound.left <width && physics_search_bound.top<height)){
        remove_all_physics();
        return;
    }
    //UtilityFunctions::print("Left ",physics_search_bound.left," right ",physics_search_bound.right," top ",physics_search_bound.top," bottom ",physics_search_bound.bottom );
    // culling
    /// Removing out of bound shapes
    int remove_count=0;
    for(int y=last_physics_search_bound.top;y<=last_physics_search_bound.bottom;y++){
        for(int x=last_physics_search_bound.left;x<=last_physics_search_bound.right;x++){
            if(!physics_search_bound.has_point(x,y)){
                for(int r=0;r<cell_instance_count;r++){
                    uint64_t uid = y*width*cell_instance_count + x*cell_instance_count + r;
                    int find_index = shapes_ids.find(uid);
                    if(find_index!=-1){
                        PS->body_remove_shape(main_physics_body,find_index);
                        shapes_ids.remove_at(find_index);
                    }
                }
            }
        }
    }
    last_physics_search_bound = physics_search_bound;
    const float* rand_buffer =(float*)rand_buffer_pull[0].ptr();
    int rand_buffer_size = rand_buffer_pull[0].size()/buffer_strid_float;
    int update_count = 0;
    for(uint32_t y=physics_search_bound.top;y<=physics_search_bound.bottom;y++){
        for(uint32_t x=physics_search_bound.left;x<=physics_search_bound.right;x++){
            if(!get_grass_by_pixel(x,y)){
                continue;
            }
            for(int r=0;r<cell_instance_count;r++){
                uint64_t uid = y*width*cell_instance_count + x*cell_instance_count + r;
                if(shapes_ids.has(uid)){
                    continue;
                }
                int rx = (x/grass_region_pixel_width);
                int ry = (y/grass_region_pixel_width);
                int rand_index = (y-ry*grass_region_pixel_width)*grass_region_pixel_width + (x-rx*grass_region_pixel_width)* cell_instance_count + r;
                //VariantUtilityFunctions::_print("grass_region_pixel_width ", grass_region_pixel_width);
                //VariantUtilityFunctions::_print("X ",x, " Y ", y, " RX ",rx, " RY ", ry);
                //VariantUtilityFunctions::_print("rand_index ",(rand_index));
                const float* ptr = rand_buffer + (rand_index%rand_buffer_size)*BUFFER_STRID_FLOAT;
                Vector3 wpos(x*grass_data->density+ptr[3],0,y*grass_data->density+ptr[11]);
                //VariantUtilityFunctions::_print("Physic pos ",wpos);
                wpos += grid->offset;
                wpos.y = grid->get_height(wpos) + ptr[7];
                if(std::isnan(wpos.y)){
                    continue;
                }
                // Godot physics not work properly with collission transformation
                // So for now we ignore transformation
                Vector3 x_axis(ptr[0],ptr[4],ptr[8]);
                Vector3 y_axis(ptr[1],ptr[5],ptr[9]);
                Vector3 z_axis(ptr[2],ptr[6],ptr[10]);
                ///
                //Vector3 x_axis(1,0,0);
                //Vector3 y_axis(0,1,0);
                //Vector3 z_axis(0,0,1);
                Basis b_no_normlized(x_axis,y_axis,z_axis);
                x_axis.normalize();
                y_axis.normalize();
                z_axis.normalize();
                Basis b(x_axis,y_axis,z_axis);
                if((shape_type==PhysicsServer3D::ShapeType::SHAPE_CONVEX_POLYGON || shape_type==PhysicsServer3D::ShapeType::SHAPE_CONCAVE_POLYGON)){
                    //As tested godot physics is ok with scaling
                    //So we make some exception here
                    Transform3D final_t(Basis(),shape_offset);
                    final_t = Transform3D(b_no_normlized,wpos)*final_t;
                    PS->body_add_shape(main_physics_body,shape->get_rid(),final_t);
                }
                else if(active_shape_resize){
                    RID s_rid;
                    Vector3 scale = b_no_normlized.get_scale();
                    Vector3 offset = shape_offset*scale; //If this has offset we should currect the offset
                    Transform3D final_t(Basis(),offset);
                    Transform3D t(b,wpos);
                    final_t = t * final_t;
                    if(shapes_rids.has(rand_index)){
                        s_rid = shapes_rids[rand_index];
                    } else {
                        s_rid = get_resized_shape(scale);
                        shapes_rids.insert(rand_index,s_rid);
                    }
                    PS->body_add_shape(main_physics_body,s_rid,final_t);
                } else{
                    Vector3 offset = shape_offset;
                    Transform3D final_t(Basis(),offset);
                    Transform3D t(b,wpos);
                    final_t = t * final_t;
                    PS->body_add_shape(main_physics_body,shape->get_rid(),final_t);
                }
                shapes_ids.push_back(uid);
                update_count++;
            }
        }
    }
}

void MGrass::remove_all_physics(){
    if(shape.is_valid()){
        PS->body_clear_shapes(main_physics_body);
        shapes_ids.clear();
    }
}

RID MGrass::get_resized_shape(Vector3 scale){
    RID new_shape;
    ERR_FAIL_COND_V(!shape.is_valid(),new_shape);
    if(shape_type==PhysicsServer3D::ShapeType::SHAPE_SPHERE){
        float max_s = scale.x > scale.y ? scale.x : scale.y;
        max_s = max_s > scale.z ? max_s : scale.z;
        float r = shape_data;
        r = r*max_s;
        new_shape = PS->sphere_shape_create();
        PS->shape_set_data(new_shape,r);
    } else if(shape_type==PhysicsServer3D::ShapeType::SHAPE_BOX){
        Vector3 d = shape_data;
        d = d*scale;
        new_shape = PS->box_shape_create();
        PS->shape_set_data(new_shape,d);
    } else if(shape_type==PhysicsServer3D::ShapeType::SHAPE_CYLINDER){
        Dictionary d = shape_data;
        float max_xz = scale.x > scale.z ? scale.x : scale.z;
        float r = d["radius"];
        r = r*max_xz;
        float h = d["height"];
        h = h*scale.y;
        Dictionary new_d;
        new_d["radius"] = r;
        new_d["height"] = h;
        new_shape = PS->cylinder_shape_create();
        PS->shape_set_data(new_shape,new_d);
    } else if(shape_type==PhysicsServer3D::ShapeType::SHAPE_CAPSULE){
        Dictionary d = shape_data;
        float max_xz = scale.x > scale.z ? scale.x : scale.z;
        float r = d["radius"];
        r = r*max_xz;
        float h = d["height"];
        h = h*scale.y;
        Dictionary new_d;
        new_d["radius"] = r;
        new_d["height"] = h;
        new_shape = PS->capsule_shape_create();
        PS->shape_set_data(new_shape,new_d);
    } else{
        WARN_PRINT("Shape Type "+itos(shape_type)+" Does not support resizing");
        new_shape = shape->get_rid();
    }
    return new_shape;
}

PackedVector3Array MGrass::get_physic_positions(Vector3 cam_pos,float radius){
    uint32_t buffer_strid_float = settings[0]->get_buffer_strid_float();
    //uint32_t buffer_strid_byte = settings[0]->get_buffer_strid_byte();
    PackedVector3Array positions;
   if(!shape.is_valid()){
        return positions;
    }
    ERR_FAIL_COND_V(!is_grass_init,positions);
    int cell_instance_count = settings[0]->cell_instance_count;
    cam_pos -= grid->offset;
    cam_pos = cam_pos / grass_data->density;
    int px_x = round(cam_pos.x);
    int px_y = round(cam_pos.z);
    int col_r = round(radius/grass_data->density);
    if(px_x < -col_r || px_y < -col_r){
        return positions;
    }
    MPixelRegion bound;
    bound.left = px_x > col_r ? px_x - col_r : 0;
    bound.top = px_y > col_r ? px_y - col_r : 0;
    if(!(bound.left <width && bound.top<height)){
        return positions;
    }
    bound.right = px_x + col_r;
    bound.right = bound.right < width ? bound.right : width - 1;
    bound.bottom = px_y + col_r;
    bound.bottom = bound.bottom < height ? bound.bottom : height - 1;
    const float* rand_buffer =(float*)rand_buffer_pull[0].ptr();
    int rand_buffer_size = rand_buffer_pull[0].size()/buffer_strid_float;
    for(uint32_t y=bound.top;y<=bound.bottom;y++){
        for(uint32_t x=bound.left;x<=bound.right;x++){
            if(!get_grass_by_pixel(x,y)){
                continue;
            }
            for(int r=0;r<cell_instance_count;r++){
                int rx = (x/grass_region_pixel_width);
                int ry = (y/grass_region_pixel_width);
                int rand_index = (y-ry*grass_region_pixel_width)*grass_region_pixel_width + (x-rx*grass_region_pixel_width) + r;
                const float* ptr = rand_buffer + (rand_index%rand_buffer_size)*buffer_strid_float;
                Vector3 wpos(x*grass_data->density+ptr[3],0,y*grass_data->density+ptr[11]);
                wpos += grid->offset;
                wpos.y = grid->get_height(wpos) + ptr[7];
                wpos += shape_offset;
                positions.push_back(wpos);
            }
        }
    }
    return positions;
}

void MGrass::_get_property_list(List<PropertyInfo> *p_list) const{
    PropertyInfo sub_lod0(Variant::INT, "LOD Settings", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_SUBGROUP);
    p_list->push_back(sub_lod0);
    for(int i=0;i<materials.size();i++){
        PropertyInfo m(Variant::OBJECT,"Setting_LOD_"+itos(i),PROPERTY_HINT_RESOURCE_TYPE,"MGrassLodSetting",PROPERTY_USAGE_EDITOR);
        p_list->push_back(m);
    }
    PropertyInfo sub_lod(Variant::INT, "Grass Materials", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_SUBGROUP);
    p_list->push_back(sub_lod);
    for(int i=0;i<materials.size();i++){
        PropertyInfo m(Variant::OBJECT,"Material_LOD_"+itos(i),PROPERTY_HINT_RESOURCE_TYPE,"StandardMaterial3D,ORMMaterial3D,ShaderMaterial",PROPERTY_USAGE_EDITOR);
        p_list->push_back(m);
    }
}

bool MGrass::_get(const StringName &p_name, Variant &r_ret) const{
    String _name = p_name;
    if(_name.begins_with("Material_LOD_")){
        PackedStringArray s = _name.split("_");
        int index = s[2].to_int();
        if(index>materials.size()-1){
            return false;
        }
        r_ret = materials[index];
        return true;
    }
    if(p_name.begins_with("Setting_LOD_")){
        PackedStringArray s = _name.split("_");
        int index = s[2].to_int();
        if(index>lod_settings.size()-1){
            return false;
        }
        r_ret = lod_settings[index];
        return true;
    }
    return false;
}
bool MGrass::_set(const StringName &p_name, const Variant &p_value){
    String _name = p_name;
    if(_name.begins_with("Material_LOD_")){
        PackedStringArray s = _name.split("_");
        int index = s[2].to_int();
        if(index>materials.size()-1){
            return false;
        }
        materials[index] = p_value;
        if(is_grass_init){
            Ref<Material> m = p_value;
            if(m.is_valid()){
                material_rids.set(index,m->get_rid());
            } else {
                material_rids.set(index,RID());
            }
            recreate_all_grass(index);
        }
        return true;
    }
    if(p_name.begins_with("Setting_LOD_")){
        PackedStringArray s = _name.split("_");
        int index = s[2].to_int();
        if(index>lod_settings.size()-1){
            return false;
        }
        lod_settings[index] = p_value;
        Ref<MGrassLodSetting> setting = p_value;
        if(setting.is_valid()){
            if(!setting->is_connected("lod_setting_changed",Callable(this,"_lod_setting_changed")))
            {
                setting->connect("lod_setting_changed",Callable(this,"_lod_setting_changed"));
            }
        }
        if(is_grass_init){
            if(!setting.is_valid()){
                setting = ResourceLoader::load("res://addons/m_terrain/default_lod_setting.res");
            }
            settings.set(index,setting);
            settings[index]->is_dirty = true;
            _lod_setting_changed();
        }
        return true;
    }
    return false;
}

Error MGrass::save_grass_data(){
    if(grass_data.is_valid()){
        return ResourceSaver::save(grass_data,grass_data->get_path());
    }
    return ERR_UNAVAILABLE;   
}

void MGrass::_recreate_all_grass(){
    recreate_all_grass();
}

void MGrass::recreate_all_grass(int lod){
    std::lock_guard<std::mutex> lock(update_mutex);
    for(HashMap<int64_t,MGrassChunk*>::Iterator it = grid_to_grass.begin();it!=grid_to_grass.end();++it){
        if(lod==-1){
            create_grass_chunk(-1,it->value);
        } else if (lod==it->value->lod)
        {
            create_grass_chunk(-1,it->value);
        }
    }
}

void MGrass::update_random_buffer_pull(int lod){
    ERR_FAIL_INDEX(lod,rand_buffer_pull.size());
    int lod_scale;
    if(settings[lod]->grid_lod >=0){
        lod_scale = pow(2,settings[lod]->grid_lod);
    } else {
        lod_scale = pow(2,lod);
    }
    float cdensity = grass_data->density*lod_scale;
    rand_buffer_pull.set(lod,settings[lod]->generate_random_number(cdensity,100));
    set_lod_setting_image_index(settings[lod]);
}

void MGrass::_lod_setting_changed(){
    if(!grid->is_created() || !is_init()){
        return;
    }
    for(int i=0;i<lod_count;i++){
        if(settings[i].is_valid()){
            if(settings[i]->is_dirty){
                update_random_buffer_pull(i);
                recreate_all_grass(i);
            }
        }
    }
    for(int i=0;i<lod_count;i++){
        if(settings[i].is_valid()){
            settings[i]->is_dirty = false;
        }
    }
}

void MGrass::check_undo(){
    ERR_FAIL_COND(!grass_data.is_valid());
    grass_data->check_undo();
}

void MGrass::undo(){
    ERR_FAIL_COND(!grass_data.is_valid());
    grass_data->undo();
    recreate_all_grass();
}

bool MGrass::is_depend_on_image(int image_index){
    for(int i=0;i<settings.size();i++){
        Ref<MGrassLodSetting> s = settings[i];
        if(s->color_img_index == image_index || s->custom_img_index){
            return true;
        }
    }
    return false;
}

float MGrass::get_shader_time(){
    double t = double(OS::get_singleton()->get_ticks_msec());
    t /= 1000.0;
    t = fmod(t,time_rollover_secs);
    return t;
}

void MGrass::update_shader_time(){
    current_shader_time = get_shader_time() - 0.07;
}

float MGrass::get_shader_time(uint32_t grass_cell_index){
    if(!cell_creation.has(grass_cell_index)){
        return std::numeric_limits<float>::quiet_NaN();
    }
    float val = cell_creation.get(grass_cell_index);
    if(std::isnan(val)){
        cell_creation.insert(grass_cell_index,current_shader_time);
        return current_shader_time;
    }
    return val;
}

void MGrass::_notification(int32_t what){
    switch (what)
    {
    case NOTIFICATION_VISIBILITY_CHANGED:
        _update_visibilty();
        break;
    case NOTIFICATION_EDITOR_PRE_SAVE:
        save_grass_data();
        break;
    default:
        break;
    }
}


bool MGrass::is_visible(){
    return visible;
}

void MGrass::_update_visibilty(){
    if(!is_inside_tree()){
        set_visibility(false);
        return;
    }
    set_visibility(is_visible_in_tree());
}

void MGrass::set_visibility(bool input){
    if(!is_init()){
        visible = input;
        return;
    }
    if(input == visible){
        return;
    }
    std::lock_guard<std::mutex> lock(update_mutex);
    for(HashMap<int64_t,MGrassChunk*>::Iterator it=grid_to_grass.begin(); it!=grid_to_grass.end(); ++it){
        MGrassChunk* c = it->value;
        if(input){
            c->unrelax();
        } else {
            c->relax();
        }
    }
    visible = input;
}
