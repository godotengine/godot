#include "mterrain.h"
#include "core/variant/variant.h"
#include "scene/resources/3d/world_3d.h"
#include "scene/main/viewport.h"
#include  "scene/3d/camera_3d.h"
#include "core/config/engine.h"
// #include <godot_cpp/classes/reg_ex.hpp>
// #include <godot_cpp/classes/reg_ex_match.hpp>
#include "core/io/file_access.h"
#include "core/io/dir_access.h"

#include "mbrush_manager.h"
#include "navmesh/mnavigation_region_3d.h"
#include "mbrush_layers.h"
#include "mtool.h"

Vector<MTerrain*> MTerrain::all_terrain_nodes;

void MTerrain::_bind_methods() {
    ClassDB::bind_method(D_METHOD("_dummy_setter","input"), &MTerrain::_dummy_setter);
    ClassDB::bind_method(D_METHOD("_dummy_getter"), &MTerrain::_dummy_getter);

    ClassDB::bind_method(D_METHOD("_finish_terrain"), &MTerrain::_finish_terrain);
    ClassDB::bind_method(D_METHOD("create_grid"), &MTerrain::create_grid);
    ClassDB::bind_method(D_METHOD("remove_grid"), &MTerrain::remove_grid);
    ClassDB::bind_method(D_METHOD("restart_grid"), &MTerrain::restart_grid);
    ClassDB::bind_method(D_METHOD("update"), &MTerrain::update);
    ClassDB::bind_method(D_METHOD("finish_update"), &MTerrain::finish_update);
    ClassDB::bind_method(D_METHOD("update_physics"), &MTerrain::update_physics);
    ClassDB::bind_method(D_METHOD("finish_update_physics"), &MTerrain::finish_update_physics);
    ClassDB::bind_method(D_METHOD("is_ram_image","uniform_name"), &MTerrain::is_ram_image);
    ClassDB::bind_method(D_METHOD("get_image_list"), &MTerrain::get_image_list);
    ClassDB::bind_method(D_METHOD("get_image_id", "uniform_name"), &MTerrain::get_image_id);
    ClassDB::bind_method(D_METHOD("set_save_config","conf"), &MTerrain::set_save_config);
    ClassDB::bind_method(D_METHOD("save_image","image_index","force_save"), &MTerrain::save_image);
    ClassDB::bind_method(D_METHOD("has_unsave_image"), &MTerrain::has_unsave_image);
    ClassDB::bind_method(D_METHOD("save_all_dirty_images"), &MTerrain::save_all_dirty_images);
    ClassDB::bind_method(D_METHOD("is_finishing_update_chunks"), &MTerrain::is_finish_updating);
    ClassDB::bind_method(D_METHOD("is_finishing_update_physics"), &MTerrain::is_finish_updating_physics);
    ClassDB::bind_method(D_METHOD("get_pixel", "x","y","image_index"), &MTerrain::get_pixel);
    ClassDB::bind_method(D_METHOD("set_pixel", "x","y","color","image_index"), &MTerrain::set_pixel);
    ClassDB::bind_method(D_METHOD("get_height_by_pixel", "x","y"), &MTerrain::get_height_by_pixel);
    ClassDB::bind_method(D_METHOD("set_height_by_pixel", "x","y","value"), &MTerrain::set_height_by_pixel);
    ClassDB::bind_method(D_METHOD("get_closest_height", "world_position"), &MTerrain::get_closest_height);
    ClassDB::bind_method(D_METHOD("get_height", "world_position"), &MTerrain::get_height);
    ClassDB::bind_method(D_METHOD("get_ray_collision_point", "ray_origin","ray_vector","step","max_step"), &MTerrain::get_ray_collision_point);
    
    ClassDB::bind_method(D_METHOD("set_dataDir","dir"), &MTerrain::set_dataDir);
    ClassDB::bind_method(D_METHOD("get_dataDir"), &MTerrain::get_dataDir);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "dataDir",PROPERTY_HINT_DIR), "set_dataDir", "get_dataDir");
    ClassDB::bind_method(D_METHOD("set_layersDataDir","input"), &MTerrain::set_layersDataDir);
    ClassDB::bind_method(D_METHOD("get_layersDataDir"), &MTerrain::get_layersDataDir);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "layersDataDir",PROPERTY_HINT_GLOBAL_DIR), "set_layersDataDir","get_layersDataDir");

    ClassDB::bind_method(D_METHOD("set_grid_create","val"), &MTerrain::set_create_grid);
    ClassDB::bind_method(D_METHOD("get_create_grid"), &MTerrain::get_create_grid);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "create"), "set_grid_create", "get_create_grid");
    ClassDB::bind_method(D_METHOD("set_terrain_material","input"), &MTerrain::set_terrain_material);
    ClassDB::bind_method(D_METHOD("get_terrain_material"), &MTerrain::get_terrain_material);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"terrain_material",PROPERTY_HINT_RESOURCE_TYPE,"MTerrainMaterial"),"set_terrain_material","get_terrain_material");
    
    ClassDB::bind_method(D_METHOD("set_heightmap_layers", "input"), &MTerrain::set_heightmap_layers);
    ClassDB::bind_method(D_METHOD("get_heightmap_layers"), &MTerrain::get_heightmap_layers);
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "heightmap_layers",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE), "set_heightmap_layers","get_heightmap_layers");

    ClassDB::bind_method(D_METHOD("set_chunks_update_interval","interval"), &MTerrain::set_chunks_update_interval);
    ClassDB::bind_method(D_METHOD("get_chunks_update_interval"), &MTerrain::get_chunks_update_interval);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "chunks_update_interval"), "set_chunks_update_interval", "get_chunks_update_interval");

    ClassDB::bind_method(D_METHOD("set_chunks_update_loop_enabled", "val"), &MTerrain::set_chunks_update_loop_enabled);
    ClassDB::bind_method(D_METHOD("get_chunks_update_loop_enabled"), &MTerrain::get_chunks_update_loop_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "chunks_update_loop_enabled"), "set_chunks_update_loop_enabled", "get_chunks_update_loop_enabled");

    ClassDB::bind_method(D_METHOD("set_physics_update_interval", "val"), &MTerrain::set_physics_update_interval);
    ClassDB::bind_method(D_METHOD("get_physics_update_interval"), &MTerrain::get_physics_update_interval);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "physics_update_interval"), "set_physics_update_interval", "get_physics_update_interval");

    ClassDB::bind_method(D_METHOD("set_physics_update_loop_enabled", "val"), &MTerrain::set_physics_update_loop_enabled);
    ClassDB::bind_method(D_METHOD("get_physics_update_loop_enabled"), &MTerrain::get_physics_update_loop_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "physics_update_loop_enabled"), "set_physics_update_loop_enabled", "get_physics_update_loop_enabled");

    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"Region Unit",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_CATEGORY),"_dummy_setter","_dummy_getter");

    ClassDB::bind_method(D_METHOD("set_regions_limit","input"), &MTerrain::set_regions_limit);
    ClassDB::bind_method(D_METHOD("get_regions_limit"), &MTerrain::get_regions_limit);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "regions_visible"), "set_regions_limit", "get_regions_limit");

    ClassDB::bind_method(D_METHOD("set_regions_processing_physics", "val"), &MTerrain::set_regions_processing_physics);
    ClassDB::bind_method(D_METHOD("get_regions_processing_physics"), &MTerrain::get_regions_processing_physics);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "regions_processing_physics"), "set_regions_processing_physics", "get_regions_processing_physics");

    ClassDB::bind_method(D_METHOD("get_terrain_region_count"), &MTerrain::get_terrain_region_count);
    ClassDB::bind_method(D_METHOD("set_terrain_region_count", "size"), &MTerrain::set_terrain_region_count);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I,"terrain_region_count",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR), "set_terrain_region_count", "get_terrain_region_count");

    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"Grid unit (min_size)",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_CATEGORY),"_dummy_setter","_dummy_getter");

    ClassDB::bind_method(D_METHOD("get_terrain_size"), &MTerrain::get_terrain_size);
    ClassDB::bind_method(D_METHOD("set_terrain_size", "size"), &MTerrain::set_terrain_size);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I,"terrain_quad_count"), "set_terrain_size", "get_terrain_size");

    ClassDB::bind_method(D_METHOD("set_region_size", "region_size"), &MTerrain::set_region_size);
    ClassDB::bind_method(D_METHOD("get_region_size"), &MTerrain::get_region_size);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"region_quad_count"), "set_region_size", "get_region_size");
    
    
    ClassDB::bind_method(D_METHOD("set_max_range", "max_range"), &MTerrain::set_max_range);
    ClassDB::bind_method(D_METHOD("get_max_range"), &MTerrain::get_max_range);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_quad_visible"), "set_max_range", "get_max_range");

    ClassDB::bind_method(D_METHOD("set_custom_camera", "camera"), &MTerrain::set_custom_camera);
    ClassDB::bind_method(D_METHOD("set_editor_camera", "camera"), &MTerrain::set_editor_camera);

    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"Meter unit",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_CATEGORY),"_dummy_setter","_dummy_getter");

    ClassDB::bind_method(D_METHOD("set_distance_update_threshold","input"), &MTerrain::set_distance_update_threshold);
    ClassDB::bind_method(D_METHOD("get_distance_update_threshold"), &MTerrain::get_distance_update_threshold);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"distance_update_threshold"),"set_distance_update_threshold","get_distance_update_threshold");

    ClassDB::bind_method(D_METHOD("set_offset", "offset"), &MTerrain::set_offset);
    ClassDB::bind_method(D_METHOD("get_offset"), &MTerrain::get_offset);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "offset"), "set_offset", "get_offset");

    ClassDB::bind_method(D_METHOD("set_min_size","index"), &MTerrain::set_min_size);
    ClassDB::bind_method(D_METHOD("get_min_size"), &MTerrain::get_min_size);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"quad_size_min",PROPERTY_HINT_ENUM, M_SIZE_LIST_STRING), "set_min_size", "get_min_size");

    ClassDB::bind_method(D_METHOD("set_max_size","index"), &MTerrain::set_max_size);
    ClassDB::bind_method(D_METHOD("get_max_size"), &MTerrain::get_max_size);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"quad_size_max",PROPERTY_HINT_ENUM, M_SIZE_LIST_STRING), "set_max_size", "get_max_size");
    
    ClassDB::bind_method(D_METHOD("set_min_h_scale","index"), &MTerrain::set_min_h_scale);
    ClassDB::bind_method(D_METHOD("get_min_h_scale"), &MTerrain::get_min_h_scale);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"min_h_scale",PROPERTY_HINT_ENUM, M_H_SCALE_LIST_STRING), "set_min_h_scale", "get_min_h_scale");

    ClassDB::bind_method(D_METHOD("set_max_h_scale","index"), &MTerrain::set_max_h_scale);
    ClassDB::bind_method(D_METHOD("get_max_h_scale"), &MTerrain::get_max_h_scale);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"max_h_scale",PROPERTY_HINT_ENUM, M_H_SCALE_LIST_STRING), "set_max_h_scale", "get_max_h_scale");

    ClassDB::bind_method(D_METHOD("set_size_info", "size_info"), &MTerrain::set_size_info);
    ClassDB::bind_method(D_METHOD("get_size_info"), &MTerrain::get_size_info);
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "size_info",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE), "set_size_info", "get_size_info");

    ClassDB::bind_method(D_METHOD("set_lod_distance", "lod_distance"), &MTerrain::set_lod_distance);
    ClassDB::bind_method(D_METHOD("get_lod_distance"), &MTerrain::get_lod_distance);
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "lod_distance",PROPERTY_HINT_NONE,"", PROPERTY_USAGE_STORAGE),"set_lod_distance","get_lod_distance");

    ADD_GROUP("Physics","");

    ClassDB::bind_method(D_METHOD("set_physics_material","input"), &MTerrain::set_physics_material);
    ClassDB::bind_method(D_METHOD("get_physics_material"), &MTerrain::get_physics_material);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"physics_material",PROPERTY_HINT_RESOURCE_TYPE,"PhysicsMaterial"),"set_physics_material","get_physics_material");

    ClassDB::bind_method(D_METHOD("get_collision_layer"), &MTerrain::get_collision_layer);
    ClassDB::bind_method(D_METHOD("set_collision_layer","input"), &MTerrain::set_collision_layer);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"collision_layer",PROPERTY_HINT_LAYERS_3D_PHYSICS),"set_collision_layer","get_collision_layer");

    ClassDB::bind_method(D_METHOD("set_collision_mask","input"), &MTerrain::set_collision_mask);
    ClassDB::bind_method(D_METHOD("get_collision_mask"), &MTerrain::get_collision_mask);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"collision_mask",PROPERTY_HINT_LAYERS_3D_PHYSICS),"set_collision_mask","get_collision_mask");

    ClassDB::bind_method(D_METHOD("get_pixel_world_pos", "x","y"), &MTerrain::get_pixel_world_pos);
    ClassDB::bind_method(D_METHOD("get_closest_pixel", "world_pos"), &MTerrain::get_closest_pixel);
    ClassDB::bind_method(D_METHOD("set_brush_manager", "brush_manager"), &MTerrain::set_brush_manager);
    ClassDB::bind_method(D_METHOD("set_brush_start_point", "brush_pos","radius"), &MTerrain::set_brush_start_point);
    ClassDB::bind_method(D_METHOD("draw_height", "brush_pos","radius","brush_id"), &MTerrain::draw_height);
    ClassDB::bind_method(D_METHOD("draw_color","brush_pos","radius","brush_id","index"),&MTerrain::draw_color);

    ClassDB::bind_method(D_METHOD("set_active_layer_by_name","layer_name"), &MTerrain::set_active_layer_by_name);
    ClassDB::bind_method(D_METHOD("add_heightmap_layer","layer_name"), &MTerrain::add_heightmap_layer);
    ClassDB::bind_method(D_METHOD("rename_heightmap_layer","old_name","new_name"), &MTerrain::rename_heightmap_layer);
    ClassDB::bind_method(D_METHOD("merge_heightmap_layer"), &MTerrain::merge_heightmap_layer);
    ClassDB::bind_method(D_METHOD("remove_heightmap_layer"), &MTerrain::remove_heightmap_layer);
    ClassDB::bind_method(D_METHOD("toggle_heightmap_layer_visibile"), &MTerrain::toggle_heightmap_layer_visibile);
    ClassDB::bind_method(D_METHOD("get_layer_visibility","input"), &MTerrain::get_layer_visibility);

    ClassDB::bind_method(D_METHOD("_terrain_ready_signal"), &MTerrain::terrain_ready_signal);
    ClassDB::bind_method(D_METHOD("terrain_child_changed"), &MTerrain::terrain_child_changed);
    ClassDB::bind_method(D_METHOD("get_region_grid_size"), &MTerrain::get_region_grid_size);
    ClassDB::bind_method(D_METHOD("get_region_id_by_world_pos","world_pos"), &MTerrain::get_region_id_by_world_pos);
    ClassDB::bind_method(D_METHOD("get_base_size"), &MTerrain::get_base_size);
    ClassDB::bind_method(D_METHOD("get_h_scale"), &MTerrain::get_h_scale);
    ClassDB::bind_method(D_METHOD("get_pixel_width"), &MTerrain::get_pixel_width);
    ClassDB::bind_method(D_METHOD("get_pixel_height"), &MTerrain::get_pixel_height);

    ClassDB::bind_method(D_METHOD("set_brush_layers","input"), &MTerrain::set_brush_layers);
    ClassDB::bind_method(D_METHOD("get_brush_layers"), &MTerrain::get_brush_layers);
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY,"brush_layers",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE),"set_brush_layers","get_brush_layers");
    ClassDB::bind_method(D_METHOD("set_brush_layers_num","input"), &MTerrain::set_brush_layers_num);
    ClassDB::bind_method(D_METHOD("get_brush_layers_num"), &MTerrain::get_brush_layers_num);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"brush_layers_groups_num",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NONE),"set_brush_layers_num","get_brush_layers_num");
    
    ADD_GROUP("util","");
    ClassDB::bind_method(D_METHOD("set_set_mtime","input"), &MTerrain::set_set_mtime);
    ClassDB::bind_method(D_METHOD("get_set_mtime"), &MTerrain::get_set_mtime);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"set_mtime"),"set_set_mtime","get_set_mtime");
    
    
    ClassDB::bind_method(D_METHOD("get_layers_info"), &MTerrain::get_layers_info);
    ClassDB::bind_method(D_METHOD("set_color_layer","index","group_index","brush_name"), &MTerrain::set_color_layer);
    ClassDB::bind_method(D_METHOD("disable_brush_mask"), &MTerrain::disable_brush_mask);
    ClassDB::bind_method(D_METHOD("set_brush_mask","mask_image"), &MTerrain::set_brush_mask);
    ClassDB::bind_method(D_METHOD("set_brush_mask_px_pos","mask_image"), &MTerrain::set_brush_mask_px_pos);
    ClassDB::bind_method(D_METHOD("set_mask_cutoff","val"), &MTerrain::set_mask_cutoff);
    ClassDB::bind_method(D_METHOD("images_add_undo_stage"), &MTerrain::images_add_undo_stage);
    ClassDB::bind_method(D_METHOD("images_undo"), &MTerrain::images_undo);
    ClassDB::bind_method(D_METHOD("get_normal_by_pixel","x","y"),&MTerrain::get_normal_by_pixel);
    ClassDB::bind_method(D_METHOD("get_normal_accurate_by_pixel","x","y"),&MTerrain::get_normal_accurate_by_pixel);
    ClassDB::bind_method(D_METHOD("get_normal","world_pos"), &MTerrain::get_normal);
    ClassDB::bind_method(D_METHOD("get_normal_accurate","world_pos"), &MTerrain::get_normal_accurate);
    ClassDB::bind_method(D_METHOD("is_grid_created"), &MTerrain::is_grid_created);
    ClassDB::bind_method(D_METHOD("update_all_dirty_image_texture","update_physics"), &MTerrain::update_all_dirty_image_texture);
    ClassDB::bind_method(D_METHOD("update_normals","left","right","top","bottom"), &MTerrain::update_normals);

    ClassDB::bind_method(D_METHOD("_update_visibility") , &MTerrain::_update_visibility);

    ClassDB::bind_static_method("MTerrain",D_METHOD("get_all_terrain_nodes"), &MTerrain::get_all_terrain_nodes);
}

TypedArray<MTerrain> MTerrain::get_all_terrain_nodes(){
    TypedArray<MTerrain> out;
    for(int i=0; i < all_terrain_nodes.size();i++){
        if(all_terrain_nodes[i]->is_inside_tree()){
            out.push_back(all_terrain_nodes[i]);
        }
    }
    return out;
}

MTerrain::MTerrain() {
    lod_distance.append(3);
    lod_distance.append(6);
    lod_distance.append(12);
    lod_distance.append(16);
    lod_distance.append(24);
    connect("tree_exited", Callable(this, "_update_visibility"));
    connect("tree_entered", Callable(this, "_update_visibility"));
    recalculate_terrain_config(true);
    grid = memnew(MGrid);
    update_chunks_timer = memnew(Timer);
    update_chunks_timer->set_wait_time(chunks_update_interval);
    update_chunks_timer->set_one_shot(true);
    add_child(update_chunks_timer);
    update_chunks_timer->connect("timeout", Callable(this, "finish_update"));

    update_physics_timer = memnew(Timer);
    update_physics_timer->set_wait_time(physics_update_interval);
    update_physics_timer->set_one_shot(true);
    add_child(update_physics_timer);
    update_physics_timer->connect("timeout", Callable(this, "finish_update_physics"));

    // Grass List update conditions
    connect("ready",Callable(this,"_terrain_ready_signal"));
    connect("child_exiting_tree",Callable(this,"terrain_child_changed"));
    connect("child_entered_tree",Callable(this,"terrain_child_changed"));
    all_terrain_nodes.push_back(this);
}

MTerrain::~MTerrain() {
    remove_grid();
    memdelete(grid);
    for(int i=0; i < all_terrain_nodes.size(); i++){
        if(this==all_terrain_nodes[i]){
            all_terrain_nodes.remove_at(i);
            break;
        }
    }
}


void MTerrain::_finish_terrain() {
    if(update_thread_chunks.valid()){
        update_thread_chunks.wait();
    }
    remove_grid();
}

void MTerrain::create_grid(){
    ERR_FAIL_COND(grid->is_created());
    ERR_FAIL_COND_EDMSG(terrain_size.x%region_size!=0,"Terrain size X component is not divisible by region size");
    ERR_FAIL_COND_EDMSG(terrain_size.y%region_size!=0,"Terrain size Y component is not divisible by region size");
    if(Engine::get_singleton()->is_editor_hint()){
        if(dataDir.is_empty() || dataDir == String("res://") || !dataDir.is_absolute_path()){
            dataDir = "res://mterrain_data";
        }
        if(layersDataDir.is_empty() || layersDataDir == String("res://") || !layersDataDir.is_absolute_path()){
            layersDataDir = dataDir.path_join("layers");
        }

        if(dataDir.is_absolute_path() && !DirAccess::dir_exists_absolute(dataDir)){
            DirAccess::make_dir_recursive_absolute(dataDir);
        }
        if(layersDataDir.is_absolute_path() && !DirAccess::dir_exists_absolute(layersDataDir)){
            DirAccess::make_dir_recursive_absolute(layersDataDir);
        }
    }
    _chunks = memnew(MChunks);
    grid->update_renderer_info();
    _chunks->create_chunks(size_list[min_size_index],size_list[max_size_index],h_scale_list[min_h_scale_index],h_scale_list[max_h_scale_index],size_info);
    grid->set_scenario(get_world_3d()->get_scenario());
    grid->space = get_world_3d()->get_space();
    grid->offset = offset;
    grid->dataDir = dataDir;
    grid->layersDataDir = layersDataDir;
    grid->region_size = region_size;
    // Loading save config if there is any
    grid->save_config.clear();
    String save_config_path = dataDir.path_join(M_SAVE_CONFIG_NAME);
    if(FileAccess::exists(save_config_path)){
        Ref<ConfigFile> save_config;
        save_config.instantiate();
        if(save_config->load(save_config_path) == Error::OK){
            set_save_config(save_config);
        } else {
            WARN_PRINT("Error loading save config "+save_config_path);
        }
    } else {
        WARN_PRINT("Can not find save config "+save_config_path);
    }
    set_heightmap_layers(grid->heightmap_layers); // To make sure we have everything currect including background image
    if(terrain_material.is_valid()){
        grid->set_terrain_material(terrain_material);
    } else {
        Ref<MTerrainMaterial> m;
        if(grid->is_opengl()){
            m = ResourceLoader::load(M_DEAFAULT_MATERIAL_OPENGL_PATH);
        } else {
            m = ResourceLoader::load(M_DEAFAULT_MATERIAL_PATH);
        }
        grid->set_terrain_material(m);
    }
    grid->lod_distance = lod_distance;
    grid->create(terrain_size.x,terrain_size.y,_chunks);
    if(!grid->is_created()){
        return;
    }
    get_cam_pos();
    grid->update_regions_bounds(cam_pos,false);
    grid->update_regions_at_load();
    grid->clear_region_bounds();
    grid->update_chunks(cam_pos);
    grid->apply_update_chunks();
    grid->update_physics(cam_pos);
    last_update_pos = cam_pos;
    // Grass Part
    terrain_ready_signal();
    for(int i=0;i<grass_list.size();i++){
        grass_list[i]->init_grass(grid);
        if(grass_list[i]->is_grass_init){
            confirm_grass_list.push_back(grass_list[i]);
            if(grass_list[i]->shape.is_valid()){
                confirm_grass_col_list.push_back(grass_list[i]);
                grass_list[i]->update_physics(cam_pos);
            }
        }
    }

    for(int i=0;i<get_child_count();i++){
        MNavigationRegion3D*  mnav= Object::cast_to<MNavigationRegion3D>(get_child(i));
        if(mnav){
            mnav->init(this,grid);
            if(mnav->is_nav_init && Engine::get_singleton()->is_editor_hint() && mnav->has_data()){
                confirm_nav.push_back(mnav);
                mnav->update_npoints();
                mnav->apply_update_npoints();
            }
        }
    }
    total_update_count = confirm_grass_list.size() + confirm_nav.size();
    if(physics_update_loop_enabled){
        update_physics();
    }
    if(chunks_update_loop_enabled){
        update();
    }
}

void MTerrain::remove_grid(){
    update_chunks_timer->stop();
    update_physics_timer->stop();
    if(update_thread_chunks.valid()){
        update_thread_chunks.wait();
        finish_updating = true;
    }
    if(update_regions_future.valid()){
        update_regions_future.wait();
    }
    is_update_regions_future_valid = false;
    if(update_thread_physics.valid()){
        update_thread_physics.wait();
        finish_updating_physics = true;
    }
    grid->clear();
    for(int i=0;i<confirm_grass_list.size();i++){
        confirm_grass_list[i]->clear_grass();
    }
    confirm_grass_list.clear();
    confirm_grass_col_list.clear();
    for(int i=0;i<get_child_count();i++){
        MNavigationRegion3D*  mnav= Object::cast_to<MNavigationRegion3D>(get_child(i));
        if(mnav){
            mnav->clear();
        }
    }
    confirm_nav.clear();
}

void MTerrain::restart_grid(){
    remove_grid();
    create_grid();
}

void MTerrain::update() {
    ERR_FAIL_COND(!finish_updating);
    ERR_FAIL_COND(!grid->is_created());
    get_cam_pos();
    finish_updating = false;
    // In case -1 is Terrain grid update turn
    if(update_stage==-1){
        if(!is_update_regions_future_valid || update_regions_future.wait_for(std::chrono::microseconds(0))==std::future_status::ready){
            if(grid->update_regions_bounds(cam_pos,true)){
                update_regions_future = std::async(std::launch::async, &MGrid::update_regions, grid);
                is_update_regions_future_valid=true;
            } else {
                is_update_regions_future_valid=false;
            }
        }
        // Finish update regions //
                // Terrain update
        last_update_pos = cam_pos;
        update_thread_chunks = std::async(std::launch::async, &MGrid::update_chunks, grid, cam_pos);
    }
    else if(update_stage>=0){
        if(update_stage>=confirm_grass_list.size()){
            int index = update_stage - confirm_grass_list.size();
            update_thread_chunks = std::async(std::launch::async, &MNavigationRegion3D::update_npoints, confirm_nav[index]);
        } else {
            update_thread_chunks = std::async(std::launch::async, &MGrass::update_grass, confirm_grass_list[update_stage]);
        }
    }
    update_chunks_timer->start();
}

void MTerrain::finish_update() {
    //VariantUtilityFunctions::_print("Finish update stage ", update_stage);
    if(update_stage == -2){
        bool finish_update_region = false;
        if(is_update_regions_future_valid && update_regions_future.wait_for(std::chrono::microseconds(0))==std::future_status::ready){
            finish_update_region = true;
            is_update_regions_future_valid = false;
        }
        get_cam_pos();
        float dis = last_update_pos.distance_to(cam_pos);
        if(dis>distance_update_threshold || finish_update_region){
            update_stage = -1;
            finish_updating = true;
            call_deferred("update");
            return;
        } else {
            update_chunks_timer->start();
        }
        return;
    }
    std::future_status status = update_thread_chunks.wait_for(std::chrono::microseconds(1));
    if(status == std::future_status::ready){
        finish_updating = true;
        if(update_stage==-1){
            grid->apply_update_chunks();
            if(confirm_grass_list.size()!=0){
                update_stage++;
                call_deferred("update");
            } else {
                update_stage = -2;
                finish_update();
            }
        }
        else if(update_stage>=0){
            if(update_stage>=confirm_grass_list.size()){
                int index = update_stage - confirm_grass_list.size();
                confirm_nav[index]->apply_update_npoints();
            } else {
                confirm_grass_list[update_stage]->apply_update_grass();
            }
            update_stage++;
            if(update_stage>=total_update_count){
                update_stage = -2;
                finish_update();
            } else {
                call_deferred("update");
            }
        }
    } else {
        update_chunks_timer->start();
    }
}

void MTerrain::update_physics(){
    ERR_FAIL_COND(!finish_updating_physics);
    ERR_FAIL_COND(!grid->is_created());
    get_cam_pos();
    finish_updating_physics = false;
    if(update_stage_physics==-1){
        update_thread_physics = std::async(std::launch::async, &MGrid::update_physics, grid, cam_pos);
    } else if(update_stage_physics>=0){
        update_thread_physics = std::async(std::launch::async, &MGrass::update_physics,confirm_grass_col_list[update_stage_physics], cam_pos);
    }
    update_physics_timer->start();
}

void MTerrain::finish_update_physics(){
    std::future_status status = update_thread_physics.wait_for(std::chrono::microseconds(1));
    if(status == std::future_status::ready){
        finish_updating_physics = true;
        update_stage_physics++;
        if(update_stage_physics>=confirm_grass_col_list.size()){
            update_stage_physics = -1;
        }
        if(physics_update_loop_enabled){
            call_deferred("update_physics");
        }
    } else {
        update_physics_timer->start();
    }
}

bool MTerrain::is_finish_updating(){
    return finish_updating;
}
bool MTerrain::is_finish_updating_physics(){
    return finish_updating_physics;
}

bool MTerrain::is_ram_image(const String& uniform_name){
    ERR_FAIL_COND_V(!grid->is_created(),false);
    ERR_FAIL_COND_V(!grid->get_terrain_material().is_valid(),false);
    int img_id = grid->get_terrain_material()->get_texture_id(uniform_name);
    ERR_FAIL_COND_V(img_id==-1,false);
    return grid->regions[0].images[img_id]->is_ram_image;
}

PackedStringArray MTerrain::get_image_list(){
    ERR_FAIL_COND_V(!grid->is_created(),PackedStringArray());
    ERR_FAIL_COND_V(!grid->get_terrain_material().is_valid(),PackedStringArray());
    return grid->get_terrain_material()->get_textures_list();
}

int MTerrain::get_image_id(String uniform_name){
    ERR_FAIL_COND_V(!grid->is_created(),false);
    ERR_FAIL_COND_V(!grid->get_terrain_material().is_valid(),-1);
    return grid->get_terrain_material()->get_texture_id(uniform_name);
}

void MTerrain::set_save_config(Ref<ConfigFile> conf){
    ERR_FAIL_COND(!grid);
	List<String> sections;
	conf->get_sections(&sections);
    for(const String& section : sections){
        if(section == "heightmap"){
            if(conf->has_section_key(section, "accuracy")){
                grid->save_config.accuracy = conf->get_value(section,"accuracy");
            } else {
                WARN_PRINT("Can not find accuraccy in save config file");
            }
            if(conf->has_section_key(section, "file_compress")){
                grid->save_config.heightmap_file_compress = (MResource::FileCompress)((int)conf->get_value(section,"file_compress"));
            } else {
                WARN_PRINT("Can not find file_compress for heightmap in save config file");
            }
            if(conf->has_section_key(section, "compress_qtq")){
                grid->save_config.heightmap_compress_qtq = ((bool)conf->get_value(section,"compress_qtq"));
            } else {
                WARN_PRINT("Can not find compress_qtq for heightmap in save config file");
            }
        } else {
            if(conf->has_section_key(section, "compress")){
                grid->save_config.data_compress.insert(section,(MResource::Compress((int)conf->get_value(section,"compress"))));
            } else {
                WARN_PRINT(section + ": can not find compress in save config");
            }
            if(conf->has_section_key(section, "file_compress")){
                grid->save_config.data_file_compress.insert(section,(MResource::FileCompress((int)conf->get_value(section,"file_compress"))));
            } else {
                WARN_PRINT(section + ": can not find file_compress in save config");
            }
        }
    }
}

void MTerrain::save_image(int image_index, bool force_save) {
    ERR_FAIL_COND(!grid->is_created());
    ERR_FAIL_COND(image_index>grid->uniforms_id.keys().size());
    grid->save_image(image_index,force_save);
}

bool MTerrain::has_unsave_image(){
    return grid->has_unsave_image();
}

void MTerrain::save_all_dirty_images(){
    if(grid->is_created()){
        grid->save_all_dirty_images();
    }
}

Color MTerrain::get_pixel(const uint32_t x,const uint32_t y, const int32_t index){
    return grid->get_pixel(x,y,index);
}
void MTerrain::set_pixel(const uint32_t x,const uint32_t y,const Color& col,const int32_t index){
    grid->set_pixel(x,y,col,index);
}
real_t MTerrain::get_height_by_pixel(const uint32_t x,const uint32_t y){
    return grid->get_height_by_pixel(x,y);
}
void MTerrain::set_height_by_pixel(const uint32_t x,const uint32_t y,const real_t value){
    grid->set_height_by_pixel(x,y,value);
}

void MTerrain::get_cam_pos() {
    if(custom_camera != nullptr){
        cam_pos = custom_camera->get_global_position();
        return;
    }
    if(Engine::get_singleton()->is_editor_hint()){
        Node3D* cam = MTool::find_editor_camera(true);
        if(cam!=nullptr){
            cam_pos = cam->get_global_position();
            return;
        }

    }
    Viewport* v = get_viewport();
    if(v!=nullptr){
        Camera3D* camera = v->get_camera_3d();
        if(camera!=nullptr){
            cam_pos = camera->get_global_position();
            return;
        }
    }
    ERR_FAIL_MSG("No camera is detected");
}

void MTerrain::set_dataDir(String input) {
    ERR_FAIL_COND_MSG(grid->is_created(),"Can not change dataDir after terrain is created!");
    dataDir = input;
    if(Engine::get_singleton()->is_editor_hint() && !dataDir.is_empty() && dataDir != String("res://") && dataDir.is_absolute_path() && layersDataDir.is_empty() && is_inside_tree()){
        // Setting automaticly layer data directory
        layersDataDir = dataDir.path_join("layers");
    }
}

String MTerrain::get_dataDir() {
    return dataDir;
}

void MTerrain::set_layersDataDir(String input){
    ERR_FAIL_COND_MSG(grid->is_created(),"Can not change layersDataDir after terrain is created!");
    layersDataDir = input;
}
String MTerrain::get_layersDataDir(){
    return layersDataDir;
}

void MTerrain::set_create_grid(bool input){
    if(!is_inside_tree()){
        return;
    }
    if(grid->is_created() && !input){
        remove_grid();
        return;
    }
    if(!grid->is_created() && input){
        create_grid();
        return;
    }
}

bool MTerrain::get_create_grid(){
    return grid->is_created();
}

void MTerrain::set_regions_limit(int input){
    if(input<1){
        return;
    }
    grid->region_limit = input;
}
int MTerrain::get_regions_limit(){
    return grid->region_limit;
}

float MTerrain::get_chunks_update_interval(){
    return chunks_update_interval;
}
void MTerrain::set_chunks_update_interval(float input){
    chunks_update_interval = input;
    if(input < 0.001){
        chunks_update_interval = 0.001;
    }
    update_chunks_timer->set_wait_time(chunks_update_interval);
}

float MTerrain::get_distance_update_threshold(){
    return distance_update_threshold;
}
void MTerrain::set_distance_update_threshold(float input){
    distance_update_threshold = input;
}

void MTerrain::set_chunks_update_loop_enabled(bool input) {
    chunks_update_loop_enabled = input;
    if(chunks_update_loop_enabled && finish_updating){
        update();
    }
}

bool MTerrain::get_chunks_update_loop_enabled() {
    return chunks_update_loop_enabled;
}

float MTerrain::get_physics_update_interval(){
    return physics_update_interval;
}
void MTerrain::set_physics_update_interval(float input){
    physics_update_interval = input;
    if(input < 0.001){
        physics_update_interval = 0.001;
    }
    update_physics_timer->set_wait_time(physics_update_interval);
}
bool MTerrain::get_physics_update_loop_enabled(){
    return physics_update_loop_enabled;
}
void MTerrain::set_physics_update_loop_enabled(bool input){
    physics_update_loop_enabled = input;
    if(physics_update_loop_enabled && finish_updating_physics){
        update_physics();
    }
}

void MTerrain::set_regions_processing_physics(int32_t input){
    if(input<0){
        grid->regions_processing_physics = 0;
    } else {
        grid->regions_processing_physics = input;
    }
}
int32_t MTerrain::get_regions_processing_physics(){
    return grid->regions_processing_physics;
}

Vector2i MTerrain::get_terrain_size(){
    return terrain_size;
}

void MTerrain::set_terrain_size(Vector2i size){
    ERR_FAIL_COND_EDMSG(size.x < 1 || size.y < 1,"Terrain size can not be zero");
    if(size == terrain_size){
        return;
    }
    terrain_size = size;
}

Vector2i MTerrain::get_terrain_region_count(){
    return terrain_size/region_size;
}

void MTerrain::set_terrain_region_count(Vector2i size){
    terrain_size = size * region_size;
}


void MTerrain::set_max_range(int32_t input) {
    ERR_FAIL_COND_EDMSG(input<1,"Max range can not be less than one");
    max_range = input;
    grid->max_range = input;
}

int32_t MTerrain::get_max_range() {
    return max_range;
}

void MTerrain::set_editor_camera(Node3D* camera_node){
    editor_camera = camera_node;
}
void MTerrain::set_custom_camera(Node3D* camera_node){
    custom_camera = camera_node;
}

void MTerrain::set_offset(Vector3 input){
    input.y = 0;
    offset = input;
}

Vector3 MTerrain::get_offset(){
    return offset;
}


void MTerrain::set_region_size(int32_t input) {
    ERR_FAIL_COND_EDMSG(input<4,"Region size can not be smaller than 4");
    region_size = input;
}

int32_t MTerrain::get_region_size() {
    return region_size;
}

void MTerrain::recalculate_terrain_config(const bool& force_calculate) {
    if(!is_inside_tree() && !force_calculate){
        return;
    }
    // Calculating max size
    max_size = (int8_t)(max_size_index - min_size_index);
    // Calculating max lod
    max_lod = (int8_t)(max_h_scale_index - min_h_scale_index);
    if(h_scale_list[max_h_scale_index] > size_list[min_size_index]){
        size_info.clear();
        notify_property_list_changed();
        ERR_FAIL_COND("min size is smaller than max h scale");
    }
    size_info.clear();
    size_info.resize(max_size+1);
    for(int i=0;i<size_info.size();i++){
        Array lod;
        lod.resize(max_lod+1);
        for(int j=0;j<lod.size();j++){
            if(j==lod.size()-1){
                lod[j] = true;
                continue;
            }
            lod[j] = i <=j;
        }
        size_info[i] = lod;
    }

    /// Calculating LOD distance
    lod_distance.resize(max_lod);
    int32_t ll = lod_distance[0];
    for(int i=1;i<lod_distance.size();i++){
        if(lod_distance[i] <= ll){
            lod_distance.write[i] = ll + 1; 
        }
        ll = lod_distance.write[i];
    }
    notify_property_list_changed();
    for(int i=0;i<grass_list.size();i++){
        grass_list[i]->recalculate_grass_config(max_lod);
    }
}

int MTerrain::get_min_size() {
    return min_size_index;
}

void MTerrain::set_min_size(int index) {
    if(index >= max_size_index){
        return;
    }
    min_size_index = index;
    recalculate_terrain_config(false);
}

int MTerrain::get_max_size() {
    return max_size_index;
}

void MTerrain::set_max_size(int index) {
    if(index <= min_size_index){
        return;
    }
    max_size_index = index;
    recalculate_terrain_config(false);
}

void MTerrain::set_min_h_scale(int index) {
    if(index >= max_h_scale_index){
        return;
    }
    min_h_scale_index = index;
    recalculate_terrain_config(false);
}

int MTerrain::get_min_h_scale() {
    return min_h_scale_index;
}

void MTerrain::set_max_h_scale(int index) {
    if(index <= min_h_scale_index){
        return;
    }
    max_h_scale_index = index;
    recalculate_terrain_config(false);
}

int MTerrain::get_max_h_scale(){
    return max_h_scale_index;
}

int MTerrain::get_collision_layer(){
    return grid->collision_layer;
}
void MTerrain::set_collision_layer(int input){
    grid->collision_layer = input;
}
int MTerrain::get_collision_mask(){
    return grid->collision_mask;
}
void MTerrain::set_collision_mask(int input){
    grid->collision_mask = input;
}
Ref<PhysicsMaterial> MTerrain::get_physics_material(){
    return grid->physics_material;
}
void MTerrain::set_physics_material(Ref<PhysicsMaterial> input){
    grid->physics_material = input;
}

void MTerrain::set_size_info(const Array& arr) {
    size_info = arr;
}
Array MTerrain::get_size_info() {
    return size_info;
}

void MTerrain::set_lod_distance(const PackedInt32Array& input){
    lod_distance = input;
}

PackedInt32Array MTerrain::get_lod_distance() {
    return lod_distance;
}

void MTerrain::_get_property_list(List<PropertyInfo> *p_list) const {
    //Adding lod distance property
    PropertyInfo sub_lod(Variant::INT, "LOD distance", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP);
    p_list->push_back(sub_lod);
    for(int i=0; i<lod_distance.size();i++){
        PropertyInfo p(Variant::INT,"M_LOD_"+itos(i),PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR);
        p_list->push_back(p);
    }
    //Adding size info property
    PropertyInfo size_group(Variant::INT, "Generating Chunks", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP);
    p_list->push_back(size_group);
    for(int size=0;size<size_info.size();size++){
        Array lod_info = size_info[size];
        PropertyInfo sub(Variant::INT, "Size "+itos(size_list[size+min_size_index]), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_SUBGROUP);
        p_list->push_back(sub);
        for(int lod=0;lod<lod_info.size();lod++){
            PropertyInfo p(Variant::BOOL,"SIZE_"+itos(size)+"_LOD_"+itos(lod)+"_HSCALE_"+itos(h_scale_list[lod+min_h_scale_index]),PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR);
            p_list->push_back(p);
        }
    }
    // Brush layers property
    PropertyInfo ccat(Variant::INT, "Brush Layers", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_CATEGORY);
    p_list->push_back(ccat);
    PropertyInfo bln(Variant::INT, "brush_layers_groups_num", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR);
    p_list->push_back(bln);
    for(int i=0;i<brush_layers.size();i++){
        PropertyInfo brl(Variant::OBJECT, "MBL_"+itos(i), PROPERTY_HINT_RESOURCE_TYPE, "MBrushLayers", PROPERTY_USAGE_EDITOR);
        p_list->push_back(brl);
    }
}

real_t MTerrain::get_closest_height(const Vector3& pos) {
    return grid->get_closest_height(pos);
}
real_t MTerrain::get_height(const Vector3& pos){
    return grid->get_height(pos);
}

Ref<MCollision> MTerrain::get_ray_collision_point(Vector3 ray_origin,Vector3 ray_vector,real_t step,int max_step){
    if(!grid->is_created()){
        Ref<MCollision> col;
        col.instantiate();
        return col;
    }
    return grid->get_ray_collision_point(ray_origin,ray_vector,step,max_step);
}

bool MTerrain::_get(const StringName &_name, Variant &r_ret) const {
    String p_name = _name;
    if(p_name.begins_with("SIZE_")){
        PackedStringArray parts = p_name.split("_");
        if(parts.size() != 6){
            return false;
        }
        int64_t size = parts[1].to_int();
        int64_t lod = parts[3].to_int();
        Array lod_info = size_info[size];
        r_ret = lod_info[lod];
        return true;
    }
    if(p_name.begins_with("M_LOD_")){
        int64_t index = p_name.get_slicec('_',2).to_int();
        r_ret = (float)lod_distance[index];
        return true;
    }
    if(p_name.begins_with("MBL_")){
        int64_t index = p_name.get_slicec('_',1).to_int();
        r_ret = brush_layers[index];
        return true;
    }
    return false;
}


bool MTerrain::_set(const StringName &_name, const Variant &p_value) {
    String p_name = _name;
    if(p_name.begins_with("SIZE_")){
        PackedStringArray parts = p_name.split("_");
        if(parts.size() != 6){
            return false;
        }
        int64_t size = parts[1].to_int();
        if(size==0){
            return false;
        }
        int64_t lod = parts[3].to_int();
        Array lod_info = size_info[size];
        lod_info[lod] = p_value;
        size_info[size] = lod_info;
        return true;
    }
    if(p_name.begins_with("M_LOD_")){
        int64_t index = p_name.get_slicec('_',2).to_int();
        int32_t val = p_value;
        if(val<0){
            return false;
        }
        lod_distance.write[index] = val;
        return true;
    }
    if(p_name.begins_with("MBL_")){
        int64_t index = p_name.get_slicec('_',1).to_int();
        brush_layers[index] = p_value;
        return true;
    }
    /// Compatibilty stuff
    if(p_name==String("max_range")){
        set_max_range(p_value);
        return true;
    }
    if(p_name==String("terrain_size")){
        set_terrain_size(p_value);
        return true;
    }
    if(p_name==String("region_size")){
        set_region_size(p_value);
        return true;
    }
    if(p_name==String("regions_limit")){
        set_regions_limit(p_value);
        return true;
    }
    if(p_name==String("min_size")){
        set_min_size(p_value);
        return true;
    }
    if(p_name==String("max_size")){
        set_max_size(p_value);
        return true;
    }
    return false;
}

Vector2i MTerrain::get_closest_pixel(const Vector3& world_pos){
    return grid->get_closest_pixel(world_pos);
}
void MTerrain::set_brush_manager(Object* input){
    ERR_FAIL_COND(!input->is_class("MBrushManager"));
    MBrushManager* brush_manager = Object::cast_to<MBrushManager>(input);
    grid->set_brush_manager(brush_manager);
}

void MTerrain::set_brush_start_point(Vector3 brush_pos,real_t radius){
    grid->set_brush_start_point(brush_pos,radius);
}

void MTerrain::draw_height(Vector3 brush_pos,real_t radius,int brush_id){
    if(!grid->is_created()) return;
    grid->draw_height(brush_pos,radius,brush_id);
}

void MTerrain::draw_color(Vector3 brush_pos,real_t radius,String brush_name, String uniform_name){
    if(!grid->is_created()) return;
    MColorBrush* brush = grid->get_brush_manager()->get_color_brush_by_name(brush_name);
    ERR_FAIL_COND(!brush);
    int id = get_image_id(uniform_name);
    ERR_FAIL_COND(id==-1);
    grid->draw_color(brush_pos,radius,brush,id);
    for(MGrass* g : confirm_grass_list){
        if(g->is_depend_on_image(id)){
            Vector2i px_pos = g->get_closest_pixel(brush_pos);
            int l = ceil(radius/g->grass_data->density);
            int left = px_pos.x - l;
            int right = px_pos.x + l;
            int top = px_pos.y - l;
            int bottom = px_pos.y + l;

            for(int x=left;x<=right;x++){
                for(int y=top;y<=bottom;y++){
                    g->make_grass_dirty_by_pixel(x,y);
                }
            }
            g->update_dirty_chunks();
        }
    }
}

Vector3 MTerrain::get_pixel_world_pos(uint32_t x,uint32_t y){
    return grid->get_pixel_world_pos(x,y);
}



void MTerrain::set_heightmap_layers(PackedStringArray input){
    ERR_FAIL_COND_MSG(grid->is_created(),"Can not change layers name when terrain grid is created!");
    grid->heightmap_layers.clear();
    if(input.size()==0){
        input.push_back("background");
    }
    if(input[0] != "background"){
        grid->heightmap_layers.clear();
        grid->heightmap_layers.push_back("background");
        grid->heightmap_layers.append_array(input);
        return;
    }
    grid->heightmap_layers = input;
}
const PackedStringArray& MTerrain::get_heightmap_layers(){
    return grid->heightmap_layers;
}

bool MTerrain::set_active_layer_by_name(String lname){
    ERR_FAIL_COND_V_MSG(!grid->is_created(),false,"Please call set_active_layer_by_name function after creating grid");
    return grid->set_active_layer(lname);
}

String MTerrain::get_active_layer_name(){
    ERR_FAIL_COND_V(!grid->is_created(),String());
    return grid->get_active_layer();
}

bool MTerrain::get_layer_visibility(String lname){
    int index = grid->heightmap_layers.find(lname);
    if(index==-1){
        return false;
    }
    if(index < grid->heightmap_layers_visibility.size()){
        return grid->heightmap_layers_visibility[index];
    }
    return false;
}

void MTerrain::add_heightmap_layer(String lname){
    ERR_FAIL_COND_EDMSG(grid->heightmap_layers.find(lname)!=-1,"Layer name must be unique");
    grid->add_heightmap_layer(lname);
    grid->heightmap_layers.push_back(lname);
}

bool MTerrain::rename_heightmap_layer(String old_name,String new_name){
    int layer_index = grid->heightmap_layers.find(old_name);
    ERR_FAIL_COND_V_MSG(layer_index==-1,false,"Can not find layer "+old_name);
    /// Renaming files
    Ref<DirAccess> dir = DirAccess::open(layersDataDir);
    PackedStringArray layers_file_names;
    // Getting a list of layer files path
    if(dir.is_valid()){
        dir->list_dir_begin();
        String fname = dir->get_next();
        while (fname != "")
        {
            if (!dir->current_is_dir()){
                layers_file_names.push_back(fname);
            }
            fname = dir->get_next();
        }
    }
    Ref<RegEx> reg;
    reg.instantiate();
    reg->compile(old_name+"_x\\d+_y\\d+\\.(res|r32)");
    for(int i=0; i < layers_file_names.size(); i++){
        String fname = layers_file_names[i];
        Ref<RegExMatch> result = reg->search(fname);
        if(result.is_valid()){
            String new_file_name = fname.replace(old_name,new_name);
            String old_path = layersDataDir.path_join(fname);
            String new_path = layersDataDir.path_join(new_file_name);
            Error err = dir->rename(old_path,new_path);
            ERR_FAIL_COND_V_MSG(err!=OK,false,"Can not rename layer file!");
        }
    }
    grid->heightmap_layers.set(layer_index,new_name);
    if(grid->is_created()){
        grid->rename_heightmap_layer(layer_index,new_name);
    }
    return true;
}

void MTerrain::merge_heightmap_layer(){
    ERR_FAIL_COND(!grid->is_created());
    grid->merge_heightmap_layer();
}

void MTerrain::remove_heightmap_layer(){
    ERR_FAIL_COND(!grid->is_created());
    grid->remove_heightmap_layer();
}

void MTerrain::toggle_heightmap_layer_visibile(){
    ERR_FAIL_COND(!grid->is_created());
    grid->toggle_heightmap_layer_visible();
}

void MTerrain::terrain_child_changed(Node* n){
    if(!is_ready){
        return;
    }
    if(n->is_class("MGrass")){
        MGrass* g = Object::cast_to<MGrass>(n);
        if(grass_list.find(g)==-1){
            terrain_ready_signal();
        }
    }
}

void MTerrain::terrain_ready_signal(){
    if(set_mtime){
        RenderingServer::get_singleton()->global_shader_parameter_add("mtime",RenderingServer::GlobalShaderParameterType::GLOBAL_VAR_TYPE_FLOAT,0.0);
        set_process(true);
    }
    // Grass part
    grass_list.clear(); // First make sure grass list is clear
    int child_count = get_child_count();
    for(int i=0;i<child_count;i++){
        if("MGrass" == get_child(i)->get_class()){
            MGrass* g = Object::cast_to<MGrass>(get_child(i));
            grass_list.push_back(g);
            g->recalculate_grass_config(max_h_scale_index - min_h_scale_index);
        }
    }
    is_ready = true;
    grid->update_renderer_info();
    /// Finish initlaztion start update
}

Vector2i MTerrain::get_region_grid_size(){
    return Vector2i(grid->get_region_grid_size().x,grid->get_region_grid_size().z);
}

int MTerrain::get_region_id_by_world_pos(const Vector3& world_pos){
    return grid->get_region_id_by_world_pos(world_pos);
}

int32_t MTerrain::get_base_size(){
    return size_list[min_size_index];
}
float MTerrain::get_h_scale(){
    return h_scale_list[min_h_scale_index];
}

int MTerrain::get_pixel_width(){
    #ifdef DEBUG_ENABLED
    if(!grid->is_created()){
        WARN_PRINT("First create grid and then get the pixel width");
    }
    #endif
    return grid->grid_pixel_region.get_width();
}

int MTerrain::get_pixel_height(){
    #ifdef DEBUG_ENABLED
    if(!grid->is_created()){
        WARN_PRINT("First create grid and then get the pixel height");
    }
    #endif
    return grid->grid_pixel_region.get_height();
}

void MTerrain::set_brush_layers(Array input){
    brush_layers = input;
}
Array MTerrain::get_brush_layers(){
    return brush_layers;
}
void MTerrain::set_brush_layers_num(int input){
    ERR_FAIL_COND(input<0);
    brush_layers.resize(input);
    notify_property_list_changed();
}
int MTerrain::get_brush_layers_num(){
    return brush_layers.size();
}

void MTerrain::set_set_mtime(bool input){
    set_mtime = input;
}
bool MTerrain::get_set_mtime(){
    return set_mtime;
}

Array MTerrain::get_layers_info(){
    Array info;
    for(int i=0;i<brush_layers.size();i++){
        Ref<MBrushLayers> layers = brush_layers[i];
        if(layers.is_valid()){
            Dictionary dic;
            dic["title"]=layers->layers_title;
            dic["index"]=i;
            dic["uniform"]=layers->uniform_name;
            dic["brush_name"]=layers->brush_name;
            dic["info"]=layers->get_layers_info();
            info.push_back(dic);
        }
    }
    return info;
}

void MTerrain::set_color_layer(int index,int group_index,String brush_name){
    ERR_FAIL_COND(!grid->is_created());
    ERR_FAIL_COND(!grid->get_brush_manager());
    MColorBrush* brush = grid->get_brush_manager()->get_color_brush_by_name(brush_name);
    ERR_FAIL_COND(!brush);
    Ref<MBrushLayers> bl = brush_layers[group_index];
    bl->set_layer(index,brush);
}

void MTerrain::disable_brush_mask(){
    grid->brush_mask_active = false;
}

void MTerrain::set_brush_mask(const Ref<Image>& img) {
    grid->brush_mask_active = true;
    grid->brush_mask = img;
}

void MTerrain::set_brush_mask_px_pos(Vector2i pos) {
    grid->brush_mask_px_pos = pos;
}

void MTerrain::set_mask_cutoff(float val){
    grid->mask_cutoff = val;
}

void MTerrain::images_add_undo_stage(){
    grid->images_add_undo_stage();
}
void MTerrain::images_undo(){
    grid->images_undo();
    VSet<int> changed_images;
    for(MImage* img : grid->last_images_undo_affected_list){
        changed_images.insert(img->index);
    }
    for(int i=0;i<changed_images.size();i++){
        int index = changed_images[i];
        for(MGrass* g : confirm_grass_list){
            if(g->is_depend_on_image(index)){
                g->recreate_all_grass();
            }
        }
    }
}

void MTerrain::set_terrain_material(Ref<MTerrainMaterial> input){
    ERR_FAIL_COND_EDMSG(grid->is_created(),"You should destroy terrain to change terrain material");
    terrain_material = input;
    if(terrain_material.is_valid()){
        terrain_material->set_grid(grid);
    }
}
Ref<MTerrainMaterial> MTerrain::get_terrain_material(){
    return terrain_material;
}

bool MTerrain::is_grid_created() {
    return grid->is_created();
}

Vector3 MTerrain::get_normal_by_pixel(uint32_t x,uint32_t y){
    ERR_FAIL_COND_V(!grid->is_created(),Vector3());
    return grid->get_normal_by_pixel(x,y);
}
Vector3 MTerrain::get_normal_accurate_by_pixel(uint32_t x,uint32_t y){
    ERR_FAIL_COND_V(!grid->is_created(),Vector3());
    return grid->get_normal_accurate_by_pixel(x,y);
}

Vector3 MTerrain::get_normal(const Vector3 world_pos){
    ERR_FAIL_COND_V(!grid->is_created(),Vector3());
    return grid->get_normal(world_pos);
}

Vector3 MTerrain::get_normal_accurate(Vector3 world_pos){
    ERR_FAIL_COND_V(!grid->is_created(),Vector3());
    return grid->get_normal_accurate(world_pos);
}


void MTerrain::update_all_dirty_image_texture(bool update_physics){
    ERR_FAIL_COND(!grid->is_created());
    grid->update_all_dirty_image_texture(update_physics);
}

void MTerrain::update_normals(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom){
    ERR_FAIL_COND(!grid->is_created());
    grid->update_normals(left, right, top,bottom);
}

void MTerrain::_notification(int32_t what){
    if(what == NOTIFICATION_PROCESS){
        if(set_mtime){
            RenderingServer::get_singleton()->global_shader_parameter_set("mtime",MGrass::get_shader_time());
        }
        return;
    }
    else if(what == NOTIFICATION_WM_CLOSE_REQUEST || what == NOTIFICATION_WM_GO_BACK_REQUEST){
        _finish_terrain();
        return;
    }
    else if(what == NOTIFICATION_VISIBILITY_CHANGED){
        _update_visibility();
    }
    
}

void MTerrain::_update_visibility(){
    if(!is_inside_tree()){
        grid->set_visibility(false);
        return;
    }
    grid->set_visibility(is_visible_in_tree());
}