#include "mgrid.h"

#include "core/variant/variant.h"
#include "scene/resources/shader.h"
#include "core/io/resource_loader.h"
#include "scene/resources/texture.h"
#include "core/io/resource_saver.h"
#include "core/variant/variant_utility.h"


#include "mbrush_manager.h"


MGrid::MGrid(){
    nvec8.append(Vector3(0,0,-1));
    nvec8.append(Vector3(-1,0,-1));
    nvec8.append(Vector3(-1,0,0));
    nvec8.append(Vector3(-1,0,1));
    nvec8.append(Vector3(0,0,1));
    nvec8.append(Vector3(1,0,1));
    nvec8.append(Vector3(1,0,0));
    nvec8.append(Vector3(1,0,-1));
    nvec8.append(Vector3(0,0,-1));
}
MGrid::~MGrid() {
    
}

uint64_t MGrid::get_update_id(){
    return _update_id;
}

void MGrid::clear() {
    if(is_dirty){
        RenderingServer* rs = RenderingServer::get_singleton();
        for(int32_t z=_search_bound.top; z <=_search_bound.bottom; z++)
        {
            for(int32_t x=_search_bound.left; x <=_search_bound.right; x++){
                if(points[z][x].has_instance){
                    rs->free(points[z][x].instance);
                    points[z][x].instance = RID();
                    points[z][x].mesh = RID();
                }
            }
        }
        memdelete_arr<MPoint>(points_row);
        memdelete_arr<MPoint*>(points);
        memdelete_arr<MRegion>(regions);
        memdelete(_chunks);
    }
    if(_terrain_material.is_valid()){
        _terrain_material->clear();
    }
    _size.x = 0;
    _size.y = 0;
    _size.z = 0;
    _grid_bound.clear();
    _search_bound.clear();
    _last_search_bound.clear();
    _region_grid_bound.clear();
    is_dirty = false;
    uniforms_id.clear();
    has_normals = false;
    _all_heightmap_image_list.clear();
    _all_image_list.clear();
    //heightmap_layers.clear();
    heightmap_layers_visibility.clear();
    active_heightmap_layer=0;
    current_undo_id =0;
    lowest_undo_id=0;
    _regions_count = 0;
    grid_pixel_region.clear();
    grid_update_info.clear();
    remove_instance_list.clear();
    update_mesh_list.clear();
    if(is_update_regions_future_valid && update_regions_future.valid()){
        update_regions_future.wait();
        is_update_regions_future_valid = false;
    }
}

bool MGrid::is_created() {
    return is_dirty;
}

MGridPos MGrid::get_size() {
    return _size;
}

void MGrid::set_scenario(RID scenario){
    _scenario = scenario;
}

RID MGrid::get_scenario(){
    return _scenario;
}

void MGrid::create(const int32_t width,const int32_t height, MChunks* chunks) {
    update_renderer_info();
    if (width == 0 || height == 0) return;
    _chunks = chunks;
    _size.x = width;
    _size.z = height;
    // Not added to one because pixels start from zero
    pixel_width = (uint32_t)(_size.x*_chunks->base_size_meter/_chunks->h_scale) + 1;
    pixel_height = (uint32_t)(_size.z*_chunks->base_size_meter/_chunks->h_scale) + 1;
    grid_pixel_region = MPixelRegion(pixel_width,pixel_height);
    _size_meter.x = width*_chunks->base_size_meter;
    _size_meter.z = height*_chunks->base_size_meter;
    _vertex_size.x = (_size_meter.x/chunks->h_scale) + 1;
    _vertex_size.z = (_size_meter.z/chunks->h_scale) + 1;
    _region_grid_size.x = _size.x/region_size + _size.x%region_size;
    _region_grid_size.z = _size.z/region_size + _size.z%region_size;
    _region_grid_bound.right  = _region_grid_size.x - 1;
    _region_grid_bound.bottom = _region_grid_size.z - 1;
    _regions_count = _region_grid_size.x*_region_grid_size.z;
    region_size_meter = region_size*_chunks->base_size_meter;
    rp = (region_size_meter/_chunks->h_scale);
    region_pixel_size = rp + 1;
    /// Checking if region pixel size is correct
    /// Also grabing the images name from the first terrain resource
    String first_res_path = dataDir.path_join("x0_y0.res");
    Array images_names;
    Ref<MResource> first_res;
    if(ResourceLoader::exists(first_res_path)){  // if not exist data directory is empty and a empty data will be created
        first_res = ResourceLoader::load(first_res_path);
        ERR_FAIL_COND_EDMSG(!first_res.is_valid(), "Data x0_y0.res is not valid MResource file ");
        images_names = first_res->get_data_names();
        for(int i=0; i < images_names.size(); i++){
            StringName _n = images_names[i];
            ERR_FAIL_COND_EDMSG(first_res->get_data_width(_n)!=region_pixel_size-1,String(_n)+" width "+itos(first_res->get_heightmap_width())+" in data directory does not match the your region size in pixel "+itos(region_pixel_size-1)+" Please change your regions size width so it match with data directory");
        }
    }
    _grid_bound = MBound(0,width-1, 0, height-1);
    regions = memnew_arr(MRegion, _regions_count);
    total_points_count = _size.z*_size.x;
    points_row = memnew_arr(MPoint, total_points_count);
    points = memnew_arr(MPoint*, _size.z);
    for (int32_t z=0; z<_size.z; z++){
        points[z] = &points_row[z*_size.x];
    }
    //Init Regions
    int index = 0;
    for(int32_t z=0; z<_region_grid_size.z; z++){
        for(int32_t x=0; x<_region_grid_size.x; x++){
            regions[index].id = index;
            regions[index].grid = this;
            regions[index].pos = MGridPos(x,0,z);
            regions[index].world_pos = get_world_pos(x*region_size,0,z*region_size);
            if(x!=0){
                regions[index].left = get_region(x-1,z);
            }
            if(x!=_region_grid_size.x-1){
                regions[index].right = get_region(x+1,z);
            }
            if(z!=0){
                regions[index].top = get_region(x,z-1);
            }
            if(z!=_region_grid_size.z-1){
                regions[index].bottom = get_region(x,z+1);
            }
            //if(_terrain_material.is_valid()){
            //    regions[index].set_material(_terrain_material->get_material(index));
            //}
            index++;
        }
    }
    is_dirty = true;
    _terrain_material->load_images(images_names,first_res);
    for(int i=0; i<_regions_count; i++){
        regions[i].configure();
    }
    _all_image_list = _terrain_material->all_images;
    _all_heightmap_image_list = _terrain_material->all_heightmap_images;
    //Initilazing Heightmap layers visibilty
    heightmap_layers_visibility.clear();
    heightmap_layers_visibility.resize(heightmap_layers.size());
    for(int i=0;i<heightmap_layers.size();i++){
        heightmap_layers_visibility.set(i,true);
        add_heightmap_layer(heightmap_layers[i]);
    }
    //for(int i=0;i<_regions_count;i++){
    //    regions[i].load();
    //}
    //generate_normals_thread(grid_pixel_region);
}

void MGrid::update_all_image_list(){
    _all_image_list.clear();
    VariantUtilityFunctions::_print("Region count is ",_regions_count );
    for(int i=0;i<_regions_count;i++){
        Vector<MImage*> rimgs = regions[i].images;
        for(int j=0;j<rimgs.size();j++){
            _all_image_list.push_back(rimgs[j]);
            if(rimgs[j]->name==HEIGHTMAP_NAME){
                _all_heightmap_image_list.push_back(rimgs[j]);
            }
        }
    }
}

Vector3 MGrid::get_world_pos(const int32_t x,const int32_t y,const int32_t z) {
    return Vector3(x,y,z)*_chunks->base_size_meter + offset;
}

Vector3 MGrid::get_world_pos(const MGridPos& pos){
    return Vector3(pos.x,pos.y,pos.z)*_chunks->base_size_meter + offset;
}

// Get point id non offset world posiotion usefull for grass for now
// in a flat x z plane
int MGrid::get_point_id_by_non_offs_ws(const Vector2& input){
    int x = ((int)(input.x))/_chunks->base_size_meter;
    int z = ((int)(input.y))/_chunks->base_size_meter;
    return z*_size.z + x;
}

int64_t MGrid::get_point_instance_id_by_point_id(int pid){
    return points_row[pid].instance.get_id();
}

MGridPos MGrid::get_grid_pos(const Vector3& pos) {
    MGridPos p;
    Vector3 _rp = pos - offset;
    p.x = ((int32_t)(_rp.x))/_chunks->base_size_meter;
    p.y = ((int32_t)(_rp.y))/_chunks->base_size_meter;
    p.z = ((int32_t)(_rp.z))/_chunks->base_size_meter;
    return p;
}

int32_t MGrid::get_regions_count(){
    return _regions_count;
}

MGridPos MGrid::get_region_grid_size(){
    return _region_grid_size;
}

int32_t MGrid::get_region_id_by_point(const int32_t x, const int32_t z) {
    return x/region_size + (z/region_size)*_region_grid_size.x;
}

MRegion* MGrid::get_region_by_point(const int32_t x, const int32_t z){
    int32_t id = x/region_size + (z/region_size)*_region_grid_size.x;
    return regions + id;
}

MBound MGrid::region_bound_to_point_bound(const MBound& rbound){
    MBound out;
    out.left = rbound.left * region_size;
    out.right = ((rbound.right + 1) * region_size) - 1;
    out.top = rbound.top * region_size;
    out.bottom = ((rbound.bottom + 1) * region_size) - 1;
    return out;
}

MRegion* MGrid::get_region(const int32_t x, const int32_t z){
    if(x < 0 || z < 0 || x >= _region_grid_size.x || z >= _region_grid_size.z ){
        return nullptr;
    }
    int32_t id = x + z*_region_grid_size.x;
    ERR_FAIL_INDEX_V(id,_regions_count,regions);
    return regions + id;
}

MGridPos MGrid::get_region_pos_by_world_pos(Vector3 world_pos){
    MGridPos p;
    world_pos -= offset;
    p.x = (int32_t)(world_pos.x/((real_t)region_size_meter));
    p.z = (int32_t)(world_pos.z/((real_t)region_size_meter));
    return p;
}

int MGrid::get_region_id_by_world_pos(Vector3 world_pos){
    world_pos -= offset;
    int32_t x = (int32_t)(world_pos.x/((real_t)region_size_meter));
    int32_t z = (int32_t)(world_pos.z/((real_t)region_size_meter));
    return x + z*_region_grid_size.x;
}

Vector2 MGrid::get_point_region_offset_ratio(int32_t x,int32_t z){
    int ofsx = x%region_size;
    int ofsz = z%region_size;
    Vector2 _offset;
    _offset.x = (double)(ofsx)/double(region_size);
    _offset.y = (double)(ofsz)/double(region_size);
    return _offset;
}

Vector3 MGrid::get_region_world_pos_by_point(int32_t x,int32_t z){
    int rx = x/region_size;
    int rz = z/region_size;
    Vector3 pos;
    pos.x = region_size_meter*rx + offset.x;
    pos.z = region_size_meter*rz + offset.z;
    return pos;
}

int8_t MGrid::get_lod_by_distance(const int32_t dis) {
    for(int8_t i=0 ; i < lod_distance.size(); i++){
        if(dis < lod_distance[i]){
            return i;
        }
    }
    return _chunks->max_lod;
}


void MGrid::set_cam_pos(const Vector3& cam_world_pos) {
    _cam_pos = get_grid_pos(cam_world_pos);
    _cam_pos_real = cam_world_pos;
}



void MGrid::update_search_bound() {
    MBound sb(_cam_pos, max_range, _size);
    _last_search_bound = _search_bound;
    _search_bound = sb;
    MBound rbl = region_bound_to_point_bound(current_region_bound);
    _search_bound.intersect(rbl);
    if(_search_bound.left == _search_bound.right || _search_bound.top == _search_bound.bottom){
        _search_bound.grow(_grid_bound,1,1);
    }
}

void MGrid::cull_out_of_bound() {
    RenderingServer* rs = RenderingServer::get_singleton();
    for(int32_t z=_last_search_bound.top; z <=_last_search_bound.bottom; z++)
    {
        for(int32_t x=_last_search_bound.left; x <=_last_search_bound.right; x++){
            if (!_search_bound.has_point(x,z)){
                if(points[z][x].has_instance){
                    //remove_instance_list.append(points[z][x].instance);
                    remove_instance_list.push_back(points[z][x].instance);
                    points[z][x].instance = RID();
                    points[z][x].has_instance = false;
                    points[z][x].mesh = RID();
                }
            }

        }
    }
}


void MGrid::update_lods() {
    /// First Some Clean up
    for(int32_t z=_search_bound.top; z <=_search_bound.bottom; z++){
        for(int32_t x=_search_bound.left; x <=_search_bound.right; x++){
            points[z][x].size = 0;
        }
    }
    //// Now Update LOD
    MGridPos closest = _grid_bound.closest_point_on_ground(_cam_pos);
    closest = get_3d_grid_pos_by_middle_point(closest);
    MBound m(closest);
    int8_t current_lod = 0;
    _cam_pos.y=0;//Discarding y on distance calculation due to NAN y value on holes
    //get_3d_grid_pos_by_middle_point function also return always point with y=0
    current_lod = get_lod_by_distance(m.center.get_distance(_cam_pos));
    if(!_grid_bound.has_point(_cam_pos))
    {
        Vector3 closest_real = get_world_pos(closest);
        Vector3 diff = _cam_pos_real - closest_real;
        m.grow_when_outside(diff.x, diff.z,_cam_pos, _search_bound,_chunks->base_size_meter);
        for(int32_t z =m.top; z <= m.bottom; z++){
            for(int32_t x=m.left; x <= m.right; x++){
                points[z][x].lod = current_lod;
                get_region_by_point(x,z)->insert_lod(current_lod);
            }
        }
    } else {
        points[m.center.z][m.center.x].lod = current_lod;
        get_region_by_point(m.center.x,m.center.z)->insert_lod(current_lod);
    }
    while (m.grow(_search_bound,1,1))
    {
        int8_t l;
        if(current_lod != _chunks->max_lod){
            MGridPos e = m.get_edge_point();
            e = get_3d_grid_pos_by_middle_point(e);
            int32_t dis = e.get_distance(_cam_pos);
            l = get_lod_by_distance(dis);
            //make sure that we are going one lod by one lod not jumping two lod
            if (l > current_lod +1){
                l = current_lod + 1;
            }
            // Also make sure to not jump back to lower lod when growing
            if(l<current_lod){
                l = current_lod;
            }
            current_lod = l;
        } else {
            l = _chunks->max_lod;
        }
        if(m.grow_left){
            for(int32_t z=m.top; z<=m.bottom;z++){
                points[z][m.left].lod = l;
                get_region_by_point(m.left,z)->insert_lod(l);
            }
        }
        if(m.grow_right){
            for(int32_t z=m.top; z<=m.bottom;z++){
                points[z][m.right].lod = l;
                get_region_by_point(m.right,z)->insert_lod(l);
            }
        }
        if(m.grow_top){
            for(int32_t x=m.left; x<=m.right; x++){
                points[m.top][x].lod = l;
                get_region_by_point(x,m.top)->insert_lod(l);
            }
        }
        if(m.grow_bottom){
            for(int32_t x=m.left; x<=m.right; x++){
                points[m.bottom][x].lod = l;
                get_region_by_point(x,m.bottom)->insert_lod(l);
            }
        }
    }
}

///////////////////////////////////////////////////////
////////////////// MERGE //////////////////////////////
void MGrid::merge_chunks() {
    for(int32_t z=_search_bound.top; z<=_search_bound.bottom; z++){
        for(int32_t x=_search_bound.left; x<=_search_bound.right; x++){
            int8_t lod = points[z][x].lod;
            MBound mb(x,z);
            int32_t region_id = get_region_id_by_point(x,z);
            #ifdef NO_MERGE
            check_bigger_size(lod,0, mb);
            num_chunks +=1;
            #else
            for(int8_t s=_chunks->max_size; s>=0; s--){
                if(_chunks->sizes[s].lods[lod].meshes.size()){
                    MBound test_bound = mb;
                    if(test_bound.grow_positive(pow(2,s) - 1, _search_bound)){
                        if(check_bigger_size(lod,s,region_id, test_bound)){
                            num_chunks +=1;
                            break;
                        }
                    }
                }
            }
            #endif
        }
    }
}

//This will check if all lod in this bound are the same if not return false
// also check if the right, bottom, top, left neighbor has the same lod level
// OR on lod level up otherwise return false
// Also if All condition are correct then we can merge to bigger size
// So this will set the size of all points except the first one to -1
// Also Here we should detrmine the edge of each mesh
bool MGrid::check_bigger_size(const int8_t lod,const int8_t size,const int32_t region_id, const MBound& bound) {
    for(int32_t z=bound.top; z<=bound.bottom; z++){
        for(int32_t x=bound.left; x<=bound.right; x++){
            if (points[z][x].lod != lod || points[z][x].size == -1 || get_region_id_by_point(x,z) != region_id)
            {
                return false;
            }
            // In this case region is not loaded or is going to be remove
            if(!regions[region_id].get_data_load_status() || regions[region_id].to_be_remove){
                if(points[z][x].has_instance){
                    points[z][x].has_instance = false;
                    remove_instance_list.push_back(points[z][x].instance);
                    points[z][x].mesh = RID();
                }
                return false;
            }
        }
    }
    // So these will determine if left, right, top, bottom edge should adjust to upper LOD
    bool right_edge = false;
    bool left_edge = false;
    bool top_edge = false;
    bool bottom_edge = false;
    // Check right neighbors
    int32_t right_neighbor = bound.right + 1;
    if (right_neighbor <= _search_bound.right){
        // Grab one sample from right neghbor
        int8_t last_right_lod = points[bound.bottom][right_neighbor].lod;
        // Now we don't care what is that all should be same
        for(int32_t z=bound.top; z<bound.bottom; z++){
            if(points[z][right_neighbor].lod != last_right_lod){
                return false;
            }
        }
        right_edge = (last_right_lod == lod + 1);
    }

    // Doing the same for bottom neighbor
    int32_t bottom_neighbor = bound.bottom + 1;
    if(bottom_neighbor <= _search_bound.bottom){
        int8_t last_bottom_lod = points[bottom_neighbor][bound.right].lod;
        for(int32_t x=bound.left; x<bound.right;x++){
            if(points[bottom_neighbor][x].lod != last_bottom_lod){
                return false;
            }
        }
        bottom_edge = (last_bottom_lod == lod + 1);
    }

    // Doing the same for left
    int32_t left_neighbor = bound.left - 1;
    if(left_neighbor >= _search_bound.left){
        int8_t last_left_lod = points[bound.bottom][left_neighbor].lod;
        for(int32_t z=bound.top;z<bound.bottom;z++){
            if(points[z][left_neighbor].lod != last_left_lod){
                return false;
            }
        }
        left_edge = (last_left_lod == lod + 1);
    }
    // WOW finally top neighbor
    int32_t top_neighbor = bound.top - 1;
    if(top_neighbor >= _search_bound.top){
        int8_t last_top_lod = points[top_neighbor][bound.right].lod;
        for(int32_t x=bound.left; x<bound.right; x++){
            if(points[top_neighbor][x].lod != last_top_lod){
                return false;
            }
        }
        top_edge = (last_top_lod == lod + 1);
    }
    // Now all the condition for creating this chunk with this size is true
    // So we start to build that
    // Top left corner will have one chunk with this size
    RenderingServer* rs = RenderingServer::get_singleton();
    RID merged_instance;
    for(int32_t z=bound.top; z<=bound.bottom; z++){
        for(int32_t x=bound.left;x<=bound.right;x++){
            if(z==bound.top && x==bound.left){
                points[z][x].size = size;
                int8_t edge_num = get_edge_num(left_edge,right_edge,top_edge,bottom_edge);
                RID mesh = _chunks->sizes[size].lods[lod].meshes[edge_num];
                if(points[z][x].mesh != mesh){
                    points[z][x].mesh = mesh;
                    if(points[z][x].has_instance){
                        //remove_instance_list.append(points[z][x].instance);
                        remove_instance_list.push_back(points[z][x].instance);
                        points[z][x].instance = RID();
                    }
                    int32_t region_id = get_region_id_by_point(x,z);
                    //MRegion* region = get_region_by_point(x,z);
                    points[z][x].create_instance(get_world_pos(x,0,z), _scenario, regions[region_id].get_material_rid(),is_visible);
                    rs->instance_set_visible(points[z][x].instance, false);
                    rs->instance_set_base(points[z][x].instance, mesh);
                    /*
                    Update info use for grass too
                    */
                   MGridUpdateInfo update_info;
                   update_info.terrain_instance = points[z][x].instance;
                   update_info.region_id = region_id;
                   update_info.region_world_pos = get_region_world_pos_by_point(x,z);
                   update_info.region_offset_ratio = get_point_region_offset_ratio(x,z);
                   update_info.lod = lod;
                   update_info.chunk_size = size;
                   MGridPos pos3d = get_3d_grid_pos_by_middle_point(MGridPos(x,0,z));
                   update_info.distance = _cam_pos.get_distance(pos3d);
                   grid_update_info.push_back(update_info);
                    update_mesh_list.push_back(points[z][x].instance);
                }
                merged_instance = points[z][x].instance;
            } else {
                points[z][x].size = -1;
                if(points[z][x].has_instance){
                    points[z][x].has_instance = false;
                    remove_instance_list.push_back(points[z][x].instance);
                    points[z][x].mesh = RID();
                }
                points[z][x].instance = merged_instance;
                points[z][x].has_instance = false;
            }
        }
    }
    return true;
}

int8_t MGrid::get_edge_num(const bool left,const bool right,const bool top,const bool bottom) {
    if(!left && !right && !top && !bottom){
        return M_MAIN;
    }
    if(left && !right && !top && !bottom){
            return M_L;
    }
    if(!left && right && !top && !bottom){
            return M_R;
    }
    if(!left && !right && top && !bottom){
            return M_T;
    }
    if(!left && !right && !top && bottom){
            return M_B;
    }
    if(left && !right && top && !bottom){
            return M_LT;
    }
    if(!left && right && top && !bottom){
            return M_RT;
    }
    if(left && !right && !top && bottom){
            return M_LB;
    }
    if(!left && right && !top && bottom){
            return M_RB;
    }
    if(left && right && top && bottom){
            return M_LRTB;
    }
    VariantUtilityFunctions::_print("Error Can not find correct Edge");
    VariantUtilityFunctions::_print(left, " ", right, " ", top, " ", bottom);
    return 0;
}

void MGrid::create_ordered_instances_distance(){
    instances_distance.clear();
    for(int32_t z=_search_bound.top; z<=_search_bound.bottom; z++){
        for(int32_t x=_search_bound.left;x<=_search_bound.right;x++){
            if(points[z][x].has_instance){
                MGridPos pos3d = get_3d_grid_pos_by_middle_point(MGridPos(x,0,z));
                int distance = _cam_pos.get_distance(pos3d);
                InstanceDistance ins_dis = {points[z][x].instance.get_id(),distance};
                instances_distance.ordered_insert(ins_dis);
            }
        }
    }
}


void MGrid::set_terrain_material(Ref<MTerrainMaterial> m) {
    m->set_grid(this);
    _terrain_material = m;
}

Ref<MTerrainMaterial> MGrid::get_terrain_material(){
    return _terrain_material;
}


MGridPos MGrid::get_3d_grid_pos_by_middle_point(MGridPos input) {
    MRegion* r = get_region_by_point(input.x,input.z);
    //Calculating middle point in chunks
    real_t half = ((real_t)_chunks->base_size_meter)/2;
    Vector3 middle_point_chunk = get_world_pos(input) + Vector3(half,0, half);
    //middle_point_chunk.y = r->get_closest_height(middle_point_chunk);
    return get_grid_pos(middle_point_chunk);
}

real_t MGrid::get_closest_height(const Vector3& pos) {
    MGridPos grid_pos = get_grid_pos(pos);
    if(!_grid_bound.has_point(grid_pos)){
        return 0;
    }
    MRegion* r = get_region_by_point(grid_pos.x, grid_pos.z);
    return r->get_closest_height(pos);
}

real_t MGrid::get_height(Vector3 pos){
    if(!is_created()){
        return 0.0;
    }
    pos -= offset;
    pos = pos/_chunks->h_scale;
    if(pos.x <0 || pos.z <0){
        return 0.0;
    }
    uint32_t x = (uint32_t)pos.x;
    uint32_t y = (uint32_t)pos.z;
    real_t hx0z0 = get_height_by_pixel(x,y);
    real_t hx1z0 = get_height_by_pixel(x+1,y);
    real_t hx0z1 = get_height_by_pixel(x,y+1);
    real_t hx1z1 = get_height_by_pixel(x+1,y+1);
    real_t factor_x = pos.x - floor(pos.x);
    real_t factor_z = pos.z - floor(pos.z);
    real_t ivaltop = (hx1z0 - hx0z0)*factor_x + hx0z0;
    real_t ivalbottom = (hx1z1 - hx0z1)*factor_x + hx0z1;
    return (ivalbottom - ivaltop)*factor_z + ivaltop;
}

Ref<MCollision> MGrid::get_ray_collision_point(Vector3 ray_origin,Vector3 ray_vector,real_t step,int max_step){
    ray_vector.normalize();
    Ref<MCollision> col;
    col.instantiate();
    float last_hieght=0.0;
    for(int i=0;i<max_step;i++){
        ray_origin += ray_vector*step;
        real_t terrain_height = get_height(ray_origin);
        //real_t terrain_height = get_closest_height(ray_origin);
        if(terrain_height > ray_origin.y){
            if(std::isnan(last_hieght)){
                col->collided = false;
                return col;
            }
            col->collided = true;
            break;
        }
        last_hieght = terrain_height;
    }
    if(col->collided){
        Vector3 las_pos = ray_origin - ray_vector*step;
        col->collision_position = las_pos.lerp(ray_origin, 0.5);
    }
    return col;
}

void MGrid::update_chunks(const Vector3& cam_pos) {
    std::lock_guard<std::mutex> lock(update_chunks_mutex);
    _update_id++;
    num_chunks = 0;
    update_mesh_list.clear();
    remove_instance_list.clear();
    grid_update_info.clear();
    set_cam_pos(cam_pos);
    update_search_bound();
    cull_out_of_bound();
    update_lods();
    for(int i=0; i < _regions_count; i++){
        regions[i].update_region();
    }
    merge_chunks();
    create_ordered_instances_distance();
}

void MGrid::update_regions(){
    for(MRegion* reg : unload_region_list){
        reg->unload();
        reg->is_data_loaded_reg_thread = false;
    }
    for(MRegion* reg : load_region_list){
        reg->load();
        reg->is_data_loaded_reg_thread = true;
    }
    for(MRegion* reg : load_region_list){
        //Should run one by one (Not in thread) as can intresect each other
        reg->correct_edges();
    }
    for(MRegion* reg : load_region_list){
        reg->is_edge_corrected = false;// This is use only correct_edges loop and out of that loop should be false
        //Should run one by one (Not in thread) as can intresect each other
        reg->recalculate_normals(true,true);
    }
    for(MRegion* reg : load_region_list){
        reg->set_data_load_status(true);
    }
}

void MGrid::update_regions_at_load(){
    for(MRegion* reg : unload_region_list){
        reg->unload();
        reg->is_data_loaded_reg_thread = false;
    }
    Vector<std::thread*> threads_pull;
    for(MRegion* reg : load_region_list){
        std::thread* t = new std::thread(&MRegion::load,reg);
        reg->is_data_loaded_reg_thread = true;
        threads_pull.push_back(t);
    }
    for(std::thread* t : threads_pull){
        t->join();
        delete t;
    }
    for(MRegion* reg : load_region_list){
        reg->correct_edges();
    }
    for(MRegion* reg : load_region_list){
        reg->is_edge_corrected = false;
        reg->recalculate_normals(true,false);
    }
    for(MRegion* reg : load_region_list){
        reg->set_data_load_status(true);
    }
}

void MGrid::apply_update_chunks() {
    for(int i=0; i < _regions_count; i++){
        regions[i].apply_update();
    }
    RenderingServer* rs = RenderingServer::get_singleton();
    for(RID rm: remove_instance_list){
        rs->free(rm);
    }
    if(is_visible){
        for(RID add : update_mesh_list){
            rs->instance_set_visible(add, true);
        }
    }
}

bool MGrid::update_regions_bounds(const Vector3& cam_pos,bool _make_neighbors_normals_dirty){
    if(_make_neighbors_normals_dirty){
        for(MRegion* reg : load_region_list){
            reg->make_neighbors_normals_dirty();
        }
    }
    for(MRegion* reg : unload_region_list){
        _terrain_material->remove_material(reg->id);
        reg->to_be_remove = false;
    }
    load_region_list.clear();
    unload_region_list.clear();
    MGridPos pos = get_region_pos_by_world_pos(cam_pos);
    MBound bound(pos);
    bound.left -= region_limit;
    bound.right += region_limit;
    bound.top -= region_limit;
    bound.bottom += region_limit;
    //Clear regions which are not in this bound but they where in last bound
    for(int32_t z=_last_region_grid_bound.top; z<=_last_region_grid_bound.bottom;z++){
        for(int32_t x=_last_region_grid_bound.left; x<=_last_region_grid_bound.right; x++){
            MGridPos p(x,0,z);
            if(_region_grid_bound.has_point(p) && !bound.has_point(p)){
                MRegion* reg = get_region(x,z);
                if(reg->get_data_load_status_relax()){
                    unload_region_list.push_back(reg);
                    reg->to_be_remove = true;
                }
            }
        }
    }
    _last_region_grid_bound = bound;
    for(int32_t z=bound.top; z<=bound.bottom;z++){
        for(int32_t x=bound.left; x<=bound.right; x++){
            MGridPos p(x,0,z);
            if(_region_grid_bound.has_point(p)){
                MRegion* reg = get_region(x,z);
                if(!reg->get_data_load_status_relax()){
                    load_region_list.push_back(reg);
                }
            }
        }
    }
    current_region_bound = bound;
    return load_region_list.size()!=0 || unload_region_list.size()!=0;
}

void MGrid::clear_region_bounds(){
    load_region_list.clear();
    unload_region_list.clear();
}

void MGrid::update_physics(const Vector3& cam_pos){
    MGridPos pos = get_region_pos_by_world_pos(cam_pos);
    MBound bound(pos);
    bound.left -= physics_update_limit;
    bound.right += physics_update_limit;
    bound.top -= physics_update_limit;
    bound.bottom += physics_update_limit;
    //Clear last physics if they are not in the current bound
    for(int32_t z=_last_physics_grid_bound.top; z<=_last_physics_grid_bound.bottom;z++){
        for(int32_t x=_last_physics_grid_bound.left; x<=_last_physics_grid_bound.right; x++){
            MGridPos p(x,0,z);
            if(_region_grid_bound.has_point(p) && !bound.has_point(p)){
                get_region(x,z)->remove_physics();
            }
        }
    }
    _last_physics_grid_bound = bound;
    for(int32_t z=bound.top; z<=bound.bottom;z++){
        for(int32_t x=bound.left; x<=bound.right; x++){
            MGridPos p(x,0,z);
            if(_region_grid_bound.has_point(p)){
                get_region(x,z)->create_physics();
            }
        }
    }
}

bool MGrid::_has_pixel(const uint32_t x,const uint32_t y){
    if(x>=pixel_width) return false;
    if(y>=pixel_height) return false;
    //if(x<0) return false; // There is no need for this as we use uint32
    //if(y<0) return false;
    return true;
}

bool MGrid::has_pixel(const uint32_t x,const uint32_t y){
    if(x>=pixel_width) return false;
    if(y>=pixel_height) return false;
    //if(x<0) return false;
    //if(y<0) return false;
    return true;
}

MImage* MGrid::get_image_by_pixel(uint32_t x,uint32_t y, const int32_t index){
    if(!_has_pixel(x,y)){
        return nullptr;
    }
    uint32_t ex = (uint32_t)(x%rp == 0 && x!=0);
    uint32_t ey = (uint32_t)(y%rp == 0 && y!=0);
    uint32_t rx = (x/rp) - ex;
    uint32_t ry = (y/rp) - ey;
    x -=rp*rx;
    y -=rp*ry;
    MRegion* r = get_region(rx,ry);
    return r->images[index];
}

Color MGrid::get_pixel(uint32_t x,uint32_t y, const int32_t index) {
    if(!_has_pixel(x,y)){
        return Color();
    }
    uint32_t ex = (uint32_t)(x%rp == 0 && x!=0);
    uint32_t ey = (uint32_t)(y%rp == 0 && y!=0);
    uint32_t rx = (x/rp) - ex;
    uint32_t ry = (y/rp) - ey;
    x -=rp*rx;
    y -=rp*ry;
    MRegion* r = get_region(rx,ry);
    return r->get_pixel(x,y,index);
}

const uint8_t* MGrid::get_pixel_by_pointer(uint32_t x,uint32_t y, const int32_t index){
    ERR_FAIL_COND_V(!_has_pixel(x,y),nullptr);
    uint32_t ex = (uint32_t)(x%rp == 0 && x!=0);
    uint32_t ey = (uint32_t)(y%rp == 0 && y!=0);
    uint32_t rx = (x/rp) - ex;
    uint32_t ry = (y/rp) - ey;
    x -=rp*rx;
    y -=rp*ry;
    return get_region(rx,ry)->images[index]->get_pixel_by_data_pointer(x,y);
}

void MGrid::set_pixel(uint32_t x,uint32_t y,const Color& col,const int32_t index) {
    if(!_has_pixel(x,y)){
        return;
    }
    bool ex = (x%rp == 0 && x!=0);
    bool ey = (y%rp == 0 && y!=0);
    uint32_t rx = (x/rp) - (uint32_t)ex;
    uint32_t ry = (y/rp) - (uint32_t)ey;
    x -=rp*rx;
    y -=rp*ry;
    MRegion* r = get_region(rx,ry);
    r->set_pixel(x,y,col,index);
    // Take care of the edges
    ex = (ex && rx != _region_grid_bound.right);
    // Same for ey
    ey = (ey && ry != _region_grid_bound.bottom);
    if(ex && ey){
        MRegion* re = get_region(rx+1,ry+1);
        re->set_pixel(0,0,col,index);
    }
    if(ex){
        MRegion* re = get_region(rx+1,ry);
        re->set_pixel(0,y,col,index);
    }
    if(ey){
        MRegion* re = get_region(rx,ry+1);
        re->set_pixel(x,0,col,index);
    }
}

void MGrid::set_pixel_by_pointer(uint32_t x,uint32_t y,uint8_t* ptr, const int32_t index){
    if(!_has_pixel(x,y)){
        return;
    }
    bool ex = (x%rp == 0 && x!=0);
    bool ey = (y%rp == 0 && y!=0);
    uint32_t rx = (x/rp) - (uint32_t)ex;
    uint32_t ry = (y/rp) - (uint32_t)ey;
    x -=rp*rx;
    y -=rp*ry;
    MRegion* r = get_region(rx,ry);
    r->images[index]->set_pixel_by_data_pointer(x,y,ptr);
    // Take care of the edges
    ex = (ex && rx != _region_grid_bound.right);
    // Same for ey
    ey = (ey && ry != _region_grid_bound.bottom);
    if(ex && ey){
        MRegion* re = get_region(rx+1,ry+1);
        re->images[index]->set_pixel_by_data_pointer(0,0,ptr);
    }
    if(ex){
        MRegion* re = get_region(rx+1,ry);
        re->images[index]->set_pixel_by_data_pointer(0,y,ptr);
    }
    if(ey){
        MRegion* re = get_region(rx,ry+1);
        re->images[index]->set_pixel_by_data_pointer(x,0,ptr);
    }
}

real_t MGrid::get_height_by_pixel(uint32_t x,uint32_t y) {
    if(!_has_pixel(x,y)){
        return 0.0;
    }
    uint32_t ex = (uint32_t)(x%rp == 0 && x!=0);
    uint32_t ey = (uint32_t)(y%rp == 0 && y!=0);
    uint32_t rx = (x/rp) - ex;
    uint32_t ry = (y/rp) - ey;
    x -=rp*rx;
    y -=rp*ry;
    MRegion* r = get_region(rx,ry);
    return r->get_height_by_pixel(x,y);
}

void MGrid::set_height_by_pixel(uint32_t x,uint32_t y,const real_t value){
    if(!_has_pixel(x,y)){
        return;
    }
    bool ex = (x%rp == 0 && x!=0);
    bool ey = (y%rp == 0 && y!=0);
    uint32_t rx = (x/rp) - (uint32_t)ex;
    uint32_t ry = (y/rp) - (uint32_t)ey;
    x -=rp*rx;
    y -=rp*ry;
    MRegion* r = get_region(rx,ry);
    r->set_height_by_pixel(x,y,value);
    // Take care of the edges
    // We dont want ex to be true at the edge of the terrain
    ex = (ex && rx != _region_grid_bound.right);
    // Same for ey
    ey = (ey && ry != _region_grid_bound.bottom);
    if(ex && ey){
        MRegion* re = get_region(rx+1,ry+1);
        re->set_height_by_pixel(0,0,value);
    }
    if(ex){
        MRegion* re = get_region(rx+1,ry);
        re->set_height_by_pixel(0,y,value);
    }
    if(ey){
        MRegion* re = get_region(rx,ry+1);
        re->set_height_by_pixel(x,0,value);
    }
}

real_t MGrid::get_height_by_pixel_in_layer(uint32_t x,uint32_t y){
    if(!_has_pixel(x,y)){
        return 0.0;
    }
    uint32_t ex = (uint32_t)(x%rp == 0 && x!=0);
    uint32_t ey = (uint32_t)(y%rp == 0 && y!=0);
    uint32_t rx = (x/rp) - ex;
    uint32_t ry = (y/rp) - ey;
    x -=rp*rx;
    y -=rp*ry;
    MRegion* r = get_region(rx,ry);
    return r->get_height_by_pixel_in_layer(x,y);
}

void MGrid::generate_normals_thread(MPixelRegion pxr) {
    Vector<MPixelRegion> px_regions = pxr.devide(4);
    Vector<std::thread*> threads_pull;
    for(int i=0;i<px_regions.size();i++){
        std::thread* t = new std::thread(&MGrid::generate_normals,this, px_regions[i]);
        threads_pull.append(t);
    }
    for(int i=0;i<threads_pull.size();i++){
        std::thread* t = threads_pull[i];
        t->join();
        delete t;
    }
}

void MGrid::generate_normals(MPixelRegion pxr) {
    //if(!uniforms_id.has("normals")) return;
    int id = _terrain_material->get_texture_id(NORMALS_NAME);
    ERR_FAIL_COND(id==-1);
    for(uint32_t y=pxr.top; y<=pxr.bottom; y++){
        for(uint32_t x=pxr.left; x<=pxr.right; x++){
            Vector2i px(x,y);
            real_t h = get_height_by_pixel(x,y);
            // Caculating face normals around point
            // and average them
            // In total there are 8 face around each point
            float heightL = _has_pixel(x-1,y) ? get_height_by_pixel(x-1,y) : h;
            float heightR = _has_pixel(x+1,y) ? get_height_by_pixel(x+1,y) : h;
            float heightU = _has_pixel(x,y-1) ? get_height_by_pixel(x,y-1) : h;
            float heightD = _has_pixel(x,y+1) ? get_height_by_pixel(x,y+1) : h;

            float heightTL = _has_pixel(x-1,y-1) ? get_height_by_pixel(x-1,y-1) : h;
            float heightTR = _has_pixel(x+1,y-1) ? get_height_by_pixel(x+1,y-1) : h;
            float heightBL = _has_pixel(x-1,y+1) ? get_height_by_pixel(x-1,y+1) : h;
            float heightBR = _has_pixel(x+1,y+1) ? get_height_by_pixel(x+1,y+1) : h;

            float nx = heightTL - heightBR + 2.0f * (heightL - heightR) + heightBL - heightTR;
            //float ny = 8.0f;
            float nz = heightTL - heightBR + 2.0f * (heightU - heightD) + heightTR - heightBL;
            float nl = sqrt(nx*nx + 64.0f + nz*nz);
            
            nx /= nl;
            float ny = 8.0f/nl;
            nz /= nl;

            nx = nx/2.0 + 0.5;
            nz = nz/2.0 + 0.5;

            // packing normals for image
            Color col(nx,ny,nz);
            set_pixel(x,y,col,id);
        }
    }
}

void MGrid::update_normals(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom){
    MPixelRegion pxr(left,right,top,bottom);
    generate_normals(pxr);
}

Vector3 MGrid::get_normal_by_pixel(uint32_t x,uint32_t y) {
    int id = _terrain_material->get_texture_id(NORMALS_NAME);
    ERR_FAIL_COND_V(id==-1,Vector3());
    Color c = get_pixel(x,y,id);
    Vector3 n(c.r*2.0 -1.0,c.g,c.b*2.0 -1.0);
    return n;
}

Vector3 MGrid::get_normal_accurate_by_pixel(uint32_t x,uint32_t y){
    Vector3 normal_vec;
    Vector2i px(x,y);
    real_t h = get_height_by_pixel(x,y);
    // Caculating face normals around point
    // and average them
    // In total there are 8 face around each point
    for(int i=0;i<nvec8.size()-1;i++){
        Vector2i px1(nvec8[i].x,nvec8[i].z);
        Vector2i px2(nvec8[i+1].x,nvec8[i+1].z);
        px1 += px;
        px2 += px;
        // Edge of the terrain
        if(!_has_pixel(px1.x,px1.y) || !_has_pixel(px2.x,px2.y)){
            continue;
        }
        Vector3 vec1 = nvec8[i];
        Vector3 vec2 = nvec8[i+1];
        vec1.y = get_height_by_pixel(px1.x,px1.y) - h;
        vec2.y = get_height_by_pixel(px2.x,px2.y) - h;
        normal_vec += vec1.cross(vec2);
    }
    normal_vec.normalize();
    return normal_vec;
}

Vector3 MGrid::get_normal(Vector3 world_pos){
    world_pos -= offset;
    world_pos = world_pos/_chunks->h_scale;
    if(world_pos.x <0 || world_pos.z <0){
        return Vector3(0,1,0);
    }
    uint32_t x = (uint32_t)world_pos.x;
    uint32_t y = (uint32_t)world_pos.z;
    Vector3 nx0z0 = get_normal_by_pixel(x,y);
    Vector3 nx1z0 = get_normal_by_pixel(x+1,y);
    Vector3 nx0z1 = get_normal_by_pixel(x,y+1);
    Vector3 nx1z1 = get_normal_by_pixel(x+1,y+1);
    real_t factor_x = world_pos.x - floor(world_pos.x);
    real_t factor_z = world_pos.z - floor(world_pos.z);
    Vector3 ivaltop = nx0z0.lerp(nx1z0,factor_x);
    Vector3 ivalbottom = nx0z1.lerp(nx1z1,factor_x);
    return ivaltop.lerp(ivalbottom,factor_z);
}

Vector3 MGrid::get_normal_accurate(Vector3 world_pos){
    world_pos -= offset;
    world_pos = world_pos/_chunks->h_scale;
    if(world_pos.x <0 || world_pos.z <0){
        return Vector3(0,1,0);
    }
    uint32_t x = (uint32_t)world_pos.x;
    uint32_t y = (uint32_t)world_pos.z;
    Vector3 nx0z0 = get_normal_accurate_by_pixel(x,y);
    Vector3 nx1z0 = get_normal_accurate_by_pixel(x+1,y);
    Vector3 nx0z1 = get_normal_accurate_by_pixel(x,y+1);
    Vector3 nx1z1 = get_normal_accurate_by_pixel(x+1,y+1);
    real_t factor_x = world_pos.x - floor(world_pos.x);
    real_t factor_z = world_pos.z - floor(world_pos.z);
    Vector3 ivaltop = nx0z0.lerp(nx1z0,factor_x);
    Vector3 ivalbottom = nx0z1.lerp(nx1z1,factor_x);
    return ivaltop.lerp(ivalbottom,factor_z);
}

void MGrid::save_image(int index,bool force_save){
    for(int i=0;i<_regions_count;i++){
        MRegion* reg = regions + i;
        String res_path = reg->get_res_path();
        Ref<MResource> mres;
        if(ResourceLoader::exists(res_path)){
            mres = ResourceLoader::load(res_path);
        }
        if(!mres.is_valid()){
            mres.instantiate();
        }
        reg->save_image(mres,index,force_save);
    }
}

bool MGrid::has_unsave_image(){
    for(int i=0;i<_all_image_list.size();i++){
        if(_all_image_list[i]->name==NORMALS_NAME){
            if(!_all_image_list[i]->is_save){
                return true;
            }
        } else {
            if(!_all_image_list[i]->is_save){
                return true;
            }
        }
    }
    return false;
}

void MGrid::save_all_dirty_images(){
   for(int i=0;i<_regions_count;i++){
        MRegion* reg = regions + i;
        String res_path = reg->get_res_path();
        Ref<MResource> mres;
        if(ResourceLoader::exists(res_path)){
            mres = ResourceLoader::load(res_path);
        }
        if(!mres.is_valid()){
            mres.instantiate();
        }
        bool is_something_saved = false;
        for(int j=0;j<reg->images.size();j++){
            MImage* img = reg->images[j];
            if(img->save(mres,false)){
                is_something_saved = true;
            }
        }
        if(is_something_saved){
            ResourceSaver::save(mres,res_path);
        }
   }
}

Vector2i MGrid::get_closest_pixel(Vector3 world_pos){
    world_pos -= offset;
    world_pos = world_pos/_chunks->h_scale;
    return Vector2i(round(world_pos.x),round(world_pos.z));
}

Vector3 MGrid::get_pixel_world_pos(uint32_t x,uint32_t y){
    Vector3 out;
    out.x = _chunks->h_scale*(real_t)x;
    out.z = _chunks->h_scale*(real_t)y;
    out += offset;
    out.y = get_height_by_pixel(x,y);
    return out;
}

void MGrid::set_brush_manager(MBrushManager* input){
    _brush_manager = input;
}

MBrushManager* MGrid::get_brush_manager(){
    return _brush_manager;
}

void MGrid::set_brush_start_point(Vector3 brush_pos,real_t radius){
    brush_world_pos_start = brush_pos;
    brush_radius = radius;
}

void MGrid::draw_height(Vector3 brush_pos,real_t radius,int brush_id){
    ERR_FAIL_COND(_brush_manager==nullptr);
    ERR_FAIL_COND_EDMSG(!heightmap_layers_visibility[active_heightmap_layer], "Can not paint on invisible layer");
    MHeightBrush* brush = _brush_manager->get_height_brush(brush_id);
    ERR_FAIL_COND_EDMSG(active_heightmap_layer!=0 && heightmap_layers[active_heightmap_layer]!="holes" && brush->_get_name() == "Hole","Only background layer or a layer with the name of holes support holes");
    ERR_FAIL_COND_EDMSG(heightmap_layers[active_heightmap_layer]=="holes" && brush->_get_name()!="Hole","In holes layer you can use only hole brush");
    Vector2i bpxpos = get_closest_pixel(brush_pos);
    if(bpxpos.x<0 || bpxpos.y<0 || bpxpos.x>grid_pixel_region.right || bpxpos.y>grid_pixel_region.bottom){
        return;
    }
    brush_px_pos_x = bpxpos.x;
    brush_px_pos_y = bpxpos.y;
    brush_px_radius = (uint32_t)(radius/_chunks->h_scale);
    brush_world_pos = brush_pos;
    brush_radius = radius;
    // Setting left right top bottom
    uint32_t left = (brush_px_pos_x>brush_px_radius) ? brush_px_pos_x - brush_px_radius : 0;
    uint32_t right = brush_px_pos_x + brush_px_radius;
    right = right > grid_pixel_region.right ? grid_pixel_region.right : right;
    uint32_t top = (brush_px_pos_y>brush_px_radius) ? brush_px_pos_y - brush_px_radius : 0;
    uint32_t bottom = brush_px_pos_y + brush_px_radius;
    bottom = (bottom>grid_pixel_region.bottom) ? grid_pixel_region.bottom : bottom;
    brush->set_grid(this);
    brush->before_draw();
    MPixelRegion _draw_pixel_region(left,right,top,bottom);
    //ERR_FAIL_COND_MSG(_draw_pixel_region.width!=_draw_pixel_region.height,"Non square brush is not supported");
    MImage* draw_image = memnew(MImage);
    {
        draw_image->create(_draw_pixel_region.get_width(),_draw_pixel_region.get_height(),Image::Format::FORMAT_RF);
        Vector<MPixelRegion> draw_pixel_regions = _draw_pixel_region.devide(4);
        Vector<MPixelRegion> local_pixel_regions;
        for(int i=0;i<draw_pixel_regions.size();i++){
            local_pixel_regions.append(_draw_pixel_region.get_local(draw_pixel_regions[i]));
        }
        Vector<std::thread*> threads_pull;
        for(int i=0;i<draw_pixel_regions.size();i++){
            std::thread* t = new std::thread(&MGrid::draw_height_region,this, draw_image,draw_pixel_regions[i],local_pixel_regions[i],brush);
            threads_pull.append(t);
        }
        for(int i=0;i<threads_pull.size();i++){
            std::thread* t = threads_pull[i];
            t->join();
            delete t;
        }
    }
    uint32_t local_x=0;
    uint32_t local_y=0;
    uint32_t x=_draw_pixel_region.left;
    uint32_t y=_draw_pixel_region.top;
    while(local_y<draw_image->height){
        while(local_x<draw_image->width){
            set_height_by_pixel(x,y,draw_image->get_pixel_RF(local_x,local_y));
            local_x++;
            x++;
        }
        local_x=0;
        x=draw_pixel_region.left;
        local_y++;
        y++;
    }
    memdelete(draw_image);
    _draw_pixel_region.grow_all_side(grid_pixel_region);
    generate_normals_thread(_draw_pixel_region);
    update_all_dirty_image_texture();
}


void MGrid::draw_height_region(MImage* img, MPixelRegion _draw_pixel_region, MPixelRegion local_pixel_region, MHeightBrush* brush){
    uint32_t local_x=local_pixel_region.left;
    uint32_t local_y=local_pixel_region.top;
    uint32_t x=_draw_pixel_region.left;
    uint32_t y=_draw_pixel_region.top;
    while(local_y<=local_pixel_region.bottom){
        while(local_x<=local_pixel_region.right){
            img->set_pixel_RF(local_x,local_y,brush->get_height(x,y));
            local_x++;
            x++;
        }
        local_x=local_pixel_region.left;
        x=_draw_pixel_region.left;
        local_y++;
        y++;
    }
}

void MGrid::draw_color(Vector3 brush_pos,real_t radius,MColorBrush* brush, int32_t index){
    ERR_FAIL_COND(_brush_manager==nullptr);
    ERR_FAIL_INDEX(index,regions[0].images.size());
    Image::Format format = regions[0].images[index]->format;
    current_paint_index = index;
    Vector2i bpxpos = get_closest_pixel(brush_pos);
    if(bpxpos.x<0 || bpxpos.y<0 || bpxpos.x>grid_pixel_region.right || bpxpos.y>grid_pixel_region.bottom){
        return;
    }
    brush_px_pos_x = bpxpos.x;
    brush_px_pos_y = bpxpos.y;
    brush_px_radius = (uint32_t)(radius/_chunks->h_scale);
    brush_world_pos = brush_pos;
    brush_radius = radius;
    // Setting left right top bottom
    uint32_t left = (brush_px_pos_x>brush_px_radius) ? brush_px_pos_x - brush_px_radius : 0;
    uint32_t right = brush_px_pos_x + brush_px_radius;
    right = right > grid_pixel_region.right ? grid_pixel_region.right : right;
    uint32_t top = (brush_px_pos_y>brush_px_radius) ? brush_px_pos_y - brush_px_radius : 0;
    uint32_t bottom = brush_px_pos_y + brush_px_radius;
    bottom = (bottom>grid_pixel_region.bottom) ? grid_pixel_region.bottom : bottom;
    // Stop here To go write a basic color brush
    brush->set_grid(this);
    draw_pixel_region = MPixelRegion(left,right,top,bottom);
    brush->before_draw();
    MImage* draw_image = memnew(MImage);
    {
        draw_image->create(draw_pixel_region.get_width(),draw_pixel_region.get_height(),format);
        Vector<MPixelRegion> draw_pixel_regions = draw_pixel_region.devide(4);
        Vector<MPixelRegion> local_pixel_regions;
        for(int i=0;i<draw_pixel_regions.size();i++){
            local_pixel_regions.append(draw_pixel_region.get_local(draw_pixel_regions[i]));
        }
        Vector<std::thread*> threads_pull;
        for(int i=0;i<draw_pixel_regions.size();i++){
            std::thread* t = new std::thread(&MGrid::draw_color_region,this, draw_image,draw_pixel_regions[i],local_pixel_regions[i],brush);
            threads_pull.append(t);
        }
        for(int i=0;i<threads_pull.size();i++){
            std::thread* t = threads_pull[i];
            t->join();
            delete t;
        }
    }
    uint32_t local_x=0;
    uint32_t local_y=0;
    uint32_t x=draw_pixel_region.left;
    uint32_t y=draw_pixel_region.top;
    int print_index = 0;
    while(local_y<draw_image->height){
        while(local_x<draw_image->width){
            uint32_t ofs = (local_x + local_y*draw_image->width)*draw_image->pixel_size;
            uint8_t* ptr = draw_image->data.ptrw() + ofs;
            set_pixel_by_pointer(x,y,ptr,index);
            local_x++;
            x++;
        }
        local_x=0;
        x=draw_pixel_region.left;
        local_y++;
        y++;
    }
    memdelete(draw_image);
    update_all_dirty_image_texture();
}
void MGrid::draw_color_region(MImage* img, MPixelRegion _draw_pixel_region, MPixelRegion local_pixel_region, MColorBrush* brush){
    uint32_t local_x=local_pixel_region.left;
    uint32_t local_y=local_pixel_region.top;
    uint32_t x=_draw_pixel_region.left;
    uint32_t y=_draw_pixel_region.top;
    while(local_y<=local_pixel_region.bottom){
        while(local_x<=local_pixel_region.right){
            brush->set_color(local_x,local_y,x,y,img);
            local_x++;
            x++;
        }
        local_x=local_pixel_region.left;
        x=_draw_pixel_region.left;
        local_y++;
        y++;
    }
}

//&MGrid::generate_normals,this, px_regions[i]
void MGrid::update_all_dirty_image_texture(bool update_physics){
    Vector<std::thread*> threads_pull;
    for(int i=0;i<_all_image_list.size();i++){
        if(_all_image_list[i]->is_dirty){
            //_all_image_list[i]->update_texture(_all_image_list[i]->current_scale,true);
            std::thread* t = new std::thread(&MImage::update_texture,_all_image_list[i],_all_image_list[i]->current_scale,true);
            threads_pull.push_back(t);
            if(_all_image_list[i]->name == HEIGHTMAP_NAME && update_physics){
                MRegion* reg = _all_image_list[i]->region;
                std::thread* pt = new std::thread(&MRegion::update_physics,reg);
                threads_pull.push_back(pt);
            }
        }
    }
    for(int i=0;i<threads_pull.size();i++){
        threads_pull[i]->join();
        delete threads_pull[i];
    }
}

bool MGrid::set_active_layer(String input){
    // We did not add background to heightmap layers so the error handling down here is ok
    int index = heightmap_layers.find(input);
    if(index != -1){
        active_heightmap_layer = index;
    } else {
        ERR_FAIL_V_MSG(false,"Active Layer "+input+"is not found");
        active_heightmap_layer = 0;
    }
    for(int i=0;i<_all_heightmap_image_list.size();i++){
        _all_heightmap_image_list[i]->set_active_layer(active_heightmap_layer);
    }
    return index!=-1;
}

String MGrid::get_active_layer(){
    ERR_FAIL_INDEX_V(active_heightmap_layer, heightmap_layers.size(),"background");
    return heightmap_layers[active_heightmap_layer];
}

// heightmap_layers will be set in MTerrain!
// This function should be called only from MTerrain
void MGrid::add_heightmap_layer(String lname){
    for(int i=0;i<_all_heightmap_image_list.size();i++){
        _all_heightmap_image_list[i]->add_layer(lname);
    }
    heightmap_layers_visibility.push_back(true);
}

// This function should be called only from MTerrain
void MGrid::rename_heightmap_layer(int layer_index,String lname){
    for(int i=0;i<_all_heightmap_image_list.size();i++){
        _all_heightmap_image_list[i]->rename_layer(layer_index,lname);
    }
    heightmap_layers.set(layer_index,lname);
}

void MGrid::merge_heightmap_layer(){
    ERR_FAIL_COND_EDMSG(active_heightmap_layer == 0,"Can't merge background layer");
    ERR_FAIL_COND_EDMSG(!heightmap_layers_visibility[active_heightmap_layer],"Make the layer visible first and then merge");
    // Merging current active layer
    for(int i=0;i<_all_heightmap_image_list.size();i++){
        _all_heightmap_image_list[i]->merge_layer();
    }
    heightmap_layers.remove_at(active_heightmap_layer);
    heightmap_layers_visibility.remove_at(active_heightmap_layer);
}

void MGrid::remove_heightmap_layer(){
    ERR_FAIL_COND_EDMSG(active_heightmap_layer == 0,"Can't remove background layer");
    // Removing current active layer
    for(int i=0;i<_all_heightmap_image_list.size();i++){
        _all_heightmap_image_list[i]->remove_layer(heightmap_layers_visibility[active_heightmap_layer]);
        // Correcting normals if is dirty
        if(_all_heightmap_image_list[i]->is_dirty){
            _all_heightmap_image_list[i]->region->recalculate_normals();
        }
    }
    update_all_dirty_image_texture();
    heightmap_layers.remove_at(active_heightmap_layer);
    heightmap_layers_visibility.remove_at(active_heightmap_layer);
}

void MGrid::toggle_heightmap_layer_visible(){
    ERR_FAIL_COND_EDMSG(active_heightmap_layer==0, "Can not Hide background layer");
    bool input = !heightmap_layers_visibility[active_heightmap_layer];
    for(int i=0;i<_all_heightmap_image_list.size();i++){
        _all_heightmap_image_list[i]->layer_visible(input);
        // Correcting normals if is dirty
        if(_all_heightmap_image_list[i]->is_dirty){
            _all_heightmap_image_list[i]->region->recalculate_normals();
        }
    }
    heightmap_layers_visibility.set(active_heightmap_layer,input);
    update_all_dirty_image_texture();
}

bool MGrid::is_layer_visible(int index){
    return heightmap_layers_visibility[index];
}

bool MGrid::is_layer_visible(const String& lname){
    int index = heightmap_layers.find(lname);
    ERR_FAIL_COND_V(index==-1,false);
    return heightmap_layers_visibility[index];
}

float MGrid::get_h_scale(){
    return _chunks->h_scale;
}

float MGrid::get_brush_mask_value(uint32_t x,uint32_t y){
    if(!brush_mask_active){
        return 1.0;
    }
    Vector2i vpos = Vector2i(x,y) - brush_mask_px_pos;
    if(vpos.x < 0 || vpos.y < 0 || vpos.x >= brush_mask->get_width() || vpos.y >= brush_mask->get_height()){
        return 0.0;
    }
    return brush_mask->get_pixel(vpos.x,vpos.y).r;
}

bool MGrid::get_brush_mask_value_bool(uint32_t x,uint32_t y){
    if(!brush_mask_active){
        return true;
    }
    Vector2i vpos = Vector2i(x,y) - brush_mask_px_pos;
    if(vpos.x < 0 || vpos.y < 0 || vpos.x >= brush_mask->get_width() || vpos.y >= brush_mask->get_height()){
        return false;
    }
    return brush_mask->get_pixel(vpos.x,vpos.y).r > mask_cutoff;
}

void MGrid::images_add_undo_stage(){
    if(current_undo_id - lowest_undo_id > 5){
        for(int i=0;i<_all_image_list.size();i++){
            _all_image_list[i]->remove_undo_data(lowest_undo_id);
        }
        lowest_undo_id++;
    }
    for(int i=0;i<_all_image_list.size();i++){
        _all_image_list[i]->current_undo_id = current_undo_id;
    }
    current_undo_id++;
}

void MGrid::images_undo(){
    last_images_undo_affected_list.clear();
    if(current_undo_id <= lowest_undo_id){
        // No more undo data
        return;
    }
    current_undo_id--;
    for(int i=0;i<_all_image_list.size();i++){
        if(_all_image_list[i]->go_to_undo(current_undo_id)) {
            last_images_undo_affected_list.push_back(_all_image_list[i]);
        }
        _all_image_list[i]->remove_undo_data(current_undo_id);
        
    }
    update_all_dirty_image_texture();
}

void MGrid::refresh_all_regions_uniforms(){
    for(int i=0;i<_regions_count;i++){
        regions[i].refresh_all_uniforms();
    }
}

void MGrid::update_renderer_info(){
    _is_opengl = RenderingServer::get_singleton()->get_rendering_device() == nullptr;
}

bool MGrid::is_opengl(){
    return _is_opengl;
}

bool MGrid::get_visibility(){
    return is_visible;
}

void MGrid::set_visibility(bool input){
    if(!is_created()){
        is_visible = input;
        return;
    }
    if(input == is_visible){
        return;
    }
    std::lock_guard<std::mutex> lock(update_chunks_mutex);
    MBound current_bound = _search_bound;
    current_bound.merge(_last_search_bound);
    for(int32_t z=current_bound.top; z <=current_bound.bottom; z++)
    {
        for(int32_t x=current_bound.left; x <=current_bound.right; x++){
            if(points[z][x].has_instance){
                RenderingServer::get_singleton()->instance_set_visible(points[z][x].instance,input);
            }
        }
    }
    is_visible = input;
}