#include "mcurve_mesh.h"

#include "servers/rendering_server.h"
#define RSS RenderingServer::get_singleton()


Vector<MCurveMesh*> MCurveMesh::all_curve_mesh_nodes;
void MCurveMesh::_bind_methods(){
    ClassDB::bind_method(D_METHOD("_update_visibilty"), &MCurveMesh::_update_visibilty);
    ClassDB::bind_method(D_METHOD("_on_connections_updated"), &MCurveMesh::_on_connections_updated);
    ClassDB::bind_method(D_METHOD("_swap_point_id","p_a","p_b"), &MCurveMesh::_swap_point_id);
    ClassDB::bind_method(D_METHOD("_id_force_update","id"), &MCurveMesh::_id_force_update);
    ClassDB::bind_method(D_METHOD("_point_force_update","p_index"), &MCurveMesh::_point_force_update);
    ClassDB::bind_method(D_METHOD("_connection_force_update","conn_id"), &MCurveMesh::_connection_force_update);
    ClassDB::bind_method(D_METHOD("_point_remove","p_index"), &MCurveMesh::_point_remove);
    ClassDB::bind_method(D_METHOD("_connection_remove","conn_id"), &MCurveMesh::_connection_remove);

    ClassDB::bind_method(D_METHOD("_on_curve_changed"), &MCurveMesh::_on_curve_changed);
    ClassDB::bind_method(D_METHOD("_generate_all_mesh_sliced_info"), &MCurveMesh::_generate_all_mesh_sliced_info);

    ClassDB::bind_method(D_METHOD("set_overrides","input"), &MCurveMesh::set_overrides);
    ClassDB::bind_method(D_METHOD("get_overrides"), &MCurveMesh::get_overrides);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"override",PROPERTY_HINT_RESOURCE_TYPE,"MCurveMeshOverride"),"set_overrides","get_overrides");

    ClassDB::bind_method(D_METHOD("set_intersections","input"), &MCurveMesh::set_intersections);
    ClassDB::bind_method(D_METHOD("get_intersections"), &MCurveMesh::get_intersections);
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY,"intersections",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE),"set_intersections","get_intersections");

    ClassDB::bind_method(D_METHOD("set_meshses","input"), &MCurveMesh::set_meshes);
    ClassDB::bind_method(D_METHOD("get_meshses"), &MCurveMesh::get_meshes);
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY,"meshes",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE),"set_meshses","get_meshses");

    ClassDB::bind_method(D_METHOD("set_materials","input"), &MCurveMesh::set_materials);
    ClassDB::bind_method(D_METHOD("get_materials"), &MCurveMesh::get_materials);
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY,"materials",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE),"set_materials","get_materials");

    ClassDB::bind_method(D_METHOD("dumy_set_restart","input"), &MCurveMesh::dumy_set_restart);
    ClassDB::bind_method(D_METHOD("dumy_get_true"), &MCurveMesh::dumy_get_true);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"restart"), "dumy_set_restart","dumy_get_true");

    ClassDB::bind_method(D_METHOD("reload"), &MCurveMesh::reload);
    ClassDB::bind_method(D_METHOD("recreate"), &MCurveMesh::recreate);

    ClassDB::bind_static_method("MCurveMesh",D_METHOD("get_all_curve_mesh_nodes"), &MCurveMesh::get_all_curve_mesh_nodes);
}

TypedArray<MCurveMesh> MCurveMesh::get_all_curve_mesh_nodes(){
    TypedArray<MCurveMesh> out;
    for(MCurveMesh* c : all_curve_mesh_nodes){
        if(c->is_inside_tree()){
            out.push_back(c);
        }
    }
    return out;
}

// after calling this sliced_pos and slice_info are invalid and should be recalculate
void MeshSlicedInfo::merge_vertex_by_distance(float merge_distance){
    return;
    ERR_FAIL_COND_MSG(sliced_pos.size() > 0,"Can not merge vertecies slice_pos is already created!");
    int32_t merge_index_first = -1;
    int32_t merge_index_second = -1;
    for(int i=0; i < vertex.size(); i++){
        for(int j=i+1; j < vertex.size(); j++){
            if(vertex[i].distance_to(vertex[j]) < merge_distance){
                merge_index_first = i;
                merge_index_second = j;
                break;
            }
        }
        if(merge_index_first!=-1){
            break;
        }
    }
    if(merge_index_first==-1){
        return; // No more mergeable
    }
    //VariantUtilityFunctions::print(merge_index_first," Removing ",merge_index_second);
    vertex.remove_at(merge_index_second);
    if(normal.size() > 0){
        Vector3 first_normal = normal[merge_index_first] + normal[merge_index_second];
        first_normal.normalize();
        normal.set(merge_index_first,first_normal);
        normal.remove_at(merge_index_second);
    }
    if(tangent.size() > 0){
        int32_t tfirst = merge_index_first * 4;
        int32_t tsecond = merge_index_second * 4;
        Vector3 first_tangent = Vector3(tangent[tfirst],tangent[tfirst+1],tangent[tfirst+2]);
        first_tangent += Vector3(tangent[tsecond],tangent[tsecond+1],tangent[tsecond+2]);
        first_tangent.normalize();
        tangent.set(tfirst,first_tangent.x);
        tangent.set(tfirst+1,first_tangent.y);
        tangent.set(tfirst+2,first_tangent.z);
        //tangent.set(tfirst+3,///Remain same);

        // Always should remove from bigger to smaller!
        tangent.remove_at(tsecond+3);
        tangent.remove_at(tsecond+2);
        tangent.remove_at(tsecond+1);
        tangent.remove_at(tsecond);
    }
    if(color.size() > 0) color.remove_at(merge_index_second);
    if(uv.size() > 0) uv.remove_at(merge_index_second);
    if(uv2.size() > 0) uv2.remove_at(merge_index_second);
    // correcting indecies
    for(int i=0; i < index.size(); i++){
        if(index[i] > merge_index_second){
            index.set(i,index[i] - 1);
        } else if(index[i] == merge_index_second){
            index.set(i,merge_index_first);
        }
    }
    merge_vertex_by_distance(merge_distance);
}

void MeshSlicedInfo::clear(){
    mesh_rid = RID();
    vertex.clear();
    normal.clear();
    tangent.clear();
    color.clear();
    uv.clear();
    uv2.clear();
    index.clear();
    sliced_pos.clear();
    sliced_info.clear();
}

int MeshSlicedInfo::slice_count() const{
    return sliced_pos.size();
}

void MeshSlicedInfo::get_color(int mesh_count,PackedColorArray& input){
    if(color.size()==0){
        input.resize(0);
        return;
    }
    input.resize(mesh_count * color.size());
    size_t block_size = color.size() * sizeof(Color);
    for(int i=0; i < mesh_count; i++){
        Color* ptrw = input.ptrw() + (color.size() * i);
        memcpy(ptrw,color.ptr(),block_size);
    }
}
void MeshSlicedInfo::get_uv(int mesh_count,PackedVector2Array& input){
    if(uv.size()==0){
        input.resize(0);
        return;
    }
    input.resize(mesh_count * uv.size());
    size_t block_size = uv.size() * sizeof(Vector2);
    for(int i=0; i < mesh_count; i++){
        Vector2* ptrw = input.ptrw() + (uv.size() * i);
        memcpy(ptrw,uv.ptr(),block_size);
    }
}
void MeshSlicedInfo::get_uv2(int mesh_count,PackedVector2Array& input){
    if(uv2.size()==0){
        input.resize(0);
        return;
    }
    input.resize(mesh_count * uv2.size());
    size_t block_size = uv2.size() * sizeof(Vector2);
    for(int i=0; i < mesh_count; i++){
        Vector2* ptrw = input.ptrw() + (uv2.size() * i);
        memcpy(ptrw,uv2.ptr(),block_size);
    }
}
void MeshSlicedInfo::get_index(int mesh_count,PackedInt32Array& input){
    ERR_FAIL_COND(index.size()==0);
    input.resize(index.size() * mesh_count);
    memcpy(input.ptrw(),index.ptr(),index.size()*sizeof(int32_t)); // copying first row as it does not change
    int vertex_count = vertex.size();
    for(int i=1; i < mesh_count; i++){
        int start_index_index = index.size() * i;
        int index_jump = vertex_count * i;
        for(int j=0; j < index.size(); j++){
            int index_index = start_index_index + j;
            input.set(index_index, index[j] + index_jump);
        }
    }
}

MCurveMesh::MCurveMesh(){
    connect("tree_exited", Callable(this, "_update_visibilty"));
    connect("tree_entered", Callable(this, "_update_visibilty"));
    all_curve_mesh_nodes.push_back(this);
}

MCurveMesh::~MCurveMesh(){
    clear();
    for(int i=0; i < all_curve_mesh_nodes.size(); i++){
        if(this == all_curve_mesh_nodes[i]){
            all_curve_mesh_nodes.remove_at(i);
            break;
        }
    }
}

void MCurveMesh::_on_connections_updated(){
    std::lock_guard<std::recursive_mutex> lock(update_mutex);
    ERR_FAIL_COND(curve.is_null());
    //ERR_FAIL_COND(!path->is_inside_tree());
    thread_task_id = WorkerThreadPool::get_singleton()->add_native_task(&MCurveMesh::thread_update,(void*)this);
    is_thread_updating = true;
    set_process(true);
}

void MCurveMesh::thread_update(void* input){
    MCurveMesh* curve_mesh = (MCurveMesh*)input;
    std::lock_guard<std::recursive_mutex> lock(curve_mesh->update_mutex);
    ERR_FAIL_COND(curve_mesh->curve.is_null());
    Ref<MCurve> curve = curve_mesh->curve;
    for(int i=0; i < curve->point_update.size(); i++){
        curve_mesh->_generate_intersection(curve->point_update[i]);
    }
    for(int i=0; i < curve->conn_update.size(); i++){
        curve_mesh->_generate_connection(curve->conn_update[i]);
    }
}


void MCurveMesh::_generate_all_mesh_sliced_info(){
    meshlod_sliced_info.clear();
    meshlod_sliced_info.resize(meshes.size());
    HashMap<RID,Ref<MeshSlicedInfo>> mesh_to_mesh_slice;
    for(int i=0; i < meshes.size(); i++){
        MeshSlicedInfoArray* info_array = meshlod_sliced_info.ptrw() + i;
        Ref<MMeshLod> mesh_lod = meshes[i];
        if(mesh_lod.is_null() || mesh_lod->get_meshes().size() == 0){
            info_array->resize(0);
            continue;
        }
        TypedArray<Mesh> p_meshes = mesh_lod->get_meshes();
        info_array->resize(p_meshes.size());
        for(int j=0; j < p_meshes.size(); j++){
            Ref<Mesh> p_mesh = p_meshes[j];
            if(p_mesh.is_null()){
                Ref<MeshSlicedInfo> empty;
                info_array->set(j,empty);
                continue;
            }
            if(mesh_to_mesh_slice.has(p_mesh->get_rid())){
                info_array->set(j,mesh_to_mesh_slice[p_mesh->get_rid()]);
                continue;
            }
            Ref<MeshSlicedInfo> s = _generate_mesh_sliced_info(p_mesh);
            info_array->set(j,s);
            mesh_to_mesh_slice.insert(p_mesh->get_rid(),s);
        }
    }
}

Ref<MeshSlicedInfo> MCurveMesh::_generate_mesh_sliced_info(Ref<Mesh> mesh){
    Ref<MeshSlicedInfo> s;
    s.instantiate();
    s->mesh_rid = mesh->get_rid();
    s->material = mesh->surface_get_material(0);
    ERR_FAIL_COND_V(mesh.is_null(),s);
    Array mesh_data = mesh->surface_get_arrays(0);
    s->vertex = mesh_data[Mesh::ARRAY_VERTEX];
    s->normal = mesh_data[Mesh::ARRAY_NORMAL];
    s->tangent = mesh_data[Mesh::ARRAY_TANGENT];
    s->color = mesh_data[Mesh::ARRAY_COLOR];
    s->uv = mesh_data[Mesh::ARRAY_TEX_UV];
    s->uv2 = mesh_data[Mesh::ARRAY_TEX_UV2];
    s->index = mesh_data[Mesh::ARRAY_INDEX];
    ERR_FAIL_COND_V(s->vertex.size() < 3,s);
    ERR_FAIL_COND_V(s->index.size() < 3,s);
    ERR_FAIL_COND_V(s->vertex.size() != s->normal.size() && s->normal.size()!=0,s);
    ERR_FAIL_COND_V(s->vertex.size() * 4 != s->tangent.size() && s->tangent.size()!=0,s);
    ERR_FAIL_COND_V(s->vertex.size() != s->color.size() && s->color.size()!=0,s);
    ERR_FAIL_COND_V(s->vertex.size() != s->uv.size() && s->uv.size()!=0,s);
    ERR_FAIL_COND_V(s->vertex.size() != s->uv2.size() && s->uv2.size()!=0,s);
    // Merging vertecies
    s->merge_vertex_by_distance();
    // Creating slices
    Vector<Pair<float,int>> sliced_pos_indicies;
    float biggest_x = s->vertex[0].x;
    for(int i=0; i < s->vertex.size(); i++){
        Vector3 vec = s->vertex[i];
        if(vec.x > biggest_x){
            biggest_x = vec.x;
        }
        bool has_slice = false;
        for(int j=0; j < sliced_pos_indicies.size(); j++){
            Pair<float,int> pi = sliced_pos_indicies[j];
            //UtilityFunctions::print("checking-----", std::abs(pi.first - vec.x) < SLICE_EPSILONE);
            if(std::abs(pi.first - vec.x) < SLICE_EPSILONE){
                Vector<int32_t>* s_ptrw = s->sliced_info.ptrw() + pi.second;
                s_ptrw->push_back(i); // pushing back vertex index
                has_slice = true;
                continue;
            }
        }
        if(has_slice){
            continue;
        }
        // Creating a new slice Vector
        int slice_index = s->sliced_info.size();
        s->sliced_info.resize(s->sliced_info.size() + 1);
        s->sliced_pos.push_back(vec.x);
        Vector<int32_t>* s_ptrw = s->sliced_info.ptrw() + slice_index;
        s_ptrw->push_back(i); // pushing back vertex index in newly created slice
        Pair<float,int> p(vec.x,slice_index);
        sliced_pos_indicies.push_back(p);
    }
    // setting the mesh lenght
    s->lenght = biggest_x;
    // setting vertex x pos to zero as we keep them inside s->sliced_pos
    for(int i=0; i < s->vertex.size(); i++){
        Vector3 vec = s->vertex[i];
        vec.x = 0;
        s->vertex.set(i,vec);
    }
    return s;
}

void MCurveMesh::_update_visibilty(){
    bool v = path->is_visible() && path->is_inside_tree() && is_inside_tree();
    for(HashMap<int64_t,Instance>::Iterator it=curve_mesh_instances.begin();it!=curve_mesh_instances.end();++it){
        RSS->instance_set_visible(it->value.instance,v);
    }
}

void MCurveMesh::_apply_update(){
    std::lock_guard<std::recursive_mutex> lock(update_mutex);
    for(int i=0; i < mesh_updated_list.size(); i++){
        ERR_CONTINUE(!curve_mesh_instances.has(mesh_updated_list[i].first));
        MCurveMesh::Instance c_instance = curve_mesh_instances[mesh_updated_list[i].first];
        if(c_instance.mesh.is_valid()){
            RSS->call_deferred("free_rid",c_instance.mesh);
        }
        RSS->instance_set_base(c_instance.instance,mesh_updated_list[i].second);
        c_instance.mesh = mesh_updated_list[i].second;
        curve_mesh_instances.insert(mesh_updated_list[i].first,c_instance);
    }
    mesh_updated_list.clear();
}

void MCurveMesh::_remove_instance(int64_t id,bool is_intersection){
    if(!curve_mesh_instances.has(id)){
        return;
    }
    if(is_intersection){
        clear_point_conn_ratio_limits(id);
    }
    MCurveMesh::Instance ii = curve_mesh_instances[id];
    if(ii.mesh.is_valid()) RSS->free(ii.mesh);
    if(ii.instance.is_valid()) RSS->free(ii.instance);
    curve_mesh_instances.erase(id);
}

void MCurveMesh::clear(){
    for(HashMap<int64_t,Instance>::Iterator it=curve_mesh_instances.begin();it!=curve_mesh_instances.end();++it){
        _remove_instance(it->key);
    }
    curve_mesh_instances.clear();
    conn_ratio_limits.clear();
}

void MCurveMesh::_remove_mesh(int64_t id,bool is_intersection){
    if(!curve_mesh_instances.has(id)){
        return;
    }
    if(is_intersection){
        clear_point_conn_ratio_limits(id);
    }
    MCurveMesh::Instance ii = curve_mesh_instances[id];
    if(ii.mesh.is_valid()) RSS->free(ii.mesh);
    ii.mesh = RID();
    ii.original_mesh_rid = RID();
    curve_mesh_instances.insert(id,ii);
}

void MCurveMesh::restart(){
    _generate_all_mesh_sliced_info();
    _generate_all_intersections_info();
    recreate();
}

void MCurveMesh::reload(){
    for(HashMap<int64_t,Instance>::Iterator it=curve_mesh_instances.begin();it!=curve_mesh_instances.end();++it){
        _id_force_update(it->key);
    }
}

void MCurveMesh::recreate(){
    clear();
    if(curve.is_valid()){
        PackedInt32Array apoints = curve->get_active_points();
        PackedInt64Array aconns = curve->get_active_conns();
        for(int i=0; i < apoints.size(); i++){
            _point_force_update(apoints[i]);
        }
        for(int i=0; i < aconns.size(); i++){
            _connection_force_update(aconns[i]);
        }
    }
}

void MCurveMesh::_swap_point_id(int64_t p_a,int64_t p_b){
    std::lock_guard<std::recursive_mutex> lock(update_mutex);
    bool has_p_a = curve_mesh_instances.has(p_a);
    bool has_p_b = curve_mesh_instances.has(p_b);
    MCurveMesh::Instance ia;
    MCurveMesh::Instance ib;
    if(has_p_a){
        ia = curve_mesh_instances[p_a];
    }
    if(has_p_b){
        ib = curve_mesh_instances[p_b];
    }
    if(has_p_a){
        curve_mesh_instances.insert(p_b,ia);
    } else {
        curve_mesh_instances.erase(p_b);
    }

    if(has_p_b){
        curve_mesh_instances.insert(p_a,ib);
    } else {
        curve_mesh_instances.erase(p_a);
    }
}

void MCurveMesh::_id_force_update(int64_t id){
    MCurve::Conn cc(id);
    if(cc.is_connection()){
        _connection_force_update(id);
    } else {
        _point_force_update(id);
    }
}

void MCurveMesh::_point_force_update(int32_t point_id){
    std::lock_guard<std::recursive_mutex> lock(update_mutex);
    ERR_FAIL_COND(curve.is_null());
    _remove_mesh(point_id);
    MCurve::PointUpdateInfo pu;
    pu.last_lod = -1;
    pu.current_lod = curve->get_point_lod(point_id);
    pu.point_id = point_id;
    _generate_intersection(pu,true);
}

void MCurveMesh::_connection_force_update(int64_t conn_id){
    std::lock_guard<std::recursive_mutex> lock(update_mutex);
    ERR_FAIL_COND(curve.is_null());
    _remove_mesh(conn_id);
    MCurve::ConnUpdateInfo cu;
    cu.last_lod = -1;
    cu.current_lod = curve->get_conn_lod(conn_id);
    cu.conn_id = conn_id;
    _generate_connection(cu,true);
}

void MCurveMesh::_point_remove(int32_t point_id){
    std::lock_guard<std::recursive_mutex> lock(update_mutex);
    _remove_instance(point_id);
}

void MCurveMesh::_connection_remove(int64_t conn_id){
    std::lock_guard<std::recursive_mutex> lock(update_mutex);
    conn_ratio_limits.erase(conn_id);
    _remove_instance(conn_id);
}

void MCurveMesh::_generate_connection(const MCurve::ConnUpdateInfo& update_info,bool immediate_update){
    ERR_FAIL_COND(meshlod_sliced_info.size() == 0);
    int64_t cid = update_info.conn_id;
    int lod = update_info.current_lod;
    int last_lod = update_info.last_lod;
    MCurveMeshOverride::Override c_override = ov.is_valid() ? ov->get_override(cid) : MCurveMeshOverride::Override();
    if(lod==-1 || c_override.mesh == -2){
        _remove_instance(cid);
        return;
    }
    /////////////////////////////////////////////////////
    /////// Grabing Correct MeshSliced //////////////////
    /////////////////////////////////////////////////////
    // mesh override
    int mesh_slice_index = c_override.mesh > 0 && c_override.mesh < meshlod_sliced_info.size() ? c_override.mesh : 0;
    MeshSlicedInfoArray mesh_slice_array = meshlod_sliced_info[mesh_slice_index]; 
    ERR_FAIL_COND(mesh_slice_array.size() == 0);
    lod = std::min(lod,mesh_slice_array.size() - 1);
    Ref<MeshSlicedInfo> mesh_sliced = mesh_slice_array.get(lod);
    if(mesh_sliced.is_null()){
        _remove_instance(cid);
        return;
    }
    if(!curve_mesh_instances.has(cid)){
        MCurveMesh::Instance c_instance;
        c_instance.instance = RSS->instance_create();
        RSS->instance_set_scenario(c_instance.instance,path->get_scenario());
        curve_mesh_instances.insert(cid,c_instance);
    } else { // Checking the last generated mesh if is the same exiting unless force_update is true
        MCurveMesh::Instance c_instance = curve_mesh_instances[cid];
        if(c_instance.original_mesh_rid == mesh_sliced->mesh_rid){ /// is same
            return;
        }
    }
    ERR_FAIL_COND(mesh_sliced.is_null());
    if(last_lod!=-1){
        last_lod = std::min(last_lod,mesh_slice_array.size() - 1);
        Ref<MeshSlicedInfo> last_mesh_sliced = mesh_slice_array.get(last_lod);
        if(last_mesh_sliced.is_valid() && last_mesh_sliced->mesh_rid == mesh_sliced->mesh_rid){
            return; // Mesh are same nothing to do here
        }
    }
    ERR_FAIL_COND(mesh_sliced->vertex.size() < 3 || mesh_sliced->index.size() < 3 || mesh_sliced->slice_count()==0);    
    /////////////////////////////////////////////////////
    ///////// Calculating transforms and mesh count /////
    /////////////////////////////////////////////////////
    int mesh_count = 0;
    Vector<Transform3D> transforms;
    {
        Vector<float> slice_ratios;
        Pair<float,float> c_t_limits = get_conn_ratio_limits(cid);
        Pair<float,float> dis_limit = curve->conn_ratio_limit_to_dis_limit(cid,c_t_limits);
        float curve_lenght = dis_limit.second - dis_limit.first;
        mesh_count = round(curve_lenght/mesh_sliced->lenght);
        if(mesh_count <=0 ){
            mesh_count = 1;
        }
        float mesh_curve_lenght = curve_lenght/((float)mesh_count); // more accurate mesh lenght in this curve
        float curve_lenght_ratio = mesh_curve_lenght/mesh_sliced->lenght;
        Vector<float> slice_curve_pos;
        slice_curve_pos.resize(mesh_sliced->slice_count());
        for(int i=0; i < mesh_sliced->slice_count();i++){
            slice_curve_pos.set(i,mesh_sliced->sliced_pos[i]*curve_lenght_ratio);
        }
        Vector<float> slice_distances;
        slice_distances.resize(mesh_sliced->slice_count() * mesh_count);
        for(int i=0 ; i < mesh_count; i++){
            for(int j=0; j < mesh_sliced->slice_count(); j++){
                int s_index = i * mesh_sliced->slice_count() + j;
                float s_distnace = i * mesh_curve_lenght + slice_curve_pos[j] + dis_limit.first;
                slice_distances.set(s_index,s_distnace);
            }
        }
        Pair<int,int> smallest_biggest = curve->get_conn_distances_ratios(cid,slice_distances,slice_ratios);
        // Making sure cover entire connection without gap
        slice_ratios.set(smallest_biggest.first,c_t_limits.first);
        slice_ratios.set(smallest_biggest.second,c_t_limits.second);
        // Correcting the start and end point
        curve->get_conn_transforms(cid,slice_ratios,transforms);
    }
    /////////////////////////////////////////////////////
    ///////// Generating Mesh ///////////////////////////
    /////////////////////////////////////////////////////
    ////// Creating the instance, mesh will be set during apply update
    PackedVector3Array vertex;
    PackedVector3Array normal;
    PackedFloat32Array tangent;
    PackedColorArray color;
    PackedVector2Array uv;
    PackedVector2Array uv2;
    PackedInt32Array index;
    // Getting some constant data with size of our mesh_count
    mesh_sliced->get_color(mesh_count,color);
    mesh_sliced->get_uv(mesh_count,uv);
    mesh_sliced->get_uv2(mesh_count,uv2);
    mesh_sliced->get_index(mesh_count,index);
    // mesh_sliced ----- We need to calculate this part
    vertex.resize(mesh_sliced->vertex.size() * mesh_count);
    bool has_normal = false;
    bool has_tangent = false;
    if(mesh_sliced->normal.size() == mesh_sliced->vertex.size()){
        normal.resize(mesh_sliced->normal.size() * mesh_count);
        has_normal = true;
    }
    if(mesh_sliced->tangent.size() == mesh_sliced->vertex.size() * 4){
        tangent.resize(mesh_sliced->tangent.size() * mesh_count);
        has_tangent = true;
    }

    // Creating the mesh
    ERR_FAIL_COND(transforms.size() != mesh_count * mesh_sliced->slice_count());
    // Setting Vertex Poistions
    int vertex_count = mesh_sliced->vertex.size();
    for(int mesh_index=0; mesh_index < mesh_count; mesh_index++){
        for(int j=0; j < mesh_sliced->slice_count(); j++){
            Transform3D transform = transforms[mesh_index * mesh_sliced->slice_count() + j];
            /// Looping through this slice indices
            int start_index = vertex_count * mesh_index;
            for(int index=0; index < mesh_sliced->sliced_info[j].size(); index++){
                int index_index = mesh_sliced->sliced_info[j][index];
                int vindex = start_index+index_index;
                vertex.set(vindex,transform.xform(mesh_sliced->vertex[index_index]));
                if(has_normal){
                    normal.set(vindex,transform.basis.xform(mesh_sliced->normal[index_index]));
                }
                if(has_tangent){
                    // Tangent contain 4 float number first 3 is tangent and last one is binormal
                    // This is why its index is calculated like this
                    int torg_index = index_index * 4;
                    int tindex = vindex * 4;
                    Vector3 _t_(mesh_sliced->tangent[torg_index],mesh_sliced->tangent[torg_index+1],mesh_sliced->tangent[torg_index+2]);
                    float binormal = mesh_sliced->tangent[torg_index+3];
                    _t_ = transform.basis.xform(_t_);
                    tangent.set(tindex,_t_.x);
                    tangent.set(tindex+1,_t_.y);
                    tangent.set(tindex+2,_t_.z);
                    tangent.set(tindex+3,binormal);
                }
            }
        }
    }
    
    Array mesh_data;
    mesh_data.resize(RenderingServer::ARRAY_MAX);
    mesh_data[RenderingServer::ARRAY_VERTEX] = vertex;
    if(has_normal) mesh_data[RenderingServer::ARRAY_NORMAL] = normal;
    if(has_tangent) mesh_data[RenderingServer::ARRAY_TANGENT] = tangent;
    if(color.size()>0) mesh_data[RenderingServer::ARRAY_COLOR] = color;
    if(uv.size()>0) mesh_data[RenderingServer::ARRAY_TEX_UV] = uv;
    if(uv2.size()>0) mesh_data[RenderingServer::ARRAY_TEX_UV2] = uv2;
    mesh_data[RenderingServer::ARRAY_INDEX] = index;
    
    bool is_material_set = false;
    MCurveMesh::Instance ii= curve_mesh_instances[cid];
    if(c_override.material >= 0 && c_override.material < materials.size()){
        Ref<Material> m_ov = materials[c_override.material];
        if(m_ov.is_valid()){
            RSS->instance_geometry_set_material_override(ii.instance,m_ov->get_rid());
            is_material_set = true;
        }
    }
    if(!is_material_set){
        if(mesh_sliced->material.is_valid()){
            RSS->instance_geometry_set_material_override(ii.instance,mesh_sliced->material->get_rid());
        }
    }

    if(immediate_update){
        
        if(!ii.mesh.is_valid()){
            ii.mesh = RSS->mesh_create();
            RSS->instance_set_base(ii.instance,ii.mesh);
        } else {
            RSS->mesh_clear(ii.mesh);
        }
        RSS->mesh_add_surface_from_arrays(ii.mesh,RenderingServer::PRIMITIVE_TRIANGLES,mesh_data); 
    } else{
        RID mesh = RSS->mesh_create();
        RSS->mesh_add_surface_from_arrays(mesh,RenderingServer::PRIMITIVE_TRIANGLES,mesh_data);
        mesh_updated_list.push_back({cid,mesh});
    }
    ii.original_mesh_rid = mesh_sliced->mesh_rid;
    curve_mesh_instances.insert(cid,ii);
}

void MCurveMesh::_generate_intersection(const MCurve::PointUpdateInfo& update_info,bool immediate_update){
    int32_t point_id = update_info.point_id;
    int lod = update_info.current_lod;
    MCurveMeshOverride::Override p_override = ov.is_valid() ? ov->get_override(point_id) : MCurveMeshOverride::Override();
    if(lod == -1 || p_override.mesh == -2){ // will also remove ratio limitation
        _remove_instance(point_id,true); 
        return;
    }
    int conn_count = curve->get_point_conn_count(point_id);
    if(conn_count < 3){
        _remove_instance(point_id,true);
        return;
    }
    // Finding a match intersection for number of connections
    Ref<MIntersection> finter;
    for(int i=0; i < intersections.size(); i++){
        Ref<MIntersection> ff = intersections[i];
        if(ff.is_valid() && ff->get_socket_count() == conn_count && ff->get_mesh_count() != 0){
            ERR_CONTINUE(!ff->is_init());
            finter = ff;
            break;
        }
    }
    if(finter.is_null()){
        _remove_instance(point_id,true);
        return;
    }
    lod = std::min(lod,finter->get_mesh_count()-1);
    Ref<MIntersectionInfo> inter_info = finter->get_mesh_info(lod);
    if(inter_info.is_null()){
        _remove_instance(point_id,true);
        return;
    }
    if(!curve_mesh_instances.has(point_id)){
        MCurveMesh::Instance c_instance;
        c_instance.instance = RSS->instance_create();
        RSS->instance_set_scenario(c_instance.instance,path->get_scenario());
        curve_mesh_instances.insert(point_id,c_instance);
    } else { // Checking the last generated mesh if is the same exiting unless force_update is true
        MCurveMesh::Instance c_instance = curve_mesh_instances[point_id];
        if(c_instance.original_mesh_rid == inter_info->mesh_rid){ /// is same
            return;
        }
    }
    Vector3 point_pos = curve->get_point_position(point_id);
    PackedInt32Array connected_points = curve->get_point_conn_points_exist(point_id);
    Vector<Transform3D> sockets = finter->_get_sockets();
    Vector<Transform3D> point_start_transform;
    point_start_transform.resize(connected_points.size());
    //VariantUtilityFunctions::print("Connected points ",connected_points);
    for(int i=0; i < connected_points.size(); i++){
        Transform3D pt = curve->get_point_order_transform(point_id,connected_points[i],0.0f,false,false);
        point_start_transform.set(i,pt);
    }
    // Trying to rotate each socket to the first connection
    // And finding maximum sum_dot_product
    HashMap<int32_t,int32_t> conn_point_socket; // Determine which socket is connect to which other one
    Basis initial_rotation;
    {
        float sum_dot_product = -1000.0;
        MCurve::Conn first_conn(point_id,connected_points[0]);
        for(int i=0; i < sockets.size(); i++){
            HashMap<int32_t,int32_t> this_conn_point_socket;
            float this_sum_dot_product=0;
            this_conn_point_socket.insert(connected_points[0],i);
            Transform3D s = sockets[i];
            Basis this_rot = point_start_transform[0].basis * s.basis.inverse();
            // We know if we rotate the first socket it will match the first connection
            // Now as we rotate this socket to the first connection
            // We need find the best match for other connections and calculate their conn match
            for(int j=0; j < sockets.size(); j++){
                if(i==j){
                    continue;
                }
                //VariantUtilityFunctions::print("Checking socket ---------- ====---------- ====---------- ==== ",j);
                Vector3 gxdir = this_rot.xform(sockets[j].basis.get_column(0));
                //VariantUtilityFunctions::print("x dir ",gxdir);
                // Finding the best connected point match for this gl
                // searching through connections
                // except first connection as it already matched to i socket
                int32_t best_match = 0; // start with invalid
                float g_highest_dot_product = -1000.0;
                for(int k=1; k < connected_points.size(); k++){
                    if(this_conn_point_socket.has(connected_points[k])){
                        continue; // Then this is already taken
                    }
                    //VariantUtilityFunctions::print("point ",connected_points[k], " dir x ",ktransform.basis[0]);
                    float gkdot = gxdir.dot(point_start_transform[k].basis.get_column(0));
                    //VariantUtilityFunctions::print(" dot ",gkdot);
                    if(gkdot > g_highest_dot_product){
                        best_match = connected_points[k];
                        g_highest_dot_product = gkdot;
                    }
                }
                //VariantUtilityFunctions::print("Best match ",best_match, " dot ",g_highest_dot_product);
                // j socket should have a connection pair otherwise error
                ERR_FAIL_COND(best_match==0);
                this_conn_point_socket.insert(best_match,j);
                this_sum_dot_product += g_highest_dot_product;
            }
            // Now check if this_sum_dot_product is biggest recorded one
            if(this_sum_dot_product > sum_dot_product){
                sum_dot_product = this_sum_dot_product;
                conn_point_socket = this_conn_point_socket;
                initial_rotation = this_rot;
            }
            //VariantUtilityFunctions::print(i , " sum dot ",this_sum_dot_product); 
        }
    }
    // This only use for finding a good position on curve
    Vector<Transform3D> positioned_sockets;
    positioned_sockets.resize(sockets.size());
    {
        Transform3D spos(initial_rotation,point_pos);
        for(int i=0; i < sockets.size(); i++){
            positioned_sockets.ptrw()[i] = spos * sockets[i];
        }
    }
    // Finding more accurate socket transformation
    Vector<Transform3D> socket_transforms;
    socket_transforms.resize(sockets.size());
    for(int i=0; i < connected_points.size(); i++){
        int32_t pid = connected_points[i];
        ERR_FAIL_COND(!conn_point_socket.has(pid));
        int32_t sid = conn_point_socket[pid];
        Transform3D s = sockets[sid];
        // finding closest ratio
        MCurve::Conn c(point_id,pid);
        float t = curve->get_closest_ratio_to_point(c.id,positioned_sockets[sid].origin);
        // ratio direction is smaller -> bigerr: correcting that!
        bool is_end = point_id > pid;

        set_conn_ratio_limits(c.id,t,is_end);
        t = is_end ? 1.0f - t : t;
        Transform3D ctransform = curve->get_point_order_transform(point_id,pid,t,true,true);
        //VariantUtilityFunctions::print("===================================!!!=================");
        //VariantUtilityFunctions::print(point_id," , ",pid,ctransform);
        //VariantUtilityFunctions::print("===================================!!!=================");
        socket_transforms.ptrw()[sid] = ctransform * s.inverse();
    }
    //// Apply transform on vertex base on their weights
    PackedVector3Array vertex = inter_info->vertex;
    PackedVector3Array normal = inter_info->normal;
    PackedFloat32Array tangents = inter_info->tangent;

    for(int i=0; i < inter_info->vertex.size(); i++){
        Vector3 total_pos(0.0f,0.0f,0.0f);
        Vector3 total_normal(0.0f,0.0f,0.0f);
        Vector3 total_tangent(0.0f,0.0f,0.0f);
        int ft = i * 4;
        int start_w = i * sockets.size();
        for(int j=0; j < socket_transforms.size(); j++){
            float w = inter_info->weights[start_w+j];
            total_pos += socket_transforms[j].xform(vertex[i]) * w;
            total_normal += socket_transforms[j].basis.xform(normal[i]) * w;
            if(tangents.size() > 0){
                total_tangent += socket_transforms[j].basis.xform(Vector3(tangents[ft],tangents[ft+1],tangents[ft+2]));
            }
        }
        vertex.set(i,total_pos);
        normal.set(i,total_normal);
        if(tangents.size() > 0){
            tangents.set(ft,total_tangent.x);
            tangents.set(ft+1,total_tangent.y);
            tangents.set(ft+2,total_tangent.z);
            /// tangents.set(ft+3,SAME); // This one remain same
        }
    }

    for(int i=0; i < inter_info->vertex.size(); i++){
        Vector3 v_pos = vertex[i];
        v_pos += point_pos;
        //vertex.set(i,v_pos);
    }
    
    PackedColorArray vcolor;
    vcolor.resize(vertex.size());
    memcpy(vcolor.ptrw(),inter_info->weights.ptr(),inter_info->weights.size() * sizeof(float));

    Array data_arr;
    data_arr.resize(Mesh::ARRAY_MAX);
    data_arr[RenderingServer::ARRAY_VERTEX] = vertex;
    data_arr[RenderingServer::ARRAY_NORMAL] = normal;
    if(tangents.size() > 0) data_arr[RenderingServer::ARRAY_TANGENT] = tangents;
    if(inter_info->color.size() > 0) data_arr[RenderingServer::ARRAY_COLOR] = inter_info->color;
    if(inter_info->uv.size() > 0) data_arr[RenderingServer::ARRAY_TEX_UV] = inter_info->uv;
    if(inter_info->uv2.size() > 0) data_arr[RenderingServer::ARRAY_TEX_UV2] = inter_info->uv2;
    data_arr[RenderingServer::ARRAY_INDEX] = inter_info->index;

    MCurveMesh::Instance ii= curve_mesh_instances[point_id];
    bool is_material_set = false;
    if(p_override.material>=0 && p_override.material < materials.size()){
        Ref<Material> mat = materials[p_override.material];
        if(mat.is_valid()){
            RSS->instance_geometry_set_material_override(ii.instance,mat->get_rid());
            is_material_set = true;
        }
    }
    if(!is_material_set){
        if(inter_info->material.is_valid()){
            RSS->instance_geometry_set_material_override(ii.instance,inter_info->material->get_rid());
        }
    }

    if(immediate_update){
        if(!ii.mesh.is_valid()){
            ii.mesh = RSS->mesh_create();
            RSS->instance_set_base(ii.instance,ii.mesh);
        } else {
            RSS->mesh_clear(ii.mesh);
        }
        RSS->mesh_add_surface_from_arrays(ii.mesh,RenderingServer::PRIMITIVE_TRIANGLES,data_arr); 
    } else{
        RID new_mesh = RSS->mesh_create();
        RSS->mesh_add_surface_from_arrays(new_mesh,RenderingServer::PRIMITIVE_TRIANGLES,data_arr);
        mesh_updated_list.push_back({point_id,new_mesh});
    }
    ii.original_mesh_rid = inter_info->mesh_rid;
    curve_mesh_instances.insert(point_id,ii);
}

void MCurveMesh::_process_tick(){
    if(is_thread_updating){
        if(WorkerThreadPool::get_singleton()->is_task_completed(thread_task_id)){
            WorkerThreadPool::get_singleton()->wait_for_task_completion(thread_task_id);
            is_thread_updating = false;
            ERR_FAIL_COND(curve.is_null());
            _apply_update();
            set_process(false);
            curve->user_finish_process(curve_user_id);
        }
    }
}

Pair<float,float> MCurveMesh::get_conn_ratio_limits(int64_t conn_id){
    Pair<float,float> out;
    if(conn_ratio_limits.has(conn_id)){
        out = conn_ratio_limits[conn_id];
    } else {
        out.first = 0.0f;
        out.second = 1.0;
    }
    return out;
}

void MCurveMesh::set_conn_ratio_limits(int64_t conn_id, float limit , bool is_end){
    Pair<float,float> cl = get_conn_ratio_limits(conn_id);
    if(is_end){
        cl.second = limit;
    } else {
        cl.first = limit;
    }
    conn_ratio_limits.insert(conn_id,cl);
    if(curve_mesh_instances.has(conn_id)){
        _connection_force_update(conn_id);
    }
}

void MCurveMesh::clear_point_conn_ratio_limits(int32_t point_id){
    PackedInt32Array ppconn = curve->get_point_conn_points_exist(point_id);
    for(int i=0; i < ppconn.size(); i++){
        MCurve::Conn cc(point_id,ppconn[i]);
        if(!conn_ratio_limits.has(cc.id)){
            continue;
        }
        Pair<float,float> l = conn_ratio_limits[cc.id];
        if(point_id < ppconn[i]){
            l.first = 0.0f;
        } else {
            l.second = 1.0f;
        }
        if(l.first == 0.0f && l.second == 1.0f){
            conn_ratio_limits.erase(cc.id);
            if(curve_mesh_instances.has(cc.id)){
                _connection_force_update(cc.id);
            }
            continue;
        }
        conn_ratio_limits.insert(cc.id,l);
        if(curve_mesh_instances.has(cc.id)){
            _connection_force_update(cc.id);
        }
    }
}

void MCurveMesh::set_overrides(Ref<MCurveMeshOverride> input){
    if(input.is_valid()){
        input->connect("id_updated",Callable(this,"_id_force_update"));
    }
    if(ov.is_valid()){
        ov->disconnect("id_updated",Callable(this,"_id_force_update"));
    }
    ov = input;
}

Ref<MCurveMeshOverride> MCurveMesh::get_overrides(){
    return ov;
}

void MCurveMesh::set_meshes(Array input){
    meshes = input;
}
Array MCurveMesh::get_meshes(){
    return meshes;
}

void MCurveMesh::set_intersections(Array input){
    intersections = input;
}

Array MCurveMesh::get_intersections(){
    return intersections;
}

void MCurveMesh::set_materials(Array input){
    materials = input;
}

Array MCurveMesh::get_materials(){
    return materials;
}

void MCurveMesh::_on_curve_changed(){
    MPath* new_path = Object::cast_to<MPath>(get_parent());
    if(new_path!=path){
        if(path!=nullptr){
            path->disconnect("curve_changed",Callable(this,"_on_curve_changed"));
            path->disconnect("visibility_changed",Callable(this,"_update_visibilty"));
            path->disconnect("tree_exited",Callable(this,"_update_visibilty"));
            path->disconnect("tree_entered",Callable(this,"_update_visibilty"));
        }
        if(new_path!=nullptr){
            new_path->connect("curve_changed",Callable(this,"_on_curve_changed"));
            new_path->connect("visibility_changed",Callable(this,"_update_visibilty"));
            new_path->connect("tree_exited",Callable(this,"_update_visibilty"));
            new_path->connect("tree_entered",Callable(this,"_update_visibilty"));
        }
    }
    path = new_path;
    Ref<MCurve> new_curve;
    // Handling Curve ...
    if(path!=nullptr){
        new_curve = path->curve;
    }
    if(curve != new_curve){
        if(curve.is_valid()){
            curve->disconnect("connection_updated",Callable(this,"_on_connections_updated"));
            curve->disconnect("force_update_point",Callable(this,"_point_force_update"));
            curve->disconnect("force_update_connection",Callable(this,"_connection_force_update"));
            curve->disconnect("remove_connection",Callable(this,"_connection_remove"));
            curve->disconnect("remove_point",Callable(this,"_point_remove"));
            curve->disconnect("swap_point_id",Callable(this,"_swap_point_id"));
            curve->disconnect("recreate",Callable(this,"recreate"));
            curve->remove_curve_user_id(curve_user_id);
        }
        curve = new_curve;
        if(curve.is_valid()){
            curve_user_id = curve->get_curve_users_id();
            curve->connect("connection_updated",Callable(this,"_on_connections_updated"));
            curve->connect("force_update_point",Callable(this,"_point_force_update"));
            curve->connect("force_update_connection",Callable(this,"_connection_force_update"));
            curve->connect("remove_connection",Callable(this,"_connection_remove"));
            curve->connect("remove_point",Callable(this,"_point_remove"));
            curve->connect("swap_point_id",Callable(this,"_swap_point_id"));
            curve->connect("recreate",Callable(this,"recreate"));
        }
    }
    update_configuration_warnings();
    recreate();
}

void MCurveMesh::_generate_all_intersections_info(){
    for(int i=0; i < intersections.size(); i++){
        Ref<MIntersection> inter = intersections[i];
        if(inter.is_valid()){
            inter->generate_mesh_info();
        }
    }
}

void MCurveMesh::_notification(int p_what){
    switch (p_what)
    {
    case NOTIFICATION_PROCESS:
        _process_tick();
        break;
    case NOTIFICATION_READY:
        if(!ov.is_valid()){
            ov.instantiate();
        }
        _generate_all_mesh_sliced_info();
        _generate_all_intersections_info();
        _on_curve_changed();
        break;
    case NOTIFICATION_PARENTED:
        _on_curve_changed();
        break;
    case NOTIFICATION_EDITOR_PRE_SAVE:
        if(!ov->get_path().is_empty()){
            ResourceSaver::save(ov,ov->get_path());
        }
    default:
        break;
    }
}

PackedStringArray MCurveMesh::_get_configuration_warnings() const {
    PackedStringArray out;
    if(path==nullptr){
        out.push_back("MCurveMesh should be a child of MPath node!");
        return out;
    }
    if(curve.is_null()){
        out.push_back("Please create a curve resource for MPath!");
        return out;
    }
    return out;
}

bool MCurveMesh::_set(const StringName &p_name, const Variant &p_value){
    if(p_name == String("meshes/mesh_count")){
        meshes.resize(p_value);
        meshlod_sliced_info.resize(p_value);
        notify_property_list_changed();
        return true;
    }
    if(p_name.begins_with("meshes/mesh_")){
        int index = p_name.replace("meshes/mesh_","").to_int();
        ERR_FAIL_INDEX_V(index,meshes.size(),false);
        Ref<MMeshLod> old_mesh = meshes[index];
        Ref<MMeshLod> seg = p_value;
        if(old_mesh.is_valid()){
            old_mesh->disconnect("meshes_changed",Callable(this,"_generate_all_mesh_sliced_info"));
        }
        if(seg.is_valid()){
            seg->connect("meshes_changed",Callable(this,"_generate_all_mesh_sliced_info"));
        }
        meshes[index] = p_value;
        return true;
    }
    if(p_name == String("intersections/intersection_count")){
        intersections.resize(p_value);
        notify_property_list_changed();
        return true;
    }
    if(p_name.begins_with("intersections/intersection_")){
        int index = p_name.replace("intersections/intersection_","").to_int();
        ERR_FAIL_INDEX_V(index,intersections.size(),false);
        Ref<MIntersection> seg = p_value;
        intersections[index] = seg;
        return true;
    }
    if(p_name == String("materials/material_count")){
        materials.resize(p_value);
        notify_property_list_changed();
        return true;
    }
    if(p_name.begins_with("materials/material_")){
        int index = p_name.replace("materials/material_","").to_int();
        ERR_FAIL_INDEX_V(index,materials.size(),false);
        Ref<Material> mat = p_value;
        materials[index] = mat;
        return true;
    }
    return false;
}

bool MCurveMesh::_get(const StringName &p_name, Variant &r_ret) const{
    if(p_name == String("meshes/mesh_count")){
        r_ret = meshes.size();
        return true;
    }
    if(p_name.begins_with("meshes/mesh_")){
        int index = p_name.replace("meshes/mesh_","").to_int();
        ERR_FAIL_INDEX_V(index,meshes.size(),false);
        Ref<MMeshLod> seg = meshes[index];
        r_ret = seg;
        return true;
    }
    if(p_name == String("intersections/intersection_count")){
        r_ret = intersections.size();
        return true;
    }
    if(p_name.begins_with("intersections/intersection_")){
        int index = p_name.replace("intersections/intersection_","").to_int();
        ERR_FAIL_INDEX_V(index,intersections.size(),false);
        Ref<MIntersection> seg = intersections[index];
        r_ret = seg;
        return true;
    }
    if(p_name == String("materials/material_count")){
        r_ret = materials.size();
        return true;
    }
    if(p_name.begins_with("materials/material_")){
        int index = p_name.replace("materials/material_","").to_int();
        ERR_FAIL_INDEX_V(index,intersections.size(),false);
        Ref<Material> seg = materials[index];
        r_ret = seg;
        return true;
    }
    return false;
}

void MCurveMesh::_get_property_list(List<PropertyInfo> *p_list) const{
    PropertyInfo mcounter(Variant::INT,"meshes/mesh_count",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR);
    p_list->push_back(mcounter);
    for(int i=0; i < meshes.size(); i++){
        PropertyInfo p(Variant::OBJECT,"meshes/mesh_"+itos(i),PROPERTY_HINT_RESOURCE_TYPE,"MMeshLod",PROPERTY_USAGE_EDITOR);
        p_list->push_back(p);
    }
    PropertyInfo scounter(Variant::INT,"intersections/intersection_count",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR);
    p_list->push_back(scounter);
    for(int i=0; i < intersections.size(); i++){
        PropertyInfo p(Variant::OBJECT,"intersections/intersection_"+itos(i),PROPERTY_HINT_RESOURCE_TYPE,"MIntersection",PROPERTY_USAGE_EDITOR);
        p_list->push_back(p);
    }
    PropertyInfo matcounter(Variant::INT,"materials/material_count",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR);
    p_list->push_back(matcounter);
    for(int i=0; i < materials.size(); i++){
        PropertyInfo p(Variant::OBJECT,"materials/material_"+itos(i),PROPERTY_HINT_RESOURCE_TYPE,"BaseMaterial3D,ShaderMaterial",PROPERTY_USAGE_EDITOR);
        p_list->push_back(p);
    }
}

void MCurveMesh::dumy_set_restart(bool input){
    restart();
}

bool MCurveMesh::dumy_get_true(){
    return true;
}