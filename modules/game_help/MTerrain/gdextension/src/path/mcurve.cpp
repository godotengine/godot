#include "mcurve.h"

#include <stack>
#include "core/io/marshalls.h"


#include "mpath.h"



MOctree* MCurve::octree = nullptr;

void MCurve::set_octree(MOctree* input){
    ERR_FAIL_COND_MSG(octree!=nullptr,"Only one octree can udpate MPath");
    octree = input;
}

MOctree* MCurve::get_octree(){
    return octree;
}

void MCurve::_bind_methods(){
    /// signals
    ADD_SIGNAL(MethodInfo("curve_updated"));
    ADD_SIGNAL(MethodInfo("connection_updated")); // chech after this point_update and connection_update
    //// Mostly use for editor update when moving or removing a point
    ADD_SIGNAL(MethodInfo("force_update_point",PropertyInfo(Variant::INT,"point_id")));
    ADD_SIGNAL(MethodInfo("force_update_connection",PropertyInfo(Variant::INT,"conn_id")));
    ADD_SIGNAL(MethodInfo("remove_point",PropertyInfo(Variant::INT,"point_id")));
    ADD_SIGNAL(MethodInfo("remove_connection",PropertyInfo(Variant::INT,"conn_id")));
    ADD_SIGNAL(MethodInfo("swap_point_id",PropertyInfo(Variant::INT,"p_a"),PropertyInfo(Variant::INT,"p_b")));
    ADD_SIGNAL(MethodInfo("recreate"));
    // end of signals

    ClassDB::bind_method(D_METHOD("get_points_count"), &MCurve::get_points_count);

    ClassDB::bind_method(D_METHOD("add_point","position","in","out","prev_conn"), &MCurve::add_point);
    ClassDB::bind_method(D_METHOD("add_point_conn_point","position","in","out","conn_types","conn_points"), &MCurve::add_point_conn_point);
    ClassDB::bind_method(D_METHOD("connect_points","p0","p1","conn_type"), &MCurve::connect_points);
    ClassDB::bind_method(D_METHOD("disconnect_conn","conn_id"), &MCurve::disconnect_conn);
    ClassDB::bind_method(D_METHOD("disconnect_points","p0","p1"), &MCurve::disconnect_points);
    ClassDB::bind_method(D_METHOD("remove_point","point_index"), &MCurve::remove_point);
    ClassDB::bind_method(D_METHOD("clear_points"), &MCurve::clear_points);

    ClassDB::bind_method(D_METHOD("get_conn_id","p0","p1"), &MCurve::get_conn_id);
    ClassDB::bind_method(D_METHOD("get_conn_ids_exist","points"), &MCurve::get_conn_ids_exist);
    ClassDB::bind_method(D_METHOD("get_conn_lod","conn_id"), &MCurve::get_conn_lod);
    ClassDB::bind_method(D_METHOD("get_active_points"), &MCurve::get_active_points);
    ClassDB::bind_method(D_METHOD("get_active_points_positions"), &MCurve::get_active_points_positions);
    ClassDB::bind_method(D_METHOD("get_active_conns"), &MCurve::get_active_conns);
    ClassDB::bind_method(D_METHOD("get_conn_baked_points","conn"), &MCurve::get_conn_baked_points);
    ClassDB::bind_method(D_METHOD("get_conn_baked_line","conn"), &MCurve::get_conn_baked_line);

    ClassDB::bind_method(D_METHOD("_octree_update_finish"), &MCurve::_octree_update_finish);


    ClassDB::bind_method(D_METHOD("has_point","p_index"), &MCurve::has_point);
    ClassDB::bind_method(D_METHOD("has_conn","conn_id"), &MCurve::has_conn);
    ClassDB::bind_method(D_METHOD("get_conn_type","conn_id"), &MCurve::get_conn_type);
    ClassDB::bind_method(D_METHOD("get_point_conn_count","p_index"), &MCurve::get_point_conn_count);
    ClassDB::bind_method(D_METHOD("get_point_conn_points","p_index"), &MCurve::get_point_conn_points);
    ClassDB::bind_method(D_METHOD("get_point_conn_points_recursive","p_index"), &MCurve::get_point_conn_points_recursive);
    ClassDB::bind_method(D_METHOD("get_point_conns","p_index"), &MCurve::get_point_conns);
    ClassDB::bind_method(D_METHOD("get_point_conns_inc_neighbor_points","p_index"), &MCurve::get_point_conns_inc_neighbor_points);
    ClassDB::bind_method(D_METHOD("growed_conn","conn_ids"), &MCurve::growed_conn);
    ClassDB::bind_method(D_METHOD("get_point_conn_types","p_index"), &MCurve::get_point_conn_types);
    ClassDB::bind_method(D_METHOD("get_point_position","p_index"), &MCurve::get_point_position);
    ClassDB::bind_method(D_METHOD("get_point_in","p_index"), &MCurve::get_point_in);
    ClassDB::bind_method(D_METHOD("get_point_out","p_index"), &MCurve::get_point_out);
    ClassDB::bind_method(D_METHOD("get_point_tilt","p_index"), &MCurve::get_point_tilt);
    ClassDB::bind_method(D_METHOD("set_point_tilt","p_index","val"), &MCurve::set_point_tilt);
    ClassDB::bind_method(D_METHOD("get_point_scale","p_index"), &MCurve::get_point_scale);
    ClassDB::bind_method(D_METHOD("set_point_scale","p_index","val"), &MCurve::set_point_scale);
    ClassDB::bind_method(D_METHOD("commit_point_update","p_index"), &MCurve::commit_point_update);
    ClassDB::bind_method(D_METHOD("commit_conn_update","conn_id"), &MCurve::commit_conn_update);

    ClassDB::bind_method(D_METHOD("toggle_conn_type","point","conn_id"), &MCurve::toggle_conn_type);
    ClassDB::bind_method(D_METHOD("validate_conn","conn_id"), &MCurve::validate_conn);
    ClassDB::bind_method(D_METHOD("swap_points","p_a","p_b"), &MCurve::swap_points);
    ClassDB::bind_method(D_METHOD("swap_points_with_validation","p_a","p_b"), &MCurve::swap_points_with_validation);
    ClassDB::bind_method(D_METHOD("sort_from","root_id","increasing"), &MCurve::sort_from);
    ClassDB::bind_method(D_METHOD("move_point","p_index","pos"), &MCurve::move_point);
    ClassDB::bind_method(D_METHOD("move_point_in","p_index","pos"), &MCurve::move_point_in);
    ClassDB::bind_method(D_METHOD("move_point_out","p_index","pos"), &MCurve::move_point_out);

    ClassDB::bind_method(D_METHOD("get_conn_position","conn_id","t"), &MCurve::get_conn_position);
    ClassDB::bind_method(D_METHOD("get_conn_aabb","conn_id"), &MCurve::get_conn_aabb);
    ClassDB::bind_method(D_METHOD("get_conns_aabb","conn_ids"), &MCurve::get_conns_aabb);
    ClassDB::bind_method(D_METHOD("get_closest_ratio_to_point","conn_id","pos"), &MCurve::get_closest_ratio_to_point);
    ClassDB::bind_method(D_METHOD("get_conn_transform","conn_id","t"), &MCurve::get_conn_transform);
    ClassDB::bind_method(D_METHOD("get_conn_lenght","conn_id"), &MCurve::get_conn_lenght);
    ClassDB::bind_method(D_METHOD("get_conn_distance_ratio","conn_id","distance"), &MCurve::get_conn_distance_ratio);

    ClassDB::bind_method(D_METHOD("ray_active_point_collision","org","dir","threshold"), &MCurve::ray_active_point_collision);
    ClassDB::bind_method(D_METHOD("_set_data","input"), &MCurve::_set_data);
    ClassDB::bind_method(D_METHOD("_get_data"), &MCurve::_get_data);
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY,"_data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE),"_set_data","_get_data");

    ClassDB::bind_method(D_METHOD("set_bake_interval","input"), &MCurve::set_bake_interval);
    ClassDB::bind_method(D_METHOD("get_bake_interval"), &MCurve::get_bake_interval);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"bake_interval"),"set_bake_interval","get_bake_interval");

    ClassDB::bind_method(D_METHOD("set_active_lod_limit","input"), &MCurve::set_active_lod_limit);
    ClassDB::bind_method(D_METHOD("get_active_lod_limit"), &MCurve::get_active_lod_limit);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"active_lod_limit"), "set_active_lod_limit","get_active_lod_limit");

    BIND_ENUM_CONSTANT(CONN_NONE);
    BIND_ENUM_CONSTANT(OUT_IN);
    BIND_ENUM_CONSTANT(IN_OUT);
    BIND_ENUM_CONSTANT(IN_IN);
    BIND_ENUM_CONSTANT(OUT_OUT);
}

MCurve::Point::Point(Vector3 _position,Vector3 _in,Vector3 _out):
position(_position),out(_out),in(_in){
}

MCurve::PointSave MCurve::Point::get_point_save(){
    PointSave ps;
    for(int8_t i=0; i < MAX_CONN; i++){
        ps.conn[i] = conn[i];
    }
    ps.tilt = tilt;
    ps.scale = scale;
    ps.in = in;
    ps.out = out;
    ps.position = position;
    return ps;
}

// p0 and p1 should be always positive and can't be equale
MCurve::Conn::Conn(int32_t p0, int32_t p1){
    if(p0 < p1){
        p.a = p0;
        p.b = p1;
    } else {
        p.a = p1;
        p.b = p0;
    }
}

MCurve::Conn::Conn(int64_t _id):id(_id){
}

MCurve::MCurve(){
    _increase_points_buffer_size(INIT_POINTS_BUFFER_SIZE);
}

MCurve::~MCurve(){

}

int MCurve::get_points_count(){
    // -1 because 0 index is always empty
    return points_buffer.size() - free_buffer_indicies.size() - 1;
}

void MCurve::_increase_points_buffer_size(size_t q){
    if(q<=0){
        return;
    }
    int64_t lsize = points_buffer.size();
    Error err = points_buffer.resize(lsize + q);
    ERR_FAIL_COND_MSG(err!=OK,"Can't increase point buffer size, possible fragmentation error!");
    for(int64_t i=points_buffer.size() - 1; i >= lsize ; i--){
        if(i==INVALID_POINT_INDEX){
            continue;
        }
        free_buffer_indicies.push_back(i);
    }
}

int32_t MCurve::get_curve_users_id(){
    last_curve_id++;
    curve_users.push_back(last_curve_id);
    return last_curve_id;
}
void MCurve::remove_curve_user_id(int32_t user_id){
    curve_users.erase(user_id);
    if(is_waiting_for_user){
        user_finish_process(user_id);
    }
}


int32_t MCurve::add_point(const Vector3& position,const Vector3& in,const Vector3& out, const int32_t prev_conn){
    // In case of prev_conn==INVALID_POINT_INDEX this is a single point in space
    ERR_FAIL_COND_V(!has_point(prev_conn) && prev_conn!=INVALID_POINT_INDEX,INVALID_POINT_INDEX);
    if(free_buffer_indicies.size() == 0){
        _increase_points_buffer_size(INC_POINTS_BUFFER_SIZE);
        ERR_FAIL_COND_V(free_buffer_indicies.size()==0,INVALID_POINT_INDEX);
    }
    int32_t free_index = free_buffer_indicies[free_buffer_indicies.size() - 1];
    Point new_point(position,in,out);
    if(prev_conn!=INVALID_POINT_INDEX){ // if this statement not run this means this is a single point in space
        Point* prev_point = points_buffer.ptrw() + prev_conn;
        // Check if prev_conn has free slot, As we are creating new point my slot defently has free slot
        int8_t prev_conn_free_slot = -1;
        for(int8_t i=0; i < MAX_CONN ; i++){
            if(prev_point->conn[i]==INVALID_POINT_INDEX){
                prev_conn_free_slot = i;
                break;
            }
        }
        ERR_FAIL_COND_V_EDMSG(prev_conn_free_slot==-1,INVALID_POINT_INDEX,"Maximum number of conn is "+itos(MAX_CONN));
        prev_point->conn[prev_conn_free_slot] = free_index;
        new_point.conn[0] = -prev_conn;
    }
    points_buffer.set(free_index,new_point);
    free_buffer_indicies.remove_at(free_buffer_indicies.size() - 1);
    if(is_init_insert && octree != nullptr){
        octree->insert_point(position,free_index,oct_id);
    }
    emit_signal("force_update_point",prev_conn);
    emit_signal("force_update_point",free_index);
    return free_index;
}

/*
    mostly has only for undo-redo use
*/
int32_t MCurve::add_point_conn_point(const Vector3& position,const Vector3& in,const Vector3& out,const Array& conn_types,const PackedInt32Array& conn_points){
    ERR_FAIL_COND_V(conn_types.size() != conn_points.size(),INVALID_POINT_INDEX);
    ERR_FAIL_COND_V(conn_types.size() > MAX_CONN,INVALID_POINT_INDEX);
    if(free_buffer_indicies.size() == 0){
        _increase_points_buffer_size(INC_POINTS_BUFFER_SIZE);
        ERR_FAIL_COND_V(free_buffer_indicies.size()==0,INVALID_POINT_INDEX);
    }
    int32_t free_index = free_buffer_indicies[free_buffer_indicies.size() - 1];
    Point new_points(position,in,out);
    points_buffer.set(free_index,new_points);
    free_buffer_indicies.remove_at(free_buffer_indicies.size() - 1);
    for(int8_t i=0; i < conn_points.size() ; i++){
        if(conn_points[i] == INVALID_POINT_INDEX){
            continue;
        }
        connect_points(free_index,conn_points[i],(ConnType)((int)conn_types[i]));
    }
    /// Correcting connection types
    if(is_init_insert && octree != nullptr){
        octree->insert_point(position,free_index,oct_id);
    }
    return free_index;
}

bool MCurve::connect_points(int32_t p0,int32_t p1,ConnType con_type){
    Conn conn(p0,p1); // Making the order right
    ERR_FAIL_COND_V(has_conn(conn.id), false);
    Point* a = points_buffer.ptrw() + conn.p.a;
    Point* b = points_buffer.ptrw() + conn.p.b;
    int8_t a_conn_index = -1;
    int8_t b_conn_index = -1;
    // Setting connections
    for(int8_t c=0; c < MAX_CONN; c++){
        if(a->conn[c] == INVALID_POINT_INDEX && a_conn_index == -1){
            a->conn[c] = conn.p.b; // Everything is positive here correcting connections types down
            a_conn_index = c;
        }
        if(b->conn[c] == INVALID_POINT_INDEX  && b_conn_index == -1){
            b->conn[c] = conn.p.a; // Everything is positive here correcting connections types down
            b_conn_index = c;
        }
    }
    // Removing Connection in case of error of MAX_CONN
    if(a_conn_index==-1 || b_conn_index==-1){
        if(a_conn_index!=-1){
            a->conn[a_conn_index] = INVALID_POINT_INDEX;
        }
        if(b_conn_index!=-1){
            b->conn[b_conn_index] = INVALID_POINT_INDEX;
        }
        ERR_FAIL_V_MSG("MAX Connection reached",false);
        return false;
    }

    // In case there is not error and both are set correcting types
    if(con_type == CONN_NONE){
        con_type = OUT_IN;
    }
    switch (con_type)
    {
    case OUT_IN:
        b->conn[b_conn_index] *= -1;
        break;
    case IN_OUT:
        a->conn[a_conn_index] *= -1;
        break;
    case IN_IN:
        a->conn[a_conn_index] *= -1;
        b->conn[b_conn_index] *= -1;
        break;
    //case OUT_OUT: // Nothing to do here has both will remain positive
    //    break;
    }
    /// Calculating LOD and force updateds
    int8_t clod = a->lod < b->lod ? a->lod : b->lod;
    conn_list.insert(conn.id,clod);
    if(clod <= active_lod_limit){
        active_conn.insert(conn.id);
    }
    emit_signal("force_update_point",conn.p.a);
    emit_signal("force_update_point",conn.p.b);
    emit_signal("force_update_connection",conn.id);
    emit_signal("curve_updated");
    return true;
}

bool MCurve::disconnect_conn(int64_t conn_id){
    Conn cc(conn_id);
    return disconnect_points(cc.p.a,cc.p.b);
}

bool MCurve::disconnect_points(int32_t p0,int32_t p1){
    ERR_FAIL_COND_V(!has_point(p0), false);
    ERR_FAIL_COND_V(!has_point(p1), false);
    Point* a = points_buffer.ptrw() + p0;
    Point* b = points_buffer.ptrw() + p1;
    bool is_removed = false;
    for(int8_t c=0; c < MAX_CONN; c++){
        if(abs(a->conn[c]) == p1){
            a->conn[c] = INVALID_POINT_INDEX; // Everything is positive here correcting connections types down
            is_removed = true;
        }
        if(abs(b->conn[c]) == p0){
            b->conn[c] = INVALID_POINT_INDEX; // Everything is positive here correcting connections types down
            is_removed = true;
        }
    }
    Conn conn(p0,p1);
    conn_list.erase(conn.id);
    active_conn.erase(conn.id);
    emit_signal("curve_updated");
    emit_signal("remove_connection",conn.id);
    baked_lines.erase(conn.id);
    return is_removed;
}


void MCurve::remove_point(const int32_t point_index){
    ERR_FAIL_COND(!has_point(point_index));
    const Point* p = points_buffer.ptr() + point_index;
    // Removing from conn
    for(int8_t i=0; i < MAX_CONN; i++){
        if(p->conn[i]!=INVALID_POINT_INDEX){
            int32_t conn_point_id = std::abs(p->conn[i]);
            ERR_FAIL_INDEX(conn_point_id, points_buffer.size());
            Conn conn(point_index,conn_point_id);
            active_conn.erase(conn.id);
            conn_list.erase(conn.id);
            conn_distances.erase(conn.id);
            baked_lines.erase(conn.id);
            Point* conn_p = points_buffer.ptrw() + conn_point_id;
            for(int8_t c=0; c < MAX_CONN; c++){
                if(std::abs(conn_p->conn[c]) == point_index){
                    conn_p->conn[c] = INVALID_POINT_INDEX;
                    break;
                }
            }
            emit_signal("remove_connection",conn.id);
        }
    }
    if(is_init_insert){
        ERR_FAIL_COND(octree==nullptr);
        octree->remove_point(point_index,p->position,oct_id);
    }
    free_buffer_indicies.push_back(point_index);
    active_points.erase(point_index);
    for(int8_t i=0; i < MAX_CONN; i++){
        if(p->conn[i]!=INVALID_POINT_INDEX){
            emit_signal("force_update_point",std::abs(p->conn[i]));
        }
    }
    emit_signal("curve_updated");
    emit_signal("remove_point",point_index);
}

void MCurve::clear_points(){
    //VariantUtilityFunctions::print("Clear points ");
    points_buffer.clear();
    free_buffer_indicies.clear();
    if(is_init_insert){
        ERR_FAIL_COND(octree==nullptr);
        octree->clear_oct_id(oct_id);
    }
    for(int i=active_points.size() - 1; i >= 0; i--){
        active_points.erase(active_points[i]);
    }
    conn_list.clear();
}

void MCurve::init_insert(){
    if(is_init_insert){
        return;
    }
    ERR_FAIL_COND_MSG(octree==nullptr,"No octree asigned to update curves, please asign a octree by calling enable_as_curve_updater and restart Godot");
    // inserting points into octree
    PackedVector3Array positions;
    PackedInt32Array ids;
    for (int i=0; i < points_buffer.size(); i++){
        if(free_buffer_indicies.has(i) || i == INVALID_POINT_INDEX){
            continue;
        }
        positions.push_back(points_buffer[i].position);
        ids.push_back(i);
    }
    oct_id = octree->get_oct_id();
    is_init_insert = true;
    octree->connect("update_finished", Callable(this,"_octree_update_finish"));
    octree->insert_points(positions,ids,oct_id);
}

void MCurve::_octree_update_finish(){
    Vector<MOctree::PointUpdate> update_info = octree->get_point_update(oct_id);
    if(update_info.size()==0){
        octree->call_deferred("point_process_finished",oct_id);
        return;
    }
    conn_update.clear();
    point_update.clear();
    Point* ptrw = points_buffer.ptrw();
    //HashSet<int32_t> updated_points;
    Vector<int32_t> updated_points;
    for(int i=0; i < update_info.size(); i++){
        ERR_CONTINUE(!has_point(update_info[i].id));
        // see if the lod is not active remove that!
        //int8_t lod = update_info[i].lod < active_lod_limit ? update_info[i].lod : INVALID_POINT_LOD;
        Point* p = points_buffer.ptrw() + update_info[i].id;
        // Updating LOD of point
        // Updating active points
        PointUpdateInfo point_update_info;
        if(update_info[i].lod > active_lod_limit){
            if(p->lod > active_lod_limit){
                continue; // Same as before was deactive
            }
            active_points.erase(update_info[i].id);
            point_update_info.current_lod = -1;
        } else {
            active_points.insert(update_info[i].id);
            point_update_info.current_lod = update_info[i].lod;
        }
        ////// Creating Point Update info
        point_update_info.last_lod = update_info[i].last_lod > active_lod_limit ? -1 : update_info[i].last_lod;
        point_update_info.point_id = update_info[i].id;
        point_update.push_back(point_update_info);
        updated_points.push_back(update_info[i].id);
        p->lod = update_info[i].lod;
    }
    // Now updating conns base on updated Points
    HashSet<int64_t> processed_conns; // As both side point of this conn may updated
    for(int32_t k=0; k < updated_points.size(); k++){
        int32_t cpoint = updated_points[k];
        const Point* p = points_buffer.ptr() + cpoint;
        for(int c=0; c < MAX_CONN; c++){ // Here we really using c++
            if(p->conn[c] != INVALID_POINT_INDEX){
                int32_t next_point_index = std::abs(p->conn[c]); // We don't care about conn type here! conneciton type is encode in positive and negetive of conn
                Conn conn(cpoint , next_point_index);
                if(processed_conns.has(conn.id)){
                    continue;
                }
                ERR_FAIL_COND(!has_point(next_point_index));
                Point* next_point = points_buffer.ptrw() + next_point_index;
                // bellow can be done considering the fact INVALID_POINT_LOD is a big number
                int8_t new_lod = p->lod < next_point->lod ? p->lod : next_point->lod;
                int8_t last_lod = conn_list.has(conn.id) ? conn_list[conn.id] : INVALID_POINT_LOD;
                conn_list.insert(conn.id,new_lod); // contain real lod without active_lod_limit
                new_lod = new_lod > active_lod_limit ? INVALID_POINT_LOD : new_lod;
                last_lod = last_lod > active_lod_limit ? INVALID_POINT_LOD : last_lod;
                if(new_lod == last_lod){ // NOTHING TO DO
                    processed_conns.insert(conn.id);
                    continue;
                }
                if(new_lod == INVALID_OCT_POINT_ID){
                    active_conn.erase(conn.id);
                    conn_distances.erase(conn.id);
                    baked_lines.erase(conn.id);
                } else {
                    active_conn.insert(conn.id);
                }
                ConnUpdateInfo cuinfo;
                cuinfo.current_lod = new_lod;
                cuinfo.last_lod = last_lod;
                cuinfo.conn_id = conn.id;
                conn_update.push_back(cuinfo);
                processed_conns.insert(conn.id);
            }
        }
    }
    emit_signal("curve_updated");
    is_waiting_for_user = true;
    emit_signal("connection_updated");
    if(curve_users.size()==0){
        octree->call_deferred("point_process_finished",oct_id);
    }
}

void MCurve::user_finish_process(int32_t user_id){
    ERR_FAIL_COND(!is_waiting_for_user);
    processing_users.erase(user_id);
    if(processing_users.size()==0){
        octree->call_deferred("point_process_finished",oct_id);
    }
}

int64_t MCurve::get_conn_id(int32_t p0, int32_t p1){
    Conn c(p0,p1);
    return c.id;
}

PackedInt64Array MCurve::get_conn_ids_exist(const PackedInt32Array points){
    PackedInt64Array out;
    if(points.size() < 2){
        return out;
    }
    for(int i=0; i < points.size() - 1; i++){
        int32_t pi = points[i];
        for(int j=i+1; j < points.size(); j++){
            int32_t pj = points[j];
            Conn conn(pi,pj);
            if(!out.has(conn.id) && has_conn(conn.id)){
                out.push_back(conn.id);
            }
        }
    }
    return out;
}

int8_t MCurve::get_conn_lod(int64_t conn_id){
    int8_t out = conn_list.has(conn_id) ? conn_list[conn_id] : -1;
    if(out > active_lod_limit){
        return -1;
    }
    return out;
}

int8_t MCurve::get_point_lod(int64_t p_id){
    ERR_FAIL_COND_V(!has_point(p_id),-1);
    int8_t out = points_buffer[p_id].lod;
    if(out > active_lod_limit){
        return -1;
    }
    return out;
}

PackedInt32Array MCurve::get_active_points(){
    PackedInt32Array out;
    out.resize(active_points.size());
    for(int i=0; i < out.size(); i++){
        out.set(i,active_points[i]);
    }
    return out;
}

PackedVector3Array MCurve::get_active_points_positions(){
    PackedVector3Array out;
    out.resize(active_points.size());
    for(int i=0; i < out.size(); i++){
        int32_t p_index = active_points[i];
        ERR_FAIL_INDEX_V(p_index,points_buffer.size(),out);
        out.set(i,points_buffer[p_index].position);
    }
    return out;
}

PackedInt64Array MCurve::get_active_conns(){
    PackedInt64Array out;
    out.resize(active_conn.size());
    for(int i=0; i < active_conn.size(); i++){
        out.set(i,active_conn[i]);
    }
    return out;
}

PackedVector3Array MCurve::get_conn_baked_points(int64_t input_conn){
    PackedVector3Array out;
    Conn conn(input_conn);
    ERR_FAIL_COND_V(!has_point(conn.p.a),out);
    ERR_FAIL_COND_V(!has_point(conn.p.b),out);
    const Point* a = points_buffer.ptr() + conn.p.a;
    const Point* b = points_buffer.ptr() + conn.p.b;
    // First we assume control is negetive and then in loop if we found positive we change that
    Vector3 a_control = a->in; 
    Vector3 b_control = b->in;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(a->conn[i] == conn.p.b){
            a_control = a->out;
        }
        if(b->conn[i] == conn.p.a){
            b_control = b->out;
        }
    }
    float lenght = get_length_between_basic(a,b,a_control,b_control);
    int pcount = lenght/bake_interval; // This is only for middle points
    pcount = pcount == 0 ? 1 : pcount;
    out.resize(pcount + 1); // including start and end pos
    out.set(0,a->position);
    float nl = 1.0 / (float)pcount; // normalized_interval
    for(int i=1; i < pcount; i++){
        out.set(i,a->position.bezier_interpolate(a_control,b_control,b->position,i*nl));
    }
    out.set(pcount,b->position);
    return out;
}

PackedVector3Array MCurve::get_conn_baked_line(int64_t input_conn){
    if(baked_lines.has(input_conn)){
        return baked_lines[input_conn];
    }
    PackedVector3Array points = get_conn_baked_points(input_conn);
    PackedVector3Array line;
    line.resize((points.size()-1)*2);
    int lc = 0;
    for(int i=0; i < points.size() - 1; i++){
        line.set(lc,points[i]);
        lc++;
        line.set(lc,points[i+1]);
        lc++;
    }
    baked_lines.insert(input_conn,line);
    return line;
}

bool MCurve::has_point(int p_index) const{
    if(p_index < 1 || p_index >= points_buffer.size() || free_buffer_indicies.has(p_index)){ // As we don't have index 0
        return false;
    }
    return true;
}

bool MCurve::has_conn(int64_t conn_id){
    Conn conn(conn_id);
    if(!has_point(conn.p.a) || !has_point(conn.p.b)){
        return false;
    }
    Point* a = points_buffer.ptrw() + conn.p.a;
    Point* b = points_buffer.ptrw() + conn.p.b;
    bool has_in_a = false;
    bool has_in_b = false;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(abs(a->conn[i]) == conn.p.b){
            has_in_a = true;
        }
        if(abs(b->conn[i]) == conn.p.a){
            has_in_b = true;
        }
    }
    return has_in_a && has_in_b;
}

MCurve::ConnType MCurve::get_conn_type(int64_t conn_id) const{
    Conn c(conn_id); // This make order of smaller and bigger right
    if(!has_point(c.p.a) || !has_point(c.p.b)){
        return CONN_NONE;
    }
    const Point* a = points_buffer.ptr() + c.p.a;
    const Point* b = points_buffer.ptr() + c.p.b;
    int a_b_sign = 0;
    int b_a_sign = 0;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(abs(a->conn[i]) == c.p.b){
            a_b_sign = a->conn[i] > 0 ? 1 : -1;
        }
        if(abs(b->conn[i]) == c.p.a){
            b_a_sign = b->conn[i] > 0 ? 1 : -1;
        }
    }
    if(a_b_sign == 1 && b_a_sign -1){
        return OUT_IN;
    }
    if(a_b_sign == -1 && b_a_sign == 1){
        return IN_OUT;
    }
    if(a_b_sign == -1 && b_a_sign == -1){
        return IN_IN;
    }
    if(a_b_sign == 1 && b_a_sign == 1){
        return OUT_OUT;
    }
    return CONN_NONE;
}

Array MCurve::get_point_conn_types(int32_t p_index) const{
    Array out;
    ERR_FAIL_COND_V(!has_point(p_index),out);
    const Point* p = points_buffer.ptr() + p_index;
    for(int8_t i=0; i < MAX_CONN ; i++){
        Conn c(p_index,abs(p->conn[i]));
        out.push_back(get_conn_type(c.id));
    }
    return out;
}

int MCurve::get_point_conn_count(int32_t p_index) const{
    int out = 0;
    ERR_FAIL_COND_V(!has_point(p_index),out);
    const Point* p = points_buffer.ptr() + p_index;
    for(int8_t i=0; i < MAX_CONN ; i++){
        if(p->conn[i]!=0){
            out++;
        }
    }
    return out;
}

PackedInt32Array MCurve::get_point_conn_points_exist(int32_t p_index) const{
    PackedInt32Array out;
    ERR_FAIL_COND_V(!has_point(p_index),out);
    const Point* p = points_buffer.ptr() + p_index;
    for(int8_t i=0; i < MAX_CONN ; i++){
        if(p->conn[i] !=0){
            out.push_back(abs(p->conn[i]));
        }
    }
    return out;
}

PackedInt32Array MCurve::get_point_conn_points(int32_t p_index) const{
    PackedInt32Array out;
    ERR_FAIL_COND_V(!has_point(p_index),out);
    const Point* p = points_buffer.ptr() + p_index;
    for(int8_t i=0; i < MAX_CONN ; i++){
        out.push_back(abs(p->conn[i]));
    }
    return out;
}
/*
    output does not include p_index
*/
PackedInt32Array MCurve::get_point_conn_points_recursive(int32_t p_index) const {
    PackedInt32Array out;
    ERR_FAIL_COND_V(!has_point(p_index),out);
    HashSet<int32_t> processed_points;
    PackedInt32Array stack = get_point_conn_points_exist(p_index);
    processed_points.insert(p_index);
    while (stack.size()!=0)
    {
        int32_t current_index = stack[stack.size()-1];
        const Point* current_point = points_buffer.ptr() + current_index;
        stack.remove_at(stack.size()-1);
        out.push_back(current_index);
        processed_points.insert(current_index);
        for(int i=0;i < MAX_CONN; i++){
            if(current_point->conn[i] != 0){
                int32_t cp = std::abs(current_point->conn[i]);
                if(!processed_points.has(cp)){
                    stack.push_back(cp);
                }
            }
        }
    }
    return out;
}

PackedInt64Array MCurve::get_point_conns(int32_t p_index) const{
    PackedInt64Array out;
    ERR_FAIL_COND_V(!has_point(p_index),out);
    const Point* p = points_buffer.ptr() + p_index;
    for(int8_t i=0; i < MAX_CONN ; i++){
        if(p->conn[i]!=0){
            int32_t other_index = std::abs(p->conn[i]);
            Conn cc(p_index,other_index);
            out.push_back(cc.id);
        }
    }
    return out;
}

PackedInt64Array MCurve::get_point_conns_inc_neighbor_points(int32_t p_index) const{
    PackedInt64Array out;
    ERR_FAIL_COND_V(!has_point(p_index),out);
    const Point* p = points_buffer.ptr() + p_index;
    PackedInt32Array conn_points;
    for(int8_t i=0; i < MAX_CONN ; i++){
        if(p->conn[i]!=0){
            int32_t other_index = std::abs(p->conn[i]);
            conn_points.push_back(other_index);
            Conn cc(p_index,other_index);
            out.push_back(cc.id);
        }
    }
    for(int j=0; j < conn_points.size(); j++){
        const Point* op = points_buffer.ptr() + conn_points[j];
        for(int8_t i=0; i < MAX_CONN ; i++){
            int32_t other_index = std::abs(op->conn[i]);
            if(other_index!=0 && other_index!=p_index){
                Conn cc(conn_points[j],other_index);
                if(true){
                    out.push_back(cc.id);
                }
            }
        }
    }
    return out;
}

PackedInt64Array MCurve::growed_conn(PackedInt64Array conn_ids) const{
    PackedInt64Array out;
    for(int i=0; i < conn_ids.size(); i++){
        Conn cc(conn_ids[i]);
        PackedInt64Array pc = get_point_conns(cc.p.a);
        pc.append_array(get_point_conns(cc.p.b));
        for(int j=0; j < pc.size(); j++){
            if(out.find(pc[j]==-1)){
                out.push_back(pc[j]);
            }
        }
    }
    return out;
}

Vector3 MCurve::get_point_position(int p_index){
    ERR_FAIL_COND_V(!has_point(p_index),Vector3());
    return points_buffer.get(p_index).position;
}
Vector3 MCurve::get_point_in(int p_index){
    ERR_FAIL_COND_V(!has_point(p_index),Vector3());
    return points_buffer.get(p_index).in;
}
Vector3 MCurve::get_point_out(int p_index){
    ERR_FAIL_COND_V(!has_point(p_index),Vector3());
    return points_buffer.get(p_index).out;
}
float MCurve::get_point_tilt(int p_index){
    ERR_FAIL_COND_V(!has_point(p_index),0.0f);
    return points_buffer.get(p_index).tilt;
}

void MCurve::set_point_tilt(int p_index,float input){
    ERR_FAIL_COND(!has_point(p_index));
    points_buffer.ptrw()[p_index].tilt = input;
}

float MCurve::get_point_scale(int p_index){
    ERR_FAIL_COND_V(!has_point(p_index),1.0);
    return points_buffer.get(p_index).scale;
}

void MCurve::set_point_scale(int p_index,float input){
    ERR_FAIL_COND(!has_point(p_index));
    points_buffer.ptrw()[p_index].scale = input;
}


void MCurve::commit_point_update(int p_index){
    ERR_FAIL_COND(!has_point(p_index));
    const Point* p = points_buffer.ptr() + p_index;
    VSet<int64_t> u_conn;
    VSet<int32_t> u_point;
    u_point.insert(p_index);
    for(int8_t i=0; i < MAX_CONN ; i++){
        int32_t pp = abs(p->conn[i]);
        if(pp!=0){
            ERR_CONTINUE(!has_point(p_index));
            u_point.insert(pp);
            Conn conn(pp,p_index);
            u_conn.insert(conn.id);
        }
    }
    for(int i=0; i < u_conn.size(); i++){
        // This will force them to rebake if the will needed later
        conn_distances.erase(u_conn[i]);
    }
    // Updating points
    for(int8_t i=0; i < u_point.size() ; i++){
        emit_signal("force_update_point",u_point[i]);
    }
    // updating connections
    for(int i=0; i < u_conn.size(); i++){
        emit_signal("force_update_connection",u_conn[i]);
    }
}

void MCurve::commit_conn_update(int64_t conn_id){
    ERR_FAIL_COND(has_conn(conn_id));
    emit_signal("force_update_connection",conn_id);
}

Vector3 MCurve::get_conn_position(int64_t conn_id,float t){
    ERR_FAIL_COND_V(!has_conn(conn_id), Vector3());
    Conn conn(conn_id);
    const Point* a = points_buffer.ptr() + conn.p.a;
    const Point* b = points_buffer.ptr() + conn.p.b;
    Vector3 a_control = a->in; 
    Vector3 b_control = b->in;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(a->conn[i] == conn.p.b){
            a_control = a->out;
        }
        if(b->conn[i] == conn.p.a){
            b_control = b->out;
        }
    }
    return a->position.bezier_interpolate(a_control,b_control,b->position,t);
}

////////
// https://iquilezles.org/articles/bezierbbox/
////////
AABB MCurve::get_conn_aabb(int64_t conn_id){
    ERR_FAIL_COND_V(!has_conn(conn_id),AABB());
    Conn cc(conn_id);
    const Point* a = points_buffer.ptr() + cc.p.a;
    const Point* b = points_buffer.ptr() + cc.p.b;
    Vector3 a_control = a->in; 
    Vector3 b_control = b->in;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(a->conn[i] == cc.p.b){
            a_control = a->out;
        }
        if(b->conn[i] == cc.p.a){
            b_control = b->out;
        }
    }
    //////// min max
    Vector3 mi = a->position.min(b->position);
    Vector3 ma = a->position.max(b->position);

    Vector3 _c = a_control - a->position;
    Vector3 _b = a->position - 2.0*a_control + b_control;
    Vector3 _a = -1.0*a->position + 3.0*a_control - 3.0*b_control + b->position;

    Vector3 h = _b*_b - _a*_c;
    if(h.x > 0.0){
        h.x = sqrt(h.x);
        float t = (-_b.x - h.x)/_a.x;
        if(t > 0.0 && t < 1.0){
            float s = 1.0f - t;
            float q = s*s*s*a->position.x + 3.0*s*s*t*a_control.x + 3.0*s*t*t*b_control.x + t*t*t*b->position.x;
            mi.x = MIN(mi.x,q);
            ma.x = MAX(ma.x,q);
        }
        t = (-_b.x + h.x)/_a.x;
        if(t > 0.0 && t < 1.0){
            float s = 1.0f - t;
            float q = s*s*s*a->position.x + 3.0*s*s*t*a_control.x + 3.0*s*t*t*b_control.x + t*t*t*b->position.x;
            mi.x = MIN(mi.x,q);
            ma.x = MAX(ma.x,q);
        }
    }
    if(h.y > 0.0){
        h.y = sqrt(h.y);
        float t = (-_b.y - h.y)/_a.y;
        if(t > 0.0 && t < 1.0){
            float s = 1.0f - t;
            float q = s*s*s*a->position.y + 3.0*s*s*t*a_control.y + 3.0*s*t*t*b_control.y + t*t*t*b->position.y;
            mi.y = MIN(mi.y,q);
            ma.y = MAX(ma.y,q);
        }
        t = (-_b.y + h.y)/_a.y;
        if(t > 0.0 && t < 1.0){
            float s = 1.0f - t;
            float q = s*s*s*a->position.y + 3.0*s*s*t*a_control.y + 3.0*s*t*t*b_control.y + t*t*t*b->position.y;
            mi.y = MIN(mi.y,q);
            ma.y = MAX(ma.y,q);
        }
    }
    if(h.z > 0.0){
        h.z = sqrt(h.z);
        float t = (-_b.z - h.z)/_a.z;
        if(t > 0.0 && t < 1.0){
            float s = 1.0f - t;
            float q = s*s*s*a->position.z + 3.0*s*s*t*a_control.z + 3.0*s*t*t*b_control.z + t*t*t*b->position.z;
            mi.z = MIN(mi.z,q);
            ma.z = MAX(ma.z,q);
        }
        t = (-_b.z + h.z)/_a.z;
        if(t > 0.0 && t < 1.0){
            float s = 1.0f - t;
            float q = s*s*s*a->position.z + 3.0*s*s*t*a_control.z + 3.0*s*t*t*b_control.z + t*t*t*b->position.z;
            mi.z = MIN(mi.z,q);
            ma.z = MAX(ma.z,q);
        }
    }
    return AABB(mi, ma - mi);
}

AABB MCurve::get_conns_aabb(const PackedInt64Array& conn_ids){
    ERR_FAIL_COND_V(conn_ids.size()==0,AABB());
    AABB faabb = get_conn_aabb(conn_ids[0]);
    for(int i=1; i < conn_ids.size(); i++){
        AABB taabb = get_conn_aabb(conn_ids[i]);
        faabb = faabb.merge(taabb);
    }
    return faabb;
}

float MCurve::get_closest_ratio_to_point(int64_t conn_id,Vector3 pos){
    ERR_FAIL_COND_V(!has_conn(conn_id), 0.0f);
    Conn conn(conn_id);
    const Point* a = points_buffer.ptr() + conn.p.a;
    const Point* b = points_buffer.ptr() + conn.p.b;
    Vector3 a_control = a->in; 
    Vector3 b_control = b->in;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(a->conn[i] == conn.p.b){
            a_control = a->out;
        }
        if(b->conn[i] == conn.p.a){
            b_control = b->out;
        }
    }
    real_t low = 0.0f;
    real_t high = 1.0;
    while (high - low > 0.005f)
    {
        real_t step = (high - low)/8.0;
        //samples
        real_t s[9] {low,
                    low + step,
                    low + step * 2.0f,
                    low + step * 3.0f,
                    low + step * 4.0f,
                    low + step * 5.0f,
                    low + step * 6.0f,
                    low + step * 7.0f,
                    high
                    };
        Vector3 p0 = a->position.bezier_interpolate(a_control,b_control,b->position,s[0]);
        Vector3 p1 = a->position.bezier_interpolate(a_control,b_control,b->position,s[1]);
        Vector3 p2 = a->position.bezier_interpolate(a_control,b_control,b->position,s[2]);
        Vector3 p3 = a->position.bezier_interpolate(a_control,b_control,b->position,s[3]);
        Vector3 p4 = a->position.bezier_interpolate(a_control,b_control,b->position,s[4]);
        Vector3 p5 = a->position.bezier_interpolate(a_control,b_control,b->position,s[5]);
        Vector3 p6 = a->position.bezier_interpolate(a_control,b_control,b->position,s[6]);
        Vector3 p7 = a->position.bezier_interpolate(a_control,b_control,b->position,s[7]);
        Vector3 p8 = a->position.bezier_interpolate(a_control,b_control,b->position,s[8]);
        real_t dis[9] = {
            pos.distance_squared_to(p0),
            pos.distance_squared_to(p1),
            pos.distance_squared_to(p2),
            pos.distance_squared_to(p3),
            pos.distance_squared_to(p4),
            pos.distance_squared_to(p5),
            pos.distance_squared_to(p6),
            pos.distance_squared_to(p7),
            pos.distance_squared_to(p8)
        };
        real_t smallest = dis[0];
        int smallest_index = 0;
        for(int i=1; i < 9; i++){
            if(dis[i] < smallest){
                smallest = dis[i];
                smallest_index = i;
            }
        }
        if(smallest_index==0){
            high = s[1];
            continue;
        }
        if(smallest_index==8){
            low = s[7];
            continue;
        }
        if(dis[smallest_index-1] < dis[smallest_index+1]){
            low = s[smallest_index -1];
            high = s[smallest_index];
        } else {
            high = s[smallest_index +1];
            low = s[smallest_index];
        }
    }
    return (low+high)/2.0;
}
/*
    Other function always return the direction start from smaller point id index
    to bigger one
    This has the direction of point which you define
*/
Vector3 MCurve::get_point_order_tangent(int32_t point_a,int32_t point_b,float t){
    ERR_FAIL_COND_V(!has_point(point_a), Vector3());
    ERR_FAIL_COND_V(!has_point(point_b), Vector3());
    const Point* a = points_buffer.ptr() + point_a;
    const Point* b = points_buffer.ptr() + point_b;
    Vector3 a_control = a->in; 
    Vector3 b_control = b->in;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(a->conn[i] == point_b){
            a_control = a->out;
        }
        if(b->conn[i] == point_a){
            b_control = b->out;
        }
    }
    return _get_bezier_tangent(a->position,b->position,a_control,b_control,t);
}

Vector3 MCurve::get_conn_tangent(int64_t conn_id,float t){
    ERR_FAIL_COND_V(!has_conn(conn_id), Vector3());
    Conn conn(conn_id);
    const Point* a = points_buffer.ptr() + conn.p.a;
    const Point* b = points_buffer.ptr() + conn.p.b;
    Vector3 a_control = a->in; 
    Vector3 b_control = b->in;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(a->conn[i] == conn.p.b){
            a_control = a->out;
        }
        if(b->conn[i] == conn.p.a){
            b_control = b->out;
        }
    }
    return _get_bezier_tangent(a->position,b->position,a_control,b_control,t);
}

Transform3D MCurve::get_point_order_transform(int32_t point_a,int32_t point_b,float t,bool tilt,bool scale){
    ERR_FAIL_COND_V(!has_point(point_a), Transform3D());
    ERR_FAIL_COND_V(!has_point(point_b), Transform3D());
    const Point* a = points_buffer.ptr() + point_a;
    const Point* b = points_buffer.ptr() + point_b;
    Vector3 a_control = a->in; 
    Vector3 b_control = b->in;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(a->conn[i] == point_b){
            a_control = a->out;
        }
        if(b->conn[i] == point_a){
            b_control = b->out;
        }
    }
    // See if is straight perpendiculare line which is not handled by _get_bezier_transform
    Transform3D ptrasform;
    if(a->position.is_equal_approx(a_control)
    &&b->position.is_equal_approx(b_control)
    && Math::is_equal_approx(a->position.x,b->position.x)
    && Math::is_equal_approx(a->position.z,b->position.z)){
        ptrasform = _get_bezier_transform(a->position,b->position,a_control,b_control,Vector3(0,0,1),t);
    }
    ptrasform = _get_bezier_transform(a->position,b->position,a_control,b_control,Vector3(0,1,0),t);
    // Applying tilt
    if(tilt){
        float current_tilt = Math::lerp(a->tilt,b->tilt,t);
        ptrasform.basis.rotate(ptrasform.basis[0],current_tilt); 
    }        
    // Applying scale
    if(scale){
        float current_scale = Math::lerp(a->scale,b->scale,t);
        ptrasform.basis.scale(Vector3(current_scale,current_scale,current_scale));
    }
    return ptrasform;
}

Transform3D MCurve::get_conn_transform(int64_t conn_id,float t,bool apply_tilt,bool apply_scale){
    ERR_FAIL_COND_V(!has_conn(conn_id), Transform3D());
    Conn conn(conn_id);
    const Point* a = points_buffer.ptr() + conn.p.a;
    const Point* b = points_buffer.ptr() + conn.p.b;
    Vector3 a_control = a->in; 
    Vector3 b_control = b->in;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(a->conn[i] == conn.p.b){
            a_control = a->out;
        }
        if(b->conn[i] == conn.p.a){
            b_control = b->out;
        }
    }
    // See if is straight perpendiculare line which is not handled by _get_bezier_transform
    Transform3D ptrasform;
    if(a->position.is_equal_approx(a_control)
    &&b->position.is_equal_approx(b_control)
    && Math::is_equal_approx(a->position.x,b->position.x)
    && Math::is_equal_approx(a->position.z,b->position.z)){
        ptrasform = _get_bezier_transform(a->position,b->position,a_control,b_control,Vector3(0,0,1),t);
    }
    ptrasform = _get_bezier_transform(a->position,b->position,a_control,b_control,Vector3(0,1,0),t);
    if(apply_tilt)
    {
        float current_tilt = Math::lerp(a->tilt,b->tilt,t);
        ptrasform.basis.rotate(ptrasform.basis[0],current_tilt); 
    }                                            
    if(apply_scale){
        float current_scale = Math::lerp(a->scale,b->scale,t);
        ptrasform.basis.scale(Vector3(current_scale,current_scale,current_scale));
    }
    return ptrasform;
}

void MCurve::get_conn_transforms(int64_t conn_id,const Vector<float>& t,Vector<Transform3D>& transforms,bool apply_tilt,bool apply_scale){
    ERR_FAIL_COND(!has_conn(conn_id));
    transforms.resize(t.size());
    Conn conn(conn_id);
    const Point* a = points_buffer.ptr() + conn.p.a;
    const Point* b = points_buffer.ptr() + conn.p.b;
    Vector3 a_control = a->in; 
    Vector3 b_control = b->in;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(a->conn[i] == conn.p.b){
            a_control = a->out;
        }
        if(b->conn[i] == conn.p.a){
            b_control = b->out;
        }
    }
    // See if is straight perpendiculare line which is not handled by _get_bezier_transform
    if(a->position.is_equal_approx(a_control)
    &&b->position.is_equal_approx(b_control)
    && Math::is_equal_approx(a->position.x,b->position.x)
    && Math::is_equal_approx(a->position.z,b->position.z)){
        for(int i=0; i < t.size(); i++){
            Transform3D ptrasform = _get_bezier_transform(a->position,b->position,a_control,b_control,Vector3(0,0,1),t[i]);
            float current_tilt = Math::lerp(a->tilt,b->tilt,t[i]);
            float current_scale = Math::lerp(a->scale,b->scale,t[i]);
            ptrasform.basis.rotate(ptrasform.basis[0],current_tilt);
            ptrasform.basis.scale(Vector3(current_scale,current_scale,current_scale));
            transforms.set(i,ptrasform);
        }
        return;
    }
    for(int i=0; i < t.size(); i++){
        Transform3D ptrasform = _get_bezier_transform(a->position,b->position,a_control,b_control,Vector3(0,1,0),t[i]);
        if(apply_tilt){
            float current_tilt = Math::lerp(a->tilt,b->tilt,t[i]);
            ptrasform.basis.rotate(ptrasform.basis[0],current_tilt);
        }
        if(apply_scale){
            float current_scale = Math::lerp(a->scale,b->scale,t[i]);
            ptrasform.basis.scale(Vector3(current_scale,current_scale,current_scale));
        }
        transforms.set(i,ptrasform);
    }
}

float MCurve::get_conn_lenght(int64_t conn_id){
    float* dis;
    if(conn_distances.has(conn_id)){
        dis = conn_distances[conn_id].dis;
    } else {
        dis = _bake_conn_distance(conn_id);
    }
    ERR_FAIL_COND_V(dis==nullptr,0.0f);
    return dis[DISTANCE_BAKE_TOTAL - 1];
}
Pair<float,float> MCurve::conn_ratio_limit_to_dis_limit(int64_t conn_id,const Pair<float,float>& limits){
    Pair<float,float> out;
    ERR_FAIL_COND_V(limits.first > limits.second,out);
    if(Math::is_equal_approx(limits.first,0.0f) && Math::is_equal_approx(limits.second,1.0f)){
        out.first = 0.0f;
        out.second = get_conn_lenght(conn_id);
        return out;
    }
    float* dis;
    if(conn_distances.has(conn_id)){
        dis = conn_distances[conn_id].dis;
    } else {
        dis = _bake_conn_distance(conn_id);
    }
    out.first = _get_conn_ratio_distance(dis,limits.first);
    out.second = _get_conn_ratio_distance(dis,limits.second);
    return out;
}
/*
    Not with order of connection
    in order of points provided
*/
float MCurve::get_point_order_distance_ratio(int32_t point_a,int32_t point_b,float distance){
    float* dis;
    Conn c(point_a,point_b);
    if(conn_distances.has(c.id)){
        dis = conn_distances[c.id].dis;
    } else {
        dis = _bake_conn_distance(c.id);
    }
    ERR_FAIL_COND_V(dis==nullptr,0.0f);
    distance = point_a > point_b ? dis[DISTANCE_BAKE_TOTAL - 1] - distance : distance;
    float t = _get_conn_distance_ratios(dis,distance);
    t = point_a > point_b ?  1.0f - t: t;
    return t;
}

float MCurve::get_conn_distance_ratio(int64_t conn_id,float distance) {
    float* dis;
    if(conn_distances.has(conn_id)){
        dis = conn_distances[conn_id].dis;
    } else {
        dis = _bake_conn_distance(conn_id);
    }
    ERR_FAIL_COND_V(dis==nullptr,0.0f);
    return _get_conn_distance_ratios(dis,distance);
}
/*
    return smallest and biggest ratio in order with Pair
*/
Pair<int,int> MCurve::get_conn_distances_ratios(int64_t conn_id,const Vector<float>& distances,Vector<float>& t){
    t.resize(distances.size());
    float* dis;
    if(conn_distances.has(conn_id)){
        dis = conn_distances[conn_id].dis;
    } else {
        dis = _bake_conn_distance(conn_id);
    }
    Pair<int,int> out;
    ERR_FAIL_COND_V(dis==nullptr,out);
    float smallest_ratio = 10.0;
    int smallest_ration_index = -1;
    float biggest_ratio = 0.0;
    int biggest_ratio_index = -1;
    for(int i=0; i < distances.size(); i++){
        float ratio = _get_conn_distance_ratios(dis,distances[i]);
        t.set(i,ratio);
        if(ratio < smallest_ratio){
            smallest_ratio = ratio;
            smallest_ration_index = i;
        }
        if(ratio > biggest_ratio){
            biggest_ratio = ratio;
            biggest_ratio_index = i;
        }
    }
    ERR_FAIL_COND_V(smallest_ration_index==-1 || biggest_ratio_index==-1,out);
    out.first = smallest_ration_index;
    out.second = biggest_ratio_index;
    return out;
}

float MCurve::_get_conn_ratio_distance(const float* baked_dis,const float ratio) const{
    if(ratio < 0.001){
        return 0.0f;
    }
    if(ratio > 0.999){
        return baked_dis[DISTANCE_BAKE_TOTAL - 1];
    }
    int ratio_index = ratio * (DISTANCE_BAKE_TOTAL - 1);
    float ratio_remainder = std::fmod(ratio , RATIO_BAKE_INTERVAL);
    return Math::lerp(baked_dis[ratio_index],baked_dis[ratio_index+1],ratio_remainder);
}

float MCurve::_get_conn_distance_ratios(const float* baked_dis,const float distance) const{
    if(distance <= 0){
        return 0.0;
    }
    if(distance >= baked_dis[DISTANCE_BAKE_TOTAL - 1]){
        return 1.0;
    }
    int low = 0;
    int high = DISTANCE_BAKE_TOTAL - 1;
    int middle;
    while (high >= low)
    {
        middle = (high + low) / 2;
        if(baked_dis[middle] < distance){
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    };
    #ifdef DEBUG_ENABLED
    ERR_FAIL_COND_V(high >= DISTANCE_BAKE_TOTAL - 1.0,0.0);
    #endif
    // our distance should be between these two
    float a;
    float b;
    float a_ratio;
    /// Despite its name hight is the lower bound here
    if(high < 0){ // is before point dis[0] and the zero lenght or start point
        a = 0;
        b = baked_dis[0];
        a_ratio = 0.0f;
    } else {
        a = baked_dis[high];
        b = baked_dis[high+1];
        a_ratio = (high + 1) * RATIO_BAKE_INTERVAL;
    }
    //return high * RATIO_BAKE_INTERVAL + RATIO_BAKE_INTERVAL;
    float dis_ratio = ((distance - a)/(b-a)) * RATIO_BAKE_INTERVAL;
    #ifdef DEBUG_ENABLED
    ERR_FAIL_COND_V(distance < a || distance > b, 0.0);
    #endif
    return dis_ratio + a_ratio;
}

/*
    Return the a pointer to ConnDistances float dis element
    if LENGTH_POINT_SAMPLE_COUNT = N
    a.pos --------------------------- b.pos
            0...1...2...3...,...,.....N-1
    There is not baked distnace at a.pos as it is always zero distance

*/
_FORCE_INLINE_ float* MCurve::_bake_conn_distance(int64_t conn_id){
    ERR_FAIL_COND_V(!has_conn(conn_id), nullptr);
    Conn conn(conn_id);
    const Point* a = points_buffer.ptr() + conn.p.a;
    const Point* b = points_buffer.ptr() + conn.p.b;
    Vector3 a_control = a->in; 
    Vector3 b_control = b->in;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(a->conn[i] == conn.p.b){
            a_control = a->out;
        }
        if(b->conn[i] == conn.p.a){
            b_control = b->out;
        }
    }

    ConnDistances conn_d;
    float _interval = 1.0f/LENGTH_POINT_SAMPLE_COUNT;
    float lenght = 0;
    Vector3 last_pos = a->position;
    for(int i=1; i < LENGTH_POINT_SAMPLE_COUNT; i++){
        Vector3 current_pos = a->position.bezier_interpolate(a_control,b_control,b->position,_interval*i);
        lenght += last_pos.distance_to(current_pos);
        last_pos = current_pos;
        if(i%DISTANCE_BAKE_INTERVAL == 0){
            conn_d.dis[(i/DISTANCE_BAKE_INTERVAL) -1] = lenght;
        }
    }
    lenght += last_pos.distance_to(b->position);
    conn_d.dis[DISTANCE_BAKE_TOTAL - 1] = lenght;
    conn_distances.insert(conn_id,conn_d);
    return conn_distances[conn_id].dis;
}
/*
    bellow rename_ ... methods has only internal use and should not be called
    for now it used for swaping two point
*/
void MCurve::toggle_conn_type(int32_t point, int64_t conn_id){
    ERR_FAIL_COND(!has_point(point));
    Conn c(conn_id);
    int32_t other_point = 0;
    if(point==c.p.a){
        other_point = c.p.b;
    } else if(point==c.p.b) {
        other_point = c.p.a;
    } else{
        return;
    }
    ERR_FAIL_COND(!has_point(other_point));
    Point* tp = points_buffer.ptrw() + point;
    for(int8_t i=0; i < MAX_CONN; i++){
        if(std::abs(tp->conn[i]) == other_point){
            tp->conn[i] = -tp->conn[i];
            Conn cc(point,std::abs(tp->conn[i]));
            baked_lines.erase(cc.id);
            return;
        }
    }
    WARN_PRINT("Can't find conn between "+itos(point)+" and "+itos(other_point));
}
/*
    ------------ will remove and recreate connection ----------------------
    If conn does not exist it will take care of it and will remove in all place were need
    It do the same if conn exist it will calculate its LOD and it will create that everywere is needed
    Also if conn exist and its lod is not right it will recaculate that
*/
void MCurve::validate_conn(int64_t conn_id,bool send_signal){
    if(send_signal){
        emit_signal("remove_connection",conn_id);
    }
    conn_distances.erase(conn_id);
    baked_lines.erase(conn_id);
    if(!has_conn(conn_id)){
        active_conn.erase(conn_id);
        conn_list.erase(conn_id);
        return;
    }
    Conn conn(conn_id);
    ERR_FAIL_COND(!has_point(conn.p.a)||!has_point(conn.p.a));
    int8_t a_lod = points_buffer[conn.p.a].lod;
    int8_t b_lod = points_buffer[conn.p.b].lod;
    int8_t conn_lod = a_lod < b_lod ? a_lod : b_lod;
    if(conn_lod > active_lod_limit){
        active_conn.erase(conn_id);
    } else {
        active_conn.insert(conn_id);
    }
    conn_list.insert(conn_id,conn_lod);
    if(send_signal){
        emit_signal("force_update_connection",conn_id);
    }
}
/*
    After calling this the connection id connected to these points will be invalid
*/
void MCurve::swap_points(const int32_t p_a,const int32_t p_b){
    if(p_a==p_b){
        return;
    }
    ERR_FAIL_COND(!has_point(p_a)||!has_point(p_b));
    Point a = points_buffer[p_a];
    Point b = points_buffer[p_b];
    // Correcting connection for p_a
    for(int i=0; i < MAX_CONN; i++){
        if(a.conn[i]!=INVALID_POINT_INDEX){
            int32_t other_index = std::abs(a.conn[i]);
            Point* other = points_buffer.ptrw() + other_index;
            if(other_index==p_b){ // in this conn_id will not change
                a.conn[i] = p_a * Math::sign(a.conn[i]);
                continue;
            }
            Conn old_conn(p_a,other_index);
            Conn new_conn(p_b,other_index);
            for(int j=0; j < MAX_CONN; j++){
                if(std::abs(other->conn[j])==p_a){
                    int32_t sign = Math::sign(other->conn[j]);
                    other->conn[j] = p_b * sign;
                    break;
                }
            }
        }
    }
    // Correcting connection for p_b
    for(int i=0; i < MAX_CONN; i++){
        if(b.conn[i]!=INVALID_POINT_INDEX){
            int32_t other_index = std::abs(b.conn[i]);
            Point* other = points_buffer.ptrw() + other_index;
            if(other_index==p_a){ // in this conn_id will not change
                b.conn[i] = p_b * Math::sign(b.conn[i]);
                continue;
            }
            Conn old_conn(p_b,other_index);
            Conn new_conn(p_a,other_index);
            for(int j=0; j < MAX_CONN; j++){
                if(std::abs(other->conn[j])==p_b){
                    int32_t sign = Math::sign(other->conn[j]);
                    other->conn[j] = p_a * sign;
                    break;
                }
            }
        }
    }
    bool is_a_active = active_points.has(p_a);
    bool is_b_active = active_points.has(p_b);
    if(is_a_active){
        active_points.insert(p_b);
    } else {
        active_points.erase(p_b);
    }
    if(is_b_active){
        active_points.insert(p_a);
    } else {
        active_points.erase(p_a);
    }
    if(octree && is_init_insert){
        octree->change_point_id(oct_id,a.position,p_a,p_b);
        octree->change_point_id(oct_id,b.position,p_b,p_a);
    }
    points_buffer.set(p_a,b);
    points_buffer.set(p_b,a);
    emit_signal("swap_point_id",p_a,p_b);
    emit_signal("curve_updated");
}

void MCurve::swap_points_with_validation(const int32_t p_a,const int32_t p_b){
    ERR_FAIL_COND(!has_point(p_a)||!has_point(p_b));
    HashSet<int64_t> validate_conn_list;
    for(int i=0; i < MAX_CONN; i++){
        if(points_buffer[p_a].conn[i]!=0){
            Conn tconn(p_a,std::abs(points_buffer[p_a].conn[i]));
            validate_conn_list.insert(tconn.id);
        }
        if(points_buffer[p_b].conn[i]!=0){
            Conn tconn(p_b,std::abs(points_buffer[p_b].conn[i]));
            validate_conn_list.insert(tconn.id);
        }
    }
    swap_points(p_a,p_b);
    for(int i=0; i < MAX_CONN; i++){
        if(points_buffer[p_a].conn[i]!=0){
            Conn tconn(p_a,std::abs(points_buffer[p_a].conn[i]));
            validate_conn_list.insert(tconn.id);
        }
        if(points_buffer[p_b].conn[i]!=0){
            Conn tconn(p_b,std::abs(points_buffer[p_b].conn[i]));
            validate_conn_list.insert(tconn.id);
        }
    }
    // validating
    for(HashSet<int64_t>::Iterator it=validate_conn_list.begin();it!=validate_conn_list.end();++it){
        validate_conn(*it);
    }
}
/*
    sort increasing or decreasing
    this will return new root point as that will change during sorting
*/
int32_t MCurve::sort_from(int32_t root_point,bool increasing){
    ERR_FAIL_COND_V(!has_point(root_point),root_point);
    PackedInt32Array all_points = get_point_conn_points_recursive(root_point);
    if(all_points.size() == 0){
        return root_point;
    }
    all_points.insert(0,root_point);
    PackedInt32Array all_points_sorted = all_points.duplicate();
    PackedInt64Array conns_before = get_conn_ids_exist(all_points);
    all_points_sorted.sort();
    if(!increasing){
        all_points_sorted.reverse();
    }
    for(int i=0; i < all_points.size(); i++){
        int32_t pid = all_points[i];
        int32_t sorted_pid = all_points_sorted[i];
        if( (sorted_pid < pid && increasing) || (sorted_pid > pid && !increasing) ){
            swap_points(pid,sorted_pid);
            // swaping in all_point
            // as we pass this index we don't check it we need to only change other index
            int oswap = all_points.find(sorted_pid);
            ERR_FAIL_COND_V(oswap==-1,root_point);
            all_points.set(oswap,pid);
        }
    }
    // Must use all_point_sorted as all_points is modified
    PackedInt64Array conns_after = get_conn_ids_exist(all_points_sorted);
    HashSet<int64_t> processed_conn;
    for(int i=0; i < conns_before.size(); i++){
        validate_conn(conns_before[i],false);
        processed_conn.insert(conns_before[i]);
    }
    for(int i=0; i < conns_after.size(); i++){
        if(!processed_conn.has(conns_after[i])){
            validate_conn(conns_after[i],false);
        }
    }
    emit_signal("recreate");
    return all_points_sorted[0];
}

void MCurve::move_point(int p_index,const Vector3& pos){
    ERR_FAIL_INDEX(p_index,points_buffer.size());
    Point* p = points_buffer.ptrw() + p_index;
    if(octree && is_init_insert){
        MOctree::PointMoveReq req(p_index,oct_id,p->position,pos);
        octree->add_move_req(req);
    }
    Vector3 diff = pos - p->position;
    p->position = pos;
    p->in += diff;
    p->out += diff;
    for(int i=0 ; i < MAX_CONN; i++){
        if(p->conn[i]!=0){
            Conn cc(std::abs(p->conn[i]),p_index);
            baked_lines.erase(cc.id);
        }
    }
    emit_signal("curve_updated");
}

void MCurve::move_point_in(int p_index,const Vector3& pos){
    ERR_FAIL_INDEX(p_index,points_buffer.size());
    Point* p = points_buffer.ptrw() + p_index;
    p->in = pos;
    for(int i=0 ; i < MAX_CONN; i++){
        if(p->conn[i]!=0){
            Conn cc(std::abs(p->conn[i]),p_index);
            baked_lines.erase(cc.id);
        }
    }
    emit_signal("curve_updated");
}

void MCurve::move_point_out(int p_index,const Vector3& pos){
    ERR_FAIL_INDEX(p_index,points_buffer.size());
    Point* p = points_buffer.ptrw() + p_index;
    p->out = pos;
    for(int i=0 ; i < MAX_CONN; i++){
        if(p->conn[i]!=0){
            Conn cc(std::abs(p->conn[i]),p_index);
            baked_lines.erase(cc.id);
        }
    }
    emit_signal("curve_updated");
}

int32_t MCurve::ray_active_point_collision(const Vector3& org,Vector3 dir,float threshold){
    ERR_FAIL_COND_V(threshold < 0.999,INVALID_POINT_INDEX);
    dir.normalize();
    for(int i=0; i < active_points.size(); i++){
        Vector3 pto = points_buffer[active_points[i]].position - org;
        pto.normalize();
        float dot = pto.dot(dir);
        if (dot > threshold){
            return active_points[i];
        }
        pto = points_buffer[active_points[i]].in - org;
        pto.normalize();
        dot = pto.dot(dir);
        if (dot > threshold){
            return active_points[i];
        }
        pto = points_buffer[active_points[i]].out - org;
        pto.normalize();
        dot = pto.dot(dir);
        if (dot > threshold){
            return active_points[i];
        }
    }
    return INVALID_POINT_INDEX;
}
/*
    Header in order -> Total header size 16 Byte
    uint32_t -> PointSave struct size
    uint32_t -> point index type size (currently should be int32_t which is 4 byte)
    uint32_t -> free_buffer_indicies size or count (Not size in byte)
    uint32_t -> points buffer size or count (Not size in byte)
*/
#define MCURVE_DATA_HEADER_SIZE 16
static int64_t PackedByteArray_decode_u32(const PackedByteArray *p_instance, int64_t p_offset) {
    uint64_t size = p_instance->size();
    ERR_FAIL_COND_V(p_offset < 0 || p_offset > (int64_t(size) - 4), 0);
    const uint8_t *r = p_instance->ptr();
    return decode_uint32(&r[p_offset]);
}
static void PackedByteArray_encode_u32(PackedByteArray *p_instance, int64_t p_offset, int64_t p_value) {
    uint64_t size = p_instance->size();
    ERR_FAIL_COND(p_offset < 0 || p_offset > int64_t(size) - 4);
    uint8_t *w = p_instance->ptrw();
    encode_uint32((uint32_t)p_value, &w[p_offset]);
}
void MCurve::_set_data(const PackedByteArray& data){
    ERR_FAIL_COND(data.size() < MCURVE_DATA_HEADER_SIZE);
    // Header
    ERR_FAIL_COND(PackedByteArray_decode_u32(&data,0)!=(uint32_t)sizeof(MCurve::PointSave));
    ERR_FAIL_COND(PackedByteArray_decode_u32(&data,4)!=(uint32_t)sizeof(int32_t));
    uint32_t free_indicies_count = PackedByteArray_decode_u32(&data,8);
    uint32_t points_buffer_count = PackedByteArray_decode_u32(&data,12);
    size_t size_free_indicies_byte = free_indicies_count * sizeof(int32_t);
    size_t size_points_buffer_byte = points_buffer_count * sizeof(MCurve::PointSave);
    ERR_FAIL_COND(data.size()!= MCURVE_DATA_HEADER_SIZE + size_free_indicies_byte + size_points_buffer_byte);

    // Finish header
    points_buffer.resize(points_buffer_count);
    free_buffer_indicies.resize(free_indicies_count);

    int64_t byte_offset = MCURVE_DATA_HEADER_SIZE;
    memcpy(free_buffer_indicies.ptrw(),data.ptr()+byte_offset,size_free_indicies_byte);
    byte_offset += size_free_indicies_byte;
    // Points buffer
    Vector<PointSave> points_save;
    points_save.resize(points_buffer_count);
    memcpy(points_save.ptrw(),data.ptr()+byte_offset,size_points_buffer_byte);
    points_buffer.resize(points_buffer_count);
    for(int i=0; i < points_buffer.size(); i++){
        points_buffer.set(i,points_save.get(i).get_point());
    }
}

PackedByteArray MCurve::_get_data(){
    PackedByteArray data;
    size_t size_free_indicies_byte = free_buffer_indicies.size() * sizeof(int32_t);
    size_t size_points_buffer_byte = points_buffer.size() * sizeof(PointSave);
    data.resize(MCURVE_DATA_HEADER_SIZE + size_free_indicies_byte + size_points_buffer_byte);
    // Header
    PackedByteArray_encode_u32(&data,0,(uint32_t)sizeof(MCurve::PointSave));
    PackedByteArray_encode_u32(&data,4,(uint32_t)sizeof(int32_t));
    PackedByteArray_encode_u32(&data,8,(uint32_t)free_buffer_indicies.size());
    PackedByteArray_encode_u32(&data,12,(uint32_t)points_buffer.size());
    int64_t byte_offset = MCURVE_DATA_HEADER_SIZE;
    //Finish header
    //copy size_free_indicies_byte
    memcpy(data.ptrw() + byte_offset,free_buffer_indicies.ptr(),size_free_indicies_byte);
    byte_offset += size_free_indicies_byte;
    //copy size_free_indicies_byte
    Vector<PointSave> points_save;
    points_save.resize(points_buffer.size());
    for(int i=0; i < points_buffer.size(); i++){
        points_save.set(i,points_buffer.get(i).get_point_save());
    }
    memcpy(data.ptrw()+byte_offset,points_save.ptr(),size_points_buffer_byte);
   return data;
}

void MCurve::set_bake_interval(float input){
    bake_interval = input;
    baked_lines.clear();
}

float MCurve::get_bake_interval(){
    return bake_interval;
}

void MCurve::set_active_lod_limit(int input){
    ERR_FAIL_INDEX(input,127);
    if(input == active_lod_limit){
        return;
    }
    int old_limit = active_lod_limit;
    active_lod_limit = input;
    Vector<int32_t> updated_points;
    for(int i=0;i<points_buffer.size();i++){
        if(free_buffer_indicies.has(i)){
            continue;
        }
        int8_t old_lod = points_buffer[i].lod > old_limit ? -1 : points_buffer[i].lod;
        int8_t new_lod = points_buffer[i].lod > input ? -1 : points_buffer[i].lod;
        if(new_lod == old_lod){
            continue;
        }
        updated_points.push_back(i);
        if(new_lod != -1){
            active_points.insert(i);
        } else {
            active_points.erase(i);
        }
        emit_signal("force_update_point",i);
    }
    HashSet<int64_t> processed_conn;
    for(int k=0; k < updated_points.size(); k++){
        int i = updated_points[k];
        for(int c=0;c < MAX_CONN; c++){
            if(points_buffer[i].conn[c] == INVALID_POINT_INDEX){
                continue;
            }
            int32_t j = std::abs(points_buffer[i].conn[c]);
            Conn cc(i,j);
            if(processed_conn.has(cc.id)){
                continue;
            }
            int8_t c_lod = points_buffer[i].lod < points_buffer[j].lod ? points_buffer[i].lod : points_buffer[j].lod;
            int8_t old_lod = c_lod > old_limit ? -1 : c_lod;
            int8_t new_lod = c_lod > input ? -1 : c_lod;
            if(old_lod == new_lod){
                processed_conn.insert(cc.id);
                continue;
            }
            if(new_lod==-1){
                active_conn.erase(cc.id);
            } else {
                active_conn.insert(cc.id);
            }
            emit_signal("force_update_connection",cc.id);
        }
    }
    emit_signal("curve_updated");
}

int MCurve::get_active_lod_limit(){
    return active_lod_limit;
}

float  MCurve::get_length_between_basic(const Point* a, const Point* b, const Vector3& a_control, const Vector3& b_control){
    float lenght = 0;
    Vector3 last_position = a->position;
    if(LENGTH_POINT_SAMPLE_COUNT_BASIC >= 1){
        float p_interval = 1.0 / LENGTH_POINT_SAMPLE_COUNT_BASIC;
        for(int i=1; i <= LENGTH_POINT_SAMPLE_COUNT_BASIC; i++){
            Vector3 current_position = a->position.bezier_interpolate(a_control,b_control,b->position,p_interval*i);
            lenght += current_position.distance_to(last_position);
            last_position = current_position;
        }
    }
    lenght += b->position.distance_to(last_position);
    
    return lenght;
}