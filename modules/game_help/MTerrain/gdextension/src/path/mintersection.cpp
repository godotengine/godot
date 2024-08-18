#include "mintersection.h"



void MIntersection::_bind_methods(){
    ClassDB::bind_method(D_METHOD("set_mesh","input"), &MIntersection::set_mesh);
    ClassDB::bind_method(D_METHOD("get_mesh"), &MIntersection::get_mesh);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"mesh",PROPERTY_HINT_RESOURCE_TYPE,"MMeshLod"),"set_mesh","get_mesh");

    ClassDB::bind_method(D_METHOD("set_sockets","input"), &MIntersection::set_sockets);
    ClassDB::bind_method(D_METHOD("get_sockets"), &MIntersection::get_sockets);
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY,"sockets",PROPERTY_HINT_RESOURCE_TYPE,"MMeshLod"),"set_sockets","get_sockets");

    ClassDB::bind_method(D_METHOD("get_debug_mesh"), &MIntersection::get_debug_mesh);
    ClassDB::bind_method(D_METHOD("generate_mesh_info"), &MIntersection::generate_mesh_info);
}

bool MIntersection::is_init(){
    return _is_init;
}

Ref<MIntersectionInfo> MIntersection::get_mesh_info(int lod){
    return mesh_info[lod];
}

void MIntersection::generate_mesh_info(){
    mesh_info.clear();
    if(!mesh.is_valid()){
        return;
    }
    TypedArray<Mesh> p_meshes = mesh->get_meshes();
    mesh_info.resize(p_meshes.size());
    HashMap<RID,Ref<MIntersectionInfo>> hm;
    for(int i=0; i < p_meshes.size(); i++){
        Ref<Mesh> m = p_meshes[i];
        if(m.is_valid()){
            Ref<MIntersectionInfo> ii;
            if(hm.has(m->get_rid())){
                ii = hm[m->get_rid()];
            } else{
                ii.instantiate();
                ii->mesh_rid = m->get_rid();
                _generate_mesh_info(m,ii);
                hm.insert(m->get_rid(),ii);
            }
            mesh_info.set(i,ii);
        }
    }
    _is_init = true;
}
/*
    All distance are squared distance
*/
void MIntersection::_generate_mesh_info(Ref<Mesh> m,Ref<MIntersectionInfo> info){
    ERR_FAIL_COND(sockets.size() == 0);
    ERR_FAIL_COND(m.is_null());
    info->material = m->surface_get_material(0);
    Array surf_info = m->surface_get_arrays(0);
    info->vertex = surf_info[Mesh::ARRAY_VERTEX];
    info->normal = surf_info[Mesh::ARRAY_NORMAL];
    info->tangent = surf_info[Mesh::ARRAY_TANGENT];
    info->color = surf_info[Mesh::ARRAY_COLOR];
    info->uv = surf_info[Mesh::ARRAY_TEX_UV];
    info->uv2 = surf_info[Mesh::ARRAY_TEX_UV2];
    info->index = surf_info[Mesh::ARRAY_INDEX];
    int num_sockets = sockets.size();
    info->weights.resize(info->vertex.size() * num_sockets);
    info->weights.fill(0.0f);
    
    struct SocketDistance
    {
        int socket_index;
        float distance;
        float inv_distnace;
        bool operator<(const SocketDistance& other) const {
            return distance < other.distance;
        }
    };

    struct VertexSockets
    {
        bool is_set_as_main = false;
        float inv_distance_sum = 0;
        Vector<SocketDistance> distance_socket;
    };
    Vector<VertexSockets> vertex_socket;
    vertex_socket.resize(info->vertex.size());
    // Determining the socket distances and distance sum
    for(int i=0; i < info->vertex.size(); i++){
        for(int j=0; j < num_sockets; j++){
            Transform3D s = sockets[j];
            Vector3 s_pos = s.origin;
            s_pos.y = 0.0f;
            Vector3 v_pos = info->vertex[i];
            v_pos.y = 0.0f;
            float d = v_pos.distance_squared_to(s_pos);
            float inv_d = d < 0.001f ? 1.0f/0.001f : 1.0f/d;
            vertex_socket.ptrw()[i].distance_socket.push_back({j,d,inv_d});
            vertex_socket.ptrw()[i].inv_distance_sum += inv_d;
        }
    }
    // Sorting
    for(int i=0; i < info->vertex.size(); i++){
        vertex_socket.ptrw()[i].distance_socket.sort();
    }
    // Determingin main sockets vertex
    for(int i=0; i < info->vertex.size(); i++){
        Vector3 v_pos = info->vertex[i];
        v_pos.y = 0; // we determine everything in flat plane
        // We start from closest one!!!
        for(int j=0; j < vertex_socket.ptrw()[i].distance_socket.size(); j++){
            if(vertex_socket.ptrw()[i].is_set_as_main){
                continue; // if already set nothing to do
            }
            int socket_index = vertex_socket.ptrw()[i].distance_socket[j].socket_index;
            Transform3D s = sockets[socket_index];
            s.origin.y = 0;
            Vector3 dir = s.origin - v_pos;
            dir.normalize();
            Vector3 z = s.basis.get_column(2);
            float adot = std::abs(dir.dot(z)); // Dot product to z direction of basis
            if(adot > 0.999 || v_pos.is_equal_approx(s.origin)){ // Setting this socket index as main controller
                vertex_socket.ptrw()[i].is_set_as_main = true;
                info->weights.ptrw()[(i * num_sockets) + socket_index] = 1.0;
            }
        }
    }
    // Determingin other vertecies weight
    // Weight is determined by square distnace
    for(int i=0; i < info->vertex.size(); i++){
        if(vertex_socket.ptrw()[i].is_set_as_main){
            continue; // if already set nothing to do
        }
        float inv_dis_sum = vertex_socket.ptrw()[i].inv_distance_sum;
        for(int j=0; j < vertex_socket.ptrw()[i].distance_socket.size(); j++){
            int socket_index = vertex_socket.ptrw()[i].distance_socket[j].socket_index;
            float socket_inv_dis = vertex_socket.ptrw()[i].distance_socket[j].inv_distnace;
            
            float w = socket_inv_dis/inv_dis_sum;
            info->weights.set((i * num_sockets) + socket_index, w);
            
        }
        float w_sum=0;
        for(int j=0; j < vertex_socket.ptrw()[i].distance_socket.size(); j++){
            int socket_index = vertex_socket.ptrw()[i].distance_socket[j].socket_index;
            float dis = vertex_socket.ptrw()[i].distance_socket[j].distance;
            float w = info->weights[(i * num_sockets) + socket_index];
            w_sum += w;
        }
    }

    return;
    /// Debug mesh creating
    if(debug_mesh.is_null()){
        debug_mesh.instantiate();
    }
    PackedColorArray vcol;
    vcol.resize(info->vertex.size());
    memcpy(vcol.ptrw(),info->weights.ptr(),info->weights.size() * sizeof(float));
    Array msurf_info;
    msurf_info.resize(Mesh::ARRAY_MAX);
    msurf_info[Mesh::ARRAY_VERTEX] = info->vertex;
    msurf_info[Mesh::ARRAY_COLOR] = vcol;
    msurf_info[Mesh::ARRAY_INDEX] = info->index;

    debug_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES,msurf_info);

}

Ref<ArrayMesh> MIntersection::get_debug_mesh(){
    return debug_mesh;
}


void MIntersection::set_mesh(Ref<MMeshLod> input){
    mesh = input;
}
Ref<MMeshLod> MIntersection::get_mesh(){
    return mesh;
}
int MIntersection::get_mesh_count(){
    if(mesh.is_valid()){
        return mesh->get_meshes().size();
    }
    return 0;
}
void MIntersection::set_sockets(TypedArray<Transform3D> input){
    sockets = input;
}
TypedArray<Transform3D> MIntersection::get_sockets(){
    return sockets;
}

int MIntersection::get_socket_count(){
    return sockets.size();
}

Vector<Transform3D> MIntersection::_get_sockets(){
    Vector<Transform3D> out;
    out.resize(sockets.size());
    for(int i=0; i < sockets.size(); i++){
        out.set(i,sockets[i]);
    }
    return out;
}