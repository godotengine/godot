#include "mcurve_mesh_override.h"


void MCurveMeshOverride::_bind_methods(){
    ADD_SIGNAL(MethodInfo("id_updated",PropertyInfo(Variant::INT,"id")));

    ClassDB::bind_method(D_METHOD("set_mesh_override","id","index"), &MCurveMeshOverride::set_mesh_override);
    ClassDB::bind_method(D_METHOD("set_material_override","id","index"), &MCurveMeshOverride::set_material_override);
    ClassDB::bind_method(D_METHOD("get_mesh_override","id"), &MCurveMeshOverride::get_mesh_override);
    ClassDB::bind_method(D_METHOD("get_material_override","id"), &MCurveMeshOverride::get_material_override);
    ClassDB::bind_method(D_METHOD("clear_mesh_override","id"), &MCurveMeshOverride::clear_mesh_override);
    ClassDB::bind_method(D_METHOD("clear_material_override","id"), &MCurveMeshOverride::clear_material_override);
    ClassDB::bind_method(D_METHOD("clear_override","id"), &MCurveMeshOverride::clear_override);

    ClassDB::bind_method(D_METHOD("_set_data","input"), &MCurveMeshOverride::set_data);
    ClassDB::bind_method(D_METHOD("_get_data"), &MCurveMeshOverride::get_data);
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY,"_data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE),"_set_data","_get_data");
}

void MCurveMeshOverride::set_mesh_override(int64_t id,int mesh){
    Override ov;
    if(data.has(id)){
        ov = data[id];
    }
    ov.mesh = mesh;
    data.insert(id,ov);
    emit_signal("id_updated",id);
}

void MCurveMeshOverride::set_material_override(int64_t id,int material){
    Override ov;
    if(data.has(id)){
        ov = data[id];
    }
    ov.material = material;
    data.insert(id,ov);
    emit_signal("id_updated",id);
}

int MCurveMeshOverride::get_mesh_override(int64_t id){
    if(!data.has(id)){
        return -1;
    }
    return data[id].mesh;
}
int MCurveMeshOverride::get_material_override(int64_t id){
    if(!data.has(id)){
        return -1;
    }
    return data[id].material;
}

MCurveMeshOverride::Override MCurveMeshOverride::get_override(int64_t id){
    if(data.has(id)){
        return data[id];
    }
    Override ov;
    return ov;
}

void MCurveMeshOverride::clear_mesh_override(int64_t id){
    if(!data.has(id)){
        return;
    }
    MCurveMeshOverride::Override ov = data[id];
    if(ov.material==-1){
        data.erase(id);
        emit_signal("id_updated",id);
        return;
    }
    ov.mesh = -1;
    data.insert(id,ov);
    emit_signal("id_updated",id);
}

void MCurveMeshOverride::clear_material_override(int64_t id){
    if(!data.has(id)){
        return;
    }
    MCurveMeshOverride::Override ov = data[id];
    if(ov.mesh==-1){
        data.erase(id);
        emit_signal("id_updated",id);
        return;
    }
    ov.material = -1;
    data.insert(id,ov);
    emit_signal("id_updated",id);
}

void MCurveMeshOverride::clear_override(int64_t id){
    data.erase(id);
    emit_signal("id_updated",id);
}

void MCurveMeshOverride::set_data(const PackedByteArray& input){
    data.clear();
    if(input.size()==0){
        return;
    }
    int sb = sizeof(Pair<int64_t,Override>);
    ERR_FAIL_COND(input.size()%sb!=0);
    int dsize = input.size()/sb;
    for(int i=0; i < dsize; i++){
        Pair<int64_t,Override> ov;
        int pos = i*sb;
        memcpy(&ov,input.ptr()+pos,sb);
        data.insert(ov.first,ov.second);
    }
}

PackedByteArray MCurveMeshOverride::get_data(){
    PackedByteArray out;
    int sb = sizeof(Pair<int64_t,Override>);
    int counter = 0;
    for(HashMap<int64_t,Override>::Iterator it=data.begin();it!=data.end();++it){
        int pos = counter*sb;
        out.resize(pos+sb);
        Pair<int64_t,Override> dp(it->key,it->value);
        memcpy(out.ptrw()+pos,&dp,sb);
        counter++;
    }
    return out;
}

void MCurveMeshOverride::clear(){
    data.clear();
}