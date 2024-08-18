#include "mmesh_lod.h"


void MMeshLod::_bind_methods(){
    ADD_SIGNAL(MethodInfo("meshes_changed"));

    ClassDB::bind_method(D_METHOD("set_meshes","input"), &MMeshLod::set_meshes);
    ClassDB::bind_method(D_METHOD("get_meshes"), &MMeshLod::get_meshes);
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY,"meshes",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE), "set_meshes" , "get_meshes");
}

MMeshLod::MMeshLod(){
    meshes.resize(1);
}

MMeshLod::~MMeshLod(){
}

RID MMeshLod::get_mesh_rid(int8_t lod){
    if(meshes.size() > lod && lod >= 0){
        Ref<Mesh> mesh = meshes[lod];
        if(mesh.is_valid()){
            return mesh->get_rid();
        }
    }
    return RID();
}

Ref<Mesh> MMeshLod::get_mesh(int8_t lod){
    Ref<Mesh> out;
    if(meshes.size() > lod && lod >= 0){
        out = meshes[lod];
    }
    return out;
}

void MMeshLod::set_meshes(TypedArray<Mesh> input){
    meshes = input;
}

TypedArray<Mesh> MMeshLod::get_meshes(){
    return meshes;
}

bool MMeshLod::_set(const StringName &_name, const Variant &p_value){
    String p_name = _name;
    if(p_name==String("lod_count")){
        int new_size = p_value;
        if(new_size > 0 && new_size < 127 && new_size!=meshes.size()){
            meshes.resize(new_size);
            notify_property_list_changed();
            emit_signal("meshes_changed");
            return true;
        }
        return false;
    }
    if(p_name.begins_with("Mesh_LOD_")){
        Ref<Mesh> new_mesh = p_value;
        PackedStringArray s = p_name.split("_");
        int index = s[2].to_int();
        Ref<Mesh> old_mesh = meshes[index];
        meshes[index] = new_mesh;
        emit_signal("meshes_changed");
        return true;
    }
    return false;
}

bool MMeshLod::_get(const StringName &_name, Variant &r_ret) const{
    String p_name = _name;
    if(p_name==String("lod_count")){
        r_ret = meshes.size();
        return true;
    }
    if(p_name.begins_with("Mesh_LOD_")){
        PackedStringArray s = p_name.split("_");
        int index = s[2].to_int();
        if(index >= meshes.size()){
            return false;
        }
        r_ret = meshes[index];
        return true;
    }
    return false;
}
void MMeshLod::_get_property_list(List<PropertyInfo> *p_list) const{
    PropertyInfo l(Variant::INT, "lod_count");
    p_list->push_back(l);
    for(int8_t i=0; i < meshes.size() ; i++){
        PropertyInfo m(Variant::OBJECT, "Mesh_LOD_"+itos(i), PROPERTY_HINT_RESOURCE_TYPE, "Mesh", PROPERTY_USAGE_EDITOR);
        p_list->push_back(m);
    }
}

bool MMeshLod::_property_can_revert(const StringName &p_name) const {
    if(p_name.begins_with("Mesh_LOD_")){
        return false;
    }
    return true;
}