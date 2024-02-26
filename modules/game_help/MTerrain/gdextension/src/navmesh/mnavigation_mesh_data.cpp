#include "mnavigation_mesh_data.h"


void MNavigationMeshData::_bind_methods(){
    ClassDB::bind_method(D_METHOD("set_data","input"), &MNavigationMeshData::set_data);
    ClassDB::bind_method(D_METHOD("get_data"), &MNavigationMeshData::get_data);
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY,"data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE|PROPERTY_USAGE_READ_ONLY),"set_data","get_data");
    ClassDB::bind_method(D_METHOD("set_on_all_at_creation","input"), &MNavigationMeshData::set_on_all_at_creation);
    ClassDB::bind_method(D_METHOD("get_on_all_at_creation"), &MNavigationMeshData::get_on_all_at_creation);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"on_all_at_creation"), "set_on_all_at_creation","get_on_all_at_creation");
}



void MNavigationMeshData::set_data(const PackedByteArray& d){
    data = d;
}

PackedByteArray MNavigationMeshData::get_data(){
    return data;
}


void MNavigationMeshData::set_on_all_at_creation(bool input){
    on_all_at_creation = input;
}

bool MNavigationMeshData::get_on_all_at_creation(){
    return on_all_at_creation;
}