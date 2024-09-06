#include "mgrass_data.h"

#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"


void MGrassData::_bind_methods(){
    ClassDB::bind_method(D_METHOD("set_data","input"), &MGrassData::set_data);
    ClassDB::bind_method(D_METHOD("get_data"), &MGrassData::get_data);
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY,"data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE),"set_data","get_data");
    
    ClassDB::bind_method(D_METHOD("set_backup_data","input"), &MGrassData::set_backup_data);
    ClassDB::bind_method(D_METHOD("get_backup_data"), &MGrassData::get_backup_data);
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY,"backup_data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE),"set_backup_data","get_backup_data");    
    
    ClassDB::bind_method(D_METHOD("set_density","input"), &MGrassData::set_density);
    ClassDB::bind_method(D_METHOD("get_density"), &MGrassData::get_density);
    // For compatibilty
    ADD_PROPERTY(PropertyInfo(Variant::INT,"density",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NONE),"set_density","get_density");
    ADD_PROPERTY(PropertyInfo(Variant::INT,"grass_cell_size",PROPERTY_HINT_ENUM,M_H_SCALE_LIST_STRING),"set_density","get_density");
}

MGrassData::MGrassData(){

}
MGrassData::~MGrassData(){
    for(HashMap<int,MGrassUndoData>::Iterator it=undo_data.begin();it!=undo_data.end();++it){
        it->value.free();
    }
}

void MGrassData::set_data(const PackedByteArray& d){
    data = d;
}

const PackedByteArray& MGrassData::get_data(){
    return data;
}

void MGrassData::set_backup_data(const PackedByteArray& d){
    backup_data = d;
}

const PackedByteArray& MGrassData::get_backup_data(){
    return backup_data;
}

void MGrassData::set_density(int input){
    float l[] = M_H_SCALE_LIST;
    density = l[input];
    density_index = input;
}

int MGrassData::get_density(){
    return density_index;
}

bool MGrassData::backup_exist(){
    return backup_data.size() > 0;
}

void MGrassData::backup_create(){
    ERR_FAIL_COND(backup_exist());
    ERR_FAIL_COND(data.size()==0);
    backup_data = data.duplicate();
}

void MGrassData::backup_merge(){
    ERR_FAIL_COND(!backup_exist());
    backup_data.resize(0);
}

void MGrassData::backup_restore(){
    ERR_FAIL_COND(!backup_exist());
    data = backup_data.duplicate();
    backup_data.resize(0);
}

void MGrassData::check_undo(){
    if(current_undo_id - lowest_undo_id > 4){
        if(undo_data.has(lowest_undo_id)){
            MGrassUndoData d = undo_data[lowest_undo_id];
            d.free();
            undo_data.erase(lowest_undo_id);
        }
        lowest_undo_id++;
    }
    if(!undo_data.has(current_undo_id)){
        MGrassUndoData d;
        d.data = memnew_arr(uint8_t,data.size());
        memcpy(d.data,data.ptr(),data.size());
        if(backup_exist()){
            d.backup_data = memnew_arr(uint8_t,backup_data.size());
            memcpy(d.backup_data,backup_data.ptr(),backup_data.size());
        }
        undo_data.insert(current_undo_id,d);
    }
    current_undo_id++;
}

void MGrassData::clear_undo(){
    for(HashMap<int,MGrassUndoData>::Iterator it=undo_data.begin(); it!=undo_data.end(); ++it){
        it->value.free();
    }
    current_undo_id = 0;
    lowest_undo_id = 0;
}

void MGrassData::undo(){
    if(current_undo_id<=lowest_undo_id){
        return;
    }
    current_undo_id--;
    if(undo_data.has(current_undo_id)){
        MGrassUndoData d = undo_data[current_undo_id];
        memcpy(data.ptrw(),d.data,data.size());
        if(d.backup_data!=nullptr){
            backup_data.resize(data.size());
            memcpy(backup_data.ptrw(),d.backup_data,backup_data.size());
        }
        d.free();
        undo_data.erase(current_undo_id);
    }
}

