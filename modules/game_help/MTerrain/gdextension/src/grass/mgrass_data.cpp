#include "mgrass_data.h"

#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"


void MGrassData::_bind_methods(){
    ClassDB::bind_method(D_METHOD("add","d"), &MGrassData::add);
    ClassDB::bind_method(D_METHOD("print_all_data"), &MGrassData::print_all_data);

    ClassDB::bind_method(D_METHOD("set_data","input"), &MGrassData::set_data);
    ClassDB::bind_method(D_METHOD("get_data"), &MGrassData::get_data);
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY,"data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE|PROPERTY_USAGE_READ_ONLY),"set_data","get_data");
    ClassDB::bind_method(D_METHOD("set_density","input"), &MGrassData::set_density);
    ClassDB::bind_method(D_METHOD("get_density"), &MGrassData::get_density);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"density",PROPERTY_HINT_ENUM,M_H_SCALE_LIST_STRING),"set_density","get_density");
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

void MGrassData::set_density(int input){
    float l[] = M_H_SCALE_LIST;
    density = l[input];
    density_index = input;
}

int MGrassData::get_density(){
    return density_index;
}

void MGrassData::add(int d) {
    data.push_back((uint8_t)d);
}

void MGrassData::print_all_data() {
    for(int i=0; i< data.size();i++){
        VariantUtilityFunctions::_print("i ",itos(i), " --> ", itos(data[i]));
    }
}


void MGrassData::check_undo(){
    if(current_undo_id - lowest_undo_id > 6){
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
        undo_data.insert(current_undo_id,d);
    }
    current_undo_id++;
}

void MGrassData::undo(){
    if(current_undo_id<=lowest_undo_id){
        return;
    }
    current_undo_id--;
    if(undo_data.has(current_undo_id)){
        MGrassUndoData d = undo_data[current_undo_id];
        memcpy(data.ptrw(),d.data,data.size());
        d.free();
        undo_data.erase(current_undo_id);
    }
}

