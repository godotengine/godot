#include "mremove_layer.h"
#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"






MRemoveLayer::MRemoveLayer(){

}
MRemoveLayer::~MRemoveLayer(){

}
String MRemoveLayer::_get_name(){
    return "Remove Layer";
}
Array MRemoveLayer::_get_property_list(){
    Array props;
    // p1
    Dictionary p1;
    p1["name"] = "hardness";
    p1["type"] = Variant::FLOAT;
    p1["hint"] = "range";
    p1["hint_string"] = "0.001";
    p1["default_value"] = hardness;
    p1["min"] = 0.0;
    p1["max"] = 0.95;
    //p2
    Dictionary p2;
    p2["name"] = "amount";
    p2["type"] = Variant::FLOAT;
    p2["hint"] = "range";
    p2["hint_string"] = "0.01";
    p2["default_value"] = amount;
    p2["min"] = 0;
    p2["max"] = 1;
    props.push_back(p1);
    props.push_back(p2);
    return props;
}
void MRemoveLayer::_set_property(String prop_name, Variant value){
    if (prop_name == "hardness"){
        hardness = value;
        return;
    }
    if (prop_name == "amount")
    {
        amount = value;
        return;
    }
}
bool MRemoveLayer::is_two_point_brush(){
    return false;
}
void MRemoveLayer::before_draw(){

}
float MRemoveLayer::get_height(uint32_t x,uint32_t y){
    float height = grid->get_height_by_pixel(x,y);
    float base_height = height - grid->get_height_by_pixel_in_layer(x,y);
    Vector3 world_pos = grid->get_pixel_world_pos(x,y);
    real_t dis = grid->brush_world_pos.distance_to(world_pos);
    dis = dis/grid->brush_radius;
    float ratio = VariantUtilityFunctions::smoothstep(1,hardness,dis)*grid->get_brush_mask_value(x,y)*amount;
    return (base_height - height)*ratio + height;
}