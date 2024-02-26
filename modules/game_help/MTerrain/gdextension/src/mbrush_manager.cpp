#include "mbrush_manager.h"
#include "core/variant/variant.h"

#include "height_brushes/mraise.h"
#include "height_brushes/mtoheight.h"
#include "height_brushes/msmooth.h"
#include "height_brushes/mhole.h"
#include "height_brushes/mremove_layer.h"

#include "color_brushes/mpaint_color.h"
#include "color_brushes/mchannel_painter.h"
#include "color_brushes/mbitwise_brush.h"
#include "color_brushes/mpaint_256.h"
#include "color_brushes/mpaint_16.h"


MBrushManager::MBrushManager(){
    add_height_brush(memnew(MRaise));
    add_height_brush(memnew(MToHeight));
    add_height_brush(memnew(MSmooth));
    add_height_brush(memnew(MRemoveLayer));
    add_height_brush(memnew(MHole));

    add_color_brush(memnew(MPaintColor));
    add_color_brush(memnew(MChannelPainter));
    add_color_brush(memnew(MBitwiseBrush));
    add_color_brush(memnew(MPaint256));
    add_color_brush(memnew(MPaint16));
}

void MBrushManager::_bind_methods(){
    ClassDB::bind_method(D_METHOD("get_height_brush_list"), &MBrushManager::get_height_brush_list);
    ClassDB::bind_method(D_METHOD("get_height_brush_id", "brush_name"), &MBrushManager::get_height_brush_id);
    ClassDB::bind_method(D_METHOD("get_height_brush_property", "brush_id"), &MBrushManager::get_height_brush_property);
    ClassDB::bind_method(D_METHOD("set_height_brush_property", "property_name","value","brush_id"), &MBrushManager::set_height_brush_propert);
}

MBrushManager::~MBrushManager(){
    for(int i=0;i<height_brushes.size();i++){
        memdelete(height_brushes[i]);
    }
}

void MBrushManager::add_height_brush(MHeightBrush* brush){\
    ERR_FAIL_COND_MSG(height_brush_map.has(brush->_get_name()), "Duplicate brush name: "+brush->_get_name());
    height_brushes.append(brush);
    height_brush_map.insert(brush->_get_name(), height_brushes.size() -1);
}

void MBrushManager::add_color_brush(MColorBrush* brush){
    ERR_FAIL_COND_MSG(color_brush_map.has(brush->_get_name()), "Duplicate brush name: "+brush->_get_name());
    color_brushes.append(brush);
    color_brush_map.insert(brush->_get_name(), color_brushes.size() -1);
}

MHeightBrush* MBrushManager::get_height_brush(int brush_id){
    ERR_FAIL_COND_V(brush_id >= height_brushes.size(), nullptr);
    return height_brushes[brush_id];
}

PackedStringArray MBrushManager::get_height_brush_list(){
    PackedStringArray out;
    for(int i=0;i<height_brushes.size();i++){
        out.append(height_brushes[i]->_get_name());
    }
    return out;
}

int MBrushManager::get_height_brush_id(String brush_name){
    if(!height_brush_map.has(brush_name)) return -1;
    return height_brush_map[brush_name];
}

Array MBrushManager::get_height_brush_property(int brush_id){
    return height_brushes[brush_id]->_get_property_list();
}

void MBrushManager::set_height_brush_propert(String prop_name,Variant value,int brush_id){
    height_brushes[brush_id]->_set_property(prop_name,value);
}

MColorBrush* MBrushManager::get_color_brush(int brush_id){
    ERR_FAIL_COND_V(brush_id >= color_brushes.size(), nullptr);
    return color_brushes[brush_id];
}

MColorBrush* MBrushManager::get_color_brush_by_name(String brush_name){
    ERR_FAIL_COND_V(!color_brush_map.has(brush_name),nullptr);
    int id = color_brush_map.get(brush_name);
    return color_brushes[id];
}

PackedStringArray MBrushManager::get_color_brush_list(){
    PackedStringArray out;
    for(int i=0;i<color_brushes.size();i++){
        out.append(color_brushes[i]->_get_name());
    }
    return out;
}