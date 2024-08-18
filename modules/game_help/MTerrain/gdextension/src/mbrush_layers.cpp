#include "mbrush_layers.h"
#include "core/variant/variant.h"

#include "core/io/image.h"
#include "mcolor_brush.h"
#define BRUSH_NAMES "Color Paint,Channel Painter,Bitwise Brush,Paint 256,Paint 16"


void MBrushLayers::_bind_methods(){
    ClassDB::bind_method(D_METHOD("set_layers_title","input"), &MBrushLayers::set_layers_title);
    ClassDB::bind_method(D_METHOD("get_layers_title"), &MBrushLayers::get_layers_title);
    ADD_PROPERTY(PropertyInfo(Variant::STRING,"layers_title"), "set_layers_title","get_layers_title");
    ClassDB::bind_method(D_METHOD("set_uniform_name","input"), &MBrushLayers::set_uniform_name);
    ClassDB::bind_method(D_METHOD("get_uniform_name"), &MBrushLayers::get_uniform_name);
    ADD_PROPERTY(PropertyInfo(Variant::STRING,"uniform_name"), "set_uniform_name","get_uniform_name");
    ClassDB::bind_method(D_METHOD("set_brush_name","input"), &MBrushLayers::set_brush_name);
    ClassDB::bind_method(D_METHOD("get_brush_name"), &MBrushLayers::get_brush_name);
    ADD_PROPERTY(PropertyInfo(Variant::STRING,"brush_name",PROPERTY_HINT_ENUM,BRUSH_NAMES), "set_brush_name","get_brush_name");
    ClassDB::bind_method(D_METHOD("set_layers","input"), &MBrushLayers::set_layers);
    ClassDB::bind_method(D_METHOD("get_layers"), &MBrushLayers::get_layers);
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY,"layers",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE), "set_layers","get_layers");
    ClassDB::bind_method(D_METHOD("set_layers_num","input"), &MBrushLayers::set_layers_num);
    ClassDB::bind_method(D_METHOD("get_layers_num"), &MBrushLayers::get_layers_num);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"layers_num",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NONE), "set_layers_num","get_layers_num");
}

MBrushLayers::MBrushLayers(){
    {
        Vector<LayerProps> color_brush;
        LayerProps hardness = {PropertyInfo(Variant::FLOAT,"hardness",PROPERTY_HINT_RANGE,"0,1",PROPERTY_USAGE_EDITOR),Variant(0.9)};
        LayerProps color = {PropertyInfo(Variant::COLOR,"color",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR),Variant(Color(1.0,0.0,0.0,1.0))};
        color_brush.push_back(hardness);
        color_brush.push_back(color);
        layer_props.insert("Color Paint",color_brush);
    }
    // Channel Painter
    {
        Vector<LayerProps> channel_paiter;
        LayerProps hardness = {PropertyInfo(Variant::FLOAT,"hardness",PROPERTY_HINT_RANGE,"0,1",PROPERTY_USAGE_EDITOR),Variant(0)};
        LayerProps red = {PropertyInfo(Variant::BOOL,"red",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR),Variant(false)};
        LayerProps green = {PropertyInfo(Variant::BOOL,"green",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR),Variant(false)};
        LayerProps blue = {PropertyInfo(Variant::BOOL,"blue",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR),Variant(false)};
        LayerProps alpha = {PropertyInfo(Variant::BOOL,"alpha",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR),Variant(false)};
        LayerProps red_value = {PropertyInfo(Variant::FLOAT,"red-value",PROPERTY_HINT_RANGE,"0,1",PROPERTY_USAGE_EDITOR),Variant(0)};
        LayerProps green_value = {PropertyInfo(Variant::FLOAT,"green-value",PROPERTY_HINT_RANGE,"0,1",PROPERTY_USAGE_EDITOR),Variant(0)};
        LayerProps blue_value = {PropertyInfo(Variant::FLOAT,"blue-value",PROPERTY_HINT_RANGE,"0,1",PROPERTY_USAGE_EDITOR),Variant(0)};
        LayerProps alpha_value = {PropertyInfo(Variant::FLOAT,"alpha-value",PROPERTY_HINT_RANGE,"0,1",PROPERTY_USAGE_EDITOR),Variant(0)};
        channel_paiter.push_back(hardness);
        channel_paiter.push_back(red);
        channel_paiter.push_back(red_value);
        channel_paiter.push_back(green);
        channel_paiter.push_back(green_value);
        channel_paiter.push_back(blue);
        channel_paiter.push_back(blue_value);
        channel_paiter.push_back(alpha);
        channel_paiter.push_back(alpha_value);
        layer_props.insert("Channel Painter",channel_paiter);
    }
    // Channel Painter
    {
        Vector<LayerProps> bit_painter;
        LayerProps value = {PropertyInfo(Variant::BOOL,"value",PROPERTY_HINT_RANGE,"",PROPERTY_USAGE_EDITOR),Variant(false)};
        LayerProps bit = {PropertyInfo(Variant::INT,"bit",PROPERTY_HINT_RANGE,"0,127",PROPERTY_USAGE_EDITOR),Variant(0)};
        bit_painter.push_back(value);
        bit_painter.push_back(bit);
        layer_props.insert("Bitwise Brush",bit_painter);
    }
    // Paint 256
    {
        Vector<LayerProps> painter;
        LayerProps bit = {PropertyInfo(Variant::INT,"paint-layer",PROPERTY_HINT_RANGE,"0,255",PROPERTY_USAGE_EDITOR),Variant(0)};
        painter.push_back(bit);
        layer_props.insert("Paint 256",painter);
    }
    // Paint 16
    {
        Vector<LayerProps> painter;
        LayerProps bit = {PropertyInfo(Variant::INT,"paint-layer",PROPERTY_HINT_RANGE,"0,15",PROPERTY_USAGE_EDITOR),Variant(0)};
        painter.push_back(bit);
        layer_props.insert("Paint 16",painter);
    }
}
MBrushLayers::~MBrushLayers(){

}

void MBrushLayers::set_layers_title(String input){
    layers_title = input;
}

String MBrushLayers::get_layers_title(){
    return layers_title;
}

void MBrushLayers::set_uniform_name(String input){
    uniform_name = input;
}
String MBrushLayers::get_uniform_name(){
    return uniform_name;
}

void MBrushLayers::set_brush_name(String input){
    if(input == brush_name){
        return;
    }
    brush_name = input;
    for(int i=0;i<layers.size();i++){
        Dictionary org = layers[i];
        Dictionary dic;
        dic["NAME"]=org["NAME"];
        dic["ICON"]=org["ICON"];
        if(layer_props.has(brush_name)){
            Vector<LayerProps> p = layer_props.get(brush_name);
            for(int j=0;j<p.size();j++){
                dic[p[j].pinfo.name] = p[j].def_value;
            }
        }
        layers[i] = dic;
    }
    notify_property_list_changed();
}

String MBrushLayers::get_brush_name(){
    return brush_name;
}

void MBrushLayers::set_layers_num(int input){
    ERR_FAIL_COND(input<0);
    layers.resize(input);
    for(int i=0;i<layers.size();i++){
        if(layers[i].get_type() == Variant::NIL){
            Dictionary dic;
            dic["NAME"]="";
            dic["ICON"]="";
            if(layer_props.has(brush_name)){
                Vector<LayerProps> p = layer_props.get(brush_name);
                for(int j=0;j<p.size();j++){
                    dic[p[j].pinfo.name] = p[j].def_value;
                }
            }
            layers[i] = dic;
        }
    }
    notify_property_list_changed();
}

int MBrushLayers::get_layers_num(){
    return layers.size();
}

void MBrushLayers::set_layers(Array input){
    layers = input;
}
Array MBrushLayers::get_layers(){
    return layers;
}



void MBrushLayers::_get_property_list(List<PropertyInfo> *p_list) const {
    PropertyInfo lnum(Variant::INT, "layers_num",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR);
    p_list->push_back(lnum);
    for(int i=0;i<layers.size();i++){
        PropertyInfo lsub(Variant::INT, "layers "+itos(i),PROPERTY_HINT_NONE,"",PROPERTY_USAGE_SUBGROUP);
        PropertyInfo lname(Variant::STRING, "L_NAME_"+itos(i),PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR);
        PropertyInfo licon(Variant::STRING, "L_ICON_"+itos(i),PROPERTY_HINT_GLOBAL_FILE,"",PROPERTY_USAGE_EDITOR);
        p_list->push_back(lsub);
        p_list->push_back(lname);
        p_list->push_back(licon);
        if(layer_props.has(brush_name)){
            Vector<LayerProps> p = layer_props.get(brush_name);
            for(int k=0;k<p.size();k++){
                PropertyInfo pinfo = p[k].pinfo;
                pinfo.name = "L_"+pinfo.name+"_"+itos(i);
                p_list->push_back(pinfo);
            }
        }
    }
}
bool MBrushLayers::_get(const StringName &p_name, Variant &r_ret) const{
    String _name = p_name;
    if(_name.begins_with("L_")){
        PackedStringArray parts = _name.split("_");
        int index = parts[2].to_int();
        Dictionary dic = layers[index];
        String key = parts[1].strip_edges();
        r_ret = dic[key];
        return true;
    }
    return false;
}
bool MBrushLayers::_set(const StringName &p_name, const Variant &p_value){
    String _name = p_name;
    if(_name.begins_with("L_")){
        PackedStringArray parts = _name.split("_");
        int index = parts[2].to_int();
        Dictionary dic = layers[index];
        String key = parts[1].strip_edges();
        dic[key] = p_value;
        layers[index] = dic;
        return true;
    }
    return false;
}

Array MBrushLayers::get_layers_info(){
    Array out;
    HashMap<String,Ref<ImageTexture>> current_textures;
    for(int i=0;i<layers.size();i++){
        Dictionary l = layers[i];
        Dictionary dic;
        dic["name"]=l["NAME"];
        dic["icon-color"]=get_layer_color(i);
        if(textures.has(l["ICON"])){
            current_textures.insert(l["ICON"],textures.get(l["ICON"]));
            dic["icon"]=textures.get(l["ICON"]);
        } else {
            Ref<Image> img = Image::load_from_file(l["ICON"]);
            Ref<ImageTexture> tex;
            if(img.is_valid()){
                img->resize(64,64);
                tex = ImageTexture::create_from_image(img);
            }
            textures.insert(l["ICON"],tex);
            current_textures.insert(l["ICON"],tex);
            dic["icon"]=tex;
        }
        out.push_back(dic);
    }
    for(HashMap<String,Ref<ImageTexture>>::Iterator it=textures.begin();it!=textures.end();++it){
        if(!current_textures.has(it->key)){
            textures.erase(it->key);
        }
    }
    return out;
}

Color MBrushLayers::get_layer_color(int index){
    Color col(0.5,0.5,0.5,1.0);
    Dictionary dic = layers[index];
    if(brush_name=="Color Paint"){
        return dic["color"];
    }
    if(brush_name=="Channel Painter"){
        bool red = dic["red"];
        bool green = dic["green"];
        bool blue = dic["blue"];
        bool alpha = dic["alpha"];
        if(alpha && !green && !blue && !red){
            float alpha_val = dic["blue-value"];
            col = Color(alpha,alpha,alpha);
            return col;
        }
        if(red){
            col.r = dic["red-value"];
        }
        if(green){
            col.g = dic["green"];
        }
        if(blue){
            col.g = dic["blue-value"];
        }
    }
    return col;
}

void MBrushLayers::set_layer(int index,MColorBrush* brush){
    Dictionary info = layers[index];
    Array keys = info.keys();
    for(int i=0;i<keys.size();i++){
        if(keys[i]!="NAME" && keys[i]!="ICON"){
            brush->_set_property(keys[i],info[keys[i]]);
        }
    }
}