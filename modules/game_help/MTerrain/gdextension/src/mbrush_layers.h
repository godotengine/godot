#ifndef MBRUSHLAYERS
#define MBRUSHLAYERS


#include "core/io/resource.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/image_texture.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"




class MColorBrush;

struct LayerProps
{
    PropertyInfo pinfo;
    Variant def_value;
};


class MBrushLayers : public Resource {
    GDCLASS(MBrushLayers,Resource);

    protected:
    static void _bind_methods();

    public:
    HashMap<String,Vector<LayerProps>> layer_props;
    String layers_title;
    String uniform_name;
    String brush_name="Color Paint";
    Array layers;
    HashMap<String,Ref<ImageTexture>> textures;

    MBrushLayers();
    ~MBrushLayers();

    void set_layers_title(String input);
    String get_layers_title();

    void set_uniform_name(String input);
    String get_uniform_name();

    void set_brush_name(String input);
    String get_brush_name();

    void set_layers_num(int input);
    int get_layers_num();

    void set_layers(Array input);
    Array get_layers();


    void _get_property_list(List<PropertyInfo> *p_list) const;
    bool _get(const StringName &p_name, Variant &r_ret) const;
    bool _set(const StringName &p_name, const Variant &p_value);

    Array get_layers_info();
    Color get_layer_color(int index);
    void set_layer(int index,MColorBrush* brush);
    

};
#endif