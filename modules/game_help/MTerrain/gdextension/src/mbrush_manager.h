#ifndef MBRUSHMANAGER
#define MBRUSHMANAGER

#include "core/object/object.h"
#include "core/templates/vector.h"
#include "core/templates/hash_map.h"

#include "mheight_brush.h"
#include "mcolor_brush.h"




class MBrushManager : public Object {
    GDCLASS(MBrushManager,Object);
    private:
    Vector<MHeightBrush*> height_brushes;
    HashMap<String,int> height_brush_map;

    Vector<MColorBrush*> color_brushes;
    HashMap<String,int> color_brush_map;

    void add_height_brush(MHeightBrush* brush);
    void add_color_brush(MColorBrush* brush);

    protected:
    static void _bind_methods();

    public:
    MBrushManager* get_singelton();
    MBrushManager();
    ~MBrushManager();
    MHeightBrush* get_height_brush(int brush_id);
    PackedStringArray get_height_brush_list();
    int get_height_brush_id(String brush_name);
    Array get_height_brush_property(int brush_id);
    void set_height_brush_propert(String prop_name,Variant value,int brush_id);

    MColorBrush* get_color_brush(int brush_id);
    MColorBrush* get_color_brush_by_name(String brush_name);
    PackedStringArray get_color_brush_list();
    void set_color_brush_propert(String prop_name,Variant value,int brush_id);
};
#endif