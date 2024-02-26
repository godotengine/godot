#ifndef MHEIGHTBRUSH
#define MHEIGHTBRUSH

#include "core/variant/variant.h"
#include "core/string/ustring.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"
#include "core/object/object.h"
#include "mconfig.h"

#include "mgrid.h"







/*
Property Format
array = [
    {"name":"name of props", type:Variant_type,hint:"Type hint",hint_string:"hint string" default:default_value, min:min_value, max:max_value}
    .
    .
    .
]
Do not confuse hint and hint string with the hint and hint string in godot
float can have hint="range" and hint_string=slider_step
*/

class MHeightBrush {
    protected:
    MGrid* grid;
    public:
    void set_grid(MGrid* _grid){grid = _grid;};
    virtual ~MHeightBrush(){};
    virtual String _get_name()=0;
    virtual Array _get_property_list()=0;
    virtual void _set_property(String prop_name, Variant value)=0;
    virtual bool is_two_point_brush()=0;
    // Will be called before start to draw
    // a initilized type before each draw
    virtual void before_draw()=0;
    // x,y -> position of current pixel to be modified
    // grid -> grid class in mterrain to access information about all pixel, normals, height or anything that you need for your brush
    virtual float get_height(uint32_t x,uint32_t y)=0;
};
#endif