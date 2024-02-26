#ifndef MPAINT16
#define MPAINT16

#include "../mcolor_brush.h"


class MPaint16 : public MColorBrush{
    uint8_t paint_layer;
    String _get_name();
    void _set_property(String prop_name, Variant value);
    bool is_two_point_brush();
    void before_draw();
    void set_color(uint32_t local_x,uint32_t local_y,uint32_t x,uint32_t y,MImage* img);
};
#endif