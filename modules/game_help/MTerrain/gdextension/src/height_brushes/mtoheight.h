#ifndef MTOHEIGHT
#define MTOHEIGHT

#include "../mheight_brush.h"



class MToHeight : public MHeightBrush {
    public:
    float weight=0.4;
    float hardness=0.5;
    float offset=0.0;
    int mode=1;
    float start_height;
    MToHeight();
    ~MToHeight();
    String _get_name();
    Array _get_property_list();
    void _set_property(String prop_name, Variant value);
    bool is_two_point_brush();
    void before_draw();
    float get_height(uint32_t x,uint32_t y);
};
#endif