#ifndef RAISEBRUSH
#define RAISEBRUSH

#include "../mheight_brush.h"

class MRaise : public MHeightBrush {
    public:
    float hardness=0.5;
    float amount=0.2;
    float revers=false;
    float final_amount;
    MRaise();
    ~MRaise();
    String _get_name();
    Array _get_property_list();
    void _set_property(String prop_name, Variant value);
    bool is_two_point_brush();
    void before_draw();
    float get_height(uint32_t x,uint32_t y);
};
#endif