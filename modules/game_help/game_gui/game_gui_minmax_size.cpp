#include "game_gui_minmax_size.h"

void GUIMinMaxSize::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_min_size", "value"), &GUIMinMaxSize::set_min_size);
    ClassDB::bind_method(D_METHOD("get_min_size"), &GUIMinMaxSize::get_min_size);
    ClassDB::bind_method(D_METHOD("set_max_size", "value"), &GUIMinMaxSize::set_max_size);
    ClassDB::bind_method(D_METHOD("get_max_size"), &GUIMinMaxSize::get_max_size);

    
    ADD_GROUP("Min Max", "minmax_");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "minmax_min_size"), "set_min_size", "get_min_size");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "minmax_max_size"), "set_max_size", "get_max_size");

}