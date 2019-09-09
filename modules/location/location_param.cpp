/* location_param.cpp */

#include "location_param.h"

void LocationParam::_bind_methods() {

    ClassDB::bind_method(D_METHOD("get_interval"), &LocationParam::get_interval);
    ClassDB::bind_method(D_METHOD("get_max_wait_time"), &LocationParam::get_max_wait_time);
    
    ClassDB::bind_method(D_METHOD("set_interval"), &LocationParam::set_interval);
    ClassDB::bind_method(D_METHOD("set_max_wait_time"), &LocationParam::set_max_wait_time);
	
    ADD_PROPERTY(PropertyInfo(Variant::INT, "interval"), "set_interval", "get_interval");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_wait_time"), "set_max_wait_time", "get_max_wait_time");

}

void LocationParam::set_interval(int p_interval) {
    interval = p_interval;
}

void LocationParam::set_max_wait_time(int p_max_wait_time) {
    max_wait_time = p_max_wait_time;
}

int LocationParam::get_interval() {
    return interval;
}

int LocationParam::get_max_wait_time() {
    return max_wait_time;
}


LocationParam::LocationParam() {

}