#include "auto_delete.h"
#include "body_main.h"

void AutoDelete::_bind_methods() {
    
    GDVIRTUAL_BIND(_is_running, "p_delta");
}

AutoDelete::AutoDelete() {
}
AutoDelete::~AutoDelete() {
}
void AutoDelete::start(StringName p_name,ObjectID p_owner,ObjectID p_item,double p_duration) {
    name = p_name;
    owner = p_owner;
    item = p_item;
    start_time = 0;
    duration = p_duration;
}
bool AutoDelete::is_running(float p_delta) {
    start_time += p_delta;    
    if (GDVIRTUAL_IS_OVERRIDDEN(_is_running)) {
        bool rs;
        GDVIRTUAL_CALL(_is_running, p_delta,rs);
        return rs;
    }
    if(start_time > duration) {
        return true;
    }
    return false;
}
void AutoDelete::stop() {
    Object *obj = ObjectDB::get_instance(item);
    Node *node = Object::cast_to<Node>( obj);
    if(node) {
        node->queue_free();
    }

    item = ObjectID();
    start_time = 0;
}