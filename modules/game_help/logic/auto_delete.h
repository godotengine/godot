#pragma once
#include "core/object/ref_counted.h"
#include "scene/main/node.h"

class BufferItemBase : public RefCounted
{
    GDCLASS(BufferItemBase, RefCounted);
    static void _bind_methods();
public:
    void process(class CharacterBodyMain* p_owner);
    
};

class AutoDelete : public RefCounted
{
    GDCLASS(AutoDelete, RefCounted);
    static void _bind_methods();
public:
	AutoDelete();
    ~AutoDelete();
    void start(StringName p_name,ObjectID p_owner,ObjectID p_item,double p_duration);
    void re_enable() {
        start_time = 0;
    }
    virtual bool is_running(float p_delta);
    void stop();

    StringName get_name() {
        return name;
    }

    
	GDVIRTUAL1R(bool,_is_running,float);
    StringName name;
    ObjectID owner;
    ObjectID item;
    double start_time;
    double curr_time;
    double duration;
};
