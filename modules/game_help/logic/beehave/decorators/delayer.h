#pragma once
#include "../beehave_node.h"
class BeehaveDecoratorDelayer : public BeehaveDecorator
{
    GDCLASS(BeehaveDecoratorDelayer, BeehaveDecorator);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_wait_time","time"),&BeehaveDecoratorDelayer::set_wait_time);
        ClassDB::bind_method(D_METHOD("get_wait_time"),&BeehaveDecoratorDelayer::get_wait_time);

        ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"wait_time"),"set_wait_time","get_wait_time");
    }
    public:
    virtual String get_tooltip()override
    {
        return String(L"延迟装饰器将在执行其子级之前返回“RUNNING”一段时间。\n当计时器及其子级均未处于“RUNNING”状态时,计时器将重置等待时间（以秒为单位）");
    }
    virtual String get_lable_name()
    {
        return String(L"延迟装饰器");
    }
    virtual StringName get_icon()
    {
        return SNAME("delayer");
    }
    virtual int tick(Node * actor, Blackboard* blackboard) override
    {
        if(get_child_count() == 0)
        {
            return SUCCESS;
        }
        if (total_time < wait_time)
        {

            total_time += (float)blackboard->get_var(SNAME("delta_time"), 0.0);
            return RUNNING;

        }
        if(child_state[0] == 0)
        {
            children[0]->before_run(actor,blackboard);
            child_state[0] = 1;
        }
        int rs = children[0]->tick(actor,blackboard);
        
        children[0]->set_status(rs);
        return rs;
    }
public:
    void set_wait_time(float time)
    {
        wait_time = time;
    }
    float get_wait_time()
    {
        return wait_time;
    }
protected:
    float wait_time = 0.0;
    float total_time = 0.0;
};