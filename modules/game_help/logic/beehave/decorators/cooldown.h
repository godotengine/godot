#pragma once
#include "../beehave_node.h"
class BeehaveDecoratorCooldown : public BeehaveDecorator
{
    GDCLASS(BeehaveDecoratorCooldown, BeehaveDecorator);
public:
    virtual String get_tooltip()override
    {
        return String(L"冷却装饰器将在执行其子进程后在规定的时间内返回“FAILURE”。\n下次执行其子进程且其未处于“RUNNING”状态时,计时器将重置");
    }
    virtual String get_lable_name()
    {
        return String(L"冷却装饰器");
    }
    virtual TypedArray<StringName> get_class_name()override
    {
        TypedArray<StringName> rs = base_class_type::get_class_name();
        rs.push_back("BeehaveSequence");
        return rs;
    }
    virtual StringName get_icon()override
    {
        return SNAME("cooldown");
    }
    virtual void interrupt(Node * actor, Blackboard* blackboard)override
    {
        base_class_type::interrupt(actor,blackboard);
        is_init = true;
        remaining_time = wait_time;
    }
    int tick(Node * actor, Blackboard* blackboard)override
    {
        Ref<BeehaveNode> child = get_child(0);
        if(child.is_null())
        {
            return FAILURE;
        }
        int response ;
        if(!is_init)
        {
            interrupt(actor,blackboard);
        }
        if(remaining_time > 0)
        {
            response = FAILURE;
            remaining_time -= (double)blackboard->get_var(SNAME("delta_time"), 0.0);
        }
        else
        {
            response = child->tick(actor,blackboard);
        }
        return response;
    }
    float wait_time = 0;
    float remaining_time = 0;
    bool is_init = false;
};