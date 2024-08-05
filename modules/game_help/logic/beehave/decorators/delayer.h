#pragma once
#include "../beehave_node.h"
class BeehaveDecoratorDelayer : public BeehaveDecorator
{
    GDCLASS(BeehaveDecoratorDelayer, BeehaveDecorator);
    public:
    virtual String get_tooltip()override
    {
        return String(L"延迟装饰器将在执行其子级之前返回“RUNNING”一段时间。\n当计时器及其子级均未处于“RUNNING”状态时,计时器将重置等待时间（以秒为单位）");
    }
    virtual String get_lable_name()
    {
        return String(L"延迟装饰器");
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
        return children[0]->tick(actor,blackboard);
    }
    float wait_time = 0.0;
    float total_time = 0.0;
};