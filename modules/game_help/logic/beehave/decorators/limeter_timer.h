#pragma once
#include "../beehave_node.h"
class BeehaveDecoratorLimiterTimer : public BeehaveDecorator
{
    GDCLASS(BeehaveDecoratorLimiterTimer, BeehaveDecorator);

public:
    virtual String get_tooltip()override
    {
        return String(L"时间限制装饰器将为其处于“RUNNING”状态的子级提供一定时间来完成,\n如果子级还未返回“RUNNING”状态，此装饰器将在等待时间结束后返回“FAILURE”。");
    }

    virtual String get_lable_name()
    {
        return L"时间限制装饰器";
    } 
    virtual void interrupt(Node * actor, Blackboard* blackboard)override
    {
        base_class_type::interrupt(actor,blackboard);
    }

    virtual void before_run(Node * actor, Blackboard* blackboard)override
    {
        base_class_type::before_run(actor,blackboard);
        current_time = 0;
    }
    virtual int tick(Node * actor, Blackboard* blackboard)override
    {
        if(get_child_count() == 0)
        {
            return FAILURE;
        }
        if(current_time < max_time)
        {
            current_time += (float)blackboard->get_var("delta_time");
            if(child_state[0] == 0)
            {
                children[0]->before_run(actor,blackboard);
                child_state[0] = 1;
            }
            int rs = children[0]->tick(actor,blackboard);
            if(rs == SUCCESS || rs == FAILURE)
            {
                for(int i = 0; i < get_child_count(); ++i)
                {
                    if(child_state[i] == 1)
                    {
                        children[i]->after_run(actor,blackboard);
                        child_state[i] = 2;
                    }
                }
                return rs;
            }
        }
        else
        {
            children[0]->after_run(actor,blackboard);
            child_state[0] = 2;
            return FAILURE;
        }
        return FAILURE;
    }
    float max_time = 0.0f;
    float current_time = 0.0f;
};