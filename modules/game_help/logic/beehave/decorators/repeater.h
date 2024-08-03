#pragma once
#include "../beehave_node.h"

class BeehaveDecoratorRepeater : public BeehaveDecorator
{
    GDCLASS(BeehaveDecoratorRepeater, BeehaveDecorator);
    public:protected:
    static void _bind_methods();
public:
    virtual String get_tooltip()override
    {
        return L"转发器将执行其子进程，直到它返回一定次数的 `SUCCESS`当达到最大刻度数时，它将返回 `SUCCESS` 状态代码。如果子进程返回 `FAILURE`，转发器将立即返回 `FAILURE`。";
    }

    virtual String get_lable_name()
    {
        return "重复执行装饰器";
    }
    virtual void interrupt(Node * actor, Blackboard* blackboard)override
    {
        __supper::interrupt(actor,blackboard);
    }

    virtual void before_run(Node * actor, Blackboard* blackboard)override
    {
        __supper::before_run(actor,blackboard);
        current_count = 0;
    }
    virtual int tick(Node * actor, Blackboard* blackboard)override
    {
        if(get_child_count() == 0)
        {
            return FAILURE;
        }
        if(current_count < max_count)
        {
            current_count++;
            if(child_state[0] == 0)
            {
                children[0]->before_run(actor,blackboard);
                child_state[0] = 1;
            }
            int rs = children[0]->tick(actor,blackboard);
            if(rs == RUNNING)
            {
                return rs;
            }
            current_count += 1;
            child_state[0] = 0;
            children[0]->after_run(actor,blackboard);
            if(rs == FAILURE)
            {
                return FAILURE;
            }
        }
        else
        {
            children[0]->after_run(actor,blackboard);
            child_state[0] = 2;
            return FAILURE;
        }
        return SUCCESS;
    }

    int max_count = 0;
    int current_count = 0;
};