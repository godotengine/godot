#pragma once
#include "../beehave_node.h"
class BeehaveDecoratorLimiterCount : public BeehaveDecorator
{
    GDCLASS(BeehaveDecoratorLimiterCount, BeehaveDecorator);
protected:
    static void _bind_methods();
public:
    virtual String get_tooltip()override
    {
        return L"限制器将执行其“正在运行”的子项“x”次。\n当达到最大刻度数时,它将返回“FAILURE”状态代码。\n下次子项不再处于“正在运行”状态时,计数将重置";
    }

    virtual String get_lable_name()
    {
        return "执行次数限制装饰器";
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

    int max_count = 0;
    int current_count = 0;
};