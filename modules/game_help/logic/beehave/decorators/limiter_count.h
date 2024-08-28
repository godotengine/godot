#pragma once
#include "../beehave_node.h"
class BeehaveDecoratorLimiterCount : public BeehaveDecorator
{
    GDCLASS(BeehaveDecoratorLimiterCount, BeehaveDecorator);
protected:
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_max_count","max_count"),&BeehaveDecoratorLimiterCount::set_max_count);
        ClassDB::bind_method(D_METHOD("get_max_count"),&BeehaveDecoratorLimiterCount::get_max_count);

        ADD_PROPERTY(PropertyInfo(Variant::INT,"max_time"),"set_max_time","get_max_time");        
        
    }
public:
    virtual String get_tooltip()override
    {
        return L"限制器将执行其“正在运行”的子项“x”次。\n当达到最大刻度数时,它将返回“FAILURE”状态代码。\n下次子项不再处于“正在运行”状态时,计数将重置";
    }

    virtual String get_lable_name()
    {
        return L"次数限制装饰器";
    }
    virtual void interrupt(const Ref<BeehaveRuncontext>& run_context)override
    {
        base_class_type::interrupt(run_context);
    }

    virtual void before_run(const Ref<BeehaveRuncontext>& run_context)override
    {
        base_class_type::before_run(run_context);
		Dictionary prop = run_context->get_property(this);
		prop[SNAME("current_count")] = 0;
    }
    virtual int tick(const Ref<BeehaveRuncontext>& run_context)override
    {
        if(get_child_count() == 0)
        {
            return FAILURE;
        }
		Dictionary prop = run_context->get_property(this);
		int current_count = prop.get(SNAME("current_count"), 0);
        if(current_count < max_count)
        {
            current_count++;
			if (run_context->get_init_status(children[0].ptr()) == 0)
            {
                children[0]->before_run(run_context);
            }
            int rs = children[0]->process(run_context);
            if(rs == NONE_PROCESS)
            {
				// 执行到断点,直接返回
                return rs;
            }
			run_context->set_run_state(this, rs);
            if(rs == SUCCESS || rs == FAILURE)
            {
                for(int i = 0; i < get_child_count(); ++i)
                {
					if (run_context->get_init_status(children[i].ptr()) == 1)
                    {
                        children[i]->after_run(run_context);
                    }
                }
                return rs;
            }
        }
        else
        {
            children[0]->after_run(run_context);
            return FAILURE;
        }
        return FAILURE;
    }

public:
    void set_max_count(int count)
    {
        max_count = count;
    }

    int get_max_count()
    {
        return max_count;
    }
protected:

    int max_count = 0;
};
