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
    virtual int tick(const Ref<BeehaveRuncontext>& run_context) override
    {
        if(get_child_count() == 0)
        {
            return SUCCESS;
        }
		Dictionary prop = run_context->get_property(this);
		float total_time = prop.get(SNAME("total_time"),0.0f);
        if (total_time < wait_time)
        {

			total_time += run_context->delta;
			prop[SNAME("total_time")] = total_time;
            return RUNNING;

        }
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
        if(rs == SUCCESS || rs == FAILURE)
        {
            children[0]->after_run(run_context);
        }
        
		run_context->set_run_state(children[0].ptr(),rs);
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
};
