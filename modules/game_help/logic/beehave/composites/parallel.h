#pragma once

#include "../beehave_node.h"
// 并行组合节点
class BeehaveCompositeParallel : public BeehaveComposite
{
    GDCLASS(BeehaveCompositeParallel, BeehaveComposite);
    static void _bind_methods()
    {
        
    }
public:
    virtual String get_tooltip()override
    {
        return String(L"并行处理所有的节点。第一个节点为主节点。其他节点状态将被忽略。\n如果主节点返回“成功”或“失败”,则此节点将中断.");
    }
    virtual String get_lable_name()
    {
        return String(L"并行组合节点");
    }
    virtual StringName get_icon()
    {
        return SNAME("BTParallel");
    }
    virtual void interrupt(const Ref<BeehaveRuncontext>& run_context)override
    {
        base_class_type::interrupt(run_context);
    }

    virtual TypedArray<StringName> get_class_name()override
    {
        TypedArray<StringName> rs = base_class_type::get_class_name();
        rs.push_back(StringName("BeehaveCompositeParallel"));
        return rs;  
    }
    virtual int tick(const Ref<BeehaveRuncontext>& run_context)override
    {
        if(get_child_count() == 0)
        {
            return SUCCESS;
        }

		Dictionary prop = run_context->get_property(this);
        int child_rs = 0;
		int rs = 0;
        for(int i = 0; i < get_child_count(); ++i)
        {
			if (run_context->get_init_status(children[i].ptr()) == 0)
			{
				children[i]->before_run(run_context);
			}
			if (run_context->get_init_status(children[i].ptr()) == 1)
            {
                child_rs = children[i]->process(run_context);
				if (child_rs == NONE_PROCESS)
				{
					// 执行到断点,直接返回
					return child_rs;
				}
                if(child_rs == SUCCESS || child_rs == FAILURE)
                {
                    children[i]->after_run(run_context);
                }
				run_context->set_run_state(children[i].ptr(), child_rs);
            }
			if (i == 0)
			{
				rs = child_rs;
			}
        }
		if (rs == SUCCESS || rs == FAILURE)
		{
			// 如果主节点结束了,所有子节点也要结束
			for (int i = 1; i < get_child_count(); ++i)
			{
				if (run_context->get_init_status(children[i].ptr()) == 1)
				{
					children[i]->after_run(run_context);
					run_context->set_run_state(children[i].ptr(), SUCCESS);
				}

			}

		}

        return rs;
    }

};
