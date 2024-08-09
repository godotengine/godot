#pragma once

#include "../beehave_node.h"
// 并行组合节点
class BeehaveCompositeParallel : public BeehaveComposite
{
    GDCLASS(BeehaveCompositeParallel, BeehaveComposite);
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
    virtual void interrupt(Node * actor, Blackboard* blackboard)override
    {
        base_class_type::interrupt(actor,blackboard);
    }

    virtual TypedArray<StringName> get_class_name()override
    {
        TypedArray<StringName> rs = base_class_type::get_class_name();
        rs.push_back(StringName("BeehaveCompositeParallel"));
        return rs;  
    }
    virtual int tick(Node * actor, Blackboard* blackboard)override
    {
        if(get_child_count() == 0)
        {
            return SUCCESS;
        }
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
        for(int i = 1; i < get_child_count(); ++i)
        {
            if(child_state[i] == 0)
            {
                children[i]->before_run(actor,blackboard);
                child_state[i] = 1;
            }
        }
        int child_rs;
        for(int i = 1; i < get_child_count(); ++i)
        {
            if(child_state[i] == 1)  
            {
                child_rs = children[i]->tick(actor,blackboard);
                if(child_rs == SUCCESS || child_rs == FAILURE)
                {
                    children[i]->after_run(actor,blackboard);
                    child_state[i] = 2;
                }
            }    
        }

        return rs;
    }

};