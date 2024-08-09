#pragma once

#include "beehave_node.h"


class BeehaveTree : public Resource
{
    GDCLASS(BeehaveTree, Resource);
    enum Status { SUCCESS, FAILURE, RUNNING };
public:
    // 初始化
    void init(Node * actor, Blackboard* blackboard)
    {
        last_tick = 0;
        _can_send_message = false;
        _process_time_metric_value = 0;
        status = RUNNING;
        if(root_node.is_valid())
        {
            root_node->interrupt(actor,blackboard);
        }
    }

    void process(Node * actor, Blackboard* blackboard, double delta)
    {
        if (last_tick < tick_rate - 1)
        {
            last_tick += 1;
            return;

        }
        last_tick = 0;
        float start_time = OS::get_singleton()->get_ticks_usec();

	    blackboard->set_var("can_send_message", _can_send_message);

        // if _can_send_message:
        // 	BeehaveDebuggerMessages.process_begin(get_instance_id())
        tick(actor, blackboard);

        // if _can_send_message:
        // 	BeehaveDebuggerMessages.process_end(get_instance_id())

	    _process_time_metric_value = OS::get_singleton()->get_ticks_usec() - start_time;
    }
    int tick(Node * actor, Blackboard* blackboard)
    {	
        if (actor == nullptr || root_node.is_null())
		{
            return FAILURE;
        }
        if (status != RUNNING)
        {
            root_node->before_run(actor, blackboard);
        }
        status = root_node->tick(actor, blackboard);

        // 	if _can_send_message:
		// BeehaveDebuggerMessages.process_tick(child.get_instance_id(), status)
		// BeehaveDebuggerMessages.process_tick(get_instance_id(), status)
        if (status != RUNNING)
        {
            root_node->after_run(actor, blackboard);
        }
        return status;
    }

    void set_root_node(const Ref<BeehaveNode> &p_root_node)
    {
        root_node = p_root_node;
    }

    Ref<BeehaveNode> get_root_node()
    {
        return root_node;
    }

public:

    Ref<BeehaveNode> root_node;
    float tick_rate = 0.1f;
    float last_tick = 0.0f;
    float _process_time_metric_value = 0.0f;
    int status = -1;
    bool _can_send_message = false;
};

