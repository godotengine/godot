


#include "beehave_node.h"
#include "core/object/object.h"

#include "beehave_tree.h"

int BeehaveNode::process(const Ref<BeehaveRuncontext>& run_context)
{
    if(run_context->tree->debug_break_node == this)
    {
        return run_context->get_run_state(this);
    }
    if(get_debug_enabled())
    {
        run_context->tree->debug_break_node = this;
        return run_context->get_run_state(this);
    }
    return tick(run_context);
}