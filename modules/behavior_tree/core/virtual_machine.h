#ifndef BEHAVIOR_TREE_VIRTUAL_MACHINE_H
#define BEHAVIOR_TREE_VIRTUAL_MACHINE_H

#include "node.h"

namespace BehaviorTree
{

struct VMRunningData
{
    struct RunningNode
    {
        Node* node;
        NodeData data;
    };

    BTVector<RunningNode> running_nodes;
    BTVector<IndexType> this_tick_running;
    BTVector<IndexType> last_tick_running;
    IndexType index_marker;

    void tick_begin();
    void tick_end();

    void sort_last_running_nodes();
    bool is_current_node_running_on_last_tick() const;
    void pop_last_running_behavior();
    void add_running_node(Node* node, NodeData node_data);

};

class VirtualMachine
{
private:
    VMRunningData& running_data;
    NodeList& node_list;
    const BTStructure& structure_data;

public:
    VirtualMachine(VMRunningData& running_data_, NodeList& node_list_, const BTStructure& structure_data_)
        : running_data(running_data_), node_list(node_list_), structure_data(structure_data_)
    {}

    // execute the whole behavior tree.
    void tick(void* context);
    // running behavior tree step by step.
    void step(void* context);

    inline void increase_index() { ++running_data.index_marker; }
    void move_index_to_node_end(IndexType index);
    IndexType move_index_to_running_child();

    inline NodeData get_node_data(IndexType index) const { return structure_data[index]; }
    inline NodeData get_current_running_node() const {
        BT_ASSERT(!running_data.running_nodes.empty());
        return running_data.running_nodes.back().data;
    }

    bool is_child(IndexType parent_index, IndexType child_index) const;

private:
    void cancel_skipped_behaviors(void* context);
    void cancel_behavior(void* context);
    void run_composites(E_State state, void* context);
    E_State run_action(Node& node, void* context);
};

} /* BehaviorTree */ 

#endif
