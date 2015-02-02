#ifndef BT_ROOT_NODE
#define BT_ROOT_NODE

#include "bt_decorator_node.h"

class BtRootNode : public BtDecoratorNode
{
    OBJ_TYPE(BtRootNode, BtDecoratorNode);

    BehaviorTree::BTStructure bt_structure_data;
    BehaviorTree::NodeList bt_node_list;
    BehaviorTree::VMRunningData bt_running_data;
    BehaviorTree::VirtualMachine vm;
    Variant context;
	bool request_reset;

public:
    BtRootNode();

    virtual void add_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) override;
    virtual void remove_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) override;
    virtual void move_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) override;

    void set_context(const Variant& context);
    Variant& get_context() { return context; }
    void tick();
    void step();
	void reset();
	void clear();

protected:
    static void _bind_methods();

private:
    void fetch_node_data_list_from_node_hierarchy(
            const Vector<BehaviorTree::Node*>& node_hierarchy,
            Vector<BehaviorTree::IndexType>& node_hierarchy_index) const;

    BehaviorTree::IndexType find_child_index(BehaviorTree::IndexType parent_index, BehaviorTree::Node* child) const;
    BehaviorTree::IndexType find_node_index_from_node_hierarchy(const Vector<BehaviorTree::Node*>& node_hierarchy) const;
};

#endif
