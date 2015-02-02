#ifndef BTNODE_H
#define BTNODE_H

#include "scene/main/node.h"
#include "virtual_machine.h"

class BtNode : public Node
{
	OBJ_TYPE(BtNode, Node);

public:
	virtual BehaviorTree::Node* get_behavior_node() = 0;

	virtual void add_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) = 0;
	virtual void remove_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) = 0;
	virtual void move_child_node(BtNode &child, Vector<BehaviorTree::Node*>& node_hierarchy) = 0;

protected:
	static void _bind_methods();

private:
	virtual void add_child_notify(Node *p_child) override;
	virtual void remove_child_notify(Node *p_child) override;
	virtual void move_child_notify(Node *p_child, int pos) override;

};

#endif
