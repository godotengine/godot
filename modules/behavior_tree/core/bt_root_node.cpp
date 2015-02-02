#include "bt_root_node.h"
#include "bt_utils.h"

BtRootNode::BtRootNode()
    : vm(bt_running_data, bt_node_list, bt_structure_data)
	, request_reset(false)
{
    BehaviorTree::NodeData node_data;
    node_data.begin = 0;
    node_data.end = 1;
    bt_structure_data.push_back(node_data);
    bt_node_list.push_back(get_behavior_node());
}

void BtRootNode::_bind_methods() {
	ObjectTypeDB::bind_method(_MD("set_context", "context"), &BtRootNode::set_context);
	ObjectTypeDB::bind_method(_MD("get_context"), &BtRootNode::get_context);
	ObjectTypeDB::bind_method(_MD("tick"), &BtRootNode::tick);
	ObjectTypeDB::bind_method(_MD("step"), &BtRootNode::step);
	ObjectTypeDB::bind_method(_MD("reset"), &BtRootNode::reset);
	ObjectTypeDB::bind_method(_MD("clear"), &BtRootNode::clear);
}

static HashMap<uint32_t, BehaviorTree::VMRunningData> running_datas;

void BtRootNode::set_context(const Variant& context) {

	this->context = context;
}

void BtRootNode::tick() {

	uint32_t key = context.hash();
	bt_running_data = running_datas[key];
	vm.tick(&context);
	if (request_reset) {

		bt_running_data.last_tick_running.clear();
		bt_running_data.running_nodes.clear();
		bt_running_data.index_marker = 0;

		request_reset = false;
	}
	running_datas[key] = bt_running_data;
}

void BtRootNode::step() {

	vm.step(&context);
}

void BtRootNode::reset() {

	request_reset = true;
}

void BtRootNode::clear() {

	running_datas.clear();
	request_reset = false;
}

void BtRootNode::add_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) {
    Vector<BehaviorTree::IndexType> node_hierarchy_index;
    fetch_node_data_list_from_node_hierarchy(node_hierarchy, node_hierarchy_index);

    BehaviorTree::BTStructure temp_bt_structure_data;
    BehaviorTree::NodeList temp_bt_node_list;
    // add new child at the end of its parent.
    int child_index = bt_structure_data[node_hierarchy_index[0]].end;
    create_bt_structure(temp_bt_structure_data, temp_bt_node_list, child, child_index);
    BehaviorTree::IndexType children_count = temp_bt_node_list.size();

    int old_size = bt_structure_data.size();
    int new_size = old_size + children_count;
    
    bt_structure_data.resize(new_size);
    bt_node_list.resize(new_size);

    for (int i = old_size-1; i >= child_index; --i) {
        BehaviorTree::NodeData& node_data = bt_structure_data[i];
        node_data.begin += children_count;
        node_data.end += children_count;
        bt_structure_data[i+children_count] = node_data;
        bt_node_list[i+children_count] = bt_node_list[i];
    }

    for (int i = 0; i < children_count; ++i) {
        ERR_EXPLAIN("Index of child is not correct.");
        ERR_FAIL_COND(child_index+i != temp_bt_structure_data[i].index);
        bt_structure_data[child_index+i] = temp_bt_structure_data[i];
        bt_node_list[child_index+i] = temp_bt_node_list[i];
    }

    int parents_count = node_hierarchy_index.size();
    for (int i = 0; i < parents_count; ++i) {
        bt_structure_data[node_hierarchy_index[i]].end += children_count;
    }
}

void BtRootNode::remove_child_node(BtNode& , Vector<BehaviorTree::Node*>& node_hierarchy) {
    Vector<BehaviorTree::IndexType> node_hierarchy_index;
    fetch_node_data_list_from_node_hierarchy(node_hierarchy, node_hierarchy_index);

    BehaviorTree::NodeData child_node_data = bt_structure_data[node_hierarchy_index[0]];
    BehaviorTree::IndexType children_count = child_node_data.end - child_node_data.begin;

    int old_size = bt_structure_data.size();
    int new_size = old_size - children_count;
    
    for (int i = child_node_data.end; i < old_size; ++i) {
        BehaviorTree::NodeData& node_data = bt_structure_data[i];
        node_data.begin -= children_count;
        node_data.end -= children_count;
        bt_structure_data[i-children_count] = node_data;
        bt_node_list[i-children_count] = bt_node_list[i];
    }

    bt_structure_data.resize(new_size);
    bt_node_list.resize(new_size);

    int parents_count = node_hierarchy_index.size();
    // first one is child itself.
    for (int i = 1; i < parents_count; ++i) {
        bt_structure_data[node_hierarchy_index[i]].end -= children_count;
    }
}

void BtRootNode::move_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) {
    BtNode* parent_node = child.get_parent() ? child.get_parent()->cast_to<BtNode>() : NULL;
    if (!parent_node) {
        ERR_EXPLAIN("Cannot find a parent node for child.");
        ERR_FAIL();
        return;
    }

    BehaviorTree::IndexType parent_index = find_node_index_from_node_hierarchy(node_hierarchy);
    BehaviorTree::NodeData parent_node_data = bt_structure_data[parent_index];

    BehaviorTree::BTStructure temp_bt_structure_data;
    BehaviorTree::NodeList temp_bt_node_list;
    create_bt_structure(temp_bt_structure_data, temp_bt_node_list, *parent_node, parent_index);

    if (temp_bt_node_list.size() != parent_node_data.end - parent_node_data.begin ||
        temp_bt_node_list.size() != temp_bt_structure_data.size()) {
        ERR_EXPLAIN("Move child cannot change total number of node.");
        ERR_FAIL();
        return;
    }

    for (BehaviorTree::IndexType i = parent_node_data.begin; i < parent_node_data.end; ++i) {
        bt_structure_data[i] = temp_bt_structure_data[i - parent_node_data.begin];
        bt_node_list[i] = temp_bt_node_list[i - parent_node_data.begin];
    }
}

void BtRootNode::fetch_node_data_list_from_node_hierarchy(
        const Vector<BehaviorTree::Node*>& node_hierarchy,
        Vector<BehaviorTree::IndexType>& node_hierarchy_index) const {

    int node_hierarchy_size = node_hierarchy.size();
    BehaviorTree::IndexType node_data_index = 0;
    node_hierarchy_index.resize(node_hierarchy_size+1); // plus a root node
    node_hierarchy_index[node_hierarchy_size] = 0;

    for (int i = node_hierarchy_size-1; i >= 0; --i) {
        BehaviorTree::Node* node = node_hierarchy[i];
        node_data_index = find_child_index(node_data_index, node);
        node_hierarchy_index[i] = node_data_index;
        ERR_EXPLAIN("Cannot find child index.");
        ERR_FAIL_COND( node_data_index != node_hierarchy_index[i] );
    }
}

BehaviorTree::IndexType BtRootNode::find_child_index(BehaviorTree::IndexType parent_index, BehaviorTree::Node* child) const {
    BehaviorTree::IndexType parent_end = bt_structure_data[parent_index].end;
    BehaviorTree::NodeData node_data = bt_structure_data[parent_index+1];
    while (node_data.end <= parent_end) {
        if (bt_node_list[node_data.index] == child)
            return node_data.index;
        else
            node_data = bt_structure_data[node_data.end];
    }
    return parent_index;
}

BehaviorTree::IndexType BtRootNode::find_node_index_from_node_hierarchy(const Vector<BehaviorTree::Node*>& node_hierarchy) const {
    BehaviorTree::IndexType node_index = 0;
    for (int i = node_hierarchy.size()-1; i >= 0; --i) {
        BehaviorTree::Node* node = node_hierarchy[i];
        BehaviorTree::IndexType child_node_index = find_child_index(node_index, node);
        ERR_EXPLAIN("Cannot find child index.");
        ERR_FAIL_COND_V( node_index != child_node_index, child_node_index );
        node_index = child_node_index;
    }
    return node_index;
}
