#ifndef BT_UTILS_H
#define BT_UTILS_H

#include "virtual_machine.h"

class BtNode;

void create_bt_structure(BehaviorTree::BtStructure& structure, BehaviorTree::NodeList& node_list, BtNode& node, int begin);

#endif
