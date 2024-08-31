#pragma once

#include "../beehave_node.h"
#include "scene/3d/node_3d.h"

class BeehaveLeafMovePosition : public BeehaveLeaf
{
    GDCLASS(BeehaveLeafMovePosition, BeehaveLeaf);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_move_direction", "move_direction"), &BeehaveLeafMovePosition::set_move_direction);
        ClassDB::bind_method(D_METHOD("get_move_direction"), &BeehaveLeafMovePosition::get_move_direction);

        ClassDB::bind_method(D_METHOD("set_by_blackboard_property", "by_blackboard_property"), &BeehaveLeafMovePosition::set_by_blackboard_property);
        ClassDB::bind_method(D_METHOD("get_by_blackboard_property"), &BeehaveLeafMovePosition::get_by_blackboard_property);

        ClassDB::bind_method(D_METHOD("get_move_direction_blackboard_property"), &BeehaveLeafMovePosition::get_move_direction_blackcoard_property);
        ClassDB::bind_method(D_METHOD("set_move_direction_blackboard_property", "move_direction_blackboard_property"), &BeehaveLeafMovePosition::set_move_direction_blackboard_property);


        ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "move_direction"), "set_move_direction", "get_move_direction");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "by_blackboard_property"), "set_by_blackboard_property", "get_by_blackboard_property");
        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "move_direction_blackboard_property"), "set_move_direction_blackboard_property", "get_move_direction_blackboard_property");
        
    }

public:
    virtual String get_tooltip()override
    {
        return String(L"移动节点的位置,这个节点将不会被中断。\n如果需要结束,需要外部的装饰器强制结束");
    }
    virtual String get_lable_name()
    {
        return String(L"移动节点位置");
    }
    virtual StringName get_icon()
    {
        return SNAME("action");
    }
    virtual int tick(const Ref<BeehaveRuncontext>& run_context) 
    {
        Node3D *node = Object::cast_to<Node3D>(run_context->actor);
        if(node == nullptr) return FAILURE;
		if (by_blackboard_property)
		{
			Vector3 dir = run_context->blackboard->getvar(move_direction_blackboard_property);
			node->set_position(node->get_position() + dir * run_context->delta);

		}
		else
		{
			node->set_position(node->get_position() + move_direction * run_context->delta);
		}
        return RUNNING;

    }
public:
    void set_move_direction(const Vector3& _move_direction)
    {
        move_direction = _move_direction;
    }
    Vector3 get_move_direction()
    {
        return move_direction;
    }

    void set_by_blackboard_property(bool _by_blackboard_property)
    {
        by_blackboard_property = _by_blackboard_property;
    }
    bool get_by_blackboard_property()
    {
        return by_blackboard_property;
    }
    void set_move_direction_blackboard_property(StringName _move_direction_blackboard_property)
    {
		move_direction_blackboard_property = _move_direction_blackboard_property;
    }
    StringName get_move_direction_blackcoard_property()
    {
        return move_direction_blackboard_property;
    }
	
protected:
    Vector3 move_direction;
	bool by_blackboard_property;
	StringName move_direction_blackboard_property;
};
