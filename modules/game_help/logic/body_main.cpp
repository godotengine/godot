#include "body_main.h"
#include "data_table_manager.h"
#include "scene/3d/path_3d.h"
#include "character_ai/character_ai.h"


void CharacterBodyMain::_bind_methods()
{
    
	ClassDB::bind_method(D_METHOD("restart"), &CharacterBodyMain::restart);
	ClassDB::bind_method(D_METHOD("init_main_body","p_skeleton_file_path","p_animation_group"), &CharacterBodyMain::init_main_body);


	ClassDB::bind_method(D_METHOD("set_behavior_tree", "behavior_tree"), &CharacterBodyMain::set_behavior_tree);
	ClassDB::bind_method(D_METHOD("get_behavior_tree"), &CharacterBodyMain::get_behavior_tree);
	ClassDB::bind_method(D_METHOD("set_update_mode", "update_mode"), &CharacterBodyMain::set_update_mode);
	ClassDB::bind_method(D_METHOD("get_update_mode"), &CharacterBodyMain::get_update_mode);
	ClassDB::bind_method(D_METHOD("set_blackboard", "blackboard"), &CharacterBodyMain::set_blackboard);
	ClassDB::bind_method(D_METHOD("get_blackboard"), &CharacterBodyMain::_get_blackboard);

	ClassDB::bind_method(D_METHOD("set_blackboard_plan", "plan"), &CharacterBodyMain::set_blackboard_plan);
	ClassDB::bind_method(D_METHOD("get_blackboard_plan"), &CharacterBodyMain::get_blackboard_plan);

    ClassDB::bind_method(D_METHOD("set_controller", "controller"), &CharacterBodyMain::set_controller);
    ClassDB::bind_method(D_METHOD("get_controller"), &CharacterBodyMain::get_controller);

    ClassDB::bind_method(D_METHOD("set_navigation_agent", "navigation_agent"), &CharacterBodyMain::set_navigation_agent);
    ClassDB::bind_method(D_METHOD("get_navigation_agent"), &CharacterBodyMain::get_navigation_agent);

    ClassDB::bind_method(D_METHOD("set_skeleton", "skeleton"), &CharacterBodyMain::set_skeleton);
    ClassDB::bind_method(D_METHOD("get_skeleton"), &CharacterBodyMain::get_skeleton);

    ClassDB::bind_method(D_METHOD("set_animation_library", "animation_library"), &CharacterBodyMain::set_animation_library);
    ClassDB::bind_method(D_METHOD("get_animation_library"), &CharacterBodyMain::get_animation_library);

    ClassDB::bind_method(D_METHOD("set_animator", "animator"), &CharacterBodyMain::set_animator);
    ClassDB::bind_method(D_METHOD("get_animator"), &CharacterBodyMain::get_animator);

    ClassDB::bind_method(D_METHOD("set_main_shape", "shape"), &CharacterBodyMain::set_main_shape);
    ClassDB::bind_method(D_METHOD("get_main_shape"), &CharacterBodyMain::get_main_shape);

    ClassDB::bind_method(D_METHOD("set_check_area", "check_area"), &CharacterBodyMain::set_check_area);
    ClassDB::bind_method(D_METHOD("get_check_area"), &CharacterBodyMain::get_check_area);
    ClassDB::bind_method(D_METHOD("get_check_area_by_name", "name"), &CharacterBodyMain::get_check_area_by_name);

    ClassDB::bind_method(D_METHOD("init_body_part_array", "part_array"), &CharacterBodyMain::init_body_part_array);
    ClassDB::bind_method(D_METHOD("set_body_part", "part"), &CharacterBodyMain::set_body_part);
    ClassDB::bind_method(D_METHOD("get_body_part"), &CharacterBodyMain::get_body_part);

    ClassDB::bind_method(D_METHOD("set_character_ai", "ai"), &CharacterBodyMain::set_character_ai);
    ClassDB::bind_method(D_METHOD("get_character_ai"), &CharacterBodyMain::get_character_ai);


    


    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "behavior_tree", PROPERTY_HINT_RESOURCE_TYPE, "BehaviorTree"), "set_behavior_tree", "get_behavior_tree");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "character_ai", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAI"), "set_character_ai", "get_character_ai");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "update_mode", PROPERTY_HINT_ENUM, "Idle,Physics,Manual"), "set_update_mode", "get_update_mode");	
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard", PROPERTY_HINT_RESOURCE_TYPE, "Blackboard",PROPERTY_USAGE_DEFAULT ), "set_blackboard", "get_blackboard");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan", PROPERTY_USAGE_DEFAULT ), "set_blackboard_plan", "get_blackboard_plan");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "controller", PROPERTY_HINT_RESOURCE_TYPE, "CharacterController"), "set_controller", "get_controller");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navigation_agent", PROPERTY_HINT_RESOURCE_TYPE, "CharacterNavigationAgent3D"), "set_navigation_agent", "get_navigation_agent");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "skeleton", PROPERTY_HINT_NODE_TYPE, "Skeleton",PROPERTY_USAGE_EDITOR), "set_skeleton", "get_skeleton");    
    // 只能編輯不能保存
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animation_library", PROPERTY_HINT_RESOURCE_TYPE, "AnimationLibrary",PROPERTY_USAGE_DEFAULT ), "set_animation_library", "get_animation_library");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animator", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimator",PROPERTY_USAGE_DEFAULT ), "set_animator", "get_animator");

    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "main_shape", PROPERTY_HINT_RESOURCE_TYPE, "CollisionObject3DConnection",PROPERTY_USAGE_DEFAULT), "set_main_shape", "get_main_shape");
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "check_area", PROPERTY_HINT_RESOURCE_TYPE, RESOURCE_TYPE_HINT("CharacterCheckArea3D"),PROPERTY_USAGE_DEFAULT), "set_check_area", "get_check_area");
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "body_part", PROPERTY_HINT_NONE,"",PROPERTY_USAGE_DEFAULT), "set_body_part", "get_body_part");


	ADD_SIGNAL(MethodInfo("behavior_tree_finished", PropertyInfo(Variant::INT, "status")));
	ADD_SIGNAL(MethodInfo("behavior_tree_updated", PropertyInfo(Variant::INT, "status")));
    
	ADD_SIGNAL(MethodInfo("skill_tree_finished", PropertyInfo(Variant::INT, "status")));
	ADD_SIGNAL(MethodInfo("skill_tree_updated", PropertyInfo(Variant::INT, "status")));

    

}

void CharacterBodyMain::clear_all()
{
    if(skeleton)
    {
        memdelete(skeleton);
        skeleton = nullptr;
        
    }
    if(player)
    {
        memdelete(player);
        player = nullptr;
        
    }
}
void CharacterBodyMain::_notification( int p_notification )
{
    if(character_agent.is_valid())
    {
        character_agent->set_body_main(this);
        character_agent->_notification(p_notification);

    }
	switch (p_notification) {
		case NOTIFICATION_PROCESS: {
            // 更新玩家位置
            GDVIRTUAL_CALL(_update_player_position);
            if(character_ai.is_valid())
            {
                character_ai->execute(this,get_blackboard().ptr(),&ai_context);
            }
            // 更新動畫
            if(animator.is_valid())
            {
                animator->update_animation(get_process_delta_time());
            }
            for(uint32_t i = 0; i < check_area.size();++i)
            {
                if(check_area[i].is_valid())
                {
                    check_area[i]->update_world_move(get_global_position());
                }
            }
            _process_move();
        } break;
    }

}
void CharacterBodyMain::_process_move()
{
    // 处理角色移动
    bool is_walk = get_blackboard()->get_var("move/using_navigation_target",false);
    if(is_walk )
    {
        // 处理导航行走
        if(character_agent.is_valid())
        {
            if(is_walk)
            {
                Vector3 target_pos = get_blackboard()->get_var("move/navigation_target_pos",Vector3());
                // 设置角色移动速度
                character_agent->set_velocity(get_velocity());
                character_agent->set_target_position(target_pos);
            }
            else
            {
                character_agent->set_navigation_finished(true);
            }
        }
    }
    else
    {
        move_and_slide();
    }

}

// 初始化身體
void CharacterBodyMain::init_main_body(String p_skeleton_file_path,StringName p_animation_group)
{
    skeleton_res = p_skeleton_file_path;
    animation_group = p_animation_group;
    load_skeleton();

}

void CharacterBodyMain::set_character_ai(const Ref<CharacterAI> &p_ai)
{
    character_ai = p_ai;
}
Ref<CharacterAI> CharacterBodyMain::get_character_ai()
{
    return character_ai;
}
void CharacterBodyMain::load_skeleton()
{
    clear_all();
    Ref<PackedScene> scene = ResourceLoader::load(skeleton_res);
    if(!scene.is_valid())
    {
        ERR_FAIL_MSG("load skeleton failed:" + skeleton_res);
        return ;
    }
    Node* ins = scene->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE);
    if (ins == nullptr) {
        ERR_FAIL_MSG("init skeleton instantiate failed:" + skeleton_res);
        return;
    }
    skeleton = Object::cast_to<Skeleton3D>(ins); 
    if(skeleton == nullptr)
    {
        ERR_FAIL_MSG("scene is not Skeleton3D:" + skeleton_res);
        memdelete(ins);
        return ;
    }
    skeleton->set_name("Skeleton3D");

    add_child(skeleton);
    skeleton->set_owner(this);
    player = memnew(AnimationPlayer);
    player->set_name("AnimationPlayer");
    add_child(player);
    player->set_owner(this);
    AnimationManager::setup_animation_tree(this,animation_group);
}
void CharacterBodyMain::load_mesh(const StringName& part_name,String p_mesh_file_path)
{
    auto old_ins = bodyPart.find(part_name);
    if(old_ins == bodyPart.end())
    {
        ERR_FAIL_MSG("not found body part:" + part_name.str());
        return;
    }
    Ref<CharacterBodyPart> mesh = ResourceLoader::load(p_mesh_file_path);
    if(!mesh.is_valid())
    {
        return;
    }
    Ref<CharacterBodyPartInstane> ins = old_ins->value;
    ins->set_part(mesh);
}
void CharacterBodyMain::behavior_tree_finished(int last_status)
{
    emit_signal("behavior_tree_finished", last_status);
}
void CharacterBodyMain::behavior_tree_update(int last_status)
{
    emit_signal("updated", last_status);
}
void CharacterBodyMain::skill_tree_finished(int last_status)
{
    emit_signal("skill_tree_finished", last_status);
}
void CharacterBodyMain::skill_tree_update(int last_status)
{
    emit_signal("skill_tree_updated", last_status);
}
BTPlayer * CharacterBodyMain::get_bt_player()
{
    if(btPlayer == nullptr)
    {
        btPlayer = memnew(BTPlayer);
        btPlayer->set_name("BTPlayer");
        btPlayer->connect("behavior_tree_finished", callable_mp(this, &CharacterBodyMain::behavior_tree_finished));
        btPlayer->connect("updated", callable_mp(this, &CharacterBodyMain::behavior_tree_update));
        btPlayer->get_blackboard()->set_parent(_get_blackboard());
        add_child(btPlayer);
        btPlayer->set_owner(this);
    }
    return btPlayer;
}

void CharacterBodyMain::set_blackboard(const Ref<Blackboard> &p_blackboard) 
{ 
    if(player_blackboard.is_null())
    {
        player_blackboard = p_blackboard;
    }
    else
    {
        player_blackboard = p_blackboard;
    }
    if(btPlayer != nullptr)
    {                
        btPlayer->get_blackboard()->set_parent(player_blackboard);
    }
}

void CharacterBodyMain::set_controller(const Ref<CharacterController> &p_controller) 
{
    controller = p_controller; 
}
Ref<CharacterController> CharacterBodyMain::get_controller()
{
    return controller; 
}
void CharacterBodyMain::set_navigation_agent(const Ref<CharacterNavigationAgent3D> &p_navigation_agent)
{
    if(character_agent == p_navigation_agent)
    {
        return;
    }
    if(character_agent.is_valid())
    {
        character_agent->set_body_main(nullptr);
    }
    character_agent = p_navigation_agent;
    if(character_agent.is_valid())
    {
        character_agent->set_body_main(this);
    }
}
Ref<CharacterNavigationAgent3D> CharacterBodyMain::get_navigation_agent()
{
    return character_agent;
}
bool CharacterBodyMain::play_skill(String p_skill_name)
{
    if(btSkillPlayer != nullptr)
    {
        return false;
    }
    btSkillPlayer = memnew(BTPlayer);
    btSkillPlayer->set_owner((Node*)this);
    btSkillPlayer->set_name("BTPlayer_Skill");
    btSkillPlayer->connect("behavior_tree_finished", callable_mp(this, &CharacterBodyMain::skill_tree_finished));
    btSkillPlayer->connect("updated", callable_mp(this, &CharacterBodyMain::skill_tree_update));
    add_child(btSkillPlayer);
    btSkillPlayer->set_owner(this);
    if(has_method("skill_tree_init"))
    {
        call("skill_tree_init",p_skill_name);
    }
    btSkillPlayer->get_blackboard()->set_parent(get_blackboard());

    get_blackboard()->set_var("skill_name",p_skill_name);
    get_blackboard()->set_var("skill_play",true);
    return true;
}
void CharacterBodyMain::stop_skill()
{
    get_blackboard()->set_var("skill_name","");
    get_blackboard()->set_var("skill_play",false);
    callable_mp(this, &CharacterBodyMain::_stop_skill).call_deferred();
}


void CharacterBodyMain::init_body_part_array(const Array& p_part_array)
{
    for(auto & a : bodyPart)
    {
        a.value->clear();
    }
    bodyPart.clear();

    for(int i = 0;i < p_part_array.size();i++)
    {
        StringName part_name = p_part_array[i];    
        if(part_name.is_empty())
        {
            continue;
        }
        Ref<CharacterBodyPartInstane> p;
        p.instantiate();
        p->set_skeleton(skeleton);
        bodyPart[part_name] = p;
    }
}
void CharacterBodyMain::init_blackboard_plan(Ref<BlackboardPlan> p_plan)
{
    if(!p_plan.is_valid())
    {
        return;
    }
    if(!p_plan->has_var("prop/curr_life"))
    {
        p_plan->add_var("prop/curr_life",BBVariable(Variant::FLOAT,0.0f));
    }
    if(!p_plan->has_var("prop/max_life"))
    {
        p_plan->add_var("prop/max_life",BBVariable(Variant::FLOAT,0.0f));
    }
    if(!p_plan->has_var("ai/curr_logic_node_name"))
    {
        p_plan->add_var("ai/curr_logic_node_name",BBVariable(Variant::STRING_NAME,StringName()));
    }


    if(!p_plan->has_var("move/look_target_pos"))
    {
        p_plan->add_var("move/look_target_pos",BBVariable(Variant::VECTOR3,Vector3()));
    }
    if(!p_plan->has_var("move/move_path"))
    {
        p_plan->add_var("move/move_path",BBVariable(Variant::OBJECT,Ref<Curve3D>()));
    }
    if(!p_plan->has_var("move/atack_target_pos"))
    {
        p_plan->add_var("move/atack_target_pos",BBVariable(Variant::VECTOR3,Vector3()));
    }
    if(!p_plan->has_var("move/navigation_target_pos"))
    {
        p_plan->add_var("move/navigation_target_pos",BBVariable(Variant::VECTOR3,Vector3()));
    }
    if(!p_plan->has_var("move/using_navigation_target"))
    {
        p_plan->add_var("move/using_navigation_target",BBVariable(Variant::BOOL,false));
    }

    if(!p_plan->has_var("move/request"))
    {
        p_plan->add_var("move/request",BBVariable(Variant::BOOL,false));
    }

    if(!p_plan->has_var("move/speed"))
    {
        p_plan->add_var("move/speed",BBVariable(Variant::FLOAT,0));
    }
    if(!p_plan->has_var("move/curr_speed"))
    {
        p_plan->add_var("move/curr_speed",BBVariable(Variant::FLOAT,0));
    }
    // 出生点
    if(!p_plan->has_var("move/birth_point"))
    {
        p_plan->add_var("move/birth_point",BBVariable(Variant::FLOAT,0));
    }
    // 巡逻范围
    if(!p_plan->has_var("move/patrol_range"))
    {
        p_plan->add_var("move/patrol_range",BBVariable(Variant::FLOAT,0));
    }

    if(!p_plan->has_var("phys/is_ground"))
    {
        p_plan->add_var("phys/is_ground",BBVariable(Variant::BOOL,false));
    }
    if(!p_plan->has_var("phys/to_ground_distance"))
    {
        p_plan->add_var("phys/to_ground_distance",BBVariable(Variant::FLOAT,0));
    }
    if(!p_plan->has_var("phys/ground_pos"))
    {
        p_plan->add_var("phys/ground_pos",BBVariable(Variant::VECTOR3,Vector3()));
    }
    if(!p_plan->has_var("phys/ground_normal"))
    {
        p_plan->add_var("phys/ground_normal",BBVariable(Variant::VECTOR3,Vector3()));
    }
    if(!p_plan->has_var("phys/ground_object_id"))
    {
        p_plan->add_var("phys/ground_object_id",BBVariable(Variant::INT,0));
    }
    if(!p_plan->has_var("phys/ground_collider_layer"))
    {
        p_plan->add_var("phys/ground_collider_layer",BBVariable(Variant::INT,0));
    }
    if(!p_plan->has_var("phys/is_fall"))
    {
        p_plan->add_var("phys/is_fall",BBVariable(Variant::BOOL,false));
    }
    if(!p_plan->has_var("phys/is_on_air"))
    {
        p_plan->add_var("phys/is_on_air",BBVariable(Variant::BOOL,false));
    }

    
    if(!p_plan->has_var("jump/request_jump"))
    {
        p_plan->add_var("jump/request_jump",BBVariable(Variant::BOOL,false));
    }  

    if(!p_plan->has_var("jump/is_jump"))
    {
        p_plan->add_var("jump/is_jump",BBVariable(Variant::BOOL,false));
    }   
    if(!p_plan->has_var("jump/is_jump2"))
    {
        p_plan->add_var("jump/is_jump2",BBVariable(Variant::BOOL,false));
    }  
    if(!p_plan->has_var("jump/jump_count"))
    {
        p_plan->add_var("jump/jump_count",BBVariable(Variant::INT,0));
    }
    // 请求角色复活
    if(!p_plan->has_var("dead/is_dead"))
    {
        p_plan->add_var("dead/is_dead",BBVariable(Variant::BOOL,false));
    }
    if(!p_plan->has_var("dead/request_revive"))
    {
        p_plan->add_var("dead/request_revive",BBVariable(Variant::BOOL,false));
    }
    if(!p_plan->has_var("dead/request_revive_life"))
    {
        p_plan->add_var("dead/request_revive_life",BBVariable(Variant::FLOAT,0.0f));
    }
}

void CharacterBodyMain::set_body_part(const Dictionary& part)
{
    Array keys = part.keys();
    
    HashMap<StringName,Ref<CharacterBodyPartInstane>> old_bodyPart = bodyPart;
    bodyPart.clear();
    for(int i = 0;i < keys.size();i++)
    {
        StringName part_name = keys[i];

        Ref<CharacterBodyPartInstane> p = part[part_name];
        if(old_bodyPart.has(part_name))
        {
            Ref<CharacterBodyPartInstane> mesh = old_bodyPart[part_name];
            if(p->get_part() != mesh->get_part())
            {
                mesh->set_part(p->get_part());
            }
            bodyPart[part_name] = mesh;        
            old_bodyPart.erase(part_name);       

        }
        else
        {
            // 克隆一份
            if(!p.is_valid())
            {
                p.instantiate();
            }
            else
            {
                p = p->duplicate();
            }
            p->set_skeleton(skeleton);
            bodyPart[part_name] = p; 
        }
    }
    for(auto & a : old_bodyPart)
    {
        a.value->clear();
    }
}
Dictionary CharacterBodyMain::get_body_part()
{
    Dictionary ret;
    for(auto & a : bodyPart)
    {
        ret[a.key] = a.value;
    }
    return ret;
}


CharacterBodyMain::CharacterBodyMain()
{
    character_movement.instantiate();
    mainCollision = memnew(CollisionShape3D);
    mainCollision->set_name("MainCollision");
    add_child(mainCollision);
    mainCollision->set_owner(this);
}
CharacterBodyMain::~CharacterBodyMain()
{
    
    for(uint32_t i = 0; i < check_area.size();++i)
    {
        if(check_area[i].is_valid())
        {
            check_area[i]->set_body_main(nullptr);
        }
    }
    check_area.clear();
}





void CharacterController::load_test()
{
    Ref<DataTableItem> data = DataTableManager::get_singleton()->get_data_table(DataTableManager::get_singleton()->get_body_table_name());
    if(data.is_null())
    {
        ERR_FAIL_MSG("data not found:" + DataTableManager::get_singleton()->get_body_table_name().str());
    }

    if(!data->data.has(load_test_id))
    {
        ERR_FAIL_MSG("data not found:" + itos(load_test_id));
    }

    CharacterBodyMain*body = get_load_test_player();
    if(body)
    {
        startup(body,data->data[load_test_id]);
    }


}

void CharacterController::log_player()
{
    CharacterBodyMain*body = get_load_test_player();
    if(body)
    {
        print_line(body->log_node());
    }
}


