#include "body_main.h"
#include "data_table_manager.h"


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

    ClassDB::bind_method(D_METHOD("set_skeleton", "skeleton"), &CharacterBodyMain::set_skeleton);
    ClassDB::bind_method(D_METHOD("get_skeleton"), &CharacterBodyMain::get_skeleton);

    ClassDB::bind_method(D_METHOD("set_animation_library", "animation_library"), &CharacterBodyMain::set_animation_library);
    ClassDB::bind_method(D_METHOD("get_animation_library"), &CharacterBodyMain::get_animation_library);

    ClassDB::bind_method(D_METHOD("set_animator", "animator"), &CharacterBodyMain::set_animator);
    ClassDB::bind_method(D_METHOD("get_animator"), &CharacterBodyMain::get_animator);

    ClassDB::bind_method(D_METHOD("set_main_shape", "shape"), &CharacterBodyMain::set_main_shape);
    ClassDB::bind_method(D_METHOD("get_main_shape"), &CharacterBodyMain::get_main_shape);

    ClassDB::bind_method(D_METHOD("init_body_part_array", "part_array"), &CharacterBodyMain::init_body_part_array);
    ClassDB::bind_method(D_METHOD("set_body_part", "part"), &CharacterBodyMain::set_body_part);
    ClassDB::bind_method(D_METHOD("get_body_part"), &CharacterBodyMain::get_body_part);


    


    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "behavior_tree", PROPERTY_HINT_RESOURCE_TYPE, "BehaviorTree"), "set_behavior_tree", "get_behavior_tree");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "update_mode", PROPERTY_HINT_ENUM, "Idle,Physics,Manual"), "set_update_mode", "get_update_mode");	
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard", PROPERTY_HINT_RESOURCE_TYPE, "Blackboard",PROPERTY_USAGE_DEFAULT ), "set_blackboard", "get_blackboard");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan", PROPERTY_USAGE_DEFAULT ), "set_blackboard_plan", "get_blackboard_plan");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "controller", PROPERTY_HINT_RESOURCE_TYPE, "CharacterController"), "set_controller", "get_controller");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "skeleton", PROPERTY_HINT_NODE_TYPE, "Skeleton",PROPERTY_USAGE_EDITOR), "set_skeleton", "get_skeleton");    
    // 只能編輯不能保存
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animation_library", PROPERTY_HINT_RESOURCE_TYPE, "AnimationLibrary",PROPERTY_USAGE_DEFAULT ), "set_animation_library", "get_animation_library");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animator", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimator",PROPERTY_USAGE_DEFAULT ), "set_animator", "get_animator");

    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "main_shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape3D",PROPERTY_USAGE_DEFAULT), "set_main_shape", "get_main_shape");
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
	switch (p_notification) {
		case NOTIFICATION_PROCESS: {
            // 更新玩家位置
            GDVIRTUAL_CALL(_update_player_position);
            // 更新動畫
            if(animator.is_valid())
            {
                animator->update_animation(get_process_delta_time());
            }
		} break;
    }

}
// 初始化身體
void CharacterBodyMain::init_main_body(String p_skeleton_file_path,StringName p_animation_group)
{
    skeleton_res = p_skeleton_file_path;
    animation_group = p_animation_group;
    load_skeleton();

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
        if(btPlayer == nullptr)
        {                
            btPlayer->get_blackboard()->set_parent(_get_blackboard());
        }
    }
    else
    {
        player_blackboard->copy_form(p_blackboard);
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
    mainShape = memnew(CollisionShape3D);
    mainShape->set_name("MainCollision");
    add_child(mainShape);
    mainShape->set_owner(this);
}
CharacterBodyMain::~CharacterBodyMain()
{
    
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


