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
	ClassDB::bind_method(D_METHOD("get_blackboard"), &CharacterBodyMain::get_blackboard);

	ClassDB::bind_method(D_METHOD("set_blackboard_plan", "plan"), &CharacterBodyMain::set_blackboard_plan);
	ClassDB::bind_method(D_METHOD("get_blackboard_plan"), &CharacterBodyMain::get_blackboard_plan);

    ClassDB::bind_method(D_METHOD("set_controller", "controller"), &CharacterBodyMain::set_controller);
    ClassDB::bind_method(D_METHOD("get_controller"), &CharacterBodyMain::get_controller);

    ClassDB::bind_method(D_METHOD("set_skeleton", "skeleton"), &CharacterBodyMain::set_skeleton);
    ClassDB::bind_method(D_METHOD("get_skeleton"), &CharacterBodyMain::get_skeleton);

    ClassDB::bind_method(D_METHOD("set_animator", "animator"), &CharacterBodyMain::set_animator);
    ClassDB::bind_method(D_METHOD("get_animator"), &CharacterBodyMain::get_animator);

    


    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "behavior_tree", PROPERTY_HINT_RESOURCE_TYPE, "BehaviorTree"), "set_behavior_tree", "get_behavior_tree");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "update_mode", PROPERTY_HINT_ENUM, "Idle,Physics,Manual"), "set_update_mode", "get_update_mode");	
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard", PROPERTY_HINT_RESOURCE_TYPE, "Blackboard",PROPERTY_USAGE_NONE), "set_blackboard", "get_blackboard");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT), "set_blackboard_plan", "get_blackboard_plan");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "controller", PROPERTY_HINT_RESOURCE_TYPE, "CharacterController"), "set_controller", "get_controller");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "skeleton", PROPERTY_HINT_NODE_TYPE, "Skeleton",PROPERTY_USAGE_EDITOR), "set_skeleton", "get_skeleton");    
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animator", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimator"), "set_animator", "get_animator");


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
    if(old_ins != bodyPart.end())
    {
        old_ins->value.clear();
        bodyPart.remove(old_ins);
    }
    Ref<CharacterBodyPart> mesh = ResourceLoader::load(p_mesh_file_path);
    if(!mesh.is_valid())
    {
        return;
    }
    CharacterBodyPartInstane ins;
    ins.init(this,mesh,skeleton);
    bodyPart.insert(mesh->get_name(),ins);
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
        btPlayer->set_owner((Node*)this);
        btPlayer->set_name("BTPlayer");
        btPlayer->connect("behavior_tree_finished", callable_mp(this, &CharacterBodyMain::behavior_tree_finished));
        btPlayer->connect("updated", callable_mp(this, &CharacterBodyMain::behavior_tree_update));
        btPlayer->get_blackboard()->set_parent(player_blackboard);
        add_child(btPlayer);
        btPlayer->set_owner(this);
    }
    return btPlayer;
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
    btSkillPlayer->get_blackboard()->set_parent(player_blackboard);

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

CharacterBodyMain::CharacterBodyMain()
{
    player_blackboard.instantiate();
    animator.instantiate();
    animator->set_body(this);
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


