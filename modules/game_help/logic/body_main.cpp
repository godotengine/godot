#include "body_main.h"
#include "data_table_manager.h"
#include "scene/3d/path_3d.h"
#include "scene/resources/3d/capsule_shape_3d.h"
#include "character_ai/character_ai.h"
#include "character_manager.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#define EDITOR_OPTIMIZE_ANIMATION 0

CharacterAIContext::CharacterAIContext()
{
	beehave_run_context.instantiate();
}

void CharacterBodyMain::init()
{
    if(character_ai.is_null())
    {
		character_ai.instantiate();
    }
	character_ai->init();
    if(animator.is_null())
    {
        animator.instantiate();
    }
    animator->set_body(this);
    animator->init();

    // 创建外形
    if(mainShape.is_null())
    {
        mainShape.instantiate();
        Ref<CapsuleShape3D> shape;
        shape.instantiate();
        shape->set_radius(0.5f);
        shape->set_height(2.0f);

        mainShape->set_shape(shape);
		mainShape->set_position(Vector3(0, 1, 0));
        mainShape->set_link_target(this);        
    }
}
void CharacterBodyMain::clear_all()
{
    if(bone_label != nullptr) {
        bone_label->clear();
    }
    Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(skeletonID));
    if(skeleton == nullptr)
    {
        return;
    }
    if(ik.is_valid()) {
        ik.unref();
    }
    if(skeleton)
    {
        remove_child(skeleton);
        skeleton->queue_free();
        skeleton = nullptr;
        
    }
    bodyPart.clear();
    if(animator.is_valid()) {
        animator->set_body(nullptr);
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
        case NOTIFICATION_ENTER_TREE: {
			if(CharacterManager::get_singleton() != nullptr)
				CharacterManager::get_singleton()->register_character(this);
        } break;
        case NOTIFICATION_EXIT_TREE: {
			if (CharacterManager::get_singleton() != nullptr)
				CharacterManager::get_singleton()->unregister_character(this);            
        }
        break;
    }

}

void CharacterBodyMain::audio_add_socket() {
    if(audio_socket_name.str().is_empty()) {
        return;
    }
    if(audio_players.has(audio_socket_name)) {
        return;
    }
    Ref<AudioStreamPlayer3DCompoent> player = memnew(AudioStreamPlayer3DCompoent);
    player->set_owenr(this);
    audio_players[audio_socket_name] = player;
}

void CharacterBodyMain::init_ai_context()
{
	if (ai_context.is_null()) {
		ai_context.instantiate();
		ai_context->beehave_run_context->blackboard = get_blackboard();
		ai_context->beehave_run_context->actor = this;
	}

}
void CharacterBodyMain::_update(double p_delta)
{

    // 更新玩家位置
    GDVIRTUAL_CALL(_update_player_position);
    for(uint32_t i = 0; i < check_area.size();++i)
    {
        if(check_area[i].is_valid())
        {
            check_area[i]->update_world_move(get_global_position());
        }
    }
    _process_move();

}
void CharacterBodyMain::_update_ai()
{
#ifdef TOOLS_ENABLED
    if(!run_ai)
    {
        return;
    }
#endif
	init_ai_context();
    if(character_ai.is_valid())
    {
        character_ai->execute(this,ai_context.ptr());
    }
}
void CharacterBodyMain::_process_animator(double time_delta)
{
    if(editor_pause_animation) {
        return ;
    }
    if(animator.is_valid())
    {
        animator->_thread_update_animator(time_delta * editor_animation_speed);
    }
}
void CharacterBodyMain::_process_animation()
{
    if(animator.is_valid())
    {
        animator->_thread_update_animation(get_process_delta_time());
    }
}
void CharacterBodyMain::_process_ik()
{
    float delta = get_process_delta_time();
    if(ik.is_valid())
    {
        ik->update_ik();
        ik->update_placement(delta);
    }

    if(bone_label != nullptr) 
    {
        bone_label->update();
    }
}

void CharacterBodyMain::_process_move()
{
	if (animator.is_null()) {
		return;
	}
    // 处理角色移动
    bool is_walk = get_blackboard()->get_var("move/using_navigation_target",false);
    Vector3 _velocity = Vector3();
    if(animator.is_valid()) {
        const CharacterRootMotion& root_motion = animator->get_root_motion();

        Transform3D rot =get_transform();


        rot.basis = root_motion.root_motion_rotation_add * rot.basis;
        set_transform(rot);
        Vector3 forward = rot.basis.xform(Vector3(0,0,1));

		_velocity = root_motion.get_velocity(forward,animator->get_time_delta(),is_on_floor());

    }
    if(is_walk )
    {
        // 处理导航行走
        if(character_agent.is_valid())
        {
            if(is_walk)
            {
                Vector3 target_pos = get_blackboard()->get_var("move/navigation_target_pos",Vector3());
                // 设置角色移动速度
                character_agent->set_velocity(_velocity * editor_animation_speed);
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
        set_velocity(_velocity);

        move_and_slide(animator->get_time_delta());
    }

}
void CharacterBodyMain::set_character_ai(const Ref<CharacterAI> &p_ai)
{
    character_ai = p_ai;
}
Ref<CharacterAI> CharacterBodyMain::get_character_ai()
{
    return character_ai;
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



void CharacterBodyMain::set_body_prefab(const Ref<CharacterBodyPrefab> &p_body_prefab)
{
    if(p_body_prefab.is_null() || p_body_prefab == body_prefab)
    {
        return;
    }
    if(body_prefab.is_valid())
    {
        body_prefab->disconnect_changed(callable_mp(this, &CharacterBodyMain::load_prefab));
    }
    body_prefab = p_body_prefab;
    
    load_prefab();


}
void CharacterBodyMain::load_prefab()
{
	_init_body();
}
void CharacterBodyMain::_init_body()
{
	// 存储一下引用,避免当前再次引用的资产不会立即释放
	clear_all();
	HashMap<StringName, Ref<CharacterBodyPartInstane>> old_part = bodyPart;
	if (body_prefab.is_valid())
	{
		body_prefab->connect_changed(callable_mp(this, &CharacterBodyMain::load_prefab));
		Ref<PackedScene> scene;
		String skeleton_res = body_prefab->skeleton_path;
		if (skeleton_res.begins_with("res://"))
			scene = ResourceLoader::load(skeleton_res);
		else
			scene = ResourceLoader::load(CharacterManager::get_singleton()->get_skeleton_root_path(body_prefab->get_is_human()).path_join(skeleton_res));
		if (!scene.is_valid())
		{
			ERR_FAIL_MSG("load skeleton failed:" + skeleton_res);
			skeleton_res = "";
			return;
		}
		Node* ins = scene->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE);
		if (ins == nullptr) {
			ERR_FAIL_MSG("init skeleton instantiate failed:" + skeleton_res);
			skeleton_res = "";
			return;
		}
		Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(ins);
		if (skeleton == nullptr)
		{
			ERR_FAIL_MSG("scene is not Skeleton3D:" + skeleton_res);
			skeleton->queue_free();
			skeleton_res = "";
			return;
		}
		skeleton->set_name("Skeleton3D");

		add_child(skeleton);
		skeleton->set_owner(this);
		skeleton->set_dont_save(true);



		if (skeleton)
		{
            ik.instantiate();
			ik->_initialize(skeleton);
		}
		skeletonID = skeleton->get_instance_id();
		// 
		TypedArray<CharacterBodyPart> part_array = body_prefab->load_part();
		int size = part_array.size();
		for (int i = 0; i < size; i++)
		{

			Ref<CharacterBodyPartInstane> p;
			p.instantiate();
			p->set_skeleton(skeleton);
			p->set_part(part_array[i]);
            p->set_show_mesh(editor_show_mesh);
			Ref< CharacterBodyPart> part = part_array[i];
			bodyPart[part->get_name()] = p;
		}
        if(animator.is_valid()) {
            animator->set_body(this);
        }
		//update_bone_visble();
		notify_property_list_changed();
	}

}
Ref<CharacterBodyPrefab> CharacterBodyMain::get_body_prefab()
{
    return body_prefab;
}



CharacterBodyMain::CharacterBodyMain()
{
    character_movement.instantiate();
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

void CharacterBodyMain::_bind_methods()
{
    
	ClassDB::bind_method(D_METHOD("restart"), &CharacterBodyMain::restart);

    ClassDB::bind_method(D_METHOD("set_audio_play_component", "audio_play_component"), &CharacterBodyMain::set_audio_play_component);
    ClassDB::bind_method(D_METHOD("get_audio_play_component"), &CharacterBodyMain::get_audio_play_component);


    ClassDB::bind_method(D_METHOD("play_audio", "audio_socket", "stream"), &CharacterBodyMain::play_audio);

    ClassDB::bind_method(D_METHOD("get_audio_player"), &CharacterBodyMain::get_audio_player);

    ClassDB::bind_method(D_METHOD("get_animation_Group"), &CharacterBodyMain::get_animation_Group);

	ClassDB::bind_method(D_METHOD("set_blackboard_plan", "plan"), &CharacterBodyMain::set_blackboard_plan);
	ClassDB::bind_method(D_METHOD("get_blackboard_plan"), &CharacterBodyMain::get_blackboard_plan);


    ClassDB::bind_method(D_METHOD("set_navigation_agent", "navigation_agent"), &CharacterBodyMain::set_navigation_agent);
    ClassDB::bind_method(D_METHOD("get_navigation_agent"), &CharacterBodyMain::get_navigation_agent);

    ClassDB::bind_method(D_METHOD("set_animation_library", "animation_library"), &CharacterBodyMain::set_animation_library);
    ClassDB::bind_method(D_METHOD("get_animation_library"), &CharacterBodyMain::get_animation_library);

    ClassDB::bind_method(D_METHOD("set_animator", "animator"), &CharacterBodyMain::set_animator);
    ClassDB::bind_method(D_METHOD("get_animator"), &CharacterBodyMain::get_animator);

    ClassDB::bind_method(D_METHOD("set_ik", "ik"), &CharacterBodyMain::set_ik);
    ClassDB::bind_method(D_METHOD("get_ik"), &CharacterBodyMain::get_ik);

    ClassDB::bind_method(D_METHOD("set_main_shape", "shape"), &CharacterBodyMain::set_main_shape);
    ClassDB::bind_method(D_METHOD("get_main_shape"), &CharacterBodyMain::get_main_shape);

    ClassDB::bind_method(D_METHOD("set_check_area", "check_area"), &CharacterBodyMain::set_check_area);
    ClassDB::bind_method(D_METHOD("get_check_area"), &CharacterBodyMain::get_check_area);
    ClassDB::bind_method(D_METHOD("get_check_area_by_name", "name"), &CharacterBodyMain::get_check_area_by_name);


    ClassDB::bind_method(D_METHOD("set_body_prefab", "body_prefab"), &CharacterBodyMain::set_body_prefab);
    ClassDB::bind_method(D_METHOD("get_body_prefab"), &CharacterBodyMain::get_body_prefab);


    ClassDB::bind_method(D_METHOD("set_character_ai", "ai"), &CharacterBodyMain::set_character_ai);
    ClassDB::bind_method(D_METHOD("get_character_ai"), &CharacterBodyMain::get_character_ai);

    
    ClassDB::bind_method(D_METHOD("set_editor_form_mesh_file_path", "editor_form_mesh_file_path"), &CharacterBodyMain::set_editor_form_mesh_file_path);
    ClassDB::bind_method(D_METHOD("get_editor_form_mesh_file_path"), &CharacterBodyMain::get_editor_form_mesh_file_path);


    ClassDB::bind_method(D_METHOD("set_editor_animation_file_path", "path"), &CharacterBodyMain::set_editor_animation_file_path);
    ClassDB::bind_method(D_METHOD("get_editor_animation_file_path"), &CharacterBodyMain::get_editor_animation_file_path);


	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan", PROPERTY_USAGE_DEFAULT ), "set_blackboard_plan", "get_blackboard_plan");


    ADD_SUBGROUP("audio", "audio_");
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "audio_play_component" ), "set_audio_play_component", "get_audio_play_component");
    IMP_GODOT_PROPERTY(StringName,audio_socket_name)
    ADD_MEMBER_BUTTON(audio_add_socket,L"增加音频插槽",CharacterBodyMain);

    ADD_SUBGROUP("editor", "editor_");

    ClassDB::bind_method(D_METHOD("set_editor_show_mesh", "editor_show_mesh"), &CharacterBodyMain::set_editor_show_mesh);
    ClassDB::bind_method(D_METHOD("get_editor_show_mesh"), &CharacterBodyMain::get_editor_show_mesh);

    ClassDB::bind_method(D_METHOD("set_editor_is_skeleton_human", "editor_is_skeleton_human"), &CharacterBodyMain::set_editor_is_skeleton_human);
    ClassDB::bind_method(D_METHOD("get_editor_is_skeleton_human"), &CharacterBodyMain::get_editor_is_skeleton_human);

    ClassDB::bind_method(D_METHOD("set_editor_animation_group"), &CharacterBodyMain::set_editor_animation_group);
    ClassDB::bind_method(D_METHOD("get_editor_animation_group"), &CharacterBodyMain::get_editor_animation_group);

    ClassDB::bind_method(D_METHOD("set_editor_convert_animations_path", "path"), &CharacterBodyMain::set_editor_convert_animations_path);
    ClassDB::bind_method(D_METHOD("get_editor_convert_animations_path"), &CharacterBodyMain::get_editor_convert_animations_path);

    ClassDB::bind_method(D_METHOD("set_play_animation", "play_animation"), &CharacterBodyMain::set_play_animation);
    ClassDB::bind_method(D_METHOD("get_play_animation"), &CharacterBodyMain::get_play_animation);

    ClassDB::bind_method(D_METHOD("set_play_animayion_speed", "speed"), &CharacterBodyMain::set_play_animayion_speed);
    ClassDB::bind_method(D_METHOD("get_play_animayion_speed"), &CharacterBodyMain::get_play_animayion_speed);


    ClassDB::bind_method(D_METHOD("set_is_position_by_hip_bone", "is_positiobn_by_hip_bone"), &CharacterBodyMain::set_is_position_by_hip_bone);
    ClassDB::bind_method(D_METHOD("get_is_position_by_hip_bone"), &CharacterBodyMain::get_is_position_by_hip_bone);



    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_show_mesh", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_editor_show_mesh", "get_editor_show_mesh");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "editor_mesh_file_path"), "set_editor_form_mesh_file_path", "get_editor_form_mesh_file_path");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_is_skeleton_human"), "set_editor_is_skeleton_human", "get_editor_is_skeleton_human");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_is_position_by_hip_bone"), "set_is_position_by_hip_bone", "get_is_position_by_hip_bone");
    ADD_MEMBER_BUTTON(editor_build_form_mesh_file_path,L"根据模型初始化",CharacterBodyMain);

    

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "editor_animation_file_path",PROPERTY_HINT_FILE,"tres,*.tres"), "set_editor_animation_file_path", "get_editor_animation_file_path");


	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "editor_animation_group", PROPERTY_HINT_ENUM_DYNAMIC_LIST, "get_animation_Group",PROPERTY_USAGE_EDITOR), "set_editor_animation_group", "get_editor_animation_group");
    ADD_MEMBER_BUTTON(editor_build_animation,L"构建动画文件信息",CharacterBodyMain);


    ADD_PROPERTY(PropertyInfo(Variant::STRING, "editor_convert_animations_path", PROPERTY_HINT_DIR), "set_editor_convert_animations_path", "get_editor_convert_animations_path");

    ADD_MEMBER_BUTTON(editor_convert_animations_bt,L"转换动画文件夹",CharacterBodyMain);

    ADD_SUBGROUP("animation_test", "animation_test");
    
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animation_test_play_animation", PROPERTY_HINT_RESOURCE_TYPE, "Animation"), "set_play_animation", "get_play_animation");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "animation_test_play_animation_speed", PROPERTY_HINT_RANGE, "0,2,0.01", PROPERTY_USAGE_EDITOR), "set_play_animayion_speed", "get_play_animayion_speed");
    ADD_MEMBER_BUTTON(animation_test_play_select_animation,L"播放动画",CharacterBodyMain);


    ADD_SUBGROUP("humanizer", "humanizer");

    ADD_MEMBER_BUTTON(humanizer_install_mkhm,L"安装mkhm包",CharacterBodyMain);





    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "body_prefab", PROPERTY_HINT_RESOURCE_TYPE, "CharacterBodyPrefab",PROPERTY_USAGE_DEFAULT ), "set_body_prefab", "get_body_prefab");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animator", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimator",PROPERTY_USAGE_DEFAULT ), "set_animator", "get_animator"); 
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animation_library", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimationLibrary",PROPERTY_USAGE_DEFAULT ), "set_animation_library", "get_animation_library");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "ik", PROPERTY_HINT_RESOURCE_TYPE, "RenIK",PROPERTY_USAGE_DEFAULT ), "set_ik", "get_ik");


    ADD_SUBGROUP("logic", "");

    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "character_ai", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAI"), "set_character_ai", "get_character_ai");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navigation_agent", PROPERTY_HINT_RESOURCE_TYPE, "CharacterNavigationAgent3D"), "set_navigation_agent", "get_navigation_agent");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "main_shape", PROPERTY_HINT_RESOURCE_TYPE, "CollisionObject3DConnectionShape",PROPERTY_USAGE_DEFAULT), "set_main_shape", "get_main_shape");
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "check_area", PROPERTY_HINT_RESOURCE_TYPE, MAKE_RESOURCE_TYPE_HINT("CharacterCheckArea3D"),PROPERTY_USAGE_DEFAULT), "set_check_area", "get_check_area");


	ADD_SIGNAL(MethodInfo("behavior_tree_finished", PropertyInfo(Variant::INT, "status")));
	ADD_SIGNAL(MethodInfo("behavior_tree_updated", PropertyInfo(Variant::INT, "status")));
    
	ADD_SIGNAL(MethodInfo("skill_tree_finished", PropertyInfo(Variant::INT, "status")));
	ADD_SIGNAL(MethodInfo("skill_tree_updated", PropertyInfo(Variant::INT, "status")));

    

}

/*********************************************************************************************************************************************************************************************************************************/


ObjectID& CharacterBodyMain::get_curr_editor_player()
{
    static ObjectID curr_editor_player;
    return curr_editor_player;
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

    // 人物情绪
    if(!p_plan->has_var("ai/emotion"))
    {
        p_plan->add_var("ai/emotion",BBVariable(Variant::INT,0,PROPERTY_HINT_ENUM,L"平静,感激,愤怒,激动,开心,恐惧"));
    }

    // 人物性格
    if(!p_plan->has_var("ai/personality"))
    {
        p_plan->add_var("ai/personality",BBVariable(Variant::INT,0,PROPERTY_HINT_ENUM,L"和平,好斗,胆小,忠诚,复仇心强,狡猾"));
    }
}
Array CharacterBodyMain::get_animation_Group() const {
    Array arr;
    CharacterManager::get_singleton()->get_animation_groups(&arr);
    return arr;
}



// 保存模型资源
static void save_fbx_res( const String& group_name,const String& sub_path,const Ref<Resource>& p_resource,String& save_path, bool is_resource = true)
{
	String export_root_path = "res://Assets/public";
	if (!DirAccess::exists("res://Assets"))
	{
		DirAccess::make_dir_absolute("res://Assets");
	}
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	export_root_path  = export_root_path.path_join(group_name);
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	export_root_path = export_root_path.path_join(sub_path);
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	save_path = export_root_path.path_join(p_resource->get_name() + (is_resource ? ".res" :".scn"));
	ResourceSaver::save(p_resource, save_path, ResourceSaver::FLAG_CHANGE_PATH);
	ResourceCache::set_ref(save_path, p_resource.ptr());
	print_line(L"CharacterBodyMain.save_fbx_res: 存储资源 :" + save_path);
    save_path = sub_path.path_join(p_resource->get_name() + (is_resource ? ".res" :".scn"));
}
static void save_fbx_tres( const String& group_name,const String& sub_path,const Ref<Resource>& p_resource,String& save_path, bool is_resource = true)
{
	String export_root_path = "res://Assets/public";
	if (!DirAccess::exists("res://Assets"))
	{
		DirAccess::make_dir_absolute("res://Assets");
	}
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	export_root_path  = export_root_path.path_join(group_name);
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	export_root_path = export_root_path.path_join(sub_path);
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	save_path = export_root_path.path_join(p_resource->get_name() + (is_resource ? ".tres" :".tscn"));
	ResourceSaver::save(p_resource, save_path, ResourceSaver::FLAG_CHANGE_PATH);
	print_line(L"CharacterBodyMain.save_fbx_res: 存储资源 :" + save_path);
    save_path = sub_path.path_join(p_resource->get_name() + (is_resource ? ".tres" :".tscn"));
}
static void get_fbx_meshs(Node *p_node,HashMap<String,MeshInstance3D* > &meshs)
{

	for(int i=0;i<p_node->get_child_count();i++)
	{
		Node * child = p_node->get_child(i);
		if(child->get_class() == "MeshInstance3D")
		{
			MeshInstance3D * mesh = Object::cast_to<MeshInstance3D>(child);
			if(!meshs.has(mesh->get_name())){
				meshs[mesh->get_name()] = mesh;
			}
			else{
				String name = mesh->get_name();
				int index = 1;
				while(meshs.has(name +"_"+ itos(index))){
					name = mesh->get_name().str() + "_" + itos(index);
					index++;
				}
				meshs[name] = mesh;
			}
		}
		get_fbx_meshs(child,meshs);
	}
}
void reset_owenr(Node* node, Node* owenr)
{
	for (int i = 0; i < node->get_child_count(); ++i)
	{
		Node* c = node->get_child(i);
		c->set_owner(nullptr);
		reset_owenr(c, owenr);
	}
}
Ref<CharacterBodyPrefab> CharacterBodyMain::build_prefab(const String& mesh_path,bool p_is_skeleton_human)
{
	if (!FileAccess::exists(mesh_path))
	{
		return Ref<CharacterBodyPrefab>();
	}

	// 加载模型
	Ref<PackedScene> scene = ResourceLoader::load(mesh_path);

	if (scene.is_null())
	{
		print_line(L"CharacterBodyMain: 路径不存在 :" + mesh_path);
		return Ref<CharacterBodyPrefab>();
	}
	Node* p_node = scene->instantiate(PackedScene::GEN_EDIT_STATE_DISABLED);
	String p_group = mesh_path.get_file().get_basename();

	Node* node = p_node->find_child("Skeleton3D");
	Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(node);

	Dictionary bone_map;
	String ske_save_path, bone_map_save_path;
	if (skeleton != nullptr)
	{
		bone_map = skeleton->get_human_bone_mapping();
        Vector<String> bone_names = skeleton->get_bone_names();
        
        Ref<HumanBoneConfig> config;

		skeleton->set_human_bone_mapping(bone_map);
        
        if(p_is_skeleton_human)
        {
			config.instantiate();
			HashMap<String,String> _bone_label = HumanAnim::HumanAnimmation::get_bone_label();
			HumanAnim::HumanAnimmation::build_virtual_pose(skeleton, *config.ptr(), _bone_label);
            skeleton->set_human_config(config);
            config = skeleton->get_human_config();
            config->set_name("human_config");
		    save_fbx_res("human_config", p_group, config, ske_save_path, true);
        }

		// 存储骨骼映射
		Ref<CharacterBoneMap> bone_map_ref;
		bone_map_ref.instantiate();
		bone_map_ref->set_name("bone_map");
		bone_map_ref->set_bone_map(bone_map);
        bone_map_ref->set_bone_names(bone_names);
        bone_map_ref->set_human_config(config);
        bone_map_ref->set_skeleton_path(p_group.path_join("skeleton.scn" ));
        if(p_is_skeleton_human) {
		    save_fbx_res("human_bone_map", p_group, bone_map_ref, bone_map_save_path, true);
        }
        else {
		    save_fbx_res("bone_map", p_group, bone_map_ref, bone_map_save_path, true);            
        }


		skeleton->set_owner(nullptr);
		reset_owenr(skeleton, skeleton);

		// 存儲骨架信息
		Ref<PackedScene> packed_scene;
		packed_scene.instantiate();
		packed_scene->pack(skeleton);
		packed_scene->set_name("skeleton");
        if(p_is_skeleton_human) {
		    save_fbx_res("human_skeleton", p_group, packed_scene, ske_save_path, false);
        }
        else {
		    save_fbx_res("skeleton", p_group, packed_scene, ske_save_path, false);
        }

	}
	// 生成预制体
	Ref<CharacterBodyPrefab> _body_prefab;
	_body_prefab.instantiate();
	_body_prefab->set_name(p_group);
	HashMap<String, MeshInstance3D* > meshs;
	// 便利存儲模型文件
	get_fbx_meshs(p_node, meshs);
	for (auto it = meshs.begin(); it != meshs.end(); ++it) {
		Ref<CharacterBodyPart> part;
		part.instantiate();
		MeshInstance3D* mesh = it->value;
		part->init_form_mesh_instance(mesh, bone_map);
        
		part->set_name(it->key);
		String save_path;
        if(p_is_skeleton_human) {
		    save_fbx_res("human_meshs", p_group, part, save_path, true);
        }
        else {
		    save_fbx_res("meshs", p_group, part, save_path, true);
        }
		_body_prefab->parts[save_path] = true;
	}
	// 保存预制体
	_body_prefab->skeleton_path = ske_save_path;
	_body_prefab->set_is_human(p_is_skeleton_human);
    if(p_is_skeleton_human) {
	    save_fbx_res("human_prefab", p_group, _body_prefab, bone_map_save_path, true);
    }
    else {
	    save_fbx_res("prefab", p_group, _body_prefab, bone_map_save_path, true);
    }


	p_node->queue_free();
	return _body_prefab;
}
void CharacterBodyMain::editor_build_form_mesh_file_path()
{
	Ref<CharacterBodyPrefab> prefab = build_prefab(editor_form_mesh_file_path,is_skeleton_human);
    // 设置预制体
    set_body_prefab(prefab);
    
}
void CharacterBodyMain::animation_test_play_select_animation() {
    init();
    if(play_animation.is_null()) {
        return;
    }
    if(animator.is_null()) {
        return;
    }
    animator->editor_play_animation(play_animation);
}
void CharacterBodyMain::update_bone_visble()
{
    if(bone_label != nullptr) {
        bone_label->clear();
    }
    Skeleton3D* skeleton = get_skeleton();
    if (skeleton == nullptr)
    {
        return;
    }
    if(play_animation.is_null()) {
        return;
    }
    if(bone_label == nullptr) {
        bone_label = memnew(HumanBoneVisble);
        add_child(bone_label);
        bone_label->set_owner(this);
    }
	bone_label->init(get_skeleton(), play_animation->get_bone_map());
    
}


static void node_to_bone_skeleton(Skeleton3D* p_ske, Node3D* p_node, int bode_parent) {
	int index = bode_parent;
	index = p_ske->add_bone(p_node->get_name());
	p_ske->set_bone_parent(index, bode_parent);
	Transform3D trans = p_node->get_transform();
	p_ske->set_bone_pose(index, trans);
	

	for (int i = 0; i < p_node->get_child_count(); ++i) {
		Node3D* node = Object::cast_to<Node3D>(p_node->get_child(i));
		if (node != nullptr) {
			node_to_bone_skeleton(p_ske, node, index);

		}
	}
	
}

void CharacterBodyMain::editor_build_animation() {
    if(editor_animation_group.is_empty()) {
        WARN_PRINT("请先设置动画组名");
        return;
    }
    editor_build_animation_form_path(editor_animation_file_name);
}
void CharacterBodyMain::editor_build_animation_form_path(String p_file_path)
{
    if(!FileAccess::exists(p_file_path))
    {
		print_line(L"CharacterBodyMain: 路径不存在 :" + p_file_path);
        return;
    }
	Ref<PackedScene> scene = ResourceLoader::load(p_file_path);
	if (scene.is_null())
	{
		print_line(L"CharacterBodyMain: 路径不存在 :" + p_file_path);
        return;
	}
	Node* p_node = scene->instantiate(PackedScene::GEN_EDIT_STATE_DISABLED);
    Node* node = p_node->find_child("Skeleton3D");
    Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(node);

	Node* anim_node = p_node->find_child("AnimationPlayer");
    if(anim_node == nullptr)
    {
        print_line(L"CharacterBodyMain: 路径不存在动画信息:" + p_file_path);
        return;
    }

    AnimationPlayer* player = Object::cast_to<AnimationPlayer>(anim_node);
    if(player == nullptr)
    {
        print_line(L"CharacterBodyMain: 路径不存在动画信息:" + p_file_path);
        return;
    }
	String p_group = p_file_path.get_file().get_basename();
    List<StringName> p_animations;
    player->get_animation_list(&p_animations);


	bool is_node_skeleton = false;
	Skeleton3D* bone_map_skeleton;

	HashMap<String, int> human_bone_name_index;
	Dictionary bone_map;
	Vector<String> bone_names;
	Ref<HumanBoneConfig> animation_human_config;
	//if (skeleton == nullptr)
	{
		is_node_skeleton = true;

		HashSet<String> node_name;
		for (const StringName& E : p_animations) {
			Ref<Animation> animation = player->get_animation(E);
			animation->get_node_names(node_name);
		}

		bone_map_skeleton = memnew(Skeleton3D);

		for (int i = 0; i < p_node->get_child_count(); ++i) {
			Node3D* child = Object::cast_to<Node3D>(p_node->get_child(i));
			if (child != nullptr) {
				if (child->get_child_count() > 0) {
					node_to_bone_skeleton(bone_map_skeleton, child, -1);
					break;
				}
			}
		}
		bone_names = bone_map_skeleton->get_bone_names();

		bone_map = bone_map_skeleton->get_human_bone_mapping();
		bone_map_skeleton->set_human_bone_mapping(bone_map);
	}
	//else

	// 有些动画的骨架可能存在多份,选择骨头最多的当做身体
	if(skeleton != nullptr && skeleton->get_bone_count() > bone_map.size())
	{
		auto new_bone_map = skeleton->get_human_bone_mapping();
		if (new_bone_map.size() > bone_map.size()) {
			bone_map = new_bone_map;
			bone_names = skeleton->get_bone_names();

			skeleton->set_human_bone_mapping(bone_map);
			bone_map_skeleton = skeleton;
		}
	}
	if (bone_map.size() < 2) {
			print_line(L"CharacterBodyMain: 动画的骨架不支持:" + p_file_path);
		return;
	}




    animation_human_config.instantiate();
    HashMap<String, String> _bone_label = HumanAnim::HumanAnimmation::get_bone_label();
    HumanAnim::HumanAnimmation::build_virtual_pose(bone_map_skeleton, *animation_human_config.ptr(), _bone_label);
	for (int i = 0; i < bone_names.size(); ++i) {
		human_bone_name_index[bone_names[i]] = i;
	}



    for (const StringName &E : p_animations) {
        Ref<Animation> animation = player->get_animation(E);
        if(animation.is_valid())
        {
            Ref<Animation> new_animation;
			new_animation = animation->duplicate();
            if(skeleton == nullptr)
            {
                new_animation->remap_node_to_bone_name(bone_names);
            }

			// 如果存在人形动作配置,转换动画为人形动画
			if (animation_human_config.is_valid()) {
				new_animation = HumanAnim::HumanAnimmation::build_human_animation(bone_map_skeleton, *animation_human_config.ptr(), new_animation, bone_map, is_position_by_hip_bone);
			}
            new_animation->set_animation_group(editor_animation_group);
            new_animation->optimize();
#if EDITOR_OPTIMIZE_ANIMATION
            new_animation->compress();
#endif
            play_animation = new_animation;
			String group = p_group;
			if (p_animations.size() == 1)
			{
				Vector<String> names = p_group.split("@");
				if (names.size() == 2)
				{
					group = names[0];
				}
				String name;
				if (names.size() > 0)
				{
					name = names[names.size() - 1];
				}
				else
				{
					name = E;
				}
				new_animation->set_name(name);
			}
			else
			{
				new_animation->set_name(E);
			}
            String save_path;
            if(animation_human_config.is_valid())  {
			    save_fbx_res("human_animation", group, new_animation, save_path, true);
            }
            else {
			    save_fbx_res("animation", group, new_animation, save_path, true);
            }
            
        }
    }
	if (is_node_skeleton) {
		memdelete(bone_map_skeleton);
		bone_map_skeleton = nullptr;
	}
    p_node->queue_free();
}
void CharacterBodyMain::editor_convert_animations_bt() {

    if( !DirAccess::exists(editor_convert_animations_path) ) {
        return;
    }
    if(editor_animation_group.is_empty()) {
        WARN_PRINT("请先设置动画组名");
        return;
    }
    editor_convert_animations(editor_convert_animations_path);
}
void CharacterBodyMain::editor_convert_animations(String p_file_path)
{

    PackedStringArray files = DirAccess::get_files_at(p_file_path);

    for (int i = 0; i < files.size(); ++i) {
        String file = files[i];
        String ext = file.get_extension().to_lower();
        if (ext == "fbx" || ext == "gltf" || ext == "glb") {
            editor_build_animation_form_path(p_file_path.path_join(file));
        }
    }
    PackedStringArray dirs = DirAccess::get_directories_at(p_file_path);
    for (int i = 0; i < dirs.size(); ++i) {
        String dir = p_file_path.path_join(dirs[i]);
        editor_convert_animations(dir);
    }

}

#include "modules/zip/zip_reader.h"


static void make_res_path(const String& p_path) {
    if(!p_path.begins_with("res://")) {
        return ;
    }
    String path = p_path.replace("res://", "");
    path = path.replace("\\", "/");
    Vector<String> paths = path.split("/");

    String curr_path = "res://";
    for(int i = 0; i < paths.size(); i++) {
        curr_path += paths[i];
        if(!DirAccess::exists(curr_path)) {
            DirAccess::make_dir_absolute(curr_path);
        }
        curr_path += "/";
    }

}

static void make_res_path_form_filepath(const String& p_path) {
    if(!p_path.begins_with("res://")) {
        return ;
    }
    String path = p_path.replace("res://", "");
    path = path.replace("\\", "/");
    Vector<String> paths = path.split("/");

    String curr_path = "res://";
    for(int i = 0; i < paths.size() - 1; i++) {
        curr_path += paths[i];
        if(!DirAccess::exists(curr_path)) {
            DirAccess::make_dir_absolute(curr_path);
        }
        curr_path += "/";
    }

}




static bool install_mkhm_zip(const String& p_zip_file, const String& p_save_path) {
    Ref<ZIPReader> zip_reader = memnew(ZIPReader);
    Error err = zip_reader->open(p_zip_file);
    if(err == OK) {
        Vector<String> files = zip_reader->get_files();
        HashMap<String,String> body_part_set = {        
            {"body","Body"},
            {"righteye","RightEye"},
            {"lefteye","LeftEye"},
            {"righteyebrow","RightEyebrow"},
            {"leftryebrow","LeftEyebrow"},
            {"righteyelash","RightEyelash"},
            {"lefteyelash","LeftEyelash"},
            {"hair","Hair"},
            {"tongue","Tongue"},
            {"teeth","Teeth"}
        };
        HashMap<String,Vector<String>> body_part_file;

        Vector<String> clothing ;
        Vector<String> targets ;

        for(int i = 0; i < files.size(); i++) {
            String file = files[i];
            file = file.replace("\\","/");
            Vector<String> path = file.split("/");
            if(path.size() > 0) {
                String name = path[0].to_lower();
                if(body_part_set.has(name)) {
                    body_part_file[body_part_set[name]].push_back(file);
                    continue;
                }
                if(name == "clothes") {
                    clothing.push_back(file);
                    continue;
                }
                if(name == "targets") {
                    targets.push_back(file);
                    continue;
                }

            }
        }

        String save_path = "res://Assets/public/mkhm/";
        make_res_path(save_path);

        String pack_name = p_zip_file.get_file();
        pack_name = pack_name.get_basename();

        save_path = save_path.path_join(pack_name);
        if (!DirAccess::exists(save_path))
        {
            DirAccess::make_dir_absolute(save_path);
        }
        String body_parts_save_path = save_path.path_join("body_parts");
        if (!DirAccess::exists(body_parts_save_path))
        {
            DirAccess::make_dir_absolute(body_parts_save_path);
        }
        bool result = false;
        // 保存身体部位
        for(auto& it : body_part_file) {
            Vector<String>& save_file_list = it.value;
            for(int i = 0; i < save_file_list.size(); i++) {
                String file = save_file_list[i];
                file = file.replace("\\","/");
                Vector<String> path_list = file.split("/");
                if(path_list.size() > 1) {
                    String path = body_parts_save_path.path_join(it.key);
                    for(int j = 1; j < path_list.size(); j++) {
                        path = path.path_join(path_list[j]);
                    }
                    auto buffer =  zip_reader->read_file(save_file_list[i],false);

                    if(buffer.size() > 0) {
                        make_res_path_form_filepath(path);
                        Ref<FileAccess> f = FileAccess::open(path, FileAccess::WRITE);
                        f->store_buffer(buffer);
                        f->close();
                        result = true;
                    }
                }
            }

        }

        // 保存衣服
        for(int i = 0; i < clothing.size(); i++) {
            String file = clothing[i];
            String path = save_path.path_join(file);
            auto buffer =  zip_reader->read_file(clothing[i],false);
            if(buffer.size() > 0) {
                make_res_path_form_filepath(path);
				Ref<FileAccess> f = FileAccess::open(path, FileAccess::WRITE);
                f->store_buffer(buffer);
                f->close();
                    result = true;
            }
        }

        // 保存目标
        for(int i = 0; i < targets.size(); i++) {
            String file = targets[i];
			String path = save_path.path_join(file);
            auto buffer =  zip_reader->read_file(targets[i],false);
            if(buffer.size() > 0) {
                make_res_path_form_filepath(path);
				Ref<FileAccess> f = FileAccess::open(path, FileAccess::WRITE);
                f->store_buffer(buffer);
                f->close();
                result = true;
            }
        }
        return result;


    }
    return false;
}
// 初始化所有的mkhm的包
static void init_all_mkhm_pack() {
    String root_path = "res://Assets/mkhm_pack/";
    if (!DirAccess::exists(root_path)){
        return;
    }

    Ref<JSON> json = memnew(JSON);
    Vector<String> files = DirAccess::get_files_at(root_path);
    for(int i = 0; i < files.size(); i++) {
        String file = files[i];
        if(file.get_extension() == "zip") {
            if(install_mkhm_zip(root_path.path_join(file),root_path))
            {
                String pack_name = file.get_file();
                pack_name = pack_name.get_basename();
                String path_path = "res://Assets/public/mkhm/" + pack_name;
                json->set(pack_name, path_path);
                print_line("初始化mkhm包:" + pack_name + "路径:" + path_path);
            }
        }
    }
    // 保存配置文件
    String config_path = root_path.path_join("config.json");
    Ref<FileAccess> f = FileAccess::open(config_path, FileAccess::WRITE);
    f->store_string(json->get_parsed_text());
    f->close();
}
void CharacterBodyMain::humanizer_install_mkhm() {
    init_all_mkhm_pack();
}



