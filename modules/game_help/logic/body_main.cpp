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

void CharacterBodyMain::init(bool p_is_only_mesh)
{
    if(animator.is_null())
    {
        animator.instantiate();
    }
    animator->set_body(this);
    animator->init();

    editor_only_mesh = p_is_only_mesh;
    if(!p_is_only_mesh) {
        if(character_ai.is_null())
        {
            character_ai.instantiate();
        }
        character_ai->init();
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
        if(audio_players.has(StringName("body"))) {
            Ref<AudioStreamPlayer3DCompoent> player;
            player.instantiate();
            player->set_owenr(this);
            audio_players["body"] = player;
        }
        if(audio_players.has(StringName("footstep"))) {
            Ref<AudioStreamPlayer3DCompoent> player;
            player.instantiate();
            player->set_owenr(this);
            audio_players["footstep"] = player;
            
        }    
        if(check_area.size() == 0) {
            Ref<CharacterCheckArea3D> _area;
            _area.instantiate();
            _area->set_name(StringName(L"周围人物检查区域"));
            _area->set_body_main(this);
            _area->init();
            check_area.push_back(_area);
        }
        
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
    _process_move();
    for(uint32_t i = 0; i < check_area.size();++i)
    {
        if(check_area[i].is_valid())
        {
            check_area[i]->update_world_move(get_global_position());
        }
    }

}
void CharacterBodyMain::_update_ai()
{
    if(editor_only_mesh) {
        return;
    }
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
    if(editor_only_mesh) {
        return;
    }
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

    // 处理射线检测
    for(auto& it : raycast) {
        Ref<RayCastCompoent3D> raycast_compoent = it.value;
        if(raycast_compoent.is_valid()) {
            raycast_compoent->force_raycast_update();
        }
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


        if(!editor_only_mesh) {
            if (skeleton)
            {
                ik.instantiate();
                ik->_initialize(skeleton);
            }
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
            //p->set_show_mesh(editor_show_mesh);
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
    ClassDB::bind_method(D_METHOD("get_animation_Tags"), &CharacterBodyMain::get_animation_Tags);

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

    ClassDB::bind_method(D_METHOD("set_raycast", "raycast"), &CharacterBodyMain::set_raycast);
    ClassDB::bind_method(D_METHOD("get_raycast"), &CharacterBodyMain::get_raycast);


    ClassDB::bind_method(D_METHOD("set_body_prefab", "body_prefab"), &CharacterBodyMain::set_body_prefab);
    ClassDB::bind_method(D_METHOD("get_body_prefab"), &CharacterBodyMain::get_body_prefab);


    ClassDB::bind_method(D_METHOD("set_character_ai", "ai"), &CharacterBodyMain::set_character_ai);
    ClassDB::bind_method(D_METHOD("get_character_ai"), &CharacterBodyMain::get_character_ai);

    


	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan", PROPERTY_USAGE_DEFAULT ), "set_blackboard_plan", "get_blackboard_plan");


    ADD_SUBGROUP("audio", "audio_");
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "audio_play_component" ), "set_audio_play_component", "get_audio_play_component");
    IMP_GODOT_PROPERTY(StringName,audio_socket_name)
    ADD_MEMBER_BUTTON(audio_add_socket,L"增加音频插槽",CharacterBodyMain);






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
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "raycast", PROPERTY_HINT_RESOURCE_TYPE, MAKE_RESOURCE_TYPE_HINT("RayCastCompoent3D"),PROPERTY_USAGE_DEFAULT), "set_raycast", "get_raycast");


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
Array CharacterBodyMain::get_animation_Tags() const {
    Array arr;
    CharacterManager::get_singleton()->get_animation_tags(&arr);
    return arr;
    
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



