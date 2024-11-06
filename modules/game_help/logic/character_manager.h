#pragma once

#include "core/object/worker_thread_pool.h"
#include "scene/main/scene_tree.h"

class CharacterManager : public GlolaleTicker
{
    GDCLASS(CharacterManager, Object);
    static void _bind_methods()
    {

    }
public:
    static CharacterManager* singleton;
    static CharacterManager* get_singleton()
    {
        return singleton;
    }
public:
    void pre_tick(float delta) override;
    void tick(float delta) override;

    void post_tick(float delta) override;
public:
    void set_main_player(class CharacterBodyMain* character);
    class CharacterBodyMain* get_main_player();

    void register_character(class CharacterBodyMain* character);
    void unregister_character(class CharacterBodyMain* character);

    // 更新所有的角色ai
    void update_ai();
    // 更新所有的动画控制
    void update_animator();

   const String& get_mesh_root_path(bool is_human){
        if(is_human)
            return human_mesh_root_path;
        return mesh_root_path;
    }
    const String& get_skeleton_root_path(bool is_human){
        if(is_human) {
            return human_skeleton_root_path;
        }
        return skeleton_root_path;
    }
    const String get_animation_root_path(bool is_human){
        if(is_human)
            return human_animation_root_path;
        return animation_root_path;
    }
    const String get_bone_map_root_path(bool is_human){
        if(is_human) {
            return human_bone_map_root_path;
        }
        return bone_map_root_path;
    }
    const String get_prefab_root_path(bool is_human){
        if(is_human)
            return human_prefab_root_path;
        return prefab_root_path;
    }
    static void _process_animator(void* p_user,uint32_t p_index);
    static void _process_animation(void* p_user,uint32_t p_index);
    static void _process_ik(void* p_user,uint32_t p_index);
    void update_finish();
    
    void get_animation_groups(Array *arr)
    {
        for(const StringName& group : animation_groups)
        {
            arr->push_back(group.str());
        }
    }
    CharacterManager();
    ~CharacterManager();
protected:
    HashSet<class CharacterBodyMain*> characters;
    HashSet<StringName> animation_groups;
    ObjectID main_player_id;
    Ref<TaskJobHandle> task_handle;
    String mesh_root_path = "res://Assets/public/meshs";
    String skeleton_root_path = "res://Assets/public/skeleton";
    String animation_root_path = "res://Assets/public/animation";
    String bone_map_root_path = "res://Assets/public/bone_map";
    String prefab_root_path = "res://Assets/public/prefab";
    
    // 人形的配置
    String human_mesh_root_path = "res://Assets/public/human_meshs";
    String human_skeleton_root_path = "res://Assets/public/human_skeleton";
    String human_animation_root_path = "res://Assets/public/human_animation";
    String human_bone_map_root_path = "res://Assets/public/human_bone_map";
    String human_prefab_root_path = "res://Assets/public/human_prefab";

	double last_time = 0;
    

};
