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

    void register_character(class CharacterBodyMain* character);
    void unregister_character(class CharacterBodyMain* character);

    // 更新所有的角色ai
    void update_ai();
    // 更新所有的动画控制
    void update_animator();
   const String& get_mesh_root_path(){
        return mesh_root_path;
    }
    const String& get_skeleton_root_path(){
        return skeleton_root_path;
    }
    const String get_animation_root_path(){
        return animation_root_path;
    }
    const String get_bone_map_root_path(){
        return bone_map_root_path;
    }
    const String get_prefab_root_path(){
        return prefab_root_path;
    }
    static void _process_animator(void* p_user,uint32_t p_index);
    static void _process_animation(void* p_user,uint32_t p_index);
    static void _process_ik(void* p_user,uint32_t p_index);
    void update_finish();
protected:
    HashSet<class CharacterBodyMain*> characters;
    Ref<TaskJobHandle> task_handle;
    String mesh_root_path = "res://Assets/public/meshs";
    String skeleton_root_path = "res://Assets/public/skeleton";
    String animation_root_path = "res://Assets/public/animation";
    String bone_map_root_path = "res://Assets/public/bone_map";
    String prefab_root_path = "res://Assets/public/prefab";
    

};
