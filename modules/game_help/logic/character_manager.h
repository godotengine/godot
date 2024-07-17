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

    void register_character(class CharacterBodyMain* character);
    void unregister_character(class CharacterBodyMain* character);
    void tick(float delta);

    // 更新所有的角色ai
    void update_ai();
    // 更新所有的动画控制
    void update_animator();
    static void _process_animator(void* p_user,uint32_t p_index);
    static void _process_animation(void* p_user,uint32_t p_index);
    static void _process_ik(void* p_user,uint32_t p_index);
    void update_finish();
protected:
    HashSet<class CharacterBodyMain*> characters;
    Ref<TaskJobHandle> task_handle;
    

};