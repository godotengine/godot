#pragma once

#include "body_main.h"
#include "CSV_EditorImportPlugin.h"
#include "core/object/worker_thread_pool.h"

class CharacterManager : public Object
{
    GDCLASS(CharacterManager, Object);
    static void _bind_methods()
    {

    }

    public:

    void register_character(class CharacterBodyMain* character)
    {
        characters.insert(character);
    }
    void unregister_character(class CharacterBodyMain* character)
    {
        characters.erase(character);
    }
    void update(class CharacterBodyMain* character)
    {
        update_ai();
        update_animator();
    }

    // 更新所有的角色ai
    void update_ai()
    {
        for(CharacterBodyMain* character : characters)
        {
            character->update_ai();
            character->_process_move();
        }
    }
    // 更新所有的动画控制
    void update_animator()
    {
        TypedArray<TaskJobHandle> handles;
        handles.resize(characters.size());
        int index = 0;
        WorkerTaskPool * worker_task_pool = WorkerTaskPool::get_singleton();
        for(CharacterBodyMain* character : characters)
        {
            Ref<TaskJobHandle> h = handles[index];
            h = worker_task_pool->add_native_group_task(&_process_animator,character,1,1,h);
            h = worker_task_pool->add_native_group_task(&_process_animation,character,1,1,h);
            h = worker_task_pool->add_native_group_task(&_process_ik,character,1,1,h);
            handles[index] = h;
            index++;
        }
        task_handle = worker_task_pool->combined_job_handle(handles);
    }
    static void _process_animator(CharacterBodyMain* p_user,int p_index)
    {
        p_user->_process_animator();
    }
    static void _process_animation(CharacterBodyMain* p_user,int p_index)
    {
        p_user->_process_animation();
    }
    static void _process_ik(CharacterBodyMain* p_user,int p_index)
    {
        p_user->_process_ik();        
    }
    void update_finish()
    {
        // 等待所有线程结束
        if(task_handle.is_valid())
        {
            task_handle->wait_completion();
            task_handle.unref();
        }
    }

    HashSet<class CharacterBodyMain*> characters;
    Ref<TaskJobHandle> task_handle;
    

};