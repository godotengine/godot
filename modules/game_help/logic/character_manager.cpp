#include "body_main.h"
#include "character_manager.h"
CharacterManager* CharacterManager::singleton = nullptr;
void CharacterManager::register_character(class CharacterBodyMain* character)
{
    characters.insert(character);
}
void CharacterManager::unregister_character(class CharacterBodyMain* character)
{
    update_finish();
    characters.erase(character);
}
void CharacterManager::tick()
{
    update_ai();
    update_animator();
}

// 更新所有的角色ai
void CharacterManager::update_ai()
{
    for(CharacterBodyMain* character : characters)
    {
        character->_update_ai();
        character->_process_move();
    }
}
// 更新所有的动画控制
void CharacterManager::update_animator()
{
    TypedArray<TaskJobHandle> handles;
    handles.resize(characters.size());
    int index = 0;
    WorkerTaskPool * worker_task_pool = WorkerTaskPool::get_singleton();
    for(CharacterBodyMain* character : characters)
    {
        Ref<TaskJobHandle> h = handles[index];
        h = worker_task_pool->add_native_group_task(&_process_animator,character,1,1,h.ptr());
        h = worker_task_pool->add_native_group_task(&_process_animation,character,1,1,h.ptr());
        h = worker_task_pool->add_native_group_task(&_process_ik,character,1,1,h.ptr());
        handles[index] = h;
        index++;
    }
    task_handle = worker_task_pool->combined_job_handle(handles);
}
void CharacterManager::_process_animator(void* p_user,uint32_t p_index)
{
    CharacterBodyMain* body_main = (CharacterBodyMain*)p_user;
    body_main->_process_animator();
}
void CharacterManager::_process_animation(void* p_user,uint32_t p_index)
{
    CharacterBodyMain* body_main = (CharacterBodyMain*)p_user;
    body_main->_process_animation();
}
void CharacterManager::_process_ik(void* p_user,uint32_t p_index)
{
    CharacterBodyMain* body_main = (CharacterBodyMain*)p_user;
    body_main->_process_ik();        
}
void CharacterManager::update_finish()
{
    // 等待所有线程结束
    if(task_handle.is_valid())
    {
        task_handle->wait_completion();
        task_handle.unref();
    }
}