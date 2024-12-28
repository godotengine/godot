#include "body_main.h"
#include "character_manager.h"
#include "scene/resources/animation.h"
#include "core/message_manager.h"

CharacterManager* CharacterManager::singleton = nullptr;
static float time_delta = 0.0f;

void CharacterManager::set_main_player(class CharacterBodyMain* character) {
    main_player_id = ObjectID();
    if(character != nullptr) {
        main_player_id = character->get_instance_id();
    }

}
class CharacterBodyMain* CharacterManager::get_main_player() {
    return Object::cast_to<CharacterBodyMain>(ObjectDB::get_instance(main_player_id));    
}

void CharacterManager::register_character(class CharacterBodyMain* character)
{
    characters.insert(character);
}
void CharacterManager::unregister_character(class CharacterBodyMain* character)
{
    if(main_player_id == character->get_instance_id()) {
        main_player_id = ObjectID();
    }
    update_finish();
    characters.erase(character);
}
void CharacterManager::pre_tick(float delta) {

	double curr_time = OS::get_singleton()->get_unix_time();
	time_delta = MIN(0.1, curr_time - last_time);
	last_time = curr_time;
	

}
void CharacterManager::tick(float delta)
{
    if(MessageManager::get_singleton()) {
        MessageManager::get_singleton()->process();
    }

    update_ai();

    update_animator();
}

void CharacterManager::post_tick(float delta) {
    update_finish();
    for(CharacterBodyMain* character : characters)
    {
        _process_ik(character,0);
    }
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
    update_finish();
    TypedArray<TaskJobHandle> handles;
    handles.resize(characters.size());
    int index = 0;
    WorkerTaskPool * worker_task_pool = WorkerTaskPool::get_singleton();
    for(CharacterBodyMain* character : characters)
    {
        Ref<TaskJobHandle> h = handles[index];
        h = worker_task_pool->add_native_group_task(&_process_animator,character,1,1,h.ptr());
        h = worker_task_pool->add_native_group_task(&_process_animation,character,1,1,h.ptr());
        handles[index] = h;
        index++;
    }
    task_handle = worker_task_pool->combined_job_handle(handles);
}
void CharacterManager::_process_animator(void* p_user,uint32_t p_index)
{
    CharacterBodyMain* body_main = (CharacterBodyMain*)p_user;
    body_main->_process_animator(time_delta);
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

static void animation_get_groups(Array* arr) {
    CharacterManager::get_singleton()->get_animation_groups(arr);
}
static void animation_get_tags(Array* arr) {
    CharacterManager::get_singleton()->get_animation_tags(arr);
}
CharacterManager::CharacterManager() {
    animation_groups.insert(StringName(L"未定义"));
    animation_groups.insert(StringName(L"人形"));
    animation_groups.insert(StringName(L"人形"));
    animation_groups.insert(StringName(L"马"));
    animation_groups.insert(StringName(L"龙"));
    animation_groups.insert(StringName(L"狗"));

    animation_tags.insert(StringName(L"走路"));
    animation_tags.insert(StringName(L"休闲"));
    animation_tags.insert(StringName(L"跑步"));

    
    animation_tags.insert(StringName(L"醉.走路"));
    animation_tags.insert(StringName(L"醉.休闲"));
    animation_tags.insert(StringName(L"醉.跑步"));

    
    animation_tags.insert(StringName(L"僵尸.走路"));
    animation_tags.insert(StringName(L"僵尸.休闲"));
    animation_tags.insert(StringName(L"僵尸.跑步"));

    
    animation_tags.insert(StringName(L"跳跃"));

    
    animation_tags.insert(StringName(L"左手.攻击"));
    animation_tags.insert(StringName(L"右手.攻击"));
    animation_tags.insert(StringName(L"双手.攻击"));

    
    animation_tags.insert(StringName(L"受击"));


    animation_tags.insert(StringName(L"舞蹈"));

    Animation::set_pf_get_animation_group_names(animation_get_groups);
    Animation::set_pf_get_animation_tags(animation_get_tags);

}
CharacterManager::~CharacterManager() {
    Animation::set_pf_get_animation_group_names(nullptr);
    Animation::set_pf_get_animation_tags(nullptr);
    
}
