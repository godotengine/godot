#include "basic_scheduler.h"

SynchronizationPoint* SynchronizationPoint::singleton = nullptr;
BasicScheduler* BasicScheduler::singleton = nullptr;

SynchronizationPoint::SynchronizationPoint(){
	ERR_FAIL_COND(singleton);
	singleton = this;
	// prime_numbers.resize(PRIME_NUMBERS_LIMIT);
	// prime_numbers[0] = 0;
	// prime_numbers[1] = 2;
	// prime_numbers[2] = 3;
	// uint32_t iter = 3, sampling = 4;
	// while (iter < PRIME_NUMBERS_LIMIT){
	// 	if (is_prime(sampling)) prime_numbers[iter++] = sampling;
	// 	sampling++;
	// }
}
SynchronizationPoint::~SynchronizationPoint(){
	singleton = nullptr;
}

void SynchronizationPoint::iter_sync(const real_t& delta){
	if (delta > 0.0)
		iter_delta = delta;
	iter_start_at = OS::get_singleton()->get_ticks_usec();
	iter_end_at = iter_start_at + (uint64_t)(delta * 1000 * 1000);
}

ProcessTask::ProcessTask(const String &task_name, const StringName& func_call, const bool& autofree){
	this->task_name = task_name;
	this->func_call = func_call;
	this->autofree = autofree;
}
ProcessTask::~ProcessTask(){
	kill_task();
}
void ProcessTask::job(const uint32_t& id){
	uint64_t time_now = 0;
	while (!killsig){
		time_now = OS::get_singleton()->get_ticks_usec();
		if (!(time_now >= SynchronizationPoint::get_singleton()->get_iter_start() &&
			time_now < SynchronizationPoint::get_singleton()->get_iter_end()))
				continue;
		uint32_t iter = id;
		while (iter < id_list.size() && !killsig && task_mutex.try_lock() == OK){
			auto obj_id = id_list[iter];
			task_mutex.unlock();
			auto obj = ObjectDB::get_instance(obj_id);
			ScriptInstance *script = nullptr;
			if (obj) {
				auto va_delta = Variant(SynchronizationPoint::get_singleton()->get_iter_delta());
				// const Variant *arg[1] = { &va_delta };
				// auto err = Variant::CallError::CALL_OK;
				script = obj->get_script_instance();
				if (script) {
					script->call(func_call, va_delta);
				} else obj->call(func_call, va_delta);
				// CRASH_COND_MSG(err != Variant::CallError::CALL_OK, (String("Call failed: ") + itos((uint64_t)err)).utf8().ptr());
			}
			iter += job_interval;
		}
	}
}
void ProcessTask::kill_task(){
	killsig = true;
	MutexLock guard(task_mutex);
	for (uint32_t i = 0; i < thread_list.size(); i++){
		thread_list.write[i]->join();
		delete thread_list.write[i];
	}
	thread_list.resize(0);
	id_list.resize(0);
}
void ProcessTask::boot(const uint32_t& thread_count){
	MutexLock guard(task_mutex);
	if (thread_list.size() > 0 || thread_count == 0) return;
	CRASH_BAD_UNSIGNED_INDEX(thread_count, MAX_THREAD_PER_TASK);
	job_interval = thread_count;
	thread_list.resize(thread_count);
	for (uint32_t i = 0; i < thread_count; i++){
		thread_list.write[i] = new std::thread(&ProcessTask::job, this, i);
	}
}
void ProcessTask::add_handle(const ObjectID& id){
	ERR_FAIL_COND(killsig);
	MutexLock guard(task_mutex);
	id_list.push_back(id);
}
bool ProcessTask::has_handle(const ObjectID& id){
	ERR_FAIL_COND_V(killsig, false);
	MutexLock guard(task_mutex);
	return (id_list.find(id) != -1);
}
void ProcessTask::remove_handle(const ObjectID& id){
	ERR_FAIL_COND(killsig);
	MutexLock guard(task_mutex);
	id_list.erase(id);
}
BasicScheduler::BasicScheduler(){
	ERR_FAIL_COND(singleton);
	singleton = this;
}
BasicScheduler::~BasicScheduler(){
	free_all();
}
void BasicScheduler::_bind_methods(){
	ClassDB::bind_method(D_METHOD("add_task", "task_name", "func_call", "is_physics_process", "thread_count", "autofree"), &BasicScheduler::add_task);
	ClassDB::bind_method(D_METHOD("has_task", "task_name"), &BasicScheduler::has_task);
	ClassDB::bind_method(D_METHOD("remove_task", "task_name"), &BasicScheduler::remove_task);

	ClassDB::bind_method(D_METHOD("add_handle", "task_name", "obj"), &BasicScheduler::add_handle);
	ClassDB::bind_method(D_METHOD("has_handle", "task_name", "obj"), &BasicScheduler::has_handle);
	ClassDB::bind_method(D_METHOD("remove_handle", "task_name", "obj"), &BasicScheduler::remove_handle);
}
void BasicScheduler::free_all(){
	while (!task_list.empty()) {
		remove_task(*task_list.next(nullptr));
	}
	singleton = nullptr;
}
bool BasicScheduler::add_task(const String &task_name, const StringName& func_call, const bool& is_physics_process, const uint32_t& thread_count, const bool& autofree){
	MutexLock guard(main_mutex);
	if (task_list.has(task_name))
		return false;
	auto new_task = new ProcessTask(task_name, func_call, autofree);
	new_task->boot(thread_count);
	task_list[task_name] = new_task;
	return true;
}
bool BasicScheduler::has_task(const String& task_name){
	MutexLock guard(main_mutex);
	return (task_list.has(task_name));
}
bool BasicScheduler::remove_task(const String& task_name){
	MutexLock guard(main_mutex);
	auto task = task_list.getptr(task_name);
	if (!task) return false;
	else {
		task_list.erase(task_name);
		delete (*task);
	}
	return true;
}

void BasicScheduler::add_handle(const String &task_name, Object* obj){
	ERR_FAIL_COND(!obj);
	MutexLock guard(main_mutex);
	auto task = task_list.getptr(task_name);
	ERR_FAIL_COND(!task);
	(*task)->add_handle(obj->get_instance_id());
}
bool BasicScheduler::has_handle(const String &task_name, Object* obj){
	ERR_FAIL_COND_V(!obj, false);
	MutexLock guard(main_mutex);
	auto task = task_list.getptr(task_name);
	ERR_FAIL_COND_V(!task, false);
	return (*task)->has_handle(obj->get_instance_id());
}
void BasicScheduler::remove_handle(const String &task_name, Object* obj){
	ERR_FAIL_COND(!obj);
	main_mutex.lock();
	auto task = task_list.getptr(task_name);
	ERR_FAIL_COND(!task);
	auto real_task = *task;
	auto real_id = obj->get_instance_id();
	real_task->remove_handle(real_id);
	main_mutex.unlock();
	if ((*task)->is_removable())
		remove_task(task_name);
}