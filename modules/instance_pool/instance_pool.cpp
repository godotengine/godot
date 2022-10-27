#include "instance_pool.h"

InstancePool *InstancePool::singleton = nullptr;

InstancePool::InstancePool(){
	singleton = this;
}

InstancePool::~InstancePool(){
	worker_swarm.resize(0);
}

void InstancePool::_bind_methods(){
	ClassDB::bind_method(D_METHOD("queue_job", "function", "parameters"), &InstancePool::queue_job);
	ClassDB::bind_method(D_METHOD("create_work_pool", "pool_name", "auto_free"), &InstancePool::create_work_pool);
	ClassDB::bind_method(D_METHOD("get_work_pool", "object_id"), &InstancePool::get_work_pool);
	ClassDB::bind_method(D_METHOD("get_work_pool_by_name", "pool_name"), &InstancePool::get_work_pool_by_name);
	ClassDB::bind_method(D_METHOD("erase_work_pool", "object_id"), &InstancePool::erase_work_pool);
	ClassDB::bind_method(D_METHOD("erase_work_pool_by_name", "pool_name"), &InstancePool::erase_work_pool_by_name);
	ClassDB::bind_method(D_METHOD("get_pool_size"), &InstancePool::get_pool_size);
}

void InstancePool::dispatch_idle(const float& delta){
	auto size = worker_swarm.size();
	if (!size) return;
	workers_mutex.lock();
	for (int i = 0; i < size; i++){
		auto pool = worker_swarm[i];
		pool->dispatch();
		if (pool->is_autofree() && !pool->get_worker_status()){
			worker_swarm.remove(i);
			i--;
			size--;
		}
	}
	workers_mutex.unlock();
}

Ref<Future> InstancePool::queue_job(const Ref<FuncRef>& function, const Array& args){
	auto pool = create_work_pool("__temp_pool", true);
	if (pool->get_pool_name() != StringName("__temp_pool")) return Ref<Future>();
	auto future = pool->queue_job(function, args);
	return future;
}

Ref<WorkPool> InstancePool::create_work_pool(const StringName& pool_name, const bool& auto_free){
	Ref<WorkPool> work_pool = memnew(WorkPool());
	auto obj_id = work_pool->get_instance_id();
	ERR_FAIL_COND_V(obj_id == 0, work_pool);
	work_pool->set_autofree(auto_free);
	work_pool->set_pool_name(pool_name);
	workers_mutex.lock();
	worker_swarm.push_back(work_pool);
	workers_mutex.unlock();
	return work_pool;
}

Ref<WorkPool> InstancePool::get_work_pool(const uint64_t& object_id) const{
	for (int i = 0; i < worker_swarm.size(); i++){
		auto pool = worker_swarm[i];
		if (pool->get_instance_id() == object_id) return pool;
	}
	return Ref<WorkPool>();
}

bool InstancePool::erase_work_pool(const uint64_t& object_id){
	bool success = false;
	workers_mutex.lock();
	for (int i = 0; i < worker_swarm.size(); i++){
		auto pool = worker_swarm[i];
		if (pool->get_instance_id() == object_id) {
			worker_swarm.remove(i);
			success = true;
			break;
		}
	}
	workers_mutex.unlock();
	return success;
}

Ref<WorkPool> InstancePool::get_work_pool_by_name(const StringName& pool_name) const{
	for (int i = 0; i < worker_swarm.size(); i++){
		auto pool = worker_swarm[i];
		if (pool->get_pool_name() == pool_name) return pool;
	}
	return Ref<WorkPool>();
}
bool InstancePool::erase_work_pool_by_name(const StringName& pool_name){
		bool success = false;
	workers_mutex.lock();
	for (int i = 0; i < worker_swarm.size(); i++){
		auto pool = worker_swarm[i];
		if (pool->get_pool_name() == pool_name) {
			worker_swarm.remove(i);
			success = true;
			break;
		}
	}
	workers_mutex.unlock();
	return success;
}
