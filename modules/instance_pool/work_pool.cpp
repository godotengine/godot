#include "work_pool.h"

WorkPool::WorkPool(){
	queued_count = 0;
	auto_free = false;
	cease_and_dessits = false;
	is_worker_active = false;
}

WorkPool::~WorkPool(){
	cease_and_dessits = true;
	queue_mutex.lock();
	while (wait_queue.size() > 0){
		auto front = wait_queue.front();
		if (front) delete front;
		wait_queue.pop();
	}
	queue_mutex.unlock();
	jobs_mutex.lock();
	while (job_queue.size() > 0){
		auto front = job_queue.front();
		if (front) delete front;
		job_queue.pop();
	}
	jobs_mutex.unlock();
}

void WorkPool::_bind_methods(){
	ClassDB::bind_method(D_METHOD("get_job_count"), &WorkPool::get_job_count);
	ClassDB::bind_method(D_METHOD("get_queued_count"), &WorkPool::get_queued_count);
	ClassDB::bind_method(D_METHOD("is_worker_active"), &WorkPool::get_worker_status);
	ClassDB::bind_method(D_METHOD("dispatch", "delta"), &WorkPool::dispatch);

	ClassDB::bind_method(D_METHOD("set_autofree", "autofree"), &WorkPool::set_autofree);
	ClassDB::bind_method(D_METHOD("is_autofree"), &WorkPool::is_autofree);

	ClassDB::bind_method(D_METHOD("set_pool_name", "new_name"), &WorkPool::set_pool_name);
	ClassDB::bind_method(D_METHOD("get_pool_name"), &WorkPool::get_pool_name);

	ClassDB::bind_method(D_METHOD("queue_job", "function", "parameters"), &WorkPool::queue_job);


	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_free"), "set_autofree", "is_autofree");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "pool_name"), "set_pool_name", "get_pool_name");
}

void WorkPool::dispatch(const float& delta){
	// If there are still jobs in the queue, skip this call
	// Else spawn another thread
	queue_mutex.lock();
	while (wait_queue.size() > 0){
		jobs_mutex.lock();
		job_queue.push(wait_queue.front());
		jobs_mutex.unlock();
		wait_queue.pop();
	}
	queue_mutex.unlock();
	if (!is_worker_active && job_queue.size() > 0) {
		is_worker_active = true;
		std::thread thread(&WorkPool::worker, this);
		thread.detach();
	}
}

void WorkPool::worker(){
	while (!cease_and_dessits && job_queue.size() > 0){
		jobs_mutex.lock();
		auto front = job_queue.front();
		job_queue.pop();
		jobs_mutex.unlock();
		if (!front) continue;
		auto res = front->func_ref->call_funcv(front->parameters);
		auto obj = ObjectDB::get_instance(front->object_id);
		if (obj){
			Ref<Future> future = obj;
			future->swap_with(res);
		}
		delete front;
	}
	is_worker_active = false;
}

Ref<Future> WorkPool::queue_job(const Ref<FuncRef>& function, const Array& args){
	Ref<Future> this_future = memnew(Future());
	if (queued_count && auto_free && is_worker_active) return this_future;
	auto obj_id = this_future->get_instance_id();
	ERR_FAIL_COND_V(obj_id == 0, this_future);
	this_future->legit = true;
	auto job = new JobTicket(function, args, obj_id);
	queue_mutex.lock();
	wait_queue.push(job);
	queued_count++;
	queue_mutex.unlock();
	return this_future;
}
