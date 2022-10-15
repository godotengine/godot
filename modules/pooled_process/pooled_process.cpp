#include "pooled_process.h"

#define MAX_WORKER_PER_POOL 3000
#define THREAD_COUNT_DIVISOR 2
#define MAX_EXPECTED_FPS 1000.0
#define MIN_FRAME_DELTA (1.0 / MAX_EXPECTED_FPS)

#define GET_TICKS OS::get_singleton()->get_ticks_msec()
#define GET_TICKS_USEC OS::get_singleton()->get_ticks_usec()

PooledProcess *PooledProcess::singleton = nullptr;

PooledWorker::PooledWorker(){
	type = ThreadType::NA;
}
PooledWorker::PooledWorker(ThreadType thread_type){
	type = thread_type;
	// thread = new std::thread();
}
PooledWorker::~PooledWorker(){
	if (thread) delete thread;
	for (auto obj : handling){
		if (obj) memdelete(obj);
	}
	handling.resize(0);
}
void PooledWorker::add_object(Object* obj){
	handling.push_back(obj);
}
bool PooledWorker::remove_object(const int& obj_id){
	int size = handling.size();
	for (int i = 0; i < size; i++){
		auto obj = handling[i];
		if (!obj) continue;
		if (obj->get_instance_id() == obj_id) {
			handling.erase(handling.begin() + i);
			return true;
		}
	}
	return false;
}
ThreadType PooledWorker::get_thread_type(){
	return type;
}
std::vector<Object*>* PooledWorker::get_handling(){
	return &handling;
}
size_t PooledWorker::get_handling_count(){
	return handling.size();
}

PooledProcess::PooledProcess(){
	singleton = this;
	finished = false;
	max_thread_count = OS::get_singleton()->get_processor_count();
	max_thread_count = clampi(max_thread_count, 1, max_thread_count / THREAD_COUNT_DIVISOR);
	// thread_list.resize(max_thread_count);
	// fref_pool.resize(max_thread_count);
	// for (size_t i = 0; i < max_thread_count; i++){
	// 	auto thread = new std::thread(&PooledProcess::worker, this, i);
	// 	thread_list.push_back(thread);
	// 	args_pool.push_back(Array());
	// 	status_pool.push_back(false);
	// }
}
PooledProcess::~PooledProcess(){
	for (int i = 0; i < current_thread_count; i++){
		auto thread = thread_list[i];
		auto mutex = threads_mutex[i];
		if (thread) delete thread;
		if (mutex) delete mutex;
	}
	thread_list.reserve(0);
	threads_mutex.resize(0);
	status_pool.resize(0);
}

void PooledProcess::_bind_methods(){
	// BIND_ENUM_CONSTANT(IDLE);
	// BIND_ENUM_CONSTANT(PHYSICS);

	ClassDB::bind_method(D_METHOD("create_new_thread", "is_physics"), &PooledProcess::create_new_thread);
	ClassDB::bind_method(D_METHOD("get_thread_count"), &PooledProcess::get_thread_count);
	ClassDB::bind_method(D_METHOD("get_max_thread_count"), &PooledProcess::get_max_thread_count);
	ClassDB::bind_method(D_METHOD("get_thread_count_by_type", "is_physics"), &PooledProcess::get_thread_count_by_type);
	ClassDB::bind_method(D_METHOD("get_delta", "is_physics"), &PooledProcess::get_delta);
	ClassDB::bind_method(D_METHOD("get_handling_instance", "worker_id"), &PooledProcess::get_handling_instance);
	ClassDB::bind_method(D_METHOD("get_physics_step"), &PooledProcess::get_physics_step);
	ClassDB::bind_method(D_METHOD("add_worker_idle", "obj"), &PooledProcess::add_worker_idle);
	ClassDB::bind_method(D_METHOD("add_worker_physics", "obj"), &PooledProcess::add_worker_physics);
	ClassDB::bind_method(D_METHOD("remove_object", "object_id", "is_physics"), &PooledProcess::remove_object);
}

int PooledProcess::clampi(const int& value, const int& from, const int& to){
	if (value < from) return from;
	if (value > to) return to;
	return value;
}

void PooledProcess::notify_idle(const float& delta, const int64_t& current_frame){
	frame_count = current_frame;
	idle_delta = delta;
}
void PooledProcess::notify_iteration(const float& delta){
	physics_call_count++;
	physics_delta = delta;
}
void PooledProcess::remove_thread(const int& id){
	pool_mutex.lock();
	if (id < current_thread_count && id >= 0){
		auto worker = thread_list[id];
		auto mutex = threads_mutex[id];
		if (worker) delete worker;
		if (mutex) delete mutex;
		thread_list.erase(thread_list.begin() + id);
		threads_mutex.erase(threads_mutex.begin() + id);
		status_pool.erase(status_pool.begin() + id);
		current_thread_count--;
	}
	pool_mutex.unlock();
}
bool PooledProcess::add_worker_internal(Object *obj, ThreadType ttype){
	if (!obj){
		Hub::get_singleton()->print_fatal(String("Object is Null"));
		return false;
	}
	int index = -1;
	long min_pool_worker = MAX_WORKER_PER_POOL;
	for (int iter = 0; iter < thread_list.size(); iter++){
		auto worker = thread_list[iter];
		if (!worker) {
			remove_thread(iter);
			iter--;
			continue;
		}
		if (worker->get_thread_type() != ttype) continue;
		auto size = worker->get_handling_count();
		if (size < min_pool_worker) {
			index = iter;
			min_pool_worker = size;
		}
	}
	if (index < 0) return false;
	threads_mutex[index]->lock();
	thread_list[index]->add_object(obj);
	threads_mutex[index]->unlock();
	return true;
}
bool PooledProcess::add_worker_idle(Object* obj){
	return add_worker_internal(obj, ThreadType::IDLE);
}
bool PooledProcess::add_worker_physics(Object* obj){
	return add_worker_internal(obj, ThreadType::PHYSICS);
}
void PooledProcess::idle_worker(const int& id){
	int64_t curr_frame = frame_count;
	while (!finished){
		while ((curr_frame == frame_count) && !finished){
			std::this_thread::sleep_for(std::chrono::seconds(int64_t(MIN_FRAME_DELTA)));
		}
		if (finished) return;
		curr_frame = frame_count;
		threads_mutex[id]->lock();
		auto worker = thread_list[id];
		if (!worker) return;
		auto handling = worker->get_handling();
		for (int i = 0; i < handling->size() && i >= 0; i++){
			auto obj = handling->operator[](i);
			if (!obj){
				handling->erase(handling->begin() + i);
				i--;
				continue;
			}
			//------------------------------------------------------
			// FuncRef fref;
			// fref.set_instance(obj);
			// fref.set_function(StringName("update_idle"));
			// Array args;
			// args.append(Variant(idle_delta));
			// args.append(Variant(frame_count));
			// fref.call_funcv(args);
			//------------------------------------------------------
			auto args = new Variant[0]();
			auto err_ok = Variant::CallError::CALL_OK;
			obj->call(StringName("update_idle"), args, 0, err_ok);
		}
		threads_mutex[id]->unlock();
	}
}
void PooledProcess::physics_worker(const int& id){
	int64_t curr_frame = physics_call_count;
	int64_t steps = 0;
	// Hub::get_singleton()->print_debug(String("Physics worker instantiated"));
	while (!finished){
		while ((curr_frame == physics_call_count) && !finished){
			std::this_thread::sleep_for(std::chrono::seconds(int64_t(MIN_FRAME_DELTA)));
		}
		// if (steps < 10) Hub::get_singleton()->print_debug(String("Physics stepped"));
		if (finished) return;
		curr_frame = physics_call_count;
		threads_mutex[id]->lock();
		auto worker = thread_list[id];
		if (!worker) return;
		auto handling = worker->get_handling();
		for (int i = 0; i < handling->size() && i >= 0; i++){
			// if (steps < 10) Hub::get_singleton()->print_debug(String("Iterating physics step"));
			auto obj = handling->operator[](i);
			if (!obj){
				handling->erase(handling->begin() + i);
				i--;
				continue;
			}
			// if (steps < 10) Hub::get_singleton()->print_debug(String("Calling obj from physics handler"));
			//------------------------------------------------------
			// FuncRef fref;
			// fref.set_instance(obj);
			// fref.set_function(StringName("update_physics"));
			// Array args;
			// args.append(Variant(physics_delta));
			// args.append(Variant(physics_call_count));
			// fref.call_funcv(args);
			//------------------------------------------------------
			auto args = new Variant[0]();
			auto err_ok = Variant::CallError::CALL_OK;
			obj->call(StringName("update_physics"), args, 0, err_ok);
		}
		threads_mutex[id]->unlock();
		steps++;
	}
}
bool PooledProcess::remove_object(const int& obj_id, bool is_physics){
	ThreadType from_pool = is_physics ? ThreadType::PHYSICS : ThreadType::IDLE;
	for (auto worker : thread_list){
		if (!worker) continue;
		if (worker->get_thread_type() != from_pool) continue;
		auto handling = worker->get_handling();
		for (int i = 0; i < handling->size(); i++){
			auto obj = handling->operator[](i);
			if (!obj) continue;
			if (obj->get_instance_id() == obj_id){
				handling->erase(handling->begin() + i);
				return true;
			}
		}
	}
	return false;
}
bool PooledProcess::create_new_thread(bool is_physics){
	ThreadType thread_type = is_physics ? ThreadType::PHYSICS : ThreadType::IDLE;
	pool_mutex.lock();
	if (current_thread_count < max_thread_count){
		auto worker = new PooledWorker(thread_type);
		std::thread *thread = nullptr;
		switch (thread_type){
			case ThreadType::IDLE:
				thread = new std::thread(&PooledProcess::idle_worker, this, current_thread_count);
				break;
			case ThreadType::PHYSICS:
				thread = new std::thread(&PooledProcess::physics_worker, this, current_thread_count);
				break;
			default:
				pool_mutex.unlock();
				delete worker;
				return false;
		}
		worker->thread = thread;
		thread_list.push_back(worker);
		threads_mutex.push_back(new std::mutex());
		status_pool.push_back(true);
		current_thread_count++;
	}
	pool_mutex.unlock();
	return false;
}
int PooledProcess::get_thread_count(){
	return current_thread_count;
}
int PooledProcess::get_max_thread_count(){
	return max_thread_count;
}
int64_t PooledProcess::get_delta(bool is_physics){
	return is_physics ? physics_delta : idle_delta;
}
int PooledProcess::get_thread_count_by_type(bool is_physics){
	ThreadType type = is_physics ? ThreadType::PHYSICS : ThreadType::IDLE;
	int count = 0;
	for (auto worker : thread_list){
		if (!worker) continue;
		if (worker->get_thread_type() == type) count++;
	}
	return count;
}
int PooledProcess::get_handling_instance(int id){
	int count = -1;
	if (id >= 0 && id < current_thread_count){
		auto worker = thread_list[id];
		if (!worker) return count;
		threads_mutex[id]->lock();
		count = worker->get_handling_count();
		threads_mutex[id]->unlock();
	}
	return count;
}
int64_t PooledProcess::get_physics_step(){
	return physics_call_count;
}

void PooledProcess::join(){
	finished = true;
	std::this_thread::sleep_for(std::chrono::milliseconds(int64_t(MIN_FRAME_DELTA * 1.5 * 1000.0)));
	// auto start_time = GET_TICKS;
	// for (auto thread : thread_list){
	// 	if (thread == nullptr) continue;
	// 	auto is_joinable = thread->joinable();
	// 	while (!is_joinable){
	// 		// if (GET_TICKS - start_time > max_wait_time_msec){
	// 		// 	Hub::get_singleton()->print_fatal(String("Pulling out"), Array());
	// 		// 	return;
	// 		// }
	// 		is_joinable = thread->joinable();
	// 	}
	// 	thread->join();
	// 	delete thread;
	// }
	// Hub::get_singleton()->print_warning(String("Joined all worker threads successfully"), Array());
}
