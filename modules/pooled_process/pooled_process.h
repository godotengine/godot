#ifndef POOLED_PROCESS
#define POOLED_PROCESS

#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <string>
#include <cstdint>

#include "core/os/os.h"
#include "core/object.h"
#include "core/ustring.h"
#include "core/string_name.h"
#include "core/reference.h"
#include "core/array.h"
#include "core/func_ref.h"
#include "core/engine.h"
#include "core/dictionary.h"
#include "modules/hub/hub.h"

enum ThreadType {
	NA,
	CONSISTENT,
	IDLE,
	PHYSICS
};

class PooledWorker {
	private:
		ThreadType type;
		std::vector<Object*> handling;
		std::vector<FuncRef*> fref_pool;
		
	public:
		PooledWorker();
		PooledWorker(ThreadType thread_type);
		~PooledWorker();

		void add_object(Object* obj);
		void remove_object_at(const int& index);
		bool remove_object(const int& obj_id);
		ThreadType get_thread_type();
		std::vector<Object*>* get_handling();
		std::vector<FuncRef*>* get_funcref_pool();
		size_t get_handling_count();

		std::thread *thread;
};

class PooledProcess : public Object {
	GDCLASS(PooledProcess, Object);
private:

	int max_wait_time_msec = 3000;
	int max_thread_count = 0;
	int current_thread_count = 0;
	std::vector<PooledWorker*> thread_list;
	std::vector<std::mutex*>  threads_mutex;
	std::vector<bool> status_pool;
	std::mutex pool_mutex;
	bool finished;

	int64_t frame_count = 0;
	int64_t physics_call_count = 0;
	float idle_delta = 0.0;
	float physics_delta = 0.0;

	int clampi(const int& value, const int& from, const int& to);
	void consistent_worker(const int& id);
	void worker_internal(const int &id, ThreadType type);
	void idle_worker(const int& id);
	void physics_worker(const int& id);
	bool add_worker_internal(Object *obj, ThreadType ttype);
protected:
	static PooledProcess* singleton;
	static void _bind_methods();
public:
	PooledProcess();
	~PooledProcess();
	static PooledProcess* get_singleton() { return singleton; }

	void join();
	void notify_idle(const float& delta, const int64_t& current_frame);
	void notify_iteration(const float& delta);

	bool add_worker_consistent(Object* obj, const Array& args);
	bool add_worker_idle(Object* obj);
	bool add_worker_physics(Object* obj);
	bool remove_object(const int& obj_id, bool is_physics);
	void remove_thread(const int& id);

	int create_new_thread(bool is_physics);
	int get_thread_count();
	int get_max_thread_count();
	int get_thread_count_by_type(bool is_physics);
	float get_delta(bool is_physics);
	int get_handling_instance(int id);
	int64_t get_physics_step();

	friend class std::thread;
};

#endif