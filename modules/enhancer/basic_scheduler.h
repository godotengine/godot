#ifndef BASIC_SCHEDULER_H
#define BASIC_SCHEDULER_H

#include <thread>

#include "core/os/os.h"
#include "core/os/mutex.h"
#include "core/object.h"
#include "core/engine.h"
#include "core/hash_map.h"
#include "core/vector.h"
#include "core/variant.h"
#include "core/script_language.h"
// #include "core/math/math_defs.h"

// #define PRIME_NUMBERS_LIMIT 32
#define MAX_THREAD_PER_TASK 16

class SynchronizationPoint {
private:
	uint64_t iter_start_at = 0;
	uint64_t iter_end_at   = 0;

	// Vector<uint32_t> prime_numbers;

	real_t iter_delta = 0.0F;

	static SynchronizationPoint* singleton;
	void iter_sync(const real_t& delta);
	// static bool is_prime(const uint32_t& num){
	// 	auto threshold = (uint32_t)Math::sqrt((real_t)num);
	// 	for (uint32_t i = 2; i <= threshold; i++){
	// 		if (num % threshold == 0) return false;
	// 	}
	// 	return true;
	// }

	friend class SceneTreeHook;
public:
	SynchronizationPoint();
	~SynchronizationPoint();

	static _FORCE_INLINE_ SynchronizationPoint* get_singleton() { return singleton; }

	_FORCE_INLINE_ void _set_iter_delta(const real_t& new_delta) { iter_delta = new_delta; }

	_FORCE_INLINE_ uint64_t get_iter_start()  const { return iter_start_at; }
	_FORCE_INLINE_ uint64_t get_iter_end()    const { return iter_end_at; }
	_FORCE_INLINE_ real_t get_iter_delta() const { return iter_delta; }
	// _FORCE_INLINE_ const Vector<uint32_t>& get_prime_numbers() const {
	// 	return prime_numbers;
	// }
};

class ProcessTask {
private:
	String task_name;
	StringName func_call;
	Mutex task_mutex;
	bool killsig = false;
	bool autofree = false;
	uint32_t job_interval = 1;

	bool is_physics_process = true;
	Vector<ObjectID> id_list;
	Vector<std::thread*> thread_list;

	void job(const uint32_t& id);
	void kill_task();
public:
	ProcessTask() = delete;
	ProcessTask(const String &task_name, const StringName& func_call, const bool& autofree = false);
	~ProcessTask();

	void boot(const uint32_t& thread_count);

	void add_handle(const ObjectID& id);
	bool has_handle(const ObjectID& id);
	void remove_handle(const ObjectID& id);

	_FORCE_INLINE_ bool is_removable() const { return (id_list.size() == 0 && autofree); }
};
class BasicScheduler : public Object {
	GDCLASS(BasicScheduler, Object);
private:
	Mutex main_mutex;
	HashMap<String, ProcessTask*> task_list;

	void free_all();
protected:
	static void _bind_methods();
	static BasicScheduler* singleton;
public:
	BasicScheduler();
	~BasicScheduler();

	static _FORCE_INLINE_ BasicScheduler* get_singleton() { return singleton; }

	bool add_task(const String &task_name, const StringName& func_call, const bool& is_physics_process, const uint32_t& thread_count, const bool& autofree = false);
	bool has_task(const String& task_name);
	bool remove_task(const String& task_name);

	void add_handle(const String &task_name, Object* obj);
	bool has_handle(const String &task_name, Object* obj);
	void remove_handle(const String &task_name, Object* obj);
};

#endif