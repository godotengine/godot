#ifndef WORK_POOL_H
#define WORK_POOL_H

#include <queue>
#include <mutex>
#include <thread>
#include <cstdint>

#include "core/string_name.h"
#include "core/reference.h"
#include "core/func_ref.h"
#include "core/array.h"
#include "future.h"

struct JobTicket {
	Ref<FuncRef> func_ref;
	Array parameters;
	uint64_t object_id;
	JobTicket(const Ref<FuncRef>& fref, const Array& args, const uint64_t& id) { func_ref = fref; parameters = args; object_id = id; }
};

class WorkPool : public Reference {
	GDCLASS(WorkPool, Reference)
private:
	std::queue<JobTicket*> job_queue;
	std::queue<JobTicket*> wait_queue;

	StringName pool_name;
	bool auto_free;

	bool queued_count;
	bool cease_and_dessits;
	bool is_worker_active;
	std::mutex queue_mutex;
	std::mutex jobs_mutex;
protected:
	static void _bind_methods();

	void worker();
public:
	WorkPool();
	~WorkPool();

	friend class std::thread;
	void dispatch(const float& delta = 0.0);

	inline uint32_t get_job_count() const { return job_queue.size(); }
	inline uint32_t get_queued_count() const { return wait_queue.size(); }
	inline bool get_worker_status() const { return is_worker_active; }

	inline void set_autofree(const bool& status) { auto_free = status; }
	inline bool is_autofree() const { return auto_free; }

	inline void set_pool_name(const StringName& new_name) { pool_name = new_name; }
	inline StringName get_pool_name() const { return pool_name; }

	Ref<Future> queue_job(const Ref<FuncRef>& function, const Array& args);
};

#endif
