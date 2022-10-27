#ifndef INSTANCE_POOL
#define INSTANCE_POOL

#include <mutex>
#include <cstdint>

#include "core/vector.h"
#include "core/reference.h"
#include "work_pool.h"

class InstancePool : public Object {
	GDCLASS(InstancePool, Object)
private:
	Vector<Ref<WorkPool>> worker_swarm;

	std::mutex workers_mutex;
protected:
	static InstancePool *singleton;
	static void _bind_methods();

	void dispatch_idle(const float& delta);
public:
	InstancePool();
	~InstancePool();

	friend class SceneTreeHook;
	static InstancePool* get_singleton() { return singleton; }
	inline uint32_t get_pool_size() const { return worker_swarm.size(); }

	Ref<Future> queue_job(const Ref<FuncRef>& function, const Array& args);

	Ref<WorkPool> create_work_pool(const StringName& pool_name, const bool& auto_free);

	Ref<WorkPool> get_work_pool_by_name(const StringName& pool_name) const;
	Ref<WorkPool> get_work_pool(const uint64_t& object_id) const;
	bool erase_work_pool_by_name(const StringName& pool_name);
	bool erase_work_pool(const uint64_t& object_id);
};

#endif
