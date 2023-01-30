#ifndef COMMAND_QUEUE_H
#define COMMAND_QUEUE_H

#include <cstdlib>

#include "core/command_queue_mt.h"
#include "core/os/thread.h"
#include "core/safe_refcount.h"
#include "core/reference.h"
#include "core/string_name.h"
#include "core/print_string.h"
#include "core/os/mutex.h"
#include "core/os/rw_lock.h"
#include "core/hash_map.h"

class ExecutionLoop : public Reference {
	GDCLASS(ExecutionLoop, Reference);
private:
	mutable CommandQueueMT command_queue;
	// mutable Mutex dispatch_lock;

	REF object_ref{};
	Object* real_object{};
	SafeFlag thread_started{};
	SafeFlag exit{};
	Thread::ID server_id{};
	Thread server_thread{};

	void remove_virtual_register(int* registers) const;

	static void _thread_callback(void * _instance);
	void thread_start();
	void thread_loop();
	uint32_t thread_sync();
	void thread_exit();

protected:
	static void _bind_methods();
	
	virtual void cleanup();
	void call_dispatched_internal(int* registers) const;
	Variant call_return_internal(const Variant **p_args, int p_argcount, Variant::CallError &r_error) const;
public:
	ExecutionLoop();
	~ExecutionLoop();

	virtual bool assign_instance(Object* instance);
	void sync();
	void flush_queue() const;

	virtual Variant call_dispatched(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	virtual Variant call_sync(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	virtual Variant call_return(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
};

class SwarmExecutionLoop : public Reference {
	GDCLASS(SwarmExecutionLoop, Reference);
private:
	RWLock rwlock;
	Vector<Ref<ExecutionLoop>> all_threads;
	HashMap<ObjectID, uint32_t> object_id_cache;
protected:
	static void _bind_methods();
public:
	SwarmExecutionLoop();
	~SwarmExecutionLoop();

	bool assign_instance(Object* instance);
	bool remove_instance(const ObjectID& object_id);
	void sync();

	Variant call_dispatched(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	Variant call_sync(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
};

#endif