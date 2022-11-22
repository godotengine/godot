#ifndef DAEMON_MANAGER_H
#define DAEMON_MANAGER_H

#include <thread>
#include <mutex>
#include <string>
#include <chrono>
#include <queue>

#include "modules/hub/hub.h"

#include "core/engine.h"
#include "core/os/os.h"
#include "core/hash_map.h"
#include "core/reference.h"
#include "core/string_name.h"
#include "core/list.h"
#include "core/ustring.h"
#include "core/engine.h"
#include "core/list.h"

class Daemon;
struct DaemonTicket;
class DaemonManager;
struct DaemonQueue;
struct DaemonHandler;
// enum DaemonType;



enum DaemonType {
	DAEMON_ONESHOT,
	DAEMON_DESYNCED,
	DAEMON_SOFTSYNC,
};

struct DaemonTicket {
private:
	std::mutex ilock;
public:
	String daemon_name;
	DaemonType job_type;
	Ref<Daemon> daemon_ref;
	uint32_t batch_size = 0;
	bool is_terminated = false;
	uint32_t handing = 0;

	DaemonTicket(const Ref<Daemon>& daemon, const uint32_t& batch, DaemonType type);
	~DaemonTicket();

	// friend class DaemonManager;

	void lock() { ilock.lock(); }
	void unlock() { ilock.unlock(); }

	bool destroy_daemon();
};

struct DaemonHandler{
	RWLock rwlock;
	std::thread *worker = nullptr;
	List<DaemonTicket*>* ticket_queue = nullptr;

	uint32_t alloc_no;
	uint32_t batch_no;

	DaemonHandler();
	~DaemonHandler();

	bool add_ticket(DaemonTicket* ticket);
	bool remove_ticket(const uint8_t& id);

	void wlock() { rwlock.write_lock(); }
	void wunlock() { rwlock.write_unlock(); }
};

struct DaemonQueue{
	RWLock rwlock;
	List<DaemonHandler*> *worker_pool = nullptr;
	// List<uint8_t>* allocation_table;

	uint8_t worker_count = 0;
	uint8_t allocation_compass = 0;
	// uint8_t curr_max_alloc = 0;
	// uint8_t curr_min_alloc = 0;
	// uint8_t curr_max_alloc_id = -1;
	// uint8_t curr_min_alloc_id = -1;

	DaemonQueue();
	~DaemonQueue();

	// DaemonTicket* get_ticket();
	void add_ticket(DaemonTicket* ticket);
	void allocate_worker(const uint8_t& number);

	void wlock() { rwlock.write_lock(); }
	void wunlock() { rwlock.write_unlock(); }
};

class DaemonManager : public Object {
	GDCLASS(DaemonManager, Object)
private:
	uint64_t sync_time = 0;
	bool is_terminated = false;

	RWLock main_lock;

	DaemonQueue oneshot_queue;
	DaemonQueue synced_queue;
	DaemonQueue desynced_queue;

	void start_all();
protected:
	static void _bind_methods();
	static DaemonManager* singleton;

	static void oneshot_job (DaemonHandler* handler, const bool* termination, const uint64_t& sync_usec);
	static void synced_job  (DaemonHandler* handler, const bool* termination, const uint64_t& sync_usec);
	static void desynced_job(DaemonHandler* handler, const bool* termination, const uint64_t& sync_usec);
public:
	DaemonManager();
	~DaemonManager();

	static DaemonManager* get_singleton() { return singleton; }

	friend class std::thread;

	// Unexposed components
	bool queue_daemon(DaemonTicket* ticket);
};

class Daemon : public Reference {
	GDCLASS(Daemon, Reference)
private:
	String daemon_name;

	DaemonTicket *ticket;
protected:
	static void _bind_methods();

	void job(const uint32_t& alloc_no, const uint32_t& batch_no = 0, const bool* termination = new bool(true));
	virtual void internal_job(const uint32_t& alloc_no, const uint32_t& batch_no, const bool* termination) {}
public:
	Daemon();
	~Daemon();

	friend class DaemonManager;
	friend struct DaemonTicket;

	_FORCE_INLINE_ void set_daemon_name(const String& name) { daemon_name = name; }
	_FORCE_INLINE_ String get_daemon_name() const { return daemon_name; }

	bool create_job(DaemonType type);
	bool create_batch_job(const uint32_t& batch_size, DaemonType type);

	bool terminate_job();
};

class DaemonTestClass : public Daemon {
	GDCLASS(DaemonTestClass, Daemon)
private:
	int a = 1;
	std::mutex mut;
	void internal_job(const uint32_t& alloc_no, const uint32_t& batch_no, const bool* termination) override {
		mut.lock();
		a = a + 1;
		mut.unlock();
	}
public:
	DaemonTestClass() = default;
	~DaemonTestClass() = default;

	_FORCE_INLINE_ uint32_t get_value() const { return a; }
};

#endif

