#include "daemon_manager.h"

#define WORKER_COUNT 8
#define CALLBACK_VMETHOD StringName("_job_callback")
#define DESYNCED_ONESHOT
#define DESYNC_INTERVAL_USEC 1000

#define CLOCK OS::get_singleton()->get_ticks_usec()

VARIANT_ENUM_CAST(DaemonType);

DaemonManager* DaemonManager::singleton = nullptr;

Daemon::Daemon(){
	daemon_name = get_class();
}

Daemon::~Daemon(){
	terminate_job();
}

void Daemon::_bind_methods(){
	ClassDB::bind_method(D_METHOD("set_daemon_name", "new_name"), &Daemon::set_daemon_name);
	ClassDB::bind_method(D_METHOD("get_daemon_name"), &Daemon::get_daemon_name);

	ClassDB::bind_method(D_METHOD("create_job", "job_type"), &Daemon::create_job);
	ClassDB::bind_method(D_METHOD("create_batch_job", "batch_size", "job_type"), &Daemon::create_batch_job);

	ClassDB::bind_method(D_METHOD("terminate_job"), &Daemon::terminate_job);

	BIND_VMETHOD(MethodInfo(Variant::NIL, CALLBACK_VMETHOD, PropertyInfo(Variant::INT, "alloc_no"), PropertyInfo(Variant::INT, "batch_no"), PropertyInfo(Variant::BOOL, "termination")));

	// auto info = PropertyInfo(Variant::INT, "current_graphics_preset", PROPERTY_HINT_ENUM, "");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "daemon_name"), "set_daemon_name", "get_daemon_name");
}

void DaemonManager::_bind_methods(){
	BIND_ENUM_CONSTANT(DAEMON_ONESHOT);
	BIND_ENUM_CONSTANT(DAEMON_DESYNCED);
	BIND_ENUM_CONSTANT(DAEMON_SOFTSYNC);
}
void Daemon::job(const uint32_t& alloc_no, const uint32_t& batch_no, const bool* termination){
	internal_job(alloc_no, batch_no, termination);
	auto script = get_script_instance();
	Hub::get_singleton()->print_custom("Daemon", "May be don\'t");
	if (script){
		script->call(CALLBACK_VMETHOD, Variant(alloc_no), Variant(batch_no), Variant(*termination));
		Hub::get_singleton()->print_custom("Daemon", "Suck my tits");
	}
}

bool Daemon::create_job(DaemonType type){
	return create_batch_job(1, type);
}
bool Daemon::create_batch_job(const uint32_t& batch_size, DaemonType type){
	ticket = new DaemonTicket(Ref<Daemon>(this), batch_size, type);
	return DaemonManager::get_singleton()->queue_daemon(ticket);
}
bool Daemon::terminate_job(){
	if (ticket) delete ticket;
	else return false;
	return true;
}

DaemonTicket::DaemonTicket(const Ref<Daemon>& daemon, const uint32_t& batch, DaemonType type){
	daemon_name = daemon->get_daemon_name();
	daemon_ref = daemon;
	batch_size = batch;
	job_type = type;
}
DaemonTicket::~DaemonTicket(){
	destroy_daemon();
}

bool DaemonTicket::destroy_daemon(){
	lock();
	is_terminated = true;
	unlock();
	return true;
}

DaemonQueue::DaemonQueue(){
	// rwlock = new RWLock();
	worker_pool = new List<DaemonHandler*>();
	// allocation_table = new List<uint8_t>();
}

DaemonQueue::~DaemonQueue(){
	for (auto E = worker_pool->front(); E; E = E->next()){
		auto pool = E->get();
		if (pool) delete pool;
	}
	delete worker_pool;
	// delete allocation_table;
	// delete rwlock;
	
}

void DaemonQueue::add_ticket(DaemonTicket* ticket){
	if (worker_pool->size() == 0) return;
	rwlock.write_lock();
	auto handler = worker_pool->operator[](allocation_compass);
	auto res = handler->add_ticket(ticket);
	if (res){
		allocation_compass++;
		if (allocation_compass >= worker_count) allocation_compass = 0;
	}
	rwlock.write_unlock();
}

void DaemonQueue::allocate_worker(const uint8_t& number){
	if (worker_count > 0 || number <= 0) return;
	rwlock.write_lock();
	for (uint8_t i = 0; i < number; i++){
		worker_pool->push_back(new DaemonHandler());
	}
	worker_count = number;
	rwlock.write_unlock();
}

DaemonHandler::DaemonHandler(){
	ticket_queue = new List<DaemonTicket*>();
}
DaemonHandler::~DaemonHandler(){
	for (auto E = ticket_queue->front(); E; E = E->next()){
		auto ticket = E->get();
		if (ticket) delete ticket;
	}
	delete ticket_queue;
	if (worker) {
		worker->join();
		delete worker;
	}
}

bool DaemonHandler::add_ticket(DaemonTicket* ticket){
	wlock();
	ticket_queue->push_back(ticket);
	wunlock();
	return true;
}

bool DaemonHandler::remove_ticket(const uint8_t& id){
	if (id < 0 || id >= ticket_queue->size()) return false;
	wlock();
	uint8_t count = 0;
	for (auto E = ticket_queue->front(); E; E = E->next()){
		if (count == id){
			ticket_queue->erase(E);
			break;
		}
		count++;
	}
	wunlock();
	return true;
}

DaemonManager::DaemonManager(){
	singleton = this;
	oneshot_queue.allocate_worker(1);
	synced_queue.allocate_worker(WORKER_COUNT);
	desynced_queue.allocate_worker(WORKER_COUNT);
	sync_time = (1.0 / Engine::get_singleton()->get_iterations_per_second()) * 1000000;
	start_all();
}

DaemonManager::~DaemonManager(){
	is_terminated = true;
}

void DaemonManager::start_all(){
	oneshot_queue.wlock();
	synced_queue.wlock();
	desynced_queue.wlock();

	auto oh = oneshot_queue.worker_pool->operator[](0);
	oh->worker = new std::thread(&DaemonManager::oneshot_job,	oh, &is_terminated, sync_time);
	for (uint8_t i = 0; i < WORKER_COUNT; i++){
		auto sh = synced_queue.worker_pool->operator[](i);
		auto dh = desynced_queue.worker_pool->operator[](i);
		sh->worker = new std::thread(&DaemonManager::synced_job,	sh, &is_terminated, sync_time);
		dh->worker = new std::thread(&DaemonManager::desynced_job,	dh, &is_terminated, DESYNC_INTERVAL_USEC);
	}

	oneshot_queue.wunlock();
	synced_queue.wunlock();
	desynced_queue.wunlock();
}

void DaemonManager::oneshot_job (DaemonHandler* handler, const bool* termination, const uint64_t& sync_usec){
	auto start = CLOCK;
	while (!(*termination)){
		handler->wlock();
		for (auto E = handler->ticket_queue->front(); E; E = E->next()){
			auto ticket = E->get();
			if (ticket->is_terminated){
				handler->ticket_queue->erase(E);
				continue;
			}
			ticket->daemon_ref->job(handler->alloc_no, handler->batch_no, termination);
			handler->ticket_queue->erase(E);
		}
		handler->wunlock();
		auto now = CLOCK;
		auto delta = (now - start);
		start = now;
		if (delta < sync_usec){
			std::this_thread::sleep_for(std::chrono::microseconds(sync_usec - delta));
		} else {
			continue;
		}
	}
}
void DaemonManager::synced_job  (DaemonHandler* handler, const bool* termination, const uint64_t& sync_usec){
	auto start = CLOCK;
	while (!(*termination)){
		handler->wlock();
		for (auto E = handler->ticket_queue->front(); E; E = E->next()){
			auto ticket = E->get();   
			if (ticket->is_terminated){
				handler->ticket_queue->erase(E);
				continue;
			}
			ticket->daemon_ref->job(handler->alloc_no, handler->batch_no, termination);
		}
		handler->wunlock();
		auto now = CLOCK;
		auto delta = (now - start);
		start = now;
		if (delta < sync_usec){
			std::this_thread::sleep_for(std::chrono::microseconds(sync_usec - delta));
		} else {
			continue;
		}
	}
}
void DaemonManager::desynced_job(DaemonHandler* handler, const bool* termination, const uint64_t& sync_usec){
	while (!(*termination)){
		handler->wlock();
		for (auto E = handler->ticket_queue->front(); E; E = E->next()){
			auto ticket = E->get();
			if (ticket->is_terminated){
				handler->ticket_queue->erase(E);
				continue;
			}
			ticket->daemon_ref->job(handler->alloc_no, handler->batch_no, termination);
			// handler->ticket_queue->erase(E);
		}
		handler->wunlock();
		std::this_thread::sleep_for(std::chrono::microseconds(sync_usec));
	}
}

bool DaemonManager::queue_daemon(DaemonTicket* ticket){
	DaemonQueue *queue;
	switch (ticket->job_type){
		case DaemonType::DAEMON_ONESHOT:
			queue = &oneshot_queue;
			break;
		case DaemonType::DAEMON_DESYNCED:
			queue = &desynced_queue;
			break;
		case DaemonType::DAEMON_SOFTSYNC:
			queue = &synced_queue;
			break;
		default: return false;
	}
	while (ticket->handing < ticket->batch_size){
		queue->add_ticket(ticket);
		ticket->handing = ticket->handing + 1;
	}
	return true;
}
