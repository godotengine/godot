#include "register_types.h"
#include "core/class_db.h"
#include "core/engine.h"
#include "daemon_manager.h"

#include "core/print_string.h"

#define DISABLE_DAEMON

static DaemonManager* DaemonManagerPtr = NULL;

void test_stuff(){
#ifndef DISABLE_DAEMON
	Ref<DaemonTestClass> test_daemon = memnew(DaemonTestClass);
	test_daemon->create_batch_job(2, DaemonType::DAEMON_ONESHOT);
	std::this_thread::sleep_for(std::chrono::milliseconds(2));
	print_line(String((std::string("Stuff") + std::to_string(test_daemon->get_value())).c_str()));
#endif
}

void register_daemon_manager_types(){
#ifndef DISABLE_DAEMON
	ClassDB::register_class<DaemonManager>();
	ClassDB::register_class<Daemon>();
	DaemonManagerPtr = memnew(DaemonManager);
	Engine::get_singleton()->add_singleton(Engine::Singleton("DaemonManager", DaemonManager::get_singleton()));
	// test_stuff();
#endif
}

void unregister_daemon_manager_types(){
#ifndef DISABLE_DAEMON
	memdelete(DaemonManagerPtr);
#endif
}
