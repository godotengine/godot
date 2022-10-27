#include "register_types.h"
#include "core/class_db.h"
#include "core/engine.h"
#include "instance_pool.h"

static InstancePool* InstancePoolPtr = NULL;

void register_instance_pool_types(){
	ClassDB::register_class<Future>();
	ClassDB::register_class<WorkPool>();
	ClassDB::register_class<InstancePool>();
	InstancePoolPtr = memnew(InstancePool);
	Engine::get_singleton()->add_singleton(Engine::Singleton("InstancePool", InstancePool::get_singleton()));
}

void unregister_instance_pool_types(){
	memdelete(InstancePoolPtr);
}
