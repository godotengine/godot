#include "register_types.h"
#include "core/class_db.h"
#include "core/engine.h"
#include "pooled_process.h"

static PooledProcess* PooledProcessPtr = NULL;

void register_pooled_process_types(){
	ClassDB::register_class<PooledProcess>();
	PooledProcessPtr = memnew(PooledProcess);
	Engine::get_singleton()->add_singleton(Engine::Singleton("PooledProcess", PooledProcess::get_singleton()));
}

void unregister_pooled_process_types(){
	PooledProcessPtr->join();
	memdelete(PooledProcessPtr);
}
