#include "register_types.h"
#include "core/class_db.h"
#include "core/engine.h"
#include "execution_loop.h"
#include "atomic_types_gd.h"
#include "rwlock_cb.h"

void register_paralex_types(){
	ClassDB::register_class<ExecutionLoop>();
	ClassDB::register_class<SwarmExecutionLoop>();
	ClassDB::register_class<_RWLock>();
	ClassDB::register_class<_SafeNumeric>();
	ClassDB::register_class<_SafeBoolean>();
}
void unregister_paralex_types(){

}
