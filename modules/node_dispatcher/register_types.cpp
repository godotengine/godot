#include "register_types.h"
#include "core/class_db.h"
#include "core/engine.h"
#include "node_dispatcher.h"

static NodeDispatcher* NodeDispatcherPtr = NULL;
// static PooledProcess* PooledProcessPtr = NULL;

void register_node_dispatcher_types(){
	ClassDB::register_class<NodeDispatcher>();
	NodeDispatcherPtr = memnew(NodeDispatcher);
	Engine::get_singleton()->add_singleton(Engine::Singleton("NodeDispatcher", NodeDispatcher::get_singleton()));
	// ClassDB::register_class<PooledProcess>();
	// PooledProcessPtr = memnew(PooledProcess);
	// Engine::get_singleton()->add_singleton(Engine::Singleton("PooledProcess", PooledProcess::get_singleton()));
}

void unregister_node_dispatcher_types(){
	memdelete(NodeDispatcherPtr);
	// PooledProcessPtr->join();
	// memdelete(PooledProcessPtr);
}
