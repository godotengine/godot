#include "node_dispatcher.h"

#define PRECHECK_ALL_NODES

#define MIN_EXPECTED_FPS 20
#define MIN_THREAD 2
#define MAX_THREAD 512
#define THREAD_SUBTRACTION 1

#define DISPATCH_PHYSICS_SCRIPT_METHOD StringName("dispatched_physics")
#define DISPATCH_IDLE_SCRIPT_METHOD StringName("dispatched_idle")

#define HUB_PRINT(MSG) \
	Hub::get_singleton()->print_custom(String("NodeDispatcher"), String(MSG.c_str()))

NodeDispatcher *NodeDispatcher::singleton = nullptr;

bool NodeDispatcher::is_node_valid(Object *n){
	if (!n) return false;
	return !(n->get_instance_id() == 0 || n->is_queued_for_deletion());
}

int NodeDispatcher::get_nearest_two_exponent(const int& num){
	if (num >= MAX_THREAD) return MAX_THREAD;
	if (num <= 1) return MIN_THREAD;
	int sample = MAX_THREAD;
	while (sample > 1){
		if (num & sample) return sample;
		sample = sample >> 1;
	}
	return MIN_THREAD;
}

NodeDispatcher::NodeDispatcher() {
	singleton = this;
	auto core_count = OS::get_singleton()->get_processor_count();
	usable_threads = NodeDispatcher::get_nearest_two_exponent(core_count - THREAD_SUBTRACTION);
	min_thread_count = 2;
	max_thread_count = core_count > usable_threads ? core_count : usable_threads;
	max_physics_time = 1.0 / (float)ProjectSettings::get_singleton()->get_setting("physics/common/physics_fps");
	max_idle_time = 1.0 / MIN_EXPECTED_FPS;

	HUB_PRINT((cstr("Usable thread count: ") + std::to_string(usable_threads)));
	HUB_PRINT((cstr("Max thread count: ") + std::to_string(max_thread_count)));
	HUB_PRINT((cstr("Max idle time: ") + std::to_string(max_idle_time)));
	HUB_PRINT((cstr("Max physics time: ") + std::to_string(max_physics_time)));
}

NodeDispatcher::~NodeDispatcher(){
	// auto node_count = get_node_count();
	// for (auto i = 0; i < node_count; i++){
	// 	auto node = node_pool[i];
	// 	if (node) memdelete(node);
	// }
}

void NodeDispatcher::_bind_methods(){
	ClassDB::bind_method(D_METHOD("add_node", "node"), &NodeDispatcher::add_node);
	ClassDB::bind_method(D_METHOD("remove_node", "instance_id"), &NodeDispatcher::remove_node);
	ClassDB::bind_method(D_METHOD("queue_execution", "function", "parameters"), &NodeDispatcher::queue_execution);

	ClassDB::bind_method(D_METHOD("set_active", "trigger"), &NodeDispatcher::set_active);
	ClassDB::bind_method(D_METHOD("get_active"), &NodeDispatcher::get_active);

	ClassDB::bind_method(D_METHOD("set_usable_thread_count", "new_thread_count"), &NodeDispatcher::set_usable_thread_count);
	ClassDB::bind_method(D_METHOD("get_usable_thread_count"), &NodeDispatcher::get_usable_thread_count);

	ClassDB::bind_method(D_METHOD("get_min_thread_count"), &NodeDispatcher::get_min_thread_count);
	ClassDB::bind_method(D_METHOD("get_max_thread_count"), &NodeDispatcher::get_max_thread_count);
	ClassDB::bind_method(D_METHOD("get_node_count"), &NodeDispatcher::get_node_count);
	ClassDB::bind_method(D_METHOD("get_queued_execution_count"), &NodeDispatcher::get_queued_execution_count);
	ClassDB::bind_method(D_METHOD("get_all_nodes"), &NodeDispatcher::get_all_nodes);
	ClassDB::bind_method(D_METHOD("get_all_nodes_by_handler"), &NodeDispatcher::get_all_nodes_by_handler);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_active"), "set_active", "get_active");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "usable_thread_count"), "set_usable_thread_count", "get_usable_thread_count");
}

void NodeDispatcher::execute_external(){
	if (exec_pool.empty()) return;
	for (auto i = 0; i < exec_pool.size(); i++){
		if (i >= args_pool.size()){
			HUB_PRINT(cstr("External execution failed, no parameter for the current function found."));
			return;
		}
		auto fref = exec_pool[i];
		auto args = args_pool[i];
		fref->call_funcv(args);
	}
	int err = 1;
	fref_lock.lock();
	err += exec_pool.resize(0);
	err += args_pool.resize(0);
	fref_lock.unlock();
	if (err != 1){
		HUB_PRINT(cstr("There were errors while trying to resize exec_pool and args_pool"));
	}
}

void NodeDispatcher::clean_pool(){
	primary_lock.lock();
	for (auto i = 0; i < node_pool.size(); i++){
		auto n = node_pool[i];
		if (is_node_valid(n)) continue;
		if (n->is_inside_tree()) continue;
		node_pool.remove(i);
		i--;
	}
	primary_lock.unlock();
}

void NodeDispatcher::dispatch_all(const bool& is_physics){
	// 4 threads | 8 jobs
	// Alternating Thread Allocation:
	//				&		&		&
	//			^		^		^
	//		@		@		@		@
	//	*		*		*		*
	//	1	2	4	8	16	32	64	128
	// ------------------------------------
	//			ALLOCATION
	//	1	1	2	2	2	2	2	2
	// Each jobs MUST be handled by at most one thread
	// Segmented Thread Allocation:
	// Each threads handle a chain of jobs with the segment length of 8 / 2
	//	0		1		2		3
	//	0	1	2	3	4	5	6	7
	//							&	&
	//					^	^
	//			@	@
	//	*	*
	//	1	2	4	8	16	32	64	128
	// ------------------------------------
	//			ALLOCATION
	//	1	1	1	1	1	1	1	1
	if (get_node_count() == 0 || !is_active) return;
	// DATASET:
	// 32 total cores
	// 98 node
	//
	// 8 idle threads | 8 physics threads
	// Each threads handle 98 / 8 = 12 jobs
	//

	// HUB_PRINT((cstr("is_physics = ") + std::to_string(is_physics)));
	auto usable = usable_threads / 2;
	auto pool_size = get_node_count();
	auto threshold = min(usable, pool_size);
	// HUB_PRINT(cstr("Dispatching..."));
	PoolVector<cthread*> thread_list;
	for (auto i = 0; i < threshold; i++){
		// HUB_PRINT((cstr("Dispatching thread no.") + std::to_string(i)));
		auto thread = dispatch(i, is_physics);
		thread_list.push_back(thread);
	}
	// HUB_PRINT(cstr("Joining..."));
	for (auto i = 0; i < threshold; i++){
		// HUB_PRINT((cstr("Joning thread no.") + std::to_string(i)));
		auto thread = thread_list[i];
		thread->join();
		delete thread;
	}
	thread_list.resize(0);
}

void NodeDispatcher::dispatch_idle(){
#ifdef PRECHECK_ALL_NODES
	clean_pool();
#endif
	dispatch_all(false);
}
void NodeDispatcher::dispatch_physics(){
	execute_external();
	dispatch_all(true);
}
void NodeDispatcher::dispatch_internal(const int& tid, const bool& is_physics){
	auto calling_method = (is_physics ? DISPATCH_PHYSICS_SCRIPT_METHOD : DISPATCH_IDLE_SCRIPT_METHOD);
	auto usable = usable_threads / 2;
	auto pool_size = get_node_count();
	auto allocate_to = (int)std::ceil((float)pool_size / usable);
	allocate_to = allocate_to >= 1 ? allocate_to : 1;
	auto time_start = OS::get_singleton()->get_ticks_msec();
	auto limit_fetch = is_physics ? &max_physics_time : &max_idle_time;
	auto static_index = tid * allocate_to;
	//	0		1		2
	//	0	1	2	3	4
	//	*	*	^	^	#
	//	1	2	4	8	16
	for (auto i = 0; i < allocate_to; i++){
		auto dynamic_index = static_index + i;
		// dynamic_index = clampi(dynamic_index, 0, pool_size);
		if (dynamic_index >= pool_size) return;
		auto n = node_pool[dynamic_index];

#ifndef PRECHECK_ALL_NODES
		if (!is_node_valid(n)) continue;
		// Check if the SceneTree is paused
		// If it is, check if current node can continue to process during pause
		if (!n->is_inside_tree()) continue;
#endif
		if (!n->can_process()) continue;
		//-----------------------------------------------------
		n->call(calling_method);
		//-----------------------------------------------------
		if (!auto_thread_cancel) continue;
		auto curr_time = OS::get_singleton()->get_ticks_msec();
		float time_taken = (curr_time - time_start) / 1000.0;
		if (time_taken > *limit_fetch) return;
	}
}
cthread *NodeDispatcher::dispatch(const int& tid, const bool& is_physics){
	return new cthread(&NodeDispatcher::dispatch_internal, this, tid, is_physics);
}

bool NodeDispatcher::add_node(Node* node){
	if (!is_node_valid(node)) return false;
	if (!node->has_method(DISPATCH_IDLE_SCRIPT_METHOD) || !node->has_method(DISPATCH_PHYSICS_SCRIPT_METHOD)){
		HUB_PRINT(cstr("Given node does not have at least one dispatched method"));
		return false;
	}
	primary_lock.lock();
	auto pool_size = get_node_count();
	for (auto i = 0; i < pool_size; i++){
		auto curr = node_pool[i];
		if (!is_node_valid(curr) || !curr->is_inside_tree()) {
			node_pool.remove(i);
			i--;
			pool_size--;
			continue;
		}
		if (curr->get_instance_id() == node->get_instance_id()) {
			primary_lock.unlock();
			return false;
		}
	}
	node_pool.push_back(node);
	primary_lock.unlock();
	return true;
}
bool NodeDispatcher::remove_node(const int& instance_id){
	primary_lock.lock();
	auto pool_size = get_node_count();
	for (auto i = 0; i < pool_size; i++){
		auto node = node_pool[i];
		if (!is_node_valid(node)) continue;
		if (node->get_instance_id() == instance_id){
			node_pool.remove(i);
			primary_lock.unlock();
			return true;
		}
	}
	primary_lock.unlock();
	return false;
}

void NodeDispatcher::queue_execution(const Ref<FuncRef>& function, Array args){
	fref_lock.lock();
	exec_pool.push_back(function);
	args_pool.push_back(args);
	fref_lock.unlock();
}

void NodeDispatcher::set_active(const bool& trigger){
	primary_lock.lock();
	is_active = trigger;
	primary_lock.unlock();
}

void NodeDispatcher::set_usable_thread_count(const int& new_thread_count){
	primary_lock.lock();
	usable_threads = get_nearest_two_exponent(min_thread_count >= new_thread_count ? min_thread_count : new_thread_count);
	primary_lock.unlock();
}

Array NodeDispatcher::get_all_nodes() {
	primary_lock.lock();
	auto size = get_node_count();
	Array arr;
	arr.resize(size);
	for (int i = 0; i < size; i++){
		arr[i] = node_pool[i];
	}
	primary_lock.unlock();
	return arr;
}
Dictionary NodeDispatcher::get_all_nodes_by_handler(){
	Dictionary re;
	primary_lock.lock();
	auto usable = usable_threads / 2;
	auto pool_size = get_node_count();
	auto allocate_to = (int)ceil((float)pool_size / usable);
	allocate_to = allocate_to >= 1 ? allocate_to : 1;
	auto threshold = min(usable, pool_size);
	for (auto tid = 0; tid < threshold; tid++){
		auto static_index = tid * allocate_to;
		Array curr;
		for (auto i = tid; i < allocate_to; i++){
			auto dynamic_index = static_index + i;
			if (dynamic_index >= pool_size) break;
			curr.append(node_pool[dynamic_index]);
		}
		re[Variant(tid)] = curr;
	}
	primary_lock.unlock();
	return re;
}
