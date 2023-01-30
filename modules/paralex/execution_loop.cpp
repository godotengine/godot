#include "execution_loop.h"

ExecutionLoop::ExecutionLoop() : command_queue(true) {
	assign_instance(nullptr);
	thread_start();
}
ExecutionLoop::~ExecutionLoop(){
	cleanup();
}

void ExecutionLoop::cleanup(){
	// print_line(String("Setting exit flag..."));
	command_queue.push(this, &ExecutionLoop::thread_exit);
	// print_line(String("Waiting for Godot..."));
	server_thread.wait_to_finish();
	// print_line(String("Thread exitted"));
	assign_instance(nullptr);
}

void ExecutionLoop::_bind_methods(){
	ClassDB::bind_method(D_METHOD("assign_instance", "instance"), &ExecutionLoop::assign_instance);
	ClassDB::bind_method(D_METHOD("sync"), &ExecutionLoop::sync);
	ClassDB::bind_method(D_METHOD("flush_queue"), &ExecutionLoop::flush_queue);
	//--------------------------------------------------------------
	{
		MethodInfo mi;
		mi.name = "call_dispatched";
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_dispatched", &ExecutionLoop::call_dispatched, mi);
	}
	{
		MethodInfo mi;
		mi.name = "call_sync";
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_sync", &ExecutionLoop::call_sync, mi);
	}
	{
		MethodInfo mi;
		mi.name = "call_return";
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_return", &ExecutionLoop::call_return, mi);
	}
}

void ExecutionLoop::_thread_callback(void * _instance){
	ExecutionLoop* cq = reinterpret_cast<ExecutionLoop*>(_instance);
	cq->thread_loop();
}
void ExecutionLoop::thread_start(){
	if (thread_started.is_set()) return;
	thread_started.set();
	server_thread.start(_thread_callback, this);
}
void ExecutionLoop::thread_loop(){
	exit.clear();
	server_id = Thread::get_caller_id();
	while (!exit.is_set()){
		command_queue.wait_and_flush_one();
	}
	command_queue.flush_all();
}
uint32_t ExecutionLoop::thread_sync(){
	return 0;
}
void ExecutionLoop::thread_exit(){
	exit.set();
}

bool ExecutionLoop::assign_instance(Object* instance){
	if (instance == nullptr){
		object_ref.unref();
		real_object = nullptr;
	} else {
		if (instance->is_class("Reference")){
			// Prevent the object from being deleted
			object_ref = instance;
		}
		real_object = instance;
	}
	return true;
}
void ExecutionLoop::sync() {
	uint32_t ret;
	command_queue.push_and_ret(this, &ExecutionLoop::thread_sync, &ret);
	return;
}
void ExecutionLoop::flush_queue() const{
	command_queue.flush_all();
}

void ExecutionLoop::remove_virtual_register(int* registers) const {
	int p_argcount = *registers;
	auto p_args = (Variant*)((size_t)registers + sizeof(int));
	for (uint32_t i = 0; i < p_argcount; i++){
		p_args[i].~Variant();
	}
	Memory::free_static(registers);
	registers = nullptr;
}

#define CallCheck(p_args, p_argcount, r_error)                                   \
	ERR_FAIL_COND_V_MSG(!real_object, Variant(), "Instance has not been set");   \
	if (p_argcount < 1) {                                                        \
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;        \
		r_error.argument = 0;                                                    \
		return Variant();                                                        \
	}                                                                            \
	if (p_args[0]->get_type() != Variant::STRING) {                              \
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;         \
		r_error.argument = 0;                                                    \
		r_error.expected = Variant::STRING;                                      \
		return Variant();                                                        \
	}

void ExecutionLoop::call_dispatched_internal(int* registers) const {
	int p_argcount = *registers;
	auto p_args = (Variant*)(&registers[1]);
	
	if (!real_object || p_argcount < 1 || p_args[0].get_type() != Variant::STRING) {
		remove_virtual_register(registers);
		return;
	}

	Variant** virt_param = (Variant**)Memory::alloc_static(sizeof(Variant*) * p_argcount);
	for (uint32_t i = 0; i < p_argcount; i++){
		virt_param[i] = &p_args[i];
	}

	StringName method = p_args[0];
	Variant::CallError ce;
	const_cast<ExecutionLoop*>(this)->real_object->call(method, const_cast<const Variant**>(&virt_param[1]), p_argcount - 1, ce);
	remove_virtual_register(registers);
	Memory::free_static(virt_param);
	return;
}
Variant ExecutionLoop::call_return_internal(const Variant **p_args, int p_argcount, Variant::CallError &r_error) const {
	// CallCheck(p_args, p_argcount, r_error);
	StringName method = *p_args[0];
	return const_cast<ExecutionLoop*>(this)->real_object->call(method, &p_args[1], p_argcount - 1, r_error);
}
Variant ExecutionLoop::call_dispatched(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	// Parameters passed as references could be deallocated when
	// this function exit.
	// That's why we need to make a copy of every arguments.
	// Since they could contain reference-count info,
	// a memcpy would not suffice.
	// Hence why this function must copy every parameters manually.
	//
	int* new_reg = (int*)Memory::alloc_static(sizeof(int) + (sizeof(Variant) * p_argcount));
	*new_reg = p_argcount;
	Variant* dup_param = (Variant*)(&new_reg[1]);
	for (uint32_t i = 0; i < p_argcount; i++){
		memnew_placement(&dup_param[i], Variant(*p_args[i]));
	}
	// --------------------------------------------------
	command_queue.push(this, &ExecutionLoop::call_dispatched_internal, new_reg);
	return Variant();
}
Variant ExecutionLoop::call_sync(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	CallCheck(p_args, p_argcount, r_error);
	Variant ret{};
	command_queue.push_and_ret(this, &ExecutionLoop::call_return_internal, p_args, p_argcount, r_error, &ret);
	return Variant();
}
Variant ExecutionLoop::call_return(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	CallCheck(p_args, p_argcount, r_error);
	Variant ret{};
	command_queue.push_and_ret(this, &ExecutionLoop::call_return_internal, p_args, p_argcount, r_error, &ret);
	return ret;
}

SwarmExecutionLoop::SwarmExecutionLoop(){

}
SwarmExecutionLoop::~SwarmExecutionLoop(){
	all_threads.clear();
	object_id_cache.clear();
}
void SwarmExecutionLoop::_bind_methods(){
	ClassDB::bind_method(D_METHOD("assign_instance", "instance"), &SwarmExecutionLoop::assign_instance);
	ClassDB::bind_method(D_METHOD("remove_instance", "object_id"), &SwarmExecutionLoop::remove_instance);
	ClassDB::bind_method(D_METHOD("sync"), &SwarmExecutionLoop::sync);
	//--------------------------------------------------------------
	{
		MethodInfo mi;
		mi.name = "call_dispatched";
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_dispatched", &SwarmExecutionLoop::call_dispatched, mi);
	}
	{
		MethodInfo mi;
		mi.name = "call_sync";
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_sync", &SwarmExecutionLoop::call_sync, mi);
	}
}
bool SwarmExecutionLoop::assign_instance(Object* instance){
	rwlock.write_lock();
	ObjectID id = instance->get_instance_id();
	if (object_id_cache.has(id)) {
		rwlock.write_unlock();
		return false;
	}
	object_id_cache[id] = all_threads.size();
	Ref<ExecutionLoop> el = memnew(ExecutionLoop);
	el->assign_instance(instance);
	all_threads.push_back(el);
	rwlock.write_unlock();
	return true;
}
bool SwarmExecutionLoop::remove_instance(const ObjectID& object_id){
	rwlock.write_lock();
	if (!object_id_cache.has(object_id)) {
		rwlock.write_unlock();
		return false;
	}
	auto idx = object_id_cache[object_id];
	object_id_cache.erase(object_id);
	all_threads.remove(idx);
	rwlock.write_unlock();
	return true;
}

void SwarmExecutionLoop::sync(){
	rwlock.read_lock();
	for (uint32_t i = 0, s = all_threads.size(); i < s; i++){
		all_threads.write[i]->sync();
	}
	rwlock.read_unlock();
}
Variant SwarmExecutionLoop::call_dispatched(const Variant **p_args, int p_argcount, Variant::CallError &r_error){
	rwlock.read_lock();
	for (uint32_t i = 0, s = all_threads.size(); i < s; i++){
		all_threads.write[i]->call_dispatched(p_args, p_argcount, r_error);
	}
	rwlock.read_unlock();
	return Variant();
}
Variant SwarmExecutionLoop::call_sync(const Variant **p_args, int p_argcount, Variant::CallError &r_error){
	rwlock.read_lock();
	for (uint32_t i = 0, s = all_threads.size(); i < s; i++){
		all_threads.write[i]->call_sync(p_args, p_argcount, r_error);
	}
	rwlock.read_unlock();
	return Variant();
}