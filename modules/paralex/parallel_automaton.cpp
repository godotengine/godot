// #include "parallel_automaton.h"

// ExecutionSegment::ExecutionSegment(){

// }
// ExecutionSegment::~ExecutionSegment(){
// 	thread_exit();
// 	if (create_thread){
// 		server_thread.wait_to_finish();
// 	} else {}
// 	remove_all_state();
// }
// void ExecutionSegment::_thread_callback(void* _instance){
// 	ExecutionSegment *es = reinterpret_cast<ExecutionSegment*>(_instance);
// 	es->thread_loop();
// }
// void ExecutionSegment::thread_setup(){
// 	exit.clear();
// 	server_thread.start(&ExecutionSegment::_thread_callback, this);
// }
// void ExecutionSegment::thread_loop(){
// 	thread_id = Thread::get_caller_id();
// 	while (!exit.is_set()){
// 		command_queue->wait_and_flush_one();
// 	}
// 	command_queue->flush_all();
// }
// void ExecutionSegment::thread_exit(){
// 	exit.set();
// }
// void ExecutionSegment::add_state(const Ref<State>& new_state){
// 	MutexLock lock(main_lock);
// 	state_list.push_back(new_state);
// }
// void ExecutionSegment::remove_state(const StringName& state_name){
// 	MutexLock lock(main_lock);
// 	for (auto E = state_list.front(); E; E = E->next()){
// 		if (E->get()->get_state_name() == state_name){
// 			E->erase();
// 			return;
// 		}
// 	}
// }
// void ExecutionSegment::remove_all_state(){
// 	MutexLock lock(main_lock);
// 	state_list.clear();
// }
// void ExecutionSegment::set_autosync(const bool& yes){
// 	MutexLock lock(main_lock);
// 	autosync = yes;
// }
// bool ExecutionSegment::get_autosync() const{
// 	MutexLock lock(main_lock);
// 	return autosync;
// }
// void ExecutionSegment::compile(){
// 	MutexLock lock(main_lock);
// 	ERR_FAIL_COND(finalized);
// 	create_thread = !(state_list.size() <= 1);
// 	autosync = !(state_list.size() <= 1);
// 	command_queue = new CommandQueueMT(create_thread);
// 	finalized = true;
// 	if (!create_thread) {
// 		thread_id = Thread::get_caller_id();
// 	} else {
// 		thread_id = 0;
// 		thread_setup();
// 	}
// }
// void ExecutionSegment::execute(const Ref<StateAutomaton>& automaton){
// 	if (!create_thread) {
// 		execute_internal(automaton);
// 	} else {
// 		command_queue->push(this, &ExecutionSegment::execute_internal, automaton);
// 	}
// }
// void ExecutionSegment::sync(){
// 	if (Thread::get_caller_id() != thread_id){
// 		command_queue->push_and_sync(this, &ExecutionSegment::sync)
// 	} else {
// 		return;
// 	}
// }
