// #ifndef PARALLEL_AUTOMATON
// #define PARALLEL_AUTOMATON

// #include "modules/state_automaton/state_automaton.h"
// #include "core/command_queue_mt.h"
// #include "core/safe_refcount.h"
// #include "core/os/thread.h"
// #include "core/os/mutex.h"
// #include "core/os/rw_lock.h"
// #include "core/list.h"

// class ExecutionSegment {
// private:
// 	mutable CommandQueueMT* command_queue = nullptr;
// 	mutable Mutex main_lock;

// 	List<Ref<State>> state_list;
// 	bool finalized = false;
// 	bool autosync = false;
// 	bool create_thread = false;
// 	SafeFlag exit;
// 	Thread::ID thread_id;
// 	Thread server_thread;
// private:
// 	static void _thread_callback(void* _instance);
// 	void thread_setup();
// 	void thread_loop();
// 	void thread_exit();
// 	void execute_internal(const Ref<StateAutomaton>& automaton);
// public:
// 	ExecutionSegment();
// 	~ExecutionSegment();

// 	void add_state(const Ref<State>& new_state);
// 	void remove_state(const StringName& state_name);
// 	void remove_all_state();

// 	void set_autosync(const bool& yes);
// 	bool get_autosync() const;

// 	void compile();
// 	void execute(const Ref<StateAutomaton>& automaton);
// 	void sync();
// };

// class ParallelAutomaton : public PushdownAutomaton {
// 	GDCLASS(ParallelAutomaton, PushdownAutomaton);
// private:
// 	Mutex internal_lock;
// 	RWLock blackboard_lock;
// protected:
// 	static void _bind_methods();
// public:
// 	ParallelAutomaton();
// 	~ParallelAutomaton();

// 	virtual Ref<State> get_entry_state() { return Ref<State>(); }
// 	virtual Ref<State> get_state_by_name(const StringName& state_name) const  { return Ref<State>(); }
// 	virtual Ref<State> get_next_state(const StringName& from_state) const  { return Ref<State>(); }
// 	virtual Ref<State> get_prev_state(const StringName& from_state) const  { return Ref<State>(); }

// 	virtual bool add_state(const Ref<State>& new_state) { return false; }
// 	virtual bool remove_state(const StringName& state_name)  { return false; }

// 	virtual void set_termination(const bool& status);
// 	virtual inline bool is_terminated() const { return terminated; }

// 	virtual inline void clean_pool() {}
// 	virtual Dictionary get_all_states() const { return Dictionary(); }
// 	virtual inline int get_pool_size() const { return -1;}

// 	void define_segments(const uint32_t& segment_count);
// 	void finalize_segment();
// };

// #endif