#ifndef STATE_AUTOMATON_H
#define STATE_AUTOMATON_H

// #include "modules/state/state.h"
#include <string>

#include "modules/hub/hub.h"
#include "core/dictionary.h"
#include "core/list.h"
#include "core/variant.h"
#include "core/reference.h"
#include "core/ustring.h"
#include "core/string_name.h"
#include "core/script_language.h"
#include "core/ordered_hash_map.h"

class State;
class PushdownAutomaton;
class StateAutomaton;

typedef OrderedHashMap<StringName, Ref<State>> StateHashMap;

class State : public Reference {
	GDCLASS(State, Reference);
private:
	// Ref<PushdownAutomaton> pda;
	StringName state_name;

protected:
	static void _bind_methods();
public:
	State();
	~State();

	void internal_start();
	StringName internal_poll(const Ref<StateAutomaton>& machine);
	void internal_finalize();

	void set_state_name(const StringName& new_state_name);
	inline StringName get_state_name() const { return state_name; }
};

class PushdownAutomaton : public Reference {
	GDCLASS(PushdownAutomaton, Reference);
private:
	StateHashMap state_pool;

protected:
	static void _bind_methods();

	inline StateHashMap* get_state_pool() { return &state_pool; }
	void get_state_pool_keys(List<StringName> *klist) const;
public:
	PushdownAutomaton();
	~PushdownAutomaton();
	
	Ref<State> get_entry_state();
	Ref<State> get_state_by_name(const StringName& state_name);
	Ref<State> get_next_state(const StringName& from_state);
	Ref<State> get_prev_state(const StringName& from_state);

	friend class StateAutomaton;

	bool add_state(const Ref<State>& new_state);
	// bool add_entry_state(const Ref<State>& new_state);
	bool remove_state(const StringName& state_name);

	Dictionary get_all_states();
};

class StateAutomaton : public Reference {
	GDCLASS(StateAutomaton, Reference);
private:
	Dictionary blackboard;
	float delta_time = 0.0;
	bool debug_status = false;

	Ref<PushdownAutomaton> pda;
protected:
	static void _bind_methods();
public:
	StateAutomaton();
	~StateAutomaton();
	
	void boot();
	void poll(const float& delta = 0.0);
	void finalize();

	void set_pda(const Ref<PushdownAutomaton>& new_pda);
	inline Ref<PushdownAutomaton> get_pda() const { return pda; }

	inline void enable_debug(const bool& debugging) { debug_status = debugging; }
	inline bool is_debugging() const { return debug_status; }

	inline float get_delta() const { return delta_time; }
	Variant blackboard_get(const String& what);
	void blackboard_set(const String& what, const Variant& with);
};
#endif