#include "state_automaton.h"

#define MAX_STATE_POLL 256

#define STATE_VM_START					StringName("_start")
#define STATE_VM_POLL					StringName("_poll")
#define STATE_VM_FINALIZE				StringName("_finalize")
#define STATE_VM_POLL_STOP_VALUE		StringName("__stop")
#define STATE_VM_POLL_NEXT_VALUE		StringName("__next")
#define STATE_VM_POLL_PREV_VALUE		StringName("__prev")
#define STATE_VM_POLL_FIRST_VALUE		StringName("__first")

#define HUB_PRINT_FLAG(FLAG, MSG) \
	Hub::get_singleton()->print_custom(String(FLAG), String(MSG.c_str()))
#define HUB_PRINT_S(MSG) \
	HUB_PRINT_FLAG("State", MSG)
#define HUB_PRINT_PA(MSG) \
	HUB_PRINT_FLAG("PushdownAutomaton", MSG)
#define HUB_PRINT_SA(MSG) \
	HUB_PRINT_FLAG("StateAutomaton", MSG)

State::State() {
	state_name = "__default";
}

State::~State() {}

void State::_bind_methods(){
	ClassDB::bind_method(D_METHOD("set_state_name", "new_state_name"), &State::set_state_name);
	ClassDB::bind_method(D_METHOD("get_state_name"), &State::get_state_name);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "state_name"), "set_state_name", "get_state_name");

	BIND_VMETHOD(MethodInfo(Variant::STRING, STATE_VM_START, PropertyInfo("StateAutomaton")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, STATE_VM_POLL, PropertyInfo("StateAutomaton")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, STATE_VM_FINALIZE, PropertyInfo("StateAutomaton")));
}

void State::internal_start(const Ref<StateAutomaton>& machine){
	auto script = get_script_instance();
	if (script) script->call(STATE_VM_START, machine);
	// if (has_method(STATE_VM_START)) call(STATE_VM_START);
}

StringName State::internal_poll(const Ref<StateAutomaton>& machine){
	auto result = Variant(STATE_VM_POLL_NEXT_VALUE);
	// if (has_method(STATE_VM_POLL)) result = call(STATE_VM_POLL, machine);
	auto script = get_script_instance();
	if (script) result = script->call(STATE_VM_POLL, machine);
	if (result.get_type() == Variant::Type::STRING)
		return (String)result;
	return STATE_VM_POLL_NEXT_VALUE;
}

void State::internal_finalize(const Ref<StateAutomaton>& machine){
	// if (has_method(STATE_VM_FINALIZE)) call(STATE_VM_FINALIZE);
	auto script = get_script_instance();
	if (script) script->call(STATE_VM_FINALIZE, machine);
}

void State::set_state_name(const StringName& new_state_name){
	auto str_name = (String)new_state_name;
	if (str_name.size() < 2) return;
	if (str_name[0] == '_' && str_name[1] == '_') return;
	state_name = new_state_name;
}

PushdownAutomaton::PushdownAutomaton(){

}

PushdownAutomaton::~PushdownAutomaton(){

}

void PushdownAutomaton::_bind_methods(){
	ClassDB::bind_method(D_METHOD("get_entry_state"), &PushdownAutomaton::get_entry_state);
	ClassDB::bind_method(D_METHOD("get_state_by_name", "state_name"), &PushdownAutomaton::get_state_by_name);
	ClassDB::bind_method(D_METHOD("get_next_state", "from_state"), &PushdownAutomaton::get_next_state);
	ClassDB::bind_method(D_METHOD("get_prev_state", "from_state"), &PushdownAutomaton::get_prev_state);

	ClassDB::bind_method(D_METHOD("add_state", "new_state"), &PushdownAutomaton::add_state);
	ClassDB::bind_method(D_METHOD("remove_state", "state_name"), &PushdownAutomaton::remove_state);
	ClassDB::bind_method(D_METHOD("get_all_states"), &PushdownAutomaton::get_all_states);
	ClassDB::bind_method(D_METHOD("get_pool_size"), &PushdownAutomaton::get_pool_size);
}

Ref<State> PushdownAutomaton::get_entry_state(){
	if (state_pool.empty()) return Ref<State>();
	return state_pool.front().value();
}

Ref<State> PushdownAutomaton::get_state_by_name(const StringName& state_name){
	auto is_available = state_pool.has(state_name);
	if (!is_available) return Ref<State>();
	for (auto E = state_pool.front(); E; E = E.next()) {
		if (E.key() == state_name) return E.value();
	}
	return Ref<State>();
}

Ref<State> PushdownAutomaton::get_next_state(const StringName& from_state){
	auto is_available = state_pool.has(from_state);
	if (!is_available) return Ref<State>();
	auto is_ready = false;
	for (auto E = state_pool.front(); E; E = E.next()) {
		if (is_ready) return E.value();
		if (E.key() == from_state) is_ready = true;
	}
	return Ref<State>();
}

Ref<State> PushdownAutomaton::get_prev_state(const StringName& from_state){
	auto is_available = state_pool.has(from_state);
	if (!is_available) return Ref<State>();
	Ref<State> last;
	for (auto E = state_pool.front(); E; E = E.next()) {
		if (E.key() == from_state) return last;
		last = E.value();
	}
	return last;
}

bool PushdownAutomaton::add_state(const Ref<State>& new_state){
	auto state_name = new_state->get_state_name();
	auto is_available = state_pool.has(state_name);
	if (is_available) return false;
	state_pool[state_name] = new_state;
	return true;
}
// bool PushdownAutomaton::add_entry_state(const Ref<State>& new_state){

// }
bool PushdownAutomaton::remove_state(const StringName& state_name){
	return state_pool.erase(state_name);
}

Dictionary PushdownAutomaton::get_all_states(){
	Dictionary re;
	for (auto E = state_pool.front(); E; E = E.next()) {
		re[Variant(E.key())] = Variant(E.get());
	}
	return re;
}

StateAutomaton::StateAutomaton(){

}

StateAutomaton::~StateAutomaton(){

}

void StateAutomaton::_bind_methods(){
	ClassDB::bind_method(D_METHOD("boot"), &StateAutomaton::boot);
	ClassDB::bind_method(D_METHOD("poll", "delta"), &StateAutomaton::poll);
	ClassDB::bind_method(D_METHOD("finalize"), &StateAutomaton::finalize);

	ClassDB::bind_method(D_METHOD("set_pda", "new_pda"), &StateAutomaton::set_pda);
	ClassDB::bind_method(D_METHOD("get_pda"), &StateAutomaton::get_pda);

	ClassDB::bind_method(D_METHOD("enable_debug", "debugging"), &StateAutomaton::enable_debug);
	ClassDB::bind_method(D_METHOD("is_debugging"), &StateAutomaton::is_debugging);

	ClassDB::bind_method(D_METHOD("set_client", "new_client"), &StateAutomaton::set_client);
	ClassDB::bind_method(D_METHOD("get_client"), &StateAutomaton::get_client);

	ClassDB::bind_method(D_METHOD("get_delta"), &StateAutomaton::get_delta);
	ClassDB::bind_method(D_METHOD("blackboard_get", "what"), &StateAutomaton::blackboard_get);
	ClassDB::bind_method(D_METHOD("blackboard_set", "what", "with"), &StateAutomaton::blackboard_set);

	ADD_PROPERTY(PropertyInfo(Variant::VARIANT_MAX, "client"), "set_client", "get_client");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_status"), "enable_debug", "is_debugging");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "pda"), "set_pda", "get_pda");
}

void StateAutomaton::boot(){
	if (pda.is_null()) return;
	auto state_pool = pda->get_state_pool();
	// List<StringName> sp_klist;
	// pda->get_state_pool_keys(&sp_klist);
	int callbacks_count = 0;
	auto self_ref = Ref<StateAutomaton>(this);
	if (debug_status){
		HUB_PRINT_SA((std::string("State count: ") + std::to_string(state_pool->size())));
	}
	for (auto E = state_pool->front(); E && callbacks_count < MAX_STATE_POLL; E = E.next()){
		// auto key = E.get();
		auto curr = E.value();
		if (curr.is_null()) continue;
		if (debug_status){
			HUB_PRINT_SA((std::string("Booting: ") + std::to_string(curr->get_instance_id())));
		}
		curr->internal_start(self_ref);
		callbacks_count++;
	}
}
void StateAutomaton::poll(const float& delta){
	if (pda.is_null()) return;
	delta_time = delta;
	auto processing = pda->get_entry_state();
	auto state_name = StringName();
	auto callbacks_count = 0;
	auto self_ref = Ref<StateAutomaton>(this);
	while (processing.is_valid() && callbacks_count < MAX_STATE_POLL){
		state_name = processing->get_state_name();
		if (debug_status){
			HUB_PRINT_SA((std::string("Polling: ") + std::to_string(processing->get_instance_id())));
		}
		auto command = processing->internal_poll(self_ref);
		if (debug_status){
			Hub::get_singleton()->print_custom("StateAutomaton", String("Returned: ") + (String)command);
		}
		// auto stop_exec = false;
		if (command == STATE_VM_POLL_FIRST_VALUE){
			processing = pda->get_entry_state();
		} else if (command == STATE_VM_POLL_NEXT_VALUE){
			processing = pda->get_next_state(state_name);
		} else if (command == STATE_VM_POLL_PREV_VALUE){
			processing = pda->get_prev_state(state_name);
		} else if (command == STATE_VM_POLL_STOP_VALUE){
			// break;
			return;
		} else {
			processing = pda->get_state_by_name(command);
		}
		// I tried, didn't work
		// switch (command.hash()){
		// 	case STATE_VM_POLL_FIRST_VALUE.hash():
		// 		processing = pda->get_entry_state();
		// 		break;
		// 	case STATE_VM_POLL_NEXT_VALUE.hash():
		// 		processing = pda->get_next_state(state_name);
		// 		break;
		// 	case STATE_VM_POLL_PREV_VALUE.hash():
		// 		processing = pda->get_prev_state(state_name);
		// 		break;
		// 	case STATE_VM_POLL_STOP_VALUE.hash():
		// 		stop_exec = true;
		// 		break;
		// 	default:
		// 		processing = pda->get_state_by_name(command);
		// 		break;
		// }
		// if (stop_exec) break;
		// if (command == STATE_VM_POLL_STOP_VALUE) break;
		// else if (command == STATE_VM_POLL_NEXT_VALUE) {
		// 	processing = pda->get_next_state(state_name);
		// }
		// else{
		// 	processing = pda->get_state_by_name(command);
		// }
		callbacks_count++;
	}

}
void StateAutomaton::finalize(){
	if (pda.is_null()) return;
	auto state_pool = pda->get_state_pool();
	// List<StringName> sp_klist;
	// pda->get_state_pool_keys(&sp_klist);
	int callbacks_count = 0;
	if (debug_status){
		HUB_PRINT_SA((std::string("State count: ") + std::to_string(state_pool->size())));
	}
	auto self_ref = Ref<StateAutomaton>(this);
	for (auto E = state_pool->front(); E && callbacks_count < MAX_STATE_POLL; E = E.next()){
		// auto key = E.get();
		auto curr = E.value();
		if (curr.is_null()) continue;
		if (debug_status){
			HUB_PRINT_SA((std::string("Finalizing: ") + std::to_string(curr->get_instance_id())));
		}
		curr->internal_finalize(self_ref);
		callbacks_count++;
	}
}

void StateAutomaton::set_pda(const Ref<PushdownAutomaton>& new_pda){
	if (new_pda == pda) return;
	pda = new_pda;
}

Variant StateAutomaton::blackboard_get(const String& what){
	return blackboard[what];
}
void StateAutomaton::blackboard_set(const String& what, const Variant& with){
	blackboard[what] = with;
}
