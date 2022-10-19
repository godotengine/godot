#include "register_types.h"
#include "core/class_db.h"
// #include "core/engine.h"
#include "state_automaton.h"

void register_state_automaton_types(){
	ClassDB::register_class<StateAutomaton>();
	ClassDB::register_class<State>();
	ClassDB::register_class<PushdownAutomaton>();
}
void unregister_state_automaton_types(){

}