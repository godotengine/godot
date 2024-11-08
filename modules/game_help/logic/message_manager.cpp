#include "message_manager.h"


MessageManager* MessageManager::singleton = nullptr;
void MessageManager::_bind_methods() {

}

MessageManager::MessageManager() {

}
MessageManager::~MessageManager() {

}

void MessageManager::emit(const StringName &p_message, Array p_args ) {
    
    List<Callable>* callables = messages.getptr(p_message);
    if(callables == nullptr) {
        return;
    }
    const Variant **p_args_ptr = (const Variant **)alloca(sizeof(Variant*) * p_args.size());
    for (int i = 0; i < p_args.size(); i++) {
        p_args_ptr[i] = &p_args[i];
    }
    Variant rs;
    Callable::CallError r_error;
    auto it = callables->begin();
    while(it != callables->end()) {
        if(!it->is_valid()) {
            it = callables->erase(it);
            continue;
        }

        it->callp(p_args_ptr, p_args.size(),rs,r_error);
        if(r_error.error != Callable::CallError::CALL_OK) {
            it = callables->erase(it);
        }
        else {
            ++it;
        }
    }

}
void MessageManager::register_message(const StringName &p_message, const Callable &p_callable) {
    List<Callable>* callables = messages.getptr(p_message);
    if(callables == nullptr) {
        callables = memnew(List<Callable>);
        messages[p_message] = *callables;
    }
	List<Callable>::Element* it = callables->find(p_callable);
    if(it != nullptr) {
        callables->push_back(p_callable);        
    }

}
void MessageManager::unregister_message(const StringName &p_message, const Callable &p_callable) {
    List<Callable>* callables = messages.getptr(p_message);
    if(callables == nullptr) {
        return;
    }
	List<Callable>::Element* it = callables->find(p_callable);
    if(it != nullptr) {
        it->get().clear();
    }

}
void MessageManager::clear() {
    messages.clear();
}
