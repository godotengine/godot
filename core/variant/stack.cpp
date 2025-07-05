#include "stack.h"
#include "core/variant/variant.h"

Stack::Stack() {
}

Stack::~Stack() {
}

void Stack::_bind_methods() {
    ClassDB::bind_method(D_METHOD("push", "value"), &Stack::push);
    ClassDB::bind_method(D_METHOD("pop"), &Stack::pop);
    ClassDB::bind_method(D_METHOD("peek"), &Stack::peek);
    ClassDB::bind_method(D_METHOD("is_empty"), &Stack::is_empty);
    ClassDB::bind_method(D_METHOD("size"), &Stack::size);
    ClassDB::bind_method(D_METHOD("clear"), &Stack::clear);
    ClassDB::bind_method(D_METHOD("to_array"), &Stack::to_array);
}

void Stack::push(const Variant &p_value) {
    _data.push_back(p_value);
}

Variant Stack::pop() {
    ERR_FAIL_COND_V_MSG(_data.is_empty(), Variant(), "Cannot pop from empty stack.");
    return _data.pop_back();
}

Variant Stack::peek() const {
    ERR_FAIL_COND_V_MSG(_data.is_empty(), Variant(), "Cannot peek empty stack.");
    return _data.back();
}

bool Stack::is_empty() const {
    return _data.is_empty();
}

int Stack::size() const {
    return _data.size();
}

void Stack::clear() {
    _data.clear();
}

Array Stack::to_array() const {
    return _data;
}

String Stack::_to_string() const {
    return "Stack(" + _data._to_string() + ")";
}