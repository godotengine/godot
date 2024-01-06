#include "luaTuple.h"

void LuaTuple::_bind_methods() {
	ClassDB::bind_static_method("LuaTuple", D_METHOD("from_array", "Array"), &LuaTuple::fromArray);

	ClassDB::bind_method(D_METHOD("push_back", "var"), &LuaTuple::pushBack);
	ClassDB::bind_method(D_METHOD("push_front", "var"), &LuaTuple::pushFront);
	ClassDB::bind_method(D_METHOD("set_value", "Index", "var"), &LuaTuple::set);
	ClassDB::bind_method(D_METHOD("clear"), &LuaTuple::clear);
	ClassDB::bind_method(D_METHOD("is_empty"), &LuaTuple::isEmpty);
	ClassDB::bind_method(D_METHOD("size"), &LuaTuple::size);
	ClassDB::bind_method(D_METHOD("pop_back"), &LuaTuple::popBack);
	ClassDB::bind_method(D_METHOD("pop_front"), &LuaTuple::popFront);
	ClassDB::bind_method(D_METHOD("get_value", "Index"), &LuaTuple::get);
	ClassDB::bind_method(D_METHOD("to_array"), &LuaTuple::toArray);
}

Ref<LuaTuple> LuaTuple::fromArray(Array elms) {
	Ref<LuaTuple> tuple;
	tuple.instantiate();
	tuple->elements = elms;
	return tuple;
}

void LuaTuple::pushBack(Variant var) {
	elements.push_back(var);
}

void LuaTuple::pushFront(Variant var) {
	elements.push_front(var);
}

void LuaTuple::set(int i, Variant var) {
	elements[i] = var;
}

void LuaTuple::clear() {
	elements.clear();
}

bool LuaTuple::isEmpty() {
	return elements.is_empty();
}

int LuaTuple::size() {
	return elements.size();
}

Variant LuaTuple::popBack() {
	return elements.pop_back();
}

Variant LuaTuple::popFront() {
	return elements.pop_front();
}

Variant LuaTuple::get(int i) {
	return elements[i];
}

Array LuaTuple::toArray() {
	return elements;
}