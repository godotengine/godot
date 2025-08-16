#include "dictionary.hpp"

#include "syscalls.h"

EXTERN_SYSCALL(void, sys_vcreate, Variant *, int, int, ...);
MAKE_SYSCALL(ECALL_DICTIONARY_OPS, int, sys_dict_ops, Dictionary_Op, unsigned, ...);
EXTERN_SYSCALL(unsigned, sys_vassign, unsigned, unsigned);

Dictionary &Dictionary::operator=(const Dictionary &other) {
	this->m_idx = sys_vassign(this->m_idx, other.m_idx);
	return *this;
}

void Dictionary::clear() {
	(void)sys_dict_ops(Dictionary_Op::CLEAR, m_idx);
}

void Dictionary::erase(const Variant &key) {
	(void)sys_dict_ops(Dictionary_Op::ERASE, m_idx, &key);
}

bool Dictionary::has(const Variant &key) const {
	return sys_dict_ops(Dictionary_Op::HAS, m_idx, &key);
}

int Dictionary::size() const {
	return sys_dict_ops(Dictionary_Op::GET_SIZE, m_idx);
}

Variant Dictionary::get(const Variant &key) const {
	Variant v;
	(void)sys_dict_ops(Dictionary_Op::GET, m_idx, &key, &v);
	return v;
}
void Dictionary::set(const Variant &key, const Variant &value) {
	(void)sys_dict_ops(Dictionary_Op::SET, m_idx, &key, &value);
}
Variant Dictionary::get_or_add(const Variant &key, const Variant &default_value) {
	Variant v;
	(void)sys_dict_ops(Dictionary_Op::GET_OR_ADD, m_idx, &key, &v, &default_value);
	return v;
}

void Dictionary::merge(const Dictionary &other) {
	Variant v(other);
	(void)sys_dict_ops(Dictionary_Op::MERGE, m_idx, &v);
}

Dictionary Dictionary::Create() {
	Variant v;
	sys_vcreate(&v, Variant::DICTIONARY, 0);
	Dictionary d;
	d.m_idx = v.get_internal_index();
	return d;
}
