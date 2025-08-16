#include "object.hpp"

#include "syscalls.h"
#include "variant.hpp"
#include <stdexcept>

MAKE_SYSCALL(ECALL_GET_OBJ, uint64_t, sys_get_obj, const char *, size_t);
MAKE_SYSCALL(ECALL_OBJ, void, sys_obj, Object_Op, uint64_t, Variant *);
MAKE_SYSCALL(ECALL_OBJ_CALLP, void, sys_obj_callp, uint64_t, const char *, size_t, bool, Variant *, const Variant *, unsigned);
MAKE_SYSCALL(ECALL_OBJ_PROP_GET, void, sys_obj_property_get, uint64_t, const char *, size_t, Variant *);
MAKE_SYSCALL(ECALL_OBJ_PROP_SET, void, sys_obj_property_set, uint64_t, const char *, size_t, const Variant *);

static_assert(sizeof(std::vector<std::string>) == 24, "std::vector<std::string> is not 24 bytes");

Object::Object(const std::string &name) :
		m_address{ sys_get_obj(name.c_str(), name.size()) } {
}

std::vector<std::string> Object::get_method_list() const {
	if constexpr (sizeof(std::string) == 32) {
		std::vector<std::string> methods;
		sys_obj(Object_Op::GET_METHOD_LIST, address(), (Variant *)&methods);
		return methods;
	} else {
		throw std::runtime_error("Fast-path std::string is not supported on this platform");
	}
}

Variant Object::get(std::string_view name) const {
	Variant var;
#if 0
	sys_obj_property_get(address(), name.data(), name.size(), &var);
#else
	register uint64_t object asm("a0") = address();
	register const char *property asm("a1") = name.data();
	register size_t property_size asm("a2") = name.size();
	register Variant *var_ptr asm("a3") = &var;
	register int syscall_number asm("a7") = ECALL_OBJ_PROP_GET;

	asm volatile(
		"ecall"
		: "=m"(*var_ptr)
		: "r"(object), "r"(property), "m"(*property), "r"(property_size), "r"(var_ptr), "r"(syscall_number)
	);
#endif
	return var;
}

void Object::set(std::string_view name, const Variant &value) {
#if 0
	sys_obj_property_set(address(), name.data(), name.size(), &value);
#else
	register uint64_t object asm("a0") = address();
	register const char *property asm("a1") = name.data();
	register size_t property_size asm("a2") = name.size();
	register const Variant *value_ptr asm("a3") = &value;
	register int syscall_number asm("a7") = ECALL_OBJ_PROP_SET;

	asm volatile(
		"ecall"
		:
		: "r"(object), "r"(property), "m"(*property), "r"(property_size), "r"(value_ptr), "m"(*value_ptr), "r"(syscall_number)
	);
#endif
}

std::vector<std::string> Object::get_property_list() const {
	if constexpr (sizeof(std::string) == 32) {
		std::vector<std::string> properties;
		sys_obj(Object_Op::GET_PROPERTY_LIST, address(), (Variant *)&properties);
		return properties;
	} else {
		throw std::runtime_error("Fast-path std::string is not supported on this platform");
	}
}

void Object::connect(Object target, const std::string &signal, std::string_view method) {
	Variant vars[3];
	vars[0] = target.address();
	vars[1] = signal;
	vars[2] = std::string(method);
	sys_obj(Object_Op::CONNECT, address(), vars);
}

void Object::disconnect(Object target, const std::string &signal, std::string_view method) {
	Variant vars[3];
	vars[0] = target.address();
	vars[1] = signal;
	vars[2] = std::string(method);
	sys_obj(Object_Op::DISCONNECT, address(), vars);
}

std::vector<std::string> Object::get_signal_list() const {
	if constexpr (sizeof(std::string) == 32) {
		std::vector<std::string> signals;
		sys_obj(Object_Op::GET_SIGNAL_LIST, address(), (Variant *)&signals);
		return signals;
	} else {
		throw std::runtime_error("Fast-path std::string is not supported on this platform");
	}
}
