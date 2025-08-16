#pragma once
#include "callable.hpp"
#include "string.hpp"
#include "syscalls_fwd.hpp"

struct Object {
	/// @brief Construct an Object object from an allowed global object.
	explicit Object(const std::string &name);

	/// @brief Construct an Object object from an existing in-scope Object object.
	/// @param addr The address of the Object object.
	constexpr Object(uint64_t addr) : m_address{addr} {}

	/// Call a method on the node.
	/// @param method The method to call.
	/// @param deferred If true, the method will be called next frame.
	/// @param args The arguments to pass to the method.
	/// @return The return value of the method.
	Variant callv(std::string_view method, bool deferred, const Variant *argv, unsigned argc);

	/// Call a method on the node without returning a value.
	/// @param method The method to call.
	/// @param deferred If true, the method will be called next frame.
	/// @param args The arguments to pass to the method.
	void voidcallv(std::string_view method, bool deferred, const Variant *argv, unsigned argc);

	template <typename... Args>
	Variant call(std::string_view method, Args... args);

	template <typename... Args>
	Variant call(std::string_view method, Args... args) const;

	template <typename... Args>
	void voidcall(std::string_view method, Args... args);

	template <typename... Args>
	Variant operator () (std::string_view method, Args... args);

	template <typename... Args>
	void call_deferred(std::string_view method, Args... args);

	/// @brief Get a list of methods available on the object.
	/// @return A list of method names.
	std::vector<std::string> get_method_list() const;

	/// @brief Get a property of the object.
	/// @param name The name of the property.
	/// @return The value of the property.
	Variant get(std::string_view name) const;

	/// @brief Set a property of the object.
	/// @param name The name of the property.
	/// @param value The value to set the property to.
	void set(std::string_view name, const Variant &value);

	/// @brief Assign a value to a property of the object, at the end of the frame.
	/// @param property The name of the property.
	/// @param value The value to set the property to.
	void set_deferred(const String &property, const Variant &value) {
		voidcall("set_deferred", property, value);
	}

	// Get a list of properties available on the object.
	// @return A list of property names.
	std::vector<std::string> get_property_list() const;

	// Connect a signal to a method on another object.
	// @param signal The signal to connect.
	// @param target The object to connect to.
	// @param method The method to call when the signal is emitted.
	void connect(Object target, const std::string &signal, std::string_view method);
	void connect(const std::string &signal, std::string_view method);
	void connect(const std::string &signal, Callable method);

	// Disconnect a signal from a method on another object.
	// @param signal The signal to disconnect.
	// @param target The object to disconnect from.
	// @param method The method to disconnect.
	void disconnect(Object target, const std::string &signal, std::string_view method);
	void disconnect(const std::string &signal, std::string_view method);

	// Get a list of signals available on the object.
	// @return A list of signal names.
	std::vector<std::string> get_signal_list() const;

	// Various casts.
	Node as_node() const;
	Node2D as_node2d() const;
	Node3D as_node3d() const;

	bool operator== (const Object &other) const {
		return address() == other.address();
	}
	bool operator!= (const Object &other) const {
		return address() != other.address();
	}
	bool operator< (const Object &other) const {
		return address() < other.address();
	}

	// Other member functions
	String get_class() const {
		return this->call("get_class");
	}
	bool is_class(const String &name) const {
		return this->call("is_class", name);
	}
	String to_string() const {
		return this->call("to_string");
	}

	METHOD(Variant, _get);
	METHOD(Variant, _get_property_list);
	METHOD(void, _init);
	METHOD(void, _notification);
	METHOD(bool, _property_can_revert);
	METHOD(Variant, _property_get_revert);
	METHOD(bool, _set);
	METHOD(String, _to_string);
	METHOD(void, _validate_property);
	METHOD(void, add_user_signal);
	METHOD(bool, can_translate_messages);
	METHOD(void, cancel_free);
	METHOD(Variant, emit_signal);
	METHOD(void, free);
	METHOD(Variant, get_incoming_connections);
	METHOD(Variant, get_indexed);
	METHOD(int, get_instance_id);
	METHOD(Variant, get_meta);
	METHOD(Variant, get_meta_list);
	METHOD(int, get_method_argument_count);
	METHOD(Variant, get_script);
	METHOD(Variant, get_signal_connection_list);
	METHOD(bool, has_meta);
	METHOD(bool, has_method);
	METHOD(bool, has_signal);
	METHOD(bool, has_user_signal);
	METHOD(bool, is_blocking_signals);
	METHOD(bool, is_connected);
	METHOD(bool, is_queued_for_deletion);
	METHOD(void, notification);
	METHOD(void, notify_property_list_changed);
	METHOD(bool, property_can_revert);
	METHOD(Variant, property_get_revert);
	METHOD(void, remove_meta);
	METHOD(void, remove_user_signal);
	METHOD(void, set_block_signals);
	METHOD(void, set_indexed);
	METHOD(void, set_message_translation);
	METHOD(void, set_meta);
	METHOD(void, set_script);
	METHOD(String, tr);
	METHOD(String, tr_n);

	// Get the object identifier.
	constexpr uint64_t address() const { return m_address; }

	// Check if the node is valid.
	constexpr bool is_valid() const { return m_address != 0; }

protected:
	uint64_t m_address;
};

inline Object Variant::as_object() const {
	if (get_type() == Variant::OBJECT)
		return Object{uintptr_t(v.i)};
	api_throw("std::bad_cast", "Variant is not an Object", this);
}

inline Variant::operator Object() const {
	return as_object();
}

template <typename T>
static inline T cast_to(const Object &obj) {
	return T{obj.address()};
}

inline Variant::Variant(const Object &obj) {
	m_type = OBJECT;
	v.i = obj.address();
}

template <typename... Args>
inline Variant Object::call(std::string_view method, Args... args) {
	Variant argv[] = {args...};
	return callv(method, false, argv, sizeof...(Args));
}

template <typename... Args>
inline Variant Object::call(std::string_view method, Args... args) const {
	Variant argv[] = {args...};
	return const_cast<Object *>(this)->callv(method, false, argv, sizeof...(Args));
}

template <typename... Args>
inline void Object::voidcall(std::string_view method, Args... args) {
	Variant argv[] = {args...};
	this->voidcallv(method, false, argv, sizeof...(Args));
}

template <typename... Args>
inline Variant Object::operator () (std::string_view method, Args... args) {
	return call(method, args...);
}

template <typename... Args>
inline void Object::call_deferred(std::string_view method, Args... args) {
	Variant argv[] = {args...};
	this->voidcallv(method, true, argv, sizeof...(Args));
}

inline void Object::connect(const std::string &signal, std::string_view method) {
	this->connect(*this, signal, method);
}
inline void Object::disconnect(const std::string &signal, std::string_view method) {
	this->disconnect(*this, signal, method);
}

// This is one of the most heavily used functions in the API, so it's worth optimizing.
inline Variant Object::callv(std::string_view method, bool deferred, const Variant *argv, unsigned argc) {
	static constexpr int ECALL_OBJ_CALLP = 506; // Call a method on an object
	Variant var;
	// We will attempt to call the method using inline assembly.
	register uint64_t object asm("a0") = address();
	register const char *method_ptr asm("a1") = method.begin();
	register size_t method_size asm("a2") = method.size();
	register bool deferred_flag asm("a3") = deferred;
	register Variant *var_ptr asm("a4") = &var;
	register const Variant *argv_ptr asm("a5") = argv;
	register unsigned argc_reg asm("a6") = argc;
	register int syscall_number asm("a7") = ECALL_OBJ_CALLP;

	asm volatile(
		"ecall"
		: "=m"(*var_ptr)
		: "r"(object), "r"(method_ptr), "r"(method_size), "r"(deferred_flag), "r"(var_ptr), "r"(argv_ptr), "m"(*argv_ptr), "r"(argc_reg), "r"(syscall_number)
	);
	return var;
}

// This variation of the function is used when the return value is not needed. We simply don't pass a pointer to the return value.
inline void Object::voidcallv(std::string_view method, bool deferred, const Variant *argv, unsigned argc) {
	static constexpr int ECALL_OBJ_CALLP = 506; // Call a method on an object
	// We will attempt to call the method using inline assembly.
	register uint64_t object asm("a0") = address();
	register const char *method_ptr asm("a1") = method.begin();
	register size_t method_size asm("a2") = method.size();
	register bool deferred_flag asm("a3") = deferred;
	register Variant *var_ptr asm("a4") = nullptr;
	register const Variant *argv_ptr asm("a5") = argv;
	register unsigned argc_reg asm("a6") = argc;
	register int syscall_number asm("a7") = ECALL_OBJ_CALLP;

	asm volatile(
		"ecall"
		: /* no outputs */
		: "r"(object), "r"(method_ptr), "r"(method_size), "r"(deferred_flag), "r"(var_ptr), "r"(argv_ptr), "m"(*argv_ptr), "r"(argc_reg), "r"(syscall_number)
	);
}

inline void Object::connect(const std::string &signal, Callable method) {
	this->call("connect", signal, method);
}

template <typename T, typename P = Variant>
struct PropertyProxy {
	const T &obj;
	const std::string_view property;

	constexpr PropertyProxy(const T &obj, std::string_view property)
			: obj(obj), property(property) {}

	PropertyProxy &operator=(const P &v) {
		const_cast<T&>(obj).set(property, v);
		return *this;
	}

	bool operator== (const P &v) const { return P(obj.get(property)) == v; }
	bool operator!=(const P &v) const { return P(obj.get(property)) != v; }

	operator P() {
		return obj.get(property);
	}
};

#define PROPERTY(name, Type)      \
	auto name() { return PropertyProxy<decltype(*this), Type>(*this, #name); } \
	auto name() const { return PropertyProxy<decltype(*this), Type>(*this, #name); }

#define CUSTOM_PROPERTY(name, Type, getter, setter)      \
	auto name() { return PropertyProxy(*this, #name); } \
	auto name() const { return PropertyProxy(*this, #name); } \
	void setter(const Type &v) { Object::set(#name, v); } \
	Type getter() const { return Object::get(#name); }
