#ifndef GODOT_HPP
#define GODOT_HPP

#include <cstdlib>
#include <cstring>

#include <gdnative_api_struct.gen.h>
#include <nativescript/godot_nativescript.h>
#include <typeinfo>

#include "CoreTypes.hpp"
#include "Ref.hpp"
#include "TagDB.hpp"
#include "Variant.hpp"

#include "Object.hpp"

#include "GodotGlobal.hpp"

namespace godot {
namespace detail {

// Godot classes are wrapped by heap-allocated instances mimicking them through the C API.
// They all inherit `_Wrapped`.
template <class T>
T *get_wrapper(godot_object *obj) {
	return (T *)godot::nativescript_1_1_api->godot_nativescript_get_instance_binding_data(godot::_RegisterState::language_index, obj);
}

// Custom class instances are not obtainable by just casting the pointer to the base class they inherit,
// partly because in Godot, scripts are not instances of the classes themselves, they are only attached to them.
// Yet we want to "fake" it as if they were the same entity.
template <class T>
T *get_custom_class_instance(const Object *obj) {
	return (obj) ? (T *)godot::nativescript_api->godot_nativescript_get_userdata(obj->_owner) : nullptr;
}

template <class T>
inline T *create_custom_class_instance() {
	// Usually, script instances hold a reference to their NativeScript resource.
	// that resource is obtained from a `.gdns` file, which in turn exists because
	// of the resource system of Godot. We can't cleanly hardcode that here,
	// so the easiest for now (though not really clean) is to create new resource instances,
	// individually attached to the script instances.

	// We cannot use wrappers because of https://github.com/godotengine/godot/issues/39181
	//	godot::NativeScript *script = godot::NativeScript::_new();
	//	script->set_library(get_wrapper<godot::GDNativeLibrary>((godot_object *)godot::gdnlib));
	//	script->set_class_name(T::___get_type_name());

	// So we use the C API directly.
	static godot_class_constructor script_constructor = godot::api->godot_get_class_constructor("NativeScript");
	static godot_method_bind *mb_set_library = godot::api->godot_method_bind_get_method("NativeScript", "set_library");
	static godot_method_bind *mb_set_class_name = godot::api->godot_method_bind_get_method("NativeScript", "set_class_name");
	godot_object *script = script_constructor();
	{
		const void *args[] = { godot::gdnlib };
		godot::api->godot_method_bind_ptrcall(mb_set_library, script, args, nullptr);
	}
	{
		const String class_name = T::___get_type_name();
		const void *args[] = { &class_name };
		godot::api->godot_method_bind_ptrcall(mb_set_class_name, script, args, nullptr);
	}

	// Now to instanciate T, we initially did this, however in case of Reference it returns a variant with refcount
	// already initialized, which woud cause inconsistent behavior compared to other classes (we still have to return a pointer).
	//Variant instance_variant = script->new_();
	//T *instance = godot::get_custom_class_instance<T>(instance_variant);

	// So we should do this instead, however while convenient, it uses unnecessary wrapper objects.
	//	Object *base_obj = T::___new_godot_base();
	//	base_obj->set_script(script);
	//	return get_custom_class_instance<T>(base_obj);

	// Again using the C API to do exactly what we have to do.
	static godot_class_constructor base_constructor = godot::api->godot_get_class_constructor(T::___get_godot_base_class_name());
	static godot_method_bind *mb_set_script = godot::api->godot_method_bind_get_method("Object", "set_script");
	godot_object *base_obj = base_constructor();
	{
		const void *args[] = { script };
		godot::api->godot_method_bind_ptrcall(mb_set_script, base_obj, args, nullptr);
	}

	return (T *)godot::nativescript_api->godot_nativescript_get_userdata(base_obj);
}

} // namespace detail

// Used in the definition of a custom class where the base is a Godot class
#define GODOT_CLASS(Name, Base)                                                             \
                                                                                            \
public:                                                                                     \
	inline static const char *___get_type_name() { return #Name; }                          \
	enum { ___CLASS_IS_SCRIPT = 1 };                                                        \
	inline static const char *___get_godot_base_class_name() {                              \
		return Base::___get_class_name();                                                   \
	}                                                                                       \
	inline static Name *_new() {                                                            \
		return godot::detail::create_custom_class_instance<Name>();                         \
	}                                                                                       \
	inline static size_t ___get_id() { return typeid(Name).hash_code(); }                   \
	inline static size_t ___get_base_id() { return Base::___get_id(); }                     \
	inline static const char *___get_base_type_name() { return Base::___get_class_name(); } \
	inline static godot::Object *___get_from_variant(godot::Variant a) {                    \
		return (godot::Object *)godot::detail::get_custom_class_instance<Name>(             \
				godot::Object::___get_from_variant(a));                                     \
	}                                                                                       \
                                                                                            \
private:

// Used in the definition of a custom class where the base is another custom class
#define GODOT_SUBCLASS(Name, Base)                                                         \
                                                                                           \
public:                                                                                    \
	inline static const char *___get_type_name() { return #Name; }                         \
	enum { ___CLASS_IS_SCRIPT = 1 };                                                       \
	inline static const char *___get_godot_base_class_name() {                             \
		return Base::___get_godot_base_class_name();                                       \
	}                                                                                      \
	inline static Name *_new() {                                                           \
		return godot::detail::create_custom_class_instance<Name>();                        \
	}                                                                                      \
	inline static size_t ___get_id() { return typeid(Name).hash_code(); };                 \
	inline static size_t ___get_base_id() { return typeid(Base).hash_code(); };            \
	inline static const char *___get_base_type_name() { return Base::___get_type_name(); } \
	inline static godot::Object *___get_from_variant(godot::Variant a) {                   \
		return (godot::Object *)godot::detail::get_custom_class_instance<Name>(            \
				godot::Object::___get_from_variant(a));                                    \
	}                                                                                      \
                                                                                           \
private:

template <class T>
struct _ArgCast {
	static T _arg_cast(Variant a) {
		return a;
	}
};

template <class T>
struct _ArgCast<T *> {
	static T *_arg_cast(Variant a) {
		return (T *)T::___get_from_variant(a);
	}
};

template <>
struct _ArgCast<Variant> {
	static Variant _arg_cast(Variant a) {
		return a;
	}
};

// instance and destroy funcs

template <class T>
void *_godot_class_instance_func(godot_object *p, void *method_data) {
	T *d = new T();
	d->_owner = p;
	d->_type_tag = typeid(T).hash_code();
	d->_init();
	return d;
}

template <class T>
void _godot_class_destroy_func(godot_object *p, void *method_data, void *data) {
	T *d = (T *)data;
	delete d;
}

template <class T>
void register_class() {
	godot_instance_create_func create = {};
	create.create_func = _godot_class_instance_func<T>;

	godot_instance_destroy_func destroy = {};
	destroy.destroy_func = _godot_class_destroy_func<T>;

	_TagDB::register_type(T::___get_id(), T::___get_base_id());

	godot::nativescript_api->godot_nativescript_register_class(godot::_RegisterState::nativescript_handle, T::___get_type_name(), T::___get_base_type_name(), create, destroy);
	godot::nativescript_1_1_api->godot_nativescript_set_type_tag(godot::_RegisterState::nativescript_handle, T::___get_type_name(), (const void *)typeid(T).hash_code());
	T::_register_methods();
}

template <class T>
void register_tool_class() {
	godot_instance_create_func create = {};
	create.create_func = _godot_class_instance_func<T>;

	godot_instance_destroy_func destroy = {};
	destroy.destroy_func = _godot_class_destroy_func<T>;

	_TagDB::register_type(T::___get_id(), T::___get_base_id());

	godot::nativescript_api->godot_nativescript_register_tool_class(godot::_RegisterState::nativescript_handle, T::___get_type_name(), T::___get_base_type_name(), create, destroy);
	godot::nativescript_1_1_api->godot_nativescript_set_type_tag(godot::_RegisterState::nativescript_handle, T::___get_type_name(), (const void *)typeid(T).hash_code());
	T::_register_methods();
}

// method registering

typedef godot_variant (*__godot_wrapper_method)(godot_object *, void *, void *, int, godot_variant **);

template <class T, class R, class... args>
const char *___get_method_class_name(R (T::*p)(args... a)) {
	return T::___get_type_name();
}

template <class T, class R, class... args>
const char *___get_method_class_name(R (T::*p)(args... a) const) {
	return T::___get_type_name();
}

// Okay, time for some template magic.
// Many thanks to manpat from the GDL Discord Server.

// This is stuff that's available in C++14 I think, but whatever.

template <int... I>
struct __Sequence {};

template <int N, int... I>
struct __construct_sequence {
	using type = typename __construct_sequence<N - 1, N - 1, I...>::type;
};

template <int... I>
struct __construct_sequence<0, I...> {
	using type = __Sequence<I...>;
};

// Now the wrapping part.
template <class T, class R, class... As>
struct _WrappedMethod {
	R(T::*f)
	(As...);

	template <int... I>
	void apply(Variant *ret, T *obj, Variant **args, __Sequence<I...>) {
		*ret = (obj->*f)(_ArgCast<As>::_arg_cast(*args[I])...);
	}
};

template <class T, class... As>
struct _WrappedMethod<T, void, As...> {
	void (T::*f)(As...);

	template <int... I>
	void apply(Variant *ret, T *obj, Variant **args, __Sequence<I...>) {
		(obj->*f)(_ArgCast<As>::_arg_cast(*args[I])...);
	}
};

template <class T, class R, class... As>
godot_variant __wrapped_method(godot_object *, void *method_data, void *user_data, int num_args, godot_variant **args) {
	godot_variant v;
	godot::api->godot_variant_new_nil(&v);

	T *obj = (T *)user_data;
	_WrappedMethod<T, R, As...> *method = (_WrappedMethod<T, R, As...> *)method_data;

	Variant *var = (Variant *)&v;
	Variant **arg = (Variant **)args;

	method->apply(var, obj, arg, typename __construct_sequence<sizeof...(As)>::type{});

	return v;
}

template <class T, class R, class... As>
void *___make_wrapper_function(R (T::*f)(As...)) {
	using MethodType = _WrappedMethod<T, R, As...>;
	MethodType *p = (MethodType *)godot::api->godot_alloc(sizeof(MethodType));
	p->f = f;
	return (void *)p;
}

template <class T, class R, class... As>
__godot_wrapper_method ___get_wrapper_function(R (T::*f)(As...)) {
	return (__godot_wrapper_method)&__wrapped_method<T, R, As...>;
}

template <class T, class R, class... A>
void *___make_wrapper_function(R (T::*f)(A...) const) {
	return ___make_wrapper_function((R(T::*)(A...))f);
}

template <class T, class R, class... A>
__godot_wrapper_method ___get_wrapper_function(R (T::*f)(A...) const) {
	return ___get_wrapper_function((R(T::*)(A...))f);
}

template <class M>
void register_method(const char *name, M method_ptr, godot_method_rpc_mode rpc_type = GODOT_METHOD_RPC_MODE_DISABLED) {
	godot_instance_method method = {};
	method.method_data = ___make_wrapper_function(method_ptr);
	method.free_func = godot::api->godot_free;
	method.method = (__godot_wrapper_method)___get_wrapper_function(method_ptr);

	godot_method_attributes attr = {};
	attr.rpc_type = rpc_type;

	godot::nativescript_api->godot_nativescript_register_method(godot::_RegisterState::nativescript_handle, ___get_method_class_name(method_ptr), name, attr, method);
}

// User can specify a derived class D to register the method for, instead of it being inferred.
template <class D, class B, class R, class... As>
void register_method_explicit(const char *name, R (B::*method_ptr)(As...), godot_method_rpc_mode rpc_type = GODOT_METHOD_RPC_MODE_DISABLED) {
	static_assert(std::is_base_of<B, D>::value, "Explicit class must derive from method class");
	register_method(name, static_cast<R (D::*)(As...)>(method_ptr), rpc_type);
}

template <class T, class P>
struct _PropertySetFunc {
	void (T::*f)(P);
	static void _wrapped_setter(godot_object *object, void *method_data, void *user_data, godot_variant *value) {
		_PropertySetFunc<T, P> *set_func = (_PropertySetFunc<T, P> *)method_data;
		T *obj = (T *)user_data;

		Variant *v = (Variant *)value;

		(obj->*(set_func->f))(_ArgCast<P>::_arg_cast(*v));
	}
};

template <class T, class P>
struct _PropertyGetFunc {
	P(T::*f)
	();
	static godot_variant _wrapped_getter(godot_object *object, void *method_data, void *user_data) {
		_PropertyGetFunc<T, P> *get_func = (_PropertyGetFunc<T, P> *)method_data;
		T *obj = (T *)user_data;

		godot_variant var;
		godot::api->godot_variant_new_nil(&var);

		Variant *v = (Variant *)&var;

		*v = (obj->*(get_func->f))();

		return var;
	}
};

template <class T, class P>
struct _PropertyDefaultSetFunc {
	P(T::*f);
	static void _wrapped_setter(godot_object *object, void *method_data, void *user_data, godot_variant *value) {
		_PropertyDefaultSetFunc<T, P> *set_func = (_PropertyDefaultSetFunc<T, P> *)method_data;
		T *obj = (T *)user_data;

		Variant *v = (Variant *)value;

		(obj->*(set_func->f)) = _ArgCast<P>::_arg_cast(*v);
	}
};

template <class T, class P>
struct _PropertyDefaultGetFunc {
	P(T::*f);
	static godot_variant _wrapped_getter(godot_object *object, void *method_data, void *user_data) {
		_PropertyDefaultGetFunc<T, P> *get_func = (_PropertyDefaultGetFunc<T, P> *)method_data;
		T *obj = (T *)user_data;

		godot_variant var;
		godot::api->godot_variant_new_nil(&var);

		Variant *v = (Variant *)&var;

		*v = (obj->*(get_func->f));

		return var;
	}
};

template <class T, class P>
void register_property(const char *name, P(T::*var), P default_value, godot_method_rpc_mode rpc_mode = GODOT_METHOD_RPC_MODE_DISABLED, godot_property_usage_flags usage = GODOT_PROPERTY_USAGE_DEFAULT, godot_property_hint hint = GODOT_PROPERTY_HINT_NONE, String hint_string = "") {
	Variant def_val = default_value;

	usage = (godot_property_usage_flags)((int)usage | GODOT_PROPERTY_USAGE_SCRIPT_VARIABLE);

	if (def_val.get_type() == Variant::OBJECT) {
		Object *o = detail::get_wrapper<Object>(def_val.operator godot_object *());
		if (o && o->is_class("Resource")) {
			hint = (godot_property_hint)((int)hint | GODOT_PROPERTY_HINT_RESOURCE_TYPE);
			hint_string = o->get_class();
		}
	}

	godot_string *_hint_string = (godot_string *)&hint_string;

	godot_property_attributes attr = {};
	if (def_val.get_type() == Variant::NIL) {
		attr.type = Variant::OBJECT;
	} else {
		attr.type = def_val.get_type();
		attr.default_value = *(godot_variant *)&def_val;
	}

	attr.hint = hint;
	attr.rset_type = rpc_mode;
	attr.usage = usage;
	attr.hint_string = *_hint_string;

	_PropertyDefaultSetFunc<T, P> *wrapped_set = (_PropertyDefaultSetFunc<T, P> *)godot::api->godot_alloc(sizeof(_PropertyDefaultSetFunc<T, P>));
	wrapped_set->f = var;

	_PropertyDefaultGetFunc<T, P> *wrapped_get = (_PropertyDefaultGetFunc<T, P> *)godot::api->godot_alloc(sizeof(_PropertyDefaultGetFunc<T, P>));
	wrapped_get->f = var;

	godot_property_set_func set_func = {};
	set_func.method_data = (void *)wrapped_set;
	set_func.free_func = godot::api->godot_free;
	set_func.set_func = &_PropertyDefaultSetFunc<T, P>::_wrapped_setter;

	godot_property_get_func get_func = {};
	get_func.method_data = (void *)wrapped_get;
	get_func.free_func = godot::api->godot_free;
	get_func.get_func = &_PropertyDefaultGetFunc<T, P>::_wrapped_getter;

	godot::nativescript_api->godot_nativescript_register_property(godot::_RegisterState::nativescript_handle, T::___get_type_name(), name, &attr, set_func, get_func);
}

template <class T, class P>
void register_property(const char *name, void (T::*setter)(P), P (T::*getter)(), P default_value, godot_method_rpc_mode rpc_mode = GODOT_METHOD_RPC_MODE_DISABLED, godot_property_usage_flags usage = GODOT_PROPERTY_USAGE_DEFAULT, godot_property_hint hint = GODOT_PROPERTY_HINT_NONE, String hint_string = "") {
	Variant def_val = default_value;

	godot_string *_hint_string = (godot_string *)&hint_string;

	godot_property_attributes attr = {};
	if (def_val.get_type() == Variant::NIL) {
		attr.type = Variant::OBJECT;
	} else {
		attr.type = def_val.get_type();
		attr.default_value = *(godot_variant *)&def_val;
	}
	attr.hint = hint;
	attr.rset_type = rpc_mode;
	attr.usage = usage;
	attr.hint_string = *_hint_string;

	_PropertySetFunc<T, P> *wrapped_set = (_PropertySetFunc<T, P> *)godot::api->godot_alloc(sizeof(_PropertySetFunc<T, P>));
	wrapped_set->f = setter;

	_PropertyGetFunc<T, P> *wrapped_get = (_PropertyGetFunc<T, P> *)godot::api->godot_alloc(sizeof(_PropertyGetFunc<T, P>));
	wrapped_get->f = getter;

	godot_property_set_func set_func = {};
	set_func.method_data = (void *)wrapped_set;
	set_func.free_func = godot::api->godot_free;
	set_func.set_func = &_PropertySetFunc<T, P>::_wrapped_setter;

	godot_property_get_func get_func = {};
	get_func.method_data = (void *)wrapped_get;
	get_func.free_func = godot::api->godot_free;
	get_func.get_func = &_PropertyGetFunc<T, P>::_wrapped_getter;

	godot::nativescript_api->godot_nativescript_register_property(godot::_RegisterState::nativescript_handle, T::___get_type_name(), name, &attr, set_func, get_func);
}

template <class T, class P>
void register_property(const char *name, void (T::*setter)(P), P (T::*getter)() const, P default_value, godot_method_rpc_mode rpc_mode = GODOT_METHOD_RPC_MODE_DISABLED, godot_property_usage_flags usage = GODOT_PROPERTY_USAGE_DEFAULT, godot_property_hint hint = GODOT_PROPERTY_HINT_NONE, String hint_string = "") {
	register_property(name, setter, (P(T::*)())getter, default_value, rpc_mode, usage, hint, hint_string);
}

template <class T>
void register_signal(String name, Dictionary args = Dictionary()) {
	godot_signal signal = {};
	signal.name = *(godot_string *)&name;
	signal.num_args = args.size();
	signal.num_default_args = 0;

	// Need to check because malloc(0) is platform-dependent. Zero arguments will leave args to nullptr.
	if (signal.num_args != 0) {
		signal.args = (godot_signal_argument *)godot::api->godot_alloc(sizeof(godot_signal_argument) * signal.num_args);
		memset((void *)signal.args, 0, sizeof(godot_signal_argument) * signal.num_args);
	}

	for (int i = 0; i < signal.num_args; i++) {
		// Array entry = args[i];
		// String name = entry[0];
		String name = args.keys()[i];
		godot_string *_key = (godot_string *)&name;
		godot::api->godot_string_new_copy(&signal.args[i].name, _key);

		// if (entry.size() > 1) {
		// 	signal.args[i].type = entry[1];
		// }
		signal.args[i].type = args.values()[i];
	}

	godot::nativescript_api->godot_nativescript_register_signal(godot::_RegisterState::nativescript_handle, T::___get_type_name(), &signal);

	for (int i = 0; i < signal.num_args; i++) {
		godot::api->godot_string_destroy(&signal.args[i].name);
	}

	if (signal.args) {
		godot::api->godot_free(signal.args);
	}
}

template <class T, class... Args>
void register_signal(String name, Args... varargs) {
	register_signal<T>(name, Dictionary::make(varargs...));
}

#ifndef GODOT_CPP_NO_OBJECT_CAST
template <class T>
T *Object::cast_to(const Object *obj) {
	if (!obj)
		return nullptr;

	if (T::___CLASS_IS_SCRIPT) {
		size_t have_tag = (size_t)godot::nativescript_1_1_api->godot_nativescript_get_type_tag(obj->_owner);
		if (have_tag) {
			if (!godot::_TagDB::is_type_known((size_t)have_tag)) {
				have_tag = 0;
			}
		}

		if (!have_tag) {
			have_tag = obj->_type_tag;
		}

		if (godot::_TagDB::is_type_compatible(T::___get_id(), have_tag)) {
			return detail::get_custom_class_instance<T>(obj);
		}
	} else {
		if (godot::core_1_2_api->godot_object_cast_to(obj->_owner, (void *)T::___get_id())) {
			return (T *)obj;
		}
	}

	return nullptr;
}
#endif

} // namespace godot

#endif // GODOT_H
