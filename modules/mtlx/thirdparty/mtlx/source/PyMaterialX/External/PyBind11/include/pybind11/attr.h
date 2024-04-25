/*
    pybind11/attr.h: Infrastructure for processing custom
    type and function attributes

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/common.h"
#include "cast.h"

#include <functional>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/// \addtogroup annotations
/// @{

/// Annotation for methods
struct is_method {
    handle class_;
    explicit is_method(const handle &c) : class_(c) {}
};

/// Annotation for operators
struct is_operator {};

/// Annotation for classes that cannot be subclassed
struct is_final {};

/// Annotation for parent scope
struct scope {
    handle value;
    explicit scope(const handle &s) : value(s) {}
};

/// Annotation for documentation
struct doc {
    const char *value;
    explicit doc(const char *value) : value(value) {}
};

/// Annotation for function names
struct name {
    const char *value;
    explicit name(const char *value) : value(value) {}
};

/// Annotation indicating that a function is an overload associated with a given "sibling"
struct sibling {
    handle value;
    explicit sibling(const handle &value) : value(value.ptr()) {}
};

/// Annotation indicating that a class derives from another given type
template <typename T>
struct base {

    PYBIND11_DEPRECATED(
        "base<T>() was deprecated in favor of specifying 'T' as a template argument to class_")
    base() = default;
};

/// Keep patient alive while nurse lives
template <size_t Nurse, size_t Patient>
struct keep_alive {};

/// Annotation indicating that a class is involved in a multiple inheritance relationship
struct multiple_inheritance {};

/// Annotation which enables dynamic attributes, i.e. adds `__dict__` to a class
struct dynamic_attr {};

/// Annotation which enables the buffer protocol for a type
struct buffer_protocol {};

/// Annotation which requests that a special metaclass is created for a type
struct metaclass {
    handle value;

    PYBIND11_DEPRECATED("py::metaclass() is no longer required. It's turned on by default now.")
    metaclass() = default;

    /// Override pybind11's default metaclass
    explicit metaclass(handle value) : value(value) {}
};

/// Specifies a custom callback with signature `void (PyHeapTypeObject*)` that
/// may be used to customize the Python type.
///
/// The callback is invoked immediately before `PyType_Ready`.
///
/// Note: This is an advanced interface, and uses of it may require changes to
/// work with later versions of pybind11.  You may wish to consult the
/// implementation of `make_new_python_type` in `detail/classes.h` to understand
/// the context in which the callback will be run.
struct custom_type_setup {
    using callback = std::function<void(PyHeapTypeObject *heap_type)>;

    explicit custom_type_setup(callback value) : value(std::move(value)) {}

    callback value;
};

/// Annotation that marks a class as local to the module:
struct module_local {
    const bool value;
    constexpr explicit module_local(bool v = true) : value(v) {}
};

/// Annotation to mark enums as an arithmetic type
struct arithmetic {};

/// Mark a function for addition at the beginning of the existing overload chain instead of the end
struct prepend {};

/** \rst
    A call policy which places one or more guard variables (``Ts...``) around the function call.

    For example, this definition:

    .. code-block:: cpp

        m.def("foo", foo, py::call_guard<T>());

    is equivalent to the following pseudocode:

    .. code-block:: cpp

        m.def("foo", [](args...) {
            T scope_guard;
            return foo(args...); // forwarded arguments
        });
 \endrst */
template <typename... Ts>
struct call_guard;

template <>
struct call_guard<> {
    using type = detail::void_type;
};

template <typename T>
struct call_guard<T> {
    static_assert(std::is_default_constructible<T>::value,
                  "The guard type must be default constructible");

    using type = T;
};

template <typename T, typename... Ts>
struct call_guard<T, Ts...> {
    struct type {
        T guard{}; // Compose multiple guard types with left-to-right default-constructor order
        typename call_guard<Ts...>::type next{};
    };
};

/// @} annotations

PYBIND11_NAMESPACE_BEGIN(detail)
/* Forward declarations */
enum op_id : int;
enum op_type : int;
struct undefined_t;
template <op_id id, op_type ot, typename L = undefined_t, typename R = undefined_t>
struct op_;
void keep_alive_impl(size_t Nurse, size_t Patient, function_call &call, handle ret);

/// Internal data structure which holds metadata about a keyword argument
struct argument_record {
    const char *name;  ///< Argument name
    const char *descr; ///< Human-readable version of the argument value
    handle value;      ///< Associated Python object
    bool convert : 1;  ///< True if the argument is allowed to convert when loading
    bool none : 1;     ///< True if None is allowed when loading

    argument_record(const char *name, const char *descr, handle value, bool convert, bool none)
        : name(name), descr(descr), value(value), convert(convert), none(none) {}
};

/// Internal data structure which holds metadata about a bound function (signature, overloads,
/// etc.)
struct function_record {
    function_record()
        : is_constructor(false), is_new_style_constructor(false), is_stateless(false),
          is_operator(false), is_method(false), has_args(false), has_kwargs(false),
          prepend(false) {}

    /// Function name
    char *name = nullptr; /* why no C++ strings? They generate heavier code.. */

    // User-specified documentation string
    char *doc = nullptr;

    /// Human-readable version of the function signature
    char *signature = nullptr;

    /// List of registered keyword arguments
    std::vector<argument_record> args;

    /// Pointer to lambda function which converts arguments and performs the actual call
    handle (*impl)(function_call &) = nullptr;

    /// Storage for the wrapped function pointer and captured data, if any
    void *data[3] = {};

    /// Pointer to custom destructor for 'data' (if needed)
    void (*free_data)(function_record *ptr) = nullptr;

    /// Return value policy associated with this function
    return_value_policy policy = return_value_policy::automatic;

    /// True if name == '__init__'
    bool is_constructor : 1;

    /// True if this is a new-style `__init__` defined in `detail/init.h`
    bool is_new_style_constructor : 1;

    /// True if this is a stateless function pointer
    bool is_stateless : 1;

    /// True if this is an operator (__add__), etc.
    bool is_operator : 1;

    /// True if this is a method
    bool is_method : 1;

    /// True if the function has a '*args' argument
    bool has_args : 1;

    /// True if the function has a '**kwargs' argument
    bool has_kwargs : 1;

    /// True if this function is to be inserted at the beginning of the overload resolution chain
    bool prepend : 1;

    /// Number of arguments (including py::args and/or py::kwargs, if present)
    std::uint16_t nargs;

    /// Number of leading positional arguments, which are terminated by a py::args or py::kwargs
    /// argument or by a py::kw_only annotation.
    std::uint16_t nargs_pos = 0;

    /// Number of leading arguments (counted in `nargs`) that are positional-only
    std::uint16_t nargs_pos_only = 0;

    /// Python method object
    PyMethodDef *def = nullptr;

    /// Python handle to the parent scope (a class or a module)
    handle scope;

    /// Python handle to the sibling function representing an overload chain
    handle sibling;

    /// Pointer to next overload
    function_record *next = nullptr;
};

/// Special data structure which (temporarily) holds metadata about a bound class
struct type_record {
    PYBIND11_NOINLINE type_record()
        : multiple_inheritance(false), dynamic_attr(false), buffer_protocol(false),
          default_holder(true), module_local(false), is_final(false) {}

    /// Handle to the parent scope
    handle scope;

    /// Name of the class
    const char *name = nullptr;

    // Pointer to RTTI type_info data structure
    const std::type_info *type = nullptr;

    /// How large is the underlying C++ type?
    size_t type_size = 0;

    /// What is the alignment of the underlying C++ type?
    size_t type_align = 0;

    /// How large is the type's holder?
    size_t holder_size = 0;

    /// The global operator new can be overridden with a class-specific variant
    void *(*operator_new)(size_t) = nullptr;

    /// Function pointer to class_<..>::init_instance
    void (*init_instance)(instance *, const void *) = nullptr;

    /// Function pointer to class_<..>::dealloc
    void (*dealloc)(detail::value_and_holder &) = nullptr;

    /// List of base classes of the newly created type
    list bases;

    /// Optional docstring
    const char *doc = nullptr;

    /// Custom metaclass (optional)
    handle metaclass;

    /// Custom type setup.
    custom_type_setup::callback custom_type_setup_callback;

    /// Multiple inheritance marker
    bool multiple_inheritance : 1;

    /// Does the class manage a __dict__?
    bool dynamic_attr : 1;

    /// Does the class implement the buffer protocol?
    bool buffer_protocol : 1;

    /// Is the default (unique_ptr) holder type used?
    bool default_holder : 1;

    /// Is the class definition local to the module shared object?
    bool module_local : 1;

    /// Is the class inheritable from python classes?
    bool is_final : 1;

    PYBIND11_NOINLINE void add_base(const std::type_info &base, void *(*caster)(void *) ) {
        auto *base_info = detail::get_type_info(base, false);
        if (!base_info) {
            std::string tname(base.name());
            detail::clean_type_id(tname);
            pybind11_fail("generic_type: type \"" + std::string(name)
                          + "\" referenced unknown base type \"" + tname + "\"");
        }

        if (default_holder != base_info->default_holder) {
            std::string tname(base.name());
            detail::clean_type_id(tname);
            pybind11_fail("generic_type: type \"" + std::string(name) + "\" "
                          + (default_holder ? "does not have" : "has")
                          + " a non-default holder type while its base \"" + tname + "\" "
                          + (base_info->default_holder ? "does not" : "does"));
        }

        bases.append((PyObject *) base_info->type);

#if PY_VERSION_HEX < 0x030B0000
        dynamic_attr |= base_info->type->tp_dictoffset != 0;
#else
        dynamic_attr |= (base_info->type->tp_flags & Py_TPFLAGS_MANAGED_DICT) != 0;
#endif

        if (caster) {
            base_info->implicit_casts.emplace_back(type, caster);
        }
    }
};

inline function_call::function_call(const function_record &f, handle p) : func(f), parent(p) {
    args.reserve(f.nargs);
    args_convert.reserve(f.nargs);
}

/// Tag for a new-style `__init__` defined in `detail/init.h`
struct is_new_style_constructor {};

/**
 * Partial template specializations to process custom attributes provided to
 * cpp_function_ and class_. These are either used to initialize the respective
 * fields in the type_record and function_record data structures or executed at
 * runtime to deal with custom call policies (e.g. keep_alive).
 */
template <typename T, typename SFINAE = void>
struct process_attribute;

template <typename T>
struct process_attribute_default {
    /// Default implementation: do nothing
    static void init(const T &, function_record *) {}
    static void init(const T &, type_record *) {}
    static void precall(function_call &) {}
    static void postcall(function_call &, handle) {}
};

/// Process an attribute specifying the function's name
template <>
struct process_attribute<name> : process_attribute_default<name> {
    static void init(const name &n, function_record *r) { r->name = const_cast<char *>(n.value); }
};

/// Process an attribute specifying the function's docstring
template <>
struct process_attribute<doc> : process_attribute_default<doc> {
    static void init(const doc &n, function_record *r) { r->doc = const_cast<char *>(n.value); }
};

/// Process an attribute specifying the function's docstring (provided as a C-style string)
template <>
struct process_attribute<const char *> : process_attribute_default<const char *> {
    static void init(const char *d, function_record *r) { r->doc = const_cast<char *>(d); }
    static void init(const char *d, type_record *r) { r->doc = d; }
};
template <>
struct process_attribute<char *> : process_attribute<const char *> {};

/// Process an attribute indicating the function's return value policy
template <>
struct process_attribute<return_value_policy> : process_attribute_default<return_value_policy> {
    static void init(const return_value_policy &p, function_record *r) { r->policy = p; }
};

/// Process an attribute which indicates that this is an overloaded function associated with a
/// given sibling
template <>
struct process_attribute<sibling> : process_attribute_default<sibling> {
    static void init(const sibling &s, function_record *r) { r->sibling = s.value; }
};

/// Process an attribute which indicates that this function is a method
template <>
struct process_attribute<is_method> : process_attribute_default<is_method> {
    static void init(const is_method &s, function_record *r) {
        r->is_method = true;
        r->scope = s.class_;
    }
};

/// Process an attribute which indicates the parent scope of a method
template <>
struct process_attribute<scope> : process_attribute_default<scope> {
    static void init(const scope &s, function_record *r) { r->scope = s.value; }
};

/// Process an attribute which indicates that this function is an operator
template <>
struct process_attribute<is_operator> : process_attribute_default<is_operator> {
    static void init(const is_operator &, function_record *r) { r->is_operator = true; }
};

template <>
struct process_attribute<is_new_style_constructor>
    : process_attribute_default<is_new_style_constructor> {
    static void init(const is_new_style_constructor &, function_record *r) {
        r->is_new_style_constructor = true;
    }
};

inline void check_kw_only_arg(const arg &a, function_record *r) {
    if (r->args.size() > r->nargs_pos && (!a.name || a.name[0] == '\0')) {
        pybind11_fail("arg(): cannot specify an unnamed argument after a kw_only() annotation or "
                      "args() argument");
    }
}

inline void append_self_arg_if_needed(function_record *r) {
    if (r->is_method && r->args.empty()) {
        r->args.emplace_back("self", nullptr, handle(), /*convert=*/true, /*none=*/false);
    }
}

/// Process a keyword argument attribute (*without* a default value)
template <>
struct process_attribute<arg> : process_attribute_default<arg> {
    static void init(const arg &a, function_record *r) {
        append_self_arg_if_needed(r);
        r->args.emplace_back(a.name, nullptr, handle(), !a.flag_noconvert, a.flag_none);

        check_kw_only_arg(a, r);
    }
};

/// Process a keyword argument attribute (*with* a default value)
template <>
struct process_attribute<arg_v> : process_attribute_default<arg_v> {
    static void init(const arg_v &a, function_record *r) {
        if (r->is_method && r->args.empty()) {
            r->args.emplace_back(
                "self", /*descr=*/nullptr, /*parent=*/handle(), /*convert=*/true, /*none=*/false);
        }

        if (!a.value) {
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            std::string descr("'");
            if (a.name) {
                descr += std::string(a.name) + ": ";
            }
            descr += a.type + "'";
            if (r->is_method) {
                if (r->name) {
                    descr += " in method '" + (std::string) str(r->scope) + "."
                             + (std::string) r->name + "'";
                } else {
                    descr += " in method of '" + (std::string) str(r->scope) + "'";
                }
            } else if (r->name) {
                descr += " in function '" + (std::string) r->name + "'";
            }
            pybind11_fail("arg(): could not convert default argument " + descr
                          + " into a Python object (type not registered yet?)");
#else
            pybind11_fail("arg(): could not convert default argument "
                          "into a Python object (type not registered yet?). "
                          "#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for "
                          "more information.");
#endif
        }
        r->args.emplace_back(a.name, a.descr, a.value.inc_ref(), !a.flag_noconvert, a.flag_none);

        check_kw_only_arg(a, r);
    }
};

/// Process a keyword-only-arguments-follow pseudo argument
template <>
struct process_attribute<kw_only> : process_attribute_default<kw_only> {
    static void init(const kw_only &, function_record *r) {
        append_self_arg_if_needed(r);
        if (r->has_args && r->nargs_pos != static_cast<std::uint16_t>(r->args.size())) {
            pybind11_fail("Mismatched args() and kw_only(): they must occur at the same relative "
                          "argument location (or omit kw_only() entirely)");
        }
        r->nargs_pos = static_cast<std::uint16_t>(r->args.size());
    }
};

/// Process a positional-only-argument maker
template <>
struct process_attribute<pos_only> : process_attribute_default<pos_only> {
    static void init(const pos_only &, function_record *r) {
        append_self_arg_if_needed(r);
        r->nargs_pos_only = static_cast<std::uint16_t>(r->args.size());
        if (r->nargs_pos_only > r->nargs_pos) {
            pybind11_fail("pos_only(): cannot follow a py::args() argument");
        }
        // It also can't follow a kw_only, but a static_assert in pybind11.h checks that
    }
};

/// Process a parent class attribute.  Single inheritance only (class_ itself already guarantees
/// that)
template <typename T>
struct process_attribute<T, enable_if_t<is_pyobject<T>::value>>
    : process_attribute_default<handle> {
    static void init(const handle &h, type_record *r) { r->bases.append(h); }
};

/// Process a parent class attribute (deprecated, does not support multiple inheritance)
template <typename T>
struct process_attribute<base<T>> : process_attribute_default<base<T>> {
    static void init(const base<T> &, type_record *r) { r->add_base(typeid(T), nullptr); }
};

/// Process a multiple inheritance attribute
template <>
struct process_attribute<multiple_inheritance> : process_attribute_default<multiple_inheritance> {
    static void init(const multiple_inheritance &, type_record *r) {
        r->multiple_inheritance = true;
    }
};

template <>
struct process_attribute<dynamic_attr> : process_attribute_default<dynamic_attr> {
    static void init(const dynamic_attr &, type_record *r) { r->dynamic_attr = true; }
};

template <>
struct process_attribute<custom_type_setup> {
    static void init(const custom_type_setup &value, type_record *r) {
        r->custom_type_setup_callback = value.value;
    }
};

template <>
struct process_attribute<is_final> : process_attribute_default<is_final> {
    static void init(const is_final &, type_record *r) { r->is_final = true; }
};

template <>
struct process_attribute<buffer_protocol> : process_attribute_default<buffer_protocol> {
    static void init(const buffer_protocol &, type_record *r) { r->buffer_protocol = true; }
};

template <>
struct process_attribute<metaclass> : process_attribute_default<metaclass> {
    static void init(const metaclass &m, type_record *r) { r->metaclass = m.value; }
};

template <>
struct process_attribute<module_local> : process_attribute_default<module_local> {
    static void init(const module_local &l, type_record *r) { r->module_local = l.value; }
};

/// Process a 'prepend' attribute, putting this at the beginning of the overload chain
template <>
struct process_attribute<prepend> : process_attribute_default<prepend> {
    static void init(const prepend &, function_record *r) { r->prepend = true; }
};

/// Process an 'arithmetic' attribute for enums (does nothing here)
template <>
struct process_attribute<arithmetic> : process_attribute_default<arithmetic> {};

template <typename... Ts>
struct process_attribute<call_guard<Ts...>> : process_attribute_default<call_guard<Ts...>> {};

/**
 * Process a keep_alive call policy -- invokes keep_alive_impl during the
 * pre-call handler if both Nurse, Patient != 0 and use the post-call handler
 * otherwise
 */
template <size_t Nurse, size_t Patient>
struct process_attribute<keep_alive<Nurse, Patient>>
    : public process_attribute_default<keep_alive<Nurse, Patient>> {
    template <size_t N = Nurse, size_t P = Patient, enable_if_t<N != 0 && P != 0, int> = 0>
    static void precall(function_call &call) {
        keep_alive_impl(Nurse, Patient, call, handle());
    }
    template <size_t N = Nurse, size_t P = Patient, enable_if_t<N != 0 && P != 0, int> = 0>
    static void postcall(function_call &, handle) {}
    template <size_t N = Nurse, size_t P = Patient, enable_if_t<N == 0 || P == 0, int> = 0>
    static void precall(function_call &) {}
    template <size_t N = Nurse, size_t P = Patient, enable_if_t<N == 0 || P == 0, int> = 0>
    static void postcall(function_call &call, handle ret) {
        keep_alive_impl(Nurse, Patient, call, ret);
    }
};

/// Recursively iterate over variadic template arguments
template <typename... Args>
struct process_attributes {
    static void init(const Args &...args, function_record *r) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(r);
        PYBIND11_WORKAROUND_INCORRECT_GCC_UNUSED_BUT_SET_PARAMETER(r);
        using expander = int[];
        (void) expander{
            0, ((void) process_attribute<typename std::decay<Args>::type>::init(args, r), 0)...};
    }
    static void init(const Args &...args, type_record *r) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(r);
        PYBIND11_WORKAROUND_INCORRECT_GCC_UNUSED_BUT_SET_PARAMETER(r);
        using expander = int[];
        (void) expander{0,
                        (process_attribute<typename std::decay<Args>::type>::init(args, r), 0)...};
    }
    static void precall(function_call &call) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(call);
        using expander = int[];
        (void) expander{0,
                        (process_attribute<typename std::decay<Args>::type>::precall(call), 0)...};
    }
    static void postcall(function_call &call, handle fn_ret) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(call, fn_ret);
        PYBIND11_WORKAROUND_INCORRECT_GCC_UNUSED_BUT_SET_PARAMETER(fn_ret);
        using expander = int[];
        (void) expander{
            0, (process_attribute<typename std::decay<Args>::type>::postcall(call, fn_ret), 0)...};
    }
};

template <typename T>
using is_call_guard = is_instantiation<call_guard, T>;

/// Extract the ``type`` from the first `call_guard` in `Extras...` (or `void_type` if none found)
template <typename... Extra>
using extract_guard_t = typename exactly_one_t<is_call_guard, call_guard<>, Extra...>::type;

/// Check the number of named arguments at compile time
template <typename... Extra,
          size_t named = constexpr_sum(std::is_base_of<arg, Extra>::value...),
          size_t self = constexpr_sum(std::is_same<is_method, Extra>::value...)>
constexpr bool expected_num_args(size_t nargs, bool has_args, bool has_kwargs) {
    PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(nargs, has_args, has_kwargs);
    return named == 0 || (self + named + size_t(has_args) + size_t(has_kwargs)) == nargs;
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
