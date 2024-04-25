/*
    pybind11/detail/type_caster_base.h (originally first part of pybind11/cast.h)

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "../pytypes.h"
#include "common.h"
#include "descr.h"
#include "internals.h"
#include "typeid.h"

#include <cstdint>
#include <iterator>
#include <new>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/// A life support system for temporary objects created by `type_caster::load()`.
/// Adding a patient will keep it alive up until the enclosing function returns.
class loader_life_support {
private:
    loader_life_support *parent = nullptr;
    std::unordered_set<PyObject *> keep_alive;

#if defined(WITH_THREAD)
    // Store stack pointer in thread-local storage.
    static PYBIND11_TLS_KEY_REF get_stack_tls_key() {
#    if PYBIND11_INTERNALS_VERSION == 4
        return get_local_internals().loader_life_support_tls_key;
#    else
        return get_internals().loader_life_support_tls_key;
#    endif
    }
    static loader_life_support *get_stack_top() {
        return static_cast<loader_life_support *>(PYBIND11_TLS_GET_VALUE(get_stack_tls_key()));
    }
    static void set_stack_top(loader_life_support *value) {
        PYBIND11_TLS_REPLACE_VALUE(get_stack_tls_key(), value);
    }
#else
    // Use single global variable for stack.
    static loader_life_support **get_stack_pp() {
        static loader_life_support *global_stack = nullptr;
        return global_stack;
    }
    static loader_life_support *get_stack_top() { return *get_stack_pp(); }
    static void set_stack_top(loader_life_support *value) { *get_stack_pp() = value; }
#endif

public:
    /// A new patient frame is created when a function is entered
    loader_life_support() : parent{get_stack_top()} { set_stack_top(this); }

    /// ... and destroyed after it returns
    ~loader_life_support() {
        if (get_stack_top() != this) {
            pybind11_fail("loader_life_support: internal error");
        }
        set_stack_top(parent);
        for (auto *item : keep_alive) {
            Py_DECREF(item);
        }
    }

    /// This can only be used inside a pybind11-bound function, either by `argument_loader`
    /// at argument preparation time or by `py::cast()` at execution time.
    PYBIND11_NOINLINE static void add_patient(handle h) {
        loader_life_support *frame = get_stack_top();
        if (!frame) {
            // NOTE: It would be nice to include the stack frames here, as this indicates
            // use of pybind11::cast<> outside the normal call framework, finding such
            // a location is challenging. Developers could consider printing out
            // stack frame addresses here using something like __builtin_frame_address(0)
            throw cast_error("When called outside a bound function, py::cast() cannot "
                             "do Python -> C++ conversions which require the creation "
                             "of temporary values");
        }

        if (frame->keep_alive.insert(h.ptr()).second) {
            Py_INCREF(h.ptr());
        }
    }
};

// Gets the cache entry for the given type, creating it if necessary.  The return value is the pair
// returned by emplace, i.e. an iterator for the entry and a bool set to `true` if the entry was
// just created.
inline std::pair<decltype(internals::registered_types_py)::iterator, bool>
all_type_info_get_cache(PyTypeObject *type);

// Populates a just-created cache entry.
PYBIND11_NOINLINE void all_type_info_populate(PyTypeObject *t, std::vector<type_info *> &bases) {
    std::vector<PyTypeObject *> check;
    for (handle parent : reinterpret_borrow<tuple>(t->tp_bases)) {
        check.push_back((PyTypeObject *) parent.ptr());
    }

    auto const &type_dict = get_internals().registered_types_py;
    for (size_t i = 0; i < check.size(); i++) {
        auto *type = check[i];
        // Ignore Python2 old-style class super type:
        if (!PyType_Check((PyObject *) type)) {
            continue;
        }

        // Check `type` in the current set of registered python types:
        auto it = type_dict.find(type);
        if (it != type_dict.end()) {
            // We found a cache entry for it, so it's either pybind-registered or has pre-computed
            // pybind bases, but we have to make sure we haven't already seen the type(s) before:
            // we want to follow Python/virtual C++ rules that there should only be one instance of
            // a common base.
            for (auto *tinfo : it->second) {
                // NB: Could use a second set here, rather than doing a linear search, but since
                // having a large number of immediate pybind11-registered types seems fairly
                // unlikely, that probably isn't worthwhile.
                bool found = false;
                for (auto *known : bases) {
                    if (known == tinfo) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    bases.push_back(tinfo);
                }
            }
        } else if (type->tp_bases) {
            // It's some python type, so keep follow its bases classes to look for one or more
            // registered types
            if (i + 1 == check.size()) {
                // When we're at the end, we can pop off the current element to avoid growing
                // `check` when adding just one base (which is typical--i.e. when there is no
                // multiple inheritance)
                check.pop_back();
                i--;
            }
            for (handle parent : reinterpret_borrow<tuple>(type->tp_bases)) {
                check.push_back((PyTypeObject *) parent.ptr());
            }
        }
    }
}

/**
 * Extracts vector of type_info pointers of pybind-registered roots of the given Python type.  Will
 * be just 1 pybind type for the Python type of a pybind-registered class, or for any Python-side
 * derived class that uses single inheritance.  Will contain as many types as required for a Python
 * class that uses multiple inheritance to inherit (directly or indirectly) from multiple
 * pybind-registered classes.  Will be empty if neither the type nor any base classes are
 * pybind-registered.
 *
 * The value is cached for the lifetime of the Python type.
 */
inline const std::vector<detail::type_info *> &all_type_info(PyTypeObject *type) {
    auto ins = all_type_info_get_cache(type);
    if (ins.second) {
        // New cache entry: populate it
        all_type_info_populate(type, ins.first->second);
    }

    return ins.first->second;
}

/**
 * Gets a single pybind11 type info for a python type.  Returns nullptr if neither the type nor any
 * ancestors are pybind11-registered.  Throws an exception if there are multiple bases--use
 * `all_type_info` instead if you want to support multiple bases.
 */
PYBIND11_NOINLINE detail::type_info *get_type_info(PyTypeObject *type) {
    const auto &bases = all_type_info(type);
    if (bases.empty()) {
        return nullptr;
    }
    if (bases.size() > 1) {
        pybind11_fail(
            "pybind11::detail::get_type_info: type has multiple pybind11-registered bases");
    }
    return bases.front();
}

inline detail::type_info *get_local_type_info(const std::type_index &tp) {
    auto &locals = get_local_internals().registered_types_cpp;
    auto it = locals.find(tp);
    if (it != locals.end()) {
        return it->second;
    }
    return nullptr;
}

inline detail::type_info *get_global_type_info(const std::type_index &tp) {
    auto &types = get_internals().registered_types_cpp;
    auto it = types.find(tp);
    if (it != types.end()) {
        return it->second;
    }
    return nullptr;
}

/// Return the type info for a given C++ type; on lookup failure can either throw or return
/// nullptr.
PYBIND11_NOINLINE detail::type_info *get_type_info(const std::type_index &tp,
                                                   bool throw_if_missing = false) {
    if (auto *ltype = get_local_type_info(tp)) {
        return ltype;
    }
    if (auto *gtype = get_global_type_info(tp)) {
        return gtype;
    }

    if (throw_if_missing) {
        std::string tname = tp.name();
        detail::clean_type_id(tname);
        pybind11_fail("pybind11::detail::get_type_info: unable to find type info for \""
                      + std::move(tname) + '"');
    }
    return nullptr;
}

PYBIND11_NOINLINE handle get_type_handle(const std::type_info &tp, bool throw_if_missing) {
    detail::type_info *type_info = get_type_info(tp, throw_if_missing);
    return handle(type_info ? ((PyObject *) type_info->type) : nullptr);
}

// Searches the inheritance graph for a registered Python instance, using all_type_info().
PYBIND11_NOINLINE handle find_registered_python_instance(void *src,
                                                         const detail::type_info *tinfo) {
    auto it_instances = get_internals().registered_instances.equal_range(src);
    for (auto it_i = it_instances.first; it_i != it_instances.second; ++it_i) {
        for (auto *instance_type : detail::all_type_info(Py_TYPE(it_i->second))) {
            if (instance_type && same_type(*instance_type->cpptype, *tinfo->cpptype)) {
                return handle((PyObject *) it_i->second).inc_ref();
            }
        }
    }
    return handle();
}

struct value_and_holder {
    instance *inst = nullptr;
    size_t index = 0u;
    const detail::type_info *type = nullptr;
    void **vh = nullptr;

    // Main constructor for a found value/holder:
    value_and_holder(instance *i, const detail::type_info *type, size_t vpos, size_t index)
        : inst{i}, index{index}, type{type}, vh{inst->simple_layout
                                                    ? inst->simple_value_holder
                                                    : &inst->nonsimple.values_and_holders[vpos]} {}

    // Default constructor (used to signal a value-and-holder not found by get_value_and_holder())
    value_and_holder() = default;

    // Used for past-the-end iterator
    explicit value_and_holder(size_t index) : index{index} {}

    template <typename V = void>
    V *&value_ptr() const {
        return reinterpret_cast<V *&>(vh[0]);
    }
    // True if this `value_and_holder` has a non-null value pointer
    explicit operator bool() const { return value_ptr() != nullptr; }

    template <typename H>
    H &holder() const {
        return reinterpret_cast<H &>(vh[1]);
    }
    bool holder_constructed() const {
        return inst->simple_layout
                   ? inst->simple_holder_constructed
                   : (inst->nonsimple.status[index] & instance::status_holder_constructed) != 0u;
    }
    // NOLINTNEXTLINE(readability-make-member-function-const)
    void set_holder_constructed(bool v = true) {
        if (inst->simple_layout) {
            inst->simple_holder_constructed = v;
        } else if (v) {
            inst->nonsimple.status[index] |= instance::status_holder_constructed;
        } else {
            inst->nonsimple.status[index] &= (std::uint8_t) ~instance::status_holder_constructed;
        }
    }
    bool instance_registered() const {
        return inst->simple_layout
                   ? inst->simple_instance_registered
                   : ((inst->nonsimple.status[index] & instance::status_instance_registered) != 0);
    }
    // NOLINTNEXTLINE(readability-make-member-function-const)
    void set_instance_registered(bool v = true) {
        if (inst->simple_layout) {
            inst->simple_instance_registered = v;
        } else if (v) {
            inst->nonsimple.status[index] |= instance::status_instance_registered;
        } else {
            inst->nonsimple.status[index] &= (std::uint8_t) ~instance::status_instance_registered;
        }
    }
};

// Container for accessing and iterating over an instance's values/holders
struct values_and_holders {
private:
    instance *inst;
    using type_vec = std::vector<detail::type_info *>;
    const type_vec &tinfo;

public:
    explicit values_and_holders(instance *inst)
        : inst{inst}, tinfo(all_type_info(Py_TYPE(inst))) {}

    struct iterator {
    private:
        instance *inst = nullptr;
        const type_vec *types = nullptr;
        value_and_holder curr;
        friend struct values_and_holders;
        iterator(instance *inst, const type_vec *tinfo)
            : inst{inst}, types{tinfo},
              curr(inst /* instance */,
                   types->empty() ? nullptr : (*types)[0] /* type info */,
                   0, /* vpos: (non-simple types only): the first vptr comes first */
                   0 /* index */) {}
        // Past-the-end iterator:
        explicit iterator(size_t end) : curr(end) {}

    public:
        bool operator==(const iterator &other) const { return curr.index == other.curr.index; }
        bool operator!=(const iterator &other) const { return curr.index != other.curr.index; }
        iterator &operator++() {
            if (!inst->simple_layout) {
                curr.vh += 1 + (*types)[curr.index]->holder_size_in_ptrs;
            }
            ++curr.index;
            curr.type = curr.index < types->size() ? (*types)[curr.index] : nullptr;
            return *this;
        }
        value_and_holder &operator*() { return curr; }
        value_and_holder *operator->() { return &curr; }
    };

    iterator begin() { return iterator(inst, &tinfo); }
    iterator end() { return iterator(tinfo.size()); }

    iterator find(const type_info *find_type) {
        auto it = begin(), endit = end();
        while (it != endit && it->type != find_type) {
            ++it;
        }
        return it;
    }

    size_t size() { return tinfo.size(); }
};

/**
 * Extracts C++ value and holder pointer references from an instance (which may contain multiple
 * values/holders for python-side multiple inheritance) that match the given type.  Throws an error
 * if the given type (or ValueType, if omitted) is not a pybind11 base of the given instance.  If
 * `find_type` is omitted (or explicitly specified as nullptr) the first value/holder are returned,
 * regardless of type (and the resulting .type will be nullptr).
 *
 * The returned object should be short-lived: in particular, it must not outlive the called-upon
 * instance.
 */
PYBIND11_NOINLINE value_and_holder
instance::get_value_and_holder(const type_info *find_type /*= nullptr default in common.h*/,
                               bool throw_if_missing /*= true in common.h*/) {
    // Optimize common case:
    if (!find_type || Py_TYPE(this) == find_type->type) {
        return value_and_holder(this, find_type, 0, 0);
    }

    detail::values_and_holders vhs(this);
    auto it = vhs.find(find_type);
    if (it != vhs.end()) {
        return *it;
    }

    if (!throw_if_missing) {
        return value_and_holder();
    }

#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
    pybind11_fail("pybind11::detail::instance::get_value_and_holder: `"
                  + get_fully_qualified_tp_name(find_type->type)
                  + "' is not a pybind11 base of the given `"
                  + get_fully_qualified_tp_name(Py_TYPE(this)) + "' instance");
#else
    pybind11_fail(
        "pybind11::detail::instance::get_value_and_holder: "
        "type is not a pybind11 base of the given instance "
        "(#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for type details)");
#endif
}

PYBIND11_NOINLINE void instance::allocate_layout() {
    const auto &tinfo = all_type_info(Py_TYPE(this));

    const size_t n_types = tinfo.size();

    if (n_types == 0) {
        pybind11_fail(
            "instance allocation failed: new instance has no pybind11-registered base types");
    }

    simple_layout
        = n_types == 1 && tinfo.front()->holder_size_in_ptrs <= instance_simple_holder_in_ptrs();

    // Simple path: no python-side multiple inheritance, and a small-enough holder
    if (simple_layout) {
        simple_value_holder[0] = nullptr;
        simple_holder_constructed = false;
        simple_instance_registered = false;
    } else { // multiple base types or a too-large holder
        // Allocate space to hold: [v1*][h1][v2*][h2]...[bb...] where [vN*] is a value pointer,
        // [hN] is the (uninitialized) holder instance for value N, and [bb...] is a set of bool
        // values that tracks whether each associated holder has been initialized.  Each [block] is
        // padded, if necessary, to an integer multiple of sizeof(void *).
        size_t space = 0;
        for (auto *t : tinfo) {
            space += 1;                      // value pointer
            space += t->holder_size_in_ptrs; // holder instance
        }
        size_t flags_at = space;
        space += size_in_ptrs(n_types); // status bytes (holder_constructed and
                                        // instance_registered)

        // Allocate space for flags, values, and holders, and initialize it to 0 (flags and values,
        // in particular, need to be 0).  Use Python's memory allocation
        // functions: Python is using pymalloc, which is designed to be
        // efficient for small allocations like the one we're doing here;
        // for larger allocations they are just wrappers around malloc.
        // TODO: is this still true for pure Python 3.6?
        nonsimple.values_and_holders = (void **) PyMem_Calloc(space, sizeof(void *));
        if (!nonsimple.values_and_holders) {
            throw std::bad_alloc();
        }
        nonsimple.status
            = reinterpret_cast<std::uint8_t *>(&nonsimple.values_and_holders[flags_at]);
    }
    owned = true;
}

// NOLINTNEXTLINE(readability-make-member-function-const)
PYBIND11_NOINLINE void instance::deallocate_layout() {
    if (!simple_layout) {
        PyMem_Free(nonsimple.values_and_holders);
    }
}

PYBIND11_NOINLINE bool isinstance_generic(handle obj, const std::type_info &tp) {
    handle type = detail::get_type_handle(tp, false);
    if (!type) {
        return false;
    }
    return isinstance(obj, type);
}

PYBIND11_NOINLINE handle get_object_handle(const void *ptr, const detail::type_info *type) {
    auto &instances = get_internals().registered_instances;
    auto range = instances.equal_range(ptr);
    for (auto it = range.first; it != range.second; ++it) {
        for (const auto &vh : values_and_holders(it->second)) {
            if (vh.type == type) {
                return handle((PyObject *) it->second);
            }
        }
    }
    return handle();
}

inline PyThreadState *get_thread_state_unchecked() {
#if defined(PYPY_VERSION)
    return PyThreadState_GET();
#else
    return _PyThreadState_UncheckedGet();
#endif
}

// Forward declarations
void keep_alive_impl(handle nurse, handle patient);
inline PyObject *make_new_instance(PyTypeObject *type);

class type_caster_generic {
public:
    PYBIND11_NOINLINE explicit type_caster_generic(const std::type_info &type_info)
        : typeinfo(get_type_info(type_info)), cpptype(&type_info) {}

    explicit type_caster_generic(const type_info *typeinfo)
        : typeinfo(typeinfo), cpptype(typeinfo ? typeinfo->cpptype : nullptr) {}

    bool load(handle src, bool convert) { return load_impl<type_caster_generic>(src, convert); }

    PYBIND11_NOINLINE static handle cast(const void *_src,
                                         return_value_policy policy,
                                         handle parent,
                                         const detail::type_info *tinfo,
                                         void *(*copy_constructor)(const void *),
                                         void *(*move_constructor)(const void *),
                                         const void *existing_holder = nullptr) {
        if (!tinfo) { // no type info: error will be set already
            return handle();
        }

        void *src = const_cast<void *>(_src);
        if (src == nullptr) {
            return none().release();
        }

        if (handle registered_inst = find_registered_python_instance(src, tinfo)) {
            return registered_inst;
        }

        auto inst = reinterpret_steal<object>(make_new_instance(tinfo->type));
        auto *wrapper = reinterpret_cast<instance *>(inst.ptr());
        wrapper->owned = false;
        void *&valueptr = values_and_holders(wrapper).begin()->value_ptr();

        switch (policy) {
            case return_value_policy::automatic:
            case return_value_policy::take_ownership:
                valueptr = src;
                wrapper->owned = true;
                break;

            case return_value_policy::automatic_reference:
            case return_value_policy::reference:
                valueptr = src;
                wrapper->owned = false;
                break;

            case return_value_policy::copy:
                if (copy_constructor) {
                    valueptr = copy_constructor(src);
                } else {
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
                    std::string type_name(tinfo->cpptype->name());
                    detail::clean_type_id(type_name);
                    throw cast_error("return_value_policy = copy, but type " + type_name
                                     + " is non-copyable!");
#else
                    throw cast_error("return_value_policy = copy, but type is "
                                     "non-copyable! (#define PYBIND11_DETAILED_ERROR_MESSAGES or "
                                     "compile in debug mode for details)");
#endif
                }
                wrapper->owned = true;
                break;

            case return_value_policy::move:
                if (move_constructor) {
                    valueptr = move_constructor(src);
                } else if (copy_constructor) {
                    valueptr = copy_constructor(src);
                } else {
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
                    std::string type_name(tinfo->cpptype->name());
                    detail::clean_type_id(type_name);
                    throw cast_error("return_value_policy = move, but type " + type_name
                                     + " is neither movable nor copyable!");
#else
                    throw cast_error("return_value_policy = move, but type is neither "
                                     "movable nor copyable! "
                                     "(#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in "
                                     "debug mode for details)");
#endif
                }
                wrapper->owned = true;
                break;

            case return_value_policy::reference_internal:
                valueptr = src;
                wrapper->owned = false;
                keep_alive_impl(inst, parent);
                break;

            default:
                throw cast_error("unhandled return_value_policy: should not happen!");
        }

        tinfo->init_instance(wrapper, existing_holder);

        return inst.release();
    }

    // Base methods for generic caster; there are overridden in copyable_holder_caster
    void load_value(value_and_holder &&v_h) {
        auto *&vptr = v_h.value_ptr();
        // Lazy allocation for unallocated values:
        if (vptr == nullptr) {
            const auto *type = v_h.type ? v_h.type : typeinfo;
            if (type->operator_new) {
                vptr = type->operator_new(type->type_size);
            } else {
#if defined(__cpp_aligned_new) && (!defined(_MSC_VER) || _MSC_VER >= 1912)
                if (type->type_align > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
                    vptr = ::operator new(type->type_size, std::align_val_t(type->type_align));
                } else {
                    vptr = ::operator new(type->type_size);
                }
#else
                vptr = ::operator new(type->type_size);
#endif
            }
        }
        value = vptr;
    }
    bool try_implicit_casts(handle src, bool convert) {
        for (const auto &cast : typeinfo->implicit_casts) {
            type_caster_generic sub_caster(*cast.first);
            if (sub_caster.load(src, convert)) {
                value = cast.second(sub_caster.value);
                return true;
            }
        }
        return false;
    }
    bool try_direct_conversions(handle src) {
        for (auto &converter : *typeinfo->direct_conversions) {
            if (converter(src.ptr(), value)) {
                return true;
            }
        }
        return false;
    }
    void check_holder_compat() {}

    PYBIND11_NOINLINE static void *local_load(PyObject *src, const type_info *ti) {
        auto caster = type_caster_generic(ti);
        if (caster.load(src, false)) {
            return caster.value;
        }
        return nullptr;
    }

    /// Try to load with foreign typeinfo, if available. Used when there is no
    /// native typeinfo, or when the native one wasn't able to produce a value.
    PYBIND11_NOINLINE bool try_load_foreign_module_local(handle src) {
        constexpr auto *local_key = PYBIND11_MODULE_LOCAL_ID;
        const auto pytype = type::handle_of(src);
        if (!hasattr(pytype, local_key)) {
            return false;
        }

        type_info *foreign_typeinfo = reinterpret_borrow<capsule>(getattr(pytype, local_key));
        // Only consider this foreign loader if actually foreign and is a loader of the correct cpp
        // type
        if (foreign_typeinfo->module_local_load == &local_load
            || (cpptype && !same_type(*cpptype, *foreign_typeinfo->cpptype))) {
            return false;
        }

        if (auto *result = foreign_typeinfo->module_local_load(src.ptr(), foreign_typeinfo)) {
            value = result;
            return true;
        }
        return false;
    }

    // Implementation of `load`; this takes the type of `this` so that it can dispatch the relevant
    // bits of code between here and copyable_holder_caster where the two classes need different
    // logic (without having to resort to virtual inheritance).
    template <typename ThisT>
    PYBIND11_NOINLINE bool load_impl(handle src, bool convert) {
        if (!src) {
            return false;
        }
        if (!typeinfo) {
            return try_load_foreign_module_local(src);
        }

        auto &this_ = static_cast<ThisT &>(*this);
        this_.check_holder_compat();

        PyTypeObject *srctype = Py_TYPE(src.ptr());

        // Case 1: If src is an exact type match for the target type then we can reinterpret_cast
        // the instance's value pointer to the target type:
        if (srctype == typeinfo->type) {
            this_.load_value(reinterpret_cast<instance *>(src.ptr())->get_value_and_holder());
            return true;
        }
        // Case 2: We have a derived class
        if (PyType_IsSubtype(srctype, typeinfo->type)) {
            const auto &bases = all_type_info(srctype);
            bool no_cpp_mi = typeinfo->simple_type;

            // Case 2a: the python type is a Python-inherited derived class that inherits from just
            // one simple (no MI) pybind11 class, or is an exact match, so the C++ instance is of
            // the right type and we can use reinterpret_cast.
            // (This is essentially the same as case 2b, but because not using multiple inheritance
            // is extremely common, we handle it specially to avoid the loop iterator and type
            // pointer lookup overhead)
            if (bases.size() == 1 && (no_cpp_mi || bases.front()->type == typeinfo->type)) {
                this_.load_value(reinterpret_cast<instance *>(src.ptr())->get_value_and_holder());
                return true;
            }
            // Case 2b: the python type inherits from multiple C++ bases.  Check the bases to see
            // if we can find an exact match (or, for a simple C++ type, an inherited match); if
            // so, we can safely reinterpret_cast to the relevant pointer.
            if (bases.size() > 1) {
                for (auto *base : bases) {
                    if (no_cpp_mi ? PyType_IsSubtype(base->type, typeinfo->type)
                                  : base->type == typeinfo->type) {
                        this_.load_value(
                            reinterpret_cast<instance *>(src.ptr())->get_value_and_holder(base));
                        return true;
                    }
                }
            }

            // Case 2c: C++ multiple inheritance is involved and we couldn't find an exact type
            // match in the registered bases, above, so try implicit casting (needed for proper C++
            // casting when MI is involved).
            if (this_.try_implicit_casts(src, convert)) {
                return true;
            }
        }

        // Perform an implicit conversion
        if (convert) {
            for (const auto &converter : typeinfo->implicit_conversions) {
                auto temp = reinterpret_steal<object>(converter(src.ptr(), typeinfo->type));
                if (load_impl<ThisT>(temp, false)) {
                    loader_life_support::add_patient(temp);
                    return true;
                }
            }
            if (this_.try_direct_conversions(src)) {
                return true;
            }
        }

        // Failed to match local typeinfo. Try again with global.
        if (typeinfo->module_local) {
            if (auto *gtype = get_global_type_info(*typeinfo->cpptype)) {
                typeinfo = gtype;
                return load(src, false);
            }
        }

        // Global typeinfo has precedence over foreign module_local
        if (try_load_foreign_module_local(src)) {
            return true;
        }

        // Custom converters didn't take None, now we convert None to nullptr.
        if (src.is_none()) {
            // Defer accepting None to other overloads (if we aren't in convert mode):
            if (!convert) {
                return false;
            }
            value = nullptr;
            return true;
        }

        return false;
    }

    // Called to do type lookup and wrap the pointer and type in a pair when a dynamic_cast
    // isn't needed or can't be used.  If the type is unknown, sets the error and returns a pair
    // with .second = nullptr.  (p.first = nullptr is not an error: it becomes None).
    PYBIND11_NOINLINE static std::pair<const void *, const type_info *>
    src_and_type(const void *src,
                 const std::type_info &cast_type,
                 const std::type_info *rtti_type = nullptr) {
        if (auto *tpi = get_type_info(cast_type)) {
            return {src, const_cast<const type_info *>(tpi)};
        }

        // Not found, set error:
        std::string tname = rtti_type ? rtti_type->name() : cast_type.name();
        detail::clean_type_id(tname);
        std::string msg = "Unregistered type : " + tname;
        PyErr_SetString(PyExc_TypeError, msg.c_str());
        return {nullptr, nullptr};
    }

    const type_info *typeinfo = nullptr;
    const std::type_info *cpptype = nullptr;
    void *value = nullptr;
};

/**
 * Determine suitable casting operator for pointer-or-lvalue-casting type casters.  The type caster
 * needs to provide `operator T*()` and `operator T&()` operators.
 *
 * If the type supports moving the value away via an `operator T&&() &&` method, it should use
 * `movable_cast_op_type` instead.
 */
template <typename T>
using cast_op_type = conditional_t<std::is_pointer<remove_reference_t<T>>::value,
                                   typename std::add_pointer<intrinsic_t<T>>::type,
                                   typename std::add_lvalue_reference<intrinsic_t<T>>::type>;

/**
 * Determine suitable casting operator for a type caster with a movable value.  Such a type caster
 * needs to provide `operator T*()`, `operator T&()`, and `operator T&&() &&`.  The latter will be
 * called in appropriate contexts where the value can be moved rather than copied.
 *
 * These operator are automatically provided when using the PYBIND11_TYPE_CASTER macro.
 */
template <typename T>
using movable_cast_op_type
    = conditional_t<std::is_pointer<typename std::remove_reference<T>::type>::value,
                    typename std::add_pointer<intrinsic_t<T>>::type,
                    conditional_t<std::is_rvalue_reference<T>::value,
                                  typename std::add_rvalue_reference<intrinsic_t<T>>::type,
                                  typename std::add_lvalue_reference<intrinsic_t<T>>::type>>;

// std::is_copy_constructible isn't quite enough: it lets std::vector<T> (and similar) through when
// T is non-copyable, but code containing such a copy constructor fails to actually compile.
template <typename T, typename SFINAE = void>
struct is_copy_constructible : std::is_copy_constructible<T> {};

// Specialization for types that appear to be copy constructible but also look like stl containers
// (we specifically check for: has `value_type` and `reference` with `reference = value_type&`): if
// so, copy constructability depends on whether the value_type is copy constructible.
template <typename Container>
struct is_copy_constructible<
    Container,
    enable_if_t<
        all_of<std::is_copy_constructible<Container>,
               std::is_same<typename Container::value_type &, typename Container::reference>,
               // Avoid infinite recursion
               negation<std::is_same<Container, typename Container::value_type>>>::value>>
    : is_copy_constructible<typename Container::value_type> {};

// Likewise for std::pair
// (after C++17 it is mandatory that the copy constructor not exist when the two types aren't
// themselves copy constructible, but this can not be relied upon when T1 or T2 are themselves
// containers).
template <typename T1, typename T2>
struct is_copy_constructible<std::pair<T1, T2>>
    : all_of<is_copy_constructible<T1>, is_copy_constructible<T2>> {};

// The same problems arise with std::is_copy_assignable, so we use the same workaround.
template <typename T, typename SFINAE = void>
struct is_copy_assignable : std::is_copy_assignable<T> {};
template <typename Container>
struct is_copy_assignable<Container,
                          enable_if_t<all_of<std::is_copy_assignable<Container>,
                                             std::is_same<typename Container::value_type &,
                                                          typename Container::reference>>::value>>
    : is_copy_assignable<typename Container::value_type> {};
template <typename T1, typename T2>
struct is_copy_assignable<std::pair<T1, T2>>
    : all_of<is_copy_assignable<T1>, is_copy_assignable<T2>> {};

PYBIND11_NAMESPACE_END(detail)

// polymorphic_type_hook<itype>::get(src, tinfo) determines whether the object pointed
// to by `src` actually is an instance of some class derived from `itype`.
// If so, it sets `tinfo` to point to the std::type_info representing that derived
// type, and returns a pointer to the start of the most-derived object of that type
// (in which `src` is a subobject; this will be the same address as `src` in most
// single inheritance cases). If not, or if `src` is nullptr, it simply returns `src`
// and leaves `tinfo` at its default value of nullptr.
//
// The default polymorphic_type_hook just returns src. A specialization for polymorphic
// types determines the runtime type of the passed object and adjusts the this-pointer
// appropriately via dynamic_cast<void*>. This is what enables a C++ Animal* to appear
// to Python as a Dog (if Dog inherits from Animal, Animal is polymorphic, Dog is
// registered with pybind11, and this Animal is in fact a Dog).
//
// You may specialize polymorphic_type_hook yourself for types that want to appear
// polymorphic to Python but do not use C++ RTTI. (This is a not uncommon pattern
// in performance-sensitive applications, used most notably in LLVM.)
//
// polymorphic_type_hook_base allows users to specialize polymorphic_type_hook with
// std::enable_if. User provided specializations will always have higher priority than
// the default implementation and specialization provided in polymorphic_type_hook_base.
template <typename itype, typename SFINAE = void>
struct polymorphic_type_hook_base {
    static const void *get(const itype *src, const std::type_info *&) { return src; }
};
template <typename itype>
struct polymorphic_type_hook_base<itype, detail::enable_if_t<std::is_polymorphic<itype>::value>> {
    static const void *get(const itype *src, const std::type_info *&type) {
        type = src ? &typeid(*src) : nullptr;
        return dynamic_cast<const void *>(src);
    }
};
template <typename itype, typename SFINAE = void>
struct polymorphic_type_hook : public polymorphic_type_hook_base<itype> {};

PYBIND11_NAMESPACE_BEGIN(detail)

/// Generic type caster for objects stored on the heap
template <typename type>
class type_caster_base : public type_caster_generic {
    using itype = intrinsic_t<type>;

public:
    static constexpr auto name = const_name<type>();

    type_caster_base() : type_caster_base(typeid(type)) {}
    explicit type_caster_base(const std::type_info &info) : type_caster_generic(info) {}

    static handle cast(const itype &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic
            || policy == return_value_policy::automatic_reference) {
            policy = return_value_policy::copy;
        }
        return cast(&src, policy, parent);
    }

    static handle cast(itype &&src, return_value_policy, handle parent) {
        return cast(&src, return_value_policy::move, parent);
    }

    // Returns a (pointer, type_info) pair taking care of necessary type lookup for a
    // polymorphic type (using RTTI by default, but can be overridden by specializing
    // polymorphic_type_hook). If the instance isn't derived, returns the base version.
    static std::pair<const void *, const type_info *> src_and_type(const itype *src) {
        const auto &cast_type = typeid(itype);
        const std::type_info *instance_type = nullptr;
        const void *vsrc = polymorphic_type_hook<itype>::get(src, instance_type);
        if (instance_type && !same_type(cast_type, *instance_type)) {
            // This is a base pointer to a derived type. If the derived type is registered
            // with pybind11, we want to make the full derived object available.
            // In the typical case where itype is polymorphic, we get the correct
            // derived pointer (which may be != base pointer) by a dynamic_cast to
            // most derived type. If itype is not polymorphic, we won't get here
            // except via a user-provided specialization of polymorphic_type_hook,
            // and the user has promised that no this-pointer adjustment is
            // required in that case, so it's OK to use static_cast.
            if (const auto *tpi = get_type_info(*instance_type)) {
                return {vsrc, tpi};
            }
        }
        // Otherwise we have either a nullptr, an `itype` pointer, or an unknown derived pointer,
        // so don't do a cast
        return type_caster_generic::src_and_type(src, cast_type, instance_type);
    }

    static handle cast(const itype *src, return_value_policy policy, handle parent) {
        auto st = src_and_type(src);
        return type_caster_generic::cast(st.first,
                                         policy,
                                         parent,
                                         st.second,
                                         make_copy_constructor(src),
                                         make_move_constructor(src));
    }

    static handle cast_holder(const itype *src, const void *holder) {
        auto st = src_and_type(src);
        return type_caster_generic::cast(st.first,
                                         return_value_policy::take_ownership,
                                         {},
                                         st.second,
                                         nullptr,
                                         nullptr,
                                         holder);
    }

    template <typename T>
    using cast_op_type = detail::cast_op_type<T>;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator itype *() { return (type *) value; }
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator itype &() {
        if (!value) {
            throw reference_cast_error();
        }
        return *((itype *) value);
    }

protected:
    using Constructor = void *(*) (const void *);

    /* Only enabled when the types are {copy,move}-constructible *and* when the type
       does not have a private operator new implementation. A comma operator is used in the
       decltype argument to apply SFINAE to the public copy/move constructors.*/
    template <typename T, typename = enable_if_t<is_copy_constructible<T>::value>>
    static auto make_copy_constructor(const T *)
        -> decltype(new T(std::declval<const T>()), Constructor{}) {
        return [](const void *arg) -> void * { return new T(*reinterpret_cast<const T *>(arg)); };
    }

    template <typename T, typename = enable_if_t<std::is_move_constructible<T>::value>>
    static auto make_move_constructor(const T *)
        -> decltype(new T(std::declval<T &&>()), Constructor{}) {
        return [](const void *arg) -> void * {
            return new T(std::move(*const_cast<T *>(reinterpret_cast<const T *>(arg))));
        };
    }

    static Constructor make_copy_constructor(...) { return nullptr; }
    static Constructor make_move_constructor(...) { return nullptr; }
};

PYBIND11_NOINLINE std::string type_info_description(const std::type_info &ti) {
    if (auto *type_data = get_type_info(ti)) {
        handle th((PyObject *) type_data->type);
        return th.attr("__module__").cast<std::string>() + '.'
               + th.attr("__qualname__").cast<std::string>();
    }
    return clean_type_id(ti.name());
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
