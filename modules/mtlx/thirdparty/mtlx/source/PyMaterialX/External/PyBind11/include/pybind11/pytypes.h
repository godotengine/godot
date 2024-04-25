/*
    pybind11/pytypes.h: Convenience wrapper classes for basic Python types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/common.h"
#include "buffer_info.h"

#include <assert.h>
#include <cstddef>
#include <exception>
#include <frameobject.h>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#if defined(PYBIND11_HAS_OPTIONAL)
#    include <optional>
#endif

#ifdef PYBIND11_HAS_STRING_VIEW
#    include <string_view>
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_WARNING_DISABLE_MSVC(4127)

/* A few forward declarations */
class handle;
class object;
class str;
class iterator;
class type;
struct arg;
struct arg_v;

PYBIND11_NAMESPACE_BEGIN(detail)
class args_proxy;
bool isinstance_generic(handle obj, const std::type_info &tp);

// Accessor forward declarations
template <typename Policy>
class accessor;
namespace accessor_policies {
struct obj_attr;
struct str_attr;
struct generic_item;
struct sequence_item;
struct list_item;
struct tuple_item;
} // namespace accessor_policies
using obj_attr_accessor = accessor<accessor_policies::obj_attr>;
using str_attr_accessor = accessor<accessor_policies::str_attr>;
using item_accessor = accessor<accessor_policies::generic_item>;
using sequence_accessor = accessor<accessor_policies::sequence_item>;
using list_accessor = accessor<accessor_policies::list_item>;
using tuple_accessor = accessor<accessor_policies::tuple_item>;

/// Tag and check to identify a class which implements the Python object API
class pyobject_tag {};
template <typename T>
using is_pyobject = std::is_base_of<pyobject_tag, remove_reference_t<T>>;

/** \rst
    A mixin class which adds common functions to `handle`, `object` and various accessors.
    The only requirement for `Derived` is to implement ``PyObject *Derived::ptr() const``.
\endrst */
template <typename Derived>
class object_api : public pyobject_tag {
    const Derived &derived() const { return static_cast<const Derived &>(*this); }

public:
    /** \rst
        Return an iterator equivalent to calling ``iter()`` in Python. The object
        must be a collection which supports the iteration protocol.
    \endrst */
    iterator begin() const;
    /// Return a sentinel which ends iteration.
    iterator end() const;

    /** \rst
        Return an internal functor to invoke the object's sequence protocol. Casting
        the returned ``detail::item_accessor`` instance to a `handle` or `object`
        subclass causes a corresponding call to ``__getitem__``. Assigning a `handle`
        or `object` subclass causes a call to ``__setitem__``.
    \endrst */
    item_accessor operator[](handle key) const;
    /// See above (the only difference is that the key's reference is stolen)
    item_accessor operator[](object &&key) const;
    /// See above (the only difference is that the key is provided as a string literal)
    item_accessor operator[](const char *key) const;

    /** \rst
        Return an internal functor to access the object's attributes. Casting the
        returned ``detail::obj_attr_accessor`` instance to a `handle` or `object`
        subclass causes a corresponding call to ``getattr``. Assigning a `handle`
        or `object` subclass causes a call to ``setattr``.
    \endrst */
    obj_attr_accessor attr(handle key) const;
    /// See above (the only difference is that the key's reference is stolen)
    obj_attr_accessor attr(object &&key) const;
    /// See above (the only difference is that the key is provided as a string literal)
    str_attr_accessor attr(const char *key) const;

    /** \rst
        Matches * unpacking in Python, e.g. to unpack arguments out of a ``tuple``
        or ``list`` for a function call. Applying another * to the result yields
        ** unpacking, e.g. to unpack a dict as function keyword arguments.
        See :ref:`calling_python_functions`.
    \endrst */
    args_proxy operator*() const;

    /// Check if the given item is contained within this object, i.e. ``item in obj``.
    template <typename T>
    bool contains(T &&item) const;

    /** \rst
        Assuming the Python object is a function or implements the ``__call__``
        protocol, ``operator()`` invokes the underlying function, passing an
        arbitrary set of parameters. The result is returned as a `object` and
        may need to be converted back into a Python object using `handle::cast()`.

        When some of the arguments cannot be converted to Python objects, the
        function will throw a `cast_error` exception. When the Python function
        call fails, a `error_already_set` exception is thrown.
    \endrst */
    template <return_value_policy policy = return_value_policy::automatic_reference,
              typename... Args>
    object operator()(Args &&...args) const;
    template <return_value_policy policy = return_value_policy::automatic_reference,
              typename... Args>
    PYBIND11_DEPRECATED("call(...) was deprecated in favor of operator()(...)")
    object call(Args &&...args) const;

    /// Equivalent to ``obj is other`` in Python.
    bool is(object_api const &other) const { return derived().ptr() == other.derived().ptr(); }
    /// Equivalent to ``obj is None`` in Python.
    bool is_none() const { return derived().ptr() == Py_None; }
    /// Equivalent to obj == other in Python
    bool equal(object_api const &other) const { return rich_compare(other, Py_EQ); }
    bool not_equal(object_api const &other) const { return rich_compare(other, Py_NE); }
    bool operator<(object_api const &other) const { return rich_compare(other, Py_LT); }
    bool operator<=(object_api const &other) const { return rich_compare(other, Py_LE); }
    bool operator>(object_api const &other) const { return rich_compare(other, Py_GT); }
    bool operator>=(object_api const &other) const { return rich_compare(other, Py_GE); }

    object operator-() const;
    object operator~() const;
    object operator+(object_api const &other) const;
    object operator+=(object_api const &other);
    object operator-(object_api const &other) const;
    object operator-=(object_api const &other);
    object operator*(object_api const &other) const;
    object operator*=(object_api const &other);
    object operator/(object_api const &other) const;
    object operator/=(object_api const &other);
    object operator|(object_api const &other) const;
    object operator|=(object_api const &other);
    object operator&(object_api const &other) const;
    object operator&=(object_api const &other);
    object operator^(object_api const &other) const;
    object operator^=(object_api const &other);
    object operator<<(object_api const &other) const;
    object operator<<=(object_api const &other);
    object operator>>(object_api const &other) const;
    object operator>>=(object_api const &other);

    PYBIND11_DEPRECATED("Use py::str(obj) instead")
    pybind11::str str() const;

    /// Get or set the object's docstring, i.e. ``obj.__doc__``.
    str_attr_accessor doc() const;

    /// Return the object's current reference count
    int ref_count() const { return static_cast<int>(Py_REFCNT(derived().ptr())); }

    // TODO PYBIND11_DEPRECATED(
    //     "Call py::type::handle_of(h) or py::type::of(h) instead of h.get_type()")
    handle get_type() const;

private:
    bool rich_compare(object_api const &other, int value) const;
};

template <typename T>
using is_pyobj_ptr_or_nullptr_t = detail::any_of<std::is_same<T, PyObject *>,
                                                 std::is_same<T, PyObject *const>,
                                                 std::is_same<T, std::nullptr_t>>;

PYBIND11_NAMESPACE_END(detail)

#if !defined(PYBIND11_HANDLE_REF_DEBUG) && !defined(NDEBUG)
#    define PYBIND11_HANDLE_REF_DEBUG
#endif

/** \rst
    Holds a reference to a Python object (no reference counting)

    The `handle` class is a thin wrapper around an arbitrary Python object (i.e. a
    ``PyObject *`` in Python's C API). It does not perform any automatic reference
    counting and merely provides a basic C++ interface to various Python API functions.

    .. seealso::
        The `object` class inherits from `handle` and adds automatic reference
        counting features.
\endrst */
class handle : public detail::object_api<handle> {
public:
    /// The default constructor creates a handle with a ``nullptr``-valued pointer
    handle() = default;

    /// Enable implicit conversion from ``PyObject *`` and ``nullptr``.
    /// Not using ``handle(PyObject *ptr)`` to avoid implicit conversion from ``0``.
    template <typename T,
              detail::enable_if_t<detail::is_pyobj_ptr_or_nullptr_t<T>::value, int> = 0>
    // NOLINTNEXTLINE(google-explicit-constructor)
    handle(T ptr) : m_ptr(ptr) {}

    /// Enable implicit conversion through ``T::operator PyObject *()``.
    template <
        typename T,
        detail::enable_if_t<detail::all_of<detail::none_of<std::is_base_of<handle, T>,
                                                           detail::is_pyobj_ptr_or_nullptr_t<T>>,
                                           std::is_convertible<T, PyObject *>>::value,
                            int>
        = 0>
    // NOLINTNEXTLINE(google-explicit-constructor)
    handle(T &obj) : m_ptr(obj) {}

    /// Return the underlying ``PyObject *`` pointer
    PyObject *ptr() const { return m_ptr; }
    PyObject *&ptr() { return m_ptr; }

    /** \rst
        Manually increase the reference count of the Python object. Usually, it is
        preferable to use the `object` class which derives from `handle` and calls
        this function automatically. Returns a reference to itself.
    \endrst */
    const handle &inc_ref() const & {
#ifdef PYBIND11_HANDLE_REF_DEBUG
        inc_ref_counter(1);
#endif
#ifdef PYBIND11_ASSERT_GIL_HELD_INCREF_DECREF
        if (m_ptr != nullptr && !PyGILState_Check()) {
            throw_gilstate_error("pybind11::handle::inc_ref()");
        }
#endif
        Py_XINCREF(m_ptr);
        return *this;
    }

    /** \rst
        Manually decrease the reference count of the Python object. Usually, it is
        preferable to use the `object` class which derives from `handle` and calls
        this function automatically. Returns a reference to itself.
    \endrst */
    const handle &dec_ref() const & {
#ifdef PYBIND11_ASSERT_GIL_HELD_INCREF_DECREF
        if (m_ptr != nullptr && !PyGILState_Check()) {
            throw_gilstate_error("pybind11::handle::dec_ref()");
        }
#endif
        Py_XDECREF(m_ptr);
        return *this;
    }

    /** \rst
        Attempt to cast the Python object into the given C++ type. A `cast_error`
        will be throw upon failure.
    \endrst */
    template <typename T>
    T cast() const;
    /// Return ``true`` when the `handle` wraps a valid Python object
    explicit operator bool() const { return m_ptr != nullptr; }
    /** \rst
        Deprecated: Check that the underlying pointers are the same.
        Equivalent to ``obj1 is obj2`` in Python.
    \endrst */
    PYBIND11_DEPRECATED("Use obj1.is(obj2) instead")
    bool operator==(const handle &h) const { return m_ptr == h.m_ptr; }
    PYBIND11_DEPRECATED("Use !obj1.is(obj2) instead")
    bool operator!=(const handle &h) const { return m_ptr != h.m_ptr; }
    PYBIND11_DEPRECATED("Use handle::operator bool() instead")
    bool check() const { return m_ptr != nullptr; }

protected:
    PyObject *m_ptr = nullptr;

private:
#ifdef PYBIND11_ASSERT_GIL_HELD_INCREF_DECREF
    void throw_gilstate_error(const std::string &function_name) const {
        fprintf(
            stderr,
            "%s is being called while the GIL is either not held or invalid. Please see "
            "https://pybind11.readthedocs.io/en/stable/advanced/"
            "misc.html#common-sources-of-global-interpreter-lock-errors for debugging advice.\n",
            function_name.c_str());
        fflush(stderr);
        if (Py_TYPE(m_ptr)->tp_name != nullptr) {
            fprintf(stderr,
                    "The failing %s call was triggered on a %s object.\n",
                    function_name.c_str(),
                    Py_TYPE(m_ptr)->tp_name);
            fflush(stderr);
        }
        throw std::runtime_error(function_name + " PyGILState_Check() failure.");
    }
#endif

#ifdef PYBIND11_HANDLE_REF_DEBUG
    static std::size_t inc_ref_counter(std::size_t add) {
        thread_local std::size_t counter = 0;
        counter += add;
        return counter;
    }

public:
    static std::size_t inc_ref_counter() { return inc_ref_counter(0); }
#endif
};

/** \rst
    Holds a reference to a Python object (with reference counting)

    Like `handle`, the `object` class is a thin wrapper around an arbitrary Python
    object (i.e. a ``PyObject *`` in Python's C API). In contrast to `handle`, it
    optionally increases the object's reference count upon construction, and it
    *always* decreases the reference count when the `object` instance goes out of
    scope and is destructed. When using `object` instances consistently, it is much
    easier to get reference counting right at the first attempt.
\endrst */
class object : public handle {
public:
    object() = default;
    PYBIND11_DEPRECATED("Use reinterpret_borrow<object>() or reinterpret_steal<object>()")
    object(handle h, bool is_borrowed) : handle(h) {
        if (is_borrowed) {
            inc_ref();
        }
    }
    /// Copy constructor; always increases the reference count
    object(const object &o) : handle(o) { inc_ref(); }
    /// Move constructor; steals the object from ``other`` and preserves its reference count
    object(object &&other) noexcept : handle(other) { other.m_ptr = nullptr; }
    /// Destructor; automatically calls `handle::dec_ref()`
    ~object() { dec_ref(); }

    /** \rst
        Resets the internal pointer to ``nullptr`` without decreasing the
        object's reference count. The function returns a raw handle to the original
        Python object.
    \endrst */
    handle release() {
        PyObject *tmp = m_ptr;
        m_ptr = nullptr;
        return handle(tmp);
    }

    object &operator=(const object &other) {
        // Skip inc_ref and dec_ref if both objects are the same
        if (!this->is(other)) {
            other.inc_ref();
            // Use temporary variable to ensure `*this` remains valid while
            // `Py_XDECREF` executes, in case `*this` is accessible from Python.
            handle temp(m_ptr);
            m_ptr = other.m_ptr;
            temp.dec_ref();
        }
        return *this;
    }

    object &operator=(object &&other) noexcept {
        if (this != &other) {
            handle temp(m_ptr);
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
            temp.dec_ref();
        }
        return *this;
    }

#define PYBIND11_INPLACE_OP(iop)                                                                  \
    object iop(object_api const &other) { return operator=(handle::iop(other)); }

    PYBIND11_INPLACE_OP(operator+=)
    PYBIND11_INPLACE_OP(operator-=)
    PYBIND11_INPLACE_OP(operator*=)
    PYBIND11_INPLACE_OP(operator/=)
    PYBIND11_INPLACE_OP(operator|=)
    PYBIND11_INPLACE_OP(operator&=)
    PYBIND11_INPLACE_OP(operator^=)
    PYBIND11_INPLACE_OP(operator<<=)
    PYBIND11_INPLACE_OP(operator>>=)
#undef PYBIND11_INPLACE_OP

    // Calling cast() on an object lvalue just copies (via handle::cast)
    template <typename T>
    T cast() const &;
    // Calling on an object rvalue does a move, if needed and/or possible
    template <typename T>
    T cast() &&;

protected:
    // Tags for choosing constructors from raw PyObject *
    struct borrowed_t {};
    struct stolen_t {};

    /// @cond BROKEN
    template <typename T>
    friend T reinterpret_borrow(handle);
    template <typename T>
    friend T reinterpret_steal(handle);
    /// @endcond

public:
    // Only accessible from derived classes and the reinterpret_* functions
    object(handle h, borrowed_t) : handle(h) { inc_ref(); }
    object(handle h, stolen_t) : handle(h) {}
};

/** \rst
    Declare that a `handle` or ``PyObject *`` is a certain type and borrow the reference.
    The target type ``T`` must be `object` or one of its derived classes. The function
    doesn't do any conversions or checks. It's up to the user to make sure that the
    target type is correct.

    .. code-block:: cpp

        PyObject *p = PyList_GetItem(obj, index);
        py::object o = reinterpret_borrow<py::object>(p);
        // or
        py::tuple t = reinterpret_borrow<py::tuple>(p); // <-- `p` must be already be a `tuple`
\endrst */
template <typename T>
T reinterpret_borrow(handle h) {
    return {h, object::borrowed_t{}};
}

/** \rst
    Like `reinterpret_borrow`, but steals the reference.

     .. code-block:: cpp

        PyObject *p = PyObject_Str(obj);
        py::str s = reinterpret_steal<py::str>(p); // <-- `p` must be already be a `str`
\endrst */
template <typename T>
T reinterpret_steal(handle h) {
    return {h, object::stolen_t{}};
}

PYBIND11_NAMESPACE_BEGIN(detail)

// Equivalent to obj.__class__.__name__ (or obj.__name__ if obj is a class).
inline const char *obj_class_name(PyObject *obj) {
    if (PyType_Check(obj)) {
        return reinterpret_cast<PyTypeObject *>(obj)->tp_name;
    }
    return Py_TYPE(obj)->tp_name;
}

std::string error_string();

struct error_fetch_and_normalize {
    // Immediate normalization is long-established behavior (starting with
    // https://github.com/pybind/pybind11/commit/135ba8deafb8bf64a15b24d1513899eb600e2011
    // from Sep 2016) and safest. Normalization could be deferred, but this could mask
    // errors elsewhere, the performance gain is very minor in typical situations
    // (usually the dominant bottleneck is EH unwinding), and the implementation here
    // would be more complex.
    explicit error_fetch_and_normalize(const char *called) {
        PyErr_Fetch(&m_type.ptr(), &m_value.ptr(), &m_trace.ptr());
        if (!m_type) {
            pybind11_fail("Internal error: " + std::string(called)
                          + " called while "
                            "Python error indicator not set.");
        }
        const char *exc_type_name_orig = detail::obj_class_name(m_type.ptr());
        if (exc_type_name_orig == nullptr) {
            pybind11_fail("Internal error: " + std::string(called)
                          + " failed to obtain the name "
                            "of the original active exception type.");
        }
        m_lazy_error_string = exc_type_name_orig;
        // PyErr_NormalizeException() may change the exception type if there are cascading
        // failures. This can potentially be extremely confusing.
        PyErr_NormalizeException(&m_type.ptr(), &m_value.ptr(), &m_trace.ptr());
        if (m_type.ptr() == nullptr) {
            pybind11_fail("Internal error: " + std::string(called)
                          + " failed to normalize the "
                            "active exception.");
        }
        const char *exc_type_name_norm = detail::obj_class_name(m_type.ptr());
        if (exc_type_name_norm == nullptr) {
            pybind11_fail("Internal error: " + std::string(called)
                          + " failed to obtain the name "
                            "of the normalized active exception type.");
        }
#if defined(PYPY_VERSION_NUM) && PYPY_VERSION_NUM < 0x07030a00
        // This behavior runs the risk of masking errors in the error handling, but avoids a
        // conflict with PyPy, which relies on the normalization here to change OSError to
        // FileNotFoundError (https://github.com/pybind/pybind11/issues/4075).
        m_lazy_error_string = exc_type_name_norm;
#else
        if (exc_type_name_norm != m_lazy_error_string) {
            std::string msg = std::string(called)
                              + ": MISMATCH of original and normalized "
                                "active exception types: ";
            msg += "ORIGINAL ";
            msg += m_lazy_error_string;
            msg += " REPLACED BY ";
            msg += exc_type_name_norm;
            msg += ": " + format_value_and_trace();
            pybind11_fail(msg);
        }
#endif
    }

    error_fetch_and_normalize(const error_fetch_and_normalize &) = delete;
    error_fetch_and_normalize(error_fetch_and_normalize &&) = delete;

    std::string format_value_and_trace() const {
        std::string result;
        std::string message_error_string;
        if (m_value) {
            auto value_str = reinterpret_steal<object>(PyObject_Str(m_value.ptr()));
            constexpr const char *message_unavailable_exc
                = "<MESSAGE UNAVAILABLE DUE TO ANOTHER EXCEPTION>";
            if (!value_str) {
                message_error_string = detail::error_string();
                result = message_unavailable_exc;
            } else {
                // Not using `value_str.cast<std::string>()`, to not potentially throw a secondary
                // error_already_set that will then result in process termination (#4288).
                auto value_bytes = reinterpret_steal<object>(
                    PyUnicode_AsEncodedString(value_str.ptr(), "utf-8", "backslashreplace"));
                if (!value_bytes) {
                    message_error_string = detail::error_string();
                    result = message_unavailable_exc;
                } else {
                    char *buffer = nullptr;
                    Py_ssize_t length = 0;
                    if (PyBytes_AsStringAndSize(value_bytes.ptr(), &buffer, &length) == -1) {
                        message_error_string = detail::error_string();
                        result = message_unavailable_exc;
                    } else {
                        result = std::string(buffer, static_cast<std::size_t>(length));
                    }
                }
            }
        } else {
            result = "<MESSAGE UNAVAILABLE>";
        }
        if (result.empty()) {
            result = "<EMPTY MESSAGE>";
        }

        bool have_trace = false;
        if (m_trace) {
#if !defined(PYPY_VERSION)
            auto *tb = reinterpret_cast<PyTracebackObject *>(m_trace.ptr());

            // Get the deepest trace possible.
            while (tb->tb_next) {
                tb = tb->tb_next;
            }

            PyFrameObject *frame = tb->tb_frame;
            Py_XINCREF(frame);
            result += "\n\nAt:\n";
            while (frame) {
#    if PY_VERSION_HEX >= 0x030900B1
                PyCodeObject *f_code = PyFrame_GetCode(frame);
#    else
                PyCodeObject *f_code = frame->f_code;
                Py_INCREF(f_code);
#    endif
                int lineno = PyFrame_GetLineNumber(frame);
                result += "  ";
                result += handle(f_code->co_filename).cast<std::string>();
                result += '(';
                result += std::to_string(lineno);
                result += "): ";
                result += handle(f_code->co_name).cast<std::string>();
                result += '\n';
                Py_DECREF(f_code);
#    if PY_VERSION_HEX >= 0x030900B1
                auto *b_frame = PyFrame_GetBack(frame);
#    else
                auto *b_frame = frame->f_back;
                Py_XINCREF(b_frame);
#    endif
                Py_DECREF(frame);
                frame = b_frame;
            }

            have_trace = true;
#endif //! defined(PYPY_VERSION)
        }

        if (!message_error_string.empty()) {
            if (!have_trace) {
                result += '\n';
            }
            result += "\nMESSAGE UNAVAILABLE DUE TO EXCEPTION: " + message_error_string;
        }

        return result;
    }

    std::string const &error_string() const {
        if (!m_lazy_error_string_completed) {
            m_lazy_error_string += ": " + format_value_and_trace();
            m_lazy_error_string_completed = true;
        }
        return m_lazy_error_string;
    }

    void restore() {
        if (m_restore_called) {
            pybind11_fail("Internal error: pybind11::detail::error_fetch_and_normalize::restore() "
                          "called a second time. ORIGINAL ERROR: "
                          + error_string());
        }
        PyErr_Restore(m_type.inc_ref().ptr(), m_value.inc_ref().ptr(), m_trace.inc_ref().ptr());
        m_restore_called = true;
    }

    bool matches(handle exc) const {
        return (PyErr_GivenExceptionMatches(m_type.ptr(), exc.ptr()) != 0);
    }

    // Not protecting these for simplicity.
    object m_type, m_value, m_trace;

private:
    // Only protecting invariants.
    mutable std::string m_lazy_error_string;
    mutable bool m_lazy_error_string_completed = false;
    mutable bool m_restore_called = false;
};

inline std::string error_string() {
    return error_fetch_and_normalize("pybind11::detail::error_string").error_string();
}

PYBIND11_NAMESPACE_END(detail)

/// Fetch and hold an error which was already set in Python.  An instance of this is typically
/// thrown to propagate python-side errors back through C++ which can either be caught manually or
/// else falls back to the function dispatcher (which then raises the captured error back to
/// python).
class PYBIND11_EXPORT_EXCEPTION error_already_set : public std::exception {
public:
    /// Fetches the current Python exception (using PyErr_Fetch()), which will clear the
    /// current Python error indicator.
    error_already_set()
        : m_fetched_error{new detail::error_fetch_and_normalize("pybind11::error_already_set"),
                          m_fetched_error_deleter} {}

    /// The what() result is built lazily on demand.
    /// WARNING: This member function needs to acquire the Python GIL. This can lead to
    ///          crashes (undefined behavior) if the Python interpreter is finalizing.
    const char *what() const noexcept override;

    /// Restores the currently-held Python error (which will clear the Python error indicator first
    /// if already set).
    /// NOTE: This member function will always restore the normalized exception, which may or may
    ///       not be the original Python exception.
    /// WARNING: The GIL must be held when this member function is called!
    void restore() { m_fetched_error->restore(); }

    /// If it is impossible to raise the currently-held error, such as in a destructor, we can
    /// write it out using Python's unraisable hook (`sys.unraisablehook`). The error context
    /// should be some object whose `repr()` helps identify the location of the error. Python
    /// already knows the type and value of the error, so there is no need to repeat that.
    void discard_as_unraisable(object err_context) {
        restore();
        PyErr_WriteUnraisable(err_context.ptr());
    }
    /// An alternate version of `discard_as_unraisable()`, where a string provides information on
    /// the location of the error. For example, `__func__` could be helpful.
    /// WARNING: The GIL must be held when this member function is called!
    void discard_as_unraisable(const char *err_context) {
        discard_as_unraisable(reinterpret_steal<object>(PYBIND11_FROM_STRING(err_context)));
    }

    // Does nothing; provided for backwards compatibility.
    PYBIND11_DEPRECATED("Use of error_already_set.clear() is deprecated")
    void clear() {}

    /// Check if the currently trapped error type matches the given Python exception class (or a
    /// subclass thereof).  May also be passed a tuple to search for any exception class matches in
    /// the given tuple.
    bool matches(handle exc) const { return m_fetched_error->matches(exc); }

    const object &type() const { return m_fetched_error->m_type; }
    const object &value() const { return m_fetched_error->m_value; }
    const object &trace() const { return m_fetched_error->m_trace; }

private:
    std::shared_ptr<detail::error_fetch_and_normalize> m_fetched_error;

    /// WARNING: This custom deleter needs to acquire the Python GIL. This can lead to
    ///          crashes (undefined behavior) if the Python interpreter is finalizing.
    static void m_fetched_error_deleter(detail::error_fetch_and_normalize *raw_ptr);
};

/// Replaces the current Python error indicator with the chosen error, performing a
/// 'raise from' to indicate that the chosen error was caused by the original error.
inline void raise_from(PyObject *type, const char *message) {
    // Based on _PyErr_FormatVFromCause:
    // https://github.com/python/cpython/blob/467ab194fc6189d9f7310c89937c51abeac56839/Python/errors.c#L405
    // See https://github.com/pybind/pybind11/pull/2112 for details.
    PyObject *exc = nullptr, *val = nullptr, *val2 = nullptr, *tb = nullptr;

    assert(PyErr_Occurred());
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_NormalizeException(&exc, &val, &tb);
    if (tb != nullptr) {
        PyException_SetTraceback(val, tb);
        Py_DECREF(tb);
    }
    Py_DECREF(exc);
    assert(!PyErr_Occurred());

    PyErr_SetString(type, message);

    PyErr_Fetch(&exc, &val2, &tb);
    PyErr_NormalizeException(&exc, &val2, &tb);
    Py_INCREF(val);
    PyException_SetCause(val2, val);
    PyException_SetContext(val2, val);
    PyErr_Restore(exc, val2, tb);
}

/// Sets the current Python error indicator with the chosen error, performing a 'raise from'
/// from the error contained in error_already_set to indicate that the chosen error was
/// caused by the original error.
inline void raise_from(error_already_set &err, PyObject *type, const char *message) {
    err.restore();
    raise_from(type, message);
}

/** \defgroup python_builtins const_name
    Unless stated otherwise, the following C++ functions behave the same
    as their Python counterparts.
 */

/** \ingroup python_builtins
    \rst
    Return true if ``obj`` is an instance of ``T``. Type ``T`` must be a subclass of
    `object` or a class which was exposed to Python as ``py::class_<T>``.
\endrst */
template <typename T, detail::enable_if_t<std::is_base_of<object, T>::value, int> = 0>
bool isinstance(handle obj) {
    return T::check_(obj);
}

template <typename T, detail::enable_if_t<!std::is_base_of<object, T>::value, int> = 0>
bool isinstance(handle obj) {
    return detail::isinstance_generic(obj, typeid(T));
}

template <>
inline bool isinstance<handle>(handle) = delete;
template <>
inline bool isinstance<object>(handle obj) {
    return obj.ptr() != nullptr;
}

/// \ingroup python_builtins
/// Return true if ``obj`` is an instance of the ``type``.
inline bool isinstance(handle obj, handle type) {
    const auto result = PyObject_IsInstance(obj.ptr(), type.ptr());
    if (result == -1) {
        throw error_already_set();
    }
    return result != 0;
}

/// \addtogroup python_builtins
/// @{
inline bool hasattr(handle obj, handle name) {
    return PyObject_HasAttr(obj.ptr(), name.ptr()) == 1;
}

inline bool hasattr(handle obj, const char *name) {
    return PyObject_HasAttrString(obj.ptr(), name) == 1;
}

inline void delattr(handle obj, handle name) {
    if (PyObject_DelAttr(obj.ptr(), name.ptr()) != 0) {
        throw error_already_set();
    }
}

inline void delattr(handle obj, const char *name) {
    if (PyObject_DelAttrString(obj.ptr(), name) != 0) {
        throw error_already_set();
    }
}

inline object getattr(handle obj, handle name) {
    PyObject *result = PyObject_GetAttr(obj.ptr(), name.ptr());
    if (!result) {
        throw error_already_set();
    }
    return reinterpret_steal<object>(result);
}

inline object getattr(handle obj, const char *name) {
    PyObject *result = PyObject_GetAttrString(obj.ptr(), name);
    if (!result) {
        throw error_already_set();
    }
    return reinterpret_steal<object>(result);
}

inline object getattr(handle obj, handle name, handle default_) {
    if (PyObject *result = PyObject_GetAttr(obj.ptr(), name.ptr())) {
        return reinterpret_steal<object>(result);
    }
    PyErr_Clear();
    return reinterpret_borrow<object>(default_);
}

inline object getattr(handle obj, const char *name, handle default_) {
    if (PyObject *result = PyObject_GetAttrString(obj.ptr(), name)) {
        return reinterpret_steal<object>(result);
    }
    PyErr_Clear();
    return reinterpret_borrow<object>(default_);
}

inline void setattr(handle obj, handle name, handle value) {
    if (PyObject_SetAttr(obj.ptr(), name.ptr(), value.ptr()) != 0) {
        throw error_already_set();
    }
}

inline void setattr(handle obj, const char *name, handle value) {
    if (PyObject_SetAttrString(obj.ptr(), name, value.ptr()) != 0) {
        throw error_already_set();
    }
}

inline ssize_t hash(handle obj) {
    auto h = PyObject_Hash(obj.ptr());
    if (h == -1) {
        throw error_already_set();
    }
    return h;
}

/// @} python_builtins

PYBIND11_NAMESPACE_BEGIN(detail)
inline handle get_function(handle value) {
    if (value) {
        if (PyInstanceMethod_Check(value.ptr())) {
            value = PyInstanceMethod_GET_FUNCTION(value.ptr());
        } else if (PyMethod_Check(value.ptr())) {
            value = PyMethod_GET_FUNCTION(value.ptr());
        }
    }
    return value;
}

// Reimplementation of python's dict helper functions to ensure that exceptions
// aren't swallowed (see #2862)

// copied from cpython _PyDict_GetItemStringWithError
inline PyObject *dict_getitemstring(PyObject *v, const char *key) {
    PyObject *kv = nullptr, *rv = nullptr;
    kv = PyUnicode_FromString(key);
    if (kv == nullptr) {
        throw error_already_set();
    }

    rv = PyDict_GetItemWithError(v, kv);
    Py_DECREF(kv);
    if (rv == nullptr && PyErr_Occurred()) {
        throw error_already_set();
    }
    return rv;
}

inline PyObject *dict_getitem(PyObject *v, PyObject *key) {
    PyObject *rv = PyDict_GetItemWithError(v, key);
    if (rv == nullptr && PyErr_Occurred()) {
        throw error_already_set();
    }
    return rv;
}

// Helper aliases/functions to support implicit casting of values given to python
// accessors/methods. When given a pyobject, this simply returns the pyobject as-is; for other C++
// type, the value goes through pybind11::cast(obj) to convert it to an `object`.
template <typename T, enable_if_t<is_pyobject<T>::value, int> = 0>
auto object_or_cast(T &&o) -> decltype(std::forward<T>(o)) {
    return std::forward<T>(o);
}
// The following casting version is implemented in cast.h:
template <typename T, enable_if_t<!is_pyobject<T>::value, int> = 0>
object object_or_cast(T &&o);
// Match a PyObject*, which we want to convert directly to handle via its converting constructor
inline handle object_or_cast(PyObject *ptr) { return ptr; }

PYBIND11_WARNING_PUSH
PYBIND11_WARNING_DISABLE_MSVC(4522) // warning C4522: multiple assignment operators specified
template <typename Policy>
class accessor : public object_api<accessor<Policy>> {
    using key_type = typename Policy::key_type;

public:
    accessor(handle obj, key_type key) : obj(obj), key(std::move(key)) {}
    accessor(const accessor &) = default;
    accessor(accessor &&) noexcept = default;

    // accessor overload required to override default assignment operator (templates are not
    // allowed to replace default compiler-generated assignments).
    void operator=(const accessor &a) && { std::move(*this).operator=(handle(a)); }
    void operator=(const accessor &a) & { operator=(handle(a)); }

    template <typename T>
    void operator=(T &&value) && {
        Policy::set(obj, key, object_or_cast(std::forward<T>(value)));
    }
    template <typename T>
    void operator=(T &&value) & {
        get_cache() = ensure_object(object_or_cast(std::forward<T>(value)));
    }

    template <typename T = Policy>
    PYBIND11_DEPRECATED(
        "Use of obj.attr(...) as bool is deprecated in favor of pybind11::hasattr(obj, ...)")
    explicit
    operator enable_if_t<std::is_same<T, accessor_policies::str_attr>::value
                             || std::is_same<T, accessor_policies::obj_attr>::value,
                         bool>() const {
        return hasattr(obj, key);
    }
    template <typename T = Policy>
    PYBIND11_DEPRECATED("Use of obj[key] as bool is deprecated in favor of obj.contains(key)")
    explicit
    operator enable_if_t<std::is_same<T, accessor_policies::generic_item>::value, bool>() const {
        return obj.contains(key);
    }

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator object() const { return get_cache(); }
    PyObject *ptr() const { return get_cache().ptr(); }
    template <typename T>
    T cast() const {
        return get_cache().template cast<T>();
    }

private:
    static object ensure_object(object &&o) { return std::move(o); }
    static object ensure_object(handle h) { return reinterpret_borrow<object>(h); }

    object &get_cache() const {
        if (!cache) {
            cache = Policy::get(obj, key);
        }
        return cache;
    }

private:
    handle obj;
    key_type key;
    mutable object cache;
};
PYBIND11_WARNING_POP

PYBIND11_NAMESPACE_BEGIN(accessor_policies)
struct obj_attr {
    using key_type = object;
    static object get(handle obj, handle key) { return getattr(obj, key); }
    static void set(handle obj, handle key, handle val) { setattr(obj, key, val); }
};

struct str_attr {
    using key_type = const char *;
    static object get(handle obj, const char *key) { return getattr(obj, key); }
    static void set(handle obj, const char *key, handle val) { setattr(obj, key, val); }
};

struct generic_item {
    using key_type = object;

    static object get(handle obj, handle key) {
        PyObject *result = PyObject_GetItem(obj.ptr(), key.ptr());
        if (!result) {
            throw error_already_set();
        }
        return reinterpret_steal<object>(result);
    }

    static void set(handle obj, handle key, handle val) {
        if (PyObject_SetItem(obj.ptr(), key.ptr(), val.ptr()) != 0) {
            throw error_already_set();
        }
    }
};

struct sequence_item {
    using key_type = size_t;

    template <typename IdxType, detail::enable_if_t<std::is_integral<IdxType>::value, int> = 0>
    static object get(handle obj, const IdxType &index) {
        PyObject *result = PySequence_GetItem(obj.ptr(), ssize_t_cast(index));
        if (!result) {
            throw error_already_set();
        }
        return reinterpret_steal<object>(result);
    }

    template <typename IdxType, detail::enable_if_t<std::is_integral<IdxType>::value, int> = 0>
    static void set(handle obj, const IdxType &index, handle val) {
        // PySequence_SetItem does not steal a reference to 'val'
        if (PySequence_SetItem(obj.ptr(), ssize_t_cast(index), val.ptr()) != 0) {
            throw error_already_set();
        }
    }
};

struct list_item {
    using key_type = size_t;

    template <typename IdxType, detail::enable_if_t<std::is_integral<IdxType>::value, int> = 0>
    static object get(handle obj, const IdxType &index) {
        PyObject *result = PyList_GetItem(obj.ptr(), ssize_t_cast(index));
        if (!result) {
            throw error_already_set();
        }
        return reinterpret_borrow<object>(result);
    }

    template <typename IdxType, detail::enable_if_t<std::is_integral<IdxType>::value, int> = 0>
    static void set(handle obj, const IdxType &index, handle val) {
        // PyList_SetItem steals a reference to 'val'
        if (PyList_SetItem(obj.ptr(), ssize_t_cast(index), val.inc_ref().ptr()) != 0) {
            throw error_already_set();
        }
    }
};

struct tuple_item {
    using key_type = size_t;

    template <typename IdxType, detail::enable_if_t<std::is_integral<IdxType>::value, int> = 0>
    static object get(handle obj, const IdxType &index) {
        PyObject *result = PyTuple_GetItem(obj.ptr(), ssize_t_cast(index));
        if (!result) {
            throw error_already_set();
        }
        return reinterpret_borrow<object>(result);
    }

    template <typename IdxType, detail::enable_if_t<std::is_integral<IdxType>::value, int> = 0>
    static void set(handle obj, const IdxType &index, handle val) {
        // PyTuple_SetItem steals a reference to 'val'
        if (PyTuple_SetItem(obj.ptr(), ssize_t_cast(index), val.inc_ref().ptr()) != 0) {
            throw error_already_set();
        }
    }
};
PYBIND11_NAMESPACE_END(accessor_policies)

/// STL iterator template used for tuple, list, sequence and dict
template <typename Policy>
class generic_iterator : public Policy {
    using It = generic_iterator;

public:
    using difference_type = ssize_t;
    using iterator_category = typename Policy::iterator_category;
    using value_type = typename Policy::value_type;
    using reference = typename Policy::reference;
    using pointer = typename Policy::pointer;

    generic_iterator() = default;
    generic_iterator(handle seq, ssize_t index) : Policy(seq, index) {}

    // NOLINTNEXTLINE(readability-const-return-type) // PR #3263
    reference operator*() const { return Policy::dereference(); }
    // NOLINTNEXTLINE(readability-const-return-type) // PR #3263
    reference operator[](difference_type n) const { return *(*this + n); }
    pointer operator->() const { return **this; }

    It &operator++() {
        Policy::increment();
        return *this;
    }
    It operator++(int) {
        auto copy = *this;
        Policy::increment();
        return copy;
    }
    It &operator--() {
        Policy::decrement();
        return *this;
    }
    It operator--(int) {
        auto copy = *this;
        Policy::decrement();
        return copy;
    }
    It &operator+=(difference_type n) {
        Policy::advance(n);
        return *this;
    }
    It &operator-=(difference_type n) {
        Policy::advance(-n);
        return *this;
    }

    friend It operator+(const It &a, difference_type n) {
        auto copy = a;
        return copy += n;
    }
    friend It operator+(difference_type n, const It &b) { return b + n; }
    friend It operator-(const It &a, difference_type n) {
        auto copy = a;
        return copy -= n;
    }
    friend difference_type operator-(const It &a, const It &b) { return a.distance_to(b); }

    friend bool operator==(const It &a, const It &b) { return a.equal(b); }
    friend bool operator!=(const It &a, const It &b) { return !(a == b); }
    friend bool operator<(const It &a, const It &b) { return b - a > 0; }
    friend bool operator>(const It &a, const It &b) { return b < a; }
    friend bool operator>=(const It &a, const It &b) { return !(a < b); }
    friend bool operator<=(const It &a, const It &b) { return !(a > b); }
};

PYBIND11_NAMESPACE_BEGIN(iterator_policies)
/// Quick proxy class needed to implement ``operator->`` for iterators which can't return pointers
template <typename T>
struct arrow_proxy {
    T value;

    // NOLINTNEXTLINE(google-explicit-constructor)
    arrow_proxy(T &&value) noexcept : value(std::move(value)) {}
    T *operator->() const { return &value; }
};

/// Lightweight iterator policy using just a simple pointer: see ``PySequence_Fast_ITEMS``
class sequence_fast_readonly {
protected:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = handle;
    using reference = const handle; // PR #3263
    using pointer = arrow_proxy<const handle>;

    sequence_fast_readonly(handle obj, ssize_t n) : ptr(PySequence_Fast_ITEMS(obj.ptr()) + n) {}

    // NOLINTNEXTLINE(readability-const-return-type) // PR #3263
    reference dereference() const { return *ptr; }
    void increment() { ++ptr; }
    void decrement() { --ptr; }
    void advance(ssize_t n) { ptr += n; }
    bool equal(const sequence_fast_readonly &b) const { return ptr == b.ptr; }
    ssize_t distance_to(const sequence_fast_readonly &b) const { return ptr - b.ptr; }

private:
    PyObject **ptr;
};

/// Full read and write access using the sequence protocol: see ``detail::sequence_accessor``
class sequence_slow_readwrite {
protected:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = object;
    using reference = sequence_accessor;
    using pointer = arrow_proxy<const sequence_accessor>;

    sequence_slow_readwrite(handle obj, ssize_t index) : obj(obj), index(index) {}

    reference dereference() const { return {obj, static_cast<size_t>(index)}; }
    void increment() { ++index; }
    void decrement() { --index; }
    void advance(ssize_t n) { index += n; }
    bool equal(const sequence_slow_readwrite &b) const { return index == b.index; }
    ssize_t distance_to(const sequence_slow_readwrite &b) const { return index - b.index; }

private:
    handle obj;
    ssize_t index;
};

/// Python's dictionary protocol permits this to be a forward iterator
class dict_readonly {
protected:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<handle, handle>;
    using reference = const value_type; // PR #3263
    using pointer = arrow_proxy<const value_type>;

    dict_readonly() = default;
    dict_readonly(handle obj, ssize_t pos) : obj(obj), pos(pos) { increment(); }

    // NOLINTNEXTLINE(readability-const-return-type) // PR #3263
    reference dereference() const { return {key, value}; }
    void increment() {
        if (PyDict_Next(obj.ptr(), &pos, &key, &value) == 0) {
            pos = -1;
        }
    }
    bool equal(const dict_readonly &b) const { return pos == b.pos; }

private:
    handle obj;
    PyObject *key = nullptr, *value = nullptr;
    ssize_t pos = -1;
};
PYBIND11_NAMESPACE_END(iterator_policies)

#if !defined(PYPY_VERSION)
using tuple_iterator = generic_iterator<iterator_policies::sequence_fast_readonly>;
using list_iterator = generic_iterator<iterator_policies::sequence_fast_readonly>;
#else
using tuple_iterator = generic_iterator<iterator_policies::sequence_slow_readwrite>;
using list_iterator = generic_iterator<iterator_policies::sequence_slow_readwrite>;
#endif

using sequence_iterator = generic_iterator<iterator_policies::sequence_slow_readwrite>;
using dict_iterator = generic_iterator<iterator_policies::dict_readonly>;

inline bool PyIterable_Check(PyObject *obj) {
    PyObject *iter = PyObject_GetIter(obj);
    if (iter) {
        Py_DECREF(iter);
        return true;
    }
    PyErr_Clear();
    return false;
}

inline bool PyNone_Check(PyObject *o) { return o == Py_None; }
inline bool PyEllipsis_Check(PyObject *o) { return o == Py_Ellipsis; }

#ifdef PYBIND11_STR_LEGACY_PERMISSIVE
inline bool PyUnicode_Check_Permissive(PyObject *o) {
    return PyUnicode_Check(o) || PYBIND11_BYTES_CHECK(o);
}
#    define PYBIND11_STR_CHECK_FUN detail::PyUnicode_Check_Permissive
#else
#    define PYBIND11_STR_CHECK_FUN PyUnicode_Check
#endif

inline bool PyStaticMethod_Check(PyObject *o) { return o->ob_type == &PyStaticMethod_Type; }

class kwargs_proxy : public handle {
public:
    explicit kwargs_proxy(handle h) : handle(h) {}
};

class args_proxy : public handle {
public:
    explicit args_proxy(handle h) : handle(h) {}
    kwargs_proxy operator*() const { return kwargs_proxy(*this); }
};

/// Python argument categories (using PEP 448 terms)
template <typename T>
using is_keyword = std::is_base_of<arg, T>;
template <typename T>
using is_s_unpacking = std::is_same<args_proxy, T>; // * unpacking
template <typename T>
using is_ds_unpacking = std::is_same<kwargs_proxy, T>; // ** unpacking
template <typename T>
using is_positional = satisfies_none_of<T, is_keyword, is_s_unpacking, is_ds_unpacking>;
template <typename T>
using is_keyword_or_ds = satisfies_any_of<T, is_keyword, is_ds_unpacking>;

// Call argument collector forward declarations
template <return_value_policy policy = return_value_policy::automatic_reference>
class simple_collector;
template <return_value_policy policy = return_value_policy::automatic_reference>
class unpacking_collector;

PYBIND11_NAMESPACE_END(detail)

// TODO: After the deprecated constructors are removed, this macro can be simplified by
//       inheriting ctors: `using Parent::Parent`. It's not an option right now because
//       the `using` statement triggers the parent deprecation warning even if the ctor
//       isn't even used.
#define PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun)                                            \
public:                                                                                           \
    PYBIND11_DEPRECATED("Use reinterpret_borrow<" #Name ">() or reinterpret_steal<" #Name ">()")  \
    Name(handle h, bool is_borrowed)                                                              \
        : Parent(is_borrowed ? Parent(h, borrowed_t{}) : Parent(h, stolen_t{})) {}                \
    Name(handle h, borrowed_t) : Parent(h, borrowed_t{}) {}                                       \
    Name(handle h, stolen_t) : Parent(h, stolen_t{}) {}                                           \
    PYBIND11_DEPRECATED("Use py::isinstance<py::python_type>(obj) instead")                       \
    bool check() const { return m_ptr != nullptr && (CheckFun(m_ptr) != 0); }                     \
    static bool check_(handle h) { return h.ptr() != nullptr && CheckFun(h.ptr()); }              \
    template <typename Policy_> /* NOLINTNEXTLINE(google-explicit-constructor) */                 \
    Name(const ::pybind11::detail::accessor<Policy_> &a) : Name(object(a)) {}

#define PYBIND11_OBJECT_CVT(Name, Parent, CheckFun, ConvertFun)                                   \
    PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun)                                                \
    /* This is deliberately not 'explicit' to allow implicit conversion from object: */           \
    /* NOLINTNEXTLINE(google-explicit-constructor) */                                             \
    Name(const object &o)                                                                         \
        : Parent(check_(o) ? o.inc_ref().ptr() : ConvertFun(o.ptr()), stolen_t{}) {               \
        if (!m_ptr)                                                                               \
            throw ::pybind11::error_already_set();                                                \
    }                                                                                             \
    /* NOLINTNEXTLINE(google-explicit-constructor) */                                             \
    Name(object &&o) : Parent(check_(o) ? o.release().ptr() : ConvertFun(o.ptr()), stolen_t{}) {  \
        if (!m_ptr)                                                                               \
            throw ::pybind11::error_already_set();                                                \
    }

#define PYBIND11_OBJECT_CVT_DEFAULT(Name, Parent, CheckFun, ConvertFun)                           \
    PYBIND11_OBJECT_CVT(Name, Parent, CheckFun, ConvertFun)                                       \
    Name() = default;

#define PYBIND11_OBJECT_CHECK_FAILED(Name, o_ptr)                                                 \
    ::pybind11::type_error("Object of type '"                                                     \
                           + ::pybind11::detail::get_fully_qualified_tp_name(Py_TYPE(o_ptr))      \
                           + "' is not an instance of '" #Name "'")

#define PYBIND11_OBJECT(Name, Parent, CheckFun)                                                   \
    PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun)                                                \
    /* This is deliberately not 'explicit' to allow implicit conversion from object: */           \
    /* NOLINTNEXTLINE(google-explicit-constructor) */                                             \
    Name(const object &o) : Parent(o) {                                                           \
        if (m_ptr && !check_(m_ptr))                                                              \
            throw PYBIND11_OBJECT_CHECK_FAILED(Name, m_ptr);                                      \
    }                                                                                             \
    /* NOLINTNEXTLINE(google-explicit-constructor) */                                             \
    Name(object &&o) : Parent(std::move(o)) {                                                     \
        if (m_ptr && !check_(m_ptr))                                                              \
            throw PYBIND11_OBJECT_CHECK_FAILED(Name, m_ptr);                                      \
    }

#define PYBIND11_OBJECT_DEFAULT(Name, Parent, CheckFun)                                           \
    PYBIND11_OBJECT(Name, Parent, CheckFun)                                                       \
    Name() = default;

/// \addtogroup pytypes
/// @{

/** \rst
    Wraps a Python iterator so that it can also be used as a C++ input iterator

    Caveat: copying an iterator does not (and cannot) clone the internal
    state of the Python iterable. This also applies to the post-increment
    operator. This iterator should only be used to retrieve the current
    value using ``operator*()``.
\endrst */
class iterator : public object {
public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = ssize_t;
    using value_type = handle;
    using reference = const handle; // PR #3263
    using pointer = const handle *;

    PYBIND11_OBJECT_DEFAULT(iterator, object, PyIter_Check)

    iterator &operator++() {
        advance();
        return *this;
    }

    iterator operator++(int) {
        auto rv = *this;
        advance();
        return rv;
    }

    // NOLINTNEXTLINE(readability-const-return-type) // PR #3263
    reference operator*() const {
        if (m_ptr && !value.ptr()) {
            auto &self = const_cast<iterator &>(*this);
            self.advance();
        }
        return value;
    }

    pointer operator->() const {
        operator*();
        return &value;
    }

    /** \rst
         The value which marks the end of the iteration. ``it == iterator::sentinel()``
         is equivalent to catching ``StopIteration`` in Python.

         .. code-block:: cpp

             void foo(py::iterator it) {
                 while (it != py::iterator::sentinel()) {
                    // use `*it`
                    ++it;
                 }
             }
    \endrst */
    static iterator sentinel() { return {}; }

    friend bool operator==(const iterator &a, const iterator &b) { return a->ptr() == b->ptr(); }
    friend bool operator!=(const iterator &a, const iterator &b) { return a->ptr() != b->ptr(); }

private:
    void advance() {
        value = reinterpret_steal<object>(PyIter_Next(m_ptr));
        if (value.ptr() == nullptr && PyErr_Occurred()) {
            throw error_already_set();
        }
    }

private:
    object value = {};
};

class type : public object {
public:
    PYBIND11_OBJECT(type, object, PyType_Check)

    /// Return a type handle from a handle or an object
    static handle handle_of(handle h) { return handle((PyObject *) Py_TYPE(h.ptr())); }

    /// Return a type object from a handle or an object
    static type of(handle h) { return type(type::handle_of(h), borrowed_t{}); }

    // Defined in pybind11/cast.h
    /// Convert C++ type to handle if previously registered. Does not convert
    /// standard types, like int, float. etc. yet.
    /// See https://github.com/pybind/pybind11/issues/2486
    template <typename T>
    static handle handle_of();

    /// Convert C++ type to type if previously registered. Does not convert
    /// standard types, like int, float. etc. yet.
    /// See https://github.com/pybind/pybind11/issues/2486
    template <typename T>
    static type of() {
        return type(type::handle_of<T>(), borrowed_t{});
    }
};

class iterable : public object {
public:
    PYBIND11_OBJECT_DEFAULT(iterable, object, detail::PyIterable_Check)
};

class bytes;

class str : public object {
public:
    PYBIND11_OBJECT_CVT(str, object, PYBIND11_STR_CHECK_FUN, raw_str)

    template <typename SzType, detail::enable_if_t<std::is_integral<SzType>::value, int> = 0>
    str(const char *c, const SzType &n)
        : object(PyUnicode_FromStringAndSize(c, ssize_t_cast(n)), stolen_t{}) {
        if (!m_ptr) {
            if (PyErr_Occurred()) {
                throw error_already_set();
            }
            pybind11_fail("Could not allocate string object!");
        }
    }

    // 'explicit' is explicitly omitted from the following constructors to allow implicit
    // conversion to py::str from C++ string-like objects
    // NOLINTNEXTLINE(google-explicit-constructor)
    str(const char *c = "") : object(PyUnicode_FromString(c), stolen_t{}) {
        if (!m_ptr) {
            if (PyErr_Occurred()) {
                throw error_already_set();
            }
            pybind11_fail("Could not allocate string object!");
        }
    }

    // NOLINTNEXTLINE(google-explicit-constructor)
    str(const std::string &s) : str(s.data(), s.size()) {}

#ifdef PYBIND11_HAS_STRING_VIEW
    // enable_if is needed to avoid "ambiguous conversion" errors (see PR #3521).
    template <typename T, detail::enable_if_t<std::is_same<T, std::string_view>::value, int> = 0>
    // NOLINTNEXTLINE(google-explicit-constructor)
    str(T s) : str(s.data(), s.size()) {}

#    ifdef PYBIND11_HAS_U8STRING
    // reinterpret_cast here is safe (C++20 guarantees char8_t has the same size/alignment as char)
    // NOLINTNEXTLINE(google-explicit-constructor)
    str(std::u8string_view s) : str(reinterpret_cast<const char *>(s.data()), s.size()) {}
#    endif

#endif

    explicit str(const bytes &b);

    /** \rst
        Return a string representation of the object. This is analogous to
        the ``str()`` function in Python.
    \endrst */
    explicit str(handle h) : object(raw_str(h.ptr()), stolen_t{}) {
        if (!m_ptr) {
            throw error_already_set();
        }
    }

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator std::string() const {
        object temp = *this;
        if (PyUnicode_Check(m_ptr)) {
            temp = reinterpret_steal<object>(PyUnicode_AsUTF8String(m_ptr));
            if (!temp) {
                throw error_already_set();
            }
        }
        char *buffer = nullptr;
        ssize_t length = 0;
        if (PyBytes_AsStringAndSize(temp.ptr(), &buffer, &length) != 0) {
            throw error_already_set();
        }
        return std::string(buffer, (size_t) length);
    }

    template <typename... Args>
    str format(Args &&...args) const {
        return attr("format")(std::forward<Args>(args)...);
    }

private:
    /// Return string representation -- always returns a new reference, even if already a str
    static PyObject *raw_str(PyObject *op) {
        PyObject *str_value = PyObject_Str(op);
        return str_value;
    }
};
/// @} pytypes

inline namespace literals {
/** \rst
    String literal version of `str`
 \endrst */
inline str operator"" _s(const char *s, size_t size) { return {s, size}; }
} // namespace literals

/// \addtogroup pytypes
/// @{
class bytes : public object {
public:
    PYBIND11_OBJECT(bytes, object, PYBIND11_BYTES_CHECK)

    // Allow implicit conversion:
    // NOLINTNEXTLINE(google-explicit-constructor)
    bytes(const char *c = "") : object(PYBIND11_BYTES_FROM_STRING(c), stolen_t{}) {
        if (!m_ptr) {
            pybind11_fail("Could not allocate bytes object!");
        }
    }

    template <typename SzType, detail::enable_if_t<std::is_integral<SzType>::value, int> = 0>
    bytes(const char *c, const SzType &n)
        : object(PYBIND11_BYTES_FROM_STRING_AND_SIZE(c, ssize_t_cast(n)), stolen_t{}) {
        if (!m_ptr) {
            pybind11_fail("Could not allocate bytes object!");
        }
    }

    // Allow implicit conversion:
    // NOLINTNEXTLINE(google-explicit-constructor)
    bytes(const std::string &s) : bytes(s.data(), s.size()) {}

    explicit bytes(const pybind11::str &s);

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator std::string() const { return string_op<std::string>(); }

#ifdef PYBIND11_HAS_STRING_VIEW
    // enable_if is needed to avoid "ambiguous conversion" errors (see PR #3521).
    template <typename T, detail::enable_if_t<std::is_same<T, std::string_view>::value, int> = 0>
    // NOLINTNEXTLINE(google-explicit-constructor)
    bytes(T s) : bytes(s.data(), s.size()) {}

    // Obtain a string view that views the current `bytes` buffer value.  Note that this is only
    // valid so long as the `bytes` instance remains alive and so generally should not outlive the
    // lifetime of the `bytes` instance.
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator std::string_view() const { return string_op<std::string_view>(); }
#endif
private:
    template <typename T>
    T string_op() const {
        char *buffer = nullptr;
        ssize_t length = 0;
        if (PyBytes_AsStringAndSize(m_ptr, &buffer, &length) != 0) {
            throw error_already_set();
        }
        return {buffer, static_cast<size_t>(length)};
    }
};
// Note: breathe >= 4.17.0 will fail to build docs if the below two constructors
// are included in the doxygen group; close here and reopen after as a workaround
/// @} pytypes

inline bytes::bytes(const pybind11::str &s) {
    object temp = s;
    if (PyUnicode_Check(s.ptr())) {
        temp = reinterpret_steal<object>(PyUnicode_AsUTF8String(s.ptr()));
        if (!temp) {
            throw error_already_set();
        }
    }
    char *buffer = nullptr;
    ssize_t length = 0;
    if (PyBytes_AsStringAndSize(temp.ptr(), &buffer, &length) != 0) {
        throw error_already_set();
    }
    auto obj = reinterpret_steal<object>(PYBIND11_BYTES_FROM_STRING_AND_SIZE(buffer, length));
    if (!obj) {
        pybind11_fail("Could not allocate bytes object!");
    }
    m_ptr = obj.release().ptr();
}

inline str::str(const bytes &b) {
    char *buffer = nullptr;
    ssize_t length = 0;
    if (PyBytes_AsStringAndSize(b.ptr(), &buffer, &length) != 0) {
        throw error_already_set();
    }
    auto obj = reinterpret_steal<object>(PyUnicode_FromStringAndSize(buffer, length));
    if (!obj) {
        if (PyErr_Occurred()) {
            throw error_already_set();
        }
        pybind11_fail("Could not allocate string object!");
    }
    m_ptr = obj.release().ptr();
}

/// \addtogroup pytypes
/// @{
class bytearray : public object {
public:
    PYBIND11_OBJECT_CVT(bytearray, object, PyByteArray_Check, PyByteArray_FromObject)

    template <typename SzType, detail::enable_if_t<std::is_integral<SzType>::value, int> = 0>
    bytearray(const char *c, const SzType &n)
        : object(PyByteArray_FromStringAndSize(c, ssize_t_cast(n)), stolen_t{}) {
        if (!m_ptr) {
            pybind11_fail("Could not allocate bytearray object!");
        }
    }

    bytearray() : bytearray("", 0) {}

    explicit bytearray(const std::string &s) : bytearray(s.data(), s.size()) {}

    size_t size() const { return static_cast<size_t>(PyByteArray_Size(m_ptr)); }

    explicit operator std::string() const {
        char *buffer = PyByteArray_AS_STRING(m_ptr);
        ssize_t size = PyByteArray_GET_SIZE(m_ptr);
        return std::string(buffer, static_cast<size_t>(size));
    }
};
// Note: breathe >= 4.17.0 will fail to build docs if the below two constructors
// are included in the doxygen group; close here and reopen after as a workaround
/// @} pytypes

/// \addtogroup pytypes
/// @{
class none : public object {
public:
    PYBIND11_OBJECT(none, object, detail::PyNone_Check)
    none() : object(Py_None, borrowed_t{}) {}
};

class ellipsis : public object {
public:
    PYBIND11_OBJECT(ellipsis, object, detail::PyEllipsis_Check)
    ellipsis() : object(Py_Ellipsis, borrowed_t{}) {}
};

class bool_ : public object {
public:
    PYBIND11_OBJECT_CVT(bool_, object, PyBool_Check, raw_bool)
    bool_() : object(Py_False, borrowed_t{}) {}
    // Allow implicit conversion from and to `bool`:
    // NOLINTNEXTLINE(google-explicit-constructor)
    bool_(bool value) : object(value ? Py_True : Py_False, borrowed_t{}) {}
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator bool() const { return (m_ptr != nullptr) && PyLong_AsLong(m_ptr) != 0; }

private:
    /// Return the truth value of an object -- always returns a new reference
    static PyObject *raw_bool(PyObject *op) {
        const auto value = PyObject_IsTrue(op);
        if (value == -1) {
            return nullptr;
        }
        return handle(value != 0 ? Py_True : Py_False).inc_ref().ptr();
    }
};

PYBIND11_NAMESPACE_BEGIN(detail)
// Converts a value to the given unsigned type.  If an error occurs, you get back (Unsigned) -1;
// otherwise you get back the unsigned long or unsigned long long value cast to (Unsigned).
// (The distinction is critically important when casting a returned -1 error value to some other
// unsigned type: (A)-1 != (B)-1 when A and B are unsigned types of different sizes).
template <typename Unsigned>
Unsigned as_unsigned(PyObject *o) {
    if (sizeof(Unsigned) <= sizeof(unsigned long)) {
        unsigned long v = PyLong_AsUnsignedLong(o);
        return v == (unsigned long) -1 && PyErr_Occurred() ? (Unsigned) -1 : (Unsigned) v;
    }
    unsigned long long v = PyLong_AsUnsignedLongLong(o);
    return v == (unsigned long long) -1 && PyErr_Occurred() ? (Unsigned) -1 : (Unsigned) v;
}
PYBIND11_NAMESPACE_END(detail)

class int_ : public object {
public:
    PYBIND11_OBJECT_CVT(int_, object, PYBIND11_LONG_CHECK, PyNumber_Long)
    int_() : object(PyLong_FromLong(0), stolen_t{}) {}
    // Allow implicit conversion from C++ integral types:
    template <typename T, detail::enable_if_t<std::is_integral<T>::value, int> = 0>
    // NOLINTNEXTLINE(google-explicit-constructor)
    int_(T value) {
        if (sizeof(T) <= sizeof(long)) {
            if (std::is_signed<T>::value) {
                m_ptr = PyLong_FromLong((long) value);
            } else {
                m_ptr = PyLong_FromUnsignedLong((unsigned long) value);
            }
        } else {
            if (std::is_signed<T>::value) {
                m_ptr = PyLong_FromLongLong((long long) value);
            } else {
                m_ptr = PyLong_FromUnsignedLongLong((unsigned long long) value);
            }
        }
        if (!m_ptr) {
            pybind11_fail("Could not allocate int object!");
        }
    }

    template <typename T, detail::enable_if_t<std::is_integral<T>::value, int> = 0>
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator T() const {
        return std::is_unsigned<T>::value  ? detail::as_unsigned<T>(m_ptr)
               : sizeof(T) <= sizeof(long) ? (T) PyLong_AsLong(m_ptr)
                                           : (T) PYBIND11_LONG_AS_LONGLONG(m_ptr);
    }
};

class float_ : public object {
public:
    PYBIND11_OBJECT_CVT(float_, object, PyFloat_Check, PyNumber_Float)
    // Allow implicit conversion from float/double:
    // NOLINTNEXTLINE(google-explicit-constructor)
    float_(float value) : object(PyFloat_FromDouble((double) value), stolen_t{}) {
        if (!m_ptr) {
            pybind11_fail("Could not allocate float object!");
        }
    }
    // NOLINTNEXTLINE(google-explicit-constructor)
    float_(double value = .0) : object(PyFloat_FromDouble((double) value), stolen_t{}) {
        if (!m_ptr) {
            pybind11_fail("Could not allocate float object!");
        }
    }
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator float() const { return (float) PyFloat_AsDouble(m_ptr); }
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator double() const { return (double) PyFloat_AsDouble(m_ptr); }
};

class weakref : public object {
public:
    PYBIND11_OBJECT_CVT_DEFAULT(weakref, object, PyWeakref_Check, raw_weakref)
    explicit weakref(handle obj, handle callback = {})
        : object(PyWeakref_NewRef(obj.ptr(), callback.ptr()), stolen_t{}) {
        if (!m_ptr) {
            if (PyErr_Occurred()) {
                throw error_already_set();
            }
            pybind11_fail("Could not allocate weak reference!");
        }
    }

private:
    static PyObject *raw_weakref(PyObject *o) { return PyWeakref_NewRef(o, nullptr); }
};

class slice : public object {
public:
    PYBIND11_OBJECT_DEFAULT(slice, object, PySlice_Check)
    slice(handle start, handle stop, handle step)
        : object(PySlice_New(start.ptr(), stop.ptr(), step.ptr()), stolen_t{}) {
        if (!m_ptr) {
            pybind11_fail("Could not allocate slice object!");
        }
    }

#ifdef PYBIND11_HAS_OPTIONAL
    slice(std::optional<ssize_t> start, std::optional<ssize_t> stop, std::optional<ssize_t> step)
        : slice(index_to_object(start), index_to_object(stop), index_to_object(step)) {}
#else
    slice(ssize_t start_, ssize_t stop_, ssize_t step_)
        : slice(int_(start_), int_(stop_), int_(step_)) {}
#endif

    bool
    compute(size_t length, size_t *start, size_t *stop, size_t *step, size_t *slicelength) const {
        return PySlice_GetIndicesEx((PYBIND11_SLICE_OBJECT *) m_ptr,
                                    (ssize_t) length,
                                    (ssize_t *) start,
                                    (ssize_t *) stop,
                                    (ssize_t *) step,
                                    (ssize_t *) slicelength)
               == 0;
    }
    bool compute(
        ssize_t length, ssize_t *start, ssize_t *stop, ssize_t *step, ssize_t *slicelength) const {
        return PySlice_GetIndicesEx(
                   (PYBIND11_SLICE_OBJECT *) m_ptr, length, start, stop, step, slicelength)
               == 0;
    }

private:
    template <typename T>
    static object index_to_object(T index) {
        return index ? object(int_(*index)) : object(none());
    }
};

class capsule : public object {
public:
    PYBIND11_OBJECT_DEFAULT(capsule, object, PyCapsule_CheckExact)
    PYBIND11_DEPRECATED("Use reinterpret_borrow<capsule>() or reinterpret_steal<capsule>()")
    capsule(PyObject *ptr, bool is_borrowed)
        : object(is_borrowed ? object(ptr, borrowed_t{}) : object(ptr, stolen_t{})) {}

    explicit capsule(const void *value,
                     const char *name = nullptr,
                     PyCapsule_Destructor destructor = nullptr)
        : object(PyCapsule_New(const_cast<void *>(value), name, destructor), stolen_t{}) {
        if (!m_ptr) {
            throw error_already_set();
        }
    }

    PYBIND11_DEPRECATED("Please use the ctor with value, name, destructor args")
    capsule(const void *value, PyCapsule_Destructor destructor)
        : object(PyCapsule_New(const_cast<void *>(value), nullptr, destructor), stolen_t{}) {
        if (!m_ptr) {
            throw error_already_set();
        }
    }

    capsule(const void *value, void (*destructor)(void *)) {
        m_ptr = PyCapsule_New(const_cast<void *>(value), nullptr, [](PyObject *o) {
            // guard if destructor called while err indicator is set
            error_scope error_guard;
            auto destructor = reinterpret_cast<void (*)(void *)>(PyCapsule_GetContext(o));
            if (destructor == nullptr && PyErr_Occurred()) {
                throw error_already_set();
            }
            const char *name = get_name_in_error_scope(o);
            void *ptr = PyCapsule_GetPointer(o, name);
            if (ptr == nullptr) {
                throw error_already_set();
            }

            if (destructor != nullptr) {
                destructor(ptr);
            }
        });

        if (!m_ptr || PyCapsule_SetContext(m_ptr, reinterpret_cast<void *>(destructor)) != 0) {
            throw error_already_set();
        }
    }

    explicit capsule(void (*destructor)()) {
        m_ptr = PyCapsule_New(reinterpret_cast<void *>(destructor), nullptr, [](PyObject *o) {
            const char *name = get_name_in_error_scope(o);
            auto destructor = reinterpret_cast<void (*)()>(PyCapsule_GetPointer(o, name));
            if (destructor == nullptr) {
                throw error_already_set();
            }
            destructor();
        });

        if (!m_ptr) {
            throw error_already_set();
        }
    }

    template <typename T>
    operator T *() const { // NOLINT(google-explicit-constructor)
        return get_pointer<T>();
    }

    /// Get the pointer the capsule holds.
    template <typename T = void>
    T *get_pointer() const {
        const auto *name = this->name();
        T *result = static_cast<T *>(PyCapsule_GetPointer(m_ptr, name));
        if (!result) {
            throw error_already_set();
        }
        return result;
    }

    /// Replaces a capsule's pointer *without* calling the destructor on the existing one.
    void set_pointer(const void *value) {
        if (PyCapsule_SetPointer(m_ptr, const_cast<void *>(value)) != 0) {
            throw error_already_set();
        }
    }

    const char *name() const {
        const char *name = PyCapsule_GetName(m_ptr);
        if ((name == nullptr) && PyErr_Occurred()) {
            throw error_already_set();
        }
        return name;
    }

    /// Replaces a capsule's name *without* calling the destructor on the existing one.
    void set_name(const char *new_name) {
        if (PyCapsule_SetName(m_ptr, new_name) != 0) {
            throw error_already_set();
        }
    }

private:
    static const char *get_name_in_error_scope(PyObject *o) {
        error_scope error_guard;

        const char *name = PyCapsule_GetName(o);
        if ((name == nullptr) && PyErr_Occurred()) {
            // write out and consume error raised by call to PyCapsule_GetName
            PyErr_WriteUnraisable(o);
        }

        return name;
    }
};

class tuple : public object {
public:
    PYBIND11_OBJECT_CVT(tuple, object, PyTuple_Check, PySequence_Tuple)
    template <typename SzType = ssize_t,
              detail::enable_if_t<std::is_integral<SzType>::value, int> = 0>
    // Some compilers generate link errors when using `const SzType &` here:
    explicit tuple(SzType size = 0) : object(PyTuple_New(ssize_t_cast(size)), stolen_t{}) {
        if (!m_ptr) {
            pybind11_fail("Could not allocate tuple object!");
        }
    }
    size_t size() const { return (size_t) PyTuple_Size(m_ptr); }
    bool empty() const { return size() == 0; }
    detail::tuple_accessor operator[](size_t index) const { return {*this, index}; }
    template <typename T, detail::enable_if_t<detail::is_pyobject<T>::value, int> = 0>
    detail::item_accessor operator[](T &&o) const {
        return object::operator[](std::forward<T>(o));
    }
    detail::tuple_iterator begin() const { return {*this, 0}; }
    detail::tuple_iterator end() const { return {*this, PyTuple_GET_SIZE(m_ptr)}; }
};

// We need to put this into a separate function because the Intel compiler
// fails to compile enable_if_t<all_of<is_keyword_or_ds<Args>...>::value> part below
// (tested with ICC 2021.1 Beta 20200827).
template <typename... Args>
constexpr bool args_are_all_keyword_or_ds() {
    return detail::all_of<detail::is_keyword_or_ds<Args>...>::value;
}

class dict : public object {
public:
    PYBIND11_OBJECT_CVT(dict, object, PyDict_Check, raw_dict)
    dict() : object(PyDict_New(), stolen_t{}) {
        if (!m_ptr) {
            pybind11_fail("Could not allocate dict object!");
        }
    }
    template <typename... Args,
              typename = detail::enable_if_t<args_are_all_keyword_or_ds<Args...>()>,
              // MSVC workaround: it can't compile an out-of-line definition, so defer the
              // collector
              typename collector = detail::deferred_t<detail::unpacking_collector<>, Args...>>
    explicit dict(Args &&...args) : dict(collector(std::forward<Args>(args)...).kwargs()) {}

    size_t size() const { return (size_t) PyDict_Size(m_ptr); }
    bool empty() const { return size() == 0; }
    detail::dict_iterator begin() const { return {*this, 0}; }
    detail::dict_iterator end() const { return {}; }
    void clear() /* py-non-const */ { PyDict_Clear(ptr()); }
    template <typename T>
    bool contains(T &&key) const {
        auto result = PyDict_Contains(m_ptr, detail::object_or_cast(std::forward<T>(key)).ptr());
        if (result == -1) {
            throw error_already_set();
        }
        return result == 1;
    }

private:
    /// Call the `dict` Python type -- always returns a new reference
    static PyObject *raw_dict(PyObject *op) {
        if (PyDict_Check(op)) {
            return handle(op).inc_ref().ptr();
        }
        return PyObject_CallFunctionObjArgs((PyObject *) &PyDict_Type, op, nullptr);
    }
};

class sequence : public object {
public:
    PYBIND11_OBJECT_DEFAULT(sequence, object, PySequence_Check)
    size_t size() const {
        ssize_t result = PySequence_Size(m_ptr);
        if (result == -1) {
            throw error_already_set();
        }
        return (size_t) result;
    }
    bool empty() const { return size() == 0; }
    detail::sequence_accessor operator[](size_t index) const { return {*this, index}; }
    template <typename T, detail::enable_if_t<detail::is_pyobject<T>::value, int> = 0>
    detail::item_accessor operator[](T &&o) const {
        return object::operator[](std::forward<T>(o));
    }
    detail::sequence_iterator begin() const { return {*this, 0}; }
    detail::sequence_iterator end() const { return {*this, PySequence_Size(m_ptr)}; }
};

class list : public object {
public:
    PYBIND11_OBJECT_CVT(list, object, PyList_Check, PySequence_List)
    template <typename SzType = ssize_t,
              detail::enable_if_t<std::is_integral<SzType>::value, int> = 0>
    // Some compilers generate link errors when using `const SzType &` here:
    explicit list(SzType size = 0) : object(PyList_New(ssize_t_cast(size)), stolen_t{}) {
        if (!m_ptr) {
            pybind11_fail("Could not allocate list object!");
        }
    }
    size_t size() const { return (size_t) PyList_Size(m_ptr); }
    bool empty() const { return size() == 0; }
    detail::list_accessor operator[](size_t index) const { return {*this, index}; }
    template <typename T, detail::enable_if_t<detail::is_pyobject<T>::value, int> = 0>
    detail::item_accessor operator[](T &&o) const {
        return object::operator[](std::forward<T>(o));
    }
    detail::list_iterator begin() const { return {*this, 0}; }
    detail::list_iterator end() const { return {*this, PyList_GET_SIZE(m_ptr)}; }
    template <typename T>
    void append(T &&val) /* py-non-const */ {
        if (PyList_Append(m_ptr, detail::object_or_cast(std::forward<T>(val)).ptr()) != 0) {
            throw error_already_set();
        }
    }
    template <typename IdxType,
              typename ValType,
              detail::enable_if_t<std::is_integral<IdxType>::value, int> = 0>
    void insert(const IdxType &index, ValType &&val) /* py-non-const */ {
        if (PyList_Insert(m_ptr,
                          ssize_t_cast(index),
                          detail::object_or_cast(std::forward<ValType>(val)).ptr())
            != 0) {
            throw error_already_set();
        }
    }
};

class args : public tuple {
    PYBIND11_OBJECT_DEFAULT(args, tuple, PyTuple_Check)
};
class kwargs : public dict {
    PYBIND11_OBJECT_DEFAULT(kwargs, dict, PyDict_Check)
};

class anyset : public object {
public:
    PYBIND11_OBJECT(anyset, object, PyAnySet_Check)
    size_t size() const { return static_cast<size_t>(PySet_Size(m_ptr)); }
    bool empty() const { return size() == 0; }
    template <typename T>
    bool contains(T &&val) const {
        auto result = PySet_Contains(m_ptr, detail::object_or_cast(std::forward<T>(val)).ptr());
        if (result == -1) {
            throw error_already_set();
        }
        return result == 1;
    }
};

class set : public anyset {
public:
    PYBIND11_OBJECT_CVT(set, anyset, PySet_Check, PySet_New)
    set() : anyset(PySet_New(nullptr), stolen_t{}) {
        if (!m_ptr) {
            pybind11_fail("Could not allocate set object!");
        }
    }
    template <typename T>
    bool add(T &&val) /* py-non-const */ {
        return PySet_Add(m_ptr, detail::object_or_cast(std::forward<T>(val)).ptr()) == 0;
    }
    void clear() /* py-non-const */ { PySet_Clear(m_ptr); }
};

class frozenset : public anyset {
public:
    PYBIND11_OBJECT_CVT(frozenset, anyset, PyFrozenSet_Check, PyFrozenSet_New)
};

class function : public object {
public:
    PYBIND11_OBJECT_DEFAULT(function, object, PyCallable_Check)
    handle cpp_function() const {
        handle fun = detail::get_function(m_ptr);
        if (fun && PyCFunction_Check(fun.ptr())) {
            return fun;
        }
        return handle();
    }
    bool is_cpp_function() const { return (bool) cpp_function(); }
};

class staticmethod : public object {
public:
    PYBIND11_OBJECT_CVT(staticmethod, object, detail::PyStaticMethod_Check, PyStaticMethod_New)
};

class buffer : public object {
public:
    PYBIND11_OBJECT_DEFAULT(buffer, object, PyObject_CheckBuffer)

    buffer_info request(bool writable = false) const {
        int flags = PyBUF_STRIDES | PyBUF_FORMAT;
        if (writable) {
            flags |= PyBUF_WRITABLE;
        }
        auto *view = new Py_buffer();
        if (PyObject_GetBuffer(m_ptr, view, flags) != 0) {
            delete view;
            throw error_already_set();
        }
        return buffer_info(view);
    }
};

class memoryview : public object {
public:
    PYBIND11_OBJECT_CVT(memoryview, object, PyMemoryView_Check, PyMemoryView_FromObject)

    /** \rst
        Creates ``memoryview`` from ``buffer_info``.

        ``buffer_info`` must be created from ``buffer::request()``. Otherwise
        throws an exception.

        For creating a ``memoryview`` from objects that support buffer protocol,
        use ``memoryview(const object& obj)`` instead of this constructor.
     \endrst */
    explicit memoryview(const buffer_info &info) {
        if (!info.view()) {
            pybind11_fail("Prohibited to create memoryview without Py_buffer");
        }
        // Note: PyMemoryView_FromBuffer never increments obj reference.
        m_ptr = (info.view()->obj) ? PyMemoryView_FromObject(info.view()->obj)
                                   : PyMemoryView_FromBuffer(info.view());
        if (!m_ptr) {
            pybind11_fail("Unable to create memoryview from buffer descriptor");
        }
    }

    /** \rst
        Creates ``memoryview`` from static buffer.

        This method is meant for providing a ``memoryview`` for C/C++ buffer not
        managed by Python. The caller is responsible for managing the lifetime
        of ``ptr`` and ``format``, which MUST outlive the memoryview constructed
        here.

        See also: Python C API documentation for `PyMemoryView_FromBuffer`_.

        .. _PyMemoryView_FromBuffer:
           https://docs.python.org/c-api/memoryview.html#c.PyMemoryView_FromBuffer

        :param ptr: Pointer to the buffer.
        :param itemsize: Byte size of an element.
        :param format: Pointer to the null-terminated format string. For
            homogeneous Buffers, this should be set to
            ``format_descriptor<T>::value``.
        :param shape: Shape of the tensor (1 entry per dimension).
        :param strides: Number of bytes between adjacent entries (for each
            per dimension).
        :param readonly: Flag to indicate if the underlying storage may be
            written to.
     \endrst */
    static memoryview from_buffer(void *ptr,
                                  ssize_t itemsize,
                                  const char *format,
                                  detail::any_container<ssize_t> shape,
                                  detail::any_container<ssize_t> strides,
                                  bool readonly = false);

    static memoryview from_buffer(const void *ptr,
                                  ssize_t itemsize,
                                  const char *format,
                                  detail::any_container<ssize_t> shape,
                                  detail::any_container<ssize_t> strides) {
        return memoryview::from_buffer(
            const_cast<void *>(ptr), itemsize, format, std::move(shape), std::move(strides), true);
    }

    template <typename T>
    static memoryview from_buffer(T *ptr,
                                  detail::any_container<ssize_t> shape,
                                  detail::any_container<ssize_t> strides,
                                  bool readonly = false) {
        return memoryview::from_buffer(reinterpret_cast<void *>(ptr),
                                       sizeof(T),
                                       format_descriptor<T>::value,
                                       std::move(shape),
                                       std::move(strides),
                                       readonly);
    }

    template <typename T>
    static memoryview from_buffer(const T *ptr,
                                  detail::any_container<ssize_t> shape,
                                  detail::any_container<ssize_t> strides) {
        return memoryview::from_buffer(
            const_cast<T *>(ptr), std::move(shape), std::move(strides), true);
    }

    /** \rst
        Creates ``memoryview`` from static memory.

        This method is meant for providing a ``memoryview`` for C/C++ buffer not
        managed by Python. The caller is responsible for managing the lifetime
        of ``mem``, which MUST outlive the memoryview constructed here.

        See also: Python C API documentation for `PyMemoryView_FromBuffer`_.

        .. _PyMemoryView_FromMemory:
           https://docs.python.org/c-api/memoryview.html#c.PyMemoryView_FromMemory
     \endrst */
    static memoryview from_memory(void *mem, ssize_t size, bool readonly = false) {
        PyObject *ptr = PyMemoryView_FromMemory(
            reinterpret_cast<char *>(mem), size, (readonly) ? PyBUF_READ : PyBUF_WRITE);
        if (!ptr) {
            pybind11_fail("Could not allocate memoryview object!");
        }
        return memoryview(object(ptr, stolen_t{}));
    }

    static memoryview from_memory(const void *mem, ssize_t size) {
        return memoryview::from_memory(const_cast<void *>(mem), size, true);
    }

#ifdef PYBIND11_HAS_STRING_VIEW
    static memoryview from_memory(std::string_view mem) {
        return from_memory(const_cast<char *>(mem.data()), static_cast<ssize_t>(mem.size()), true);
    }
#endif
};

/// @cond DUPLICATE
inline memoryview memoryview::from_buffer(void *ptr,
                                          ssize_t itemsize,
                                          const char *format,
                                          detail::any_container<ssize_t> shape,
                                          detail::any_container<ssize_t> strides,
                                          bool readonly) {
    size_t ndim = shape->size();
    if (ndim != strides->size()) {
        pybind11_fail("memoryview: shape length doesn't match strides length");
    }
    ssize_t size = ndim != 0u ? 1 : 0;
    for (size_t i = 0; i < ndim; ++i) {
        size *= (*shape)[i];
    }
    Py_buffer view;
    view.buf = ptr;
    view.obj = nullptr;
    view.len = size * itemsize;
    view.readonly = static_cast<int>(readonly);
    view.itemsize = itemsize;
    view.format = const_cast<char *>(format);
    view.ndim = static_cast<int>(ndim);
    view.shape = shape->data();
    view.strides = strides->data();
    view.suboffsets = nullptr;
    view.internal = nullptr;
    PyObject *obj = PyMemoryView_FromBuffer(&view);
    if (!obj) {
        throw error_already_set();
    }
    return memoryview(object(obj, stolen_t{}));
}
/// @endcond
/// @} pytypes

/// \addtogroup python_builtins
/// @{

/// Get the length of a Python object.
inline size_t len(handle h) {
    ssize_t result = PyObject_Length(h.ptr());
    if (result < 0) {
        throw error_already_set();
    }
    return (size_t) result;
}

/// Get the length hint of a Python object.
/// Returns 0 when this cannot be determined.
inline size_t len_hint(handle h) {
    ssize_t result = PyObject_LengthHint(h.ptr(), 0);
    if (result < 0) {
        // Sometimes a length can't be determined at all (eg generators)
        // In which case simply return 0
        PyErr_Clear();
        return 0;
    }
    return (size_t) result;
}

inline str repr(handle h) {
    PyObject *str_value = PyObject_Repr(h.ptr());
    if (!str_value) {
        throw error_already_set();
    }
    return reinterpret_steal<str>(str_value);
}

inline iterator iter(handle obj) {
    PyObject *result = PyObject_GetIter(obj.ptr());
    if (!result) {
        throw error_already_set();
    }
    return reinterpret_steal<iterator>(result);
}
/// @} python_builtins

PYBIND11_NAMESPACE_BEGIN(detail)
template <typename D>
iterator object_api<D>::begin() const {
    return iter(derived());
}
template <typename D>
iterator object_api<D>::end() const {
    return iterator::sentinel();
}
template <typename D>
item_accessor object_api<D>::operator[](handle key) const {
    return {derived(), reinterpret_borrow<object>(key)};
}
template <typename D>
item_accessor object_api<D>::operator[](object &&key) const {
    return {derived(), std::move(key)};
}
template <typename D>
item_accessor object_api<D>::operator[](const char *key) const {
    return {derived(), pybind11::str(key)};
}
template <typename D>
obj_attr_accessor object_api<D>::attr(handle key) const {
    return {derived(), reinterpret_borrow<object>(key)};
}
template <typename D>
obj_attr_accessor object_api<D>::attr(object &&key) const {
    return {derived(), std::move(key)};
}
template <typename D>
str_attr_accessor object_api<D>::attr(const char *key) const {
    return {derived(), key};
}
template <typename D>
args_proxy object_api<D>::operator*() const {
    return args_proxy(derived().ptr());
}
template <typename D>
template <typename T>
bool object_api<D>::contains(T &&item) const {
    return attr("__contains__")(std::forward<T>(item)).template cast<bool>();
}

template <typename D>
pybind11::str object_api<D>::str() const {
    return pybind11::str(derived());
}

template <typename D>
str_attr_accessor object_api<D>::doc() const {
    return attr("__doc__");
}

template <typename D>
handle object_api<D>::get_type() const {
    return type::handle_of(derived());
}

template <typename D>
bool object_api<D>::rich_compare(object_api const &other, int value) const {
    int rv = PyObject_RichCompareBool(derived().ptr(), other.derived().ptr(), value);
    if (rv == -1) {
        throw error_already_set();
    }
    return rv == 1;
}

#define PYBIND11_MATH_OPERATOR_UNARY(op, fn)                                                      \
    template <typename D>                                                                         \
    object object_api<D>::op() const {                                                            \
        object result = reinterpret_steal<object>(fn(derived().ptr()));                           \
        if (!result.ptr())                                                                        \
            throw error_already_set();                                                            \
        return result;                                                                            \
    }

#define PYBIND11_MATH_OPERATOR_BINARY(op, fn)                                                     \
    template <typename D>                                                                         \
    object object_api<D>::op(object_api const &other) const {                                     \
        object result = reinterpret_steal<object>(fn(derived().ptr(), other.derived().ptr()));    \
        if (!result.ptr())                                                                        \
            throw error_already_set();                                                            \
        return result;                                                                            \
    }

#define PYBIND11_MATH_OPERATOR_BINARY_INPLACE(iop, fn)                                            \
    template <typename D>                                                                         \
    object object_api<D>::iop(object_api const &other) {                                          \
        object result = reinterpret_steal<object>(fn(derived().ptr(), other.derived().ptr()));    \
        if (!result.ptr())                                                                        \
            throw error_already_set();                                                            \
        return result;                                                                            \
    }

PYBIND11_MATH_OPERATOR_UNARY(operator~, PyNumber_Invert)
PYBIND11_MATH_OPERATOR_UNARY(operator-, PyNumber_Negative)
PYBIND11_MATH_OPERATOR_BINARY(operator+, PyNumber_Add)
PYBIND11_MATH_OPERATOR_BINARY_INPLACE(operator+=, PyNumber_InPlaceAdd)
PYBIND11_MATH_OPERATOR_BINARY(operator-, PyNumber_Subtract)
PYBIND11_MATH_OPERATOR_BINARY_INPLACE(operator-=, PyNumber_InPlaceSubtract)
PYBIND11_MATH_OPERATOR_BINARY(operator*, PyNumber_Multiply)
PYBIND11_MATH_OPERATOR_BINARY_INPLACE(operator*=, PyNumber_InPlaceMultiply)
PYBIND11_MATH_OPERATOR_BINARY(operator/, PyNumber_TrueDivide)
PYBIND11_MATH_OPERATOR_BINARY_INPLACE(operator/=, PyNumber_InPlaceTrueDivide)
PYBIND11_MATH_OPERATOR_BINARY(operator|, PyNumber_Or)
PYBIND11_MATH_OPERATOR_BINARY_INPLACE(operator|=, PyNumber_InPlaceOr)
PYBIND11_MATH_OPERATOR_BINARY(operator&, PyNumber_And)
PYBIND11_MATH_OPERATOR_BINARY_INPLACE(operator&=, PyNumber_InPlaceAnd)
PYBIND11_MATH_OPERATOR_BINARY(operator^, PyNumber_Xor)
PYBIND11_MATH_OPERATOR_BINARY_INPLACE(operator^=, PyNumber_InPlaceXor)
PYBIND11_MATH_OPERATOR_BINARY(operator<<, PyNumber_Lshift)
PYBIND11_MATH_OPERATOR_BINARY_INPLACE(operator<<=, PyNumber_InPlaceLshift)
PYBIND11_MATH_OPERATOR_BINARY(operator>>, PyNumber_Rshift)
PYBIND11_MATH_OPERATOR_BINARY_INPLACE(operator>>=, PyNumber_InPlaceRshift)

#undef PYBIND11_MATH_OPERATOR_UNARY
#undef PYBIND11_MATH_OPERATOR_BINARY
#undef PYBIND11_MATH_OPERATOR_BINARY_INPLACE

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
