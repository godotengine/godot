/*
    pybind11/embed.h: Support for embedding the interpreter

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "eval.h"

#include <memory>
#include <vector>

#if defined(PYPY_VERSION)
#    error Embedding the interpreter is not supported with PyPy
#endif

#define PYBIND11_EMBEDDED_MODULE_IMPL(name)                                                       \
    extern "C" PyObject *pybind11_init_impl_##name();                                             \
    extern "C" PyObject *pybind11_init_impl_##name() { return pybind11_init_wrapper_##name(); }

/** \rst
    Add a new module to the table of builtins for the interpreter. Must be
    defined in global scope. The first macro parameter is the name of the
    module (without quotes). The second parameter is the variable which will
    be used as the interface to add functions and classes to the module.

    .. code-block:: cpp

        PYBIND11_EMBEDDED_MODULE(example, m) {
            // ... initialize functions and classes here
            m.def("foo", []() {
                return "Hello, World!";
            });
        }
 \endrst */
#define PYBIND11_EMBEDDED_MODULE(name, variable)                                                  \
    static ::pybind11::module_::module_def PYBIND11_CONCAT(pybind11_module_def_, name);           \
    static void PYBIND11_CONCAT(pybind11_init_, name)(::pybind11::module_ &);                     \
    static PyObject PYBIND11_CONCAT(*pybind11_init_wrapper_, name)() {                            \
        auto m = ::pybind11::module_::create_extension_module(                                    \
            PYBIND11_TOSTRING(name), nullptr, &PYBIND11_CONCAT(pybind11_module_def_, name));      \
        try {                                                                                     \
            PYBIND11_CONCAT(pybind11_init_, name)(m);                                             \
            return m.ptr();                                                                       \
        }                                                                                         \
        PYBIND11_CATCH_INIT_EXCEPTIONS                                                            \
    }                                                                                             \
    PYBIND11_EMBEDDED_MODULE_IMPL(name)                                                           \
    ::pybind11::detail::embedded_module PYBIND11_CONCAT(pybind11_module_, name)(                  \
        PYBIND11_TOSTRING(name), PYBIND11_CONCAT(pybind11_init_impl_, name));                     \
    void PYBIND11_CONCAT(pybind11_init_, name)(::pybind11::module_                                \
                                               & variable) // NOLINT(bugprone-macro-parentheses)

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/// Python 2.7/3.x compatible version of `PyImport_AppendInittab` and error checks.
struct embedded_module {
    using init_t = PyObject *(*) ();
    embedded_module(const char *name, init_t init) {
        if (Py_IsInitialized() != 0) {
            pybind11_fail("Can't add new modules after the interpreter has been initialized");
        }

        auto result = PyImport_AppendInittab(name, init);
        if (result == -1) {
            pybind11_fail("Insufficient memory to add a new module");
        }
    }
};

struct wide_char_arg_deleter {
    void operator()(wchar_t *ptr) const {
        // API docs: https://docs.python.org/3/c-api/sys.html#c.Py_DecodeLocale
        PyMem_RawFree(ptr);
    }
};

inline wchar_t *widen_chars(const char *safe_arg) {
    wchar_t *widened_arg = Py_DecodeLocale(safe_arg, nullptr);
    return widened_arg;
}

inline void precheck_interpreter() {
    if (Py_IsInitialized() != 0) {
        pybind11_fail("The interpreter is already running");
    }
}

#if !defined(PYBIND11_PYCONFIG_SUPPORT_PY_VERSION_HEX)
#    define PYBIND11_PYCONFIG_SUPPORT_PY_VERSION_HEX (0x03080000)
#endif

#if PY_VERSION_HEX < PYBIND11_PYCONFIG_SUPPORT_PY_VERSION_HEX
inline void initialize_interpreter_pre_pyconfig(bool init_signal_handlers,
                                                int argc,
                                                const char *const *argv,
                                                bool add_program_dir_to_path) {
    detail::precheck_interpreter();
    Py_InitializeEx(init_signal_handlers ? 1 : 0);
#    if defined(WITH_THREAD) && PY_VERSION_HEX < 0x03070000
    PyEval_InitThreads();
#    endif

    // Before it was special-cased in python 3.8, passing an empty or null argv
    // caused a segfault, so we have to reimplement the special case ourselves.
    bool special_case = (argv == nullptr || argc <= 0);

    const char *const empty_argv[]{"\0"};
    const char *const *safe_argv = special_case ? empty_argv : argv;
    if (special_case) {
        argc = 1;
    }

    auto argv_size = static_cast<size_t>(argc);
    // SetArgv* on python 3 takes wchar_t, so we have to convert.
    std::unique_ptr<wchar_t *[]> widened_argv(new wchar_t *[argv_size]);
    std::vector<std::unique_ptr<wchar_t[], detail::wide_char_arg_deleter>> widened_argv_entries;
    widened_argv_entries.reserve(argv_size);
    for (size_t ii = 0; ii < argv_size; ++ii) {
        widened_argv_entries.emplace_back(detail::widen_chars(safe_argv[ii]));
        if (!widened_argv_entries.back()) {
            // A null here indicates a character-encoding failure or the python
            // interpreter out of memory. Give up.
            return;
        }
        widened_argv[ii] = widened_argv_entries.back().get();
    }

    auto *pysys_argv = widened_argv.get();

    PySys_SetArgvEx(argc, pysys_argv, static_cast<int>(add_program_dir_to_path));
}
#endif

PYBIND11_NAMESPACE_END(detail)

#if PY_VERSION_HEX >= PYBIND11_PYCONFIG_SUPPORT_PY_VERSION_HEX
inline void initialize_interpreter(PyConfig *config,
                                   int argc = 0,
                                   const char *const *argv = nullptr,
                                   bool add_program_dir_to_path = true) {
    detail::precheck_interpreter();
    PyStatus status = PyConfig_SetBytesArgv(config, argc, const_cast<char *const *>(argv));
    if (PyStatus_Exception(status) != 0) {
        // A failure here indicates a character-encoding failure or the python
        // interpreter out of memory. Give up.
        PyConfig_Clear(config);
        throw std::runtime_error(PyStatus_IsError(status) != 0 ? status.err_msg
                                                               : "Failed to prepare CPython");
    }
    status = Py_InitializeFromConfig(config);
    if (PyStatus_Exception(status) != 0) {
        PyConfig_Clear(config);
        throw std::runtime_error(PyStatus_IsError(status) != 0 ? status.err_msg
                                                               : "Failed to init CPython");
    }
    if (add_program_dir_to_path) {
        PyRun_SimpleString("import sys, os.path; "
                           "sys.path.insert(0, "
                           "os.path.abspath(os.path.dirname(sys.argv[0])) "
                           "if sys.argv and os.path.exists(sys.argv[0]) else '')");
    }
    PyConfig_Clear(config);
}
#endif

/** \rst
    Initialize the Python interpreter. No other pybind11 or CPython API functions can be
    called before this is done; with the exception of `PYBIND11_EMBEDDED_MODULE`. The
    optional `init_signal_handlers` parameter can be used to skip the registration of
    signal handlers (see the `Python documentation`_ for details). Calling this function
    again after the interpreter has already been initialized is a fatal error.

    If initializing the Python interpreter fails, then the program is terminated.  (This
    is controlled by the CPython runtime and is an exception to pybind11's normal behavior
    of throwing exceptions on errors.)

    The remaining optional parameters, `argc`, `argv`, and `add_program_dir_to_path` are
    used to populate ``sys.argv`` and ``sys.path``.
    See the |PySys_SetArgvEx documentation|_ for details.

    .. _Python documentation: https://docs.python.org/3/c-api/init.html#c.Py_InitializeEx
    .. |PySys_SetArgvEx documentation| replace:: ``PySys_SetArgvEx`` documentation
    .. _PySys_SetArgvEx documentation: https://docs.python.org/3/c-api/init.html#c.PySys_SetArgvEx
 \endrst */
inline void initialize_interpreter(bool init_signal_handlers = true,
                                   int argc = 0,
                                   const char *const *argv = nullptr,
                                   bool add_program_dir_to_path = true) {
#if PY_VERSION_HEX < PYBIND11_PYCONFIG_SUPPORT_PY_VERSION_HEX
    detail::initialize_interpreter_pre_pyconfig(
        init_signal_handlers, argc, argv, add_program_dir_to_path);
#else
    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);
    config.isolated = 0;
    config.use_environment = 1;
    config.install_signal_handlers = init_signal_handlers ? 1 : 0;
    initialize_interpreter(&config, argc, argv, add_program_dir_to_path);
#endif
}

/** \rst
    Shut down the Python interpreter. No pybind11 or CPython API functions can be called
    after this. In addition, pybind11 objects must not outlive the interpreter:

    .. code-block:: cpp

        { // BAD
            py::initialize_interpreter();
            auto hello = py::str("Hello, World!");
            py::finalize_interpreter();
        } // <-- BOOM, hello's destructor is called after interpreter shutdown

        { // GOOD
            py::initialize_interpreter();
            { // scoped
                auto hello = py::str("Hello, World!");
            } // <-- OK, hello is cleaned up properly
            py::finalize_interpreter();
        }

        { // BETTER
            py::scoped_interpreter guard{};
            auto hello = py::str("Hello, World!");
        }

    .. warning::

        The interpreter can be restarted by calling `initialize_interpreter` again.
        Modules created using pybind11 can be safely re-initialized. However, Python
        itself cannot completely unload binary extension modules and there are several
        caveats with regard to interpreter restarting. All the details can be found
        in the CPython documentation. In short, not all interpreter memory may be
        freed, either due to reference cycles or user-created global data.

 \endrst */
inline void finalize_interpreter() {
    handle builtins(PyEval_GetBuiltins());
    const char *id = PYBIND11_INTERNALS_ID;

    // Get the internals pointer (without creating it if it doesn't exist).  It's possible for the
    // internals to be created during Py_Finalize() (e.g. if a py::capsule calls `get_internals()`
    // during destruction), so we get the pointer-pointer here and check it after Py_Finalize().
    detail::internals **internals_ptr_ptr = detail::get_internals_pp();
    // It could also be stashed in builtins, so look there too:
    if (builtins.contains(id) && isinstance<capsule>(builtins[id])) {
        internals_ptr_ptr = capsule(builtins[id]);
    }
    // Local internals contains data managed by the current interpreter, so we must clear them to
    // avoid undefined behaviors when initializing another interpreter
    detail::get_local_internals().registered_types_cpp.clear();
    detail::get_local_internals().registered_exception_translators.clear();

    Py_Finalize();

    if (internals_ptr_ptr) {
        delete *internals_ptr_ptr;
        *internals_ptr_ptr = nullptr;
    }
}

/** \rst
    Scope guard version of `initialize_interpreter` and `finalize_interpreter`.
    This a move-only guard and only a single instance can exist.

    See `initialize_interpreter` for a discussion of its constructor arguments.

    .. code-block:: cpp

        #include <pybind11/embed.h>

        int main() {
            py::scoped_interpreter guard{};
            py::print(Hello, World!);
        } // <-- interpreter shutdown
 \endrst */
class scoped_interpreter {
public:
    explicit scoped_interpreter(bool init_signal_handlers = true,
                                int argc = 0,
                                const char *const *argv = nullptr,
                                bool add_program_dir_to_path = true) {
        initialize_interpreter(init_signal_handlers, argc, argv, add_program_dir_to_path);
    }

#if PY_VERSION_HEX >= PYBIND11_PYCONFIG_SUPPORT_PY_VERSION_HEX
    explicit scoped_interpreter(PyConfig *config,
                                int argc = 0,
                                const char *const *argv = nullptr,
                                bool add_program_dir_to_path = true) {
        initialize_interpreter(config, argc, argv, add_program_dir_to_path);
    }
#endif

    scoped_interpreter(const scoped_interpreter &) = delete;
    scoped_interpreter(scoped_interpreter &&other) noexcept { other.is_valid = false; }
    scoped_interpreter &operator=(const scoped_interpreter &) = delete;
    scoped_interpreter &operator=(scoped_interpreter &&) = delete;

    ~scoped_interpreter() {
        if (is_valid) {
            finalize_interpreter();
        }
    }

private:
    bool is_valid = true;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
