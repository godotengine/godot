/*
    pybind11/gil.h: RAII helpers for managing the GIL

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/common.h"

#if defined(WITH_THREAD) && !defined(PYBIND11_SIMPLE_GIL_MANAGEMENT)
#    include "detail/internals.h"
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)

// forward declarations
PyThreadState *get_thread_state_unchecked();

PYBIND11_NAMESPACE_END(detail)

#if defined(WITH_THREAD)

#    if !defined(PYBIND11_SIMPLE_GIL_MANAGEMENT)

/* The functions below essentially reproduce the PyGILState_* API using a RAII
 * pattern, but there are a few important differences:
 *
 * 1. When acquiring the GIL from an non-main thread during the finalization
 *    phase, the GILState API blindly terminates the calling thread, which
 *    is often not what is wanted. This API does not do this.
 *
 * 2. The gil_scoped_release function can optionally cut the relationship
 *    of a PyThreadState and its associated thread, which allows moving it to
 *    another thread (this is a fairly rare/advanced use case).
 *
 * 3. The reference count of an acquired thread state can be controlled. This
 *    can be handy to prevent cases where callbacks issued from an external
 *    thread would otherwise constantly construct and destroy thread state data
 *    structures.
 *
 * See the Python bindings of NanoGUI (http://github.com/wjakob/nanogui) for an
 * example which uses features 2 and 3 to migrate the Python thread of
 * execution to another thread (to run the event loop on the original thread,
 * in this case).
 */

class gil_scoped_acquire {
public:
    PYBIND11_NOINLINE gil_scoped_acquire() {
        auto &internals = detail::get_internals();
        tstate = (PyThreadState *) PYBIND11_TLS_GET_VALUE(internals.tstate);

        if (!tstate) {
            /* Check if the GIL was acquired using the PyGILState_* API instead (e.g. if
               calling from a Python thread). Since we use a different key, this ensures
               we don't create a new thread state and deadlock in PyEval_AcquireThread
               below. Note we don't save this state with internals.tstate, since we don't
               create it we would fail to clear it (its reference count should be > 0). */
            tstate = PyGILState_GetThisThreadState();
        }

        if (!tstate) {
            tstate = PyThreadState_New(internals.istate);
#        if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            if (!tstate) {
                pybind11_fail("scoped_acquire: could not create thread state!");
            }
#        endif
            tstate->gilstate_counter = 0;
            PYBIND11_TLS_REPLACE_VALUE(internals.tstate, tstate);
        } else {
            release = detail::get_thread_state_unchecked() != tstate;
        }

        if (release) {
            PyEval_AcquireThread(tstate);
        }

        inc_ref();
    }

    gil_scoped_acquire(const gil_scoped_acquire &) = delete;
    gil_scoped_acquire &operator=(const gil_scoped_acquire &) = delete;

    void inc_ref() { ++tstate->gilstate_counter; }

    PYBIND11_NOINLINE void dec_ref() {
        --tstate->gilstate_counter;
#        if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
        if (detail::get_thread_state_unchecked() != tstate) {
            pybind11_fail("scoped_acquire::dec_ref(): thread state must be current!");
        }
        if (tstate->gilstate_counter < 0) {
            pybind11_fail("scoped_acquire::dec_ref(): reference count underflow!");
        }
#        endif
        if (tstate->gilstate_counter == 0) {
#        if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            if (!release) {
                pybind11_fail("scoped_acquire::dec_ref(): internal error!");
            }
#        endif
            PyThreadState_Clear(tstate);
            if (active) {
                PyThreadState_DeleteCurrent();
            }
            PYBIND11_TLS_DELETE_VALUE(detail::get_internals().tstate);
            release = false;
        }
    }

    /// This method will disable the PyThreadState_DeleteCurrent call and the
    /// GIL won't be acquired. This method should be used if the interpreter
    /// could be shutting down when this is called, as thread deletion is not
    /// allowed during shutdown. Check _Py_IsFinalizing() on Python 3.7+, and
    /// protect subsequent code.
    PYBIND11_NOINLINE void disarm() { active = false; }

    PYBIND11_NOINLINE ~gil_scoped_acquire() {
        dec_ref();
        if (release) {
            PyEval_SaveThread();
        }
    }

private:
    PyThreadState *tstate = nullptr;
    bool release = true;
    bool active = true;
};

class gil_scoped_release {
public:
    explicit gil_scoped_release(bool disassoc = false) : disassoc(disassoc) {
        // `get_internals()` must be called here unconditionally in order to initialize
        // `internals.tstate` for subsequent `gil_scoped_acquire` calls. Otherwise, an
        // initialization race could occur as multiple threads try `gil_scoped_acquire`.
        auto &internals = detail::get_internals();
        // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
        tstate = PyEval_SaveThread();
        if (disassoc) {
            // Python >= 3.7 can remove this, it's an int before 3.7
            // NOLINTNEXTLINE(readability-qualified-auto)
            auto key = internals.tstate;
            PYBIND11_TLS_DELETE_VALUE(key);
        }
    }

    gil_scoped_release(const gil_scoped_release &) = delete;
    gil_scoped_release &operator=(const gil_scoped_release &) = delete;

    /// This method will disable the PyThreadState_DeleteCurrent call and the
    /// GIL won't be acquired. This method should be used if the interpreter
    /// could be shutting down when this is called, as thread deletion is not
    /// allowed during shutdown. Check _Py_IsFinalizing() on Python 3.7+, and
    /// protect subsequent code.
    PYBIND11_NOINLINE void disarm() { active = false; }

    ~gil_scoped_release() {
        if (!tstate) {
            return;
        }
        // `PyEval_RestoreThread()` should not be called if runtime is finalizing
        if (active) {
            PyEval_RestoreThread(tstate);
        }
        if (disassoc) {
            // Python >= 3.7 can remove this, it's an int before 3.7
            // NOLINTNEXTLINE(readability-qualified-auto)
            auto key = detail::get_internals().tstate;
            PYBIND11_TLS_REPLACE_VALUE(key, tstate);
        }
    }

private:
    PyThreadState *tstate;
    bool disassoc;
    bool active = true;
};

#    else // PYBIND11_SIMPLE_GIL_MANAGEMENT

class gil_scoped_acquire {
    PyGILState_STATE state;

public:
    gil_scoped_acquire() : state{PyGILState_Ensure()} {}
    gil_scoped_acquire(const gil_scoped_acquire &) = delete;
    gil_scoped_acquire &operator=(const gil_scoped_acquire &) = delete;
    ~gil_scoped_acquire() { PyGILState_Release(state); }
    void disarm() {}
};

class gil_scoped_release {
    PyThreadState *state;

public:
    gil_scoped_release() : state{PyEval_SaveThread()} {}
    gil_scoped_release(const gil_scoped_release &) = delete;
    gil_scoped_release &operator=(const gil_scoped_release &) = delete;
    ~gil_scoped_release() { PyEval_RestoreThread(state); }
    void disarm() {}
};

#    endif // PYBIND11_SIMPLE_GIL_MANAGEMENT

#else // WITH_THREAD

class gil_scoped_acquire {
public:
    gil_scoped_acquire() {
        // Trick to suppress `unused variable` error messages (at call sites).
        (void) (this != (this + 1));
    }
    gil_scoped_acquire(const gil_scoped_acquire &) = delete;
    gil_scoped_acquire &operator=(const gil_scoped_acquire &) = delete;
    void disarm() {}
};

class gil_scoped_release {
public:
    gil_scoped_release() {
        // Trick to suppress `unused variable` error messages (at call sites).
        (void) (this != (this + 1));
    }
    gil_scoped_release(const gil_scoped_release &) = delete;
    gil_scoped_release &operator=(const gil_scoped_release &) = delete;
    void disarm() {}
};

#endif // WITH_THREAD

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
