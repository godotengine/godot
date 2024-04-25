/*
    pybind11/iostream.h -- Tools to assist with redirecting cout and cerr to Python

    Copyright (c) 2017 Henry F. Schreiner

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.

    WARNING: The implementation in this file is NOT thread safe. Multiple
    threads writing to a redirected ostream concurrently cause data races
    and potentially buffer overflows. Therefore it is currently a requirement
    that all (possibly) concurrent redirected ostream writes are protected by
    a mutex.
    #HelpAppreciated: Work on iostream.h thread safety.
    For more background see the discussions under
    https://github.com/pybind/pybind11/pull/2982 and
    https://github.com/pybind/pybind11/pull/2995.
*/

#pragma once

#include "pybind11.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <streambuf>
#include <string>
#include <utility>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

// Buffer that writes to Python instead of C++
class pythonbuf : public std::streambuf {
private:
    using traits_type = std::streambuf::traits_type;

    const size_t buf_size;
    std::unique_ptr<char[]> d_buffer;
    object pywrite;
    object pyflush;

    int overflow(int c) override {
        if (!traits_type::eq_int_type(c, traits_type::eof())) {
            *pptr() = traits_type::to_char_type(c);
            pbump(1);
        }
        return sync() == 0 ? traits_type::not_eof(c) : traits_type::eof();
    }

    // Computes how many bytes at the end of the buffer are part of an
    // incomplete sequence of UTF-8 bytes.
    // Precondition: pbase() < pptr()
    size_t utf8_remainder() const {
        const auto rbase = std::reverse_iterator<char *>(pbase());
        const auto rpptr = std::reverse_iterator<char *>(pptr());
        auto is_ascii = [](char c) { return (static_cast<unsigned char>(c) & 0x80) == 0x00; };
        auto is_leading = [](char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0xC0; };
        auto is_leading_2b = [](char c) { return static_cast<unsigned char>(c) <= 0xDF; };
        auto is_leading_3b = [](char c) { return static_cast<unsigned char>(c) <= 0xEF; };
        // If the last character is ASCII, there are no incomplete code points
        if (is_ascii(*rpptr)) {
            return 0;
        }
        // Otherwise, work back from the end of the buffer and find the first
        // UTF-8 leading byte
        const auto rpend = rbase - rpptr >= 3 ? rpptr + 3 : rbase;
        const auto leading = std::find_if(rpptr, rpend, is_leading);
        if (leading == rbase) {
            return 0;
        }
        const auto dist = static_cast<size_t>(leading - rpptr);
        size_t remainder = 0;

        if (dist == 0) {
            remainder = 1; // 1-byte code point is impossible
        } else if (dist == 1) {
            remainder = is_leading_2b(*leading) ? 0 : dist + 1;
        } else if (dist == 2) {
            remainder = is_leading_3b(*leading) ? 0 : dist + 1;
        }
        // else if (dist >= 3), at least 4 bytes before encountering an UTF-8
        // leading byte, either no remainder or invalid UTF-8.
        // Invalid UTF-8 will cause an exception later when converting
        // to a Python string, so that's not handled here.
        return remainder;
    }

    // This function must be non-virtual to be called in a destructor.
    int _sync() {
        if (pbase() != pptr()) { // If buffer is not empty
            gil_scoped_acquire tmp;
            // This subtraction cannot be negative, so dropping the sign.
            auto size = static_cast<size_t>(pptr() - pbase());
            size_t remainder = utf8_remainder();

            if (size > remainder) {
                str line(pbase(), size - remainder);
                pywrite(std::move(line));
                pyflush();
            }

            // Copy the remainder at the end of the buffer to the beginning:
            if (remainder > 0) {
                std::memmove(pbase(), pptr() - remainder, remainder);
            }
            setp(pbase(), epptr());
            pbump(static_cast<int>(remainder));
        }
        return 0;
    }

    int sync() override { return _sync(); }

public:
    explicit pythonbuf(const object &pyostream, size_t buffer_size = 1024)
        : buf_size(buffer_size), d_buffer(new char[buf_size]), pywrite(pyostream.attr("write")),
          pyflush(pyostream.attr("flush")) {
        setp(d_buffer.get(), d_buffer.get() + buf_size - 1);
    }

    pythonbuf(pythonbuf &&) = default;

    /// Sync before destroy
    ~pythonbuf() override { _sync(); }
};

PYBIND11_NAMESPACE_END(detail)

/** \rst
    This a move-only guard that redirects output.

    .. code-block:: cpp

        #include <pybind11/iostream.h>

        ...

        {
            py::scoped_ostream_redirect output;
            std::cout << "Hello, World!"; // Python stdout
        } // <-- return std::cout to normal

    You can explicitly pass the c++ stream and the python object,
    for example to guard stderr instead.

    .. code-block:: cpp

        {
            py::scoped_ostream_redirect output{
                std::cerr, py::module::import("sys").attr("stderr")};
            std::cout << "Hello, World!";
        }
 \endrst */
class scoped_ostream_redirect {
protected:
    std::streambuf *old;
    std::ostream &costream;
    detail::pythonbuf buffer;

public:
    explicit scoped_ostream_redirect(std::ostream &costream = std::cout,
                                     const object &pyostream
                                     = module_::import("sys").attr("stdout"))
        : costream(costream), buffer(pyostream) {
        old = costream.rdbuf(&buffer);
    }

    ~scoped_ostream_redirect() { costream.rdbuf(old); }

    scoped_ostream_redirect(const scoped_ostream_redirect &) = delete;
    scoped_ostream_redirect(scoped_ostream_redirect &&other) = default;
    scoped_ostream_redirect &operator=(const scoped_ostream_redirect &) = delete;
    scoped_ostream_redirect &operator=(scoped_ostream_redirect &&) = delete;
};

/** \rst
    Like `scoped_ostream_redirect`, but redirects cerr by default. This class
    is provided primary to make ``py::call_guard`` easier to make.

    .. code-block:: cpp

     m.def("noisy_func", &noisy_func,
           py::call_guard<scoped_ostream_redirect,
                          scoped_estream_redirect>());

\endrst */
class scoped_estream_redirect : public scoped_ostream_redirect {
public:
    explicit scoped_estream_redirect(std::ostream &costream = std::cerr,
                                     const object &pyostream
                                     = module_::import("sys").attr("stderr"))
        : scoped_ostream_redirect(costream, pyostream) {}
};

PYBIND11_NAMESPACE_BEGIN(detail)

// Class to redirect output as a context manager. C++ backend.
class OstreamRedirect {
    bool do_stdout_;
    bool do_stderr_;
    std::unique_ptr<scoped_ostream_redirect> redirect_stdout;
    std::unique_ptr<scoped_estream_redirect> redirect_stderr;

public:
    explicit OstreamRedirect(bool do_stdout = true, bool do_stderr = true)
        : do_stdout_(do_stdout), do_stderr_(do_stderr) {}

    void enter() {
        if (do_stdout_) {
            redirect_stdout.reset(new scoped_ostream_redirect());
        }
        if (do_stderr_) {
            redirect_stderr.reset(new scoped_estream_redirect());
        }
    }

    void exit() {
        redirect_stdout.reset();
        redirect_stderr.reset();
    }
};

PYBIND11_NAMESPACE_END(detail)

/** \rst
    This is a helper function to add a C++ redirect context manager to Python
    instead of using a C++ guard. To use it, add the following to your binding code:

    .. code-block:: cpp

        #include <pybind11/iostream.h>

        ...

        py::add_ostream_redirect(m, "ostream_redirect");

    You now have a Python context manager that redirects your output:

    .. code-block:: python

        with m.ostream_redirect():
            m.print_to_cout_function()

    This manager can optionally be told which streams to operate on:

    .. code-block:: python

        with m.ostream_redirect(stdout=true, stderr=true):
            m.noisy_function_with_error_printing()

 \endrst */
inline class_<detail::OstreamRedirect>
add_ostream_redirect(module_ m, const std::string &name = "ostream_redirect") {
    return class_<detail::OstreamRedirect>(std::move(m), name.c_str(), module_local())
        .def(init<bool, bool>(), arg("stdout") = true, arg("stderr") = true)
        .def("__enter__", &detail::OstreamRedirect::enter)
        .def("__exit__", [](detail::OstreamRedirect &self_, const args &) { self_.exit(); });
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
