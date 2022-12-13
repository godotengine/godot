// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2011, SIL International, All rights reserved.

// JSON pretty printer for graphite font debug output logging.
// Created on: 15 Dec 2011
//     Author: Tim Eves

#pragma once

#include "inc/Main.h"
#include <cassert>
#include <cstdio>
#include <cstdint>
#include "inc/List.h"

namespace graphite2 {

class json
{
    // Prevent copying
    json(const json &);
    json & operator = (const json &);

    typedef void (*_context_t)(json &);

    FILE * const    _stream;
    char            _contexts[128], // context stack
                  * _context,       // current context (top of stack)
                  * _flatten;       // if !0 points to context above which
                                    //  pretty printed output should occur.
    Vector<void *>  _env;

    void context(const char current) throw();
    void indent(const int d=0) throw();
    void push_context(const char, const char) throw();
    void pop_context() throw();

public:
    class closer;

    using string = const char *;
    using number = double;
    enum class integer : std::intmax_t {};
    enum class integer_u : std::uintmax_t {};
    using boolean = bool;
    static const std::nullptr_t  null;

    void setenv(unsigned int index, void *val) { _env.reserve(index + 1); if (index >= _env.size()) _env.insert(_env.end(), _env.size() - index + 1, 0); _env[index] = val; }
    void *getenv(unsigned int index) const { return _env[index]; }
    const Vector<void *> &getenvs() const { return _env; }

    static void flat(json &) throw();
    static void close(json &) throw();
    static void object(json &) throw();
    static void array(json &) throw();
    static void item(json &) throw();

    json(FILE * stream) throw();
    ~json() throw ();

    FILE * stream() const throw();

    json & operator << (string) throw();
    json & operator << (number) throw();
    json & operator << (integer) throw();
    json & operator << (integer_u) throw();
    json & operator << (boolean) throw();
    json & operator << (std::nullptr_t) throw();
    json & operator << (_context_t) throw();

    operator bool() const throw();
    bool good() const throw();
    bool eof() const throw();

    CLASS_NEW_DELETE;
};

class json::closer
{
    // Prevent copying.
    closer(const closer &);
    closer & operator = (const closer &);

    json * const    _j;
public:
    closer(json * const j) : _j(j) {}
    ~closer() throw() { if (_j)  *_j << close; }
};

inline
json::json(FILE * s) throw()
: _stream(s), _context(_contexts), _flatten(0)
{
    if (good())
        fflush(s);
}


inline
json::~json() throw ()
{
    while (_context > _contexts)    pop_context();
}

inline
FILE * json::stream() const throw()     { return _stream; }


inline
json & json::operator << (json::_context_t ctxt) throw()
{
    ctxt(*this);
    return *this;
}

inline
json & operator << (json & j, signed char d) throw()   { return j << json::integer(d); }

inline
json & operator << (json & j, unsigned char d) throw() { return j << json::integer_u(d); }

inline
json & operator << (json & j, short int d) throw()   { return j << json::integer(d); }

inline
json & operator << (json & j, unsigned short int d) throw() { return j << json::integer_u(d); }

inline
json & operator << (json & j, int d) throw()         { return j << json::integer(d); }

inline
json & operator << (json & j, unsigned int d) throw()       { return j << json::integer_u(d); }

inline
json & operator << (json & j, long int d) throw()         { return j << json::integer(d); }

inline
json & operator << (json & j, unsigned long int d) throw()       { return j << json::integer_u(d); }

inline
json & operator << (json & j, long long int d) throw()         { return j << json::integer(d); }

inline
json & operator << (json & j, unsigned long long int d) throw()       { return j << json::integer_u(d); }

inline
json::operator bool() const throw()     { return good(); }

inline
bool json::good() const throw()         { return _stream && ferror(_stream) == 0; }

inline
bool json::eof() const throw()          { return feof(_stream) != 0; }

} // namespace graphite2
