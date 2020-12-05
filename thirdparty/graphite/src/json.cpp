/*  GRAPHITE2 LICENSING

    Copyright 2011, SIL International
    All rights reserved.

    This library is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation; either version 2.1 of License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should also have received a copy of the GNU Lesser General Public
    License along with this library in the file named "LICENSE".
    If not, write to the Free Software Foundation, 51 Franklin Street,
    Suite 500, Boston, MA 02110-1335, USA or visit their web page on the
    internet at http://www.fsf.org/licenses/lgpl.html.

Alternatively, the contents of this file may be used under the terms of the
Mozilla Public License (http://mozilla.org/MPL) or the GNU General Public
License, as published by the Free Software Foundation, either version 2
of the License or (at your option) any later version.
*/
// JSON debug logging
// Author: Tim Eves

#if !defined GRAPHITE2_NTRACING

#include <cstdio>
#include <limits>
#include "inc/json.h"

#if defined(_MSC_VER)
#define FORMAT_INTMAX "%lli"
#define FORMAT_UINTMAX "%llu"
#else
#define FORMAT_INTMAX "%ji"
#define FORMAT_UINTMAX "%ju"
#endif

using namespace graphite2;

namespace
{
    enum
    {
        seq = ',',
        obj='}', member=':', empty_obj='{',
        arr=']', empty_arr='['
    };
}

const std::nullptr_t json::null = nullptr;

inline
void json::context(const char current) throw()
{
    fprintf(_stream, "%c", *_context);
    indent();
    *_context = current;
}


void json::indent(const int d) throw()
{
    if (*_context == member || (_flatten  && _flatten < _context))
        fputc(' ', _stream);
    else
        fprintf(_stream,  "\n%*s",  4*int(_context - _contexts + d), "");
}


inline
void json::push_context(const char prefix, const char suffix) throw()
{
    assert(_context - _contexts < ptrdiff_t(sizeof _contexts));

    if (_context == _contexts)
        *_context = suffix;
    else
        context(suffix);
    *++_context = prefix;
}


void json::pop_context() throw()
{
    assert(_context > _contexts);

    if (*_context == seq)   indent(-1);
    else                    fputc(*_context, _stream);

    fputc(*--_context, _stream);
    if (_context == _contexts)  fputc('\n', _stream);
    fflush(_stream);

    if (_flatten >= _context)   _flatten = 0;
    *_context = seq;
}


// These four functions cannot be inlined as pointers to these
// functions are needed for operator << (_context_t) to work.
void json::flat(json & j) throw()   { if (!j._flatten)  j._flatten = j._context; }
void json::close(json & j) throw()  { j.pop_context(); }
void json::object(json & j) throw() { j.push_context('{', '}'); }
void json::array(json & j) throw()  { j.push_context('[', ']'); }
void json::item(json & j) throw()
{
    while (j._context > j._contexts+1 && j._context[-1] != arr)
        j.pop_context();
}


json & json::operator << (json::string s) throw()
{
    const char ctxt = _context[-1] == obj ? *_context == member ? seq : member : seq;
    context(ctxt);
    fprintf(_stream, "\"%s\"", s);
    if (ctxt == member) fputc(' ', _stream);

    return *this;
}

json & json::operator << (json::number f) throw()
{
    context(seq);
    if (std::numeric_limits<json::number>::infinity() == f)
        fputs("Infinity", _stream);
    else if (-std::numeric_limits<json::number>::infinity() == f)
        fputs("-Infinity", _stream);
    else if (std::numeric_limits<json::number>::quiet_NaN() == f ||
            std::numeric_limits<json::number>::signaling_NaN() == f)
        fputs("NaN", _stream);
    else
        fprintf(_stream, "%g", f);
    return *this;
}
json & json::operator << (json::integer d) throw()  { context(seq); fprintf(_stream, FORMAT_INTMAX, intmax_t(d)); return *this; }
json & json::operator << (json::integer_u d) throw()  { context(seq); fprintf(_stream, FORMAT_UINTMAX, uintmax_t(d)); return *this; }
json & json::operator << (json::boolean b) throw()  { context(seq); fputs(b ? "true" : "false", _stream); return *this; }
json & json::operator << (std::nullptr_t)  throw()  { context(seq); fputs("null",_stream); return *this; }

#endif
