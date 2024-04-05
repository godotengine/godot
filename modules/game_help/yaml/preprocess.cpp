#include "preprocess.hpp"
#include "./detail/parser_dbg.hpp"

/** @file preprocess.hpp Functions for preprocessing YAML prior to parsing. */

namespace c4 {
namespace yml {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

namespace {
C4_ALWAYS_INLINE bool _is_idchar(char c)
{
    return (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z')
        || (c >= '0' && c <= '9')
        || (c == '_' || c == '-' || c == '~' || c == '$');
}

typedef enum { kReadPending = 0, kKeyPending = 1, kValPending = 2 } _ppstate;
C4_ALWAYS_INLINE _ppstate _next(_ppstate s)
{
    int n = (int)s + 1;
    return (_ppstate)(n <= (int)kValPending ? n : 0);
}
} // empty namespace


//-----------------------------------------------------------------------------

size_t preprocess_rxmap(csubstr s, substr buf)
{
    detail::_SubstrWriter writer(buf);
    _ppstate state = kReadPending;
    size_t last = 0;

    if(s.begins_with('{'))
    {
        RYML_CHECK(s.ends_with('}'));
        s = s.offs(1, 1);
    }

    writer.append('{');

    for(size_t i = 0; i < s.len; ++i)
    {
        const char curr = s[i];
        const char next = i+1 < s.len ? s[i+1] : '\0';

        if(curr == '\'' || curr == '"')
        {
            csubstr ss = s.sub(i).pair_range_esc(curr, '\\');
            i += static_cast<size_t>(ss.end() - (s.str + i));
            state = _next(state);
        }
        else if(state == kReadPending && _is_idchar(curr))
        {
            state = _next(state);
        }

        switch(state)
        {
        case kKeyPending:
        {
            if(curr == ':' && next == ' ')
            {
                state = _next(state);
            }
            else if(curr == ',' && next == ' ')
            {
                writer.append(s.range(last, i));
                writer.append(": 1, ");
                last = i + 2;
            }
            break;
        }
        case kValPending:
        {
            if(curr == '[' || curr == '{' || curr == '(')
            {
                csubstr ss = s.sub(i).pair_range_nested(curr, '\\');
                i += static_cast<size_t>(ss.end() - (s.str + i));
                state = _next(state);
            }
            else if(curr == ',' && next == ' ')
            {
                state = _next(state);
            }
            break;
        }
        default:
            // nothing to do
            break;
        }
    }

    writer.append(s.sub(last));
    if(state == kKeyPending)
        writer.append(": 1");
    writer.append('}');

    return writer.pos;
}

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace yml
} // namespace c4
