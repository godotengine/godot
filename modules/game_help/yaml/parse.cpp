#include "parse.hpp"
#include "c4/error.hpp"
#include "c4/utf.hpp"
#include "c4/dump.hpp"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>

#include "detail/parser_dbg.hpp"
#ifdef RYML_DBG
#include "detail/print.hpp"
#endif

#ifndef RYML_ERRMSG_SIZE
    #define RYML_ERRMSG_SIZE 1024
#endif

//#define RYML_WITH_TAB_TOKENS
#ifdef RYML_WITH_TAB_TOKENS
#define _RYML_WITH_TAB_TOKENS(...) __VA_ARGS__
#define _RYML_WITH_OR_WITHOUT_TAB_TOKENS(with, without) with
#else
#define _RYML_WITH_TAB_TOKENS(...)
#define _RYML_WITH_OR_WITHOUT_TAB_TOKENS(with, without) without
#endif


#if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable: 4296/*expression is always 'boolean_value'*/)
#elif defined(__clang__)
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wtype-limits" // to remove a warning on an assertion that a size_t >= 0. Later on, this size_t will turn into a template argument, and then it can become < 0.
#   pragma clang diagnostic ignored "-Wformat-nonliteral"
#   pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wtype-limits" // to remove a warning on an assertion that a size_t >= 0. Later on, this size_t will turn into a template argument, and then it can become < 0.
#   pragma GCC diagnostic ignored "-Wformat-nonliteral"
#   pragma GCC diagnostic ignored "-Wold-style-cast"
#   if __GNUC__ >= 7
#       pragma GCC diagnostic ignored "-Wduplicated-branches"
#   endif
#endif

namespace c4 {
namespace yml {

namespace {

template<class DumpFn, class ...Args>
void _parse_dump(DumpFn dumpfn, c4::csubstr fmt, Args&& ...args)
{
    char writebuf[256];
    auto results = c4::format_dump_resume(dumpfn, writebuf, fmt, std::forward<Args>(args)...);
    // resume writing if the results failed to fit the buffer
    if(C4_UNLIKELY(results.bufsize > sizeof(writebuf))) // bufsize will be that of the largest element serialized. Eg int(1), will require 1 byte.
    {
        results = format_dump_resume(dumpfn, results, writebuf, fmt, std::forward<Args>(args)...);
        if(C4_UNLIKELY(results.bufsize > sizeof(writebuf)))
        {
            results = format_dump_resume(dumpfn, results, writebuf, fmt, std::forward<Args>(args)...);
        }
    }
}

bool _is_scalar_next__runk(csubstr s)
{
    return !(s.begins_with(": ") || s.begins_with_any("#,{}[]%&") || s.begins_with("? ") || s == "-" || s.begins_with("- ") || s.begins_with(":\"") || s.begins_with(":'"));
}

bool _is_scalar_next__rseq_rval(csubstr s)
{
    return !(s.begins_with_any("[{!&") || s.begins_with("? ") || s.begins_with("- ") || s == "-");
}

bool _is_scalar_next__rmap(csubstr s)
{
    return !(s.begins_with(": ") || s.begins_with_any("#,!&") || s.begins_with("? ") _RYML_WITH_TAB_TOKENS(|| s.begins_with(":\t")));
}

bool _is_scalar_next__rmap_val(csubstr s)
{
    return !(s.begins_with("- ") || s.begins_with_any("{[") || s == "-");
}

bool _is_doc_sep(csubstr s)
{
    constexpr const csubstr dashes = "---";
    constexpr const csubstr ellipsis = "...";
    constexpr const csubstr whitesp = " \t";
    if(s.begins_with(dashes))
        return s == dashes || s.sub(3).begins_with_any(whitesp);
    else if(s.begins_with(ellipsis))
        return s == ellipsis || s.sub(3).begins_with_any(whitesp);
    return false;
}

/** @p i is set to the first non whitespace character after the line
 * @return the number of empty lines after the initial position */
size_t count_following_newlines(csubstr r, size_t *C4_RESTRICT i, size_t indentation)
{
    RYML_ASSERT(r[*i] == '\n');
    size_t numnl_following = 0;
    ++(*i);
    for( ; *i < r.len; ++(*i))
    {
        if(r.str[*i] == '\n')
        {
            ++numnl_following;
            if(indentation) // skip the indentation after the newline
            {
                size_t stop = *i + indentation;
                for( ; *i < r.len; ++(*i))
                {
                    if(r.str[*i] != ' ' && r.str[*i] != '\r')
                        break;
                    RYML_ASSERT(*i < stop);
                }
                C4_UNUSED(stop);
            }
        }
        else if(r.str[*i] == ' ' || r.str[*i] == '\t' || r.str[*i] == '\r')  // skip leading whitespace
            ;
        else
            break;
    }
    return numnl_following;
}

} // anon namespace


//-----------------------------------------------------------------------------

Parser::~Parser()
{
    _free();
    _clr();
}

Parser::Parser(Callbacks const& cb, ParserOptions opts)
    : m_options(opts)
    , m_file()
    , m_buf()
    , m_root_id(NONE)
    , m_tree()
    , m_stack(cb)
    , m_state()
    , m_key_tag_indentation(0)
    , m_key_tag2_indentation(0)
    , m_key_tag()
    , m_key_tag2()
    , m_val_tag_indentation(0)
    , m_val_tag()
    , m_key_anchor_was_before(false)
    , m_key_anchor_indentation(0)
    , m_key_anchor()
    , m_val_anchor_indentation(0)
    , m_val_anchor()
    , m_filter_arena()
    , m_newline_offsets()
    , m_newline_offsets_size(0)
    , m_newline_offsets_capacity(0)
    , m_newline_offsets_buf()
{
    m_stack.push(State{});
    m_state = &m_stack.top();
}

Parser::Parser(Parser &&that)
    : m_options(that.m_options)
    , m_file(that.m_file)
    , m_buf(that.m_buf)
    , m_root_id(that.m_root_id)
    , m_tree(that.m_tree)
    , m_stack(std::move(that.m_stack))
    , m_state(&m_stack.top())
    , m_key_tag_indentation(that.m_key_tag_indentation)
    , m_key_tag2_indentation(that.m_key_tag2_indentation)
    , m_key_tag(that.m_key_tag)
    , m_key_tag2(that.m_key_tag2)
    , m_val_tag_indentation(that.m_val_tag_indentation)
    , m_val_tag(that.m_val_tag)
    , m_key_anchor_was_before(that.m_key_anchor_was_before)
    , m_key_anchor_indentation(that.m_key_anchor_indentation)
    , m_key_anchor(that.m_key_anchor)
    , m_val_anchor_indentation(that.m_val_anchor_indentation)
    , m_val_anchor(that.m_val_anchor)
    , m_filter_arena(that.m_filter_arena)
    , m_newline_offsets(that.m_newline_offsets)
    , m_newline_offsets_size(that.m_newline_offsets_size)
    , m_newline_offsets_capacity(that.m_newline_offsets_capacity)
    , m_newline_offsets_buf(that.m_newline_offsets_buf)
{
    that._clr();
}

Parser::Parser(Parser const& that)
    : m_options(that.m_options)
    , m_file(that.m_file)
    , m_buf(that.m_buf)
    , m_root_id(that.m_root_id)
    , m_tree(that.m_tree)
    , m_stack(that.m_stack)
    , m_state(&m_stack.top())
    , m_key_tag_indentation(that.m_key_tag_indentation)
    , m_key_tag2_indentation(that.m_key_tag2_indentation)
    , m_key_tag(that.m_key_tag)
    , m_key_tag2(that.m_key_tag2)
    , m_val_tag_indentation(that.m_val_tag_indentation)
    , m_val_tag(that.m_val_tag)
    , m_key_anchor_was_before(that.m_key_anchor_was_before)
    , m_key_anchor_indentation(that.m_key_anchor_indentation)
    , m_key_anchor(that.m_key_anchor)
    , m_val_anchor_indentation(that.m_val_anchor_indentation)
    , m_val_anchor(that.m_val_anchor)
    , m_filter_arena()
    , m_newline_offsets()
    , m_newline_offsets_size()
    , m_newline_offsets_capacity()
    , m_newline_offsets_buf()
{
    if(that.m_newline_offsets_capacity)
    {
        _resize_locations(that.m_newline_offsets_capacity);
        _RYML_CB_CHECK(m_stack.m_callbacks, m_newline_offsets_capacity == that.m_newline_offsets_capacity);
        memcpy(m_newline_offsets, that.m_newline_offsets, that.m_newline_offsets_size * sizeof(size_t));
        m_newline_offsets_size = that.m_newline_offsets_size;
    }
    if(that.m_filter_arena.len)
    {
        _resize_filter_arena(that.m_filter_arena.len);
    }
}

Parser& Parser::operator=(Parser &&that)
{
    _free();
    m_options = (that.m_options);
    m_file = (that.m_file);
    m_buf = (that.m_buf);
    m_root_id = (that.m_root_id);
    m_tree = (that.m_tree);
    m_stack = std::move(that.m_stack);
    m_state = (&m_stack.top());
    m_key_tag_indentation = (that.m_key_tag_indentation);
    m_key_tag2_indentation = (that.m_key_tag2_indentation);
    m_key_tag = (that.m_key_tag);
    m_key_tag2 = (that.m_key_tag2);
    m_val_tag_indentation = (that.m_val_tag_indentation);
    m_val_tag = (that.m_val_tag);
    m_key_anchor_was_before = (that.m_key_anchor_was_before);
    m_key_anchor_indentation = (that.m_key_anchor_indentation);
    m_key_anchor = (that.m_key_anchor);
    m_val_anchor_indentation = (that.m_val_anchor_indentation);
    m_val_anchor = (that.m_val_anchor);
    m_filter_arena = that.m_filter_arena;
    m_newline_offsets = (that.m_newline_offsets);
    m_newline_offsets_size = (that.m_newline_offsets_size);
    m_newline_offsets_capacity = (that.m_newline_offsets_capacity);
    m_newline_offsets_buf = (that.m_newline_offsets_buf);
    that._clr();
    return *this;
}

Parser& Parser::operator=(Parser const& that)
{
    _free();
    m_options = (that.m_options);
    m_file = (that.m_file);
    m_buf = (that.m_buf);
    m_root_id = (that.m_root_id);
    m_tree = (that.m_tree);
    m_stack = that.m_stack;
    m_state = &m_stack.top();
    m_key_tag_indentation = (that.m_key_tag_indentation);
    m_key_tag2_indentation = (that.m_key_tag2_indentation);
    m_key_tag = (that.m_key_tag);
    m_key_tag2 = (that.m_key_tag2);
    m_val_tag_indentation = (that.m_val_tag_indentation);
    m_val_tag = (that.m_val_tag);
    m_key_anchor_was_before = (that.m_key_anchor_was_before);
    m_key_anchor_indentation = (that.m_key_anchor_indentation);
    m_key_anchor = (that.m_key_anchor);
    m_val_anchor_indentation = (that.m_val_anchor_indentation);
    m_val_anchor = (that.m_val_anchor);
    if(that.m_filter_arena.len > 0)
        _resize_filter_arena(that.m_filter_arena.len);
    if(that.m_newline_offsets_capacity > m_newline_offsets_capacity)
        _resize_locations(that.m_newline_offsets_capacity);
    _RYML_CB_CHECK(m_stack.m_callbacks, m_newline_offsets_capacity >= that.m_newline_offsets_capacity);
    _RYML_CB_CHECK(m_stack.m_callbacks, m_newline_offsets_capacity >= that.m_newline_offsets_size);
    memcpy(m_newline_offsets, that.m_newline_offsets, that.m_newline_offsets_size * sizeof(size_t));
    m_newline_offsets_size = that.m_newline_offsets_size;
    m_newline_offsets_buf = that.m_newline_offsets_buf;
    return *this;
}

void Parser::_clr()
{
    m_options = {};
    m_file = {};
    m_buf = {};
    m_root_id = {};
    m_tree = {};
    m_stack.clear();
    m_state = {};
    m_key_tag_indentation = {};
    m_key_tag2_indentation = {};
    m_key_tag = {};
    m_key_tag2 = {};
    m_val_tag_indentation = {};
    m_val_tag = {};
    m_key_anchor_was_before = {};
    m_key_anchor_indentation = {};
    m_key_anchor = {};
    m_val_anchor_indentation = {};
    m_val_anchor = {};
    m_filter_arena = {};
    m_newline_offsets = {};
    m_newline_offsets_size = {};
    m_newline_offsets_capacity = {};
    m_newline_offsets_buf = {};
}

void Parser::_free()
{
    if(m_newline_offsets)
    {
        _RYML_CB_FREE(m_stack.m_callbacks, m_newline_offsets, size_t, m_newline_offsets_capacity);
        m_newline_offsets = nullptr;
        m_newline_offsets_size = 0u;
        m_newline_offsets_capacity = 0u;
        m_newline_offsets_buf = 0u;
    }
    if(m_filter_arena.len)
    {
        _RYML_CB_FREE(m_stack.m_callbacks, m_filter_arena.str, char, m_filter_arena.len);
        m_filter_arena = {};
    }
    m_stack._free();
}


//-----------------------------------------------------------------------------
void Parser::_reset()
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_stack.size() == 1);
    m_stack.clear();
    m_stack.push({});
    m_state = &m_stack.top();
    m_state->reset(m_file.str, m_root_id);

    m_key_tag_indentation = 0;
    m_key_tag2_indentation = 0;
    m_key_tag.clear();
    m_key_tag2.clear();
    m_val_tag_indentation = 0;
    m_val_tag.clear();
    m_key_anchor_was_before = false;
    m_key_anchor_indentation = 0;
    m_key_anchor.clear();
    m_val_anchor_indentation = 0;
    m_val_anchor.clear();

    if(m_options.locations())
    {
        _prepare_locations();
    }
}

//-----------------------------------------------------------------------------
template<class DumpFn>
void Parser::_fmt_msg(DumpFn &&dumpfn) const
{
    auto const& lc = m_state->line_contents;
    csubstr contents = lc.stripped;
    if(contents.len)
    {
        // print the yaml src line
        size_t offs = 3u + to_chars(substr{}, m_state->pos.line) + to_chars(substr{}, m_state->pos.col);
        if(m_file.len)
        {
            _parse_dump(dumpfn, "{}:", m_file);
            offs += m_file.len + 1;
        }
        _parse_dump(dumpfn, "{}:{}: ", m_state->pos.line, m_state->pos.col);
        csubstr maybe_full_content = (contents.len < 80u ? contents : contents.first(80u));
        csubstr maybe_ellipsis = (contents.len < 80u ? csubstr{} : csubstr("..."));
        _parse_dump(dumpfn, "{}{}  (size={})\n", maybe_full_content, maybe_ellipsis, contents.len);
        // highlight the remaining portion of the previous line
        size_t firstcol = (size_t)(lc.rem.begin() - lc.full.begin());
        size_t lastcol = firstcol + lc.rem.len;
        for(size_t i = 0; i < offs + firstcol; ++i)
            dumpfn(" ");
        dumpfn("^");
        for(size_t i = 1, e = (lc.rem.len < 80u ? lc.rem.len : 80u); i < e; ++i)
            dumpfn("~");
        _parse_dump(dumpfn, "{}  (cols {}-{})\n", maybe_ellipsis, firstcol+1, lastcol+1);
    }
    else
    {
        dumpfn("\n");
    }

#ifdef RYML_DBG
    // next line: print the state flags
    {
        char flagbuf_[64];
        _parse_dump(dumpfn, "top state: {}\n", _prfl(flagbuf_, m_state->flags));
    }
#endif
}


//-----------------------------------------------------------------------------
template<class ...Args>
void Parser::_err(csubstr fmt, Args const& C4_RESTRICT ...args) const
{
    char errmsg[RYML_ERRMSG_SIZE];
    detail::_SubstrWriter writer(errmsg);
    auto dumpfn = [&writer](csubstr s){ writer.append(s); };
    _parse_dump(dumpfn, fmt, args...);
    writer.append('\n');
    _fmt_msg(dumpfn);
    size_t len = writer.pos < RYML_ERRMSG_SIZE ? writer.pos : RYML_ERRMSG_SIZE;
    m_tree->m_callbacks.m_error(errmsg, len, m_state->pos, m_tree->m_callbacks.m_user_data);
}

//-----------------------------------------------------------------------------
#ifdef RYML_DBG
template<class ...Args>
void Parser::_dbg(csubstr fmt, Args const& C4_RESTRICT ...args) const
{
    auto dumpfn = [](csubstr s){ fwrite(s.str, 1, s.len, stdout); };
    _parse_dump(dumpfn, fmt, args...);
    dumpfn("\n");
    _fmt_msg(dumpfn);
}
#endif

//-----------------------------------------------------------------------------
bool Parser::_finished_file() const
{
    bool ret = m_state->pos.offset >= m_buf.len;
    if(ret)
    {
        _c4dbgp("finished file!!!");
    }
    return ret;
}

//-----------------------------------------------------------------------------
bool Parser::_finished_line() const
{
    return m_state->line_contents.rem.empty();
}

//-----------------------------------------------------------------------------
void Parser::parse_in_place(csubstr file, substr buf, Tree *t, size_t node_id)
{
    m_file = file;
    m_buf = buf;
    m_root_id = node_id;
    m_tree = t;
    _reset();
    while( ! _finished_file())
    {
        _scan_line();
        while( ! _finished_line())
            _handle_line();
        if(_finished_file())
            break; // it may have finished because of multiline blocks
        _line_ended();
    }
    _handle_finished_file();
}

//-----------------------------------------------------------------------------
void Parser::_handle_finished_file()
{
    _end_stream();
}

//-----------------------------------------------------------------------------
void Parser::_handle_line()
{
    _c4dbgq("\n-----------");
    _c4dbgt("handling line={}, offset={}B", m_state->pos.line, m_state->pos.offset);
    _RYML_CB_ASSERT(m_stack.m_callbacks,  ! m_state->line_contents.rem.empty());
    if(has_any(RSEQ))
    {
        if(has_any(FLOW))
        {
            if(_handle_seq_flow())
                return;
        }
        else
        {
            if(_handle_seq_blck())
                return;
        }
    }
    else if(has_any(RMAP))
    {
        if(has_any(FLOW))
        {
            if(_handle_map_flow())
                return;
        }
        else
        {
            if(_handle_map_blck())
                return;
        }
    }
    else if(has_any(RUNK))
    {
        if(_handle_unk())
            return;
    }

    if(_handle_top())
        return;
}


//-----------------------------------------------------------------------------
bool Parser::_handle_unk()
{
    _c4dbgp("handle_unk");

    csubstr rem = m_state->line_contents.rem;
    const bool start_as_child = (node(m_state) == nullptr);

    if(C4_UNLIKELY(has_any(NDOC)))
    {
        if(rem == "---" || rem.begins_with("--- "))
        {
            _start_new_doc(rem);
            return true;
        }
        auto trimmed = rem.triml(' ');
        if(trimmed == "---" || trimmed.begins_with("--- "))
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks, rem.len >= trimmed.len);
            _line_progressed(rem.len - trimmed.len);
            _start_new_doc(trimmed);
            _save_indentation();
            return true;
        }
        else if(trimmed.begins_with("..."))
        {
            _end_stream();
        }
        else if(trimmed.first_of("#%") == csubstr::npos) // neither a doc nor a tag
        {
            _c4dbgpf("starting implicit doc to accomodate unexpected tokens: '{}'", rem);
            size_t indref = m_state->indref;
            _push_level();
            _start_doc();
            _set_indentation(indref);
        }
        _RYML_CB_ASSERT(m_stack.m_callbacks, !trimmed.empty());
    }

    _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RNXT|RSEQ|RMAP));
    if(m_state->indref > 0)
    {
        csubstr ws = rem.left_of(rem.first_not_of(' '));
        if(m_state->indref <= ws.len)
        {
            _c4dbgpf("skipping base indentation of {}", m_state->indref);
            _line_progressed(m_state->indref);
            rem = rem.sub(m_state->indref);
        }
    }

    if(rem.begins_with("- ") _RYML_WITH_TAB_TOKENS( || rem.begins_with("-\t")))
    {
        _c4dbgpf("it's a seq (as_child={})", start_as_child);
        _move_key_anchor_to_val_anchor();
        _move_key_tag_to_val_tag();
        _push_level();
        _start_seq(start_as_child);
        _save_indentation();
        _line_progressed(2);
        return true;
    }
    else if(rem == '-')
    {
        _c4dbgpf("it's a seq (as_child={})", start_as_child);
        _move_key_anchor_to_val_anchor();
        _move_key_tag_to_val_tag();
        _push_level();
        _start_seq(start_as_child);
        _save_indentation();
        _line_progressed(1);
        return true;
    }
    else if(rem.begins_with('['))
    {
        _c4dbgpf("it's a seq, flow (as_child={})", start_as_child);
        _move_key_anchor_to_val_anchor();
        _move_key_tag_to_val_tag();
        _push_level(/*explicit flow*/true);
        _start_seq(start_as_child);
        add_flags(FLOW);
        _line_progressed(1);
        return true;
    }
    else if(rem.begins_with('{'))
    {
        _c4dbgpf("it's a map, flow (as_child={})", start_as_child);
        _move_key_anchor_to_val_anchor();
        _move_key_tag_to_val_tag();
        _push_level(/*explicit flow*/true);
        _start_map(start_as_child);
        addrem_flags(FLOW|RKEY, RVAL);
        _line_progressed(1);
        return true;
    }
    else if(rem.begins_with("? "))
    {
        _c4dbgpf("it's a map (as_child={}) + this key is complex", start_as_child);
        _move_key_anchor_to_val_anchor();
        _move_key_tag_to_val_tag();
        _push_level();
        _start_map(start_as_child);
        addrem_flags(RKEY|QMRK, RVAL);
        _save_indentation();
        _line_progressed(2);
        return true;
    }
    else if(rem.begins_with(": ") && !has_any(SSCL))
    {
        _c4dbgp("it's a map with an empty key");
        _move_key_anchor_to_val_anchor();
        _move_key_tag_to_val_tag();
        _push_level();
        _start_map(start_as_child);
        _store_scalar_null(rem.str);
        addrem_flags(RVAL, RKEY);
        _save_indentation();
        _line_progressed(2);
        return true;
    }
    else if(rem == ':' && !has_any(SSCL))
    {
        _c4dbgp("it's a map with an empty key");
        _move_key_anchor_to_val_anchor();
        _move_key_tag_to_val_tag();
        _push_level();
        _start_map(start_as_child);
        _store_scalar_null(rem.str);
        addrem_flags(RVAL, RKEY);
        _save_indentation();
        _line_progressed(1);
        return true;
    }
    else if(_handle_types())
    {
        return true;
    }
    else if(!rem.begins_with('*') && _handle_key_anchors_and_refs())
    {
        return true;
    }
    else if(has_any(SSCL))
    {
        _c4dbgpf("there's a stored scalar: '{}'", m_state->scalar);

        csubstr saved_scalar;
        bool is_quoted = false;
        if(_scan_scalar_unk(&saved_scalar, &is_quoted))
        {
            rem = m_state->line_contents.rem;
            _c4dbgpf("... and there's also a scalar next! '{}'", saved_scalar);
            if(rem.begins_with_any(" \t"))
            {
                size_t n = rem.first_not_of(" \t");
                _c4dbgpf("skipping {} spaces/tabs", n);
                rem = rem.sub(n);
                _line_progressed(n);
            }
        }

        _c4dbgpf("rem='{}'", rem);

        if(rem.begins_with(", "))
        {
            _c4dbgpf("got a ',' -- it's a seq (as_child={})", start_as_child);
            _start_seq(start_as_child);
            add_flags(FLOW);
            _append_val(_consume_scalar());
            _line_progressed(2);
        }
        else if(rem.begins_with(','))
        {
            _c4dbgpf("got a ',' -- it's a seq (as_child={})", start_as_child);
            _start_seq(start_as_child);
            add_flags(FLOW);
            _append_val(_consume_scalar());
            _line_progressed(1);
        }
        else if(rem.begins_with(": ") _RYML_WITH_TAB_TOKENS( || rem.begins_with(":\t")))
        {
            _c4dbgpf("got a ': ' -- it's a map (as_child={})", start_as_child);
            _start_map_unk(start_as_child); // wait for the val scalar to append the key-val pair
            _line_progressed(2);
        }
        else if(rem == ":" || rem.begins_with(":\"") || rem.begins_with(":'"))
        {
            if(rem == ":") { _c4dbgpf("got a ':' -- it's a map (as_child={})", start_as_child); }
            else { _c4dbgpf("got a '{}' -- it's a map (as_child={})", rem.first(2), start_as_child); }
            _start_map_unk(start_as_child); // wait for the val scalar to append the key-val pair
            _line_progressed(1); // advance only 1
        }
        #ifdef RYML_NO_COVERAGE__TO_BE_DELETED
        else if(rem.begins_with('}'))
        {
            if(!has_all(RMAP|FLOW))
            {
                _c4err("invalid token: not reading a map");
            }
            if(!has_all(SSCL))
            {
                _c4err("no scalar stored");
            }
            _append_key_val(saved_scalar, is_quoted);
            _stop_map();
            _line_progressed(1);
            saved_scalar.clear();
            is_quoted = false;
        }
        #endif
        else if(rem.begins_with("..."))
        {
            _c4dbgp("got stream end '...'");
            _end_stream();
            _line_progressed(3);
        }
        else if(rem.begins_with('#'))
        {
            _c4dbgpf("it's a comment: '{}'", rem);
            _scan_comment();
            return true;
        }
        else if(_handle_key_anchors_and_refs())
        {
            return true;
        }
        else if(rem.begins_with(" ") || rem.begins_with("\t"))
        {
            size_t n = rem.first_not_of(" \t");
            if(n == npos)
                n = rem.len;
            _c4dbgpf("has {} spaces/tabs, skip...", n);
            _line_progressed(n);
            return true;
        }
        else if(rem.empty())
        {
            // nothing to do
        }
        else if(rem == "---" || rem.begins_with("--- "))
        {
            _c4dbgp("caught ---: starting doc");
            _start_new_doc(rem);
            return true;
        }
        else if(rem.begins_with('%'))
        {
            _c4dbgp("caught a directive: ignoring...");
            _line_progressed(rem.len);
            return true;
        }
        else
        {
            _c4err("parse error");
        }

        if(is_quoted || (! saved_scalar.empty()))
        {
            _store_scalar(saved_scalar, is_quoted);
        }

        return true;
    }
    else
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks,  ! has_any(SSCL));
        csubstr scalar;
        size_t indentation = m_state->line_contents.indentation; // save
        bool is_quoted;
        if(_scan_scalar_unk(&scalar, &is_quoted))
        {
            _c4dbgpf("got a {} scalar", is_quoted ? "quoted" : "");
            rem = m_state->line_contents.rem;
            {
                size_t first = rem.first_not_of(" \t");
                if(first && first != npos)
                {
                    _c4dbgpf("skip {} whitespace characters", first);
                   _line_progressed(first);
                   rem = rem.sub(first);
                }
            }
            _store_scalar(scalar, is_quoted);
            if(rem.begins_with(": ") _RYML_WITH_TAB_TOKENS( || rem.begins_with(":\t")))
            {
                _c4dbgpf("got a ': ' next -- it's a map (as_child={})", start_as_child);
                _push_level();
                _start_map(start_as_child); // wait for the val scalar to append the key-val pair
                _set_indentation(indentation);
                _line_progressed(2); // call this AFTER saving the indentation
            }
            else if(rem.begins_with(':'))
            {
                _c4dbgpf("got a ':' next -- it's a map (as_child={})", start_as_child);
                _push_level();
                _start_map(start_as_child); // wait for the val scalar to append the key-val pair
                _set_indentation(indentation);
                _line_progressed(1); // call this AFTER saving the indentation
            }
            else
            {
                // we still don't know whether it's a seq or a map
                // so just store the scalar
            }
            return true;
        }
        else if(rem.begins_with_any(" \t"))
        {
            csubstr ws = rem.left_of(rem.first_not_of(" \t"));
            rem = rem.right_of(ws);
            if(has_all(RTOP) && rem.begins_with("---"))
            {
                _c4dbgp("there's a doc starting, and it's indented");
                _set_indentation(ws.len);
            }
            _c4dbgpf("skipping {} spaces/tabs", ws.len);
            _line_progressed(ws.len);
            return true;
        }
    }

    return false;
}


//-----------------------------------------------------------------------------
C4_ALWAYS_INLINE void Parser::_skipchars(char c)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->line_contents.rem.begins_with(c));
    size_t pos = m_state->line_contents.rem.first_not_of(c);
    if(pos == npos)
        pos = m_state->line_contents.rem.len; // maybe the line is just whitespace
    _c4dbgpf("skip {} '{}'", pos, c);
    _line_progressed(pos);
}

template<size_t N>
C4_ALWAYS_INLINE void Parser::_skipchars(const char (&chars)[N])
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->line_contents.rem.begins_with_any(chars));
    size_t pos = m_state->line_contents.rem.first_not_of(chars);
    if(pos == npos)
        pos = m_state->line_contents.rem.len; // maybe the line is just whitespace
    _c4dbgpf("skip {} characters", pos);
    _line_progressed(pos);
}


//-----------------------------------------------------------------------------
bool Parser::_handle_seq_flow()
{
    _c4dbgpf("handle_seq_flow: node_id={} level={}", m_state->node_id, m_state->level);
    csubstr rem = m_state->line_contents.rem;

    _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RKEY));
    _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(RSEQ|FLOW));

    if(rem.begins_with(' '))
    {
        // with explicit flow, indentation does not matter
        _c4dbgp("starts with spaces");
        _skipchars(' ');
        return true;
    }
    _RYML_WITH_TAB_TOKENS(else if(rem.begins_with('\t'))
    {
        _c4dbgp("starts with tabs");
        _skipchars('\t');
        return true;
    })
    else if(rem.begins_with('#'))
    {
        _c4dbgp("it's a comment");
        rem = _scan_comment(); // also progresses the line
        return true;
    }
    else if(rem.begins_with(']'))
    {
        _c4dbgp("end the sequence");
        _pop_level();
        _line_progressed(1);
        if(has_all(RSEQIMAP))
        {
            _stop_seqimap();
            _pop_level();
        }
        return true;
    }

    if(has_any(RVAL))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RNXT));
        bool is_quoted;
        if(_scan_scalar_seq_flow(&rem, &is_quoted))
        {
            _c4dbgp("it's a scalar");
            addrem_flags(RNXT, RVAL);
            _append_val(rem, is_quoted);
            return true;
        }
        else if(rem.begins_with('['))
        {
            _c4dbgp("val is a child seq");
            addrem_flags(RNXT, RVAL); // before _push_level!
            _push_level(/*explicit flow*/true);
            _start_seq();
            add_flags(FLOW);
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with('{'))
        {
            _c4dbgp("val is a child map");
            addrem_flags(RNXT, RVAL); // before _push_level!
            _push_level(/*explicit flow*/true);
            _start_map();
            addrem_flags(FLOW|RKEY, RVAL);
            _line_progressed(1);
            return true;
        }
        else if(rem == ':')
        {
            _c4dbgpf("found ':' -- there's an implicit map in the seq node[{}]", m_state->node_id);
            _start_seqimap();
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with(": ") _RYML_WITH_TAB_TOKENS( || rem.begins_with(":\t")))
        {
            _c4dbgpf("found ': ' -- there's an implicit map in the seq node[{}]", m_state->node_id);
            _start_seqimap();
            _line_progressed(2);
            return true;
        }
        else if(rem.begins_with("? "))
        {
            _c4dbgpf("found '? ' -- there's an implicit map in the seq node[{}]", m_state->node_id);
            _start_seqimap();
            _line_progressed(2);
            _RYML_CB_ASSERT(m_stack.m_callbacks, has_any(SSCL) && m_state->scalar == "");
            addrem_flags(QMRK|RKEY, RVAL|SSCL);
            return true;
        }
        else if(_handle_types())
        {
            return true;
        }
        else if(_handle_val_anchors_and_refs())
        {
            return true;
        }
        else if(rem.begins_with(", "))
        {
            _c4dbgp("found ',' -- the value was null");
            _append_val_null(rem.str - 1);
            _line_progressed(2);
            return true;
        }
        else if(rem.begins_with(','))
        {
            _c4dbgp("found ',' -- the value was null");
            _append_val_null(rem.str - 1);
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with('\t'))
        {
            _skipchars('\t');
            return true;
        }
        else
        {
            _c4err("parse error");
        }
    }
    else if(has_any(RNXT))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RVAL));
        if(rem.begins_with(", "))
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(FLOW));
            _c4dbgp("seq: expect next val");
            addrem_flags(RVAL, RNXT);
            _line_progressed(2);
            return true;
        }
        else if(rem.begins_with(','))
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(FLOW));
            _c4dbgp("seq: expect next val");
            addrem_flags(RVAL, RNXT);
            _line_progressed(1);
            return true;
        }
        else if(rem == ':')
        {
            _c4dbgpf("found ':' -- there's an implicit map in the seq node[{}]", m_state->node_id);
            _start_seqimap();
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with(": ") _RYML_WITH_TAB_TOKENS( || rem.begins_with(":\t")))
        {
            _c4dbgpf("found ': ' -- there's an implicit map in the seq node[{}]", m_state->node_id);
            _start_seqimap();
            _line_progressed(2);
            return true;
        }
        else
        {
            _c4err("was expecting a comma");
        }
    }
    else
    {
        _c4err("internal error");
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Parser::_handle_seq_blck()
{
    _c4dbgpf("handle_seq_impl: node_id={} level={}", m_state->node_id, m_state->level);
    csubstr rem = m_state->line_contents.rem;

    _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(RSEQ));
    _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RKEY));
    _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(FLOW));

    if(rem.begins_with('#'))
    {
        _c4dbgp("it's a comment");
        rem = _scan_comment();
        return true;
    }
    if(has_any(RNXT))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RVAL));

        if(_handle_indentation())
            return true;

        if(rem.begins_with("- ") _RYML_WITH_TAB_TOKENS( || rem.begins_with("-\t")))
        {
            _c4dbgp("expect another val");
            addrem_flags(RVAL, RNXT);
            _line_progressed(2);
            return true;
        }
        else if(rem == '-')
        {
            _c4dbgp("expect another val");
            addrem_flags(RVAL, RNXT);
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with_any(" \t"))
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks,  ! _at_line_begin());
            _skipchars(" \t");
            return true;
        }
        else if(rem.begins_with("..."))
        {
            _c4dbgp("got stream end '...'");
            _end_stream();
            _line_progressed(3);
            return true;
        }
        else if(rem.begins_with("---"))
        {
            _c4dbgp("got document start '---'");
            _start_new_doc(rem);
            return true;
        }
        else
        {
            _c4err("parse error");
        }
    }
    else if(has_any(RVAL))
    {
        // there can be empty values
        if(_handle_indentation())
            return true;

        csubstr s;
        bool is_quoted;
        if(_scan_scalar_seq_blck(&s, &is_quoted)) // this also progresses the line
        {
            _c4dbgpf("it's a{} scalar", is_quoted ? " quoted" : "");

            rem = m_state->line_contents.rem;
            if(_RYML_WITH_OR_WITHOUT_TAB_TOKENS(rem.begins_with_any(" \t"), rem.begins_with(' ')))
            {
                _c4dbgp("skipping whitespace...");
                size_t skip = rem.first_not_of(_RYML_WITH_OR_WITHOUT_TAB_TOKENS(" \t", ' '));
                if(skip == csubstr::npos)
                    skip = rem.len; // maybe the line is just whitespace
                _line_progressed(skip);
                rem = rem.sub(skip);
            }

            _c4dbgpf("rem=[{}]~~~{}~~~", rem.len, rem);
            if(!rem.begins_with('#') && (rem.ends_with(':') || rem.begins_with(": ") _RYML_WITH_TAB_TOKENS( || rem.begins_with(":\t"))))
            {
                _c4dbgp("actually, the scalar is the first key of a map, and it opens a new scope");
                if(m_key_anchor.empty())
                    _move_val_anchor_to_key_anchor();
                if(m_key_tag.empty())
                    _move_val_tag_to_key_tag();
                addrem_flags(RNXT, RVAL); // before _push_level! This prepares the current level for popping by setting it to RNXT
                _push_level();
                _start_map();
                _store_scalar(s, is_quoted);
                if( ! _maybe_set_indentation_from_anchor_or_tag())
                {
                    _c4dbgpf("set indentation from scalar: {}", m_state->scalar_col);
                    _set_indentation(m_state->scalar_col); // this is the column where the scalar starts
                }
                _move_key_tag2_to_key_tag();
                addrem_flags(RVAL, RKEY);
                _line_progressed(1);
            }
            else
            {
                _c4dbgp("appending val to current seq");
                _append_val(s, is_quoted);
                addrem_flags(RNXT, RVAL);
            }
            return true;
        }
        else if(rem.begins_with("- ") _RYML_WITH_TAB_TOKENS( || rem.begins_with("-\t")))
        {
            if(_rval_dash_start_or_continue_seq())
                _line_progressed(2);
            return true;
        }
        else if(rem == '-')
        {
            if(_rval_dash_start_or_continue_seq())
                _line_progressed(1);
            return true;
        }
        else if(rem.begins_with('['))
        {
            _c4dbgp("val is a child seq, flow");
            addrem_flags(RNXT, RVAL); // before _push_level!
            _push_level(/*explicit flow*/true);
            _start_seq();
            add_flags(FLOW);
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with('{'))
        {
            _c4dbgp("val is a child map, flow");
            addrem_flags(RNXT, RVAL); // before _push_level!
            _push_level(/*explicit flow*/true);
            _start_map();
            addrem_flags(FLOW|RKEY, RVAL);
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with("? "))
        {
            _c4dbgp("val is a child map + this key is complex");
            addrem_flags(RNXT, RVAL); // before _push_level!
            _push_level();
            _start_map();
            addrem_flags(QMRK|RKEY, RVAL);
            _save_indentation();
            _line_progressed(2);
            return true;
        }
        else if(rem.begins_with(' '))
        {
            csubstr spc = rem.left_of(rem.first_not_of(' '));
            if(_at_line_begin())
            {
                _c4dbgpf("skipping value indentation: {} spaces", spc.len);
                _line_progressed(spc.len);
                return true;
            }
            else
            {
                _c4dbgpf("skipping {} spaces", spc.len);
                _line_progressed(spc.len);
                return true;
            }
        }
        else if(_handle_types())
        {
            return true;
        }
        else if(_handle_val_anchors_and_refs())
        {
            return true;
        }
        /* pathological case:
         * - &key : val
         * - &key :
         * - : val
         */
        else if((!has_all(SSCL)) &&
                (rem.begins_with(": ") || rem.left_of(rem.find("#")).trimr("\t") == ":"))
        {
            if(!m_val_anchor.empty() || !m_val_tag.empty())
            {
                _c4dbgp("val is a child map + this key is empty, with anchors or tags");
                addrem_flags(RNXT, RVAL); // before _push_level!
                _move_val_tag_to_key_tag();
                _move_val_anchor_to_key_anchor();
                _push_level();
                _start_map();
                _store_scalar_null(rem.str);
                addrem_flags(RVAL, RKEY);
                RYML_CHECK(_maybe_set_indentation_from_anchor_or_tag()); // one of them must exist
                _line_progressed(rem.begins_with(": ") ? 2u : 1u);
                return true;
            }
            else
            {
                _c4dbgp("val is a child map + this key is empty, no anchors or tags");
                addrem_flags(RNXT, RVAL); // before _push_level!
                size_t ind = m_state->indref;
                _push_level();
                _start_map();
                _store_scalar_null(rem.str);
                addrem_flags(RVAL, RKEY);
                _c4dbgpf("set indentation from map anchor: {}", ind + 2);
                _set_indentation(ind + 2); // this is the column where the map starts
                _line_progressed(rem.begins_with(": ") ? 2u : 1u);
                return true;
            }
        }
        else
        {
            _c4err("parse error");
        }
    }

    return false;
}

//-----------------------------------------------------------------------------

bool Parser::_rval_dash_start_or_continue_seq()
{
    size_t ind = m_state->line_contents.current_col();
    _RYML_CB_ASSERT(m_stack.m_callbacks, ind >= m_state->indref);
    size_t delta_ind = ind - m_state->indref;
    if( ! delta_ind)
    {
        _c4dbgp("prev val was empty");
        addrem_flags(RNXT, RVAL);
        _append_val_null(&m_state->line_contents.full[ind]);
        return false;
    }
    _c4dbgp("val is a nested seq, indented");
    addrem_flags(RNXT, RVAL); // before _push_level!
    _push_level();
    _start_seq();
    _save_indentation();
    return true;
}

//-----------------------------------------------------------------------------
bool Parser::_handle_map_flow()
{
    // explicit flow, ie, inside {}, separated by commas
    _c4dbgpf("handle_map_flow: node_id={}  level={}", m_state->node_id, m_state->level);
    csubstr rem = m_state->line_contents.rem;

    _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(RMAP|FLOW));

    if(rem.begins_with(' '))
    {
        // with explicit flow, indentation does not matter
        _c4dbgp("starts with spaces");
        _skipchars(' ');
        return true;
    }
    _RYML_WITH_TAB_TOKENS(else if(rem.begins_with('\t'))
    {
        // with explicit flow, indentation does not matter
        _c4dbgp("starts with tabs");
        _skipchars('\t');
        return true;
    })
    else if(rem.begins_with('#'))
    {
        _c4dbgp("it's a comment");
        rem = _scan_comment(); // also progresses the line
        return true;
    }
    else if(rem.begins_with('}'))
    {
        _c4dbgp("end the map");
        if(has_all(SSCL))
        {
            _c4dbgp("the last val was null");
            _append_key_val_null(rem.str - 1);
            rem_flags(RVAL);
        }
        _pop_level();
        _line_progressed(1);
        if(has_all(RSEQIMAP))
        {
            _c4dbgp("stopping implicitly nested 1x map");
            _stop_seqimap();
            _pop_level();
        }
        return true;
    }

    if(has_any(RNXT))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RKEY));
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RVAL));
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RSEQIMAP));

        if(rem.begins_with(", "))
        {
            _c4dbgp("seq: expect next keyval");
            addrem_flags(RKEY, RNXT);
            _line_progressed(2);
            return true;
        }
        else if(rem.begins_with(','))
        {
            _c4dbgp("seq: expect next keyval");
            addrem_flags(RKEY, RNXT);
            _line_progressed(1);
            return true;
        }
        else
        {
            _c4err("parse error");
        }
    }
    else if(has_any(RKEY))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RNXT));
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RVAL));

        bool is_quoted;
        if(has_none(SSCL) && _scan_scalar_map_flow(&rem, &is_quoted))
        {
            _c4dbgp("it's a scalar");
            _store_scalar(rem, is_quoted);
            rem = m_state->line_contents.rem;
            csubstr trimmed = rem.triml(" \t");
            if(trimmed.len && (trimmed.begins_with(": ") || trimmed.begins_with_any(":,}") _RYML_WITH_TAB_TOKENS( || rem.begins_with(":\t"))))
            {
                _RYML_CB_ASSERT(m_stack.m_callbacks, trimmed.str >= rem.str);
                size_t num = static_cast<size_t>(trimmed.str - rem.str);
                _c4dbgpf("trimming {} whitespace after the scalar: '{}' --> '{}'", num, rem, rem.sub(num));
                rem = rem.sub(num);
                _line_progressed(num);
            }
        }

        if(rem.begins_with(": ") _RYML_WITH_TAB_TOKENS( || rem.begins_with(":\t")))
        {
            _c4dbgp("wait for val");
            addrem_flags(RVAL, RKEY|QMRK);
            _line_progressed(2);
            if(!has_all(SSCL))
            {
                _c4dbgp("no key was found, defaulting to empty key ''");
                _store_scalar_null(rem.str);
            }
            return true;
        }
        else if(rem == ':')
        {
            _c4dbgp("wait for val");
            addrem_flags(RVAL, RKEY|QMRK);
            _line_progressed(1);
            if(!has_all(SSCL))
            {
                _c4dbgp("no key was found, defaulting to empty key ''");
                _store_scalar_null(rem.str);
            }
            return true;
        }
        else if(rem.begins_with('?'))
        {
            _c4dbgp("complex key");
            add_flags(QMRK);
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with(','))
        {
            _c4dbgp("prev scalar was a key with null value");
            _append_key_val_null(rem.str - 1);
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with('}'))
        {
            _c4dbgp("map terminates after a key...");
            _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(SSCL));
            _c4dbgp("the last val was null");
            _append_key_val_null(rem.str - 1);
            rem_flags(RVAL);
            if(has_all(RSEQIMAP))
            {
                _c4dbgp("stopping implicitly nested 1x map");
                _stop_seqimap();
                _pop_level();
            }
            _pop_level();
            _line_progressed(1);
            return true;
        }
        else if(_handle_types())
        {
            return true;
        }
        else if(_handle_key_anchors_and_refs())
        {
            return true;
        }
        else if(rem == "")
        {
            return true;
        }
        else
        {
            size_t pos = rem.first_not_of(" \t");
            if(pos == csubstr::npos)
               pos = 0;
            rem = rem.sub(pos);
            if(rem.begins_with(':'))
            {
                _c4dbgp("wait for val");
                addrem_flags(RVAL, RKEY|QMRK);
                _line_progressed(pos + 1);
                if(!has_all(SSCL))
                {
                    _c4dbgp("no key was found, defaulting to empty key ''");
                    _store_scalar_null(rem.str);
                }
                return true;
            }
            else if(rem.begins_with('#'))
            {
                _c4dbgp("it's a comment");
                _line_progressed(pos);
                rem = _scan_comment(); // also progresses the line
                return true;
            }
            else
            {
                _c4err("parse error");
            }
        }
    }
    else if(has_any(RVAL))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RNXT));
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RKEY));
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(SSCL));
        bool is_quoted;
        if(_scan_scalar_map_flow(&rem, &is_quoted))
        {
            _c4dbgp("it's a scalar");
            addrem_flags(RNXT, RVAL|RKEY);
            _append_key_val(rem, is_quoted);
            if(has_all(RSEQIMAP))
            {
                _c4dbgp("stopping implicitly nested 1x map");
                _stop_seqimap();
                _pop_level();
            }
            return true;
        }
        else if(rem.begins_with('['))
        {
            _c4dbgp("val is a child seq");
            addrem_flags(RNXT, RVAL|RKEY); // before _push_level!
            _push_level(/*explicit flow*/true);
            _move_scalar_from_top();
            _start_seq();
            add_flags(FLOW);
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with('{'))
        {
            _c4dbgp("val is a child map");
            addrem_flags(RNXT, RVAL|RKEY); // before _push_level!
            _push_level(/*explicit flow*/true);
            _move_scalar_from_top();
            _start_map();
            addrem_flags(FLOW|RKEY, RNXT|RVAL);
            _line_progressed(1);
            return true;
        }
        else if(_handle_types())
        {
            return true;
        }
        else if(_handle_val_anchors_and_refs())
        {
            return true;
        }
        else if(rem.begins_with(','))
        {
            _c4dbgp("appending empty val");
            _append_key_val_null(rem.str - 1);
            addrem_flags(RKEY, RVAL);
            _line_progressed(1);
            if(has_any(RSEQIMAP))
            {
                _c4dbgp("stopping implicitly nested 1x map");
                _stop_seqimap();
                _pop_level();
            }
            return true;
        }
        else if(has_any(RSEQIMAP) && rem.begins_with(']'))
        {
            _c4dbgp("stopping implicitly nested 1x map");
            if(has_any(SSCL))
            {
                _append_key_val_null(rem.str - 1);
            }
            _stop_seqimap();
            _pop_level();
            return true;
        }
        else
        {
            _c4err("parse error");
        }
    }
    else
    {
        _c4err("internal error");
    }

    return false;
}

//-----------------------------------------------------------------------------
bool Parser::_handle_map_blck()
{
    _c4dbgpf("handle_map_blck: node_id={}  level={}", m_state->node_id, m_state->level);
    csubstr rem = m_state->line_contents.rem;

    _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(RMAP));
    _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(FLOW));

    if(rem.begins_with('#'))
    {
        _c4dbgp("it's a comment");
        rem = _scan_comment();
        return true;
    }

    if(has_any(RNXT))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RKEY));
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RVAL));
        // actually, we don't need RNXT in indent-based maps.
        addrem_flags(RKEY, RNXT);
    }

    if(_handle_indentation())
    {
        _c4dbgp("indentation token");
        return true;
    }

    if(has_any(RKEY))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RNXT));
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RVAL));

        _c4dbgp("RMAP|RKEY read scalar?");
        bool is_quoted;
        if(_scan_scalar_map_blck(&rem, &is_quoted)) // this also progresses the line
        {
            _c4dbgpf("it's a{} scalar", is_quoted ? " quoted" : "");
            if(has_all(QMRK|SSCL))
            {
                _c4dbgpf("current key is QMRK; SSCL is set. so take store scalar='{}' as key and add an empty val", m_state->scalar);
                _append_key_val_null(rem.str - 1);
            }
            _store_scalar(rem, is_quoted);
            if(has_all(QMRK|RSET))
            {
                _c4dbgp("it's a complex key, so use null value '~'");
                _append_key_val_null(rem.str);
            }
            rem = m_state->line_contents.rem;

            if(rem.begins_with(':'))
            {
                _c4dbgp("wait for val");
                addrem_flags(RVAL, RKEY|QMRK);
                _line_progressed(1);
                rem = m_state->line_contents.rem;
                if(rem.begins_with_any(" \t"))
                {
                    _RYML_CB_ASSERT(m_stack.m_callbacks,  ! _at_line_begin());
                    rem = rem.left_of(rem.first_not_of(" \t"));
                    _c4dbgpf("skip {} spaces/tabs", rem.len);
                    _line_progressed(rem.len);
                }
            }
            return true;
        }
        else if(rem.begins_with_any(" \t"))
        {
            size_t pos = rem.first_not_of(" \t");
            if(pos == npos)
                pos = rem.len;
            _c4dbgpf("skip {} spaces/tabs", pos);
            _line_progressed(pos);
            return true;
        }
        else if(rem == '?' || rem.begins_with("? "))
        {
            _c4dbgp("it's a complex key");
            _line_progressed(rem.begins_with("? ") ? 2u : 1u);
            if(has_any(SSCL))
                _append_key_val_null(rem.str - 1);
            add_flags(QMRK);
            return true;
        }
        else if(has_all(QMRK) && rem.begins_with(':'))
        {
            _c4dbgp("complex key finished");
            if(!has_any(SSCL))
                _store_scalar_null(rem.str);
            addrem_flags(RVAL, RKEY|QMRK);
            _line_progressed(1);
            rem = m_state->line_contents.rem;
            if(rem.begins_with(' '))
            {
                _RYML_CB_ASSERT(m_stack.m_callbacks,  ! _at_line_begin());
                _skipchars(' ');
            }
            return true;
        }
        else if(rem == ':' || rem.begins_with(": ") _RYML_WITH_TAB_TOKENS( || rem.begins_with(":\t")))
        {
            _c4dbgp("key finished");
            if(!has_all(SSCL))
            {
                _c4dbgp("key was empty...");
                _store_scalar_null(rem.str);
                rem_flags(QMRK);
            }
            addrem_flags(RVAL, RKEY);
            _line_progressed(rem == ':' ? 1 : 2);
            return true;
        }
        else if(rem.begins_with("..."))
        {
            _c4dbgp("end current document");
            _end_stream();
            _line_progressed(3);
            return true;
        }
        else if(rem.begins_with("---"))
        {
            _c4dbgp("start new document '---'");
            _start_new_doc(rem);
            return true;
        }
        else if(_handle_types())
        {
            return true;
        }
        else if(_handle_key_anchors_and_refs())
        {
            return true;
        }
        else
        {
            _c4err("parse error");
        }
    }
    else if(has_any(RVAL))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RNXT));
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RKEY));

        _c4dbgp("RMAP|RVAL read scalar?");
        csubstr s;
        bool is_quoted;
        if(_scan_scalar_map_blck(&s, &is_quoted)) // this also progresses the line
        {
            _c4dbgpf("it's a{} scalar", is_quoted ? " quoted" : "");

            rem = m_state->line_contents.rem;

            if(rem.begins_with(": "))
            {
                _c4dbgp("actually, the scalar is the first key of a map");
                addrem_flags(RKEY, RVAL); // before _push_level! This prepares the current level for popping by setting it to RNXT
                _push_level();
                _move_scalar_from_top();
                _move_val_anchor_to_key_anchor();
                _start_map();
                _save_indentation(m_state->scalar_col);
                addrem_flags(RVAL, RKEY);
                _line_progressed(2);
            }
            else if(rem.begins_with(':'))
            {
                _c4dbgp("actually, the scalar is the first key of a map, and it opens a new scope");
                addrem_flags(RKEY, RVAL); // before _push_level! This prepares the current level for popping by setting it to RNXT
                _push_level();
                _move_scalar_from_top();
                _move_val_anchor_to_key_anchor();
                _start_map();
                _save_indentation(/*behind*/s.len);
                addrem_flags(RVAL, RKEY);
                _line_progressed(1);
            }
            else
            {
                _c4dbgp("appending keyval to current map");
                _append_key_val(s, is_quoted);
                addrem_flags(RKEY, RVAL);
            }
            return true;
        }
        else if(rem.begins_with("- ") _RYML_WITH_TAB_TOKENS( || rem.begins_with("-\t")))
        {
            _c4dbgp("val is a nested seq, indented");
            addrem_flags(RKEY, RVAL); // before _push_level!
            _push_level();
            _move_scalar_from_top();
            _start_seq();
            _save_indentation();
            _line_progressed(2);
            return true;
        }
        else if(rem == '-')
        {
            _c4dbgp("maybe a seq. start unknown, indented");
            _start_unk();
            _save_indentation();
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with('['))
        {
            _c4dbgp("val is a child seq, flow");
            addrem_flags(RKEY, RVAL); // before _push_level!
            _push_level(/*explicit flow*/true);
            _move_scalar_from_top();
            _start_seq();
            add_flags(FLOW);
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with('{'))
        {
            _c4dbgp("val is a child map, flow");
            addrem_flags(RKEY, RVAL); // before _push_level!
            _push_level(/*explicit flow*/true);
            _move_scalar_from_top();
            _start_map();
            addrem_flags(FLOW|RKEY, RVAL);
            _line_progressed(1);
            return true;
        }
        else if(rem.begins_with(' '))
        {
            csubstr spc = rem.left_of(rem.first_not_of(' '));
            if(_at_line_begin())
            {
                _c4dbgpf("skipping value indentation: {} spaces", spc.len);
                _line_progressed(spc.len);
                return true;
            }
            else
            {
                _c4dbgpf("skipping {} spaces", spc.len);
                _line_progressed(spc.len);
                return true;
            }
        }
        else if(_handle_types())
        {
            return true;
        }
        else if(_handle_val_anchors_and_refs())
        {
            return true;
        }
        else if(rem.begins_with("--- ") || rem == "---" || rem.begins_with("---\t"))
        {
            _start_new_doc(rem);
            return true;
        }
        else if(rem.begins_with("..."))
        {
            _c4dbgp("end current document");
            _end_stream();
            _line_progressed(3);
            return true;
        }
        else
        {
            _c4err("parse error");
        }
    }
    else
    {
        _c4err("internal error");
    }

    return false;
}


//-----------------------------------------------------------------------------
bool Parser::_handle_top()
{
    _c4dbgp("handle_top");
    csubstr rem = m_state->line_contents.rem;

    if(rem.begins_with('#'))
    {
        _c4dbgp("a comment line");
        _scan_comment();
        return true;
    }

    csubstr trimmed = rem.triml(' ');

    if(trimmed.begins_with('%'))
    {
        _handle_directive(trimmed);
        _line_progressed(rem.len);
        return true;
    }
    else if(trimmed.begins_with("--- ") || trimmed == "---" || trimmed.begins_with("---\t"))
    {
        _start_new_doc(rem);
        if(trimmed.len < rem.len)
        {
            _line_progressed(rem.len - trimmed.len);
            _save_indentation();
        }
        return true;
    }
    else if(trimmed.begins_with("..."))
    {
        _c4dbgp("end current document");
        _end_stream();
        if(trimmed.len < rem.len)
        {
            _line_progressed(rem.len - trimmed.len);
        }
        _line_progressed(3);
        return true;
    }
    else
    {
        _c4err("parse error");
    }

    return false;
}


//-----------------------------------------------------------------------------

bool Parser::_handle_key_anchors_and_refs()
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, !has_any(RVAL));
    const csubstr rem = m_state->line_contents.rem;
    if(rem.begins_with('&'))
    {
        _c4dbgp("found a key anchor!!!");
        if(has_all(QMRK|SSCL))
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks, has_any(RKEY));
            _c4dbgp("there is a stored key, so this anchor is for the next element");
            _append_key_val_null(rem.str - 1);
            rem_flags(QMRK);
            return true;
        }
        csubstr anchor = rem.left_of(rem.first_of(' '));
        _line_progressed(anchor.len);
        anchor = anchor.sub(1); // skip the first character
        _move_key_anchor_to_val_anchor();
        _c4dbgpf("key anchor value: '{}'", anchor);
        m_key_anchor = anchor;
        m_key_anchor_indentation = m_state->line_contents.current_col(rem);
        return true;
    }
    else if(C4_UNLIKELY(rem.begins_with('*')))
    {
        _c4err("not implemented - this should have been catched elsewhere");
        C4_NEVER_REACH();
        return false;
    }
    return false;
}

bool Parser::_handle_val_anchors_and_refs()
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, !has_any(RKEY));
    const csubstr rem = m_state->line_contents.rem;
    if(rem.begins_with('&'))
    {
        csubstr anchor = rem.left_of(rem.first_of(' '));
        _line_progressed(anchor.len);
        anchor = anchor.sub(1); // skip the first character
        _c4dbgpf("val: found an anchor: '{}', indentation={}!!!", anchor, m_state->line_contents.current_col(rem));
        if(m_val_anchor.empty())
        {
            _c4dbgpf("save val anchor: '{}'", anchor);
            m_val_anchor = anchor;
            m_val_anchor_indentation = m_state->line_contents.current_col(rem);
        }
        else
        {
            _c4dbgpf("there is a pending val anchor '{}'", m_val_anchor);
            if(m_tree->is_seq(m_state->node_id))
            {
                if(m_tree->has_children(m_state->node_id))
                {
                    _c4dbgpf("current node={} is a seq, has {} children", m_state->node_id, m_tree->num_children(m_state->node_id));
                    _c4dbgpf("... so take the new one as a key anchor '{}'", anchor);
                    m_key_anchor = anchor;
                    m_key_anchor_indentation = m_state->line_contents.current_col(rem);
                }
                else
                {
                    _c4dbgpf("current node={} is a seq, has no children", m_state->node_id);
                    if(m_tree->has_val_anchor(m_state->node_id))
                    {
                        _c4dbgpf("... node={} already has val anchor: '{}'", m_state->node_id, m_tree->val_anchor(m_state->node_id));
                        _c4dbgpf("... so take the new one as a key anchor '{}'", anchor);
                        m_key_anchor = anchor;
                        m_key_anchor_indentation = m_state->line_contents.current_col(rem);
                    }
                    else
                    {
                        _c4dbgpf("... so set pending val anchor: '{}' on current node {}", m_val_anchor, m_state->node_id);
                        m_tree->set_val_anchor(m_state->node_id, m_val_anchor);
                        m_val_anchor = anchor;
                        m_val_anchor_indentation = m_state->line_contents.current_col(rem);
                    }
                }
            }
        }
        return true;
    }
    else if(C4_UNLIKELY(rem.begins_with('*')))
    {
        _c4err("not implemented - this should have been catched elsewhere");
        C4_NEVER_REACH();
        return false;
    }
    return false;
}

void Parser::_move_key_anchor_to_val_anchor()
{
    if(m_key_anchor.empty())
        return;
    _c4dbgpf("move current key anchor to val slot: key='{}' -> val='{}'", m_key_anchor, m_val_anchor);
    if(!m_val_anchor.empty())
        _c4err("triple-pending anchor");
    m_val_anchor = m_key_anchor;
    m_val_anchor_indentation = m_key_anchor_indentation;
    m_key_anchor = {};
    m_key_anchor_indentation = {};
}

void Parser::_move_val_anchor_to_key_anchor()
{
    if(m_val_anchor.empty())
        return;
    if(!_token_is_from_this_line(m_val_anchor))
        return;
    _c4dbgpf("move current val anchor to key slot: key='{}' <- val='{}'", m_key_anchor, m_val_anchor);
    if(!m_key_anchor.empty())
        _c4err("triple-pending anchor");
    m_key_anchor = m_val_anchor;
    m_key_anchor_indentation = m_val_anchor_indentation;
    m_val_anchor = {};
    m_val_anchor_indentation = {};
}

void Parser::_move_key_tag_to_val_tag()
{
    if(m_key_tag.empty())
        return;
    _c4dbgpf("move key tag to val tag: key='{}' -> val='{}'", m_key_tag, m_val_tag);
    m_val_tag = m_key_tag;
    m_val_tag_indentation = m_key_tag_indentation;
    m_key_tag.clear();
    m_key_tag_indentation = 0;
}

void Parser::_move_val_tag_to_key_tag()
{
    if(m_val_tag.empty())
        return;
    if(!_token_is_from_this_line(m_val_tag))
        return;
    _c4dbgpf("move val tag to key tag: key='{}' <- val='{}'", m_key_tag, m_val_tag);
    m_key_tag = m_val_tag;
    m_key_tag_indentation = m_val_tag_indentation;
    m_val_tag.clear();
    m_val_tag_indentation = 0;
}

void Parser::_move_key_tag2_to_key_tag()
{
    if(m_key_tag2.empty())
        return;
    _c4dbgpf("move key tag2 to key tag: key='{}' <- key2='{}'", m_key_tag, m_key_tag2);
    m_key_tag = m_key_tag2;
    m_key_tag_indentation = m_key_tag2_indentation;
    m_key_tag2.clear();
    m_key_tag2_indentation = 0;
}


//-----------------------------------------------------------------------------

bool Parser::_handle_types()
{
    csubstr rem = m_state->line_contents.rem.triml(' ');
    csubstr t;

    if(rem.begins_with("!!"))
    {
        _c4dbgp("begins with '!!'");
        t = rem.left_of(rem.first_of(" ,"));
        _RYML_CB_ASSERT(m_stack.m_callbacks, t.len >= 2);
        //t = t.sub(2);
        if(t == "!!set")
            add_flags(RSET);
    }
    else if(rem.begins_with("!<"))
    {
        _c4dbgp("begins with '!<'");
        t = rem.left_of(rem.first_of('>'), true);
        _RYML_CB_ASSERT(m_stack.m_callbacks, t.len >= 2);
        //t = t.sub(2, t.len-1);
    }
    else if(rem.begins_with("!h!"))
    {
        _c4dbgp("begins with '!h!'");
        t = rem.left_of(rem.first_of(' '));
        _RYML_CB_ASSERT(m_stack.m_callbacks, t.len >= 3);
        //t = t.sub(3);
    }
    else if(rem.begins_with('!'))
    {
        _c4dbgp("begins with '!'");
        t = rem.left_of(rem.first_of(' '));
        _RYML_CB_ASSERT(m_stack.m_callbacks, t.len >= 1);
        //t = t.sub(1);
    }

    if(t.empty())
        return false;

    if(has_all(QMRK|SSCL))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_any(RKEY));
        _c4dbgp("there is a stored key, so this tag is for the next element");
        _append_key_val_null(rem.str - 1);
        rem_flags(QMRK);
    }

    #ifdef RYML_NO_COVERAGE__TO_BE_DELETED
    const char *tag_beginning = rem.str;
    #endif
    size_t tag_indentation = m_state->line_contents.current_col(t);
    _c4dbgpf("there was a tag: '{}', indentation={}", t, tag_indentation);
    _RYML_CB_ASSERT(m_stack.m_callbacks, t.end() > m_state->line_contents.rem.begin());
    _line_progressed(static_cast<size_t>(t.end() - m_state->line_contents.rem.begin()));
    {
        size_t pos = m_state->line_contents.rem.first_not_of(" \t");
        if(pos != csubstr::npos)
            _line_progressed(pos);
    }

    if(has_all(RMAP|RKEY))
    {
        _c4dbgpf("saving map key tag '{}'", t);
        _RYML_CB_ASSERT(m_stack.m_callbacks, m_key_tag.empty());
        m_key_tag = t;
        m_key_tag_indentation = tag_indentation;
    }
    else if(has_all(RMAP|RVAL))
    {
        /* foo: !!str
         * !!str : bar  */
        rem = m_state->line_contents.rem;
        rem = rem.left_of(rem.find("#"));
        rem = rem.trimr(" \t");
        _c4dbgpf("rem='{}'", rem);
        #ifdef RYML_NO_COVERAGE__TO_BE_DELETED
        if(rem == ':' || rem.begins_with(": "))
        {
            _c4dbgp("the last val was null, and this is a tag from a null key");
            _append_key_val_null(tag_beginning - 1);
            _store_scalar_null(rem.str - 1);
            // do not change the flag to key, it is ~
            _RYML_CB_ASSERT(m_stack.m_callbacks, rem.begin() > m_state->line_contents.rem.begin());
            size_t token_len = rem == ':' ? 1 : 2;
            _line_progressed(static_cast<size_t>(token_len + rem.begin() - m_state->line_contents.rem.begin()));
        }
        #endif
        _c4dbgpf("saving map val tag '{}'", t);
        _RYML_CB_ASSERT(m_stack.m_callbacks, m_val_tag.empty());
        m_val_tag = t;
        m_val_tag_indentation = tag_indentation;
    }
    else if(has_all(RSEQ|RVAL) || has_all(RTOP|RUNK|NDOC))
    {
        if(m_val_tag.empty())
        {
            _c4dbgpf("saving seq/doc val tag '{}'", t);
            m_val_tag = t;
            m_val_tag_indentation = tag_indentation;
        }
        else
        {
            _c4dbgpf("saving seq/doc key tag '{}'", t);
            m_key_tag = t;
            m_key_tag_indentation = tag_indentation;
        }
    }
    else if(has_all(RTOP|RUNK) || has_any(RUNK))
    {
        rem = m_state->line_contents.rem;
        rem = rem.left_of(rem.find("#"));
        rem = rem.trimr(" \t");
        if(rem.empty())
        {
            _c4dbgpf("saving val tag '{}'", t);
            _RYML_CB_ASSERT(m_stack.m_callbacks, m_val_tag.empty());
            m_val_tag = t;
            m_val_tag_indentation = tag_indentation;
        }
        else
        {
            _c4dbgpf("saving key tag '{}'", t);
            if(m_key_tag.empty())
            {
                m_key_tag = t;
                m_key_tag_indentation = tag_indentation;
            }
            else
            {
                /* handle this case:
                 * !!str foo: !!map
                 *   !!int 1: !!float 20.0
                 *   !!int 3: !!float 40.0
                 *
                 * (m_key_tag would be !!str and m_key_tag2 would be !!int)
                 */
                m_key_tag2 = t;
                m_key_tag2_indentation = tag_indentation;
            }
        }
    }
    else
    {
        _c4err("internal error");
    }

    if(m_val_tag.not_empty())
    {
        YamlTag_e tag = to_tag(t);
        if(tag == TAG_STR)
        {
            _c4dbgpf("tag '{}' is a str-type tag", t);
            if(has_all(RTOP|RUNK|NDOC))
            {
                _c4dbgpf("docval. slurping the string. pos={}", m_state->pos.offset);
                csubstr scalar = _slurp_doc_scalar();
                _c4dbgpf("docval. after slurp: {}, at node {}: '{}'", m_state->pos.offset, m_state->node_id, scalar);
                m_tree->to_val(m_state->node_id, scalar, DOC);
                _c4dbgpf("docval. val tag {} -> {}", m_val_tag, normalize_tag(m_val_tag));
                m_tree->set_val_tag(m_state->node_id, normalize_tag(m_val_tag));
                m_val_tag.clear();
                if(!m_val_anchor.empty())
                {
                    _c4dbgpf("setting val anchor[{}]='{}'", m_state->node_id, m_val_anchor);
                    m_tree->set_val_anchor(m_state->node_id, m_val_anchor);
                    m_val_anchor.clear();
                }
                _end_stream();
            }
        }
    }
    return true;
}

//-----------------------------------------------------------------------------
csubstr Parser::_slurp_doc_scalar()
{
    csubstr s = m_state->line_contents.rem;
    size_t pos = m_state->pos.offset;
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->line_contents.full.find("---") != csubstr::npos);
    _c4dbgpf("slurp 0 '{}'. REM='{}'", s, m_buf.sub(m_state->pos.offset));
    if(s.len == 0)
    {
        _line_ended();
        _scan_line();
        s = m_state->line_contents.rem;
        pos = m_state->pos.offset;
    }

    size_t skipws = s.first_not_of(" \t");
    _c4dbgpf("slurp 1 '{}'. REM='{}'", s, m_buf.sub(m_state->pos.offset));
    if(skipws != npos)
    {
        _line_progressed(skipws);
        s = m_state->line_contents.rem;
        pos = m_state->pos.offset;
        _c4dbgpf("slurp 2 '{}'. REM='{}'", s, m_buf.sub(m_state->pos.offset));
    }

    _RYML_CB_ASSERT(m_stack.m_callbacks, m_val_anchor.empty());
    _handle_val_anchors_and_refs();
    if(!m_val_anchor.empty())
    {
        s = m_state->line_contents.rem;
        skipws = s.first_not_of(" \t");
        if(skipws != npos)
        {
            _line_progressed(skipws);
        }
        s = m_state->line_contents.rem;
        pos = m_state->pos.offset;
        _c4dbgpf("slurp 3 '{}'. REM='{}'", s, m_buf.sub(m_state->pos.offset));
    }

    if(s.begins_with('\''))
    {
        m_state->scalar_col = m_state->line_contents.current_col(s);
        return _scan_squot_scalar();
    }
    else if(s.begins_with('"'))
    {
        m_state->scalar_col = m_state->line_contents.current_col(s);
        return _scan_dquot_scalar();
    }
    else if(s.begins_with('|') || s.begins_with('>'))
    {
        return _scan_block();
    }

    _c4dbgpf("slurp 4 '{}'. REM='{}'", s, m_buf.sub(m_state->pos.offset));

    m_state->scalar_col = m_state->line_contents.current_col(s);
    _RYML_CB_ASSERT(m_stack.m_callbacks, s.end() >= m_buf.begin() + pos);
    _line_progressed(static_cast<size_t>(s.end() - (m_buf.begin() + pos)));

    _c4dbgpf("slurp 5 '{}'. REM='{}'", s, m_buf.sub(m_state->pos.offset));

    if(_at_line_end())
    {
        _c4dbgpf("at line end. curr='{}'", s);
        s = _extend_scanned_scalar(s);
    }

    _c4dbgpf("scalar was '{}'", s);

    return s;
}


//-----------------------------------------------------------------------------

bool Parser::_scan_scalar_seq_blck(csubstr *C4_RESTRICT scalar, bool *C4_RESTRICT quoted)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks,  has_any(RSEQ));
    _RYML_CB_ASSERT(m_stack.m_callbacks,  has_any(RVAL));
    _RYML_CB_ASSERT(m_stack.m_callbacks,  ! has_any(RKEY));
    _RYML_CB_ASSERT(m_stack.m_callbacks,  ! has_any(FLOW));

    csubstr s = m_state->line_contents.rem;
    if(s.len == 0)
        return false;
    s = s.trim(" \t");
    if(s.len == 0)
        return false;

    if(s.begins_with('\''))
    {
        _c4dbgp("got a ': scanning single-quoted scalar");
        m_state->scalar_col = m_state->line_contents.current_col(s);
        *scalar = _scan_squot_scalar();
        *quoted = true;
        return true;
    }
    else if(s.begins_with('"'))
    {
        _c4dbgp("got a \": scanning double-quoted scalar");
        m_state->scalar_col = m_state->line_contents.current_col(s);
        *scalar = _scan_dquot_scalar();
        *quoted = true;
        return true;
    }
    else if(s.begins_with('|') || s.begins_with('>'))
    {
        *scalar = _scan_block();
        *quoted = true;
        return true;
    }
    else if(has_any(RTOP) && _is_doc_sep(s))
    {
        return false;
    }

    _c4dbgp("RSEQ|RVAL");
    if( ! _is_scalar_next__rseq_rval(s))
        return false;
    _RYML_WITH_TAB_TOKENS(else if(s.begins_with("-\t"))
        return false;
    )

    if(s.ends_with(':'))
    {
        --s.len;
    }
    else
    {
        auto first = s.first_of_any(": " _RYML_WITH_TAB_TOKENS( , ":\t"), " #");
        if(first)
            s.len = first.pos;
    }
    s = s.trimr(_RYML_WITH_OR_WITHOUT_TAB_TOKENS(" \t", ' '));

    if(s.empty())
        return false;

    m_state->scalar_col = m_state->line_contents.current_col(s);
    _RYML_CB_ASSERT(m_stack.m_callbacks, s.str >= m_state->line_contents.rem.str);
    _line_progressed(static_cast<size_t>(s.str - m_state->line_contents.rem.str) + s.len);

    if(_at_line_end() && s != '~')
    {
        _c4dbgpf("at line end. curr='{}'", s);
        s = _extend_scanned_scalar(s);
    }

    _c4dbgpf("scalar was '{}'", s);

    *scalar = s;
    *quoted = false;
    return true;
}

bool Parser::_scan_scalar_map_blck(csubstr *C4_RESTRICT scalar, bool *C4_RESTRICT quoted)
{
    _c4dbgp("_scan_scalar_map_blck");
    _RYML_CB_ASSERT(m_stack.m_callbacks,  has_any(RMAP));
    _RYML_CB_ASSERT(m_stack.m_callbacks,  ! has_any(FLOW));
    _RYML_CB_ASSERT(m_stack.m_callbacks,  has_any(RKEY|RVAL));

    csubstr s = m_state->line_contents.rem;
    #ifdef RYML_NO_COVERAGE__TO_BE_DELETED__OR_REFACTORED
    if(s.len == 0)
        return false;
    #endif
    s = s.trim(" \t");
    if(s.len == 0)
        return false;

    if(s.begins_with('\''))
    {
        _c4dbgp("got a ': scanning single-quoted scalar");
        m_state->scalar_col = m_state->line_contents.current_col(s);
        *scalar = _scan_squot_scalar();
        *quoted = true;
        return true;
    }
    else if(s.begins_with('"'))
    {
        _c4dbgp("got a \": scanning double-quoted scalar");
        m_state->scalar_col = m_state->line_contents.current_col(s);
        *scalar = _scan_dquot_scalar();
        *quoted = true;
        return true;
    }
    else if(s.begins_with('|') || s.begins_with('>'))
    {
        *scalar = _scan_block();
        *quoted = true;
        return true;
    }
    else if(has_any(RTOP) && _is_doc_sep(s))
    {
        return false;
    }

    if( ! _is_scalar_next__rmap(s))
        return false;

    size_t colon_token = s.find(": ");
    if(colon_token == npos)
    {
        _RYML_WITH_OR_WITHOUT_TAB_TOKENS(
            // with tab tokens
            colon_token = s.find(":\t");
            if(colon_token == npos)
            {
                _RYML_CB_ASSERT(m_stack.m_callbacks, s.len > 0);
                colon_token = s.find(':');
                if(colon_token != s.len-1)
                    colon_token = npos;
            }
            ,
            // without tab tokens
            colon_token = s.find(':');
            _RYML_CB_ASSERT(m_stack.m_callbacks, s.len > 0);
            if(colon_token != s.len-1)
                colon_token = npos;
        )
    }

    if(has_all(RKEY))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, !s.begins_with(' '));
        if(has_any(QMRK))
        {
            _c4dbgp("RMAP|RKEY|CPLX");
            _RYML_CB_ASSERT(m_stack.m_callbacks, has_any(RMAP));
            if(s.begins_with("? ") || s == '?')
                return false;
            s = s.left_of(colon_token);
            s = s.left_of(s.first_of("#"));
            s = s.trimr(" \t");
            if(s.begins_with("---"))
                return false;
            else if(s.begins_with("..."))
                return false;
        }
        else
        {
            _c4dbgp("RMAP|RKEY");
            _RYML_CB_CHECK(m_stack.m_callbacks, !s.begins_with('{'));
            if(s.begins_with("? ") || s == '?')
                return false;
            s = s.left_of(colon_token);
            s = s.trimr(_RYML_WITH_OR_WITHOUT_TAB_TOKENS(" \t", ' '));
            if(s.begins_with("---"))
            {
                return false;
            }
            else if(s.begins_with("..."))
            {
                return false;
            }
        }
    }
    else if(has_all(RVAL))
    {
        _c4dbgp("RMAP|RVAL");
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(QMRK));
        if( ! _is_scalar_next__rmap_val(s))
            return false;
        _RYML_WITH_TAB_TOKENS(
        else if(s.begins_with("-\t"))
            return false;
        )
        _c4dbgp("RMAP|RVAL: scalar");
        s = s.left_of(s.find(" #")); // is there a comment?
        s = s.left_of(s.find("\t#")); // is there a comment?
        s = s.trim(_RYML_WITH_OR_WITHOUT_TAB_TOKENS(" \t", ' '));
        if(s.begins_with("---"))
            return false;
        #ifdef RYML_NO_COVERAGE__TO_BE_DELETED__OR_REFACTORED
        else if(s.begins_with("..."))
            return false;
        #endif
    }

    if(s.empty())
        return false;

    m_state->scalar_col = m_state->line_contents.current_col(s);
    _RYML_CB_ASSERT(m_stack.m_callbacks, s.str >= m_state->line_contents.rem.str);
    _line_progressed(static_cast<size_t>(s.str - m_state->line_contents.rem.str) + s.len);

    if(_at_line_end() && s != '~')
    {
        _c4dbgpf("at line end. curr='{}'", s);
        s = _extend_scanned_scalar(s);
    }

    _c4dbgpf("scalar was '{}'", s);

    *scalar = s;
    *quoted = false;
    return true;
}

bool Parser::_scan_scalar_seq_flow(csubstr *C4_RESTRICT scalar, bool *C4_RESTRICT quoted)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks,  has_any(RSEQ));
    _RYML_CB_ASSERT(m_stack.m_callbacks,  has_any(FLOW));
    _RYML_CB_ASSERT(m_stack.m_callbacks,  has_any(RVAL));
    _RYML_CB_ASSERT(m_stack.m_callbacks,  ! has_any(RKEY));

    csubstr s = m_state->line_contents.rem;
    if(s.len == 0)
        return false;
    s = s.trim(" \t");
    if(s.len == 0)
        return false;

    if(s.begins_with('\''))
    {
        _c4dbgp("got a ': scanning single-quoted scalar");
        m_state->scalar_col = m_state->line_contents.current_col(s);
        *scalar = _scan_squot_scalar();
        *quoted = true;
        return true;
    }
    else if(s.begins_with('"'))
    {
        _c4dbgp("got a \": scanning double-quoted scalar");
        m_state->scalar_col = m_state->line_contents.current_col(s);
        *scalar = _scan_dquot_scalar();
        *quoted = true;
        return true;
    }

    if(has_all(RVAL))
    {
        _c4dbgp("RSEQ|RVAL");
        if( ! _is_scalar_next__rseq_rval(s))
            return false;
        _RYML_WITH_TAB_TOKENS(else if(s.begins_with("-\t"))
            return false;
        )
        _c4dbgp("RSEQ|RVAL|FLOW");
        s = s.left_of(s.first_of(",]"));
        if(s.ends_with(':'))
        {
            --s.len;
        }
        else
        {
            auto first = s.first_of_any(": " _RYML_WITH_TAB_TOKENS( , ":\t"), " #");
            if(first)
                s.len = first.pos;
        }
        s = s.trimr(_RYML_WITH_OR_WITHOUT_TAB_TOKENS(" \t", ' '));
    }

    if(s.empty())
        return false;

    m_state->scalar_col = m_state->line_contents.current_col(s);
    _RYML_CB_ASSERT(m_stack.m_callbacks, s.str >= m_state->line_contents.rem.str);
    _line_progressed(static_cast<size_t>(s.str - m_state->line_contents.rem.str) + s.len);

    if(_at_line_end() && s != '~')
    {
        _c4dbgpf("at line end. curr='{}'", s);
        s = _extend_scanned_scalar(s);
    }

    _c4dbgpf("scalar was '{}'", s);

    *scalar = s;
    *quoted = false;
    return true;
}

bool Parser::_scan_scalar_map_flow(csubstr *C4_RESTRICT scalar, bool *C4_RESTRICT quoted)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks,  has_any(RMAP));
    _RYML_CB_ASSERT(m_stack.m_callbacks,  has_any(FLOW));
    _RYML_CB_ASSERT(m_stack.m_callbacks,  has_any(RKEY|RVAL));

    csubstr s = m_state->line_contents.rem;
    if(s.len == 0)
        return false;
    s = s.trim(" \t");
    if(s.len == 0)
        return false;

    if(s.begins_with('\''))
    {
        _c4dbgp("got a ': scanning single-quoted scalar");
        m_state->scalar_col = m_state->line_contents.current_col(s);
        *scalar = _scan_squot_scalar();
        *quoted = true;
        return true;
    }
    else if(s.begins_with('"'))
    {
        _c4dbgp("got a \": scanning double-quoted scalar");
        m_state->scalar_col = m_state->line_contents.current_col(s);
        *scalar = _scan_dquot_scalar();
        *quoted = true;
        return true;
    }

    if( ! _is_scalar_next__rmap(s))
        return false;

    if(has_all(RKEY))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, !s.begins_with(' '));
        size_t colon_token = s.find(": ");
        if(colon_token == npos)
        {
            _RYML_WITH_OR_WITHOUT_TAB_TOKENS(
                // with tab tokens
                colon_token = s.find(":\t");
                if(colon_token == npos)
                {
                    _RYML_CB_ASSERT(m_stack.m_callbacks, s.len > 0);
                    colon_token = s.find(':');
                    if(colon_token != s.len-1)
                        colon_token = npos;
                }
                ,
                // without tab tokens
                colon_token = s.find(':');
                _RYML_CB_ASSERT(m_stack.m_callbacks, s.len > 0);
                if(colon_token != s.len-1)
                    colon_token = npos;
            )
        }
        if(s.begins_with("? ") || s == '?')
            return false;
        if(has_any(QMRK))
        {
            _c4dbgp("RMAP|RKEY|CPLX");
            _RYML_CB_ASSERT(m_stack.m_callbacks, has_any(RMAP));
            s = s.left_of(colon_token);
            s = s.left_of(s.first_of("#"));
            s = s.left_of(s.first_of(':'));
            s = s.trimr(" \t");
            if(s.begins_with("---"))
                return false;
            else if(s.begins_with("..."))
                return false;
        }
        else
        {
            _RYML_CB_CHECK(m_stack.m_callbacks, !s.begins_with('{'));
            _c4dbgp("RMAP|RKEY");
            s = s.left_of(colon_token);
            s = s.trimr(_RYML_WITH_OR_WITHOUT_TAB_TOKENS(" \t", ' '));
            _c4dbgpf("RMAP|RKEY|FLOW: '{}'", s);
            s = s.left_of(s.first_of(",}"));
            if(s.ends_with(':'))
                --s.len;
        }
    }
    else if(has_all(RVAL))
    {
        _c4dbgp("RMAP|RVAL");
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(QMRK));
        if( ! _is_scalar_next__rmap_val(s))
            return false;
        _RYML_WITH_TAB_TOKENS(else if(s.begins_with("-\t"))
            return false;
        )
        _c4dbgp("RMAP|RVAL|FLOW");
        if(has_none(RSEQIMAP))
            s = s.left_of(s.first_of(",}"));
        else
            s = s.left_of(s.first_of(",]"));
        s = s.left_of(s.find(" #")); // is there a comment?
        s = s.left_of(s.find("\t#")); // is there a comment?
        s = s.trim(_RYML_WITH_OR_WITHOUT_TAB_TOKENS(" \t", ' '));
    }

    if(s.empty())
        return false;

    m_state->scalar_col = m_state->line_contents.current_col(s);
    _RYML_CB_ASSERT(m_stack.m_callbacks, s.str >= m_state->line_contents.rem.str);
    _line_progressed(static_cast<size_t>(s.str - m_state->line_contents.rem.str) + s.len);

    if(_at_line_end() && s != '~')
    {
        _c4dbgpf("at line end. curr='{}'", s);
        s = _extend_scanned_scalar(s);
    }

    _c4dbgpf("scalar was '{}'", s);

    *scalar = s;
    *quoted = false;
    return true;
}

bool Parser::_scan_scalar_unk(csubstr *C4_RESTRICT scalar, bool *C4_RESTRICT quoted)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks,  has_any(RUNK));

    csubstr s = m_state->line_contents.rem;
    if(s.len == 0)
        return false;
    s = s.trim(" \t");
    if(s.len == 0)
        return false;

    if(s.begins_with('\''))
    {
        _c4dbgp("got a ': scanning single-quoted scalar");
        m_state->scalar_col = m_state->line_contents.current_col(s);
        *scalar = _scan_squot_scalar();
        *quoted = true;
        return true;
    }
    else if(s.begins_with('"'))
    {
        _c4dbgp("got a \": scanning double-quoted scalar");
        m_state->scalar_col = m_state->line_contents.current_col(s);
        *scalar = _scan_dquot_scalar();
        *quoted = true;
        return true;
    }
    else if(s.begins_with('|') || s.begins_with('>'))
    {
        *scalar = _scan_block();
        *quoted = true;
        return true;
    }
    else if(has_any(RTOP) && _is_doc_sep(s))
    {
        return false;
    }

    _c4dbgpf("RUNK '[{}]~~~{}~~~", s.len, s);
    if( ! _is_scalar_next__runk(s))
    {
        _c4dbgp("RUNK: no scalar next");
        return false;
    }
    size_t pos = s.find(" #");
    if(pos != npos)
    {
        _c4dbgpf("RUNK: found ' #' at {}", pos);
        s = s.left_of(pos);
    }
    pos = s.find(": ");
    if(pos != npos)
    {
        _c4dbgpf("RUNK: found ': ' at {}", pos);
        s = s.left_of(pos);
    }
    else if(s.ends_with(':'))
    {
        _c4dbgp("RUNK: ends with ':'");
        s = s.left_of(s.len-1);
    }
    _RYML_WITH_TAB_TOKENS(
    else if((pos = s.find(":\t")) != npos) // TABS
    {
        _c4dbgp("RUNK: ends with ':\\t'");
        s = s.left_of(pos);
    })
    else
    {
        _c4dbgp("RUNK: trimming left of ,");
        s = s.left_of(s.first_of(','));
    }
    s = s.trim(" \t");
    _c4dbgpf("RUNK: scalar=[{}]~~~{}~~~", s.len, s);

    if(s.empty())
        return false;

    m_state->scalar_col = m_state->line_contents.current_col(s);
    _RYML_CB_ASSERT(m_stack.m_callbacks, s.str >= m_state->line_contents.rem.str);
    _line_progressed(static_cast<size_t>(s.str - m_state->line_contents.rem.str) + s.len);

    if(_at_line_end() && s != '~')
    {
        _c4dbgpf("at line end. curr=[{}]~~~{}~~", s.len, s);
        s = _extend_scanned_scalar(s);
    }

    _c4dbgpf("scalar was [{}]~~~{}~~~", s.len, s);

    *scalar = s;
    *quoted = false;
    return true;
}


//-----------------------------------------------------------------------------

csubstr Parser::_extend_scanned_scalar(csubstr s)
{
    if(has_all(RMAP|RKEY|QMRK))
    {
        size_t scalar_indentation = has_any(FLOW) ? 0 : m_state->scalar_col;
        _c4dbgpf("extend_scalar: explicit key! indref={} scalar_indentation={} scalar_col={}", m_state->indref, scalar_indentation, m_state->scalar_col);
        csubstr n = _scan_to_next_nonempty_line(scalar_indentation);
        if(!n.empty())
        {
            substr full = _scan_complex_key(s, n).trimr(" \t\r\n");
            if(full != s)
                s = _filter_plain_scalar(full, scalar_indentation);
        }
    }
    // deal with plain (unquoted) scalars that continue to the next line
    else if(!s.begins_with_any("*")) // cannot be a plain scalar if it starts with * (that's an anchor reference)
    {
        _c4dbgpf("extend_scalar: line ended, scalar='{}'", s);
        if(has_none(FLOW))
        {
            size_t scalar_indentation = m_state->indref + 1;
            if(has_all(RUNK) && scalar_indentation == 1)
                scalar_indentation = 0;
            csubstr n = _scan_to_next_nonempty_line(scalar_indentation);
            if(!n.empty())
            {
                _c4dbgpf("rscalar[IMPL]: state_indref={} state_indentation={} scalar_indentation={}", m_state->indref, m_state->line_contents.indentation, scalar_indentation);
                _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->line_contents.full.is_super(n));
                substr full = _scan_plain_scalar_blck(s, n, scalar_indentation);
                if(full.len >= s.len)
                    s = _filter_plain_scalar(full, scalar_indentation);
            }
        }
        else
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(FLOW));
            csubstr n = _scan_to_next_nonempty_line(/*indentation*/0);
            if(!n.empty())
            {
                _c4dbgp("rscalar[FLOW]");
                substr full = _scan_plain_scalar_flow(s, n);
                s = _filter_plain_scalar(full, /*indentation*/0);
            }
        }
    }

    return s;
}


//-----------------------------------------------------------------------------

substr Parser::_scan_plain_scalar_flow(csubstr currscalar, csubstr peeked_line)
{
    static constexpr const csubstr chars = "[]{}?#,";
    size_t pos = peeked_line.first_of(chars);
    bool first = true;
    while(pos != 0)
    {
        if(has_all(RMAP|RKEY) || has_any(RUNK))
        {
            csubstr tpkl = peeked_line.triml(' ').trimr("\r\n");
            if(tpkl.begins_with(": ") || tpkl == ':')
            {
                _c4dbgpf("rscalar[FLOW]: map value starts on the peeked line: '{}'", peeked_line);
                peeked_line = peeked_line.first(0);
                break;
            }
            else
            {
                auto colon_pos = peeked_line.first_of_any(": ", ":");
                if(colon_pos && colon_pos.pos < pos)
                {
                    peeked_line = peeked_line.first(colon_pos.pos);
                    _c4dbgpf("rscalar[FLOW]: found colon at {}. peeked='{}'", colon_pos.pos, peeked_line);
                    _RYML_CB_ASSERT(m_stack.m_callbacks, peeked_line.end() >= m_state->line_contents.rem.begin());
                    _line_progressed(static_cast<size_t>(peeked_line.end() - m_state->line_contents.rem.begin()));
                    break;
                }
            }
        }
        if(pos != npos)
        {
            _c4dbgpf("rscalar[FLOW]: found special character '{}' at {}, stopping: '{}'", peeked_line[pos], pos, peeked_line.left_of(pos).trimr("\r\n"));
            peeked_line = peeked_line.left_of(pos);
            _RYML_CB_ASSERT(m_stack.m_callbacks, peeked_line.end() >= m_state->line_contents.rem.begin());
            _line_progressed(static_cast<size_t>(peeked_line.end() - m_state->line_contents.rem.begin()));
            break;
        }
        _c4dbgpf("rscalar[FLOW]: append another line, full: '{}'", peeked_line.trimr("\r\n"));
        if(!first)
        {
            RYML_CHECK(_advance_to_peeked());
        }
        peeked_line = _scan_to_next_nonempty_line(/*indentation*/0);
        if(peeked_line.empty())
        {
            _c4err("expected token or continuation");
        }
        pos = peeked_line.first_of(chars);
        first = false;
    }
    substr full(m_buf.str + (currscalar.str - m_buf.str), m_buf.begin() + m_state->pos.offset);
    full = full.trimr("\n\r ");
    return full;
}


//-----------------------------------------------------------------------------

substr Parser::_scan_plain_scalar_blck(csubstr currscalar, csubstr peeked_line, size_t indentation)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_buf.is_super(currscalar));
    // NOTE. there's a problem with _scan_to_next_nonempty_line(), as it counts newlines twice
    // size_t offs = m_state->pos.offset;   // so we workaround by directly counting from the end of the given scalar
    _RYML_CB_ASSERT(m_stack.m_callbacks, currscalar.end() >= m_buf.begin());
    size_t offs = static_cast<size_t>(currscalar.end() - m_buf.begin());
    _RYML_CB_ASSERT(m_stack.m_callbacks, peeked_line.begins_with(' ', indentation));
    while(true)
    {
        _c4dbgpf("rscalar[IMPL]: continuing... ref_indentation={}", indentation);
        if(peeked_line.begins_with("...") || peeked_line.begins_with("---"))
        {
            _c4dbgpf("rscalar[IMPL]: document termination next -- bail now '{}'", peeked_line.trimr("\r\n"));
            break;
        }
        else if(( ! peeked_line.begins_with(' ', indentation))) // is the line deindented?
        {
            if(!peeked_line.trim(" \r\n\t").empty()) // is the line not blank?
            {
                _c4dbgpf("rscalar[IMPL]: deindented line, not blank -- bail now '{}'", peeked_line.trimr("\r\n"));
                break;
            }
            _c4dbgpf("rscalar[IMPL]: line is blank and has less indentation: ref={} line={}: '{}'", indentation, peeked_line.first_not_of(' ') == csubstr::npos ? 0 : peeked_line.first_not_of(' '), peeked_line.trimr("\r\n"));
            _c4dbgpf("rscalar[IMPL]: ... searching for a line starting at indentation {}", indentation);
            csubstr next_peeked = _scan_to_next_nonempty_line(indentation);
            if(next_peeked.empty())
            {
                _c4dbgp("rscalar[IMPL]: ... finished.");
                break;
            }
            _c4dbgp("rscalar[IMPL]: ... continuing.");
            peeked_line = next_peeked;
        }

        _c4dbgpf("rscalar[IMPL]: line contents: '{}'", peeked_line.right_of(indentation, true).trimr("\r\n"));
        size_t token_pos;
        if(peeked_line.find(": ") != npos)
        {
            _line_progressed(peeked_line.find(": "));
            _c4err("': ' is not a valid token in plain flow (unquoted) scalars");
        }
        else if(peeked_line.ends_with(':'))
        {
            _line_progressed(peeked_line.find(':'));
            _c4err("lines cannot end with ':' in plain flow (unquoted) scalars");
        }
        else if((token_pos = peeked_line.find(" #")) != npos)
        {
            _line_progressed(token_pos);
            break;
            //_c4err("' #' is not a valid token in plain flow (unquoted) scalars");
        }

        _c4dbgpf("rscalar[IMPL]: append another line: (len={})'{}'", peeked_line.len, peeked_line.trimr("\r\n"));
        if(!_advance_to_peeked())
        {
            _c4dbgp("rscalar[IMPL]: file finishes after the scalar");
            break;
        }
        peeked_line = m_state->line_contents.rem;
    }
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->pos.offset >= offs);
    substr full(m_buf.str + (currscalar.str - m_buf.str),
                currscalar.len + (m_state->pos.offset - offs));
    full = full.trimr("\r\n ");
    return full;
}

substr Parser::_scan_complex_key(csubstr currscalar, csubstr peeked_line)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_buf.is_super(currscalar));
    // NOTE. there's a problem with _scan_to_next_nonempty_line(), as it counts newlines twice
    // size_t offs = m_state->pos.offset;   // so we workaround by directly counting from the end of the given scalar
    _RYML_CB_ASSERT(m_stack.m_callbacks, currscalar.end() >= m_buf.begin());
    size_t offs = static_cast<size_t>(currscalar.end() - m_buf.begin());
    while(true)
    {
        _c4dbgp("rcplxkey: continuing...");
        if(peeked_line.begins_with("...") || peeked_line.begins_with("---"))
        {
            _c4dbgpf("rcplxkey: document termination next -- bail now '{}'", peeked_line.trimr("\r\n"));
            break;
        }
        else
        {
            size_t pos = peeked_line.first_of("?:[]{}");
            if(pos == csubstr::npos)
            {
                pos = peeked_line.find("- ");
            }
            if(pos != csubstr::npos)
            {
                _c4dbgpf("rcplxkey: found special characters at pos={}: '{}'", pos, peeked_line.trimr("\r\n"));
                _line_progressed(pos);
                break;
            }
        }

        _c4dbgpf("rcplxkey: no special chars found '{}'", peeked_line.trimr("\r\n"));
        csubstr next_peeked = _scan_to_next_nonempty_line(0);
        if(next_peeked.empty())
        {
            _c4dbgp("rcplxkey: empty ... finished.");
            break;
        }
        _c4dbgp("rcplxkey: ... continuing.");
        peeked_line = next_peeked;

        _c4dbgpf("rcplxkey: line contents: '{}'", peeked_line.trimr("\r\n"));
        size_t colpos;
        if((colpos = peeked_line.find(": ")) != npos)
        {
            _c4dbgp("rcplxkey: found ': ', stopping.");
            _line_progressed(colpos);
            break;
        }
        #ifdef RYML_NO_COVERAGE__TO_BE_DELETED
        else if((colpos = peeked_line.ends_with(':')))
        {
            _c4dbgp("rcplxkey: ends with ':', stopping.");
            _line_progressed(colpos);
            break;
        }
        #endif
        _c4dbgpf("rcplxkey: append another line: (len={})'{}'", peeked_line.len, peeked_line.trimr("\r\n"));
        if(!_advance_to_peeked())
        {
            _c4dbgp("rcplxkey: file finishes after the scalar");
            break;
        }
        peeked_line = m_state->line_contents.rem;
    }
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->pos.offset >= offs);
    substr full(m_buf.str + (currscalar.str - m_buf.str),
                currscalar.len + (m_state->pos.offset - offs));
    return full;
}

//! scans to the next non-blank line starting with the given indentation
csubstr Parser::_scan_to_next_nonempty_line(size_t indentation)
{
    csubstr next_peeked;
    while(true)
    {
        _c4dbgpf("rscalar: ... curr offset: {} indentation={}", m_state->pos.offset, indentation);
        next_peeked = _peek_next_line(m_state->pos.offset);
        csubstr next_peeked_triml = next_peeked.triml(' ');
        _c4dbgpf("rscalar: ... next peeked line='{}'", next_peeked.trimr("\r\n"));
        if(next_peeked_triml.begins_with('#'))
        {
            _c4dbgp("rscalar: ... first non-space character is #");
            return {};
        }
        else if(next_peeked.begins_with(' ', indentation))
        {
            _c4dbgpf("rscalar: ... begins at same indentation {}, assuming continuation", indentation);
            _advance_to_peeked();
            return next_peeked;
        }
        else   // check for de-indentation
        {
            csubstr trimmed = next_peeked_triml.trimr("\t\r\n");
            _c4dbgpf("rscalar: ... deindented! trimmed='{}'", trimmed);
            if(!trimmed.empty())
            {
                _c4dbgp("rscalar: ... and not empty. bailing out.");
                return {};
            }
        }
        if(!_advance_to_peeked())
        {
            _c4dbgp("rscalar: file finished");
            return {};
        }
    }
    return {};
}

// returns false when the file finished
bool Parser::_advance_to_peeked()
{
    _line_progressed(m_state->line_contents.rem.len);
    _line_ended(); // advances to the peeked-at line, consuming all remaining (probably newline) characters on the current line
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->line_contents.rem.first_of("\r\n") == csubstr::npos);
    _c4dbgpf("advance to peeked: scan more... pos={} len={}", m_state->pos.offset, m_buf.len);
    _scan_line();  // puts the peeked-at line in the buffer
    if(_finished_file())
    {
        _c4dbgp("rscalar: finished file!");
        return false;
    }
    return true;
}

//-----------------------------------------------------------------------------

C4_ALWAYS_INLINE size_t _extend_from_combined_newline(char nl, char following)
{
    return (nl == '\n' && following == '\r') || (nl == '\r' && following == '\n');
}

//! look for the next newline chars, and jump to the right of those
csubstr from_next_line(csubstr rem)
{
    size_t nlpos = rem.first_of("\r\n");
    if(nlpos == csubstr::npos)
        return {};
    const char nl = rem[nlpos];
    rem = rem.right_of(nlpos);
    if(rem.empty())
        return {};
    if(_extend_from_combined_newline(nl, rem.front()))
        rem = rem.sub(1);
    return rem;
}

csubstr Parser::_peek_next_line(size_t pos) const
{
    csubstr rem{}; // declare here because of the goto
    size_t nlpos{}; // declare here because of the goto
    pos = pos == npos ? m_state->pos.offset : pos;
    if(pos >= m_buf.len)
        goto next_is_empty;

    // look for the next newline chars, and jump to the right of those
    rem = from_next_line(m_buf.sub(pos));
    if(rem.empty())
        goto next_is_empty;

    // now get everything up to and including the following newline chars
    nlpos = rem.first_of("\r\n");
    if((nlpos != csubstr::npos) && (nlpos + 1 < rem.len))
        nlpos += _extend_from_combined_newline(rem[nlpos], rem[nlpos+1]);
    rem = rem.left_of(nlpos, /*include_pos*/true);

    _c4dbgpf("peek next line @ {}: (len={})'{}'", pos, rem.len, rem.trimr("\r\n"));
    return rem;

next_is_empty:
    _c4dbgpf("peek next line @ {}: (len=0)''", pos);
    return {};
}


//-----------------------------------------------------------------------------
void Parser::LineContents::reset_with_next_line(csubstr buf, size_t offset)
{
    RYML_ASSERT(offset <= buf.len);
    char const* C4_RESTRICT b = &buf[offset];
    char const* C4_RESTRICT e = b;
    // get the current line stripped of newline chars
    while(e < buf.end() && (*e != '\n' && *e != '\r'))
        ++e;
    RYML_ASSERT(e >= b);
    const csubstr stripped_ = buf.sub(offset, static_cast<size_t>(e - b));
    // advance pos to include the first line ending
    if(e != buf.end() && *e == '\r')
        ++e;
    if(e != buf.end() && *e == '\n')
        ++e;
    RYML_ASSERT(e >= b);
    const csubstr full_ = buf.sub(offset, static_cast<size_t>(e - b));
    reset(full_, stripped_);
}

void Parser::_scan_line()
{
    if(m_state->pos.offset >= m_buf.len)
    {
        m_state->line_contents.reset(m_buf.last(0), m_buf.last(0));
        return;
    }
    m_state->line_contents.reset_with_next_line(m_buf, m_state->pos.offset);
}


//-----------------------------------------------------------------------------
void Parser::_line_progressed(size_t ahead)
{
    _c4dbgpf("line[{}] ({} cols) progressed by {}:  col {}-->{}   offset {}-->{}", m_state->pos.line, m_state->line_contents.full.len, ahead, m_state->pos.col, m_state->pos.col+ahead, m_state->pos.offset, m_state->pos.offset+ahead);
    m_state->pos.offset += ahead;
    m_state->pos.col += ahead;
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->pos.col <= m_state->line_contents.stripped.len+1);
    m_state->line_contents.rem = m_state->line_contents.rem.sub(ahead);
}

void Parser::_line_ended()
{
    _c4dbgpf("line[{}] ({} cols) ended! offset {}-->{}", m_state->pos.line, m_state->line_contents.full.len, m_state->pos.offset, m_state->pos.offset+m_state->line_contents.full.len - m_state->line_contents.stripped.len);
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->pos.col == m_state->line_contents.stripped.len+1);
    m_state->pos.offset += m_state->line_contents.full.len - m_state->line_contents.stripped.len;
    ++m_state->pos.line;
    m_state->pos.col = 1;
}

void Parser::_line_ended_undo()
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->pos.col == 1u);
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->pos.line > 0u);
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->pos.offset >= m_state->line_contents.full.len - m_state->line_contents.stripped.len);
    size_t delta = m_state->line_contents.full.len - m_state->line_contents.stripped.len;
    _c4dbgpf("line[{}] undo ended! line {}-->{}, offset {}-->{}", m_state->pos.line, m_state->pos.line, m_state->pos.line - 1, m_state->pos.offset, m_state->pos.offset - delta);
    m_state->pos.offset -= delta;
    --m_state->pos.line;
    m_state->pos.col = m_state->line_contents.stripped.len + 1u;
    // don't forget to undo also the changes to the remainder of the line
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->pos.offset >= m_buf.len || m_buf[m_state->pos.offset] == '\n' || m_buf[m_state->pos.offset] == '\r');
    m_state->line_contents.rem = m_buf.sub(m_state->pos.offset, 0);
}


//-----------------------------------------------------------------------------
void Parser::_set_indentation(size_t indentation)
{
    m_state->indref = indentation;
    _c4dbgpf("state[{}]: saving indentation: {}", m_state-m_stack.begin(), m_state->indref);
}

void Parser::_save_indentation(size_t behind)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->line_contents.rem.begin() >= m_state->line_contents.full.begin());
    m_state->indref = static_cast<size_t>(m_state->line_contents.rem.begin() - m_state->line_contents.full.begin());
    _RYML_CB_ASSERT(m_stack.m_callbacks, behind <= m_state->indref);
    m_state->indref -= behind;
    _c4dbgpf("state[{}]: saving indentation: {}", m_state-m_stack.begin(), m_state->indref);
}

bool Parser::_maybe_set_indentation_from_anchor_or_tag()
{
    if(m_key_anchor.not_empty())
    {
        _c4dbgpf("set indentation from key anchor: {}", m_key_anchor_indentation);
        _set_indentation(m_key_anchor_indentation); // this is the column where the anchor starts
        return true;
    }
    else if(m_key_tag.not_empty())
    {
        _c4dbgpf("set indentation from key tag: {}", m_key_tag_indentation);
        _set_indentation(m_key_tag_indentation); // this is the column where the tag starts
        return true;
    }
    return false;
}


//-----------------------------------------------------------------------------
void Parser::_write_key_anchor(size_t node_id)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_tree->has_key(node_id));
    if( ! m_key_anchor.empty())
    {
        _c4dbgpf("node={}: set key anchor to '{}'", node_id, m_key_anchor);
        m_tree->set_key_anchor(node_id, m_key_anchor);
        m_key_anchor.clear();
        m_key_anchor_was_before = false;
        m_key_anchor_indentation = 0;
    }
    else if( ! m_tree->is_key_quoted(node_id))
    {
        csubstr r = m_tree->key(node_id);
        if(r.begins_with('*'))
        {
            _c4dbgpf("node={}: set key reference: '{}'", node_id, r);
            m_tree->set_key_ref(node_id, r.sub(1));
        }
        else if(r == "<<")
        {
            m_tree->set_key_ref(node_id, r);
            _c4dbgpf("node={}: it's an inheriting reference", node_id);
            if(m_tree->is_seq(node_id))
            {
                _c4dbgpf("node={}: inheriting from seq of {}", node_id, m_tree->num_children(node_id));
                for(size_t i = m_tree->first_child(node_id); i != NONE; i = m_tree->next_sibling(i))
                {
                    if( ! (m_tree->val(i).begins_with('*')))
                        _c4err("malformed reference: '{}'", m_tree->val(i));
                }
            }
            else if( ! m_tree->val(node_id).begins_with('*'))
            {
                 _c4err("malformed reference: '{}'", m_tree->val(node_id));
            }
            //m_tree->set_key_ref(node_id, r);
        }
    }
}

//-----------------------------------------------------------------------------
void Parser::_write_val_anchor(size_t node_id)
{
    if( ! m_val_anchor.empty())
    {
        _c4dbgpf("node={}: set val anchor to '{}'", node_id, m_val_anchor);
        m_tree->set_val_anchor(node_id, m_val_anchor);
        m_val_anchor.clear();
    }
    csubstr r = m_tree->has_val(node_id) ? m_tree->val(node_id) : "";
    if(!m_tree->is_val_quoted(node_id) && r.begins_with('*'))
    {
        _c4dbgpf("node={}: set val reference: '{}'", node_id, r);
        RYML_CHECK(!m_tree->has_val_anchor(node_id));
        m_tree->set_val_ref(node_id, r.sub(1));
    }
}

//-----------------------------------------------------------------------------
void Parser::_push_level(bool explicit_flow_chars)
{
    _c4dbgpf("pushing level! currnode={}  currlevel={} stacksize={} stackcap={}", m_state->node_id, m_state->level, m_stack.size(), m_stack.capacity());
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state == &m_stack.top());
    if(node(m_state) == nullptr)
    {
        _c4dbgp("pushing level! actually no, current node is null");
        //_RYML_CB_ASSERT(m_stack.m_callbacks,  ! explicit_flow_chars);
        return;
    }
    flag_t st = RUNK;
    if(explicit_flow_chars || has_all(FLOW))
    {
        st |= FLOW;
    }
    m_stack.push_top();
    m_state = &m_stack.top();
    set_flags(st);
    m_state->node_id = (size_t)NONE;
    m_state->indref = (size_t)NONE;
    ++m_state->level;
    _c4dbgpf("pushing level: now, currlevel={}", m_state->level);
}

void Parser::_pop_level()
{
    _c4dbgpf("popping level! currnode={} currlevel={}", m_state->node_id, m_state->level);
    if(has_any(RMAP) || m_tree->is_map(m_state->node_id))
    {
        _stop_map();
    }
    if(has_any(RSEQ) || m_tree->is_seq(m_state->node_id))
    {
        _stop_seq();
    }
    if(m_tree->is_doc(m_state->node_id))
    {
        _stop_doc();
    }
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_stack.size() > 1);
    _prepare_pop();
    m_stack.pop();
    m_state = &m_stack.top();
    /*if(has_any(RMAP))
    {
        _toggle_key_val();
    }*/
    if(m_state->line_contents.indentation == 0)
    {
        //_RYML_CB_ASSERT(m_stack.m_callbacks, has_none(RTOP));
        add_flags(RTOP);
    }
    _c4dbgpf("popping level: now, currnode={} currlevel={}", m_state->node_id, m_state->level);
}

//-----------------------------------------------------------------------------
void Parser::_start_unk(bool /*as_child*/)
{
    _c4dbgp("start_unk");
    _push_level();
    _move_scalar_from_top();
}

//-----------------------------------------------------------------------------
void Parser::_start_doc(bool as_child)
{
    _c4dbgpf("start_doc (as child={})", as_child);
    _RYML_CB_ASSERT(m_stack.m_callbacks, node(m_stack.bottom()) == node(m_root_id));
    size_t parent_id = m_stack.size() < 2 ? m_root_id : m_stack.top(1).node_id;
    _RYML_CB_ASSERT(m_stack.m_callbacks, parent_id != NONE);
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_tree->is_root(parent_id));
    _RYML_CB_ASSERT(m_stack.m_callbacks, node(m_state) == nullptr || node(m_state) == node(m_root_id));
    if(as_child)
    {
        _c4dbgpf("start_doc: parent={}", parent_id);
        if( ! m_tree->is_stream(parent_id))
        {
            _c4dbgp("start_doc: rearranging with root as STREAM");
            m_tree->set_root_as_stream();
        }
        m_state->node_id = m_tree->append_child(parent_id);
        m_tree->to_doc(m_state->node_id);
    }
    #ifdef RYML_NO_COVERAGE__TO_BE_DELETED
    else
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, m_tree->is_seq(parent_id) || m_tree->empty(parent_id));
        m_state->node_id = parent_id;
        if( ! m_tree->is_doc(parent_id))
        {
            m_tree->to_doc(parent_id, DOC);
        }
    }
    #endif
    _c4dbgpf("start_doc: id={}", m_state->node_id);
    add_flags(RUNK|RTOP|NDOC);
    _handle_types();
    rem_flags(NDOC);
}

void Parser::_stop_doc()
{
    size_t doc_node = m_state->node_id;
    _c4dbgpf("stop_doc[{}]", doc_node);
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_tree->is_doc(doc_node));
    if(!m_tree->is_seq(doc_node) && !m_tree->is_map(doc_node) && !m_tree->is_val(doc_node))
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(SSCL));
        _c4dbgpf("stop_doc[{}]: there was nothing; adding null val", doc_node);
        m_tree->to_val(doc_node, {}, DOC);
    }
}

void Parser::_end_stream()
{
    _c4dbgpf("end_stream, level={} node_id={}", m_state->level, m_state->node_id);
    _RYML_CB_ASSERT(m_stack.m_callbacks,  ! m_stack.empty());
    NodeData *added = nullptr;
    if(has_any(SSCL))
    {
        if(m_tree->is_seq(m_state->node_id))
        {
            _c4dbgp("append val...");
            added = _append_val(_consume_scalar());
        }
        else if(m_tree->is_map(m_state->node_id))
        {
            _c4dbgp("append null key val...");
            added = _append_key_val_null(m_state->line_contents.rem.str);
            #ifdef RYML_NO_COVERAGE__TO_BE_DELETED
            if(has_any(RSEQIMAP))
            {
                _stop_seqimap();
                _pop_level();
            }
            #endif
        }
        else if(m_tree->is_doc(m_state->node_id) || m_tree->type(m_state->node_id) == NOTYPE)
        {
            NodeType_e quoted = has_any(QSCL) ? VALQUO : NOTYPE; // do this before consuming the scalar
            csubstr scalar = _consume_scalar();
            _c4dbgpf("node[{}]: to docval '{}'{}", m_state->node_id, scalar, quoted == VALQUO ? ", quoted" : "");
            m_tree->to_val(m_state->node_id, scalar, DOC|quoted);
            added = m_tree->get(m_state->node_id);
        }
        else
        {
            _c4err("internal error");
        }
    }
    else if(has_all(RSEQ|RVAL) && has_none(FLOW))
    {
        _c4dbgp("add last...");
        added = _append_val_null(m_state->line_contents.rem.str);
    }
    else if(!m_val_tag.empty() && (m_tree->is_doc(m_state->node_id) || m_tree->type(m_state->node_id) == NOTYPE))
    {
        csubstr scalar = m_state->line_contents.rem.first(0);
        _c4dbgpf("node[{}]: add null scalar as docval", m_state->node_id);
        m_tree->to_val(m_state->node_id, scalar, DOC);
        added = m_tree->get(m_state->node_id);
    }

    if(added)
    {
        size_t added_id = m_tree->id(added);
        if(m_tree->is_seq(m_state->node_id) || m_tree->is_doc(m_state->node_id))
        {
            if(!m_key_anchor.empty())
            {
                _c4dbgpf("node[{}]: move key to val anchor: '{}'", added_id, m_key_anchor);
                m_val_anchor = m_key_anchor;
                m_key_anchor = {};
            }
            if(!m_key_tag.empty())
            {
                _c4dbgpf("node[{}]: move key to val tag: '{}'", added_id, m_key_tag);
                m_val_tag = m_key_tag;
                m_key_tag = {};
            }
        }
        #ifdef RYML_NO_COVERAGE__TO_BE_DELETED
        if(!m_key_anchor.empty())
        {
            _c4dbgpf("node[{}]: set key anchor='{}'", added_id, m_key_anchor);
            m_tree->set_key_anchor(added_id, m_key_anchor);
            m_key_anchor = {};
        }
        #endif
        if(!m_val_anchor.empty())
        {
            _c4dbgpf("node[{}]: set val anchor='{}'", added_id, m_val_anchor);
            m_tree->set_val_anchor(added_id, m_val_anchor);
            m_val_anchor = {};
        }
        #ifdef RYML_NO_COVERAGE__TO_BE_DELETED
        if(!m_key_tag.empty())
        {
            _c4dbgpf("node[{}]: set key tag='{}' -> '{}'", added_id, m_key_tag, normalize_tag(m_key_tag));
            m_tree->set_key_tag(added_id, normalize_tag(m_key_tag));
            m_key_tag = {};
        }
        #endif
        if(!m_val_tag.empty())
        {
            _c4dbgpf("node[{}]: set val tag='{}' -> '{}'", added_id, m_val_tag, normalize_tag(m_val_tag));
            m_tree->set_val_tag(added_id, normalize_tag(m_val_tag));
            m_val_tag = {};
        }
    }

    while(m_stack.size() > 1)
    {
        _c4dbgpf("popping level: {} (stack sz={})", m_state->level, m_stack.size());
        _RYML_CB_ASSERT(m_stack.m_callbacks,  ! has_any(SSCL, &m_stack.top()));
        if(has_all(RSEQ|FLOW))
            _c4err("closing ] not found");
        _pop_level();
    }
    add_flags(NDOC);
}

void Parser::_start_new_doc(csubstr rem)
{
    _c4dbgp("_start_new_doc");
    _RYML_CB_ASSERT(m_stack.m_callbacks, rem.begins_with("---"));
    C4_UNUSED(rem);

    _end_stream();

    size_t indref = m_state->indref;
    _c4dbgpf("start a document, indentation={}", indref);
    _line_progressed(3);
    _push_level();
    _start_doc();
    _set_indentation(indref);
}


//-----------------------------------------------------------------------------
void Parser::_start_map(bool as_child)
{
    _c4dbgpf("start_map (as child={})", as_child);
    addrem_flags(RMAP|RVAL, RKEY|RUNK);
    _RYML_CB_ASSERT(m_stack.m_callbacks, node(m_stack.bottom()) == node(m_root_id));
    size_t parent_id = m_stack.size() < 2 ? m_root_id : m_stack.top(1).node_id;
    _RYML_CB_ASSERT(m_stack.m_callbacks, parent_id != NONE);
    _RYML_CB_ASSERT(m_stack.m_callbacks, node(m_state) == nullptr || node(m_state) == node(m_root_id));
    if(as_child)
    {
        m_state->node_id = m_tree->append_child(parent_id);
        if(has_all(SSCL))
        {
            type_bits key_quoted = NOTYPE;
            if(m_state->flags & QSCL) // before consuming the scalar
                key_quoted |= KEYQUO;
            csubstr key = _consume_scalar();
            m_tree->to_map(m_state->node_id, key, key_quoted);
            _c4dbgpf("start_map: id={} key='{}'", m_state->node_id, m_tree->key(m_state->node_id));
            _write_key_anchor(m_state->node_id);
            if( ! m_key_tag.empty())
            {
                _c4dbgpf("node[{}]: set key tag='{}' -> '{}'", m_state->node_id, m_key_tag, normalize_tag(m_key_tag));
                m_tree->set_key_tag(m_state->node_id, normalize_tag(m_key_tag));
                m_key_tag.clear();
            }
        }
        else
        {
            m_tree->to_map(m_state->node_id);
            _c4dbgpf("start_map: id={}", m_state->node_id);
        }
        m_tree->_p(m_state->node_id)->m_val.scalar.str = m_state->line_contents.rem.str;
        _write_val_anchor(m_state->node_id);
    }
    else
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, parent_id != NONE);
        m_state->node_id = parent_id;
        _c4dbgpf("start_map: id={}", m_state->node_id);
        type_bits as_doc = 0;
        if(m_tree->is_doc(m_state->node_id))
            as_doc |= DOC;
        if(!m_tree->is_map(parent_id))
        {
            RYML_CHECK(!m_tree->has_children(parent_id));
            m_tree->to_map(parent_id, as_doc);
        }
        else
        {
            m_tree->_add_flags(parent_id, as_doc);
        }
        _move_scalar_from_top();
        if(m_key_anchor.not_empty())
            m_key_anchor_was_before = true;
        _write_val_anchor(parent_id);
        if(m_stack.size() >= 2)
        {
            State const& parent_state = m_stack.top(1);
            if(parent_state.flags & RSET)
                add_flags(RSET);
        }
        m_tree->_p(parent_id)->m_val.scalar.str = m_state->line_contents.rem.str;
    }
    if( ! m_val_tag.empty())
    {
        _c4dbgpf("node[{}]: set val tag='{}' -> '{}'", m_state->node_id, m_val_tag, normalize_tag(m_val_tag));
        m_tree->set_val_tag(m_state->node_id, normalize_tag(m_val_tag));
        m_val_tag.clear();
    }
}

void Parser::_start_map_unk(bool as_child)
{
    _c4dbgpf("start_map_unk (as child={})", as_child);
    if(!m_key_anchor_was_before)
    {
        _c4dbgpf("stash key anchor before starting map... '{}'", m_key_anchor);
        csubstr ka = m_key_anchor;
        m_key_anchor = {};
        _start_map(as_child);
        m_key_anchor = ka;
    }
    else
    {
        _start_map(as_child);
        m_key_anchor_was_before = false;
    }
    if(m_key_tag2.not_empty())
    {
        m_key_tag = m_key_tag2;
        m_key_tag_indentation = m_key_tag2_indentation;
        m_key_tag2.clear();
        m_key_tag2_indentation = 0;
    }
}

void Parser::_stop_map()
{
    _c4dbgpf("stop_map[{}]", m_state->node_id);
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_tree->is_map(m_state->node_id));
    if(has_all(QMRK|RKEY) && !has_all(SSCL))
    {
        _c4dbgpf("stop_map[{}]: RKEY", m_state->node_id);
        _store_scalar_null(m_state->line_contents.rem.str);
        _append_key_val_null(m_state->line_contents.rem.str);
    }
}


//-----------------------------------------------------------------------------
void Parser::_start_seq(bool as_child)
{
    _c4dbgpf("start_seq (as child={})", as_child);
    if(has_all(RTOP|RUNK))
    {
        _c4dbgpf("start_seq: moving key tag to val tag: '{}'", m_key_tag);
        m_val_tag = m_key_tag;
        m_key_tag.clear();
    }
    addrem_flags(RSEQ|RVAL, RUNK);
    _RYML_CB_ASSERT(m_stack.m_callbacks, node(m_stack.bottom()) == node(m_root_id));
    size_t parent_id = m_stack.size() < 2 ? m_root_id : m_stack.top(1).node_id;
    _RYML_CB_ASSERT(m_stack.m_callbacks, parent_id != NONE);
    _RYML_CB_ASSERT(m_stack.m_callbacks, node(m_state) == nullptr || node(m_state) == node(m_root_id));
    if(as_child)
    {
        m_state->node_id = m_tree->append_child(parent_id);
        if(has_all(SSCL))
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks, m_tree->is_map(parent_id));
            type_bits key_quoted = 0;
            if(m_state->flags & QSCL) // before consuming the scalar
                key_quoted |= KEYQUO;
            csubstr key = _consume_scalar();
            m_tree->to_seq(m_state->node_id, key, key_quoted);
            _c4dbgpf("start_seq: id={} name='{}'", m_state->node_id, m_tree->key(m_state->node_id));
            _write_key_anchor(m_state->node_id);
            if( ! m_key_tag.empty())
            {
                _c4dbgpf("start_seq[{}]: set key tag='{}' -> '{}'", m_state->node_id, m_key_tag, normalize_tag(m_key_tag));
                m_tree->set_key_tag(m_state->node_id, normalize_tag(m_key_tag));
                m_key_tag.clear();
            }
        }
        else
        {
            type_bits as_doc = 0;
            _RYML_CB_ASSERT(m_stack.m_callbacks, !m_tree->is_doc(m_state->node_id));
            m_tree->to_seq(m_state->node_id, as_doc);
            _c4dbgpf("start_seq: id={}{}", m_state->node_id, as_doc ? " as doc" : "");
        }
        _write_val_anchor(m_state->node_id);
        m_tree->_p(m_state->node_id)->m_val.scalar.str = m_state->line_contents.rem.str;
    }
    else
    {
        m_state->node_id = parent_id;
        type_bits as_doc = 0;
        if(m_tree->is_doc(m_state->node_id))
            as_doc |= DOC;
        if(!m_tree->is_seq(parent_id))
        {
            RYML_CHECK(!m_tree->has_children(parent_id));
            m_tree->to_seq(parent_id, as_doc);
        }
        else
        {
            m_tree->_add_flags(parent_id, as_doc);
        }
        _move_scalar_from_top();
        _c4dbgpf("start_seq: id={}{}", m_state->node_id, as_doc ? " as_doc" : "");
        _write_val_anchor(parent_id);
        m_tree->_p(parent_id)->m_val.scalar.str = m_state->line_contents.rem.str;
    }
    if( ! m_val_tag.empty())
    {
        _c4dbgpf("start_seq[{}]: set val tag='{}' -> '{}'", m_state->node_id, m_val_tag, normalize_tag(m_val_tag));
        m_tree->set_val_tag(m_state->node_id, normalize_tag(m_val_tag));
        m_val_tag.clear();
    }
}

void Parser::_stop_seq()
{
    _c4dbgp("stop_seq");
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_tree->is_seq(m_state->node_id));
}


//-----------------------------------------------------------------------------
void Parser::_start_seqimap()
{
    _c4dbgpf("start_seqimap at node={}. has_children={}", m_state->node_id, m_tree->has_children(m_state->node_id));
    _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(RSEQ|FLOW));
    // create a map, and turn the last scalar of this sequence
    // into the key of the map's first child. This scalar was
    // understood to be a value in the sequence, but it is
    // actually a key of a map, implicitly opened here.
    // Eg [val, key: val]
    //
    // Yep, YAML is crazy.
    if(m_tree->has_children(m_state->node_id) && m_tree->has_val(m_tree->last_child(m_state->node_id)))
    {
        size_t prev = m_tree->last_child(m_state->node_id);
        NodeType ty = m_tree->_p(prev)->m_type; // don't use type() because it masks out the quotes
        NodeScalar tmp = m_tree->valsc(prev);
        _c4dbgpf("has children and last child={} has val. saving the scalars, val='{}' quoted={}", prev, tmp.scalar, ty.is_val_quoted());
        m_tree->remove(prev);
        _push_level();
        _start_map();
        _store_scalar(tmp.scalar, ty.is_val_quoted());
        m_key_anchor = tmp.anchor;
        m_key_tag = tmp.tag;
    }
    else
    {
        _c4dbgpf("node {} has no children yet, using empty key", m_state->node_id);
        _push_level();
        _start_map();
        _store_scalar_null(m_state->line_contents.rem.str);
    }
    add_flags(RSEQIMAP|FLOW);
}

void Parser::_stop_seqimap()
{
    _c4dbgp("stop_seqimap");
    _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(RSEQIMAP));
}


//-----------------------------------------------------------------------------
NodeData* Parser::_append_val(csubstr val, flag_t quoted)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks,  ! has_all(SSCL));
    _RYML_CB_ASSERT(m_stack.m_callbacks, node(m_state) != nullptr);
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_tree->is_seq(m_state->node_id));
    type_bits additional_flags = quoted ? VALQUO : NOTYPE;
    _c4dbgpf("append val: '{}' to parent id={} (level={}){}", val, m_state->node_id, m_state->level, quoted ? " VALQUO!" : "");
    size_t nid = m_tree->append_child(m_state->node_id);
    m_tree->to_val(nid, val, additional_flags);
    _c4dbgpf("append val: id={} val='{}'", nid, m_tree->get(nid)->m_val.scalar);
    if( ! m_val_tag.empty())
    {
        _c4dbgpf("append val[{}]: set val tag='{}' -> '{}'", nid, m_val_tag, normalize_tag(m_val_tag));
        m_tree->set_val_tag(nid, normalize_tag(m_val_tag));
        m_val_tag.clear();
    }
    _write_val_anchor(nid);
    return m_tree->get(nid);
}

NodeData* Parser::_append_key_val(csubstr val, flag_t val_quoted)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_tree->is_map(m_state->node_id));
    type_bits additional_flags = 0;
    if(m_state->flags & QSCL)
        additional_flags |= KEYQUO;
    if(val_quoted)
        additional_flags |= VALQUO;
    csubstr key = _consume_scalar();
    _c4dbgpf("append keyval: '{}' '{}' to parent id={} (level={}){}{}", key, val, m_state->node_id, m_state->level, (additional_flags & KEYQUO) ? " KEYQUO!" : "", (additional_flags & VALQUO) ? " VALQUO!" : "");
    size_t nid = m_tree->append_child(m_state->node_id);
    m_tree->to_keyval(nid, key, val, additional_flags);
    _c4dbgpf("append keyval: id={} key='{}' val='{}'", nid, m_tree->key(nid), m_tree->val(nid));
    if( ! m_key_tag.empty())
    {
        _c4dbgpf("append keyval[{}]: set key tag='{}' -> '{}'", nid, m_key_tag, normalize_tag(m_key_tag));
        m_tree->set_key_tag(nid, normalize_tag(m_key_tag));
        m_key_tag.clear();
    }
    if( ! m_val_tag.empty())
    {
        _c4dbgpf("append keyval[{}]: set val tag='{}' -> '{}'", nid, m_val_tag, normalize_tag(m_val_tag));
        m_tree->set_val_tag(nid, normalize_tag(m_val_tag));
        m_val_tag.clear();
    }
    _write_key_anchor(nid);
    _write_val_anchor(nid);
    rem_flags(QMRK);
    return m_tree->get(nid);
}


//-----------------------------------------------------------------------------
void Parser::_store_scalar(csubstr s, flag_t is_quoted)
{
    _c4dbgpf("state[{}]: storing scalar '{}' (flag: {}) (old scalar='{}')",
             m_state-m_stack.begin(), s, m_state->flags & SSCL, m_state->scalar);
    RYML_CHECK(has_none(SSCL));
    add_flags(SSCL | (is_quoted * QSCL));
    m_state->scalar = s;
}

csubstr Parser::_consume_scalar()
{
    _c4dbgpf("state[{}]: consuming scalar '{}' (flag: {}))", m_state-m_stack.begin(), m_state->scalar, m_state->flags & SSCL);
    RYML_CHECK(m_state->flags & SSCL);
    csubstr s = m_state->scalar;
    rem_flags(SSCL | QSCL);
    m_state->scalar.clear();
    return s;
}

void Parser::_move_scalar_from_top()
{
    if(m_stack.size() < 2) return;
    State &prev = m_stack.top(1);
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state == &m_stack.top());
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state != &prev);
    if(prev.flags & SSCL)
    {
        _c4dbgpf("moving scalar '{}' from state[{}] to state[{}] (overwriting '{}')", prev.scalar, &prev-m_stack.begin(), m_state-m_stack.begin(), m_state->scalar);
        add_flags(prev.flags & (SSCL | QSCL));
        m_state->scalar = prev.scalar;
        rem_flags(SSCL | QSCL, &prev);
        prev.scalar.clear();
    }
}

//-----------------------------------------------------------------------------
/** @todo this function is a monster and needs love. Likely, it needs
 * to be split like _scan_scalar_*() */
bool Parser::_handle_indentation()
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(FLOW));
    if( ! _at_line_begin())
        return false;

    size_t ind = m_state->line_contents.indentation;
    csubstr rem = m_state->line_contents.rem;
    /** @todo instead of trimming, we should use the indentation index from above */
    csubstr remt = rem.triml(' ');

    if(remt.empty() || remt.begins_with('#')) // this is a blank or comment line
    {
        _line_progressed(rem.size());
        return true;
    }

    _c4dbgpf("indentation? ind={} indref={}", ind, m_state->indref);
    if(ind == m_state->indref)
    {
        _c4dbgpf("same indentation: {}", ind);
        if(!rem.sub(ind).begins_with('-'))
        {
            _c4dbgp("does not begin with -");
            if(has_any(RMAP))
            {
                if(has_all(SSCL|RVAL))
                {
                    _c4dbgp("add with null val");
                    _append_key_val_null(rem.str + ind - 1);
                    addrem_flags(RKEY, RVAL);
                }
            }
            else if(has_any(RSEQ))
            {
                if(m_stack.size() > 2) // do not pop to root level
                {
                    if(has_any(RNXT))
                    {
                        _c4dbgp("end the indentless seq");
                        _pop_level();
                        return true;
                    }
                    else if(has_any(RVAL))
                    {
                        _c4dbgp("add with null val");
                        _append_val_null(rem.str);
                        _c4dbgp("end the indentless seq");
                        _pop_level();
                        return true;
                    }
                }
            }
        }
        _line_progressed(ind);
        return ind > 0;
    }
    else if(ind < m_state->indref)
    {
        _c4dbgpf("smaller indentation ({} < {})!!!", ind, m_state->indref);
        if(has_all(RVAL))
        {
            _c4dbgp("there was an empty val -- appending");
            if(has_all(RMAP))
            {
                _RYML_CB_ASSERT(m_stack.m_callbacks, has_all(SSCL));
                _append_key_val_null(rem.sub(ind).str - 1);
            }
            else if(has_all(RSEQ))
            {
                _RYML_CB_ASSERT(m_stack.m_callbacks, has_none(SSCL));
                _append_val_null(rem.sub(ind).str - 1);
            }
        }
        // search the stack frame to jump to based on its indentation
        State const* popto = nullptr;
        _RYML_CB_ASSERT(m_stack.m_callbacks, m_stack.is_contiguous()); // this search relies on the stack being contiguous
        for(State const* s = m_state-1; s >= m_stack.begin(); --s)
        {
            _c4dbgpf("searching for state with indentation {}. curr={} (level={},node={})", ind, s->indref, s->level, s->node_id);
            if(s->indref == ind)
            {
                _c4dbgpf("gotit!!! level={} node={}", s->level, s->node_id);
                popto = s;
                // while it may be tempting to think we're done at this
                // point, we must still determine whether we're jumping to a
                // parent with the same indentation. Consider this case with
                // an indentless sequence:
                //
                // product:
                // - sku: BL394D
                //   quantity: 4
                //   description: Basketball
                //   price: 450.00
                // - sku: BL4438H
                //   quantity: 1
                //   description: Super Hoop
                //   price: 2392.00  # jumping one level here would be wrong.
                // tax: 1234.5       # we must jump two levels
                if(popto > m_stack.begin())
                {
                    auto parent = popto - 1;
                    if(parent->indref == popto->indref)
                    {
                        _c4dbgpf("the parent (level={},node={}) has the same indentation ({}). is this in an indentless sequence?", parent->level, parent->node_id, popto->indref);
                        _c4dbgpf("isseq(popto)={} ismap(parent)={}", m_tree->is_seq(popto->node_id), m_tree->is_map(parent->node_id));
                        if(m_tree->is_seq(popto->node_id) && m_tree->is_map(parent->node_id))
                        {
                            if( ! remt.begins_with('-'))
                            {
                                _c4dbgp("this is an indentless sequence");
                                popto = parent;
                            }
                            else
                            {
                                _c4dbgp("not an indentless sequence");
                            }
                        }
                    }
                }
                break;
            }
        }
        if(!popto || popto >= m_state || popto->level >= m_state->level)
        {
            _c4err("parse error: incorrect indentation?");
        }
        _c4dbgpf("popping {} levels: from level {} to level {}", m_state->level-popto->level, m_state->level, popto->level);
        while(m_state != popto)
        {
            _c4dbgpf("popping level {} (indentation={})", m_state->level, m_state->indref);
            _pop_level();
        }
        _RYML_CB_ASSERT(m_stack.m_callbacks, ind == m_state->indref);
        _line_progressed(ind);
        return true;
    }
    else
    {
        _c4dbgpf("larger indentation ({} > {})!!!", ind, m_state->indref);
        _RYML_CB_ASSERT(m_stack.m_callbacks, ind > m_state->indref);
        if(has_all(RMAP|RVAL))
        {
            if(_is_scalar_next__rmap_val(remt) && (!remt.first_of_any(": ", "? ")) && (!remt.ends_with(":")))
            {
                _c4dbgpf("actually it seems a value: '{}'", remt);
            }
            else
            {
                addrem_flags(RKEY, RVAL);
                _start_unk();
                //_move_scalar_from_top();
                _line_progressed(ind);
                _save_indentation();
                return true;
            }
        }
        else if(has_all(RSEQ|RVAL))
        {
            // nothing to do here
        }
        else
        {
            _c4err("parse error - indentation should not increase at this point");
        }
    }

    return false;
}

//-----------------------------------------------------------------------------
csubstr Parser::_scan_comment()
{
    csubstr s = m_state->line_contents.rem;
    _RYML_CB_ASSERT(m_stack.m_callbacks, s.begins_with('#'));
    _line_progressed(s.len);
    // skip the # character
    s = s.sub(1);
    // skip leading whitespace
    s = s.right_of(s.first_not_of(' '), /*include_pos*/true);
    _c4dbgpf("comment was '{}'", s);
    return s;
}

//-----------------------------------------------------------------------------
csubstr Parser::_scan_squot_scalar()
{
    // quoted scalars can spread over multiple lines!
    // nice explanation here: http://yaml-multiline.info/

    // a span to the end of the file
    size_t b = m_state->pos.offset;
    substr s = m_buf.sub(b);
    if(s.begins_with(' '))
    {
        s = s.triml(' ');
        _RYML_CB_ASSERT(m_stack.m_callbacks, m_buf.sub(b).is_super(s));
        _RYML_CB_ASSERT(m_stack.m_callbacks, s.begin() >= m_buf.sub(b).begin());
        _line_progressed((size_t)(s.begin() - m_buf.sub(b).begin()));
    }
    b = m_state->pos.offset; // take this into account
    _RYML_CB_ASSERT(m_stack.m_callbacks, s.begins_with('\''));

    // skip the opening quote
    _line_progressed(1);
    s = s.sub(1);

    bool needs_filter = false;

    size_t numlines = 1; // we already have one line
    size_t pos = npos; // find the pos of the matching quote
    while( ! _finished_file())
    {
        const csubstr line = m_state->line_contents.rem;
        bool line_is_blank = true;
        _c4dbgpf("scanning single quoted scalar @ line[{}]: ~~~{}~~~", m_state->pos.line, line);
        for(size_t i = 0; i < line.len; ++i)
        {
            const char curr = line.str[i];
            if(curr == '\'') // single quotes are escaped with two single quotes
            {
                const char next = i+1 < line.len ? line.str[i+1] : '~';
                if(next != '\'') // so just look for the first quote
                {                // without another after it
                    pos = i;
                    break;
                }
                else
                {
                    needs_filter = true; // needs filter to remove escaped quotes
                    ++i; // skip the escaped quote
                }
            }
            else if(curr != ' ')
            {
                line_is_blank = false;
            }
        }

        // leading whitespace also needs filtering
        needs_filter = needs_filter
            || (numlines > 1)
            || line_is_blank
            || (_at_line_begin() && line.begins_with(' '));

        if(pos == npos)
        {
            _line_progressed(line.len);
            ++numlines;
        }
        else
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks, pos >= 0 && pos < m_buf.len);
            _RYML_CB_ASSERT(m_stack.m_callbacks, m_buf[m_state->pos.offset + pos] == '\'');
            _line_progressed(pos + 1); // progress beyond the quote
            pos = m_state->pos.offset - b - 1; // but we stop before it
            break;
        }

        _line_ended();
        _scan_line();
    }

    if(pos == npos)
    {
        _c4err("reached end of file while looking for closing quote");
    }
    else
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, pos > 0);
        _RYML_CB_ASSERT(m_stack.m_callbacks, s.end() >= m_buf.begin() && s.end() <= m_buf.end());
        _RYML_CB_ASSERT(m_stack.m_callbacks, s.end() == m_buf.end() || *s.end() == '\'');
        s = s.sub(0, pos-1);
    }

    if(needs_filter)
    {
        csubstr ret = _filter_squot_scalar(s);
        _RYML_CB_ASSERT(m_stack.m_callbacks, ret.len <= s.len || s.empty() || s.trim(' ').empty());
        _c4dbgpf("final scalar: \"{}\"", ret);
        return ret;
    }

    _c4dbgpf("final scalar: \"{}\"", s);

    return s;
}

//-----------------------------------------------------------------------------
csubstr Parser::_scan_dquot_scalar()
{
    // quoted scalars can spread over multiple lines!
    // nice explanation here: http://yaml-multiline.info/

    // a span to the end of the file
    size_t b = m_state->pos.offset;
    substr s = m_buf.sub(b);
    if(s.begins_with(' '))
    {
        s = s.triml(' ');
        _RYML_CB_ASSERT(m_stack.m_callbacks, m_buf.sub(b).is_super(s));
        _RYML_CB_ASSERT(m_stack.m_callbacks, s.begin() >= m_buf.sub(b).begin());
        _line_progressed((size_t)(s.begin() - m_buf.sub(b).begin()));
    }
    b = m_state->pos.offset; // take this into account
    _RYML_CB_ASSERT(m_stack.m_callbacks, s.begins_with('"'));

    // skip the opening quote
    _line_progressed(1);
    s = s.sub(1);

    bool needs_filter = false;

    size_t numlines = 1; // we already have one line
    size_t pos = npos; // find the pos of the matching quote
    while( ! _finished_file())
    {
        const csubstr line = m_state->line_contents.rem;
        bool line_is_blank = true;
        _c4dbgpf("scanning double quoted scalar @ line[{}]:  line='{}'", m_state->pos.line, line);
        for(size_t i = 0; i < line.len; ++i)
        {
            const char curr = line.str[i];
            if(curr != ' ')
                line_is_blank = false;
            // every \ is an escape
            if(curr == '\\')
            {
                const char next = i+1 < line.len ? line.str[i+1] : '~';
                needs_filter = true;
                if(next == '"' || next == '\\')
                    ++i;
            }
            else if(curr == '"')
            {
                pos = i;
                break;
            }
        }

        // leading whitespace also needs filtering
        needs_filter = needs_filter
            || (numlines > 1)
            || line_is_blank
            || (_at_line_begin() && line.begins_with(' '));

        if(pos == npos)
        {
            _line_progressed(line.len);
            ++numlines;
        }
        else
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks, pos >= 0 && pos < m_buf.len);
            _RYML_CB_ASSERT(m_stack.m_callbacks, m_buf[m_state->pos.offset + pos] == '"');
            _line_progressed(pos + 1); // progress beyond the quote
            pos = m_state->pos.offset - b - 1; // but we stop before it
            break;
        }

        _line_ended();
        _scan_line();
    }

    if(pos == npos)
    {
        _c4err("reached end of file looking for closing quote");
    }
    else
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, pos > 0);
        _RYML_CB_ASSERT(m_stack.m_callbacks, s.end() == m_buf.end() || *s.end() == '"');
        _RYML_CB_ASSERT(m_stack.m_callbacks, s.end() >= m_buf.begin() && s.end() <= m_buf.end());
        s = s.sub(0, pos-1);
    }

    if(needs_filter)
    {
        csubstr ret = _filter_dquot_scalar(s);
        _c4dbgpf("final scalar: [{}]\"{}\"", ret.len, ret);
        _RYML_CB_ASSERT(m_stack.m_callbacks, ret.len <= s.len || s.empty() || s.trim(' ').empty());
        return ret;
    }

    _c4dbgpf("final scalar: \"{}\"", s);

    return s;
}

//-----------------------------------------------------------------------------
csubstr Parser::_scan_block()
{
    // nice explanation here: http://yaml-multiline.info/
    csubstr s = m_state->line_contents.rem;
    csubstr trimmed = s.triml(' ');
    if(trimmed.str > s.str)
    {
        _c4dbgp("skipping whitespace");
        _RYML_CB_ASSERT(m_stack.m_callbacks, trimmed.str >= s.str);
        _line_progressed(static_cast<size_t>(trimmed.str - s.str));
        s = trimmed;
    }
    _RYML_CB_ASSERT(m_stack.m_callbacks, s.begins_with('|') || s.begins_with('>'));

    _c4dbgpf("scanning block: specs=\"{}\"", s);

    // parse the spec
    BlockStyle_e newline = s.begins_with('>') ? BLOCK_FOLD : BLOCK_LITERAL;
    BlockChomp_e chomp = CHOMP_CLIP; // default to clip unless + or - are used
    size_t indentation = npos; // have to find out if no spec is given
    csubstr digits;
    if(s.len > 1)
    {
        _RYML_CB_ASSERT(m_stack.m_callbacks, s.begins_with_any("|>"));
        csubstr t = s.sub(1);
        _c4dbgpf("scanning block: spec is multichar: '{}'", t);
        _RYML_CB_ASSERT(m_stack.m_callbacks, t.len >= 1);
        size_t pos = t.first_of("-+");
        _c4dbgpf("scanning block: spec chomp char at {}", pos);
        if(pos != npos)
        {
            if(t[pos] == '-')
                chomp = CHOMP_STRIP;
            else if(t[pos] == '+')
                chomp = CHOMP_KEEP;
            if(pos == 0)
                t = t.sub(1);
            else
                t = t.first(pos);
        }
        // from here to the end, only digits are considered
        digits = t.left_of(t.first_not_of("0123456789"));
        if( ! digits.empty())
        {
            if( ! c4::atou(digits, &indentation))
                _c4err("parse error: could not read decimal");
            _c4dbgpf("scanning block: indentation specified: {}. add {} from curr state -> {}", indentation, m_state->indref, indentation+m_state->indref);
            indentation += m_state->indref;
        }
    }

    // finish the current line
    _line_progressed(s.len);
    _line_ended();
    _scan_line();

    _c4dbgpf("scanning block: style={}  chomp={}  indentation={}", newline==BLOCK_FOLD ? "fold" : "literal", chomp==CHOMP_CLIP ? "clip" : (chomp==CHOMP_STRIP ? "strip" : "keep"), indentation);

    // start with a zero-length block, already pointing at the right place
    substr raw_block(m_buf.data() + m_state->pos.offset, size_t(0));// m_state->line_contents.full.sub(0, 0);
    _RYML_CB_ASSERT(m_stack.m_callbacks, raw_block.begin() == m_state->line_contents.full.begin());

    // read every full line into a raw block,
    // from which newlines are to be stripped as needed.
    //
    // If no explicit indentation was given, pick it from the first
    // non-empty line. See
    // https://yaml.org/spec/1.2.2/#8111-block-indentation-indicator
    size_t num_lines = 0, first = m_state->pos.line, provisional_indentation = npos;
    LineContents lc;
    while(( ! _finished_file()))
    {
        // peek next line, but do not advance immediately
        lc.reset_with_next_line(m_buf, m_state->pos.offset);
        _c4dbgpf("scanning block: peeking at '{}'", lc.stripped);
        // evaluate termination conditions
        if(indentation != npos)
        {
            // stop when the line is deindented and not empty
            if(lc.indentation < indentation && ( ! lc.rem.trim(" \t\r\n").empty()))
            {
                if(raw_block.len)
                {
                    _c4dbgpf("scanning block: indentation decreased ref={} thisline={}", indentation, lc.indentation);
                }
                else
                {
                    _c4err("indentation decreased without any scalar");
                }
                break;
            }
            else if(indentation == 0)
            {
                if((lc.rem == "..." || lc.rem.begins_with("... "))
                    ||
                   (lc.rem == "---" || lc.rem.begins_with("--- ")))
                {
                    _c4dbgp("scanning block: stop. indentation=0 and stream ended");
                    break;
                }
            }
        }
        else
        {
            _c4dbgpf("scanning block: indentation ref not set. firstnonws={}", lc.stripped.first_not_of(' '));
            if(lc.stripped.first_not_of(' ') != npos) // non-empty line
            {
                _c4dbgpf("scanning block: line not empty. indref={} indprov={} indentation={}", m_state->indref, provisional_indentation, lc.indentation);
                if(provisional_indentation == npos)
                {
                    if(lc.indentation < m_state->indref)
                    {
                        _c4dbgpf("scanning block: block terminated indentation={} < indref={}", lc.indentation, m_state->indref);
                        if(raw_block.len == 0)
                        {
                            _c4dbgp("scanning block: was empty, undo next line");
                            _line_ended_undo();
                        }
                        break;
                    }
                    else if(lc.indentation == m_state->indref)
                    {
                        if(has_any(RSEQ|RMAP))
                        {
                            _c4dbgpf("scanning block: block terminated. reading container and indentation={}==indref={}", lc.indentation, m_state->indref);
                            break;
                        }
                    }
                    _c4dbgpf("scanning block: set indentation ref from this line: ref={}", lc.indentation);
                    indentation = lc.indentation;
                }
                else
                {
                    if(lc.indentation >= provisional_indentation)
                    {
                        _c4dbgpf("scanning block: set indentation ref from provisional indentation: provisional_ref={}, thisline={}", provisional_indentation, lc.indentation);
                        //indentation = provisional_indentation ? provisional_indentation : lc.indentation;
                        indentation = lc.indentation;
                    }
                    else
                    {
                        break;
                        //_c4err("parse error: first non-empty block line should have at least the original indentation");
                    }
                }
            }
            else // empty line
            {
                _c4dbgpf("scanning block: line empty or {} spaces. line_indentation={} prov_indentation={}", lc.stripped.len, lc.indentation, provisional_indentation);
                if(provisional_indentation != npos)
                {
                    if(lc.stripped.len >= provisional_indentation)
                    {
                        _c4dbgpf("scanning block: increase provisional_ref {} -> {}", provisional_indentation, lc.stripped.len);
                        provisional_indentation = lc.stripped.len;
                    }
                    #ifdef RYML_NO_COVERAGE__TO_BE_DELETED
                    else if(lc.indentation >= provisional_indentation && lc.indentation != npos)
                    {
                        _c4dbgpf("scanning block: increase provisional_ref {} -> {}", provisional_indentation, lc.indentation);
                        provisional_indentation = lc.indentation;
                    }
                    #endif
                }
                else
                {
                    provisional_indentation = lc.indentation ? lc.indentation : has_any(RSEQ|RVAL);
                    _c4dbgpf("scanning block: initialize provisional_ref={}", provisional_indentation);
                    if(provisional_indentation == npos)
                    {
                        provisional_indentation = lc.stripped.len ? lc.stripped.len : has_any(RSEQ|RVAL);
                        _c4dbgpf("scanning block: initialize provisional_ref={}", provisional_indentation);
                    }
                }
            }
        }
        // advance now that we know the folded scalar continues
        m_state->line_contents = lc;
        _c4dbgpf("scanning block: append '{}'", m_state->line_contents.rem);
        raw_block.len += m_state->line_contents.full.len;
        _line_progressed(m_state->line_contents.rem.len);
        _line_ended();
        ++num_lines;
    }
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_state->pos.line == (first + num_lines) || (raw_block.len == 0));
    C4_UNUSED(num_lines);
    C4_UNUSED(first);

    if(indentation == npos)
    {
        _c4dbgpf("scanning block: set indentation from provisional: {}", provisional_indentation);
        indentation = provisional_indentation;
    }

    if(num_lines)
        _line_ended_undo();

    _c4dbgpf("scanning block: raw=~~~{}~~~", raw_block);

    // ok! now we strip the newlines and spaces according to the specs
    s = _filter_block_scalar(raw_block, newline, chomp, indentation);

    _c4dbgpf("scanning block: final=~~~{}~~~", s);

    return s;
}


//-----------------------------------------------------------------------------

template<bool backslash_is_escape, bool keep_trailing_whitespace>
bool Parser::_filter_nl(substr r, size_t *C4_RESTRICT i, size_t *C4_RESTRICT pos, size_t indentation)
{
    // a debugging scaffold:
    #if 0
    #define _c4dbgfnl(fmt, ...) _c4dbgpf("filter_nl[{}]: " fmt, *i, __VA_ARGS__)
    #else
    #define _c4dbgfnl(...)
    #endif

    const char curr = r[*i];
    bool replaced = false;

    _RYML_CB_ASSERT(m_stack.m_callbacks, indentation != npos);
    _RYML_CB_ASSERT(m_stack.m_callbacks, curr == '\n');

    _c4dbgfnl("found newline. sofar=[{}]~~~{}~~~", *pos, m_filter_arena.first(*pos));
    size_t ii = *i;
    size_t numnl_following = count_following_newlines(r, &ii, indentation);
    if(numnl_following)
    {
        _c4dbgfnl("{} consecutive (empty) lines {} in the middle. totalws={}", 1+numnl_following, ii < r.len ? "in the middle" : "at the end", ii - *i);
        for(size_t j = 0; j < numnl_following; ++j)
            m_filter_arena.str[(*pos)++] = '\n';
    }
    else
    {
        if(r.first_not_of(" \t", *i+1) != npos)
        {
            m_filter_arena.str[(*pos)++] = ' ';
            _c4dbgfnl("single newline. convert to space. ii={}/{}. sofar=[{}]~~~{}~~~", ii, r.len, *pos, m_filter_arena.first(*pos));
            replaced = true;
        }
        else
        {
            if C4_IF_CONSTEXPR (keep_trailing_whitespace)
            {
                m_filter_arena.str[(*pos)++] = ' ';
                _c4dbgfnl("single newline. convert to space. ii={}/{}. sofar=[{}]~~~{}~~~", ii, r.len, *pos, m_filter_arena.first(*pos));
                replaced = true;
            }
            else
            {
                _c4dbgfnl("last newline, everything else is whitespace. ii={}/{}", ii, r.len);
                *i = r.len;
            }
        }
        if C4_IF_CONSTEXPR (backslash_is_escape)
        {
            if(ii < r.len && r.str[ii] == '\\')
            {
                const char next = ii+1 < r.len ? r.str[ii+1] : '\0';
                if(next == ' ' || next == '\t')
                {
                    _c4dbgfnl("extend skip to backslash{}", "");
                    ++ii;
                }
            }
        }
    }
    *i = ii - 1; // correct for the loop increment

    #undef _c4dbgfnl

    return replaced;
}


//-----------------------------------------------------------------------------

template<bool keep_trailing_whitespace>
void Parser::_filter_ws(substr r, size_t *C4_RESTRICT i, size_t *C4_RESTRICT pos)
{
    // a debugging scaffold:
    #if 0
    #define _c4dbgfws(fmt, ...) _c4dbgpf("filt_nl[{}]: " fmt, *i, __VA_ARGS__)
    #else
    #define _c4dbgfws(...)
    #endif

    const char curr = r[*i];
    _c4dbgfws("found whitespace '{}'", _c4prc(curr));
    _RYML_CB_ASSERT(m_stack.m_callbacks, curr == ' ' || curr == '\t');

    size_t first = *i > 0 ? r.first_not_of(" \t", *i) : r.first_not_of(' ', *i);
    if(first != npos)
    {
        if(r[first] == '\n' || r[first] == '\r') // skip trailing whitespace
        {
            _c4dbgfws("whitespace is trailing on line. firstnonws='{}'@{}", _c4prc(r[first]), first);
            *i = first - 1; // correct for the loop increment
        }
        else // a legit whitespace
        {
            m_filter_arena.str[(*pos)++] = curr;
            _c4dbgfws("legit whitespace. sofar=[{}]~~~{}~~~", *pos, m_filter_arena.first(*pos));
        }
    }
    else
    {
        _c4dbgfws("... everything else is trailing whitespace{}", "");
        if C4_IF_CONSTEXPR (keep_trailing_whitespace)
            for(size_t j = *i; j < r.len; ++j)
                m_filter_arena.str[(*pos)++] = r[j];
        *i = r.len;
    }

    #undef _c4dbgfws
}


//-----------------------------------------------------------------------------
csubstr Parser::_filter_plain_scalar(substr s, size_t indentation)
{
    // a debugging scaffold:
    #if 0
    #define _c4dbgfps(...) _c4dbgpf("filt_plain_scalar" __VA_ARGS__)
    #else
    #define _c4dbgfps(...)
    #endif

    _c4dbgfps("before=~~~{}~~~", s);

    substr r = s.triml(" \t");
    _grow_filter_arena(r.len);
    size_t pos = 0; // the filtered size
    bool filtered_chars = false;
    for(size_t i = 0; i < r.len; ++i)
    {
        const char curr = r.str[i];
        _c4dbgfps("[{}]: '{}'", i, _c4prc(curr));
        if(curr == ' ' || curr == '\t')
        {
            _filter_ws</*keep_trailing_ws*/false>(r, &i, &pos);
        }
        else if(curr == '\n')
        {
            filtered_chars = _filter_nl</*backslash_is_escape*/false, /*keep_trailing_ws*/false>(r, &i, &pos, indentation);
        }
        else if(curr == '\r')  // skip \r --- https://stackoverflow.com/questions/1885900
        {
            ;
        }
        else
        {
            m_filter_arena.str[pos++] = r[i];
        }
    }

    _RYML_CB_ASSERT(m_stack.m_callbacks, pos <= m_filter_arena.len);
    if(pos < r.len || filtered_chars)
    {
        r = _finish_filter_arena(r, pos);
    }

    _RYML_CB_ASSERT(m_stack.m_callbacks, s.len >= r.len);
    _c4dbgfps("#filteredchars={} after=~~~{}~~~", s.len - r.len, r);

    #undef _c4dbgfps
    return r;
}


//-----------------------------------------------------------------------------
csubstr Parser::_filter_squot_scalar(substr s)
{
    // a debugging scaffold:
    #if 0
    #define _c4dbgfsq(...) _c4dbgpf("filt_squo_scalar")
    #else
    #define _c4dbgfsq(...)
    #endif

    // from the YAML spec for double-quoted scalars:
    // https://yaml.org/spec/1.2-old/spec.html#style/flow/single-quoted

    _c4dbgfsq(": before=~~~{}~~~", s);

    _grow_filter_arena(s.len);
    substr r = s;
    size_t pos = 0; // the filtered size
    bool filtered_chars = false;
    for(size_t i = 0; i < r.len; ++i)
    {
        const char curr = r[i];
        _c4dbgfsq("[{}]: '{}'", i, _c4prc(curr));
        if(curr == ' ' || curr == '\t')
        {
            _filter_ws</*keep_trailing_ws*/true>(r, &i, &pos);
        }
        else if(curr == '\n')
        {
            filtered_chars = _filter_nl</*backslash_is_escape*/false, /*keep_trailing_ws*/true>(r, &i, &pos, /*indentation*/0);
        }
        else if(curr == '\r')  // skip \r --- https://stackoverflow.com/questions/1885900
        {
            ;
        }
        else if(curr == '\'')
        {
            char next = i+1 < r.len ? r[i+1] : '\0';
            if(next == '\'')
            {
                _c4dbgfsq("[{}]: two consecutive quotes", i);
                filtered_chars = true;
                m_filter_arena.str[pos++] = '\'';
                ++i;
            }
        }
        else
        {
            m_filter_arena.str[pos++] = curr;
        }
    }

    _RYML_CB_ASSERT(m_stack.m_callbacks, pos <= m_filter_arena.len);
    if(pos < r.len || filtered_chars)
    {
        r = _finish_filter_arena(r, pos);
    }

    _RYML_CB_ASSERT(m_stack.m_callbacks, s.len >= r.len);
    _c4dbgpf(": #filteredchars={} after=~~~{}~~~", s.len - r.len, r);

    #undef _c4dbgfsq
    return r;
}


//-----------------------------------------------------------------------------
csubstr Parser::_filter_dquot_scalar(substr s)
{
    // a debugging scaffold:
    #if 0
    #define _c4dbgfdq(...) _c4dbgpf("filt_dquo_scalar" __VA_ARGS__)
    #else
    #define _c4dbgfdq(...)
    #endif

    _c4dbgfdq(": before=~~~{}~~~", s);

    // from the YAML spec for double-quoted scalars:
    // https://yaml.org/spec/1.2-old/spec.html#style/flow/double-quoted
    //
    // All leading and trailing white space characters are excluded
    // from the content. Each continuation line must therefore contain
    // at least one non-space character. Empty lines, if any, are
    // consumed as part of the line folding.

    _grow_filter_arena(s.len + 2u * s.count('\\'));
    substr r = s;
    size_t pos = 0; // the filtered size
    bool filtered_chars = false;
    for(size_t i = 0; i < r.len; ++i)
    {
        const char curr = r[i];
        _c4dbgfdq("[{}]: '{}'", i, _c4prc(curr));
        if(curr == ' ' || curr == '\t')
        {
            _filter_ws</*keep_trailing_ws*/true>(r, &i, &pos);
        }
        else if(curr == '\n')
        {
            filtered_chars = _filter_nl</*backslash_is_escape*/true, /*keep_trailing_ws*/true>(r, &i, &pos, /*indentation*/0);
        }
        else if(curr == '\r')  // skip \r --- https://stackoverflow.com/questions/1885900
        {
            ;
        }
        else if(curr == '\\')
        {
            char next = i+1 < r.len ? r[i+1] : '\0';
            _c4dbgfdq("[{}]: backslash, next='{}'", i, _c4prc(next));
            filtered_chars = true;
            if(next == '\r')
            {
                if(i+2 < r.len && r[i+2] == '\n')
                {
                    ++i; // newline escaped with \ -- skip both (add only one as i is loop-incremented)
                    next = '\n';
                    _c4dbgfdq("[{}]: was \\r\\n, now next='\\n'", i);
                }
            }
            // remember the loop will also increment i
            if(next == '\n')
            {
                size_t ii = i + 2;
                for( ; ii < r.len; ++ii)
                {
                    if(r.str[ii] == ' ' || r.str[ii] == '\t')  // skip leading whitespace
                        ;
                    else
                        break;
                }
                i += ii - i - 1;
            }
            else if(next == '"' || next == '/'  || next == ' ' || next == '\t') // escapes for json compatibility
            {
                m_filter_arena.str[pos++] = next;
                ++i;
            }
            else if(next == '\r')
            {
                //++i;
            }
            else if(next == 'n')
            {
                m_filter_arena.str[pos++] = '\n';
                ++i;
            }
            else if(next == 'r')
            {
                m_filter_arena.str[pos++] = '\r';
                ++i; // skip
            }
            else if(next == 't')
            {
                m_filter_arena.str[pos++] = '\t';
                ++i;
            }
            else if(next == '\\')
            {
                m_filter_arena.str[pos++] = '\\';
                ++i;
            }
            else if(next == 'x') // UTF8
            {
                if(i + 1u + 2u >= r.len)
                    _c4err("\\x requires 2 hex digits");
                uint8_t byteval = {};
                if(!read_hex(r.sub(i + 2u, 2u), &byteval))
                    _c4err("failed to read \\x codepoint");
                m_filter_arena.str[pos++] = *(char*)&byteval;
                i += 1u + 2u;
            }
            else if(next == 'u') // UTF16
            {
                if(i + 1u + 4u >= r.len)
                    _c4err("\\u requires 4 hex digits");
                char readbuf[8];
                csubstr codepoint = r.sub(i + 2u, 4u);
                uint32_t codepoint_val = {};
                if(!read_hex(codepoint, &codepoint_val))
                    _c4err("failed to parse \\u codepoint");
                size_t numbytes = decode_code_point((uint8_t*)readbuf, sizeof(readbuf), codepoint_val);
                C4_ASSERT(numbytes <= 4);
                memcpy(m_filter_arena.str + pos, readbuf, numbytes);
                pos += numbytes;
                i += 1u + 4u;
            }
            else if(next == 'U') // UTF32
            {
                if(i + 1u + 8u >= r.len)
                    _c4err("\\U requires 8 hex digits");
                char readbuf[8];
                csubstr codepoint = r.sub(i + 2u, 8u);
                uint32_t codepoint_val = {};
                if(!read_hex(codepoint, &codepoint_val))
                    _c4err("failed to parse \\U codepoint");
                size_t numbytes = decode_code_point((uint8_t*)readbuf, sizeof(readbuf), codepoint_val);
                C4_ASSERT(numbytes <= 4);
                memcpy(m_filter_arena.str + pos, readbuf, numbytes);
                pos += numbytes;
                i += 1u + 8u;
            }
            // https://yaml.org/spec/1.2.2/#rule-c-ns-esc-char
            else if(next == '0')
            {
                m_filter_arena.str[pos++] = '\0';
                ++i;
            }
            else if(next == 'b') // backspace
            {
                m_filter_arena.str[pos++] = '\b';
                ++i;
            }
            else if(next == 'f') // form feed
            {
                m_filter_arena.str[pos++] = '\f';
                ++i;
            }
            else if(next == 'a') // bell character
            {
                m_filter_arena.str[pos++] = '\a';
                ++i;
            }
            else if(next == 'v') // vertical tab
            {
                m_filter_arena.str[pos++] = '\v';
                ++i;
            }
            else if(next == 'e') // escape character
            {
                m_filter_arena.str[pos++] = '\x1b';
                ++i;
            }
            else if(next == '_') // unicode non breaking space \u00a0
            {
                // https://www.compart.com/en/unicode/U+00a0
                m_filter_arena.str[pos++] = _RYML_CHCONST(-0x3e, 0xc2);
                m_filter_arena.str[pos++] = _RYML_CHCONST(-0x60, 0xa0);
                ++i;
            }
            else if(next == 'N') // unicode next line \u0085
            {
                // https://www.compart.com/en/unicode/U+0085
                m_filter_arena.str[pos++] = _RYML_CHCONST(-0x3e, 0xc2);
                m_filter_arena.str[pos++] = _RYML_CHCONST(-0x7b, 0x85);
                ++i;
            }
            else if(next == 'L') // unicode line separator \u2028
            {
                // https://www.utf8-chartable.de/unicode-utf8-table.pl?start=8192&number=1024&names=-&utf8=0x&unicodeinhtml=hex
                m_filter_arena.str[pos++] = _RYML_CHCONST(-0x1e, 0xe2);
                m_filter_arena.str[pos++] = _RYML_CHCONST(-0x80, 0x80);
                m_filter_arena.str[pos++] = _RYML_CHCONST(-0x58, 0xa8);
                ++i;
            }
            else if(next == 'P') // unicode paragraph separator \u2029
            {
                // https://www.utf8-chartable.de/unicode-utf8-table.pl?start=8192&number=1024&names=-&utf8=0x&unicodeinhtml=hex
                m_filter_arena.str[pos++] = _RYML_CHCONST(-0x1e, 0xe2);
                m_filter_arena.str[pos++] = _RYML_CHCONST(-0x80, 0x80);
                m_filter_arena.str[pos++] = _RYML_CHCONST(-0x57, 0xa9);
                ++i;
            }
            _c4dbgfdq("[{}]: backslash...sofar=[{}]~~~{}~~~", i, pos, m_filter_arena.first(pos));
        }
        else
        {
            m_filter_arena.str[pos++] = curr;
        }
    }

    _RYML_CB_ASSERT(m_stack.m_callbacks, pos <= m_filter_arena.len);
    if(pos < r.len || filtered_chars)
    {
        r = _finish_filter_arena(r, pos);
    }

    _RYML_CB_ASSERT(m_stack.m_callbacks, s.len >= r.len);
    _c4dbgpf(": #filteredchars={} after=~~~{}~~~", s.len - r.len, r);

    #undef _c4dbgfdq

    return r;
}


//-----------------------------------------------------------------------------
bool Parser::_apply_chomp(substr buf, size_t *C4_RESTRICT pos, BlockChomp_e chomp)
{
    substr trimmed = buf.first(*pos).trimr('\n');
    bool added_newline = false;
    switch(chomp)
    {
    case CHOMP_KEEP:
        if(trimmed.len == *pos)
        {
            _c4dbgpf("chomp=KEEP: add missing newline @{}", *pos);
            //m_filter_arena.str[(*pos)++] = '\n';
            added_newline = true;
        }
        break;
    case CHOMP_CLIP:
        if(trimmed.len == *pos)
        {
            _c4dbgpf("chomp=CLIP: add missing newline @{}", *pos);
            m_filter_arena.str[(*pos)++] = '\n';
            added_newline = true;
        }
        else
        {
            _c4dbgpf("chomp=CLIP: include single trailing newline @{}", trimmed.len+1);
            *pos = trimmed.len + 1;
        }
        break;
    case CHOMP_STRIP:
        _c4dbgpf("chomp=STRIP: strip {}-{}-{} newlines", *pos, trimmed.len, *pos-trimmed.len);
        *pos = trimmed.len;
        break;
    default:
        _c4err("unknown chomp style");
    }
    return added_newline;
}


//-----------------------------------------------------------------------------
csubstr Parser::_filter_block_scalar(substr s, BlockStyle_e style, BlockChomp_e chomp, size_t indentation)
{
    // a debugging scaffold:
    #if 0
    #define _c4dbgfbl(fmt, ...) _c4dbgpf("filt_block" fmt, __VA_ARGS__)
    #else
    #define _c4dbgfbl(...)
    #endif

    _c4dbgfbl(": indentation={} before=[{}]~~~{}~~~", indentation, s.len, s);

    if(chomp != CHOMP_KEEP && s.trim(" \n\r").len == 0u)
    {
        _c4dbgp("filt_block: empty scalar");
        return s.first(0);
    }

    substr r = s;

    switch(style)
    {
    case BLOCK_LITERAL:
        {
            _c4dbgp("filt_block: style=literal");
            // trim leading whitespace up to indentation
            {
                size_t numws = r.first_not_of(' ');
                if(numws != npos)
                {
                    if(numws > indentation)
                        r = r.sub(indentation);
                    else
                        r = r.sub(numws);
                    _c4dbgfbl(": after triml=[{}]~~~{}~~~", r.len, r);
                }
                else
                {
                    if(chomp != CHOMP_KEEP || r.len == 0)
                    {
                        _c4dbgfbl(": all spaces {}, return empty", r.len);
                        return r.first(0);
                    }
                    else
                    {
                        r[0] = '\n';
                        return r.first(1);
                    }
                }
            }
            _grow_filter_arena(s.len + 2u);  // use s.len! because we may need to add a newline at the end, so the leading indentation will allow space for that newline
            size_t pos = 0; // the filtered size
            for(size_t i = 0; i < r.len; ++i)
            {
                const char curr = r.str[i];
                _c4dbgfbl("[{}]='{}'  pos={}", i, _c4prc(curr), pos);
                if(curr == '\r')
                    continue;
                m_filter_arena.str[pos++] = curr;
                if(curr == '\n')
                {
                    _c4dbgfbl("[{}]: found newline", i);
                    // skip indentation on the next line
                    csubstr rem = r.sub(i+1);
                    size_t first = rem.first_not_of(' ');
                    if(first != npos)
                    {
                        _RYML_CB_ASSERT(m_stack.m_callbacks, first < rem.len);
                        _RYML_CB_ASSERT(m_stack.m_callbacks, i+1+first < r.len);
                        _c4dbgfbl("[{}]: {} spaces follow before next nonws character @ [{}]='{}'", i, first, i+1+first, rem.str[first]);
                        if(first < indentation)
                        {
                            _c4dbgfbl("[{}]: skip {}<{} spaces from indentation", i, first, indentation);
                            i += first;
                        }
                        else
                        {
                            _c4dbgfbl("[{}]: skip {} spaces from indentation", i, indentation);
                            i += indentation;
                        }
                    }
                    else
                    {
                        _RYML_CB_ASSERT(m_stack.m_callbacks, i+1 <= r.len);
                        first = rem.len;
                        _c4dbgfbl("[{}]: {} spaces to the end", i, first);
                        if(first)
                        {
                            if(first < indentation)
                            {
                                _c4dbgfbl("[{}]: skip everything", i);
                                --pos;
                                break;
                            }
                            else
                            {
                                _c4dbgfbl("[{}]: skip {} spaces from indentation", i, indentation);
                                i += indentation;
                            }
                        }
                        else if(i+1 == r.len)
                        {
                            if(chomp == CHOMP_STRIP)
                                --pos;
                            break;
                        }
                    }
                }
            }
            _RYML_CB_ASSERT(m_stack.m_callbacks, s.len >= pos);
            _c4dbgfbl(": #filteredchars={} after=~~~{}~~~", s.len - r.len, r);
            bool changed = _apply_chomp(m_filter_arena, &pos, chomp);
            _RYML_CB_ASSERT(m_stack.m_callbacks, pos <= m_filter_arena.len);
            _RYML_CB_ASSERT(m_stack.m_callbacks, pos <= s.len);
            if(pos < r.len || changed)
            {
                r = _finish_filter_arena(s, pos); // write into s
            }
            break;
        }
    case BLOCK_FOLD:
        {
            _c4dbgp("filt_block: style=fold");
            _grow_filter_arena(r.len + 2);
            size_t pos = 0; // the filtered size
            bool filtered_chars = false;
            bool started = false;
            bool is_indented = false;
            size_t i = r.first_not_of(' ');
            _c4dbgfbl(": first non space at {}", i);
            if(i > indentation)
            {
                is_indented = true;
                i = indentation;
            }
            _c4dbgfbl(": start folding at {}, is_indented={}", i, (int)is_indented);
            auto on_change_indentation = [&](size_t numnl_following, size_t last_newl, size_t first_non_whitespace){
                _c4dbgfbl("[{}]: add 1+{} newlines", i, numnl_following);
                for(size_t j = 0; j < 1 + numnl_following; ++j)
                    m_filter_arena.str[pos++] = '\n';
                for(i = last_newl + 1 + indentation; i < first_non_whitespace; ++i)
                {
                    if(r.str[i] == '\r')
                        continue;
                    _c4dbgfbl("[{}]: add '{}'", i, _c4prc(r.str[i]));
                    m_filter_arena.str[pos++] = r.str[i];
                }
                --i;
            };
            for( ; i < r.len; ++i)
            {
                const char curr = r.str[i];
                _c4dbgfbl("[{}]='{}'", i, _c4prc(curr));
                if(curr == '\n')
                {
                    filtered_chars = true;
                    // skip indentation on the next line, and advance over the next non-indented blank lines as well
                    size_t first_non_whitespace;
                    size_t numnl_following = (size_t)-1;
                    while(r[i] == '\n')
                    {
                        ++numnl_following;
                        csubstr rem = r.sub(i+1);
                        size_t first = rem.first_not_of(' ');
                        _c4dbgfbl("[{}]: found newline. first={} rem.len={}", i, first, rem.len);
                        if(first != npos)
                        {
                            first_non_whitespace = first + i+1;
                            while(first_non_whitespace < r.len && r[first_non_whitespace] == '\r')
                                ++first_non_whitespace;
                            _RYML_CB_ASSERT(m_stack.m_callbacks, first < rem.len);
                            _RYML_CB_ASSERT(m_stack.m_callbacks, i+1+first < r.len);
                            _c4dbgfbl("[{}]: {} spaces follow before next nonws character @ [{}]='{}'", i, first, i+1+first, _c4prc(rem.str[first]));
                            if(first < indentation)
                            {
                                _c4dbgfbl("[{}]: skip {}<{} spaces from indentation", i, first, indentation);
                                i += first;
                            }
                            else
                            {
                                _c4dbgfbl("[{}]: skip {} spaces from indentation", i, indentation);
                                i += indentation;
                                if(first > indentation)
                                {
                                    _c4dbgfbl("[{}]: {} further indented than {}, stop newlining", i, first, indentation);
                                    goto finished_counting_newlines;
                                }
                            }
                            // prepare the next while loop iteration
                            // by setting i at the next newline after
                            // an empty line
                            if(r[first_non_whitespace] == '\n')
                                i = first_non_whitespace;
                            else
                                goto finished_counting_newlines;
                        }
                        else
                        {
                            _RYML_CB_ASSERT(m_stack.m_callbacks, i+1 <= r.len);
                            first = rem.len;
                            first_non_whitespace = first + i+1;
                            if(first)
                            {
                                _c4dbgfbl("[{}]: {} spaces to the end", i, first);
                                if(first < indentation)
                                {
                                    _c4dbgfbl("[{}]: skip everything", i);
                                    i += first;
                                }
                                else
                                {
                                    _c4dbgfbl("[{}]: skip {} spaces from indentation", i, indentation);
                                    i += indentation;
                                    if(first > indentation)
                                    {
                                        _c4dbgfbl("[{}]: {} spaces missing. not done yet", i, indentation - first);
                                        goto finished_counting_newlines;
                                    }
                                }
                            }
                            else // if(i+1 == r.len)
                            {
                                _c4dbgfbl("[{}]: it's the final newline", i);
                                _RYML_CB_ASSERT(m_stack.m_callbacks, i+1 == r.len);
                                _RYML_CB_ASSERT(m_stack.m_callbacks, rem.len == 0);
                            }
                            goto end_of_scalar;
                        }
                    }
                end_of_scalar:
                    // Write all the trailing newlines. Since we're
                    // at the end no folding is needed, so write every
                    // newline (add 1).
                    _c4dbgfbl("[{}]: add {} trailing newlines", i, 1+numnl_following);
                    for(size_t j = 0; j < 1 + numnl_following; ++j)
                        m_filter_arena.str[pos++] = '\n';
                    break;
                finished_counting_newlines:
                    _c4dbgfbl("[{}]: #newlines={} firstnonws={}", i, numnl_following, first_non_whitespace);
                    while(first_non_whitespace < r.len && r[first_non_whitespace] == '\t')
                        ++first_non_whitespace;
                    _c4dbgfbl("[{}]: #newlines={} firstnonws={}", i, numnl_following, first_non_whitespace);
                    _RYML_CB_ASSERT(m_stack.m_callbacks, first_non_whitespace <= r.len);
                    size_t last_newl = r.last_of('\n', first_non_whitespace);
                    size_t this_indentation = first_non_whitespace - last_newl - 1;
                    _c4dbgfbl("[{}]: #newlines={} firstnonws={} lastnewl={} this_indentation={} vs indentation={}", i, numnl_following, first_non_whitespace, last_newl, this_indentation, indentation);
                    _RYML_CB_ASSERT(m_stack.m_callbacks, first_non_whitespace >= last_newl + 1);
                    _RYML_CB_ASSERT(m_stack.m_callbacks, this_indentation >= indentation);
                    if(!started)
                    {
                        _c4dbgfbl("[{}]: #newlines={}. write all leading newlines", i, numnl_following);
                        for(size_t j = 0; j < 1 + numnl_following; ++j)
                            m_filter_arena.str[pos++] = '\n';
                        if(this_indentation > indentation)
                        {
                            is_indented = true;
                            _c4dbgfbl("[{}]: advance ->{}", i, last_newl + indentation);
                            i = last_newl + indentation;
                        }
                        else
                        {
                            i = first_non_whitespace - 1;
                            _c4dbgfbl("[{}]: advance ->{}", i, first_non_whitespace);
                        }
                    }
                    else if(this_indentation == indentation)
                    {
                        _c4dbgfbl("[{}]: same indentation", i);
                        if(!is_indented)
                        {
                            if(numnl_following == 0)
                            {
                                _c4dbgfbl("[{}]: fold!", i);
                                m_filter_arena.str[pos++] = ' ';
                            }
                            else
                            {
                                _c4dbgfbl("[{}]: add {} newlines", i, 1 + numnl_following);
                                for(size_t j = 0; j < numnl_following; ++j)
                                    m_filter_arena.str[pos++] = '\n';
                            }
                            i = first_non_whitespace - 1;
                            _c4dbgfbl("[{}]: advance {}->{}", i, i, first_non_whitespace);
                        }
                        else
                        {
                            _c4dbgfbl("[{}]: back to ref indentation", i);
                            is_indented = false;
                            on_change_indentation(numnl_following, last_newl, first_non_whitespace);
                            _c4dbgfbl("[{}]: advance {}->{}", i, i, first_non_whitespace);
                        }
                    }
                    else
                    {
                        _c4dbgfbl("[{}]: increased indentation.", i);
                        is_indented = true;
                        _RYML_CB_ASSERT(m_stack.m_callbacks, this_indentation > indentation);
                        on_change_indentation(numnl_following, last_newl, first_non_whitespace);
                        _c4dbgfbl("[{}]: advance {}->{}", i, i, first_non_whitespace);
                    }
                }
                else if(curr != '\r')
                {
                    if(curr != '\t')
                        started = true;
                    m_filter_arena.str[pos++] = curr;
                }
            }
            _RYML_CB_ASSERT(m_stack.m_callbacks, pos <= m_filter_arena.len);
            _c4dbgfbl(": #filteredchars={} after=[{}]~~~{}~~~", (int)s.len - (int)pos, pos, m_filter_arena.first(pos));
            bool changed = _apply_chomp(m_filter_arena, &pos, chomp);
            if(pos < r.len || filtered_chars || changed)
            {
                r = _finish_filter_arena(s, pos); // write into s
            }
        }
        break;
    default:
        _c4err("unknown block style");
    }

    _c4dbgfbl(": final=[{}]~~~{}~~~", r.len, r);

    #undef _c4dbgfbl

    return r;
}

//-----------------------------------------------------------------------------
size_t Parser::_count_nlines(csubstr src)
{
    return 1 + src.count('\n');
}

//-----------------------------------------------------------------------------
void Parser::_handle_directive(csubstr directive_)
{
    csubstr directive = directive_;
    if(directive.begins_with("%TAG"))
    {
        TagDirective td;
        _c4dbgpf("%TAG directive: {}", directive_);
        directive = directive.sub(4);
        if(!directive.begins_with(' '))
            _c4err("malformed tag directive: {}", directive_);
        directive = directive.triml(' ');
        size_t pos = directive.find(' ');
        if(pos == npos)
            _c4err("malformed tag directive: {}", directive_);
        td.handle = directive.first(pos);
        directive = directive.sub(td.handle.len).triml(' ');
        pos = directive.find(' ');
        if(pos != npos)
            directive = directive.first(pos);
        td.prefix = directive;
        td.next_node_id = m_tree->size();
        if(m_tree->size() > 0)
        {
            size_t prev = m_tree->size() - 1;
            if(m_tree->is_root(prev) && m_tree->type(prev) != NOTYPE && !m_tree->is_stream(prev))
                ++td.next_node_id;
        }
        _c4dbgpf("%TAG: handle={} prefix={} next_node={}", td.handle, td.prefix, td.next_node_id);
        m_tree->add_tag_directive(td);
    }
    else if(directive.begins_with("%YAML"))
    {
        _c4dbgpf("%YAML directive! ignoring...: {}", directive);
    }
}

//-----------------------------------------------------------------------------
void Parser::set_flags(flag_t f, State * s)
{
#ifdef RYML_DBG
    char buf1_[64], buf2_[64];
    csubstr buf1 = _prfl(buf1_, f);
    csubstr buf2 = _prfl(buf2_, s->flags);
    _c4dbgpf("state[{}]: setting flags to {}: before={}", s-m_stack.begin(), buf1, buf2);
#endif
    s->flags = f;
}

void Parser::add_flags(flag_t on, State * s)
{
#ifdef RYML_DBG
    char buf1_[64], buf2_[64], buf3_[64];
    csubstr buf1 = _prfl(buf1_, on);
    csubstr buf2 = _prfl(buf2_, s->flags);
    csubstr buf3 = _prfl(buf3_, s->flags|on);
    _c4dbgpf("state[{}]: adding flags {}: before={} after={}", s-m_stack.begin(), buf1, buf2, buf3);
#endif
    s->flags |= on;
}

void Parser::addrem_flags(flag_t on, flag_t off, State * s)
{
#ifdef RYML_DBG
    char buf1_[64], buf2_[64], buf3_[64], buf4_[64];
    csubstr buf1 = _prfl(buf1_, on);
    csubstr buf2 = _prfl(buf2_, off);
    csubstr buf3 = _prfl(buf3_, s->flags);
    csubstr buf4 = _prfl(buf4_, ((s->flags|on)&(~off)));
    _c4dbgpf("state[{}]: adding flags {} / removing flags {}: before={} after={}", s-m_stack.begin(), buf1, buf2, buf3, buf4);
#endif
    s->flags |= on;
    s->flags &= ~off;
}

void Parser::rem_flags(flag_t off, State * s)
{
#ifdef RYML_DBG
    char buf1_[64], buf2_[64], buf3_[64];
    csubstr buf1 = _prfl(buf1_, off);
    csubstr buf2 = _prfl(buf2_, s->flags);
    csubstr buf3 = _prfl(buf3_, s->flags&(~off));
    _c4dbgpf("state[{}]: removing flags {}: before={} after={}", s-m_stack.begin(), buf1, buf2, buf3);
#endif
    s->flags &= ~off;
}

//-----------------------------------------------------------------------------

csubstr Parser::_prfl(substr buf, flag_t flags)
{
    size_t pos = 0;
    bool gotone = false;

    #define _prflag(fl)                                     \
    if((flags & fl) == (fl))                                \
    {                                                       \
        if(gotone)                                          \
        {                                                   \
            if(pos + 1 < buf.len)                           \
                buf[pos] = '|';                             \
            ++pos;                                          \
        }                                                   \
        csubstr fltxt = #fl;                                \
        if(pos + fltxt.len <= buf.len)                      \
            memcpy(buf.str + pos, fltxt.str, fltxt.len);    \
        pos += fltxt.len;                                   \
        gotone = true;                                      \
    }

    _prflag(RTOP);
    _prflag(RUNK);
    _prflag(RMAP);
    _prflag(RSEQ);
    _prflag(FLOW);
    _prflag(QMRK);
    _prflag(RKEY);
    _prflag(RVAL);
    _prflag(RNXT);
    _prflag(SSCL);
    _prflag(QSCL);
    _prflag(RSET);
    _prflag(NDOC);
    _prflag(RSEQIMAP);

    #undef _prflag

    RYML_ASSERT(pos <= buf.len);

    return buf.first(pos);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

void Parser::_grow_filter_arena(size_t num_characters_needed)
{
    _c4dbgpf("grow: arena={} numchars={}", m_filter_arena.len, num_characters_needed);
    if(num_characters_needed <= m_filter_arena.len)
        return;
    size_t sz = m_filter_arena.len << 1;
    _c4dbgpf("grow: sz={}", sz);
    sz = num_characters_needed > sz ? num_characters_needed : sz;
    _c4dbgpf("grow: sz={}", sz);
    sz = sz < 128u ? 128u : sz;
    _c4dbgpf("grow: sz={}", sz);
    _RYML_CB_ASSERT(m_stack.m_callbacks, sz >= num_characters_needed);
    _resize_filter_arena(sz);
}

void Parser::_resize_filter_arena(size_t num_characters)
{
    if(num_characters > m_filter_arena.len)
    {
        _c4dbgpf("resize: sz={}", num_characters);
        char *prev = m_filter_arena.str;
        if(m_filter_arena.str)
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks, m_filter_arena.len > 0);
            _RYML_CB_FREE(m_stack.m_callbacks, m_filter_arena.str, char, m_filter_arena.len);
        }
        m_filter_arena.str = _RYML_CB_ALLOC_HINT(m_stack.m_callbacks, char, num_characters, prev);
        m_filter_arena.len = num_characters;
    }
}

substr Parser::_finish_filter_arena(substr dst, size_t pos)
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, pos <= m_filter_arena.len);
    _RYML_CB_ASSERT(m_stack.m_callbacks, pos <= dst.len);
    memcpy(dst.str, m_filter_arena.str, pos);
    return dst.first(pos);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

csubstr Parser::location_contents(Location const& loc) const
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, loc.offset < m_buf.len);
    return m_buf.sub(loc.offset);
}

Location Parser::location(ConstNodeRef node) const
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, node.valid());
    return location(*node.tree(), node.id());
}

Location Parser::location(Tree const& tree, size_t node) const
{
    // try hard to avoid getting the location from a null string.
    Location loc;
    if(_location_from_node(tree, node, &loc, 0))
        return loc;
    return val_location(m_buf.str);
}

bool Parser::_location_from_node(Tree const& tree, size_t node, Location *C4_RESTRICT loc, size_t level) const
{
    if(tree.has_key(node))
    {
        csubstr k = tree.key(node);
        if(C4_LIKELY(k.str != nullptr))
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks, k.is_sub(m_buf));
            _RYML_CB_ASSERT(m_stack.m_callbacks, m_buf.is_super(k));
            *loc = val_location(k.str);
            return true;
        }
    }

    if(tree.has_val(node))
    {
        csubstr v = tree.val(node);
        if(C4_LIKELY(v.str != nullptr))
        {
            _RYML_CB_ASSERT(m_stack.m_callbacks, v.is_sub(m_buf));
            _RYML_CB_ASSERT(m_stack.m_callbacks, m_buf.is_super(v));
            *loc = val_location(v.str);
            return true;
        }
    }

    if(tree.is_container(node))
    {
        if(_location_from_cont(tree, node, loc))
            return true;
    }

    if(tree.type(node) != NOTYPE && level == 0)
    {
        // try the prev sibling
        {
            const size_t prev = tree.prev_sibling(node);
            if(prev != NONE)
            {
                if(_location_from_node(tree, prev, loc, level+1))
                    return true;
            }
        }
        // try the next sibling
        {
            const size_t next = tree.next_sibling(node);
            if(next != NONE)
            {
                if(_location_from_node(tree, next, loc, level+1))
                    return true;
            }
        }
        // try the parent
        {
            const size_t parent = tree.parent(node);
            if(parent != NONE)
            {
                if(_location_from_node(tree, parent, loc, level+1))
                    return true;
            }
        }
    }

    return false;
}

bool Parser::_location_from_cont(Tree const& tree, size_t node, Location *C4_RESTRICT loc) const
{
    _RYML_CB_ASSERT(m_stack.m_callbacks, tree.is_container(node));
    if(!tree.is_stream(node))
    {
        const char *node_start = tree._p(node)->m_val.scalar.str;  // this was stored in the container
        if(tree.has_children(node))
        {
            size_t child = tree.first_child(node);
            if(tree.has_key(child))
            {
                // when a map starts, the container was set after the key
                csubstr k = tree.key(child);
                if(k.str && node_start > k.str)
                    node_start = k.str;
            }
        }
        *loc = val_location(node_start);
        return true;
    }
    else // it's a stream
    {
        *loc = val_location(m_buf.str); // just return the front of the buffer
    }
    return true;
}


Location Parser::val_location(const char *val) const
{
    if(C4_UNLIKELY(val == nullptr))
        return {m_file, 0, 0, 0};

    _RYML_CB_CHECK(m_stack.m_callbacks, m_options.locations());
    // NOTE: if any of these checks fails, the parser needs to be
    // instantiated with locations enabled.
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_buf.str == m_newline_offsets_buf.str);
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_buf.len == m_newline_offsets_buf.len);
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_options.locations());
    _RYML_CB_ASSERT(m_stack.m_callbacks, !_locations_dirty());
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_newline_offsets != nullptr);
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_newline_offsets_size > 0);
    // NOTE: the pointer needs to belong to the buffer that was used to parse.
    csubstr src = m_buf;
    _RYML_CB_CHECK(m_stack.m_callbacks, val != nullptr || src.str == nullptr);
    _RYML_CB_CHECK(m_stack.m_callbacks, (val >= src.begin() && val <= src.end()) || (src.str == nullptr && val == nullptr));
    // ok. search the first stored newline after the given ptr
    using lineptr_type = size_t const* C4_RESTRICT;
    lineptr_type lineptr = nullptr;
    size_t offset = (size_t)(val - src.begin());
    if(m_newline_offsets_size < 30) // TODO magic number
    {
        // just do a linear search if the size is small.
        for(lineptr_type curr = m_newline_offsets, last = m_newline_offsets + m_newline_offsets_size; curr < last; ++curr)
        {
            if(*curr > offset)
            {
                lineptr = curr;
                break;
            }
        }
    }
    else
    {
        // do a bisection search if the size is not small.
        //
        // We could use std::lower_bound but this is simple enough and
        // spares the include of <algorithm>.
        size_t count = m_newline_offsets_size;
        size_t step;
        lineptr_type it;
        lineptr = m_newline_offsets;
        while(count)
        {
            step = count >> 1;
            it = lineptr + step;
            if(*it < offset)
            {
                lineptr = ++it;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
    }
    _RYML_CB_ASSERT(m_stack.m_callbacks, lineptr >= m_newline_offsets);
    _RYML_CB_ASSERT(m_stack.m_callbacks, lineptr <= m_newline_offsets + m_newline_offsets_size);
    _RYML_CB_ASSERT(m_stack.m_callbacks, *lineptr > offset);
    Location loc;
    loc.name = m_file;
    loc.offset = offset;
    loc.line = (size_t)(lineptr - m_newline_offsets);
    if(lineptr > m_newline_offsets)
        loc.col = (offset - *(lineptr-1) - 1u);
    else
        loc.col = offset;
    return loc;
}

void Parser::_prepare_locations()
{
    m_newline_offsets_buf = m_buf;
    size_t numnewlines = 1u + m_buf.count('\n');
    _resize_locations(numnewlines);
    m_newline_offsets_size = 0;
    for(size_t i = 0; i < m_buf.len; i++)
        if(m_buf[i] == '\n')
            m_newline_offsets[m_newline_offsets_size++] = i;
    m_newline_offsets[m_newline_offsets_size++] = m_buf.len;
    _RYML_CB_ASSERT(m_stack.m_callbacks, m_newline_offsets_size == numnewlines);
}

void Parser::_resize_locations(size_t numnewlines)
{
    if(numnewlines > m_newline_offsets_capacity)
    {
        if(m_newline_offsets)
            _RYML_CB_FREE(m_stack.m_callbacks, m_newline_offsets, size_t, m_newline_offsets_capacity);
        m_newline_offsets = _RYML_CB_ALLOC_HINT(m_stack.m_callbacks, size_t, numnewlines, m_newline_offsets);
        m_newline_offsets_capacity = numnewlines;
    }
}

bool Parser::_locations_dirty() const
{
    return !m_newline_offsets_size;
}

} // namespace yml
} // namespace c4


#if defined(_MSC_VER)
#   pragma warning(pop)
#elif defined(__clang__)
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#endif
