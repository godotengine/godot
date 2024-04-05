#ifndef _C4_YML_PARSE_HPP_
#define _C4_YML_PARSE_HPP_

#ifndef _C4_YML_TREE_HPP_
#include "tree.hpp"
#endif

#ifndef _C4_YML_NODE_HPP_
#include "node.hpp"
#endif

#ifndef _C4_YML_DETAIL_STACK_HPP_
#include "detail/stack.hpp"
#endif

#include <stdarg.h>

#if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable: 4251/*needs to have dll-interface to be used by clients of struct*/)
#endif

namespace c4 {
namespace yml {

struct RYML_EXPORT ParserOptions
{
private:

    typedef enum : uint32_t {
        LOCATIONS = (1 << 0),
        DEFAULTS = 0,
    } Flags_e;

    uint32_t flags = DEFAULTS;
public:
    ParserOptions() = default;

    /** @name source location tracking */
    /** @{ */

    /** enable/disable source location tracking */
    ParserOptions& locations(bool enabled)
    {
        if(enabled)
            flags |= LOCATIONS;
        else
            flags &= ~LOCATIONS;
        return *this;
    }
    bool locations() const { return (flags & LOCATIONS) != 0u; }

    /** @} */
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
class RYML_EXPORT Parser
{
public:

    /** @name construction and assignment */
    /** @{ */

    Parser(Callbacks const& cb, ParserOptions opts={});
    Parser(ParserOptions opts={}) : Parser(get_callbacks(), opts) {}
    ~Parser();

    Parser(Parser &&);
    Parser(Parser const&);
    Parser& operator=(Parser &&);
    Parser& operator=(Parser const&);

    /** @} */

public:

    /** @name modifiers */
    /** @{ */

    /** Reserve a certain capacity for the parsing stack.
     * This should be larger than the expected depth of the parsed
     * YAML tree.
     *
     * The parsing stack is the only (potential) heap memory used by
     * the parser.
     *
     * If the requested capacity is below the default
     * stack size of 16, the memory is used directly in the parser
     * object; otherwise it will be allocated from the heap.
     *
     * @note this reserves memory only for the parser itself; all the
     * allocations for the parsed tree will go through the tree's
     * allocator.
     *
     * @note the tree and the arena can (and should) also be reserved. */
    void reserve_stack(size_t capacity)
    {
        m_stack.reserve(capacity);
    }

    /** Reserve a certain capacity for the array used to track node
     * locations in the source buffer. */
    void reserve_locations(size_t num_source_lines)
    {
        _resize_locations(num_source_lines);
    }

    /** Reserve a certain capacity for the character arena used to
     * filter scalars. */
    void reserve_filter_arena(size_t num_characters)
    {
        _resize_filter_arena(num_characters);
    }

    /** @} */

public:

    /** @name getters and modifiers */
    /** @{ */

    /** Get the current callbacks in the parser. */
    Callbacks callbacks() const { return m_stack.m_callbacks; }

    /** Get the name of the latest file parsed by this object. */
    csubstr filename() const { return m_file; }

    /** Get the latest YAML buffer parsed by this object. */
    csubstr source() const { return m_buf; }

    size_t stack_capacity() const { return m_stack.capacity(); }
    size_t locations_capacity() const { return m_newline_offsets_capacity; }
    size_t filter_arena_capacity() const { return m_filter_arena.len; }

    ParserOptions const& options() const { return m_options; }

    /** @} */

public:

    /** @name parse_in_place */
    /** @{ */

    /** Create a new tree and parse into its root.
     * The tree is created with the callbacks currently in the parser. */
    Tree parse_in_place(csubstr filename, substr src)
    {
        Tree t(callbacks());
        t.reserve(_estimate_capacity(src));
        this->parse_in_place(filename, src, &t, t.root_id());
        return t;
    }

    /** Parse into an existing tree, starting at its root node.
     * The callbacks in the tree are kept, and used to allocate
     * the tree members, if any allocation is required. */
    void parse_in_place(csubstr filename, substr src, Tree *t)
    {
        this->parse_in_place(filename, src, t, t->root_id());
    }

    /** Parse into an existing node.
     * The callbacks in the tree are kept, and used to allocate
     * the tree members, if any allocation is required. */
    void parse_in_place(csubstr filename, substr src, Tree *t, size_t node_id);
    //   ^^^^^^^^^^^^^ this is the workhorse overload; everything else is syntactic candy

    /** Parse into an existing node.
     * The callbacks in the tree are kept, and used to allocate
     * the tree members, if any allocation is required. */
    void parse_in_place(csubstr filename, substr src, NodeRef node)
    {
        this->parse_in_place(filename, src, node.tree(), node.id());
    }

    RYML_DEPRECATED("use parse_in_place() instead") Tree parse(csubstr filename, substr src) { return parse_in_place(filename, src); }
    RYML_DEPRECATED("use parse_in_place() instead") void parse(csubstr filename, substr src, Tree *t) { parse_in_place(filename, src, t); }
    RYML_DEPRECATED("use parse_in_place() instead") void parse(csubstr filename, substr src, Tree *t, size_t node_id) { parse_in_place(filename, src, t, node_id); }
    RYML_DEPRECATED("use parse_in_place() instead") void parse(csubstr filename, substr src, NodeRef node) { parse_in_place(filename, src, node); }

    /** @} */

public:

    /** @name parse_in_arena: copy the YAML source buffer to the
     * tree's arena, then parse the copy in situ
     *
     * @note overloads receiving a substr YAML buffer are intentionally
     * left undefined, such that calling parse_in_arena() with a substr
     * will cause a linker error. This is to prevent an accidental
     * copy of the source buffer to the tree's arena, because substr
     * is implicitly convertible to csubstr. If you really intend to parse
     * a mutable buffer in the tree's arena, convert it first to immutable
     * by assigning the substr to a csubstr prior to calling parse_in_arena().
     * This is not needed for parse_in_place() because csubstr is not
     * implicitly convertible to substr. */
    /** @{ */

    // READ THE NOTE ABOVE!
    #define RYML_DONT_PARSE_SUBSTR_IN_ARENA "Do not pass a (mutable) substr to parse_in_arena(); if you have a substr, it should be parsed in place. Consider using parse_in_place() instead, or convert the buffer to csubstr prior to calling. This function is deliberately left undefined and will cause a linker error."
    RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) Tree parse_in_arena(csubstr filename, substr csrc);
    RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) void parse_in_arena(csubstr filename, substr csrc, Tree *t);
    RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) void parse_in_arena(csubstr filename, substr csrc, Tree *t, size_t node_id);
    RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) void parse_in_arena(csubstr filename, substr csrc, NodeRef node);

    /** Create a new tree and parse into its root.
     * The immutable YAML source is first copied to the tree's arena,
     * and parsed from there.
     * The callbacks in the tree are kept, and used to allocate
     * the tree members, if any allocation is required. */
    Tree parse_in_arena(csubstr filename, csubstr csrc)
    {
        Tree t(callbacks());
        substr src = t.copy_to_arena(csrc);
        t.reserve(_estimate_capacity(csrc));
        this->parse_in_place(filename, src, &t, t.root_id());
        return t;
    }

    /** Parse into an existing tree, starting at its root node.
     * The immutable YAML source is first copied to the tree's arena,
     * and parsed from there.
     * The callbacks in the tree are kept, and used to allocate
     * the tree members, if any allocation is required. */
    void parse_in_arena(csubstr filename, csubstr csrc, Tree *t)
    {
        substr src = t->copy_to_arena(csrc);
        this->parse_in_place(filename, src, t, t->root_id());
    }

    /** Parse into a specific node in an existing tree.
     * The immutable YAML source is first copied to the tree's arena,
     * and parsed from there.
     * The callbacks in the tree are kept, and used to allocate
     * the tree members, if any allocation is required. */
    void parse_in_arena(csubstr filename, csubstr csrc, Tree *t, size_t node_id)
    {
        substr src = t->copy_to_arena(csrc);
        this->parse_in_place(filename, src, t, node_id);
    }

    /** Parse into a specific node in an existing tree.
     * The immutable YAML source is first copied to the tree's arena,
     * and parsed from there.
     * The callbacks in the tree are kept, and used to allocate
     * the tree members, if any allocation is required. */
    void parse_in_arena(csubstr filename, csubstr csrc, NodeRef node)
    {
        substr src = node.tree()->copy_to_arena(csrc);
        this->parse_in_place(filename, src, node.tree(), node.id());
    }

    RYML_DEPRECATED("use parse_in_arena() instead") Tree parse(csubstr filename, csubstr csrc) { return parse_in_arena(filename, csrc); }
    RYML_DEPRECATED("use parse_in_arena() instead") void parse(csubstr filename, csubstr csrc, Tree *t) { parse_in_arena(filename, csrc, t); }
    RYML_DEPRECATED("use parse_in_arena() instead") void parse(csubstr filename, csubstr csrc, Tree *t, size_t node_id) { parse_in_arena(filename, csrc, t, node_id); }
    RYML_DEPRECATED("use parse_in_arena() instead") void parse(csubstr filename, csubstr csrc, NodeRef node) { parse_in_arena(filename, csrc, node); }

    /** @} */

public:

    /** @name locations */
    /** @{ */

    /** Get the location of a node of the last tree to be parsed by this parser. */
    Location location(Tree const& tree, size_t node_id) const;
    /** Get the location of a node of the last tree to be parsed by this parser. */
    Location location(ConstNodeRef node) const;
    /** Get the string starting at a particular location, to the end
     * of the parsed source buffer. */
    csubstr location_contents(Location const& loc) const;
    /** Given a pointer to a buffer position, get the location. @p val
     * must be pointing to somewhere in the source buffer that was
     * last parsed by this object. */
    Location val_location(const char *val) const;

    /** @} */

private:

    typedef enum {
        BLOCK_LITERAL, //!< keep newlines (|)
        BLOCK_FOLD     //!< replace newline with single space (>)
    } BlockStyle_e;

    typedef enum {
        CHOMP_CLIP,    //!< single newline at end (default)
        CHOMP_STRIP,   //!< no newline at end     (-)
        CHOMP_KEEP     //!< all newlines from end (+)
    } BlockChomp_e;

private:

    using flag_t = int;

    static size_t _estimate_capacity(csubstr src) { size_t c = _count_nlines(src); c = c >= 16 ? c : 16; return c; }

    void  _reset();

    bool  _finished_file() const;
    bool  _finished_line() const;

    csubstr _peek_next_line(size_t pos=npos) const;
    bool    _advance_to_peeked();
    void    _scan_line();

    csubstr _slurp_doc_scalar();

    /**
     * @param [out] quoted
     * Will only be written to if this method returns true.
     * Will be set to true if the scanned scalar was quoted, by '', "", > or |.
     */
    bool    _scan_scalar_seq_blck(csubstr *C4_RESTRICT scalar, bool *C4_RESTRICT quoted);
    bool    _scan_scalar_map_blck(csubstr *C4_RESTRICT scalar, bool *C4_RESTRICT quoted);
    bool    _scan_scalar_seq_flow(csubstr *C4_RESTRICT scalar, bool *C4_RESTRICT quoted);
    bool    _scan_scalar_map_flow(csubstr *C4_RESTRICT scalar, bool *C4_RESTRICT quoted);
    bool    _scan_scalar_unk(csubstr *C4_RESTRICT scalar, bool *C4_RESTRICT quoted);

    csubstr _scan_comment();
    csubstr _scan_squot_scalar();
    csubstr _scan_dquot_scalar();
    csubstr _scan_block();
    substr  _scan_plain_scalar_blck(csubstr currscalar, csubstr peeked_line, size_t indentation);
    substr  _scan_plain_scalar_flow(csubstr currscalar, csubstr peeked_line);
    substr  _scan_complex_key(csubstr currscalar, csubstr peeked_line);
    csubstr _scan_to_next_nonempty_line(size_t indentation);
    csubstr _extend_scanned_scalar(csubstr currscalar);

    csubstr _filter_squot_scalar(const substr s);
    csubstr _filter_dquot_scalar(substr s);
    csubstr _filter_plain_scalar(substr s, size_t indentation);
    csubstr _filter_block_scalar(substr s, BlockStyle_e style, BlockChomp_e chomp, size_t indentation);
    template<bool backslash_is_escape, bool keep_trailing_whitespace>
    bool    _filter_nl(substr scalar, size_t *C4_RESTRICT pos, size_t *C4_RESTRICT filter_arena_pos, size_t indentation);
    template<bool keep_trailing_whitespace>
    void    _filter_ws(substr scalar, size_t *C4_RESTRICT pos, size_t *C4_RESTRICT filter_arena_pos);
    bool    _apply_chomp(substr buf, size_t *C4_RESTRICT pos, BlockChomp_e chomp);

    void  _handle_finished_file();
    void  _handle_line();

    bool  _handle_indentation();

    bool  _handle_unk();
    bool  _handle_map_flow();
    bool  _handle_map_blck();
    bool  _handle_seq_flow();
    bool  _handle_seq_blck();
    bool  _handle_top();
    bool  _handle_types();
    bool  _handle_key_anchors_and_refs();
    bool  _handle_val_anchors_and_refs();
    void  _move_val_tag_to_key_tag();
    void  _move_key_tag_to_val_tag();
    void  _move_key_tag2_to_key_tag();
    void  _move_val_anchor_to_key_anchor();
    void  _move_key_anchor_to_val_anchor();

    void  _push_level(bool explicit_flow_chars = false);
    void  _pop_level();

    void  _start_unk(bool as_child=true);

    void  _start_map(bool as_child=true);
    void  _start_map_unk(bool as_child);
    void  _stop_map();

    void  _start_seq(bool as_child=true);
    void  _stop_seq();

    void  _start_seqimap();
    void  _stop_seqimap();

    void  _start_doc(bool as_child=true);
    void  _stop_doc();
    void  _start_new_doc(csubstr rem);
    void  _end_stream();

    NodeData* _append_val(csubstr val, flag_t quoted=false);
    NodeData* _append_key_val(csubstr val, flag_t val_quoted=false);
    bool  _rval_dash_start_or_continue_seq();

    void  _store_scalar(csubstr s, flag_t is_quoted);
    csubstr _consume_scalar();
    void  _move_scalar_from_top();

    inline NodeData* _append_val_null(const char *str) { _RYML_CB_ASSERT(m_stack.m_callbacks, str >= m_buf.begin() && str <= m_buf.end()); return _append_val({nullptr, size_t(0)}); }
    inline NodeData* _append_key_val_null(const char *str) { _RYML_CB_ASSERT(m_stack.m_callbacks, str >= m_buf.begin() && str <= m_buf.end()); return _append_key_val({nullptr, size_t(0)}); }
    inline void      _store_scalar_null(const char *str) {  _RYML_CB_ASSERT(m_stack.m_callbacks, str >= m_buf.begin() && str <= m_buf.end()); _store_scalar({nullptr, size_t(0)}, false); }

    void  _set_indentation(size_t behind);
    void  _save_indentation(size_t behind=0);
    bool  _maybe_set_indentation_from_anchor_or_tag();

    void  _write_key_anchor(size_t node_id);
    void  _write_val_anchor(size_t node_id);

    void _handle_directive(csubstr directive);

    void _skipchars(char c);
    template<size_t N>
    void _skipchars(const char (&chars)[N]);

private:

    static size_t _count_nlines(csubstr src);

private:

    typedef enum : flag_t {
        RTOP = 0x01 <<  0,   ///< reading at top level
        RUNK = 0x01 <<  1,   ///< reading an unknown: must determine whether scalar, map or seq
        RMAP = 0x01 <<  2,   ///< reading a map
        RSEQ = 0x01 <<  3,   ///< reading a seq
        FLOW = 0x01 <<  4,   ///< reading is inside explicit flow chars: [] or {}
        QMRK = 0x01 <<  5,   ///< reading an explicit key (`? key`)
        RKEY = 0x01 <<  6,   ///< reading a scalar as key
        RVAL = 0x01 <<  7,   ///< reading a scalar as val
        RNXT = 0x01 <<  8,   ///< read next val or keyval
        SSCL = 0x01 <<  9,   ///< there's a stored scalar
        QSCL = 0x01 << 10,   ///< stored scalar was quoted
        RSET = 0x01 << 11,   ///< the (implicit) map being read is a !!set. @see https://yaml.org/type/set.html
        NDOC = 0x01 << 12,   ///< no document mode. a document has ended and another has not started yet.
        //! reading an implicit map nested in an explicit seq.
        //! eg, {key: [key2: value2, key3: value3]}
        //! is parsed as {key: [{key2: value2}, {key3: value3}]}
        RSEQIMAP = 0x01 << 13,
    } State_e;

    struct LineContents
    {
        csubstr  full;        ///< the full line, including newlines on the right
        csubstr  stripped;    ///< the stripped line, excluding newlines on the right
        csubstr  rem;         ///< the stripped line remainder; initially starts at the first non-space character
        size_t   indentation; ///< the number of spaces on the beginning of the line

        LineContents() : full(), stripped(), rem(), indentation() {}

        void reset_with_next_line(csubstr buf, size_t pos);

        void reset(csubstr full_, csubstr stripped_)
        {
            full = full_;
            stripped = stripped_;
            rem = stripped_;
            // find the first column where the character is not a space
            indentation = full.first_not_of(' ');
        }

        size_t current_col() const
        {
            return current_col(rem);
        }

        size_t current_col(csubstr s) const
        {
            RYML_ASSERT(s.str >= full.str);
            RYML_ASSERT(full.is_super(s));
            size_t col = static_cast<size_t>(s.str - full.str);
            return col;
        }
    };

    struct State
    {
        flag_t       flags;
        size_t       level;
        size_t       node_id; // don't hold a pointer to the node as it will be relocated during tree resizes
        csubstr      scalar;
        size_t       scalar_col; // the column where the scalar (or its quotes) begin

        Location     pos;
        LineContents line_contents;
        size_t       indref;

        State() : flags(), level(), node_id(), scalar(), scalar_col(), pos(), line_contents(), indref() {}

        void reset(const char *file, size_t node_id_)
        {
            flags = RUNK|RTOP;
            level = 0;
            pos.name = to_csubstr(file);
            pos.offset = 0;
            pos.line = 1;
            pos.col = 1;
            node_id = node_id_;
            scalar_col = 0;
            scalar.clear();
            indref = 0;
        }
    };

    void _line_progressed(size_t ahead);
    void _line_ended();
    void _line_ended_undo();

    void _prepare_pop()
    {
        RYML_ASSERT(m_stack.size() > 1);
        State const& curr = m_stack.top();
        State      & next = m_stack.top(1);
        next.pos = curr.pos;
        next.line_contents = curr.line_contents;
        next.scalar = curr.scalar;
    }

    inline bool _at_line_begin() const
    {
        return m_state->line_contents.rem.begin() == m_state->line_contents.full.begin();
    }
    inline bool _at_line_end() const
    {
        csubstr r = m_state->line_contents.rem;
        return r.empty() || r.begins_with(' ', r.len);
    }
    inline bool _token_is_from_this_line(csubstr token) const
    {
        return token.is_sub(m_state->line_contents.full);
    }

    inline NodeData * node(State const* s) const { return m_tree->get(s->node_id); }
    inline NodeData * node(State const& s) const { return m_tree->get(s .node_id); }
    inline NodeData * node(size_t node_id) const { return m_tree->get(   node_id); }

    inline bool has_all(flag_t f) const { return (m_state->flags & f) == f; }
    inline bool has_any(flag_t f) const { return (m_state->flags & f) != 0; }
    inline bool has_none(flag_t f) const { return (m_state->flags & f) == 0; }

    static inline bool has_all(flag_t f, State const* s) { return (s->flags & f) == f; }
    static inline bool has_any(flag_t f, State const* s) { return (s->flags & f) != 0; }
    static inline bool has_none(flag_t f, State const* s) { return (s->flags & f) == 0; }

    inline void set_flags(flag_t f) { set_flags(f, m_state); }
    inline void add_flags(flag_t on) { add_flags(on, m_state); }
    inline void addrem_flags(flag_t on, flag_t off) { addrem_flags(on, off, m_state); }
    inline void rem_flags(flag_t off) { rem_flags(off, m_state); }

    void set_flags(flag_t f, State * s);
    void add_flags(flag_t on, State * s);
    void addrem_flags(flag_t on, flag_t off, State * s);
    void rem_flags(flag_t off, State * s);

    void _resize_filter_arena(size_t num_characters);
    void _grow_filter_arena(size_t num_characters);
    substr _finish_filter_arena(substr dst, size_t pos);

    void _prepare_locations();
    void _resize_locations(size_t sz);
    bool _locations_dirty() const;

    bool _location_from_cont(Tree const& tree, size_t node, Location *C4_RESTRICT loc) const;
    bool _location_from_node(Tree const& tree, size_t node, Location *C4_RESTRICT loc, size_t level) const;

private:

    void _free();
    void _clr();
    void _cp(Parser const* that);
    void _mv(Parser *that);

#ifdef RYML_DBG
    template<class ...Args> void _dbg(csubstr fmt, Args const& C4_RESTRICT ...args) const;
#endif
    template<class ...Args> void _err(csubstr fmt, Args const& C4_RESTRICT ...args) const;
    template<class DumpFn>  void _fmt_msg(DumpFn &&dumpfn) const;
    static csubstr _prfl(substr buf, flag_t v);

private:

    ParserOptions m_options;

    csubstr m_file;
     substr m_buf;

    size_t  m_root_id;
    Tree *  m_tree;

    detail::stack<State> m_stack;
    State * m_state;

    size_t  m_key_tag_indentation;
    size_t  m_key_tag2_indentation;
    csubstr m_key_tag;
    csubstr m_key_tag2;
    size_t  m_val_tag_indentation;
    csubstr m_val_tag;

    bool    m_key_anchor_was_before;
    size_t  m_key_anchor_indentation;
    csubstr m_key_anchor;
    size_t  m_val_anchor_indentation;
    csubstr m_val_anchor;

    substr m_filter_arena;

    size_t *m_newline_offsets;
    size_t  m_newline_offsets_size;
    size_t  m_newline_offsets_capacity;
    csubstr m_newline_offsets_buf;
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** @name parse_in_place
 *
 * @desc parse a mutable YAML source buffer.
 *
 * @note These freestanding functions use a temporary parser object,
 * and are convenience functions to easily parse YAML without the need
 * to instantiate a separate parser. Note that some properties
 * (notably node locations in the original source code) are only
 * available through the parser object after it has parsed the
 * code. If you need access to any of these properties, use
 * Parser::parse_in_place() */
/** @{ */

inline Tree parse_in_place(                  substr yaml                         ) { Parser np; return np.parse_in_place({}      , yaml); } //!< parse in-situ a modifiable YAML source buffer.
inline Tree parse_in_place(csubstr filename, substr yaml                         ) { Parser np; return np.parse_in_place(filename, yaml); } //!< parse in-situ a modifiable YAML source buffer, providing a filename for error messages.
inline void parse_in_place(                  substr yaml, Tree *t                ) { Parser np; np.parse_in_place({}      , yaml, t); } //!< reusing the YAML tree, parse in-situ a modifiable YAML source buffer
inline void parse_in_place(csubstr filename, substr yaml, Tree *t                ) { Parser np; np.parse_in_place(filename, yaml, t); } //!< reusing the YAML tree, parse in-situ a modifiable YAML source buffer, providing a filename for error messages.
inline void parse_in_place(                  substr yaml, Tree *t, size_t node_id) { Parser np; np.parse_in_place({}      , yaml, t, node_id); } //!< reusing the YAML tree, parse in-situ a modifiable YAML source buffer
inline void parse_in_place(csubstr filename, substr yaml, Tree *t, size_t node_id) { Parser np; np.parse_in_place(filename, yaml, t, node_id); } //!< reusing the YAML tree, parse in-situ a modifiable YAML source buffer, providing a filename for error messages.
inline void parse_in_place(                  substr yaml, NodeRef node           ) { Parser np; np.parse_in_place({}      , yaml, node); } //!< reusing the YAML tree, parse in-situ a modifiable YAML source buffer
inline void parse_in_place(csubstr filename, substr yaml, NodeRef node           ) { Parser np; np.parse_in_place(filename, yaml, node); } //!< reusing the YAML tree, parse in-situ a modifiable YAML source buffer, providing a filename for error messages.

RYML_DEPRECATED("use parse_in_place() instead") inline Tree parse(                  substr yaml                         ) { Parser np; return np.parse_in_place({}      , yaml); }
RYML_DEPRECATED("use parse_in_place() instead") inline Tree parse(csubstr filename, substr yaml                         ) { Parser np; return np.parse_in_place(filename, yaml); }
RYML_DEPRECATED("use parse_in_place() instead") inline void parse(                  substr yaml, Tree *t                ) { Parser np; np.parse_in_place({}      , yaml, t); }
RYML_DEPRECATED("use parse_in_place() instead") inline void parse(csubstr filename, substr yaml, Tree *t                ) { Parser np; np.parse_in_place(filename, yaml, t); }
RYML_DEPRECATED("use parse_in_place() instead") inline void parse(                  substr yaml, Tree *t, size_t node_id) { Parser np; np.parse_in_place({}      , yaml, t, node_id); }
RYML_DEPRECATED("use parse_in_place() instead") inline void parse(csubstr filename, substr yaml, Tree *t, size_t node_id) { Parser np; np.parse_in_place(filename, yaml, t, node_id); }
RYML_DEPRECATED("use parse_in_place() instead") inline void parse(                  substr yaml, NodeRef node           ) { Parser np; np.parse_in_place({}      , yaml, node); }
RYML_DEPRECATED("use parse_in_place() instead") inline void parse(csubstr filename, substr yaml, NodeRef node           ) { Parser np; np.parse_in_place(filename, yaml, node); }

/** @} */


//-----------------------------------------------------------------------------

/** @name parse_in_arena
 * @desc parse a read-only YAML source buffer, copying it first to the tree's arena.
 *
 * @note These freestanding functions use a temporary parser object,
 * and are convenience functions to easily parse YAML without the need
 * to instantiate a separate parser. Note that some properties
 * (notably node locations in the original source code) are only
 * available through the parser object after it has parsed the
 * code. If you need access to any of these properties, use
 * Parser::parse_in_arena().
 *
 * @note overloads receiving a substr YAML buffer are intentionally
 * left undefined, such that calling parse_in_arena() with a substr
 * will cause a linker error. This is to prevent an accidental
 * copy of the source buffer to the tree's arena, because substr
 * is implicitly convertible to csubstr. If you really intend to parse
 * a mutable buffer in the tree's arena, convert it first to immutable
 * by assigning the substr to a csubstr prior to calling parse_in_arena().
 * This is not needed for parse_in_place() because csubstr is not
 * implicitly convertible to substr. */
/** @{ */

/* READ THE NOTE ABOVE! */
RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) Tree parse_in_arena(                  substr yaml                         );
RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) Tree parse_in_arena(csubstr filename, substr yaml                         );
RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) void parse_in_arena(                  substr yaml, Tree *t                );
RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) void parse_in_arena(csubstr filename, substr yaml, Tree *t                );
RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) void parse_in_arena(                  substr yaml, Tree *t, size_t node_id);
RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) void parse_in_arena(csubstr filename, substr yaml, Tree *t, size_t node_id);
RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) void parse_in_arena(                  substr yaml, NodeRef node           );
RYML_DEPRECATED(RYML_DONT_PARSE_SUBSTR_IN_ARENA) void parse_in_arena(csubstr filename, substr yaml, NodeRef node           );

inline Tree parse_in_arena(                  csubstr yaml                         ) { Parser np; return np.parse_in_arena({}      , yaml); } //!< parse a read-only YAML source buffer, copying it first to the tree's source arena.
inline Tree parse_in_arena(csubstr filename, csubstr yaml                         ) { Parser np; return np.parse_in_arena(filename, yaml); } //!< parse a read-only YAML source buffer, copying it first to the tree's source arena, providing a filename for error messages.
inline void parse_in_arena(                  csubstr yaml, Tree *t                ) { Parser np; np.parse_in_arena({}      , yaml, t); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena.
inline void parse_in_arena(csubstr filename, csubstr yaml, Tree *t                ) { Parser np; np.parse_in_arena(filename, yaml, t); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena, providing a filename for error messages.
inline void parse_in_arena(                  csubstr yaml, Tree *t, size_t node_id) { Parser np; np.parse_in_arena({}      , yaml, t, node_id); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena.
inline void parse_in_arena(csubstr filename, csubstr yaml, Tree *t, size_t node_id) { Parser np; np.parse_in_arena(filename, yaml, t, node_id); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena, providing a filename for error messages.
inline void parse_in_arena(                  csubstr yaml, NodeRef node           ) { Parser np; np.parse_in_arena({}      , yaml, node); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena.
inline void parse_in_arena(csubstr filename, csubstr yaml, NodeRef node           ) { Parser np; np.parse_in_arena(filename, yaml, node); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena, providing a filename for error messages.

RYML_DEPRECATED("use parse_in_arena() instead") inline Tree parse(                  csubstr yaml                         ) { Parser np; return np.parse_in_arena({}      , yaml); } //!< parse a read-only YAML source buffer, copying it first to the tree's source arena.
RYML_DEPRECATED("use parse_in_arena() instead") inline Tree parse(csubstr filename, csubstr yaml                         ) { Parser np; return np.parse_in_arena(filename, yaml); } //!< parse a read-only YAML source buffer, copying it first to the tree's source arena, providing a filename for error messages.
RYML_DEPRECATED("use parse_in_arena() instead") inline void parse(                  csubstr yaml, Tree *t                ) { Parser np; np.parse_in_arena({}      , yaml, t); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena.
RYML_DEPRECATED("use parse_in_arena() instead") inline void parse(csubstr filename, csubstr yaml, Tree *t                ) { Parser np; np.parse_in_arena(filename, yaml, t); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena, providing a filename for error messages.
RYML_DEPRECATED("use parse_in_arena() instead") inline void parse(                  csubstr yaml, Tree *t, size_t node_id) { Parser np; np.parse_in_arena({}      , yaml, t, node_id); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena.
RYML_DEPRECATED("use parse_in_arena() instead") inline void parse(csubstr filename, csubstr yaml, Tree *t, size_t node_id) { Parser np; np.parse_in_arena(filename, yaml, t, node_id); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena, providing a filename for error messages.
RYML_DEPRECATED("use parse_in_arena() instead") inline void parse(                  csubstr yaml, NodeRef node           ) { Parser np; np.parse_in_arena({}      , yaml, node); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena.
RYML_DEPRECATED("use parse_in_arena() instead") inline void parse(csubstr filename, csubstr yaml, NodeRef node           ) { Parser np; np.parse_in_arena(filename, yaml, node); } //!< reusing the YAML tree, parse a read-only YAML source buffer, copying it first to the tree's source arena, providing a filename for error messages.

/** @} */

} // namespace yml
} // namespace c4

#if defined(_MSC_VER)
#   pragma warning(pop)
#endif

#endif /* _C4_YML_PARSE_HPP_ */
