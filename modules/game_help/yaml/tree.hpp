#ifndef _C4_YML_TREE_HPP_
#define _C4_YML_TREE_HPP_


#include "c4/error.hpp"
#include "c4/types.hpp"
#ifndef _C4_YML_COMMON_HPP_
#include "common.hpp"
#endif

#include "c4/charconv.hpp"
#include <cmath>
#include <limits>


C4_SUPPRESS_WARNING_MSVC_PUSH
C4_SUPPRESS_WARNING_MSVC(4251) // needs to have dll-interface to be used by clients of struct
C4_SUPPRESS_WARNING_MSVC(4296) // expression is always 'boolean_value'
C4_SUPPRESS_WARNING_GCC_CLANG_PUSH
C4_SUPPRESS_WARNING_GCC_CLANG("-Wold-style-cast")
C4_SUPPRESS_WARNING_GCC("-Wtype-limits")


namespace c4 {
namespace yml {

struct NodeScalar;
struct NodeInit;
struct NodeData;
class NodeRef;
class ConstNodeRef;
class Tree;


/** encode a floating point value to a string. */
template<class T>
size_t to_chars_float(substr buf, T val)
{
    C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wfloat-equal");
    static_assert(std::is_floating_point<T>::value, "must be floating point");
    if(C4_UNLIKELY(std::isnan(val)))
        return to_chars(buf, csubstr(".nan"));
    else if(C4_UNLIKELY(val == std::numeric_limits<T>::infinity()))
        return to_chars(buf, csubstr(".inf"));
    else if(C4_UNLIKELY(val == -std::numeric_limits<T>::infinity()))
        return to_chars(buf, csubstr("-.inf"));
    return to_chars(buf, val);
    C4_SUPPRESS_WARNING_GCC_CLANG_POP
}


/** decode a floating point from string. Accepts special values: .nan,
 * .inf, -.inf */
template<class T>
bool from_chars_float(csubstr buf, T *C4_RESTRICT val)
{
    static_assert(std::is_floating_point<T>::value, "must be floating point");
    if(C4_LIKELY(from_chars(buf, val)))
    {
        return true;
    }
    else if(C4_UNLIKELY(buf == ".nan" || buf == ".NaN" || buf == ".NAN"))
    {
        *val = std::numeric_limits<T>::quiet_NaN();
        return true;
    }
    else if(C4_UNLIKELY(buf == ".inf" || buf == ".Inf" || buf == ".INF"))
    {
        *val = std::numeric_limits<T>::infinity();
        return true;
    }
    else if(C4_UNLIKELY(buf == "-.inf" || buf == "-.Inf" || buf == "-.INF"))
    {
        *val = -std::numeric_limits<T>::infinity();
        return true;
    }
    else
    {
        return false;
    }
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** the integral type necessary to cover all the bits marking node tags */
using tag_bits = uint16_t;

/** a bit mask for marking tags for types */
typedef enum : tag_bits {
    // container types
    TAG_NONE      =  0,
    TAG_MAP       =  1, /**< !!map   Unordered set of key: value pairs without duplicates. @see https://yaml.org/type/map.html */
    TAG_OMAP      =  2, /**< !!omap  Ordered sequence of key: value pairs without duplicates. @see https://yaml.org/type/omap.html */
    TAG_PAIRS     =  3, /**< !!pairs Ordered sequence of key: value pairs allowing duplicates. @see https://yaml.org/type/pairs.html */
    TAG_SET       =  4, /**< !!set   Unordered set of non-equal values. @see https://yaml.org/type/set.html */
    TAG_SEQ       =  5, /**< !!seq   Sequence of arbitrary values. @see https://yaml.org/type/seq.html */
    // scalar types
    TAG_BINARY    =  6, /**< !!binary A sequence of zero or more octets (8 bit values). @see https://yaml.org/type/binary.html */
    TAG_BOOL      =  7, /**< !!bool   Mathematical Booleans. @see https://yaml.org/type/bool.html */
    TAG_FLOAT     =  8, /**< !!float  Floating-point approximation to real numbers. https://yaml.org/type/float.html */
    TAG_INT       =  9, /**< !!float  Mathematical integers. https://yaml.org/type/int.html */
    TAG_MERGE     = 10, /**< !!merge  Specify one or more mapping to be merged with the current one. https://yaml.org/type/merge.html */
    TAG_NULL      = 11, /**< !!null   Devoid of value. https://yaml.org/type/null.html */
    TAG_STR       = 12, /**< !!str    A sequence of zero or more Unicode characters. https://yaml.org/type/str.html */
    TAG_TIMESTAMP = 13, /**< !!timestamp A point in time https://yaml.org/type/timestamp.html */
    TAG_VALUE     = 14, /**< !!value  Specify the default value of a mapping https://yaml.org/type/value.html */
    TAG_YAML      = 15, /**< !!yaml   Specify the default value of a mapping https://yaml.org/type/yaml.html */
} YamlTag_e;

YamlTag_e to_tag(csubstr tag);
csubstr from_tag(YamlTag_e tag);
csubstr from_tag_long(YamlTag_e tag);
csubstr normalize_tag(csubstr tag);
csubstr normalize_tag_long(csubstr tag);

struct TagDirective
{
    /** Eg `!e!` in `%TAG !e! tag:example.com,2000:app/` */
    csubstr handle;
    /** Eg `tag:example.com,2000:app/` in `%TAG !e! tag:example.com,2000:app/` */
    csubstr prefix;
    /** The next node to which this tag directive applies */
    size_t next_node_id;
};

#ifndef RYML_MAX_TAG_DIRECTIVES
/** the maximum number of tag directives in a Tree */
#define RYML_MAX_TAG_DIRECTIVES 4
#endif



//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


/** the integral type necessary to cover all the bits marking node types */
using type_bits = uint64_t;


/** a bit mask for marking node types */
typedef enum : type_bits {
    // a convenience define, undefined below
    #define c4bit(v) (type_bits(1) << v)
    NOTYPE  = 0,            ///< no node type is set
    VAL     = c4bit(0),     ///< a leaf node, has a (possibly empty) value
    KEY     = c4bit(1),     ///< is member of a map, must have non-empty key
    MAP     = c4bit(2),     ///< a map: a parent of keyvals
    SEQ     = c4bit(3),     ///< a seq: a parent of vals
    DOC     = c4bit(4),     ///< a document
    STREAM  = c4bit(5)|SEQ, ///< a stream: a seq of docs
    KEYREF  = c4bit(6),     ///< a *reference: the key references an &anchor
    VALREF  = c4bit(7),     ///< a *reference: the val references an &anchor
    KEYANCH = c4bit(8),     ///< the key has an &anchor
    VALANCH = c4bit(9),     ///< the val has an &anchor
    KEYTAG  = c4bit(10),    ///< the key has an explicit tag/type
    VALTAG  = c4bit(11),    ///< the val has an explicit tag/type
    _TYMASK = c4bit(12)-1,  // all the bits up to here
    VALQUO  = c4bit(12),    ///< the val is quoted by '', "", > or |
    KEYQUO  = c4bit(13),    ///< the key is quoted by '', "", > or |
    KEYVAL  = KEY|VAL,
    KEYSEQ  = KEY|SEQ,
    KEYMAP  = KEY|MAP,
    DOCMAP  = DOC|MAP,
    DOCSEQ  = DOC|SEQ,
    DOCVAL  = DOC|VAL,
    _KEYMASK = KEY | KEYQUO | KEYANCH | KEYREF | KEYTAG,
    _VALMASK = VAL | VALQUO | VALANCH | VALREF | VALTAG,
    // these flags are from a work in progress and should be used with care
    _WIP_STYLE_FLOW_SL = c4bit(14), ///< mark container with single-line flow format (seqs as '[val1,val2], maps as '{key: val, key2: val2}')
    _WIP_STYLE_FLOW_ML = c4bit(15), ///< mark container with multi-line flow format (seqs as '[val1,\nval2], maps as '{key: val,\nkey2: val2}')
    _WIP_STYLE_BLOCK   = c4bit(16), ///< mark container with block format (seqs as '- val\n', maps as 'key: val')
    _WIP_KEY_LITERAL   = c4bit(17), ///< mark key scalar as multiline, block literal |
    _WIP_VAL_LITERAL   = c4bit(18), ///< mark val scalar as multiline, block literal |
    _WIP_KEY_FOLDED    = c4bit(19), ///< mark key scalar as multiline, block folded >
    _WIP_VAL_FOLDED    = c4bit(20), ///< mark val scalar as multiline, block folded >
    _WIP_KEY_SQUO      = c4bit(21), ///< mark key scalar as single quoted
    _WIP_VAL_SQUO      = c4bit(22), ///< mark val scalar as single quoted
    _WIP_KEY_DQUO      = c4bit(23), ///< mark key scalar as double quoted
    _WIP_VAL_DQUO      = c4bit(24), ///< mark val scalar as double quoted
    _WIP_KEY_PLAIN     = c4bit(25), ///< mark key scalar as plain scalar (unquoted, even when multiline)
    _WIP_VAL_PLAIN     = c4bit(26), ///< mark val scalar as plain scalar (unquoted, even when multiline)
    _WIP_KEY_STYLE     = _WIP_KEY_LITERAL|_WIP_KEY_FOLDED|_WIP_KEY_SQUO|_WIP_KEY_DQUO|_WIP_KEY_PLAIN,
    _WIP_VAL_STYLE     = _WIP_VAL_LITERAL|_WIP_VAL_FOLDED|_WIP_VAL_SQUO|_WIP_VAL_DQUO|_WIP_VAL_PLAIN,
    _WIP_KEY_FT_NL     = c4bit(27), ///< features: mark key scalar as having \n in its contents
    _WIP_VAL_FT_NL     = c4bit(28), ///< features: mark val scalar as having \n in its contents
    _WIP_KEY_FT_SQ     = c4bit(29), ///< features: mark key scalar as having single quotes in its contents
    _WIP_VAL_FT_SQ     = c4bit(30), ///< features: mark val scalar as having single quotes in its contents
    _WIP_KEY_FT_DQ     = c4bit(31), ///< features: mark key scalar as having double quotes in its contents
    _WIP_VAL_FT_DQ     = c4bit(32), ///< features: mark val scalar as having double quotes in its contents
    #undef c4bit
} NodeType_e;


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** wraps a NodeType_e element with some syntactic sugar and predicates */
struct NodeType
{
public:

    NodeType_e type;

public:

    C4_ALWAYS_INLINE NodeType() : type(NOTYPE) {}
    C4_ALWAYS_INLINE NodeType(NodeType_e t) : type(t) {}
    C4_ALWAYS_INLINE NodeType(type_bits t) : type((NodeType_e)t) {}

    C4_ALWAYS_INLINE const char *type_str() const { return type_str(type); }
    static const char* type_str(NodeType_e t);

    C4_ALWAYS_INLINE void set(NodeType_e t) { type = t; }
    C4_ALWAYS_INLINE void set(type_bits  t) { type = (NodeType_e)t; }

    C4_ALWAYS_INLINE void add(NodeType_e t) { type = (NodeType_e)(type|t); }
    C4_ALWAYS_INLINE void add(type_bits  t) { type = (NodeType_e)(type|t); }

    C4_ALWAYS_INLINE void rem(NodeType_e t) { type = (NodeType_e)(type & ~t); }
    C4_ALWAYS_INLINE void rem(type_bits  t) { type = (NodeType_e)(type & ~t); }

    C4_ALWAYS_INLINE void clear() { type = NOTYPE; }

public:

    C4_ALWAYS_INLINE operator NodeType_e      & C4_RESTRICT ()       { return type; }
    C4_ALWAYS_INLINE operator NodeType_e const& C4_RESTRICT () const { return type; }

    C4_ALWAYS_INLINE bool operator== (NodeType_e t) const { return type == t; }
    C4_ALWAYS_INLINE bool operator!= (NodeType_e t) const { return type != t; }

public:

    #if defined(__clang__)
    #   pragma clang diagnostic push
    #   pragma clang diagnostic ignored "-Wnull-dereference"
    #elif defined(__GNUC__)
    #   pragma GCC diagnostic push
    #   if __GNUC__ >= 6
    #       pragma GCC diagnostic ignored "-Wnull-dereference"
    #   endif
    #endif

    C4_ALWAYS_INLINE bool is_notype() const { return type == NOTYPE; }
    C4_ALWAYS_INLINE bool is_stream() const { return ((type & STREAM) == STREAM) != 0; }
    C4_ALWAYS_INLINE bool is_doc() const { return (type & DOC) != 0; }
    C4_ALWAYS_INLINE bool is_container() const { return (type & (MAP|SEQ|STREAM)) != 0; }
    C4_ALWAYS_INLINE bool is_map() const { return (type & MAP) != 0; }
    C4_ALWAYS_INLINE bool is_seq() const { return (type & SEQ) != 0; }
    C4_ALWAYS_INLINE bool has_key() const { return (type & KEY) != 0; }
    C4_ALWAYS_INLINE bool has_val() const { return (type & VAL) != 0; }
    C4_ALWAYS_INLINE bool is_val() const { return (type & KEYVAL) == VAL; }
    C4_ALWAYS_INLINE bool is_keyval() const { return (type & KEYVAL) == KEYVAL; }
    C4_ALWAYS_INLINE bool has_key_tag() const { return (type & (KEY|KEYTAG)) == (KEY|KEYTAG); }
    C4_ALWAYS_INLINE bool has_val_tag() const { return ((type & VALTAG) && (type & (VAL|MAP|SEQ))); }
    C4_ALWAYS_INLINE bool has_key_anchor() const { return (type & (KEY|KEYANCH)) == (KEY|KEYANCH); }
    C4_ALWAYS_INLINE bool is_key_anchor() const { return (type & (KEY|KEYANCH)) == (KEY|KEYANCH); }
    C4_ALWAYS_INLINE bool has_val_anchor() const { return (type & VALANCH) != 0 && (type & (VAL|SEQ|MAP)) != 0; }
    C4_ALWAYS_INLINE bool is_val_anchor() const { return (type & VALANCH) != 0 && (type & (VAL|SEQ|MAP)) != 0; }
    C4_ALWAYS_INLINE bool has_anchor() const { return (type & (KEYANCH|VALANCH)) != 0; }
    C4_ALWAYS_INLINE bool is_anchor() const { return (type & (KEYANCH|VALANCH)) != 0; }
    C4_ALWAYS_INLINE bool is_key_ref() const { return (type & KEYREF) != 0; }
    C4_ALWAYS_INLINE bool is_val_ref() const { return (type & VALREF) != 0; }
    C4_ALWAYS_INLINE bool is_ref() const { return (type & (KEYREF|VALREF)) != 0; }
    C4_ALWAYS_INLINE bool is_anchor_or_ref() const { return (type & (KEYANCH|VALANCH|KEYREF|VALREF)) != 0; }
    C4_ALWAYS_INLINE bool is_key_quoted() const { return (type & (KEY|KEYQUO)) == (KEY|KEYQUO); }
    C4_ALWAYS_INLINE bool is_val_quoted() const { return (type & (VAL|VALQUO)) == (VAL|VALQUO); }
    C4_ALWAYS_INLINE bool is_quoted() const { return (type & (KEY|KEYQUO)) == (KEY|KEYQUO) || (type & (VAL|VALQUO)) == (VAL|VALQUO); }

    // these predicates are a work in progress and subject to change. Don't use yet.
    C4_ALWAYS_INLINE bool default_block() const { return (type & (_WIP_STYLE_BLOCK|_WIP_STYLE_FLOW_ML|_WIP_STYLE_FLOW_SL)) == 0; }
    C4_ALWAYS_INLINE bool marked_block() const { return (type & (_WIP_STYLE_BLOCK)) != 0; }
    C4_ALWAYS_INLINE bool marked_flow_sl() const { return (type & (_WIP_STYLE_FLOW_SL)) != 0; }
    C4_ALWAYS_INLINE bool marked_flow_ml() const { return (type & (_WIP_STYLE_FLOW_ML)) != 0; }
    C4_ALWAYS_INLINE bool marked_flow() const { return (type & (_WIP_STYLE_FLOW_ML|_WIP_STYLE_FLOW_SL)) != 0; }
    C4_ALWAYS_INLINE bool key_marked_literal() const { return (type & (_WIP_KEY_LITERAL)) != 0; }
    C4_ALWAYS_INLINE bool val_marked_literal() const { return (type & (_WIP_VAL_LITERAL)) != 0; }
    C4_ALWAYS_INLINE bool key_marked_folded() const { return (type & (_WIP_KEY_FOLDED)) != 0; }
    C4_ALWAYS_INLINE bool val_marked_folded() const { return (type & (_WIP_VAL_FOLDED)) != 0; }
    C4_ALWAYS_INLINE bool key_marked_squo() const { return (type & (_WIP_KEY_SQUO)) != 0; }
    C4_ALWAYS_INLINE bool val_marked_squo() const { return (type & (_WIP_VAL_SQUO)) != 0; }
    C4_ALWAYS_INLINE bool key_marked_dquo() const { return (type & (_WIP_KEY_DQUO)) != 0; }
    C4_ALWAYS_INLINE bool val_marked_dquo() const { return (type & (_WIP_VAL_DQUO)) != 0; }
    C4_ALWAYS_INLINE bool key_marked_plain() const { return (type & (_WIP_KEY_PLAIN)) != 0; }
    C4_ALWAYS_INLINE bool val_marked_plain() const { return (type & (_WIP_VAL_PLAIN)) != 0; }

    #if defined(__clang__)
    #   pragma clang diagnostic pop
    #elif defined(__GNUC__)
    #   pragma GCC diagnostic pop
    #endif

};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** a node scalar is a csubstr, which may be tagged and anchored. */
struct NodeScalar
{
    csubstr tag;
    csubstr scalar;
    csubstr anchor;

public:

    /// initialize as an empty scalar
    inline NodeScalar() noexcept : tag(), scalar(), anchor() {}

    /// initialize as an untagged scalar
    template<size_t N>
    inline NodeScalar(const char (&s)[N]) noexcept : tag(), scalar(s), anchor() {}
    inline NodeScalar(csubstr      s    ) noexcept : tag(), scalar(s), anchor() {}

    /// initialize as a tagged scalar
    template<size_t N, size_t M>
    inline NodeScalar(const char (&t)[N], const char (&s)[N]) noexcept : tag(t), scalar(s), anchor() {}
    inline NodeScalar(csubstr      t    , csubstr      s    ) noexcept : tag(t), scalar(s), anchor() {}

public:

    ~NodeScalar() noexcept = default;
    NodeScalar(NodeScalar &&) noexcept = default;
    NodeScalar(NodeScalar const&) noexcept = default;
    NodeScalar& operator= (NodeScalar &&) noexcept = default;
    NodeScalar& operator= (NodeScalar const&) noexcept = default;

public:

    bool empty() const noexcept { return tag.empty() && scalar.empty() && anchor.empty(); }

    void clear() noexcept { tag.clear(); scalar.clear(); anchor.clear(); }

    void set_ref_maybe_replacing_scalar(csubstr ref, bool has_scalar) noexcept
    {
        csubstr trimmed = ref.begins_with('*') ? ref.sub(1) : ref;
        anchor = trimmed;
        if((!has_scalar) || !scalar.ends_with(trimmed))
            scalar = ref;
    }
};
C4_MUST_BE_TRIVIAL_COPY(NodeScalar);


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** convenience class to initialize nodes */
struct NodeInit
{

    NodeType   type;
    NodeScalar key;
    NodeScalar val;

public:

    /// initialize as an empty node
    NodeInit() : type(NOTYPE), key(), val() {}
    /// initialize as a typed node
    NodeInit(NodeType_e t) : type(t), key(), val() {}
    /// initialize as a sequence member
    NodeInit(NodeScalar const& v) : type(VAL), key(), val(v) { _add_flags(); }
    /// initialize as a mapping member
    NodeInit(              NodeScalar const& k, NodeScalar const& v) : type(KEYVAL), key(k.tag, k.scalar), val(v.tag, v.scalar) { _add_flags(); }
    /// initialize as a mapping member with explicit type
    NodeInit(NodeType_e t, NodeScalar const& k, NodeScalar const& v) : type(t     ), key(k.tag, k.scalar), val(v.tag, v.scalar) { _add_flags(); }
    /// initialize as a mapping member with explicit type (eg SEQ or MAP)
    NodeInit(NodeType_e t, NodeScalar const& k                     ) : type(t     ), key(k.tag, k.scalar), val(               ) { _add_flags(KEY); }

public:

    void clear()
    {
        type.clear();
        key.clear();
        val.clear();
    }

    void _add_flags(type_bits more_flags=0)
    {
        type = (type|more_flags);
        if( ! key.tag.empty())
            type = (type|KEYTAG);
        if( ! val.tag.empty())
            type = (type|VALTAG);
        if( ! key.anchor.empty())
            type = (type|KEYANCH);
        if( ! val.anchor.empty())
            type = (type|VALANCH);
    }

    bool _check() const
    {
        // key cannot be empty
        RYML_ASSERT(key.scalar.empty() == ((type & KEY) == 0));
        // key tag cannot be empty
        RYML_ASSERT(key.tag.empty() == ((type & KEYTAG) == 0));
        // val may be empty even though VAL is set. But when VAL is not set, val must be empty
        RYML_ASSERT(((type & VAL) != 0) || val.scalar.empty());
        // val tag cannot be empty
        RYML_ASSERT(val.tag.empty() == ((type & VALTAG) == 0));
        return true;
    }
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** contains the data for each YAML node. */
struct NodeData
{
    NodeType   m_type;

    NodeScalar m_key;
    NodeScalar m_val;

    size_t     m_parent;
    size_t     m_first_child;
    size_t     m_last_child;
    size_t     m_next_sibling;
    size_t     m_prev_sibling;
};
C4_MUST_BE_TRIVIAL_COPY(NodeData);


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class RYML_EXPORT Tree
{
public:

    /** @name construction and assignment */
    /** @{ */

    Tree() : Tree(get_callbacks()) {}
    Tree(Callbacks const& cb);
    Tree(size_t node_capacity, size_t arena_capacity=0) : Tree(node_capacity, arena_capacity, get_callbacks()) {}
    Tree(size_t node_capacity, size_t arena_capacity, Callbacks const& cb);

    ~Tree();

    Tree(Tree const& that) noexcept;
    Tree(Tree     && that) noexcept;

    Tree& operator= (Tree const& that) noexcept;
    Tree& operator= (Tree     && that) noexcept;

    /** @} */

public:

    /** @name memory and sizing */
    /** @{ */

    void reserve(size_t node_capacity);

    /** clear the tree and zero every node
     * @note does NOT clear the arena
     * @see clear_arena() */
    void clear();
    inline void clear_arena() { m_arena_pos = 0; }

    inline bool   empty() const { return m_size == 0; }

    inline size_t size() const { return m_size; }
    inline size_t capacity() const { return m_cap; }
    inline size_t slack() const { RYML_ASSERT(m_cap >= m_size); return m_cap - m_size; }

    Callbacks const& callbacks() const { return m_callbacks; }
    void callbacks(Callbacks const& cb) { m_callbacks = cb; }

    /** @} */

public:

    /** @name node getters */
    /** @{ */

    //! get the index of a node belonging to this tree.
    //! @p n can be nullptr, in which case a
    size_t id(NodeData const* n) const
    {
        if( ! n)
        {
            return NONE;
        }
        RYML_ASSERT(n >= m_buf && n < m_buf + m_cap);
        return static_cast<size_t>(n - m_buf);
    }

    //! get a pointer to a node's NodeData.
    //! i can be NONE, in which case a nullptr is returned
    inline NodeData *get(size_t i)
    {
        if(i == NONE)
            return nullptr;
        RYML_ASSERT(i >= 0 && i < m_cap);
        return m_buf + i;
    }
    //! get a pointer to a node's NodeData.
    //! i can be NONE, in which case a nullptr is returned.
    inline NodeData const *get(size_t i) const
    {
        if(i == NONE)
            return nullptr;
        RYML_ASSERT(i >= 0 && i < m_cap);
        return m_buf + i;
    }

    //! An if-less form of get() that demands a valid node index.
    //! This function is implementation only; use at your own risk.
    inline NodeData       * _p(size_t i)       { RYML_ASSERT(i != NONE && i >= 0 && i < m_cap); return m_buf + i; }
    //! An if-less form of get() that demands a valid node index.
    //! This function is implementation only; use at your own risk.
    inline NodeData const * _p(size_t i) const { RYML_ASSERT(i != NONE && i >= 0 && i < m_cap); return m_buf + i; }

    //! Get the id of the root node
    size_t root_id()       { if(m_cap == 0) { reserve(16); } RYML_ASSERT(m_cap > 0 && m_size > 0); return 0; }
    //! Get the id of the root node
    size_t root_id() const {                                 RYML_ASSERT(m_cap > 0 && m_size > 0); return 0; }

    //! Get a NodeRef of a node by id
    NodeRef      ref(size_t id);
    //! Get a NodeRef of a node by id
    ConstNodeRef ref(size_t id) const;
    //! Get a NodeRef of a node by id
    ConstNodeRef cref(size_t id);
    //! Get a NodeRef of a node by id
    ConstNodeRef cref(size_t id) const;

    //! Get the root as a NodeRef
    NodeRef      rootref();
    //! Get the root as a NodeRef
    ConstNodeRef rootref() const;
    //! Get the root as a NodeRef
    ConstNodeRef crootref();
    //! Get the root as a NodeRef
    ConstNodeRef crootref() const;

    //! find a root child by name, return it as a NodeRef
    //! @note requires the root to be a map.
    NodeRef      operator[] (csubstr key);
    //! find a root child by name, return it as a NodeRef
    //! @note requires the root to be a map.
    ConstNodeRef operator[] (csubstr key) const;

    //! find a root child by index: return the root node's @p i-th child as a NodeRef
    //! @note @i is NOT the node id, but the child's position
    NodeRef      operator[] (size_t i);
    //! find a root child by index: return the root node's @p i-th child as a NodeRef
    //! @note @i is NOT the node id, but the child's position
    ConstNodeRef operator[] (size_t i) const;

    //! get the i-th document of the stream
    //! @note @i is NOT the node id, but the doc position within the stream
    NodeRef      docref(size_t i);
    //! get the i-th document of the stream
    //! @note @i is NOT the node id, but the doc position within the stream
    ConstNodeRef docref(size_t i) const;

    /** @} */

public:

    /** @name node property getters */
    /** @{ */

    NodeType type(size_t node) const { return _p(node)->m_type; }
    const char* type_str(size_t node) const { return NodeType::type_str(_p(node)->m_type); }

    csubstr    const& key       (size_t node) const { RYML_ASSERT(has_key(node)); return _p(node)->m_key.scalar; }
    csubstr    const& key_tag   (size_t node) const { RYML_ASSERT(has_key_tag(node)); return _p(node)->m_key.tag; }
    csubstr    const& key_ref   (size_t node) const { RYML_ASSERT(is_key_ref(node) && ! has_key_anchor(node)); return _p(node)->m_key.anchor; }
    csubstr    const& key_anchor(size_t node) const { RYML_ASSERT( ! is_key_ref(node) && has_key_anchor(node)); return _p(node)->m_key.anchor; }
    NodeScalar const& keysc     (size_t node) const { RYML_ASSERT(has_key(node)); return _p(node)->m_key; }

    csubstr    const& val       (size_t node) const { RYML_ASSERT(has_val(node)); return _p(node)->m_val.scalar; }
    csubstr    const& val_tag   (size_t node) const { RYML_ASSERT(has_val_tag(node)); return _p(node)->m_val.tag; }
    csubstr    const& val_ref   (size_t node) const { RYML_ASSERT(is_val_ref(node) && ! has_val_anchor(node)); return _p(node)->m_val.anchor; }
    csubstr    const& val_anchor(size_t node) const { RYML_ASSERT( ! is_val_ref(node) && has_val_anchor(node)); return _p(node)->m_val.anchor; }
    NodeScalar const& valsc     (size_t node) const { RYML_ASSERT(has_val(node)); return _p(node)->m_val; }

    /** @} */

public:

    /** @name node predicates */
    /** @{ */

    C4_ALWAYS_INLINE bool is_stream(size_t node) const { return _p(node)->m_type.is_stream(); }
    C4_ALWAYS_INLINE bool is_doc(size_t node) const { return _p(node)->m_type.is_doc(); }
    C4_ALWAYS_INLINE bool is_container(size_t node) const { return _p(node)->m_type.is_container(); }
    C4_ALWAYS_INLINE bool is_map(size_t node) const { return _p(node)->m_type.is_map(); }
    C4_ALWAYS_INLINE bool is_seq(size_t node) const { return _p(node)->m_type.is_seq(); }
    C4_ALWAYS_INLINE bool has_key(size_t node) const { return _p(node)->m_type.has_key(); }
    C4_ALWAYS_INLINE bool has_val(size_t node) const { return _p(node)->m_type.has_val(); }
    C4_ALWAYS_INLINE bool is_val(size_t node) const { return _p(node)->m_type.is_val(); }
    C4_ALWAYS_INLINE bool is_keyval(size_t node) const { return _p(node)->m_type.is_keyval(); }
    C4_ALWAYS_INLINE bool has_key_tag(size_t node) const { return _p(node)->m_type.has_key_tag(); }
    C4_ALWAYS_INLINE bool has_val_tag(size_t node) const { return _p(node)->m_type.has_val_tag(); }
    C4_ALWAYS_INLINE bool has_key_anchor(size_t node) const { return _p(node)->m_type.has_key_anchor(); }
    C4_ALWAYS_INLINE bool is_key_anchor(size_t node) const { return _p(node)->m_type.is_key_anchor(); }
    C4_ALWAYS_INLINE bool has_val_anchor(size_t node) const { return _p(node)->m_type.has_val_anchor(); }
    C4_ALWAYS_INLINE bool is_val_anchor(size_t node) const { return _p(node)->m_type.is_val_anchor(); }
    C4_ALWAYS_INLINE bool has_anchor(size_t node) const { return _p(node)->m_type.has_anchor(); }
    C4_ALWAYS_INLINE bool is_anchor(size_t node) const { return _p(node)->m_type.is_anchor(); }
    C4_ALWAYS_INLINE bool is_key_ref(size_t node) const { return _p(node)->m_type.is_key_ref(); }
    C4_ALWAYS_INLINE bool is_val_ref(size_t node) const { return _p(node)->m_type.is_val_ref(); }
    C4_ALWAYS_INLINE bool is_ref(size_t node) const { return _p(node)->m_type.is_ref(); }
    C4_ALWAYS_INLINE bool is_anchor_or_ref(size_t node) const { return _p(node)->m_type.is_anchor_or_ref(); }
    C4_ALWAYS_INLINE bool is_key_quoted(size_t node) const { return _p(node)->m_type.is_key_quoted(); }
    C4_ALWAYS_INLINE bool is_val_quoted(size_t node) const { return _p(node)->m_type.is_val_quoted(); }
    C4_ALWAYS_INLINE bool is_quoted(size_t node) const { return _p(node)->m_type.is_quoted(); }

    C4_ALWAYS_INLINE bool parent_is_seq(size_t node) const { RYML_ASSERT(has_parent(node)); return is_seq(_p(node)->m_parent); }
    C4_ALWAYS_INLINE bool parent_is_map(size_t node) const { RYML_ASSERT(has_parent(node)); return is_map(_p(node)->m_parent); }

    /** true when key and val are empty, and has no children */
    C4_ALWAYS_INLINE bool empty(size_t node) const { return ! has_children(node) && _p(node)->m_key.empty() && (( ! (_p(node)->m_type & VAL)) || _p(node)->m_val.empty()); }
    /** true when the node has an anchor named a */
    C4_ALWAYS_INLINE bool has_anchor(size_t node, csubstr a) const { return _p(node)->m_key.anchor == a || _p(node)->m_val.anchor == a; }

    C4_ALWAYS_INLINE bool key_is_null(size_t node) const { RYML_ASSERT(has_key(node)); NodeData const* C4_RESTRICT n = _p(node); return !n->m_type.is_key_quoted() && scalar_is_null(n->m_key.scalar); }
    C4_ALWAYS_INLINE bool val_is_null(size_t node) const { RYML_ASSERT(has_val(node)); NodeData const* C4_RESTRICT n = _p(node); return !n->m_type.is_val_quoted() && scalar_is_null(n->m_val.scalar); }

    /** @todo move this function to node_type.hpp */
    static bool scalar_is_null(csubstr s) noexcept
    {
        return s.str == nullptr ||
            s == "~" ||
            s == "null" ||
            s == "Null" ||
            s == "NULL";
    }

    /** @} */

public:

    /** @name hierarchy predicates */
    /** @{ */

    bool is_root(size_t node) const { RYML_ASSERT(_p(node)->m_parent != NONE || node == 0); return _p(node)->m_parent == NONE; }

    bool has_parent(size_t node) const { return _p(node)->m_parent != NONE; }

    /** true if @p node has a child with id @p ch */
    bool has_child(size_t node, size_t ch) const { return _p(ch)->m_parent == node; }
    /** true if @p node has a child with key @p key */
    bool has_child(size_t node, csubstr key) const { return find_child(node, key) != npos; }
    /** true if @p node has any children key */
    bool has_children(size_t node) const { return _p(node)->m_first_child != NONE; }

    /** true if @p node has a sibling with id @p sib */
    bool has_sibling(size_t node, size_t sib) const { return _p(node)->m_parent == _p(sib)->m_parent; }
    /** true if one of the node's siblings has the given key */
    bool has_sibling(size_t node, csubstr key) const { return find_sibling(node, key) != npos; }
    /** true if node is not a single child */
    bool has_other_siblings(size_t node) const
    {
        NodeData const *n = _p(node);
        if(C4_LIKELY(n->m_parent != NONE))
        {
            n = _p(n->m_parent);
            return n->m_first_child != n->m_last_child;
        }
        return false;
    }

    RYML_DEPRECATED("use has_other_siblings()") bool has_siblings(size_t /*node*/) const { return true; }

    /** @} */

public:

    /** @name hierarchy getters */
    /** @{ */

    size_t parent(size_t node) const { return _p(node)->m_parent; }

    size_t prev_sibling(size_t node) const { return _p(node)->m_prev_sibling; }
    size_t next_sibling(size_t node) const { return _p(node)->m_next_sibling; }

    /** O(#num_children) */
    size_t num_children(size_t node) const;
    size_t child_pos(size_t node, size_t ch) const;
    size_t first_child(size_t node) const { return _p(node)->m_first_child; }
    size_t last_child(size_t node) const { return _p(node)->m_last_child; }
    size_t child(size_t node, size_t pos) const;
    size_t find_child(size_t node, csubstr const& key) const;

    /** O(#num_siblings) */
    /** counts with this */
    size_t num_siblings(size_t node) const { return is_root(node) ? 1 : num_children(_p(node)->m_parent); }
    /** does not count with this */
    size_t num_other_siblings(size_t node) const { size_t ns = num_siblings(node); RYML_ASSERT(ns > 0); return ns-1; }
    size_t sibling_pos(size_t node, size_t sib) const { RYML_ASSERT( ! is_root(node) || node == root_id()); return child_pos(_p(node)->m_parent, sib); }
    size_t first_sibling(size_t node) const { return is_root(node) ? node : _p(_p(node)->m_parent)->m_first_child; }
    size_t last_sibling(size_t node) const { return is_root(node) ? node : _p(_p(node)->m_parent)->m_last_child; }
    size_t sibling(size_t node, size_t pos) const { return child(_p(node)->m_parent, pos); }
    size_t find_sibling(size_t node, csubstr const& key) const { return find_child(_p(node)->m_parent, key); }

    size_t doc(size_t i) const { size_t rid = root_id(); RYML_ASSERT(is_stream(rid)); return child(rid, i); } //!< gets the @p i document node index. requires that the root node is a stream.

    /** @} */

public:

    /** @name node modifiers */
    /** @{ */

    void to_keyval(size_t node, csubstr key, csubstr val, type_bits more_flags=0);
    void to_map(size_t node, csubstr key, type_bits more_flags=0);
    void to_seq(size_t node, csubstr key, type_bits more_flags=0);
    void to_val(size_t node, csubstr val, type_bits more_flags=0);
    void to_map(size_t node, type_bits more_flags=0);
    void to_seq(size_t node, type_bits more_flags=0);
    void to_doc(size_t node, type_bits more_flags=0);
    void to_stream(size_t node, type_bits more_flags=0);

    void set_key(size_t node, csubstr key) { RYML_ASSERT(has_key(node)); _p(node)->m_key.scalar = key; }
    void set_val(size_t node, csubstr val) { RYML_ASSERT(has_val(node)); _p(node)->m_val.scalar = val; }

    void set_key_tag(size_t node, csubstr tag) { RYML_ASSERT(has_key(node)); _p(node)->m_key.tag = tag; _add_flags(node, KEYTAG); }
    void set_val_tag(size_t node, csubstr tag) { RYML_ASSERT(has_val(node) || is_container(node)); _p(node)->m_val.tag = tag; _add_flags(node, VALTAG); }

    void set_key_anchor(size_t node, csubstr anchor) { RYML_ASSERT( ! is_key_ref(node)); _p(node)->m_key.anchor = anchor.triml('&'); _add_flags(node, KEYANCH); }
    void set_val_anchor(size_t node, csubstr anchor) { RYML_ASSERT( ! is_val_ref(node)); _p(node)->m_val.anchor = anchor.triml('&'); _add_flags(node, VALANCH); }
    void set_key_ref   (size_t node, csubstr ref   ) { RYML_ASSERT( ! has_key_anchor(node)); NodeData* C4_RESTRICT n = _p(node); n->m_key.set_ref_maybe_replacing_scalar(ref, n->m_type.has_key()); _add_flags(node, KEY|KEYREF); }
    void set_val_ref   (size_t node, csubstr ref   ) { RYML_ASSERT( ! has_val_anchor(node)); NodeData* C4_RESTRICT n = _p(node); n->m_val.set_ref_maybe_replacing_scalar(ref, n->m_type.has_val()); _add_flags(node, VAL|VALREF); }

    void rem_key_anchor(size_t node) { _p(node)->m_key.anchor.clear(); _rem_flags(node, KEYANCH); }
    void rem_val_anchor(size_t node) { _p(node)->m_val.anchor.clear(); _rem_flags(node, VALANCH); }
    void rem_key_ref   (size_t node) { _p(node)->m_key.anchor.clear(); _rem_flags(node, KEYREF); }
    void rem_val_ref   (size_t node) { _p(node)->m_val.anchor.clear(); _rem_flags(node, VALREF); }
    void rem_anchor_ref(size_t node) { _p(node)->m_key.anchor.clear(); _p(node)->m_val.anchor.clear(); _rem_flags(node, KEYANCH|VALANCH|KEYREF|VALREF); }

    /** @} */

public:

    /** @name tree modifiers */
    /** @{ */

    /** reorder the tree in memory so that all the nodes are stored
     * in a linear sequence when visited in depth-first order.
     * This will invalidate existing ids, since the node id is its
     * position in the node array. */
    void reorder();

    /** Resolve references (aliases <- anchors) in the tree.
     *
     * Dereferencing is opt-in; after parsing, Tree::resolve()
     * has to be called explicitly for obtaining resolved references in the
     * tree. This method will resolve all references and substitute the
     * anchored values in place of the reference.
     *
     * This method first does a full traversal of the tree to gather all
     * anchors and references in a separate collection, then it goes through
     * that collection to locate the names, which it does by obeying the YAML
     * standard diktat that "an alias node refers to the most recent node in
     * the serialization having the specified anchor"
     *
     * So, depending on the number of anchor/alias nodes, this is a
     * potentially expensive operation, with a best-case linear complexity
     * (from the initial traversal). This potential cost is the reason for
     * requiring an explicit call.
     */
    void resolve();

    /** @} */

public:

    /** @name tag directives */
    /** @{ */

    void resolve_tags();

    size_t num_tag_directives() const;
    size_t add_tag_directive(TagDirective const& td);
    void clear_tag_directives();

    size_t resolve_tag(substr output, csubstr tag, size_t node_id) const;
    csubstr resolve_tag_sub(substr output, csubstr tag, size_t node_id) const
    {
        size_t needed = resolve_tag(output, tag, node_id);
        return needed <= output.len ? output.first(needed) : output;
    }

    using tag_directive_const_iterator = TagDirective const*;
    tag_directive_const_iterator begin_tag_directives() const { return m_tag_directives; }
    tag_directive_const_iterator end_tag_directives() const { return m_tag_directives + num_tag_directives(); }

    struct TagDirectiveProxy
    {
        tag_directive_const_iterator b, e;
        tag_directive_const_iterator begin() const { return b; }
        tag_directive_const_iterator end() const { return e; }
    };

    TagDirectiveProxy tag_directives() const { return TagDirectiveProxy{begin_tag_directives(), end_tag_directives()}; }

    /** @} */

public:

    /** @name modifying hierarchy */
    /** @{ */

    /** create and insert a new child of @p parent. insert after the (to-be)
     * sibling @p after, which must be a child of @p parent. To insert as the
     * first child, set after to NONE */
    C4_ALWAYS_INLINE size_t insert_child(size_t parent, size_t after)
    {
        RYML_ASSERT(parent != NONE);
        RYML_ASSERT(is_container(parent) || is_root(parent));
        RYML_ASSERT(after == NONE || (_p(after)->m_parent == parent));
        size_t child = _claim();
        _set_hierarchy(child, parent, after);
        return child;
    }
    /** create and insert a node as the first child of @p parent */
    C4_ALWAYS_INLINE size_t prepend_child(size_t parent) { return insert_child(parent, NONE); }
    /** create and insert a node as the last child of @p parent */
    C4_ALWAYS_INLINE size_t  append_child(size_t parent) { return insert_child(parent, _p(parent)->m_last_child); }

public:

    #if defined(__clang__)
    #   pragma clang diagnostic push
    #   pragma clang diagnostic ignored "-Wnull-dereference"
    #elif defined(__GNUC__)
    #   pragma GCC diagnostic push
    #   if __GNUC__ >= 6
    #       pragma GCC diagnostic ignored "-Wnull-dereference"
    #   endif
    #endif

    //! create and insert a new sibling of n. insert after "after"
    C4_ALWAYS_INLINE size_t insert_sibling(size_t node, size_t after)
    {
        return insert_child(_p(node)->m_parent, after);
    }
    /** create and insert a node as the first node of @p parent */
    C4_ALWAYS_INLINE size_t prepend_sibling(size_t node) { return prepend_child(_p(node)->m_parent); }
    C4_ALWAYS_INLINE size_t  append_sibling(size_t node) { return append_child(_p(node)->m_parent); }

public:

    /** remove an entire branch at once: ie remove the children and the node itself */
    inline void remove(size_t node)
    {
        remove_children(node);
        _release(node);
    }

    /** remove all the node's children, but keep the node itself */
    void remove_children(size_t node);

    /** change the @p type of the node to one of MAP, SEQ or VAL.  @p
     * type must have one and only one of MAP,SEQ,VAL; @p type may
     * possibly have KEY, but if it does, then the @p node must also
     * have KEY. Changing to the same type is a no-op. Otherwise,
     * changing to a different type will initialize the node with an
     * empty value of the desired type: changing to VAL will
     * initialize with a null scalar (~), changing to MAP will
     * initialize with an empty map ({}), and changing to SEQ will
     * initialize with an empty seq ([]). */
    bool change_type(size_t node, NodeType type);

    bool change_type(size_t node, type_bits type)
    {
        return change_type(node, (NodeType)type);
    }

    #if defined(__clang__)
    #   pragma clang diagnostic pop
    #elif defined(__GNUC__)
    #   pragma GCC diagnostic pop
    #endif

public:

    /** change the node's position in the parent */
    void move(size_t node, size_t after);

    /** change the node's parent and position */
    void move(size_t node, size_t new_parent, size_t after);

    /** change the node's parent and position to a different tree
     * @return the index of the new node in the destination tree */
    size_t move(Tree * src, size_t node, size_t new_parent, size_t after);

    /** ensure the first node is a stream. Eg, change this tree
     *
     *  DOCMAP
     *    MAP
     *      KEYVAL
     *      KEYVAL
     *    SEQ
     *      VAL
     *
     * to
     *
     *  STREAM
     *    DOCMAP
     *      MAP
     *        KEYVAL
     *        KEYVAL
     *      SEQ
     *        VAL
     *
     * If the root is already a stream, this is a no-op.
     */
    void set_root_as_stream();

public:

    /** recursively duplicate a node from this tree into a new parent,
     * placing it after one of its children
     * @return the index of the copy */
    size_t duplicate(size_t node, size_t new_parent, size_t after);
    /** recursively duplicate a node from a different tree into a new parent,
     * placing it after one of its children
     * @return the index of the copy */
    size_t duplicate(Tree const* src, size_t node, size_t new_parent, size_t after);

    /** recursively duplicate the node's children (but not the node)
     * @return the index of the last duplicated child */
    size_t duplicate_children(size_t node, size_t parent, size_t after);
    /** recursively duplicate the node's children (but not the node), where
     * the node is from a different tree
     * @return the index of the last duplicated child */
    size_t duplicate_children(Tree const* src, size_t node, size_t parent, size_t after);

    void duplicate_contents(size_t node, size_t where);
    void duplicate_contents(Tree const* src, size_t node, size_t where);

    /** duplicate the node's children (but not the node) in a new parent, but
     * omit repetitions where a duplicated node has the same key (in maps) or
     * value (in seqs). If one of the duplicated children has the same key
     * (in maps) or value (in seqs) as one of the parent's children, the one
     * that is placed closest to the end will prevail. */
    size_t duplicate_children_no_rep(size_t node, size_t parent, size_t after);
    size_t duplicate_children_no_rep(Tree const* src, size_t node, size_t parent, size_t after);

public:

    void merge_with(Tree const* src, size_t src_node=NONE, size_t dst_root=NONE);

    /** @} */

public:

    /** @name internal string arena */
    /** @{ */

    /** get the current size of the tree's internal arena */
    RYML_DEPRECATED("use arena_size() instead") size_t arena_pos() const { return m_arena_pos; }
    /** get the current size of the tree's internal arena */
    inline size_t arena_size() const { return m_arena_pos; }
    /** get the current capacity of the tree's internal arena */
    inline size_t arena_capacity() const { return m_arena.len; }
    /** get the current slack of the tree's internal arena */
    inline size_t arena_slack() const { RYML_ASSERT(m_arena.len >= m_arena_pos); return m_arena.len - m_arena_pos; }

    /** get the current arena */
    substr arena() const { return m_arena.first(m_arena_pos); }

    /** return true if the given substring is part of the tree's string arena */
    bool in_arena(csubstr s) const
    {
        return m_arena.is_super(s);
    }

    /** serialize the given floating-point variable to the tree's
     * arena, growing it as needed to accomodate the serialization.
     *
     * @note Growing the arena may cause relocation of the entire
     * existing arena, and thus change the contents of individual
     * nodes, and thus cost O(numnodes)+O(arenasize). To avoid this
     * cost, ensure that the arena is reserved to an appropriate size
     * using .reserve_arena()
     *
     * @see alloc_arena() */
    template<class T>
    typename std::enable_if<std::is_floating_point<T>::value, csubstr>::type
    to_arena(T const& C4_RESTRICT a)
    {
        substr rem(m_arena.sub(m_arena_pos));
        size_t num = to_chars_float(rem, a);
        if(num > rem.len)
        {
            rem = _grow_arena(num);
            num = to_chars_float(rem, a);
            RYML_ASSERT(num <= rem.len);
        }
        rem = _request_span(num);
        return rem;
    }
    /** serialize the given non-floating-point variable to the tree's
     * arena, growing it as needed to accomodate the serialization.
     *
     * @note Growing the arena may cause relocation of the entire
     * existing arena, and thus change the contents of individual
     * nodes, and thus cost O(numnodes)+O(arenasize). To avoid this
     * cost, ensure that the arena is reserved to an appropriate size
     * using .reserve_arena()
     *
     * @see alloc_arena() */
    template<class T>
    typename std::enable_if<!std::is_floating_point<T>::value, csubstr>::type
    to_arena(T const& C4_RESTRICT a)
    {
        substr rem(m_arena.sub(m_arena_pos));
        size_t num = to_chars(rem, a);
        if(num > rem.len)
        {
            rem = _grow_arena(num);
            num = to_chars(rem, a);
            RYML_ASSERT(num <= rem.len);
        }
        rem = _request_span(num);
        return rem;
    }

    /** serialize the given csubstr to the tree's arena, growing the
     * arena as needed to accomodate the serialization.
     *
     * @note Growing the arena may cause relocation of the entire
     * existing arena, and thus change the contents of individual
     * nodes, and thus cost O(numnodes)+O(arenasize). To avoid this
     * cost, ensure that the arena is reserved to an appropriate size
     * using .reserve_arena()
     *
     * @see alloc_arena() */
    csubstr to_arena(csubstr a)
    {
        if(a.len > 0)
        {
            substr rem(m_arena.sub(m_arena_pos));
            size_t num = to_chars(rem, a);
            if(num > rem.len)
            {
                rem = _grow_arena(num);
                num = to_chars(rem, a);
                RYML_ASSERT(num <= rem.len);
            }
            return _request_span(num);
        }
        else
        {
            if(a.str == nullptr)
            {
                return csubstr{};
            }
            else if(m_arena.str == nullptr)
            {
                // Arena is empty and we want to store a non-null
                // zero-length string.
                // Even though the string has zero length, we need
                // some "memory" to store a non-nullptr string
                _grow_arena(1);
            }
            return _request_span(0);
        }
    }
    C4_ALWAYS_INLINE csubstr to_arena(const char *s)
    {
        return to_arena(to_csubstr(s));
    }
    C4_ALWAYS_INLINE csubstr to_arena(std::nullptr_t)
    {
        return csubstr{};
    }

    /** copy the given substr to the tree's arena, growing it by the
     * required size
     *
     * @note Growing the arena may cause relocation of the entire
     * existing arena, and thus change the contents of individual
     * nodes, and thus cost O(numnodes)+O(arenasize). To avoid this
     * cost, ensure that the arena is reserved to an appropriate size
     * using .reserve_arena()
     *
     * @see alloc_arena() */
    substr copy_to_arena(csubstr s)
    {
        substr cp = alloc_arena(s.len);
        RYML_ASSERT(cp.len == s.len);
        RYML_ASSERT(!s.overlaps(cp));
        #if (!defined(__clang__)) && (defined(__GNUC__) && __GNUC__ >= 10)
        C4_SUPPRESS_WARNING_GCC_PUSH
        C4_SUPPRESS_WARNING_GCC("-Wstringop-overflow=") // no need for terminating \0
        C4_SUPPRESS_WARNING_GCC( "-Wrestrict") // there's an assert to ensure no violation of restrict behavior
        #endif
        if(s.len)
            memcpy(cp.str, s.str, s.len);
        #if (!defined(__clang__)) && (defined(__GNUC__) && __GNUC__ >= 10)
        C4_SUPPRESS_WARNING_GCC_POP
        #endif
        return cp;
    }

    /** grow the tree's string arena by the given size and return a substr
     * of the added portion
     *
     * @note Growing the arena may cause relocation of the entire
     * existing arena, and thus change the contents of individual
     * nodes, and thus cost O(numnodes)+O(arenasize). To avoid this
     * cost, ensure that the arena is reserved to an appropriate size
     * using .reserve_arena().
     *
     * @see reserve_arena() */
    substr alloc_arena(size_t sz)
    {
        if(sz > arena_slack())
            _grow_arena(sz - arena_slack());
        substr s = _request_span(sz);
        return s;
    }

    /** ensure the tree's internal string arena is at least the given capacity
     * @note This operation has a potential complexity of O(numNodes)+O(arenasize).
     * Growing the arena may cause relocation of the entire
     * existing arena, and thus change the contents of individual nodes. */
    void reserve_arena(size_t arena_cap)
    {
        if(arena_cap > m_arena.len)
        {
            substr buf;
            buf.str = (char*) m_callbacks.m_allocate(arena_cap, m_arena.str, m_callbacks.m_user_data);
            buf.len = arena_cap;
            if(m_arena.str)
            {
                RYML_ASSERT(m_arena.len >= 0);
                _relocate(buf); // does a memcpy and changes nodes using the arena
                m_callbacks.m_free(m_arena.str, m_arena.len, m_callbacks.m_user_data);
            }
            m_arena = buf;
        }
    }

    /** @} */

private:

    substr _grow_arena(size_t more)
    {
        size_t cap = m_arena.len + more;
        cap = cap < 2 * m_arena.len ? 2 * m_arena.len : cap;
        cap = cap < 64 ? 64 : cap;
        reserve_arena(cap);
        return m_arena.sub(m_arena_pos);
    }

    substr _request_span(size_t sz)
    {
        substr s;
        s = m_arena.sub(m_arena_pos, sz);
        m_arena_pos += sz;
        return s;
    }

    substr _relocated(csubstr s, substr next_arena) const
    {
        RYML_ASSERT(m_arena.is_super(s));
        RYML_ASSERT(m_arena.sub(0, m_arena_pos).is_super(s));
        auto pos = (s.str - m_arena.str);
        substr r(next_arena.str + pos, s.len);
        RYML_ASSERT(r.str - next_arena.str == pos);
        RYML_ASSERT(next_arena.sub(0, m_arena_pos).is_super(r));
        return r;
    }

public:

    /** @name lookup */
    /** @{ */

    struct lookup_result
    {
        size_t  target;
        size_t  closest;
        size_t  path_pos;
        csubstr path;

        inline operator bool() const { return target != NONE; }

        lookup_result() : target(NONE), closest(NONE), path_pos(0), path() {}
        lookup_result(csubstr path_, size_t start) : target(NONE), closest(start), path_pos(0), path(path_) {}

        /** get the part ot the input path that was resolved */
        csubstr resolved() const;
        /** get the part ot the input path that was unresolved */
        csubstr unresolved() const;
    };

    /** for example foo.bar[0].baz */
    lookup_result lookup_path(csubstr path, size_t start=NONE) const;

    /** defaulted lookup: lookup @p path; if the lookup fails, recursively modify
     * the tree so that the corresponding lookup_path() would return the
     * default value.
     * @see lookup_path() */
    size_t lookup_path_or_modify(csubstr default_value, csubstr path, size_t start=NONE);

    /** defaulted lookup: lookup @p path; if the lookup fails, recursively modify
     * the tree so that the corresponding lookup_path() would return the
     * branch @p src_node (from the tree @p src).
     * @see lookup_path() */
    size_t lookup_path_or_modify(Tree const *src, size_t src_node, csubstr path, size_t start=NONE);

    /** @} */

private:

    struct _lookup_path_token
    {
        csubstr value;
        NodeType type;
        _lookup_path_token() : value(), type() {}
        _lookup_path_token(csubstr v, NodeType t) : value(v), type(t) {}
        inline operator bool() const { return type != NOTYPE; }
        bool is_index() const { return value.begins_with('[') && value.ends_with(']'); }
    };

    size_t _lookup_path_or_create(csubstr path, size_t start);

    void   _lookup_path       (lookup_result *r) const;
    void   _lookup_path_modify(lookup_result *r);

    size_t _next_node       (lookup_result *r, _lookup_path_token *parent) const;
    size_t _next_node_modify(lookup_result *r, _lookup_path_token *parent);

    void   _advance(lookup_result *r, size_t more) const;

    _lookup_path_token _next_token(lookup_result *r, _lookup_path_token const& parent) const;

private:

    void _clear();
    void _free();
    void _copy(Tree const& that);
    void _move(Tree      & that);

    void _relocate(substr next_arena);

public:

    #if ! RYML_USE_ASSERT
    C4_ALWAYS_INLINE void _check_next_flags(size_t, type_bits) {}
    #else
    void _check_next_flags(size_t node, type_bits f)
    {
        auto n = _p(node);
        type_bits o = n->m_type; // old
        C4_UNUSED(o);
        if(f & MAP)
        {
            RYML_ASSERT_MSG((f & SEQ) == 0, "cannot mark simultaneously as map and seq");
            RYML_ASSERT_MSG((f & VAL) == 0, "cannot mark simultaneously as map and val");
            RYML_ASSERT_MSG((o & SEQ) == 0, "cannot turn a seq into a map; clear first");
            RYML_ASSERT_MSG((o & VAL) == 0, "cannot turn a val into a map; clear first");
        }
        else if(f & SEQ)
        {
            RYML_ASSERT_MSG((f & MAP) == 0, "cannot mark simultaneously as seq and map");
            RYML_ASSERT_MSG((f & VAL) == 0, "cannot mark simultaneously as seq and val");
            RYML_ASSERT_MSG((o & MAP) == 0, "cannot turn a map into a seq; clear first");
            RYML_ASSERT_MSG((o & VAL) == 0, "cannot turn a val into a seq; clear first");
        }
        if(f & KEY)
        {
            RYML_ASSERT(!is_root(node));
            auto pid = parent(node); C4_UNUSED(pid);
            RYML_ASSERT(is_map(pid));
        }
        if((f & VAL) && !is_root(node))
        {
            auto pid = parent(node); C4_UNUSED(pid);
            RYML_ASSERT(is_map(pid) || is_seq(pid));
        }
    }
    #endif

    inline void _set_flags(size_t node, NodeType_e f) { _check_next_flags(node, f); _p(node)->m_type = f; }
    inline void _set_flags(size_t node, type_bits  f) { _check_next_flags(node, f); _p(node)->m_type = f; }

    inline void _add_flags(size_t node, NodeType_e f) { NodeData *d = _p(node); type_bits fb = f |  d->m_type; _check_next_flags(node, fb); d->m_type = (NodeType_e) fb; }
    inline void _add_flags(size_t node, type_bits  f) { NodeData *d = _p(node);                f |= d->m_type; _check_next_flags(node,  f); d->m_type = f; }

    inline void _rem_flags(size_t node, NodeType_e f) { NodeData *d = _p(node); type_bits fb = d->m_type & ~f; _check_next_flags(node, fb); d->m_type = (NodeType_e) fb; }
    inline void _rem_flags(size_t node, type_bits  f) { NodeData *d = _p(node);            f = d->m_type & ~f; _check_next_flags(node,  f); d->m_type = f; }

    void _set_key(size_t node, csubstr key, type_bits more_flags=0)
    {
        _p(node)->m_key.scalar = key;
        _add_flags(node, KEY|more_flags);
    }
    void _set_key(size_t node, NodeScalar const& key, type_bits more_flags=0)
    {
        _p(node)->m_key = key;
        _add_flags(node, KEY|more_flags);
    }

    void _set_val(size_t node, csubstr val, type_bits more_flags=0)
    {
        RYML_ASSERT(num_children(node) == 0);
        RYML_ASSERT(!is_seq(node) && !is_map(node));
        _p(node)->m_val.scalar = val;
        _add_flags(node, VAL|more_flags);
    }
    void _set_val(size_t node, NodeScalar const& val, type_bits more_flags=0)
    {
        RYML_ASSERT(num_children(node) == 0);
        RYML_ASSERT( ! is_container(node));
        _p(node)->m_val = val;
        _add_flags(node, VAL|more_flags);
    }

    void _set(size_t node, NodeInit const& i)
    {
        RYML_ASSERT(i._check());
        NodeData *n = _p(node);
        RYML_ASSERT(n->m_key.scalar.empty() || i.key.scalar.empty() || i.key.scalar == n->m_key.scalar);
        _add_flags(node, i.type);
        if(n->m_key.scalar.empty())
        {
            if( ! i.key.scalar.empty())
            {
                _set_key(node, i.key.scalar);
            }
        }
        n->m_key.tag = i.key.tag;
        n->m_val = i.val;
    }

    void _set_parent_as_container_if_needed(size_t in)
    {
        NodeData const* n = _p(in);
        size_t ip = parent(in);
        if(ip != NONE)
        {
            if( ! (is_seq(ip) || is_map(ip)))
            {
                if((in == first_child(ip)) && (in == last_child(ip)))
                {
                    if( ! n->m_key.empty() || has_key(in))
                    {
                        _add_flags(ip, MAP);
                    }
                    else
                    {
                        _add_flags(ip, SEQ);
                    }
                }
            }
        }
    }

    void _seq2map(size_t node)
    {
        RYML_ASSERT(is_seq(node));
        for(size_t i = first_child(node); i != NONE; i = next_sibling(i))
        {
            NodeData *C4_RESTRICT ch = _p(i);
            if(ch->m_type.is_keyval())
                continue;
            ch->m_type.add(KEY);
            ch->m_key = ch->m_val;
        }
        auto *C4_RESTRICT n = _p(node);
        n->m_type.rem(SEQ);
        n->m_type.add(MAP);
    }

    size_t _do_reorder(size_t *node, size_t count);

    void _swap(size_t n_, size_t m_);
    void _swap_props(size_t n_, size_t m_);
    void _swap_hierarchy(size_t n_, size_t m_);
    void _copy_hierarchy(size_t dst_, size_t src_);

    inline void _copy_props(size_t dst_, size_t src_)
    {
        _copy_props(dst_, this, src_);
    }

    inline void _copy_props_wo_key(size_t dst_, size_t src_)
    {
        _copy_props_wo_key(dst_, this, src_);
    }

    void _copy_props(size_t dst_, Tree const* that_tree, size_t src_)
    {
        auto      & C4_RESTRICT dst = *_p(dst_);
        auto const& C4_RESTRICT src = *that_tree->_p(src_);
        dst.m_type = src.m_type;
        dst.m_key  = src.m_key;
        dst.m_val  = src.m_val;
    }

    void _copy_props_wo_key(size_t dst_, Tree const* that_tree, size_t src_)
    {
        auto      & C4_RESTRICT dst = *_p(dst_);
        auto const& C4_RESTRICT src = *that_tree->_p(src_);
        dst.m_type = (src.m_type & ~_KEYMASK) | (dst.m_type & _KEYMASK);
        dst.m_val  = src.m_val;
    }

    inline void _clear_type(size_t node)
    {
        _p(node)->m_type = NOTYPE;
    }

    inline void _clear(size_t node)
    {
        auto *C4_RESTRICT n = _p(node);
        n->m_type = NOTYPE;
        n->m_key.clear();
        n->m_val.clear();
        n->m_parent = NONE;
        n->m_first_child = NONE;
        n->m_last_child = NONE;
    }

    inline void _clear_key(size_t node)
    {
        _p(node)->m_key.clear();
        _rem_flags(node, KEY);
    }

    inline void _clear_val(size_t node)
    {
        _p(node)->m_val.clear();
        _rem_flags(node, VAL);
    }

private:

    void _clear_range(size_t first, size_t num);

    size_t _claim();
    void   _claim_root();
    void   _release(size_t node);
    void   _free_list_add(size_t node);
    void   _free_list_rem(size_t node);

    void _set_hierarchy(size_t node, size_t parent, size_t after_sibling);
    void _rem_hierarchy(size_t node);

public:

    // members are exposed, but you should NOT access them directly

    NodeData * m_buf;
    size_t m_cap;

    size_t m_size;

    size_t m_free_head;
    size_t m_free_tail;

    substr m_arena;
    size_t m_arena_pos;

    Callbacks m_callbacks;

    TagDirective m_tag_directives[RYML_MAX_TAG_DIRECTIVES];

};

} // namespace yml
} // namespace c4


C4_SUPPRESS_WARNING_MSVC_POP
C4_SUPPRESS_WARNING_GCC_CLANG_POP


#endif /* _C4_YML_TREE_HPP_ */
