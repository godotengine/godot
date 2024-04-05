#ifndef _C4_YML_EMIT_HPP_
#define _C4_YML_EMIT_HPP_

#ifndef _C4_YML_WRITER_HPP_
#include "./writer.hpp"
#endif

#ifndef _C4_YML_TREE_HPP_
#include "./tree.hpp"
#endif

#ifndef _C4_YML_NODE_HPP_
#include "./node.hpp"
#endif


#define RYML_DEPRECATE_EMIT                                             \
    RYML_DEPRECATED("use emit_yaml() instead. See https://github.com/biojppm/rapidyaml/issues/120")
#ifdef emit
#error "emit is defined, likely from a Qt include. This will cause a compilation error. See https://github.com/biojppm/rapidyaml/issues/120"
#endif
#define RYML_DEPRECATE_EMITRS                                           \
    RYML_DEPRECATED("use emitrs_yaml() instead. See https://github.com/biojppm/rapidyaml/issues/120")


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

namespace c4 {
namespace yml {

template<class Writer> class Emitter;

template<class OStream>
using EmitterOStream = Emitter<WriterOStream<OStream>>;
using EmitterFile = Emitter<WriterFile>;
using EmitterBuf  = Emitter<WriterBuf>;

typedef enum {
    EMIT_YAML = 0,
    EMIT_JSON = 1
} EmitType_e;


/** mark a tree or node to be emitted as json */
struct as_json
{
    Tree const* tree;
    size_t node;
    as_json(Tree const& t) : tree(&t), node(t.empty() ? NONE : t.root_id()) {}
    as_json(Tree const& t, size_t id) : tree(&t), node(id) {}
    as_json(ConstNodeRef const& n) : tree(n.tree()), node(n.id()) {}
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

template<class Writer>
class Emitter : public Writer
{
public:

    using Writer::Writer;

    /** emit!
     *
     * When writing to a buffer, returns a substr of the emitted YAML.
     * If the given buffer has insufficient space, the returned span will
     * be null and its size will be the needed space. No writes are done
     * after the end of the buffer.
     *
     * When writing to a file, the returned substr will be null, but its
     * length will be set to the number of bytes written. */
    substr emit_as(EmitType_e type, Tree const& t, size_t id, bool error_on_excess);
    /** emit starting at the root node */
    substr emit_as(EmitType_e type, Tree const& t, bool error_on_excess=true);
    /** emit the given node */
    substr emit_as(EmitType_e type, ConstNodeRef const& n, bool error_on_excess=true);

private:

    Tree const* C4_RESTRICT m_tree;

    void _emit_yaml(size_t id);
    void _do_visit_flow_sl(size_t id, size_t ilevel=0);
    void _do_visit_flow_ml(size_t id, size_t ilevel=0, size_t do_indent=1);
    void _do_visit_block(size_t id, size_t ilevel=0, size_t do_indent=1);
    void _do_visit_block_container(size_t id, size_t next_level, size_t do_indent);
    void _do_visit_json(size_t id);

private:

    void _write(NodeScalar const& C4_RESTRICT sc, NodeType flags, size_t level);
    void _write_json(NodeScalar const& C4_RESTRICT sc, NodeType flags);

    void _write_doc(size_t id);
    void _write_scalar(csubstr s, bool was_quoted);
    void _write_scalar_json(csubstr s, bool as_key, bool was_quoted);
    void _write_scalar_literal(csubstr s, size_t level, bool as_key, bool explicit_indentation=false);
    void _write_scalar_folded(csubstr s, size_t level, bool as_key);
    void _write_scalar_squo(csubstr s, size_t level);
    void _write_scalar_dquo(csubstr s, size_t level);
    void _write_scalar_plain(csubstr s, size_t level);

    void _write_tag(csubstr tag)
    {
        if(!tag.begins_with('!'))
            this->Writer::_do_write('!');
        this->Writer::_do_write(tag);
    }

    enum : type_bits {
        _keysc =  (KEY|KEYREF|KEYANCH|KEYQUO|_WIP_KEY_STYLE) | ~(VAL|VALREF|VALANCH|VALQUO|_WIP_VAL_STYLE),
        _valsc = ~(KEY|KEYREF|KEYANCH|KEYQUO|_WIP_KEY_STYLE) |  (VAL|VALREF|VALANCH|VALQUO|_WIP_VAL_STYLE),
        _keysc_json =  (KEY)  | ~(VAL),
        _valsc_json = ~(KEY)  |  (VAL),
    };

    C4_ALWAYS_INLINE void _writek(size_t id, size_t level) { _write(m_tree->keysc(id), m_tree->_p(id)->m_type.type & ~_valsc, level); }
    C4_ALWAYS_INLINE void _writev(size_t id, size_t level) { _write(m_tree->valsc(id), m_tree->_p(id)->m_type.type & ~_keysc, level); }

    C4_ALWAYS_INLINE void _writek_json(size_t id) { _write_json(m_tree->keysc(id), m_tree->_p(id)->m_type.type & ~(VAL)); }
    C4_ALWAYS_INLINE void _writev_json(size_t id) { _write_json(m_tree->valsc(id), m_tree->_p(id)->m_type.type & ~(KEY)); }

};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** emit YAML to the given file. A null file defaults to stdout.
 * Return the number of bytes written. */
inline size_t emit_yaml(Tree const& t, size_t id, FILE *f)
{
    EmitterFile em(f);
    return em.emit_as(EMIT_YAML, t, id, /*error_on_excess*/true).len;
}
RYML_DEPRECATE_EMIT inline size_t emit(Tree const& t, size_t id, FILE *f)
{
    return emit_yaml(t, id, f);
}

/** emit JSON to the given file. A null file defaults to stdout.
 * Return the number of bytes written. */
inline size_t emit_json(Tree const& t, size_t id, FILE *f)
{
    EmitterFile em(f);
    return em.emit_as(EMIT_JSON, t, id, /*error_on_excess*/true).len;
}


/** emit YAML to the given file. A null file defaults to stdout.
 * Return the number of bytes written.
 * @overload */
inline size_t emit_yaml(Tree const& t, FILE *f=nullptr)
{
    EmitterFile em(f);
    return em.emit_as(EMIT_YAML, t, /*error_on_excess*/true).len;
}
RYML_DEPRECATE_EMIT inline size_t emit(Tree const& t, FILE *f=nullptr)
{
    return emit_yaml(t, f);
}

/** emit JSON to the given file. A null file defaults to stdout.
 * Return the number of bytes written.
 * @overload */
inline size_t emit_json(Tree const& t, FILE *f=nullptr)
{
    EmitterFile em(f);
    return em.emit_as(EMIT_JSON, t, /*error_on_excess*/true).len;
}


/** emit YAML to the given file. A null file defaults to stdout.
 * Return the number of bytes written.
 * @overload */
inline size_t emit_yaml(ConstNodeRef const& r, FILE *f=nullptr)
{
    EmitterFile em(f);
    return em.emit_as(EMIT_YAML, r, /*error_on_excess*/true).len;
}
RYML_DEPRECATE_EMIT inline size_t emit(ConstNodeRef const& r, FILE *f=nullptr)
{
    return emit_yaml(r, f);
}

/** emit JSON to the given file. A null file defaults to stdout.
 * Return the number of bytes written.
 * @overload */
inline size_t emit_json(ConstNodeRef const& r, FILE *f=nullptr)
{
    EmitterFile em(f);
    return em.emit_as(EMIT_JSON, r, /*error_on_excess*/true).len;
}


//-----------------------------------------------------------------------------

/** emit YAML to an STL-like ostream */
template<class OStream>
inline OStream& operator<< (OStream& s, Tree const& t)
{
    EmitterOStream<OStream> em(s);
    em.emit_as(EMIT_YAML, t);
    return s;
}

/** emit YAML to an STL-like ostream
 * @overload */
template<class OStream>
inline OStream& operator<< (OStream& s, ConstNodeRef const& n)
{
    EmitterOStream<OStream> em(s);
    em.emit_as(EMIT_YAML, n);
    return s;
}

/** emit json to an STL-like stream */
template<class OStream>
inline OStream& operator<< (OStream& s, as_json const& j)
{
    EmitterOStream<OStream> em(s);
    em.emit_as(EMIT_JSON, *j.tree, j.node, true);
    return s;
}


//-----------------------------------------------------------------------------


/** emit YAML to the given buffer. Return a substr trimmed to the emitted YAML.
 * @param error_on_excess Raise an error if the space in the buffer is insufficient.
 * @overload */
inline substr emit_yaml(Tree const& t, size_t id, substr buf, bool error_on_excess=true)
{
    EmitterBuf em(buf);
    return em.emit_as(EMIT_YAML, t, id, error_on_excess);
}
RYML_DEPRECATE_EMIT inline substr emit(Tree const& t, size_t id, substr buf, bool error_on_excess=true)
{
    return emit_yaml(t, id, buf, error_on_excess);
}

/** emit JSON to the given buffer. Return a substr trimmed to the emitted JSON.
 * @param error_on_excess Raise an error if the space in the buffer is insufficient.
 * @overload */
inline substr emit_json(Tree const& t, size_t id, substr buf, bool error_on_excess=true)
{
    EmitterBuf em(buf);
    return em.emit_as(EMIT_JSON, t, id, error_on_excess);
}


/** emit YAML to the given buffer. Return a substr trimmed to the emitted YAML.
 * @param error_on_excess Raise an error if the space in the buffer is insufficient.
 * @overload */
inline substr emit_yaml(Tree const& t, substr buf, bool error_on_excess=true)
{
    EmitterBuf em(buf);
    return em.emit_as(EMIT_YAML, t, error_on_excess);
}
RYML_DEPRECATE_EMIT inline substr emit(Tree const& t, substr buf, bool error_on_excess=true)
{
    return emit_yaml(t, buf, error_on_excess);
}

/** emit JSON to the given buffer. Return a substr trimmed to the emitted JSON.
 * @param error_on_excess Raise an error if the space in the buffer is insufficient.
 * @overload */
inline substr emit_json(Tree const& t, substr buf, bool error_on_excess=true)
{
    EmitterBuf em(buf);
    return em.emit_as(EMIT_JSON, t, error_on_excess);
}


/** emit YAML to the given buffer. Return a substr trimmed to the emitted YAML.
 * @param error_on_excess Raise an error if the space in the buffer is insufficient.
 * @overload
 */
inline substr emit_yaml(ConstNodeRef const& r, substr buf, bool error_on_excess=true)
{
    EmitterBuf em(buf);
    return em.emit_as(EMIT_YAML, r, error_on_excess);
}
RYML_DEPRECATE_EMIT inline substr emit(ConstNodeRef const& r, substr buf, bool error_on_excess=true)
{
    return emit_yaml(r, buf, error_on_excess);
}

/** emit JSON to the given buffer. Return a substr trimmed to the emitted JSON.
 * @param error_on_excess Raise an error if the space in the buffer is insufficient.
 * @overload
 */
inline substr emit_json(ConstNodeRef const& r, substr buf, bool error_on_excess=true)
{
    EmitterBuf em(buf);
    return em.emit_as(EMIT_JSON, r, error_on_excess);
}


//-----------------------------------------------------------------------------

/** emit+resize: emit YAML to the given std::string/std::vector-like
 * container, resizing it as needed to fit the emitted YAML. */
template<class CharOwningContainer>
substr emitrs_yaml(Tree const& t, size_t id, CharOwningContainer * cont)
{
    substr buf = to_substr(*cont);
    substr ret = emit_yaml(t, id, buf, /*error_on_excess*/false);
    if(ret.str == nullptr && ret.len > 0)
    {
        cont->resize(ret.len);
        buf = to_substr(*cont);
        ret = emit_yaml(t, id, buf, /*error_on_excess*/true);
    }
    return ret;
}
template<class CharOwningContainer>
RYML_DEPRECATE_EMITRS substr emitrs(Tree const& t, size_t id, CharOwningContainer * cont)
{
    return emitrs_yaml(t, id, cont);
}

/** emit+resize: emit JSON to the given std::string/std::vector-like
 * container, resizing it as needed to fit the emitted JSON. */
template<class CharOwningContainer>
substr emitrs_json(Tree const& t, size_t id, CharOwningContainer * cont)
{
    substr buf = to_substr(*cont);
    substr ret = emit_json(t, id, buf, /*error_on_excess*/false);
    if(ret.str == nullptr && ret.len > 0)
    {
        cont->resize(ret.len);
        buf = to_substr(*cont);
        ret = emit_json(t, id, buf, /*error_on_excess*/true);
    }
    return ret;
}


/** emit+resize: emit YAML to the given std::string/std::vector-like
 * container, resizing it as needed to fit the emitted YAML. */
template<class CharOwningContainer>
CharOwningContainer emitrs_yaml(Tree const& t, size_t id)
{
    CharOwningContainer c;
    emitrs_yaml(t, id, &c);
    return c;
}
template<class CharOwningContainer>
RYML_DEPRECATE_EMITRS CharOwningContainer emitrs(Tree const& t, size_t id)
{
    CharOwningContainer c;
    emitrs_yaml(t, id, &c);
    return c;
}

/** emit+resize: emit JSON to the given std::string/std::vector-like
 * container, resizing it as needed to fit the emitted JSON. */
template<class CharOwningContainer>
CharOwningContainer emitrs_json(Tree const& t, size_t id)
{
    CharOwningContainer c;
    emitrs_json(t, id, &c);
    return c;
}


/** emit+resize: YAML to the given std::string/std::vector-like
 * container, resizing it as needed to fit the emitted YAML. */
template<class CharOwningContainer>
substr emitrs_yaml(Tree const& t, CharOwningContainer * cont)
{
    if(t.empty())
        return {};
    return emitrs_yaml(t, t.root_id(), cont);
}
template<class CharOwningContainer>
RYML_DEPRECATE_EMITRS substr emitrs(Tree const& t, CharOwningContainer * cont)
{
    return emitrs_yaml(t, cont);
}

/** emit+resize: JSON to the given std::string/std::vector-like
 * container, resizing it as needed to fit the emitted JSON. */
template<class CharOwningContainer>
substr emitrs_json(Tree const& t, CharOwningContainer * cont)
{
    if(t.empty())
        return {};
    return emitrs_json(t, t.root_id(), cont);
}


/** emit+resize: YAML to the given std::string/std::vector-like container,
 * resizing it as needed to fit the emitted YAML. */
template<class CharOwningContainer>
CharOwningContainer emitrs_yaml(Tree const& t)
{
    CharOwningContainer c;
    if(t.empty())
        return c;
    emitrs_yaml(t, t.root_id(), &c);
    return c;
}
template<class CharOwningContainer>
RYML_DEPRECATE_EMITRS CharOwningContainer emitrs(Tree const& t)
{
    return emitrs_yaml<CharOwningContainer>(t);
}

/** emit+resize: JSON to the given std::string/std::vector-like container,
 * resizing it as needed to fit the emitted JSON. */
template<class CharOwningContainer>
CharOwningContainer emitrs_json(Tree const& t)
{
    CharOwningContainer c;
    if(t.empty())
        return c;
    emitrs_json(t, t.root_id(), &c);
    return c;
}


/** emit+resize: YAML to the given std::string/std::vector-like container,
 * resizing it as needed to fit the emitted YAML. */
template<class CharOwningContainer>
substr emitrs_yaml(ConstNodeRef const& n, CharOwningContainer * cont)
{
    _RYML_CB_CHECK(n.tree()->callbacks(), n.valid());
    return emitrs_yaml(*n.tree(), n.id(), cont);
}
template<class CharOwningContainer>
RYML_DEPRECATE_EMITRS substr emitrs(ConstNodeRef const& n, CharOwningContainer * cont)
{
    return emitrs_yaml(n, cont);
}

/** emit+resize: JSON to the given std::string/std::vector-like container,
 * resizing it as needed to fit the emitted JSON. */
template<class CharOwningContainer>
substr emitrs_json(ConstNodeRef const& n, CharOwningContainer * cont)
{
    _RYML_CB_CHECK(n.tree()->callbacks(), n.valid());
    return emitrs_json(*n.tree(), n.id(), cont);
}


/** emit+resize: YAML to the given std::string/std::vector-like container,
 * resizing it as needed to fit the emitted YAML. */
template<class CharOwningContainer>
CharOwningContainer emitrs_yaml(ConstNodeRef const& n)
{
    _RYML_CB_CHECK(n.tree()->callbacks(), n.valid());
    CharOwningContainer c;
    emitrs_yaml(*n.tree(), n.id(), &c);
    return c;
}
template<class CharOwningContainer>
RYML_DEPRECATE_EMITRS CharOwningContainer emitrs(ConstNodeRef const& n)
{
    return emitrs_yaml<CharOwningContainer>(n);
}

/** emit+resize: JSON to the given std::string/std::vector-like container,
 * resizing it as needed to fit the emitted JSON. */
template<class CharOwningContainer>
CharOwningContainer emitrs_json(ConstNodeRef const& n)
{
    _RYML_CB_CHECK(n.tree()->callbacks(), n.valid());
    CharOwningContainer c;
    emitrs_json(*n.tree(), n.id(), &c);
    return c;
}

} // namespace yml
} // namespace c4

#undef RYML_DEPRECATE_EMIT
#undef RYML_DEPRECATE_EMITRS

#include "c4/yml/emit.def.hpp"

#endif /* _C4_YML_EMIT_HPP_ */
