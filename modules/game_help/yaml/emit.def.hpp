#ifndef _C4_YML_EMIT_DEF_HPP_
#define _C4_YML_EMIT_DEF_HPP_

#ifndef _C4_YML_EMIT_HPP_
#include "emit.hpp"
#endif

namespace c4 {
namespace yml {

template<class Writer>
substr Emitter<Writer>::emit_as(EmitType_e type, Tree const& t, size_t id, bool error_on_excess)
{
    if(t.empty())
    {
        _RYML_CB_ASSERT(t.callbacks(), id == NONE);
        return {};
    }
    _RYML_CB_CHECK(t.callbacks(), id < t.capacity());
    m_tree = &t;
    if(type == EMIT_YAML)
        _emit_yaml(id);
    else if(type == EMIT_JSON)
        _do_visit_json(id);
    else
        _RYML_CB_ERR(m_tree->callbacks(), "unknown emit type");
    return this->Writer::_get(error_on_excess);
}

template<class Writer>
substr Emitter<Writer>::emit_as(EmitType_e type, Tree const& t, bool error_on_excess)
{
    if(t.empty())
        return {};
    return this->emit_as(type, t, t.root_id(), error_on_excess);
}

template<class Writer>
substr Emitter<Writer>::emit_as(EmitType_e type, ConstNodeRef const& n, bool error_on_excess)
{
    _RYML_CB_CHECK(n.tree()->callbacks(), n.valid());
    return this->emit_as(type, *n.tree(), n.id(), error_on_excess);
}


//-----------------------------------------------------------------------------

template<class Writer>
void Emitter<Writer>::_emit_yaml(size_t id)
{
    // save branches in the visitor by doing the initial stream/doc
    // logic here, sparing the need to check stream/val/keyval inside
    // the visitor functions
    auto dispatch = [this](size_t node){
        NodeType ty = m_tree->type(node);
        if(ty.marked_flow_sl())
            _do_visit_flow_sl(node, 0);
        else if(ty.marked_flow_ml())
            _do_visit_flow_ml(node, 0);
        else
        {
            _do_visit_block(node, 0);
        }
    };
    if(!m_tree->is_root(id))
    {
        if(m_tree->is_container(id) && !m_tree->type(id).marked_flow())
        {
            size_t ilevel = 0;
            if(m_tree->has_key(id))
            {
                this->Writer::_do_write(m_tree->key(id));
                this->Writer::_do_write(":\n");
                ++ilevel;
            }
            _do_visit_block_container(id, ilevel, ilevel);
            return;
        }
    }

    auto *btd = m_tree->tag_directives().b;
    auto *etd = m_tree->tag_directives().e;
    auto write_tag_directives = [&btd, etd, this](size_t next_node){
        auto end = btd;
        while(end < etd)
        {
            if(end->next_node_id > next_node)
                break;
            ++end;
        }
        for( ; btd != end; ++btd)
        {
            if(next_node != m_tree->first_child(m_tree->parent(next_node)))
                this->Writer::_do_write("...\n");
            this->Writer::_do_write("%TAG ");
            this->Writer::_do_write(btd->handle);
            this->Writer::_do_write(' ');
            this->Writer::_do_write(btd->prefix);
            this->Writer::_do_write('\n');
        }
    };
    if(m_tree->is_stream(id))
    {
        if(m_tree->first_child(id) != NONE)
            write_tag_directives(m_tree->first_child(id));
        for(size_t child = m_tree->first_child(id); child != NONE; child = m_tree->next_sibling(child))
        {
            dispatch(child);
            if(m_tree->next_sibling(child) != NONE)
                write_tag_directives(m_tree->next_sibling(child));
        }
    }
    else if(m_tree->is_container(id))
    {
        dispatch(id);
    }
    else if(m_tree->is_doc(id))
    {
        _RYML_CB_ASSERT(m_tree->callbacks(), !m_tree->is_container(id)); // checked above
        _RYML_CB_ASSERT(m_tree->callbacks(), m_tree->is_val(id)); // so it must be a val
        _write_doc(id);
    }
    else if(m_tree->is_keyval(id))
    {
        _writek(id, 0);
        this->Writer::_do_write(": ");
        _writev(id, 0);
        if(!m_tree->type(id).marked_flow())
            this->Writer::_do_write('\n');
    }
    else if(m_tree->is_val(id))
    {
        //this->Writer::_do_write("- ");
        _writev(id, 0);
        if(!m_tree->type(id).marked_flow())
            this->Writer::_do_write('\n');
    }
    else if(m_tree->type(id) == NOTYPE)
    {
        ;
    }
    else
    {
        _RYML_CB_ERR(m_tree->callbacks(), "unknown type");
    }
}

template<class Writer>
void Emitter<Writer>::_write_doc(size_t id)
{
    RYML_ASSERT(m_tree->is_doc(id));
    if(!m_tree->is_root(id))
    {
        RYML_ASSERT(m_tree->is_stream(m_tree->parent(id)));
        this->Writer::_do_write("---");
    }
    if(!m_tree->has_val(id)) // this is more frequent
    {
        if(m_tree->has_val_tag(id))
        {
            if(!m_tree->is_root(id))
                this->Writer::_do_write(' ');
            _write_tag(m_tree->val_tag(id));
        }
        if(m_tree->has_val_anchor(id))
        {
            if(!m_tree->is_root(id))
                this->Writer::_do_write(' ');
            this->Writer::_do_write('&');
            this->Writer::_do_write(m_tree->val_anchor(id));
        }
    }
    else // docval
    {
        RYML_ASSERT(m_tree->has_val(id));
        RYML_ASSERT(!m_tree->has_key(id));
        if(!m_tree->is_root(id))
            this->Writer::_do_write(' ');
        _writev(id, 0);
    }
    this->Writer::_do_write('\n');
}

template<class Writer>
void Emitter<Writer>::_do_visit_flow_sl(size_t node, size_t ilevel)
{
    RYML_ASSERT(!m_tree->is_stream(node));
    RYML_ASSERT(m_tree->is_container(node) || m_tree->is_doc(node));
    RYML_ASSERT(m_tree->is_root(node) || (m_tree->parent_is_map(node) || m_tree->parent_is_seq(node)));

    if(m_tree->is_doc(node))
    {
        _write_doc(node);
        if(!m_tree->has_children(node))
            return;
    }
    else if(m_tree->is_container(node))
    {
        RYML_ASSERT(m_tree->is_map(node) || m_tree->is_seq(node));

        bool spc = false; // write a space

        if(m_tree->has_key(node))
        {
            _writek(node, ilevel);
            this->Writer::_do_write(':');
            spc = true;
        }

        if(m_tree->has_val_tag(node))
        {
            if(spc)
                this->Writer::_do_write(' ');
            _write_tag(m_tree->val_tag(node));
            spc = true;
        }

        if(m_tree->has_val_anchor(node))
        {
            if(spc)
                this->Writer::_do_write(' ');
            this->Writer::_do_write('&');
            this->Writer::_do_write(m_tree->val_anchor(node));
            spc = true;
        }

        if(spc)
            this->Writer::_do_write(' ');

        if(m_tree->is_map(node))
        {
            this->Writer::_do_write('{');
        }
        else
        {
            _RYML_CB_ASSERT(m_tree->callbacks(), m_tree->is_seq(node));
            this->Writer::_do_write('[');
        }
    } // container

    for(size_t child = m_tree->first_child(node), count = 0; child != NONE; child = m_tree->next_sibling(child))
    {
        if(count++)
            this->Writer::_do_write(',');
        if(m_tree->is_keyval(child))
        {
            _writek(child, ilevel);
            this->Writer::_do_write(": ");
            _writev(child, ilevel);
        }
        else if(m_tree->is_val(child))
        {
            _writev(child, ilevel);
        }
        else
        {
            // with single-line flow, we can never go back to block
            _do_visit_flow_sl(child, ilevel + 1);
        }
    }

    if(m_tree->is_map(node))
    {
        this->Writer::_do_write('}');
    }
    else if(m_tree->is_seq(node))
    {
        this->Writer::_do_write(']');
    }
}

template<class Writer>
void Emitter<Writer>::_do_visit_flow_ml(size_t id, size_t ilevel, size_t do_indent)
{
    C4_UNUSED(id);
    C4_UNUSED(ilevel);
    C4_UNUSED(do_indent);
    RYML_CHECK(false/*not implemented*/);
}

template<class Writer>
void Emitter<Writer>::_do_visit_block_container(size_t node, size_t next_level, size_t do_indent)
{
    RepC ind = indent_to(do_indent * next_level);

    if(m_tree->is_seq(node))
    {
        for(size_t child = m_tree->first_child(node); child != NONE; child = m_tree->next_sibling(child))
        {
            _RYML_CB_ASSERT(m_tree->callbacks(), !m_tree->has_key(child));
            if(m_tree->is_val(child))
            {
                this->Writer::_do_write(ind);
                this->Writer::_do_write("- ");
                _writev(child, next_level);
                this->Writer::_do_write('\n');
            }
            else
            {
                _RYML_CB_ASSERT(m_tree->callbacks(), m_tree->is_container(child));
                NodeType ty = m_tree->type(child);
                if(ty.marked_flow_sl())
                {
                    this->Writer::_do_write(ind);
                    this->Writer::_do_write("- ");
                    _do_visit_flow_sl(child, 0u);
                    this->Writer::_do_write('\n');
                }
                else if(ty.marked_flow_ml())
                {
                    this->Writer::_do_write(ind);
                    this->Writer::_do_write("- ");
                    _do_visit_flow_ml(child, next_level, do_indent);
                    this->Writer::_do_write('\n');
                }
                else
                {
                    _do_visit_block(child, next_level, do_indent);
                }
            }
            do_indent = true;
            ind = indent_to(do_indent * next_level);
        }
    }
    else // map
    {
        _RYML_CB_ASSERT(m_tree->callbacks(), m_tree->is_map(node));
        for(size_t ich = m_tree->first_child(node); ich != NONE; ich = m_tree->next_sibling(ich))
        {
            _RYML_CB_ASSERT(m_tree->callbacks(), m_tree->has_key(ich));
            if(m_tree->is_keyval(ich))
            {
                this->Writer::_do_write(ind);
                _writek(ich, next_level);
                this->Writer::_do_write(": ");
                _writev(ich, next_level);
                this->Writer::_do_write('\n');
            }
            else
            {
                _RYML_CB_ASSERT(m_tree->callbacks(), m_tree->is_container(ich));
                NodeType ty = m_tree->type(ich);
                if(ty.marked_flow_sl())
                {
                    this->Writer::_do_write(ind);
                    _do_visit_flow_sl(ich, 0u);
                    this->Writer::_do_write('\n');
                }
                else if(ty.marked_flow_ml())
                {
                    this->Writer::_do_write(ind);
                    _do_visit_flow_ml(ich, 0u);
                    this->Writer::_do_write('\n');
                }
                else
                {
                    _do_visit_block(ich, next_level, do_indent);
                }
            }
            do_indent = true;
            ind = indent_to(do_indent * next_level);
        }
    }
}

template<class Writer>
void Emitter<Writer>::_do_visit_block(size_t node, size_t ilevel, size_t do_indent)
{
    RYML_ASSERT(!m_tree->is_stream(node));
    RYML_ASSERT(m_tree->is_container(node) || m_tree->is_doc(node));
    RYML_ASSERT(m_tree->is_root(node) || (m_tree->parent_is_map(node) || m_tree->parent_is_seq(node)));
    RepC ind = indent_to(do_indent * ilevel);

    if(m_tree->is_doc(node))
    {
        _write_doc(node);
        if(!m_tree->has_children(node))
            return;
    }
    else if(m_tree->is_container(node))
    {
        RYML_ASSERT(m_tree->is_map(node) || m_tree->is_seq(node));

        bool spc = false; // write a space
        bool nl = false;  // write a newline

        if(m_tree->has_key(node))
        {
            this->Writer::_do_write(ind);
            _writek(node, ilevel);
            this->Writer::_do_write(':');
            spc = true;
        }
        else if(!m_tree->is_root(node))
        {
            this->Writer::_do_write(ind);
            this->Writer::_do_write('-');
            spc = true;
        }

        if(m_tree->has_val_tag(node))
        {
            if(spc)
                this->Writer::_do_write(' ');
            _write_tag(m_tree->val_tag(node));
            spc = true;
            nl = true;
        }

        if(m_tree->has_val_anchor(node))
        {
            if(spc)
                this->Writer::_do_write(' ');
            this->Writer::_do_write('&');
            this->Writer::_do_write(m_tree->val_anchor(node));
            spc = true;
            nl = true;
        }

        if(m_tree->has_children(node))
        {
            if(m_tree->has_key(node))
                nl = true;
            else
                if(!m_tree->is_root(node) && !nl)
                    spc = true;
        }
        else
        {
            if(m_tree->is_seq(node))
                this->Writer::_do_write(" []\n");
            else if(m_tree->is_map(node))
                this->Writer::_do_write(" {}\n");
            return;
        }

        if(spc && !nl)
            this->Writer::_do_write(' ');

        do_indent = 0;
        if(nl)
        {
            this->Writer::_do_write('\n');
            do_indent = 1;
        }
    } // container

    size_t next_level = ilevel + 1;
    if(m_tree->is_root(node) || m_tree->is_doc(node))
        next_level = ilevel; // do not indent at top level

    _do_visit_block_container(node, next_level, do_indent);
}

template<class Writer>
void Emitter<Writer>::_do_visit_json(size_t id)
{
    _RYML_CB_CHECK(m_tree->callbacks(), !m_tree->is_stream(id)); // JSON does not have streams
    if(m_tree->is_keyval(id))
    {
        _writek_json(id);
        this->Writer::_do_write(": ");
        _writev_json(id);
    }
    else if(m_tree->is_val(id))
    {
        _writev_json(id);
    }
    else if(m_tree->is_container(id))
    {
        if(m_tree->has_key(id))
        {
            _writek_json(id);
            this->Writer::_do_write(": ");
        }
        if(m_tree->is_seq(id))
            this->Writer::_do_write('[');
        else if(m_tree->is_map(id))
            this->Writer::_do_write('{');
    }  // container

    for(size_t ich = m_tree->first_child(id); ich != NONE; ich = m_tree->next_sibling(ich))
    {
        if(ich != m_tree->first_child(id))
            this->Writer::_do_write(',');
        _do_visit_json(ich);
    }

    if(m_tree->is_seq(id))
        this->Writer::_do_write(']');
    else if(m_tree->is_map(id))
        this->Writer::_do_write('}');
}

template<class Writer>
void Emitter<Writer>::_write(NodeScalar const& C4_RESTRICT sc, NodeType flags, size_t ilevel)
{
    if( ! sc.tag.empty())
    {
        _write_tag(sc.tag);
        this->Writer::_do_write(' ');
    }
    if(flags.has_anchor())
    {
        RYML_ASSERT(flags.is_ref() != flags.has_anchor());
        RYML_ASSERT( ! sc.anchor.empty());
        this->Writer::_do_write('&');
        this->Writer::_do_write(sc.anchor);
        this->Writer::_do_write(' ');
    }
    else if(flags.is_ref())
    {
        if(sc.anchor != "<<")
            this->Writer::_do_write('*');
        this->Writer::_do_write(sc.anchor);
        return;
    }

    // ensure the style flags only have one of KEY or VAL
    _RYML_CB_ASSERT(m_tree->callbacks(), ((flags & (_WIP_KEY_STYLE|_WIP_VAL_STYLE)) == 0) || (((flags&_WIP_KEY_STYLE) == 0) != ((flags&_WIP_VAL_STYLE) == 0)));

    auto style_marks = flags & (_WIP_KEY_STYLE|_WIP_VAL_STYLE);
    if(style_marks & (_WIP_KEY_LITERAL|_WIP_VAL_LITERAL))
    {
        _write_scalar_literal(sc.scalar, ilevel, flags.has_key());
    }
    else if(style_marks & (_WIP_KEY_FOLDED|_WIP_VAL_FOLDED))
    {
        _write_scalar_folded(sc.scalar, ilevel, flags.has_key());
    }
    else if(style_marks & (_WIP_KEY_SQUO|_WIP_VAL_SQUO))
    {
        _write_scalar_squo(sc.scalar, ilevel);
    }
    else if(style_marks & (_WIP_KEY_DQUO|_WIP_VAL_DQUO))
    {
        _write_scalar_dquo(sc.scalar, ilevel);
    }
    else if(style_marks & (_WIP_KEY_PLAIN|_WIP_VAL_PLAIN))
    {
        _write_scalar_plain(sc.scalar, ilevel);
    }
    else if(!style_marks)
    {
        size_t first_non_nl = sc.scalar.first_not_of('\n');
        bool all_newlines = first_non_nl == npos;
        bool has_leading_ws = (!all_newlines) && sc.scalar.sub(first_non_nl).begins_with_any(" \t");
        bool do_literal = ((!sc.scalar.empty() && all_newlines) || (has_leading_ws && !sc.scalar.trim(' ').empty()));
        if(do_literal)
        {
            _write_scalar_literal(sc.scalar, ilevel, flags.has_key(), /*explicit_indentation*/has_leading_ws);
        }
        else
        {
            for(size_t i = 0; i < sc.scalar.len; ++i)
            {
                if(sc.scalar.str[i] == '\n')
                {
                    _write_scalar_literal(sc.scalar, ilevel, flags.has_key(), /*explicit_indentation*/has_leading_ws);
                    goto wrote_special;
                }
                // todo: check for escaped characters requiring double quotes
            }
            _write_scalar(sc.scalar, flags.is_quoted());
        wrote_special:
            ;
        }
    }
    else
    {
        _RYML_CB_ERR(m_tree->callbacks(), "not implemented");
    }
}
template<class Writer>
void Emitter<Writer>::_write_json(NodeScalar const& C4_RESTRICT sc, NodeType flags)
{
    if(C4_UNLIKELY( ! sc.tag.empty()))
        _RYML_CB_ERR(m_tree->callbacks(), "JSON does not have tags");
    if(C4_UNLIKELY(flags.has_anchor()))
        _RYML_CB_ERR(m_tree->callbacks(), "JSON does not have anchors");
    _write_scalar_json(sc.scalar, flags.has_key(), flags.is_quoted());
}

#define _rymlindent_nextline() for(size_t lv = 0; lv < ilevel+1; ++lv) { this->Writer::_do_write(' '); this->Writer::_do_write(' '); }

template<class Writer>
void Emitter<Writer>::_write_scalar_literal(csubstr s, size_t ilevel, bool explicit_key, bool explicit_indentation)
{
    if(explicit_key)
        this->Writer::_do_write("? ");
    csubstr trimmed = s.trimr("\n\r");
    size_t numnewlines_at_end = s.len - trimmed.len - s.sub(trimmed.len).count('\r');
    //
    if(!explicit_indentation)
        this->Writer::_do_write('|');
    else
        this->Writer::_do_write("|2");
    //
    if(numnewlines_at_end > 1 || (trimmed.len == 0 && s.len > 0)/*only newlines*/)
        this->Writer::_do_write("+\n");
    else if(numnewlines_at_end == 1)
        this->Writer::_do_write('\n');
    else
        this->Writer::_do_write("-\n");
    //
    if(trimmed.len)
    {
        size_t pos = 0; // tracks the last character that was already written
        for(size_t i = 0; i < trimmed.len; ++i)
        {
            if(trimmed[i] != '\n')
                continue;
            // write everything up to this point
            csubstr since_pos = trimmed.range(pos, i+1); // include the newline
            _rymlindent_nextline()
            this->Writer::_do_write(since_pos);
            pos = i+1; // already written
        }
        if(pos < trimmed.len)
        {
            _rymlindent_nextline()
            this->Writer::_do_write(trimmed.sub(pos));
        }
        if(numnewlines_at_end)
        {
            this->Writer::_do_write('\n');
            --numnewlines_at_end;
        }
    }
    for(size_t i = 0; i < numnewlines_at_end; ++i)
    {
        _rymlindent_nextline()
        if(i+1 < numnewlines_at_end || explicit_key)
            this->Writer::_do_write('\n');
    }
    if(explicit_key && !numnewlines_at_end)
        this->Writer::_do_write('\n');
}

template<class Writer>
void Emitter<Writer>::_write_scalar_folded(csubstr s, size_t ilevel, bool explicit_key)
{
    if(explicit_key)
    {
        this->Writer::_do_write("? ");
    }
    RYML_ASSERT(s.find("\r") == csubstr::npos);
    csubstr trimmed = s.trimr('\n');
    size_t numnewlines_at_end = s.len - trimmed.len;
    if(numnewlines_at_end == 0)
    {
        this->Writer::_do_write(">-\n");
    }
    else if(numnewlines_at_end == 1)
    {
        this->Writer::_do_write(">\n");
    }
    else if(numnewlines_at_end > 1)
    {
        this->Writer::_do_write(">+\n");
    }
    if(trimmed.len)
    {
        size_t pos = 0; // tracks the last character that was already written
        for(size_t i = 0; i < trimmed.len; ++i)
        {
            if(trimmed[i] != '\n')
                continue;
            // write everything up to this point
            csubstr since_pos = trimmed.range(pos, i+1); // include the newline
            pos = i+1; // because of the newline
            _rymlindent_nextline()
            this->Writer::_do_write(since_pos);
            this->Writer::_do_write('\n'); // write the newline twice
        }
        if(pos < trimmed.len)
        {
            _rymlindent_nextline()
            this->Writer::_do_write(trimmed.sub(pos));
        }
        if(numnewlines_at_end)
        {
            this->Writer::_do_write('\n');
            --numnewlines_at_end;
        }
    }
    for(size_t i = 0; i < numnewlines_at_end; ++i)
    {
        _rymlindent_nextline()
        if(i+1 < numnewlines_at_end || explicit_key)
            this->Writer::_do_write('\n');
    }
    if(explicit_key && !numnewlines_at_end)
        this->Writer::_do_write('\n');
}

template<class Writer>
void Emitter<Writer>::_write_scalar_squo(csubstr s, size_t ilevel)
{
    size_t pos = 0; // tracks the last character that was already written
    this->Writer::_do_write('\'');
    for(size_t i = 0; i < s.len; ++i)
    {
        if(s[i] == '\n')
        {
            csubstr sub = s.range(pos, i+1);
            this->Writer::_do_write(sub);  // write everything up to (including) this char
            this->Writer::_do_write('\n'); // write the character again
            if(i + 1 < s.len)
                _rymlindent_nextline()     // indent the next line
            pos = i+1;
        }
        else if(s[i] == '\'')
        {
            csubstr sub = s.range(pos, i+1);
            this->Writer::_do_write(sub); // write everything up to (including) this char
            this->Writer::_do_write('\''); // write the character again
            pos = i+1;
        }
    }
    // write missing characters at the end of the string
    if(pos < s.len)
        this->Writer::_do_write(s.sub(pos));
    this->Writer::_do_write('\'');
}

template<class Writer>
void Emitter<Writer>::_write_scalar_dquo(csubstr s, size_t ilevel)
{
    size_t pos = 0; // tracks the last character that was already written
    this->Writer::_do_write('"');
    for(size_t i = 0; i < s.len; ++i)
    {
        const char curr = s.str[i];
        if(curr == '"' || curr == '\\')
        {
            csubstr sub = s.range(pos, i);
            this->Writer::_do_write(sub);  // write everything up to (excluding) this char
            this->Writer::_do_write('\\'); // write the escape
            this->Writer::_do_write(curr); // write the char
            pos = i+1;
        }
        else if(s[i] == '\n')
        {
            csubstr sub = s.range(pos, i+1);
            this->Writer::_do_write(sub);  // write everything up to (including) this newline
            this->Writer::_do_write('\n'); // write the newline again
            if(i + 1 < s.len)
                _rymlindent_nextline()     // indent the next line
            pos = i+1;
            if(i+1 < s.len) // escape leading whitespace after the newline
            {
                const char next = s.str[i+1];
                if(next == ' ' || next == '\t')
                    this->Writer::_do_write('\\');
            }
        }
        else if(curr == ' ' || curr == '\t')
        {
            // escape trailing whitespace before a newline
            size_t next = s.first_not_of(" \t\r", i);
            if(next != npos && s[next] == '\n')
            {
                csubstr sub = s.range(pos, i);
                this->Writer::_do_write(sub);  // write everything up to (excluding) this char
                this->Writer::_do_write('\\'); // escape the whitespace
                pos = i;
            }
        }
        else if(C4_UNLIKELY(curr == '\r'))
        {
            csubstr sub = s.range(pos, i);
            this->Writer::_do_write(sub);  // write everything up to (excluding) this char
            this->Writer::_do_write("\\r"); // write the escaped char
            pos = i+1;
        }
    }
    // write missing characters at the end of the string
    if(pos < s.len)
    {
        csubstr sub = s.sub(pos);
        this->Writer::_do_write(sub);
    }
    this->Writer::_do_write('"');
}

template<class Writer>
void Emitter<Writer>::_write_scalar_plain(csubstr s, size_t ilevel)
{
    size_t pos = 0; // tracks the last character that was already written
    for(size_t i = 0; i < s.len; ++i)
    {
        const char curr = s.str[i];
        if(curr == '\n')
        {
            csubstr sub = s.range(pos, i+1);
            this->Writer::_do_write(sub);  // write everything up to (including) this newline
            this->Writer::_do_write('\n'); // write the newline again
            if(i + 1 < s.len)
                _rymlindent_nextline()     // indent the next line
            pos = i+1;
        }
    }
    // write missing characters at the end of the string
    if(pos < s.len)
    {
        csubstr sub = s.sub(pos);
        this->Writer::_do_write(sub);
    }
}

#undef _rymlindent_nextline

template<class Writer>
void Emitter<Writer>::_write_scalar(csubstr s, bool was_quoted)
{
    // this block of code needed to be moved to before the needs_quotes
    // assignment to work around a g++ optimizer bug where (s.str != nullptr)
    // was evaluated as true even if s.str was actually a nullptr (!!!)
    if(s.len == size_t(0))
    {
        if(was_quoted || s.str != nullptr)
            this->Writer::_do_write("''");
        return;
    }

    const bool needs_quotes = (
        was_quoted
        ||
        (
            ( ! s.is_number())
            &&
            (
                // has leading whitespace
                // looks like reference or anchor
                // would be treated as a directive
                // see https://www.yaml.info/learn/quote.html#noplain
                s.begins_with_any(" \n\t\r*&%@`")
                ||
                s.begins_with("<<")
                ||
                // has trailing whitespace
                s.ends_with_any(" \n\t\r")
                ||
                // has special chars
                (s.first_of("#:-?,\n{}[]'\"") != npos)
            )
        )
    );

    if( ! needs_quotes)
    {
        this->Writer::_do_write(s);
    }
    else
    {
        const bool has_dquotes = s.first_of( '"') != npos;
        const bool has_squotes = s.first_of('\'') != npos;
        if(!has_squotes && has_dquotes)
        {
            this->Writer::_do_write('\'');
            this->Writer::_do_write(s);
            this->Writer::_do_write('\'');
        }
        else if(has_squotes && !has_dquotes)
        {
            RYML_ASSERT(s.count('\n') == 0);
            this->Writer::_do_write('"');
            this->Writer::_do_write(s);
            this->Writer::_do_write('"');
        }
        else
        {
            _write_scalar_squo(s, /*FIXME FIXME FIXME*/0);
        }
    }
}
template<class Writer>
void Emitter<Writer>::_write_scalar_json(csubstr s, bool as_key, bool use_quotes)
{
    if((!use_quotes)
       // json keys require quotes
       && (!as_key)
       && (
           // do not quote special cases
           (s == "true" || s == "false" || s == "null")
           || (
               // do not quote numbers
               (s.is_number()
                && (
                    // quote integral numbers if they have a leading 0
                    // https://github.com/biojppm/rapidyaml/issues/291
                    (!(s.len > 1 && s.begins_with('0')))
                    // do not quote reals with leading 0
                    // https://github.com/biojppm/rapidyaml/issues/313
                    || (s.find('.') != csubstr::npos) ))
               )
           )
        )
    {
        this->Writer::_do_write(s);
    }
    else
    {
        size_t pos = 0;
        this->Writer::_do_write('"');
        for(size_t i = 0; i < s.len; ++i)
        {
            switch(s.str[i])
            {
            case '"':
              this->Writer ::_do_write(s.range(pos, i));
              this->Writer ::_do_write("\\\"");
              pos = i + 1;
              break;
            case '\n':
              this->Writer ::_do_write(s.range(pos, i));
              this->Writer ::_do_write("\\n");
              pos = i + 1;
              break;
            case '\t':
              this->Writer ::_do_write(s.range(pos, i));
              this->Writer ::_do_write("\\t");
              pos = i + 1;
              break;
            case '\\':
              this->Writer ::_do_write(s.range(pos, i));
              this->Writer ::_do_write("\\\\");
              pos = i + 1;
              break;
            case '\r':
              this->Writer ::_do_write(s.range(pos, i));
              this->Writer ::_do_write("\\r");
              pos = i + 1;
              break;
            case '\b':
              this->Writer ::_do_write(s.range(pos, i));
              this->Writer ::_do_write("\\b");
              pos = i + 1;
              break;
            case '\f':
              this->Writer ::_do_write(s.range(pos, i));
              this->Writer ::_do_write("\\f");
              pos = i + 1;
              break;
            }
        }
        if(pos < s.len)
        {
            csubstr sub = s.sub(pos);
            this->Writer::_do_write(sub);
        }
        this->Writer::_do_write('"');
    }
}

} // namespace yml
} // namespace c4

#endif /* _C4_YML_EMIT_DEF_HPP_ */
