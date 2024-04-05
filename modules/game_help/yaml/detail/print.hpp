#ifndef C4_YML_DETAIL_PRINT_HPP_
#define C4_YML_DETAIL_PRINT_HPP_

#include "../tree.hpp"
#include "../node.hpp"


namespace c4 {
namespace yml {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

inline size_t print_node(Tree const& p, size_t node, int level, size_t count, bool print_children)
{
    printf("[%zd]%*s[%zd] %p", count, (2*level), "", node, (void const*)p.get(node));
    if(p.is_root(node))
    {
        printf(" [ROOT]");
    }
    printf(" %s:", p.type_str(node));
    if(p.has_key(node))
    {
        if(p.has_key_anchor(node))
        {
            csubstr ka = p.key_anchor(node);
            printf(" &%.*s", (int)ka.len, ka.str);
        }
        if(p.has_key_tag(node))
        {
            csubstr kt = p.key_tag(node);
            csubstr k  = p.key(node);
            printf(" %.*s '%.*s'", (int)kt.len, kt.str, (int)k.len, k.str);
        }
        else
        {
            csubstr k  = p.key(node);
            printf(" '%.*s'", (int)k.len, k.str);
        }
    }
    else
    {
        RYML_ASSERT( ! p.has_key_tag(node));
    }
    if(p.has_val(node))
    {
        if(p.has_val_tag(node))
        {
            csubstr vt = p.val_tag(node);
            csubstr v  = p.val(node);
            printf(" %.*s '%.*s'", (int)vt.len, vt.str, (int)v.len, v.str);
        }
        else
        {
            csubstr v  = p.val(node);
            printf(" '%.*s'", (int)v.len, v.str);
        }
    }
    else
    {
        if(p.has_val_tag(node))
        {
            csubstr vt = p.val_tag(node);
            printf(" %.*s", (int)vt.len, vt.str);
        }
    }
    if(p.has_val_anchor(node))
    {
        auto &a = p.val_anchor(node);
        printf(" valanchor='&%.*s'", (int)a.len, a.str);
    }
    printf(" (%zd sibs)", p.num_siblings(node));

    ++count;

    if(p.is_container(node))
    {
        printf(" %zd children:\n", p.num_children(node));
        if(print_children)
        {
            for(size_t i = p.first_child(node); i != NONE; i = p.next_sibling(i))
            {
                count = print_node(p, i, level+1, count, print_children);
            }
        }
    }
    else
    {
        printf("\n");
    }

    return count;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

inline void print_node(ConstNodeRef const& p, int level=0)
{
    print_node(*p.tree(), p.id(), level, 0, true);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

inline size_t print_tree(Tree const& p, size_t node=NONE)
{
    printf("--------------------------------------\n");
    size_t ret = 0;
    if(!p.empty())
    {
        if(node == NONE)
            node = p.root_id();
        ret = print_node(p, node, 0, 0, true);
    }
    printf("#nodes=%zd vs #printed=%zd\n", p.size(), ret);
    printf("--------------------------------------\n");
    return ret;
}

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} /* namespace yml */
} /* namespace c4 */


#endif /* C4_YML_DETAIL_PRINT_HPP_ */
