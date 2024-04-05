#ifndef _C4_YML_NODE_HPP_
#define _C4_YML_NODE_HPP_

/** @file node.hpp
 * @see NodeRef */

#include <cstddef>

#include "tree.hpp"
#include "./c4/base64.hpp"

#ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wtype-limits"
#   pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wtype-limits"
#   pragma GCC diagnostic ignored "-Wold-style-cast"
#elif defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable: 4251/*needs to have dll-interface to be used by clients of struct*/)
#   pragma warning(disable: 4296/*expression is always 'boolean_value'*/)
#endif

namespace c4 {
namespace yml {

template<class K> struct Key { K & k; };
template<> struct Key<fmt::const_base64_wrapper> { fmt::const_base64_wrapper wrapper; };
template<> struct Key<fmt::base64_wrapper> { fmt::base64_wrapper wrapper; };

template<class K> C4_ALWAYS_INLINE Key<K> key(K & k) { return Key<K>{k}; }
C4_ALWAYS_INLINE Key<fmt::const_base64_wrapper> key(fmt::const_base64_wrapper w) { return {w}; }
C4_ALWAYS_INLINE Key<fmt::base64_wrapper> key(fmt::base64_wrapper w) { return {w}; }

template<class T> void write(NodeRef *n, T const& v);

template<class T>
typename std::enable_if< ! std::is_floating_point<T>::value, bool>::type
read(NodeRef const& n, T *v);

template<class T>
typename std::enable_if<   std::is_floating_point<T>::value, bool>::type
read(NodeRef const& n, T *v);


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

// forward decls
class NodeRef;
class ConstNodeRef;

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

namespace detail {

template<class NodeRefType>
struct child_iterator
{
    using value_type = NodeRefType;
    using tree_type = typename NodeRefType::tree_type;

    tree_type * C4_RESTRICT m_tree;
    size_t m_child_id;

    child_iterator(tree_type * t, size_t id) : m_tree(t), m_child_id(id) {}

    child_iterator& operator++ () { RYML_ASSERT(m_child_id != NONE); m_child_id = m_tree->next_sibling(m_child_id); return *this; }
    child_iterator& operator-- () { RYML_ASSERT(m_child_id != NONE); m_child_id = m_tree->prev_sibling(m_child_id); return *this; }

    NodeRefType operator*  () const { return NodeRefType(m_tree, m_child_id); }
    NodeRefType operator-> () const { return NodeRefType(m_tree, m_child_id); }

    bool operator!= (child_iterator that) const { RYML_ASSERT(m_tree == that.m_tree); return m_child_id != that.m_child_id; }
    bool operator== (child_iterator that) const { RYML_ASSERT(m_tree == that.m_tree); return m_child_id == that.m_child_id; }
};

template<class NodeRefType>
struct children_view_
{
    using n_iterator = child_iterator<NodeRefType>;

    n_iterator b, e;

    inline children_view_(n_iterator const& C4_RESTRICT b_,
                          n_iterator const& C4_RESTRICT e_) : b(b_), e(e_) {}

    inline n_iterator begin() const { return b; }
    inline n_iterator end  () const { return e; }
};

template<class NodeRefType, class Visitor>
bool _visit(NodeRefType &node, Visitor fn, size_t indentation_level, bool skip_root=false)
{
    size_t increment = 0;
    if( ! (node.is_root() && skip_root))
    {
        if(fn(node, indentation_level))
            return true;
        ++increment;
    }
    if(node.has_children())
    {
        for(auto ch : node.children())
        {
            if(_visit(ch, fn, indentation_level + increment, false)) // no need to forward skip_root as it won't be root
            {
                return true;
            }
        }
    }
    return false;
}

template<class NodeRefType, class Visitor>
bool _visit_stacked(NodeRefType &node, Visitor fn, size_t indentation_level, bool skip_root=false)
{
    size_t increment = 0;
    if( ! (node.is_root() && skip_root))
    {
        if(fn(node, indentation_level))
        {
            return true;
        }
        ++increment;
    }
    if(node.has_children())
    {
        fn.push(node, indentation_level);
        for(auto ch : node.children())
        {
            if(_visit_stacked(ch, fn, indentation_level + increment, false)) // no need to forward skip_root as it won't be root
            {
                fn.pop(node, indentation_level);
                return true;
            }
        }
        fn.pop(node, indentation_level);
    }
    return false;
}


//-----------------------------------------------------------------------------

/** a CRTP base for read-only node methods */
template<class Impl, class ConstImpl>
struct RoNodeMethods
{
    C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wcast-align")
    // helper CRTP macros, undefined at the end
    #define tree_ ((ConstImpl const* C4_RESTRICT)this)->m_tree
    #define id_ ((ConstImpl const* C4_RESTRICT)this)->m_id
    #define tree__ ((Impl const* C4_RESTRICT)this)->m_tree
    #define id__ ((Impl const* C4_RESTRICT)this)->m_id
    // require valid
    #define _C4RV()                                       \
        RYML_ASSERT(tree_ != nullptr);                    \
        _RYML_CB_ASSERT(tree_->m_callbacks, id_ != NONE)
    #define _C4_IF_MUTABLE(ty) typename std::enable_if<!std::is_same<U, ConstImpl>::value, ty>::type

public:

    /** @name node property getters */
    /** @{ */

    /** returns the data or null when the id is NONE */
    C4_ALWAYS_INLINE C4_PURE NodeData const* get() const noexcept { RYML_ASSERT(tree_ != nullptr); return tree_->get(id_); }
    /** returns the data or null when the id is NONE */
    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto get() noexcept -> _C4_IF_MUTABLE(NodeData*) { RYML_ASSERT(tree_ != nullptr); return tree__->get(id__); }

    C4_ALWAYS_INLINE C4_PURE NodeType    type() const noexcept { _C4RV(); return tree_->type(id_); }
    C4_ALWAYS_INLINE C4_PURE const char* type_str() const noexcept { return tree_->type_str(id_); }

    C4_ALWAYS_INLINE C4_PURE csubstr key()        const noexcept { _C4RV(); return tree_->key(id_); }
    C4_ALWAYS_INLINE C4_PURE csubstr key_tag()    const noexcept { _C4RV(); return tree_->key_tag(id_); }
    C4_ALWAYS_INLINE C4_PURE csubstr key_ref()    const noexcept { _C4RV(); return tree_->key_ref(id_); }
    C4_ALWAYS_INLINE C4_PURE csubstr key_anchor() const noexcept { _C4RV(); return tree_->key_anchor(id_); }

    C4_ALWAYS_INLINE C4_PURE csubstr val()        const noexcept { _C4RV(); return tree_->val(id_); }
    C4_ALWAYS_INLINE C4_PURE csubstr val_tag()    const noexcept { _C4RV(); return tree_->val_tag(id_); }
    C4_ALWAYS_INLINE C4_PURE csubstr val_ref()    const noexcept { _C4RV(); return tree_->val_ref(id_); }
    C4_ALWAYS_INLINE C4_PURE csubstr val_anchor() const noexcept { _C4RV(); return tree_->val_anchor(id_); }

    C4_ALWAYS_INLINE C4_PURE NodeScalar const& keysc() const noexcept { _C4RV(); return tree_->keysc(id_); }
    C4_ALWAYS_INLINE C4_PURE NodeScalar const& valsc() const noexcept { _C4RV(); return tree_->valsc(id_); }

    C4_ALWAYS_INLINE C4_PURE bool key_is_null() const noexcept { _C4RV(); return tree_->key_is_null(id_); }
    C4_ALWAYS_INLINE C4_PURE bool val_is_null() const noexcept { _C4RV(); return tree_->val_is_null(id_); }

    /** @} */

public:

    /** @name node property predicates */
    /** @{ */

    C4_ALWAYS_INLINE C4_PURE bool empty()            const noexcept { _C4RV(); return tree_->empty(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_stream()        const noexcept { _C4RV(); return tree_->is_stream(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_doc()           const noexcept { _C4RV(); return tree_->is_doc(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_container()     const noexcept { _C4RV(); return tree_->is_container(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_map()           const noexcept { _C4RV(); return tree_->is_map(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_seq()           const noexcept { _C4RV(); return tree_->is_seq(id_); }
    C4_ALWAYS_INLINE C4_PURE bool has_val()          const noexcept { _C4RV(); return tree_->has_val(id_); }
    C4_ALWAYS_INLINE C4_PURE bool has_key()          const noexcept { _C4RV(); return tree_->has_key(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_val()           const noexcept { _C4RV(); return tree_->is_val(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_keyval()        const noexcept { _C4RV(); return tree_->is_keyval(id_); }
    C4_ALWAYS_INLINE C4_PURE bool has_key_tag()      const noexcept { _C4RV(); return tree_->has_key_tag(id_); }
    C4_ALWAYS_INLINE C4_PURE bool has_val_tag()      const noexcept { _C4RV(); return tree_->has_val_tag(id_); }
    C4_ALWAYS_INLINE C4_PURE bool has_key_anchor()   const noexcept { _C4RV(); return tree_->has_key_anchor(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_key_anchor()    const noexcept { _C4RV(); return tree_->is_key_anchor(id_); }
    C4_ALWAYS_INLINE C4_PURE bool has_val_anchor()   const noexcept { _C4RV(); return tree_->has_val_anchor(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_val_anchor()    const noexcept { _C4RV(); return tree_->is_val_anchor(id_); }
    C4_ALWAYS_INLINE C4_PURE bool has_anchor()       const noexcept { _C4RV(); return tree_->has_anchor(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_anchor()        const noexcept { _C4RV(); return tree_->is_anchor(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_key_ref()       const noexcept { _C4RV(); return tree_->is_key_ref(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_val_ref()       const noexcept { _C4RV(); return tree_->is_val_ref(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_ref()           const noexcept { _C4RV(); return tree_->is_ref(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_anchor_or_ref() const noexcept { _C4RV(); return tree_->is_anchor_or_ref(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_key_quoted()    const noexcept { _C4RV(); return tree_->is_key_quoted(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_val_quoted()    const noexcept { _C4RV(); return tree_->is_val_quoted(id_); }
    C4_ALWAYS_INLINE C4_PURE bool is_quoted()        const noexcept { _C4RV(); return tree_->is_quoted(id_); }
    C4_ALWAYS_INLINE C4_PURE bool parent_is_seq()    const noexcept { _C4RV(); return tree_->parent_is_seq(id_); }
    C4_ALWAYS_INLINE C4_PURE bool parent_is_map()    const noexcept { _C4RV(); return tree_->parent_is_map(id_); }

    /** @} */

public:

    /** @name hierarchy predicates */
    /** @{ */

    C4_ALWAYS_INLINE C4_PURE bool is_root()    const noexcept { _C4RV(); return tree_->is_root(id_); }
    C4_ALWAYS_INLINE C4_PURE bool has_parent() const noexcept { _C4RV(); return tree_->has_parent(id_); }

    C4_ALWAYS_INLINE C4_PURE bool has_child(ConstImpl const& ch) const noexcept { _C4RV(); return tree_->has_child(id_, ch.m_id); }
    C4_ALWAYS_INLINE C4_PURE bool has_child(csubstr name) const noexcept { _C4RV(); return tree_->has_child(id_, name); }
    C4_ALWAYS_INLINE C4_PURE bool has_children() const noexcept { _C4RV(); return tree_->has_children(id_); }

    C4_ALWAYS_INLINE C4_PURE bool has_sibling(ConstImpl const& n) const noexcept { _C4RV(); return tree_->has_sibling(id_, n.m_id); }
    C4_ALWAYS_INLINE C4_PURE bool has_sibling(csubstr name) const noexcept { _C4RV(); return tree_->has_sibling(id_, name); }
    /** counts with this */
    C4_ALWAYS_INLINE C4_PURE bool has_siblings() const noexcept { _C4RV(); return tree_->has_siblings(id_); }
    /** does not count with this */
    C4_ALWAYS_INLINE C4_PURE bool has_other_siblings() const noexcept { _C4RV(); return tree_->has_other_siblings(id_); }

    /** @} */

public:

    /** @name hierarchy getters */
    /** @{ */


    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto doc(size_t num) noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->doc(num)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl doc(size_t num) const noexcept { _C4RV(); return {tree_, tree_->doc(num)}; }


    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto parent() noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->parent(id__)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl parent() const noexcept { _C4RV(); return {tree_, tree_->parent(id_)}; }


    /** O(#num_children) */
    C4_ALWAYS_INLINE C4_PURE size_t child_pos(ConstImpl const& n) const noexcept { _C4RV(); return tree_->child_pos(id_, n.m_id); }
    C4_ALWAYS_INLINE C4_PURE size_t num_children() const noexcept { _C4RV(); return tree_->num_children(id_); }

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto first_child() noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->first_child(id__)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl first_child() const noexcept { _C4RV(); return {tree_, tree_->first_child(id_)}; }

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto last_child() noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->last_child(id__)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl last_child () const noexcept { _C4RV(); return {tree_, tree_->last_child (id_)}; }

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto child(size_t pos) noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->child(id__, pos)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl child(size_t pos) const noexcept { _C4RV(); return {tree_, tree_->child(id_, pos)}; }

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto find_child(csubstr name)  noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->find_child(id__, name)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl find_child(csubstr name) const noexcept { _C4RV(); return {tree_, tree_->find_child(id_, name)}; }


    /** O(#num_siblings) */
    C4_ALWAYS_INLINE C4_PURE size_t num_siblings() const noexcept { _C4RV(); return tree_->num_siblings(id_); }
    C4_ALWAYS_INLINE C4_PURE size_t num_other_siblings() const noexcept { _C4RV(); return tree_->num_other_siblings(id_); }
    C4_ALWAYS_INLINE C4_PURE size_t sibling_pos(ConstImpl const& n) const noexcept { _C4RV(); return tree_->child_pos(tree_->parent(id_), n.m_id); }

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto prev_sibling() noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->prev_sibling(id__)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl prev_sibling() const noexcept { _C4RV(); return {tree_, tree_->prev_sibling(id_)}; }

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto next_sibling() noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->next_sibling(id__)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl next_sibling() const noexcept { _C4RV(); return {tree_, tree_->next_sibling(id_)}; }

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto first_sibling() noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->first_sibling(id__)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl first_sibling() const noexcept { _C4RV(); return {tree_, tree_->first_sibling(id_)}; }

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto last_sibling() noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->last_sibling(id__)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl last_sibling () const noexcept { _C4RV(); return {tree_, tree_->last_sibling(id_)}; }

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto sibling(size_t pos) noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->sibling(id__, pos)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl sibling(size_t pos) const noexcept { _C4RV(); return {tree_, tree_->sibling(id_, pos)}; }

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto find_sibling(csubstr name) noexcept -> _C4_IF_MUTABLE(Impl) { _C4RV(); return {tree__, tree__->find_sibling(id__, name)}; }
    C4_ALWAYS_INLINE C4_PURE ConstImpl find_sibling(csubstr name) const noexcept { _C4RV(); return {tree_, tree_->find_sibling(id_, name)}; }


    /** O(num_children) */
    C4_ALWAYS_INLINE C4_PURE ConstImpl operator[] (csubstr k) const noexcept
    {
        _C4RV();
        size_t ch = tree_->find_child(id_, k);
        _RYML_CB_ASSERT(tree_->m_callbacks, ch != NONE);
        return {tree_, ch};
    }
    /** Find child by key. O(num_children). returns a seed node if no such child is found.  */
    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto operator[] (csubstr k) noexcept -> _C4_IF_MUTABLE(Impl)
    {
        _C4RV();
        size_t ch = tree__->find_child(id__, k);
        return ch != NONE ? Impl(tree__, ch) : NodeRef(tree__, id__, k);
    }

    /** O(num_children) */
    C4_ALWAYS_INLINE C4_PURE ConstImpl operator[] (size_t pos) const noexcept
    {
        _C4RV();
        size_t ch = tree_->child(id_, pos);
        _RYML_CB_ASSERT(tree_->m_callbacks, ch != NONE);
        return {tree_, ch};
    }

    /** Find child by position. O(pos). returns a seed node if no such child is found.  */
    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto operator[] (size_t pos) noexcept -> _C4_IF_MUTABLE(Impl)
    {
        _C4RV();
        size_t ch = tree__->child(id__, pos);
        return ch != NONE ? Impl(tree__, ch) : NodeRef(tree__, id__, pos);
    }

    /** @} */

public:

    /** deserialization */
    /** @{ */

    template<class T>
    ConstImpl const& operator>> (T &v) const
    {
        _C4RV();
        if( ! read((ConstImpl const&)*this, &v))
            _RYML_CB_ERR(tree_->m_callbacks, "could not deserialize value");
        return *((ConstImpl const*)this);
    }

    /** deserialize the node's key to the given variable */
    template<class T>
    ConstImpl const& operator>> (Key<T> v) const
    {
        _C4RV();
        if( ! from_chars(key(), &v.k))
            _RYML_CB_ERR(tree_->m_callbacks, "could not deserialize key");
        return *((ConstImpl const*)this);
    }

    /** deserialize the node's key as base64 */
    ConstImpl const& operator>> (Key<fmt::base64_wrapper> w) const
    {
        deserialize_key(w.wrapper);
        return *((ConstImpl const*)this);
    }

    /** deserialize the node's val as base64 */
    ConstImpl const& operator>> (fmt::base64_wrapper w) const
    {
        deserialize_val(w);
        return *((ConstImpl const*)this);
    }

    /** decode the base64-encoded key and assign the
     * decoded blob to the given buffer/
     * @return the size of base64-decoded blob */
    size_t deserialize_key(fmt::base64_wrapper v) const
    {
        _C4RV();
        return from_chars(key(), &v);
    }
    /** decode the base64-encoded key and assign the
     * decoded blob to the given buffer/
     * @return the size of base64-decoded blob */
    size_t deserialize_val(fmt::base64_wrapper v) const
    {
        _C4RV();
        return from_chars(val(), &v);
    };

    template<class T>
    bool get_if(csubstr name, T *var) const
    {
        auto ch = find_child(name);
        if(!ch.valid())
            return false;
        ch >> *var;
        return true;
    }

    template<class T>
    bool get_if(csubstr name, T *var, T const& fallback) const
    {
        auto ch = find_child(name);
        if(ch.valid())
        {
            ch >> *var;
            return true;
        }
        else
        {
            *var = fallback;
            return false;
        }
    }

    /** @} */

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

    /** @name iteration */
    /** @{ */

    using iterator = detail::child_iterator<Impl>;
    using const_iterator = detail::child_iterator<ConstImpl>;
    using children_view = detail::children_view_<Impl>;
    using const_children_view = detail::children_view_<ConstImpl>;

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto begin() noexcept -> _C4_IF_MUTABLE(iterator) { _C4RV(); return iterator(tree__, tree__->first_child(id__)); }
    C4_ALWAYS_INLINE C4_PURE const_iterator begin() const noexcept { _C4RV(); return const_iterator(tree_, tree_->first_child(id_)); }
    C4_ALWAYS_INLINE C4_PURE const_iterator cbegin() const noexcept { _C4RV(); return const_iterator(tree_, tree_->first_child(id_)); }

    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto end() noexcept -> _C4_IF_MUTABLE(iterator) { _C4RV(); return iterator(tree__, NONE); }
    C4_ALWAYS_INLINE C4_PURE const_iterator end() const noexcept { _C4RV(); return const_iterator(tree_, NONE); }
    C4_ALWAYS_INLINE C4_PURE const_iterator cend() const noexcept { _C4RV(); return const_iterator(tree_, tree_->first_child(id_)); }

    /** get an iterable view over children */
    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto children() noexcept -> _C4_IF_MUTABLE(children_view) { _C4RV(); return children_view(begin(), end()); }
    /** get an iterable view over children */
    C4_ALWAYS_INLINE C4_PURE const_children_view children() const noexcept { _C4RV(); return const_children_view(begin(), end()); }
    /** get an iterable view over children */
    C4_ALWAYS_INLINE C4_PURE const_children_view cchildren() const noexcept { _C4RV(); return const_children_view(begin(), end()); }

    /** get an iterable view over all siblings (including the calling node) */
    template<class U=Impl>
    C4_ALWAYS_INLINE C4_PURE auto siblings() noexcept -> _C4_IF_MUTABLE(children_view)
    {
        _C4RV();
        NodeData const *nd = tree__->get(id__);
        return (nd->m_parent != NONE) ? // does it have a parent?
            children_view(iterator(tree__, tree_->get(nd->m_parent)->m_first_child), iterator(tree__, NONE))
            :
            children_view(end(), end());
    }
    /** get an iterable view over all siblings (including the calling node) */
    C4_ALWAYS_INLINE C4_PURE const_children_view siblings() const noexcept
    {
        _C4RV();
        NodeData const *nd = tree_->get(id_);
        return (nd->m_parent != NONE) ? // does it have a parent?
            const_children_view(const_iterator(tree_, tree_->get(nd->m_parent)->m_first_child), const_iterator(tree_, NONE))
            :
            const_children_view(end(), end());
    }
    /** get an iterable view over all siblings (including the calling node) */
    C4_ALWAYS_INLINE C4_PURE const_children_view csiblings() const noexcept { return siblings(); }

    /** visit every child node calling fn(node) */
    template<class Visitor>
    C4_ALWAYS_INLINE bool visit(Visitor fn, size_t indentation_level=0, bool skip_root=true) const noexcept
    {
        return detail::_visit(*(ConstImpl const*)this, fn, indentation_level, skip_root);
    }
    /** visit every child node calling fn(node) */
    template<class Visitor, class U=Impl>
    auto visit(Visitor fn, size_t indentation_level=0, bool skip_root=true) noexcept
        -> _C4_IF_MUTABLE(bool)
    {
        return detail::_visit(*(Impl*)this, fn, indentation_level, skip_root);
    }

    /** visit every child node calling fn(node, level) */
    template<class Visitor>
    C4_ALWAYS_INLINE bool visit_stacked(Visitor fn, size_t indentation_level=0, bool skip_root=true) const noexcept
    {
        return detail::_visit_stacked(*(ConstImpl const*)this, fn, indentation_level, skip_root);
    }
    /** visit every child node calling fn(node, level) */
    template<class Visitor, class U=Impl>
    auto visit_stacked(Visitor fn, size_t indentation_level=0, bool skip_root=true) noexcept
        -> _C4_IF_MUTABLE(bool)
    {
        return detail::_visit_stacked(*(Impl*)this, fn, indentation_level, skip_root);
    }

    /** @} */

    #if defined(__clang__)
    #   pragma clang diagnostic pop
    #elif defined(__GNUC__)
    #   pragma GCC diagnostic pop
    #endif

    #undef _C4_IF_MUTABLE
    #undef _C4RV
    #undef tree_
    #undef tree__
    #undef id_
    #undef id__

    C4_SUPPRESS_WARNING_GCC_CLANG_POP
};

} // namespace detail


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
class RYML_EXPORT ConstNodeRef : public detail::RoNodeMethods<ConstNodeRef, ConstNodeRef>
{
public:

    using tree_type = Tree const;

public:

    Tree const* C4_RESTRICT m_tree;
    size_t m_id;

    friend NodeRef;
    friend struct detail::RoNodeMethods<ConstNodeRef, ConstNodeRef>;

public:

    /** @name construction */
    /** @{ */

    ConstNodeRef() : m_tree(nullptr), m_id(NONE) {}
    ConstNodeRef(Tree const &t) : m_tree(&t), m_id(t .root_id()) {}
    ConstNodeRef(Tree const *t) : m_tree(t ), m_id(t->root_id()) {}
    ConstNodeRef(Tree const *t, size_t id) : m_tree(t), m_id(id) {}
    ConstNodeRef(std::nullptr_t) : m_tree(nullptr), m_id(NONE) {}

    ConstNodeRef(ConstNodeRef const&) = default;
    ConstNodeRef(ConstNodeRef     &&) = default;

    ConstNodeRef(NodeRef const&);
    ConstNodeRef(NodeRef     &&);

    /** @} */

public:

    /** @name assignment */
    /** @{ */

    ConstNodeRef& operator= (std::nullptr_t) { m_tree = nullptr; m_id = NONE; return *this; }

    ConstNodeRef& operator= (ConstNodeRef const&) = default;
    ConstNodeRef& operator= (ConstNodeRef     &&) = default;

    ConstNodeRef& operator= (NodeRef const&);
    ConstNodeRef& operator= (NodeRef     &&);


    /** @} */

public:

    /** @name state queries */
    /** @{ */

    C4_ALWAYS_INLINE C4_PURE bool valid() const noexcept { return m_tree != nullptr && m_id != NONE; }

    /** @} */

public:

    /** @name member getters */
    /** @{ */

    C4_ALWAYS_INLINE C4_PURE Tree const* tree() const noexcept { return m_tree; }
    C4_ALWAYS_INLINE C4_PURE size_t id() const noexcept { return m_id; }

    /** @} */

public:

    /** @name comparisons */
    /** @{ */

    C4_ALWAYS_INLINE C4_PURE bool operator== (ConstNodeRef const& that) const noexcept { RYML_ASSERT(that.m_tree == m_tree); return m_id == that.m_id; }
    C4_ALWAYS_INLINE C4_PURE bool operator!= (ConstNodeRef const& that) const noexcept { RYML_ASSERT(that.m_tree == m_tree); return ! this->operator==(that); }

    C4_ALWAYS_INLINE C4_PURE bool operator== (std::nullptr_t) const noexcept { return m_tree == nullptr || m_id == NONE; }
    C4_ALWAYS_INLINE C4_PURE bool operator!= (std::nullptr_t) const noexcept { return ! this->operator== (nullptr); }

    C4_ALWAYS_INLINE C4_PURE bool operator== (csubstr val) const noexcept { RYML_ASSERT(has_val()); return m_tree->val(m_id) == val; }
    C4_ALWAYS_INLINE C4_PURE bool operator!= (csubstr val) const noexcept { RYML_ASSERT(has_val()); return m_tree->val(m_id) != val; }

    /** @} */

};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** a reference to a node in an existing yaml tree, offering a more
 * convenient API than the index-based API used in the tree. */
class RYML_EXPORT NodeRef : public detail::RoNodeMethods<NodeRef, ConstNodeRef>
{
public:

    using tree_type = Tree;
    using base_type = detail::RoNodeMethods<NodeRef, ConstNodeRef>;

private:

    Tree *C4_RESTRICT m_tree;
    size_t m_id;

    /** This member is used to enable lazy operator[] writing. When a child
     * with a key or index is not found, m_id is set to the id of the parent
     * and the asked-for key or index are stored in this member until a write
     * does happen. Then it is given as key or index for creating the child.
     * When a key is used, the csubstr stores it (so the csubstr's string is
     * non-null and the csubstr's size is different from NONE). When an index is
     * used instead, the csubstr's string is set to null, and only the csubstr's
     * size is set to a value different from NONE. Otherwise, when operator[]
     * does find the child then this member is empty: the string is null and
     * the size is NONE. */
    csubstr m_seed;

    friend ConstNodeRef;
    friend struct detail::RoNodeMethods<NodeRef, ConstNodeRef>;

    // require valid: a helper macro, undefined at the end
    #define _C4RV()                                                         \
        RYML_ASSERT(m_tree != nullptr);                                     \
        _RYML_CB_ASSERT(m_tree->m_callbacks, m_id != NONE && !is_seed())

public:

    /** @name construction */
    /** @{ */

    NodeRef() : m_tree(nullptr), m_id(NONE), m_seed() { _clear_seed(); }
    NodeRef(Tree &t) : m_tree(&t), m_id(t .root_id()), m_seed() { _clear_seed(); }
    NodeRef(Tree *t) : m_tree(t ), m_id(t->root_id()), m_seed() { _clear_seed(); }
    NodeRef(Tree *t, size_t id) : m_tree(t), m_id(id), m_seed() { _clear_seed(); }
    NodeRef(Tree *t, size_t id, size_t seed_pos) : m_tree(t), m_id(id), m_seed() { m_seed.str = nullptr; m_seed.len = seed_pos; }
    NodeRef(Tree *t, size_t id, csubstr  seed_key) : m_tree(t), m_id(id), m_seed(seed_key) {}
    NodeRef(std::nullptr_t) : m_tree(nullptr), m_id(NONE), m_seed() {}

    /** @} */

public:

    /** @name assignment */
    /** @{ */

    NodeRef(NodeRef const&) = default;
    NodeRef(NodeRef     &&) = default;

    NodeRef& operator= (NodeRef const&) = default;
    NodeRef& operator= (NodeRef     &&) = default;

    /** @} */

public:

    /** @name state queries */
    /** @{ */

    inline bool valid() const { return m_tree != nullptr && m_id != NONE; }
    inline bool is_seed() const { return m_seed.str != nullptr || m_seed.len != NONE; }

    inline void _clear_seed() { /*do this manually or an assert is triggered*/ m_seed.str = nullptr; m_seed.len = NONE; }

    /** @} */

public:

    /** @name comparisons */
    /** @{ */

    inline bool operator== (NodeRef const& that) const { _C4RV(); RYML_ASSERT(that.valid() && !that.is_seed()); RYML_ASSERT(that.m_tree == m_tree); return m_id == that.m_id; }
    inline bool operator!= (NodeRef const& that) const { return ! this->operator==(that); }

    inline bool operator== (ConstNodeRef const& that) const { _C4RV(); RYML_ASSERT(that.valid()); RYML_ASSERT(that.m_tree == m_tree); return m_id == that.m_id; }
    inline bool operator!= (ConstNodeRef const& that) const { return ! this->operator==(that); }

    inline bool operator== (std::nullptr_t) const { return m_tree == nullptr || m_id == NONE || is_seed(); }
    inline bool operator!= (std::nullptr_t) const { return m_tree != nullptr && m_id != NONE && !is_seed(); }

    inline bool operator== (csubstr val) const { _C4RV(); RYML_ASSERT(has_val()); return m_tree->val(m_id) == val; }
    inline bool operator!= (csubstr val) const { _C4RV(); RYML_ASSERT(has_val()); return m_tree->val(m_id) != val; }

    //inline operator bool () const { return m_tree == nullptr || m_id == NONE || is_seed(); }

    /** @} */

public:

    /** @name node property getters */
    /** @{ */

    C4_ALWAYS_INLINE C4_PURE Tree * tree() noexcept { return m_tree; }
    C4_ALWAYS_INLINE C4_PURE Tree const* tree() const noexcept { return m_tree; }

    C4_ALWAYS_INLINE C4_PURE size_t id() const noexcept { return m_id; }

    /** @} */

public:

    /** @name node modifiers */
    /** @{ */

    void change_type(NodeType t) { _C4RV(); m_tree->change_type(m_id, t); }

    void set_type(NodeType t) { _C4RV(); m_tree->_set_flags(m_id, t); }
    void set_key(csubstr key) { _C4RV(); m_tree->_set_key(m_id, key); }
    void set_val(csubstr val) { _C4RV(); m_tree->_set_val(m_id, val); }
    void set_key_tag(csubstr key_tag) { _C4RV(); m_tree->set_key_tag(m_id, key_tag); }
    void set_val_tag(csubstr val_tag) { _C4RV(); m_tree->set_val_tag(m_id, val_tag); }
    void set_key_anchor(csubstr key_anchor) { _C4RV(); m_tree->set_key_anchor(m_id, key_anchor); }
    void set_val_anchor(csubstr val_anchor) { _C4RV(); m_tree->set_val_anchor(m_id, val_anchor); }
    void set_key_ref(csubstr key_ref) { _C4RV(); m_tree->set_key_ref(m_id, key_ref); }
    void set_val_ref(csubstr val_ref) { _C4RV(); m_tree->set_val_ref(m_id, val_ref); }

    template<class T>
    size_t set_key_serialized(T const& C4_RESTRICT k)
    {
        _C4RV();
        csubstr s = m_tree->to_arena(k);
        m_tree->_set_key(m_id, s);
        return s.len;
    }
    template<class T>
    size_t set_val_serialized(T const& C4_RESTRICT v)
    {
        _C4RV();
        csubstr s = m_tree->to_arena(v);
        m_tree->_set_val(m_id, s);
        return s.len;
    }
    size_t set_val_serialized(std::nullptr_t)
    {
        _C4RV();
        m_tree->_set_val(m_id, csubstr{});
        return 0;
    }

    /** encode a blob as base64, then assign the result to the node's key
     * @return the size of base64-encoded blob */
    size_t set_key_serialized(fmt::const_base64_wrapper w);
    /** encode a blob as base64, then assign the result to the node's val
     * @return the size of base64-encoded blob */
    size_t set_val_serialized(fmt::const_base64_wrapper w);

public:

    inline void clear()
    {
        if(is_seed())
            return;
        m_tree->remove_children(m_id);
        m_tree->_clear(m_id);
    }

    inline void clear_key()
    {
        if(is_seed())
            return;
        m_tree->_clear_key(m_id);
    }

    inline void clear_val()
    {
        if(is_seed())
            return;
        m_tree->_clear_val(m_id);
    }

    inline void clear_children()
    {
        if(is_seed())
            return;
        m_tree->remove_children(m_id);
    }

    void create() { _apply_seed(); }

    inline void operator= (NodeType_e t)
    {
        _apply_seed();
        m_tree->_add_flags(m_id, t);
    }

    inline void operator|= (NodeType_e t)
    {
        _apply_seed();
        m_tree->_add_flags(m_id, t);
    }

    inline void operator= (NodeInit const& v)
    {
        _apply_seed();
        _apply(v);
    }

    inline void operator= (NodeScalar const& v)
    {
        _apply_seed();
        _apply(v);
    }

    inline void operator= (std::nullptr_t)
    {
        _apply_seed();
        _apply(csubstr{});
    }

    inline void operator= (csubstr v)
    {
        _apply_seed();
        _apply(v);
    }

    template<size_t N>
    inline void operator= (const char (&v)[N])
    {
        _apply_seed();
        csubstr sv;
        sv.assign<N>(v);
        _apply(sv);
    }

    /** @} */

public:

    /** @name serialization */
    /** @{ */

    /** serialize a variable to the arena */
    template<class T>
    inline csubstr to_arena(T const& C4_RESTRICT s)
    {
        _C4RV();
        return m_tree->to_arena(s);
    }

    /** serialize a variable, then assign the result to the node's val */
    inline NodeRef& operator<< (csubstr s)
    {
        // this overload is needed to prevent ambiguity (there's also
        // operator<< for writing a substr to a stream)
        _apply_seed();
        write(this, s);
        RYML_ASSERT(val() == s);
        return *this;
    }

    template<class T>
    inline NodeRef& operator<< (T const& C4_RESTRICT v)
    {
        _apply_seed();
        write(this, v);
        return *this;
    }

    /** serialize a variable, then assign the result to the node's key */
    template<class T>
    inline NodeRef& operator<< (Key<const T> const& C4_RESTRICT v)
    {
        _apply_seed();
        set_key_serialized(v.k);
        return *this;
    }

    /** serialize a variable, then assign the result to the node's key */
    template<class T>
    inline NodeRef& operator<< (Key<T> const& C4_RESTRICT v)
    {
        _apply_seed();
        set_key_serialized(v.k);
        return *this;
    }

    NodeRef& operator<< (Key<fmt::const_base64_wrapper> w)
    {
        set_key_serialized(w.wrapper);
        return *this;
    }

    NodeRef& operator<< (fmt::const_base64_wrapper w)
    {
        set_val_serialized(w);
        return *this;
    }

    /** @} */

private:

    void _apply_seed()
    {
        if(m_seed.str) // we have a seed key: use it to create the new child
        {
            //RYML_ASSERT(i.key.scalar.empty() || m_key == i.key.scalar || m_key.empty());
            m_id = m_tree->append_child(m_id);
            m_tree->_set_key(m_id, m_seed);
            m_seed.str = nullptr;
            m_seed.len = NONE;
        }
        else if(m_seed.len != NONE) // we have a seed index: create a child at that position
        {
            RYML_ASSERT(m_tree->num_children(m_id) == m_seed.len);
            m_id = m_tree->append_child(m_id);
            m_seed.str = nullptr;
            m_seed.len = NONE;
        }
        else
        {
            RYML_ASSERT(valid());
        }
    }

    inline void _apply(csubstr v)
    {
        m_tree->_set_val(m_id, v);
    }

    inline void _apply(NodeScalar const& v)
    {
        m_tree->_set_val(m_id, v);
    }

    inline void _apply(NodeInit const& i)
    {
        m_tree->_set(m_id, i);
    }

public:

    /** @name modification of hierarchy */
    /** @{ */

    inline NodeRef insert_child(NodeRef after)
    {
        _C4RV();
        RYML_ASSERT(after.m_tree == m_tree);
        NodeRef r(m_tree, m_tree->insert_child(m_id, after.m_id));
        return r;
    }

    inline NodeRef insert_child(NodeInit const& i, NodeRef after)
    {
        _C4RV();
        RYML_ASSERT(after.m_tree == m_tree);
        NodeRef r(m_tree, m_tree->insert_child(m_id, after.m_id));
        r._apply(i);
        return r;
    }

    inline NodeRef prepend_child()
    {
        _C4RV();
        NodeRef r(m_tree, m_tree->insert_child(m_id, NONE));
        return r;
    }

    inline NodeRef prepend_child(NodeInit const& i)
    {
        _C4RV();
        NodeRef r(m_tree, m_tree->insert_child(m_id, NONE));
        r._apply(i);
        return r;
    }

    inline NodeRef append_child()
    {
        _C4RV();
        NodeRef r(m_tree, m_tree->append_child(m_id));
        return r;
    }

    inline NodeRef append_child(NodeInit const& i)
    {
        _C4RV();
        NodeRef r(m_tree, m_tree->append_child(m_id));
        r._apply(i);
        return r;
    }

public:

    inline NodeRef insert_sibling(ConstNodeRef const& after)
    {
        _C4RV();
        RYML_ASSERT(after.m_tree == m_tree);
        NodeRef r(m_tree, m_tree->insert_sibling(m_id, after.m_id));
        return r;
    }

    inline NodeRef insert_sibling(NodeInit const& i, ConstNodeRef const& after)
    {
        _C4RV();
        RYML_ASSERT(after.m_tree == m_tree);
        NodeRef r(m_tree, m_tree->insert_sibling(m_id, after.m_id));
        r._apply(i);
        return r;
    }

    inline NodeRef prepend_sibling()
    {
        _C4RV();
        NodeRef r(m_tree, m_tree->prepend_sibling(m_id));
        return r;
    }

    inline NodeRef prepend_sibling(NodeInit const& i)
    {
        _C4RV();
        NodeRef r(m_tree, m_tree->prepend_sibling(m_id));
        r._apply(i);
        return r;
    }

    inline NodeRef append_sibling()
    {
        _C4RV();
        NodeRef r(m_tree, m_tree->append_sibling(m_id));
        return r;
    }

    inline NodeRef append_sibling(NodeInit const& i)
    {
        _C4RV();
        NodeRef r(m_tree, m_tree->append_sibling(m_id));
        r._apply(i);
        return r;
    }

public:

    inline void remove_child(NodeRef & child)
    {
        _C4RV();
        RYML_ASSERT(has_child(child));
        RYML_ASSERT(child.parent().id() == id());
        m_tree->remove(child.id());
        child.clear();
    }

    //! remove the nth child of this node
    inline void remove_child(size_t pos)
    {
        _C4RV();
        RYML_ASSERT(pos >= 0 && pos < num_children());
        size_t child = m_tree->child(m_id, pos);
        RYML_ASSERT(child != NONE);
        m_tree->remove(child);
    }

    //! remove a child by name
    inline void remove_child(csubstr key)
    {
        _C4RV();
        size_t child = m_tree->find_child(m_id, key);
        RYML_ASSERT(child != NONE);
        m_tree->remove(child);
    }

public:

    /** change the node's position within its parent, placing it after
     * @p after. To move to the first position in the parent, simply
     * pass an empty or default-constructed reference like this:
     * `n.move({})`. */
    inline void move(ConstNodeRef const& after)
    {
        _C4RV();
        m_tree->move(m_id, after.m_id);
    }

    /** move the node to a different @p parent (which may belong to a
     * different tree), placing it after @p after. When the
     * destination parent is in a new tree, then this node's tree
     * pointer is reset to the tree of the parent node. */
    inline void move(NodeRef const& parent, ConstNodeRef const& after)
    {
        _C4RV();
        if(parent.m_tree == m_tree)
        {
            m_tree->move(m_id, parent.m_id, after.m_id);
        }
        else
        {
            parent.m_tree->move(m_tree, m_id, parent.m_id, after.m_id);
            m_tree = parent.m_tree;
        }
    }

    /** duplicate the current node somewhere within its parent, and
     * place it after the node @p after. To place into the first
     * position of the parent, simply pass an empty or
     * default-constructed reference like this: `n.move({})`. */
    inline NodeRef duplicate(ConstNodeRef const& after) const
    {
        _C4RV();
        RYML_ASSERT(m_tree == after.m_tree || after.m_id == NONE);
        size_t dup = m_tree->duplicate(m_id, m_tree->parent(m_id), after.m_id);
        NodeRef r(m_tree, dup);
        return r;
    }

    /** duplicate the current node somewhere into a different @p parent
     * (possibly from a different tree), and place it after the node
     * @p after. To place into the first position of the parent,
     * simply pass an empty or default-constructed reference like
     * this: `n.move({})`. */
    inline NodeRef duplicate(NodeRef const& parent, ConstNodeRef const& after) const
    {
        _C4RV();
        RYML_ASSERT(parent.m_tree == after.m_tree || after.m_id == NONE);
        if(parent.m_tree == m_tree)
        {
            size_t dup = m_tree->duplicate(m_id, parent.m_id, after.m_id);
            NodeRef r(m_tree, dup);
            return r;
        }
        else
        {
            size_t dup = parent.m_tree->duplicate(m_tree, m_id, parent.m_id, after.m_id);
            NodeRef r(parent.m_tree, dup);
            return r;
        }
    }

    inline void duplicate_children(NodeRef const& parent, ConstNodeRef const& after) const
    {
        _C4RV();
        RYML_ASSERT(parent.m_tree == after.m_tree);
        if(parent.m_tree == m_tree)
        {
            m_tree->duplicate_children(m_id, parent.m_id, after.m_id);
        }
        else
        {
            parent.m_tree->duplicate_children(m_tree, m_id, parent.m_id, after.m_id);
        }
    }

    /** @} */

#undef _C4RV
};


//-----------------------------------------------------------------------------

inline ConstNodeRef::ConstNodeRef(NodeRef const& that)
    : m_tree(that.m_tree)
    , m_id(!that.is_seed() ? that.id() : NONE)
{
}

inline ConstNodeRef::ConstNodeRef(NodeRef && that)
    : m_tree(that.m_tree)
    , m_id(!that.is_seed() ? that.id() : NONE)
{
}


inline ConstNodeRef& ConstNodeRef::operator= (NodeRef const& that)
{
    m_tree = (that.m_tree);
    m_id = (!that.is_seed() ? that.id() : NONE);
    return *this;
}

inline ConstNodeRef& ConstNodeRef::operator= (NodeRef && that)
{
    m_tree = (that.m_tree);
    m_id = (!that.is_seed() ? that.id() : NONE);
    return *this;
}


//-----------------------------------------------------------------------------

template<class T>
inline void write(NodeRef *n, T const& v)
{
    n->set_val_serialized(v);
}

template<class T>
typename std::enable_if< ! std::is_floating_point<T>::value, bool>::type
inline read(NodeRef const& n, T *v)
{
    return from_chars(n.val(), v);
}
template<class T>
typename std::enable_if< ! std::is_floating_point<T>::value, bool>::type
inline read(ConstNodeRef const& n, T *v)
{
    return from_chars(n.val(), v);
}

template<class T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
inline read(NodeRef const& n, T *v)
{
    return from_chars_float(n.val(), v);
}
template<class T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
inline read(ConstNodeRef const& n, T *v)
{
    return from_chars_float(n.val(), v);
}


} // namespace yml
} // namespace c4



#ifdef __clang__
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#   pragma warning(pop)
#endif

#endif /* _C4_YML_NODE_HPP_ */
