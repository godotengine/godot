#ifndef _C4_YML_DETAIL_STACK_HPP_
#define _C4_YML_DETAIL_STACK_HPP_

#ifndef _C4_YML_COMMON_HPP_
#include "../common.hpp"
#endif

#ifdef RYML_DBG
#   include <type_traits>
#endif

#include <string.h>

namespace c4 {
namespace yml {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

namespace detail {

/** A lightweight contiguous stack with SSO. This avoids a dependency on std. */
template<class T, size_t N=16>
class stack
{
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    static_assert(std::is_trivially_destructible<T>::value, "T must be trivially destructible");

    enum : size_t { sso_size = N };

public:

    T         m_buf[N];
    T *       m_stack;
    size_t    m_size;
    size_t    m_capacity;
    Callbacks m_callbacks;

public:

    constexpr static bool is_contiguous() { return true; }

    stack(Callbacks const& cb)
        : m_buf()
        , m_stack(m_buf)
        , m_size(0)
        , m_capacity(N)
        , m_callbacks(cb) {}
    stack() : stack(get_callbacks()) {}
    ~stack()
    {
        _free();
    }

    stack(stack const& that) noexcept : stack(that.m_callbacks)
    {
        resize(that.m_size);
        _cp(&that);
    }

    stack(stack &&that) noexcept : stack(that.m_callbacks)
    {
        _mv(&that);
    }

    stack& operator= (stack const& that) noexcept
    {
        _cb(that.m_callbacks);
        resize(that.m_size);
        _cp(&that);
        return *this;
    }

    stack& operator= (stack &&that) noexcept
    {
        _cb(that.m_callbacks);
        _mv(&that);
        return *this;
    }

public:

    size_t size() const { return m_size; }
    size_t empty() const { return m_size == 0; }
    size_t capacity() const { return m_capacity; }

    void clear()
    {
        m_size = 0;
    }

    void resize(size_t sz)
    {
        reserve(sz);
        m_size = sz;
    }

    void reserve(size_t sz);

    void push(T const& C4_RESTRICT n)
    {
        RYML_ASSERT((const char*)&n + sizeof(T) < (const char*)m_stack || &n > m_stack + m_capacity);
        if(m_size == m_capacity)
        {
            size_t cap = m_capacity == 0 ? N : 2 * m_capacity;
            reserve(cap);
        }
        m_stack[m_size] = n;
        ++m_size;
    }

    void push_top()
    {
        RYML_ASSERT(m_size > 0);
        if(m_size == m_capacity)
        {
            size_t cap = m_capacity == 0 ? N : 2 * m_capacity;
            reserve(cap);
        }
        m_stack[m_size] = m_stack[m_size - 1];
        ++m_size;
    }

    T const& C4_RESTRICT pop()
    {
        RYML_ASSERT(m_size > 0);
        --m_size;
        return m_stack[m_size];
    }

    C4_ALWAYS_INLINE T const& C4_RESTRICT top() const { RYML_ASSERT(m_size > 0); return m_stack[m_size - 1]; }
    C4_ALWAYS_INLINE T      & C4_RESTRICT top()       { RYML_ASSERT(m_size > 0); return m_stack[m_size - 1]; }

    C4_ALWAYS_INLINE T const& C4_RESTRICT bottom() const { RYML_ASSERT(m_size > 0); return m_stack[0]; }
    C4_ALWAYS_INLINE T      & C4_RESTRICT bottom()       { RYML_ASSERT(m_size > 0); return m_stack[0]; }

    C4_ALWAYS_INLINE T const& C4_RESTRICT top(size_t i) const { RYML_ASSERT(i < m_size); return m_stack[m_size - 1 - i]; }
    C4_ALWAYS_INLINE T      & C4_RESTRICT top(size_t i)       { RYML_ASSERT(i < m_size); return m_stack[m_size - 1 - i]; }

    C4_ALWAYS_INLINE T const& C4_RESTRICT bottom(size_t i) const { RYML_ASSERT(i < m_size); return m_stack[i]; }
    C4_ALWAYS_INLINE T      & C4_RESTRICT bottom(size_t i)       { RYML_ASSERT(i < m_size); return m_stack[i]; }

    C4_ALWAYS_INLINE T const& C4_RESTRICT operator[](size_t i) const { RYML_ASSERT(i < m_size); return m_stack[i]; }
    C4_ALWAYS_INLINE T      & C4_RESTRICT operator[](size_t i)       { RYML_ASSERT(i < m_size); return m_stack[i]; }

public:

    using       iterator = T       *;
    using const_iterator = T const *;

    iterator begin() { return m_stack; }
    iterator end  () { return m_stack + m_size; }

    const_iterator begin() const { return (const_iterator)m_stack; }
    const_iterator end  () const { return (const_iterator)m_stack + m_size; }

public:
    void _free();
    void _cp(stack const* C4_RESTRICT that);
    void _mv(stack * that);
    void _cb(Callbacks const& cb);
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

template<class T, size_t N>
void stack<T, N>::reserve(size_t sz)
{
    if(sz <= m_size)
        return;
    if(sz <= N)
    {
        m_stack = m_buf;
        m_capacity = N;
        return;
    }
    T *buf = (T*) m_callbacks.m_allocate(sz * sizeof(T), m_stack, m_callbacks.m_user_data);
    memcpy(buf, m_stack, m_size * sizeof(T));
    if(m_stack != m_buf)
    {
        m_callbacks.m_free(m_stack, m_capacity * sizeof(T), m_callbacks.m_user_data);
    }
    m_stack = buf;
    m_capacity = sz;
}


//-----------------------------------------------------------------------------

template<class T, size_t N>
void stack<T, N>::_free()
{
    RYML_ASSERT(m_stack != nullptr); // this structure cannot be memset() to zero
    if(m_stack != m_buf)
    {
        m_callbacks.m_free(m_stack, m_capacity * sizeof(T), m_callbacks.m_user_data);
        m_stack = m_buf;
        m_size = N;
        m_capacity = N;
    }
    else
    {
        RYML_ASSERT(m_capacity == N);
    }
}


//-----------------------------------------------------------------------------

template<class T, size_t N>
void stack<T, N>::_cp(stack const* C4_RESTRICT that)
{
    if(that->m_stack != that->m_buf)
    {
        RYML_ASSERT(that->m_capacity > N);
        RYML_ASSERT(that->m_size <= that->m_capacity);
    }
    else
    {
        RYML_ASSERT(that->m_capacity <= N);
        RYML_ASSERT(that->m_size <= that->m_capacity);
    }
    memcpy(m_stack, that->m_stack, that->m_size * sizeof(T));
    m_size = that->m_size;
    m_capacity = that->m_size < N ? N : that->m_size;
    m_callbacks = that->m_callbacks;
}


//-----------------------------------------------------------------------------

template<class T, size_t N>
void stack<T, N>::_mv(stack * that)
{
    if(that->m_stack != that->m_buf)
    {
        RYML_ASSERT(that->m_capacity > N);
        RYML_ASSERT(that->m_size <= that->m_capacity);
        m_stack = that->m_stack;
    }
    else
    {
        RYML_ASSERT(that->m_capacity <= N);
        RYML_ASSERT(that->m_size <= that->m_capacity);
        memcpy(m_buf, that->m_buf, that->m_size * sizeof(T));
        m_stack = m_buf;
    }
    m_size = that->m_size;
    m_capacity = that->m_capacity;
    m_callbacks = that->m_callbacks;
    // make sure no deallocation happens on destruction
    RYML_ASSERT(that->m_stack != m_buf);
    that->m_stack = that->m_buf;
    that->m_capacity = N;
    that->m_size = 0;
}


//-----------------------------------------------------------------------------

template<class T, size_t N>
void stack<T, N>::_cb(Callbacks const& cb)
{
    if(cb != m_callbacks)
    {
        _free();
        m_callbacks = cb;
    }
}

} // namespace detail

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace yml
} // namespace c4

#endif /* _C4_YML_DETAIL_STACK_HPP_ */
