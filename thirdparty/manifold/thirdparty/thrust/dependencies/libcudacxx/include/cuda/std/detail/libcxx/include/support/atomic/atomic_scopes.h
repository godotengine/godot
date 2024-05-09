#ifndef __LIBCUDACXX_ATOMIC_SCOPES_H
#define __LIBCUDACXX_ATOMIC_SCOPES_H

#ifndef __ATOMIC_BLOCK
#define __ATOMIC_SYSTEM 0 // 0 indicates default
#define __ATOMIC_DEVICE 1
#define __ATOMIC_BLOCK 2
#define __ATOMIC_THREAD 10
#endif //__ATOMIC_BLOCK

enum thread_scope {
    thread_scope_system = __ATOMIC_SYSTEM,
    thread_scope_device = __ATOMIC_DEVICE,
    thread_scope_block = __ATOMIC_BLOCK,
    thread_scope_thread = __ATOMIC_THREAD
};

#define _LIBCUDACXX_ATOMIC_SCOPE_TYPE ::cuda::thread_scope
#define _LIBCUDACXX_ATOMIC_SCOPE_DEFAULT ::cuda::thread_scope::system

struct __thread_scope_thread_tag { };
struct __thread_scope_block_tag { };
struct __thread_scope_device_tag { };
struct __thread_scope_system_tag { };

template<int _Scope>  struct __scope_enum_to_tag { };
/* This would be the implementation once an actual thread-scope backend exists.
template<> struct __scope_enum_to_tag<(int)thread_scope_thread> {
    using type = __thread_scope_thread_tag; };
Until then: */
template<> struct __scope_enum_to_tag<(int)thread_scope_thread> {
    using type = __thread_scope_block_tag; };
template<> struct __scope_enum_to_tag<(int)thread_scope_block> {
    using type = __thread_scope_block_tag; };
template<> struct __scope_enum_to_tag<(int)thread_scope_device> {
    using type = __thread_scope_device_tag; };
template<> struct __scope_enum_to_tag<(int)thread_scope_system> {
    using type = __thread_scope_system_tag; };

template <int _Scope>
_LIBCUDACXX_INLINE_VISIBILITY auto constexpr __scope_tag() ->
        typename __scope_enum_to_tag<_Scope>::type {
    return typename __scope_enum_to_tag<_Scope>::type();
}

#endif // __LIBCUDACXX_ATOMIC_SCOPES_H
