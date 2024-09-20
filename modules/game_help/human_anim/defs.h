#pragma once

/***
* defs.h - definitions/declarations for some commonly standard declaration
*
*
* Purpose:
*       This file defines the following ma keywords:
*   RESTRICT
*   MECANIM_FORCE_INLINE
*   STATIC_INLINE
*   ATTRIBUTE_ALIGN(a)
*   EXPLICIT_TYPENAME
*   EXPLICIT_TEMPLATE
*   DLL_IMPORT
*   DLL_EXPORT
*   DECLARE_C
*
****/

#if defined(__INTEL_COMPILER) || defined(__ICL)
    #include <cstddef>
    #define RESTRICT                __restrict
    #define MECANIM_FORCE_INLINE    __forceinline
    #define ATTRIBUTE_ALIGN(a)      __declspec(align(a))
    #define EXPLICIT_TEMPLATE       template
    #define DLL_IMPORT              __declspec(dllimport)
    #define DLL_EXPORT              __declspec(dllexport)
    #define ALIGN4F                 16
    #define DECLARE_C               __cdecl

#elif defined(_MSC_VER)
    #include <cstddef>
    #define RESTRICT                __restrict
    #define MECANIM_FORCE_INLINE    __forceinline
    #define ATTRIBUTE_ALIGN(a)      __declspec(align(a))
    #define EXPLICIT_TEMPLATE       template
    #define DLL_IMPORT              __declspec(dllimport)
    #define DLL_EXPORT              __declspec(dllexport)
    #define ALIGN4F                 16
    #define DECLARE_C               __cdecl

    #pragma warning( disable : 4996)

#elif defined(__GNUC__)
    #include <cstddef>
    #if ((__GNUC__ >= 3) && (__GNUC_MINOR__ >= 1)) || (__GNUC__ >= 4)
        #ifdef _DEBUG
            #ifndef MECANIM_FORCE_INLINE
                #define MECANIM_FORCE_INLINE        inline
            #endif
            #define STATIC_INLINE       inline
        #else
            #ifndef MECANIM_FORCE_INLINE
                #define MECANIM_FORCE_INLINE        inline __attribute__((always_inline))
            #endif
            #define STATIC_INLINE       inline __attribute__((always_inline))
        #endif
    #else
        #define STATIC_INLINE           extern inline
    #endif

    #define ATTRIBUTE_ALIGN(a)          __attribute__ ((aligned(a)))
    #define ALIGN4F                     16

    #if ((__GNUC__ >= 3) && (__GNUC_MINOR__ >= 4)) || (__GNUC__ >= 4)
        #define EXPLICIT_TEMPLATE   template
    #endif
#endif

#if defined(__GNUC__) && ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 2))
 #define TEMPLATE_SPEC(L, R) template<L,R>
#else
 #define TEMPLATE_SPEC(L, R) template<>
#endif

#ifndef RESTRICT
    #define RESTRICT
#endif

#ifndef MECANIM_FORCE_INLINE
    #define MECANIM_FORCE_INLINE        inline
#endif

#ifndef STATIC_INLINE
    #define STATIC_INLINE       static inline
#endif

#ifndef ATTRIBUTE_ALIGN
    #define ATTRIBUTE_ALIGN(a)
#endif

#ifndef EXPLICIT_TYPENAME
    #define EXPLICIT_TYPENAME   typename
#endif

#ifndef EXPLICIT_TEMPLATE
    #define EXPLICIT_TEMPLATE
#endif

#ifndef DLL_IMPORT
    #define DLL_IMPORT
#endif

#ifndef DLL_EXPORT
    #define DLL_EXPORT
#endif

#ifndef ALIGN4F
    #define ALIGN4F                     16
#endif

#ifndef DECLARE_C
    #define DECLARE_C
#endif

template<typename TYPE>
class OffsetPtr
{
public:
    typedef TYPE        value_type;
    typedef TYPE*       ptr_type;
    typedef TYPE const* const_ptr_type;
    typedef TYPE&       reference_type;
    typedef TYPE const& const_reference_type;
    typedef size_t      offset_type;

    OffsetPtr() : m_Offset(0)
#if UNITY_EDITOR
        , m_DebugPtr(0)
#endif
    {
    }

    OffsetPtr(const OffsetPtr<value_type>& offsetPtr)
    {
        Reset(const_cast<ptr_type>(offsetPtr.Get()));
    }

    OffsetPtr& operator=(const OffsetPtr<value_type>& offsetPtr)
    {
        Reset(const_cast<ptr_type>(offsetPtr.Get()));
        return *this;
    }

    void Reset(ptr_type ptr)
    {
        m_Offset = ptr != 0 ? reinterpret_cast<size_t>(ptr) - reinterpret_cast<size_t>(this) : 0;
#if UNITY_EDITOR
        m_DebugPtr = ptr;
#endif
    }

    void ValidatePointer() const
    {
    }

    OffsetPtr& operator=(const ptr_type ptr)
    {
        Reset(ptr);
        return *this;
    }

    ptr_type operator->()
    {
        ValidatePointer();

        ptr_type ptr = reinterpret_cast<ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
#if UNITY_EDITOR
        m_DebugPtr = ptr;
#endif
        return ptr;
    }

    const_ptr_type operator->() const
    {
        ValidatePointer();

        return reinterpret_cast<const_ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
    }

    reference_type operator*()
    {
        ValidatePointer();

        ptr_type ptr = reinterpret_cast<ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
#if UNITY_EDITOR
        m_DebugPtr = ptr;
#endif
        return *ptr;
    }

    const_reference_type operator*() const
    {
        ValidatePointer();

        return *reinterpret_cast<const_ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
    }

    value_type& operator[](std::size_t i)
    {
        ValidatePointer();

        ptr_type ptr = reinterpret_cast<ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
#if UNITY_EDITOR
        m_DebugPtr = ptr;
#endif
        return ptr[i];
    }

    value_type const& operator[](std::size_t i) const
    {
        ValidatePointer();

        const_ptr_type ptr = reinterpret_cast<const_ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
        return ptr[i];
    }

    bool IsNull() const
    {
        return m_Offset == 0;
    }

    ptr_type Get()
    {
        if (IsNull())
            return NULL;

#if UNITY_EDITOR
        m_DebugPtr = reinterpret_cast<ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
#endif
        return reinterpret_cast<ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
    }

    const_ptr_type Get() const
    {
        if (IsNull())
            return NULL;

#if UNITY_EDITOR
        m_DebugPtr = reinterpret_cast<ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
#endif
        return reinterpret_cast<ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
    }

    size_t get_size() const
    {
        return sizeof(TYPE);
    }

protected:
    offset_type         m_Offset;
#if UNITY_EDITOR
    mutable ptr_type    m_DebugPtr;
#endif

    // Those accessor are needed by the serialization system to write value
    ptr_type Get_Unsafe()
    {
        return reinterpret_cast<ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
    }

    const_ptr_type Get_Unsafe() const
    {
        return reinterpret_cast<ptr_type>(reinterpret_cast<std::size_t>(this) + m_Offset);
    }
};

