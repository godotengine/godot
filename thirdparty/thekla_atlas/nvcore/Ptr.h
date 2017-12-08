// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#ifndef NV_CORE_PTR_H
#define NV_CORE_PTR_H

#include "nvcore.h"
#include "Debug.h"

#include "RefCounted.h"

namespace nv
{
    class WeakProxy;

    /** Simple auto pointer template class.
    *
    * This is very similar to the standard auto_ptr class, but with some 
    * additional limitations to make its use less error prone:
    * - Copy constructor and assignment operator are disabled.
    * - reset method is removed.
    * 
    * The semantics of the standard auto_ptr are not clear and change depending
    * on the std implementation. For a discussion of the problems of auto_ptr read:
    * http://www.awprofessional.com/content/images/020163371X/autoptrupdate\auto_ptr_update.html
    */
    template <class T>
    class AutoPtr
    {
        NV_FORBID_COPY(AutoPtr);
        NV_FORBID_HEAPALLOC();
    public:

        /// Ctor.
        AutoPtr(T * p = NULL) : m_ptr(p) { }

        template <class Q>
        AutoPtr(Q * p) : m_ptr(static_cast<T *>(p)) { }

        /// Dtor. Deletes owned pointer.
        ~AutoPtr() {
            delete m_ptr;
            m_ptr = NULL;
        }

        /// Delete owned pointer and assign new one.
        void operator=( T * p ) {
            if (p != m_ptr)
            {
                delete m_ptr;
                m_ptr = p;
            }
        }

        template <class Q>
        void operator=( Q * p ) {
            if (p != m_ptr)
            {
                delete m_ptr;
                m_ptr = static_cast<T *>(p);
            }
        }

        /// Member access.
        T * operator -> () const {
            nvDebugCheck(m_ptr != NULL);
            return m_ptr;
        }

        /// Get reference.
        T & operator*() const {
            nvDebugCheck(m_ptr != NULL);
            return *m_ptr;
        }

        /// Get pointer.
        T * ptr() const { return m_ptr; }

        /// Relinquish ownership of the underlying pointer and returns that pointer.
        T * release() {
            T * tmp = m_ptr;
            m_ptr = NULL;
            return tmp;
        }

        /// Const pointer equal comparation.
        friend bool operator == (const AutoPtr<T> & ap, const T * const p) {
            return (ap.ptr() == p);
        }

        /// Const pointer nequal comparation.
        friend bool operator != (const AutoPtr<T> & ap, const T * const p) {
            return (ap.ptr() != p);
        }

        /// Const pointer equal comparation.
        friend bool operator == (const T * const p, const AutoPtr<T> & ap) {
            return (ap.ptr() == p);
        }

        /// Const pointer nequal comparation.
        friend bool operator != (const T * const p, const AutoPtr<T> & ap) {
            return (ap.ptr() != p);
        }

    private:
        T * m_ptr;
    };


    /// Smart pointer template class.
    template <class BaseClass>
    class SmartPtr {
    public:

        // BaseClass must implement addRef() and release().
        typedef SmartPtr<BaseClass> ThisType;

        /// Default ctor.
        SmartPtr() : m_ptr(NULL) 
        {
        }

        /// Other type assignment.
        template <class OtherBase>
        SmartPtr( const SmartPtr<OtherBase> & tc )
        {
            m_ptr = static_cast<BaseClass *>( tc.ptr() );
            if (m_ptr) {
                m_ptr->addRef();
            }
        }

        /// Copy ctor.
        SmartPtr( const ThisType & bc )
        {
            m_ptr = bc.ptr();
            if (m_ptr) {
                m_ptr->addRef();
            }
        }

        /// Copy cast ctor. SmartPtr(NULL) is valid.
        explicit SmartPtr( BaseClass * bc )
        {
            m_ptr = bc;
            if (m_ptr) {
                m_ptr->addRef();
            }
        }

        /// Dtor.
        ~SmartPtr()
        {
            set(NULL);
        }


        /// -> operator.
        BaseClass * operator -> () const
        {
            nvCheck( m_ptr != NULL );
            return m_ptr;
        }

        /// * operator.
        BaseClass & operator*() const
        {
            nvCheck( m_ptr != NULL );
            return *m_ptr;
        }

        /// Get pointer.
        BaseClass * ptr() const
        {
            return m_ptr;
        }

        /// Other type assignment.
        template <class OtherBase>
        void operator = ( const SmartPtr<OtherBase> & tc )
        {
            set( static_cast<BaseClass *>(tc.ptr()) );
        }

        /// This type assignment.
        void operator = ( const ThisType & bc )
        {
            set( bc.ptr() );
        }

        /// Pointer assignment.
        void operator = ( BaseClass * bc )
        {
            set( bc );
        }


        /// Other type equal comparation.
        template <class OtherBase>
        bool operator == ( const SmartPtr<OtherBase> & other ) const
        {
            return m_ptr == other.ptr();
        }

        /// This type equal comparation.
        bool operator == ( const ThisType & bc ) const
        {
            return m_ptr == bc.ptr();
        }

        /// Const pointer equal comparation.
        bool operator == ( const BaseClass * const bc ) const
        {
            return m_ptr == bc;
        }

        /// Other type not equal comparation.
        template <class OtherBase>
        bool operator != ( const SmartPtr<OtherBase> & other ) const
        {
            return m_ptr != other.ptr();
        }

        /// Other type not equal comparation.
        bool operator != ( const ThisType & bc ) const
        {
            return m_ptr != bc.ptr();
        }

        /// Const pointer not equal comparation.
        bool operator != (const BaseClass * const bc) const
        {
            return m_ptr != bc;
        }

        /// This type lower than comparation.
        bool operator < (const ThisType & p) const
        {
            return m_ptr < p.ptr();
        }

        bool isValid() const {
            return isValidPtr(m_ptr);
        }

    private:

        // Set this pointer.
        void set( BaseClass * p )
        {
            if (p) p->addRef();
            if (m_ptr) m_ptr->release();
            m_ptr = p;
        }

    private:

        BaseClass * m_ptr;

    };


    /// Smart pointer template class.
    template <class T>
    class WeakPtr {
    public:

        WeakPtr() {}

        WeakPtr(T * p)  { operator=(p); }
        WeakPtr(const SmartPtr<T> & p) { operator=(p.ptr()); }

        // Default constructor and assignment from weak_ptr<T> are OK.

        void operator=(T * p)
        {
            if (p) {
                m_proxy = p->getWeakProxy();
                nvDebugCheck(m_proxy != NULL);
                nvDebugCheck(m_proxy->ptr() == p);
            }
            else {
                m_proxy = NULL;
            }
        }

        void operator=(const SmartPtr<T> & ptr) { operator=(ptr.ptr()); }

        bool operator==(const SmartPtr<T> & p) const { return ptr() == p.ptr(); }
        bool operator!=(const SmartPtr<T> & p) const { return ptr() != p.ptr(); }

        bool operator==(const WeakPtr<T> & p) const { return ptr() == p.ptr(); }
        bool operator!=(const WeakPtr<T> & p) const { return ptr() != p.ptr(); }

        bool operator==(T * p) const { return ptr() == p; }
        bool operator!=(T * p) const { return ptr() != p; }

        T * operator->() const
        {
            T * p = ptr();
            nvDebugCheck(p != NULL);
            return p;
        }

        T * ptr() const
        {
            if (m_proxy != NULL) {
                return static_cast<T *>(m_proxy->ptr());
            }
            return NULL;
        }

    private:

        mutable SmartPtr<WeakProxy> m_proxy;

    };


} // nv namespace

#endif // NV_CORE_PTR_H
