// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#ifndef NV_CORE_REFCOUNTED_H
#define NV_CORE_REFCOUNTED_H

#include "nvcore.h"
#include "Debug.h"

#define NV_DECLARE_PTR(Class) \
    template <class T> class SmartPtr; \
    typedef SmartPtr<class Class> Class ## Ptr; \
    typedef SmartPtr<const class Class> Class ## ConstPtr


namespace nv
{
    /// Weak proxy.
    class WeakProxy
    {
        NV_FORBID_COPY(WeakProxy);
    public:
	    /// Ctor.
	    WeakProxy(void * ptr) : m_count(0), m_ptr(ptr) { }

        /// Dtor.
        ~WeakProxy()
        {
            nvCheck( m_count == 0 );
        }

        /// Increase reference count.
        uint addRef() const
        {
            m_count++;
            return m_count;
        }

        /// Decrease reference count and remove when 0.
        uint release() const
        {
            nvCheck( m_count > 0 );

            m_count--;
            if( m_count == 0 ) {
                delete this;
                return 0;
            }
            return m_count;
        }

	    /// WeakPtr's call this to determine if their pointer is valid or not.
	    bool isAlive() const {
		    return m_ptr != NULL;
	    }

	    /// Only the actual object should call this.
	    void notifyObjectDied() {
		    m_ptr = NULL;
	    }

        /// Return proxy pointer.
        void * ptr() const {
            return m_ptr;
        }

    private:
        mutable int m_count;
	    void * m_ptr;
    };


    /// Reference counted base class to be used with SmartPtr and WeakPtr.
    class RefCounted
    {
        NV_FORBID_COPY(RefCounted);
    public:

        /// Ctor.
        RefCounted() : m_count(0), m_weak_proxy(NULL)
        {
        }

        /// Virtual dtor.
        virtual ~RefCounted()
        {
            nvCheck( m_count == 0 );
            releaseWeakProxy();
        }


        /// Increase reference count.
        uint addRef() const
        {
            m_count++;
            return m_count;
        }


        /// Decrease reference count and remove when 0.
        uint release() const
        {
            nvCheck( m_count > 0 );

            m_count--;
            if( m_count == 0 ) {
                delete this;
                return 0;
            }
            return m_count;
        }

        /// Get weak proxy.
        WeakProxy * getWeakProxy() const
        {
            if (m_weak_proxy == NULL) {
                m_weak_proxy = new WeakProxy((void *)this);
                m_weak_proxy->addRef();
            }
            return m_weak_proxy;
        }

        /// Release the weak proxy.	
        void releaseWeakProxy() const
        {
            if (m_weak_proxy != NULL) {
                m_weak_proxy->notifyObjectDied();
                m_weak_proxy->release();
                m_weak_proxy = NULL;
            }
        }

        /// Get reference count.
        int refCount() const
        {
            return m_count;
        }


    private:

        mutable int m_count;
        mutable WeakProxy * m_weak_proxy;

    };

} // nv namespace


#endif // NV_CORE_REFCOUNTED_H
