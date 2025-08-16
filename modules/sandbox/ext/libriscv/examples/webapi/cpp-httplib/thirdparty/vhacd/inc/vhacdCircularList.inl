#pragma once
#ifndef HACD_CIRCULAR_LIST_INL
#define HACD_CIRCULAR_LIST_INL
namespace VHACD
{
    template < typename T > 
    inline bool CircularList<T>::Delete(CircularListElement<T> * element)
    {
        if (!element)
        {
            return false;
        }
        if (m_size > 1)
        {
            CircularListElement<T> * next = element->GetNext();
            CircularListElement<T> * prev = element->GetPrev();
            delete element;
            m_size--;
            if (element == m_head)
            {
                m_head = next;
            }
            next->GetPrev() = prev;
            prev->GetNext() = next;
            return true;
        }
        else if (m_size == 1)
        {
            delete m_head;
            m_size--;
            m_head = 0;
            return true;
        }
        else
        {
            return false;
        }
    }
    
    template < typename T > 
    inline bool CircularList<T>::Delete()
    {
        if (m_size > 1)
        {
            CircularListElement<T> * next = m_head->GetNext();
            CircularListElement<T> * prev = m_head->GetPrev();
            delete m_head;
            m_size--;
            m_head = next;
            next->GetPrev() = prev;
            prev->GetNext() = next;
            return true;
        }
        else if (m_size == 1)
        {
            delete m_head;
            m_size--;
            m_head = 0;
            return true;
        }
        else
        {
            return false;
        }
    }
    template < typename T > 
    inline CircularListElement<T> * CircularList<T>::Add(const T * data)
    {
        if (m_size == 0)
        {
            if (data)
            {
                m_head = new CircularListElement<T>(*data);
            }
            else
            {
                m_head = new CircularListElement<T>();
            }
            m_head->GetNext() = m_head->GetPrev() = m_head;
        }
        else
        {
            CircularListElement<T> * next = m_head->GetNext();
            CircularListElement<T> * element = m_head;
            if (data)
            {
                m_head = new CircularListElement<T>(*data);
            }
            else
            {
                m_head = new CircularListElement<T>();
            }
            m_head->GetNext() = next;
            m_head->GetPrev() = element;
            element->GetNext() = m_head;
            next->GetPrev() = m_head;
        }
        m_size++;
        return m_head;
    }
    template < typename T > 
    inline CircularListElement<T> * CircularList<T>::Add(const T & data)
    {
        const T * pData = &data;
        return Add(pData);
    }
    template < typename T > 
    inline bool CircularList<T>::Next()
    {
        if (m_size == 0)
        {
            return false;
        }
        m_head = m_head->GetNext();
        return true;
    }
    template < typename T > 
    inline bool CircularList<T>::Prev()
    {
        if (m_size == 0)
        {
            return false;
        }
        m_head = m_head->GetPrev();
        return true;
    }
    template < typename T > 
    inline CircularList<T>::CircularList(const CircularList& rhs)
    {
        if (rhs.m_size > 0)
        {
            CircularListElement<T> * current = rhs.m_head;
            do
            {
                current = current->GetNext();
                Add(current->GetData());
            }
            while ( current != rhs.m_head );
        }
    }
    template < typename T > 
    inline const CircularList<T>& CircularList<T>::operator=(const CircularList& rhs)
    {
        if (&rhs != this)
        {
            Clear();
            if (rhs.m_size > 0)
            {
                CircularListElement<T> * current = rhs.m_head;
                do
                {
                    current = current->GetNext();
                    Add(current->GetData());
                }
                while ( current != rhs.m_head );
            }
        }
        return (*this);
    }
}
#endif