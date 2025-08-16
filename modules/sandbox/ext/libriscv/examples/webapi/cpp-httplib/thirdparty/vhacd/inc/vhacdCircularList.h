/* Copyright (c) 2011 Khaled Mamou (kmamou at gmail dot com)
 All rights reserved.
 
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 3. The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#ifndef VHACD_CIRCULAR_LIST_H
#define VHACD_CIRCULAR_LIST_H
#include <stdlib.h>
namespace VHACD {
//!    CircularListElement class.
template <typename T>
class CircularListElement {
public:
    T& GetData() { return m_data; }
    const T& GetData() const { return m_data; }
    CircularListElement<T>*& GetNext() { return m_next; }
    CircularListElement<T>*& GetPrev() { return m_prev; }
    const CircularListElement<T>*& GetNext() const { return m_next; }
    const CircularListElement<T>*& GetPrev() const { return m_prev; }
    //!    Constructor
    CircularListElement(const T& data) { m_data = data; }
    CircularListElement(void) {}
    //! Destructor
    ~CircularListElement(void) {}
private:
    T m_data;
    CircularListElement<T>* m_next;
    CircularListElement<T>* m_prev;

    CircularListElement(const CircularListElement& rhs);
};
//!    CircularList class.
template <typename T>
class CircularList {
public:
    CircularListElement<T>*& GetHead() { return m_head; }
    const CircularListElement<T>* GetHead() const { return m_head; }
    bool IsEmpty() const { return (m_size == 0); }
    size_t GetSize() const { return m_size; }
    const T& GetData() const { return m_head->GetData(); }
    T& GetData() { return m_head->GetData(); }
    bool Delete();
    bool Delete(CircularListElement<T>* element);
    CircularListElement<T>* Add(const T* data = 0);
    CircularListElement<T>* Add(const T& data);
    bool Next();
    bool Prev();
    void Clear()
    {
        while (Delete())
            ;
    };
    const CircularList& operator=(const CircularList& rhs);
    //!    Constructor
    CircularList()
    {
        m_head = 0;
        m_size = 0;
    }
    CircularList(const CircularList& rhs);
    //! Destructor
    ~CircularList(void) { Clear(); };
private:
    CircularListElement<T>* m_head; //!< a pointer to the head of the circular list
    size_t m_size; //!< number of element in the circular list
};
}
#include "vhacdCircularList.inl"
#endif // VHACD_CIRCULAR_LIST_H