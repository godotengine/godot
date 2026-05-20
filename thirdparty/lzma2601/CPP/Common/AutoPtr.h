// Common/AutoPtr.h

#ifndef ZIP7_INC_COMMON_AUTOPTR_H
#define ZIP7_INC_COMMON_AUTOPTR_H

template<class T> class CMyUniquePtr
// CMyAutoPtr
{
  T *_p;
  
  CMyUniquePtr(CMyUniquePtr<T>& p); // : _p(p.release()) {}
  CMyUniquePtr<T>& operator=(T *p);
  CMyUniquePtr<T>& operator=(CMyUniquePtr<T>& p);
  /*
  {
    reset(p.release());
    return (*this);
  }
  */
  void reset(T* p = NULL)
  {
    if (p != _p)
      delete _p;
    _p = p;
  }
public:
  CMyUniquePtr(T *p = NULL) : _p(p) {}
  ~CMyUniquePtr() { delete _p; }
  T& operator*() const { return *_p; }
  T* operator->() const { return _p; }
  // operator bool() const { return _p != NULL; }
  T* get() const { return _p; }
  T* release()
  {
    T *tmp = _p;
    _p = NULL;
    return tmp;
  }
  void Create_if_Empty()
  {
    if (!_p)
      _p = new T;
  }
};

#endif
