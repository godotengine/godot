#pragma once
#ifndef VHACD_VECTOR_INL
#define VHACD_VECTOR_INL
namespace VHACD
{
    template <typename T> 
    inline Vec3<T> operator*(T lhs, const Vec3<T> & rhs)
    {
        return Vec3<T>(lhs * rhs.X(), lhs * rhs.Y(), lhs * rhs.Z());
    }
    template <typename T> 
    inline T & Vec3<T>::X() 
    {
        return m_data[0];
    }
    template <typename T> 
    inline  T &    Vec3<T>::Y() 
    {
        return m_data[1];
    }
    template <typename T>    
    inline  T &    Vec3<T>::Z() 
    {
        return m_data[2];
    }
    template <typename T> 
    inline  const T & Vec3<T>::X() const 
    {
        return m_data[0];
    }
    template <typename T> 
    inline  const T & Vec3<T>::Y() const 
    {
        return m_data[1];
    }
    template <typename T> 
    inline  const T & Vec3<T>::Z() const 
    {
        return m_data[2];
    }
    template <typename T> 
    inline  void Vec3<T>::Normalize()
    {
        T n = sqrt(m_data[0]*m_data[0]+m_data[1]*m_data[1]+m_data[2]*m_data[2]);
        if (n != 0.0) (*this) /= n;
    }
    template <typename T> 
    inline  T Vec3<T>::GetNorm() const 
    { 
        return sqrt(m_data[0]*m_data[0]+m_data[1]*m_data[1]+m_data[2]*m_data[2]);
    }
    template <typename T> 
    inline  void Vec3<T>::operator= (const Vec3 & rhs)
    { 
        this->m_data[0] = rhs.m_data[0]; 
        this->m_data[1] = rhs.m_data[1]; 
        this->m_data[2] = rhs.m_data[2]; 
    }
    template <typename T> 
    inline  void Vec3<T>::operator+=(const Vec3 & rhs)
    { 
        this->m_data[0] += rhs.m_data[0]; 
        this->m_data[1] += rhs.m_data[1]; 
        this->m_data[2] += rhs.m_data[2]; 
    }     
    template <typename T>  
    inline void Vec3<T>::operator-=(const Vec3 & rhs)
    { 
        this->m_data[0] -= rhs.m_data[0]; 
        this->m_data[1] -= rhs.m_data[1]; 
        this->m_data[2] -= rhs.m_data[2]; 
    }
    template <typename T> 
    inline void Vec3<T>::operator-=(T a)
    { 
        this->m_data[0] -= a; 
        this->m_data[1] -= a; 
        this->m_data[2] -= a; 
    }
    template <typename T> 
    inline void Vec3<T>::operator+=(T a)
    { 
        this->m_data[0] += a; 
        this->m_data[1] += a; 
        this->m_data[2] += a; 
    }
    template <typename T> 
    inline void Vec3<T>::operator/=(T a)
    { 
        this->m_data[0] /= a; 
        this->m_data[1] /= a; 
        this->m_data[2] /= a; 
    }
    template <typename T>  
    inline void Vec3<T>::operator*=(T a)
    { 
        this->m_data[0] *= a; 
        this->m_data[1] *= a; 
        this->m_data[2] *= a; 
    }  
    template <typename T> 
    inline Vec3<T> Vec3<T>::operator^ (const Vec3<T> & rhs) const
    {
        return Vec3<T>(m_data[1] * rhs.m_data[2] - m_data[2] * rhs.m_data[1],
                       m_data[2] * rhs.m_data[0] - m_data[0] * rhs.m_data[2],
                       m_data[0] * rhs.m_data[1] - m_data[1] * rhs.m_data[0]);
    }
    template <typename T>
    inline T Vec3<T>::operator*(const Vec3<T> & rhs) const
    {
        return (m_data[0] * rhs.m_data[0] + m_data[1] * rhs.m_data[1] + m_data[2] * rhs.m_data[2]);
    }        
    template <typename T>
    inline Vec3<T> Vec3<T>::operator+(const Vec3<T> & rhs) const
    {
        return Vec3<T>(m_data[0] + rhs.m_data[0],m_data[1] + rhs.m_data[1],m_data[2] + rhs.m_data[2]);
    }
    template <typename T> 
    inline  Vec3<T> Vec3<T>::operator-(const Vec3<T> & rhs) const
    {
        return Vec3<T>(m_data[0] - rhs.m_data[0],m_data[1] - rhs.m_data[1],m_data[2] - rhs.m_data[2]) ;
    }     
    template <typename T> 
    inline  Vec3<T> Vec3<T>::operator-() const
    {
        return Vec3<T>(-m_data[0],-m_data[1],-m_data[2]) ;
    }     

    template <typename T> 
    inline Vec3<T> Vec3<T>::operator*(T rhs) const
    {
        return Vec3<T>(rhs * this->m_data[0], rhs * this->m_data[1], rhs * this->m_data[2]);
    }
    template <typename T>
    inline Vec3<T> Vec3<T>::operator/ (T rhs) const
    {
        return Vec3<T>(m_data[0] / rhs, m_data[1] / rhs, m_data[2] / rhs);
    }
    template <typename T>
    inline Vec3<T>::Vec3(T a) 
    { 
        m_data[0] = m_data[1] = m_data[2] = a; 
    }
    template <typename T>
    inline Vec3<T>::Vec3(T x, T y, T z)
    {
        m_data[0] = x;
        m_data[1] = y;
        m_data[2] = z;
    }
    template <typename T>
    inline Vec3<T>::Vec3(const Vec3 & rhs)
    {        
        m_data[0] = rhs.m_data[0];
        m_data[1] = rhs.m_data[1];
        m_data[2] = rhs.m_data[2];
    }
    template <typename T>
    inline Vec3<T>::~Vec3(void){};

    template <typename T>
    inline Vec3<T>::Vec3() {}
    
    template<typename T>
    inline const bool Colinear(const Vec3<T> & a, const Vec3<T> & b, const Vec3<T> & c)
    {
        return  ((c.Z() - a.Z()) * (b.Y() - a.Y()) - (b.Z() - a.Z()) * (c.Y() - a.Y()) == 0.0 /*EPS*/) &&
                ((b.Z() - a.Z()) * (c.X() - a.X()) - (b.X() - a.X()) * (c.Z() - a.Z()) == 0.0 /*EPS*/) &&
                ((b.X() - a.X()) * (c.Y() - a.Y()) - (b.Y() - a.Y()) * (c.X() - a.X()) == 0.0 /*EPS*/);
    }
    
    template<typename T>
    inline const T ComputeVolume4(const Vec3<T> & a, const Vec3<T> & b, const Vec3<T> & c, const Vec3<T> & d)
    {
        return (a-d) * ((b-d) ^ (c-d));
    }

    template <typename T> 
    inline bool Vec3<T>::operator<(const Vec3 & rhs) const
    {
        if (X() == rhs[0])
        {
            if (Y() == rhs[1])
            {
                return (Z()<rhs[2]);
            }
            return (Y()<rhs[1]);
        }
        return (X()<rhs[0]);
    }
    template <typename T> 
    inline  bool Vec3<T>::operator>(const Vec3 & rhs) const
    {
        if (X() == rhs[0])
        {
            if (Y() == rhs[1])
            {
                return (Z()>rhs[2]);
            }
            return (Y()>rhs[1]);
        }
        return (X()>rhs[0]);
    } 
    template <typename T> 
    inline Vec2<T> operator*(T lhs, const Vec2<T> & rhs)
    {
        return Vec2<T>(lhs * rhs.X(), lhs * rhs.Y());
    }
    template <typename T> 
    inline T & Vec2<T>::X() 
    {
        return m_data[0];
    }
    template <typename T>    
    inline  T &    Vec2<T>::Y() 
    {
        return m_data[1];
    }
    template <typename T>    
    inline  const T & Vec2<T>::X() const 
    {
        return m_data[0];
    }
    template <typename T>    
    inline  const T & Vec2<T>::Y() const 
    {
        return m_data[1];
    }
    template <typename T>    
    inline  void Vec2<T>::Normalize()
    {
        T n = sqrt(m_data[0]*m_data[0]+m_data[1]*m_data[1]);
        if (n != 0.0) (*this) /= n;
    }
    template <typename T>    
    inline  T Vec2<T>::GetNorm() const 
    { 
        return sqrt(m_data[0]*m_data[0]+m_data[1]*m_data[1]);
    }
    template <typename T>    
    inline  void Vec2<T>::operator= (const Vec2 & rhs)
    { 
        this->m_data[0] = rhs.m_data[0]; 
        this->m_data[1] = rhs.m_data[1]; 
    }
    template <typename T>    
    inline  void Vec2<T>::operator+=(const Vec2 & rhs)
    { 
        this->m_data[0] += rhs.m_data[0]; 
        this->m_data[1] += rhs.m_data[1]; 
    }     
    template <typename T>  
    inline void Vec2<T>::operator-=(const Vec2 & rhs)
    { 
        this->m_data[0] -= rhs.m_data[0]; 
        this->m_data[1] -= rhs.m_data[1]; 
    }
    template <typename T>  
    inline void Vec2<T>::operator-=(T a)
    { 
        this->m_data[0] -= a; 
        this->m_data[1] -= a; 
    }
    template <typename T>  
    inline void Vec2<T>::operator+=(T a)
    { 
        this->m_data[0] += a; 
        this->m_data[1] += a; 
    }
    template <typename T>  
    inline void Vec2<T>::operator/=(T a)
    { 
        this->m_data[0] /= a; 
        this->m_data[1] /= a; 
    }
    template <typename T>  
    inline void Vec2<T>::operator*=(T a)
    { 
        this->m_data[0] *= a; 
        this->m_data[1] *= a; 
    }  
    template <typename T> 
    inline T Vec2<T>::operator^ (const Vec2<T> & rhs) const
    {
        return m_data[0] * rhs.m_data[1] - m_data[1] * rhs.m_data[0];
    }
    template <typename T>
    inline T Vec2<T>::operator*(const Vec2<T> & rhs) const
    {
        return (m_data[0] * rhs.m_data[0] + m_data[1] * rhs.m_data[1]);
    }        
    template <typename T>
    inline Vec2<T> Vec2<T>::operator+(const Vec2<T> & rhs) const
    {
        return Vec2<T>(m_data[0] + rhs.m_data[0],m_data[1] + rhs.m_data[1]);
    }
    template <typename T> 
    inline  Vec2<T> Vec2<T>::operator-(const Vec2<T> & rhs) const
    {
        return Vec2<T>(m_data[0] - rhs.m_data[0],m_data[1] - rhs.m_data[1]);
    }     
    template <typename T> 
    inline  Vec2<T> Vec2<T>::operator-() const
    {
        return Vec2<T>(-m_data[0],-m_data[1]) ;
    }     

    template <typename T> 
    inline Vec2<T> Vec2<T>::operator*(T rhs) const
    {
        return Vec2<T>(rhs * this->m_data[0], rhs * this->m_data[1]);
    }
    template <typename T>
    inline Vec2<T> Vec2<T>::operator/ (T rhs) const
    {
        return Vec2<T>(m_data[0] / rhs, m_data[1] / rhs);
    }
    template <typename T>
    inline Vec2<T>::Vec2(T a) 
    { 
        m_data[0] = m_data[1] = a; 
    }
    template <typename T>
    inline Vec2<T>::Vec2(T x, T y)
    {
        m_data[0] = x;
        m_data[1] = y;
    }
    template <typename T>
    inline Vec2<T>::Vec2(const Vec2 & rhs)
    {        
        m_data[0] = rhs.m_data[0];
        m_data[1] = rhs.m_data[1];
    }
    template <typename T>
    inline Vec2<T>::~Vec2(void){};

    template <typename T>
    inline Vec2<T>::Vec2() {}

   /*
     InsideTriangle decides if a point P is Inside of the triangle
     defined by A, B, C.
   */
    template<typename T>
    inline const bool InsideTriangle(const Vec2<T> & a, const Vec2<T> & b, const Vec2<T> & c, const Vec2<T> & p)
    {
        T ax, ay, bx, by, cx, cy, apx, apy, bpx, bpy, cpx, cpy;
        T cCROSSap, bCROSScp, aCROSSbp;
        ax = c.X() - b.X();  ay = c.Y() - b.Y();
        bx = a.X() - c.X();  by = a.Y() - c.Y();
        cx = b.X() - a.X();  cy = b.Y() - a.Y();
        apx= p.X() - a.X();  apy= p.Y() - a.Y();
        bpx= p.X() - b.X();  bpy= p.Y() - b.Y();
        cpx= p.X() - c.X();  cpy= p.Y() - c.Y();
        aCROSSbp = ax*bpy - ay*bpx;
        cCROSSap = cx*apy - cy*apx;
        bCROSScp = bx*cpy - by*cpx;
        return ((aCROSSbp >= 0.0) && (bCROSScp >= 0.0) && (cCROSSap >= 0.0));
    }
}
#endif //VHACD_VECTOR_INL