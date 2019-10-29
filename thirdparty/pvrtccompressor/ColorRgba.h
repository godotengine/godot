#pragma once

namespace Javelin {

template<typename T>
class ColorRgb {
public:
    T b;
    T g;
    T r;
    

    ColorRgb()
        : b(0)
        , g(0)
        , r(0) {
    }

    ColorRgb(T red, T green, T blue)
        : b(blue)
        , g(green)
        , r(red) {
    }

    ColorRgb(const ColorRgb<T> &x)
        : b(x.b)
        , g(x.g)
        , r(x.r) {
    }

    ColorRgb<int> operator *(int x) {
        return ColorRgb<int>(r * x, g * x, b * x);
    }

    ColorRgb<int> operator +(const ColorRgb<T> &x) const {
        return ColorRgb<int>(r + (int)x.r, g + (int)x.g, b + (int)x.b);
    }

    ColorRgb<int> operator -(const ColorRgb<T> &x) const {
        return ColorRgb<int>(r - (int)x.r, g - (int)x.g, b - (int)x.b);
    }

    int operator %(const ColorRgb<T> &x) const {
        return r * (int)x.r + g * (int)x.g + b * (int)x.b;
    }

    bool operator ==(const ColorRgb<T> &x) const {
        return r == x.r && g == x.g && b == x.b;
    }

    bool operator !=(const ColorRgb<T> &x) const {
        return r != x.r || g != x.g || b != x.b;
    }

    void SetMin(const ColorRgb<T> &x) {
        if (x.r < r) {
            r = x.r;
        }
        if (x.g < g) {
            g = x.g;
        }
        if (x.b < b) {
            b = x.b;
        }
    }

    void SetMax(const ColorRgb<T> &x) {
        if (x.r > r) {
            r = x.r;
        }
        if (x.g > g) {
            g = x.g;
        }
        if (x.b > b) {
            b = x.b;
        }
    }
};

template<typename T>
class ColorRgba : public ColorRgb<T> {
public:
    T a;

    ColorRgba() :
        a(0) {
    }

    ColorRgba(T red, T green, T blue, T alpha)
        : ColorRgb<T>(red, green, blue)
        , a(alpha) {
    }

    ColorRgba(const ColorRgba<T> &x)
        : ColorRgb<T>(x.r, x.g, x.b)
        , a(x.a) {
    }

    ColorRgba<int> operator *(int x) {
        return ColorRgba<T>(ColorRgb<T>::r * x, 
                            ColorRgb<T>::g * x, 
                            ColorRgb<T>::b * x, 
                            a * x);
    }

    ColorRgba<int> operator +(const ColorRgba<T> &x) {
        return ColorRgba<T>(ColorRgb<T>::r + (int)x.r, 
                            ColorRgb<T>::g + (int)x.g, 
                            ColorRgb<T>::b + (int)x.b, 
                            a + (int)x.a);
    }

    ColorRgba<int> operator -(const ColorRgba<T> &x) {
        return ColorRgba<T>(ColorRgb<T>::r - (int)x.r, 
                            ColorRgb<T>::g - (int)x.g, 
                            ColorRgb<T>::b - (int)x.b, 
                            a - (int)x.a);
    }

    int operator %(const ColorRgba<T> &x) {
        return ColorRgb<T>::r * (int)x.r + 
               ColorRgb<T>::g * (int)x.g + 
               ColorRgb<T>::b * (int)x.b + 
               a * (int)x.a;
    }

    bool operator ==(const ColorRgba<T> &x) {
        return ColorRgb<T>::r == x.r && ColorRgb<T>::g == x.g && 
               ColorRgb<T>::b == x.b && a == x.a;
    }

    bool operator !=(const ColorRgba<T> &x) {
        return ColorRgb<T>::r != x.r || ColorRgb<T>::g != x.g || 
               ColorRgb<T>::b != x.b || a != x.a;
    }

    void SetMin(const ColorRgba<T> &x) {
        ColorRgb<T>::SetMin(x);
        if (x.a < a) {
            a = x.a;
        }
    }

    void SetMax(const ColorRgba<T> &x) {
        ColorRgb<T>::SetMax(x);
        if (x.a > a) {
            a = x.a;
        }
    }
};

}
