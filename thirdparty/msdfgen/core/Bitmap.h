
#pragma once

#include "YAxisOrientation.h"
#include "BitmapRef.hpp"

namespace msdfgen {

/// A 2D image bitmap with N channels of type T. Pixel memory is managed by the class.
template <typename T, int N = 1>
class Bitmap {

public:
    Bitmap();
    Bitmap(int width, int height, YAxisOrientation yOrientation = MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION);
    explicit Bitmap(const BitmapConstRef<T, N> &orig);
    explicit Bitmap(const BitmapConstSection<T, N> &orig);
    Bitmap(const Bitmap<T, N> &orig);
#ifdef MSDFGEN_USE_CPP11
    Bitmap(Bitmap<T, N> &&orig);
#endif
    ~Bitmap();
    Bitmap<T, N> &operator=(const BitmapConstRef<T, N> &orig);
    Bitmap<T, N> &operator=(const BitmapConstSection<T, N> &orig);
    Bitmap<T, N> &operator=(const Bitmap<T, N> &orig);
#ifdef MSDFGEN_USE_CPP11
    Bitmap<T, N> &operator=(Bitmap<T, N> &&orig);
#endif
    /// Bitmap width in pixels.
    int width() const;
    /// Bitmap height in pixels.
    int height() const;
    T *operator()(int x, int y);
    const T *operator()(int x, int y) const;
#ifdef MSDFGEN_USE_CPP11
    explicit operator T *();
    explicit operator const T *() const;
#else
    operator T *();
    operator const T *() const;
#endif
    operator BitmapRef<T, N>();
    operator BitmapConstRef<T, N>() const;
    operator BitmapSection<T, N>();
    operator BitmapConstSection<T, N>() const;
    /// Returns a reference to a rectangular section of the bitmap specified by bounds (excluding xMax, yMax).
    BitmapSection<T, N> getSection(int xMin, int yMin, int xMax, int yMax);
    /// Returns a constant reference to a rectangular section of the bitmap specified by bounds (excluding xMax, yMax).
    BitmapConstSection<T, N> getConstSection(int xMin, int yMin, int xMax, int yMax) const;

private:
    T *pixels;
    int w, h;
    YAxisOrientation yOrientation;

};

}

#include "Bitmap.hpp"
