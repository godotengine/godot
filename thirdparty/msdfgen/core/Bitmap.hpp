
#include "Bitmap.h"

#include <cstdlib>
#include <cstring>

namespace msdfgen {

template <typename T, int N>
Bitmap<T, N>::Bitmap() : pixels(NULL), w(0), h(0), yOrientation(MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION) { }

template <typename T, int N>
Bitmap<T, N>::Bitmap(int width, int height, YAxisOrientation yOrientation) : w(width), h(height), yOrientation(yOrientation) {
    pixels = new T[N*w*h];
}

template <typename T, int N>
Bitmap<T, N>::Bitmap(const BitmapConstRef<T, N> &orig) : w(orig.width), h(orig.height), yOrientation(orig.yOrientation) {
    pixels = new T[N*w*h];
    memcpy(pixels, orig.pixels, sizeof(T)*N*w*h);
}

template <typename T, int N>
Bitmap<T, N>::Bitmap(const BitmapConstSection<T, N> &orig) : w(orig.width), h(orig.height), yOrientation(orig.yOrientation) {
    pixels = new T[N*w*h];
    T *dst = pixels;
    const T *src = orig.pixels;
    int rowLength = N*w;
    for (int y = 0; y < h; ++y) {
        memcpy(dst, src, sizeof(T)*rowLength);
        dst += rowLength;
        src += orig.rowStride;
    }
}

template <typename T, int N>
Bitmap<T, N>::Bitmap(const Bitmap<T, N> &orig) : w(orig.w), h(orig.h), yOrientation(orig.yOrientation) {
    pixels = new T[N*w*h];
    memcpy(pixels, orig.pixels, sizeof(T)*N*w*h);
}

#ifdef MSDFGEN_USE_CPP11
template <typename T, int N>
Bitmap<T, N>::Bitmap(Bitmap<T, N> &&orig) : pixels(orig.pixels), w(orig.w), h(orig.h), yOrientation(orig.yOrientation) {
    orig.pixels = NULL;
    orig.w = 0, orig.h = 0;
}
#endif

template <typename T, int N>
Bitmap<T, N>::~Bitmap() {
    delete[] pixels;
}

template <typename T, int N>
Bitmap<T, N> &Bitmap<T, N>::operator=(const BitmapConstRef<T, N> &orig) {
    if (pixels != orig.pixels) {
        delete[] pixels;
        w = orig.width, h = orig.height;
        yOrientation = orig.yOrientation;
        pixels = new T[N*w*h];
        memcpy(pixels, orig.pixels, sizeof(T)*N*w*h);
    }
    return *this;
}

template <typename T, int N>
Bitmap<T, N> &Bitmap<T, N>::operator=(const BitmapConstSection<T, N> &orig) {
    if (orig.pixels && orig.pixels >= pixels && orig.pixels < pixels+N*w*h)
        return *this = Bitmap<T, N>(orig);
    delete[] pixels;
    w = orig.width, h = orig.height;
    yOrientation = orig.yOrientation;
    pixels = new T[N*w*h];
    T *dst = pixels;
    const T *src = orig.pixels;
    int rowLength = N*w;
    for (int y = 0; y < h; ++y) {
        memcpy(dst, src, sizeof(T)*rowLength);
        dst += rowLength;
        src += orig.rowStride;
    }
    return *this;
}

template <typename T, int N>
Bitmap<T, N> &Bitmap<T, N>::operator=(const Bitmap<T, N> &orig) {
    if (this != &orig) {
        delete[] pixels;
        w = orig.w, h = orig.h;
        yOrientation = orig.yOrientation;
        pixels = new T[N*w*h];
        memcpy(pixels, orig.pixels, sizeof(T)*N*w*h);
    }
    return *this;
}

#ifdef MSDFGEN_USE_CPP11
template <typename T, int N>
Bitmap<T, N> &Bitmap<T, N>::operator=(Bitmap<T, N> &&orig) {
    if (this != &orig) {
        delete[] pixels;
        pixels = orig.pixels;
        w = orig.w, h = orig.h;
        yOrientation = orig.yOrientation;
        orig.pixels = NULL;
    }
    return *this;
}
#endif

template <typename T, int N>
int Bitmap<T, N>::width() const {
    return w;
}

template <typename T, int N>
int Bitmap<T, N>::height() const {
    return h;
}

template <typename T, int N>
T *Bitmap<T, N>::operator()(int x, int y) {
    return pixels+N*(w*y+x);
}

template <typename T, int N>
const T *Bitmap<T, N>::operator()(int x, int y) const {
    return pixels+N*(w*y+x);
}

template <typename T, int N>
Bitmap<T, N>::operator T *() {
    return pixels;
}

template <typename T, int N>
Bitmap<T, N>::operator const T *() const {
    return pixels;
}

template <typename T, int N>
Bitmap<T, N>::operator BitmapRef<T, N>() {
    return BitmapRef<T, N>(pixels, w, h, yOrientation);
}

template <typename T, int N>
Bitmap<T, N>::operator BitmapConstRef<T, N>() const {
    return BitmapConstRef<T, N>(pixels, w, h, yOrientation);
}

template <typename T, int N>
Bitmap<T, N>::operator BitmapSection<T, N>() {
    return BitmapSection<T, N>(pixels, w, h, yOrientation);
}

template <typename T, int N>
Bitmap<T, N>::operator BitmapConstSection<T, N>() const {
    return BitmapConstSection<T, N>(pixels, w, h, yOrientation);
}

template <typename T, int N>
BitmapSection<T, N> Bitmap<T, N>::getSection(int xMin, int yMin, int xMax, int yMax) {
    return BitmapSection<T, N>(pixels+N*(w*yMin+xMin), xMax-xMin, yMax-yMin, N*w, yOrientation);
}

template <typename T, int N>
BitmapConstSection<T, N> Bitmap<T, N>::getConstSection(int xMin, int yMin, int xMax, int yMax) const {
    return BitmapConstSection<T, N>(pixels+N*(w*yMin+xMin), xMax-xMin, yMax-yMin, N*w, yOrientation);
}

}
