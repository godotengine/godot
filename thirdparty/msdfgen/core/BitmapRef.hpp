
#pragma once

#include "YAxisOrientation.h"

namespace msdfgen {

/// Reference to a 2D image bitmap or a buffer acting as one. Pixel storage not owned or managed by the object.
template <typename T, int N = 1>
struct BitmapRef;
/// Constant reference to a 2D image bitmap or a buffer acting as one. Pixel storage not owned or managed by the object.
template <typename T, int N = 1>
struct BitmapConstRef;
/// Reference to a 2D image bitmap with non-contiguous rows of pixels. Pixel storage not owned or managed by the object. Can represent e.g. a section of a larger bitmap, bitmap with padded rows, or vertically flipped bitmap (rowStride can be negative).
template <typename T, int N = 1>
struct BitmapSection;
/// Constant reference to a 2D image bitmap with non-contiguous rows of pixels. Pixel storage not owned or managed by the object. Can represent e.g. a section of a larger bitmap, bitmap with padded rows, or vertically flipped bitmap (rowStride can be negative).
template <typename T, int N = 1>
struct BitmapConstSection;

template <typename T, int N>
struct BitmapRef {

    T *pixels;
    int width, height;
    YAxisOrientation yOrientation;

    inline BitmapRef() : pixels(NULL), width(0), height(0), yOrientation(MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION) { }
    inline BitmapRef(T *pixels, int width, int height, YAxisOrientation yOrientation = MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION) : pixels(pixels), width(width), height(height), yOrientation(yOrientation) { }

    inline T *operator()(int x, int y) const {
        return pixels+N*(width*y+x);
    }

    /// Returns a reference to a rectangular section of the bitmap specified by bounds (excluding xMax, yMax).
    inline BitmapSection<T, N> getSection(int xMin, int yMin, int xMax, int yMax) const {
        return BitmapSection<T, N>(pixels+N*(width*yMin+xMin), xMax-xMin, yMax-yMin, N*width, yOrientation);
    }

    /// Returns a constant reference to a rectangular section of the bitmap specified by bounds (excluding xMax, yMax).
    inline BitmapConstSection<T, N> getConstSection(int xMin, int yMin, int xMax, int yMax) const {
        return BitmapConstSection<T, N>(pixels+N*(width*yMin+xMin), xMax-xMin, yMax-yMin, N*width, yOrientation);
    }

};

template <typename T, int N>
struct BitmapConstRef {

    const T *pixels;
    int width, height;
    YAxisOrientation yOrientation;

    inline BitmapConstRef() : pixels(NULL), width(0), height(0), yOrientation(MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION) { }
    inline BitmapConstRef(const T *pixels, int width, int height, YAxisOrientation yOrientation = MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION) : pixels(pixels), width(width), height(height), yOrientation(yOrientation) { }
    inline BitmapConstRef(const BitmapRef<T, N> &orig) : pixels(orig.pixels), width(orig.width), height(orig.height), yOrientation(orig.yOrientation) { }

    inline const T *operator()(int x, int y) const {
        return pixels+N*(width*y+x);
    }

    /// Returns a constant reference to a rectangular section of the bitmap specified by bounds (excluding xMax, yMax).
    inline BitmapConstSection<T, N> getSection(int xMin, int yMin, int xMax, int yMax) const {
        return BitmapConstSection<T, N>(pixels+N*(width*yMin+xMin), xMax-xMin, yMax-yMin, N*width, yOrientation);
    }

    /// Returns a constant reference to a rectangular section of the bitmap specified by bounds (excluding xMax, yMax).
    inline BitmapConstSection<T, N> getConstSection(int xMin, int yMin, int xMax, int yMax) const {
        return getSection(xMin, yMin, xMax, yMax);
    }

};

template <typename T, int N>
struct BitmapSection {

    T *pixels;
    int width, height;
    /// Specifies the difference between the beginnings of adjacent pixel rows as the number of T elements, can be negative.
    int rowStride;
    YAxisOrientation yOrientation;

    inline BitmapSection() : pixels(NULL), width(0), height(0), rowStride(0), yOrientation(MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION) { }
    inline BitmapSection(T *pixels, int width, int height, YAxisOrientation yOrientation = MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION) : pixels(pixels), width(width), height(height), rowStride(N*width), yOrientation(yOrientation) { }
    inline BitmapSection(T *pixels, int width, int height, int rowStride, YAxisOrientation yOrientation = MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION) : pixels(pixels), width(width), height(height), rowStride(rowStride), yOrientation(yOrientation) { }
    inline BitmapSection(const BitmapRef<T, N> &orig) : pixels(orig.pixels), width(orig.width), height(orig.height), rowStride(N*orig.width), yOrientation(orig.yOrientation) { }

    inline T *operator()(int x, int y) const {
        return pixels+rowStride*y+N*x;
    }

    /// Returns a reference to a rectangular subsection of the bitmap specified by bounds (excluding xMax, yMax).
    inline BitmapSection<T, N> getSection(int xMin, int yMin, int xMax, int yMax) const {
        return BitmapSection<T, N>(pixels+rowStride*yMin+N*xMin, xMax-xMin, yMax-yMin, rowStride, yOrientation);
    }

    /// Returns a constant reference to a rectangular subsection of the bitmap specified by bounds (excluding xMax, yMax).
    inline BitmapConstSection<T, N> getConstSection(int xMin, int yMin, int xMax, int yMax) const {
        return BitmapConstSection<T, N>(pixels+rowStride*yMin+N*xMin, xMax-xMin, yMax-yMin, rowStride, yOrientation);
    }

    /// Makes sure that the section's Y-axis orientation matches the argument by potentially reordering its rows.
    inline void reorient(YAxisOrientation newYAxisOrientation) {
        if (yOrientation != newYAxisOrientation) {
            pixels += rowStride*(height-1);
            rowStride = -rowStride;
            yOrientation = newYAxisOrientation;
        }
    }

};

template <typename T, int N>
struct BitmapConstSection {

    const T *pixels;
    int width, height;
    /// Specifies the difference between the beginnings of adjacent pixel rows as the number of T elements, can be negative.
    int rowStride;
    YAxisOrientation yOrientation;

    inline BitmapConstSection() : pixels(NULL), width(0), height(0), rowStride(0), yOrientation(MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION) { }
    inline BitmapConstSection(const T *pixels, int width, int height, YAxisOrientation yOrientation = MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION) : pixels(pixels), width(width), height(height), rowStride(N*width), yOrientation(yOrientation) { }
    inline BitmapConstSection(const T *pixels, int width, int height, int rowStride, YAxisOrientation yOrientation = MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION) : pixels(pixels), width(width), height(height), rowStride(rowStride), yOrientation(yOrientation) { }
    inline BitmapConstSection(const BitmapRef<T, N> &orig) : pixels(orig.pixels), width(orig.width), height(orig.height), rowStride(N*orig.width), yOrientation(orig.yOrientation) { }
    inline BitmapConstSection(const BitmapConstRef<T, N> &orig) : pixels(orig.pixels), width(orig.width), height(orig.height), rowStride(N*orig.width), yOrientation(orig.yOrientation) { }
    inline BitmapConstSection(const BitmapSection<T, N> &orig) : pixels(orig.pixels), width(orig.width), height(orig.height), rowStride(orig.rowStride), yOrientation(orig.yOrientation) { }

    inline const T *operator()(int x, int y) const {
        return pixels+rowStride*y+N*x;
    }

    /// Returns a constant reference to a rectangular subsection of the bitmap specified by bounds (excluding xMax, yMax).
    inline BitmapConstSection<T, N> getSection(int xMin, int yMin, int xMax, int yMax) const {
        return BitmapConstSection<T, N>(pixels+rowStride*yMin+N*xMin, xMax-xMin, yMax-yMin, rowStride, yOrientation);
    }

    /// Returns a constant reference to a rectangular subsection of the bitmap specified by bounds (excluding xMax, yMax).
    inline BitmapConstSection<T, N> getConstSection(int xMin, int yMin, int xMax, int yMax) const {
        return getSection(xMin, yMin, xMax, yMax);
    }

    /// Makes sure that the section's Y-axis orientation matches the argument by potentially reordering its rows.
    inline void reorient(YAxisOrientation newYAxisOrientation) {
        if (yOrientation != newYAxisOrientation) {
            pixels += rowStride*(height-1);
            rowStride = -rowStride;
            yOrientation = newYAxisOrientation;
        }
    }

};

}
