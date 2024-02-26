/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkRRect_DEFINED
#define SkRRect_DEFINED

#include "include/core/SkPoint.h"
#include "include/core/SkRect.h"

class SkPath;
class SkMatrix;
class SkString;

/** \class SkRRect
    SkRRect describes a rounded rectangle with a bounds and a pair of radii for each corner.
    The bounds and radii can be set so that SkRRect describes: a rectangle with sharp corners;
    a circle; an oval; or a rectangle with one or more rounded corners.

    SkRRect allows implementing CSS properties that describe rounded corners.
    SkRRect may have up to eight different radii, one for each axis on each of its four
    corners.

    SkRRect may modify the provided parameters when initializing bounds and radii.
    If either axis radii is zero or less: radii are stored as zero; corner is square.
    If corner curves overlap, radii are proportionally reduced to fit within bounds.
*/
class SK_API SkRRect {
public:

    /** Initializes bounds at (0, 0), the origin, with zero width and height.
        Initializes corner radii to (0, 0), and sets type of kEmpty_Type.

        @return  empty SkRRect
    */
    SkRRect() = default;

    /** Initializes to copy of rrect bounds and corner radii.

        @param rrect  bounds and corner to copy
        @return       copy of rrect
    */
    SkRRect(const SkRRect& rrect) = default;

    /** Copies rrect bounds and corner radii.

        @param rrect  bounds and corner to copy
        @return       copy of rrect
    */
    SkRRect& operator=(const SkRRect& rrect) = default;

    /** \enum SkRRect::Type
        Type describes possible specializations of SkRRect. Each Type is
        exclusive; a SkRRect may only have one type.

        Type members become progressively less restrictive; larger values of
        Type have more degrees of freedom than smaller values.
    */
    enum Type {
        kEmpty_Type,                     //!< zero width or height
        kRect_Type,                      //!< non-zero width and height, and zeroed radii
        kOval_Type,                      //!< non-zero width and height filled with radii
        kSimple_Type,                    //!< non-zero width and height with equal radii
        kNinePatch_Type,                 //!< non-zero width and height with axis-aligned radii
        kComplex_Type,                   //!< non-zero width and height with arbitrary radii
        kLastType       = kComplex_Type, //!< largest Type value
    };

    Type getType() const {
        SkASSERT(this->isValid());
        return static_cast<Type>(fType);
    }

    Type type() const { return this->getType(); }

    inline bool isEmpty() const { return kEmpty_Type == this->getType(); }
    inline bool isRect() const { return kRect_Type == this->getType(); }
    inline bool isOval() const { return kOval_Type == this->getType(); }
    inline bool isSimple() const { return kSimple_Type == this->getType(); }
    inline bool isNinePatch() const { return kNinePatch_Type == this->getType(); }
    inline bool isComplex() const { return kComplex_Type == this->getType(); }

    /** Returns span on the x-axis. This does not check if result fits in 32-bit float;
        result may be infinity.

        @return  rect().fRight minus rect().fLeft
    */
    SkScalar width() const { return fRect.width(); }

    /** Returns span on the y-axis. This does not check if result fits in 32-bit float;
        result may be infinity.

        @return  rect().fBottom minus rect().fTop
    */
    SkScalar height() const { return fRect.height(); }

    /** Returns top-left corner radii. If type() returns kEmpty_Type, kRect_Type,
        kOval_Type, or kSimple_Type, returns a value representative of all corner radii.
        If type() returns kNinePatch_Type or kComplex_Type, at least one of the
        remaining three corners has a different value.

        @return  corner radii for simple types
    */
    SkVector getSimpleRadii() const {
        return fRadii[0];
    }

    /** Sets bounds to zero width and height at (0, 0), the origin. Sets
        corner radii to zero and sets type to kEmpty_Type.
    */
    void setEmpty() { *this = SkRRect(); }

    /** Sets bounds to sorted rect, and sets corner radii to zero.
        If set bounds has width and height, and sets type to kRect_Type;
        otherwise, sets type to kEmpty_Type.

        @param rect  bounds to set
    */
    void setRect(const SkRect& rect) {
        if (!this->initializeRect(rect)) {
            return;
        }

        memset(fRadii, 0, sizeof(fRadii));
        fType = kRect_Type;

        SkASSERT(this->isValid());
    }

    /** Initializes bounds at (0, 0), the origin, with zero width and height.
        Initializes corner radii to (0, 0), and sets type of kEmpty_Type.

        @return  empty SkRRect
    */
    static SkRRect MakeEmpty() { return SkRRect(); }

    /** Initializes to copy of r bounds and zeroes corner radii.

        @param r  bounds to copy
        @return   copy of r
    */
    static SkRRect MakeRect(const SkRect& r) {
        SkRRect rr;
        rr.setRect(r);
        return rr;
    }

    /** Sets bounds to oval, x-axis radii to half oval.width(), and all y-axis radii
        to half oval.height(). If oval bounds is empty, sets to kEmpty_Type.
        Otherwise, sets to kOval_Type.

        @param oval  bounds of oval
        @return      oval
    */
    static SkRRect MakeOval(const SkRect& oval) {
        SkRRect rr;
        rr.setOval(oval);
        return rr;
    }

    /** Sets to rounded rectangle with the same radii for all four corners.
        If rect is empty, sets to kEmpty_Type.
        Otherwise, if xRad and yRad are zero, sets to kRect_Type.
        Otherwise, if xRad is at least half rect.width() and yRad is at least half
        rect.height(), sets to kOval_Type.
        Otherwise, sets to kSimple_Type.

        @param rect  bounds of rounded rectangle
        @param xRad  x-axis radius of corners
        @param yRad  y-axis radius of corners
        @return      rounded rectangle
    */
    static SkRRect MakeRectXY(const SkRect& rect, SkScalar xRad, SkScalar yRad) {
        SkRRect rr;
        rr.setRectXY(rect, xRad, yRad);
        return rr;
    }

    /** Sets bounds to oval, x-axis radii to half oval.width(), and all y-axis radii
        to half oval.height(). If oval bounds is empty, sets to kEmpty_Type.
        Otherwise, sets to kOval_Type.

        @param oval  bounds of oval
    */
    void setOval(const SkRect& oval);

    /** Sets to rounded rectangle with the same radii for all four corners.
        If rect is empty, sets to kEmpty_Type.
        Otherwise, if xRad or yRad is zero, sets to kRect_Type.
        Otherwise, if xRad is at least half rect.width() and yRad is at least half
        rect.height(), sets to kOval_Type.
        Otherwise, sets to kSimple_Type.

        @param rect  bounds of rounded rectangle
        @param xRad  x-axis radius of corners
        @param yRad  y-axis radius of corners

        example: https://fiddle.skia.org/c/@RRect_setRectXY
    */
    void setRectXY(const SkRect& rect, SkScalar xRad, SkScalar yRad);

    /** Sets bounds to rect. Sets radii to (leftRad, topRad), (rightRad, topRad),
        (rightRad, bottomRad), (leftRad, bottomRad).

        If rect is empty, sets to kEmpty_Type.
        Otherwise, if leftRad and rightRad are zero, sets to kRect_Type.
        Otherwise, if topRad and bottomRad are zero, sets to kRect_Type.
        Otherwise, if leftRad and rightRad are equal and at least half rect.width(), and
        topRad and bottomRad are equal at least half rect.height(), sets to kOval_Type.
        Otherwise, if leftRad and rightRad are equal, and topRad and bottomRad are equal,
        sets to kSimple_Type. Otherwise, sets to kNinePatch_Type.

        Nine patch refers to the nine parts defined by the radii: one center rectangle,
        four edge patches, and four corner patches.

        @param rect       bounds of rounded rectangle
        @param leftRad    left-top and left-bottom x-axis radius
        @param topRad     left-top and right-top y-axis radius
        @param rightRad   right-top and right-bottom x-axis radius
        @param bottomRad  left-bottom and right-bottom y-axis radius
    */
    void setNinePatch(const SkRect& rect, SkScalar leftRad, SkScalar topRad,
                      SkScalar rightRad, SkScalar bottomRad);

    /** Sets bounds to rect. Sets radii array for individual control of all for corners.

        If rect is empty, sets to kEmpty_Type.
        Otherwise, if one of each corner radii are zero, sets to kRect_Type.
        Otherwise, if all x-axis radii are equal and at least half rect.width(), and
        all y-axis radii are equal at least half rect.height(), sets to kOval_Type.
        Otherwise, if all x-axis radii are equal, and all y-axis radii are equal,
        sets to kSimple_Type. Otherwise, sets to kNinePatch_Type.

        @param rect   bounds of rounded rectangle
        @param radii  corner x-axis and y-axis radii

        example: https://fiddle.skia.org/c/@RRect_setRectRadii
    */
    void setRectRadii(const SkRect& rect, const SkVector radii[4]);

    /** \enum SkRRect::Corner
        The radii are stored: top-left, top-right, bottom-right, bottom-left.
    */
    enum Corner {
        kUpperLeft_Corner,  //!< index of top-left corner radii
        kUpperRight_Corner, //!< index of top-right corner radii
        kLowerRight_Corner, //!< index of bottom-right corner radii
        kLowerLeft_Corner,  //!< index of bottom-left corner radii
    };

    /** Returns bounds. Bounds may have zero width or zero height. Bounds right is
        greater than or equal to left; bounds bottom is greater than or equal to top.
        Result is identical to getBounds().

        @return  bounding box
    */
    const SkRect& rect() const { return fRect; }

    /** Returns scalar pair for radius of curve on x-axis and y-axis for one corner.
        Both radii may be zero. If not zero, both are positive and finite.

        @return        x-axis and y-axis radii for one corner
    */
    SkVector radii(Corner corner) const { return fRadii[corner]; }

    /** Returns bounds. Bounds may have zero width or zero height. Bounds right is
        greater than or equal to left; bounds bottom is greater than or equal to top.
        Result is identical to rect().

        @return  bounding box
    */
    const SkRect& getBounds() const { return fRect; }

    /** Returns true if bounds and radii in a are equal to bounds and radii in b.

        a and b are not equal if either contain NaN. a and b are equal if members
        contain zeroes with different signs.

        @param a  SkRect bounds and radii to compare
        @param b  SkRect bounds and radii to compare
        @return   true if members are equal
    */
    friend bool operator==(const SkRRect& a, const SkRRect& b) {
        return a.fRect == b.fRect && SkScalarsEqual(&a.fRadii[0].fX, &b.fRadii[0].fX, 8);
    }

    /** Returns true if bounds and radii in a are not equal to bounds and radii in b.

        a and b are not equal if either contain NaN. a and b are equal if members
        contain zeroes with different signs.

        @param a  SkRect bounds and radii to compare
        @param b  SkRect bounds and radii to compare
        @return   true if members are not equal
    */
    friend bool operator!=(const SkRRect& a, const SkRRect& b) {
        return a.fRect != b.fRect || !SkScalarsEqual(&a.fRadii[0].fX, &b.fRadii[0].fX, 8);
    }

    /** Copies SkRRect to dst, then insets dst bounds by dx and dy, and adjusts dst
        radii by dx and dy. dx and dy may be positive, negative, or zero. dst may be
        SkRRect.

        If either corner radius is zero, the corner has no curvature and is unchanged.
        Otherwise, if adjusted radius becomes negative, pins radius to zero.
        If dx exceeds half dst bounds width, dst bounds left and right are set to
        bounds x-axis center. If dy exceeds half dst bounds height, dst bounds top and
        bottom are set to bounds y-axis center.

        If dx or dy cause the bounds to become infinite, dst bounds is zeroed.

        @param dx   added to rect().fLeft, and subtracted from rect().fRight
        @param dy   added to rect().fTop, and subtracted from rect().fBottom
        @param dst  insets bounds and radii

        example: https://fiddle.skia.org/c/@RRect_inset
    */
    void inset(SkScalar dx, SkScalar dy, SkRRect* dst) const;

    /** Insets bounds by dx and dy, and adjusts radii by dx and dy. dx and dy may be
        positive, negative, or zero.

        If either corner radius is zero, the corner has no curvature and is unchanged.
        Otherwise, if adjusted radius becomes negative, pins radius to zero.
        If dx exceeds half bounds width, bounds left and right are set to
        bounds x-axis center. If dy exceeds half bounds height, bounds top and
        bottom are set to bounds y-axis center.

        If dx or dy cause the bounds to become infinite, bounds is zeroed.

        @param dx  added to rect().fLeft, and subtracted from rect().fRight
        @param dy  added to rect().fTop, and subtracted from rect().fBottom
    */
    void inset(SkScalar dx, SkScalar dy) {
        this->inset(dx, dy, this);
    }

    /** Outsets dst bounds by dx and dy, and adjusts radii by dx and dy. dx and dy may be
        positive, negative, or zero.

        If either corner radius is zero, the corner has no curvature and is unchanged.
        Otherwise, if adjusted radius becomes negative, pins radius to zero.
        If dx exceeds half dst bounds width, dst bounds left and right are set to
        bounds x-axis center. If dy exceeds half dst bounds height, dst bounds top and
        bottom are set to bounds y-axis center.

        If dx or dy cause the bounds to become infinite, dst bounds is zeroed.

        @param dx   subtracted from rect().fLeft, and added to rect().fRight
        @param dy   subtracted from rect().fTop, and added to rect().fBottom
        @param dst  outset bounds and radii
    */
    void outset(SkScalar dx, SkScalar dy, SkRRect* dst) const {
        this->inset(-dx, -dy, dst);
    }

    /** Outsets bounds by dx and dy, and adjusts radii by dx and dy. dx and dy may be
        positive, negative, or zero.

        If either corner radius is zero, the corner has no curvature and is unchanged.
        Otherwise, if adjusted radius becomes negative, pins radius to zero.
        If dx exceeds half bounds width, bounds left and right are set to
        bounds x-axis center. If dy exceeds half bounds height, bounds top and
        bottom are set to bounds y-axis center.

        If dx or dy cause the bounds to become infinite, bounds is zeroed.

        @param dx  subtracted from rect().fLeft, and added to rect().fRight
        @param dy  subtracted from rect().fTop, and added to rect().fBottom
    */
    void outset(SkScalar dx, SkScalar dy) {
        this->inset(-dx, -dy, this);
    }

    /** Translates SkRRect by (dx, dy).

        @param dx  offset added to rect().fLeft and rect().fRight
        @param dy  offset added to rect().fTop and rect().fBottom
    */
    void offset(SkScalar dx, SkScalar dy) {
        fRect.offset(dx, dy);
    }

    /** Returns SkRRect translated by (dx, dy).

        @param dx  offset added to rect().fLeft and rect().fRight
        @param dy  offset added to rect().fTop and rect().fBottom
        @return    SkRRect bounds offset by (dx, dy), with unchanged corner radii
    */
    SkRRect SK_WARN_UNUSED_RESULT makeOffset(SkScalar dx, SkScalar dy) const {
        return SkRRect(fRect.makeOffset(dx, dy), fRadii, fType);
    }

    /** Returns true if rect is inside the bounds and corner radii, and if
        SkRRect and rect are not empty.

        @param rect  area tested for containment
        @return      true if SkRRect contains rect

        example: https://fiddle.skia.org/c/@RRect_contains
    */
    bool contains(const SkRect& rect) const;

    /** Returns true if bounds and radii values are finite and describe a SkRRect
        SkRRect::Type that matches getType(). All SkRRect methods construct valid types,
        even if the input values are not valid. Invalid SkRRect data can only
        be generated by corrupting memory.

        @return  true if bounds and radii match type()

        example: https://fiddle.skia.org/c/@RRect_isValid
    */
    bool isValid() const;

    static constexpr size_t kSizeInMemory = 12 * sizeof(SkScalar);

    /** Writes SkRRect to buffer. Writes kSizeInMemory bytes, and returns
        kSizeInMemory, the number of bytes written.

        @param buffer  storage for SkRRect
        @return        bytes written, kSizeInMemory

        example: https://fiddle.skia.org/c/@RRect_writeToMemory
    */
    size_t writeToMemory(void* buffer) const;

    /** Reads SkRRect from buffer, reading kSizeInMemory bytes.
        Returns kSizeInMemory, bytes read if length is at least kSizeInMemory.
        Otherwise, returns zero.

        @param buffer  memory to read from
        @param length  size of buffer
        @return        bytes read, or 0 if length is less than kSizeInMemory

        example: https://fiddle.skia.org/c/@RRect_readFromMemory
    */
    size_t readFromMemory(const void* buffer, size_t length);

    /** Transforms by SkRRect by matrix, storing result in dst.
        Returns true if SkRRect transformed can be represented by another SkRRect.
        Returns false if matrix contains transformations that are not axis aligned.

        Asserts in debug builds if SkRRect equals dst.

        @param matrix  SkMatrix specifying the transform
        @param dst     SkRRect to store the result
        @return        true if transformation succeeded.

        example: https://fiddle.skia.org/c/@RRect_transform
    */
    bool transform(const SkMatrix& matrix, SkRRect* dst) const;

    /** Writes text representation of SkRRect to standard output.
        Set asHex true to generate exact binary representations
        of floating point numbers.

        @param asHex  true if SkScalar values are written as hexadecimal

        example: https://fiddle.skia.org/c/@RRect_dump
    */
// -- GODOT start --
    //void dump(bool asHex) const;
    //SkString dumpToString(bool asHex) const;
// -- GODOT end --

    /** Writes text representation of SkRRect to standard output. The representation
        may be directly compiled as C++ code. Floating point values are written
        with limited precision; it may not be possible to reconstruct original
        SkRRect from output.
    */
// -- GODOT start --
    //void dump() const { this->dump(false); }
// -- GODOT end --

    /** Writes text representation of SkRRect to standard output. The representation
        may be directly compiled as C++ code. Floating point values are written
        in hexadecimal to preserve their exact bit pattern. The output reconstructs the
        original SkRRect.
    */
// -- GODOT start --
    //void dumpHex() const { this->dump(true); }
// -- GODOT end --
private:
    static bool AreRectAndRadiiValid(const SkRect&, const SkVector[4]);

    SkRRect(const SkRect& rect, const SkVector radii[4], int32_t type)
        : fRect(rect)
        , fRadii{radii[0], radii[1], radii[2], radii[3]}
        , fType(type) {}

    /**
     * Initializes fRect. If the passed in rect is not finite or empty the rrect will be fully
     * initialized and false is returned. Otherwise, just fRect is initialized and true is returned.
     */
    bool initializeRect(const SkRect&);

    void computeType();
    bool checkCornerContainment(SkScalar x, SkScalar y) const;
    // Returns true if the radii had to be scaled to fit rect
    bool scaleRadii();

    SkRect fRect = SkRect::MakeEmpty();
    // Radii order is UL, UR, LR, LL. Use Corner enum to index into fRadii[]
    SkVector fRadii[4] = {{0, 0}, {0, 0}, {0,0}, {0,0}};
    // use an explicitly sized type so we're sure the class is dense (no uninitialized bytes)
    int32_t fType = kEmpty_Type;
    // TODO: add padding so we can use memcpy for flattening and not copy uninitialized data

    // to access fRadii directly
    friend class SkPath;
    friend class SkRRectPriv;
};

#endif
