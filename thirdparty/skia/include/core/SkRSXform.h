/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkRSXform_DEFINED
#define SkRSXform_DEFINED

#include "include/core/SkPoint.h"
#include "include/core/SkSize.h"

/**
 *  A compressed form of a rotation+scale matrix.
 *
 *  [ fSCos     -fSSin    fTx ]
 *  [ fSSin      fSCos    fTy ]
 *  [     0          0      1 ]
 */
struct SK_API SkRSXform {
    static SkRSXform Make(SkScalar scos, SkScalar ssin, SkScalar tx, SkScalar ty) {
        SkRSXform xform = { scos, ssin, tx, ty };
        return xform;
    }

    /*
     *  Initialize a new xform based on the scale, rotation (in radians), final tx,ty location
     *  and anchor-point ax,ay within the src quad.
     *
     *  Note: the anchor point is not normalized (e.g. 0...1) but is in pixels of the src image.
     */
    static SkRSXform MakeFromRadians(SkScalar scale, SkScalar radians, SkScalar tx, SkScalar ty,
                                     SkScalar ax, SkScalar ay) {
        const SkScalar s = SkScalarSin(radians) * scale;
        const SkScalar c = SkScalarCos(radians) * scale;
        return Make(c, s, tx + -c * ax + s * ay, ty + -s * ax - c * ay);
    }

    SkScalar fSCos;
    SkScalar fSSin;
    SkScalar fTx;
    SkScalar fTy;

    bool rectStaysRect() const {
        return 0 == fSCos || 0 == fSSin;
    }

    void setIdentity() {
        fSCos = 1;
        fSSin = fTx = fTy = 0;
    }

    void set(SkScalar scos, SkScalar ssin, SkScalar tx, SkScalar ty) {
        fSCos = scos;
        fSSin = ssin;
        fTx = tx;
        fTy = ty;
    }

    void toQuad(SkScalar width, SkScalar height, SkPoint quad[4]) const;
    void toQuad(const SkSize& size, SkPoint quad[4]) const {
        this->toQuad(size.width(), size.height(), quad);
    }
    void toTriStrip(SkScalar width, SkScalar height, SkPoint strip[4]) const;
};

#endif

