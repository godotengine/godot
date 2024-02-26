/*
 * Copyright 2013 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/private/SkPathRef.h"

#include "include/core/SkPath.h"
#include "include/core/SkRRect.h"
#include "include/private/SkOnce.h"
#include "include/private/SkTo.h"
#include "include/private/SkVx.h"
// -- GODOT start --
//#include "src/core/SkBuffer.h"
// -- GODOT end --
#include "src/core/SkPathPriv.h"
#include "src/core/SkSafeMath.h"

//////////////////////////////////////////////////////////////////////////////
SkPathRef::Editor::Editor(sk_sp<SkPathRef>* pathRef,
                          int incReserveVerbs,
                          int incReservePoints)
{
    SkASSERT(incReserveVerbs >= 0);
    SkASSERT(incReservePoints >= 0);

    if ((*pathRef)->unique()) {
        (*pathRef)->incReserve(incReserveVerbs, incReservePoints);
    } else {
        SkPathRef* copy = new SkPathRef;
        copy->copy(**pathRef, incReserveVerbs, incReservePoints);
        pathRef->reset(copy);
    }
    fPathRef = pathRef->get();
    fPathRef->callGenIDChangeListeners();
    fPathRef->fGenerationID = 0;
    fPathRef->fBoundsIsDirty = true;
    SkDEBUGCODE(fPathRef->fEditorsAttached++;)
}

// Sort of like makeSpace(0) but the the additional requirement that we actively shrink the
// allocations to just fit the current needs. makeSpace() will only grow, but never shrinks.
//
void SkPath::shrinkToFit() {
    // Since this can relocate the allocated arrays, we have to defensively copy ourselves if
    // we're not the only owner of the pathref... since relocating the arrays will invalidate
    // any existing iterators.
    if (!fPathRef->unique()) {
        SkPathRef* pr = new SkPathRef;
        pr->copy(*fPathRef, 0, 0);
        fPathRef.reset(pr);
    }
    fPathRef->fPoints.shrinkToFit();
    fPathRef->fVerbs.shrinkToFit();
    fPathRef->fConicWeights.shrinkToFit();
    SkDEBUGCODE(fPathRef->validate();)
}

//////////////////////////////////////////////////////////////////////////////

size_t SkPathRef::approximateBytesUsed() const {
    return sizeof(SkPathRef)
         + fPoints      .reserved() * sizeof(fPoints      [0])
         + fVerbs       .reserved() * sizeof(fVerbs       [0])
         + fConicWeights.reserved() * sizeof(fConicWeights[0]);
}

SkPathRef::~SkPathRef() {
    // Deliberately don't validate() this path ref, otherwise there's no way
    // to read one that's not valid and then free its memory without asserting.
    SkDEBUGCODE(fGenerationID = 0xEEEEEEEE;)
    SkDEBUGCODE(fEditorsAttached.store(0x7777777);)
}

static SkPathRef* gEmpty = nullptr;

SkPathRef* SkPathRef::CreateEmpty() {
    static SkOnce once;
    once([]{
        gEmpty = new SkPathRef;
        gEmpty->computeBounds();   // Avoids races later to be the first to do this.
    });
    return SkRef(gEmpty);
}

static void transform_dir_and_start(const SkMatrix& matrix, bool isRRect, bool* isCCW,
                                    unsigned* start) {
    int inStart = *start;
    int rm = 0;
    if (isRRect) {
        // Degenerate rrect indices to oval indices and remember the remainder.
        // Ovals have one index per side whereas rrects have two.
        rm = inStart & 0b1;
        inStart /= 2;
    }
    // Is the antidiagonal non-zero (otherwise the diagonal is zero)
    int antiDiag;
    // Is the non-zero value in the top row (either kMScaleX or kMSkewX) negative
    int topNeg;
    // Are the two non-zero diagonal or antidiagonal values the same sign.
    int sameSign;
    if (matrix.get(SkMatrix::kMScaleX) != 0) {
        antiDiag = 0b00;
        if (matrix.get(SkMatrix::kMScaleX) > 0) {
            topNeg = 0b00;
            sameSign = matrix.get(SkMatrix::kMScaleY) > 0 ? 0b01 : 0b00;
        } else {
            topNeg = 0b10;
            sameSign = matrix.get(SkMatrix::kMScaleY) > 0 ? 0b00 : 0b01;
        }
    } else {
        antiDiag = 0b01;
        if (matrix.get(SkMatrix::kMSkewX) > 0) {
            topNeg = 0b00;
            sameSign = matrix.get(SkMatrix::kMSkewY) > 0 ? 0b01 : 0b00;
        } else {
            topNeg = 0b10;
            sameSign = matrix.get(SkMatrix::kMSkewY) > 0 ? 0b00 : 0b01;
        }
    }
    if (sameSign != antiDiag) {
        // This is a rotation (and maybe scale). The direction is unchanged.
        // Trust me on the start computation (or draw yourself some pictures)
        *start = (inStart + 4 - (topNeg | antiDiag)) % 4;
        SkASSERT(*start < 4);
        if (isRRect) {
            *start = 2 * *start + rm;
        }
    } else {
        // This is a mirror (and maybe scale). The direction is reversed.
        *isCCW = !*isCCW;
        // Trust me on the start computation (or draw yourself some pictures)
        *start = (6 + (topNeg | antiDiag) - inStart) % 4;
        SkASSERT(*start < 4);
        if (isRRect) {
            *start = 2 * *start + (rm ? 0 : 1);
        }
    }
}

void SkPathRef::CreateTransformedCopy(sk_sp<SkPathRef>* dst,
                                      const SkPathRef& src,
                                      const SkMatrix& matrix) {
    SkDEBUGCODE(src.validate();)
    if (matrix.isIdentity()) {
        if (dst->get() != &src) {
            src.ref();
            dst->reset(const_cast<SkPathRef*>(&src));
            SkDEBUGCODE((*dst)->validate();)
        }
        return;
    }

    sk_sp<const SkPathRef> srcKeepAlive;
    if (!(*dst)->unique()) {
        // If dst and src are the same then we are about to drop our only ref on the common path
        // ref. Some other thread may have owned src when we checked unique() above but it may not
        // continue to do so. Add another ref so we continue to be an owner until we're done.
        if (dst->get() == &src) {
            srcKeepAlive.reset(SkRef(&src));
        }
        dst->reset(new SkPathRef);
    }

    if (dst->get() != &src) {
        (*dst)->fVerbs = src.fVerbs;
        (*dst)->fConicWeights = src.fConicWeights;
        (*dst)->callGenIDChangeListeners();
        (*dst)->fGenerationID = 0;  // mark as dirty
        // don't copy, just allocate the points
        (*dst)->fPoints.setCount(src.fPoints.count());
    }
    matrix.mapPoints((*dst)->fPoints.begin(), src.fPoints.begin(), src.fPoints.count());

    // Need to check this here in case (&src == dst)
    bool canXformBounds = !src.fBoundsIsDirty && matrix.rectStaysRect() && src.countPoints() > 1;

    /*
     *  Here we optimize the bounds computation, by noting if the bounds are
     *  already known, and if so, we just transform those as well and mark
     *  them as "known", rather than force the transformed path to have to
     *  recompute them.
     *
     *  Special gotchas if the path is effectively empty (<= 1 point) or
     *  if it is non-finite. In those cases bounds need to stay empty,
     *  regardless of the matrix.
     */
    if (canXformBounds) {
        (*dst)->fBoundsIsDirty = false;
        if (src.fIsFinite) {
            matrix.mapRect(&(*dst)->fBounds, src.fBounds);
            if (!((*dst)->fIsFinite = (*dst)->fBounds.isFinite())) {
                (*dst)->fBounds.setEmpty();
            }
        } else {
            (*dst)->fIsFinite = false;
            (*dst)->fBounds.setEmpty();
        }
    } else {
        (*dst)->fBoundsIsDirty = true;
    }

    (*dst)->fSegmentMask = src.fSegmentMask;

    // It's an oval only if it stays a rect.
    bool rectStaysRect = matrix.rectStaysRect();
    (*dst)->fIsOval = src.fIsOval && rectStaysRect;
    (*dst)->fIsRRect = src.fIsRRect && rectStaysRect;
    if ((*dst)->fIsOval || (*dst)->fIsRRect) {
        unsigned start = src.fRRectOrOvalStartIdx;
        bool isCCW = SkToBool(src.fRRectOrOvalIsCCW);
        transform_dir_and_start(matrix, (*dst)->fIsRRect, &isCCW, &start);
        (*dst)->fRRectOrOvalIsCCW = isCCW;
        (*dst)->fRRectOrOvalStartIdx = start;
    }

    if (dst->get() == &src) {
        (*dst)->callGenIDChangeListeners();
        (*dst)->fGenerationID = 0;
    }

    SkDEBUGCODE((*dst)->validate();)
}

void SkPathRef::Rewind(sk_sp<SkPathRef>* pathRef) {
    if ((*pathRef)->unique()) {
        SkDEBUGCODE((*pathRef)->validate();)
        (*pathRef)->callGenIDChangeListeners();
        (*pathRef)->fBoundsIsDirty = true;  // this also invalidates fIsFinite
        (*pathRef)->fGenerationID = 0;
        (*pathRef)->fPoints.rewind();
        (*pathRef)->fVerbs.rewind();
        (*pathRef)->fConicWeights.rewind();
        (*pathRef)->fSegmentMask = 0;
        (*pathRef)->fIsOval = false;
        (*pathRef)->fIsRRect = false;
        SkDEBUGCODE((*pathRef)->validate();)
    } else {
        int oldVCnt = (*pathRef)->countVerbs();
        int oldPCnt = (*pathRef)->countPoints();
        pathRef->reset(new SkPathRef);
        (*pathRef)->resetToSize(0, 0, 0, oldVCnt, oldPCnt);
    }
}

bool SkPathRef::operator== (const SkPathRef& ref) const {
    SkDEBUGCODE(this->validate();)
    SkDEBUGCODE(ref.validate();)

    // We explicitly check fSegmentMask as a quick-reject. We could skip it,
    // since it is only a cache of info in the fVerbs, but its a fast way to
    // notice a difference
    if (fSegmentMask != ref.fSegmentMask) {
        return false;
    }

    bool genIDMatch = fGenerationID && fGenerationID == ref.fGenerationID;
#ifdef SK_RELEASE
    if (genIDMatch) {
        return true;
    }
#endif
    if (fPoints != ref.fPoints || fConicWeights != ref.fConicWeights || fVerbs != ref.fVerbs) {
        SkASSERT(!genIDMatch);
        return false;
    }
    if (ref.fVerbs.count() == 0) {
        SkASSERT(ref.fPoints.count() == 0);
    }
    return true;
}

// -- GODOT start --
/*
void SkPathRef::writeToBuffer(SkWBuffer* buffer) const {
    SkDEBUGCODE(this->validate();)
    SkDEBUGCODE(size_t beforePos = buffer->pos();)

    // Call getBounds() to ensure (as a side-effect) that fBounds
    // and fIsFinite are computed.
    const SkRect& bounds = this->getBounds();

    // We store fSegmentMask for older readers, but current readers can't trust it, so they
    // don't read it.
    int32_t packed = ((fIsFinite & 1) << kIsFinite_SerializationShift) |
                     (fSegmentMask << kSegmentMask_SerializationShift);
    buffer->write32(packed);

    // TODO: write gen ID here. Problem: We don't know if we're cross process or not from
    // SkWBuffer. Until this is fixed we write 0.
    buffer->write32(0);
    buffer->write32(fVerbs.count());
    buffer->write32(fPoints.count());
    buffer->write32(fConicWeights.count());
    buffer->write(fVerbs.begin(), fVerbs.bytes());
    buffer->write(fPoints.begin(), fVerbs.bytes());
    buffer->write(fConicWeights.begin(), fConicWeights.bytes());
    buffer->write(&bounds, sizeof(bounds));

    SkASSERT(buffer->pos() - beforePos == (size_t) this->writeSize());
}

uint32_t SkPathRef::writeSize() const {
    return uint32_t(5 * sizeof(uint32_t) +
                    fVerbs.bytes() + fPoints.bytes() + fConicWeights.bytes() +
                    sizeof(SkRect));
}
*/
// -- GODOT end --

void SkPathRef::copy(const SkPathRef& ref,
                     int additionalReserveVerbs,
                     int additionalReservePoints) {
    SkDEBUGCODE(this->validate();)
    this->resetToSize(ref.fVerbs.count(), ref.fPoints.count(), ref.fConicWeights.count(),
                      additionalReserveVerbs, additionalReservePoints);
    fVerbs = ref.fVerbs;
    fPoints = ref.fPoints;
    fConicWeights = ref.fConicWeights;
    fBoundsIsDirty = ref.fBoundsIsDirty;
    if (!fBoundsIsDirty) {
        fBounds = ref.fBounds;
        fIsFinite = ref.fIsFinite;
    }
    fSegmentMask = ref.fSegmentMask;
    fIsOval = ref.fIsOval;
    fIsRRect = ref.fIsRRect;
    fRRectOrOvalIsCCW = ref.fRRectOrOvalIsCCW;
    fRRectOrOvalStartIdx = ref.fRRectOrOvalStartIdx;
    SkDEBUGCODE(this->validate();)
}

void SkPathRef::interpolate(const SkPathRef& ending, SkScalar weight, SkPathRef* out) const {
    const SkScalar* inValues = &ending.getPoints()->fX;
    SkScalar* outValues = &out->getWritablePoints()->fX;
    int count = out->countPoints() * 2;
    for (int index = 0; index < count; ++index) {
        outValues[index] = outValues[index] * weight + inValues[index] * (1 - weight);
    }
    out->fBoundsIsDirty = true;
    out->fIsOval = false;
    out->fIsRRect = false;
}

std::tuple<SkPoint*, SkScalar*> SkPathRef::growForVerbsInPath(const SkPathRef& path) {
    SkDEBUGCODE(this->validate();)

    fSegmentMask |= path.fSegmentMask;
    fBoundsIsDirty = true;  // this also invalidates fIsFinite
    fIsOval = false;
    fIsRRect = false;

    if (int numVerbs = path.countVerbs()) {
        memcpy(fVerbs.append(numVerbs), path.fVerbs.begin(), numVerbs * sizeof(fVerbs[0]));
    }

    SkPoint* pts = nullptr;
    if (int numPts = path.countPoints()) {
        pts = fPoints.append(numPts);
    }

    SkScalar* weights = nullptr;
    if (int numConics = path.countWeights()) {
        weights = fConicWeights.append(numConics);
    }

    SkDEBUGCODE(this->validate();)
    return {pts, weights};
}

SkPoint* SkPathRef::growForRepeatedVerb(int /*SkPath::Verb*/ verb,
                                        int numVbs,
                                        SkScalar** weights) {
    SkDEBUGCODE(this->validate();)
    int pCnt;
    switch (verb) {
        case SkPath::kMove_Verb:
            pCnt = numVbs;
            break;
        case SkPath::kLine_Verb:
            fSegmentMask |= SkPath::kLine_SegmentMask;
            pCnt = numVbs;
            break;
        case SkPath::kQuad_Verb:
            fSegmentMask |= SkPath::kQuad_SegmentMask;
            pCnt = 2 * numVbs;
            break;
        case SkPath::kConic_Verb:
            fSegmentMask |= SkPath::kConic_SegmentMask;
            pCnt = 2 * numVbs;
            break;
        case SkPath::kCubic_Verb:
            fSegmentMask |= SkPath::kCubic_SegmentMask;
            pCnt = 3 * numVbs;
            break;
        case SkPath::kClose_Verb:
            SkDEBUGFAIL("growForRepeatedVerb called for kClose_Verb");
            pCnt = 0;
            break;
        case SkPath::kDone_Verb:
            SkDEBUGFAIL("growForRepeatedVerb called for kDone");
            pCnt = 0;
            break;
        default:
            SkDEBUGFAIL("default should not be reached");
            pCnt = 0;
            break;
    }

    fBoundsIsDirty = true;  // this also invalidates fIsFinite
    fIsOval = false;
    fIsRRect = false;

    memset(fVerbs.append(numVbs), verb, numVbs);
    if (SkPath::kConic_Verb == verb) {
        SkASSERT(weights);
        *weights = fConicWeights.append(numVbs);
    }
    SkPoint* pts = fPoints.append(pCnt);

    SkDEBUGCODE(this->validate();)
    return pts;
}

SkPoint* SkPathRef::growForVerb(int /* SkPath::Verb*/ verb, SkScalar weight) {
    SkDEBUGCODE(this->validate();)
    int pCnt;
    unsigned mask = 0;
    switch (verb) {
        case SkPath::kMove_Verb:
            pCnt = 1;
            break;
        case SkPath::kLine_Verb:
            mask = SkPath::kLine_SegmentMask;
            pCnt = 1;
            break;
        case SkPath::kQuad_Verb:
            mask = SkPath::kQuad_SegmentMask;
            pCnt = 2;
            break;
        case SkPath::kConic_Verb:
            mask = SkPath::kConic_SegmentMask;
            pCnt = 2;
            break;
        case SkPath::kCubic_Verb:
            mask = SkPath::kCubic_SegmentMask;
            pCnt = 3;
            break;
        case SkPath::kClose_Verb:
            pCnt = 0;
            break;
        case SkPath::kDone_Verb:
            SkDEBUGFAIL("growForVerb called for kDone");
            pCnt = 0;
            break;
        default:
            SkDEBUGFAIL("default is not reached");
            pCnt = 0;
            break;
    }

    fSegmentMask |= mask;
    fBoundsIsDirty = true;  // this also invalidates fIsFinite
    fIsOval = false;
    fIsRRect = false;

    *fVerbs.append() = verb;
    if (SkPath::kConic_Verb == verb) {
        *fConicWeights.append() = weight;
    }
    SkPoint* pts = fPoints.append(pCnt);

    SkDEBUGCODE(this->validate();)
    return pts;
}

uint32_t SkPathRef::genID() const {
    SkASSERT(fEditorsAttached.load() == 0);
    static const uint32_t kMask = (static_cast<int64_t>(1) << SkPathPriv::kPathRefGenIDBitCnt) - 1;

    if (fGenerationID == 0) {
        if (fPoints.count() == 0 && fVerbs.count() == 0) {
            fGenerationID = kEmptyGenID;
        } else {
            static std::atomic<uint32_t> nextID{kEmptyGenID + 1};
            do {
                fGenerationID = nextID.fetch_add(1, std::memory_order_relaxed) & kMask;
            } while (fGenerationID == 0 || fGenerationID == kEmptyGenID);
        }
    }
    return fGenerationID;
}

void SkPathRef::addGenIDChangeListener(sk_sp<SkIDChangeListener> listener) {
    if (this == gEmpty) {
        return;
    }
    fGenIDChangeListeners.add(std::move(listener));
}

int SkPathRef::genIDChangeListenerCount() { return fGenIDChangeListeners.count(); }

// we need to be called *before* the genID gets changed or zerod
void SkPathRef::callGenIDChangeListeners() {
    fGenIDChangeListeners.changed();
}

SkRRect SkPathRef::getRRect() const {
    const SkRect& bounds = this->getBounds();
    SkVector radii[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    Iter iter(*this);
    SkPoint pts[4];
    uint8_t verb = iter.next(pts);
    SkASSERT(SkPath::kMove_Verb == verb);
    while ((verb = iter.next(pts)) != SkPath::kDone_Verb) {
        if (SkPath::kConic_Verb == verb) {
            SkVector v1_0 = pts[1] - pts[0];
            SkVector v2_1 = pts[2] - pts[1];
            SkVector dxdy;
            if (v1_0.fX) {
                SkASSERT(!v2_1.fX && !v1_0.fY);
                dxdy.set(SkScalarAbs(v1_0.fX), SkScalarAbs(v2_1.fY));
            } else if (!v1_0.fY) {
                SkASSERT(!v2_1.fX || !v2_1.fY);
                dxdy.set(SkScalarAbs(v2_1.fX), SkScalarAbs(v2_1.fY));
            } else {
                SkASSERT(!v2_1.fY);
                dxdy.set(SkScalarAbs(v2_1.fX), SkScalarAbs(v1_0.fY));
            }
            SkRRect::Corner corner =
                    pts[1].fX == bounds.fLeft ?
                        pts[1].fY == bounds.fTop ?
                            SkRRect::kUpperLeft_Corner : SkRRect::kLowerLeft_Corner :
                    pts[1].fY == bounds.fTop ?
                            SkRRect::kUpperRight_Corner : SkRRect::kLowerRight_Corner;
            SkASSERT(!radii[corner].fX && !radii[corner].fY);
            radii[corner] = dxdy;
        } else {
            SkASSERT((verb == SkPath::kLine_Verb
                    && (!(pts[1].fX - pts[0].fX) || !(pts[1].fY - pts[0].fY)))
                    || verb == SkPath::kClose_Verb);
        }
    }
    SkRRect rrect;
    rrect.setRectRadii(bounds, radii);
    return rrect;
}

bool SkPathRef::isRRect(SkRRect* rrect, bool* isCCW, unsigned* start) const {
        if (fIsRRect) {
            if (rrect) {
                *rrect = this->getRRect();
            }
            if (isCCW) {
                *isCCW = SkToBool(fRRectOrOvalIsCCW);
            }
            if (start) {
                *start = fRRectOrOvalStartIdx;
            }
        }
        return SkToBool(fIsRRect);
    }

///////////////////////////////////////////////////////////////////////////////

SkPathRef::Iter::Iter() {
#ifdef SK_DEBUG
    fPts = nullptr;
    fConicWeights = nullptr;
#endif
    // need to init enough to make next() harmlessly return kDone_Verb
    fVerbs = nullptr;
    fVerbStop = nullptr;
}

SkPathRef::Iter::Iter(const SkPathRef& path) {
    this->setPathRef(path);
}

void SkPathRef::Iter::setPathRef(const SkPathRef& path) {
    fPts = path.points();
    fVerbs = path.verbsBegin();
    fVerbStop = path.verbsEnd();
    fConicWeights = path.conicWeights();
    if (fConicWeights) {
        fConicWeights -= 1;  // begin one behind
    }

    // Don't allow iteration through non-finite points.
    if (!path.isFinite()) {
        fVerbStop = fVerbs;
    }
}

uint8_t SkPathRef::Iter::next(SkPoint pts[4]) {
    SkASSERT(pts);

    SkDEBUGCODE(unsigned peekResult = this->peek();)

    if (fVerbs == fVerbStop) {
        SkASSERT(peekResult == SkPath::kDone_Verb);
        return (uint8_t) SkPath::kDone_Verb;
    }

    // fVerbs points one beyond next verb so decrement first.
    unsigned verb = *fVerbs++;
    const SkPoint* srcPts = fPts;

    switch (verb) {
        case SkPath::kMove_Verb:
            pts[0] = srcPts[0];
            srcPts += 1;
            break;
        case SkPath::kLine_Verb:
            pts[0] = srcPts[-1];
            pts[1] = srcPts[0];
            srcPts += 1;
            break;
        case SkPath::kConic_Verb:
            fConicWeights += 1;
            [[fallthrough]];
        case SkPath::kQuad_Verb:
            pts[0] = srcPts[-1];
            pts[1] = srcPts[0];
            pts[2] = srcPts[1];
            srcPts += 2;
            break;
        case SkPath::kCubic_Verb:
            pts[0] = srcPts[-1];
            pts[1] = srcPts[0];
            pts[2] = srcPts[1];
            pts[3] = srcPts[2];
            srcPts += 3;
            break;
        case SkPath::kClose_Verb:
            break;
        case SkPath::kDone_Verb:
            SkASSERT(fVerbs == fVerbStop);
            break;
    }
    fPts = srcPts;
    SkASSERT(peekResult == verb);
    return (uint8_t) verb;
}

uint8_t SkPathRef::Iter::peek() const {
    return fVerbs < fVerbStop ? *fVerbs : (uint8_t) SkPath::kDone_Verb;
}


bool SkPathRef::isValid() const {
    if (fIsOval || fIsRRect) {
        // Currently we don't allow both of these to be set, even though ovals are ro
        if (fIsOval == fIsRRect) {
            return false;
        }
        if (fIsOval) {
            if (fRRectOrOvalStartIdx >= 4) {
                return false;
            }
        } else {
            if (fRRectOrOvalStartIdx >= 8) {
                return false;
            }
        }
    }

    if (!fBoundsIsDirty && !fBounds.isEmpty()) {
        bool isFinite = true;
        auto leftTop = skvx::float2(fBounds.fLeft, fBounds.fTop);
        auto rightBot = skvx::float2(fBounds.fRight, fBounds.fBottom);
        for (int i = 0; i < fPoints.count(); ++i) {
            auto point = skvx::float2(fPoints[i].fX, fPoints[i].fY);
#ifdef SK_DEBUG
            if (fPoints[i].isFinite() && (any(point < leftTop)|| any(point > rightBot))) {
                SkDebugf("bad SkPathRef bounds: %g %g %g %g\n",
                         fBounds.fLeft, fBounds.fTop, fBounds.fRight, fBounds.fBottom);
                for (int j = 0; j < fPoints.count(); ++j) {
                    if (i == j) {
                        SkDebugf("*** bounds do not contain: ");
                    }
                    SkDebugf("%g %g\n", fPoints[j].fX, fPoints[j].fY);
                }
                return false;
            }
#endif

            if (fPoints[i].isFinite() && any(point < leftTop) && !any(point > rightBot))
                return false;
            if (!fPoints[i].isFinite()) {
                isFinite = false;
            }
        }
        if (SkToBool(fIsFinite) != isFinite) {
            return false;
        }
    }
    return true;
}

bool SkPathRef::dataMatchesVerbs() const {
    const auto info = sk_path_analyze_verbs(fVerbs.begin(), fVerbs.count());
    return info.valid                          &&
           info.segmentMask == fSegmentMask    &&
           info.points      == fPoints.count() &&
           info.weights     == fConicWeights.count();
}
//////////////////////////////////////////////////////////////////////////////////////////////////

SkPathEdgeIter::SkPathEdgeIter(const SkPath& path) {
    fMoveToPtr = fPts = path.fPathRef->points();
    fVerbs = path.fPathRef->verbsBegin();
    fVerbsStop = path.fPathRef->verbsEnd();
    fConicWeights = path.fPathRef->conicWeights();
    if (fConicWeights) {
        fConicWeights -= 1;  // begin one behind
    }

    fNeedsCloseLine = false;
    fNextIsNewContour = false;
    SkDEBUGCODE(fIsConic = false;)
}
