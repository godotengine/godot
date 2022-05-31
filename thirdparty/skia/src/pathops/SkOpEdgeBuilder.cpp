/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/core/SkGeometry.h"
#include "src/core/SkPathPriv.h"
#include "src/core/SkTSort.h"
#include "src/pathops/SkOpEdgeBuilder.h"
#include "src/pathops/SkReduceOrder.h"

void SkOpEdgeBuilder::init() {
    fOperand = false;
    fXorMask[0] = fXorMask[1] = ((int)fPath->getFillType() & 1) ? kEvenOdd_PathOpsMask
            : kWinding_PathOpsMask;
    fUnparseable = false;
    fSecondHalf = preFetch();
}

// very tiny points cause numerical instability : don't allow them
static SkPoint force_small_to_zero(const SkPoint& pt) {
    SkPoint ret = pt;
    if (SkScalarAbs(ret.fX) < FLT_EPSILON_ORDERABLE_ERR) {
        ret.fX = 0;
    }
    if (SkScalarAbs(ret.fY) < FLT_EPSILON_ORDERABLE_ERR) {
        ret.fY = 0;
    }
    return ret;
}

static bool can_add_curve(SkPath::Verb verb, SkPoint* curve) {
    if (SkPath::kMove_Verb == verb) {
        return false;
    }
    for (int index = 0; index <= SkPathOpsVerbToPoints(verb); ++index) {
        curve[index] = force_small_to_zero(curve[index]);
    }
    return SkPath::kLine_Verb != verb || !SkDPoint::ApproximatelyEqual(curve[0], curve[1]);
}

void SkOpEdgeBuilder::addOperand(const SkPath& path) {
    SkASSERT(fPathVerbs.count() > 0 && fPathVerbs.end()[-1] == SkPath::kDone_Verb);
    fPathVerbs.pop();
    fPath = &path;
    fXorMask[1] = ((int)fPath->getFillType() & 1) ? kEvenOdd_PathOpsMask
            : kWinding_PathOpsMask;
    preFetch();
}

bool SkOpEdgeBuilder::finish() {
    fOperand = false;
    if (fUnparseable || !walk()) {
        return false;
    }
    complete();
    SkOpContour* contour = fContourBuilder.contour();
    if (contour && !contour->count()) {
        fContoursHead->remove(contour);
    }
    return true;
}

void SkOpEdgeBuilder::closeContour(const SkPoint& curveEnd, const SkPoint& curveStart) {
    if (!SkDPoint::ApproximatelyEqual(curveEnd, curveStart)) {
        *fPathVerbs.append() = SkPath::kLine_Verb;
        *fPathPts.append() = curveStart;
    } else {
        int verbCount = fPathVerbs.count();
        int ptsCount = fPathPts.count();
        if (SkPath::kLine_Verb == fPathVerbs[verbCount - 1]
                && fPathPts[ptsCount - 2] == curveStart) {
            fPathVerbs.pop();
            fPathPts.pop();
        } else {
            fPathPts[ptsCount - 1] = curveStart;
        }
    }
    *fPathVerbs.append() = SkPath::kClose_Verb;
}

int SkOpEdgeBuilder::preFetch() {
    if (!fPath->isFinite()) {
        fUnparseable = true;
        return 0;
    }
    SkPoint curveStart;
    SkPoint curve[4];
    bool lastCurve = false;
    for (auto [pathVerb, pts, w] : SkPathPriv::Iterate(*fPath)) {
        auto verb = static_cast<SkPath::Verb>(pathVerb);
        switch (verb) {
            case SkPath::kMove_Verb:
                if (!fAllowOpenContours && lastCurve) {
                    closeContour(curve[0], curveStart);
                }
                *fPathVerbs.append() = verb;
                curve[0] = force_small_to_zero(pts[0]);
                *fPathPts.append() = curve[0];
                curveStart = curve[0];
                lastCurve = false;
                continue;
            case SkPath::kLine_Verb:
                curve[1] = force_small_to_zero(pts[1]);
                if (SkDPoint::ApproximatelyEqual(curve[0], curve[1])) {
                    uint8_t lastVerb = fPathVerbs.top();
                    if (lastVerb != SkPath::kLine_Verb && lastVerb != SkPath::kMove_Verb) {
                        fPathPts.top() = curve[0] = curve[1];
                    }
                    continue;  // skip degenerate points
                }
                break;
            case SkPath::kQuad_Verb:
                curve[1] = force_small_to_zero(pts[1]);
                curve[2] = force_small_to_zero(pts[2]);
                verb = SkReduceOrder::Quad(curve, curve);
                if (verb == SkPath::kMove_Verb) {
                    continue;  // skip degenerate points
                }
                break;
            case SkPath::kConic_Verb:
                curve[1] = force_small_to_zero(pts[1]);
                curve[2] = force_small_to_zero(pts[2]);
                verb = SkReduceOrder::Quad(curve, curve);
                if (SkPath::kQuad_Verb == verb && 1 != *w) {
                  verb = SkPath::kConic_Verb;
                } else if (verb == SkPath::kMove_Verb) {
                    continue;  // skip degenerate points
                }
                break;
            case SkPath::kCubic_Verb:
                curve[1] = force_small_to_zero(pts[1]);
                curve[2] = force_small_to_zero(pts[2]);
                curve[3] = force_small_to_zero(pts[3]);
                verb = SkReduceOrder::Cubic(curve, curve);
                if (verb == SkPath::kMove_Verb) {
                    continue;  // skip degenerate points
                }
                break;
            case SkPath::kClose_Verb:
                closeContour(curve[0], curveStart);
                lastCurve = false;
                continue;
            case SkPath::kDone_Verb:
                continue;
        }
        *fPathVerbs.append() = verb;
        int ptCount = SkPathOpsVerbToPoints(verb);
        fPathPts.append(ptCount, &curve[1]);
        if (verb == SkPath::kConic_Verb) {
            *fWeights.append() = *w;
        }
        curve[0] = curve[ptCount];
        lastCurve = true;
    }
    if (!fAllowOpenContours && lastCurve) {
        closeContour(curve[0], curveStart);
    }
    *fPathVerbs.append() = SkPath::kDone_Verb;
    return fPathVerbs.count() - 1;
}

bool SkOpEdgeBuilder::close() {
    complete();
    return true;
}

bool SkOpEdgeBuilder::walk() {
    uint8_t* verbPtr = fPathVerbs.begin();
    uint8_t* endOfFirstHalf = &verbPtr[fSecondHalf];
    SkPoint* pointsPtr = fPathPts.begin();
    SkScalar* weightPtr = fWeights.begin();
    SkPath::Verb verb;
    SkOpContour* contour = fContourBuilder.contour();
    int moveToPtrBump = 0;
    while ((verb = (SkPath::Verb) *verbPtr) != SkPath::kDone_Verb) {
        if (verbPtr == endOfFirstHalf) {
            fOperand = true;
        }
        verbPtr++;
        switch (verb) {
            case SkPath::kMove_Verb:
                if (contour && contour->count()) {
                    if (fAllowOpenContours) {
                        complete();
                    } else if (!close()) {
                        return false;
                    }
                }
                if (!contour) {
                    fContourBuilder.setContour(contour = fContoursHead->appendContour());
                }
                contour->init(fGlobalState, fOperand,
                    fXorMask[fOperand] == kEvenOdd_PathOpsMask);
                pointsPtr += moveToPtrBump;
                moveToPtrBump = 1;
                continue;
            case SkPath::kLine_Verb:
                fContourBuilder.addLine(pointsPtr);
                break;
            case SkPath::kQuad_Verb:
                {
                    SkVector vec1 = pointsPtr[1] - pointsPtr[0];
                    SkVector vec2 = pointsPtr[2] - pointsPtr[1];
                    if (vec1.dot(vec2) < 0) {
                        SkPoint pair[5];
                        if (SkChopQuadAtMaxCurvature(pointsPtr, pair) == 1) {
                            goto addOneQuad;
                        }
                        if (!SkScalarsAreFinite(&pair[0].fX, SK_ARRAY_COUNT(pair) * 2)) {
                            return false;
                        }
                        for (unsigned index = 0; index < SK_ARRAY_COUNT(pair); ++index) {
                            pair[index] = force_small_to_zero(pair[index]);
                        }
                        SkPoint cStorage[2][2];
                        SkPath::Verb v1 = SkReduceOrder::Quad(&pair[0], cStorage[0]);
                        SkPath::Verb v2 = SkReduceOrder::Quad(&pair[2], cStorage[1]);
                        SkPoint* curve1 = v1 != SkPath::kLine_Verb ? &pair[0] : cStorage[0];
                        SkPoint* curve2 = v2 != SkPath::kLine_Verb ? &pair[2] : cStorage[1];
                        if (can_add_curve(v1, curve1) && can_add_curve(v2, curve2)) {
                            fContourBuilder.addCurve(v1, curve1);
                            fContourBuilder.addCurve(v2, curve2);
                            break;
                        }
                    }
                }
            addOneQuad:
                fContourBuilder.addQuad(pointsPtr);
                break;
            case SkPath::kConic_Verb: {
                SkVector vec1 = pointsPtr[1] - pointsPtr[0];
                SkVector vec2 = pointsPtr[2] - pointsPtr[1];
                SkScalar weight = *weightPtr++;
                if (vec1.dot(vec2) < 0) {
                    // FIXME: max curvature for conics hasn't been implemented; use placeholder
                    SkScalar maxCurvature = SkFindQuadMaxCurvature(pointsPtr);
                    if (0 < maxCurvature && maxCurvature < 1) {
                        SkConic conic(pointsPtr, weight);
                        SkConic pair[2];
                        if (!conic.chopAt(maxCurvature, pair)) {
                            // if result can't be computed, use original
                            fContourBuilder.addConic(pointsPtr, weight);
                            break;
                        }
                        SkPoint cStorage[2][3];
                        SkPath::Verb v1 = SkReduceOrder::Conic(pair[0], cStorage[0]);
                        SkPath::Verb v2 = SkReduceOrder::Conic(pair[1], cStorage[1]);
                        SkPoint* curve1 = v1 != SkPath::kLine_Verb ? pair[0].fPts : cStorage[0];
                        SkPoint* curve2 = v2 != SkPath::kLine_Verb ? pair[1].fPts : cStorage[1];
                        if (can_add_curve(v1, curve1) && can_add_curve(v2, curve2)) {
                            fContourBuilder.addCurve(v1, curve1, pair[0].fW);
                            fContourBuilder.addCurve(v2, curve2, pair[1].fW);
                            break;
                        }
                    }
                }
                fContourBuilder.addConic(pointsPtr, weight);
                } break;
            case SkPath::kCubic_Verb:
                {
                    // Split complex cubics (such as self-intersecting curves or
                    // ones with difficult curvature) in two before proceeding.
                    // This can be required for intersection to succeed.
                    SkScalar splitT[3];
                    int breaks = SkDCubic::ComplexBreak(pointsPtr, splitT);
                    if (!breaks) {
                        fContourBuilder.addCubic(pointsPtr);
                        break;
                    }
                    SkASSERT(breaks <= (int) SK_ARRAY_COUNT(splitT));
                    struct Splitsville {
                        double fT[2];
                        SkPoint fPts[4];
                        SkPoint fReduced[4];
                        SkPath::Verb fVerb;
                        bool fCanAdd;
                    } splits[4];
                    SkASSERT(SK_ARRAY_COUNT(splits) == SK_ARRAY_COUNT(splitT) + 1);
                    SkTQSort(splitT, splitT + breaks);
                    for (int index = 0; index <= breaks; ++index) {
                        Splitsville* split = &splits[index];
                        split->fT[0] = index ? splitT[index - 1] : 0;
                        split->fT[1] = index < breaks ? splitT[index] : 1;
                        SkDCubic part = SkDCubic::SubDivide(pointsPtr, split->fT[0], split->fT[1]);
                        if (!part.toFloatPoints(split->fPts)) {
                            return false;
                        }
                        split->fVerb = SkReduceOrder::Cubic(split->fPts, split->fReduced);
                        SkPoint* curve = SkPath::kCubic_Verb == split->fVerb
                                ? split->fPts : split->fReduced;
                        split->fCanAdd = can_add_curve(split->fVerb, curve);
                    }
                    for (int index = 0; index <= breaks; ++index) {
                        Splitsville* split = &splits[index];
                        if (!split->fCanAdd) {
                            continue;
                        }
                        int prior = index;
                        while (prior > 0 && !splits[prior - 1].fCanAdd) {
                            --prior;
                        }
                        if (prior < index) {
                            split->fT[0] = splits[prior].fT[0];
                            split->fPts[0] = splits[prior].fPts[0];
                        }
                        int next = index;
                        int breakLimit = std::min(breaks, (int) SK_ARRAY_COUNT(splits) - 1);
                        while (next < breakLimit && !splits[next + 1].fCanAdd) {
                            ++next;
                        }
                        if (next > index) {
                            split->fT[1] = splits[next].fT[1];
                            split->fPts[3] = splits[next].fPts[3];
                        }
                        if (prior < index || next > index) {
                            split->fVerb = SkReduceOrder::Cubic(split->fPts, split->fReduced);
                        }
                        SkPoint* curve = SkPath::kCubic_Verb == split->fVerb
                                ? split->fPts : split->fReduced;
                        if (!can_add_curve(split->fVerb, curve)) {
                            return false;
                        }
                        fContourBuilder.addCurve(split->fVerb, curve);
                    }
                }
                break;
            case SkPath::kClose_Verb:
                SkASSERT(contour);
                if (!close()) {
                    return false;
                }
                contour = nullptr;
                continue;
            default:
                SkDEBUGFAIL("bad verb");
                return false;
        }
        SkASSERT(contour);
        if (contour->count()) {
            contour->debugValidate();
        }
        pointsPtr += SkPathOpsVerbToPoints(verb);
    }
    fContourBuilder.flush();
    if (contour && contour->count() &&!fAllowOpenContours && !close()) {
        return false;
    }
    return true;
}
