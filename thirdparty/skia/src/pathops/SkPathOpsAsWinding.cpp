/*
 * Copyright 2018 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "include/core/SkRect.h"
#include "src/core/SkPathPriv.h"
#include "src/pathops/SkOpEdgeBuilder.h"
#include "src/pathops/SkPathOpsCommon.h"
#include <algorithm>
#include <vector>

using std::vector;

struct Contour {
    enum class Direction {  // SkPathDirection doesn't have 'none' state
        kCCW = -1,
        kNone,
        kCW,
    };

    Contour(const SkRect& bounds, int lastStart, int verbStart)
        : fBounds(bounds)
        , fVerbStart(lastStart)
        , fVerbEnd(verbStart) {
    }

    vector<Contour*> fChildren;
    const SkRect fBounds;
    SkPoint fMinXY{SK_ScalarMax, SK_ScalarMax};
    const int fVerbStart;
    const int fVerbEnd;
    Direction fDirection{Direction::kNone};
    bool fContained{false};
    bool fReverse{false};
};

static const int kPtCount[] = { 1, 1, 2, 2, 3, 0 };
static const int kPtIndex[] = { 0, 1, 1, 1, 1, 0 };

static Contour::Direction to_direction(SkScalar dy) {
    return dy > 0 ? Contour::Direction::kCCW : dy < 0 ? Contour::Direction::kCW :
            Contour::Direction::kNone;
}

static int contains_edge(SkPoint pts[4], SkPath::Verb verb, SkScalar weight, const SkPoint& edge) {
    SkRect bounds;
    bounds.setBounds(pts, kPtCount[verb] + 1);
    if (bounds.fTop > edge.fY) {
        return 0;
    }
    if (bounds.fBottom <= edge.fY) {  // check to see if y is at line end to avoid double counting
        return 0;
    }
    if (bounds.fLeft >= edge.fX) {
        return 0;
    }
    int winding = 0;
    double tVals[3];
    Contour::Direction directions[3];
    // must intersect horz ray with curve in case it intersects more than once
    int count = (*CurveIntercept[verb * 2])(pts, weight, edge.fY, tVals);
    SkASSERT(between(0, count, 3));
    // remove results to the right of edge
    for (int index = 0; index < count; ) {
        SkScalar intersectX = (*CurvePointAtT[verb])(pts, weight, tVals[index]).fX;
        if (intersectX < edge.fX) {
            ++index;
            continue;
        }
        if (intersectX > edge.fX) {
            tVals[index] = tVals[--count];
            continue;
        }
        // if intersect x equals edge x, we need to determine if pts is to the left or right of edge
        if (pts[0].fX < edge.fX && pts[kPtCount[verb]].fX < edge.fX) {
            ++index;
            continue;
        }
        // TODO : other cases need discriminating. need op angle code to figure it out
        // example: edge ends 45 degree diagonal going up. If pts is to the left of edge, keep.
        // if pts is to the right of edge, discard. With code as is, can't distiguish the two cases.
        tVals[index] = tVals[--count];
    }
    // use first derivative to determine if intersection is contributing +1 or -1 to winding
    for (int index = 0; index < count; ++index) {
        directions[index] = to_direction((*CurveSlopeAtT[verb])(pts, weight, tVals[index]).fY);
    }
    for (int index = 0; index < count; ++index) {
        // skip intersections that end at edge and go up
        if (zero_or_one(tVals[index]) && Contour::Direction::kCCW != directions[index]) {
            continue;
        }
        winding += (int) directions[index];
    }
    return winding;  // note winding indicates containership, not contour direction
}

static SkScalar conic_weight(const SkPath::Iter& iter, SkPath::Verb verb) {
    return SkPath::kConic_Verb == verb ? iter.conicWeight() : 1;
}

static SkPoint left_edge(SkPoint pts[4], SkPath::Verb verb, SkScalar weight,
        Contour::Direction* direction) {
    SkASSERT(SkPath::kLine_Verb <= verb && verb <= SkPath::kCubic_Verb);
    SkPoint result;
    double dy;
    double t SK_INIT_TO_AVOID_WARNING;
    int roots = 0;
    if (SkPath::kLine_Verb == verb) {
        result = pts[0].fX < pts[1].fX ? pts[0] : pts[1];
        dy = pts[1].fY - pts[0].fY;
    } else if (SkPath::kQuad_Verb == verb) {
        SkDQuad quad;
        quad.set(pts);
        if (!quad.monotonicInX()) {
            roots = SkDQuad::FindExtrema(&quad[0].fX, &t);
        }
        if (roots) {
            result = quad.ptAtT(t).asSkPoint();
        } else {
            result = pts[0].fX < pts[2].fX ? pts[0] : pts[2];
            t = pts[0].fX < pts[2].fX ? 0 : 1;
        }
        dy = quad.dxdyAtT(t).fY;
    } else if (SkPath::kConic_Verb == verb) {
        SkDConic conic;
        conic.set(pts, weight);
        if (!conic.monotonicInX()) {
            roots = SkDConic::FindExtrema(&conic[0].fX, weight, &t);
        }
        if (roots) {
            result = conic.ptAtT(t).asSkPoint();
        } else {
            result = pts[0].fX < pts[2].fX ? pts[0] : pts[2];
            t = pts[0].fX < pts[2].fX ? 0 : 1;
        }
        dy = conic.dxdyAtT(t).fY;
    } else {
        SkASSERT(SkPath::kCubic_Verb == verb);
        SkDCubic cubic;
        cubic.set(pts);
        if (!cubic.monotonicInX()) {
            double tValues[2];
            roots = SkDCubic::FindExtrema(&cubic[0].fX, tValues);
            SkASSERT(roots <= 2);
            for (int index = 0; index < roots; ++index) {
                SkPoint temp = cubic.ptAtT(tValues[index]).asSkPoint();
                if (0 == index || result.fX > temp.fX) {
                    result = temp;
                    t = tValues[index];
                }
            }
        }
        if (roots) {
            result = cubic.ptAtT(t).asSkPoint();
        } else {
            result = pts[0].fX < pts[3].fX ? pts[0] : pts[3];
            t = pts[0].fX < pts[3].fX ? 0 : 1;
        }
        dy = cubic.dxdyAtT(t).fY;
    }
    *direction = to_direction(dy);
    return result;
}

class OpAsWinding {
public:
    enum class Edge {
        kInitial,
        kCompare,
    };

    OpAsWinding(const SkPath& path)
        : fPath(path) {
    }

    void contourBounds(vector<Contour>* containers) {
        SkRect bounds;
        bounds.setEmpty();
        int lastStart = 0;
        int verbStart = 0;
        for (auto [verb, pts, w] : SkPathPriv::Iterate(fPath)) {
            if (SkPathVerb::kMove == verb) {
                if (!bounds.isEmpty()) {
                    containers->emplace_back(bounds, lastStart, verbStart);
                    lastStart = verbStart;
               }
               bounds.setBounds(&pts[kPtIndex[SkPath::kMove_Verb]], kPtCount[SkPath::kMove_Verb]);
            }
            if (SkPathVerb::kLine <= verb && verb <= SkPathVerb::kCubic) {
                SkRect verbBounds;
                verbBounds.setBounds(&pts[kPtIndex[(int)verb]], kPtCount[(int)verb]);
                bounds.joinPossiblyEmptyRect(verbBounds);
            }
            ++verbStart;
        }
        if (!bounds.isEmpty()) {
            containers->emplace_back(bounds, lastStart, ++verbStart);
        }
    }

    int nextEdge(Contour& contour, Edge edge) {
        SkPath::Iter iter(fPath, true);
        SkPoint pts[4];
        SkPath::Verb verb;
        int verbCount = -1;
        int winding = 0;
        do {
            verb = iter.next(pts);
            if (++verbCount < contour.fVerbStart) {
                continue;
            }
            if (verbCount >= contour.fVerbEnd) {
                continue;
            }
            if (SkPath::kLine_Verb > verb || verb > SkPath::kCubic_Verb) {
                continue;
            }
            bool horizontal = true;
            for (int index = 1; index <= kPtCount[verb]; ++index) {
                if (pts[0].fY != pts[index].fY) {
                    horizontal = false;
                    break;
                }
            }
            if (horizontal) {
                continue;
            }
            if (edge == Edge::kCompare) {
                winding += contains_edge(pts, verb, conic_weight(iter, verb), contour.fMinXY);
                continue;
            }
            SkASSERT(edge == Edge::kInitial);
            Contour::Direction direction;
            SkPoint minXY = left_edge(pts, verb, conic_weight(iter, verb), &direction);
            if (minXY.fX > contour.fMinXY.fX) {
                continue;
            }
            if (minXY.fX == contour.fMinXY.fX) {
                if (minXY.fY != contour.fMinXY.fY) {
                    continue;
                }
                if (direction == contour.fDirection) {
                    continue;
                }
                // incomplete: must sort edges to find the one most to left
                // File a bug if this code path is triggered and AsWinding was
                // expected to succeed.
                SkDEBUGF("incomplete\n");
                // TODO: add edges as opangle and sort
            }
            contour.fMinXY = minXY;
            contour.fDirection = direction;
        } while (SkPath::kDone_Verb != verb);
        return winding;
    }

    bool containerContains(Contour& contour, Contour& test) {
        // find outside point on lesser contour
        // arbitrarily, choose non-horizontal edge where point <= bounds left
        // note that if leftmost point is control point, may need tight bounds
            // to find edge with minimum-x
        if (SK_ScalarMax == test.fMinXY.fX) {
            this->nextEdge(test, Edge::kInitial);
        }
        // find all edges on greater equal or to the left of one on lesser
        contour.fMinXY = test.fMinXY;
        int winding = this->nextEdge(contour, Edge::kCompare);
        // if edge is up, mark contour cw, otherwise, ccw
        // sum of greater edges direction should be cw, 0, ccw
        test.fContained = winding != 0;
        return -1 <= winding && winding <= 1;
    }

    void inParent(Contour& contour, Contour& parent) {
        // move contour into sibling list contained by parent
        for (auto test : parent.fChildren) {
            if (test->fBounds.contains(contour.fBounds)) {
                inParent(contour, *test);
                return;
            }
        }
        // move parent's children into contour's children if contained by contour
        for (auto iter = parent.fChildren.begin(); iter != parent.fChildren.end(); ) {
            if (contour.fBounds.contains((*iter)->fBounds)) {
                contour.fChildren.push_back(*iter);
                iter = parent.fChildren.erase(iter);
                continue;
            }
            ++iter;
        }
        parent.fChildren.push_back(&contour);
    }

    bool checkContainerChildren(Contour* parent, Contour* child) {
        for (auto grandChild : child->fChildren) {
            if (!checkContainerChildren(child, grandChild)) {
                return false;
            }
        }
        if (parent) {
            if (!containerContains(*parent, *child)) {
                return false;
            }
        }
        return true;
    }

    bool markReverse(Contour* parent, Contour* child) {
        bool reversed = false;
        for (auto grandChild : child->fChildren) {
            reversed |= markReverse(grandChild->fContained ? child : parent, grandChild);
        }
        if (parent && parent->fDirection == child->fDirection) {
            child->fReverse = true;
            child->fDirection = (Contour::Direction) -(int) child->fDirection;
            return true;
        }
        return reversed;
    }

    SkPath reverseMarkedContours(vector<Contour>& contours, SkPathFillType fillType) {
        SkPathPriv::Iterate iterate(fPath);
        auto iter = iterate.begin();
        int verbCount = 0;

        SkPathBuilder result;
        result.setFillType(fillType);
        for (const Contour& contour : contours) {
            SkPathBuilder reverse;
            SkPathBuilder* temp = contour.fReverse ? &reverse : &result;
            for (; iter != iterate.end() && verbCount < contour.fVerbEnd; ++iter, ++verbCount) {
                auto [verb, pts, w] = *iter;
                switch (verb) {
                    case SkPathVerb::kMove:
                        temp->moveTo(pts[0]);
                        break;
                    case SkPathVerb::kLine:
                        temp->lineTo(pts[1]);
                        break;
                    case SkPathVerb::kQuad:
                        temp->quadTo(pts[1], pts[2]);
                        break;
                    case SkPathVerb::kConic:
                        temp->conicTo(pts[1], pts[2], *w);
                        break;
                    case SkPathVerb::kCubic:
                        temp->cubicTo(pts[1], pts[2], pts[3]);
                        break;
                    case SkPathVerb::kClose:
                        temp->close();
                        break;
                }
            }
            if (contour.fReverse) {
                SkASSERT(temp == &reverse);
                SkPathPriv::ReverseAddPath(&result, reverse.detach());
            }
        }
        return result.detach();
    }

private:
    const SkPath& fPath;
};

static bool set_result_path(SkPath* result, const SkPath& path, SkPathFillType fillType) {
    *result = path;
    result->setFillType(fillType);
    return true;
}

bool AsWinding(const SkPath& path, SkPath* result) {
    if (!path.isFinite()) {
        return false;
    }
    SkPathFillType fillType = path.getFillType();
    if (fillType == SkPathFillType::kWinding
            || fillType == SkPathFillType::kInverseWinding ) {
        return set_result_path(result, path, fillType);
    }
    fillType = path.isInverseFillType() ? SkPathFillType::kInverseWinding :
            SkPathFillType::kWinding;
    if (path.isEmpty() || path.isConvex()) {
        return set_result_path(result, path, fillType);
    }
    // count contours
    vector<Contour> contours;   // one per contour
    OpAsWinding winder(path);
    winder.contourBounds(&contours);
    if (contours.size() <= 1) {
        return set_result_path(result, path, fillType);
    }
    // create contour bounding box tree
    Contour sorted(SkRect(), 0, 0);
    for (auto& contour : contours) {
        winder.inParent(contour, sorted);
    }
    // if sorted has no grandchildren, no child has to fix its children's winding
    if (std::all_of(sorted.fChildren.begin(), sorted.fChildren.end(),
            [](const Contour* contour) -> bool { return !contour->fChildren.size(); } )) {
        return set_result_path(result, path, fillType);
    }
    // starting with outermost and moving inward, see if one path contains another
    for (auto contour : sorted.fChildren) {
        winder.nextEdge(*contour, OpAsWinding::Edge::kInitial);
        if (!winder.checkContainerChildren(nullptr, contour)) {
            return false;
        }
    }
    // starting with outermost and moving inward, mark paths to reverse
    bool reversed = false;
    for (auto contour : sorted.fChildren) {
        reversed |= winder.markReverse(nullptr, contour);
    }
    if (!reversed) {
        return set_result_path(result, path, fillType);
    }
    *result = winder.reverseMarkedContours(contours, fillType);
    return true;
}
