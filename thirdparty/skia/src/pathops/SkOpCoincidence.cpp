/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/pathops/SkOpCoincidence.h"
#include "src/pathops/SkOpSegment.h"
#include "src/pathops/SkPathOpsTSect.h"

#include <utility>

// returns true if coincident span's start and end are the same
bool SkCoincidentSpans::collapsed(const SkOpPtT* test) const {
    return (fCoinPtTStart == test && fCoinPtTEnd->contains(test))
        || (fCoinPtTEnd == test && fCoinPtTStart->contains(test))
        || (fOppPtTStart == test && fOppPtTEnd->contains(test))
        || (fOppPtTEnd == test && fOppPtTStart->contains(test));
}

// out of line since this function is referenced by address
const SkOpPtT* SkCoincidentSpans::coinPtTEnd() const {
    return fCoinPtTEnd;
}

// out of line since this function is referenced by address
const SkOpPtT* SkCoincidentSpans::coinPtTStart() const {
    return fCoinPtTStart;
}

// sets the span's end to the ptT referenced by the previous-next
void SkCoincidentSpans::correctOneEnd(
        const SkOpPtT* (SkCoincidentSpans::* getEnd)() const,
        void (SkCoincidentSpans::*setEnd)(const SkOpPtT* ptT) ) {
    const SkOpPtT* origPtT = (this->*getEnd)();
    const SkOpSpanBase* origSpan = origPtT->span();
    const SkOpSpan* prev = origSpan->prev();
    const SkOpPtT* testPtT = prev ? prev->next()->ptT()
            : origSpan->upCast()->next()->prev()->ptT();
    if (origPtT != testPtT) {
        (this->*setEnd)(testPtT);
    }
}

/* Please keep this in sync with debugCorrectEnds */
// FIXME: member pointers have fallen out of favor and can be replaced with
// an alternative approach.
// makes all span ends agree with the segment's spans that define them
void SkCoincidentSpans::correctEnds() {
    this->correctOneEnd(&SkCoincidentSpans::coinPtTStart, &SkCoincidentSpans::setCoinPtTStart);
    this->correctOneEnd(&SkCoincidentSpans::coinPtTEnd, &SkCoincidentSpans::setCoinPtTEnd);
    this->correctOneEnd(&SkCoincidentSpans::oppPtTStart, &SkCoincidentSpans::setOppPtTStart);
    this->correctOneEnd(&SkCoincidentSpans::oppPtTEnd, &SkCoincidentSpans::setOppPtTEnd);
}

/* Please keep this in sync with debugExpand */
// expand the range by checking adjacent spans for coincidence
bool SkCoincidentSpans::expand() {
    bool expanded = false;
    const SkOpSegment* segment = coinPtTStart()->segment();
    const SkOpSegment* oppSegment = oppPtTStart()->segment();
    do {
        const SkOpSpan* start = coinPtTStart()->span()->upCast();
        const SkOpSpan* prev = start->prev();
        const SkOpPtT* oppPtT;
        if (!prev || !(oppPtT = prev->contains(oppSegment))) {
            break;
        }
        double midT = (prev->t() + start->t()) / 2;
        if (!segment->isClose(midT, oppSegment)) {
            break;
        }
        setStarts(prev->ptT(), oppPtT);
        expanded = true;
    } while (true);
    do {
        const SkOpSpanBase* end = coinPtTEnd()->span();
        SkOpSpanBase* next = end->final() ? nullptr : end->upCast()->next();
        if (next && next->deleted()) {
            break;
        }
        const SkOpPtT* oppPtT;
        if (!next || !(oppPtT = next->contains(oppSegment))) {
            break;
        }
        double midT = (end->t() + next->t()) / 2;
        if (!segment->isClose(midT, oppSegment)) {
            break;
        }
        setEnds(next->ptT(), oppPtT);
        expanded = true;
    } while (true);
    return expanded;
}

// increase the range of this span
bool SkCoincidentSpans::extend(const SkOpPtT* coinPtTStart, const SkOpPtT* coinPtTEnd,
        const SkOpPtT* oppPtTStart, const SkOpPtT* oppPtTEnd) {
    bool result = false;
    if (fCoinPtTStart->fT > coinPtTStart->fT || (this->flipped()
            ? fOppPtTStart->fT < oppPtTStart->fT : fOppPtTStart->fT > oppPtTStart->fT)) {
        this->setStarts(coinPtTStart, oppPtTStart);
        result = true;
    }
    if (fCoinPtTEnd->fT < coinPtTEnd->fT || (this->flipped()
            ? fOppPtTEnd->fT > oppPtTEnd->fT : fOppPtTEnd->fT < oppPtTEnd->fT)) {
        this->setEnds(coinPtTEnd, oppPtTEnd);
        result = true;
    }
    return result;
}

// set the range of this span
void SkCoincidentSpans::set(SkCoincidentSpans* next, const SkOpPtT* coinPtTStart,
        const SkOpPtT* coinPtTEnd, const SkOpPtT* oppPtTStart, const SkOpPtT* oppPtTEnd) {
    SkASSERT(SkOpCoincidence::Ordered(coinPtTStart, oppPtTStart));
    fNext = next;
    this->setStarts(coinPtTStart, oppPtTStart);
    this->setEnds(coinPtTEnd, oppPtTEnd);
}

// returns true if both points are inside this
bool SkCoincidentSpans::contains(const SkOpPtT* s, const SkOpPtT* e) const {
    if (s->fT > e->fT) {
        using std::swap;
        swap(s, e);
    }
    if (s->segment() == fCoinPtTStart->segment()) {
        return fCoinPtTStart->fT <= s->fT && e->fT <= fCoinPtTEnd->fT;
    } else {
        SkASSERT(s->segment() == fOppPtTStart->segment());
        double oppTs = fOppPtTStart->fT;
        double oppTe = fOppPtTEnd->fT;
        if (oppTs > oppTe) {
            using std::swap;
            swap(oppTs, oppTe);
        }
        return oppTs <= s->fT && e->fT <= oppTe;
    }
}

// out of line since this function is referenced by address
const SkOpPtT* SkCoincidentSpans::oppPtTStart() const {
    return fOppPtTStart;
}

// out of line since this function is referenced by address
const SkOpPtT* SkCoincidentSpans::oppPtTEnd() const {
    return fOppPtTEnd;
}

// A coincident span is unordered if the pairs of points in the main and opposite curves'
// t values do not ascend or descend. For instance, if a tightly arced quadratic is
// coincident with another curve, it may intersect it out of order.
bool SkCoincidentSpans::ordered(bool* result) const {
    const SkOpSpanBase* start = this->coinPtTStart()->span();
    const SkOpSpanBase* end = this->coinPtTEnd()->span();
    const SkOpSpanBase* next = start->upCast()->next();
    if (next == end) {
        *result = true;
        return true;
    }
    bool flipped = this->flipped();
    const SkOpSegment* oppSeg = this->oppPtTStart()->segment();
    double oppLastT = fOppPtTStart->fT;
    do {
        const SkOpPtT* opp = next->contains(oppSeg);
        if (!opp) {
//            SkOPOBJASSERT(start, 0);  // may assert if coincident span isn't fully processed
            return false;
        }
        if ((oppLastT > opp->fT) != flipped) {
            *result = false;
            return true;
        }
        oppLastT = opp->fT;
        if (next == end) {
            break;
        }
        if (!next->upCastable()) {
            *result = false;
            return true;
        }
        next = next->upCast()->next();
    } while (true);
    *result = true;
    return true;
}

// if there is an existing pair that overlaps the addition, extend it
bool SkOpCoincidence::extend(const SkOpPtT* coinPtTStart, const SkOpPtT* coinPtTEnd,
        const SkOpPtT* oppPtTStart, const SkOpPtT* oppPtTEnd) {
    SkCoincidentSpans* test = fHead;
    if (!test) {
        return false;
    }
    const SkOpSegment* coinSeg = coinPtTStart->segment();
    const SkOpSegment* oppSeg = oppPtTStart->segment();
    if (!Ordered(coinPtTStart, oppPtTStart)) {
        using std::swap;
        swap(coinSeg, oppSeg);
        swap(coinPtTStart, oppPtTStart);
        swap(coinPtTEnd, oppPtTEnd);
        if (coinPtTStart->fT > coinPtTEnd->fT) {
            swap(coinPtTStart, coinPtTEnd);
            swap(oppPtTStart, oppPtTEnd);
        }
    }
    double oppMinT = std::min(oppPtTStart->fT, oppPtTEnd->fT);
    SkDEBUGCODE(double oppMaxT = std::max(oppPtTStart->fT, oppPtTEnd->fT));
    do {
        if (coinSeg != test->coinPtTStart()->segment()) {
            continue;
        }
        if (oppSeg != test->oppPtTStart()->segment()) {
            continue;
        }
        double oTestMinT = std::min(test->oppPtTStart()->fT, test->oppPtTEnd()->fT);
        double oTestMaxT = std::max(test->oppPtTStart()->fT, test->oppPtTEnd()->fT);
        // if debug check triggers, caller failed to check if extended already exists
        SkASSERT(test->coinPtTStart()->fT > coinPtTStart->fT
                || coinPtTEnd->fT > test->coinPtTEnd()->fT
                || oTestMinT > oppMinT || oppMaxT > oTestMaxT);
        if ((test->coinPtTStart()->fT <= coinPtTEnd->fT
                && coinPtTStart->fT <= test->coinPtTEnd()->fT)
                || (oTestMinT <= oTestMaxT && oppMinT <= oTestMaxT)) {
            test->extend(coinPtTStart, coinPtTEnd, oppPtTStart, oppPtTEnd);
            return true;
        }
    } while ((test = test->next()));
    return false;
}

// verifies that the coincidence hasn't already been added
static void DebugCheckAdd(const SkCoincidentSpans* check, const SkOpPtT* coinPtTStart,
        const SkOpPtT* coinPtTEnd, const SkOpPtT* oppPtTStart, const SkOpPtT* oppPtTEnd) {
#if DEBUG_COINCIDENCE
    while (check) {
        SkASSERT(check->coinPtTStart() != coinPtTStart || check->coinPtTEnd() != coinPtTEnd
                || check->oppPtTStart() != oppPtTStart || check->oppPtTEnd() != oppPtTEnd);
        SkASSERT(check->coinPtTStart() != oppPtTStart || check->coinPtTEnd() != oppPtTEnd
                || check->oppPtTStart() != coinPtTStart || check->oppPtTEnd() != coinPtTEnd);
        check = check->next();
    }
#endif
}

// adds a new coincident pair
void SkOpCoincidence::add(SkOpPtT* coinPtTStart, SkOpPtT* coinPtTEnd, SkOpPtT* oppPtTStart,
        SkOpPtT* oppPtTEnd) {
    // OPTIMIZE: caller should have already sorted
    if (!Ordered(coinPtTStart, oppPtTStart)) {
        if (oppPtTStart->fT < oppPtTEnd->fT) {
            this->add(oppPtTStart, oppPtTEnd, coinPtTStart, coinPtTEnd);
        } else {
            this->add(oppPtTEnd, oppPtTStart, coinPtTEnd, coinPtTStart);
        }
        return;
    }
    SkASSERT(Ordered(coinPtTStart, oppPtTStart));
    // choose the ptT at the front of the list to track
    coinPtTStart = coinPtTStart->span()->ptT();
    coinPtTEnd = coinPtTEnd->span()->ptT();
    oppPtTStart = oppPtTStart->span()->ptT();
    oppPtTEnd = oppPtTEnd->span()->ptT();
    SkOPASSERT(coinPtTStart->fT < coinPtTEnd->fT);
    SkOPASSERT(oppPtTStart->fT != oppPtTEnd->fT);
    SkOPASSERT(!coinPtTStart->deleted());
    SkOPASSERT(!coinPtTEnd->deleted());
    SkOPASSERT(!oppPtTStart->deleted());
    SkOPASSERT(!oppPtTEnd->deleted());
    DebugCheckAdd(fHead, coinPtTStart, coinPtTEnd, oppPtTStart, oppPtTEnd);
    DebugCheckAdd(fTop, coinPtTStart, coinPtTEnd, oppPtTStart, oppPtTEnd);
    SkCoincidentSpans* coinRec = this->globalState()->allocator()->make<SkCoincidentSpans>();
    coinRec->init(SkDEBUGCODE(fGlobalState));
    coinRec->set(this->fHead, coinPtTStart, coinPtTEnd, oppPtTStart, oppPtTEnd);
    fHead = coinRec;
}

// description below
bool SkOpCoincidence::addEndMovedSpans(const SkOpSpan* base, const SkOpSpanBase* testSpan) {
    const SkOpPtT* testPtT = testSpan->ptT();
    const SkOpPtT* stopPtT = testPtT;
    const SkOpSegment* baseSeg = base->segment();
    int escapeHatch = 100000;  // this is 100 times larger than the debugLoopLimit test
    while ((testPtT = testPtT->next()) != stopPtT) {
        if (--escapeHatch <= 0) {
            return false;  // if triggered (likely by a fuzz-generated test) too complex to succeed
        }
        const SkOpSegment* testSeg = testPtT->segment();
        if (testPtT->deleted()) {
            continue;
        }
        if (testSeg == baseSeg) {
            continue;
        }
        if (testPtT->span()->ptT() != testPtT) {
            continue;
        }
        if (this->contains(baseSeg, testSeg, testPtT->fT)) {
            continue;
        }
        // intersect perp with base->ptT() with testPtT->segment()
        SkDVector dxdy = baseSeg->dSlopeAtT(base->t());
        const SkPoint& pt = base->pt();
        SkDLine ray = {{{pt.fX, pt.fY}, {pt.fX + dxdy.fY, pt.fY - dxdy.fX}}};
        SkIntersections i  SkDEBUGCODE((this->globalState()));
        (*CurveIntersectRay[testSeg->verb()])(testSeg->pts(), testSeg->weight(), ray, &i);
        for (int index = 0; index < i.used(); ++index) {
            double t = i[0][index];
            if (!between(0, t, 1)) {
                continue;
            }
            SkDPoint oppPt = i.pt(index);
            if (!oppPt.approximatelyEqual(pt)) {
                continue;
            }
            SkOpSegment* writableSeg = const_cast<SkOpSegment*>(testSeg);
            SkOpPtT* oppStart = writableSeg->addT(t);
            if (oppStart == testPtT) {
                continue;
            }
            SkOpSpan* writableBase = const_cast<SkOpSpan*>(base);
            oppStart->span()->addOpp(writableBase);
            if (oppStart->deleted()) {
                continue;
            }
            SkOpSegment* coinSeg = base->segment();
            SkOpSegment* oppSeg = oppStart->segment();
            double coinTs, coinTe, oppTs, oppTe;
            if (Ordered(coinSeg, oppSeg)) {
                coinTs = base->t();
                coinTe = testSpan->t();
                oppTs = oppStart->fT;
                oppTe = testPtT->fT;
            } else {
                using std::swap;
                swap(coinSeg, oppSeg);
                coinTs = oppStart->fT;
                coinTe = testPtT->fT;
                oppTs = base->t();
                oppTe = testSpan->t();
            }
            if (coinTs > coinTe) {
                using std::swap;
                swap(coinTs, coinTe);
                swap(oppTs, oppTe);
            }
            bool added;
            FAIL_IF(!this->addOrOverlap(coinSeg, oppSeg, coinTs, coinTe, oppTs, oppTe, &added));
        }
    }
    return true;
}

// description below
bool SkOpCoincidence::addEndMovedSpans(const SkOpPtT* ptT) {
    FAIL_IF(!ptT->span()->upCastable());
    const SkOpSpan* base = ptT->span()->upCast();
    const SkOpSpan* prev = base->prev();
    FAIL_IF(!prev);
    if (!prev->isCanceled()) {
        if (!this->addEndMovedSpans(base, base->prev())) {
            return false;
        }
    }
    if (!base->isCanceled()) {
        if (!this->addEndMovedSpans(base, base->next())) {
            return false;
        }
    }
    return true;
}

/*  If A is coincident with B and B includes an endpoint, and A's matching point
    is not the endpoint (i.e., there's an implied line connecting B-end and A)
    then assume that the same implied line may intersect another curve close to B.
    Since we only care about coincidence that was undetected, look at the
    ptT list on B-segment adjacent to the B-end/A ptT loop (not in the loop, but
    next door) and see if the A matching point is close enough to form another
    coincident pair. If so, check for a new coincident span between B-end/A ptT loop
    and the adjacent ptT loop.
*/
bool SkOpCoincidence::addEndMovedSpans(DEBUG_COIN_DECLARE_ONLY_PARAMS()) {
    DEBUG_SET_PHASE();
    SkCoincidentSpans* span = fHead;
    if (!span) {
        return true;
    }
    fTop = span;
    fHead = nullptr;
    do {
        if (span->coinPtTStart()->fPt != span->oppPtTStart()->fPt) {
            FAIL_IF(1 == span->coinPtTStart()->fT);
            bool onEnd = span->coinPtTStart()->fT == 0;
            bool oOnEnd = zero_or_one(span->oppPtTStart()->fT);
            if (onEnd) {
                if (!oOnEnd) {  // if both are on end, any nearby intersect was already found
                    if (!this->addEndMovedSpans(span->oppPtTStart())) {
                        return false;
                    }
                }
            } else if (oOnEnd) {
                if (!this->addEndMovedSpans(span->coinPtTStart())) {
                    return false;
                }
            }
        }
        if (span->coinPtTEnd()->fPt != span->oppPtTEnd()->fPt) {
            bool onEnd = span->coinPtTEnd()->fT == 1;
            bool oOnEnd = zero_or_one(span->oppPtTEnd()->fT);
            if (onEnd) {
                if (!oOnEnd) {
                    if (!this->addEndMovedSpans(span->oppPtTEnd())) {
                        return false;
                    }
                }
            } else if (oOnEnd) {
                if (!this->addEndMovedSpans(span->coinPtTEnd())) {
                    return false;
                }
            }
        }
    } while ((span = span->next()));
    this->restoreHead();
    return true;
}

/* Please keep this in sync with debugAddExpanded */
// for each coincident pair, match the spans
// if the spans don't match, add the missing pt to the segment and loop it in the opposite span
bool SkOpCoincidence::addExpanded(DEBUG_COIN_DECLARE_ONLY_PARAMS()) {
    DEBUG_SET_PHASE();
    SkCoincidentSpans* coin = this->fHead;
    if (!coin) {
        return true;
    }
    do {
        const SkOpPtT* startPtT = coin->coinPtTStart();
        const SkOpPtT* oStartPtT = coin->oppPtTStart();
        double priorT = startPtT->fT;
        double oPriorT = oStartPtT->fT;
        FAIL_IF(!startPtT->contains(oStartPtT));
        SkOPASSERT(coin->coinPtTEnd()->contains(coin->oppPtTEnd()));
        const SkOpSpanBase* start = startPtT->span();
        const SkOpSpanBase* oStart = oStartPtT->span();
        const SkOpSpanBase* end = coin->coinPtTEnd()->span();
        const SkOpSpanBase* oEnd = coin->oppPtTEnd()->span();
        FAIL_IF(oEnd->deleted());
        FAIL_IF(!start->upCastable());
        const SkOpSpanBase* test = start->upCast()->next();
        FAIL_IF(!coin->flipped() && !oStart->upCastable());
        const SkOpSpanBase* oTest = coin->flipped() ? oStart->prev() : oStart->upCast()->next();
        FAIL_IF(!oTest);
        SkOpSegment* seg = start->segment();
        SkOpSegment* oSeg = oStart->segment();
        while (test != end || oTest != oEnd) {
            const SkOpPtT* containedOpp = test->ptT()->contains(oSeg);
            const SkOpPtT* containedThis = oTest->ptT()->contains(seg);
            if (!containedOpp || !containedThis) {
                // choose the ends, or the first common pt-t list shared by both
                double nextT, oNextT;
                if (containedOpp) {
                    nextT = test->t();
                    oNextT = containedOpp->fT;
                } else if (containedThis) {
                    nextT = containedThis->fT;
                    oNextT = oTest->t();
                } else {
                    // iterate through until a pt-t list found that contains the other
                    const SkOpSpanBase* walk = test;
                    const SkOpPtT* walkOpp;
                    do {
                        FAIL_IF(!walk->upCastable());
                        walk = walk->upCast()->next();
                    } while (!(walkOpp = walk->ptT()->contains(oSeg))
                            && walk != coin->coinPtTEnd()->span());
                    FAIL_IF(!walkOpp);
                    nextT = walk->t();
                    oNextT = walkOpp->fT;
                }
                // use t ranges to guess which one is missing
                double startRange = nextT - priorT;
                FAIL_IF(!startRange);
                double startPart = (test->t() - priorT) / startRange;
                double oStartRange = oNextT - oPriorT;
                FAIL_IF(!oStartRange);
                double oStartPart = (oTest->t() - oPriorT) / oStartRange;
                FAIL_IF(startPart == oStartPart);
                bool addToOpp = !containedOpp && !containedThis ? startPart < oStartPart
                        : !!containedThis;
                bool startOver = false;
                bool success = addToOpp ? oSeg->addExpanded(
                        oPriorT + oStartRange * startPart, test, &startOver)
                        : seg->addExpanded(
                        priorT + startRange * oStartPart, oTest, &startOver);
                FAIL_IF(!success);
                if (startOver) {
                    test = start;
                    oTest = oStart;
                }
                end = coin->coinPtTEnd()->span();
                oEnd = coin->oppPtTEnd()->span();
            }
            if (test != end) {
                FAIL_IF(!test->upCastable());
                priorT = test->t();
                test = test->upCast()->next();
            }
            if (oTest != oEnd) {
                oPriorT = oTest->t();
                if (coin->flipped()) {
                    oTest = oTest->prev();
                } else {
                    FAIL_IF(!oTest->upCastable());
                    oTest = oTest->upCast()->next();
                }
                FAIL_IF(!oTest);
            }

        }
    } while ((coin = coin->next()));
    return true;
}

// given a t span, map the same range on the coincident span
/*
the curves may not scale linearly, so interpolation may only happen within known points
remap over1s, over1e, cointPtTStart, coinPtTEnd to smallest range that captures over1s
then repeat to capture over1e
*/
double SkOpCoincidence::TRange(const SkOpPtT* overS, double t,
       const SkOpSegment* coinSeg  SkDEBUGPARAMS(const SkOpPtT* overE)) {
    const SkOpSpanBase* work = overS->span();
    const SkOpPtT* foundStart = nullptr;
    const SkOpPtT* foundEnd = nullptr;
    const SkOpPtT* coinStart = nullptr;
    const SkOpPtT* coinEnd = nullptr;
    do {
        const SkOpPtT* contained = work->contains(coinSeg);
        if (!contained) {
            if (work->final()) {
                break;
            }
            continue;
        }
        if (work->t() <= t) {
            coinStart = contained;
            foundStart = work->ptT();
        }
        if (work->t() >= t) {
            coinEnd = contained;
            foundEnd = work->ptT();
            break;
        }
        SkASSERT(work->ptT() != overE);
    } while ((work = work->upCast()->next()));
    if (!coinStart || !coinEnd) {
        return 1;
    }
    // while overS->fT <=t and overS contains coinSeg
    double denom = foundEnd->fT - foundStart->fT;
    double sRatio = denom ? (t - foundStart->fT) / denom : 1;
    return coinStart->fT + (coinEnd->fT - coinStart->fT) * sRatio;
}

// return true if span overlaps existing and needs to adjust the coincident list
bool SkOpCoincidence::checkOverlap(SkCoincidentSpans* check,
        const SkOpSegment* coinSeg, const SkOpSegment* oppSeg,
        double coinTs, double coinTe, double oppTs, double oppTe,
        SkTDArray<SkCoincidentSpans*>* overlaps) const {
    if (!Ordered(coinSeg, oppSeg)) {
        if (oppTs < oppTe) {
            return this->checkOverlap(check, oppSeg, coinSeg, oppTs, oppTe, coinTs, coinTe,
                    overlaps);
        }
        return this->checkOverlap(check, oppSeg, coinSeg, oppTe, oppTs, coinTe, coinTs, overlaps);
    }
    bool swapOpp = oppTs > oppTe;
    if (swapOpp) {
        using std::swap;
        swap(oppTs, oppTe);
    }
    do {
        if (check->coinPtTStart()->segment() != coinSeg) {
            continue;
        }
        if (check->oppPtTStart()->segment() != oppSeg) {
            continue;
        }
        double checkTs = check->coinPtTStart()->fT;
        double checkTe = check->coinPtTEnd()->fT;
        bool coinOutside = coinTe < checkTs || coinTs > checkTe;
        double oCheckTs = check->oppPtTStart()->fT;
        double oCheckTe = check->oppPtTEnd()->fT;
        if (swapOpp) {
            if (oCheckTs <= oCheckTe) {
                return false;
            }
            using std::swap;
            swap(oCheckTs, oCheckTe);
        }
        bool oppOutside = oppTe < oCheckTs || oppTs > oCheckTe;
        if (coinOutside && oppOutside) {
            continue;
        }
        bool coinInside = coinTe <= checkTe && coinTs >= checkTs;
        bool oppInside = oppTe <= oCheckTe && oppTs >= oCheckTs;
        if (coinInside && oppInside) {  // already included, do nothing
            return false;
        }
        *overlaps->append() = check; // partial overlap, extend existing entry
    } while ((check = check->next()));
    return true;
}

/* Please keep this in sync with debugAddIfMissing() */
// note that over1s, over1e, over2s, over2e are ordered
bool SkOpCoincidence::addIfMissing(const SkOpPtT* over1s, const SkOpPtT* over2s,
        double tStart, double tEnd, SkOpSegment* coinSeg, SkOpSegment* oppSeg, bool* added
        SkDEBUGPARAMS(const SkOpPtT* over1e) SkDEBUGPARAMS(const SkOpPtT* over2e)) {
    SkASSERT(tStart < tEnd);
    SkASSERT(over1s->fT < over1e->fT);
    SkASSERT(between(over1s->fT, tStart, over1e->fT));
    SkASSERT(between(over1s->fT, tEnd, over1e->fT));
    SkASSERT(over2s->fT < over2e->fT);
    SkASSERT(between(over2s->fT, tStart, over2e->fT));
    SkASSERT(between(over2s->fT, tEnd, over2e->fT));
    SkASSERT(over1s->segment() == over1e->segment());
    SkASSERT(over2s->segment() == over2e->segment());
    SkASSERT(over1s->segment() == over2s->segment());
    SkASSERT(over1s->segment() != coinSeg);
    SkASSERT(over1s->segment() != oppSeg);
    SkASSERT(coinSeg != oppSeg);
    double coinTs, coinTe, oppTs, oppTe;
    coinTs = TRange(over1s, tStart, coinSeg  SkDEBUGPARAMS(over1e));
    coinTe = TRange(over1s, tEnd, coinSeg  SkDEBUGPARAMS(over1e));
    SkOpSpanBase::Collapsed result = coinSeg->collapsed(coinTs, coinTe);
    if (SkOpSpanBase::Collapsed::kNo != result) {
        return SkOpSpanBase::Collapsed::kYes == result;
    }
    oppTs = TRange(over2s, tStart, oppSeg  SkDEBUGPARAMS(over2e));
    oppTe = TRange(over2s, tEnd, oppSeg  SkDEBUGPARAMS(over2e));
    result = oppSeg->collapsed(oppTs, oppTe);
    if (SkOpSpanBase::Collapsed::kNo != result) {
        return SkOpSpanBase::Collapsed::kYes == result;
    }
    if (coinTs > coinTe) {
        using std::swap;
        swap(coinTs, coinTe);
        swap(oppTs, oppTe);
    }
    (void) this->addOrOverlap(coinSeg, oppSeg, coinTs, coinTe, oppTs, oppTe, added);
    return true;
}

/* Please keep this in sync with debugAddOrOverlap() */
// If this is called by addEndMovedSpans(), a returned false propogates out to an abort.
// If this is called by AddIfMissing(), a returned false indicates there was nothing to add
bool SkOpCoincidence::addOrOverlap(SkOpSegment* coinSeg, SkOpSegment* oppSeg,
        double coinTs, double coinTe, double oppTs, double oppTe, bool* added) {
    SkTDArray<SkCoincidentSpans*> overlaps;
    FAIL_IF(!fTop);
    if (!this->checkOverlap(fTop, coinSeg, oppSeg, coinTs, coinTe, oppTs, oppTe, &overlaps)) {
        return true;
    }
    if (fHead && !this->checkOverlap(fHead, coinSeg, oppSeg, coinTs,
            coinTe, oppTs, oppTe, &overlaps)) {
        return true;
    }
    SkCoincidentSpans* overlap = overlaps.count() ? overlaps[0] : nullptr;
    for (int index = 1; index < overlaps.count(); ++index) { // combine overlaps before continuing
        SkCoincidentSpans* test = overlaps[index];
        if (overlap->coinPtTStart()->fT > test->coinPtTStart()->fT) {
            overlap->setCoinPtTStart(test->coinPtTStart());
        }
        if (overlap->coinPtTEnd()->fT < test->coinPtTEnd()->fT) {
            overlap->setCoinPtTEnd(test->coinPtTEnd());
        }
        if (overlap->flipped()
                ? overlap->oppPtTStart()->fT < test->oppPtTStart()->fT
                : overlap->oppPtTStart()->fT > test->oppPtTStart()->fT) {
            overlap->setOppPtTStart(test->oppPtTStart());
        }
        if (overlap->flipped()
                ? overlap->oppPtTEnd()->fT > test->oppPtTEnd()->fT
                : overlap->oppPtTEnd()->fT < test->oppPtTEnd()->fT) {
            overlap->setOppPtTEnd(test->oppPtTEnd());
        }
        if (!fHead || !this->release(fHead, test)) {
            SkAssertResult(this->release(fTop, test));
        }
    }
    const SkOpPtT* cs = coinSeg->existing(coinTs, oppSeg);
    const SkOpPtT* ce = coinSeg->existing(coinTe, oppSeg);
    if (overlap && cs && ce && overlap->contains(cs, ce)) {
        return true;
    }
    FAIL_IF(cs == ce && cs);
    const SkOpPtT* os = oppSeg->existing(oppTs, coinSeg);
    const SkOpPtT* oe = oppSeg->existing(oppTe, coinSeg);
    if (overlap && os && oe && overlap->contains(os, oe)) {
        return true;
    }
    FAIL_IF(cs && cs->deleted());
    FAIL_IF(os && os->deleted());
    FAIL_IF(ce && ce->deleted());
    FAIL_IF(oe && oe->deleted());
    const SkOpPtT* csExisting = !cs ? coinSeg->existing(coinTs, nullptr) : nullptr;
    const SkOpPtT* ceExisting = !ce ? coinSeg->existing(coinTe, nullptr) : nullptr;
    FAIL_IF(csExisting && csExisting == ceExisting);
//    FAIL_IF(csExisting && (csExisting == ce ||
//            csExisting->contains(ceExisting ? ceExisting : ce)));
    FAIL_IF(ceExisting && (ceExisting == cs ||
            ceExisting->contains(csExisting ? csExisting : cs)));
    const SkOpPtT* osExisting = !os ? oppSeg->existing(oppTs, nullptr) : nullptr;
    const SkOpPtT* oeExisting = !oe ? oppSeg->existing(oppTe, nullptr) : nullptr;
    FAIL_IF(osExisting && osExisting == oeExisting);
    FAIL_IF(osExisting && (osExisting == oe ||
            osExisting->contains(oeExisting ? oeExisting : oe)));
    FAIL_IF(oeExisting && (oeExisting == os ||
            oeExisting->contains(osExisting ? osExisting : os)));
    // extra line in debug code
    this->debugValidate();
    if (!cs || !os) {
        SkOpPtT* csWritable = cs ? const_cast<SkOpPtT*>(cs)
            : coinSeg->addT(coinTs);
        if (csWritable == ce) {
            return true;
        }
        SkOpPtT* osWritable = os ? const_cast<SkOpPtT*>(os)
            : oppSeg->addT(oppTs);
        FAIL_IF(!csWritable || !osWritable);
        csWritable->span()->addOpp(osWritable->span());
        cs = csWritable;
        os = osWritable->active();
        FAIL_IF(!os);
        FAIL_IF((ce && ce->deleted()) || (oe && oe->deleted()));
    }
    if (!ce || !oe) {
        SkOpPtT* ceWritable = ce ? const_cast<SkOpPtT*>(ce)
            : coinSeg->addT(coinTe);
        SkOpPtT* oeWritable = oe ? const_cast<SkOpPtT*>(oe)
            : oppSeg->addT(oppTe);
        FAIL_IF(!ceWritable->span()->addOpp(oeWritable->span()));
        ce = ceWritable;
        oe = oeWritable;
    }
    this->debugValidate();
    FAIL_IF(cs->deleted());
    FAIL_IF(os->deleted());
    FAIL_IF(ce->deleted());
    FAIL_IF(oe->deleted());
    FAIL_IF(cs->contains(ce) || os->contains(oe));
    bool result = true;
    if (overlap) {
        if (overlap->coinPtTStart()->segment() == coinSeg) {
            result = overlap->extend(cs, ce, os, oe);
        } else {
            if (os->fT > oe->fT) {
                using std::swap;
                swap(cs, ce);
                swap(os, oe);
            }
            result = overlap->extend(os, oe, cs, ce);
        }
#if DEBUG_COINCIDENCE_VERBOSE
        if (result) {
            overlaps[0]->debugShow();
        }
#endif
    } else {
        this->add(cs, ce, os, oe);
#if DEBUG_COINCIDENCE_VERBOSE
        fHead->debugShow();
#endif
    }
    this->debugValidate();
    if (result) {
        *added = true;
    }
    return true;
}

// Please keep this in sync with debugAddMissing()
/* detects overlaps of different coincident runs on same segment */
/* does not detect overlaps for pairs without any segments in common */
// returns true if caller should loop again
bool SkOpCoincidence::addMissing(bool* added  DEBUG_COIN_DECLARE_PARAMS()) {
    SkCoincidentSpans* outer = fHead;
    *added = false;
    if (!outer) {
        return true;
    }
    fTop = outer;
    fHead = nullptr;
    do {
    // addifmissing can modify the list that this is walking
    // save head so that walker can iterate over old data unperturbed
    // addifmissing adds to head freely then add saved head in the end
        const SkOpPtT* ocs = outer->coinPtTStart();
        FAIL_IF(ocs->deleted());
        const SkOpSegment* outerCoin = ocs->segment();
        FAIL_IF(outerCoin->done());
        const SkOpPtT* oos = outer->oppPtTStart();
        if (oos->deleted()) {
            return true;
        }
        const SkOpSegment* outerOpp = oos->segment();
        SkOPASSERT(!outerOpp->done());
        SkOpSegment* outerCoinWritable = const_cast<SkOpSegment*>(outerCoin);
        SkOpSegment* outerOppWritable = const_cast<SkOpSegment*>(outerOpp);
        SkCoincidentSpans* inner = outer;
#ifdef SK_BUILD_FOR_FUZZER
        int safetyNet = 1000;
#endif
        while ((inner = inner->next())) {
#ifdef SK_BUILD_FOR_FUZZER
            if (!--safetyNet) {
                return false;
            }
#endif
            this->debugValidate();
            double overS, overE;
            const SkOpPtT* ics = inner->coinPtTStart();
            FAIL_IF(ics->deleted());
            const SkOpSegment* innerCoin = ics->segment();
            FAIL_IF(innerCoin->done());
            const SkOpPtT* ios = inner->oppPtTStart();
            FAIL_IF(ios->deleted());
            const SkOpSegment* innerOpp = ios->segment();
            SkOPASSERT(!innerOpp->done());
            SkOpSegment* innerCoinWritable = const_cast<SkOpSegment*>(innerCoin);
            SkOpSegment* innerOppWritable = const_cast<SkOpSegment*>(innerOpp);
            if (outerCoin == innerCoin) {
                const SkOpPtT* oce = outer->coinPtTEnd();
                if (oce->deleted()) {
                    return true;
                }
                const SkOpPtT* ice = inner->coinPtTEnd();
                FAIL_IF(ice->deleted());
                if (outerOpp != innerOpp && this->overlap(ocs, oce, ics, ice, &overS, &overE)) {
                    FAIL_IF(!this->addIfMissing(ocs->starter(oce), ics->starter(ice),
                            overS, overE, outerOppWritable, innerOppWritable, added
                            SkDEBUGPARAMS(ocs->debugEnder(oce))
                            SkDEBUGPARAMS(ics->debugEnder(ice))));
                }
            } else if (outerCoin == innerOpp) {
                const SkOpPtT* oce = outer->coinPtTEnd();
                FAIL_IF(oce->deleted());
                const SkOpPtT* ioe = inner->oppPtTEnd();
                FAIL_IF(ioe->deleted());
                if (outerOpp != innerCoin && this->overlap(ocs, oce, ios, ioe, &overS, &overE)) {
                    FAIL_IF(!this->addIfMissing(ocs->starter(oce), ios->starter(ioe),
                            overS, overE, outerOppWritable, innerCoinWritable, added
                            SkDEBUGPARAMS(ocs->debugEnder(oce))
                            SkDEBUGPARAMS(ios->debugEnder(ioe))));
                }
            } else if (outerOpp == innerCoin) {
                const SkOpPtT* ooe = outer->oppPtTEnd();
                FAIL_IF(ooe->deleted());
                const SkOpPtT* ice = inner->coinPtTEnd();
                FAIL_IF(ice->deleted());
                SkASSERT(outerCoin != innerOpp);
                if (this->overlap(oos, ooe, ics, ice, &overS, &overE)) {
                    FAIL_IF(!this->addIfMissing(oos->starter(ooe), ics->starter(ice),
                            overS, overE, outerCoinWritable, innerOppWritable, added
                            SkDEBUGPARAMS(oos->debugEnder(ooe))
                            SkDEBUGPARAMS(ics->debugEnder(ice))));
                }
            } else if (outerOpp == innerOpp) {
                const SkOpPtT* ooe = outer->oppPtTEnd();
                FAIL_IF(ooe->deleted());
                const SkOpPtT* ioe = inner->oppPtTEnd();
                if (ioe->deleted()) {
                    return true;
                }
                SkASSERT(outerCoin != innerCoin);
                if (this->overlap(oos, ooe, ios, ioe, &overS, &overE)) {
                    FAIL_IF(!this->addIfMissing(oos->starter(ooe), ios->starter(ioe),
                            overS, overE, outerCoinWritable, innerCoinWritable, added
                            SkDEBUGPARAMS(oos->debugEnder(ooe))
                            SkDEBUGPARAMS(ios->debugEnder(ioe))));
                }
            }
            this->debugValidate();
        }
    } while ((outer = outer->next()));
    this->restoreHead();
    return true;
}

bool SkOpCoincidence::addOverlap(const SkOpSegment* seg1, const SkOpSegment* seg1o,
        const SkOpSegment* seg2, const SkOpSegment* seg2o,
        const SkOpPtT* overS, const SkOpPtT* overE) {
    const SkOpPtT* s1 = overS->find(seg1);
    const SkOpPtT* e1 = overE->find(seg1);
    FAIL_IF(!s1);
    FAIL_IF(!e1);
    if (!s1->starter(e1)->span()->upCast()->windValue()) {
        s1 = overS->find(seg1o);
        e1 = overE->find(seg1o);
        FAIL_IF(!s1);
        FAIL_IF(!e1);
        if (!s1->starter(e1)->span()->upCast()->windValue()) {
            return true;
        }
    }
    const SkOpPtT* s2 = overS->find(seg2);
    const SkOpPtT* e2 = overE->find(seg2);
    FAIL_IF(!s2);
    FAIL_IF(!e2);
    if (!s2->starter(e2)->span()->upCast()->windValue()) {
        s2 = overS->find(seg2o);
        e2 = overE->find(seg2o);
        FAIL_IF(!s2);
        FAIL_IF(!e2);
        if (!s2->starter(e2)->span()->upCast()->windValue()) {
            return true;
        }
    }
    if (s1->segment() == s2->segment()) {
        return true;
    }
    if (s1->fT > e1->fT) {
        using std::swap;
        swap(s1, e1);
        swap(s2, e2);
    }
    this->add(s1, e1, s2, e2);
    return true;
}

bool SkOpCoincidence::contains(const SkOpSegment* seg, const SkOpSegment* opp, double oppT) const {
    if (this->contains(fHead, seg, opp, oppT)) {
        return true;
    }
    if (this->contains(fTop, seg, opp, oppT)) {
        return true;
    }
    return false;
}

bool SkOpCoincidence::contains(const SkCoincidentSpans* coin, const SkOpSegment* seg,
        const SkOpSegment* opp, double oppT) const {
    if (!coin) {
        return false;
   }
    do {
        if (coin->coinPtTStart()->segment() == seg && coin->oppPtTStart()->segment() == opp
                && between(coin->oppPtTStart()->fT, oppT, coin->oppPtTEnd()->fT)) {
            return true;
        }
        if (coin->oppPtTStart()->segment() == seg && coin->coinPtTStart()->segment() == opp
                && between(coin->coinPtTStart()->fT, oppT, coin->coinPtTEnd()->fT)) {
            return true;
        }
    } while ((coin = coin->next()));
    return false;
}

bool SkOpCoincidence::contains(const SkOpPtT* coinPtTStart, const SkOpPtT* coinPtTEnd,
        const SkOpPtT* oppPtTStart, const SkOpPtT* oppPtTEnd) const {
    const SkCoincidentSpans* test = fHead;
    if (!test) {
        return false;
    }
    const SkOpSegment* coinSeg = coinPtTStart->segment();
    const SkOpSegment* oppSeg = oppPtTStart->segment();
    if (!Ordered(coinPtTStart, oppPtTStart)) {
        using std::swap;
        swap(coinSeg, oppSeg);
        swap(coinPtTStart, oppPtTStart);
        swap(coinPtTEnd, oppPtTEnd);
        if (coinPtTStart->fT > coinPtTEnd->fT) {
            swap(coinPtTStart, coinPtTEnd);
            swap(oppPtTStart, oppPtTEnd);
        }
    }
    double oppMinT = std::min(oppPtTStart->fT, oppPtTEnd->fT);
    double oppMaxT = std::max(oppPtTStart->fT, oppPtTEnd->fT);
    do {
        if (coinSeg != test->coinPtTStart()->segment()) {
            continue;
        }
        if (coinPtTStart->fT < test->coinPtTStart()->fT) {
            continue;
        }
        if (coinPtTEnd->fT > test->coinPtTEnd()->fT) {
            continue;
        }
        if (oppSeg != test->oppPtTStart()->segment()) {
            continue;
        }
        if (oppMinT < std::min(test->oppPtTStart()->fT, test->oppPtTEnd()->fT)) {
            continue;
        }
        if (oppMaxT > std::max(test->oppPtTStart()->fT, test->oppPtTEnd()->fT)) {
            continue;
        }
        return true;
    } while ((test = test->next()));
    return false;
}

void SkOpCoincidence::correctEnds(DEBUG_COIN_DECLARE_ONLY_PARAMS()) {
    DEBUG_SET_PHASE();
    SkCoincidentSpans* coin = fHead;
    if (!coin) {
        return;
    }
    do {
        coin->correctEnds();
    } while ((coin = coin->next()));
}

// walk span sets in parallel, moving winding from one to the other
bool SkOpCoincidence::apply(DEBUG_COIN_DECLARE_ONLY_PARAMS()) {
    DEBUG_SET_PHASE();
    SkCoincidentSpans* coin = fHead;
    if (!coin) {
        return true;
    }
    do {
        SkOpSpanBase* startSpan = coin->coinPtTStartWritable()->span();
        FAIL_IF(!startSpan->upCastable());
        SkOpSpan* start = startSpan->upCast();
        if (start->deleted()) {
            continue;
        }
        const SkOpSpanBase* end = coin->coinPtTEnd()->span();
        FAIL_IF(start != start->starter(end));
        bool flipped = coin->flipped();
        SkOpSpanBase* oStartBase = (flipped ? coin->oppPtTEndWritable()
                : coin->oppPtTStartWritable())->span();
        FAIL_IF(!oStartBase->upCastable());
        SkOpSpan* oStart = oStartBase->upCast();
        if (oStart->deleted()) {
            continue;
        }
        const SkOpSpanBase* oEnd = (flipped ? coin->oppPtTStart() : coin->oppPtTEnd())->span();
        SkASSERT(oStart == oStart->starter(oEnd));
        SkOpSegment* segment = start->segment();
        SkOpSegment* oSegment = oStart->segment();
        bool operandSwap = segment->operand() != oSegment->operand();
        if (flipped) {
            if (oEnd->deleted()) {
                continue;
            }
            do {
                SkOpSpanBase* oNext = oStart->next();
                if (oNext == oEnd) {
                    break;
                }
                FAIL_IF(!oNext->upCastable());
                oStart = oNext->upCast();
            } while (true);
        }
        do {
            int windValue = start->windValue();
            int oppValue = start->oppValue();
            int oWindValue = oStart->windValue();
            int oOppValue = oStart->oppValue();
            // winding values are added or subtracted depending on direction and wind type
            // same or opposite values are summed depending on the operand value
            int windDiff = operandSwap ? oOppValue : oWindValue;
            int oWindDiff = operandSwap ? oppValue : windValue;
            if (!flipped) {
                windDiff = -windDiff;
                oWindDiff = -oWindDiff;
            }
            bool addToStart = windValue && (windValue > windDiff || (windValue == windDiff
                    && oWindValue <= oWindDiff));
            if (addToStart ? start->done() : oStart->done()) {
                addToStart ^= true;
            }
            if (addToStart) {
                if (operandSwap) {
                    using std::swap;
                    swap(oWindValue, oOppValue);
                }
                if (flipped) {
                    windValue -= oWindValue;
                    oppValue -= oOppValue;
                } else {
                    windValue += oWindValue;
                    oppValue += oOppValue;
                }
                if (segment->isXor()) {
                    windValue &= 1;
                }
                if (segment->oppXor()) {
                    oppValue &= 1;
                }
                oWindValue = oOppValue = 0;
            } else {
                if (operandSwap) {
                    using std::swap;
                    swap(windValue, oppValue);
                }
                if (flipped) {
                    oWindValue -= windValue;
                    oOppValue -= oppValue;
                } else {
                    oWindValue += windValue;
                    oOppValue += oppValue;
                }
                if (oSegment->isXor()) {
                    oWindValue &= 1;
                }
                if (oSegment->oppXor()) {
                    oOppValue &= 1;
                }
                windValue = oppValue = 0;
            }
#if 0 && DEBUG_COINCIDENCE
            SkDebugf("seg=%d span=%d windValue=%d oppValue=%d\n", segment->debugID(),
                    start->debugID(), windValue, oppValue);
            SkDebugf("seg=%d span=%d windValue=%d oppValue=%d\n", oSegment->debugID(),
                    oStart->debugID(), oWindValue, oOppValue);
#endif
            FAIL_IF(windValue <= -1);
            start->setWindValue(windValue);
            start->setOppValue(oppValue);
            FAIL_IF(oWindValue <= -1);
            oStart->setWindValue(oWindValue);
            oStart->setOppValue(oOppValue);
            if (!windValue && !oppValue) {
                segment->markDone(start);
            }
            if (!oWindValue && !oOppValue) {
                oSegment->markDone(oStart);
            }
            SkOpSpanBase* next = start->next();
            SkOpSpanBase* oNext = flipped ? oStart->prev() : oStart->next();
            if (next == end) {
                break;
            }
            FAIL_IF(!next->upCastable());
            start = next->upCast();
            // if the opposite ran out too soon, just reuse the last span
            if (!oNext || !oNext->upCastable()) {
               oNext = oStart;
            }
            oStart = oNext->upCast();
        } while (true);
    } while ((coin = coin->next()));
    return true;
}

// Please keep this in sync with debugRelease()
bool SkOpCoincidence::release(SkCoincidentSpans* coin, SkCoincidentSpans* remove)  {
    SkCoincidentSpans* head = coin;
    SkCoincidentSpans* prev = nullptr;
    SkCoincidentSpans* next;
    do {
        next = coin->next();
        if (coin == remove) {
            if (prev) {
                prev->setNext(next);
            } else if (head == fHead) {
                fHead = next;
            } else {
                fTop = next;
            }
            break;
        }
        prev = coin;
    } while ((coin = next));
    return coin != nullptr;
}

void SkOpCoincidence::releaseDeleted(SkCoincidentSpans* coin) {
    if (!coin) {
        return;
    }
    SkCoincidentSpans* head = coin;
    SkCoincidentSpans* prev = nullptr;
    SkCoincidentSpans* next;
    do {
        next = coin->next();
        if (coin->coinPtTStart()->deleted()) {
            SkOPASSERT(coin->flipped() ? coin->oppPtTEnd()->deleted() :
                    coin->oppPtTStart()->deleted());
            if (prev) {
                prev->setNext(next);
            } else if (head == fHead) {
                fHead = next;
            } else {
                fTop = next;
            }
        } else {
             SkOPASSERT(coin->flipped() ? !coin->oppPtTEnd()->deleted() :
                    !coin->oppPtTStart()->deleted());
            prev = coin;
        }
    } while ((coin = next));
}

void SkOpCoincidence::releaseDeleted() {
    this->releaseDeleted(fHead);
    this->releaseDeleted(fTop);
}

void SkOpCoincidence::restoreHead() {
    SkCoincidentSpans** headPtr = &fHead;
    while (*headPtr) {
        headPtr = (*headPtr)->nextPtr();
    }
    *headPtr = fTop;
    fTop = nullptr;
    // segments may have collapsed in the meantime; remove empty referenced segments
    headPtr = &fHead;
    while (*headPtr) {
        SkCoincidentSpans* test = *headPtr;
        if (test->coinPtTStart()->segment()->done() || test->oppPtTStart()->segment()->done()) {
            *headPtr = test->next();
            continue;
        }
        headPtr = (*headPtr)->nextPtr();
    }
}

// Please keep this in sync with debugExpand()
// expand the range by checking adjacent spans for coincidence
bool SkOpCoincidence::expand(DEBUG_COIN_DECLARE_ONLY_PARAMS()) {
    DEBUG_SET_PHASE();
    SkCoincidentSpans* coin = fHead;
    if (!coin) {
        return false;
    }
    bool expanded = false;
    do {
        if (coin->expand()) {
            // check to see if multiple spans expanded so they are now identical
            SkCoincidentSpans* test = fHead;
            do {
                if (coin == test) {
                    continue;
                }
                if (coin->coinPtTStart() == test->coinPtTStart()
                        && coin->oppPtTStart() == test->oppPtTStart()) {
                    this->release(fHead, test);
                    break;
                }
            } while ((test = test->next()));
            expanded = true;
        }
    } while ((coin = coin->next()));
    return expanded;
}

bool SkOpCoincidence::findOverlaps(SkOpCoincidence* overlaps  DEBUG_COIN_DECLARE_PARAMS()) const {
    DEBUG_SET_PHASE();
    overlaps->fHead = overlaps->fTop = nullptr;
    SkCoincidentSpans* outer = fHead;
    while (outer) {
        const SkOpSegment* outerCoin = outer->coinPtTStart()->segment();
        const SkOpSegment* outerOpp = outer->oppPtTStart()->segment();
        SkCoincidentSpans* inner = outer;
        while ((inner = inner->next())) {
            const SkOpSegment* innerCoin = inner->coinPtTStart()->segment();
            if (outerCoin == innerCoin) {
                continue;  // both winners are the same segment, so there's no additional overlap
            }
            const SkOpSegment* innerOpp = inner->oppPtTStart()->segment();
            const SkOpPtT* overlapS;
            const SkOpPtT* overlapE;
            if ((outerOpp == innerCoin && SkOpPtT::Overlaps(outer->oppPtTStart(),
                    outer->oppPtTEnd(),inner->coinPtTStart(), inner->coinPtTEnd(), &overlapS,
                    &overlapE))
                    || (outerCoin == innerOpp && SkOpPtT::Overlaps(outer->coinPtTStart(),
                    outer->coinPtTEnd(), inner->oppPtTStart(), inner->oppPtTEnd(),
                    &overlapS, &overlapE))
                    || (outerOpp == innerOpp && SkOpPtT::Overlaps(outer->oppPtTStart(),
                    outer->oppPtTEnd(), inner->oppPtTStart(), inner->oppPtTEnd(),
                    &overlapS, &overlapE))) {
                if (!overlaps->addOverlap(outerCoin, outerOpp, innerCoin, innerOpp,
                        overlapS, overlapE)) {
                    return false;
                }
             }
        }
        outer = outer->next();
    }
    return true;
}

void SkOpCoincidence::fixUp(SkOpPtT* deleted, const SkOpPtT* kept) {
    SkOPASSERT(deleted != kept);
    if (fHead) {
        this->fixUp(fHead, deleted, kept);
    }
    if (fTop) {
        this->fixUp(fTop, deleted, kept);
    }
}

void SkOpCoincidence::fixUp(SkCoincidentSpans* coin, SkOpPtT* deleted, const SkOpPtT* kept) {
    SkCoincidentSpans* head = coin;
    do {
        if (coin->coinPtTStart() == deleted) {
            if (coin->coinPtTEnd()->span() == kept->span()) {
                this->release(head, coin);
                continue;
            }
            coin->setCoinPtTStart(kept);
        }
        if (coin->coinPtTEnd() == deleted) {
            if (coin->coinPtTStart()->span() == kept->span()) {
                this->release(head, coin);
                continue;
            }
            coin->setCoinPtTEnd(kept);
       }
        if (coin->oppPtTStart() == deleted) {
            if (coin->oppPtTEnd()->span() == kept->span()) {
                this->release(head, coin);
                continue;
            }
            coin->setOppPtTStart(kept);
        }
        if (coin->oppPtTEnd() == deleted) {
            if (coin->oppPtTStart()->span() == kept->span()) {
                this->release(head, coin);
                continue;
            }
            coin->setOppPtTEnd(kept);
        }
    } while ((coin = coin->next()));
}

// Please keep this in sync with debugMark()
/* this sets up the coincidence links in the segments when the coincidence crosses multiple spans */
bool SkOpCoincidence::mark(DEBUG_COIN_DECLARE_ONLY_PARAMS()) {
    DEBUG_SET_PHASE();
    SkCoincidentSpans* coin = fHead;
    if (!coin) {
        return true;
    }
    do {
        SkOpSpanBase* startBase = coin->coinPtTStartWritable()->span();
        FAIL_IF(!startBase->upCastable());
        SkOpSpan* start = startBase->upCast();
        FAIL_IF(start->deleted());
        SkOpSpanBase* end = coin->coinPtTEndWritable()->span();
        SkOPASSERT(!end->deleted());
        SkOpSpanBase* oStart = coin->oppPtTStartWritable()->span();
        SkOPASSERT(!oStart->deleted());
        SkOpSpanBase* oEnd = coin->oppPtTEndWritable()->span();
        FAIL_IF(oEnd->deleted());
        bool flipped = coin->flipped();
        if (flipped) {
            using std::swap;
            swap(oStart, oEnd);
        }
        /* coin and opp spans may not match up. Mark the ends, and then let the interior
           get marked as many times as the spans allow */
        FAIL_IF(!oStart->upCastable());
        start->insertCoincidence(oStart->upCast());
        end->insertCoinEnd(oEnd);
        const SkOpSegment* segment = start->segment();
        const SkOpSegment* oSegment = oStart->segment();
        SkOpSpanBase* next = start;
        SkOpSpanBase* oNext = oStart;
        bool ordered;
        FAIL_IF(!coin->ordered(&ordered));
        while ((next = next->upCast()->next()) != end) {
            FAIL_IF(!next->upCastable());
            FAIL_IF(!next->upCast()->insertCoincidence(oSegment, flipped, ordered));
        }
        while ((oNext = oNext->upCast()->next()) != oEnd) {
            FAIL_IF(!oNext->upCastable());
            FAIL_IF(!oNext->upCast()->insertCoincidence(segment, flipped, ordered));
        }
    } while ((coin = coin->next()));
    return true;
}

// Please keep in sync with debugMarkCollapsed()
void SkOpCoincidence::markCollapsed(SkCoincidentSpans* coin, SkOpPtT* test) {
    SkCoincidentSpans* head = coin;
    while (coin) {
        if (coin->collapsed(test)) {
            if (zero_or_one(coin->coinPtTStart()->fT) && zero_or_one(coin->coinPtTEnd()->fT)) {
                coin->coinPtTStartWritable()->segment()->markAllDone();
            }
            if (zero_or_one(coin->oppPtTStart()->fT) && zero_or_one(coin->oppPtTEnd()->fT)) {
                coin->oppPtTStartWritable()->segment()->markAllDone();
            }
            this->release(head, coin);
        }
        coin = coin->next();
    }
}

// Please keep in sync with debugMarkCollapsed()
void SkOpCoincidence::markCollapsed(SkOpPtT* test) {
    markCollapsed(fHead, test);
    markCollapsed(fTop, test);
}

bool SkOpCoincidence::Ordered(const SkOpSegment* coinSeg, const SkOpSegment* oppSeg) {
    if (coinSeg->verb() < oppSeg->verb()) {
        return true;
    }
    if (coinSeg->verb() > oppSeg->verb()) {
        return false;
    }
    int count = (SkPathOpsVerbToPoints(coinSeg->verb()) + 1) * 2;
    const SkScalar* cPt = &coinSeg->pts()[0].fX;
    const SkScalar* oPt = &oppSeg->pts()[0].fX;
    for (int index = 0; index < count; ++index) {
        if (*cPt < *oPt) {
            return true;
        }
        if (*cPt > *oPt) {
            return false;
        }
        ++cPt;
        ++oPt;
    }
    return true;
}

bool SkOpCoincidence::overlap(const SkOpPtT* coin1s, const SkOpPtT* coin1e,
        const SkOpPtT* coin2s, const SkOpPtT* coin2e, double* overS, double* overE) const {
    SkASSERT(coin1s->segment() == coin2s->segment());
    *overS = std::max(std::min(coin1s->fT, coin1e->fT), std::min(coin2s->fT, coin2e->fT));
    *overE = std::min(std::max(coin1s->fT, coin1e->fT), std::max(coin2s->fT, coin2e->fT));
    return *overS < *overE;
}

// Commented-out lines keep this in sync with debugRelease()
void SkOpCoincidence::release(const SkOpSegment* deleted) {
    SkCoincidentSpans* coin = fHead;
    if (!coin) {
        return;
    }
    do {
        if (coin->coinPtTStart()->segment() == deleted
                || coin->coinPtTEnd()->segment() == deleted
                || coin->oppPtTStart()->segment() == deleted
                || coin->oppPtTEnd()->segment() == deleted) {
            this->release(fHead, coin);
        }
    } while ((coin = coin->next()));
}
