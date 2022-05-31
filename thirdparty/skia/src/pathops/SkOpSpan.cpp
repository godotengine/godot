/*
 * Copyright 2014 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/pathops/SkOpCoincidence.h"
#include "src/pathops/SkOpContour.h"
#include "src/pathops/SkOpSegment.h"
#include "src/pathops/SkPathWriter.h"

bool SkOpPtT::alias() const {
    return this->span()->ptT() != this;
}

const SkOpPtT* SkOpPtT::active() const {
    if (!fDeleted) {
        return this;
    }
    const SkOpPtT* ptT = this;
    const SkOpPtT* stopPtT = ptT;
    while ((ptT = ptT->next()) != stopPtT) {
        if (ptT->fSpan == fSpan && !ptT->fDeleted) {
            return ptT;
        }
    }
    return nullptr; // should never return deleted; caller must abort
}

bool SkOpPtT::contains(const SkOpPtT* check) const {
    SkOPASSERT(this != check);
    const SkOpPtT* ptT = this;
    const SkOpPtT* stopPtT = ptT;
    while ((ptT = ptT->next()) != stopPtT) {
        if (ptT == check) {
            return true;
        }
    }
    return false;
}

bool SkOpPtT::contains(const SkOpSegment* segment, const SkPoint& pt) const {
    SkASSERT(this->segment() != segment);
    const SkOpPtT* ptT = this;
    const SkOpPtT* stopPtT = ptT;
    while ((ptT = ptT->next()) != stopPtT) {
        if (ptT->fPt == pt && ptT->segment() == segment) {
            return true;
        }
    }
    return false;
}

bool SkOpPtT::contains(const SkOpSegment* segment, double t) const {
    const SkOpPtT* ptT = this;
    const SkOpPtT* stopPtT = ptT;
    while ((ptT = ptT->next()) != stopPtT) {
        if (ptT->fT == t && ptT->segment() == segment) {
            return true;
        }
    }
    return false;
}

const SkOpPtT* SkOpPtT::contains(const SkOpSegment* check) const {
    SkASSERT(this->segment() != check);
    const SkOpPtT* ptT = this;
    const SkOpPtT* stopPtT = ptT;
    while ((ptT = ptT->next()) != stopPtT) {
        if (ptT->segment() == check && !ptT->deleted()) {
            return ptT;
        }
    }
    return nullptr;
}

SkOpContour* SkOpPtT::contour() const {
    return segment()->contour();
}

const SkOpPtT* SkOpPtT::find(const SkOpSegment* segment) const {
    const SkOpPtT* ptT = this;
    const SkOpPtT* stopPtT = ptT;
    do {
        if (ptT->segment() == segment && !ptT->deleted()) {
            return ptT;
        }
        ptT = ptT->fNext;
    } while (stopPtT != ptT);
//    SkASSERT(0);
    return nullptr;
}

SkOpGlobalState* SkOpPtT::globalState() const {
    return contour()->globalState();
}

void SkOpPtT::init(SkOpSpanBase* span, double t, const SkPoint& pt, bool duplicate) {
    fT = t;
    fPt = pt;
    fSpan = span;
    fNext = this;
    fDuplicatePt = duplicate;
    fDeleted = false;
    fCoincident = false;
    SkDEBUGCODE(fID = span->globalState()->nextPtTID());
}

bool SkOpPtT::onEnd() const {
    const SkOpSpanBase* span = this->span();
    if (span->ptT() != this) {
        return false;
    }
    const SkOpSegment* segment = this->segment();
    return span == segment->head() || span == segment->tail();
}

bool SkOpPtT::ptAlreadySeen(const SkOpPtT* check) const {
    while (this != check) {
        if (this->fPt == check->fPt) {
            return true;
        }
        check = check->fNext;
    }
    return false;
}

SkOpPtT* SkOpPtT::prev() {
    SkOpPtT* result = this;
    SkOpPtT* next = this;
    while ((next = next->fNext) != this) {
        result = next;
    }
    SkASSERT(result->fNext == this);
    return result;
}

const SkOpSegment* SkOpPtT::segment() const {
    return span()->segment();
}

SkOpSegment* SkOpPtT::segment() {
    return span()->segment();
}

void SkOpPtT::setDeleted() {
    SkASSERT(this->span()->debugDeleted() || this->span()->ptT() != this);
    SkOPASSERT(!fDeleted);
    fDeleted = true;
}

bool SkOpSpanBase::addOpp(SkOpSpanBase* opp) {
    SkOpPtT* oppPrev = this->ptT()->oppPrev(opp->ptT());
    if (!oppPrev) {
        return true;
    }
    FAIL_IF(!this->mergeMatches(opp));
    this->ptT()->addOpp(opp->ptT(), oppPrev);
    this->checkForCollapsedCoincidence();
    return true;
}

SkOpSpanBase::Collapsed SkOpSpanBase::collapsed(double s, double e) const {
    const SkOpPtT* start = &fPtT;
    const SkOpPtT* startNext = nullptr;
    const SkOpPtT* walk = start;
    double min = walk->fT;
    double max = min;
    const SkOpSegment* segment = this->segment();
    int safetyNet = 100000;
    while ((walk = walk->next()) != start) {
        if (!--safetyNet) {
            return Collapsed::kError;
        }
        if (walk == startNext) {
            return Collapsed::kError;
        }
        if (walk->segment() != segment) {
            continue;
        }
        min = std::min(min, walk->fT);
        max = std::max(max, walk->fT);
        if (between(min, s, max) && between(min, e, max)) {
            return Collapsed::kYes;
        }
        startNext = start->next();
    }
    return Collapsed::kNo;
}

bool SkOpSpanBase::contains(const SkOpSpanBase* span) const {
    const SkOpPtT* start = &fPtT;
    const SkOpPtT* check = &span->fPtT;
    SkOPASSERT(start != check);
    const SkOpPtT* walk = start;
    while ((walk = walk->next()) != start) {
        if (walk == check) {
            return true;
        }
    }
    return false;
}

const SkOpPtT* SkOpSpanBase::contains(const SkOpSegment* segment) const {
    const SkOpPtT* start = &fPtT;
    const SkOpPtT* walk = start;
    while ((walk = walk->next()) != start) {
        if (walk->deleted()) {
            continue;
        }
        if (walk->segment() == segment && walk->span()->ptT() == walk) {
            return walk;
        }
    }
    return nullptr;
}

bool SkOpSpanBase::containsCoinEnd(const SkOpSegment* segment) const {
    SkASSERT(this->segment() != segment);
    const SkOpSpanBase* next = this;
    while ((next = next->fCoinEnd) != this) {
        if (next->segment() == segment) {
            return true;
        }
    }
    return false;
}

SkOpContour* SkOpSpanBase::contour() const {
    return segment()->contour();
}

SkOpGlobalState* SkOpSpanBase::globalState() const {
    return contour()->globalState();
}

void SkOpSpanBase::initBase(SkOpSegment* segment, SkOpSpan* prev, double t, const SkPoint& pt) {
    fSegment = segment;
    fPtT.init(this, t, pt, false);
    fCoinEnd = this;
    fFromAngle = nullptr;
    fPrev = prev;
    fSpanAdds = 0;
    fAligned = true;
    fChased = false;
    SkDEBUGCODE(fCount = 1);
    SkDEBUGCODE(fID = globalState()->nextSpanID());
    SkDEBUGCODE(fDebugDeleted = false);
}

// this pair of spans share a common t value or point; merge them and eliminate duplicates
// this does not compute the best t or pt value; this merely moves all data into a single list
void SkOpSpanBase::merge(SkOpSpan* span) {
    SkOpPtT* spanPtT = span->ptT();
    SkASSERT(this->t() != spanPtT->fT);
    SkASSERT(!zero_or_one(spanPtT->fT));
    span->release(this->ptT());
    if (this->contains(span)) {
        SkOPASSERT(0);  // check to see if this ever happens -- should have been found earlier
        return;  // merge is already in the ptT loop
    }
    SkOpPtT* remainder = spanPtT->next();
    this->ptT()->insert(spanPtT);
    while (remainder != spanPtT) {
        SkOpPtT* next = remainder->next();
        SkOpPtT* compare = spanPtT->next();
        while (compare != spanPtT) {
            SkOpPtT* nextC = compare->next();
            if (nextC->span() == remainder->span() && nextC->fT == remainder->fT) {
                goto tryNextRemainder;
            }
            compare = nextC;
        }
        spanPtT->insert(remainder);
tryNextRemainder:
        remainder = next;
    }
    fSpanAdds += span->fSpanAdds;
}

// please keep in sync with debugCheckForCollapsedCoincidence()
void SkOpSpanBase::checkForCollapsedCoincidence() {
    SkOpCoincidence* coins = this->globalState()->coincidence();
    if (coins->isEmpty()) {
        return;
    }
// the insert above may have put both ends of a coincident run in the same span
// for each coincident ptT in loop; see if its opposite in is also in the loop
// this implementation is the motivation for marking that a ptT is referenced by a coincident span
    SkOpPtT* head = this->ptT();
    SkOpPtT* test = head;
    do {
        if (!test->coincident()) {
            continue;
        }
        coins->markCollapsed(test);
    } while ((test = test->next()) != head);
    coins->releaseDeleted();
}

// please keep in sync with debugMergeMatches()
// Look to see if pt-t linked list contains same segment more than once
// if so, and if each pt-t is directly pointed to by spans in that segment,
// merge them
// keep the points, but remove spans so that the segment doesn't have 2 or more
// spans pointing to the same pt-t loop at different loop elements
bool SkOpSpanBase::mergeMatches(SkOpSpanBase* opp) {
    SkOpPtT* test = &fPtT;
    SkOpPtT* testNext;
    const SkOpPtT* stop = test;
    int safetyHatch = 1000000;
    do {
        if (!--safetyHatch) {
            return false;
        }
        testNext = test->next();
        if (test->deleted()) {
            continue;
        }
        SkOpSpanBase* testBase = test->span();
        SkASSERT(testBase->ptT() == test);
        SkOpSegment* segment = test->segment();
        if (segment->done()) {
            continue;
        }
        SkOpPtT* inner = opp->ptT();
        const SkOpPtT* innerStop = inner;
        do {
            if (inner->segment() != segment) {
                continue;
            }
            if (inner->deleted()) {
                continue;
            }
            SkOpSpanBase* innerBase = inner->span();
            SkASSERT(innerBase->ptT() == inner);
            // when the intersection is first detected, the span base is marked if there are
            // more than one point in the intersection.
            if (!zero_or_one(inner->fT)) {
                innerBase->upCast()->release(test);
            } else {
                SkOPASSERT(inner->fT != test->fT);
                if (!zero_or_one(test->fT)) {
                    testBase->upCast()->release(inner);
                } else {
                    segment->markAllDone();  // mark segment as collapsed
                    SkDEBUGCODE(testBase->debugSetDeleted());
                    test->setDeleted();
                    SkDEBUGCODE(innerBase->debugSetDeleted());
                    inner->setDeleted();
                }
            }
#ifdef SK_DEBUG   // assert if another undeleted entry points to segment
            const SkOpPtT* debugInner = inner;
            while ((debugInner = debugInner->next()) != innerStop) {
                if (debugInner->segment() != segment) {
                    continue;
                }
                if (debugInner->deleted()) {
                    continue;
                }
                SkOPASSERT(0);
            }
#endif
            break;
        } while ((inner = inner->next()) != innerStop);
    } while ((test = testNext) != stop);
    this->checkForCollapsedCoincidence();
    return true;
}

int SkOpSpan::computeWindSum() {
    SkOpGlobalState* globals = this->globalState();
    SkOpContour* contourHead = globals->contourHead();
    int windTry = 0;
    while (!this->sortableTop(contourHead) && ++windTry < SkOpGlobalState::kMaxWindingTries) {
    }
    return this->windSum();
}

bool SkOpSpan::containsCoincidence(const SkOpSegment* segment) const {
    SkASSERT(this->segment() != segment);
    const SkOpSpan* next = fCoincident;
    do {
        if (next->segment() == segment) {
            return true;
        }
    } while ((next = next->fCoincident) != this);
    return false;
}

void SkOpSpan::init(SkOpSegment* segment, SkOpSpan* prev, double t, const SkPoint& pt) {
    SkASSERT(t != 1);
    initBase(segment, prev, t, pt);
    fCoincident = this;
    fToAngle = nullptr;
    fWindSum = fOppSum = SK_MinS32;
    fWindValue = 1;
    fOppValue = 0;
    fTopTTry = 0;
    fChased = fDone = false;
    segment->bumpCount();
    fAlreadyAdded = false;
}

// Please keep this in sync with debugInsertCoincidence()
bool SkOpSpan::insertCoincidence(const SkOpSegment* segment, bool flipped, bool ordered) {
    if (this->containsCoincidence(segment)) {
        return true;
    }
    SkOpPtT* next = &fPtT;
    while ((next = next->next()) != &fPtT) {
        if (next->segment() == segment) {
            SkOpSpan* span;
            SkOpSpanBase* base = next->span();
            if (!ordered) {
                const SkOpPtT* spanEndPtT = fNext->contains(segment);
                FAIL_IF(!spanEndPtT);
                const SkOpSpanBase* spanEnd = spanEndPtT->span();
                const SkOpPtT* start = base->ptT()->starter(spanEnd->ptT());
                FAIL_IF(!start->span()->upCastable());
                span = const_cast<SkOpSpan*>(start->span()->upCast());
            } else if (flipped) {
                span = base->prev();
                FAIL_IF(!span);
            } else {
                FAIL_IF(!base->upCastable());
                span = base->upCast();
            }
            this->insertCoincidence(span);
            return true;
        }
    }
#if DEBUG_COINCIDENCE
    SkASSERT(0); // FIXME? if we get here, the span is missing its opposite segment...
#endif
    return true;
}

void SkOpSpan::release(const SkOpPtT* kept) {
    SkDEBUGCODE(fDebugDeleted = true);
    SkOPASSERT(kept->span() != this);
    SkASSERT(!final());
    SkOpSpan* prev = this->prev();
    SkASSERT(prev);
    SkOpSpanBase* next = this->next();
    SkASSERT(next);
    prev->setNext(next);
    next->setPrev(prev);
    this->segment()->release(this);
    SkOpCoincidence* coincidence = this->globalState()->coincidence();
    if (coincidence) {
        coincidence->fixUp(this->ptT(), kept);
    }
    this->ptT()->setDeleted();
    SkOpPtT* stopPtT = this->ptT();
    SkOpPtT* testPtT = stopPtT;
    const SkOpSpanBase* keptSpan = kept->span();
    do {
        if (this == testPtT->span()) {
            testPtT->setSpan(keptSpan);
        }
    } while ((testPtT = testPtT->next()) != stopPtT);
}

void SkOpSpan::setOppSum(int oppSum) {
    SkASSERT(!final());
    if (fOppSum != SK_MinS32 && fOppSum != oppSum) {
        this->globalState()->setWindingFailed();
        return;
    }
    SkASSERT(!DEBUG_LIMIT_WIND_SUM || SkTAbs(oppSum) <= DEBUG_LIMIT_WIND_SUM);
    fOppSum = oppSum;
}

void SkOpSpan::setWindSum(int windSum) {
    SkASSERT(!final());
    if (fWindSum != SK_MinS32 && fWindSum != windSum) {
        this->globalState()->setWindingFailed();
        return;
    }
    SkASSERT(!DEBUG_LIMIT_WIND_SUM || SkTAbs(windSum) <= DEBUG_LIMIT_WIND_SUM);
    fWindSum = windSum;
}
