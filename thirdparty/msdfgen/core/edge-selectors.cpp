
#include "edge-selectors.h"

#include "arithmetics.hpp"

namespace msdfgen {

#define DISTANCE_DELTA_FACTOR 1.001

TrueDistanceSelector::EdgeCache::EdgeCache() : absDistance(0) { }

void TrueDistanceSelector::reset(const Point2 &p) {
    double delta = DISTANCE_DELTA_FACTOR*(p-this->p).length();
    minDistance.distance += nonZeroSign(minDistance.distance)*delta;
    this->p = p;
}

void TrueDistanceSelector::addEdge(EdgeCache &cache, const EdgeSegment *prevEdge, const EdgeSegment *edge, const EdgeSegment *nextEdge) {
    double delta = DISTANCE_DELTA_FACTOR*(p-cache.point).length();
    if (cache.absDistance-delta <= fabs(minDistance.distance)) {
        double dummy;
        SignedDistance distance = edge->signedDistance(p, dummy);
        if (distance < minDistance)
            minDistance = distance;
        cache.point = p;
        cache.absDistance = fabs(distance.distance);
    }
}

void TrueDistanceSelector::merge(const TrueDistanceSelector &other) {
    if (other.minDistance < minDistance)
        minDistance = other.minDistance;
}

TrueDistanceSelector::DistanceType TrueDistanceSelector::distance() const {
    return minDistance.distance;
}

PseudoDistanceSelectorBase::EdgeCache::EdgeCache() : absDistance(0), aDomainDistance(0), bDomainDistance(0), aPseudoDistance(0), bPseudoDistance(0) { }

bool PseudoDistanceSelectorBase::getPseudoDistance(double &distance, const Vector2 &ep, const Vector2 &edgeDir) {
    double ts = dotProduct(ep, edgeDir);
    if (ts > 0) {
        double pseudoDistance = crossProduct(ep, edgeDir);
        if (fabs(pseudoDistance) < fabs(distance)) {
            distance = pseudoDistance;
            return true;
        }
    }
    return false;
}

PseudoDistanceSelectorBase::PseudoDistanceSelectorBase() : minNegativePseudoDistance(-fabs(minTrueDistance.distance)), minPositivePseudoDistance(fabs(minTrueDistance.distance)), nearEdge(NULL), nearEdgeParam(0) { }

void PseudoDistanceSelectorBase::reset(double delta) {
    minTrueDistance.distance += nonZeroSign(minTrueDistance.distance)*delta;
    minNegativePseudoDistance = -fabs(minTrueDistance.distance);
    minPositivePseudoDistance = fabs(minTrueDistance.distance);
    nearEdge = NULL;
    nearEdgeParam = 0;
}

bool PseudoDistanceSelectorBase::isEdgeRelevant(const EdgeCache &cache, const EdgeSegment *edge, const Point2 &p) const {
    double delta = DISTANCE_DELTA_FACTOR*(p-cache.point).length();
    return (
        cache.absDistance-delta <= fabs(minTrueDistance.distance) ||
        fabs(cache.aDomainDistance) < delta ||
        fabs(cache.bDomainDistance) < delta ||
        (cache.aDomainDistance > 0 && (cache.aPseudoDistance < 0 ?
            cache.aPseudoDistance+delta >= minNegativePseudoDistance :
            cache.aPseudoDistance-delta <= minPositivePseudoDistance
        )) ||
        (cache.bDomainDistance > 0 && (cache.bPseudoDistance < 0 ?
            cache.bPseudoDistance+delta >= minNegativePseudoDistance :
            cache.bPseudoDistance-delta <= minPositivePseudoDistance
        ))
    );
}

void PseudoDistanceSelectorBase::addEdgeTrueDistance(const EdgeSegment *edge, const SignedDistance &distance, double param) {
    if (distance < minTrueDistance) {
        minTrueDistance = distance;
        nearEdge = edge;
        nearEdgeParam = param;
    }
}

void PseudoDistanceSelectorBase::addEdgePseudoDistance(double distance) {
    if (distance <= 0 && distance > minNegativePseudoDistance)
        minNegativePseudoDistance = distance;
    if (distance >= 0 && distance < minPositivePseudoDistance)
        minPositivePseudoDistance = distance;
}

void PseudoDistanceSelectorBase::merge(const PseudoDistanceSelectorBase &other) {
    if (other.minTrueDistance < minTrueDistance) {
        minTrueDistance = other.minTrueDistance;
        nearEdge = other.nearEdge;
        nearEdgeParam = other.nearEdgeParam;
    }
    if (other.minNegativePseudoDistance > minNegativePseudoDistance)
        minNegativePseudoDistance = other.minNegativePseudoDistance;
    if (other.minPositivePseudoDistance < minPositivePseudoDistance)
        minPositivePseudoDistance = other.minPositivePseudoDistance;
}

double PseudoDistanceSelectorBase::computeDistance(const Point2 &p) const {
    double minDistance = minTrueDistance.distance < 0 ? minNegativePseudoDistance : minPositivePseudoDistance;
    if (nearEdge) {
        SignedDistance distance = minTrueDistance;
        nearEdge->distanceToPseudoDistance(distance, p, nearEdgeParam);
        if (fabs(distance.distance) < fabs(minDistance))
            minDistance = distance.distance;
    }
    return minDistance;
}

SignedDistance PseudoDistanceSelectorBase::trueDistance() const {
    return minTrueDistance;
}

void PseudoDistanceSelector::reset(const Point2 &p) {
    double delta = DISTANCE_DELTA_FACTOR*(p-this->p).length();
    PseudoDistanceSelectorBase::reset(delta);
    this->p = p;
}

void PseudoDistanceSelector::addEdge(EdgeCache &cache, const EdgeSegment *prevEdge, const EdgeSegment *edge, const EdgeSegment *nextEdge) {
    if (isEdgeRelevant(cache, edge, p)) {
        double param;
        SignedDistance distance = edge->signedDistance(p, param);
        addEdgeTrueDistance(edge, distance, param);
        cache.point = p;
        cache.absDistance = fabs(distance.distance);

        Vector2 ap = p-edge->point(0);
        Vector2 bp = p-edge->point(1);
        Vector2 aDir = edge->direction(0).normalize(true);
        Vector2 bDir = edge->direction(1).normalize(true);
        Vector2 prevDir = prevEdge->direction(1).normalize(true);
        Vector2 nextDir = nextEdge->direction(0).normalize(true);
        double add = dotProduct(ap, (prevDir+aDir).normalize(true));
        double bdd = -dotProduct(bp, (bDir+nextDir).normalize(true));
        if (add > 0) {
            double pd = distance.distance;
            if (getPseudoDistance(pd, ap, -aDir))
                addEdgePseudoDistance(pd = -pd);
            cache.aPseudoDistance = pd;
        }
        if (bdd > 0) {
            double pd = distance.distance;
            if (getPseudoDistance(pd, bp, bDir))
                addEdgePseudoDistance(pd);
            cache.bPseudoDistance = pd;
        }
        cache.aDomainDistance = add;
        cache.bDomainDistance = bdd;
    }
}

PseudoDistanceSelector::DistanceType PseudoDistanceSelector::distance() const {
    return computeDistance(p);
}

void MultiDistanceSelector::reset(const Point2 &p) {
    double delta = DISTANCE_DELTA_FACTOR*(p-this->p).length();
    r.reset(delta);
    g.reset(delta);
    b.reset(delta);
    this->p = p;
}

void MultiDistanceSelector::addEdge(EdgeCache &cache, const EdgeSegment *prevEdge, const EdgeSegment *edge, const EdgeSegment *nextEdge) {
    if (
        (edge->color&RED && r.isEdgeRelevant(cache, edge, p)) ||
        (edge->color&GREEN && g.isEdgeRelevant(cache, edge, p)) ||
        (edge->color&BLUE && b.isEdgeRelevant(cache, edge, p))
    ) {
        double param;
        SignedDistance distance = edge->signedDistance(p, param);
        if (edge->color&RED)
            r.addEdgeTrueDistance(edge, distance, param);
        if (edge->color&GREEN)
            g.addEdgeTrueDistance(edge, distance, param);
        if (edge->color&BLUE)
            b.addEdgeTrueDistance(edge, distance, param);
        cache.point = p;
        cache.absDistance = fabs(distance.distance);

        Vector2 ap = p-edge->point(0);
        Vector2 bp = p-edge->point(1);
        Vector2 aDir = edge->direction(0).normalize(true);
        Vector2 bDir = edge->direction(1).normalize(true);
        Vector2 prevDir = prevEdge->direction(1).normalize(true);
        Vector2 nextDir = nextEdge->direction(0).normalize(true);
        double add = dotProduct(ap, (prevDir+aDir).normalize(true));
        double bdd = -dotProduct(bp, (bDir+nextDir).normalize(true));
        if (add > 0) {
            double pd = distance.distance;
            if (PseudoDistanceSelectorBase::getPseudoDistance(pd, ap, -aDir)) {
                pd = -pd;
                if (edge->color&RED)
                    r.addEdgePseudoDistance(pd);
                if (edge->color&GREEN)
                    g.addEdgePseudoDistance(pd);
                if (edge->color&BLUE)
                    b.addEdgePseudoDistance(pd);
            }
            cache.aPseudoDistance = pd;
        }
        if (bdd > 0) {
            double pd = distance.distance;
            if (PseudoDistanceSelectorBase::getPseudoDistance(pd, bp, bDir)) {
                if (edge->color&RED)
                    r.addEdgePseudoDistance(pd);
                if (edge->color&GREEN)
                    g.addEdgePseudoDistance(pd);
                if (edge->color&BLUE)
                    b.addEdgePseudoDistance(pd);
            }
            cache.bPseudoDistance = pd;
        }
        cache.aDomainDistance = add;
        cache.bDomainDistance = bdd;
    }
}

void MultiDistanceSelector::merge(const MultiDistanceSelector &other) {
    r.merge(other.r);
    g.merge(other.g);
    b.merge(other.b);
}

MultiDistanceSelector::DistanceType MultiDistanceSelector::distance() const {
    MultiDistance multiDistance;
    multiDistance.r = r.computeDistance(p);
    multiDistance.g = g.computeDistance(p);
    multiDistance.b = b.computeDistance(p);
    return multiDistance;
}

SignedDistance MultiDistanceSelector::trueDistance() const {
    SignedDistance distance = r.trueDistance();
    if (g.trueDistance() < distance)
        distance = g.trueDistance();
    if (b.trueDistance() < distance)
        distance = b.trueDistance();
    return distance;
}

MultiAndTrueDistanceSelector::DistanceType MultiAndTrueDistanceSelector::distance() const {
    MultiDistance multiDistance = MultiDistanceSelector::distance();
    MultiAndTrueDistance mtd;
    mtd.r = multiDistance.r;
    mtd.g = multiDistance.g;
    mtd.b = multiDistance.b;
    mtd.a = trueDistance().distance;
    return mtd;
}

}
