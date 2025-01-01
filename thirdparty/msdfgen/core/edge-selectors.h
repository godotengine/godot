
#pragma once

#include "Vector2.hpp"
#include "SignedDistance.hpp"
#include "edge-segments.h"

namespace msdfgen {

struct MultiDistance {
    double r, g, b;
};
struct MultiAndTrueDistance : MultiDistance {
    double a;
};

/// Selects the nearest edge by its true distance.
class TrueDistanceSelector {

public:
    typedef double DistanceType;

    struct EdgeCache {
        Point2 point;
        double absDistance;

        EdgeCache();
    };

    void reset(const Point2 &p);
    void addEdge(EdgeCache &cache, const EdgeSegment *prevEdge, const EdgeSegment *edge, const EdgeSegment *nextEdge);
    void merge(const TrueDistanceSelector &other);
    DistanceType distance() const;

private:
    Point2 p;
    SignedDistance minDistance;

};

class PseudoDistanceSelectorBase {

public:
    struct EdgeCache {
        Point2 point;
        double absDistance;
        double aDomainDistance, bDomainDistance;
        double aPseudoDistance, bPseudoDistance;

        EdgeCache();
    };

    static bool getPseudoDistance(double &distance, const Vector2 &ep, const Vector2 &edgeDir);

    PseudoDistanceSelectorBase();
    void reset(double delta);
    bool isEdgeRelevant(const EdgeCache &cache, const EdgeSegment *edge, const Point2 &p) const;
    void addEdgeTrueDistance(const EdgeSegment *edge, const SignedDistance &distance, double param);
    void addEdgePseudoDistance(double distance);
    void merge(const PseudoDistanceSelectorBase &other);
    double computeDistance(const Point2 &p) const;
    SignedDistance trueDistance() const;

private:
    SignedDistance minTrueDistance;
    double minNegativePseudoDistance;
    double minPositivePseudoDistance;
    const EdgeSegment *nearEdge;
    double nearEdgeParam;

};

/// Selects the nearest edge by its pseudo-distance.
class PseudoDistanceSelector : public PseudoDistanceSelectorBase {

public:
    typedef double DistanceType;

    void reset(const Point2 &p);
    void addEdge(EdgeCache &cache, const EdgeSegment *prevEdge, const EdgeSegment *edge, const EdgeSegment *nextEdge);
    DistanceType distance() const;

private:
    Point2 p;

};

/// Selects the nearest edge for each of the three channels by its pseudo-distance.
class MultiDistanceSelector {

public:
    typedef MultiDistance DistanceType;
    typedef PseudoDistanceSelectorBase::EdgeCache EdgeCache;

    void reset(const Point2 &p);
    void addEdge(EdgeCache &cache, const EdgeSegment *prevEdge, const EdgeSegment *edge, const EdgeSegment *nextEdge);
    void merge(const MultiDistanceSelector &other);
    DistanceType distance() const;
    SignedDistance trueDistance() const;

private:
    Point2 p;
    PseudoDistanceSelectorBase r, g, b;

};

/// Selects the nearest edge for each of the three color channels by its pseudo-distance and by true distance for the alpha channel.
class MultiAndTrueDistanceSelector : public MultiDistanceSelector {

public:
    typedef MultiAndTrueDistance DistanceType;

    DistanceType distance() const;

};

}
