
#include "Contour.h"

#include "arithmetics.hpp"

namespace msdfgen {

static double shoelace(const Point2 &a, const Point2 &b) {
    return (b.x-a.x)*(a.y+b.y);
}

void Contour::addEdge(const EdgeHolder &edge) {
    edges.push_back(edge);
}

#ifdef MSDFGEN_USE_CPP11
void Contour::addEdge(EdgeHolder &&edge) {
    edges.push_back((EdgeHolder &&) edge);
}
#endif

EdgeHolder &Contour::addEdge() {
    edges.resize(edges.size()+1);
    return edges.back();
}

static void boundPoint(double &l, double &b, double &r, double &t, Point2 p) {
    if (p.x < l) l = p.x;
    if (p.y < b) b = p.y;
    if (p.x > r) r = p.x;
    if (p.y > t) t = p.y;
}

void Contour::bound(double &l, double &b, double &r, double &t) const {
    for (std::vector<EdgeHolder>::const_iterator edge = edges.begin(); edge != edges.end(); ++edge)
        (*edge)->bound(l, b, r, t);
}

void Contour::boundMiters(double &l, double &b, double &r, double &t, double border, double miterLimit, int polarity) const {
    if (edges.empty())
        return;
    Vector2 prevDir = edges.back()->direction(1).normalize(true);
    for (std::vector<EdgeHolder>::const_iterator edge = edges.begin(); edge != edges.end(); ++edge) {
        Vector2 dir = -(*edge)->direction(0).normalize(true);
        if (polarity*crossProduct(prevDir, dir) >= 0) {
            double miterLength = miterLimit;
            double q = .5*(1-dotProduct(prevDir, dir));
            if (q > 0)
                miterLength = min(1/sqrt(q), miterLimit);
            Point2 miter = (*edge)->point(0)+border*miterLength*(prevDir+dir).normalize(true);
            boundPoint(l, b, r, t, miter);
        }
        prevDir = (*edge)->direction(1).normalize(true);
    }
}

int Contour::winding() const {
    if (edges.empty())
        return 0;
    double total = 0;
    if (edges.size() == 1) {
        Point2 a = edges[0]->point(0), b = edges[0]->point(1/3.), c = edges[0]->point(2/3.);
        total += shoelace(a, b);
        total += shoelace(b, c);
        total += shoelace(c, a);
    } else if (edges.size() == 2) {
        Point2 a = edges[0]->point(0), b = edges[0]->point(.5), c = edges[1]->point(0), d = edges[1]->point(.5);
        total += shoelace(a, b);
        total += shoelace(b, c);
        total += shoelace(c, d);
        total += shoelace(d, a);
    } else {
        Point2 prev = edges.back()->point(0);
        for (std::vector<EdgeHolder>::const_iterator edge = edges.begin(); edge != edges.end(); ++edge) {
            Point2 cur = (*edge)->point(0);
            total += shoelace(prev, cur);
            prev = cur;
        }
    }
    return sign(total);
}

void Contour::reverse() {
    for (int i = (int) edges.size()/2; i > 0; --i)
        EdgeHolder::swap(edges[i-1], edges[edges.size()-i]);
    for (std::vector<EdgeHolder>::iterator edge = edges.begin(); edge != edges.end(); ++edge)
        (*edge)->reverse();
}

}
