
#include "Scanline.h"

#include <algorithm>
#include "arithmetics.hpp"

namespace msdfgen {

static int compareIntersections(const void *a, const void *b) {
    return sign(reinterpret_cast<const Scanline::Intersection *>(a)->x-reinterpret_cast<const Scanline::Intersection *>(b)->x);
}

bool interpretFillRule(int intersections, FillRule fillRule) {
    switch (fillRule) {
        case FILL_NONZERO:
            return intersections != 0;
        case FILL_ODD:
            return intersections&1;
        case FILL_POSITIVE:
            return intersections > 0;
        case FILL_NEGATIVE:
            return intersections < 0;
    }
    return false;
}

double Scanline::overlap(const Scanline &a, const Scanline &b, double xFrom, double xTo, FillRule fillRule) {
    double total = 0;
    bool aInside = false, bInside = false;
    int ai = 0, bi = 0;
    double ax = !a.intersections.empty() ? a.intersections[ai].x : xTo;
    double bx = !b.intersections.empty() ? b.intersections[bi].x : xTo;
    while (ax < xFrom || bx < xFrom) {
        double xNext = min(ax, bx);
        if (ax == xNext && ai < (int) a.intersections.size()) {
            aInside = interpretFillRule(a.intersections[ai].direction, fillRule);
            ax = ++ai < (int) a.intersections.size() ? a.intersections[ai].x : xTo;
        }
        if (bx == xNext && bi < (int) b.intersections.size()) {
            bInside = interpretFillRule(b.intersections[bi].direction, fillRule);
            bx = ++bi < (int) b.intersections.size() ? b.intersections[bi].x : xTo;
        }
    }
    double x = xFrom;
    while (ax < xTo || bx < xTo) {
        double xNext = min(ax, bx);
        if (aInside == bInside)
            total += xNext-x;
        if (ax == xNext && ai < (int) a.intersections.size()) {
            aInside = interpretFillRule(a.intersections[ai].direction, fillRule);
            ax = ++ai < (int) a.intersections.size() ? a.intersections[ai].x : xTo;
        }
        if (bx == xNext && bi < (int) b.intersections.size()) {
            bInside = interpretFillRule(b.intersections[bi].direction, fillRule);
            bx = ++bi < (int) b.intersections.size() ? b.intersections[bi].x : xTo;
        }
        x = xNext;
    }
    if (aInside == bInside)
        total += xTo-x;
    return total;
}

Scanline::Scanline() : lastIndex(0) { }

void Scanline::preprocess() {
    lastIndex = 0;
    if (!intersections.empty()) {
        qsort(&intersections[0], intersections.size(), sizeof(Intersection), compareIntersections);
        int totalDirection = 0;
        for (std::vector<Intersection>::iterator intersection = intersections.begin(); intersection != intersections.end(); ++intersection) {
            totalDirection += intersection->direction;
            intersection->direction = totalDirection;
        }
    }
}

void Scanline::setIntersections(const std::vector<Intersection> &intersections) {
    this->intersections = intersections;
    preprocess();
}

#ifdef MSDFGEN_USE_CPP11
void Scanline::setIntersections(std::vector<Intersection> &&intersections) {
    this->intersections = (std::vector<Intersection> &&) intersections;
    preprocess();
}
#endif

int Scanline::moveTo(double x) const {
    if (intersections.empty())
        return -1;
    int index = lastIndex;
    if (x < intersections[index].x) {
        do {
            if (index == 0) {
                lastIndex = 0;
                return -1;
            }
            --index;
        } while (x < intersections[index].x);
    } else {
        while (index < (int) intersections.size()-1 && x >= intersections[index+1].x)
            ++index;
    }
    lastIndex = index;
    return index;
}

int Scanline::countIntersections(double x) const {
    return moveTo(x)+1;
}

int Scanline::sumIntersections(double x) const {
    int index = moveTo(x);
    if (index >= 0)
        return intersections[index].direction;
    return 0;
}

bool Scanline::filled(double x, FillRule fillRule) const {
    return interpretFillRule(sumIntersections(x), fillRule);
}

}
