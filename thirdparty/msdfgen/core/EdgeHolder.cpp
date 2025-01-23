
#include "EdgeHolder.h"

namespace msdfgen {

void EdgeHolder::swap(EdgeHolder &a, EdgeHolder &b) {
    EdgeSegment *tmp = a.edgeSegment;
    a.edgeSegment = b.edgeSegment;
    b.edgeSegment = tmp;
}

EdgeHolder::EdgeHolder() : edgeSegment(NULL) { }

EdgeHolder::EdgeHolder(EdgeSegment *segment) : edgeSegment(segment) { }

EdgeHolder::EdgeHolder(Point2 p0, Point2 p1, EdgeColor edgeColor) : edgeSegment(new LinearSegment(p0, p1, edgeColor)) { }

EdgeHolder::EdgeHolder(Point2 p0, Point2 p1, Point2 p2, EdgeColor edgeColor) : edgeSegment(new QuadraticSegment(p0, p1, p2, edgeColor)) { }

EdgeHolder::EdgeHolder(Point2 p0, Point2 p1, Point2 p2, Point2 p3, EdgeColor edgeColor) : edgeSegment(new CubicSegment(p0, p1, p2, p3, edgeColor)) { }

EdgeHolder::EdgeHolder(const EdgeHolder &orig) : edgeSegment(orig.edgeSegment ? orig.edgeSegment->clone() : NULL) { }

#ifdef MSDFGEN_USE_CPP11
EdgeHolder::EdgeHolder(EdgeHolder &&orig) : edgeSegment(orig.edgeSegment) {
    orig.edgeSegment = NULL;
}
#endif

EdgeHolder::~EdgeHolder() {
    delete edgeSegment;
}

EdgeHolder &EdgeHolder::operator=(const EdgeHolder &orig) {
    if (this != &orig) {
        delete edgeSegment;
        edgeSegment = orig.edgeSegment ? orig.edgeSegment->clone() : NULL;
    }
    return *this;
}

#ifdef MSDFGEN_USE_CPP11
EdgeHolder &EdgeHolder::operator=(EdgeHolder &&orig) {
    if (this != &orig) {
        delete edgeSegment;
        edgeSegment = orig.edgeSegment;
        orig.edgeSegment = NULL;
    }
    return *this;
}
#endif

EdgeSegment &EdgeHolder::operator*() {
    return *edgeSegment;
}

const EdgeSegment &EdgeHolder::operator*() const {
    return *edgeSegment;
}

EdgeSegment *EdgeHolder::operator->() {
    return edgeSegment;
}

const EdgeSegment *EdgeHolder::operator->() const {
    return edgeSegment;
}

EdgeHolder::operator EdgeSegment *() {
    return edgeSegment;
}

EdgeHolder::operator const EdgeSegment *() const {
    return edgeSegment;
}

}
