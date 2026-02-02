#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace mapbox {

namespace util {

template <std::size_t I, typename T> struct nth {
    inline static typename std::tuple_element<I, T>::type
    get(const T& t) { return std::get<I>(t); };
};

}

namespace detail {

template <typename N = uint32_t>
class Earcut {
public:
    std::vector<N> indices;
    std::size_t vertices = 0;

    template <typename Polygon>
    void operator()(const Polygon& points);

private:
    struct Node {
        Node(N index, double x_, double y_) : i(index), x(x_), y(y_) {}
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;
        Node(Node&&) = delete;
        Node& operator=(Node&&) = delete;

        const N i;
        const double x;
        const double y;

        // previous and next vertice nodes in a polygon ring
        Node* prev = nullptr;
        Node* next = nullptr;

        // z-order curve value
        int32_t z = 0;

        // previous and next nodes in z-order
        Node* prevZ = nullptr;
        Node* nextZ = nullptr;

        // indicates whether this is a steiner point
        bool steiner = false;
    };

    template <typename Ring> Node* linkedList(const Ring& points, const bool clockwise);
    Node* filterPoints(Node* start, Node* end = nullptr);
    void earcutLinked(Node* ear, int pass = 0);
    bool isEar(Node* ear);
    bool isEarHashed(Node* ear);
    Node* cureLocalIntersections(Node* start);
    void splitEarcut(Node* start);
    template <typename Polygon> Node* eliminateHoles(const Polygon& points, Node* outerNode);
    Node* eliminateHole(Node* hole, Node* outerNode);
    Node* findHoleBridge(Node* hole, Node* outerNode);
    bool sectorContainsSector(const Node* m, const Node* p);
    void indexCurve(Node* start);
    Node* sortLinked(Node* list);
    int32_t zOrder(const double x_, const double y_);
    Node* getLeftmost(Node* start);
    bool pointInTriangle(double ax, double ay, double bx, double by, double cx, double cy, double px, double py) const;
    bool isValidDiagonal(Node* a, Node* b);
    double area(const Node* p, const Node* q, const Node* r) const;
    bool equals(const Node* p1, const Node* p2);
    bool intersects(const Node* p1, const Node* q1, const Node* p2, const Node* q2);
    bool onSegment(const Node* p, const Node* q, const Node* r);
    int sign(double val);
    bool intersectsPolygon(const Node* a, const Node* b);
    bool locallyInside(const Node* a, const Node* b);
    bool middleInside(const Node* a, const Node* b);
    Node* splitPolygon(Node* a, Node* b);
    template <typename Point> Node* insertNode(std::size_t i, const Point& p, Node* last);
    void removeNode(Node* p);

    bool hashing;
    double minX, maxX;
    double minY, maxY;
    double inv_size = 0;

    template <typename T, typename Alloc = std::allocator<T>>
    class ObjectPool {
    public:
        ObjectPool() { }
        ObjectPool(std::size_t blockSize_) {
            reset(blockSize_);
        }
        ~ObjectPool() {
            clear();
        }
        template <typename... Args>
        T* construct(Args&&... args) {
            if (currentIndex >= blockSize) {
                currentBlock = alloc_traits::allocate(alloc, blockSize);
                allocations.emplace_back(currentBlock);
                currentIndex = 0;
            }
            T* object = &currentBlock[currentIndex++];
            alloc_traits::construct(alloc, object, std::forward<Args>(args)...);
            return object;
        }
        void reset(std::size_t newBlockSize) {
            for (auto allocation : allocations) {
                alloc_traits::deallocate(alloc, allocation, blockSize);
            }
            allocations.clear();
            blockSize = std::max<std::size_t>(1, newBlockSize);
            currentBlock = nullptr;
            currentIndex = blockSize;
        }
        void clear() { reset(blockSize); }
    private:
        T* currentBlock = nullptr;
        std::size_t currentIndex = 1;
        std::size_t blockSize = 1;
        std::vector<T*> allocations;
        Alloc alloc;
        typedef typename std::allocator_traits<Alloc> alloc_traits;
    };
    ObjectPool<Node> nodes;
};

template <typename N> template <typename Polygon>
void Earcut<N>::operator()(const Polygon& points) {
    // reset
    indices.clear();
    vertices = 0;

    if (points.empty()) return;

    double x;
    double y;
    int threshold = 80;
    std::size_t len = 0;

    for (size_t i = 0; threshold >= 0 && i < points.size(); i++) {
        threshold -= static_cast<int>(points[i].size());
        len += points[i].size();
    }

    //estimate size of nodes and indices
    nodes.reset(len * 3 / 2);
    indices.reserve(len + points[0].size());

    Node* outerNode = linkedList(points[0], true);
    if (!outerNode || outerNode->prev == outerNode->next) return;

    if (points.size() > 1) outerNode = eliminateHoles(points, outerNode);

    // if the shape is not too simple, we'll use z-order curve hash later; calculate polygon bbox
    hashing = threshold < 0;
    if (hashing) {
        Node* p = outerNode->next;
        minX = maxX = outerNode->x;
        minY = maxY = outerNode->y;
        do {
            x = p->x;
            y = p->y;
            minX = std::min<double>(minX, x);
            minY = std::min<double>(minY, y);
            maxX = std::max<double>(maxX, x);
            maxY = std::max<double>(maxY, y);
            p = p->next;
        } while (p != outerNode);

        // minX, minY and inv_size are later used to transform coords into integers for z-order calculation
        inv_size = std::max<double>(maxX - minX, maxY - minY);
        inv_size = inv_size != .0 ? (32767. / inv_size) : .0;
    }

    earcutLinked(outerNode);

    nodes.clear();
}

// create a circular doubly linked list from polygon points in the specified winding order
template <typename N> template <typename Ring>
typename Earcut<N>::Node*
Earcut<N>::linkedList(const Ring& points, const bool clockwise) {
    using Point = typename Ring::value_type;
    double sum = 0;
    const std::size_t len = points.size();
    std::size_t i, j;
    Node* last = nullptr;

    // calculate original winding order of a polygon ring
    for (i = 0, j = len > 0 ? len - 1 : 0; i < len; j = i++) {
        const auto& p1 = points[i];
        const auto& p2 = points[j];
        const double p20 = util::nth<0, Point>::get(p2);
        const double p10 = util::nth<0, Point>::get(p1);
        const double p11 = util::nth<1, Point>::get(p1);
        const double p21 = util::nth<1, Point>::get(p2);
        sum += (p20 - p10) * (p11 + p21);
    }

    // link points into circular doubly-linked list in the specified winding order
    if (clockwise == (sum > 0)) {
        for (i = 0; i < len; i++) last = insertNode(vertices + i, points[i], last);
    } else {
        for (i = len; i-- > 0;) last = insertNode(vertices + i, points[i], last);
    }

    if (last && equals(last, last->next)) {
        removeNode(last);
        last = last->next;
    }

    vertices += len;

    return last;
}

// eliminate colinear or duplicate points
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::filterPoints(Node* start, Node* end) {
    if (!end) end = start;

    Node* p = start;
    bool again;
    do {
        again = false;

        if (!p->steiner && (equals(p, p->next) || area(p->prev, p, p->next) == 0)) {
            removeNode(p);
            p = end = p->prev;

            if (p == p->next) break;
            again = true;

        } else {
            p = p->next;
        }
    } while (again || p != end);

    return end;
}

// main ear slicing loop which triangulates a polygon (given as a linked list)
template <typename N>
void Earcut<N>::earcutLinked(Node* ear, int pass) {
    if (!ear) return;

    // interlink polygon nodes in z-order
    if (!pass && hashing) indexCurve(ear);

    Node* stop = ear;
    Node* prev;
    Node* next;

    int iterations = 0;

    // iterate through ears, slicing them one by one
    while (ear->prev != ear->next) {
        iterations++;
        prev = ear->prev;
        next = ear->next;

        if (hashing ? isEarHashed(ear) : isEar(ear)) {
            // cut off the triangle
            indices.emplace_back(prev->i);
            indices.emplace_back(ear->i);
            indices.emplace_back(next->i);

            removeNode(ear);

            // skipping the next vertice leads to less sliver triangles
            ear = next->next;
            stop = next->next;

            continue;
        }

        ear = next;

        // if we looped through the whole remaining polygon and can't find any more ears
        if (ear == stop) {
            // try filtering points and slicing again
            if (!pass) earcutLinked(filterPoints(ear), 1);

            // if this didn't work, try curing all small self-intersections locally
            else if (pass == 1) {
                ear = cureLocalIntersections(filterPoints(ear));
                earcutLinked(ear, 2);

            // as a last resort, try splitting the remaining polygon into two
            } else if (pass == 2) splitEarcut(ear);

            break;
        }
    }
}

// check whether a polygon node forms a valid ear with adjacent nodes
template <typename N>
bool Earcut<N>::isEar(Node* ear) {
    const Node* a = ear->prev;
    const Node* b = ear;
    const Node* c = ear->next;

    if (area(a, b, c) >= 0) return false; // reflex, can't be an ear

    // now make sure we don't have other points inside the potential ear
    Node* p = ear->next->next;

    while (p != ear->prev) {
        if (pointInTriangle(a->x, a->y, b->x, b->y, c->x, c->y, p->x, p->y) &&
            area(p->prev, p, p->next) >= 0) return false;
        p = p->next;
    }

    return true;
}

template <typename N>
bool Earcut<N>::isEarHashed(Node* ear) {
    const Node* a = ear->prev;
    const Node* b = ear;
    const Node* c = ear->next;

    if (area(a, b, c) >= 0) return false; // reflex, can't be an ear

    // triangle bbox; min & max are calculated like this for speed
    const double minTX = std::min<double>(a->x, std::min<double>(b->x, c->x));
    const double minTY = std::min<double>(a->y, std::min<double>(b->y, c->y));
    const double maxTX = std::max<double>(a->x, std::max<double>(b->x, c->x));
    const double maxTY = std::max<double>(a->y, std::max<double>(b->y, c->y));

    // z-order range for the current triangle bbox;
    const int32_t minZ = zOrder(minTX, minTY);
    const int32_t maxZ = zOrder(maxTX, maxTY);

    // first look for points inside the triangle in increasing z-order
    Node* p = ear->nextZ;

    while (p && p->z <= maxZ) {
        if (p != ear->prev && p != ear->next &&
            pointInTriangle(a->x, a->y, b->x, b->y, c->x, c->y, p->x, p->y) &&
            area(p->prev, p, p->next) >= 0) return false;
        p = p->nextZ;
    }

    // then look for points in decreasing z-order
    p = ear->prevZ;

    while (p && p->z >= minZ) {
        if (p != ear->prev && p != ear->next &&
            pointInTriangle(a->x, a->y, b->x, b->y, c->x, c->y, p->x, p->y) &&
            area(p->prev, p, p->next) >= 0) return false;
        p = p->prevZ;
    }

    return true;
}

// go through all polygon nodes and cure small local self-intersections
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::cureLocalIntersections(Node* start) {
    Node* p = start;
    do {
        Node* a = p->prev;
        Node* b = p->next->next;

        // a self-intersection where edge (v[i-1],v[i]) intersects (v[i+1],v[i+2])
        if (!equals(a, b) && intersects(a, p, p->next, b) && locallyInside(a, b) && locallyInside(b, a)) {
            indices.emplace_back(a->i);
            indices.emplace_back(p->i);
            indices.emplace_back(b->i);

            // remove two nodes involved
            removeNode(p);
            removeNode(p->next);

            p = start = b;
        }
        p = p->next;
    } while (p != start);

    return filterPoints(p);
}

// try splitting polygon into two and triangulate them independently
template <typename N>
void Earcut<N>::splitEarcut(Node* start) {
    // look for a valid diagonal that divides the polygon into two
    Node* a = start;
    do {
        Node* b = a->next->next;
        while (b != a->prev) {
            if (a->i != b->i && isValidDiagonal(a, b)) {
                // split the polygon in two by the diagonal
                Node* c = splitPolygon(a, b);

                // filter colinear points around the cuts
                a = filterPoints(a, a->next);
                c = filterPoints(c, c->next);

                // run earcut on each half
                earcutLinked(a);
                earcutLinked(c);
                return;
            }
            b = b->next;
        }
        a = a->next;
    } while (a != start);
}

// link every hole into the outer loop, producing a single-ring polygon without holes
template <typename N> template <typename Polygon>
typename Earcut<N>::Node*
Earcut<N>::eliminateHoles(const Polygon& points, Node* outerNode) {
    const size_t len = points.size();

    std::vector<Node*> queue;
    for (size_t i = 1; i < len; i++) {
        Node* list = linkedList(points[i], false);
        if (list) {
            if (list == list->next) list->steiner = true;
            queue.push_back(getLeftmost(list));
        }
    }
    std::sort(queue.begin(), queue.end(), [](const Node* a, const Node* b) {
        return a->x < b->x;
    });

    // process holes from left to right
    for (size_t i = 0; i < queue.size(); i++) {
        outerNode = eliminateHole(queue[i], outerNode);
    }

    return outerNode;
}

// find a bridge between vertices that connects hole with an outer ring and and link it
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::eliminateHole(Node* hole, Node* outerNode) {
    Node* bridge = findHoleBridge(hole, outerNode);
    if (!bridge) {
        return outerNode;
    }

    Node* bridgeReverse = splitPolygon(bridge, hole);

    // filter collinear points around the cuts
    filterPoints(bridgeReverse, bridgeReverse->next);

    // Check if input node was removed by the filtering
    return filterPoints(bridge, bridge->next);
}

// David Eberly's algorithm for finding a bridge between hole and outer polygon
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::findHoleBridge(Node* hole, Node* outerNode) {
    Node* p = outerNode;
    double hx = hole->x;
    double hy = hole->y;
    double qx = -std::numeric_limits<double>::infinity();
    Node* m = nullptr;

    // find a segment intersected by a ray from the hole's leftmost Vertex to the left;
    // segment's endpoint with lesser x will be potential connection Vertex
    do {
        if (hy <= p->y && hy >= p->next->y && p->next->y != p->y) {
          double x = p->x + (hy - p->y) * (p->next->x - p->x) / (p->next->y - p->y);
          if (x <= hx && x > qx) {
            qx = x;
            m = p->x < p->next->x ? p : p->next;
            if (x == hx) return m; // hole touches outer segment; pick leftmost endpoint
          }
        }
        p = p->next;
    } while (p != outerNode);

    if (!m) return 0;

    // look for points inside the triangle of hole Vertex, segment intersection and endpoint;
    // if there are no points found, we have a valid connection;
    // otherwise choose the Vertex of the minimum angle with the ray as connection Vertex

    const Node* stop = m;
    double tanMin = std::numeric_limits<double>::infinity();
    double tanCur = 0;

    p = m;
    double mx = m->x;
    double my = m->y;

    do {
        if (hx >= p->x && p->x >= mx && hx != p->x &&
            pointInTriangle(hy < my ? hx : qx, hy, mx, my, hy < my ? qx : hx, hy, p->x, p->y)) {

            tanCur = std::abs(hy - p->y) / (hx - p->x); // tangential

            if (locallyInside(p, hole) &&
                (tanCur < tanMin || (tanCur == tanMin && (p->x > m->x || sectorContainsSector(m, p))))) {
                m = p;
                tanMin = tanCur;
            }
        }

        p = p->next;
    } while (p != stop);

    return m;
}

// whether sector in vertex m contains sector in vertex p in the same coordinates
template <typename N>
bool Earcut<N>::sectorContainsSector(const Node* m, const Node* p) {
    return area(m->prev, m, p->prev) < 0 && area(p->next, m, m->next) < 0;
}

// interlink polygon nodes in z-order
template <typename N>
void Earcut<N>::indexCurve(Node* start) {
    assert(start);
    Node* p = start;

    do {
        p->z = p->z ? p->z : zOrder(p->x, p->y);
        p->prevZ = p->prev;
        p->nextZ = p->next;
        p = p->next;
    } while (p != start);

    p->prevZ->nextZ = nullptr;
    p->prevZ = nullptr;

    sortLinked(p);
}

// Simon Tatham's linked list merge sort algorithm
// http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::sortLinked(Node* list) {
    assert(list);
    Node* p;
    Node* q;
    Node* e;
    Node* tail;
    int i, numMerges, pSize, qSize;
    int inSize = 1;

    for (;;) {
        p = list;
        list = nullptr;
        tail = nullptr;
        numMerges = 0;

        while (p) {
            numMerges++;
            q = p;
            pSize = 0;
            for (i = 0; i < inSize; i++) {
                pSize++;
                q = q->nextZ;
                if (!q) break;
            }

            qSize = inSize;

            while (pSize > 0 || (qSize > 0 && q)) {

                if (pSize == 0) {
                    e = q;
                    q = q->nextZ;
                    qSize--;
                } else if (qSize == 0 || !q) {
                    e = p;
                    p = p->nextZ;
                    pSize--;
                } else if (p->z <= q->z) {
                    e = p;
                    p = p->nextZ;
                    pSize--;
                } else {
                    e = q;
                    q = q->nextZ;
                    qSize--;
                }

                if (tail) tail->nextZ = e;
                else list = e;

                e->prevZ = tail;
                tail = e;
            }

            p = q;
        }

        tail->nextZ = nullptr;

        if (numMerges <= 1) return list;

        inSize *= 2;
    }
}

// z-order of a Vertex given coords and size of the data bounding box
template <typename N>
int32_t Earcut<N>::zOrder(const double x_, const double y_) {
    // coords are transformed into non-negative 15-bit integer range
    int32_t x = static_cast<int32_t>((x_ - minX) * inv_size);
    int32_t y = static_cast<int32_t>((y_ - minY) * inv_size);

    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    return x | (y << 1);
}

// find the leftmost node of a polygon ring
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::getLeftmost(Node* start) {
    Node* p = start;
    Node* leftmost = start;
    do {
        if (p->x < leftmost->x || (p->x == leftmost->x && p->y < leftmost->y))
            leftmost = p;
        p = p->next;
    } while (p != start);

    return leftmost;
}

// check if a point lies within a convex triangle
template <typename N>
bool Earcut<N>::pointInTriangle(double ax, double ay, double bx, double by, double cx, double cy, double px, double py) const {
    return (cx - px) * (ay - py) >= (ax - px) * (cy - py) &&
           (ax - px) * (by - py) >= (bx - px) * (ay - py) &&
           (bx - px) * (cy - py) >= (cx - px) * (by - py);
}

// check if a diagonal between two polygon nodes is valid (lies in polygon interior)
template <typename N>
bool Earcut<N>::isValidDiagonal(Node* a, Node* b) {
    return a->next->i != b->i && a->prev->i != b->i && !intersectsPolygon(a, b) && // dones't intersect other edges
           ((locallyInside(a, b) && locallyInside(b, a) && middleInside(a, b) && // locally visible
            (area(a->prev, a, b->prev) != 0.0 || area(a, b->prev, b) != 0.0)) || // does not create opposite-facing sectors
            (equals(a, b) && area(a->prev, a, a->next) > 0 && area(b->prev, b, b->next) > 0)); // special zero-length case
}

// signed area of a triangle
template <typename N>
double Earcut<N>::area(const Node* p, const Node* q, const Node* r) const {
    return (q->y - p->y) * (r->x - q->x) - (q->x - p->x) * (r->y - q->y);
}

// check if two points are equal
template <typename N>
bool Earcut<N>::equals(const Node* p1, const Node* p2) {
    return p1->x == p2->x && p1->y == p2->y;
}

// check if two segments intersect
template <typename N>
bool Earcut<N>::intersects(const Node* p1, const Node* q1, const Node* p2, const Node* q2) {
    int o1 = sign(area(p1, q1, p2));
    int o2 = sign(area(p1, q1, q2));
    int o3 = sign(area(p2, q2, p1));
    int o4 = sign(area(p2, q2, q1));

    if (o1 != o2 && o3 != o4) return true; // general case

    if (o1 == 0 && onSegment(p1, p2, q1)) return true; // p1, q1 and p2 are collinear and p2 lies on p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) return true; // p1, q1 and q2 are collinear and q2 lies on p1q1
    if (o3 == 0 && onSegment(p2, p1, q2)) return true; // p2, q2 and p1 are collinear and p1 lies on p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) return true; // p2, q2 and q1 are collinear and q1 lies on p2q2

    return false;
}

// for collinear points p, q, r, check if point q lies on segment pr
template <typename N>
bool Earcut<N>::onSegment(const Node* p, const Node* q, const Node* r) {
    return q->x <= std::max<double>(p->x, r->x) &&
        q->x >= std::min<double>(p->x, r->x) &&
        q->y <= std::max<double>(p->y, r->y) &&
        q->y >= std::min<double>(p->y, r->y);
}

template <typename N>
int Earcut<N>::sign(double val) {
    return (0.0 < val) - (val < 0.0);
}

// check if a polygon diagonal intersects any polygon segments
template <typename N>
bool Earcut<N>::intersectsPolygon(const Node* a, const Node* b) {
    const Node* p = a;
    do {
        if (p->i != a->i && p->next->i != a->i && p->i != b->i && p->next->i != b->i &&
                intersects(p, p->next, a, b)) return true;
        p = p->next;
    } while (p != a);

    return false;
}

// check if a polygon diagonal is locally inside the polygon
template <typename N>
bool Earcut<N>::locallyInside(const Node* a, const Node* b) {
    return area(a->prev, a, a->next) < 0 ?
        area(a, b, a->next) >= 0 && area(a, a->prev, b) >= 0 :
        area(a, b, a->prev) < 0 || area(a, a->next, b) < 0;
}

// check if the middle Vertex of a polygon diagonal is inside the polygon
template <typename N>
bool Earcut<N>::middleInside(const Node* a, const Node* b) {
    const Node* p = a;
    bool inside = false;
    double px = (a->x + b->x) / 2;
    double py = (a->y + b->y) / 2;
    do {
        if (((p->y > py) != (p->next->y > py)) && p->next->y != p->y &&
                (px < (p->next->x - p->x) * (py - p->y) / (p->next->y - p->y) + p->x))
            inside = !inside;
        p = p->next;
    } while (p != a);

    return inside;
}

// link two polygon vertices with a bridge; if the vertices belong to the same ring, it splits
// polygon into two; if one belongs to the outer ring and another to a hole, it merges it into a
// single ring
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::splitPolygon(Node* a, Node* b) {
    Node* a2 = nodes.construct(a->i, a->x, a->y);
    Node* b2 = nodes.construct(b->i, b->x, b->y);
    Node* an = a->next;
    Node* bp = b->prev;

    a->next = b;
    b->prev = a;

    a2->next = an;
    an->prev = a2;

    b2->next = a2;
    a2->prev = b2;

    bp->next = b2;
    b2->prev = bp;

    return b2;
}

// create a node and util::optionally link it with previous one (in a circular doubly linked list)
template <typename N> template <typename Point>
typename Earcut<N>::Node*
Earcut<N>::insertNode(std::size_t i, const Point& pt, Node* last) {
    Node* p = nodes.construct(static_cast<N>(i), util::nth<0, Point>::get(pt), util::nth<1, Point>::get(pt));

    if (!last) {
        p->prev = p;
        p->next = p;

    } else {
        assert(last);
        p->next = last->next;
        p->prev = last;
        last->next->prev = p;
        last->next = p;
    }
    return p;
}

template <typename N>
void Earcut<N>::removeNode(Node* p) {
    p->next->prev = p->prev;
    p->prev->next = p->next;

    if (p->prevZ) p->prevZ->nextZ = p->nextZ;
    if (p->nextZ) p->nextZ->prevZ = p->prevZ;
}
}

template <typename N = uint32_t, typename Polygon>
std::vector<N> earcut(const Polygon& poly) {
    mapbox::detail::Earcut<N> earcut;
    earcut(poly);
    return std::move(earcut.indices);
}
}
