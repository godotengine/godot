// Copyright 2021 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "polygon.h"

#include <map>
#include <set>

#include "collider.h"
#include "optional_assert.h"
#include "utils.h"

namespace {
using namespace manifold;

static ExecutionParams params;

constexpr float kBest = -std::numeric_limits<float>::infinity();

// it seems that MSVC cannot optimize glm::determinant(glm::mat2(a, b))
constexpr float determinant2x2(glm::vec2 a, glm::vec2 b) {
  return a.x * b.y - a.y * b.x;
}

#ifdef MANIFOLD_DEBUG
struct PolyEdge {
  int startVert, endVert;
};

std::vector<PolyEdge> Polygons2Edges(const PolygonsIdx &polys) {
  std::vector<PolyEdge> halfedges;
  for (const auto &poly : polys) {
    for (size_t i = 1; i < poly.size(); ++i) {
      halfedges.push_back({poly[i - 1].idx, poly[i].idx});
    }
    halfedges.push_back({poly.back().idx, poly[0].idx});
  }
  return halfedges;
}

std::vector<PolyEdge> Triangles2Edges(
    const std::vector<glm::ivec3> &triangles) {
  std::vector<PolyEdge> halfedges;
  halfedges.reserve(triangles.size() * 3);
  for (const glm::ivec3 &tri : triangles) {
    halfedges.push_back({tri[0], tri[1]});
    halfedges.push_back({tri[1], tri[2]});
    halfedges.push_back({tri[2], tri[0]});
  }
  return halfedges;
}

void CheckTopology(const std::vector<PolyEdge> &halfedges) {
  DEBUG_ASSERT(halfedges.size() % 2 == 0, topologyErr,
               "Odd number of halfedges.");
  size_t n_edges = halfedges.size() / 2;
  std::vector<PolyEdge> forward(halfedges.size()), backward(halfedges.size());

  auto end = std::copy_if(halfedges.begin(), halfedges.end(), forward.begin(),
                          [](PolyEdge e) { return e.endVert > e.startVert; });
  DEBUG_ASSERT(
      static_cast<size_t>(std::distance(forward.begin(), end)) == n_edges,
      topologyErr, "Half of halfedges should be forward.");
  forward.resize(n_edges);

  end = std::copy_if(halfedges.begin(), halfedges.end(), backward.begin(),
                     [](PolyEdge e) { return e.endVert < e.startVert; });
  DEBUG_ASSERT(
      static_cast<size_t>(std::distance(backward.begin(), end)) == n_edges,
      topologyErr, "Half of halfedges should be backward.");
  backward.resize(n_edges);

  std::for_each(backward.begin(), backward.end(),
                [](PolyEdge &e) { std::swap(e.startVert, e.endVert); });
  auto cmp = [](const PolyEdge &a, const PolyEdge &b) {
    return a.startVert < b.startVert ||
           (a.startVert == b.startVert && a.endVert < b.endVert);
  };
  std::stable_sort(forward.begin(), forward.end(), cmp);
  std::stable_sort(backward.begin(), backward.end(), cmp);
  for (size_t i = 0; i < n_edges; ++i) {
    DEBUG_ASSERT(forward[i].startVert == backward[i].startVert &&
                     forward[i].endVert == backward[i].endVert,
                 topologyErr, "Not manifold.");
  }
}

void CheckTopology(const std::vector<glm::ivec3> &triangles,
                   const PolygonsIdx &polys) {
  std::vector<PolyEdge> halfedges = Triangles2Edges(triangles);
  std::vector<PolyEdge> openEdges = Polygons2Edges(polys);
  for (PolyEdge e : openEdges) {
    halfedges.push_back({e.endVert, e.startVert});
  }
  CheckTopology(halfedges);
}

void CheckGeometry(const std::vector<glm::ivec3> &triangles,
                   const PolygonsIdx &polys, float precision) {
  std::unordered_map<int, glm::vec2> vertPos;
  for (const auto &poly : polys) {
    for (size_t i = 0; i < poly.size(); ++i) {
      vertPos[poly[i].idx] = poly[i].pos;
    }
  }
  DEBUG_ASSERT(std::all_of(triangles.begin(), triangles.end(),
                           [&vertPos, precision](const glm::ivec3 &tri) {
                             return CCW(vertPos[tri[0]], vertPos[tri[1]],
                                        vertPos[tri[2]], precision) >= 0;
                           }),
               geometryErr, "triangulation is not entirely CCW!");
}

void Dump(const PolygonsIdx &polys) {
  for (auto poly : polys) {
    std::cout << "polys.push_back({" << std::setprecision(9) << std::endl;
    for (auto v : poly) {
      std::cout << "    {" << v.pos.x << ", " << v.pos.y << "},  //"
                << std::endl;
    }
    std::cout << "});" << std::endl;
  }
  for (auto poly : polys) {
    std::cout << "show(array([" << std::endl;
    for (auto v : poly) {
      std::cout << "  [" << v.pos.x << ", " << v.pos.y << "]," << std::endl;
    }
    std::cout << "]))" << std::endl;
  }
}

void PrintFailure(const std::exception &e, const PolygonsIdx &polys,
                  std::vector<glm::ivec3> &triangles, float precision) {
  std::cout << "-----------------------------------" << std::endl;
  std::cout << "Triangulation failed! Precision = " << precision << std::endl;
  std::cout << e.what() << std::endl;
  Dump(polys);
  std::cout << "produced this triangulation:" << std::endl;
  for (size_t j = 0; j < triangles.size(); ++j) {
    std::cout << triangles[j][0] << ", " << triangles[j][1] << ", "
              << triangles[j][2] << std::endl;
  }
}

#define PRINT(msg) \
  if (params.verbose) std::cout << msg << std::endl;
#else
#define PRINT(msg)
#endif

/**
 * Tests if the input polygons are convex by searching for any reflex vertices.
 * Exactly colinear edges and zero-length edges are treated conservatively as
 * reflex. Does not check for overlaps.
 */
bool IsConvex(const PolygonsIdx &polys, float precision) {
  for (const SimplePolygonIdx &poly : polys) {
    const glm::vec2 firstEdge = poly[0].pos - poly[poly.size() - 1].pos;
    // Zero-length edges comes out NaN, which won't trip the early return, but
    // it's okay because that zero-length edge will also get tested
    // non-normalized and will trip det == 0.
    glm::vec2 lastEdge = glm::normalize(firstEdge);
    for (size_t v = 0; v < poly.size(); ++v) {
      const glm::vec2 edge =
          v + 1 < poly.size() ? poly[v + 1].pos - poly[v].pos : firstEdge;
      const float det = determinant2x2(lastEdge, edge);
      if (det <= 0 ||
          (glm::abs(det) < precision && glm::dot(lastEdge, edge) < 0))
        return false;
      lastEdge = glm::normalize(edge);
    }
  }
  return true;
}

/**
 * Triangulates a set of convex polygons by alternating instead of a fan, to
 * avoid creating high-degree vertices.
 */
std::vector<glm::ivec3> TriangulateConvex(const PolygonsIdx &polys) {
  const size_t numTri = manifold::transform_reduce(
      polys.begin(), polys.end(), 0_z, [](size_t a, size_t b) { return a + b; },
      [](const SimplePolygonIdx &poly) { return poly.size() - 2; });
  std::vector<glm::ivec3> triangles;
  triangles.reserve(numTri);
  for (const SimplePolygonIdx &poly : polys) {
    size_t i = 0;
    size_t k = poly.size() - 1;
    bool right = true;
    while (i + 1 < k) {
      const size_t j = right ? i + 1 : k - 1;
      triangles.push_back({poly[i].idx, poly[j].idx, poly[k].idx});
      if (right) {
        i = j;
      } else {
        k = j;
      }
      right = !right;
    }
  }
  return triangles;
}

/**
 * Ear-clipping triangulator based on David Eberly's approach from Geometric
 * Tools, but adjusted to handle epsilon-valid polygons, and including a
 * fallback that ensures a manifold triangulation even for overlapping polygons.
 * This is an O(n^2) algorithm, but hopefully this is not a big problem as the
 * number of edges in a given polygon is generally much less than the number of
 * triangles in a mesh, and relatively few faces even need triangulation.
 *
 * The main adjustments for robustness involve clipping the sharpest ears first
 * (a known technique to get higher triangle quality), and doing an exhaustive
 * search to determine ear convexity exactly if the first geometric result is
 * within precision.
 */

class EarClip {
 public:
  EarClip(const PolygonsIdx &polys, float precision) : precision_(precision) {
    ZoneScoped;

    size_t numVert = 0;
    for (const SimplePolygonIdx &poly : polys) {
      numVert += poly.size();
    }
    polygon_.reserve(numVert + 2 * polys.size());

    std::vector<VertItr> starts = Initialize(polys);

    for (VertItr v = polygon_.begin(); v != polygon_.end(); ++v) {
      ClipIfDegenerate(v);
    }

    for (const VertItr first : starts) {
      FindStart(first);
    }
  }

  std::vector<glm::ivec3> Triangulate() {
    ZoneScoped;

    for (const VertItr start : holes_) {
      CutKeyhole(start);
    }

    for (const VertItr start : simples_) {
      TriangulatePoly(start);
    }

    return triangles_;
  }

  float GetPrecision() const { return precision_; }

 private:
  struct Vert;
  typedef std::vector<Vert>::iterator VertItr;
  typedef std::vector<Vert>::const_iterator VertItrC;
  struct MaxX {
    bool operator()(const VertItr &a, const VertItr &b) const {
      return a->pos.x > b->pos.x;
    }
  };
  struct MinCost {
    bool operator()(const VertItr &a, const VertItr &b) const {
      return a->cost < b->cost;
    }
  };
  typedef std::set<VertItr, MinCost>::iterator qItr;

  // The flat list where all the Verts are stored. Not used much for traversal.
  std::vector<Vert> polygon_;
  // The set of right-most starting points, one for each negative-area contour.
  std::multiset<VertItr, MaxX> holes_;
  // The set of starting points, one for each positive-area contour.
  std::vector<VertItr> outers_;
  // The set of starting points, one for each simple polygon.
  std::vector<VertItr> simples_;
  // Maps each hole (by way of starting point) to its bounding box.
  std::map<VertItr, Rect> hole2BBox_;
  // A priority queue of valid ears - the multiset allows them to be updated.
  std::multiset<VertItr, MinCost> earsQueue_;
  // The output triangulation.
  std::vector<glm::ivec3> triangles_;
  // Bounding box of the entire set of polygons
  Rect bBox_;
  // Working precision: max of float error and input value.
  float precision_;

  struct IdxCollider {
    Collider collider;
    std::vector<VertItr> itr;
  };

  // A circularly-linked list representing the polygon(s) that still need to be
  // triangulated. This gets smaller as ears are clipped until it degenerates to
  // two points and terminates.
  struct Vert {
    int mesh_idx;
    float cost;
    qItr ear;
    glm::vec2 pos, rightDir;
    VertItr left, right;

    // Shorter than half of precision, to be conservative so that it doesn't
    // cause CW triangles that exceed precision due to rounding error.
    bool IsShort(float precision) const {
      const glm::vec2 edge = right->pos - pos;
      return glm::dot(edge, edge) * 4 < precision * precision;
    }

    // Like CCW, returns 1 if v is on the inside of the angle formed at this
    // vert, -1 on the outside, and 0 if it's within precision of the boundary.
    // Ensure v is more than precision from pos, as case this will not return 0.
    int Interior(glm::vec2 v, float precision) const {
      const glm::vec2 diff = v - pos;
      if (glm::dot(diff, diff) < precision * precision) {
        return 0;
      }
      return CCW(pos, left->pos, right->pos, precision) +
             CCW(pos, right->pos, v, precision) +
             CCW(pos, v, left->pos, precision);
    }

    // Returns true if Vert is on the inside of the edge that goes from tail to
    // tail->right. This will walk the edges if necessary until a clear answer
    // is found (beyond precision). If toLeft is true, this Vert will walk its
    // edges to the left. This should be chosen so that the edges walk in the
    // same general direction - tail always walks to the right.
    bool InsideEdge(VertItr tail, float precision, bool toLeft) const {
      const float p2 = precision * precision;
      VertItr nextL = left->right;
      VertItr nextR = tail->right;
      VertItr center = tail;
      VertItr last = center;

      while (nextL != nextR && tail != nextR &&
             nextL != (toLeft ? right : left)) {
        const glm::vec2 edgeL = nextL->pos - center->pos;
        const float l2 = glm::dot(edgeL, edgeL);
        if (l2 <= p2) {
          nextL = toLeft ? nextL->left : nextL->right;
          continue;
        }

        const glm::vec2 edgeR = nextR->pos - center->pos;
        const float r2 = glm::dot(edgeR, edgeR);
        if (r2 <= p2) {
          nextR = nextR->right;
          continue;
        }

        const glm::vec2 vecLR = nextR->pos - nextL->pos;
        const float lr2 = glm::dot(vecLR, vecLR);
        if (lr2 <= p2) {
          last = center;
          center = nextL;
          nextL = toLeft ? nextL->left : nextL->right;
          if (nextL == nextR) break;
          nextR = nextR->right;
          continue;
        }

        int convexity = CCW(nextL->pos, center->pos, nextR->pos, precision);
        if (center != last) {
          convexity += CCW(last->pos, center->pos, nextL->pos, precision) +
                       CCW(nextR->pos, center->pos, last->pos, precision);
        }
        if (convexity != 0) return convexity > 0;

        if (l2 < r2) {
          center = nextL;
          nextL = toLeft ? nextL->left : nextL->right;
        } else {
          center = nextR;
          nextR = nextR->right;
        }
        last = center;
      }
      // The whole polygon is degenerate - consider this to be convex.
      return true;
    }

    // A major key to robustness is to only clip convex ears, but this is
    // difficult to determine when an edge is folded back on itself. This
    // function walks down the kinks in a degenerate portion of a polygon until
    // it finds a clear geometric result. In the vast majority of cases the loop
    // will only need one or two iterations.
    bool IsConvex(float precision) const {
      const int convexity = CCW(left->pos, pos, right->pos, precision);
      if (convexity != 0) {
        return convexity > 0;
      }
      if (glm::dot(left->pos - pos, right->pos - pos) <= 0) {
        return true;
      }
      return left->InsideEdge(left->right, precision, true);
    }

    // Subtly different from !IsConvex because IsConvex will return true for
    // colinear non-folded verts, while IsReflex will always check until actual
    // certainty is determined.
    bool IsReflex(float precision) const {
      return !left->InsideEdge(left->right, precision, true);
    }

    // This function is the core of finding a proper place to keyhole. It runs
    // on this Vert, which represents the edge from this to right. It returns
    // an iterator to the vert to connect to (either this or right) and a bool
    // denoting if the edge is a valid option for a keyhole (must be upwards and
    // cross the start.y-value).
    //
    // If the edge terminates within the precision band, it checks the next edge
    // to ensure validity. No while loop is necessary because short edges have
    // already been removed. The onTop value is 1 if the start.y-value is at the
    // top of the polygon's bounding box, -1 if it's at the bottom, and 0
    // otherwise. This allows proper handling of horizontal edges.
    std::pair<VertItr, bool> InterpY2X(glm::vec2 start, int onTop,
                                       float precision) const {
      const auto none = std::make_pair(left, false);
      if (pos.y < start.y && right->pos.y >= start.y &&
          (pos.x > start.x - precision || right->pos.x > start.x - precision)) {
        return std::make_pair(left->right, true);
      } else if (onTop != 0 && pos.x > start.x - precision &&
                 pos.y > start.y - precision && pos.y < start.y + precision &&
                 Interior(start, precision) >= 0) {
        if (onTop > 0 && left->pos.x < pos.x &&
            left->pos.y > start.y - precision) {
          return none;
        }
        if (onTop < 0 && right->pos.x < pos.x &&
            right->pos.y < start.y + precision) {
          return none;
        }
        const VertItr p = pos.x < right->pos.x ? right : left->right;
        return std::make_pair(p, true);
      }
      // Edge does not cross start.y going up
      return none;
    }

    // This finds the cost of this vert relative to one of the two closed sides
    // of the ear. Points are valid even when they touch, so long as their edge
    // goes to the outside. No need to check the other side, since all verts are
    // processed in the EarCost loop.
    float SignedDist(VertItr v, glm::vec2 unit, float precision) const {
      float d = determinant2x2(unit, v->pos - pos);
      if (std::abs(d) < precision) {
        float dR = determinant2x2(unit, v->right->pos - pos);
        if (std::abs(dR) > precision) return dR;
        float dL = determinant2x2(unit, v->left->pos - pos);
        if (std::abs(dL) > precision) return dL;
      }
      return d;
    }

    // Find the cost of Vert v within this ear, where openSide is the unit
    // vector from Verts right to left - passed in for reuse.
    float Cost(VertItr v, glm::vec2 openSide, float precision) const {
      float cost = glm::min(SignedDist(v, rightDir, precision),
                            SignedDist(v, left->rightDir, precision));

      const float openCost = determinant2x2(openSide, v->pos - right->pos);
      return glm::min(cost, openCost);
    }

    // For verts outside the ear, apply a cost based on the Delaunay condition
    // to aid in prioritization and produce cleaner triangulations. This doesn't
    // affect robustness, but may be adjusted to improve output.
    static float DelaunayCost(glm::vec2 diff, float scale, float precision) {
      return -precision - scale * glm::dot(diff, diff);
    }

    // This is the expensive part of the algorithm, checking this ear against
    // every Vert to ensure none are inside. The Collider brings the total
    // triangulator cost down from O(n^2) to O(nlogn) for most large polygons.
    //
    // Think of a cost as vaguely a distance metric - 0 is right on the edge of
    // being invalid. cost > precision is definitely invalid. Cost < -precision
    // is definitely valid, so all improvement costs are designed to always give
    // values < -precision so they will never affect validity. The first
    // totalCost is designed to give priority to sharper angles. Any cost < (-1
    // - precision) has satisfied the Delaunay condition.
    float EarCost(float precision, const IdxCollider &collider) const {
      glm::vec2 openSide = left->pos - right->pos;
      const glm::vec2 center = 0.5f * (left->pos + right->pos);
      const float scale = 4 / glm::dot(openSide, openSide);
      const float radius = glm::length(openSide) / 2;
      openSide = glm::normalize(openSide);

      float totalCost = glm::dot(left->rightDir, rightDir) - 1 - precision;
      if (CCW(pos, left->pos, right->pos, precision) == 0) {
        // Clip folded ears first
        return totalCost;
      }

      Vec<Box> earBox;
      earBox.push_back({glm::vec3(center.x - radius, center.y - radius, 0),
                        glm::vec3(center.x + radius, center.y + radius, 0)});
      earBox.back().Union(glm::vec3(pos, 0));
      const SparseIndices toTest = collider.collider.Collisions(earBox.cview());

      const int lid = left->mesh_idx;
      const int rid = right->mesh_idx;
      for (size_t i = 0; i < toTest.size(); ++i) {
        const VertItr test = collider.itr[toTest.Get(i, true)];
        if (!Clipped(test) && test->mesh_idx != mesh_idx &&
            test->mesh_idx != lid &&
            test->mesh_idx != rid) {  // Skip duplicated verts
          float cost = Cost(test, openSide, precision);
          if (cost < -precision) {
            cost = DelaunayCost(test->pos - center, scale, precision);
          }
          totalCost = glm::max(totalCost, cost);
        }
      }

      return totalCost;
    }

    void PrintVert() const {
#ifdef MANIFOLD_DEBUG
      if (!params.verbose) return;
      std::cout << "vert: " << mesh_idx << ", left: " << left->mesh_idx
                << ", right: " << right->mesh_idx << ", cost: " << cost
                << std::endl;
#endif
    }
  };

  static glm::vec2 SafeNormalize(glm::vec2 v) {
    glm::vec2 n = glm::normalize(v);
    return glm::isfinite(n.x) ? n : glm::vec2(0, 0);
  }

  // This function and JoinPolygons are the only functions that affect the
  // circular list data structure. This helps ensure it remains circular.
  static void Link(VertItr left, VertItr right) {
    left->right = right;
    right->left = left;
    left->rightDir = SafeNormalize(right->pos - left->pos);
  }

  // When an ear vert is clipped, its neighbors get linked, so they get unlinked
  // from it, but it is still linked to them.
  static bool Clipped(VertItr v) { return v->right->left != v; }

  // Apply func to each un-clipped vert in a polygon and return an un-clipped
  // vert.
  VertItrC Loop(VertItr first, std::function<void(VertItr)> func) const {
    VertItr v = first;
    do {
      if (Clipped(v)) {
        // Update first to an un-clipped vert so we will return to it instead
        // of infinite-looping.
        first = v->right->left;
        if (!Clipped(first)) {
          v = first;
          if (v->right == v->left) {
            return polygon_.end();
          }
          func(v);
        }
      } else {
        if (v->right == v->left) {
          return polygon_.end();
        }
        func(v);
      }
      v = v->right;
    } while (v != first);
    return v;
  }

  // Remove this vert from the circular list and output a corresponding
  // triangle.
  void ClipEar(VertItrC ear) {
    Link(ear->left, ear->right);
    if (ear->left->mesh_idx != ear->mesh_idx &&
        ear->mesh_idx != ear->right->mesh_idx &&
        ear->right->mesh_idx != ear->left->mesh_idx) {
      // Filter out topological degenerates, which can form in bad
      // triangulations of polygons with holes, due to vert duplication.
      triangles_.push_back(
          {ear->left->mesh_idx, ear->mesh_idx, ear->right->mesh_idx});
    } else {
      PRINT("Topological degenerate!");
    }
  }

  // If an ear will make a degenerate triangle, clip it early to avoid
  // difficulty in key-holing. This function is recursive, as the process of
  // clipping may cause the neighbors to degenerate. Reflex degenerates *must
  // not* be clipped, unless they have a short edge.
  void ClipIfDegenerate(VertItr ear) {
    if (Clipped(ear)) {
      return;
    }
    if (ear->left == ear->right) {
      return;
    }
    if (ear->IsShort(precision_) ||
        (CCW(ear->left->pos, ear->pos, ear->right->pos, precision_) == 0 &&
         glm::dot(ear->left->pos - ear->pos, ear->right->pos - ear->pos) > 0 &&
         ear->IsConvex(precision_))) {
      ClipEar(ear);
      ClipIfDegenerate(ear->left);
      ClipIfDegenerate(ear->right);
    }
  }

  // Build the circular list polygon structures.
  std::vector<VertItr> Initialize(const PolygonsIdx &polys) {
    std::vector<VertItr> starts;
    for (const SimplePolygonIdx &poly : polys) {
      auto vert = poly.begin();
      polygon_.push_back({vert->idx, 0.0f, earsQueue_.end(), vert->pos});
      const VertItr first = std::prev(polygon_.end());

      bBox_.Union(first->pos);
      VertItr last = first;
      // This is not the real rightmost start, but just an arbitrary vert for
      // now to identify each polygon.
      starts.push_back(first);

      for (++vert; vert != poly.end(); ++vert) {
        bBox_.Union(vert->pos);

        polygon_.push_back({vert->idx, 0.0f, earsQueue_.end(), vert->pos});
        VertItr next = std::prev(polygon_.end());

        Link(last, next);
        last = next;
      }
      Link(last, first);
    }

    if (precision_ < 0) precision_ = bBox_.Scale() * kTolerance;

    // Slightly more than enough, since each hole can cause two extra triangles.
    triangles_.reserve(polygon_.size() + 2 * starts.size());
    return starts;
  }

  // Find the actual rightmost starts after degenerate removal. Also calculate
  // the polygon bounding boxes.
  void FindStart(VertItr first) {
    const glm::vec2 origin = first->pos;

    VertItr start = first;
    float maxX = -std::numeric_limits<float>::infinity();
    Rect bBox;
    // Kahan summation
    double area = 0;
    double areaCompensation = 0;

    auto AddPoint = [&](VertItr v) {
      bBox.Union(v->pos);
      const double area1 =
          determinant2x2(v->pos - origin, v->right->pos - origin);
      const double t1 = area + area1;
      areaCompensation += (area - t1) + area1;
      area = t1;

      if (v->pos.x > maxX) {
        maxX = v->pos.x;
        start = v;
      }
    };

    if (Loop(first, AddPoint) == polygon_.end()) {
      // No polygon left if all ears were degenerate and already clipped.
      return;
    }

    area += areaCompensation;
    const glm::vec2 size = bBox.Size();
    const float minArea = precision_ * glm::max(size.x, size.y);

    if (glm::isfinite(maxX) && area < -minArea) {
      holes_.insert(start);
      hole2BBox_.insert({start, bBox});
    } else {
      simples_.push_back(start);
      if (area > minArea) {
        outers_.push_back(start);
      }
    }
  }

  // All holes must be key-holed (attached to an outer polygon) before ear
  // clipping can commence. Instead of relying on sorting, which may be
  // incorrect due to precision, we check for polygon edges both ahead and
  // behind to ensure all valid options are found.
  void CutKeyhole(const VertItr start) {
    const Rect bBox = hole2BBox_[start];
    const int onTop = start->pos.y >= bBox.max.y - precision_   ? 1
                      : start->pos.y <= bBox.min.y + precision_ ? -1
                                                                : 0;
    VertItr connector = polygon_.end();

    auto CheckEdge = [&](VertItr edge) {
      const std::pair<VertItr, bool> pair =
          edge->InterpY2X(start->pos, onTop, precision_);
      if (pair.second && start->InsideEdge(pair.first, precision_, true) &&
          (connector == polygon_.end() ||
           (connector->pos.y < pair.first->pos.y
                ? pair.first->InsideEdge(connector, precision_, false)
                : !connector->InsideEdge(pair.first, precision_, false)))) {
        connector = pair.first;
      }
    };

    for (const VertItr first : outers_) {
      Loop(first, CheckEdge);
    }

    if (connector == polygon_.end()) {
      PRINT("hole did not find an outer contour!");
      simples_.push_back(start);
      return;
    }

    connector = FindCloserBridge(start, connector, onTop);

    JoinPolygons(start, connector);

#ifdef MANIFOLD_DEBUG
    if (params.verbose) {
      std::cout << "connected " << start->mesh_idx << " to "
                << connector->mesh_idx << std::endl;
    }
#endif
  }

  // This converts the initial guess for the keyhole location into the final one
  // and returns it. It does so by finding any reflex verts inside the triangle
  // containing the best connection and the initial horizontal line.
  VertItr FindCloserBridge(VertItr start, VertItr edge, int onTop) {
    VertItr best = edge->pos.x > edge->right->pos.x ? edge : edge->right;
    const float maxX = best->pos.x;
    const float above = best->pos.y > start->pos.y ? 1 : -1;

    auto CheckVert = [&](VertItr vert) {
      const float inside = above * CCW(start->pos, vert->pos, best->pos, 0);
      if (vert->pos.x > start->pos.x - precision_ &&
          vert->pos.x < maxX + precision_ &&
          vert->pos.y * above > start->pos.y * above - precision_ &&
          (inside > 0 || (inside == 0 && vert->pos.x < best->pos.x)) &&
          vert->InsideEdge(edge, precision_, true) &&
          vert->IsReflex(precision_)) {
        if (vert->pos.y > start->pos.y - precision_ &&
            vert->pos.y < start->pos.y + precision_) {
          if (onTop > 0 && vert->left->pos.x < vert->pos.x &&
              vert->left->pos.y > start->pos.y - precision_) {
            return;
          }
          if (onTop < 0 && vert->right->pos.x < vert->pos.x &&
              vert->right->pos.y < start->pos.y + precision_) {
            return;
          }
        }
        best = vert;
      }
    };

    for (const VertItr first : outers_) {
      Loop(first, CheckVert);
    }

    return best;
  }

  // Creates a keyhole between the start vert of a hole and the connector vert
  // of an outer polygon. To do this, both verts are duplicated and reattached.
  // This process may create degenerate ears, so these are clipped if necessary
  // to keep from confusing subsequent key-holing operations.
  void JoinPolygons(VertItr start, VertItr connector) {
    polygon_.push_back(*start);
    const VertItr newStart = std::prev(polygon_.end());
    polygon_.push_back(*connector);
    const VertItr newConnector = std::prev(polygon_.end());

    start->right->left = newStart;
    connector->left->right = newConnector;
    Link(start, connector);
    Link(newConnector, newStart);

    ClipIfDegenerate(start);
    ClipIfDegenerate(newStart);
    ClipIfDegenerate(connector);
    ClipIfDegenerate(newConnector);
  }

  // Recalculate the cost of the Vert v ear, updating it in the queue by
  // removing and reinserting it.
  void ProcessEar(VertItr v, const IdxCollider &collider) {
    if (v->ear != earsQueue_.end()) {
      earsQueue_.erase(v->ear);
      v->ear = earsQueue_.end();
    }
    if (v->IsShort(precision_)) {
      v->cost = kBest;
      v->ear = earsQueue_.insert(v);
    } else if (v->IsConvex(2 * precision_)) {
      v->cost = v->EarCost(precision_, collider);
      v->ear = earsQueue_.insert(v);
    } else {
      v->cost = 1;  // not used, but marks reflex verts for debug
    }
  }

  // Create a collider of all vertices in this polygon, each expanded by
  // precision_. Each ear uses this BVH to quickly find a subset of vertices to
  // check for cost.
  IdxCollider VertCollider(VertItr start) const {
    Vec<Box> vertBox;
    Vec<uint32_t> vertMorton;
    std::vector<VertItr> itr;
    const Box box(glm::vec3(bBox_.min, 0), glm::vec3(bBox_.max, 0));

    Loop(start, [&vertBox, &vertMorton, &itr, &box, this](VertItr v) {
      itr.push_back(v);
      const glm::vec3 pos(v->pos, 0);
      vertBox.push_back({pos - precision_, pos + precision_});
      vertMorton.push_back(Collider::MortonCode(pos, box));
    });

    if (itr.empty()) {
      return {Collider(), itr};
    }

    const int numVert = itr.size();
    Vec<int> vertNew2Old(numVert);
    sequence(vertNew2Old.begin(), vertNew2Old.end());

    stable_sort(vertNew2Old.begin(), vertNew2Old.end(),
                [&vertMorton](const int a, const int b) {
                  return vertMorton[a] < vertMorton[b];
                });
    Permute(vertMorton, vertNew2Old);
    Permute(vertBox, vertNew2Old);
    Permute(itr, vertNew2Old);

    return {Collider(vertBox, vertMorton), itr};
  }

  // The main ear-clipping loop. This is called once for each simple polygon -
  // all holes have already been key-holed and joined to an outer polygon.
  void TriangulatePoly(VertItr start) {
    ZoneScoped;

    const IdxCollider vertCollider = VertCollider(start);

    if (vertCollider.itr.empty()) {
      PRINT("Empty poly");
      return;
    }

    // A simple polygon always creates two fewer triangles than it has verts.
    int numTri = -2;
    earsQueue_.clear();

    auto QueueVert = [&](VertItr v) {
      ProcessEar(v, vertCollider);
      ++numTri;
      v->PrintVert();
    };

    VertItrC v = Loop(start, QueueVert);
    if (v == polygon_.end()) return;
    Dump(v);

    while (numTri > 0) {
      const qItr ear = earsQueue_.begin();
      if (ear != earsQueue_.end()) {
        v = *ear;
        // Cost should always be negative, generally < -precision.
        v->PrintVert();
        earsQueue_.erase(ear);
      } else {
        PRINT("No ear found!");
      }

      ClipEar(v);
      --numTri;

      ProcessEar(v->left, vertCollider);
      ProcessEar(v->right, vertCollider);
      // This is a backup vert that is used if the queue is empty (geometrically
      // invalid polygon), to ensure manifoldness.
      v = v->right;
    }

    DEBUG_ASSERT(v->right == v->left, logicErr, "Triangulator error!");
    PRINT("Finished poly");
  }

  void Dump(VertItrC start) const {
#ifdef MANIFOLD_DEBUG
    if (!params.verbose) return;
    VertItrC v = start;
    std::cout << "show(array([" << std::endl;
    do {
      std::cout << "  [" << v->pos.x << ", " << v->pos.y << "],# "
                << v->mesh_idx << ", cost: " << v->cost << std::endl;
      v = v->right;
    } while (v != start);
    std::cout << "  [" << v->pos.x << ", " << v->pos.y << "],# " << v->mesh_idx
              << std::endl;
    std::cout << "]))" << std::endl;

    v = start;
    std::cout << "polys.push_back({" << std::setprecision(9) << std::endl;
    do {
      std::cout << "    {" << v->pos.x << ", " << v->pos.y << "},  //"
                << std::endl;
      v = v->right;
    } while (v != start);
    std::cout << "});" << std::endl;
#endif
  }
};
}  // namespace

namespace manifold {

/**
 * @brief Triangulates a set of &epsilon;-valid polygons. If the input is not
 * &epsilon;-valid, the triangulation may overlap, but will always return a
 * manifold result that matches the input edge directions.
 *
 * @param polys The set of polygons, wound CCW and representing multiple
 * polygons and/or holes. These have 2D-projected positions as well as
 * references back to the original vertices.
 * @param precision The value of &epsilon;, bounding the uncertainty of the
 * input.
 * @return std::vector<glm::ivec3> The triangles, referencing the original
 * vertex indicies.
 */
std::vector<glm::ivec3> TriangulateIdx(const PolygonsIdx &polys,
                                       float precision) {
  std::vector<glm::ivec3> triangles;
  float updatedPrecision = precision;
#if MANIFOLD_EXCEPTION
  try {
#endif
    if (IsConvex(polys, precision)) {  // fast path
      triangles = TriangulateConvex(polys);
    } else {
      EarClip triangulator(polys, precision);
      triangles = triangulator.Triangulate();
      updatedPrecision = triangulator.GetPrecision();
    }
#if MANIFOLD_EXCEPTION
#ifdef MANIFOLD_DEBUG
    if (params.intermediateChecks) {
      CheckTopology(triangles, polys);
      if (!params.processOverlaps) {
        CheckGeometry(triangles, polys, 2 * updatedPrecision);
      }
    }
  } catch (const geometryErr &e) {
    if (!params.suppressErrors) {
      PrintFailure(e, polys, triangles, updatedPrecision);
    }
    throw;
  } catch (const std::exception &e) {
    PrintFailure(e, polys, triangles, updatedPrecision);
    throw;
#else
  } catch (const std::exception &e) {
#endif
  }
#endif
  return triangles;
}

/**
 * @brief Triangulates a set of &epsilon;-valid polygons. If the input is not
 * &epsilon;-valid, the triangulation may overlap, but will always return a
 * manifold result that matches the input edge directions.
 *
 * @param polygons The set of polygons, wound CCW and representing multiple
 * polygons and/or holes.
 * @param precision The value of &epsilon;, bounding the uncertainty of the
 * input.
 * @return std::vector<glm::ivec3> The triangles, referencing the original
 * polygon points in order.
 */
std::vector<glm::ivec3> Triangulate(const Polygons &polygons, float precision) {
  int idx = 0;
  PolygonsIdx polygonsIndexed;
  for (const auto &poly : polygons) {
    SimplePolygonIdx simpleIndexed;
    for (const glm::vec2 &polyVert : poly) {
      simpleIndexed.push_back({polyVert, idx++});
    }
    polygonsIndexed.push_back(simpleIndexed);
  }
  return TriangulateIdx(polygonsIndexed, precision);
}

ExecutionParams &PolygonParams() { return params; }

}  // namespace manifold
