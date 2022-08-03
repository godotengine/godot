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

#include <../third_party/thrust/thrust/sequence.h>

#include "csg_tree.h"
#include "graph.h"
#include "impl.h"
#include "par.h"
#include "polygon.h"

namespace {
using namespace manifold;
using namespace thrust::placeholders;

struct ToSphere {
  float length;
  __host__ __device__ void operator()(glm::vec3& v) {
    v = glm::cos(glm::half_pi<float>() * (1.0f - v));
    v = length * glm::normalize(v);
    if (isnan(v.x)) v = glm::vec3(0.0);
  }
};

struct Equals {
  int val;
  __host__ __device__ bool operator()(int x) { return x == val; }
};

struct RemoveFace {
  const Halfedge* halfedge;
  const int* vertLabel;
  const int keepLabel;

  __host__ __device__ bool operator()(int face) {
    return vertLabel[halfedge[3 * face].startVert] != keepLabel;
  }
};
}  // namespace

namespace manifold {
/**
 * Constructs a smooth version of the input mesh by creating tangents; this
 * method will throw if you have supplied tangnets with your mesh already. The
 * actual triangle resolution is unchanged; use the Refine() method to
 * interpolate to a higher-resolution curve.
 *
 * By default, every edge is calculated for maximum smoothness (very much
 * approximately), attempting to minimize the maximum mean Curvature magnitude.
 * No higher-order derivatives are considered, as the interpolation is
 * independent per triangle, only sharing constraints on their boundaries.
 *
 * @param mesh input Mesh.
 * @param sharpenedEdges If desired, you can supply a vector of sharpened
 * halfedges, which should in general be a small subset of all halfedges. Order
 * of entries doesn't matter, as each one specifies the desired smoothness
 * (between zero and one, with one the default for all unspecified halfedges)
 * and the halfedge index (3 * triangle index + [0,1,2] where 0 is the edge
 * between triVert 0 and 1, etc).
 *
 * At a smoothness value of zero, a sharp crease is made. The smoothness is
 * interpolated along each edge, so the specified value should be thought of as
 * an average. Where exactly two sharpened edges meet at a vertex, their
 * tangents are rotated to be colinear so that the sharpened edge can be
 * continuous. Vertices with only one sharpened edge are completely smooth,
 * allowing sharpened edges to smoothly vanish at termination. A single vertex
 * can be sharpened by sharping all edges that are incident on it, allowing
 * cones to be formed.
 */
Manifold Manifold::Smooth(const Mesh& mesh,
                          const std::vector<Smoothness>& sharpenedEdges) {
  ASSERT(mesh.halfedgeTangent.empty(), std::runtime_error,
         "when supplying tangents, the normal constructor should be used "
         "rather than Smooth().");

  std::shared_ptr<Impl> impl = std::make_shared<Impl>(mesh);
  impl->CreateTangents(sharpenedEdges);
  return Manifold(impl);
}

/**
 * Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
 * and the rest at similarly symmetric points.
 */
Manifold Manifold::Tetrahedron() {
  return Manifold(std::make_shared<Impl>(Impl::Shape::TETRAHEDRON));
}

/**
 * Constructs a unit cube (edge lengths all one), by default in the first
 * octant, touching the origin.
 *
 * @param size The X, Y, and Z dimensions of the box.
 * @param center Set to true to shift the center to the origin.
 */
Manifold Manifold::Cube(glm::vec3 size, bool center) {
  auto cube = Manifold(std::make_shared<Impl>(Impl::Shape::CUBE));
  cube = cube.Scale(size);
  if (center) cube = cube.Translate(-size / 2.0f);
  return cube;
}

/**
 * A convenience constructor for the common case of extruding a circle. Can also
 * form cones if both radii are specified.
 *
 * @param height Z-extent
 * @param radiusLow Radius of bottom circle. Must be positive.
 * @param radiusHigh Radius of top circle. Can equal zero. Default is equal to
 * radiusLow.
 * @param circularSegments How many line segments to use around the circle.
 * Default is calculated by the static Defaults.
 * @param center Set to true to shift the center to the origin. Default is
 * origin at the bottom.
 */
Manifold Manifold::Cylinder(float height, float radiusLow, float radiusHigh,
                            int circularSegments, bool center) {
  float scale = radiusHigh >= 0.0f ? radiusHigh / radiusLow : 1.0f;
  float radius = fmax(radiusLow, radiusHigh);
  int n = circularSegments > 2 ? circularSegments : GetCircularSegments(radius);
  Polygons circle(1);
  float dPhi = 360.0f / n;
  for (int i = 0; i < n; ++i) {
    circle[0].push_back(
        {radiusLow * glm::vec2(cosd(dPhi * i), sind(dPhi * i)), 0});
  }
  Manifold cylinder =
      Manifold::Extrude(circle, height, 0, 0.0f, glm::vec2(scale));
  if (center)
    cylinder = cylinder.Translate(glm::vec3(0.0f, 0.0f, -height / 2.0f));
  return cylinder;
}

/**
 * Constructs a geodesic sphere of a given radius.
 *
 * @param radius Radius of the sphere. Must be positive.
 * @param circularSegments Number of segments along its
 * diameter. This number will always be rounded up to the nearest factor of
 * four, as this sphere is constructed by refining an octahedron. This means
 * there are a circle of vertices on all three of the axis planes. Default is
 * calculated by the static Defaults.
 */
Manifold Manifold::Sphere(float radius, int circularSegments) {
  int n = circularSegments > 0 ? (circularSegments + 3) / 4
                               : GetCircularSegments(radius) / 4;
  auto pImpl_ = std::make_shared<Impl>(Impl::Shape::OCTAHEDRON);
  pImpl_->Subdivide(n);
  for_each_n(autoPolicy(pImpl_->NumVert()), pImpl_->vertPos_.begin(),
             pImpl_->NumVert(), ToSphere({radius}));
  pImpl_->Finish();
  // Ignore preceding octahedron.
  pImpl_->ReinitializeReference(Impl::meshIDCounter_.fetch_add(1));
  return Manifold(pImpl_);
}

/**
 * Constructs a manifold from a set of polygons by extruding them along the
 * Z-axis.
 *
 * @param crossSection A set of non-overlapping polygons to extrude.
 * @param height Z-extent of extrusion.
 * @param nDivisions Number of extra copies of the crossSection to insert into
 * the shape vertically; especially useful in combnation with twistDegrees to
 * avoid interpolation artifacts. Default is none.
 * @param twistDegrees Amount to twist the top crossSection relative to the
 * bottom, interpolated linearly for the divisions in between.
 * @param scaleTop Amount to scale the top (independently in X and Y). If the
 * scale is {0, 0}, a pure cone is formed with only a single vertex at the top.
 * Default {1, 1}.
 */
Manifold Manifold::Extrude(Polygons crossSection, float height, int nDivisions,
                           float twistDegrees, glm::vec2 scaleTop) {
  scaleTop.x = glm::max(scaleTop.x, 0.0f);
  scaleTop.y = glm::max(scaleTop.y, 0.0f);

  auto pImpl_ = std::make_shared<Impl>();
  ++nDivisions;
  auto& vertPos = pImpl_->vertPos_;
  VecDH<glm::ivec3> triVertsDH;
  auto& triVerts = triVertsDH;
  int nCrossSection = 0;
  bool isCone = scaleTop.x == 0.0 && scaleTop.y == 0.0;
  int idx = 0;
  for (auto& poly : crossSection) {
    nCrossSection += poly.size();
    for (PolyVert& polyVert : poly) {
      vertPos.push_back({polyVert.pos.x, polyVert.pos.y, 0.0f});
      polyVert.idx = idx++;
    }
  }
  for (int i = 1; i < nDivisions + 1; ++i) {
    float alpha = i / float(nDivisions);
    float phi = alpha * twistDegrees;
    glm::mat2 transform(cosd(phi), sind(phi), -sind(phi), cosd(phi));
    glm::vec2 scale = glm::mix(glm::vec2(1.0f), scaleTop, alpha);
    transform = transform * glm::mat2(scale.x, 0.0f, 0.0f, scale.y);
    int j = 0;
    int idx = 0;
    for (const auto& poly : crossSection) {
      for (int vert = 0; vert < poly.size(); ++vert) {
        int offset = idx + nCrossSection * i;
        int thisVert = vert + offset;
        int lastVert = (vert == 0 ? poly.size() : vert) - 1 + offset;
        if (i == nDivisions && isCone) {
          triVerts.push_back({nCrossSection * i + j, lastVert - nCrossSection,
                              thisVert - nCrossSection});
        } else {
          glm::vec2 pos = transform * poly[vert].pos;
          vertPos.push_back({pos.x, pos.y, height * alpha});
          triVerts.push_back({thisVert, lastVert, thisVert - nCrossSection});
          triVerts.push_back(
              {lastVert, lastVert - nCrossSection, thisVert - nCrossSection});
        }
      }
      ++j;
      idx += poly.size();
    }
  }
  if (isCone)
    for (int j = 0; j < crossSection.size(); ++j)  // Duplicate vertex for Genus
      vertPos.push_back({0.0f, 0.0f, height});
  std::vector<glm::ivec3> top = Triangulate(crossSection);
  for (const glm::ivec3& tri : top) {
    triVerts.push_back({tri[0], tri[2], tri[1]});
    if (!isCone) triVerts.push_back(tri + nCrossSection * nDivisions);
  }

  pImpl_->CreateHalfedges(triVertsDH);
  pImpl_->Finish();
  pImpl_->InitializeNewReference();
  return Manifold(pImpl_);
}

/**
 * Constructs a manifold from a set of polygons by revolving this cross-section
 * around its Y-axis and then setting this as the Z-axis of the resulting
 * manifold. If the polygons cross the Y-axis, only the part on the positive X
 * side is used. Geometrically valid input will result in geometrically valid
 * output.
 *
 * @param crossSection A set of non-overlapping polygons to revolve.
 * @param circularSegments Number of segments along its diameter. Default is
 * calculated by the static Defaults.
 */
Manifold Manifold::Revolve(const Polygons& crossSection, int circularSegments) {
  float radius = 0.0f;
  for (const auto& poly : crossSection) {
    for (const auto& vert : poly) {
      radius = fmax(radius, vert.pos.x);
    }
  }
  int nDivisions =
      circularSegments > 2 ? circularSegments : GetCircularSegments(radius);
  auto pImpl_ = std::make_shared<Impl>();
  auto& vertPos = pImpl_->vertPos_;
  VecDH<glm::ivec3> triVertsDH;
  auto& triVerts = triVertsDH;
  float dPhi = 360.0f / nDivisions;
  for (const auto& poly : crossSection) {
    int start = -1;
    for (int polyVert = 0; polyVert < poly.size(); ++polyVert) {
      if (poly[polyVert].pos.x <= 0) {
        start = polyVert;
        break;
      }
    }
    if (start == -1) {  // poly all positive
      for (int polyVert = 0; polyVert < poly.size(); ++polyVert) {
        int startVert = vertPos.size();
        int lastStart =
            startVert +
            (polyVert == 0 ? nDivisions * (poly.size() - 1) : -nDivisions);
        for (int slice = 0; slice < nDivisions; ++slice) {
          int lastSlice = (slice == 0 ? nDivisions : slice) - 1;
          float phi = slice * dPhi;
          glm::vec2 pos = poly[polyVert].pos;
          vertPos.push_back({pos.x * cosd(phi), pos.x * sind(phi), pos.y});
          triVerts.push_back({startVert + slice, startVert + lastSlice,
                              lastStart + lastSlice});
          triVerts.push_back(
              {lastStart + lastSlice, lastStart + slice, startVert + slice});
        }
      }
    } else {  // poly crosses zero
      int polyVert = start;
      glm::vec2 pos = poly[polyVert].pos;
      do {
        glm::vec2 lastPos = pos;
        polyVert = (polyVert + 1) % poly.size();
        pos = poly[polyVert].pos;
        if (pos.x > 0) {
          if (lastPos.x <= 0) {
            float a = pos.x / (pos.x - lastPos.x);
            vertPos.push_back({0.0f, 0.0f, glm::mix(pos.y, lastPos.y, a)});
          }
          int startVert = vertPos.size();
          for (int slice = 0; slice < nDivisions; ++slice) {
            int lastSlice = (slice == 0 ? nDivisions : slice) - 1;
            float phi = slice * dPhi;
            glm::vec2 pos = poly[polyVert].pos;
            vertPos.push_back({pos.x * cosd(phi), pos.x * sind(phi), pos.y});
            if (lastPos.x > 0) {
              triVerts.push_back({startVert + slice, startVert + lastSlice,
                                  startVert - nDivisions + lastSlice});
              triVerts.push_back({startVert - nDivisions + lastSlice,
                                  startVert - nDivisions + slice,
                                  startVert + slice});
            } else {
              triVerts.push_back(
                  {startVert - 1, startVert + slice, startVert + lastSlice});
            }
          }
        } else if (lastPos.x > 0) {
          int startVert = vertPos.size();
          float a = pos.x / (pos.x - lastPos.x);
          vertPos.push_back({0.0f, 0.0f, glm::mix(pos.y, lastPos.y, a)});
          for (int slice = 0; slice < nDivisions; ++slice) {
            int lastSlice = (slice == 0 ? nDivisions : slice) - 1;
            triVerts.push_back({startVert, startVert - nDivisions + lastSlice,
                                startVert - nDivisions + slice});
          }
        }
      } while (polyVert != start);
    }
  }

  pImpl_->CreateHalfedges(triVertsDH);
  pImpl_->Finish();
  pImpl_->InitializeNewReference();
  return Manifold(pImpl_);
}

/**
 * Constructs a new manifold from a vector of other manifolds. This is a purely
 * topological operation, so care should be taken to avoid creating
 * overlapping results. It is the inverse operation of Decompose().
 *
 * @param manifolds A vector of Manifolds to lazy-union together.
 */
Manifold Manifold::Compose(const std::vector<Manifold>& manifolds) {
  std::vector<std::shared_ptr<CsgLeafNode>> children;
  for (const auto& manifold : manifolds) {
    children.push_back(manifold.pNode_->ToLeafNode());
  }
  return Manifold(std::make_shared<Impl>(CsgLeafNode::Compose(children)));
}

/**
 * This operation returns a vector of Manifolds that are topologically
 * disconnected. If everything is connected, the vector is length one,
 * containing a copy of the original. It is the inverse operation of Compose().
 */
std::vector<Manifold> Manifold::Decompose() const {
  Graph graph;
  auto pImpl_ = GetCsgLeafNode().GetImpl();
  for (int i = 0; i < NumVert(); ++i) {
    graph.add_nodes(i);
  }
  for (const Halfedge& halfedge : pImpl_->halfedge_) {
    if (halfedge.IsForward())
      graph.add_edge(halfedge.startVert, halfedge.endVert);
  }
  std::vector<int> components;
  const int numLabel = ConnectedComponents(components, graph);

  if (numLabel == 1) {
    std::vector<Manifold> meshes(1);
    meshes[0] = *this;
    return meshes;
  }
  VecDH<int> vertLabel(components);

  std::vector<Manifold> meshes;
  for (int i = 0; i < numLabel; ++i) {
    auto impl = std::make_shared<Impl>();
    // inherit original object's precision
    impl->precision_ = pImpl_->precision_;
    impl->vertPos_.resize(NumVert());
    VecDH<int> vertNew2Old(NumVert());
    auto policy = autoPolicy(NumVert());
    auto start = zip(impl->vertPos_.begin(), vertNew2Old.begin());
    int nVert =
        copy_if<decltype(start)>(
            policy, zip(pImpl_->vertPos_.begin(), countAt(0)),
            zip(pImpl_->vertPos_.end(), countAt(NumVert())), vertLabel.begin(),
            zip(impl->vertPos_.begin(), vertNew2Old.begin()), Equals({i})) -
        start;
    impl->vertPos_.resize(nVert);

    VecDH<int> faceNew2Old(NumTri());
    sequence(policy, faceNew2Old.begin(), faceNew2Old.end());

    int nFace =
        remove_if<decltype(faceNew2Old.begin())>(
            policy, faceNew2Old.begin(), faceNew2Old.end(),
            RemoveFace({pImpl_->halfedge_.cptrD(), vertLabel.cptrD(), i})) -
        faceNew2Old.begin();
    faceNew2Old.resize(nFace);

    impl->GatherFaces(*pImpl_, faceNew2Old);
    impl->ReindexVerts(vertNew2Old, pImpl_->NumVert());

    impl->Finish();

    meshes.push_back(Manifold(impl));
  }
  return meshes;
}
}  // namespace manifold
