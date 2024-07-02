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

#include <thrust/sequence.h>

#include "cross_section.h"
#include "csg_tree.h"
#include "impl.h"
#include "par.h"
#include "polygon.h"

namespace {
using namespace manifold;
using namespace thrust::placeholders;

struct ToSphere {
  float length;
  void operator()(glm::vec3& v) {
    v = glm::cos(glm::half_pi<float>() * (1.0f - v));
    v = length * glm::normalize(v);
    if (isnan(v.x)) v = glm::vec3(0.0);
  }
};

struct Equals {
  int val;
  bool operator()(int x) { return x == val; }
};

struct RemoveFace {
  VecView<const Halfedge> halfedge;
  VecView<const int> vertLabel;
  const int keepLabel;

  bool operator()(int face) {
    return vertLabel[halfedge[3 * face].startVert] != keepLabel;
  }
};
}  // namespace

namespace manifold {
/**
 * Constructs a smooth version of the input mesh by creating tangents; this
 * method will throw if you have supplied tangents with your mesh already. The
 * actual triangle resolution is unchanged; use the Refine() method to
 * interpolate to a higher-resolution curve.
 *
 * By default, every edge is calculated for maximum smoothness (very much
 * approximately), attempting to minimize the maximum mean Curvature magnitude.
 * No higher-order derivatives are considered, as the interpolation is
 * independent per triangle, only sharing constraints on their boundaries.
 *
 * @param meshGL input MeshGL.
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
Manifold Manifold::Smooth(const MeshGL& meshGL,
                          const std::vector<Smoothness>& sharpenedEdges) {
  ASSERT(meshGL.halfedgeTangent.empty(), std::runtime_error,
         "when supplying tangents, the normal constructor should be used "
         "rather than Smooth().");

  // Don't allow any triangle merging.
  std::vector<float> propertyTolerance(meshGL.numProp - 3, -1);
  std::shared_ptr<Impl> impl =
      std::make_shared<Impl>(meshGL, propertyTolerance);
  impl->CreateTangents(impl->UpdateSharpenedEdges(sharpenedEdges));
  return Manifold(impl);
}

/**
 * Constructs a smooth version of the input mesh by creating tangents; this
 * method will throw if you have supplied tangents with your mesh already. The
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

  Impl::MeshRelationD relation = {(int)ReserveIDs(1)};
  std::shared_ptr<Impl> impl = std::make_shared<Impl>(mesh, relation);
  impl->CreateTangents(impl->UpdateSharpenedEdges(sharpenedEdges));
  return Manifold(impl);
}

/**
 * Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
 * and the rest at similarly symmetric points.
 */
Manifold Manifold::Tetrahedron() {
  return Manifold(std::make_shared<Impl>(Impl::Shape::Tetrahedron));
}

/**
 * Constructs a unit cube (edge lengths all one), by default in the first
 * octant, touching the origin. If any dimensions in size are negative, or if
 * all are zero, an empty Manifold will be returned.
 *
 * @param size The X, Y, and Z dimensions of the box.
 * @param center Set to true to shift the center to the origin.
 */
Manifold Manifold::Cube(glm::vec3 size, bool center) {
  if (size.x < 0.0f || size.y < 0.0f || size.z < 0.0f ||
      glm::length(size) == 0.) {
    return Invalid();
  }
  glm::mat4x3 m =
      glm::translate(center ? (-size / 2.0f) : glm::vec3(0)) * glm::scale(size);
  return Manifold(std::make_shared<Impl>(Manifold::Impl::Shape::Cube, m));
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
  if (height <= 0.0f || radiusLow <= 0.0f) {
    return Invalid();
  }
  float scale = radiusHigh >= 0.0f ? radiusHigh / radiusLow : 1.0f;
  float radius = fmax(radiusLow, radiusHigh);
  int n = circularSegments > 2 ? circularSegments
                               : Quality::GetCircularSegments(radius);

  CrossSection circle = CrossSection::Circle(radiusLow, n);
  Manifold cylinder =
      Manifold::Extrude(circle, height, 0, 0.0f, glm::vec2(scale));
  if (center)
    cylinder =
        cylinder.Translate(glm::vec3(0.0f, 0.0f, -height / 2.0f)).AsOriginal();
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
  if (radius <= 0.0f) {
    return Invalid();
  }
  int n = circularSegments > 0 ? (circularSegments + 3) / 4
                               : Quality::GetCircularSegments(radius) / 4;
  auto pImpl_ = std::make_shared<Impl>(Impl::Shape::Octahedron);
  pImpl_->Subdivide([n](glm::vec3 edge) { return n - 1; });
  for_each_n(autoPolicy(pImpl_->NumVert()), pImpl_->vertPos_.begin(),
             pImpl_->NumVert(), ToSphere({radius}));
  pImpl_->Finish();
  // Ignore preceding octahedron.
  pImpl_->InitializeOriginal();
  return Manifold(pImpl_);
}

/**
 * Constructs a manifold from a set of polygons by extruding them along the
 * Z-axis.
 * Note that high twistDegrees with small nDivisions may cause
 * self-intersection. This is not checked here and it is up to the user to
 * choose the correct parameters.
 *
 * @param crossSection A set of non-overlapping polygons to extrude.
 * @param height Z-extent of extrusion.
 * @param nDivisions Number of extra copies of the crossSection to insert into
 * the shape vertically; especially useful in combination with twistDegrees to
 * avoid interpolation artifacts. Default is none.
 * @param twistDegrees Amount to twist the top crossSection relative to the
 * bottom, interpolated linearly for the divisions in between.
 * @param scaleTop Amount to scale the top (independently in X and Y). If the
 * scale is {0, 0}, a pure cone is formed with only a single vertex at the top.
 * Note that scale is applied after twist.
 * Default {1, 1}.
 */
Manifold Manifold::Extrude(const CrossSection& crossSection, float height,
                           int nDivisions, float twistDegrees,
                           glm::vec2 scaleTop) {
  ZoneScoped;
  auto polygons = crossSection.ToPolygons();
  if (polygons.size() == 0 || height <= 0.0f) {
    return Invalid();
  }

  scaleTop.x = glm::max(scaleTop.x, 0.0f);
  scaleTop.y = glm::max(scaleTop.y, 0.0f);

  auto pImpl_ = std::make_shared<Impl>();
  ++nDivisions;
  auto& vertPos = pImpl_->vertPos_;
  Vec<glm::ivec3> triVertsDH;
  auto& triVerts = triVertsDH;
  int nCrossSection = 0;
  bool isCone = scaleTop.x == 0.0 && scaleTop.y == 0.0;
  int idx = 0;
  PolygonsIdx polygonsIndexed;
  for (auto& poly : polygons) {
    nCrossSection += poly.size();
    SimplePolygonIdx simpleIndexed;
    for (const glm::vec2& polyVert : poly) {
      vertPos.push_back({polyVert.x, polyVert.y, 0.0f});
      simpleIndexed.push_back({polyVert, idx++});
    }
    polygonsIndexed.push_back(simpleIndexed);
  }
  for (int i = 1; i < nDivisions + 1; ++i) {
    float alpha = i / float(nDivisions);
    float phi = alpha * twistDegrees;
    glm::vec2 scale = glm::mix(glm::vec2(1.0f), scaleTop, alpha);
    glm::mat2 rotation(cosd(phi), sind(phi), -sind(phi), cosd(phi));
    glm::mat2 transform = glm::mat2(scale.x, 0.0f, 0.0f, scale.y) * rotation;
    int j = 0;
    int idx = 0;
    for (const auto& poly : polygons) {
      for (int vert = 0; vert < poly.size(); ++vert) {
        int offset = idx + nCrossSection * i;
        int thisVert = vert + offset;
        int lastVert = (vert == 0 ? poly.size() : vert) - 1 + offset;
        if (i == nDivisions && isCone) {
          triVerts.push_back({nCrossSection * i + j, lastVert - nCrossSection,
                              thisVert - nCrossSection});
        } else {
          glm::vec2 pos = transform * poly[vert];
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
    for (int j = 0; j < polygons.size(); ++j)  // Duplicate vertex for Genus
      vertPos.push_back({0.0f, 0.0f, height});
  std::vector<glm::ivec3> top = TriangulateIdx(polygonsIndexed);
  for (const glm::ivec3& tri : top) {
    triVerts.push_back({tri[0], tri[2], tri[1]});
    if (!isCone) triVerts.push_back(tri + nCrossSection * nDivisions);
  }

  pImpl_->CreateHalfedges(triVertsDH);
  pImpl_->Finish();
  pImpl_->meshRelation_.originalID = ReserveIDs(1);
  pImpl_->InitializeOriginal();
  pImpl_->CreateFaces();
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
 * @param revolveDegrees Number of degrees to revolve. Default is 360 degrees.
 */
Manifold Manifold::Revolve(const CrossSection& crossSection,
                           int circularSegments, float revolveDegrees) {
  ZoneScoped;
  Polygons polygons = crossSection.ToPolygons();

  if (polygons.size() == 0) {
    return Invalid();
  }

  const Rect bounds = crossSection.Bounds();
  const float radius = bounds.max.x;

  if (radius <= 0) {
    return Invalid();
  } else if (bounds.min.x < 0) {
    // Take the x>=0 slice.
    glm::vec2 min = bounds.min;
    glm::vec2 max = bounds.max;
    CrossSection posBoundingBox = CrossSection(
        {{0.0, min.y}, {max.x, min.y}, {max.x, max.y}, {0.0, max.y}});

    polygons = (crossSection ^ posBoundingBox).ToPolygons();
  }

  if (revolveDegrees > 360.0f) {
    revolveDegrees = 360.0f;
  }
  const bool isFullRevolution = revolveDegrees == 360.0f;

  const int nDivisions =
      circularSegments > 2
          ? circularSegments
          : Quality::GetCircularSegments(radius) * revolveDegrees / 360;

  auto pImpl_ = std::make_shared<Impl>();
  auto& vertPos = pImpl_->vertPos_;
  Vec<glm::ivec3> triVertsDH;
  auto& triVerts = triVertsDH;

  std::vector<int> startPoses;
  std::vector<int> endPoses;

  const float dPhi = revolveDegrees / nDivisions;
  // first and last slice are distinguished if not a full revolution.
  const int nSlices = isFullRevolution ? nDivisions : nDivisions + 1;

  for (const auto& poly : polygons) {
    std::size_t nPosVerts = 0;
    std::size_t nRevolveAxisVerts = 0;
    for (auto& pt : poly) {
      if (pt.x > 0) {
        nPosVerts++;
      } else {
        nRevolveAxisVerts++;
      }
    }

    for (int polyVert = 0; polyVert < poly.size(); ++polyVert) {
      const int startPosIndex = vertPos.size();

      if (!isFullRevolution) startPoses.push_back(startPosIndex);

      const glm::vec2 currPolyVertex = poly[polyVert];
      const glm::vec2 prevPolyVertex =
          poly[polyVert == 0 ? poly.size() - 1 : polyVert - 1];

      const int prevStartPosIndex =
          startPosIndex +
          (polyVert == 0 ? nRevolveAxisVerts + (nSlices * nPosVerts) : 0) +
          (prevPolyVertex.x == 0.0 ? -1 : -nSlices);

      for (int slice = 0; slice < nSlices; ++slice) {
        const float phi = slice * dPhi;
        if (slice == 0 || currPolyVertex.x > 0) {
          vertPos.push_back({currPolyVertex.x * cosd(phi),
                             currPolyVertex.x * sind(phi), currPolyVertex.y});
        }

        if (isFullRevolution || slice > 0) {
          const int lastSlice = (slice == 0 ? nDivisions : slice) - 1;
          if (currPolyVertex.x > 0.0) {
            triVerts.push_back(
                {startPosIndex + slice, startPosIndex + lastSlice,
                 // "Reuse" vertex of first slice if it lies on the revolve axis
                 (prevPolyVertex.x == 0.0 ? prevStartPosIndex
                                          : prevStartPosIndex + lastSlice)});
          }

          if (prevPolyVertex.x > 0.0) {
            triVerts.push_back(
                {prevStartPosIndex + lastSlice, prevStartPosIndex + slice,
                 (currPolyVertex.x == 0.0 ? startPosIndex
                                          : startPosIndex + slice)});
          }
        }
      }
      if (!isFullRevolution) endPoses.push_back(vertPos.size() - 1);
    }
  }

  // Add front and back triangles if not a full revolution.
  if (!isFullRevolution) {
    std::vector<glm::ivec3> frontTriangles =
        Triangulate(polygons, pImpl_->precision_);
    for (auto& t : frontTriangles) {
      triVerts.push_back({startPoses[t.x], startPoses[t.y], startPoses[t.z]});
    }

    for (auto& t : frontTriangles) {
      triVerts.push_back({endPoses[t.z], endPoses[t.y], endPoses[t.x]});
    }
  }

  pImpl_->CreateHalfedges(triVertsDH);
  pImpl_->Finish();
  pImpl_->meshRelation_.originalID = ReserveIDs(1);
  pImpl_->InitializeOriginal();
  pImpl_->CreateFaces();
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
  ZoneScoped;
  UnionFind<> uf(NumVert());
  // Graph graph;
  auto pImpl_ = GetCsgLeafNode().GetImpl();
  for (const Halfedge& halfedge : pImpl_->halfedge_) {
    if (halfedge.IsForward()) uf.unionXY(halfedge.startVert, halfedge.endVert);
  }
  std::vector<int> componentIndices;
  const int numComponents = uf.connectedComponents(componentIndices);

  if (numComponents == 1) {
    std::vector<Manifold> meshes(1);
    meshes[0] = *this;
    return meshes;
  }
  Vec<int> vertLabel(componentIndices);

  std::vector<Manifold> meshes;
  for (int i = 0; i < numComponents; ++i) {
    auto impl = std::make_shared<Impl>();
    // inherit original object's precision
    impl->precision_ = pImpl_->precision_;
    impl->vertPos_.resize(NumVert());
    Vec<int> vertNew2Old(NumVert());
    auto policy = autoPolicy(NumVert());
    auto start = zip(impl->vertPos_.begin(), vertNew2Old.begin());
    int nVert =
        copy_if<decltype(start)>(
            policy, zip(pImpl_->vertPos_.begin(), countAt(0)),
            zip(pImpl_->vertPos_.end(), countAt(NumVert())), vertLabel.begin(),
            zip(impl->vertPos_.begin(), vertNew2Old.begin()), Equals({i})) -
        start;
    impl->vertPos_.resize(nVert);

    Vec<int> faceNew2Old(NumTri());
    sequence(policy, faceNew2Old.begin(), faceNew2Old.end());

    int nFace = remove_if<decltype(faceNew2Old.begin())>(
                    policy, faceNew2Old.begin(), faceNew2Old.end(),
                    RemoveFace({pImpl_->halfedge_, vertLabel, i})) -
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
