#include "impl.h"

namespace {
using namespace manifold;

inline int FlipHalfedge(int halfedge) {
  const int tri = halfedge / 3;
  const int vert = 2 - (halfedge - 3 * tri);
  return 3 * tri + vert;
}

struct TransformNormals {
  const glm::mat3 transform;

  glm::vec3 operator()(glm::vec3 normal) {
    normal = glm::normalize(transform * normal);
    if (isnan(normal.x)) normal = glm::vec3(0.0f);
    return normal;
  }
};

struct TransformTangents {
  const glm::mat3 transform;
  const bool invert;
  VecView<const glm::vec4> oldTangents;
  VecView<const Halfedge> halfedge;

  void operator()(thrust::tuple<glm::vec4&, int> inOut) {
    glm::vec4& tangent = thrust::get<0>(inOut);
    int edge = thrust::get<1>(inOut);
    if (invert) {
      edge = halfedge[FlipHalfedge(edge)].pairedHalfedge;
    }

    tangent = glm::vec4(transform * glm::vec3(oldTangents[edge]),
                        oldTangents[edge].w);
  }
};

struct FlipTris {
  VecView<Halfedge> halfedge;

  void operator()(thrust::tuple<TriRef&, int> inOut) {
    TriRef& bary = thrust::get<0>(inOut);
    const int tri = thrust::get<1>(inOut);

    thrust::swap(halfedge[3 * tri], halfedge[3 * tri + 2]);

    for (const int i : {0, 1, 2}) {
      thrust::swap(halfedge[3 * tri + i].startVert,
                   halfedge[3 * tri + i].endVert);
      halfedge[3 * tri + i].pairedHalfedge =
          FlipHalfedge(halfedge[3 * tri + i].pairedHalfedge);
    }
  }
};
}  // namespace
