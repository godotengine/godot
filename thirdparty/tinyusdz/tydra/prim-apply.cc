#include "prim-apply.hh"

#include "prim-types.hh"
#include "usdGeom.hh"
#include "usdSkel.hh"
#include "usdLux.hh"

namespace tinyusdz {
namespace tydra {

bool ApplyToGPrim(
  const Stage &stage, const Prim &prim,
  std::function<bool(const Stage &stage, const GPrim *gprim)> fn) {

  (void)stage;

  if ((value::TypeId::TYPE_ID_GPRIM <= prim.type_id() &&
      (value::TypeId::TYPE_ID_GEOM_END > prim.type_id()))) {
    // gprim
  } else {
    return false;
  }

#define APPLY_FUN(__ty) { \
  const auto *v = prim.as<__ty>(); \
  if (v) { \
    return fn(stage, v); \
  } \
  }

  APPLY_FUN(GPrim)
  APPLY_FUN(Xform)
  APPLY_FUN(GeomMesh)
  APPLY_FUN(GeomSphere)
  APPLY_FUN(GeomCapsule)
  APPLY_FUN(GeomCube)
  APPLY_FUN(GeomPoints)
  APPLY_FUN(GeomCylinder)
  APPLY_FUN(GeomBasisCurves)

#undef APPLY_FUN

  return false;

}

bool ApplyToMaterialBinding(
  const Stage &stage, const Prim &prim,
  std::function<bool(const Stage &stage, const MaterialBinding *mb)> fn) {

  if ((value::TypeId::TYPE_ID_GPRIM <= prim.type_id() &&
      (value::TypeId::TYPE_ID_GEOM_END > prim.type_id()))) {
    // gprim or geomsubset
  } else {
    return false;
  }

#define APPLY_FUN(__ty) { \
  const auto *v = prim.as<__ty>(); \
  if (v) { \
    return fn(stage, v); \
  } \
  }

  // TODO: Model/Scope
  APPLY_FUN(Model)
  APPLY_FUN(Scope)
  APPLY_FUN(GPrim)
  APPLY_FUN(Xform)
  APPLY_FUN(GeomMesh)
  APPLY_FUN(GeomSphere)
  APPLY_FUN(GeomCapsule)
  APPLY_FUN(GeomCube)
  APPLY_FUN(GeomPoints)
  APPLY_FUN(GeomCylinder)
  APPLY_FUN(GeomBasisCurves)
  APPLY_FUN(GeomSubset)

#undef APPLY_FUN

  return false;

}

bool ApplyToCollection(
  const Prim &prim,
  std::function<bool(const Collection *col)> fn) {

  if ((value::TypeId::TYPE_ID_GPRIM <= prim.type_id() &&
      (value::TypeId::TYPE_ID_GEOM_END > prim.type_id()))) {
    // gprim or geomsubset
  } else if ((value::TypeId::TYPE_ID_LUX_BEGIN <= prim.type_id() &&
      (value::TypeId::TYPE_ID_LUX_END > prim.type_id()))) {
    // usdLux
  } else {
    return false;
  }

#define APPLY_FUN(__ty) { \
  const auto *v = prim.as<__ty>(); \
  if (v) { \
    return fn(v); \
  } \
  }

  APPLY_FUN(Model)
  APPLY_FUN(Scope)
  APPLY_FUN(GPrim)
  APPLY_FUN(Xform)
  APPLY_FUN(GeomMesh)
  APPLY_FUN(GeomSphere)
  APPLY_FUN(GeomCapsule)
  APPLY_FUN(GeomCube)
  APPLY_FUN(GeomPoints)
  APPLY_FUN(GeomCylinder)
  APPLY_FUN(GeomBasisCurves)
  APPLY_FUN(GeomSubset)
  APPLY_FUN(SphereLight)

#undef APPLY_FUN

  return false;

}

bool ApplyToXformable(
  const Stage &stage, const Prim &prim,
  std::function<bool(const Stage &stage, const Xformable *xformable)> fn) {

  (void)stage;

#define APPLY_FUN(__ty) { \
  const auto *v = prim.as<__ty>(); \
  if (v) { \
    return fn(stage, v); \
  } \
  }

  APPLY_FUN(GPrim)
  APPLY_FUN(Xform)
  APPLY_FUN(GeomMesh)
  APPLY_FUN(GeomSphere)
  APPLY_FUN(GeomCapsule)
  APPLY_FUN(GeomCube)
  APPLY_FUN(GeomPoints)
  APPLY_FUN(GeomCylinder)
  APPLY_FUN(GeomBasisCurves)
  APPLY_FUN(SkelRoot)

#undef APPLY_FUN

  return false;

}

} // namespace tydra
} // namespace tinyusdz
