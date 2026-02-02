// Apply __FUNC to Prim type(class).

#define APPLY_FUNC_TO_PRIM_TYPES(__FUNC) \
  __FUNC(Model) \
  __FUNC(Scope) \
  __FUNC(Xform) \
  __FUNC(GeomMesh) \
  __FUNC(GeomPoints) \
  __FUNC(GeomCylinder) \
  __FUNC(GeomCube) \
  __FUNC(GeomCone) \
  __FUNC(GeomSphere) \
  __FUNC(GeomCapsule) \
  __FUNC(GeomBasisCurves) \
  __FUNC(GeomNurbsCurves) \
  __FUNC(GeomCamera) \
  __FUNC(PointInstancer) \
  __FUNC(GeomSubset) \
  __FUNC(SphereLight) \
  __FUNC(DomeLight) \
  __FUNC(CylinderLight) \
  __FUNC(DiskLight) \
  __FUNC(DistantLight) \
  __FUNC(RectLight) \
  __FUNC(PortalLight) \
  __FUNC(SkelRoot) \
  __FUNC(Skeleton) \
  __FUNC(SkelAnimation) \
  __FUNC(BlendShape) \
  __FUNC(Shader) \
  __FUNC(Material)
