// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "primitive.h"
#include "curveNv.h"
#include "curveNi.h"
#include "curveNi_mb.h"
#include "linei.h"
#include "triangle.h"
#include "trianglev.h"
#include "trianglev_mb.h"
#include "trianglei.h"
#include "quadv.h"
#include "quadi.h"
#include "subdivpatch1.h"
#include "object.h"
#include "instance.h"
#include "instance_array.h"
#include "subgrid.h"

namespace embree
{
  /********************** Curve4v **************************/

  template<>
  const char* Curve4v::Type::name () const {
    return "curve4v";
  }

  template<>
  size_t Curve4v::Type::sizeActive(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return ((Line4i*)This)->size();
    else
      return ((Curve4v*)This)->N;
  }

  template<>
  size_t Curve4v::Type::sizeTotal(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return 4;
    else
      return ((Curve4v*)This)->N;
  }

  template<>
  size_t Curve4v::Type::getBytes(const char* This) const
  {
     if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return Line4i::bytes(sizeActive(This));
     else
        return Curve4v::bytes(sizeActive(This));
  }

  /********************** Curve4i **************************/

  template<>
  const char* Curve4i::Type::name () const {
    return "curve4i";
  }

  template<>
  size_t Curve4i::Type::sizeActive(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return ((Line4i*)This)->size();
    else
      return ((Curve4i*)This)->N;
  }

  template<>
  size_t Curve4i::Type::sizeTotal(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return 4;
    else
      return ((Curve4i*)This)->N;
  }

  template<>
  size_t Curve4i::Type::getBytes(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return Line4i::bytes(sizeActive(This));
    else
      return Curve4i::bytes(sizeActive(This));
  }

  /********************** Curve4iMB **************************/

  template<>
  const char* Curve4iMB::Type::name () const {
    return "curve4imb";
  }

  template<>
  size_t Curve4iMB::Type::sizeActive(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return ((Line4i*)This)->size();
    else
      return ((Curve4iMB*)This)->N;
  }

  template<>
  size_t Curve4iMB::Type::sizeTotal(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return 4;
    else
      return ((Curve4iMB*)This)->N;
  }

  template<>
  size_t Curve4iMB::Type::getBytes(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return Line4i::bytes(sizeActive(This));
    else
      return Curve4iMB::bytes(sizeActive(This));
  }

  /********************** Line4i **************************/

  template<>
  const char* Line4i::Type::name () const {
    return "line4i";
  }

  template<>
  size_t Line4i::Type::sizeActive(const char* This) const {
    return ((Line4i*)This)->size();
  }

  template<>
  size_t Line4i::Type::sizeTotal(const char* This) const {
    return 4;
  }

  template<>
  size_t Line4i::Type::getBytes(const char* This) const {
    return sizeof(Line4i);
  }

  /********************** Triangle4 **************************/

  template<>
  const char* Triangle4::Type::name () const {
    return "triangle4";
  }

  template<>
  size_t Triangle4::Type::sizeActive(const char* This) const {
    return ((Triangle4*)This)->size();
  }

  template<>
  size_t Triangle4::Type::sizeTotal(const char* This) const {
    return 4;
  }

  template<>
  size_t Triangle4::Type::getBytes(const char* This) const {
    return sizeof(Triangle4);
  }

  /********************** Triangle4v **************************/

  template<>
  const char* Triangle4v::Type::name () const {
    return "triangle4v";
  }

  template<>
  size_t Triangle4v::Type::sizeActive(const char* This) const {
    return ((Triangle4v*)This)->size();
  }

  template<>
  size_t Triangle4v::Type::sizeTotal(const char* This) const {
    return 4;
  }

  template<>
  size_t Triangle4v::Type::getBytes(const char* This) const {
    return sizeof(Triangle4v);
  }

  /********************** Triangle4i **************************/

  template<>
  const char* Triangle4i::Type::name () const {
    return "triangle4i";
  }

  template<>
  size_t Triangle4i::Type::sizeActive(const char* This) const {
    return ((Triangle4i*)This)->size();
  }

  template<>
  size_t Triangle4i::Type::sizeTotal(const char* This) const {
    return 4;
  }

  template<>
  size_t Triangle4i::Type::getBytes(const char* This) const {
    return sizeof(Triangle4i);
  }

  /********************** Triangle4vMB **************************/

  template<>
  const char* Triangle4vMB::Type::name () const {
    return  "triangle4vmb";
  }

  template<>
  size_t Triangle4vMB::Type::sizeActive(const char* This) const {
    return ((Triangle4vMB*)This)->size();
  }

  template<>
  size_t Triangle4vMB::Type::sizeTotal(const char* This) const {
    return 4;
  }

  template<>
  size_t Triangle4vMB::Type::getBytes(const char* This) const {
    return sizeof(Triangle4vMB);
  }

  /********************** Quad4v **************************/

  template<>
  const char* Quad4v::Type::name () const {
    return "quad4v";
  }

  template<>
  size_t Quad4v::Type::sizeActive(const char* This) const {
    return ((Quad4v*)This)->size();
  }

  template<>
  size_t Quad4v::Type::sizeTotal(const char* This) const {
    return 4;
  }

  template<>
  size_t Quad4v::Type::getBytes(const char* This) const {
    return sizeof(Quad4v);
  }

  /********************** Quad4i **************************/

  template<>
  const char* Quad4i::Type::name () const {
    return "quad4i";
  }

  template<>
  size_t Quad4i::Type::sizeActive(const char* This) const {
    return ((Quad4i*)This)->size();
  }

  template<>
  size_t Quad4i::Type::sizeTotal(const char* This) const {
    return 4;
  }

  template<>
  size_t Quad4i::Type::getBytes(const char* This) const {
    return sizeof(Quad4i);
  }

  /********************** SubdivPatch1 **************************/

  const char* SubdivPatch1::Type::name () const {
    return "subdivpatch1";
  }

  size_t SubdivPatch1::Type::sizeActive(const char* This) const {
    return 1;
  }

  size_t SubdivPatch1::Type::sizeTotal(const char* This) const {
    return 1;
  }

  size_t SubdivPatch1::Type::getBytes(const char* This) const {
    return sizeof(SubdivPatch1);
  }

  SubdivPatch1::Type SubdivPatch1::type;

  /********************** Virtual Object **************************/

  const char* Object::Type::name () const {
    return "object";
  }

  size_t Object::Type::sizeActive(const char* This) const {
    return 1;
  }

  size_t Object::Type::sizeTotal(const char* This) const {
    return 1;
  }

  size_t Object::Type::getBytes(const char* This) const {
    return sizeof(Object);
  }

  Object::Type Object::type;

  /********************** Instance **************************/

  const char* InstancePrimitive::Type::name () const {
    return "instance";
  }

  size_t InstancePrimitive::Type::sizeActive(const char* This) const {
    return 1;
  }

  size_t InstancePrimitive::Type::sizeTotal(const char* This) const {
    return 1;
  }

  size_t InstancePrimitive::Type::getBytes(const char* This) const {
    return sizeof(InstancePrimitive);
  }

  InstancePrimitive::Type InstancePrimitive::type;

  /********************** InstanceArray4 **************************/

  const char* InstanceArrayPrimitive::Type::name () const {
    return "instance_array";
  }

  size_t InstanceArrayPrimitive::Type::sizeActive(const char* This) const {
    return 1;
  }

  size_t InstanceArrayPrimitive::Type::sizeTotal(const char* This) const {
    return 1;
  }

  size_t InstanceArrayPrimitive::Type::getBytes(const char* This) const {
    return sizeof(InstanceArrayPrimitive);
  }

  InstanceArrayPrimitive::Type InstanceArrayPrimitive::type;

  /********************** SubGrid **************************/

  const char* SubGrid::Type::name () const {
    return "subgrid";
  }

  size_t SubGrid::Type::sizeActive(const char* This) const {
    return 1;
  }

  size_t SubGrid::Type::sizeTotal(const char* This) const {
    return 1;
  }

  size_t SubGrid::Type::getBytes(const char* This) const {
    return sizeof(SubGrid);
  }

  SubGrid::Type SubGrid::type;
  
  /********************** SubGridQBVH4 **************************/

  template<>
  const char* SubGridQBVH4::Type::name () const {
    return "SubGridQBVH4";
  }

  template<>
  size_t SubGridQBVH4::Type::sizeActive(const char* This) const {
    return 1;
  }

  template<>
  size_t SubGridQBVH4::Type::sizeTotal(const char* This) const {
    return 1;
  }

  template<>
  size_t SubGridQBVH4::Type::getBytes(const char* This) const {
    return sizeof(SubGridQBVH4);
  }
}
