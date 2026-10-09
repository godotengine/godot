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
  /********************** Curve8v **************************/

  template<>
  const char* Curve8v::Type::name () const {
    return "curve8v";
  }

  template<>
  size_t Curve8v::Type::sizeActive(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return ((Line8i*)This)->size();
    else
      return ((Curve8v*)This)->N;
  }

  template<>
  size_t Curve8v::Type::sizeTotal(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return 8;
    else
      return ((Curve8v*)This)->N;
  }

  template<>
  size_t Curve8v::Type::getBytes(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
       return Line8i::bytes(sizeActive(This));
     else
       return Curve8v::bytes(sizeActive(This));
  }

  /********************** Curve8i **************************/

  template<>
  const char* Curve8i::Type::name () const {
    return "curve8i";
  }

  template<>
  size_t Curve8i::Type::sizeActive(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return ((Line8i*)This)->size();
    else
      return ((Curve8i*)This)->N;
  }

  template<>
  size_t Curve8i::Type::sizeTotal(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return 8;
    else
      return ((Curve8i*)This)->N;
  }

  template<>
  size_t Curve8i::Type::getBytes(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
       return Line8i::bytes(sizeActive(This));
     else
       return Curve8i::bytes(sizeActive(This));
  }

  /********************** Curve8iMB **************************/

  template<>
  const char* Curve8iMB::Type::name () const {
    return "curve8imb";
  }

  template<>
  size_t Curve8iMB::Type::sizeActive(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return ((Line8i*)This)->size();
    else
      return ((Curve8iMB*)This)->N;
  }

  template<>
  size_t Curve8iMB::Type::sizeTotal(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
      return 8;
    else
      return ((Curve8iMB*)This)->N;
  }

  template<>
  size_t Curve8iMB::Type::getBytes(const char* This) const
  {
    if ((*This & Geometry::GType::GTY_BASIS_MASK) == Geometry::GType::GTY_BASIS_LINEAR)
       return Line8i::bytes(sizeActive(This));
     else
       return Curve8iMB::bytes(sizeActive(This));
  }

  /********************** SubGridQBVH8 **************************/

  template<>
  const char* SubGridQBVH8::Type::name () const {
    return "SubGridQBVH8";
  }

  template<>
  size_t SubGridQBVH8::Type::sizeActive(const char* This) const {
    return 1;
  }

  template<>
  size_t SubGridQBVH8::Type::sizeTotal(const char* This) const {
    return 1;
  }

  template<>
  size_t SubGridQBVH8::Type::getBytes(const char* This) const {
    return sizeof(SubGridQBVH8);
  }

  /********************** Instance Array **************************/
#if 0
  template<>
  const char* InstanceArrayPrimitive<8>::Type::name () const {
    return "instance_array_8";
  }

  template<>
  size_t InstanceArrayPrimitive<8>::Type::sizeActive(const char* This) const {
    return ((InstanceArrayPrimitive<8>*)This)->size();
  }

  template<>
  size_t InstanceArrayPrimitive<8>::Type::sizeTotal(const char* This) const {
    return 8;
  }

  template<>
  size_t InstanceArrayPrimitive<8>::Type::getBytes(const char* This) const {
    return sizeof(InstanceArrayPrimitive<8>);
  }
#endif

}
