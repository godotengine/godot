// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh_node_base.h"

namespace embree
{
  /*! BVHN Quantized Node */
  template<int N>
    struct __aligned(8) QuantizedBaseNode_t
  {
    typedef unsigned char T;
    static const T MIN_QUAN = 0;
    static const T MAX_QUAN = 255;
    
    /*! Clears the node. */
    __forceinline void clear() {
      for (size_t i=0; i<N; i++) lower_x[i] = lower_y[i] = lower_z[i] = MAX_QUAN;
      for (size_t i=0; i<N; i++) upper_x[i] = upper_y[i] = upper_z[i] = MIN_QUAN;
    }
    
    /*! Returns bounds of specified child. */
    __forceinline BBox3fa bounds(size_t i) const
    {
      assert(i < N);
      const Vec3fa lower(madd(scale.x,(float)lower_x[i],start.x),
                         madd(scale.y,(float)lower_y[i],start.y),
                         madd(scale.z,(float)lower_z[i],start.z));
      const Vec3fa upper(madd(scale.x,(float)upper_x[i],start.x),
                         madd(scale.y,(float)upper_y[i],start.y),
                         madd(scale.z,(float)upper_z[i],start.z));
      return BBox3fa(lower,upper);
    }
    
    /*! Returns extent of bounds of specified child. */
    __forceinline Vec3fa extent(size_t i) const {
      return bounds(i).size();
    }
    
    static __forceinline void init_dim(const vfloat<N> &lower,
                                       const vfloat<N> &upper,
                                       T lower_quant[N],
                                       T upper_quant[N],
                                       float &start,
                                       float &scale)
    {
      /* quantize bounds */
      const vbool<N> m_valid = lower != vfloat<N>(pos_inf);
      const float minF = reduce_min(lower);
      const float maxF = reduce_max(upper);
      float diff = (1.0f+2.0f*float(ulp))*(maxF - minF);
      float decode_scale = diff / float(MAX_QUAN);
      if (decode_scale == 0.0f) decode_scale = 2.0f*FLT_MIN; // result may have been flushed to zero
      assert(madd(decode_scale,float(MAX_QUAN),minF) >= maxF);
      const float encode_scale = diff > 0 ? (float(MAX_QUAN) / diff) : 0.0f;
      vint<N> ilower = max(vint<N>(floor((lower - vfloat<N>(minF))*vfloat<N>(encode_scale))),MIN_QUAN);
      vint<N> iupper = min(vint<N>(ceil ((upper - vfloat<N>(minF))*vfloat<N>(encode_scale))),MAX_QUAN);
      
      /* lower/upper correction */
      vbool<N> m_lower_correction = (madd(vfloat<N>(ilower),decode_scale,minF)) > lower;
      vbool<N> m_upper_correction = (madd(vfloat<N>(iupper),decode_scale,minF)) < upper;
      ilower = max(select(m_lower_correction,ilower-1,ilower),MIN_QUAN);
      iupper = min(select(m_upper_correction,iupper+1,iupper),MAX_QUAN);
      
      /* disable invalid lanes */
      ilower = select(m_valid,ilower,MAX_QUAN);
      iupper = select(m_valid,iupper,MIN_QUAN);
      
      /* store as uchar to memory */
      vint<N>::store(lower_quant,ilower);
      vint<N>::store(upper_quant,iupper);
      start = minF;
      scale = decode_scale;
      
#if defined(DEBUG)
      vfloat<N> extract_lower( vint<N>::loadu(lower_quant) );
      vfloat<N> extract_upper( vint<N>::loadu(upper_quant) );
      vfloat<N> final_extract_lower = madd(extract_lower,decode_scale,minF);
      vfloat<N> final_extract_upper = madd(extract_upper,decode_scale,minF);
      assert( (movemask(final_extract_lower <= lower ) & movemask(m_valid)) == movemask(m_valid));
      assert( (movemask(final_extract_upper >= upper ) & movemask(m_valid)) == movemask(m_valid));
#endif
    }
    
    __forceinline void init_dim(AABBNode_t<NodeRefPtr<N>,N>& node)
    {
      init_dim(node.lower_x,node.upper_x,lower_x,upper_x,start.x,scale.x);
      init_dim(node.lower_y,node.upper_y,lower_y,upper_y,start.y,scale.y);
      init_dim(node.lower_z,node.upper_z,lower_z,upper_z,start.z,scale.z);
    }
    
    __forceinline vbool<N> validMask() const { return vint<N>::loadu(lower_x) <= vint<N>::loadu(upper_x); }
    
#if defined(__AVX512F__) // KNL
    __forceinline vbool16 validMask16() const { return le(0xff,vint<16>::loadu(lower_x),vint<16>::loadu(upper_x)); }
#endif
    __forceinline vfloat<N> dequantizeLowerX() const { return madd(vfloat<N>(vint<N>::loadu(lower_x)),scale.x,vfloat<N>(start.x)); }
    
    __forceinline vfloat<N> dequantizeUpperX() const { return madd(vfloat<N>(vint<N>::loadu(upper_x)),scale.x,vfloat<N>(start.x)); }
    
    __forceinline vfloat<N> dequantizeLowerY() const { return madd(vfloat<N>(vint<N>::loadu(lower_y)),scale.y,vfloat<N>(start.y)); }
    
    __forceinline vfloat<N> dequantizeUpperY() const { return madd(vfloat<N>(vint<N>::loadu(upper_y)),scale.y,vfloat<N>(start.y)); }
    
    __forceinline vfloat<N> dequantizeLowerZ() const { return madd(vfloat<N>(vint<N>::loadu(lower_z)),scale.z,vfloat<N>(start.z)); }
    
    __forceinline vfloat<N> dequantizeUpperZ() const { return madd(vfloat<N>(vint<N>::loadu(upper_z)),scale.z,vfloat<N>(start.z)); }
    
    template <int M>
      __forceinline vfloat<M> dequantize(const size_t offset) const { return vfloat<M>(vint<M>::loadu(all_planes+offset)); }
    
#if defined(__AVX512F__)
    __forceinline vfloat16 dequantizeLowerUpperX(const vint16 &p) const { return madd(vfloat16(permute(vint<16>::loadu(lower_x),p)),scale.x,vfloat16(start.x)); }
    __forceinline vfloat16 dequantizeLowerUpperY(const vint16 &p) const { return madd(vfloat16(permute(vint<16>::loadu(lower_y),p)),scale.y,vfloat16(start.y)); }
    __forceinline vfloat16 dequantizeLowerUpperZ(const vint16 &p) const { return madd(vfloat16(permute(vint<16>::loadu(lower_z),p)),scale.z,vfloat16(start.z)); }      
#endif
    
    union {
      struct {
        T lower_x[N]; //!< 8bit discretized X dimension of lower bounds of all N children
        T upper_x[N]; //!< 8bit discretized X dimension of upper bounds of all N children
        T lower_y[N]; //!< 8bit discretized Y dimension of lower bounds of all N children
        T upper_y[N]; //!< 8bit discretized Y dimension of upper bounds of all N children
        T lower_z[N]; //!< 8bit discretized Z dimension of lower bounds of all N children
        T upper_z[N]; //!< 8bit discretized Z dimension of upper bounds of all N children
      };
      T all_planes[6*N];
    };
    
    Vec3f start;
    Vec3f scale;
    
    friend embree_ostream operator<<(embree_ostream o, const QuantizedBaseNode_t& n)
    {
      o << "QuantizedBaseNode { " << embree_endl;
      o << "  start   " << n.start << embree_endl;
      o << "  scale   " << n.scale << embree_endl;
      o << "  lower_x " << vuint<N>::loadu(n.lower_x) << embree_endl;
      o << "  upper_x " << vuint<N>::loadu(n.upper_x) << embree_endl;
      o << "  lower_y " << vuint<N>::loadu(n.lower_y) << embree_endl;
      o << "  upper_y " << vuint<N>::loadu(n.upper_y) << embree_endl;
      o << "  lower_z " << vuint<N>::loadu(n.lower_z) << embree_endl;
      o << "  upper_z " << vuint<N>::loadu(n.upper_z) << embree_endl;
      o << "}" << embree_endl;
      return o;
    }
    
  };

  template<typename NodeRef, int N>
    struct __aligned(8) QuantizedNode_t : public BaseNode_t<NodeRef, N>, QuantizedBaseNode_t<N>
  {
    using BaseNode_t<NodeRef,N>::children;
    using QuantizedBaseNode_t<N>::lower_x;
    using QuantizedBaseNode_t<N>::upper_x;
    using QuantizedBaseNode_t<N>::lower_y;
    using QuantizedBaseNode_t<N>::upper_y;
    using QuantizedBaseNode_t<N>::lower_z;
    using QuantizedBaseNode_t<N>::upper_z;
    using QuantizedBaseNode_t<N>::start;
    using QuantizedBaseNode_t<N>::scale;
    using QuantizedBaseNode_t<N>::init_dim;
    
    __forceinline void setRef(size_t i, const NodeRef& ref) {
      assert(i < N);
      children[i] = ref;
    }
    
    struct Create2
    {
      template<typename BuildRecord>
      __forceinline NodeRef operator() (BuildRecord* children, const size_t n, const FastAllocator::CachedAllocator& alloc) const
      {
        __aligned(64) AABBNode_t<NodeRef,N> node;
        node.clear();
        for (size_t i=0; i<n; i++) {
          node.setBounds(i,children[i].bounds());
        }
        QuantizedNode_t *qnode = (QuantizedNode_t*) alloc.malloc0(sizeof(QuantizedNode_t), NodeRef::byteAlignment);
        qnode->init(node);
        
        return (size_t)qnode | NodeRef::tyQuantizedNode;
      }
    };
    
    struct Set2
    {
      template<typename BuildRecord>
      __forceinline NodeRef operator() (const BuildRecord& precord, const BuildRecord* crecords, NodeRef ref, NodeRef* children, const size_t num) const
      {
        QuantizedNode_t* node = ref.quantizedNode();
        for (size_t i=0; i<num; i++) node->setRef(i,children[i]);
        return ref;
      }
    };
    
    __forceinline void init(AABBNode_t<NodeRef,N>& node)
    {
      for (size_t i=0;i<N;i++) children[i] = NodeRef::emptyNode;
      init_dim(node);
    }
    
  }; 
  
  /*! BVHN Quantized Node */
  template<int N>
    struct __aligned(8) QuantizedBaseNodeMB_t
  {
    QuantizedBaseNode_t<N> node0;
    QuantizedBaseNode_t<N> node1;
    
    /*! Clears the node. */
    __forceinline void clear() {
      node0.clear();
      node1.clear();
    }
    
    /*! Returns bounds of specified child. */
    __forceinline BBox3fa bounds(size_t i) const
    {
      assert(i < N);
      BBox3fa bounds0 = node0.bounds(i);
      BBox3fa bounds1 = node1.bounds(i);
      bounds0.extend(bounds1);
      return bounds0;
    }
    
    /*! Returns extent of bounds of specified child. */
    __forceinline Vec3fa extent(size_t i) const {
      return bounds(i).size();
    }
    
    __forceinline vbool<N> validMask() const { return node0.validMask(); }
    
    template<typename T>
      __forceinline vfloat<N> dequantizeLowerX(const T t) const { return lerp(node0.dequantizeLowerX(),node1.dequantizeLowerX(),t); }
    template<typename T>
      __forceinline vfloat<N> dequantizeUpperX(const T t) const { return lerp(node0.dequantizeUpperX(),node1.dequantizeUpperX(),t); }
    template<typename T>
      __forceinline vfloat<N> dequantizeLowerY(const T t) const { return lerp(node0.dequantizeLowerY(),node1.dequantizeLowerY(),t); }
    template<typename T>
      __forceinline vfloat<N> dequantizeUpperY(const T t) const { return lerp(node0.dequantizeUpperY(),node1.dequantizeUpperY(),t); }
    template<typename T>
      __forceinline vfloat<N> dequantizeLowerZ(const T t) const { return lerp(node0.dequantizeLowerZ(),node1.dequantizeLowerZ(),t); }
    template<typename T>
      __forceinline vfloat<N> dequantizeUpperZ(const T t) const { return lerp(node0.dequantizeUpperZ(),node1.dequantizeUpperZ(),t); }
    
    
    template<int M>
      __forceinline vfloat<M> dequantizeLowerX(const size_t i, const vfloat<M> &t) const { return lerp(vfloat<M>(node0.dequantizeLowerX()[i]),vfloat<M>(node1.dequantizeLowerX()[i]),t); }
    template<int M>
      __forceinline vfloat<M> dequantizeUpperX(const size_t i, const vfloat<M> &t) const { return lerp(vfloat<M>(node0.dequantizeUpperX()[i]),vfloat<M>(node1.dequantizeUpperX()[i]),t); }
    template<int M>
      __forceinline vfloat<M> dequantizeLowerY(const size_t i, const vfloat<M> &t) const { return lerp(vfloat<M>(node0.dequantizeLowerY()[i]),vfloat<M>(node1.dequantizeLowerY()[i]),t); }
    template<int M>
      __forceinline vfloat<M> dequantizeUpperY(const size_t i, const vfloat<M> &t) const { return lerp(vfloat<M>(node0.dequantizeUpperY()[i]),vfloat<M>(node1.dequantizeUpperY()[i]),t); }
    template<int M>
      __forceinline vfloat<M> dequantizeLowerZ(const size_t i, const vfloat<M> &t) const { return lerp(vfloat<M>(node0.dequantizeLowerZ()[i]),vfloat<M>(node1.dequantizeLowerZ()[i]),t); }
    template<int M>
      __forceinline vfloat<M> dequantizeUpperZ(const size_t i, const vfloat<M> &t) const { return lerp(vfloat<M>(node0.dequantizeUpperZ()[i]),vfloat<M>(node1.dequantizeUpperZ()[i]),t); }
    
  };
}
