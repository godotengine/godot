// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

namespace embree
{
  /* 4-wide SSE float type */
  template<>
  struct vfloat<4>
  {
    typedef vboolf4 Bool;
    typedef vint4   Int;
    typedef vfloat4 Float;
    
    enum  { size = 4 };                        // number of SIMD elements
    union { __m128 v; float f[4]; int i[4]; }; // data

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////
    
    __forceinline vfloat() {}
    __forceinline vfloat(const vfloat4& other) { v = other.v; }
    __forceinline vfloat4& operator =(const vfloat4& other) { v = other.v; return *this; }

    __forceinline vfloat(__m128 a) : v(a) {}
    __forceinline operator const __m128&() const { return v; }
    __forceinline operator       __m128&()       { return v; }

    __forceinline vfloat(float a) : v(_mm_set1_ps(a)) {}
    __forceinline vfloat(float a, float b, float c, float d) : v(_mm_set_ps(d, c, b, a)) {}

    __forceinline explicit vfloat(const vint4& a) : v(_mm_cvtepi32_ps(a)) {}
    __forceinline explicit vfloat(const vuint4& x) {
      const __m128i a   = _mm_and_si128(x,_mm_set1_epi32(0x7FFFFFFF));
      const __m128i b   = _mm_and_si128(_mm_srai_epi32(x,31),_mm_set1_epi32(0x4F000000)); //0x4F000000 = 2^31 
      const __m128  af  = _mm_cvtepi32_ps(a);
      const __m128  bf  = _mm_castsi128_ps(b);  
      v  = _mm_add_ps(af,bf);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vfloat(ZeroTy)   : v(_mm_setzero_ps()) {}
    __forceinline vfloat(OneTy)    : v(_mm_set1_ps(1.0f)) {}
    __forceinline vfloat(PosInfTy) : v(_mm_set1_ps(pos_inf)) {}
    __forceinline vfloat(NegInfTy) : v(_mm_set1_ps(neg_inf)) {}
    __forceinline vfloat(StepTy)   : v(_mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f)) {}
    __forceinline vfloat(NaNTy)    : v(_mm_set1_ps(nan)) {}
    __forceinline vfloat(UndefinedTy) : v(_mm_undefined_ps()) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline vfloat4 load (const void* a) { return _mm_load_ps((float*)a); }
    static __forceinline vfloat4 loadu(const void* a) { return _mm_loadu_ps((float*)a); }

    static __forceinline void store (void* ptr, const vfloat4& v) { _mm_store_ps((float*)ptr,v); }
    static __forceinline void storeu(void* ptr, const vfloat4& v) { _mm_storeu_ps((float*)ptr,v); }

#if defined(__AVX512VL__)

    static __forceinline vfloat4 compact(const vboolf4& mask, vfloat4 &v) {
      return _mm_mask_compress_ps(v, mask, v);
    }
    static __forceinline vfloat4 compact(const vboolf4& mask, vfloat4 &a, const vfloat4& b) {
      return _mm_mask_compress_ps(a, mask, b);
    }

    static __forceinline vfloat4 load (const vboolf4& mask, const void* ptr) { return _mm_mask_load_ps (_mm_setzero_ps(),mask,(float*)ptr); }
    static __forceinline vfloat4 loadu(const vboolf4& mask, const void* ptr) { return _mm_mask_loadu_ps(_mm_setzero_ps(),mask,(float*)ptr); }

    static __forceinline void store (const vboolf4& mask, void* ptr, const vfloat4& v) { _mm_mask_store_ps ((float*)ptr,mask,v); }
    static __forceinline void storeu(const vboolf4& mask, void* ptr, const vfloat4& v) { _mm_mask_storeu_ps((float*)ptr,mask,v); }
#elif defined(__AVX__)
    static __forceinline vfloat4 load (const vboolf4& mask, const void* ptr) { return _mm_maskload_ps((float*)ptr,mask); }
    static __forceinline vfloat4 loadu(const vboolf4& mask, const void* ptr) { return _mm_maskload_ps((float*)ptr,mask); }

    static __forceinline void store (const vboolf4& mask, void* ptr, const vfloat4& v) { _mm_maskstore_ps((float*)ptr,(__m128i)mask,v); }
    static __forceinline void storeu(const vboolf4& mask, void* ptr, const vfloat4& v) { _mm_maskstore_ps((float*)ptr,(__m128i)mask,v); }
#else
    static __forceinline vfloat4 load (const vboolf4& mask, const void* ptr) { return _mm_and_ps(_mm_load_ps ((float*)ptr),mask); }
    static __forceinline vfloat4 loadu(const vboolf4& mask, const void* ptr) { return _mm_and_ps(_mm_loadu_ps((float*)ptr),mask); }

    static __forceinline void store (const vboolf4& mask, void* ptr, const vfloat4& v) { store (ptr,select(mask,v,load (ptr))); }
    static __forceinline void storeu(const vboolf4& mask, void* ptr, const vfloat4& v) { storeu(ptr,select(mask,v,loadu(ptr))); }
#endif

#if defined(__AVX__)
    static __forceinline vfloat4 broadcast(const void* a) { return _mm_broadcast_ss((float*)a); }
#else
    static __forceinline vfloat4 broadcast(const void* a) { return _mm_set1_ps(*(float*)a); }
#endif

    static __forceinline vfloat4 load_nt (const float* ptr) {
#if defined (__SSE4_1__)
    return _mm_castsi128_ps(_mm_stream_load_si128((__m128i*)ptr));
#else
    return _mm_load_ps(ptr); 
#endif
  }

#if defined(__SSE4_1__)
    static __forceinline vfloat4 load(const char* ptr) {
      return _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_loadu_si128((__m128i*)ptr)));
    }
#else
    static __forceinline vfloat4 load(const char* ptr) {
      return vfloat4(ptr[0],ptr[1],ptr[2],ptr[3]);
    }
#endif

#if defined(__SSE4_1__)
    static __forceinline vfloat4 load(const unsigned char* ptr) {
      return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128((__m128i*)ptr)));
    }
#else
    static __forceinline vfloat4 load(const unsigned char* ptr) {
      //return _mm_cvtpu8_ps(*(__m64*)ptr); // don't enable, will use MMX instructions
      return vfloat4(ptr[0],ptr[1],ptr[2],ptr[3]);
    }
#endif

#if defined(__SSE4_1__)
    static __forceinline vfloat4 load(const short* ptr) {
      return _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si128((__m128i*)ptr)));
    }
#else
    static __forceinline vfloat4 load(const short* ptr) {
      return vfloat4(ptr[0],ptr[1],ptr[2],ptr[3]);
    }
#endif

    static __forceinline vfloat4 load(const unsigned short* ptr) {
      return _mm_mul_ps(vfloat4(vint4::load(ptr)),vfloat4(1.0f/65535.0f));
    }
    
    static __forceinline void store_nt(void* ptr, const vfloat4& v)
    {
#if defined (__SSE4_1__)
      _mm_stream_ps((float*)ptr,v);
#else
      _mm_store_ps((float*)ptr,v);
#endif
    }

    template<int scale = 4>
    static __forceinline vfloat4 gather(const float* ptr, const vint4& index) {
#if defined(__AVX2__)
      return _mm_i32gather_ps(ptr, index, scale);
#else
      return vfloat4(
        *(float*)(((char*)ptr)+scale*index[0]),
        *(float*)(((char*)ptr)+scale*index[1]),
        *(float*)(((char*)ptr)+scale*index[2]),
        *(float*)(((char*)ptr)+scale*index[3]));
#endif
    }

    template<int scale = 4>
    static __forceinline vfloat4 gather(const vboolf4& mask, const float* ptr, const vint4& index) {
      vfloat4 r = zero;
#if defined(__AVX512VL__)
      return _mm_mmask_i32gather_ps(r, mask, index, ptr, scale);
#elif defined(__AVX2__)
      return _mm_mask_i32gather_ps(r, ptr, index, mask, scale);
#else
      if (likely(mask[0])) r[0] = *(float*)(((char*)ptr)+scale*index[0]);
      if (likely(mask[1])) r[1] = *(float*)(((char*)ptr)+scale*index[1]);
      if (likely(mask[2])) r[2] = *(float*)(((char*)ptr)+scale*index[2]);
      if (likely(mask[3])) r[3] = *(float*)(((char*)ptr)+scale*index[3]);
      return r;
#endif
    }

    template<int scale = 4>
    static __forceinline void scatter(void* ptr, const vint4& index, const vfloat4& v)
    {
#if defined(__AVX512VL__)
      _mm_i32scatter_ps((float*)ptr, index, v, scale);
#else
      *(float*)(((char*)ptr)+scale*index[0]) = v[0];
      *(float*)(((char*)ptr)+scale*index[1]) = v[1];
      *(float*)(((char*)ptr)+scale*index[2]) = v[2];
      *(float*)(((char*)ptr)+scale*index[3]) = v[3];
#endif
    }

    template<int scale = 4>
    static __forceinline void scatter(const vboolf4& mask, void* ptr, const vint4& index, const vfloat4& v)
    {
#if defined(__AVX512VL__)
      _mm_mask_i32scatter_ps((float*)ptr ,mask, index, v, scale);
#else
      if (likely(mask[0])) *(float*)(((char*)ptr)+scale*index[0]) = v[0];
      if (likely(mask[1])) *(float*)(((char*)ptr)+scale*index[1]) = v[1];
      if (likely(mask[2])) *(float*)(((char*)ptr)+scale*index[2]) = v[2];
      if (likely(mask[3])) *(float*)(((char*)ptr)+scale*index[3]) = v[3];
#endif
    }

    static __forceinline void store(const vboolf4& mask, char* ptr, const vint4& ofs, const vfloat4& v) {
      scatter<1>(mask,ptr,ofs,v);
    }
    static __forceinline void store(const vboolf4& mask, float* ptr, const vint4& ofs, const vfloat4& v) {
      scatter<4>(mask,ptr,ofs,v);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline const float& operator [](size_t index) const { assert(index < 4); return f[index]; }
    __forceinline       float& operator [](size_t index)       { assert(index < 4); return f[index]; }

    friend __forceinline vfloat4 select(const vboolf4& m, const vfloat4& t, const vfloat4& f) {
#if defined(__AVX512VL__)
      return _mm_mask_blend_ps(m, f, t);
#elif defined(__SSE4_1__)
      return _mm_blendv_ps(f, t, m); 
#else
      return _mm_or_ps(_mm_and_ps(m, t), _mm_andnot_ps(m, f)); 
#endif
    }
  };


  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat4 asFloat(const vint4&   a) { return _mm_castsi128_ps(a); }
  __forceinline vint4   asInt  (const vfloat4& a) { return _mm_castps_si128(a); }
  __forceinline vuint4  asUInt (const vfloat4& a) { return _mm_castps_si128(a); }

  __forceinline vfloat4 operator +(const vfloat4& a) { return a; }
  __forceinline vfloat4 operator -(const vfloat4& a) { return _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x80000000))); }

  __forceinline vfloat4 abs(const vfloat4& a) { return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))); }
#if defined(__AVX512VL__)
  __forceinline vfloat4 sign(const vfloat4& a) { return _mm_mask_blend_ps(_mm_cmp_ps_mask(a, vfloat4(zero), _CMP_LT_OQ), vfloat4(one), -vfloat4(one)); }
#else
  __forceinline vfloat4 sign(const vfloat4& a) { return blendv_ps(vfloat4(one), -vfloat4(one), _mm_cmplt_ps(a, vfloat4(zero))); }
#endif
  __forceinline vfloat4 signmsk(const vfloat4& a) { return _mm_and_ps(a,_mm_castsi128_ps(_mm_set1_epi32(0x80000000))); }
  
  __forceinline vfloat4 rcp(const vfloat4& a)
  {
#if defined(__AVX512VL__)
    const vfloat4 r = _mm_rcp14_ps(a);
#else
    const vfloat4 r = _mm_rcp_ps(a);
#endif

#if defined(__AVX2__)
    return _mm_mul_ps(r,_mm_fnmadd_ps(r, a, vfloat4(2.0f)));
#else
    return _mm_mul_ps(r,_mm_sub_ps(vfloat4(2.0f), _mm_mul_ps(r, a)));
#endif
  }
  __forceinline vfloat4 sqr (const vfloat4& a) { return _mm_mul_ps(a,a); }
  __forceinline vfloat4 sqrt(const vfloat4& a) { return _mm_sqrt_ps(a); }

  __forceinline vfloat4 rsqrt(const vfloat4& a)
  {
#if defined(__AVX512VL__)
    const vfloat4 r = _mm_rsqrt14_ps(a);
#else
    const vfloat4 r = _mm_rsqrt_ps(a);
#endif

#if defined(__AVX2__)
    return _mm_fmadd_ps(_mm_set1_ps(1.5f), r,
                        _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a, _mm_set1_ps(-0.5f)), r), _mm_mul_ps(r, r)));
#else
    return _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.5f), r),
                      _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a, _mm_set1_ps(-0.5f)), r), _mm_mul_ps(r, r)));
#endif
  }

  __forceinline vboolf4 isnan(const vfloat4& a) {
    const vfloat4 b = _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)));
#if defined(__AVX512VL__)
    return _mm_cmp_epi32_mask(_mm_castps_si128(b), _mm_set1_epi32(0x7f800000), _MM_CMPINT_GT);
#else
    return _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_castps_si128(b), _mm_set1_epi32(0x7f800000)));
#endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat4 operator +(const vfloat4& a, const vfloat4& b) { return _mm_add_ps(a, b); }
  __forceinline vfloat4 operator +(const vfloat4& a, float          b) { return a + vfloat4(b); }
  __forceinline vfloat4 operator +(float          a, const vfloat4& b) { return vfloat4(a) + b; }

  __forceinline vfloat4 operator -(const vfloat4& a, const vfloat4& b) { return _mm_sub_ps(a, b); }
  __forceinline vfloat4 operator -(const vfloat4& a, float          b) { return a - vfloat4(b); }
  __forceinline vfloat4 operator -(float          a, const vfloat4& b) { return vfloat4(a) - b; }

  __forceinline vfloat4 operator *(const vfloat4& a, const vfloat4& b) { return _mm_mul_ps(a, b); }
  __forceinline vfloat4 operator *(const vfloat4& a, float          b) { return a * vfloat4(b); }
  __forceinline vfloat4 operator *(float          a, const vfloat4& b) { return vfloat4(a) * b; }

  __forceinline vfloat4 operator /(const vfloat4& a, const vfloat4& b) { return _mm_div_ps(a,b); }
  __forceinline vfloat4 operator /(const vfloat4& a, float          b) { return a/vfloat4(b); }
  __forceinline vfloat4 operator /(float          a, const vfloat4& b) { return vfloat4(a)/b; }

  __forceinline vfloat4 operator ^(const vfloat4& a, const vfloat4& b) { return _mm_xor_ps(a,b); }
  __forceinline vfloat4 operator ^(const vfloat4& a, const vint4&   b) { return _mm_xor_ps(a,_mm_castsi128_ps(b)); }

  __forceinline vfloat4 min(const vfloat4& a, const vfloat4& b) { return _mm_min_ps(a,b); }
  __forceinline vfloat4 min(const vfloat4& a, float          b) { return _mm_min_ps(a,vfloat4(b)); }
  __forceinline vfloat4 min(float          a, const vfloat4& b) { return _mm_min_ps(vfloat4(a),b); }

  __forceinline vfloat4 max(const vfloat4& a, const vfloat4& b) { return _mm_max_ps(a,b); }
  __forceinline vfloat4 max(const vfloat4& a, float          b) { return _mm_max_ps(a,vfloat4(b)); }
  __forceinline vfloat4 max(float          a, const vfloat4& b) { return _mm_max_ps(vfloat4(a),b); }

#if defined(__SSE4_1__)
    __forceinline vfloat4 mini(const vfloat4& a, const vfloat4& b) {
      const vint4 ai = _mm_castps_si128(a);
      const vint4 bi = _mm_castps_si128(b);
      const vint4 ci = _mm_min_epi32(ai,bi);
      return _mm_castsi128_ps(ci);
    }

    __forceinline vfloat4 maxi(const vfloat4& a, const vfloat4& b) {
      const vint4 ai = _mm_castps_si128(a);
      const vint4 bi = _mm_castps_si128(b);
      const vint4 ci = _mm_max_epi32(ai,bi);
      return _mm_castsi128_ps(ci);
    }

    __forceinline vfloat4 minui(const vfloat4& a, const vfloat4& b) {
      const vint4 ai = _mm_castps_si128(a);
      const vint4 bi = _mm_castps_si128(b);
      const vint4 ci = _mm_min_epu32(ai,bi);
      return _mm_castsi128_ps(ci);
    }

    __forceinline vfloat4 maxui(const vfloat4& a, const vfloat4& b) {
      const vint4 ai = _mm_castps_si128(a);
      const vint4 bi = _mm_castps_si128(b);
      const vint4 ci = _mm_max_epu32(ai,bi);
      return _mm_castsi128_ps(ci);
    }
#else
    __forceinline vfloat4 mini(const vfloat4& a, const vfloat4& b) {
      return min(a,b);
    }

    __forceinline vfloat4 maxi(const vfloat4& a, const vfloat4& b) {
      return max(a,b);
    }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX2__)
  __forceinline vfloat4 madd (const vfloat4& a, const vfloat4& b, const vfloat4& c) { return _mm_fmadd_ps(a,b,c); }
  __forceinline vfloat4 msub (const vfloat4& a, const vfloat4& b, const vfloat4& c) { return _mm_fmsub_ps(a,b,c); }
  __forceinline vfloat4 nmadd(const vfloat4& a, const vfloat4& b, const vfloat4& c) { return _mm_fnmadd_ps(a,b,c); }
  __forceinline vfloat4 nmsub(const vfloat4& a, const vfloat4& b, const vfloat4& c) { return _mm_fnmsub_ps(a,b,c); }
#else
  __forceinline vfloat4 madd (const vfloat4& a, const vfloat4& b, const vfloat4& c) { return a*b+c; }
  __forceinline vfloat4 msub (const vfloat4& a, const vfloat4& b, const vfloat4& c) { return a*b-c; }
  __forceinline vfloat4 nmadd(const vfloat4& a, const vfloat4& b, const vfloat4& c) { return -a*b+c;}
  __forceinline vfloat4 nmsub(const vfloat4& a, const vfloat4& b, const vfloat4& c) { return -a*b-c; }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat4& operator +=(vfloat4& a, const vfloat4& b) { return a = a + b; }
  __forceinline vfloat4& operator +=(vfloat4& a, float          b) { return a = a + b; }

  __forceinline vfloat4& operator -=(vfloat4& a, const vfloat4& b) { return a = a - b; }
  __forceinline vfloat4& operator -=(vfloat4& a, float          b) { return a = a - b; }

  __forceinline vfloat4& operator *=(vfloat4& a, const vfloat4& b) { return a = a * b; }
  __forceinline vfloat4& operator *=(vfloat4& a, float          b) { return a = a * b; }

  __forceinline vfloat4& operator /=(vfloat4& a, const vfloat4& b) { return a = a / b; }
  __forceinline vfloat4& operator /=(vfloat4& a, float          b) { return a = a / b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX512VL__)
  __forceinline vboolf4 operator ==(const vfloat4& a, const vfloat4& b) { return _mm_cmp_ps_mask(a, b, _MM_CMPINT_EQ); }
  __forceinline vboolf4 operator !=(const vfloat4& a, const vfloat4& b) { return _mm_cmp_ps_mask(a, b, _MM_CMPINT_NE); }
  __forceinline vboolf4 operator < (const vfloat4& a, const vfloat4& b) { return _mm_cmp_ps_mask(a, b, _MM_CMPINT_LT); }
  __forceinline vboolf4 operator >=(const vfloat4& a, const vfloat4& b) { return _mm_cmp_ps_mask(a, b, _MM_CMPINT_GE); }
  __forceinline vboolf4 operator > (const vfloat4& a, const vfloat4& b) { return _mm_cmp_ps_mask(a, b, _MM_CMPINT_GT); }
  __forceinline vboolf4 operator <=(const vfloat4& a, const vfloat4& b) { return _mm_cmp_ps_mask(a, b, _MM_CMPINT_LE); }
#else
  __forceinline vboolf4 operator ==(const vfloat4& a, const vfloat4& b) { return _mm_cmpeq_ps (a, b); }
  __forceinline vboolf4 operator !=(const vfloat4& a, const vfloat4& b) { return _mm_cmpneq_ps(a, b); }
  __forceinline vboolf4 operator < (const vfloat4& a, const vfloat4& b) { return _mm_cmplt_ps (a, b); }
  __forceinline vboolf4 operator >=(const vfloat4& a, const vfloat4& b) { return _mm_cmpnlt_ps(a, b); }
  __forceinline vboolf4 operator > (const vfloat4& a, const vfloat4& b) { return _mm_cmpnle_ps(a, b); }
  __forceinline vboolf4 operator <=(const vfloat4& a, const vfloat4& b) { return _mm_cmple_ps (a, b); }
#endif

  __forceinline vboolf4 operator ==(const vfloat4& a, float          b) { return a == vfloat4(b); }
  __forceinline vboolf4 operator ==(float          a, const vfloat4& b) { return vfloat4(a) == b; }

  __forceinline vboolf4 operator !=(const vfloat4& a, float          b) { return a != vfloat4(b); }
  __forceinline vboolf4 operator !=(float          a, const vfloat4& b) { return vfloat4(a) != b; }

  __forceinline vboolf4 operator < (const vfloat4& a, float          b) { return a <  vfloat4(b); }
  __forceinline vboolf4 operator < (float          a, const vfloat4& b) { return vfloat4(a) <  b; }
  
  __forceinline vboolf4 operator >=(const vfloat4& a, float          b) { return a >= vfloat4(b); }
  __forceinline vboolf4 operator >=(float          a, const vfloat4& b) { return vfloat4(a) >= b; }

  __forceinline vboolf4 operator > (const vfloat4& a, float          b) { return a >  vfloat4(b); }
  __forceinline vboolf4 operator > (float          a, const vfloat4& b) { return vfloat4(a) >  b; }

  __forceinline vboolf4 operator <=(const vfloat4& a, float          b) { return a <= vfloat4(b); }
  __forceinline vboolf4 operator <=(float          a, const vfloat4& b) { return vfloat4(a) <= b; }

  __forceinline vboolf4 eq(const vfloat4& a, const vfloat4& b) { return a == b; }
  __forceinline vboolf4 ne(const vfloat4& a, const vfloat4& b) { return a != b; }
  __forceinline vboolf4 lt(const vfloat4& a, const vfloat4& b) { return a <  b; }
  __forceinline vboolf4 ge(const vfloat4& a, const vfloat4& b) { return a >= b; }
  __forceinline vboolf4 gt(const vfloat4& a, const vfloat4& b) { return a >  b; }
  __forceinline vboolf4 le(const vfloat4& a, const vfloat4& b) { return a <= b; }

#if defined(__AVX512VL__)
  __forceinline vboolf4 eq(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return _mm_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_EQ); }
  __forceinline vboolf4 ne(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return _mm_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_NE); }
  __forceinline vboolf4 lt(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return _mm_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_LT); }
  __forceinline vboolf4 ge(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return _mm_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_GE); }
  __forceinline vboolf4 gt(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return _mm_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_GT); }
  __forceinline vboolf4 le(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return _mm_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_LE); }
#else
  __forceinline vboolf4 eq(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return mask & (a == b); }
  __forceinline vboolf4 ne(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return mask & (a != b); }
  __forceinline vboolf4 lt(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return mask & (a <  b); }
  __forceinline vboolf4 ge(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return mask & (a >= b); }
  __forceinline vboolf4 gt(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return mask & (a >  b); }
  __forceinline vboolf4 le(const vboolf4& mask, const vfloat4& a, const vfloat4& b) { return mask & (a <= b); }
#endif

  template<int mask>
    __forceinline vfloat4 select(const vfloat4& t, const vfloat4& f)
  {
#if defined(__SSE4_1__) 
    return _mm_blend_ps(f, t, mask);
#else
    return select(vboolf4(mask), t, f);
#endif
  }
  
  __forceinline vfloat4 lerp(const vfloat4& a, const vfloat4& b, const vfloat4& t) {
    return madd(t,b-a,a);
  }
  
  __forceinline bool isvalid(const vfloat4& v) {
    return all((v > vfloat4(-FLT_LARGE)) & (v < vfloat4(+FLT_LARGE)));
  }

  __forceinline bool is_finite(const vfloat4& a) {
    return all((a >= vfloat4(-FLT_MAX)) & (a <= vfloat4(+FLT_MAX)));
  }

  __forceinline bool is_finite(const vboolf4& valid, const vfloat4& a) {
    return all(valid, (a >= vfloat4(-FLT_MAX)) & (a <= vfloat4(+FLT_MAX)));
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Rounding Functions
  ////////////////////////////////////////////////////////////////////////////////

#if defined (__SSE4_1__)
  __forceinline vfloat4 floor(const vfloat4& a) { return _mm_round_ps(a, _MM_FROUND_TO_NEG_INF   ); }
  __forceinline vfloat4 ceil (const vfloat4& a) { return _mm_round_ps(a, _MM_FROUND_TO_POS_INF   ); }
  __forceinline vfloat4 trunc(const vfloat4& a) { return _mm_round_ps(a, _MM_FROUND_TO_ZERO      ); }
#else
  __forceinline vfloat4 floor(const vfloat4& a) { return vfloat4(floorf(a[0]),floorf(a[1]),floorf(a[2]),floorf(a[3]));  }
  __forceinline vfloat4 ceil (const vfloat4& a) { return vfloat4(ceilf (a[0]),ceilf (a[1]),ceilf (a[2]),ceilf (a[3])); }
  //__forceinline vfloat4 trunc(const vfloat4& a) { return vfloat4(truncf(a[0]),truncf(a[1]),truncf(a[2]),truncf(a[3])); }
#endif
  __forceinline vfloat4 frac(const vfloat4& a) { return a-floor(a); }

  __forceinline vint4 floori(const vfloat4& a) {
#if defined(__SSE4_1__)
    return vint4(floor(a));
#else
    return vint4(a-vfloat4(0.5f));
#endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat4 unpacklo(const vfloat4& a, const vfloat4& b) { return _mm_unpacklo_ps(a, b); }
  __forceinline vfloat4 unpackhi(const vfloat4& a, const vfloat4& b) { return _mm_unpackhi_ps(a, b); }

  template<int i0, int i1, int i2, int i3>
  __forceinline vfloat4 shuffle(const vfloat4& v) {
    return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), _MM_SHUFFLE(i3, i2, i1, i0)));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vfloat4 shuffle(const vfloat4& a, const vfloat4& b) {
    return _mm_shuffle_ps(a, b, _MM_SHUFFLE(i3, i2, i1, i0));
  }

#if defined (__SSSE3__)
  __forceinline vfloat4 shuffle8(const vfloat4& a, const vint4& shuf) {
    return _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(a), shuf)); 
  }
#endif

#if defined(__SSE3__)
  template<> __forceinline vfloat4 shuffle<0, 0, 2, 2>(const vfloat4& v) { return _mm_moveldup_ps(v); }
  template<> __forceinline vfloat4 shuffle<1, 1, 3, 3>(const vfloat4& v) { return _mm_movehdup_ps(v); }
  template<> __forceinline vfloat4 shuffle<0, 1, 0, 1>(const vfloat4& v) { return _mm_castpd_ps(_mm_movedup_pd(_mm_castps_pd(v))); }
#endif

  template<int i>
  __forceinline vfloat4 shuffle(const vfloat4& v) {
    return shuffle<i,i,i,i>(v);
  }

#if defined (__SSE4_1__) && !defined(__GNUC__)
  template<int i> __forceinline float extract(const vfloat4& a) { return _mm_cvtss_f32(_mm_extract_ps(a,i)); }
#else
  template<int i> __forceinline float extract(const vfloat4& a) { return _mm_cvtss_f32(shuffle<i,i,i,i>(a)); }
#endif
  template<> __forceinline float extract<0>(const vfloat4& a) { return _mm_cvtss_f32(a); }

#if defined (__SSE4_1__)
  template<int dst, int src, int clr> __forceinline vfloat4 insert(const vfloat4& a, const vfloat4& b) { return _mm_insert_ps(a, b, (dst << 4) | (src << 6) | clr); }
  template<int dst, int src> __forceinline vfloat4 insert(const vfloat4& a, const vfloat4& b) { return insert<dst, src, 0>(a, b); }
  template<int dst> __forceinline vfloat4 insert(const vfloat4& a, const float b) { return insert<dst, 0>(a, _mm_set_ss(b)); }
#else
  template<int dst, int src> __forceinline vfloat4 insert(const vfloat4& a, const vfloat4& b) { vfloat4 c = a; c[dst&3] = b[src&3]; return c; }
  template<int dst>  __forceinline vfloat4 insert(const vfloat4& a, float b) { vfloat4 c = a; c[dst&3] = b; return c; }
#endif

  __forceinline float toScalar(const vfloat4& v) { return _mm_cvtss_f32(v); }

  __forceinline vfloat4 broadcast4f(const vfloat4& a, size_t k) {
    return vfloat4::broadcast(&a[k]);
  }

  __forceinline vfloat4 shift_right_1(const vfloat4& x) {
    return _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(x), 4)); 
  }

#if defined (__AVX2__)
  __forceinline vfloat4 permute(const vfloat4 &a, const __m128i &index) {
    return _mm_permutevar_ps(a,index);
  }

  __forceinline vfloat4 broadcast1f(const void* a) { return _mm_broadcast_ss((float*)a); }

#endif

#if defined(__AVX512VL__)
  template<int i>
  __forceinline vfloat4 align_shift_right(const vfloat4& a, const vfloat4& b) {
    return _mm_castsi128_ps(_mm_alignr_epi32(_mm_castps_si128(a), _mm_castps_si128(b), i));
  }  
#endif


  ////////////////////////////////////////////////////////////////////////////////
  /// Sorting Network
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat4 sort_ascending(const vfloat4& v)
  {
    const vfloat4 a0 = v;
    const vfloat4 b0 = shuffle<1,0,3,2>(a0);
    const vfloat4 c0 = min(a0,b0);
    const vfloat4 d0 = max(a0,b0);
    const vfloat4 a1 = select<0x5 /* 0b0101 */>(c0,d0);
    const vfloat4 b1 = shuffle<2,3,0,1>(a1);
    const vfloat4 c1 = min(a1,b1);
    const vfloat4 d1 = max(a1,b1);
    const vfloat4 a2 = select<0x3 /* 0b0011 */>(c1,d1);
    const vfloat4 b2 = shuffle<0,2,1,3>(a2);
    const vfloat4 c2 = min(a2,b2);
    const vfloat4 d2 = max(a2,b2);
    const vfloat4 a3 = select<0x2 /* 0b0010 */>(c2,d2);
    return a3;
  }

  __forceinline vfloat4 sort_descending(const vfloat4& v)
  {
    const vfloat4 a0 = v;
    const vfloat4 b0 = shuffle<1,0,3,2>(a0);
    const vfloat4 c0 = max(a0,b0);
    const vfloat4 d0 = min(a0,b0);
    const vfloat4 a1 = select<0x5 /* 0b0101 */>(c0,d0);
    const vfloat4 b1 = shuffle<2,3,0,1>(a1);
    const vfloat4 c1 = max(a1,b1);
    const vfloat4 d1 = min(a1,b1);
    const vfloat4 a2 = select<0x3 /* 0b0011 */>(c1,d1);
    const vfloat4 b2 = shuffle<0,2,1,3>(a2);
    const vfloat4 c2 = max(a2,b2);
    const vfloat4 d2 = min(a2,b2);
    const vfloat4 a3 = select<0x2 /* 0b0010 */>(c2,d2);
    return a3;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Transpose
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline void transpose(const vfloat4& r0, const vfloat4& r1, const vfloat4& r2, const vfloat4& r3, vfloat4& c0, vfloat4& c1, vfloat4& c2, vfloat4& c3)
  {
    vfloat4 l02 = unpacklo(r0,r2);
    vfloat4 h02 = unpackhi(r0,r2);
    vfloat4 l13 = unpacklo(r1,r3);
    vfloat4 h13 = unpackhi(r1,r3);
    c0 = unpacklo(l02,l13);
    c1 = unpackhi(l02,l13);
    c2 = unpacklo(h02,h13);
    c3 = unpackhi(h02,h13);
  }

  __forceinline void transpose(const vfloat4& r0, const vfloat4& r1, const vfloat4& r2, const vfloat4& r3, vfloat4& c0, vfloat4& c1, vfloat4& c2)
  {
    vfloat4 l02 = unpacklo(r0,r2);
    vfloat4 h02 = unpackhi(r0,r2);
    vfloat4 l13 = unpacklo(r1,r3);
    vfloat4 h13 = unpackhi(r1,r3);
    c0 = unpacklo(l02,l13);
    c1 = unpackhi(l02,l13);
    c2 = unpacklo(h02,h13);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat4 vreduce_min(const vfloat4& v) { vfloat4 h = min(shuffle<1,0,3,2>(v),v); return min(shuffle<2,3,0,1>(h),h); }
  __forceinline vfloat4 vreduce_max(const vfloat4& v) { vfloat4 h = max(shuffle<1,0,3,2>(v),v); return max(shuffle<2,3,0,1>(h),h); }
  __forceinline vfloat4 vreduce_add(const vfloat4& v) { vfloat4 h = shuffle<1,0,3,2>(v)   + v ; return shuffle<2,3,0,1>(h)   + h ; }

  __forceinline float reduce_min(const vfloat4& v) { return _mm_cvtss_f32(vreduce_min(v)); }
  __forceinline float reduce_max(const vfloat4& v) { return _mm_cvtss_f32(vreduce_max(v)); }
  __forceinline float reduce_add(const vfloat4& v) { return _mm_cvtss_f32(vreduce_add(v)); }

  __forceinline size_t select_min(const vboolf4& valid, const vfloat4& v) 
  { 
    const vfloat4 a = select(valid,v,vfloat4(pos_inf)); 
    const vbool4 valid_min = valid & (a == vreduce_min(a));
    return bsf(movemask(any(valid_min) ? valid_min : valid)); 
  }
  __forceinline size_t select_max(const vboolf4& valid, const vfloat4& v) 
  { 
    const vfloat4 a = select(valid,v,vfloat4(neg_inf)); 
    const vbool4 valid_max = valid & (a == vreduce_max(a));
    return bsf(movemask(any(valid_max) ? valid_max : valid)); 
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Euclidian Space Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline float dot(const vfloat4& a, const vfloat4& b) {
    return reduce_add(a*b);
  }

  __forceinline vfloat4 cross(const vfloat4& a, const vfloat4& b)
  {
    const vfloat4 a0 = a;
    const vfloat4 b0 = shuffle<1,2,0,3>(b);
    const vfloat4 a1 = shuffle<1,2,0,3>(a);
    const vfloat4 b1 = b;
    return shuffle<1,2,0,3>(msub(a0,b0,a1*b1));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  inline std::ostream& operator <<(std::ostream& cout, const vfloat4& a) {
    return cout << "<" << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << ">";
  }

}
