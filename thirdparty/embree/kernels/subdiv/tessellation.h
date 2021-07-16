// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace embree
{
  /* adjust discret tessellation level for feature-adaptive pre-subdivision */
  __forceinline float adjustTessellationLevel(float l, const size_t sublevel)
  {
    for (size_t i=0; i<sublevel; i++) l *= 0.5f;
    float r = ceilf(l);      
    for (size_t i=0; i<sublevel; i++) r *= 2.0f;
    return r;
  }
  
  __forceinline int stitch(const int x, const int fine, const int coarse) {
    return (2*x+1)*coarse/(2*fine);
  }

  __forceinline void stitchGridEdges(const unsigned int low_rate,
                                     const unsigned int high_rate,
                                     const unsigned int x0,
                                     const unsigned int x1,
				    float * __restrict__ const uv_array,
				    const unsigned int uv_array_step)
  {
#if 1
    const float inv_low_rate = rcp((float)(low_rate-1));
    for (unsigned x=x0; x<=x1; x++) {
      uv_array[(x-x0)*uv_array_step] = float(stitch(x,high_rate-1,low_rate-1))*inv_low_rate;
    }
    if (unlikely(x1 == high_rate-1))
      uv_array[(x1-x0)*uv_array_step] = 1.0f;
#else
    assert(low_rate < high_rate);
    assert(high_rate >= 2);
    
    const float inv_low_rate = rcp((float)(low_rate-1));
    const unsigned int dy = low_rate  - 1; 
    const unsigned int dx = high_rate - 1;
    
    int p = 2*dy-dx;  
    
    unsigned int offset = 0;
    unsigned int y = 0;
    float value = 0.0f;
    for(unsigned int x=0;x<high_rate-1; x++) // '<=' would be correct but we will leave the 1.0f at the end
    {
      uv_array[offset] = value;
      
      offset += uv_array_step;      
      if (unlikely(p > 0))
      {
	y++;
	value = (float)y * inv_low_rate;
	p -= 2*dx;
      }
      p += 2*dy;
    }
#endif
  }
  
  __forceinline void stitchUVGrid(const float edge_levels[4],
                                  const unsigned int swidth,
                                  const unsigned int sheight,
                                  const unsigned int x0,
                                  const unsigned int y0,
				  const unsigned int grid_u_res,
				  const unsigned int grid_v_res,
				  float * __restrict__ const u_array,
				  float * __restrict__ const v_array)
  {
    const unsigned int x1 = x0+grid_u_res-1;
    const unsigned int y1 = y0+grid_v_res-1;
    const unsigned int int_edge_points0 = (unsigned int)edge_levels[0] + 1;
    const unsigned int int_edge_points1 = (unsigned int)edge_levels[1] + 1;
    const unsigned int int_edge_points2 = (unsigned int)edge_levels[2] + 1;
    const unsigned int int_edge_points3 = (unsigned int)edge_levels[3] + 1;
    
    if (unlikely(y0 == 0 && int_edge_points0 < swidth))
      stitchGridEdges(int_edge_points0,swidth,x0,x1,u_array,1);
    
    if (unlikely(y1 == sheight-1 && int_edge_points2 < swidth))
      stitchGridEdges(int_edge_points2,swidth,x0,x1,&u_array[(grid_v_res-1)*grid_u_res],1);
    
    if (unlikely(x0 == 0 && int_edge_points1 < sheight))
      stitchGridEdges(int_edge_points1,sheight,y0,y1,&v_array[grid_u_res-1],grid_u_res);
    
    if (unlikely(x1 == swidth-1 && int_edge_points3 < sheight))
      stitchGridEdges(int_edge_points3,sheight,y0,y1,v_array,grid_u_res);  
  }
  
  __forceinline void gridUVTessellator(const float edge_levels[4],  
                                       const unsigned int swidth,
                                       const unsigned int sheight,
                                       const unsigned int x0,
                                       const unsigned int y0,
				       const unsigned int grid_u_res,
				       const unsigned int grid_v_res,
				       float * __restrict__ const u_array,
				       float * __restrict__ const v_array)
  {
    assert( grid_u_res >= 1);
    assert( grid_v_res >= 1);
    assert( edge_levels[0] >= 1.0f );
    assert( edge_levels[1] >= 1.0f );
    assert( edge_levels[2] >= 1.0f );
    assert( edge_levels[3] >= 1.0f );
    
#if defined(__AVX__)
    const vint8 grid_u_segments = vint8(swidth)-1;
    const vint8 grid_v_segments = vint8(sheight)-1;
    
    const vfloat8 inv_grid_u_segments = rcp(vfloat8(grid_u_segments));
    const vfloat8 inv_grid_v_segments = rcp(vfloat8(grid_v_segments));
    
    unsigned int index = 0;
    vint8 v_i( zero );
    for (unsigned int y=0;y<grid_v_res;y++,index+=grid_u_res,v_i += 1)
    {
      vint8 u_i ( step );
      
      const vbool8 m_v = v_i < grid_v_segments;
      
      for (unsigned int x=0;x<grid_u_res;x+=8, u_i += 8)
      {
        const vbool8 m_u = u_i < grid_u_segments;
	const vfloat8 u = select(m_u, vfloat8(x0+u_i) * inv_grid_u_segments, 1.0f);
	const vfloat8 v = select(m_v, vfloat8(y0+v_i) * inv_grid_v_segments, 1.0f);
	vfloat8::storeu(&u_array[index + x],u);
	vfloat8::storeu(&v_array[index + x],v);	   
      }
    }       
 #else   
    const vint4 grid_u_segments = vint4(swidth)-1;
    const vint4 grid_v_segments = vint4(sheight)-1;
    
    const vfloat4 inv_grid_u_segments = rcp(vfloat4(grid_u_segments));
    const vfloat4 inv_grid_v_segments = rcp(vfloat4(grid_v_segments));
    
    unsigned int index = 0;
    vint4 v_i( zero );
    for (unsigned int y=0;y<grid_v_res;y++,index+=grid_u_res,v_i += 1)
    {
      vint4 u_i ( step );
      
      const vbool4 m_v = v_i < grid_v_segments;
      
      for (unsigned int x=0;x<grid_u_res;x+=4, u_i += 4)
      {
        const vbool4 m_u = u_i < grid_u_segments;
	const vfloat4 u = select(m_u, vfloat4(x0+u_i) * inv_grid_u_segments, 1.0f);
	const vfloat4 v = select(m_v, vfloat4(y0+v_i) * inv_grid_v_segments, 1.0f);
        vfloat4::storeu(&u_array[index + x],u);
	vfloat4::storeu(&v_array[index + x],v);	   
      }
    }       
#endif
  } 
}
