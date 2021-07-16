// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"

/* Macros to gather statistics */
#ifdef EMBREE_STAT_COUNTERS
#  define STAT(x) x
#  define STAT3(s,x,y,z) \
  STAT(Stat::get().code  .s+=x);               \
  STAT(Stat::get().active.s+=y);               \
  STAT(Stat::get().all   .s+=z);
#  define STAT_USER(i,x) Stat::get().user[i]+=x;
#else
#  define STAT(x)
#  define STAT3(s,x,y,z)
#  define STAT_USER(i,x) 
#endif

namespace embree
{
  /*! Gathers ray tracing statistics. We count 1) how often a code
   *  location is reached, 2) how many SIMD lanes are active, 3) how
   *  many SIMD lanes reach the code location */
  class Stat
  { 
  public:

    static const size_t SIZE_HISTOGRAM = 64+1;

    /*! constructs stat counter class */
    Stat ();

    /*! destructs stat counter class */
    ~Stat ();

    class Counters 
    {
    public:
      Counters () { 
        clear(); 
      }
      
      void clear() 
      { 
        all.clear();
        active.clear();
        code.clear();
        for (auto& u : user) u.store(0);
      }

    public:

	/* per packet and per ray stastics */
	struct Data
        {
          void clear () {
            normal.clear();
            shadow.clear();
            point_query.clear();
          }

	  /* normal and shadow ray statistics */
	  struct 
          {
            void clear() 
            {
              travs.store(0);
              trav_nodes.store(0);
              trav_leaves.store(0);
              trav_prims.store(0);
              trav_prim_hits.store(0);
              for (auto& v : trav_hit_boxes) v.store(0);
              trav_stack_pop.store(0);
              trav_stack_nodes.store(0); 
              trav_xfm_nodes.store(0); 
            }

          public:
	    std::atomic<size_t> travs;
	    std::atomic<size_t> trav_nodes;
	    std::atomic<size_t> trav_leaves;
	    std::atomic<size_t> trav_prims;
	    std::atomic<size_t> trav_prim_hits;
	    std::atomic<size_t> trav_hit_boxes[SIZE_HISTOGRAM+1];
	    std::atomic<size_t> trav_stack_pop;
	    std::atomic<size_t> trav_stack_nodes; 
            std::atomic<size_t> trav_xfm_nodes; 
            
	  } normal, shadow, point_query;
	} all, active, code; 

        std::atomic<size_t> user[10];
    };

  public:

    static __forceinline Counters& get() {
      return instance.cntrs;
    }
    
    static void clear() {
      instance.cntrs.clear();
    }
    
    static void print(embree_ostream cout);

  private: 
    Counters cntrs;

  private:
    static Stat instance;
  };
}
