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

#include "stat.h"

namespace embree
{
  Stat Stat::instance; 
  
  Stat::Stat () {
  }

  Stat::~Stat () 
  {
#ifdef EMBREE_STAT_COUNTERS
    Stat::print(std::cout);
#endif
  }

  void Stat::print(std::ostream& cout)
  {
    Counters& cntrs = instance.cntrs;
    Counters::Data& data = instance.cntrs.code;
    //Counters::Data& data = instance.cntrs.active;

    /* print absolute numbers */
    cout << "--------- ABSOLUTE ---------" << std::endl;
    cout << "  #normal_travs   = " << float(data.normal.travs            )*1E-6 << "M" << std::endl;
    cout << "    #nodes        = " << float(data.normal.trav_nodes       )*1E-6 << "M" << std::endl;
    cout << "    #nodes_xfm    = " << float(data.normal.trav_xfm_nodes   )*1E-6 << "M" << std::endl;
    cout << "    #leaves       = " << float(data.normal.trav_leaves      )*1E-6 << "M" << std::endl;
    cout << "    #prims        = " << float(data.normal.trav_prims       )*1E-6 << "M" << std::endl;
    cout << "    #prim_hits    = " << float(data.normal.trav_prim_hits   )*1E-6 << "M" << std::endl;

    cout << "    #stack nodes  = " << float(data.normal.trav_stack_nodes )*1E-6 << "M" << std::endl;
    cout << "    #stack pop    = " << float(data.normal.trav_stack_pop )*1E-6 << "M" << std::endl;

    size_t normal_box_hits = 0;
    size_t weighted_box_hits = 0;
    for (size_t i=0;i<SIZE_HISTOGRAM;i++) { 
      normal_box_hits += data.normal.trav_hit_boxes[i];
      weighted_box_hits += data.normal.trav_hit_boxes[i]*i;
    }
    cout << "    #hit_boxes    = " << normal_box_hits << " (total) distribution: ";
    float average = 0.0f;
    for (size_t i=0;i<SIZE_HISTOGRAM;i++) 
    {
      float value = 100.0f * data.normal.trav_hit_boxes[i] / normal_box_hits;
      cout << "[" << i << "] " << value << " ";
      average += (float)i*data.normal.trav_hit_boxes[i] / normal_box_hits;
    }
    cout << "    average = " << average << std::endl;
    for (size_t i=0;i<SIZE_HISTOGRAM;i++) cout << "[" << i << "] " << 100.0f * data.normal.trav_hit_boxes[i]*i / weighted_box_hits << " ";
    cout << std::endl;

    if (data.shadow.travs) {
      cout << "  #shadow_travs = " << float(data.shadow.travs         )*1E-6 << "M" << std::endl;
      cout << "    #nodes      = " << float(data.shadow.trav_nodes    )*1E-6 << "M" << std::endl;
      cout << "    #nodes_xfm  = " << float(data.shadow.trav_xfm_nodes)*1E-6 << "M" << std::endl;
      cout << "    #leaves     = " << float(data.shadow.trav_leaves   )*1E-6 << "M" << std::endl;
      cout << "    #prims      = " << float(data.shadow.trav_prims    )*1E-6 << "M" << std::endl;
      cout << "    #prim_hits  = " << float(data.shadow.trav_prim_hits)*1E-6 << "M" << std::endl;

      cout << "    #stack nodes = " << float(data.shadow.trav_stack_nodes )*1E-6 << "M" << std::endl;
      cout << "    #stack pop   = " << float(data.shadow.trav_stack_pop )*1E-6 << "M" << std::endl;

      size_t shadow_box_hits = 0;
      size_t weighted_shadow_box_hits = 0;

      for (size_t i=0;i<SIZE_HISTOGRAM;i++) {        
        shadow_box_hits += data.shadow.trav_hit_boxes[i];
        weighted_shadow_box_hits += data.shadow.trav_hit_boxes[i]*i;
      }
      cout << "    #hit_boxes    = ";
      for (size_t i=0;i<SIZE_HISTOGRAM;i++) cout << "[" << i << "] " << 100.0f * data.shadow.trav_hit_boxes[i] / shadow_box_hits << " ";
      cout << std::endl;
      for (size_t i=0;i<SIZE_HISTOGRAM;i++) cout << "[" << i << "] " << 100.0f * data.shadow.trav_hit_boxes[i]*i / weighted_shadow_box_hits << " ";
      cout << std::endl;
    }
    cout << std::endl;

    /* print per traversal numbers */
    cout << "--------- PER TRAVERSAL ---------" << std::endl;
    float active_normal_travs       = float(cntrs.active.normal.travs      )/float(cntrs.all.normal.travs      );
    float active_normal_trav_nodes  = float(cntrs.active.normal.trav_nodes )/float(cntrs.all.normal.trav_nodes );
    float active_normal_trav_xfm_nodes  = float(cntrs.active.normal.trav_xfm_nodes )/float(cntrs.all.normal.trav_xfm_nodes );
    float active_normal_trav_leaves = float(cntrs.active.normal.trav_leaves)/float(cntrs.all.normal.trav_leaves);
    float active_normal_trav_prims   = float(cntrs.active.normal.trav_prims  )/float(cntrs.all.normal.trav_prims  );
    float active_normal_trav_prim_hits = float(cntrs.active.normal.trav_prim_hits  )/float(cntrs.all.normal.trav_prim_hits  );
    float active_normal_trav_stack_pop = float(cntrs.active.normal.trav_stack_pop  )/float(cntrs.all.normal.trav_stack_pop  );

    cout << "  #normal_travs   = " << float(cntrs.code.normal.travs      )/float(cntrs.code.normal.travs) << ", " << 100.0f*active_normal_travs       << "% active" << std::endl;
    cout << "    #nodes        = " << float(cntrs.code.normal.trav_nodes )/float(cntrs.code.normal.travs) << ", " << 100.0f*active_normal_trav_nodes  << "% active" << std::endl;
    cout << "    #node_xfm     = " << float(cntrs.code.normal.trav_xfm_nodes )/float(cntrs.code.normal.travs) << ", " << 100.0f*active_normal_trav_xfm_nodes  << "% active" << std::endl;
    cout << "    #leaves       = " << float(cntrs.code.normal.trav_leaves)/float(cntrs.code.normal.travs) << ", " << 100.0f*active_normal_trav_leaves << "% active" << std::endl;
    cout << "    #prims        = " << float(cntrs.code.normal.trav_prims  )/float(cntrs.code.normal.travs) << ", " << 100.0f*active_normal_trav_prims   << "% active" << std::endl;
    cout << "    #prim_hits    = " << float(cntrs.code.normal.trav_prim_hits  )/float(cntrs.code.normal.travs) << ", " << 100.0f*active_normal_trav_prim_hits   << "% active" << std::endl;
    cout << "    #stack_pop    = " << float(cntrs.code.normal.trav_stack_pop  )/float(cntrs.code.normal.travs) << ", " << 100.0f*active_normal_trav_stack_pop   << "% active" << std::endl;

    if (cntrs.all.shadow.travs) {
      float active_shadow_travs       = float(cntrs.active.shadow.travs      )/float(cntrs.all.shadow.travs      );
      float active_shadow_trav_nodes  = float(cntrs.active.shadow.trav_nodes )/float(cntrs.all.shadow.trav_nodes );
      float active_shadow_trav_xfm_nodes  = float(cntrs.active.shadow.trav_xfm_nodes )/float(cntrs.all.shadow.trav_xfm_nodes );
      float active_shadow_trav_leaves = float(cntrs.active.shadow.trav_leaves)/float(cntrs.all.shadow.trav_leaves);
      float active_shadow_trav_prims   = float(cntrs.active.shadow.trav_prims  )/float(cntrs.all.shadow.trav_prims  );
      float active_shadow_trav_prim_hits = float(cntrs.active.shadow.trav_prim_hits  )/float(cntrs.all.shadow.trav_prim_hits  );

      cout << "  #shadow_travs = " << float(cntrs.code.shadow.travs      )/float(cntrs.code.shadow.travs) << ", " << 100.0f*active_shadow_travs       << "% active" << std::endl;
      cout << "    #nodes      = " << float(cntrs.code.shadow.trav_nodes )/float(cntrs.code.shadow.travs) << ", " << 100.0f*active_shadow_trav_nodes  << "% active" << std::endl;
      cout << "    #nodes_xfm  = " << float(cntrs.code.shadow.trav_xfm_nodes )/float(cntrs.code.shadow.travs) << ", " << 100.0f*active_shadow_trav_xfm_nodes  << "% active" << std::endl;
      cout << "    #leaves     = " << float(cntrs.code.shadow.trav_leaves)/float(cntrs.code.shadow.travs) << ", " << 100.0f*active_shadow_trav_leaves << "% active" << std::endl;
      cout << "    #prims      = " << float(cntrs.code.shadow.trav_prims  )/float(cntrs.code.shadow.travs) << ", " << 100.0f*active_shadow_trav_prims   << "% active" << std::endl;
      cout << "    #prim_hits  = " << float(cntrs.code.shadow.trav_prim_hits  )/float(cntrs.code.shadow.travs) << ", " << 100.0f*active_shadow_trav_prim_hits   << "% active" << std::endl;

    }
    cout << std::endl;

     /* print user counters for performance tuning */
    cout << "--------- USER ---------" << std::endl;
    for (size_t i=0; i<10; i++)
      cout << "#user" << i << " = " << float(cntrs.user[i])/float(cntrs.all.normal.travs+cntrs.all.shadow.travs) << " per traversal" << std::endl;

    cout << "#user5/user3 " << 100.0f*float(cntrs.user[5])/float(cntrs.user[3]) << "%" << std::endl;
    cout << "#user6/user3 " << 100.0f*float(cntrs.user[6])/float(cntrs.user[3]) << "%" << std::endl;
    cout << "#user7/user3 " << 100.0f*float(cntrs.user[7])/float(cntrs.user[3]) << "%" << std::endl;
    cout << std::endl;
  }
}
