/*
    Copyright (c) 2019-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_info_H
#define __TBB_info_H

#include "tbb_config.h"

#if __TBB_NUMA_SUPPORT

#include <vector>

namespace tbb {
    namespace internal {
        namespace numa_topology {
            unsigned nodes_count();
            void fill(int* indexes_array);
            int default_concurrency(int node_id);
        } //namespace numa_topology
    } // namespace internal

    typedef int numa_node_id;

    namespace info {
        inline std::vector<numa_node_id> numa_nodes() {
            std::vector<numa_node_id> nodes_indexes(tbb::internal::numa_topology::nodes_count());
            internal::numa_topology::fill(&nodes_indexes.front());
            return nodes_indexes;
        }

        inline int default_concurrency(numa_node_id id = -1) {
            return internal::numa_topology::default_concurrency(id);
        }
    } // namespace info
} // namespace tbb

#endif /*__TBB_NUMA_SUPPORT*/

#endif /*__TBB_info_H*/
