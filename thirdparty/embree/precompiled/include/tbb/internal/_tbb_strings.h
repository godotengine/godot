/*
    Copyright (c) 2005-2020 Intel Corporation

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

TBB_STRING_RESOURCE(FLOW_BROADCAST_NODE, "broadcast_node")
TBB_STRING_RESOURCE(FLOW_BUFFER_NODE, "buffer_node")
TBB_STRING_RESOURCE(FLOW_CONTINUE_NODE, "continue_node")
TBB_STRING_RESOURCE(FLOW_FUNCTION_NODE, "function_node")
TBB_STRING_RESOURCE(FLOW_JOIN_NODE_QUEUEING, "join_node (queueing)")
TBB_STRING_RESOURCE(FLOW_JOIN_NODE_RESERVING, "join_node (reserving)")
TBB_STRING_RESOURCE(FLOW_JOIN_NODE_TAG_MATCHING, "join_node (tag_matching)")
TBB_STRING_RESOURCE(FLOW_LIMITER_NODE, "limiter_node")
TBB_STRING_RESOURCE(FLOW_MULTIFUNCTION_NODE, "multifunction_node")
TBB_STRING_RESOURCE(FLOW_OR_NODE, "or_node") //no longer in use, kept for backward compatibility
TBB_STRING_RESOURCE(FLOW_OVERWRITE_NODE, "overwrite_node")
TBB_STRING_RESOURCE(FLOW_PRIORITY_QUEUE_NODE, "priority_queue_node")
TBB_STRING_RESOURCE(FLOW_QUEUE_NODE, "queue_node")
TBB_STRING_RESOURCE(FLOW_SEQUENCER_NODE, "sequencer_node")
TBB_STRING_RESOURCE(FLOW_SOURCE_NODE, "source_node")
TBB_STRING_RESOURCE(FLOW_SPLIT_NODE, "split_node")
TBB_STRING_RESOURCE(FLOW_WRITE_ONCE_NODE, "write_once_node")
TBB_STRING_RESOURCE(FLOW_BODY, "body")
TBB_STRING_RESOURCE(FLOW_GRAPH, "graph")
TBB_STRING_RESOURCE(FLOW_NODE, "node")
TBB_STRING_RESOURCE(FLOW_INPUT_PORT, "input_port")
TBB_STRING_RESOURCE(FLOW_INPUT_PORT_0, "input_port_0")
TBB_STRING_RESOURCE(FLOW_INPUT_PORT_1, "input_port_1")
TBB_STRING_RESOURCE(FLOW_INPUT_PORT_2, "input_port_2")
TBB_STRING_RESOURCE(FLOW_INPUT_PORT_3, "input_port_3")
TBB_STRING_RESOURCE(FLOW_INPUT_PORT_4, "input_port_4")
TBB_STRING_RESOURCE(FLOW_INPUT_PORT_5, "input_port_5")
TBB_STRING_RESOURCE(FLOW_INPUT_PORT_6, "input_port_6")
TBB_STRING_RESOURCE(FLOW_INPUT_PORT_7, "input_port_7")
TBB_STRING_RESOURCE(FLOW_INPUT_PORT_8, "input_port_8")
TBB_STRING_RESOURCE(FLOW_INPUT_PORT_9, "input_port_9")
TBB_STRING_RESOURCE(FLOW_OUTPUT_PORT, "output_port")
TBB_STRING_RESOURCE(FLOW_OUTPUT_PORT_0, "output_port_0")
TBB_STRING_RESOURCE(FLOW_OUTPUT_PORT_1, "output_port_1")
TBB_STRING_RESOURCE(FLOW_OUTPUT_PORT_2, "output_port_2")
TBB_STRING_RESOURCE(FLOW_OUTPUT_PORT_3, "output_port_3")
TBB_STRING_RESOURCE(FLOW_OUTPUT_PORT_4, "output_port_4")
TBB_STRING_RESOURCE(FLOW_OUTPUT_PORT_5, "output_port_5")
TBB_STRING_RESOURCE(FLOW_OUTPUT_PORT_6, "output_port_6")
TBB_STRING_RESOURCE(FLOW_OUTPUT_PORT_7, "output_port_7")
TBB_STRING_RESOURCE(FLOW_OUTPUT_PORT_8, "output_port_8")
TBB_STRING_RESOURCE(FLOW_OUTPUT_PORT_9, "output_port_9")
TBB_STRING_RESOURCE(FLOW_OBJECT_NAME, "object_name")
TBB_STRING_RESOURCE(FLOW_NULL, "null")
TBB_STRING_RESOURCE(FLOW_INDEXER_NODE, "indexer_node")
TBB_STRING_RESOURCE(FLOW_COMPOSITE_NODE, "composite_node")
TBB_STRING_RESOURCE(FLOW_ASYNC_NODE, "async_node")
TBB_STRING_RESOURCE(FLOW_OPENCL_NODE, "opencl_node")
TBB_STRING_RESOURCE(ALGORITHM, "tbb_algorithm")
TBB_STRING_RESOURCE(PARALLEL_FOR, "tbb_parallel_for")
TBB_STRING_RESOURCE(PARALLEL_DO, "tbb_parallel_do")
TBB_STRING_RESOURCE(PARALLEL_INVOKE, "tbb_parallel_invoke")
TBB_STRING_RESOURCE(PARALLEL_REDUCE, "tbb_parallel_reduce")
TBB_STRING_RESOURCE(PARALLEL_SCAN, "tbb_parallel_scan")
TBB_STRING_RESOURCE(PARALLEL_SORT, "tbb_parallel_sort")
TBB_STRING_RESOURCE(CUSTOM_CTX, "tbb_custom")
TBB_STRING_RESOURCE(FLOW_TASKS, "tbb_flow_graph")
TBB_STRING_RESOURCE(PARALLEL_FOR_TASK, "tbb_parallel_for_task")
// TODO: Drop following string prefix "fgt_" here and in FGA's collector
TBB_STRING_RESOURCE(USER_EVENT, "fgt_user_event")
#if __TBB_CPF_BUILD || (TBB_PREVIEW_FLOW_GRAPH_TRACE && TBB_USE_THREADING_TOOLS)
TBB_STRING_RESOURCE(CODE_ADDRESS, "code_address")
#endif
