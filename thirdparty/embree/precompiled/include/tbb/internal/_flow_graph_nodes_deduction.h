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

#ifndef __TBB_flow_graph_nodes_deduction_H
#define __TBB_flow_graph_nodes_deduction_H

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

namespace tbb {
namespace flow {
namespace interface11 {

template <typename Input, typename Output>
struct declare_body_types {
    using input_type = Input;
    using output_type = Output;
};

struct NoInputBody {};

template <typename Output>
struct declare_body_types<NoInputBody, Output> {
    using output_type = Output;
};

template <typename T> struct body_types;

template <typename T, typename Input, typename Output>
struct body_types<Output (T::*)(const Input&) const> : declare_body_types<Input, Output> {};

template <typename T, typename Input, typename Output>
struct body_types<Output (T::*)(const Input&)> : declare_body_types<Input, Output> {};

template <typename T, typename Input, typename Output>
struct body_types<Output (T::*)(Input&) const> : declare_body_types<Input, Output> {};

template <typename T, typename Input, typename Output>
struct body_types<Output (T::*)(Input&)> : declare_body_types<Input, Output> {};

template <typename Input, typename Output>
struct body_types<Output (*)(Input&)> : declare_body_types<Input, Output> {};

template <typename Input, typename Output>
struct body_types<Output (*)(const Input&)> : declare_body_types<Input, Output> {};

template <typename T, typename Output>
struct body_types<Output (T::*)(flow_control&) const> : declare_body_types<NoInputBody, Output> {};

template <typename T, typename Output>
struct body_types<Output (T::*)(flow_control&)> : declare_body_types<NoInputBody, Output> {};

template <typename Output>
struct body_types<Output (*)(flow_control&)> : declare_body_types<NoInputBody, Output> {};

template <typename Body>
using input_t = typename body_types<Body>::input_type;

template <typename Body>
using output_t = typename body_types<Body>::output_type;

template <typename T, typename Input, typename Output>
auto decide_on_operator_overload(Output (T::*name)(const Input&) const)->decltype(name);

template <typename T, typename Input, typename Output>
auto decide_on_operator_overload(Output (T::*name)(const Input&))->decltype(name);

template <typename T, typename Input, typename Output>
auto decide_on_operator_overload(Output (T::*name)(Input&) const)->decltype(name);

template <typename T, typename Input, typename Output>
auto decide_on_operator_overload(Output (T::*name)(Input&))->decltype(name);

template <typename Input, typename Output>
auto decide_on_operator_overload(Output (*name)(const Input&))->decltype(name);

template <typename Input, typename Output>
auto decide_on_operator_overload(Output (*name)(Input&))->decltype(name);

template <typename Body>
decltype(decide_on_operator_overload(&Body::operator())) decide_on_callable_type(int);

template <typename Body>
decltype(decide_on_operator_overload(std::declval<Body>())) decide_on_callable_type(...);

// Deduction guides for Flow Graph nodes
#if TBB_USE_SOURCE_NODE_AS_ALIAS
#if TBB_DEPRECATED_INPUT_NODE_BODY
template <typename GraphOrSet, typename Body>
source_node(GraphOrSet&&, Body)
->source_node<input_t<decltype(decide_on_callable_type<Body>(0))>>;
#else
template <typename GraphOrSet, typename Body>
source_node(GraphOrSet&&, Body)
->source_node<output_t<decltype(decide_on_callable_type<Body>(0))>>;
#endif // TBB_DEPRECATED_INPUT_NODE_BODY
#else
template <typename GraphOrSet, typename Body>
source_node(GraphOrSet&&, Body, bool = true)
->source_node<input_t<decltype(decide_on_callable_type<Body>(0))>>;
#endif

#if TBB_DEPRECATED_INPUT_NODE_BODY
template <typename GraphOrSet, typename Body>
input_node(GraphOrSet&&, Body, bool = true)
->input_node<input_t<decltype(decide_on_callable_type<Body>(0))>>;
#else
template <typename GraphOrSet, typename Body>
input_node(GraphOrSet&&, Body)
->input_node<output_t<decltype(decide_on_callable_type<Body>(0))>>;
#endif

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

template <typename NodeSet>
struct decide_on_set;

template <typename Node, typename... Nodes>
struct decide_on_set<node_set<internal::order::following, Node, Nodes...>> {
    using type = typename Node::output_type;
};

template <typename Node, typename... Nodes>
struct decide_on_set<node_set<internal::order::preceding, Node, Nodes...>> {
    using type = typename Node::input_type;
};

template <typename NodeSet>
using decide_on_set_t = typename decide_on_set<std::decay_t<NodeSet>>::type;

template <typename NodeSet>
broadcast_node(const NodeSet&)
->broadcast_node<decide_on_set_t<NodeSet>>;

template <typename NodeSet>
buffer_node(const NodeSet&)
->buffer_node<decide_on_set_t<NodeSet>>;

template <typename NodeSet>
queue_node(const NodeSet&)
->queue_node<decide_on_set_t<NodeSet>>;
#endif // __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

template <typename GraphOrProxy, typename Sequencer>
sequencer_node(GraphOrProxy&&, Sequencer)
->sequencer_node<input_t<decltype(decide_on_callable_type<Sequencer>(0))>>;

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
template <typename NodeSet, typename Compare>
priority_queue_node(const NodeSet&, const Compare&)
->priority_queue_node<decide_on_set_t<NodeSet>, Compare>;

template <typename NodeSet>
priority_queue_node(const NodeSet&)
->priority_queue_node<decide_on_set_t<NodeSet>, std::less<decide_on_set_t<NodeSet>>>;
#endif // __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

template <typename Key>
struct join_key {
    using type = Key;
};

template <typename T>
struct join_key<const T&> {
    using type = T&;
};

template <typename Key>
using join_key_t = typename join_key<Key>::type;

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
template <typename Policy, typename... Predecessors>
join_node(const node_set<internal::order::following, Predecessors...>&, Policy)
->join_node<std::tuple<typename Predecessors::output_type...>,
            Policy>;

template <typename Policy, typename Successor, typename... Successors>
join_node(const node_set<internal::order::preceding, Successor, Successors...>&, Policy)
->join_node<typename Successor::input_type, Policy>;

template <typename... Predecessors>
join_node(const node_set<internal::order::following, Predecessors...>)
->join_node<std::tuple<typename Predecessors::output_type...>,
            queueing>;

template <typename Successor, typename... Successors>
join_node(const node_set<internal::order::preceding, Successor, Successors...>)
->join_node<typename Successor::input_type, queueing>;
#endif

template <typename GraphOrProxy, typename Body, typename... Bodies>
join_node(GraphOrProxy&&, Body, Bodies...)
->join_node<std::tuple<input_t<decltype(decide_on_callable_type<Body>(0))>,
                       input_t<decltype(decide_on_callable_type<Bodies>(0))>...>,
            key_matching<join_key_t<output_t<decltype(decide_on_callable_type<Body>(0))>>>>;

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
template <typename... Predecessors>
indexer_node(const node_set<internal::order::following, Predecessors...>&)
->indexer_node<typename Predecessors::output_type...>;
#endif

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
template <typename NodeSet>
limiter_node(const NodeSet&, size_t)
->limiter_node<decide_on_set_t<NodeSet>>;

template <typename Predecessor, typename... Predecessors>
split_node(const node_set<internal::order::following, Predecessor, Predecessors...>&)
->split_node<typename Predecessor::output_type>;

template <typename... Successors>
split_node(const node_set<internal::order::preceding, Successors...>&)
->split_node<std::tuple<typename Successors::input_type...>>;

#endif

template <typename GraphOrSet, typename Body, typename Policy>
function_node(GraphOrSet&&,
              size_t, Body,
            __TBB_FLOW_GRAPH_PRIORITY_ARG1(Policy, node_priority_t = tbb::flow::internal::no_priority))
->function_node<input_t<decltype(decide_on_callable_type<Body>(0))>,
                output_t<decltype(decide_on_callable_type<Body>(0))>,
                Policy>;

template <typename GraphOrSet, typename Body>
function_node(GraphOrSet&&, size_t,
              __TBB_FLOW_GRAPH_PRIORITY_ARG1(Body, node_priority_t = tbb::flow::internal::no_priority))
->function_node<input_t<decltype(decide_on_callable_type<Body>(0))>,
                output_t<decltype(decide_on_callable_type<Body>(0))>,
                queueing>;

template <typename Output>
struct continue_output {
    using type = Output;
};

template <>
struct continue_output<void> {
    using type = continue_msg;
};

template <typename T>
using continue_output_t = typename continue_output<T>::type;

template <typename GraphOrSet, typename Body, typename Policy>
continue_node(GraphOrSet&&, Body,
              __TBB_FLOW_GRAPH_PRIORITY_ARG1(Policy, node_priority_t = tbb::flow::internal::no_priority))
->continue_node<continue_output_t<std::invoke_result_t<Body, continue_msg>>,
                Policy>;

template <typename GraphOrSet, typename Body, typename Policy>
continue_node(GraphOrSet&&,
              int, Body,
              __TBB_FLOW_GRAPH_PRIORITY_ARG1(Policy, node_priority_t = tbb::flow::internal::no_priority))
->continue_node<continue_output_t<std::invoke_result_t<Body, continue_msg>>,
                Policy>;

template <typename GraphOrSet, typename Body>
continue_node(GraphOrSet&&,
              __TBB_FLOW_GRAPH_PRIORITY_ARG1(Body, node_priority_t = tbb::flow::internal::no_priority))
->continue_node<continue_output_t<std::invoke_result_t<Body, continue_msg>>,
                internal::Policy<void>>;

template <typename GraphOrSet, typename Body>
continue_node(GraphOrSet&&, int,
              __TBB_FLOW_GRAPH_PRIORITY_ARG1(Body, node_priority_t = tbb::flow::internal::no_priority))
->continue_node<continue_output_t<std::invoke_result_t<Body, continue_msg>>,
                internal::Policy<void>>;

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

template <typename NodeSet>
overwrite_node(const NodeSet&)
->overwrite_node<decide_on_set_t<NodeSet>>;

template <typename NodeSet>
write_once_node(const NodeSet&)
->write_once_node<decide_on_set_t<NodeSet>>;
#endif // __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
} // namespace interfaceX
} // namespace flow
} // namespace tbb

#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
#endif // __TBB_flow_graph_nodes_deduction_H
