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

#ifndef __TBB_flow_graph_node_set_impl_H
#define __TBB_flow_graph_node_set_impl_H

#ifndef __TBB_flow_graph_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

// Included in namespace tbb::flow::interfaceX (in flow_graph.h)

namespace internal {

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
// Visual Studio 2019 reports an error while calling predecessor_selector::get and successor_selector::get
// Seems like the well-formed expression in trailing decltype is treated as ill-formed
// TODO: investigate problems with decltype in trailing return types or find the cross-platform solution
#define __TBB_MSVC_DISABLE_TRAILING_DECLTYPE (_MSC_VER >= 1900)

namespace order {
struct undefined {};
struct following {};
struct preceding {};
}

class get_graph_helper {
public:
    // TODO: consider making graph_reference() public and consistent interface to get a reference to the graph
    // and remove get_graph_helper
    template <typename T>
    static graph& get(const T& object) {
        return get_impl(object, std::is_base_of<graph_node, T>());
    }

private:
    // Get graph from the object of type derived from graph_node
    template <typename T>
    static graph& get_impl(const T& object, std::true_type) {
        return static_cast<const graph_node*>(&object)->my_graph;
    }

    template <typename T>
    static graph& get_impl(const T& object, std::false_type) {
        return object.graph_reference();
    }
};

template<typename Order, typename... Nodes>
struct node_set {
    typedef Order order_type;

    tbb::flow::tuple<Nodes&...> nodes;
    node_set(Nodes&... ns) : nodes(ns...) {}

    template <typename... Nodes2>
    node_set(const node_set<order::undefined, Nodes2...>& set) : nodes(set.nodes) {}

    graph& graph_reference() const {
        return get_graph_helper::get(std::get<0>(nodes));
    }
};

namespace alias_helpers {
template <typename T> using output_type = typename T::output_type;
template <typename T> using output_ports_type = typename T::output_ports_type;
template <typename T> using input_type = typename T::input_type;
template <typename T> using input_ports_type = typename T::input_ports_type;
} // namespace alias_helpers

template <typename T>
using has_output_type = tbb::internal::supports<T, alias_helpers::output_type>;

template <typename T>
using has_input_type = tbb::internal::supports<T, alias_helpers::input_type>;

template <typename T>
using has_input_ports_type = tbb::internal::supports<T, alias_helpers::input_ports_type>;

template <typename T>
using has_output_ports_type = tbb::internal::supports<T, alias_helpers::output_ports_type>;

template<typename T>
struct is_sender : std::is_base_of<sender<typename T::output_type>, T> {};

template<typename T>
struct is_receiver : std::is_base_of<receiver<typename T::input_type>, T> {};

template <typename Node>
struct is_async_node : std::false_type {};

template <typename... Args>
struct is_async_node<tbb::flow::interface11::async_node<Args...>> : std::true_type {};

template<typename FirstPredecessor, typename... Predecessors>
node_set<order::following, FirstPredecessor, Predecessors...>
follows(FirstPredecessor& first_predecessor, Predecessors&... predecessors) {
    __TBB_STATIC_ASSERT((tbb::internal::conjunction<has_output_type<FirstPredecessor>,
                                                   has_output_type<Predecessors>...>::value),
                        "Not all node's predecessors has output_type typedef");
    __TBB_STATIC_ASSERT((tbb::internal::conjunction<is_sender<FirstPredecessor>, is_sender<Predecessors>...>::value),
                        "Not all node's predecessors are senders");
    return node_set<order::following, FirstPredecessor, Predecessors...>(first_predecessor, predecessors...);
}

template<typename... Predecessors>
node_set<order::following, Predecessors...>
follows(node_set<order::undefined, Predecessors...>& predecessors_set) {
    __TBB_STATIC_ASSERT((tbb::internal::conjunction<has_output_type<Predecessors>...>::value),
                        "Not all nodes in the set has output_type typedef");
    __TBB_STATIC_ASSERT((tbb::internal::conjunction<is_sender<Predecessors>...>::value),
                        "Not all nodes in the set are senders");
    return node_set<order::following, Predecessors...>(predecessors_set);
}

template<typename FirstSuccessor, typename... Successors>
node_set<order::preceding, FirstSuccessor, Successors...>
precedes(FirstSuccessor& first_successor, Successors&... successors) {
    __TBB_STATIC_ASSERT((tbb::internal::conjunction<has_input_type<FirstSuccessor>,
                                                    has_input_type<Successors>...>::value),
                        "Not all node's successors has input_type typedef");
    __TBB_STATIC_ASSERT((tbb::internal::conjunction<is_receiver<FirstSuccessor>, is_receiver<Successors>...>::value),
                        "Not all node's successors are receivers");
    return node_set<order::preceding, FirstSuccessor, Successors...>(first_successor, successors...);
}

template<typename... Successors>
node_set<order::preceding, Successors...>
precedes(node_set<order::undefined, Successors...>& successors_set) {
    __TBB_STATIC_ASSERT((tbb::internal::conjunction<has_input_type<Successors>...>::value),
                        "Not all nodes in the set has input_type typedef");
    __TBB_STATIC_ASSERT((tbb::internal::conjunction<is_receiver<Successors>...>::value),
                        "Not all nodes in the set are receivers");
    return node_set<order::preceding, Successors...>(successors_set);
}

template <typename Node, typename... Nodes>
node_set<order::undefined, Node, Nodes...>
make_node_set(Node& first_node, Nodes&... nodes) {
    return node_set<order::undefined, Node, Nodes...>(first_node, nodes...);
}

template<size_t I>
class successor_selector {
    template <typename NodeType>
    static auto get_impl(NodeType& node, std::true_type) -> decltype(input_port<I>(node)) {
        return input_port<I>(node);
    }

    template <typename NodeType>
    static NodeType& get_impl(NodeType& node, std::false_type) { return node; }

public:
    template <typename NodeType>
#if __TBB_MSVC_DISABLE_TRAILING_DECLTYPE
    static auto& get(NodeType& node)
#else
    static auto get(NodeType& node) -> decltype(get_impl(node, has_input_ports_type<NodeType>()))
#endif
	{
        return get_impl(node, has_input_ports_type<NodeType>());
    }
};

template<size_t I>
class predecessor_selector {
    template <typename NodeType>
    static auto internal_get(NodeType& node, std::true_type) -> decltype(output_port<I>(node)) {
        return output_port<I>(node);
    }

    template <typename NodeType>
    static NodeType& internal_get(NodeType& node, std::false_type) { return node;}

    template <typename NodeType>
#if __TBB_MSVC_DISABLE_TRAILING_DECLTYPE
    static auto& get_impl(NodeType& node, std::false_type)
#else
    static auto get_impl(NodeType& node, std::false_type) -> decltype(internal_get(node, has_output_ports_type<NodeType>()))
#endif
	{
        return internal_get(node, has_output_ports_type<NodeType>());
    }

    template <typename AsyncNode>
    static AsyncNode& get_impl(AsyncNode& node, std::true_type) { return node; }

public:
    template <typename NodeType>
#if __TBB_MSVC_DISABLE_TRAILING_DECLTYPE
    static auto& get(NodeType& node)
#else
    static auto get(NodeType& node) -> decltype(get_impl(node, is_async_node<NodeType>()))
#endif
	{
        return get_impl(node, is_async_node<NodeType>());
    }
};

template<size_t I>
class make_edges_helper {
public:
    template<typename PredecessorsTuple, typename NodeType>
    static void connect_predecessors(PredecessorsTuple& predecessors, NodeType& node) {
        make_edge(std::get<I>(predecessors), successor_selector<I>::get(node));
        make_edges_helper<I - 1>::connect_predecessors(predecessors, node);
    }

    template<typename SuccessorsTuple, typename NodeType>
    static void connect_successors(NodeType& node, SuccessorsTuple& successors) {
        make_edge(predecessor_selector<I>::get(node), std::get<I>(successors));
        make_edges_helper<I - 1>::connect_successors(node, successors);
    }
};

template<>
struct make_edges_helper<0> {
    template<typename PredecessorsTuple, typename NodeType>
    static void connect_predecessors(PredecessorsTuple& predecessors, NodeType& node) {
        make_edge(std::get<0>(predecessors), successor_selector<0>::get(node));
    }

    template<typename SuccessorsTuple, typename NodeType>
    static void connect_successors(NodeType& node, SuccessorsTuple& successors) {
        make_edge(predecessor_selector<0>::get(node), std::get<0>(successors));
    }
};

// TODO: consider adding an overload for making edges between node sets
template<typename NodeType, typename OrderFlagType, typename... Args>
void make_edges(const node_set<OrderFlagType, Args...>& s, NodeType& node) {
    const std::size_t SetSize = tbb::flow::tuple_size<decltype(s.nodes)>::value;
    make_edges_helper<SetSize - 1>::connect_predecessors(s.nodes, node);
}

template <typename NodeType, typename OrderFlagType, typename... Args>
void make_edges(NodeType& node, const node_set<OrderFlagType, Args...>& s) {
    const std::size_t SetSize = tbb::flow::tuple_size<decltype(s.nodes)>::value;
    make_edges_helper<SetSize - 1>::connect_successors(node, s.nodes);
}

template <typename NodeType, typename... Nodes>
void make_edges_in_order(const node_set<order::following, Nodes...>& ns, NodeType& node) {
    make_edges(ns, node);
}

template <typename NodeType, typename... Nodes>
void make_edges_in_order(const node_set<order::preceding, Nodes...>& ns, NodeType& node) {
    make_edges(node, ns);
}

#endif // __TBB_CPP11_PRESENT

} // namespace internal

#endif // __TBB_flow_graph_node_set_impl_H
