// From https://github.com/haasdo95/graphlite commit 9cd2815e5d571a87e5b8b8ef3752e04d971f35d4
#ifndef GSK_GRAPH_LITE_H
#define GSK_GRAPH_LITE_H

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// container spec
namespace graph_lite {
// ContainerGen, supposed to be container of neighbors
enum class Container { VEC, LIST, SET, UNORDERED_SET, MULTISET, UNORDERED_MULTISET };

// self loop permission
enum class SelfLoop { ALLOWED, DISALLOWED };

// multi-edge permission
enum class MultiEdge { ALLOWED, DISALLOWED };

// directed or undirected graph
enum class EdgeDirection { DIRECTED, UNDIRECTED };

// map for adj list
enum class Map { MAP, UNORDERED_MAP };
}

// type manipulation
namespace graph_lite::detail {
template <typename T>
constexpr bool is_vector_v =
    std::is_same_v<T, std::vector<typename T::value_type, typename T::allocator_type>>;

template <typename T>
constexpr bool is_list_v =
    std::is_same_v<T, std::vector<typename T::value_type, typename T::allocator_type>>;

// determine if type is map or unordered_map
template <typename T, typename U = void>
struct is_map : std::false_type {};
template <typename T>
struct is_map<T, std::void_t<typename T::key_type, typename T::mapped_type, typename T::key_compare,
                             typename T::allocator_type>> {
  static constexpr bool value =
      std::is_same_v<T, std::map<typename T::key_type, typename T::mapped_type,
                                 typename T::key_compare, typename T::allocator_type>>;
};
template <typename T>
constexpr bool is_map_v = is_map<T>::value;

template <typename T, typename U = void>
struct is_unordered_map : std::false_type {};
template <typename T>
struct is_unordered_map<
    T, std::void_t<typename T::key_type, typename T::mapped_type, typename T::hasher,
                   typename T::key_equal, typename T::allocator_type>> {
  static constexpr bool value = std::is_same_v<
      T, std::unordered_map<typename T::key_type, typename T::mapped_type, typename T::hasher,
                            typename T::key_equal, typename T::allocator_type>>;
};
template <typename T>
constexpr bool is_unordered_map_v = is_unordered_map<T>::value;
template <typename T>
constexpr bool is_either_map_v = is_map_v<T> || is_unordered_map_v<T>;

// CREDIT: https://stackoverflow.com/questions/765148/how-to-remove-constness-of-const-iterator
template <typename ContainerType, typename ConstIterator>
typename ContainerType::iterator const_iter_to_iter(ContainerType& c, ConstIterator it) {
  return c.erase(it, it);
}
// shorthand for turning const T& into T
template <typename T>
struct remove_cv_ref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};
template <typename T>
using remove_cv_ref_t = typename remove_cv_ref<T>::type;
// END OF shorthand for turning const T& into T

// test if lhs==rhs and lhs!=rhs work
template <typename T, typename = std::void_t<>>
struct is_eq_comparable : std::false_type {};
template <typename T>
struct is_eq_comparable<T, std::void_t<decltype(std::declval<T>() == std::declval<T>())>>
    : std::true_type {};
template <typename T>
constexpr bool is_eq_comparable_v = is_eq_comparable<T>::value;

// END OF test if lhs==rhs work

// test comparability; see if a < b works
template <typename T, typename = std::void_t<>>
struct is_comparable : std::false_type {};
template <typename T>
struct is_comparable<T, std::void_t<decltype(std::declval<T>() < std::declval<T>())>>
    : std::true_type {};
template <typename T>
constexpr bool is_comparable_v = is_comparable<T>::value;
// END OF test comparability

// test streamability
template <typename T, typename = std::void_t<>>
struct is_streamable : std::false_type {};
template <typename T>
struct is_streamable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>>
    : std::true_type {};
template <typename T>
constexpr bool is_streamable_v = is_streamable<T>::value;
// END OF test streamability

// test hashability
template <typename T, typename = std::void_t<>>
struct is_std_hashable : std::false_type {};
template <typename T>
struct is_std_hashable<T, std::void_t<decltype(std::declval<std::hash<T>>()(std::declval<T>()))>>
    : std::true_type {};
template <typename T>
constexpr bool is_std_hashable_v = is_std_hashable<T>::value;
// END OF test hashability

template <Container C>
struct MultiEdgeTraits {};
template <>
struct MultiEdgeTraits<Container::VEC> {
  static constexpr MultiEdge value = MultiEdge::ALLOWED;
};
template <>
struct MultiEdgeTraits<Container::LIST> {
  static constexpr MultiEdge value = MultiEdge::ALLOWED;
};
template <>
struct MultiEdgeTraits<Container::MULTISET> {
  static constexpr MultiEdge value = MultiEdge::ALLOWED;
};
template <>
struct MultiEdgeTraits<Container::UNORDERED_MULTISET> {
  static constexpr MultiEdge value = MultiEdge::ALLOWED;
};
template <>
struct MultiEdgeTraits<Container::SET> {
  static constexpr MultiEdge value = MultiEdge::DISALLOWED;
};
template <>
struct MultiEdgeTraits<Container::UNORDERED_SET> {
  static constexpr MultiEdge value = MultiEdge::DISALLOWED;
};

template <typename T>
struct OutIn {
  T out;
  T in;
};
}

// operation on containers
namespace graph_lite::detail::container {
template <typename ContainerType, typename ValueType,
          typename = std::enable_if_t<std::is_convertible_v<remove_cv_ref_t<ValueType>,
                                                            typename ContainerType::value_type>>>
auto insert(ContainerType& c, ValueType&& v) {
  return c.insert(c.cend(), std::forward<ValueType>(v));
}

// find the first occurrence of a value
// using different find for efficiency; std::find is always linear
template <typename ContainerType, typename ValueType,
          std::enable_if_t<!is_vector_v<ContainerType> && !is_list_v<ContainerType>, bool> = true>
auto find(ContainerType& c, const ValueType& v) {
  return c.find(v);
}

template <typename ValueType, typename T>
auto find(std::vector<std::pair<ValueType, T>>& c, const ValueType& v) {
  return std::find_if(c.begin(), c.end(), [&v](const auto& p) { return p.first == v; });
}
template <typename ValueType, typename T>
auto find(std::list<std::pair<ValueType, T>>& c, const ValueType& v) {
  return std::find_if(c.begin(), c.end(), [&v](const auto& p) { return p.first == v; });
}

template <typename ValueType, typename T>
auto find(const std::vector<std::pair<ValueType, T>>& c, const ValueType& v) {
  return std::find_if(c.begin(), c.end(), [&v](const auto& p) { return p.first == v; });
}
template <typename ValueType, typename T>
auto find(const std::list<std::pair<ValueType, T>>& c, const ValueType& v) {
  return std::find_if(c.begin(), c.end(), [&v](const auto& p) { return p.first == v; });
}

template <typename ValueType, typename FullValueType,
          std::enable_if_t<std::is_convertible_v<ValueType, FullValueType>, bool> = true>
auto find(std::vector<FullValueType>& c, const ValueType& v) {
  return std::find(c.begin(), c.end(), v);
}

template <typename ValueType, typename FullValueType,
          std::enable_if_t<std::is_convertible_v<ValueType, FullValueType>, bool> = true>
auto find(std::list<FullValueType>& c, const ValueType& v) {
  return std::find(c.begin(), c.end(), v);
}

template <typename ValueType, typename FullValueType,
          std::enable_if_t<std::is_convertible_v<ValueType, FullValueType>, bool> = true>
auto find(const std::vector<FullValueType>& c, const ValueType& v) {
  return std::find(c.begin(), c.end(), v);
}

template <typename ValueType, typename FullValueType,
          std::enable_if_t<std::is_convertible_v<ValueType, FullValueType>, bool> = true>
auto find(const std::list<FullValueType>& c, const ValueType& v) {
  return std::find(c.begin(), c.end(), v);
}
// END OF find the first occurrence of a value

// count the occurrence of a value
template <typename ContainerType, typename ValueType>
std::enable_if_t<!is_vector_v<ContainerType> && !is_list_v<ContainerType>, int> count(
    const ContainerType& c, const ValueType& v) {
  return c.count(v);
}

template <typename ValueType, typename T>
int count(const std::vector<std::pair<ValueType, T>>& c, const ValueType& v) {
  return std::count_if(c.begin(), c.end(), [&v](const auto& e) { return e.first == v; });
}

template <typename ValueType, typename T>
int count(const std::list<std::pair<ValueType, T>>& c, const ValueType& v) {
  return std::count_if(c.begin(), c.end(), [&v](const auto& e) { return e.first == v; });
}

template <typename ValueType, typename FullValueType>
std::enable_if_t<std::is_convertible_v<ValueType, FullValueType>, int> count(
    const std::vector<FullValueType>& c, const ValueType& v) {
  return std::count(c.begin(), c.end(), v);
}

template <typename ValueType, typename FullValueType>
std::enable_if_t<std::is_convertible_v<ValueType, FullValueType>, int> count(
    const std::list<FullValueType>& c, const ValueType& v) {
  return std::count(c.begin(), c.end(), v);
}
// END OF count the occurrence of a value

// remove all by value
// erase always erases all with value v; again, std::remove is linear
template <typename ContainerType, typename ValueType>
std::enable_if_t<!is_vector_v<ContainerType> && !is_list_v<ContainerType>, int> erase_all(
    ContainerType& c, const ValueType& v) {
  static_assert(std::is_same_v<ContainerType, std::remove_const_t<ContainerType>>);
  return c.erase(v);
}

template <typename ValueType, typename FullValueType>
std::enable_if_t<std::is_convertible_v<ValueType, FullValueType>, int> erase_all(
    std::vector<FullValueType>& c, const ValueType& v) {
  size_t old_size = c.size();
  c.erase(std::remove(c.begin(), c.end(), v), c.end());
  return old_size - c.size();
}

template <typename ValueType, typename T>
int erase_all(std::vector<std::pair<ValueType, T>>& c, const ValueType& v) {
  size_t old_size = c.size();
  c.erase(std::remove_if(c.begin(), c.end(), [&v](const auto& p) { return p.first == v; }),
          c.end());
  return old_size - c.size();
}

template <typename ValueType, typename FullValueType>
std::enable_if_t<std::is_convertible_v<ValueType, FullValueType>, int> erase_all(
    std::list<FullValueType>& c, const ValueType& v) {
  size_t old_size = c.size();
  c.remove(v);
  return old_size - c.size();
}

template <typename ValueType, typename T>
int erase_all(std::list<std::pair<ValueType, T>>& c, const ValueType& v) {
  size_t old_size = c.size();
  c.remove_if([&v](const auto& p) { return p.first == v; });
  return old_size - c.size();
}
// END OF remove all by value

// remove one by value or position; remove AT MOST 1 element
template <typename ContainerType, typename ValueType>
int erase_one(ContainerType& c, const ValueType& v) {
  if constexpr (std::is_same_v<remove_cv_ref_t<ValueType>, typename ContainerType::iterator> ||
                std::is_same_v<remove_cv_ref_t<ValueType>,
                               typename ContainerType::const_iterator>) {  // remove by pos
    c.erase(v);
    return 1;
  } else {  // remove by value
    auto pos = find(c, v);
    if (pos == c.end()) {
      return 0;
    }
    c.erase(pos);
    return 1;
  }
}
// END OF remove one
}

// mixin base classes of Graph
namespace graph_lite::detail {
// EdgePropListBase provides optional member variable edge_prop_list
template <typename EPT>
struct EdgePropListBase {
 protected:
  std::list<EPT> edge_prop_list;
};
template <>
struct EdgePropListBase<void> {};  // empty base optimization if edge prop is not needed

// EdgeDirectionBase provides different API for directed and undirected graphs
template <typename GType, EdgeDirection direction>
struct EdgeDirectionBase {};

template <typename GType>
struct EdgeDirectionBase<GType, EdgeDirection::UNDIRECTED> {  // undirected graph only
  template <typename T>
  auto neighbors(const T& node_iv) const {
    const auto* self = static_cast<const GType*>(this);
    return self->template get_neighbors_helper<true>(node_iv);
  }
  template <typename T>
  auto neighbors(const T& node_iv) {
    auto* self = static_cast<GType*>(this);
    return self->template get_neighbors_helper<true>(node_iv);
  }
  // returns the number of neighbors of a node
  template <typename T>
  int count_neighbors(const T& node_iv) const {
    const auto* self = static_cast<const GType*>(this);
    return self->template count_neighbors_helper<true>(node_iv);
  }

  // find a node with value tgt within the neighborhood of src
  template <typename U, typename V>
  auto find_neighbor(const U& src_iv, V&& tgt_identifier) const {
    const auto* self = static_cast<const GType*>(this);
    return self->template find_neighbor_helper<true>(src_iv, std::forward<V>(tgt_identifier));
  }
  template <typename U, typename V>
  auto find_neighbor(const U& src_iv, V&& tgt_identifier) {  // non-const overload
    auto* self = static_cast<GType*>(this);
    return self->template find_neighbor_helper<true>(src_iv, std::forward<V>(tgt_identifier));
  }
};

template <typename GType>
struct EdgeDirectionBase<GType, EdgeDirection::DIRECTED> {  // directed graph only
  template <typename T>
  auto out_neighbors(const T& node_iv) const {
    const auto* self = static_cast<const GType*>(this);
    return self->template get_neighbors_helper<true>(node_iv);
  }
  template <typename T>
  auto out_neighbors(const T& node_iv) {
    auto* self = static_cast<GType*>(this);
    return self->template get_neighbors_helper<true>(node_iv);
  }

  template <typename T>
  auto in_neighbors(const T& node_iv) const {
    const auto* self = static_cast<const GType*>(this);
    return self->template get_neighbors_helper<false>(node_iv);
  }
  template <typename T>
  auto in_neighbors(const T& node_iv) {
    auto* self = static_cast<GType*>(this);
    return self->template get_neighbors_helper<false>(node_iv);
  }

  // returns the number of out neighbors of a node
  template <typename T>
  int count_out_neighbors(const T& node_iv) const {
    const auto* self = static_cast<const GType*>(this);
    return self->template count_neighbors_helper<true>(node_iv);
  }

  // returns the number of out neighbors of a node
  template <typename T>
  int count_in_neighbors(const T& node_iv) const {
    const auto* self = static_cast<const GType*>(this);
    return self->template count_neighbors_helper<false>(node_iv);
  }

  // find a node with value tgt within the out-neighborhood of src
  template <typename U, typename V>
  auto find_out_neighbor(const U& src_iv, V&& tgt_identifier) const {
    const auto* self = static_cast<const GType*>(this);
    return self->template find_neighbor_helper<true>(src_iv, std::forward<V>(tgt_identifier));
  }
  template <typename U, typename V>
  auto find_out_neighbor(const U& src_iv, V&& tgt_identifier) {  // non-const overload
    auto* self = static_cast<GType*>(this);
    return self->template find_neighbor_helper<true>(src_iv, std::forward<V>(tgt_identifier));
  }

  // find a node with value tgt within the in-neighborhood of src
  template <typename U, typename V>
  auto find_in_neighbor(const U& src_iv, V&& tgt_identifier) const {
    const auto* self = static_cast<const GType*>(this);
    return self->template find_neighbor_helper<false>(src_iv, std::forward<V>(tgt_identifier));
  }
  template <typename U, typename V>
  auto find_in_neighbor(const U& src_iv, V&& tgt_identifier) {  // non-const overload
    auto* self = static_cast<GType*>(this);
    return self->template find_neighbor_helper<false>(src_iv, std::forward<V>(tgt_identifier));
  }
};

// NodePropGraphBase provides different API depending on whether node prop is needed
template <typename GType, typename NodePropType>
struct NodePropGraphBase {
 public:  // this can seg fault if node_identifier is invalid...
  template <typename T>
  const NodePropType& node_prop(const T& node_iv) const {
    const auto* self = static_cast<const GType*>(this);
    auto pos = self->find_by_iter_or_by_value(node_iv);
    return pos->second.prop;
  }
  template <typename T>
  NodePropType& node_prop(const T& node_iv) {
    return const_cast<NodePropType&>(
        static_cast<const NodePropGraphBase*>(this)->node_prop(node_iv));
  }

  template <typename NT, typename... NPT>
  int add_node_with_prop(NT&& new_node, NPT&&... prop) noexcept {
    static_assert(std::is_same_v<remove_cv_ref_t<NT>, typename GType::node_type>);
    static_assert(std::is_constructible_v<NodePropType, NPT...>);
    auto* self = static_cast<GType*>(this);
    if (!self->adj_list.count(new_node)) {  // insert if not already existing
      // this should invoke(in-place) the constructor of PropNode
      self->adj_list.emplace(std::piecewise_construct,
                             std::forward_as_tuple(std::forward<NT>(new_node)),
                             std::forward_as_tuple(std::forward<NPT>(prop)...));
      return 1;
    }
    return 0;  // a no-op if already existing
  }
};
template <typename GType>
struct NodePropGraphBase<GType, void> {  // no node prop
  template <typename T>
  int add_nodes(T&& new_node) noexcept {  // base case
    static_assert(std::is_same_v<detail::remove_cv_ref_t<T>, typename GType::node_type>);
    auto* self = static_cast<GType*>(this);
    int old_size = self->adj_list.size();
    self->adj_list[std::forward<T>(new_node)];  // insertion here; no-op if already existing
    return self->adj_list.size() - old_size;
  }
  template <typename T, typename... Args>
  int add_nodes(T&& new_node, Args&&... args) noexcept {
    static_assert(std::is_same_v<detail::remove_cv_ref_t<T>, typename GType::node_type>);
    return add_nodes(std::forward<T>(new_node)) + add_nodes(std::forward<Args>(args)...);
  }
};

// EdgePropGraphBase provides method "add_edge" when edge prop is not needed;
// and "add_edge_with_prop" when edge prop is needed
template <typename GType, typename EdgePropType>
struct EdgePropGraphBase {
  // extract edge property given node pair
  template <typename U, typename V>
  EdgePropType& edge_prop(U&& source_iv, V&& target_iv) {
    return const_cast<EdgePropType&>(
        static_cast<const EdgePropGraphBase*>(this)->template edge_prop(
            std::forward<U>(source_iv), std::forward<V>(target_iv)));
  }

  template <typename U, typename V>
  const EdgePropType& edge_prop(U&& source_iv, V&& target_iv) const {
    const auto* self = static_cast<const GType*>(this);
    const char* src_not_found_msg = "source is not found";
    const char* tgt_not_found_msg = "target is not found";
    auto find_neighbor_iv = [self, &source_iv, &target_iv]() {
      auto&& tgt_val = self->unwrap_by_iter_or_by_value(std::forward<V>(target_iv));
      if constexpr (GType::DIRECTION == EdgeDirection::UNDIRECTED) {
        return self->find_neighbor(source_iv, std::forward<decltype(tgt_val)>(tgt_val));
      } else {
        return self->find_out_neighbor(source_iv, std::forward<decltype(tgt_val)>(tgt_val));
      }
    };
    auto find_neighbor_vi = [self, &source_iv, &target_iv]() {
      auto&& src_val = self->unwrap_by_iter_or_by_value(std::forward<U>(source_iv));
      if constexpr (GType::DIRECTION == EdgeDirection::UNDIRECTED) {
        return self->find_neighbor(target_iv, std::forward<decltype(src_val)>(src_val));
      } else {
        return self->find_in_neighbor(target_iv, std::forward<decltype(src_val)>(src_val));
      }
    };
    // this ensures that no adj list lookup will happen if either is an iterator
    if constexpr (GType::template is_iterator<U>()) {  // I & V or I & I
      auto [found, pos] = find_neighbor_iv();
      if (!found) {
        throw std::runtime_error(tgt_not_found_msg);
      }
      return pos->second.prop();
    } else {
      // V & I or V & V
      auto [found, pos] = find_neighbor_vi();
      if (!found) {
        throw std::runtime_error(src_not_found_msg);
      }
      return pos->second.prop();
    }
  }

  template <typename U, typename V, typename... EPT>
  int add_edge_with_prop(U&& source_iv, V&& target_iv, EPT&&... prop) noexcept {
    static_assert(std::is_constructible_v<EdgePropType, EPT...>);
    auto* self = static_cast<GType*>(this);
    auto src_pos = self->find_by_iter_or_by_value(source_iv);
    auto tgt_pos = self->find_by_iter_or_by_value(target_iv);
    if (src_pos == self->adj_list.end() || tgt_pos == self->adj_list.end()) {
      if (src_pos == self->adj_list.end()) {
        self->print_by_iter_or_by_value(std::cerr << "(add_edge) edge involves non-existent source",
                                        source_iv)
            << "\n";
      }
      if (tgt_pos == self->adj_list.end()) {
        self->print_by_iter_or_by_value(std::cerr << "(add_edge) edge involves non-existent target",
                                        target_iv)
            << "\n";
      }
      return 0;
    }
    const typename GType::node_type& src_full = src_pos->first;
    const typename GType::node_type& tgt_full = tgt_pos->first;  // flesh out src and tgt
    if (self->check_edge_dup(src_pos, src_full, tgt_full)) {
      return 0;
    }
    if (self->check_self_loop(src_pos, tgt_pos, src_full)) {
      return 0;
    }
    auto prop_pos = self->insert_edge_prop(std::forward<EPT>(prop)...);
    container::insert(self->get_out_neighbors(src_pos), std::make_pair(tgt_full, prop_pos));
    if (src_pos != tgt_pos || GType::DIRECTION == EdgeDirection::DIRECTED) {
      container::insert(self->get_in_neighbors(tgt_pos), std::make_pair(src_full, prop_pos));
    }
    ++self->num_of_edges;
    return 1;
  }
};
template <typename GType>
struct EdgePropGraphBase<GType, void> {
  template <typename U, typename V>
  int add_edge(U&& source_iv, V&& target_iv) noexcept {
    auto* self = static_cast<GType*>(this);
    auto src_pos = self->find_by_iter_or_by_value(source_iv);
    auto tgt_pos = self->find_by_iter_or_by_value(target_iv);
    if (src_pos == self->adj_list.end() || tgt_pos == self->adj_list.end()) {
      if (src_pos == self->adj_list.end()) {
        self->print_by_iter_or_by_value(std::cerr << "(add_edge) edge involves non-existent source",
                                        source_iv)
            << "\n";
      }
      if (tgt_pos == self->adj_list.end()) {
        self->print_by_iter_or_by_value(std::cerr << "(add_edge) edge involves non-existent target",
                                        target_iv)
            << "\n";
      }
      return 0;
    }
    const typename GType::node_type& src_full = src_pos->first;
    const typename GType::node_type& tgt_full = tgt_pos->first;  // flesh out src and tgt
    if (self->check_edge_dup(src_pos, src_full, tgt_full)) {
      return 0;
    }
    if (self->check_self_loop(src_pos, tgt_pos, src_full)) {
      return 0;
    }
    container::insert(self->get_out_neighbors(src_pos), tgt_full);
    if (src_pos != tgt_pos || GType::DIRECTION == EdgeDirection::DIRECTED) {
      container::insert(self->get_in_neighbors(tgt_pos), src_full);
    }
    ++self->num_of_edges;
    return 1;
  }
};
}

// Graph class
namespace graph_lite {
template <typename NodeType = int, typename NodePropType = void, typename EdgePropType = void,
          EdgeDirection direction = EdgeDirection::UNDIRECTED,
          MultiEdge multi_edge = MultiEdge::DISALLOWED, SelfLoop self_loop = SelfLoop::DISALLOWED,
          Map adj_list_spec = Map::UNORDERED_MAP,
          Container neighbors_container_spec = Container::UNORDERED_SET>
class Graph : private detail::EdgePropListBase<EdgePropType>,
              public detail::EdgeDirectionBase<
                  Graph<NodeType, NodePropType, EdgePropType, direction, multi_edge, self_loop,
                        adj_list_spec, neighbors_container_spec>,
                  direction>,
              public detail::NodePropGraphBase<
                  Graph<NodeType, NodePropType, EdgePropType, direction, multi_edge, self_loop,
                        adj_list_spec, neighbors_container_spec>,
                  NodePropType>,
              public detail::EdgePropGraphBase<
                  Graph<NodeType, NodePropType, EdgePropType, direction, multi_edge, self_loop,
                        adj_list_spec, neighbors_container_spec>,
                  EdgePropType> {
  // friend class with CRTP base classes
  friend detail::EdgeDirectionBase<
      Graph<NodeType, NodePropType, EdgePropType, direction, multi_edge, self_loop, adj_list_spec,
            neighbors_container_spec>,
      direction>;
  friend detail::NodePropGraphBase<
      Graph<NodeType, NodePropType, EdgePropType, direction, multi_edge, self_loop, adj_list_spec,
            neighbors_container_spec>,
      NodePropType>;
  friend detail::EdgePropGraphBase<
      Graph<NodeType, NodePropType, EdgePropType, direction, multi_edge, self_loop, adj_list_spec,
            neighbors_container_spec>,
      EdgePropType>;
  static_assert(std::is_same_v<NodeType, std::remove_reference_t<NodeType>>,
                "NodeType should not be a reference");
  static_assert(std::is_same_v<NodeType, std::remove_cv_t<NodeType>>,
                "NodeType should not be cv-qualified");
  static_assert(std::is_same_v<NodePropType, std::remove_reference_t<NodePropType>> &&
                    std::is_same_v<EdgePropType, std::remove_reference_t<EdgePropType>>,
                "Property types should not be references");
  static_assert(std::is_same_v<NodePropType, std::remove_cv_t<NodePropType>> &&
                    std::is_same_v<EdgePropType, std::remove_cv_t<EdgePropType>>,
                "Property types should not be cv-qualified");
  static_assert(detail::is_eq_comparable_v<NodeType>,
                "NodeType does not support ==; implement operator==");
  static_assert(detail::is_streamable_v<NodeType>,
                "NodeType is not streamable; implement operator<<");
  static_assert(!((neighbors_container_spec == Container::UNORDERED_SET ||
                     neighbors_container_spec == Container::UNORDERED_MULTISET ||
                     adj_list_spec == Map::UNORDERED_MAP) &&
                    !detail::is_std_hashable_v<NodeType>),
                "NodeType is not hashable");
  static_assert(!((neighbors_container_spec == Container::SET ||
                     neighbors_container_spec == Container::MULTISET ||
                     adj_list_spec == Map::MAP) &&
                    !detail::is_comparable_v<NodeType>),
                "NodeType does not support operator <");
  static_assert(!(detail::MultiEdgeTraits<neighbors_container_spec>::value ==
                        MultiEdge::DISALLOWED &&
                    multi_edge == MultiEdge::ALLOWED),
                "node container does not support multi-edge");
  static_assert(!((neighbors_container_spec == Container::MULTISET ||
                     neighbors_container_spec == Container::UNORDERED_MULTISET) &&
                    multi_edge == MultiEdge::DISALLOWED),
                "disallowing multi-edge yet still using multi-set; use set/unordered_set instead");

 public:  // exposed types and constants
  using node_type = NodeType;
  using node_prop_type = NodePropType;
  using edge_prop_type = EdgePropType;
  static constexpr EdgeDirection DIRECTION = direction;
  static constexpr MultiEdge MULTI_EDGE = multi_edge;
  static constexpr SelfLoop SELF_LOOP = self_loop;
  static constexpr Map ADJ_LIST_SPEC = adj_list_spec;
  static constexpr Container NEIGHBORS_CONTAINER_SPEC = neighbors_container_spec;

 private:  // type gymnastics
  // handle neighbors that may have property
  // PairIterator is useful only when (1) the container is VEC or LIST and (2) edge prop is needed
  template <typename ContainerType>
  class PairIterator {  // always non const, since the build-in iter can handle const
    friend class Graph;

   private:
    using It = typename ContainerType::iterator;
    using ConstIt = typename ContainerType::const_iterator;
    It it;
    using VT = typename It::value_type;
    using FirstType = typename VT::first_type;
    using SecondType = typename VT::second_type;

   public:  // mimic the iter of a std::map
    using difference_type = typename It::difference_type;
    using value_type = std::pair<const FirstType, SecondType>;
    using reference = std::pair<const FirstType&, SecondType&>;
    using pointer = std::pair<const FirstType, SecondType>*;
    using iterator_category = std::bidirectional_iterator_tag;
    PairIterator() = default;
    // can be implicitly converted FROM a non-const iter
    PairIterator(const It& it) : it{it} {}
    // can be implicitly converted TO a const iter
    // relying on the imp conv of the underlying iter
    operator ConstIt() { return it; }

    friend bool operator==(const PairIterator& lhs, const PairIterator& rhs) {
      return lhs.it == rhs.it;
    }
    friend bool operator!=(const PairIterator& lhs, const PairIterator& rhs) {
      return lhs.it != rhs.it;
    }
    friend bool operator==(const PairIterator& lhs, const ConstIt& rhs) { return lhs.it == rhs; }
    friend bool operator!=(const PairIterator& lhs, const ConstIt& rhs) { return lhs.it != rhs; }
    // symmetry
    friend bool operator==(const ConstIt& lhs, const PairIterator& rhs) { return rhs == lhs; }
    friend bool operator!=(const ConstIt& lhs, const PairIterator& rhs) { return rhs != lhs; }

    reference operator*() const { return {std::cref(it->first), std::ref(it->second)}; }
    pointer operator->() const {
      std::pair<FirstType, SecondType>* ptr = it.operator->();
      using CVT = std::pair<const FirstType, SecondType>;
      static_assert(offsetof(VT, first) == offsetof(CVT, first) &&
                    offsetof(VT, second) == offsetof(CVT, second));
      return static_cast<pointer>(static_cast<void*>(ptr));  // adding constness to first
    }

    PairIterator& operator++() {  // prefix
      ++it;
      return *this;
    }
    PairIterator& operator--() {  // prefix
      --it;
      return *this;
    }
    PairIterator operator++(int) & {  // postfix
      PairIterator tmp = *this;
      ++(*this);
      return tmp;
    }
    PairIterator operator--(int) & {  // postfix
      PairIterator tmp = *this;
      --(*this);
      return tmp;
    }
  };

  template <typename EPT>
  struct EdgePropIterWrap {
    friend class Graph;

   private:
    using Iter = typename std::list<EPT>::iterator;
    // list iterators are NOT invalidated by insertion/removal(of others), making this possible
    Iter pos;

   public:
    EdgePropIterWrap() = default;
    explicit EdgePropIterWrap(const Iter& pos) : pos{pos} {}
    const EPT& prop() const { return *(this->pos); }
    EPT& prop() { return *(this->pos); }
  };

  using NeighborType = std::conditional_t<std::is_void_v<EdgePropType>, NodeType,
                                          std::pair<NodeType, EdgePropIterWrap<EdgePropType>>>;
  // type of neighbors container; the typename NT is here only because explicit spec is not allowed
  // in a class...
  template <Container C, typename NT, typename EPT>
  struct ContainerGen {};
  template <typename NT, typename EPT>
  struct ContainerGen<Container::LIST, NT, EPT> {
    using type = std::list<NeighborType>;
  };
  template <typename NT, typename EPT>
  struct ContainerGen<Container::VEC, NT, EPT> {
    using type = std::vector<NeighborType>;
  };
  template <typename NT, typename EPT>
  struct ContainerGen<Container::SET, NT, EPT> {
    using type = std::map<NT, EdgePropIterWrap<EPT>>;
  };
  template <typename NT>
  struct ContainerGen<Container::SET, NT, void> {
    using type = std::set<NT>;
  };
  template <typename NT, typename EPT>
  struct ContainerGen<Container::MULTISET, NT, EPT> {
    using type = std::multimap<NT, EdgePropIterWrap<EPT>>;
  };
  template <typename NT>
  struct ContainerGen<Container::MULTISET, NT, void> {
    using type = std::multiset<NT>;
  };
  template <typename NT, typename EPT>
  struct ContainerGen<Container::UNORDERED_SET, NT, EPT> {
    using type = std::unordered_map<NT, EdgePropIterWrap<EPT>>;
  };
  template <typename NT>
  struct ContainerGen<Container::UNORDERED_SET, NT, void> {
    using type = std::unordered_set<NT>;
  };
  template <typename NT, typename EPT>
  struct ContainerGen<Container::UNORDERED_MULTISET, NT, EPT> {
    using type = std::unordered_multimap<NT, EdgePropIterWrap<EPT>>;
  };
  template <typename NT>
  struct ContainerGen<Container::UNORDERED_MULTISET, NT, void> {
    using type = std::unordered_multiset<NT>;
  };

 public:
  using NeighborsContainerType =
      typename ContainerGen<neighbors_container_spec, NodeType, EdgePropType>::type;

 private:
  using NeighborsType =
      std::conditional_t<direction == EdgeDirection::UNDIRECTED, NeighborsContainerType,
                         detail::OutIn<NeighborsContainerType>>;
  struct PropNode {  // node with property
    NodePropType prop;
    NeighborsType neighbors;
    PropNode() = default;  // needed for map/unordered map
    template <typename... NPT>
    explicit PropNode(NPT&&... prop) : prop{std::forward<NPT>(prop)...}, neighbors{} {
      static_assert(std::is_constructible_v<NodePropType, NPT...>);
    }
  };
  static constexpr bool has_node_prop = ! std::is_void_v<NodePropType>;
  using AdjListValueType = std::conditional_t<!has_node_prop, NeighborsType, PropNode>;
  using AdjListType =
      std::conditional_t<adj_list_spec == Map::MAP, std::map<NodeType, AdjListValueType>,
                         std::unordered_map<NodeType, AdjListValueType>>;

 public:  // iterator types
  using NeighborsConstIterator = typename NeighborsContainerType::const_iterator;

 private:
  template <typename T>
  static constexpr bool can_construct_node =
      std::is_constructible_v<NodeType, detail::remove_cv_ref_t<T>>;
  static constexpr bool has_edge_prop = !std::is_void_v<EdgePropType>;
  static constexpr bool need_pair_iter =
      has_edge_prop &&
      (neighbors_container_spec == Container::VEC || neighbors_container_spec == Container::LIST);

 public:
  using NeighborsIterator = std::conditional_t<
      has_edge_prop,
      std::conditional_t<need_pair_iter,
                         PairIterator<NeighborsContainerType>,  // make node(aka first) immutable
                         typename NeighborsContainerType::iterator>,  // the iter for map/multi-map
                                                                      // works just fine
      NeighborsConstIterator>;  // if no edge prop is needed, always const
  static_assert(std::is_convertible_v<NeighborsIterator, NeighborsConstIterator>);
  using NeighborsView = std::pair<NeighborsIterator, NeighborsIterator>;
  using NeighborsConstView = std::pair<NeighborsConstIterator, NeighborsConstIterator>;

 private:
  using AdjListIterType = typename AdjListType::iterator;
  using AdjListConstIterType = typename AdjListType::const_iterator;

 private:
  int num_of_edges{};
  AdjListType adj_list;

 private:  // iterator support
  template <bool IsConst>
  class Iter {
   private:
    template <bool>
    friend class Iter;
    friend class Graph;
    using AdjIterT = std::conditional_t<IsConst, AdjListConstIterType, AdjListIterType>;
    AdjIterT it;

   public:
    Iter() = default;
    Iter(AdjIterT it) : it{it} {};

    // enables implicit conversion from non-const to const
    template <bool WasConst, typename = std::enable_if_t<IsConst || !WasConst>>
    Iter(const Iter<WasConst>& other) : it{other.it} {}

    Iter& operator++() {  // prefix
      ++it;
      return *this;
    }
    Iter operator++(int) & {  // postfix
      Iter tmp = *this;
      ++(*this);
      return tmp;
    }
    const NodeType& operator*() const { return it->first; }
    const NodeType* operator->() const { return &(it->first); }
    friend bool operator==(const Iter& lhs, const Iter& rhs) { return lhs.it == rhs.it; }
    friend bool operator!=(const Iter& lhs, const Iter& rhs) { return lhs.it != rhs.it; }
  };

 public:
  using Iterator = Iter<false>;
  using ConstIterator = Iter<true>;
  static_assert(std::is_convertible_v<Iterator, ConstIterator>);

  Iterator begin() noexcept { return Iter<false>(adj_list.begin()); }
  Iterator end() noexcept { return Iter<false>(adj_list.end()); }
  ConstIterator begin() const noexcept { return Iter<true>(adj_list.cbegin()); }
  ConstIterator end() const noexcept { return Iter<true>(adj_list.cend()); }
  // END OF iterator support
 private:  // neighbor access helpers
  const NeighborsContainerType& get_out_neighbors(AdjListConstIterType adj_iter) const {
    if constexpr (!has_node_prop) {
      if constexpr (direction == EdgeDirection::UNDIRECTED) {
        return adj_iter->second;
      } else {
        return adj_iter->second.out;
      }
    } else {
      if constexpr (direction == EdgeDirection::UNDIRECTED) {
        return adj_iter->second.neighbors;
      } else {
        return adj_iter->second.neighbors.out;
      }
    }
  }
  NeighborsContainerType& get_out_neighbors(AdjListIterType adj_iter) {
    return const_cast<NeighborsContainerType&>(
        static_cast<const Graph*>(this)->get_out_neighbors(adj_iter));
  }

  const NeighborsContainerType& get_in_neighbors(AdjListConstIterType adj_iter) const {
    if constexpr (!has_node_prop) {
      if constexpr (direction == EdgeDirection::UNDIRECTED) {
        return adj_iter->second;
      } else {
        return adj_iter->second.in;
      }
    } else {
      if constexpr (direction == EdgeDirection::UNDIRECTED) {
        return adj_iter->second.neighbors;
      } else {
        return adj_iter->second.neighbors.in;
      }
    }
  }
  NeighborsContainerType& get_in_neighbors(AdjListIterType adj_iter) {
    return const_cast<NeighborsContainerType&>(
        static_cast<const Graph*>(this)->get_in_neighbors(adj_iter));
  }

  // helpers for EdgeDirectionBase
  template <bool is_out, typename T>
  [[nodiscard]] int count_neighbors_helper(const T& node_iv) const {
    AdjListConstIterType pos = find_by_iter_or_by_value(node_iv);
    if (pos == adj_list.end()) {
      std::string msg = is_out ? "out" : "in";
      print_by_iter_or_by_value(
          std::cerr << "(count_neighbors) counting " << msg << "-neighbors of a non-existent node",
          node_iv)
          << "\n";
      throw std::runtime_error("counting " + msg + "-neighbors of a non-existent node");
    }
    if constexpr (is_out) {
      return get_out_neighbors(pos).size();
    } else {
      return get_in_neighbors(pos).size();
    }
  }

  template <bool is_out, typename T>
  NeighborsConstView get_neighbors_helper(const T& node_iv) const {
    AdjListConstIterType pos = find_by_iter_or_by_value(node_iv);
    if (pos == adj_list.end()) {
      std::string msg = is_out ? "out" : "in";
      print_by_iter_or_by_value(
          std::cerr << "(neighbors) finding " << msg << "-neighbors of a non-existent node",
          node_iv)
          << "\n";
      throw std::runtime_error("finding " + msg + "-neighbors of a non-existent node");
    }
    const NeighborsContainerType& neighbors = [ this, &pos ]() -> auto& {
      if constexpr (is_out) {
        return get_out_neighbors(pos);
      } else {
        return get_in_neighbors(pos);
      }
    }
    ();
    return {neighbors.begin(), neighbors.end()};
  }
  template <bool is_out, typename T>
  NeighborsView get_neighbors_helper(const T& node_iv) {
    AdjListIterType pos = find_by_iter_or_by_value(node_iv);
    if (pos == adj_list.end()) {
      std::string msg = is_out ? "out" : "in";
      print_by_iter_or_by_value(
          std::cerr << "(neighbors) finding " << msg << "-neighbors of a non-existent node",
          node_iv)
          << "\n";
      throw std::runtime_error("finding " + msg + "-neighbors of a non-existent node");
    }
    NeighborsContainerType& neighbors = [ this, &pos ]() -> auto& {
      if constexpr (is_out) {
        return get_out_neighbors(pos);
      } else {
        return get_in_neighbors(pos);
      }
    }
    ();
    return {neighbors.begin(), neighbors.end()};
  }
  // END OF helpers for EdgeDirectionBase

  const NodeType& get_neighbor_node(const NeighborsConstIterator& nbr_pos) const {
    if constexpr (has_edge_prop) {
      return nbr_pos->first;
    } else {
      return *nbr_pos;
    }
  }
  // END OF neighbor access helpers
 private:  // helpers for node search
  // find a node by value
  template <typename T>
  AdjListConstIterType find_node(const T& node_identifier) const {
    static_assert(can_construct_node<T>);
    if constexpr (std::is_convertible_v<T, NodeType>) {  // implicit conversion
      return adj_list.find(node_identifier);
    } else {  // conversion has to be explicit
      NodeType node{node_identifier};
      return adj_list.find(node);
    }
  }
  template <typename T>
  AdjListIterType find_node(const T& node_identifier) {
    return detail::const_iter_to_iter(adj_list,
                                      static_cast<const Graph*>(this)->find_node(node_identifier));
  }

  template <typename T>
  static constexpr bool is_iterator() {
    return std::is_same_v<ConstIterator, detail::remove_cv_ref_t<T>> ||
           std::is_same_v<Iterator, detail::remove_cv_ref_t<T>>;
  }

  template <typename T>
  decltype(auto) unwrap_by_iter_or_by_value(T&& iter_or_val) const {
    if constexpr (is_iterator<T>()) {
      return iter_or_val.it->first;
    } else {
      static_assert(can_construct_node<T>);
      return std::forward<T>(iter_or_val);  // simply pass along
    }
  }

  // either unwrap the iterator, or find the node in adj_list
  template <typename T>
  AdjListConstIterType find_by_iter_or_by_value(const T& iter_or_val) const {
    if constexpr (is_iterator<T>()) {  // by iter
      return iter_or_val.it;
    } else {  // by value
      return find_node(iter_or_val);
    }
  }
  template <typename T>
  AdjListIterType find_by_iter_or_by_value(const T& iter_or_val) {
    return detail::const_iter_to_iter(
        adj_list, static_cast<const Graph*>(this)->find_by_iter_or_by_value(iter_or_val));
  }
  // END OF either unwrap the iterator, or find the node in adj_list

  // helper method to provide better error messages
  template <typename T>
  std::ostream& print_by_iter_or_by_value(std::ostream& os, const T& iter_or_val) const {
    if constexpr (is_iterator<T>()) {  // by iter
      return os;                       // no-op if by iter
    } else {                           // by value
      static_assert(can_construct_node<T>);
      os << ": " << NodeType{iter_or_val};
      return os;
    }
  }

  // find tgt in the neighborhood of src; returns a pair {is_found, neighbor_iterator}
  template <bool is_out, typename U, typename V>
  std::pair<bool, NeighborsConstIterator> find_neighbor_helper(const U& src_iv,
                                                               V&& tgt_identifier) const {
    static_assert(can_construct_node<V>);
    AdjListConstIterType src_pos = find_by_iter_or_by_value(src_iv);
    if (src_pos == adj_list.end()) {
      print_by_iter_or_by_value(std::cerr << "(find_neighbor) source node not found", src_iv)
          << "\n";
      throw std::runtime_error{"source node is not found"};
    }
    const NeighborsContainerType& src_neighbors = [ this, &src_pos ]() -> auto& {
      if constexpr (is_out) {
        return get_out_neighbors(src_pos);
      } else {
        return get_in_neighbors(src_pos);
      }
    }
    ();
    if constexpr (std::is_same_v<NodeType, detail::remove_cv_ref_t<V>>) {
      NeighborsConstIterator tgt_pos = detail::container::find(src_neighbors, tgt_identifier);
      return {tgt_pos != src_neighbors.end(), tgt_pos};
    } else {
      NeighborsConstIterator tgt_pos =
          detail::container::find(src_neighbors, NodeType{std::forward<V>(tgt_identifier)});
      return {tgt_pos != src_neighbors.end(), tgt_pos};
    }
  }
  template <bool is_out, typename U, typename V>
  std::pair<bool, NeighborsIterator> find_neighbor_helper(const U& src_iv, V&& tgt_identifier) {
    static_assert(can_construct_node<V>);
    AdjListIterType src_pos = find_by_iter_or_by_value(src_iv);
    if (src_pos == adj_list.end()) {
      print_by_iter_or_by_value(std::cerr << "(find_neighbor) source node not found", src_iv)
          << "\n";
      throw std::runtime_error{"source node is not found"};
    }
    NeighborsContainerType& src_neighbors = [ this, &src_pos ]() -> auto& {
      if constexpr (is_out) {
        return get_out_neighbors(src_pos);
      } else {
        return get_in_neighbors(src_pos);
      }
    }
    ();
    if constexpr (std::is_same_v<NodeType, detail::remove_cv_ref_t<V>>) {
      auto tgt_pos = detail::container::find(src_neighbors, tgt_identifier);
      return {tgt_pos != src_neighbors.end(), tgt_pos};
    } else {
      auto tgt_pos =
          detail::container::find(src_neighbors, NodeType{std::forward<V>(tgt_identifier)});
      return {tgt_pos != src_neighbors.end(), tgt_pos};
    }
  }
  // END OF find tgt in the neighborhood of src; returns a pair {is_found, neighbor_iterator}
  // END OF helpers for node search
 public:  // simple queries
  [[nodiscard]] size_t size() const noexcept { return adj_list.size(); }

  [[nodiscard]] int num_edges() const noexcept { return num_of_edges; }

  template <typename T>
  bool has_node(const T& node_identifier) const noexcept {
    auto pos = find_node(node_identifier);
    return pos != adj_list.end();
  }

  // count the number of edges between src and tgt
  template <typename U, typename V>
  int count_edges(const U& source_iv, const V& target_iv) const noexcept {
    AdjListConstIterType src_pos = find_by_iter_or_by_value(source_iv);
    AdjListConstIterType tgt_pos = find_by_iter_or_by_value(target_iv);
    if (src_pos == adj_list.end() || tgt_pos == adj_list.end()) {
      if (src_pos == adj_list.end()) {
        print_by_iter_or_by_value(std::cerr << "(count_edges) source node not found", source_iv)
            << "\n";
      }
      if (tgt_pos == adj_list.end()) {
        print_by_iter_or_by_value(std::cerr << "(count_edges) target node not found", target_iv)
            << "\n";
      }
      return 0;
    }
    return detail::container::count(get_out_neighbors(src_pos), tgt_pos->first);
  }

  // find a node in the graph by value
  template <typename T>
  ConstIterator find(const T& node_identifier) const noexcept {
    AdjListConstIterType pos = find_node(node_identifier);
    return ConstIterator{pos};
  }
  template <typename T>
  Iterator find(const T& node_identifier) noexcept {
    AdjListIterType pos = find_node(node_identifier);
    return Iterator{pos};
  }

 private:  // edge addition helpers
  template <typename... EPT>
  auto insert_edge_prop(EPT&&... prop) {
    static_assert(has_edge_prop);
    static_assert(std::is_constructible_v<EdgePropType, EPT...>);
    this->edge_prop_list.emplace_back(std::forward<EPT>(prop)...);
    return EdgePropIterWrap<EdgePropType>{
        std::prev(this->edge_prop_list.end())};  // iterator to the last element
  }

  bool check_edge_dup(AdjListConstIterType src_pos, const NodeType& src_full,
                      const NodeType& tgt_full) const {
    if constexpr (multi_edge ==
                  MultiEdge::DISALLOWED) {  // check is needed only when we disallow dup
      // this catches multi-self-loop as well
      const NeighborsContainerType& neighbors = get_out_neighbors(src_pos);
      if (detail::container::find(neighbors, tgt_full) != neighbors.end()) {
        std::cerr << "(add_edge) re-adding existing edge: (" << src_full << ", " << tgt_full
                  << ")\n";
        return true;
      }
    }
    return false;
  }

  bool check_self_loop(AdjListIterType src_pos, AdjListIterType tgt_pos, const NodeType& src_full) {
    if constexpr (self_loop == SelfLoop::DISALLOWED) {
      if (src_pos == tgt_pos) {
        std::cerr << "(add_edge) adding self loop on node: " << src_full << "\n";
        return true;
      }
    }
    return false;
  }
  // END OF edge addition helpers
 private:  // edge removal helpers
  // because every edge has double entry, this method finds the correct double entry to remove
  NeighborsConstIterator find_tgt_remove_pos(AdjListIterType src_pos,
                                             NeighborsConstIterator src_remove_pos,
                                             NeighborsContainerType& tgt_neighbors) {
    if constexpr (has_edge_prop && multi_edge == MultiEdge::ALLOWED) {
      auto prop_address =
          &(src_remove_pos->second.prop());  // finding the corresponding double entry
      auto prop_finder = [&prop_address](const auto& tgt_nbr) {
        return prop_address == &(tgt_nbr.second.prop());
      };
      if constexpr (neighbors_container_spec == Container::VEC ||
                    neighbors_container_spec == Container::LIST) {
        // NeighborsContainerType is a vector/list of pairs
        // linearly search for the correct entry and remove
        return std::find_if(tgt_neighbors.begin(), tgt_neighbors.end(), prop_finder);
      } else {
        // NeighborsContainerType is a multi_map or unordered_multi_map
        static_assert(neighbors_container_spec == Container::MULTISET ||
                      neighbors_container_spec == Container::UNORDERED_MULTISET);
        auto [eq_begin, eq_end] =
            tgt_neighbors.equal_range(src_pos->first);  // slightly optimized search
        return std::find_if(eq_begin, eq_end, prop_finder);
      }
    } else {  // either multi edge is disallowed, or we don't differentiate multi-edges
      static_assert(!has_edge_prop || multi_edge == MultiEdge::DISALLOWED);
      return detail::container::find(tgt_neighbors, src_pos->first);
    }
  }

  // this method is useful for the "remove all" operation
  std::pair<NeighborsConstIterator, NeighborsConstIterator> find_remove_range(
      NeighborsContainerType& neighbors, const NodeType& node) {
    static_assert(has_edge_prop);
    if constexpr (neighbors_container_spec == Container::VEC ||
                  neighbors_container_spec == Container::LIST) {
      NeighborsConstIterator partition_pos =
          std::partition(neighbors.begin(), neighbors.end(),
                         [&node](const auto& src_nbr) { return !(src_nbr.first == node); });
      return {partition_pos, neighbors.end()};
    } else {
      static_assert(neighbors_container_spec == Container::MULTISET ||
                    neighbors_container_spec == Container::UNORDERED_MULTISET);
      return neighbors.equal_range(node);
    }
  }
  // END OF edge removal helpers
 public:  // edge removal
  // all iterators are assumed to be valid
  int remove_edge(ConstIterator source_pos, NeighborsConstIterator target_nbr_pos) noexcept {
    AdjListIterType src_pos = detail::const_iter_to_iter(adj_list, source_pos.it);
    NeighborsContainerType& src_neighbors = get_out_neighbors(src_pos);
    assert(target_nbr_pos != src_neighbors.cend());
    AdjListIterType tgt_pos = adj_list.find(get_neighbor_node(target_nbr_pos));
    assert(tgt_pos != adj_list.end());
    if constexpr (self_loop == SelfLoop::DISALLOWED) {
      assert(src_pos != tgt_pos);
    }
    NeighborsContainerType& tgt_neighbors = get_in_neighbors(tgt_pos);
    NeighborsConstIterator tgt_remove_pos =
        find_tgt_remove_pos(src_pos, target_nbr_pos, tgt_neighbors);
    assert(tgt_remove_pos != tgt_neighbors.end());
    if constexpr (has_edge_prop) {  // need to remove edge prop, as well
      this->edge_prop_list.erase(target_nbr_pos->second.pos);
    }
    detail::container::erase_one(src_neighbors, target_nbr_pos);
    if (src_pos != tgt_pos || direction == EdgeDirection::DIRECTED) {
      // when src==tgt && UNDIRECTED, there is NO double entry
      detail::container::erase_one(tgt_neighbors, tgt_remove_pos);
    }
    --num_of_edges;
    return 1;
  }

  // remove all edges between source and target
  template <typename U, typename V>
  std::enable_if_t<!std::is_convertible_v<V, NeighborsConstIterator>, int> remove_edge(
      const U& source_iv, const V& target_iv) noexcept {
    auto src_pos = find_by_iter_or_by_value(source_iv);
    auto tgt_pos = find_by_iter_or_by_value(target_iv);
    if (src_pos == adj_list.end() || tgt_pos == adj_list.end()) {
      if (src_pos == adj_list.end()) {
        print_by_iter_or_by_value(std::cerr << "(remove_edge) edge involves non-existent node",
                                  source_iv)
            << "\n";
      }
      if (tgt_pos == adj_list.end()) {
        print_by_iter_or_by_value(std::cerr << "(remove_edge) edge involves non-existent node",
                                  target_iv)
            << "\n";
      }
      return 0;  // no-op if nodes are not found
    }
    const NodeType& src_full = src_pos->first;
    const NodeType& tgt_full = tgt_pos->first;
    if constexpr (self_loop == SelfLoop::DISALLOWED) {
      if (src_pos == tgt_pos) {  // we know self loop cannot exist
        std::cerr << "(remove_edge) cannot remove self loop on node " << src_full
                  << " when self loop is not even permitted\n";
        return 0;
      }
    }
    NeighborsContainerType& src_neighbors = get_out_neighbors(src_pos);
    if constexpr (multi_edge == MultiEdge::DISALLOWED) {  // remove at most 1
      NeighborsIterator src_remove_pos = detail::container::find(src_neighbors, tgt_full);
      if (src_remove_pos == src_neighbors.cend()) {
        std::cerr << "(remove_edge) edge (" << src_full << ", " << tgt_full << ") not found\n";
        return 0;
      }
      remove_edge(ConstIterator{src_pos}, src_remove_pos);
      --num_of_edges;
      return 1;
    } else {  // remove all edges between src and tgt, potentially removing no edge at all
      static_assert(multi_edge == MultiEdge::ALLOWED);
      static_assert(neighbors_container_spec != Container::SET &&
                    neighbors_container_spec != Container::UNORDERED_SET);
      int num_edges_removed = 0;
      if constexpr (has_edge_prop) {  // remove prop too
        const auto [src_remove_begin, src_remove_end] = find_remove_range(src_neighbors, tgt_full);
        // loop through this range to remove prop
        for (auto it = src_remove_begin; it != src_remove_end; ++it) {
          ++num_edges_removed;
          this->edge_prop_list.erase(it->second.pos);
        }
        // erase this range itself
        src_neighbors.erase(src_remove_begin, src_remove_end);
      } else {  // simply erase all
        num_edges_removed = detail::container::erase_all(src_neighbors, tgt_full);
      }
      if (src_pos != tgt_pos || direction == EdgeDirection::DIRECTED) {
        int num_tgt_removed = detail::container::erase_all(get_in_neighbors(tgt_pos), src_full);
        assert(num_edges_removed == num_tgt_removed);
      }
      if (num_edges_removed == 0) {
        std::cerr << "(remove_edge) edge (" << src_full << ", " << tgt_full << ") not found\n";
      }
      num_of_edges -= num_edges_removed;
      return num_edges_removed;
    }
  }

 private:  // node removal helper
  template <bool OutIn>
  void purge_edge_with(AdjListIterType pos) noexcept {
    const NodeType& node = pos->first;
    NeighborsContainerType& neighbors = [ this, &pos ]() -> auto& {
      if constexpr (OutIn) {
        return get_out_neighbors(pos);
      } else {
        return get_in_neighbors(pos);
      }
    }
    ();
    NeighborsIterator nbr_begin = neighbors.begin();
    NeighborsIterator nbr_end = neighbors.end();
    // this loop should be enough for undirected graphs
    for (auto it = nbr_begin; it != nbr_end; ++it) {
      // purge edges that has to do with the to-be-removed node
      AdjListIterType neighbor_pos = adj_list.find(get_neighbor_node(it));
      NeighborsContainerType& neighbors_of_neighbor = [ this, &neighbor_pos ]() -> auto& {
        if constexpr (OutIn) {
          return get_in_neighbors(neighbor_pos);
        } else {
          return get_out_neighbors(neighbor_pos);
        }
      }
      ();
      if constexpr (self_loop == SelfLoop::DISALLOWED) {
        detail::container::erase_all(neighbors_of_neighbor, node);
      } else {
        if (neighbor_pos != pos) {  // need to check for self loop
          detail::container::erase_all(neighbors_of_neighbor, node);
        }
      }
      // remove edge property if needed
      if constexpr (has_edge_prop) {
        this->edge_prop_list.erase(it->second.pos);
      }
    }
  }

 public:  // node removal
  // we can allow removal of several nodes by iterator because erase does not invalidate other
  // iterators
  template <typename T>
  int remove_nodes(const T& node_iv) noexcept {
    auto pos = find_by_iter_or_by_value(node_iv);
    if (pos == adj_list.end()) {  // no-op if not found
      print_by_iter_or_by_value(std::cerr << "(remove_nodes) removing non-existent node", node_iv)
          << "\n";
      return 0;
    }
    purge_edge_with<true>(pos);  // purge all edges going out of node
    if constexpr (direction == EdgeDirection::DIRECTED) {
      // this loop makes it work for directed graphs as well
      purge_edge_with<false>(pos);  // purge all edges coming into node
    }
    // count purged edges
    auto& out_nbrs = get_out_neighbors(pos);
    int num_edges_purged = out_nbrs.size();
    if constexpr (direction == EdgeDirection::DIRECTED) {
      num_edges_purged += get_in_neighbors(pos).size();
      if constexpr (self_loop == SelfLoop::ALLOWED) {  // we would be double counting self-edges
        num_edges_purged -= detail::container::count(out_nbrs, pos->first);
      }
    }
    num_of_edges -= num_edges_purged;
    // finally erase pos
    adj_list.erase(pos);
    return 1;
  }
  template <typename T, typename... Args>
  int remove_nodes(const T& node_iv, const Args&... args) noexcept {
    return remove_nodes(node_iv) + remove_nodes(args...);
  }
  // END OF node removal
};
}

#endif  // GSK_GRAPH_LITE_H
