#pragma once
#include "object.hpp"

struct Node : public Object {
	/// @brief Construct a Node object from an existing in-scope Node object.
	/// @param addr The address of the Node object.
	constexpr Node(uint64_t addr) : Object{addr} {}
	constexpr Node(Object obj) : Object{obj.address()} {}

	/// @brief Construct a Node object from a path.
	/// @param path The path to the Node object.
	Node(std::string_view path);

	/// @brief Get the name of the node.
	/// @return The name of the node.
	Variant get_name() const;

	/// @brief Set the name of the node.
	/// @param name The new name of the node.
	void set_name(Variant name);

	/// @brief Get the path of the node, relative to the root node.
	/// @return The path of the node.
	Variant get_path() const;

	/// @brief Get the parent of the node.
	/// @return The parent node.
	Node get_parent() const;

	/// @brief Get the Node object at the given path, relative to this node.
	/// @param path The path to the Node object.
	/// @return The Node object.
	Node get_node(std::string_view path) const;

	template <typename T>
	T get_node(std::string_view path) const {
		return T(get_node(path));
	}

	/// @brief Get the number of children of the node.
	/// @return The number of children.
	unsigned get_child_count() const;

	/// @brief Get the child of the node at the given index.
	/// @param index The index of the child.
	/// @return The child node.
	Node get_child(unsigned index) const;

	/// @brief Add a child to the node.
	/// @param child The child node to add.
	/// @param deferred If true, the child will be added next frame.
	void add_child(const Node &child, bool deferred = false);

	/// @brief Add a sibling to the node.
	/// @param sibling The sibling node to add.
	/// @param deferred If true, the sibling will be added next frame.
	void add_sibling(const Node &sibling, bool deferred = false);

	/// @brief Move a child of the node to a new index.
	/// @param child The child node to move.
	/// @param index The new index of the child.
	void move_child(const Node &child, unsigned index);

	/// @brief Remove a child from the node. The child is *not* freed.
	/// @param child The child node to remove.
	/// @param deferred If true, the child will be removed next frame.
	void remove_child(const Node &child, bool deferred = false);

	/// @brief Get a list of children of the node.
	/// @return A list of children nodes.
	std::vector<Node> get_children() const;

	/// @brief Add the node to a group.
	/// @param group The group to add the node to.
	void add_to_group(std::string_view group);

	/// @brief Remove the node from a group.
	/// @param group The group to remove the node from.
	void remove_from_group(std::string_view group);

	/// @brief Check if the node is in a group.
	/// @param group The group to check.
	/// @return True if the node is in the group, false otherwise.
	bool is_in_group(std::string_view group) const;

	/// @brief Check if the node is inside the scene tree.
	/// @return True if the node is inside the scene tree, false otherwise.
	bool is_inside_tree() const;

	/// @brief Replace the node with another node.
	/// @param node The node to replace this node with.
	void replace_by(const Node &node, bool keep_groups = false);

	/// @brief Changes the parent of this Node to the new_parent.
	/// The node needs to already have a parent.
	/// The node's owner is preserved if its owner is still reachable
	/// from the new location (i.e., the node is still a descendant
	/// of the new parent after the operation).
	/// @param new_parent The new parent node.
	/// @param keep_global_transform If true, the node's global transform is preserved.
	void reparent(const Node &new_parent, bool keep_global_transform = true);

	/// @brief Remove this node from its parent, freeing it.
	/// @note This is a potentially deferred operation.
	void queue_free();

	/// @brief  Duplicate the node.
	/// @return A new Node object with the same properties and children.
	Node duplicate(int flags = 15) const;

	/// @brief Create a new Node object.
	/// @param path The path to the Node object.
	/// @return The Node object.
	static Node Create(std::string_view path);

	//- Properties -//
	PROPERTY(name, String);
	PROPERTY(owner, Node);
	PROPERTY(unique_name_in_owner, bool);
	PROPERTY(editor_description, String);
	PROPERTY(physics_interpolation_mode, int64_t);
	PROPERTY(process_mode, int64_t);
	PROPERTY(process_priority, int64_t);

	//- Methods -//
	METHOD(bool, can_process);
	METHOD(Object, create_tween);
	METHOD(Node, find_child);
	METHOD(Variant, find_children);
	METHOD(Node, find_parent);
	METHOD(Node, get_viewport);
	METHOD(Node, get_window);
	METHOD(bool, has_node);
	METHOD(bool, has_node_and_resource);
	METHOD(bool, is_ancestor_of);
	METHOD(void, set_physics_process);
	METHOD(bool, is_physics_processing);
	METHOD(void, set_physics_process_internal);
	METHOD(bool, is_physics_processing_internal);
	METHOD(void, set_process);
	METHOD(bool, is_processing);
	METHOD(void, set_process_input);
	METHOD(bool, is_processing_input);
	METHOD(void, set_process_internal);
	METHOD(bool, is_processing_internal);
	METHOD(void, set_process_unhandled_input);
	METHOD(bool, is_processing_unhandled_input);
	METHOD(void, set_process_unhandled_key_input);
	METHOD(bool, is_processing_unhandled_key_input);
	METHOD(void, set_process_shortcut_input);
	METHOD(bool, is_processing_shortcut_input);
	METHOD(bool, is_node_ready);
	METHOD(void, set_thread_safe);
	METHOD(void, set_owner);
	METHOD(Node, get_owner);
	METHOD(void, set_scene_file_path);
	METHOD(String, get_scene_file_path);
	METHOD(void, print_tree);
	METHOD(void, print_tree_pretty);
	METHOD(void, print_orphan_nodes);
	METHOD(void, propagate_call);


	//- Signals -//

	//- Constants -//
	static constexpr int PROCESS_MODE_INHERIT = 0;
	static constexpr int PROCESS_MODE_PAUSABLE = 1;
	static constexpr int PROCESS_MODE_WHEN_PAUSED = 2;
	static constexpr int PROCESS_MODE_ALWAYS = 3;
	static constexpr int PROCESS_MODE_DISABLED = 4;

	static constexpr int PHYSICS_INTERPOLATION_MODE_INHERIT = 0;
	static constexpr int PHYSICS_INTERPOLATION_MODE_ON = 1;
	static constexpr int PHYSICS_INTERPOLATION_MODE_OFF = 2;

	static constexpr int DUPLICATE_SIGNALS = 1;
	static constexpr int DUPLICATE_GROUPS = 2;
	static constexpr int DUPLICATE_SCRIPTS = 4;
	static constexpr int DUPLICATE_USE_INSTANTIATION = 8;
};

inline Node Variant::as_node() const {
	if (get_type() == Variant::OBJECT)
		return Node{uintptr_t(v.i)};
	else if (get_type() == Variant::NODE_PATH)
		return Node{this->internal_fetch_string()};

	api_throw("std::bad_cast", "Variant is not a Node or NodePath", this);
}

inline Node Object::as_node() const {
	return Node{address()};
}

template <typename T>
static inline T cast_to(const Variant &var) {
	if (var.get_type() == Variant::OBJECT)
		return T{uintptr_t(var)};
	api_throw("std::bad_cast", "Variant is not an Object", &var);
}

inline Variant::Variant(const Node &node) {
	m_type = OBJECT;
	v.i = node.address();
}

inline Variant::operator Node() const {
	return as_node();
}
