/*************************************************************************/
/*  map.h                                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef MAP_H
#define MAP_H

#include "set.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

// based on the very nice implementation of rb-trees by:
// http://web.mit.edu/~emin/www/source_code/red_black_tree/index.html

template <class K, class V, class C = Comparator<K>, class A = DefaultAllocator>
class Map {

	enum Color {
		RED,
		BLACK
	};
	struct _Data;

public:
	class Element {

	private:
		friend class Map<K, V, C, A>;
		int color;
		Element *right;
		Element *left;
		Element *parent;
		Element *_next;
		Element *_prev;
		K _key;
		V _value;
		//_Data *data;

	public:
		const Element *next() const {

			return _next;
		}
		Element *next() {

			return _next;
		}
		const Element *prev() const {

			return _prev;
		}
		Element *prev() {

			return _prev;
		}
		const K &key() const {
			return _key;
		};
		V &value() {
			return _value;
		};
		const V &value() const {
			return _value;
		};
		V &get() {
			return _value;
		};
		const V &get() const {
			return _value;
		};
		Element() {
			color = RED;
			right = NULL;
			left = NULL;
			parent = NULL;
			_next = NULL;
			_prev = NULL;
		};
	};

private:
	struct _Data {

		Element *_root;
		Element *_nil;
		int size_cache;

		_FORCE_INLINE_ _Data() {
#ifdef GLOBALNIL_DISABLED
			_nil = memnew_allocator(Element, A);
			_nil->parent = _nil->left = _nil->right = _nil;
			_nil->color = BLACK;
#else
			_nil = (Element *)&_GlobalNilClass::_nil;
#endif
			_root = NULL;
			size_cache = 0;
		}

		void _create_root() {

			_root = memnew_allocator(Element, A);
			_root->parent = _root->left = _root->right = _nil;
			_root->color = BLACK;
		}

		void _free_root() {

			if (_root) {
				memdelete_allocator<Element, A>(_root);
				_root = NULL;
			}
		}

		~_Data() {

			_free_root();

#ifdef GLOBALNIL_DISABLED
			memdelete_allocator<Element, A>(_nil);
#endif
		}
	};

	_Data _data;

	inline void _set_color(Element *p_node, int p_color) {

		ERR_FAIL_COND(p_node == _data._nil && p_color == RED);
		p_node->color = p_color;
	}

	inline void _rotate_left(Element *p_node) {

		Element *r = p_node->right;
		p_node->right = r->left;
		if (r->left != _data._nil)
			r->left->parent = p_node;
		r->parent = p_node->parent;
		if (p_node == p_node->parent->left)
			p_node->parent->left = r;
		else
			p_node->parent->right = r;

		r->left = p_node;
		p_node->parent = r;
	}

	inline void _rotate_right(Element *p_node) {

		Element *l = p_node->left;
		p_node->left = l->right;
		if (l->right != _data._nil)
			l->right->parent = p_node;
		l->parent = p_node->parent;
		if (p_node == p_node->parent->right)
			p_node->parent->right = l;
		else
			p_node->parent->left = l;

		l->right = p_node;
		p_node->parent = l;
	}

	inline Element *_successor(Element *p_node) const {

		Element *node = p_node;

		if (node->right != _data._nil) {

			node = node->right;
			while (node->left != _data._nil) { /* returns the minium of the right subtree of node */
				node = node->left;
			}
			return node;
		} else {

			while (node == node->parent->right) {
				node = node->parent;
			}

			if (node->parent == _data._root)
				return NULL; // No successor, as p_node = last node
			return node->parent;
		}
	}

	inline Element *_predecessor(Element *p_node) const {
		Element *node = p_node;

		if (node->left != _data._nil) {

			node = node->left;
			while (node->right != _data._nil) { /* returns the minium of the left subtree of node */
				node = node->right;
			}
			return node;
		} else {

			while (node == node->parent->left) {
				node = node->parent;
			}

			if (node == _data._root)
				return NULL; // No predecessor, as p_node = first node
			return node->parent;
		}
	}

	Element *_find(const K &p_key) const {

		Element *node = _data._root->left;
		C less;

		while (node != _data._nil) {
			if (less(p_key, node->_key))
				node = node->left;
			else if (less(node->_key, p_key))
				node = node->right;
			else
				return node; // found
		}

		return NULL;
	}

	Element *_find_closest(const K &p_key) const {

		Element *node = _data._root->left;
		Element *prev = NULL;
		C less;

		while (node != _data._nil) {
			prev = node;

			if (less(p_key, node->_key))
				node = node->left;
			else if (less(node->_key, p_key))
				node = node->right;
			else
				return node; // found
		}

		if (prev == NULL)
			return NULL; // tree empty

		if (less(p_key, prev->_key))
			prev = prev->_prev;

		return prev;
	}

	void _insert_rb_fix(Element *p_new_node) {

		Element *node = p_new_node;
		Element *nparent = node->parent;
		Element *ngrand_parent;

		while (nparent->color == RED) {
			ngrand_parent = nparent->parent;

			if (nparent == ngrand_parent->left) {
				if (ngrand_parent->right->color == RED) {
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent->right, BLACK);
					_set_color(ngrand_parent, RED);
					node = ngrand_parent;
					nparent = node->parent;
				} else {
					if (node == nparent->right) {
						_rotate_left(nparent);
						node = nparent;
						nparent = node->parent;
					}
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent, RED);
					_rotate_right(ngrand_parent);
				}
			} else {
				if (ngrand_parent->left->color == RED) {
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent->left, BLACK);
					_set_color(ngrand_parent, RED);
					node = ngrand_parent;
					nparent = node->parent;
				} else {
					if (node == nparent->left) {
						_rotate_right(nparent);
						node = nparent;
						nparent = node->parent;
					}
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent, RED);
					_rotate_left(ngrand_parent);
				}
			}
		}

		_set_color(_data._root->left, BLACK);
	}

	Element *_insert(const K &p_key, const V &p_value) {

		Element *new_parent = _data._root;
		Element *node = _data._root->left;
		C less;

		while (node != _data._nil) {

			new_parent = node;

			if (less(p_key, node->_key))
				node = node->left;
			else if (less(node->_key, p_key))
				node = node->right;
			else {
				node->_value = p_value;
				return node; // Return existing node with new value
			}
		}

		Element *new_node = memnew_allocator(Element, A);
		new_node->parent = new_parent;
		new_node->right = _data._nil;
		new_node->left = _data._nil;
		new_node->_key = p_key;
		new_node->_value = p_value;
		//new_node->data=_data;

		if (new_parent == _data._root || less(p_key, new_parent->_key)) {
			new_parent->left = new_node;
		} else {
			new_parent->right = new_node;
		}

		new_node->_next = _successor(new_node);
		new_node->_prev = _predecessor(new_node);
		if (new_node->_next)
			new_node->_next->_prev = new_node;
		if (new_node->_prev)
			new_node->_prev->_next = new_node;

		_data.size_cache++;
		_insert_rb_fix(new_node);
		return new_node;
	}

	void _erase_fix_rb(Element *p_node) {

		Element *root = _data._root->left;
		Element *node = _data._nil;
		Element *sibling = p_node;
		Element *parent = sibling->parent;

		while (node != root) { // If red node found, will exit at a break
			if (sibling->color == RED) {
				_set_color(sibling, BLACK);
				_set_color(parent, RED);
				if (sibling == parent->right) {
					sibling = sibling->left;
					_rotate_left(parent);
				} else {
					sibling = sibling->right;
					_rotate_right(parent);
				}
			}
			if ((sibling->left->color == BLACK) && (sibling->right->color == BLACK)) {
				_set_color(sibling, RED);
				if (parent->color == RED) {
					_set_color(parent, BLACK);
					break;
				} else { // loop: haven't found any red nodes yet
					node = parent;
					parent = node->parent;
					sibling = (node == parent->left) ? parent->right : parent->left;
				}
			} else {
				if (sibling == parent->right) {
					if (sibling->right->color == BLACK) {
						_set_color(sibling->left, BLACK);
						_set_color(sibling, RED);
						_rotate_right(sibling);
						sibling = sibling->parent;
					}
					_set_color(sibling, parent->color);
					_set_color(parent, BLACK);
					_set_color(sibling->right, BLACK);
					_rotate_left(parent);
					break;
				} else {
					if (sibling->left->color == BLACK) {
						_set_color(sibling->right, BLACK);
						_set_color(sibling, RED);
						_rotate_left(sibling);
						sibling = sibling->parent;
					}

					_set_color(sibling, parent->color);
					_set_color(parent, BLACK);
					_set_color(sibling->left, BLACK);
					_rotate_right(parent);
					break;
				}
			}
		}

		ERR_FAIL_COND(_data._nil->color != BLACK);
	}

	void _erase(Element *p_node) {

		Element *rp = ((p_node->left == _data._nil) || (p_node->right == _data._nil)) ? p_node : p_node->_next;
		Element *node = (rp->left == _data._nil) ? rp->right : rp->left;

		Element *sibling;
		if (rp == rp->parent->left) {
			rp->parent->left = node;
			sibling = rp->parent->right;
		} else {
			rp->parent->right = node;
			sibling = rp->parent->left;
		}

		if (node->color == RED) {
			node->parent = rp->parent;
			_set_color(node, BLACK);
		} else if (rp->color == BLACK && rp->parent != _data._root) {
			_erase_fix_rb(sibling);
		}

		if (rp != p_node) {

			ERR_FAIL_COND(rp == _data._nil);

			rp->left = p_node->left;
			rp->right = p_node->right;
			rp->parent = p_node->parent;
			rp->color = p_node->color;
			if (p_node->left != _data._nil)
				p_node->left->parent = rp;
			if (p_node->right != _data._nil)
				p_node->right->parent = rp;

			if (p_node == p_node->parent->left) {
				p_node->parent->left = rp;
			} else {
				p_node->parent->right = rp;
			}
		}

		if (p_node->_next)
			p_node->_next->_prev = p_node->_prev;
		if (p_node->_prev)
			p_node->_prev->_next = p_node->_next;

		memdelete_allocator<Element, A>(p_node);
		_data.size_cache--;
		ERR_FAIL_COND(_data._nil->color == RED);
	}

	void _calculate_depth(Element *p_element, int &max_d, int d) const {

		if (p_element == _data._nil)
			return;

		_calculate_depth(p_element->left, max_d, d + 1);
		_calculate_depth(p_element->right, max_d, d + 1);

		if (d > max_d)
			max_d = d;
	}

	void _cleanup_tree(Element *p_element) {

		if (p_element == _data._nil)
			return;

		_cleanup_tree(p_element->left);
		_cleanup_tree(p_element->right);
		memdelete_allocator<Element, A>(p_element);
	}

	void _copy_from(const Map &p_map) {

		clear();
		// not the fastest way, but safeset to write.
		for (Element *I = p_map.front(); I; I = I->next()) {

			insert(I->key(), I->value());
		}
	}

public:
	const Element *find(const K &p_key) const {

		if (!_data._root)
			return NULL;

		const Element *res = _find(p_key);
		return res;
	}

	Element *find(const K &p_key) {

		if (!_data._root)
			return NULL;

		Element *res = _find(p_key);
		return res;
	}

	const Element *find_closest(const K &p_key) const {

		if (!_data._root)
			return NULL;

		const Element *res = _find_closest(p_key);
		return res;
	}

	Element *find_closest(const K &p_key) {

		if (!_data._root)
			return NULL;

		Element *res = _find_closest(p_key);
		return res;
	}

	bool has(const K &p_key) const {

		return find(p_key) != NULL;
	}

	Element *insert(const K &p_key, const V &p_value) {

		if (!_data._root)
			_data._create_root();
		return _insert(p_key, p_value);
	}

	void erase(Element *p_element) {

		if (!_data._root || !p_element)
			return;

		_erase(p_element);
		if (_data.size_cache == 0 && _data._root)
			_data._free_root();
	}

	bool erase(const K &p_key) {

		if (!_data._root)
			return false;

		Element *e = find(p_key);
		if (!e)
			return false;

		_erase(e);
		if (_data.size_cache == 0 && _data._root)
			_data._free_root();
		return true;
	}

	const V &operator[](const K &p_key) const {

		CRASH_COND(!_data._root);
		const Element *e = find(p_key);
		CRASH_COND(!e);
		return e->_value;
	}

	V &operator[](const K &p_key) {

		if (!_data._root)
			_data._create_root();

		Element *e = find(p_key);
		if (!e)
			e = insert(p_key, V());

		return e->_value;
	}

	Element *front() const {

		if (!_data._root)
			return NULL;

		Element *e = _data._root->left;
		if (e == _data._nil)
			return NULL;

		while (e->left != _data._nil)
			e = e->left;

		return e;
	}

	Element *back() const {

		if (!_data._root)
			return NULL;

		Element *e = _data._root->left;
		if (e == _data._nil)
			return NULL;

		while (e->right != _data._nil)
			e = e->right;

		return e;
	}

	inline bool empty() const { return _data.size_cache == 0; }
	inline int size() const { return _data.size_cache; }

	int calculate_depth() const {
		// used for debug mostly
		if (!_data._root)
			return 0;

		int max_d = 0;
		_calculate_depth(_data._root->left, max_d, 0);
		return max_d;
	}

	void clear() {

		if (!_data._root)
			return;

		_cleanup_tree(_data._root->left);
		_data._root->left = _data._nil;
		_data.size_cache = 0;
		_data._free_root();
	}

	void operator=(const Map &p_map) {

		_copy_from(p_map);
	}

	Map(const Map &p_map) {

		_copy_from(p_map);
	}

	_FORCE_INLINE_ Map() {
	}

	~Map() {

		clear();
	}
};

#endif
