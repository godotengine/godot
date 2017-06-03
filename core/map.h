/*************************************************************************/
/*  map.h                                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
		//Color color;
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
			//memdelete_allocator<Element,A>(_root);
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
				return NULL;
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
				if (node->parent == _data._root)
					return NULL;
				node = node->parent;
			}
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
				break; // found
		}

		return (node != _data._nil) ? node : NULL;
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
				break; // found
		}

		if (node == _data._nil) {
			if (prev == NULL)
				return NULL;
			if (less(p_key, prev->_key)) {

				prev = prev->_prev;
			}

			return prev;

		} else
			return node;
	}

	Element *_insert(const K &p_key, bool &r_exists) {

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
				r_exists = true;
				return node;
			}
		}

		Element *new_node = memnew_allocator(Element, A);

		new_node->parent = new_parent;
		new_node->right = _data._nil;
		new_node->left = _data._nil;
		new_node->_key = p_key;
		//new_node->data=_data;
		if (new_parent == _data._root || less(p_key, new_parent->_key)) {

			new_parent->left = new_node;
		} else {
			new_parent->right = new_node;
		}

		r_exists = false;

		new_node->_next = _successor(new_node);
		new_node->_prev = _predecessor(new_node);
		if (new_node->_next)
			new_node->_next->_prev = new_node;
		if (new_node->_prev)
			new_node->_prev->_next = new_node;

		return new_node;
	}

	Element *_insert_rb(const K &p_key, const V &p_value) {

		bool exists = false;
		Element *new_node = _insert(p_key, exists);

		if (new_node) {
			new_node->_value = p_value;
		}
		if (exists)
			return new_node;

		Element *node = new_node;
		_data.size_cache++;

		while (node->parent->color == RED) {

			if (node->parent == node->parent->parent->left) {

				Element *aux = node->parent->parent->right;

				if (aux->color == RED) {
					_set_color(node->parent, BLACK);
					_set_color(aux, BLACK);
					_set_color(node->parent->parent, RED);
					node = node->parent->parent;
				} else {
					if (node == node->parent->right) {
						node = node->parent;
						_rotate_left(node);
					}
					_set_color(node->parent, BLACK);
					_set_color(node->parent->parent, RED);
					_rotate_right(node->parent->parent);
				}
			} else {
				Element *aux = node->parent->parent->left;

				if (aux->color == RED) {
					_set_color(node->parent, BLACK);
					_set_color(aux, BLACK);
					_set_color(node->parent->parent, RED);
					node = node->parent->parent;
				} else {
					if (node == node->parent->left) {
						node = node->parent;
						_rotate_right(node);
					}
					_set_color(node->parent, BLACK);
					_set_color(node->parent->parent, RED);
					_rotate_left(node->parent->parent);
				}
			}
		}
		_set_color(_data._root->left, BLACK);
		return new_node;
	}

	void _erase_fix(Element *p_node) {

		Element *root = _data._root->left;
		Element *node = p_node;

		while ((node->color == BLACK) && (root != node)) {
			if (node == node->parent->left) {
				Element *aux = node->parent->right;
				if (aux->color == RED) {
					_set_color(aux, BLACK);
					_set_color(node->parent, RED);
					_rotate_left(node->parent);
					aux = node->parent->right;
				}
				if ((aux->right->color == BLACK) && (aux->left->color == BLACK)) {
					_set_color(aux, RED);
					node = node->parent;
				} else {
					if (aux->right->color == BLACK) {
						_set_color(aux->left, BLACK);
						_set_color(aux, RED);
						_rotate_right(aux);
						aux = node->parent->right;
					}
					_set_color(aux, node->parent->color);
					_set_color(node->parent, BLACK);
					_set_color(aux->right, BLACK);
					_rotate_left(node->parent);
					node = root; /* this is to exit while loop */
				}
			} else { /* the code below is has left and right switched from above */
				Element *aux = node->parent->left;
				if (aux->color == RED) {
					_set_color(aux, BLACK);
					_set_color(node->parent, RED);
					_rotate_right(node->parent);
					aux = node->parent->left;
				}
				if ((aux->right->color == BLACK) && (aux->left->color == BLACK)) {
					_set_color(aux, RED);
					node = node->parent;
				} else {
					if (aux->left->color == BLACK) {
						_set_color(aux->right, BLACK);
						_set_color(aux, RED);
						_rotate_left(aux);
						aux = node->parent->left;
					}
					_set_color(aux, node->parent->color);
					_set_color(node->parent, BLACK);
					_set_color(aux->left, BLACK);
					_rotate_right(node->parent);
					node = root;
				}
			}
		}

		_set_color(node, BLACK);

		ERR_FAIL_COND(_data._nil->color != BLACK);
	}

	void _erase(Element *p_node) {

		Element *rp = ((p_node->left == _data._nil) || (p_node->right == _data._nil)) ? p_node : _successor(p_node);
		if (!rp)
			rp = _data._nil;
		Element *node = (rp->left == _data._nil) ? rp->right : rp->left;

		if (_data._root == (node->parent = rp->parent)) {
			_data._root->left = node;
		} else {
			if (rp == rp->parent->left) {
				rp->parent->left = node;
			} else {
				rp->parent->right = node;
			}
		}

		if (rp != p_node) {

			ERR_FAIL_COND(rp == _data._nil);

			if (rp->color == BLACK)
				_erase_fix(node);

			rp->left = p_node->left;
			rp->right = p_node->right;
			rp->parent = p_node->parent;
			rp->color = p_node->color;
			p_node->left->parent = rp;
			p_node->right->parent = rp;

			if (p_node == p_node->parent->left) {
				p_node->parent->left = rp;
			} else {
				p_node->parent->right = rp;
			}
		} else {
			if (p_node->color == BLACK)
				_erase_fix(node);
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

		if (p_element == _data._nil) {
			return;
		}
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

	Element *insert(const K &p_key, const V &p_value) {

		if (!_data._root)
			_data._create_root();
		return _insert_rb(p_key, p_value);
	}

	void erase(Element *p_element) {

		if (!_data._root)
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
		return true;
	}

	bool has(const K &p_key) const {

		if (!_data._root)
			return false;
		return find(p_key) != NULL;
	}

	const V &operator[](const K &p_key) const {

		ERR_FAIL_COND_V(!_data._root, *(V *)NULL); // crash on purpose
		const Element *e = find(p_key);
		ERR_FAIL_COND_V(!e, *(V *)NULL); // crash on purpose
		return e->_value;
	}
	V &operator[](const K &p_key) {

		if (!_data._root)
			_data._create_root();

		Element *e = find(p_key);
		if (!e)
			e = insert(p_key, V());

		ERR_FAIL_COND_V(!e, *(V *)NULL); // crash on purpose
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
		_data._nil->parent = _data._nil;
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
