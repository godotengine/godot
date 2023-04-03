/*
 * Copyright © 2017 Jason Ekstrand
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef RB_TREE_H
#define RB_TREE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/** A red-black tree node
 *
 * This struct represents a node in the red-black tree.  This struct should
 * be embedded as a member in whatever structure you wish to put in the
 * tree.
 */
struct rb_node {
    /** Parent and color of this node
     *
     * The least significant bit represents the color and is est to 1 for
     * black and 0 for red.  The other bits are the pointer to the parent
     * and that pointer can be retrieved by masking off the bottom bit and
     * casting to a pointer.
     */
    uintptr_t parent;

    /** Left child of this node, null for a leaf */
    struct rb_node *left;

    /** Right child of this node, null for a leaf */
    struct rb_node *right;
};

/** Return the parent node of the given node or NULL if it is the root */
static inline struct rb_node *
rb_node_parent(struct rb_node *n)
{
    return (struct rb_node *)(n->parent & ~(uintptr_t)1);
}

/** A red-black tree
 *
 * This struct represents the red-black tree itself.  It is just a pointer
 * to the root node with no other metadata.
 */
struct rb_tree {
    struct rb_node *root;
};

/** Initialize a red-black tree */
void rb_tree_init(struct rb_tree *T);

/** Returns true if the red-black tree is empty */
static inline bool
rb_tree_is_empty(const struct rb_tree *T)
{
    return T->root == NULL;
}

/** Retrieve the data structure containing a node
 *
 * \param   type    The type of the containing data structure
 *
 * \param   node    A pointer to a rb_node
 *
 * \param   field   The rb_node field in the containing data structure
 */
#define rb_node_data(type, node, field) \
    ((type *)(((char *)(node)) - offsetof(type, field)))

/** Insert a node into a tree at a particular location
 *
 * This function should probably not be used directly as it relies on the
 * caller to ensure that the parent node is correct.  Use rb_tree_insert
 * instead.
 *
 * \param   T           The red-black tree into which to insert the new node
 *
 * \param   parent      The node in the tree that will be the parent of the
 *                      newly inserted node
 *
 * \param   node        The node to insert
 *
 * \param   insert_left If true, the new node will be the left child of
 *                      \p parent, otherwise it will be the right child
 */
void rb_tree_insert_at(struct rb_tree *T, struct rb_node *parent,
                       struct rb_node *node, bool insert_left);

/** Insert a node into a tree
 *
 * \param   T       The red-black tree into which to insert the new node
 *
 * \param   node    The node to insert
 *
 * \param   cmp     A comparison function to use to order the nodes.
 */
static inline void
rb_tree_insert(struct rb_tree *T, struct rb_node *node,
               int (*cmp)(const struct rb_node *, const struct rb_node *))
{
    /* This function is declared inline in the hopes that the compiler can
     * optimize away the comparison function pointer call.
     */
    struct rb_node *y = NULL;
    struct rb_node *x = T->root;
    bool left = false;
    while (x != NULL) {
        y = x;
        left = cmp(x, node) < 0;
        if (left)
            x = x->left;
        else
            x = x->right;
    }

    rb_tree_insert_at(T, y, node, left);
}

/** Remove a node from a tree
 *
 * \param   T       The red-black tree from which to remove the node
 *
 * \param   node    The node to remove
 */
void rb_tree_remove(struct rb_tree *T, struct rb_node *z);

/** Search the tree for a node
 *
 * If a node with a matching key exists, the first matching node found will
 * be returned.  If no matching node exists, NULL is returned.
 *
 * \param   T       The red-black tree to search
 *
 * \param   key     The key to search for
 *
 * \param   cmp     A comparison function to use to order the nodes
 */
static inline struct rb_node *
rb_tree_search(struct rb_tree *T, const void *key,
               int (*cmp)(const struct rb_node *, const void *))
{
    /* This function is declared inline in the hopes that the compiler can
     * optimize away the comparison function pointer call.
     */
    struct rb_node *x = T->root;
    while (x != NULL) {
        int c = cmp(x, key);
        if (c < 0)
            x = x->left;
        else if (c > 0)
            x = x->right;
        else
            return x;
    }

    return x;
}

/** Sloppily search the tree for a node
 *
 * This function searches the tree for a given node.  If a node with a
 * matching key exists, that first matching node found will be returned.
 * If no node with an exactly matching key exists, the node returned will
 * be either the right-most node comparing less than \p key or the
 * right-most node comparing greater than \p key.  If the tree is empty,
 * NULL is returned.
 *
 * \param   T       The red-black tree to search
 *
 * \param   key     The key to search for
 *
 * \param   cmp     A comparison function to use to order the nodes
 */
static inline struct rb_node *
rb_tree_search_sloppy(struct rb_tree *T, const void *key,
                      int (*cmp)(const struct rb_node *, const void *))
{
    /* This function is declared inline in the hopes that the compiler can
     * optimize away the comparison function pointer call.
     */
    struct rb_node *y = NULL;
    struct rb_node *x = T->root;
    while (x != NULL) {
        y = x;
        int c = cmp(x, key);
        if (c < 0)
            x = x->left;
        else if (c > 0)
            x = x->right;
        else
            return x;
    }

    return y;
}

/** Get the first (left-most) node in the tree or NULL */
struct rb_node *rb_tree_first(struct rb_tree *T);

/** Get the last (right-most) node in the tree or NULL */
struct rb_node *rb_tree_last(struct rb_tree *T);

/** Get the next node (to the right) in the tree or NULL */
struct rb_node *rb_node_next(struct rb_node *node);

/** Get the next previous (to the left) in the tree or NULL */
struct rb_node *rb_node_prev(struct rb_node *node);

#define rb_node_next_or_null(n) ((n) == NULL ? NULL : rb_node_next(n))
#define rb_node_prev_or_null(n) ((n) == NULL ? NULL : rb_node_prev(n))

/** Iterate over the nodes in the tree
 *
 * \param   type    The type of the containing data structure
 *
 * \param   node    The variable name for current node in the iteration;
 *                  this will be declared as a pointer to \p type
 *
 * \param   T       The red-black tree
 *
 * \param   field   The rb_node field in containing data structure
 */
#define rb_tree_foreach(type, iter, T, field) \
   for (type *iter, *__node = (type *)rb_tree_first(T); \
        __node != NULL && \
        (iter = rb_node_data(type, (struct rb_node *)__node, field), true); \
        __node = (type *)rb_node_next((struct rb_node *)__node))

/** Iterate over the nodes in the tree, allowing the current node to be freed
 *
 * \param   type    The type of the containing data structure
 *
 * \param   node    The variable name for current node in the iteration;
 *                  this will be declared as a pointer to \p type
 *
 * \param   T       The red-black tree
 *
 * \param   field   The rb_node field in containing data structure
 */
#define rb_tree_foreach_safe(type, iter, T, field) \
   for (type *iter, \
             *__node = (type *)rb_tree_first(T), \
             *__next = (type *)rb_node_next_or_null((struct rb_node *)__node); \
        __node != NULL && \
        (iter = rb_node_data(type, (struct rb_node *)__node, field), true); \
        __node = __next, \
        __next = (type *)rb_node_next_or_null((struct rb_node *)__node))

/** Iterate over the nodes in the tree in reverse
 *
 * \param   type    The type of the containing data structure
 *
 * \param   node    The variable name for current node in the iteration;
 *                  this will be declared as a pointer to \p type
 *
 * \param   T       The red-black tree
 *
 * \param   field   The rb_node field in containing data structure
 */
#define rb_tree_foreach_rev(type, iter, T, field) \
   for (type *iter, *__node = (type *)rb_tree_last(T); \
        __node != NULL && \
        (iter = rb_node_data(type, (struct rb_node *)__node, field), true); \
        __node = (type *)rb_node_prev((struct rb_node *)__node))

/** Iterate over the nodes in the tree in reverse, allowing the current node to be freed
 *
 * \param   type    The type of the containing data structure
 *
 * \param   node    The variable name for current node in the iteration;
 *                  this will be declared as a pointer to \p type
 *
 * \param   T       The red-black tree
 *
 * \param   field   The rb_node field in containing data structure
 */
#define rb_tree_foreach_rev_safe(type, iter, T, field) \
   for (type *iter, \
             *__node = (type *)rb_tree_last(T), \
             *__prev = (type *)rb_node_prev_or_null((struct rb_node *)__node); \
        __node != NULL && \
        (iter = rb_node_data(type, (struct rb_node *)__node, field), true); \
        __node = __prev, \
        __prev = (type *)rb_node_prev_or_null((struct rb_node *)__node))

/** Validate a red-black tree
 *
 * This function walks the tree and validates that this is a valid red-
 * black tree.  If anything is wrong, it will assert-fail.
 */
void rb_tree_validate(struct rb_tree *T);

#ifdef __cplusplus
} /* extern C */
#endif

#endif /* RB_TREE_H */
