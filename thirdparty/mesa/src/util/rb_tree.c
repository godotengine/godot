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

#include "rb_tree.h"

/** \file rb_tree.c
 *
 * An implementation of a red-black tree
 *
 * This file implements the guts of a red-black tree.  The implementation
 * is mostly based on the one in "Introduction to Algorithms", third
 * edition, by Cormen, Leiserson, Rivest, and Stein.  The primary
 * divergence in our algorithms from those presented in CLRS is that we use
 * NULL for the leaves instead of a sentinel.  This means we have to do a
 * tiny bit more tracking in our implementation of delete but it makes the
 * algorithms far more explicit than stashing stuff in the sentinel.
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>

static bool
rb_node_is_black(struct rb_node *n)
{
    /* NULL nodes are leaves and therefore black */
    return (n == NULL) || (n->parent & 1);
}

static bool
rb_node_is_red(struct rb_node *n)
{
    return !rb_node_is_black(n);
}

static void
rb_node_set_black(struct rb_node *n)
{
    n->parent |= 1;
}

static void
rb_node_set_red(struct rb_node *n)
{
    n->parent &= ~1ull;
}

static void
rb_node_copy_color(struct rb_node *dst, struct rb_node *src)
{
    dst->parent = (dst->parent & ~1ull) | (src->parent & 1);
}

static void
rb_node_set_parent(struct rb_node *n, struct rb_node *p)
{
    n->parent = (n->parent & 1) | (uintptr_t)p;
}

static struct rb_node *
rb_node_minimum(struct rb_node *node)
{
    while (node->left)
        node = node->left;
    return node;
}

static struct rb_node *
rb_node_maximum(struct rb_node *node)
{
    while (node->right)
        node = node->right;
    return node;
}

void
rb_tree_init(struct rb_tree *T)
{
    T->root = NULL;
}

/**
 * Replace the subtree of T rooted at u with the subtree rooted at v
 *
 * This is called RB-transplant in CLRS.
 *
 * The node to be replaced is assumed to be a non-leaf.
 */
static void
rb_tree_splice(struct rb_tree *T, struct rb_node *u, struct rb_node *v)
{
    assert(u);
    struct rb_node *p = rb_node_parent(u);
    if (p == NULL) {
        assert(T->root == u);
        T->root = v;
    } else if (u == p->left) {
        p->left = v;
    } else {
        assert(u == p->right);
        p->right = v;
    }
    if (v)
        rb_node_set_parent(v, p);
}

static void
rb_tree_rotate_left(struct rb_tree *T, struct rb_node *x)
{
    assert(x && x->right);

    struct rb_node *y = x->right;
    x->right = y->left;
    if (y->left)
        rb_node_set_parent(y->left, x);
    rb_tree_splice(T, x, y);
    y->left = x;
    rb_node_set_parent(x, y);
}

static void
rb_tree_rotate_right(struct rb_tree *T, struct rb_node *y)
{
    assert(y && y->left);

    struct rb_node *x = y->left;
    y->left = x->right;
    if (x->right)
        rb_node_set_parent(x->right, y);
    rb_tree_splice(T, y, x);
    x->right = y;
    rb_node_set_parent(y, x);
}

void
rb_tree_insert_at(struct rb_tree *T, struct rb_node *parent,
                  struct rb_node *node, bool insert_left)
{
    /* This sets null children, parent, and a color of red */
    memset(node, 0, sizeof(*node));

    if (parent == NULL) {
        assert(T->root == NULL);
        T->root = node;
        rb_node_set_black(node);
        return;
    }

    if (insert_left) {
        assert(parent->left == NULL);
        parent->left = node;
    } else {
        assert(parent->right == NULL);
        parent->right = node;
    }
    rb_node_set_parent(node, parent);

    /* Now we do the insertion fixup */
    struct rb_node *z = node;
    while (rb_node_is_red(rb_node_parent(z))) {
        struct rb_node *z_p = rb_node_parent(z);
        assert(z == z_p->left || z == z_p->right);
        struct rb_node *z_p_p = rb_node_parent(z_p);
        assert(z_p_p != NULL);
        if (z_p == z_p_p->left) {
            struct rb_node *y = z_p_p->right;
            if (rb_node_is_red(y)) {
                rb_node_set_black(z_p);
                rb_node_set_black(y);
                rb_node_set_red(z_p_p);
                z = z_p_p;
            } else {
                if (z == z_p->right) {
                    z = z_p;
                    rb_tree_rotate_left(T, z);
                    /* We changed z */
                    z_p = rb_node_parent(z);
                    assert(z == z_p->left || z == z_p->right);
                    z_p_p = rb_node_parent(z_p);
                }
                rb_node_set_black(z_p);
                rb_node_set_red(z_p_p);
                rb_tree_rotate_right(T, z_p_p);
            }
        } else {
            struct rb_node *y = z_p_p->left;
            if (rb_node_is_red(y)) {
                rb_node_set_black(z_p);
                rb_node_set_black(y);
                rb_node_set_red(z_p_p);
                z = z_p_p;
            } else {
                if (z == z_p->left) {
                    z = z_p;
                    rb_tree_rotate_right(T, z);
                    /* We changed z */
                    z_p = rb_node_parent(z);
                    assert(z == z_p->left || z == z_p->right);
                    z_p_p = rb_node_parent(z_p);
                }
                rb_node_set_black(z_p);
                rb_node_set_red(z_p_p);
                rb_tree_rotate_left(T, z_p_p);
            }
        }
    }
    rb_node_set_black(T->root);
}

void
rb_tree_remove(struct rb_tree *T, struct rb_node *z)
{
    /* x_p is always the parent node of X.  We have to track this
     * separately because x may be NULL.
     */
    struct rb_node *x, *x_p;
    struct rb_node *y = z;
    bool y_was_black = rb_node_is_black(y);
    if (z->left == NULL) {
        x = z->right;
        x_p = rb_node_parent(z);
        rb_tree_splice(T, z, x);
    } else if (z->right == NULL) {
        x = z->left;
        x_p = rb_node_parent(z);
        rb_tree_splice(T, z, x);
    } else {
        /* Find the minimum sub-node of z->right */
        y = rb_node_minimum(z->right);
        y_was_black = rb_node_is_black(y);

        x = y->right;
        if (rb_node_parent(y) == z) {
            x_p = y;
        } else {
            x_p = rb_node_parent(y);
            rb_tree_splice(T, y, x);
            y->right = z->right;
            rb_node_set_parent(y->right, y);
        }
        assert(y->left == NULL);
        rb_tree_splice(T, z, y);
        y->left = z->left;
        rb_node_set_parent(y->left, y);
        rb_node_copy_color(y, z);
    }

    assert(x_p == NULL || x == x_p->left || x == x_p->right);

    if (!y_was_black)
        return;

    /* Fixup RB tree after the delete */
    while (x != T->root && rb_node_is_black(x)) {
        if (x == x_p->left) {
            struct rb_node *w = x_p->right;
            if (rb_node_is_red(w)) {
                rb_node_set_black(w);
                rb_node_set_red(x_p);
                rb_tree_rotate_left(T, x_p);
                assert(x == x_p->left);
                w = x_p->right;
            }
            if (rb_node_is_black(w->left) && rb_node_is_black(w->right)) {
                rb_node_set_red(w);
                x = x_p;
            } else {
                if (rb_node_is_black(w->right)) {
                    rb_node_set_black(w->left);
                    rb_node_set_red(w);
                    rb_tree_rotate_right(T, w);
                    w = x_p->right;
                }
                rb_node_copy_color(w, x_p);
                rb_node_set_black(x_p);
                rb_node_set_black(w->right);
                rb_tree_rotate_left(T, x_p);
                x = T->root;
            }
        } else {
            struct rb_node *w = x_p->left;
            if (rb_node_is_red(w)) {
                rb_node_set_black(w);
                rb_node_set_red(x_p);
                rb_tree_rotate_right(T, x_p);
                assert(x == x_p->right);
                w = x_p->left;
            }
            if (rb_node_is_black(w->right) && rb_node_is_black(w->left)) {
                rb_node_set_red(w);
                x = x_p;
            } else {
                if (rb_node_is_black(w->left)) {
                    rb_node_set_black(w->right);
                    rb_node_set_red(w);
                    rb_tree_rotate_left(T, w);
                    w = x_p->left;
                }
                rb_node_copy_color(w, x_p);
                rb_node_set_black(x_p);
                rb_node_set_black(w->left);
                rb_tree_rotate_right(T, x_p);
                x = T->root;
            }
        }
        x_p = rb_node_parent(x);
    }
    if (x)
        rb_node_set_black(x);
}

struct rb_node *
rb_tree_first(struct rb_tree *T)
{
    return T->root ? rb_node_minimum(T->root) : NULL;
}

struct rb_node *
rb_tree_last(struct rb_tree *T)
{
    return T->root ? rb_node_maximum(T->root) : NULL;
}

struct rb_node *
rb_node_next(struct rb_node *node)
{
    if (node->right) {
        /* If we have a right child, then the next thing (compared to this
         * node) is the left-most child of our right child.
         */
        return rb_node_minimum(node->right);
    } else {
        /* If node doesn't have a right child, crawl back up the to the
         * left until we hit a parent to the right.
         */
        struct rb_node *p = rb_node_parent(node);
        while (p && node == p->right) {
            node = p;
            p = rb_node_parent(node);
        }
        assert(p == NULL || node == p->left);
        return p;
    }
}

struct rb_node *
rb_node_prev(struct rb_node *node)
{
    if (node->left) {
        /* If we have a left child, then the previous thing (compared to
         * this node) is the right-most child of our left child.
         */
        return rb_node_maximum(node->left);
    } else {
        /* If node doesn't have a left child, crawl back up the to the
         * right until we hit a parent to the left.
         */
        struct rb_node *p = rb_node_parent(node);
        while (p && node == p->left) {
            node = p;
            p = rb_node_parent(node);
        }
        assert(p == NULL || node == p->right);
        return p;
    }
}

static void
validate_rb_node(struct rb_node *n, int black_depth)
{
    if (n == NULL) {
        assert(black_depth == 0);
        return;
    }

    if (rb_node_is_black(n)) {
        black_depth--;
    } else {
        assert(rb_node_is_black(n->left));
        assert(rb_node_is_black(n->right));
    }

    validate_rb_node(n->left, black_depth);
    validate_rb_node(n->right, black_depth);
}

void
rb_tree_validate(struct rb_tree *T)
{
    if (T->root == NULL)
        return;

    assert(rb_node_is_black(T->root));

    unsigned black_depth = 0;
    for (struct rb_node *n = T->root; n; n = n->left) {
        if (rb_node_is_black(n))
            black_depth++;
    }

    validate_rb_node(T->root, black_depth);
}
