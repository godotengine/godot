/*
 * Copyright © 2008, 2010 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * \file list.h
 * \brief Doubly-linked list abstract container type.
 *
 * Each doubly-linked list has a sentinel head and tail node.  These nodes
 * contain no data.  The head sentinel can be identified by its \c prev
 * pointer being \c NULL.  The tail sentinel can be identified by its
 * \c next pointer being \c NULL.
 *
 * A list is empty if either the head sentinel's \c next pointer points to the
 * tail sentinel or the tail sentinel's \c prev poiner points to the head
 * sentinel. The head sentinel and tail sentinel nodes are allocated within the
 * list structure.
 *
 * Do note that this means that the list nodes will contain pointers into the
 * list structure itself and as a result you may not \c realloc() an  \c
 * exec_list or any structure in which an \c exec_list is embedded.
 */

#ifndef LIST_CONTAINER_H
#define LIST_CONTAINER_H

#ifndef __cplusplus
#include <stddef.h>
#endif
#include <assert.h>

#include "util/ralloc.h"

struct exec_node {
   struct exec_node *next;
   struct exec_node *prev;

#ifdef __cplusplus
   DECLARE_RZALLOC_CXX_OPERATORS(exec_node)

   exec_node() : next(NULL), prev(NULL)
   {
      /* empty */
   }

   const exec_node *get_next() const;
   exec_node *get_next();

   const exec_node *get_prev() const;
   exec_node *get_prev();

   void remove();

   /**
    * Link a node with itself
    *
    * This creates a sort of degenerate list that is occasionally useful.
    */
   void self_link();

   /**
    * Insert a node in the list after the current node
    */
   void insert_after(exec_node *after);

   /**
    * Insert another list in the list after the current node
    */
   void insert_after(struct exec_list *after);

   /**
    * Insert a node in the list before the current node
    */
   void insert_before(exec_node *before);

   /**
    * Insert another list in the list before the current node
    */
   void insert_before(struct exec_list *before);

   /**
    * Replace the current node with the given node.
    */
   void replace_with(exec_node *replacement);

   /**
    * Is this the sentinel at the tail of the list?
    */
   bool is_tail_sentinel() const;

   /**
    * Is this the sentinel at the head of the list?
    */
   bool is_head_sentinel() const;
#endif
};

static inline void
exec_node_init(struct exec_node *n)
{
   n->next = NULL;
   n->prev = NULL;
}

static inline const struct exec_node *
exec_node_get_next_const(const struct exec_node *n)
{
   return n->next;
}

static inline struct exec_node *
exec_node_get_next(struct exec_node *n)
{
   return n->next;
}

static inline const struct exec_node *
exec_node_get_prev_const(const struct exec_node *n)
{
   return n->prev;
}

static inline struct exec_node *
exec_node_get_prev(struct exec_node *n)
{
   return n->prev;
}

static inline void
exec_node_remove(struct exec_node *n)
{
   n->next->prev = n->prev;
   n->prev->next = n->next;
   n->next = NULL;
   n->prev = NULL;
}

static inline void
exec_node_self_link(struct exec_node *n)
{
   n->next = n;
   n->prev = n;
}

static inline void
exec_node_insert_after(struct exec_node *n, struct exec_node *after)
{
   after->next = n->next;
   after->prev = n;

   n->next->prev = after;
   n->next = after;
}

static inline void
exec_node_insert_node_before(struct exec_node *n, struct exec_node *before)
{
   before->next = n;
   before->prev = n->prev;

   n->prev->next = before;
   n->prev = before;
}

static inline void
exec_node_replace_with(struct exec_node *n, struct exec_node *replacement)
{
   replacement->prev = n->prev;
   replacement->next = n->next;

   n->prev->next = replacement;
   n->next->prev = replacement;
}

static inline bool
exec_node_is_tail_sentinel(const struct exec_node *n)
{
   return n->next == NULL;
}

static inline bool
exec_node_is_head_sentinel(const struct exec_node *n)
{
   return n->prev == NULL;
}

#ifdef __cplusplus
inline const exec_node *exec_node::get_next() const
{
   return exec_node_get_next_const(this);
}

inline exec_node *exec_node::get_next()
{
   return exec_node_get_next(this);
}

inline const exec_node *exec_node::get_prev() const
{
   return exec_node_get_prev_const(this);
}

inline exec_node *exec_node::get_prev()
{
   return exec_node_get_prev(this);
}

inline void exec_node::remove()
{
   exec_node_remove(this);
}

inline void exec_node::self_link()
{
   exec_node_self_link(this);
}

inline void exec_node::insert_after(exec_node *after)
{
   exec_node_insert_after(this, after);
}

inline void exec_node::insert_before(exec_node *before)
{
   exec_node_insert_node_before(this, before);
}

inline void exec_node::replace_with(exec_node *replacement)
{
   exec_node_replace_with(this, replacement);
}

inline bool exec_node::is_tail_sentinel() const
{
   return exec_node_is_tail_sentinel(this);
}

inline bool exec_node::is_head_sentinel() const
{
   return exec_node_is_head_sentinel(this);
}
#endif

#ifdef __cplusplus
/* This macro will not work correctly if `t' uses virtual inheritance.  If you
 * are using virtual inheritance, you deserve a slow and painful death.  Enjoy!
 */
#define exec_list_offsetof(t, f, p) \
   (((char *) &((t *) p)->f) - ((char *) p))
#else
#define exec_list_offsetof(t, f, p) offsetof(t, f)
#endif

/**
 * Get a pointer to the structure containing an exec_node
 *
 * Given a pointer to an \c exec_node embedded in a structure, get a pointer to
 * the containing structure.
 *
 * \param type  Base type of the structure containing the node
 * \param node  Pointer to the \c exec_node
 * \param field Name of the field in \c type that is the embedded \c exec_node
 */
#define exec_node_data(type, node, field) \
   ((type *) (((uintptr_t) node) - exec_list_offsetof(type, field, node)))

#ifdef __cplusplus
struct exec_node;
#endif

struct exec_list {
   struct exec_node head_sentinel;
   struct exec_node tail_sentinel;

#ifdef __cplusplus
   DECLARE_RALLOC_CXX_OPERATORS(exec_list)

   exec_list()
   {
      make_empty();
   }

   void make_empty();

   bool is_empty() const;

   const exec_node *get_head() const;
   exec_node *get_head();
   const exec_node *get_head_raw() const;
   exec_node *get_head_raw();

   const exec_node *get_tail() const;
   exec_node *get_tail();
   const exec_node *get_tail_raw() const;
   exec_node *get_tail_raw();

   unsigned length() const;

   void push_head(exec_node *n);
   void push_tail(exec_node *n);
   void push_degenerate_list_at_head(exec_node *n);

   /**
    * Remove the first node from a list and return it
    *
    * \return
    * The first node in the list or \c NULL if the list is empty.
    *
    * \sa exec_list::get_head
    */
   exec_node *pop_head();

   /**
    * Move all of the nodes from this list to the target list
    */
   void move_nodes_to(exec_list *target);

   /**
    * Append all nodes from the source list to the end of the target list
    */
   void append_list(exec_list *source);

   /**
    * Prepend all nodes from the source list to the beginning of the target
    * list
    */
   void prepend_list(exec_list *source);
#endif
};

static inline void
exec_list_make_empty(struct exec_list *list)
{
   list->head_sentinel.next = &list->tail_sentinel;
   list->head_sentinel.prev = NULL;
   list->tail_sentinel.next = NULL;
   list->tail_sentinel.prev = &list->head_sentinel;
}

static inline bool
exec_list_is_empty(const struct exec_list *list)
{
   /* There are three ways to test whether a list is empty or not.
    *
    * - Check to see if the head sentinel's \c next is the tail sentinel.
    * - Check to see if the tail sentinel's \c prev is the head sentinel.
    * - Check to see if the head is the sentinel node by test whether its
    *   \c next pointer is \c NULL.
    *
    * The first two methods tend to generate better code on modern systems
    * because they save a pointer dereference.
    */
   return list->head_sentinel.next == &list->tail_sentinel;
}

static inline bool
exec_list_is_singular(const struct exec_list *list)
{
   return !exec_list_is_empty(list) &&
          list->head_sentinel.next->next == &list->tail_sentinel;
}

static inline const struct exec_node *
exec_list_get_head_const(const struct exec_list *list)
{
   return !exec_list_is_empty(list) ? list->head_sentinel.next : NULL;
}

static inline struct exec_node *
exec_list_get_head(struct exec_list *list)
{
   return !exec_list_is_empty(list) ? list->head_sentinel.next : NULL;
}

static inline const struct exec_node *
exec_list_get_head_raw_const(const struct exec_list *list)
{
   return list->head_sentinel.next;
}

static inline struct exec_node *
exec_list_get_head_raw(struct exec_list *list)
{
   return list->head_sentinel.next;
}

static inline const struct exec_node *
exec_list_get_tail_const(const struct exec_list *list)
{
   return !exec_list_is_empty(list) ? list->tail_sentinel.prev : NULL;
}

static inline struct exec_node *
exec_list_get_tail(struct exec_list *list)
{
   return !exec_list_is_empty(list) ? list->tail_sentinel.prev : NULL;
}

static inline const struct exec_node *
exec_list_get_tail_raw_const(const struct exec_list *list)
{
   return list->tail_sentinel.prev;
}

static inline struct exec_node *
exec_list_get_tail_raw(struct exec_list *list)
{
   return list->tail_sentinel.prev;
}

static inline unsigned
exec_list_length(const struct exec_list *list)
{
   unsigned size = 0;
   struct exec_node *node;

   for (node = list->head_sentinel.next; node->next != NULL; node = node->next) {
      size++;
   }

   return size;
}

static inline void
exec_list_push_head(struct exec_list *list, struct exec_node *n)
{
   n->next = list->head_sentinel.next;
   n->prev = &list->head_sentinel;

   n->next->prev = n;
   list->head_sentinel.next = n;
}

static inline void
exec_list_push_tail(struct exec_list *list, struct exec_node *n)
{
   n->next = &list->tail_sentinel;
   n->prev = list->tail_sentinel.prev;

   n->prev->next = n;
   list->tail_sentinel.prev = n;
}

static inline void
exec_list_push_degenerate_list_at_head(struct exec_list *list, struct exec_node *n)
{
   assert(n->prev->next == n);

   n->prev->next = list->head_sentinel.next;
   list->head_sentinel.next->prev = n->prev;
   n->prev = &list->head_sentinel;
   list->head_sentinel.next = n;
}

static inline struct exec_node *
exec_list_pop_head(struct exec_list *list)
{
   struct exec_node *const n = exec_list_get_head(list);
   if (n != NULL)
      exec_node_remove(n);

   return n;
}

static inline void
exec_list_move_nodes_to(struct exec_list *list, struct exec_list *target)
{
   if (exec_list_is_empty(list)) {
      exec_list_make_empty(target);
   } else {
      target->head_sentinel.next = list->head_sentinel.next;
      target->head_sentinel.prev = NULL;
      target->tail_sentinel.next = NULL;
      target->tail_sentinel.prev = list->tail_sentinel.prev;

      target->head_sentinel.next->prev = &target->head_sentinel;
      target->tail_sentinel.prev->next = &target->tail_sentinel;

      exec_list_make_empty(list);
   }
}

static inline void
exec_list_append(struct exec_list *list, struct exec_list *source)
{
   if (exec_list_is_empty(source))
      return;

   /* Link the first node of the source with the last node of the target list.
    */
   list->tail_sentinel.prev->next = source->head_sentinel.next;
   source->head_sentinel.next->prev = list->tail_sentinel.prev;

   /* Make the tail of the source list be the tail of the target list.
    */
   list->tail_sentinel.prev = source->tail_sentinel.prev;
   list->tail_sentinel.prev->next = &list->tail_sentinel;

   /* Make the source list empty for good measure.
    */
   exec_list_make_empty(source);
}

static inline void
exec_node_insert_list_after(struct exec_node *n, struct exec_list *after)
{
   if (exec_list_is_empty(after))
      return;

   after->tail_sentinel.prev->next = n->next;
   after->head_sentinel.next->prev = n;

   n->next->prev = after->tail_sentinel.prev;
   n->next = after->head_sentinel.next;

   exec_list_make_empty(after);
}

static inline void
exec_list_prepend(struct exec_list *list, struct exec_list *source)
{
   exec_list_append(source, list);
   exec_list_move_nodes_to(source, list);
}

static inline void
exec_node_insert_list_before(struct exec_node *n, struct exec_list *before)
{
   if (exec_list_is_empty(before))
      return;

   before->tail_sentinel.prev->next = n;
   before->head_sentinel.next->prev = n->prev;

   n->prev->next = before->head_sentinel.next;
   n->prev = before->tail_sentinel.prev;

   exec_list_make_empty(before);
}

static inline void
exec_list_validate(const struct exec_list *list)
{
   const struct exec_node *node;

   assert(list->head_sentinel.next->prev == &list->head_sentinel);
   assert(list->head_sentinel.prev == NULL);
   assert(list->tail_sentinel.next == NULL);
   assert(list->tail_sentinel.prev->next == &list->tail_sentinel);

   /* We could try to use one of the interators below for this but they all
    * either require C++ or assume the exec_node is embedded in a structure
    * which is not the case for this function.
    */
   for (node = list->head_sentinel.next; node->next != NULL; node = node->next) {
      assert(node->next->prev == node);
      assert(node->prev->next == node);
   }
}

#ifdef __cplusplus
inline void exec_list::make_empty()
{
   exec_list_make_empty(this);
}

inline bool exec_list::is_empty() const
{
   return exec_list_is_empty(this);
}

inline const exec_node *exec_list::get_head() const
{
   return exec_list_get_head_const(this);
}

inline exec_node *exec_list::get_head()
{
   return exec_list_get_head(this);
}

inline const exec_node *exec_list::get_head_raw() const
{
   return exec_list_get_head_raw_const(this);
}

inline exec_node *exec_list::get_head_raw()
{
   return exec_list_get_head_raw(this);
}

inline const exec_node *exec_list::get_tail() const
{
   return exec_list_get_tail_const(this);
}

inline exec_node *exec_list::get_tail()
{
   return exec_list_get_tail(this);
}

inline const exec_node *exec_list::get_tail_raw() const
{
   return exec_list_get_tail_raw_const(this);
}

inline exec_node *exec_list::get_tail_raw()
{
   return exec_list_get_tail_raw(this);
}

inline unsigned exec_list::length() const
{
   return exec_list_length(this);
}

inline void exec_list::push_head(exec_node *n)
{
   exec_list_push_head(this, n);
}

inline void exec_list::push_tail(exec_node *n)
{
   exec_list_push_tail(this, n);
}

inline void exec_list::push_degenerate_list_at_head(exec_node *n)
{
   exec_list_push_degenerate_list_at_head(this, n);
}

inline exec_node *exec_list::pop_head()
{
   return exec_list_pop_head(this);
}

inline void exec_list::move_nodes_to(exec_list *target)
{
   exec_list_move_nodes_to(this, target);
}

inline void exec_list::append_list(exec_list *source)
{
   exec_list_append(this, source);
}

inline void exec_node::insert_after(exec_list *after)
{
   exec_node_insert_list_after(this, after);
}

inline void exec_list::prepend_list(exec_list *source)
{
   exec_list_prepend(this, source);
}

inline void exec_node::insert_before(exec_list *before)
{
   exec_node_insert_list_before(this, before);
}
#endif

#define exec_node_typed_forward(__node, __type) \
   (!exec_node_is_tail_sentinel(__node) ? (__type) (__node) : NULL)

#define exec_node_typed_backward(__node, __type) \
   (!exec_node_is_head_sentinel(__node) ? (__type) (__node) : NULL)

#define foreach_in_list(__type, __inst, __list)                                           \
   for (__type *__inst = exec_node_typed_forward((__list)->head_sentinel.next, __type *); \
        (__inst) != NULL;                                                                 \
        (__inst) = exec_node_typed_forward((__inst)->next, __type *))

#define foreach_in_list_reverse(__type, __inst, __list)                                      \
   for (__type *__inst = exec_node_typed_backward((__list)->tail_sentinel.prev, __type *);   \
        (__inst) != NULL;                                                                    \
        (__inst) = exec_node_typed_backward((__inst)->prev, __type *))

/**
 * This version is safe even if the current node is removed.
 */

#define foreach_in_list_safe(__type, __node, __list)                                                              \
   for (__type *__node = exec_node_typed_forward((__list)->head_sentinel.next, __type *),                         \
               *__next = (__node) ? exec_node_typed_forward((__list)->head_sentinel.next->next, __type *) : NULL; \
        (__node) != NULL;                                                                                         \
        (__node) = __next, __next = __next ? exec_node_typed_forward(__next->next, __type *) : NULL)

#define foreach_in_list_reverse_safe(__type, __node, __list)                                                        \
   for (__type *__node = exec_node_typed_backward((__list)->tail_sentinel.prev, __type *),                          \
               *__prev = (__node) ? exec_node_typed_backward((__list)->tail_sentinel.prev->prev, __type *) : NULL;  \
        (__node) != NULL;                                                                                           \
        (__node) = __prev, __prev = __prev ? exec_node_typed_backward(__prev->prev, __type *) : NULL)

#define foreach_in_list_use_after(__type, __inst, __list)                           \
   __type *__inst;                                                                  \
   for ((__inst) = exec_node_typed_forward((__list)->head_sentinel.next, __type *); \
        (__inst) != NULL;                                                           \
        (__inst) = exec_node_typed_forward((__inst)->next, __type *))

/**
 * Iterate through two lists at once.  Stops at the end of the shorter list.
 *
 * This is safe against either current node being removed or replaced.
 */
#define foreach_two_lists(__node1, __list1, __node2, __list2) \
   for (struct exec_node * __node1 = (__list1)->head_sentinel.next, \
                         * __node2 = (__list2)->head_sentinel.next, \
                         * __next1 = __node1->next,           \
                         * __next2 = __node2->next            \
	; __next1 != NULL && __next2 != NULL                  \
	; __node1 = __next1,                                  \
          __node2 = __next2,                                  \
          __next1 = __next1->next,                            \
          __next2 = __next2->next)

#define exec_node_data_forward(type, node, field) \
   (!exec_node_is_tail_sentinel(node) ? exec_node_data(type, node, field) : NULL)

#define exec_node_data_backward(type, node, field) \
   (!exec_node_is_head_sentinel(node) ? exec_node_data(type, node, field) : NULL)

#define foreach_list_typed(__type, __node, __field, __list)                     \
   for (__type * __node =                                                       \
         exec_node_data_forward(__type, (__list)->head_sentinel.next, __field); \
   (__node) != NULL;                                                            \
   (__node) = exec_node_data_forward(__type, (__node)->__field.next, __field))

#define foreach_list_typed_from(__type, __node, __field, __list, __start)        \
   for (__type * __node = exec_node_data_forward(__type, (__start), __field);    \
   (__node) != NULL;                                                             \
   (__node) = exec_node_data_forward(__type, (__node)->__field.next, __field))

#define foreach_list_typed_reverse(__type, __node, __field, __list)                 \
   for (__type * __node =                                                           \
           exec_node_data_backward(__type, (__list)->tail_sentinel.prev, __field);  \
        (__node) != NULL;                                                           \
        (__node) = exec_node_data_backward(__type, (__node)->__field.prev, __field))

#define foreach_list_typed_safe(__type, __node, __field, __list)                    \
   for (__type * __node =                                                           \
           exec_node_data_forward(__type, (__list)->head_sentinel.next, __field),   \
               * __next = (__node) ?                                                \
           exec_node_data_forward(__type, (__node)->__field.next, __field) : NULL;  \
        (__node) != NULL;                                                           \
        (__node) = __next, __next = (__next && (__next)->__field.next) ?            \
           exec_node_data_forward(__type, (__next)->__field.next, __field) : NULL)

#define foreach_list_typed_reverse_safe(__type, __node, __field, __list)            \
   for (__type * __node =                                                           \
           exec_node_data_backward(__type, (__list)->tail_sentinel.prev, __field),  \
               * __prev = (__node) ?                                                \
           exec_node_data_backward(__type, (__node)->__field.prev, __field) : NULL; \
        (__node) != NULL;                                                           \
        (__node) = __prev, __prev = (__prev && (__prev)->__field.prev) ?            \
           exec_node_data_backward(__type, (__prev)->__field.prev, __field) : NULL)

#endif /* LIST_CONTAINER_H */
