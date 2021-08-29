/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1997 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/

#include "curl_setup.h"

#include "splay.h"

/*
 * This macro compares two node keys i and j and returns:
 *
 *  negative value: when i is smaller than j
 *  zero          : when i is equal   to   j
 *  positive when : when i is larger  than j
 */
#define compare(i,j) Curl_splaycomparekeys((i),(j))

/*
 * Splay using the key i (which may or may not be in the tree.) The starting
 * root is t.
 */
struct Curl_tree *Curl_splay(struct curltime i,
                             struct Curl_tree *t)
{
  struct Curl_tree N, *l, *r, *y;

  if(!t)
    return t;
  N.smaller = N.larger = NULL;
  l = r = &N;

  for(;;) {
    long comp = compare(i, t->key);
    if(comp < 0) {
      if(!t->smaller)
        break;
      if(compare(i, t->smaller->key) < 0) {
        y = t->smaller;                           /* rotate smaller */
        t->smaller = y->larger;
        y->larger = t;
        t = y;
        if(!t->smaller)
          break;
      }
      r->smaller = t;                               /* link smaller */
      r = t;
      t = t->smaller;
    }
    else if(comp > 0) {
      if(!t->larger)
        break;
      if(compare(i, t->larger->key) > 0) {
        y = t->larger;                          /* rotate larger */
        t->larger = y->smaller;
        y->smaller = t;
        t = y;
        if(!t->larger)
          break;
      }
      l->larger = t;                              /* link larger */
      l = t;
      t = t->larger;
    }
    else
      break;
  }

  l->larger = t->smaller;                                /* assemble */
  r->smaller = t->larger;
  t->smaller = N.larger;
  t->larger = N.smaller;

  return t;
}

/* Insert key i into the tree t.  Return a pointer to the resulting tree or
 * NULL if something went wrong.
 *
 * @unittest: 1309
 */
struct Curl_tree *Curl_splayinsert(struct curltime i,
                                   struct Curl_tree *t,
                                   struct Curl_tree *node)
{
  static const struct curltime KEY_NOTUSED = {
    (time_t)-1, (unsigned int)-1
  }; /* will *NEVER* appear */

  if(!node)
    return t;

  if(t != NULL) {
    t = Curl_splay(i, t);
    if(compare(i, t->key) == 0) {
      /* There already exists a node in the tree with the very same key. Build
         a doubly-linked circular list of nodes. We add the new 'node' struct
         to the end of this list. */

      node->key = KEY_NOTUSED; /* we set the key in the sub node to NOTUSED
                                  to quickly identify this node as a subnode */
      node->samen = t;
      node->samep = t->samep;
      t->samep->samen = node;
      t->samep = node;

      return t; /* the root node always stays the same */
    }
  }

  if(!t) {
    node->smaller = node->larger = NULL;
  }
  else if(compare(i, t->key) < 0) {
    node->smaller = t->smaller;
    node->larger = t;
    t->smaller = NULL;

  }
  else {
    node->larger = t->larger;
    node->smaller = t;
    t->larger = NULL;
  }
  node->key = i;

  /* no identical nodes (yet), we are the only one in the list of nodes */
  node->samen = node;
  node->samep = node;
  return node;
}

/* Finds and deletes the best-fit node from the tree. Return a pointer to the
   resulting tree.  best-fit means the smallest node if it is not larger than
   the key */
struct Curl_tree *Curl_splaygetbest(struct curltime i,
                                    struct Curl_tree *t,
                                    struct Curl_tree **removed)
{
  static const struct curltime tv_zero = {0, 0};
  struct Curl_tree *x;

  if(!t) {
    *removed = NULL; /* none removed since there was no root */
    return NULL;
  }

  /* find smallest */
  t = Curl_splay(tv_zero, t);
  if(compare(i, t->key) < 0) {
    /* even the smallest is too big */
    *removed = NULL;
    return t;
  }

  /* FIRST! Check if there is a list with identical keys */
  x = t->samen;
  if(x != t) {
    /* there is, pick one from the list */

    /* 'x' is the new root node */

    x->key = t->key;
    x->larger = t->larger;
    x->smaller = t->smaller;
    x->samep = t->samep;
    t->samep->samen = x;

    *removed = t;
    return x; /* new root */
  }

  /* we splayed the tree to the smallest element, there is no smaller */
  x = t->larger;
  *removed = t;

  return x;
}


/* Deletes the very node we point out from the tree if it's there. Stores a
 * pointer to the new resulting tree in 'newroot'.
 *
 * Returns zero on success and non-zero on errors!
 * When returning error, it does not touch the 'newroot' pointer.
 *
 * NOTE: when the last node of the tree is removed, there's no tree left so
 * 'newroot' will be made to point to NULL.
 *
 * @unittest: 1309
 */
int Curl_splayremove(struct Curl_tree *t,
                     struct Curl_tree *removenode,
                     struct Curl_tree **newroot)
{
  static const struct curltime KEY_NOTUSED = {
    (time_t)-1, (unsigned int)-1
  }; /* will *NEVER* appear */
  struct Curl_tree *x;

  if(!t || !removenode)
    return 1;

  if(compare(KEY_NOTUSED, removenode->key) == 0) {
    /* Key set to NOTUSED means it is a subnode within a 'same' linked list
       and thus we can unlink it easily. */
    if(removenode->samen == removenode)
      /* A non-subnode should never be set to KEY_NOTUSED */
      return 3;

    removenode->samep->samen = removenode->samen;
    removenode->samen->samep = removenode->samep;

    /* Ensures that double-remove gets caught. */
    removenode->samen = removenode;

    *newroot = t; /* return the same root */
    return 0;
  }

  t = Curl_splay(removenode->key, t);

  /* First make sure that we got the same root node as the one we want
     to remove, as otherwise we might be trying to remove a node that
     isn't actually in the tree.

     We cannot just compare the keys here as a double remove in quick
     succession of a node with key != KEY_NOTUSED && same != NULL
     could return the same key but a different node. */
  if(t != removenode)
    return 2;

  /* Check if there is a list with identical sizes, as then we're trying to
     remove the root node of a list of nodes with identical keys. */
  x = t->samen;
  if(x != t) {
    /* 'x' is the new root node, we just make it use the root node's
       smaller/larger links */

    x->key = t->key;
    x->larger = t->larger;
    x->smaller = t->smaller;
    x->samep = t->samep;
    t->samep->samen = x;
  }
  else {
    /* Remove the root node */
    if(!t->smaller)
      x = t->larger;
    else {
      x = Curl_splay(removenode->key, t->smaller);
      x->larger = t->larger;
    }
  }

  *newroot = x; /* store new root pointer */

  return 0;
}
