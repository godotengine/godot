/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#if CONFIG_DEBUG
#include <assert.h>
#endif
#include <stdio.h>

#include "treecoder.h"

static void tree2tok(
    struct vp8_token_struct *const p,
    vp8_tree t,
    int i,
    int v,
    int L
)
{
    v += v;
    ++L;

    do
    {
        const vp8_tree_index j = t[i++];

        if (j <= 0)
        {
            p[-j].value = v;
            p[-j].Len = L;
        }
        else
            tree2tok(p, t, j, v, L);
    }
    while (++v & 1);
}

void vp8_tokens_from_tree(struct vp8_token_struct *p, vp8_tree t)
{
    tree2tok(p, t, 0, 0, 0);
}

void vp8_tokens_from_tree_offset(struct vp8_token_struct *p, vp8_tree t,
                                 int offset)
{
    tree2tok(p - offset, t, 0, 0, 0);
}

static void branch_counts(
    int n,                      /* n = size of alphabet */
    vp8_token tok               [ /* n */ ],
    vp8_tree tree,
    unsigned int branch_ct       [ /* n-1 */ ] [2],
    const unsigned int num_events[ /* n */ ]
)
{
    const int tree_len = n - 1;
    int t = 0;

#if CONFIG_DEBUG
    assert(tree_len);
#endif

    do
    {
        branch_ct[t][0] = branch_ct[t][1] = 0;
    }
    while (++t < tree_len);

    t = 0;

    do
    {
        int L = tok[t].Len;
        const int enc = tok[t].value;
        const unsigned int ct = num_events[t];

        vp8_tree_index i = 0;

        do
        {
            const int b = (enc >> --L) & 1;
            const int j = i >> 1;
#if CONFIG_DEBUG
            assert(j < tree_len  &&  0 <= L);
#endif

            branch_ct [j] [b] += ct;
            i = tree[ i + b];
        }
        while (i > 0);

#if CONFIG_DEBUG
        assert(!L);
#endif
    }
    while (++t < n);

}


void vp8_tree_probs_from_distribution(
    int n,                      /* n = size of alphabet */
    vp8_token tok               [ /* n */ ],
    vp8_tree tree,
    vp8_prob probs          [ /* n-1 */ ],
    unsigned int branch_ct       [ /* n-1 */ ] [2],
    const unsigned int num_events[ /* n */ ],
    unsigned int Pfac,
    int rd
)
{
    const int tree_len = n - 1;
    int t = 0;

    branch_counts(n, tok, tree, branch_ct, num_events);

    do
    {
        const unsigned int *const c = branch_ct[t];
        const unsigned int tot = c[0] + c[1];

#if CONFIG_DEBUG
        assert(tot < (1 << 24));        /* no overflow below */
#endif

        if (tot)
        {
            const unsigned int p = ((c[0] * Pfac) + (rd ? tot >> 1 : 0)) / tot;
            probs[t] = p < 256 ? (p ? p : 1) : 255; /* agree w/old version for now */
        }
        else
            probs[t] = vp8_prob_half;
    }
    while (++t < tree_len);
}
