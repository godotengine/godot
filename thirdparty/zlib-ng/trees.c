/* trees.c -- output deflated data using Huffman coding
 * Copyright (C) 1995-2024 Jean-loup Gailly
 * detect_data_type() function provided freely by Cosmin Truta, 2006
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/*
 *  ALGORITHM
 *
 *      The "deflation" process uses several Huffman trees. The more
 *      common source values are represented by shorter bit sequences.
 *
 *      Each code tree is stored in a compressed form which is itself
 * a Huffman encoding of the lengths of all the code strings (in
 * ascending order by source values).  The actual code strings are
 * reconstructed from the lengths in the inflate process, as described
 * in the deflate specification.
 *
 *  REFERENCES
 *
 *      Deutsch, L.P.,"'Deflate' Compressed Data Format Specification".
 *      Available in ftp.uu.net:/pub/archiving/zip/doc/deflate-1.1.doc
 *
 *      Storer, James A.
 *          Data Compression:  Methods and Theory, pp. 49-50.
 *          Computer Science Press, 1988.  ISBN 0-7167-8156-5.
 *
 *      Sedgewick, R.
 *          Algorithms, p290.
 *          Addison-Wesley, 1983. ISBN 0-201-06672-6.
 */

#include "zbuild.h"
#include "deflate.h"
#include "deflate_p.h"
#include "trees.h"
#include "trees_emit.h"
#include "trees_tbl.h"

/* The lengths of the bit length codes are sent in order of decreasing
 * probability, to avoid transmitting the lengths for unused bit length codes.
 */

/* ===========================================================================
 * Local data. These are initialized only once.
 */

struct static_tree_desc_s {
    const ct_data *static_tree; /* static tree or NULL */
    const int     *extra_bits;  /* extra bits for each code or NULL */
    int            extra_base;  /* base index for extra_bits */
    int            elems;       /* max number of elements in the tree */
    unsigned int   max_length;  /* max bit length for the codes */
};

static const static_tree_desc  static_l_desc =
{static_ltree, extra_lbits, LITERALS+1, L_CODES, MAX_BITS};

static const static_tree_desc  static_d_desc =
{static_dtree, extra_dbits, 0,          D_CODES, MAX_BITS};

static const static_tree_desc  static_bl_desc =
{(const ct_data *)0, extra_blbits, 0,   BL_CODES, MAX_BL_BITS};

/* ===========================================================================
 * Local (static) routines in this file.
 */

static void init_block       (deflate_state *s);
static inline void pqdownheap       (unsigned char *depth, int *heap, const int heap_len, ct_data *tree, int k);
static void build_tree       (deflate_state *s, tree_desc *desc);
static void gen_bitlen       (deflate_state *s, tree_desc *desc);
static void scan_tree        (deflate_state *s, ct_data *tree, int max_code);
static void send_tree        (deflate_state *s, ct_data *tree, int max_code);
static int  build_bl_tree    (deflate_state *s);
static void send_all_trees   (deflate_state *s, int lcodes, int dcodes, int blcodes);
static void compress_block   (deflate_state *s, const ct_data *ltree, const ct_data *dtree);
static int  detect_data_type (deflate_state *s);

/* ===========================================================================
 * Initialize the tree data structures for a new zlib stream.
 */
void Z_INTERNAL zng_tr_init(deflate_state *s) {
    s->l_desc.dyn_tree = s->dyn_ltree;
    s->l_desc.stat_desc = &static_l_desc;

    s->d_desc.dyn_tree = s->dyn_dtree;
    s->d_desc.stat_desc = &static_d_desc;

    s->bl_desc.dyn_tree = s->bl_tree;
    s->bl_desc.stat_desc = &static_bl_desc;

    s->bi_buf = 0;
    s->bi_valid = 0;
#ifdef ZLIB_DEBUG
    s->compressed_len = 0L;
    s->bits_sent = 0L;
#endif

    /* Initialize the first block of the first file: */
    init_block(s);
}

/* ===========================================================================
 * Initialize a new block.
 */
static void init_block(deflate_state *s) {
    int n; /* iterates over tree elements */

    /* Initialize the trees. */
    for (n = 0; n < L_CODES;  n++)
        s->dyn_ltree[n].Freq = 0;
    for (n = 0; n < D_CODES;  n++)
        s->dyn_dtree[n].Freq = 0;
    for (n = 0; n < BL_CODES; n++)
        s->bl_tree[n].Freq = 0;

    s->dyn_ltree[END_BLOCK].Freq = 1;
    s->opt_len = s->static_len = 0L;
    s->sym_next = s->matches = 0;
}

#define SMALLEST 1
/* Index within the heap array of least frequent node in the Huffman tree */


/* ===========================================================================
 * Compares to subtrees, using the tree depth as tie breaker when
 * the subtrees have equal frequency. This minimizes the worst case length.
 */
#define smaller(tree, n, m, depth) \
    (tree[n].Freq < tree[m].Freq || \
    (tree[n].Freq == tree[m].Freq && depth[n] <= depth[m]))

/* ===========================================================================
 * Remove the smallest element from the heap and recreate the heap with
 * one less element. Updates heap and heap_len. Used by build_tree().
 */
#define pqremove(s, depth, heap, tree, top) { \
    top = heap[SMALLEST]; \
    heap[SMALLEST] = heap[s->heap_len--]; \
    pqdownheap(depth, heap, s->heap_len, tree, SMALLEST); \
}

/* ===========================================================================
 * Restore the heap property by moving down the tree starting at node k,
 * exchanging a node with the smallest of its two sons if necessary, stopping
 * when the heap property is re-established (each father smaller than its
 * two sons). Used by build_tree().
 */
static inline void pqdownheap(unsigned char *depth, int *heap, const int heap_len, ct_data *tree, int k) {
    /* tree: the tree to restore */
    /* k: node to move down */
    int j = k << 1;  /* left son of k */
    const int v = heap[k];

    while (j <= heap_len) {
        /* Set j to the smallest of the two sons: */
        if (j < heap_len && smaller(tree, heap[j+1], heap[j], depth)) {
            j++;
        }
        /* Exit if v is smaller than both sons */
        if (smaller(tree, v, heap[j], depth))
            break;

        /* Exchange v with the smallest son */
        heap[k] = heap[j];
        k = j;

        /* And continue down the tree, setting j to the left son of k */
        j <<= 1;
    }
    heap[k] = v;
}

/* ===========================================================================
 * Construct one Huffman tree and assigns the code bit strings and lengths.
 * Update the total bit length for the current block.
 * IN assertion: the field freq is set for all tree elements.
 * OUT assertions: the fields len and code are set to the optimal bit length
 *     and corresponding code. The length opt_len is updated; static_len is
 *     also updated if stree is not null. The field max_code is set.
 */
static void build_tree(deflate_state *s, tree_desc *desc) {
    /* desc: the tree descriptor */
    unsigned char *depth  = s->depth;
    int *heap             = s->heap;
    ct_data *tree         = desc->dyn_tree;
    const ct_data *stree  = desc->stat_desc->static_tree;
    int elems             = desc->stat_desc->elems;
    int n, m;          /* iterate over heap elements */
    int max_code = -1; /* largest code with non zero frequency */
    int node;          /* new node being created */

    /* Construct the initial heap, with least frequent element in
     * heap[SMALLEST]. The sons of heap[n] are heap[2*n] and heap[2*n+1].
     * heap[0] is not used.
     */
    s->heap_len = 0;
    s->heap_max = HEAP_SIZE;

    for (n = 0; n < elems; n++) {
        if (tree[n].Freq != 0) {
            heap[++(s->heap_len)] = max_code = n;
            depth[n] = 0;
        } else {
            tree[n].Len = 0;
        }
    }

    /* The pkzip format requires that at least one distance code exists,
     * and that at least one bit should be sent even if there is only one
     * possible code. So to avoid special checks later on we force at least
     * two codes of non zero frequency.
     */
    while (s->heap_len < 2) {
        node = heap[++(s->heap_len)] = (max_code < 2 ? ++max_code : 0);
        tree[node].Freq = 1;
        depth[node] = 0;
        s->opt_len--;
        if (stree)
            s->static_len -= stree[node].Len;
        /* node is 0 or 1 so it does not have extra bits */
    }
    desc->max_code = max_code;

    /* The elements heap[heap_len/2+1 .. heap_len] are leaves of the tree,
     * establish sub-heaps of increasing lengths:
     */
    for (n = s->heap_len/2; n >= 1; n--)
        pqdownheap(depth, heap, s->heap_len, tree, n);

    /* Construct the Huffman tree by repeatedly combining the least two
     * frequent nodes.
     */
    node = elems;              /* next internal node of the tree */
    do {
        pqremove(s, depth, heap, tree, n);  /* n = node of least frequency */
        m = heap[SMALLEST]; /* m = node of next least frequency */

        heap[--(s->heap_max)] = n; /* keep the nodes sorted by frequency */
        heap[--(s->heap_max)] = m;

        /* Create a new node father of n and m */
        tree[node].Freq = tree[n].Freq + tree[m].Freq;
        depth[node] = (unsigned char)((depth[n] >= depth[m] ?
                                          depth[n] : depth[m]) + 1);
        tree[n].Dad = tree[m].Dad = (uint16_t)node;
#ifdef DUMP_BL_TREE
        if (tree == s->bl_tree) {
            fprintf(stderr, "\nnode %d(%d), sons %d(%d) %d(%d)",
                    node, tree[node].Freq, n, tree[n].Freq, m, tree[m].Freq);
        }
#endif
        /* and insert the new node in the heap */
        heap[SMALLEST] = node++;
        pqdownheap(depth, heap, s->heap_len, tree, SMALLEST);
    } while (s->heap_len >= 2);

    heap[--(s->heap_max)] = heap[SMALLEST];

    /* At this point, the fields freq and dad are set. We can now
     * generate the bit lengths.
     */
    gen_bitlen(s, (tree_desc *)desc);

    /* The field len is now set, we can generate the bit codes */
    gen_codes((ct_data *)tree, max_code, s->bl_count);
}

/* ===========================================================================
 * Compute the optimal bit lengths for a tree and update the total bit length
 * for the current block.
 * IN assertion: the fields freq and dad are set, heap[heap_max] and
 *    above are the tree nodes sorted by increasing frequency.
 * OUT assertions: the field len is set to the optimal bit length, the
 *     array bl_count contains the frequencies for each bit length.
 *     The length opt_len is updated; static_len is also updated if stree is
 *     not null. Used by build_tree().
 */
static void gen_bitlen(deflate_state *s, tree_desc *desc) {
    /* desc: the tree descriptor */
    ct_data *tree           = desc->dyn_tree;
    int max_code            = desc->max_code;
    const ct_data *stree    = desc->stat_desc->static_tree;
    const int *extra        = desc->stat_desc->extra_bits;
    int base                = desc->stat_desc->extra_base;
    unsigned int max_length = desc->stat_desc->max_length;
    int h;              /* heap index */
    int n, m;           /* iterate over the tree elements */
    unsigned int bits;  /* bit length */
    int xbits;          /* extra bits */
    uint16_t f;         /* frequency */
    int overflow = 0;   /* number of elements with bit length too large */

    for (bits = 0; bits <= MAX_BITS; bits++)
        s->bl_count[bits] = 0;

    /* In a first pass, compute the optimal bit lengths (which may
     * overflow in the case of the bit length tree).
     */
    tree[s->heap[s->heap_max]].Len = 0; /* root of the heap */

    for (h = s->heap_max + 1; h < HEAP_SIZE; h++) {
        n = s->heap[h];
        bits = tree[tree[n].Dad].Len + 1u;
        if (bits > max_length){
            bits = max_length;
            overflow++;
        }
        tree[n].Len = (uint16_t)bits;
        /* We overwrite tree[n].Dad which is no longer needed */

        if (n > max_code) /* not a leaf node */
            continue;

        s->bl_count[bits]++;
        xbits = 0;
        if (n >= base)
            xbits = extra[n-base];
        f = tree[n].Freq;
        s->opt_len += (unsigned long)f * (unsigned int)(bits + xbits);
        if (stree)
            s->static_len += (unsigned long)f * (unsigned int)(stree[n].Len + xbits);
    }
    if (overflow == 0)
        return;

    Tracev((stderr, "\nbit length overflow\n"));
    /* This happens for example on obj2 and pic of the Calgary corpus */

    /* Find the first bit length which could increase: */
    do {
        bits = max_length - 1;
        while (s->bl_count[bits] == 0)
            bits--;
        s->bl_count[bits]--;       /* move one leaf down the tree */
        s->bl_count[bits+1] += 2u; /* move one overflow item as its brother */
        s->bl_count[max_length]--;
        /* The brother of the overflow item also moves one step up,
         * but this does not affect bl_count[max_length]
         */
        overflow -= 2;
    } while (overflow > 0);

    /* Now recompute all bit lengths, scanning in increasing frequency.
     * h is still equal to HEAP_SIZE. (It is simpler to reconstruct all
     * lengths instead of fixing only the wrong ones. This idea is taken
     * from 'ar' written by Haruhiko Okumura.)
     */
    for (bits = max_length; bits != 0; bits--) {
        n = s->bl_count[bits];
        while (n != 0) {
            m = s->heap[--h];
            if (m > max_code)
                continue;
            if (tree[m].Len != bits) {
                Tracev((stderr, "code %d bits %d->%u\n", m, tree[m].Len, bits));
                s->opt_len += (unsigned long)(bits * tree[m].Freq);
                s->opt_len -= (unsigned long)(tree[m].Len * tree[m].Freq);
                tree[m].Len = (uint16_t)bits;
            }
            n--;
        }
    }
}

/* ===========================================================================
 * Generate the codes for a given tree and bit counts (which need not be
 * optimal).
 * IN assertion: the array bl_count contains the bit length statistics for
 * the given tree and the field len is set for all tree elements.
 * OUT assertion: the field code is set for all tree elements of non
 *     zero code length. Used by build_tree().
 */
Z_INTERNAL void gen_codes(ct_data *tree, int max_code, uint16_t *bl_count) {
    /* tree: the tree to decorate */
    /* max_code: largest code with non zero frequency */
    /* bl_count: number of codes at each bit length */
    uint16_t next_code[MAX_BITS+1];  /* next code value for each bit length */
    unsigned int code = 0;           /* running code value */
    int bits;                        /* bit index */
    int n;                           /* code index */

    /* The distribution counts are first used to generate the code values
     * without bit reversal.
     */
    for (bits = 1; bits <= MAX_BITS; bits++) {
        code = (code + bl_count[bits-1]) << 1;
        next_code[bits] = (uint16_t)code;
    }
    /* Check that the bit counts in bl_count are consistent. The last code
     * must be all ones.
     */
    Assert(code + bl_count[MAX_BITS]-1 == (1 << MAX_BITS)-1, "inconsistent bit counts");
    Tracev((stderr, "\ngen_codes: max_code %d ", max_code));

    for (n = 0;  n <= max_code; n++) {
        int len = tree[n].Len;
        if (len == 0)
            continue;
        /* Now reverse the bits */
        tree[n].Code = bi_reverse(next_code[len]++, len);

        Tracecv(tree != static_ltree, (stderr, "\nn %3d %c l %2d c %4x (%x) ",
             n, (isgraph(n & 0xff) ? n : ' '), len, tree[n].Code, next_code[len]-1));
    }
}

/* ===========================================================================
 * Scan a literal or distance tree to determine the frequencies of the codes
 * in the bit length tree.
 */
static void scan_tree(deflate_state *s, ct_data *tree, int max_code) {
    /* tree: the tree to be scanned */
    /* max_code: and its largest code of non zero frequency */
    int n;                     /* iterates over all tree elements */
    int prevlen = -1;          /* last emitted length */
    int curlen;                /* length of current code */
    int nextlen = tree[0].Len; /* length of next code */
    uint16_t count = 0;        /* repeat count of the current code */
    uint16_t max_count = 7;    /* max repeat count */
    uint16_t min_count = 4;    /* min repeat count */

    if (nextlen == 0)
        max_count = 138, min_count = 3;

    tree[max_code+1].Len = (uint16_t)0xffff; /* guard */

    for (n = 0; n <= max_code; n++) {
        curlen = nextlen;
        nextlen = tree[n+1].Len;
        if (++count < max_count && curlen == nextlen) {
            continue;
        } else if (count < min_count) {
            s->bl_tree[curlen].Freq += count;
        } else if (curlen != 0) {
            if (curlen != prevlen)
                s->bl_tree[curlen].Freq++;
            s->bl_tree[REP_3_6].Freq++;
        } else if (count <= 10) {
            s->bl_tree[REPZ_3_10].Freq++;
        } else {
            s->bl_tree[REPZ_11_138].Freq++;
        }
        count = 0;
        prevlen = curlen;
        if (nextlen == 0) {
            max_count = 138, min_count = 3;
        } else if (curlen == nextlen) {
            max_count = 6, min_count = 3;
        } else {
            max_count = 7, min_count = 4;
        }
    }
}

/* ===========================================================================
 * Send a literal or distance tree in compressed form, using the codes in
 * bl_tree.
 */
static void send_tree(deflate_state *s, ct_data *tree, int max_code) {
    /* tree: the tree to be scanned */
    /* max_code and its largest code of non zero frequency */
    int n;                     /* iterates over all tree elements */
    int prevlen = -1;          /* last emitted length */
    int curlen;                /* length of current code */
    int nextlen = tree[0].Len; /* length of next code */
    int count = 0;             /* repeat count of the current code */
    int max_count = 7;         /* max repeat count */
    int min_count = 4;         /* min repeat count */

    /* tree[max_code+1].Len = -1; */  /* guard already set */
    if (nextlen == 0)
        max_count = 138, min_count = 3;

    // Temp local variables
    uint32_t bi_valid = s->bi_valid;
    uint64_t bi_buf = s->bi_buf;

    for (n = 0; n <= max_code; n++) {
        curlen = nextlen;
        nextlen = tree[n+1].Len;
        if (++count < max_count && curlen == nextlen) {
            continue;
        } else if (count < min_count) {
            do {
                send_code(s, curlen, s->bl_tree, bi_buf, bi_valid);
            } while (--count != 0);

        } else if (curlen != 0) {
            if (curlen != prevlen) {
                send_code(s, curlen, s->bl_tree, bi_buf, bi_valid);
                count--;
            }
            Assert(count >= 3 && count <= 6, " 3_6?");
            send_code(s, REP_3_6, s->bl_tree, bi_buf, bi_valid);
            send_bits(s, count-3, 2, bi_buf, bi_valid);

        } else if (count <= 10) {
            send_code(s, REPZ_3_10, s->bl_tree, bi_buf, bi_valid);
            send_bits(s, count-3, 3, bi_buf, bi_valid);

        } else {
            send_code(s, REPZ_11_138, s->bl_tree, bi_buf, bi_valid);
            send_bits(s, count-11, 7, bi_buf, bi_valid);
        }
        count = 0;
        prevlen = curlen;
        if (nextlen == 0) {
            max_count = 138, min_count = 3;
        } else if (curlen == nextlen) {
            max_count = 6, min_count = 3;
        } else {
            max_count = 7, min_count = 4;
        }
    }

    // Store back temp variables
    s->bi_buf = bi_buf;
    s->bi_valid = bi_valid;
}

/* ===========================================================================
 * Construct the Huffman tree for the bit lengths and return the index in
 * bl_order of the last bit length code to send.
 */
static int build_bl_tree(deflate_state *s) {
    int max_blindex;  /* index of last bit length code of non zero freq */

    /* Determine the bit length frequencies for literal and distance trees */
    scan_tree(s, (ct_data *)s->dyn_ltree, s->l_desc.max_code);
    scan_tree(s, (ct_data *)s->dyn_dtree, s->d_desc.max_code);

    /* Build the bit length tree: */
    build_tree(s, (tree_desc *)(&(s->bl_desc)));
    /* opt_len now includes the length of the tree representations, except
     * the lengths of the bit lengths codes and the 5+5+4 bits for the counts.
     */

    /* Determine the number of bit length codes to send. The pkzip format
     * requires that at least 4 bit length codes be sent. (appnote.txt says
     * 3 but the actual value used is 4.)
     */
    for (max_blindex = BL_CODES-1; max_blindex >= 3; max_blindex--) {
        if (s->bl_tree[bl_order[max_blindex]].Len != 0)
            break;
    }
    /* Update opt_len to include the bit length tree and counts */
    s->opt_len += 3*((unsigned long)max_blindex+1) + 5+5+4;
    Tracev((stderr, "\ndyn trees: dyn %lu, stat %lu", s->opt_len, s->static_len));

    return max_blindex;
}

/* ===========================================================================
 * Send the header for a block using dynamic Huffman trees: the counts, the
 * lengths of the bit length codes, the literal tree and the distance tree.
 * IN assertion: lcodes >= 257, dcodes >= 1, blcodes >= 4.
 */
static void send_all_trees(deflate_state *s, int lcodes, int dcodes, int blcodes) {
    int rank;                    /* index in bl_order */

    Assert(lcodes >= 257 && dcodes >= 1 && blcodes >= 4, "not enough codes");
    Assert(lcodes <= L_CODES && dcodes <= D_CODES && blcodes <= BL_CODES, "too many codes");

    // Temp local variables
    uint32_t bi_valid = s->bi_valid;
    uint64_t bi_buf = s->bi_buf;

    Tracev((stderr, "\nbl counts: "));
    send_bits(s, lcodes-257, 5, bi_buf, bi_valid); /* not +255 as stated in appnote.txt */
    send_bits(s, dcodes-1,   5, bi_buf, bi_valid);
    send_bits(s, blcodes-4,  4, bi_buf, bi_valid); /* not -3 as stated in appnote.txt */
    for (rank = 0; rank < blcodes; rank++) {
        Tracev((stderr, "\nbl code %2u ", bl_order[rank]));
        send_bits(s, s->bl_tree[bl_order[rank]].Len, 3, bi_buf, bi_valid);
    }
    Tracev((stderr, "\nbl tree: sent %lu", s->bits_sent));

    // Store back temp variables
    s->bi_buf = bi_buf;
    s->bi_valid = bi_valid;

    send_tree(s, (ct_data *)s->dyn_ltree, lcodes-1); /* literal tree */
    Tracev((stderr, "\nlit tree: sent %lu", s->bits_sent));

    send_tree(s, (ct_data *)s->dyn_dtree, dcodes-1); /* distance tree */
    Tracev((stderr, "\ndist tree: sent %lu", s->bits_sent));
}

/* ===========================================================================
 * Send a stored block
 */
void Z_INTERNAL zng_tr_stored_block(deflate_state *s, char *buf, uint32_t stored_len, int last) {
    /* buf: input block */
    /* stored_len: length of input block */
    /* last: one if this is the last block for a file */
    zng_tr_emit_tree(s, STORED_BLOCK, last); /* send block type */
    zng_tr_emit_align(s);                    /* align on byte boundary */
    cmpr_bits_align(s);
    put_short(s, (uint16_t)stored_len);
    put_short(s, (uint16_t)~stored_len);
    cmpr_bits_add(s, 32);
    sent_bits_add(s, 32);
    if (stored_len) {
        memcpy(s->pending_buf + s->pending, (unsigned char *)buf, stored_len);
        s->pending += stored_len;
        cmpr_bits_add(s, stored_len << 3);
        sent_bits_add(s, stored_len << 3);
    }
}

/* ===========================================================================
 * Send one empty static block to give enough lookahead for inflate.
 * This takes 10 bits, of which 7 may remain in the bit buffer.
 */
void Z_INTERNAL zng_tr_align(deflate_state *s) {
    zng_tr_emit_tree(s, STATIC_TREES, 0);
    zng_tr_emit_end_block(s, static_ltree, 0);
    zng_tr_flush_bits(s);
}

/* ===========================================================================
 * Determine the best encoding for the current block: dynamic trees, static
 * trees or store, and write out the encoded block.
 */
void Z_INTERNAL zng_tr_flush_block(deflate_state *s, char *buf, uint32_t stored_len, int last) {
    /* buf: input block, or NULL if too old */
    /* stored_len: length of input block */
    /* last: one if this is the last block for a file */
    unsigned long opt_lenb, static_lenb; /* opt_len and static_len in bytes */
    int max_blindex = 0;  /* index of last bit length code of non zero freq */

    /* Build the Huffman trees unless a stored block is forced */
    if (UNLIKELY(s->sym_next == 0)) {
        /* Emit an empty static tree block with no codes */
        opt_lenb = static_lenb = 0;
        s->static_len = 7;
    } else if (s->level > 0) {
        /* Check if the file is binary or text */
        if (s->strm->data_type == Z_UNKNOWN)
            s->strm->data_type = detect_data_type(s);

        /* Construct the literal and distance trees */
        build_tree(s, (tree_desc *)(&(s->l_desc)));
        Tracev((stderr, "\nlit data: dyn %lu, stat %lu", s->opt_len, s->static_len));

        build_tree(s, (tree_desc *)(&(s->d_desc)));
        Tracev((stderr, "\ndist data: dyn %lu, stat %lu", s->opt_len, s->static_len));
        /* At this point, opt_len and static_len are the total bit lengths of
         * the compressed block data, excluding the tree representations.
         */

        /* Build the bit length tree for the above two trees, and get the index
         * in bl_order of the last bit length code to send.
         */
        max_blindex = build_bl_tree(s);

        /* Determine the best encoding. Compute the block lengths in bytes. */
        opt_lenb = (s->opt_len+3+7) >> 3;
        static_lenb = (s->static_len+3+7) >> 3;

        Tracev((stderr, "\nopt %lu(%lu) stat %lu(%lu) stored %u lit %u ",
                opt_lenb, s->opt_len, static_lenb, s->static_len, stored_len,
                s->sym_next / 3));

        if (static_lenb <= opt_lenb || s->strategy == Z_FIXED)
            opt_lenb = static_lenb;

    } else {
        Assert(buf != NULL, "lost buf");
        opt_lenb = static_lenb = stored_len + 5; /* force a stored block */
    }

    if (stored_len+4 <= opt_lenb && buf != NULL) {
        /* 4: two words for the lengths
         * The test buf != NULL is only necessary if LIT_BUFSIZE > WSIZE.
         * Otherwise we can't have processed more than WSIZE input bytes since
         * the last block flush, because compression would have been
         * successful. If LIT_BUFSIZE <= WSIZE, it is never too late to
         * transform a block into a stored block.
         */
        zng_tr_stored_block(s, buf, stored_len, last);

    } else if (static_lenb == opt_lenb) {
        zng_tr_emit_tree(s, STATIC_TREES, last);
        compress_block(s, (const ct_data *)static_ltree, (const ct_data *)static_dtree);
        cmpr_bits_add(s, s->static_len);
    } else {
        zng_tr_emit_tree(s, DYN_TREES, last);
        send_all_trees(s, s->l_desc.max_code+1, s->d_desc.max_code+1, max_blindex+1);
        compress_block(s, (const ct_data *)s->dyn_ltree, (const ct_data *)s->dyn_dtree);
        cmpr_bits_add(s, s->opt_len);
    }
    Assert(s->compressed_len == s->bits_sent, "bad compressed size");
    /* The above check is made mod 2^32, for files larger than 512 MB
     * and unsigned long implemented on 32 bits.
     */
    init_block(s);

    if (last) {
        zng_tr_emit_align(s);
    }
    Tracev((stderr, "\ncomprlen %lu(%lu) ", s->compressed_len>>3, s->compressed_len-7*last));
}

/* ===========================================================================
 * Send the block data compressed using the given Huffman trees
 */
static void compress_block(deflate_state *s, const ct_data *ltree, const ct_data *dtree) {
    /* ltree: literal tree */
    /* dtree: distance tree */
    unsigned dist;      /* distance of matched string */
    int lc;             /* match length or unmatched char (if dist == 0) */
    unsigned sx = 0;    /* running index in symbol buffers */

    /* Local pointers to avoid indirection */
    const unsigned int sym_next = s->sym_next;
#ifdef LIT_MEM
    uint16_t *d_buf = s->d_buf;
    unsigned char *l_buf = s->l_buf;
#else
    unsigned char *sym_buf = s->sym_buf;
#endif

    if (sym_next != 0) {
        do {
#ifdef LIT_MEM
            dist = d_buf[sx];
            lc = l_buf[sx++];
#else
            dist = sym_buf[sx++] & 0xff;
            dist += (unsigned)(sym_buf[sx++] & 0xff) << 8;
            lc = sym_buf[sx++];
#endif
            if (dist == 0) {
                zng_emit_lit(s, ltree, lc);
            } else {
                zng_emit_dist(s, ltree, dtree, lc, dist);
            } /* literal or match pair ? */

            /* Check for no overlay of pending_buf on needed symbols */
#ifdef LIT_MEM
            Assert(s->pending < 2 * (s->lit_bufsize + sx), "pending_buf overflow");
#else
            Assert(s->pending < s->lit_bufsize + sx, "pending_buf overflow");
#endif
        } while (sx < sym_next);
    }

    zng_emit_end_block(s, ltree, 0);
}

/* ===========================================================================
 * Check if the data type is TEXT or BINARY, using the following algorithm:
 * - TEXT if the two conditions below are satisfied:
 *    a) There are no non-portable control characters belonging to the
 *       "block list" (0..6, 14..25, 28..31).
 *    b) There is at least one printable character belonging to the
 *       "allow list" (9 {TAB}, 10 {LF}, 13 {CR}, 32..255).
 * - BINARY otherwise.
 * - The following partially-portable control characters form a
 *   "gray list" that is ignored in this detection algorithm:
 *   (7 {BEL}, 8 {BS}, 11 {VT}, 12 {FF}, 26 {SUB}, 27 {ESC}).
 * IN assertion: the fields Freq of dyn_ltree are set.
 */
static int detect_data_type(deflate_state *s) {
    /* block_mask is the bit mask of block-listed bytes
     * set bits 0..6, 14..25, and 28..31
     * 0xf3ffc07f = binary 11110011111111111100000001111111
     */
    unsigned long block_mask = 0xf3ffc07fUL;
    int n;

    /* Check for non-textual ("block-listed") bytes. */
    for (n = 0; n <= 31; n++, block_mask >>= 1)
        if ((block_mask & 1) && (s->dyn_ltree[n].Freq != 0))
            return Z_BINARY;

    /* Check for textual ("allow-listed") bytes. */
    if (s->dyn_ltree[9].Freq != 0 || s->dyn_ltree[10].Freq != 0 || s->dyn_ltree[13].Freq != 0)
        return Z_TEXT;
    for (n = 32; n < LITERALS; n++)
        if (s->dyn_ltree[n].Freq != 0)
            return Z_TEXT;

    /* There are no "block-listed" or "allow-listed" bytes:
     * this stream either is empty or has tolerated ("gray-listed") bytes only.
     */
    return Z_BINARY;
}

/* ===========================================================================
 * Flush the bit buffer, keeping at most 7 bits in it.
 */
void Z_INTERNAL zng_tr_flush_bits(deflate_state *s) {
    if (s->bi_valid >= 48) {
        put_uint32(s, (uint32_t)s->bi_buf);
        put_short(s, (uint16_t)(s->bi_buf >> 32));
        s->bi_buf >>= 48;
        s->bi_valid -= 48;
    } else if (s->bi_valid >= 32) {
        put_uint32(s, (uint32_t)s->bi_buf);
        s->bi_buf >>= 32;
        s->bi_valid -= 32;
    }
    if (s->bi_valid >= 16) {
        put_short(s, (uint16_t)s->bi_buf);
        s->bi_buf >>= 16;
        s->bi_valid -= 16;
    }
    if (s->bi_valid >= 8) {
        put_byte(s, s->bi_buf);
        s->bi_buf >>= 8;
        s->bi_valid -= 8;
    }
}
