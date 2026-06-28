/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009,2025           *
 * by the Xiph.Org Foundation and contributors                      *
 * https://www.xiph.org/                                            *
 *                                                                  *
 ********************************************************************

  function:

 ********************************************************************/

#include <stdlib.h>
#include <string.h>
#include <ogg/ogg.h>
#include "huffdec.h"
#include "decint.h"



/*Instead of storing every branching in the tree, subtrees can be collapsed
   into one node, with a table of size 1<<nbits pointing directly to its
   descedents nbits levels down.
  This allows more than one bit to be read at a time, and avoids following all
   the intermediate branches with next to no increased code complexity once
   the collapsed tree has been built.
  We do _not_ require that a subtree be complete to be collapsed, but instead
   store duplicate pointers in the table, and record the actual depth of the
   node below its parent.
  This tells us the number of bits to advance the stream after reaching it.

  This turns out to be equivalent to the method described in \cite{Hash95},
   without the requirement that codewords be sorted by length.
  If the codewords were sorted by length (so-called ``canonical-codes''), they
   could be decoded much faster via either Lindell and Moffat's approach or
   Hashemian's Condensed Huffman Code approach, the latter of which has an
   extremely small memory footprint.
  We can't use Choueka et al.'s finite state machine approach, which is
   extremely fast, because we can't allow multiple symbols to be output at a
   time; the codebook can and does change between symbols.
  It also has very large memory requirements, which impairs cache coherency.

  We store the tree packed in an array of 16-bit integers (words).
  Each node consists of a single word, followed consecutively by two or more
   indices of its children.
  Let n be the value of this first word.
  This is the number of bits that need to be read to traverse the node, and
   must be positive.
  1<<n entries follow in the array, each an index to a child node.
  If the child is positive, then it is the index of another internal node in
   the table.
  If the child is negative or zero, then it is a leaf node.
  These are stored directly in the child pointer to save space, since they only
   require a single word.
  If a leaf node would have been encountered before reading n bits, then it is
   duplicated the necessary number of times in this table.
  Leaf nodes pack both a token value and their actual depth in the tree.
  The token in the leaf node is (-leaf&255).
  The number of bits that need to be consumed to reach the leaf, starting from
   the current node, is (-leaf>>8).

  @ARTICLE{Hash95,
    author="Reza Hashemian",
    title="Memory Efficient and High-Speed Search {Huffman} Coding",
    journal="{IEEE} Transactions on Communications",
    volume=43,
    number=10,
    pages="2576--2581",
    month=Oct,
    year=1995
  }*/



/*The map from external spec-defined tokens to internal tokens.
  This is constructed so that any extra bits read with the original token value
   can be masked off the least significant bits of its internal token index.
  In addition, all of the tokens which require additional extra bits are placed
   at the start of the list, and grouped by type.
  OC_DCT_REPEAT_RUN3_TOKEN is placed first, as it is an extra-special case, so
   giving it index 0 may simplify comparisons on some architectures.
  These requirements require some substantial reordering.*/
static const unsigned char OC_DCT_TOKEN_MAP[TH_NDCT_TOKENS]={
  /*OC_DCT_EOB1_TOKEN (0 extra bits)*/
  15,
  /*OC_DCT_EOB2_TOKEN (0 extra bits)*/
  16,
  /*OC_DCT_EOB3_TOKEN (0 extra bits)*/
  17,
  /*OC_DCT_REPEAT_RUN0_TOKEN (2 extra bits)*/
  88,
  /*OC_DCT_REPEAT_RUN1_TOKEN (3 extra bits)*/
  80,
  /*OC_DCT_REPEAT_RUN2_TOKEN (4 extra bits)*/
   1,
  /*OC_DCT_REPEAT_RUN3_TOKEN (12 extra bits)*/
   0,
  /*OC_DCT_SHORT_ZRL_TOKEN (3 extra bits)*/
  48,
  /*OC_DCT_ZRL_TOKEN (6 extra bits)*/
  14,
  /*OC_ONE_TOKEN (0 extra bits)*/
  56,
  /*OC_MINUS_ONE_TOKEN (0 extra bits)*/
  57,
  /*OC_TWO_TOKEN (0 extra bits)*/
  58,
  /*OC_MINUS_TWO_TOKEN (0 extra bits)*/
  59,
  /*OC_DCT_VAL_CAT2 (1 extra bit)*/
  60,
  62,
  64,
  66,
  /*OC_DCT_VAL_CAT3 (2 extra bits)*/
  68,
  /*OC_DCT_VAL_CAT4 (3 extra bits)*/
  72,
  /*OC_DCT_VAL_CAT5 (4 extra bits)*/
   2,
  /*OC_DCT_VAL_CAT6 (5 extra bits)*/
   4,
  /*OC_DCT_VAL_CAT7 (6 extra bits)*/
   6,
  /*OC_DCT_VAL_CAT8 (10 extra bits)*/
   8,
  /*OC_DCT_RUN_CAT1A (1 extra bit)*/
  18,
  20,
  22,
  24,
  26,
  /*OC_DCT_RUN_CAT1B (3 extra bits)*/
  32,
  /*OC_DCT_RUN_CAT1C (4 extra bits)*/
  12,
  /*OC_DCT_RUN_CAT2A (2 extra bits)*/
  28,
  /*OC_DCT_RUN_CAT2B (3 extra bits)*/
  40
};

/*The log base 2 of number of internal tokens associated with each of the spec
   tokens (i.e., how many of the extra bits are folded into the token value).
  Increasing the maximum value beyond 3 will enlarge the amount of stack
   required for tree construction.*/
static const unsigned char OC_DCT_TOKEN_MAP_LOG_NENTRIES[TH_NDCT_TOKENS]={
  0,0,0,2,3,0,0,3,0,0,0,0,0,1,1,1,1,2,3,1,1,1,2,1,1,1,1,1,3,1,2,3
};


/*The size a lookup table is allowed to grow to relative to the number of
   unique nodes it contains.
  E.g., if OC_HUFF_SLUSH is 4, then at most 75% of the space in the tree is
   wasted (1/4 of the space must be used).
  Larger numbers can decode tokens with fewer read operations, while smaller
   numbers may save more space.
  With a sample file:
  32233473 read calls are required when no tree collapsing is done (100.0%).
  19269269 read calls are required when OC_HUFF_SLUSH is 1 (59.8%).
  11144969 read calls are required when OC_HUFF_SLUSH is 2 (34.6%).
  10538563 read calls are required when OC_HUFF_SLUSH is 4 (32.7%).
  10192578 read calls are required when OC_HUFF_SLUSH is 8 (31.6%).
  Since a value of 2 gets us the vast majority of the speed-up with only a
   small amount of wasted memory, this is what we use.
  This value must be less than 128, or you could create a tree with more than
   32767 entries, which would overflow the 16-bit words used to index it.*/
#define OC_HUFF_SLUSH (2)
/*The root of the tree is on the fast path, and a larger value here is more
   beneficial than elsewhere in the tree.
  7 appears to give the best performance, trading off between increased use of
   the single-read fast path and cache footprint for the tables, though
   obviously this will depend on your cache size.
  Using 7 here, the VP3 tables are about twice as large compared to using 2.*/
#define OC_ROOT_HUFF_SLUSH (7)



/*Unpacks a Huffman codebook.
  _opb:    The buffer to unpack from.
  _tokens: Stores a list of internal tokens, in the order they were found in
            the codebook, and the lengths of their corresponding codewords.
           This is enough to completely define the codebook, while minimizing
            stack usage and avoiding temporary allocations (for platforms
            where free() is a no-op).
  Return: The number of internal tokens in the codebook, or a negative value
   on error.*/
int oc_huff_tree_unpack(oc_pack_buf *_opb,unsigned char _tokens[256][2]){
  ogg_uint32_t code;
  int          len;
  int          ntokens;
  int          nleaves;
  code=0;
  len=ntokens=nleaves=0;
  for(;;){
    long bits;
    bits=oc_pack_read1(_opb);
    /*Only process nodes so long as there's more bits in the buffer.*/
    if(oc_pack_bytes_left(_opb)<0)return TH_EBADHEADER;
    /*Read an internal node:*/
    if(!bits){
      len++;
      /*Don't allow codewords longer than 32 bits.*/
      if(len>32)return TH_EBADHEADER;
    }
    /*Read a leaf node:*/
    else{
      ogg_uint32_t code_bit;
      int          neb;
      int          nentries;
      int          token;
      /*Don't allow more than 32 spec-tokens per codebook.*/
      if(++nleaves>32)return TH_EBADHEADER;
      bits=oc_pack_read(_opb,OC_NDCT_TOKEN_BITS);
      neb=OC_DCT_TOKEN_MAP_LOG_NENTRIES[bits];
      token=OC_DCT_TOKEN_MAP[bits];
      nentries=1<<neb;
      while(nentries-->0){
        _tokens[ntokens][0]=(unsigned char)token++;
        _tokens[ntokens][1]=(unsigned char)(len+neb);
        ntokens++;
      }
      if(len<=0)break;
      code_bit=0x80000000U>>len-1;
      while(len>0&&(code&code_bit)){
        code^=code_bit;
        code_bit<<=1;
        len--;
      }
      if(len<=0)break;
      code|=code_bit;
    }
  }
  return ntokens;
}

/*Count how many tokens would be required to fill a subtree at depth _depth.
  _tokens: A list of internal tokens, in the order they are found in the
            codebook, and the lengths of their corresponding codewords.
  _depth:  The depth of the desired node in the corresponding tree structure.
  Return: The number of tokens that belong to that subtree.*/
static int oc_huff_subtree_tokens(unsigned char _tokens[][2],int _depth){
  ogg_uint32_t code;
  int          ti;
  code=0;
  ti=0;
  do{
    if(_tokens[ti][1]-_depth<32)code+=0x80000000U>>_tokens[ti++][1]-_depth;
    else{
      /*Because of the expanded internal tokens, we can have codewords as long
         as 35 bits.
        A single recursion here is enough to advance past them.*/
      code++;
      ti+=oc_huff_subtree_tokens(_tokens+ti,_depth+31);
    }
  }
  while(code<0x80000000U);
  return ti;
}

/*Compute the number of bits to use for a collapsed tree node at the given
   depth.
  _tokens:  A list of internal tokens, in the order they are found in the
             codebook, and the lengths of their corresponding codewords.
  _ntokens: The number of tokens corresponding to this tree node.
  _depth:   The depth of this tree node.
  Return: The number of bits to use for a collapsed tree node rooted here.
          This is always at least one, even if this was a leaf node.*/
static int oc_huff_tree_collapse_depth(unsigned char _tokens[][2],
 int _ntokens,int _depth){
  int got_leaves;
  int loccupancy;
  int occupancy;
  int slush;
  int nbits;
  int best_nbits;
  slush=_depth>0?OC_HUFF_SLUSH:OC_ROOT_HUFF_SLUSH;
  /*It's legal to have a tree with just a single node, which requires no bits
     to decode and always returns the same token.
    However, no encoder actually does this (yet).
    To avoid a special case in oc_huff_token_decode(), we force the number of
     lookahead bits to be at least one.
    This will produce a tree that looks ahead one bit and then advances the
     stream zero bits.*/
  nbits=1;
  occupancy=2;
  got_leaves=1;
  do{
    int ti;
    if(got_leaves)best_nbits=nbits;
    nbits++;
    got_leaves=0;
    loccupancy=occupancy;
    for(occupancy=ti=0;ti<_ntokens;occupancy++){
      if(_tokens[ti][1]<_depth+nbits)ti++;
      else if(_tokens[ti][1]==_depth+nbits){
        got_leaves=1;
        ti++;
      }
      else ti+=oc_huff_subtree_tokens(_tokens+ti,_depth+nbits);
    }
  }
  while(occupancy>loccupancy&&occupancy*slush>=1<<nbits);
  return best_nbits;
}

/*Determines the size in words of a Huffman tree node that represents a
   subtree of depth _nbits.
  _nbits: The depth of the subtree.
          This must be greater than zero.
  Return: The number of words required to store the node.*/
static size_t oc_huff_node_size(int _nbits){
  return 1+(1<<_nbits);
}

/*Produces a collapsed-tree representation of the given token list.
  _tree: The storage for the collapsed Huffman tree.
         This may be NULL to compute the required storage size instead of
          constructing the tree.
  _tokens:  A list of internal tokens, in the order they are found in the
             codebook, and the lengths of their corresponding codewords.
  _ntokens: The number of tokens corresponding to this tree node.
  Return: The number of words required to store the tree.*/
static size_t oc_huff_tree_collapse(ogg_int16_t *_tree,
 unsigned char _tokens[][2],int _ntokens){
  ogg_int16_t   node[34];
  unsigned char depth[34];
  unsigned char last[34];
  size_t        ntree;
  int           ti;
  int           l;
  depth[0]=0;
  last[0]=(unsigned char)(_ntokens-1);
  ntree=0;
  ti=0;
  l=0;
  do{
    int nbits;
    nbits=oc_huff_tree_collapse_depth(_tokens+ti,last[l]+1-ti,depth[l]);
    node[l]=(ogg_int16_t)ntree;
    ntree+=oc_huff_node_size(nbits);
    if(_tree!=NULL)_tree[node[l]++]=(ogg_int16_t)nbits;
    do{
      while(ti<=last[l]&&_tokens[ti][1]<=depth[l]+nbits){
        if(_tree!=NULL){
          ogg_int16_t leaf;
          int         nentries;
          nentries=1<<depth[l]+nbits-_tokens[ti][1];
          leaf=(ogg_int16_t)-(_tokens[ti][1]-depth[l]<<8|_tokens[ti][0]);
          while(nentries-->0)_tree[node[l]++]=leaf;
        }
        ti++;
      }
      if(ti<=last[l]){
        /*We need to recurse*/
        depth[l+1]=(unsigned char)(depth[l]+nbits);
        if(_tree!=NULL)_tree[node[l]++]=(ogg_int16_t)ntree;
        l++;
        last[l]=
         (unsigned char)(ti+oc_huff_subtree_tokens(_tokens+ti,depth[l])-1);
        break;
      }
      /*Pop back up a level of recursion.*/
      else if(l-->0)nbits=depth[l+1]-depth[l];
    }
    while(l>=0);
  }
  while(l>=0);
  return ntree;
}

/*Unpacks a set of Huffman trees, and reduces them to a collapsed
   representation.
  _opb:   The buffer to unpack the trees from.
  _nodes: The table to fill with the Huffman trees.
  Return: 0 on success, or a negative value on error.
          The caller is responsible for cleaning up any partially initialized
           _nodes on failure.*/
int oc_huff_trees_unpack(oc_pack_buf *_opb,
 ogg_int16_t *_nodes[TH_NHUFFMAN_TABLES]){
  int i;
  for(i=0;i<TH_NHUFFMAN_TABLES;i++){
    unsigned char  tokens[256][2];
    int            ntokens;
    ogg_int16_t   *tree;
    size_t         size;
    /*Unpack the full tree into a temporary buffer.*/
    ntokens=oc_huff_tree_unpack(_opb,tokens);
    if(ntokens<0)return ntokens;
    /*Figure out how big the collapsed tree will be and allocate space for it.*/
    size=oc_huff_tree_collapse(NULL,tokens,ntokens);
    /*This should never happen; if it does it means you set OC_HUFF_SLUSH or
       OC_ROOT_HUFF_SLUSH too large.*/
    if(size>32767)return TH_EIMPL;
    tree=(ogg_int16_t *)_ogg_malloc(size*sizeof(*tree));
    if(tree==NULL)return TH_EFAULT;
    /*Construct the collapsed the tree.*/
    oc_huff_tree_collapse(tree,tokens,ntokens);
    _nodes[i]=tree;
  }
  return 0;
}

/*Determines the size in words of a Huffman subtree.
  _tree: The complete Huffman tree.
  _node: The index of the root of the desired subtree.
  Return: The number of words required to store the tree.*/
static size_t oc_huff_tree_size(const ogg_int16_t *_tree,int _node){
  size_t size;
  int    nchildren;
  int    n;
  int    i;
  n=_tree[_node];
  size=oc_huff_node_size(n);
  nchildren=1<<n;
  i=0;
  do{
    int child;
    child=_tree[_node+i+1];
    if(child<=0)i+=1<<n-(-child>>8);
    else{
      size+=oc_huff_tree_size(_tree,child);
      i++;
    }
  }
  while(i<nchildren);
  return size;
}

/*Makes a copy of the given set of Huffman trees.
  _dst: The array to store the copy in.
  _src: The array of trees to copy.*/
int oc_huff_trees_copy(ogg_int16_t *_dst[TH_NHUFFMAN_TABLES],
 const ogg_int16_t *const _src[TH_NHUFFMAN_TABLES]){
  int i;
  for(i=0;i<TH_NHUFFMAN_TABLES;i++){
    size_t size;
    size=oc_huff_tree_size(_src[i],0);
    _dst[i]=(ogg_int16_t *)_ogg_malloc(size*sizeof(*_dst[i]));
    if(_dst[i]==NULL){
      while(i-->0)_ogg_free(_dst[i]);
      return TH_EFAULT;
    }
    memcpy(_dst[i],_src[i],size*sizeof(*_dst[i]));
  }
  return 0;
}

/*Frees the memory used by a set of Huffman trees.
  _nodes: The array of trees to free.*/
void oc_huff_trees_clear(ogg_int16_t *_nodes[TH_NHUFFMAN_TABLES]){
  int i;
  for(i=0;i<TH_NHUFFMAN_TABLES;i++)_ogg_free(_nodes[i]);
}


/*Unpacks a single token using the given Huffman tree.
  _opb:  The buffer to unpack the token from.
  _node: The tree to unpack the token with.
  Return: The token value.*/
int oc_huff_token_decode_c(oc_pack_buf *_opb,const ogg_int16_t *_tree){
  const unsigned char *ptr;
  const unsigned char *stop;
  oc_pb_window         window;
  int                  available;
  long                 bits;
  int                  node;
  int                  n;
  ptr=_opb->ptr;
  window=_opb->window;
  stop=_opb->stop;
  available=_opb->bits;
  node=0;
  for(;;){
    n=_tree[node];
    if(n>available){
      unsigned shift;
      shift=OC_PB_WINDOW_SIZE-available;
      do{
        /*We don't bother setting eof because we won't check for it after we've
           started decoding DCT tokens.*/
        if(ptr>=stop){
          shift=(unsigned)-OC_LOTS_OF_BITS;
          break;
        }
        shift-=8;
        window|=(oc_pb_window)*ptr++<<shift;
      }
      while(shift>=8);
      /*Note: We never request more than 24 bits, so there's no need to fill in
         the last partial byte here.*/
      available=OC_PB_WINDOW_SIZE-shift;
    }
    bits=window>>OC_PB_WINDOW_SIZE-n;
    node=_tree[node+1+bits];
    if(node<=0)break;
    window<<=n;
    available-=n;
  }
  node=-node;
  n=node>>8;
  window<<=n;
  available-=n;
  _opb->ptr=ptr;
  _opb->window=window;
  _opb->bits=available;
  return node&255;
}
