// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Utilities for building and looking up Huffman trees.
//
// Author: Urvang Joshi (urvang@google.com)

#include <assert.h>
#include <stdlib.h>
#include "./huffman.h"
#include "../utils/utils.h"
#include "../webp/format_constants.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define NON_EXISTENT_SYMBOL (-1)

static void TreeNodeInit(HuffmanTreeNode* const node) {
  node->children_ = -1;   // means: 'unassigned so far'
}

static int NodeIsEmpty(const HuffmanTreeNode* const node) {
  return (node->children_ < 0);
}

static int IsFull(const HuffmanTree* const tree) {
  return (tree->num_nodes_ == tree->max_nodes_);
}

static void AssignChildren(HuffmanTree* const tree,
                           HuffmanTreeNode* const node) {
  HuffmanTreeNode* const children = tree->root_ + tree->num_nodes_;
  node->children_ = (int)(children - node);
  assert(children - node == (int)(children - node));
  tree->num_nodes_ += 2;
  TreeNodeInit(children + 0);
  TreeNodeInit(children + 1);
}

static int TreeInit(HuffmanTree* const tree, int num_leaves) {
  assert(tree != NULL);
  if (num_leaves == 0) return 0;
  // We allocate maximum possible nodes in the tree at once.
  // Note that a Huffman tree is a full binary tree; and in a full binary tree
  // with L leaves, the total number of nodes N = 2 * L - 1.
  tree->max_nodes_ = 2 * num_leaves - 1;
  tree->root_ = (HuffmanTreeNode*)WebPSafeMalloc((uint64_t)tree->max_nodes_,
                                                 sizeof(*tree->root_));
  if (tree->root_ == NULL) return 0;
  TreeNodeInit(tree->root_);  // Initialize root.
  tree->num_nodes_ = 1;
  return 1;
}

void HuffmanTreeRelease(HuffmanTree* const tree) {
  if (tree != NULL) {
    free(tree->root_);
    tree->root_ = NULL;
    tree->max_nodes_ = 0;
    tree->num_nodes_ = 0;
  }
}

int HuffmanCodeLengthsToCodes(const int* const code_lengths,
                              int code_lengths_size, int* const huff_codes) {
  int symbol;
  int code_len;
  int code_length_hist[MAX_ALLOWED_CODE_LENGTH + 1] = { 0 };
  int curr_code;
  int next_codes[MAX_ALLOWED_CODE_LENGTH + 1] = { 0 };
  int max_code_length = 0;

  assert(code_lengths != NULL);
  assert(code_lengths_size > 0);
  assert(huff_codes != NULL);

  // Calculate max code length.
  for (symbol = 0; symbol < code_lengths_size; ++symbol) {
    if (code_lengths[symbol] > max_code_length) {
      max_code_length = code_lengths[symbol];
    }
  }
  if (max_code_length > MAX_ALLOWED_CODE_LENGTH) return 0;

  // Calculate code length histogram.
  for (symbol = 0; symbol < code_lengths_size; ++symbol) {
    ++code_length_hist[code_lengths[symbol]];
  }
  code_length_hist[0] = 0;

  // Calculate the initial values of 'next_codes' for each code length.
  // next_codes[code_len] denotes the code to be assigned to the next symbol
  // of code length 'code_len'.
  curr_code = 0;
  next_codes[0] = -1;  // Unused, as code length = 0 implies code doesn't exist.
  for (code_len = 1; code_len <= max_code_length; ++code_len) {
    curr_code = (curr_code + code_length_hist[code_len - 1]) << 1;
    next_codes[code_len] = curr_code;
  }

  // Get symbols.
  for (symbol = 0; symbol < code_lengths_size; ++symbol) {
    if (code_lengths[symbol] > 0) {
      huff_codes[symbol] = next_codes[code_lengths[symbol]]++;
    } else {
      huff_codes[symbol] = NON_EXISTENT_SYMBOL;
    }
  }
  return 1;
}

static int TreeAddSymbol(HuffmanTree* const tree,
                         int symbol, int code, int code_length) {
  HuffmanTreeNode* node = tree->root_;
  const HuffmanTreeNode* const max_node = tree->root_ + tree->max_nodes_;
  while (code_length-- > 0) {
    if (node >= max_node) {
      return 0;
    }
    if (NodeIsEmpty(node)) {
      if (IsFull(tree)) return 0;    // error: too many symbols.
      AssignChildren(tree, node);
    } else if (HuffmanTreeNodeIsLeaf(node)) {
      return 0;  // leaf is already occupied.
    }
    node += node->children_ + ((code >> code_length) & 1);
  }
  if (NodeIsEmpty(node)) {
    node->children_ = 0;      // turn newly created node into a leaf.
  } else if (!HuffmanTreeNodeIsLeaf(node)) {
    return 0;   // trying to assign a symbol to already used code.
  }
  node->symbol_ = symbol;  // Add symbol in this node.
  return 1;
}

int HuffmanTreeBuildImplicit(HuffmanTree* const tree,
                             const int* const code_lengths,
                             int code_lengths_size) {
  int symbol;
  int num_symbols = 0;
  int root_symbol = 0;

  assert(tree != NULL);
  assert(code_lengths != NULL);

  // Find out number of symbols and the root symbol.
  for (symbol = 0; symbol < code_lengths_size; ++symbol) {
    if (code_lengths[symbol] > 0) {
      // Note: code length = 0 indicates non-existent symbol.
      ++num_symbols;
      root_symbol = symbol;
    }
  }

  // Initialize the tree. Will fail for num_symbols = 0
  if (!TreeInit(tree, num_symbols)) return 0;

  // Build tree.
  if (num_symbols == 1) {  // Trivial case.
    const int max_symbol = code_lengths_size;
    if (root_symbol < 0 || root_symbol >= max_symbol) {
      HuffmanTreeRelease(tree);
      return 0;
    }
    return TreeAddSymbol(tree, root_symbol, 0, 0);
  } else {  // Normal case.
    int ok = 0;

    // Get Huffman codes from the code lengths.
    int* const codes =
        (int*)WebPSafeMalloc((uint64_t)code_lengths_size, sizeof(*codes));
    if (codes == NULL) goto End;

    if (!HuffmanCodeLengthsToCodes(code_lengths, code_lengths_size, codes)) {
      goto End;
    }

    // Add symbols one-by-one.
    for (symbol = 0; symbol < code_lengths_size; ++symbol) {
      if (code_lengths[symbol] > 0) {
        if (!TreeAddSymbol(tree, symbol, codes[symbol], code_lengths[symbol])) {
          goto End;
        }
      }
    }
    ok = 1;
 End:
    free(codes);
    ok = ok && IsFull(tree);
    if (!ok) HuffmanTreeRelease(tree);
    return ok;
  }
}

int HuffmanTreeBuildExplicit(HuffmanTree* const tree,
                             const int* const code_lengths,
                             const int* const codes,
                             const int* const symbols, int max_symbol,
                             int num_symbols) {
  int ok = 0;
  int i;

  assert(tree != NULL);
  assert(code_lengths != NULL);
  assert(codes != NULL);
  assert(symbols != NULL);

  // Initialize the tree. Will fail if num_symbols = 0.
  if (!TreeInit(tree, num_symbols)) return 0;

  // Add symbols one-by-one.
  for (i = 0; i < num_symbols; ++i) {
    if (codes[i] != NON_EXISTENT_SYMBOL) {
      if (symbols[i] < 0 || symbols[i] >= max_symbol) {
        goto End;
      }
      if (!TreeAddSymbol(tree, symbols[i], codes[i], code_lengths[i])) {
        goto End;
      }
    }
  }
  ok = 1;
 End:
  ok = ok && IsFull(tree);
  if (!ok) HuffmanTreeRelease(tree);
  return ok;
}

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
