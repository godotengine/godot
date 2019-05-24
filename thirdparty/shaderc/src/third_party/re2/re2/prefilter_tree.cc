// Copyright 2009 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "re2/prefilter_tree.h"

#include <stddef.h>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "util/util.h"
#include "util/logging.h"
#include "util/strutil.h"
#include "re2/prefilter.h"
#include "re2/re2.h"

namespace re2 {

static const bool ExtraDebug = false;

PrefilterTree::PrefilterTree()
    : compiled_(false),
      min_atom_len_(3) {
}

PrefilterTree::PrefilterTree(int min_atom_len)
    : compiled_(false),
      min_atom_len_(min_atom_len) {
}

PrefilterTree::~PrefilterTree() {
  for (size_t i = 0; i < prefilter_vec_.size(); i++)
    delete prefilter_vec_[i];

  for (size_t i = 0; i < entries_.size(); i++)
    delete entries_[i].parents;
}

void PrefilterTree::Add(Prefilter* prefilter) {
  if (compiled_) {
    LOG(DFATAL) << "Add called after Compile.";
    return;
  }
  if (prefilter != NULL && !KeepNode(prefilter)) {
    delete prefilter;
    prefilter = NULL;
  }

  prefilter_vec_.push_back(prefilter);
}

void PrefilterTree::Compile(std::vector<std::string>* atom_vec) {
  if (compiled_) {
    LOG(DFATAL) << "Compile called already.";
    return;
  }

  // Some legacy users of PrefilterTree call Compile() before
  // adding any regexps and expect Compile() to have no effect.
  if (prefilter_vec_.empty())
    return;

  compiled_ = true;

  // TODO(junyer): Use std::unordered_set<Prefilter*> instead?
  NodeMap nodes;
  AssignUniqueIds(&nodes, atom_vec);

  // Identify nodes that are too common among prefilters and are
  // triggering too many parents. Then get rid of them if possible.
  // Note that getting rid of a prefilter node simply means they are
  // no longer necessary for their parent to trigger; that is, we do
  // not miss out on any regexps triggering by getting rid of a
  // prefilter node.
  for (size_t i = 0; i < entries_.size(); i++) {
    StdIntMap* parents = entries_[i].parents;
    if (parents->size() > 8) {
      // This one triggers too many things. If all the parents are AND
      // nodes and have other things guarding them, then get rid of
      // this trigger. TODO(vsri): Adjust the threshold appropriately,
      // make it a function of total number of nodes?
      bool have_other_guard = true;
      for (StdIntMap::iterator it = parents->begin();
           it != parents->end(); ++it) {
        have_other_guard = have_other_guard &&
            (entries_[it->first].propagate_up_at_count > 1);
      }

      if (have_other_guard) {
        for (StdIntMap::iterator it = parents->begin();
             it != parents->end(); ++it)
          entries_[it->first].propagate_up_at_count -= 1;

        parents->clear();  // Forget the parents
      }
    }
  }

  if (ExtraDebug)
    PrintDebugInfo(&nodes);
}

Prefilter* PrefilterTree::CanonicalNode(NodeMap* nodes, Prefilter* node) {
  std::string node_string = NodeString(node);
  std::map<std::string, Prefilter*>::iterator iter = nodes->find(node_string);
  if (iter == nodes->end())
    return NULL;
  return (*iter).second;
}

std::string PrefilterTree::NodeString(Prefilter* node) const {
  // Adding the operation disambiguates AND/OR/atom nodes.
  std::string s = StringPrintf("%d", node->op()) + ":";
  if (node->op() == Prefilter::ATOM) {
    s += node->atom();
  } else {
    for (size_t i = 0; i < node->subs()->size(); i++) {
      if (i > 0)
        s += ',';
      s += StringPrintf("%d", (*node->subs())[i]->unique_id());
    }
  }
  return s;
}

bool PrefilterTree::KeepNode(Prefilter* node) const {
  if (node == NULL)
    return false;

  switch (node->op()) {
    default:
      LOG(DFATAL) << "Unexpected op in KeepNode: " << node->op();
      return false;

    case Prefilter::ALL:
    case Prefilter::NONE:
      return false;

    case Prefilter::ATOM:
      return node->atom().size() >= static_cast<size_t>(min_atom_len_);

    case Prefilter::AND: {
      int j = 0;
      std::vector<Prefilter*>* subs = node->subs();
      for (size_t i = 0; i < subs->size(); i++)
        if (KeepNode((*subs)[i]))
          (*subs)[j++] = (*subs)[i];
        else
          delete (*subs)[i];

      subs->resize(j);
      return j > 0;
    }

    case Prefilter::OR:
      for (size_t i = 0; i < node->subs()->size(); i++)
        if (!KeepNode((*node->subs())[i]))
          return false;
      return true;
  }
}

void PrefilterTree::AssignUniqueIds(NodeMap* nodes,
                                    std::vector<std::string>* atom_vec) {
  atom_vec->clear();

  // Build vector of all filter nodes, sorted topologically
  // from top to bottom in v.
  std::vector<Prefilter*> v;

  // Add the top level nodes of each regexp prefilter.
  for (size_t i = 0; i < prefilter_vec_.size(); i++) {
    Prefilter* f = prefilter_vec_[i];
    if (f == NULL)
      unfiltered_.push_back(static_cast<int>(i));

    // We push NULL also on to v, so that we maintain the
    // mapping of index==regexpid for level=0 prefilter nodes.
    v.push_back(f);
  }

  // Now add all the descendant nodes.
  for (size_t i = 0; i < v.size(); i++) {
    Prefilter* f = v[i];
    if (f == NULL)
      continue;
    if (f->op() == Prefilter::AND || f->op() == Prefilter::OR) {
      const std::vector<Prefilter*>& subs = *f->subs();
      for (size_t j = 0; j < subs.size(); j++)
        v.push_back(subs[j]);
    }
  }

  // Identify unique nodes.
  int unique_id = 0;
  for (int i = static_cast<int>(v.size()) - 1; i >= 0; i--) {
    Prefilter *node = v[i];
    if (node == NULL)
      continue;
    node->set_unique_id(-1);
    Prefilter* canonical = CanonicalNode(nodes, node);
    if (canonical == NULL) {
      // Any further nodes that have the same node string
      // will find this node as the canonical node.
      nodes->emplace(NodeString(node), node);
      if (node->op() == Prefilter::ATOM) {
        atom_vec->push_back(node->atom());
        atom_index_to_id_.push_back(unique_id);
      }
      node->set_unique_id(unique_id++);
    } else {
      node->set_unique_id(canonical->unique_id());
    }
  }
  entries_.resize(nodes->size());

  // Create parent StdIntMap for the entries.
  for (int i = static_cast<int>(v.size()) - 1; i >= 0; i--) {
    Prefilter* prefilter = v[i];
    if (prefilter == NULL)
      continue;

    if (CanonicalNode(nodes, prefilter) != prefilter)
      continue;

    Entry* entry = &entries_[prefilter->unique_id()];
    entry->parents = new StdIntMap();
  }

  // Fill the entries.
  for (int i = static_cast<int>(v.size()) - 1; i >= 0; i--) {
    Prefilter* prefilter = v[i];
    if (prefilter == NULL)
      continue;

    if (CanonicalNode(nodes, prefilter) != prefilter)
      continue;

    Entry* entry = &entries_[prefilter->unique_id()];

    switch (prefilter->op()) {
      default:
      case Prefilter::ALL:
        LOG(DFATAL) << "Unexpected op: " << prefilter->op();
        return;

      case Prefilter::ATOM:
        entry->propagate_up_at_count = 1;
        break;

      case Prefilter::OR:
      case Prefilter::AND: {
        std::set<int> uniq_child;
        for (size_t j = 0; j < prefilter->subs()->size(); j++) {
          Prefilter* child = (*prefilter->subs())[j];
          Prefilter* canonical = CanonicalNode(nodes, child);
          if (canonical == NULL) {
            LOG(DFATAL) << "Null canonical node";
            return;
          }
          int child_id = canonical->unique_id();
          uniq_child.insert(child_id);
          // To the child, we want to add to parent indices.
          Entry* child_entry = &entries_[child_id];
          if (child_entry->parents->find(prefilter->unique_id()) ==
              child_entry->parents->end()) {
            (*child_entry->parents)[prefilter->unique_id()] = 1;
          }
        }
        entry->propagate_up_at_count = prefilter->op() == Prefilter::AND
                                           ? static_cast<int>(uniq_child.size())
                                           : 1;

        break;
      }
    }
  }

  // For top level nodes, populate regexp id.
  for (size_t i = 0; i < prefilter_vec_.size(); i++) {
    if (prefilter_vec_[i] == NULL)
      continue;
    int id = CanonicalNode(nodes, prefilter_vec_[i])->unique_id();
    DCHECK_LE(0, id);
    Entry* entry = &entries_[id];
    entry->regexps.push_back(static_cast<int>(i));
  }
}

// Functions for triggering during search.
void PrefilterTree::RegexpsGivenStrings(
    const std::vector<int>& matched_atoms,
    std::vector<int>* regexps) const {
  regexps->clear();
  if (!compiled_) {
    // Some legacy users of PrefilterTree call Compile() before
    // adding any regexps and expect Compile() to have no effect.
    // This kludge is a counterpart to that kludge.
    if (prefilter_vec_.empty())
      return;

    LOG(ERROR) << "RegexpsGivenStrings called before Compile.";
    for (size_t i = 0; i < prefilter_vec_.size(); i++)
      regexps->push_back(static_cast<int>(i));
  } else {
    IntMap regexps_map(static_cast<int>(prefilter_vec_.size()));
    std::vector<int> matched_atom_ids;
    for (size_t j = 0; j < matched_atoms.size(); j++)
      matched_atom_ids.push_back(atom_index_to_id_[matched_atoms[j]]);
    PropagateMatch(matched_atom_ids, &regexps_map);
    for (IntMap::iterator it = regexps_map.begin();
         it != regexps_map.end();
         ++it)
      regexps->push_back(it->index());

    regexps->insert(regexps->end(), unfiltered_.begin(), unfiltered_.end());
  }
  std::sort(regexps->begin(), regexps->end());
}

void PrefilterTree::PropagateMatch(const std::vector<int>& atom_ids,
                                   IntMap* regexps) const {
  IntMap count(static_cast<int>(entries_.size()));
  IntMap work(static_cast<int>(entries_.size()));
  for (size_t i = 0; i < atom_ids.size(); i++)
    work.set(atom_ids[i], 1);
  for (IntMap::iterator it = work.begin(); it != work.end(); ++it) {
    const Entry& entry = entries_[it->index()];
    // Record regexps triggered.
    for (size_t i = 0; i < entry.regexps.size(); i++)
      regexps->set(entry.regexps[i], 1);
    int c;
    // Pass trigger up to parents.
    for (StdIntMap::iterator it = entry.parents->begin();
         it != entry.parents->end();
         ++it) {
      int j = it->first;
      const Entry& parent = entries_[j];
      // Delay until all the children have succeeded.
      if (parent.propagate_up_at_count > 1) {
        if (count.has_index(j)) {
          c = count.get_existing(j) + 1;
          count.set_existing(j, c);
        } else {
          c = 1;
          count.set_new(j, c);
        }
        if (c < parent.propagate_up_at_count)
          continue;
      }
      // Trigger the parent.
      work.set(j, 1);
    }
  }
}

// Debugging help.
void PrefilterTree::PrintPrefilter(int regexpid) {
  LOG(ERROR) << DebugNodeString(prefilter_vec_[regexpid]);
}

void PrefilterTree::PrintDebugInfo(NodeMap* nodes) {
  LOG(ERROR) << "#Unique Atoms: " << atom_index_to_id_.size();
  LOG(ERROR) << "#Unique Nodes: " << entries_.size();

  for (size_t i = 0; i < entries_.size(); i++) {
    StdIntMap* parents = entries_[i].parents;
    const std::vector<int>& regexps = entries_[i].regexps;
    LOG(ERROR) << "EntryId: " << i
               << " N: " << parents->size() << " R: " << regexps.size();
    for (StdIntMap::iterator it = parents->begin(); it != parents->end(); ++it)
      LOG(ERROR) << it->first;
  }
  LOG(ERROR) << "Map:";
  for (std::map<std::string, Prefilter*>::const_iterator iter = nodes->begin();
       iter != nodes->end(); ++iter)
    LOG(ERROR) << "NodeId: " << (*iter).second->unique_id()
               << " Str: " << (*iter).first;
}

std::string PrefilterTree::DebugNodeString(Prefilter* node) const {
  std::string node_string = "";
  if (node->op() == Prefilter::ATOM) {
    DCHECK(!node->atom().empty());
    node_string += node->atom();
  } else {
    // Adding the operation disambiguates AND and OR nodes.
    node_string +=  node->op() == Prefilter::AND ? "AND" : "OR";
    node_string += "(";
    for (size_t i = 0; i < node->subs()->size(); i++) {
      if (i > 0)
        node_string += ',';
      node_string += StringPrintf("%d", (*node->subs())[i]->unique_id());
      node_string += ":";
      node_string += DebugNodeString((*node->subs())[i]);
    }
    node_string += ")";
  }
  return node_string;
}

}  // namespace re2
