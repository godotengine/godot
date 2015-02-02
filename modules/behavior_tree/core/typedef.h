#ifndef BEHAVIOR_TREE_TYPEDEF_H
#define BEHAVIOR_TREE_TYPEDEF_H
#define BEHAVIOR_TREE_AS_GODOT_MODULE
#if defined(BEHAVIOR_TREE_AS_GODOT_MODULE)

#include "core/vector.h"
#define BT_STATIC_ASSERT(x, y)
#define BT_ASSERT(x)
#ifndef override
#define override
#endif

#else

#include <cassert>
#include <vector>
#include <algorithm>
#define BT_STATIC_ASSERT(x, y) static_assert(x, y)
#define BT_ASSERT(x) assert(x)

#endif

namespace BehaviorTree { struct Node; }

namespace BehaviorTree
{

enum E_State { BH_ERROR = 0, BH_SUCCESS = 1, BH_FAILURE = 2, BH_RUNNING = 3 };

typedef unsigned short IndexType;
const IndexType INDEX_TYPE_MAX = 0xffff;

struct NodeData
{
    union {
    IndexType begin;
    IndexType index;
    };
    IndexType end;
};

#if defined(BEHAVIOR_TREE_AS_GODOT_MODULE)

template<typename T>
class BTVector : public Vector<T>
{
public:
    T &      back()       { return Vector<T>::operator[](Vector<T>::size()-1); }
    T const& back() const { return Vector<T>::operator[](Vector<T>::size()-1); }

    void pop_back() { Vector<T>::resize(Vector<T>::size()-1); }

    void swap(BTVector& other) { other = *this; Vector<T>::clear(); }
};

template<typename COMPARATOR, typename T>
void sort(BTVector<T>& vector) { vector.template sort_custom<COMPARATOR>(); }

#else

template<typename T> class BTVector : public std::vector<T> {};

template<typename COMPARATOR, typename T>
void sort(BTVector<T>& vector) { std::sort(vector.begin(), vector.end(), COMPARATOR()); }

#endif

typedef BTVector<NodeData> BTStructure;
typedef BTVector<Node*> NodeList;


} /* BehaviorTree */ 

#endif
