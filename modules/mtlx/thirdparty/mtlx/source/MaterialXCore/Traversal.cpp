//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXCore/Traversal.h>

#include <MaterialXCore/Node.h>

MATERIALX_NAMESPACE_BEGIN

const Edge NULL_EDGE(nullptr, nullptr, nullptr);

const TreeIterator NULL_TREE_ITERATOR(nullptr);
const GraphIterator NULL_GRAPH_ITERATOR(nullptr);
const InheritanceIterator NULL_INHERITANCE_ITERATOR(nullptr);

//
// Edge methods
//

Edge::operator bool() const
{
    return *this != NULL_EDGE;
}

string Edge::getName() const
{
    return _elemConnect ? _elemConnect->getName() : EMPTY_STRING;
}

//
// TreeIterator methods
//

const TreeIterator& TreeIterator::end()
{
    return NULL_TREE_ITERATOR;
}

TreeIterator& TreeIterator::operator++()
{
    if (_holdCount)
    {
        _holdCount--;
        return *this;
    }

    if (!_prune && _elem && !_elem->getChildren().empty())
    {
        // Traverse to the first child of this element.
        _stack.emplace_back(_elem, 0);
        _elem = _elem->getChildren()[0];
        return *this;
    }
    _prune = false;

    while (true)
    {
        if (_stack.empty())
        {
            // Traversal is complete.
            _elem = ElementPtr();
            return *this;
        }

        // Traverse to our siblings.
        StackFrame& parentFrame = _stack.back();
        const vector<ElementPtr>& siblings = parentFrame.first->getChildren();
        if (parentFrame.second + 1 < siblings.size())
        {
            _elem = siblings[++parentFrame.second];
            return *this;
        }

        // Traverse to our parent's siblings.
        _stack.pop_back();
    }
}

//
// GraphIterator methods
//

size_t GraphIterator::getNodeDepth() const
{
    size_t nodeDepth = 0;
    for (ElementPtr elem : _pathElems)
    {
        if (elem->isA<Node>())
        {
            nodeDepth++;
        }
    }
    return nodeDepth;
}

const GraphIterator& GraphIterator::end()
{
    return NULL_GRAPH_ITERATOR;
}

GraphIterator& GraphIterator::operator++()
{
    if (_holdCount)
    {
        _holdCount--;
        return *this;
    }

    if (!_prune && _upstreamElem && _upstreamElem->getUpstreamEdgeCount())
    {
        // Traverse to the first upstream edge of this element.
        _stack.emplace_back(_upstreamElem, 0);
        Edge nextEdge = _upstreamElem->getUpstreamEdge(0);
        if (nextEdge && nextEdge.getUpstreamElement())
        {
            extendPathUpstream(nextEdge.getUpstreamElement(), nextEdge.getConnectingElement());
            return *this;
        }
    }
    _prune = false;

    while (true)
    {
        if (_upstreamElem)
        {
            returnPathDownstream(_upstreamElem);
        }

        if (_stack.empty())
        {
            // Traversal is complete.
            *this = GraphIterator::end();
            return *this;
        }

        // Traverse to our siblings.
        StackFrame& parentFrame = _stack.back();
        if (parentFrame.second + 1 < parentFrame.first->getUpstreamEdgeCount())
        {
            Edge nextEdge = parentFrame.first->getUpstreamEdge(++parentFrame.second);
            if (nextEdge && nextEdge.getUpstreamElement())
            {
                extendPathUpstream(nextEdge.getUpstreamElement(), nextEdge.getConnectingElement());
                return *this;
            }
            continue;
        }

        // Traverse to our parent's siblings.
        returnPathDownstream(parentFrame.first);
        _stack.pop_back();
    }

    return *this;
}

void GraphIterator::extendPathUpstream(ElementPtr upstreamElem, ElementPtr connectingElem)
{
    // Check for cycles.
    if (_pathElems.count(upstreamElem))
    {
        throw ExceptionFoundCycle("Encountered cycle at element: " + upstreamElem->asString());
    }

    // Extend the current path to the new element.
    _pathElems.insert(upstreamElem);
    _upstreamElem = upstreamElem;
    _connectingElem = connectingElem;
}

void GraphIterator::returnPathDownstream(ElementPtr upstreamElem)
{
    _pathElems.erase(upstreamElem);
    _upstreamElem = ElementPtr();
    _connectingElem = ElementPtr();
}

//
// InheritanceIterator methods
//

const InheritanceIterator& InheritanceIterator::end()
{
    return NULL_INHERITANCE_ITERATOR;
}

InheritanceIterator& InheritanceIterator::operator++()
{
    if (_holdCount)
    {
        _holdCount--;
        return *this;
    }

    if (_elem)
    {
        ElementPtr super = _elem->getInheritsFrom();
        if (super && super->getCategory() != _elem->getCategory())
        {
            super = nullptr;
        }
        if (super)
        {
            // Check for cycles.
            if (_pathElems.count(super))
            {
                throw ExceptionFoundCycle("Encountered cycle at element: " + super->asString());
            }
            _pathElems.insert(super);
        }
        _elem = super;
    }
    return *this;
}

MATERIALX_NAMESPACE_END
