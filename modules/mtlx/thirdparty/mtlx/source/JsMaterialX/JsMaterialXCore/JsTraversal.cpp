//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Traversal.h>
#include <MaterialXCore/Material.h>

#include <emscripten.h>
#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

#define BIND_ITERABLE_PROTOCOL(NAME)                                                       \
            .function("next", ems::optional_override([](mx::NAME &it) {                    \
                bool done = ++it == it.end();                                              \
                ems::val result = ems::val::object();                                      \
                result.set("done", done);                                                  \
                if (!done) result.set("value", *it);                                       \
                return result;                                                             \
            }));                                                                           \
            EM_ASM(                                                                        \
                Module[#NAME]['prototype'][Symbol.iterator] = function() { return this; }; \
            );                                                          

EMSCRIPTEN_BINDINGS(traversal)
{
    ems::class_<mx::Edge>("Edge")
        .smart_ptr_constructor("Edge", &std::make_shared<mx::Edge, mx::ElementPtr, mx::ElementPtr, mx::ElementPtr>)
        .function("equals", ems::optional_override([](mx::Edge &self, const mx::Edge &rhs) { return self == rhs; }))
        .function("notEquals", ems::optional_override([](mx::Edge &self, const mx::Edge &rhs) { return self != rhs; }))
        .function("lessThan", ems::optional_override([](mx::Edge &self, const mx::Edge &rhs) { return self < rhs; }))
        .function("notNull", &mx::Edge::operator bool)
        .function("getDownstreamElement", &mx::Edge::getDownstreamElement)
        .function("getConnectingElement", &mx::Edge::getConnectingElement)
        .function("getUpstreamElement", &mx::Edge::getUpstreamElement)
        .function("getName", &mx::Edge::getName);

    ems::class_<mx::TreeIterator>("TreeIterator")
        .smart_ptr_constructor("TreeIterator", &std::make_shared<mx::TreeIterator, mx::ElementPtr>)
        .function("getElement", &mx::TreeIterator::getElement)
        .function("getElementDepth", &mx::TreeIterator::getElementDepth)
        .function("setPruneSubtree", &mx::TreeIterator::setPruneSubtree)
        .function("getPruneSubtree", &mx::TreeIterator::getPruneSubtree)
        BIND_ITERABLE_PROTOCOL(TreeIterator)

    ems::class_<mx::GraphIterator>("GraphIterator")
        .smart_ptr_constructor("GraphIterator", &std::make_shared<mx::GraphIterator, mx::ElementPtr>)
        .function("getDownstreamElement", &mx::GraphIterator::getDownstreamElement)
        .function("getConnectingElement", &mx::GraphIterator::getConnectingElement)
        .function("getUpstreamElement", &mx::GraphIterator::getUpstreamElement)
        .function("getUpstreamIndex", &mx::GraphIterator::getUpstreamIndex)
        .function("getElementDepth", &mx::GraphIterator::getElementDepth)
        .function("getNodeDepth", &mx::GraphIterator::getNodeDepth)
        .function("setPruneSubgraph", &mx::GraphIterator::setPruneSubgraph)
        .function("getPruneSubgraph", &mx::GraphIterator::getPruneSubgraph)
        BIND_ITERABLE_PROTOCOL(GraphIterator)

    ems::class_<mx::InheritanceIterator>("InheritanceIterator")
        .smart_ptr_constructor("InheritanceIterator", &std::make_shared<mx::InheritanceIterator, mx::ConstElementPtr>)
        BIND_ITERABLE_PROTOCOL(InheritanceIterator)

    ems::constant("NULL_EDGE", mx::NULL_EDGE);
    ems::constant("NULL_TREE_ITERATOR", mx::NULL_TREE_ITERATOR);
    ems::constant("NULL_GRAPH_ITERATOR", mx::NULL_GRAPH_ITERATOR);
    ems::constant("NULL_INHERITANCE_ITERATOR", mx::NULL_INHERITANCE_ITERATOR);
}
