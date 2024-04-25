//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Traversal.h>

#include <MaterialXCore/Material.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyTraversal(py::module& mod)
{
    py::class_<mx::Edge>(mod, "Edge")
        .def("getDownstreamElement", &mx::Edge::getDownstreamElement)
        .def("getConnectingElement", &mx::Edge::getConnectingElement)
        .def("getUpstreamElement", &mx::Edge::getUpstreamElement)
        .def("getName", &mx::Edge::getName);

    py::class_<mx::TreeIterator>(mod, "TreeIterator")
        .def("getElement", &mx::TreeIterator::getElement)
        .def("getElementDepth", &mx::TreeIterator::getElementDepth)
        .def("setPruneSubtree", &mx::TreeIterator::setPruneSubtree)
        .def("getPruneSubtree", &mx::TreeIterator::getPruneSubtree)
        .def("__iter__", [](mx::TreeIterator& it) -> mx::TreeIterator&
            {
                return it.begin(1);
            })
        .def("__next__", [](mx::TreeIterator& it)
            {
                if (++it == it.end())
                    throw py::stop_iteration();
                return *it;
            });

    py::class_<mx::GraphIterator>(mod, "GraphIterator")
        .def("getDownstreamElement", &mx::GraphIterator::getDownstreamElement)
        .def("getConnectingElement", &mx::GraphIterator::getConnectingElement)
        .def("getUpstreamElement", &mx::GraphIterator::getUpstreamElement)
        .def("getUpstreamIndex", &mx::GraphIterator::getUpstreamIndex)
        .def("getElementDepth", &mx::GraphIterator::getElementDepth)
        .def("getNodeDepth", &mx::GraphIterator::getNodeDepth)
        .def("setPruneSubgraph", &mx::GraphIterator::setPruneSubgraph)
        .def("getPruneSubgraph", &mx::GraphIterator::getPruneSubgraph)
        .def("__iter__", [](mx::GraphIterator& it) -> mx::GraphIterator&
            {
                return it.begin(1);
            })
        .def("__next__", [](mx::GraphIterator& it)
            {
                if (++it == it.end())
                    throw py::stop_iteration();
                return *it;
            });

    py::class_<mx::InheritanceIterator>(mod, "InheritanceIterator")
        .def("__iter__", [](mx::InheritanceIterator& it) -> mx::InheritanceIterator&
            {
                return it.begin(1);
            })
        .def("__next__", [](mx::InheritanceIterator& it)
            {
                if (++it == it.end())
                    throw py::stop_iteration();
                return *it;
            });

    py::register_exception<mx::ExceptionFoundCycle>(mod, "ExceptionFoundCycle");
}
