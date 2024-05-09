#!/usr/bin/env python
'''
Unit tests for MaterialX Python.
'''

import math, os, unittest

import MaterialX as mx


#--------------------------------------------------------------------------------
_testValues = (1,
               True,
               1.0,
               mx.Color3(0.1, 0.2, 0.3),
               mx.Color4(0.1, 0.2, 0.3, 0.4),
               mx.Vector2(1.0, 2.0),
               mx.Vector3(1.0, 2.0, 3.0),
               mx.Vector4(1.0, 2.0, 3.0, 4.0),
               mx.Matrix33(0.0),
               mx.Matrix44(1.0),
               'value',
               [1, 2, 3],
               [False, True, False],
               [1.0, 2.0, 3.0],
               ['one', 'two', 'three'])

_fileDir = os.path.dirname(os.path.abspath(__file__))
_libraryDir = os.path.join(_fileDir, '../../libraries/stdlib/')
_exampleDir = os.path.join(_fileDir, '../../resources/Materials/Examples/')
_searchPath = _libraryDir + mx.PATH_LIST_SEPARATOR + _exampleDir

_libraryFilenames = ('stdlib_defs.mtlx',
                     'stdlib_ng.mtlx')
_exampleFilenames = ('StandardSurface/standard_surface_brass_tiled.mtlx',
                     'StandardSurface/standard_surface_brick_procedural.mtlx',
                     'StandardSurface/standard_surface_carpaint.mtlx',
                     'StandardSurface/standard_surface_marble_solid.mtlx',
                     'StandardSurface/standard_surface_look_brass_tiled.mtlx',
                     'UsdPreviewSurface/usd_preview_surface_gold.mtlx',
                     'UsdPreviewSurface/usd_preview_surface_plastic.mtlx')

_epsilon = 1e-4


#--------------------------------------------------------------------------------
class TestMaterialX(unittest.TestCase):
    def test_Globals(self):
        self.assertTrue(mx.__version__ == mx.getVersionString())

    def test_DataTypes(self):
        for value in _testValues:
            valueString = mx.getValueString(value)
            typeString = mx.getTypeString(value)
            newValue = mx.createValueFromStrings(valueString, typeString)
            self.assertTrue(newValue == value)
            self.assertTrue(mx.getTypeString(newValue) == typeString)

    def test_Vectors(self):
        v1 = mx.Vector3(1, 2, 3)
        v2 = mx.Vector3(2, 4, 6)

        # Indexing operators
        self.assertTrue(v1[2] == 3)
        v1[2] = 4
        self.assertTrue(v1[2] == 4)
        v1[2] = 3

        # Component-wise operators
        self.assertTrue(v2 + v1 == mx.Vector3(3, 6, 9))
        self.assertTrue(v2 - v1 == mx.Vector3(1, 2, 3))
        self.assertTrue(v2 * v1 == mx.Vector3(2, 8, 18))
        self.assertTrue(v2 / v1 == mx.Vector3(2, 2, 2))
        v2 += v1
        self.assertTrue(v2 == mx.Vector3(3, 6, 9))
        v2 -= v1
        self.assertTrue(v2 == mx.Vector3(2, 4, 6))
        v2 *= v1
        self.assertTrue(v2 == mx.Vector3(2, 8, 18))
        v2 /= v1
        self.assertTrue(v2 == mx.Vector3(2, 4, 6))
        self.assertTrue(v1 * 2 == v2)
        self.assertTrue(v2 / 2 == v1)

        # Geometric methods
        v3 = mx.Vector4(4)
        self.assertTrue(v3.getMagnitude() == 8)
        self.assertTrue(v3.getNormalized().getMagnitude() == 1)
        self.assertTrue(v1.dot(v2) == 28)
        self.assertTrue(v1.cross(v2) == mx.Vector3())

        # Vector copy
        v4 = v2.copy()
        self.assertTrue(v4 == v2)
        v4[0] += 1;
        self.assertTrue(v4 != v2)

    def test_Matrices(self):
        # Translation and scale
        trans = mx.Matrix44.createTranslation(mx.Vector3(1, 2, 3))
        scale = mx.Matrix44.createScale(mx.Vector3(2))
        self.assertTrue(trans == mx.Matrix44(1, 0, 0, 0,
                                             0, 1, 0, 0,
                                             0, 0, 1, 0,
                                             1, 2, 3, 1))
        self.assertTrue(scale == mx.Matrix44(2, 0, 0, 0,
                                             0, 2, 0, 0,
                                             0, 0, 2, 0,
                                             0, 0, 0, 1))

        # Indexing operators
        self.assertTrue(trans[3, 2] == 3)
        trans[3, 2] = 4
        self.assertTrue(trans[3, 2] == 4)
        trans[3, 2] = 3

        # Matrix methods
        self.assertTrue(trans.getTranspose() == mx.Matrix44(1, 0, 0, 1,
                                                            0, 1, 0, 2,
                                                            0, 0, 1, 3,
                                                            0, 0, 0, 1))
        self.assertTrue(scale.getTranspose() == scale)
        self.assertTrue(trans.getDeterminant() == 1)
        self.assertTrue(scale.getDeterminant() == 8)
        self.assertTrue(trans.getInverse() ==
                        mx.Matrix44.createTranslation(mx.Vector3(-1, -2, -3)))

        # Matrix product
        prod1 = trans * scale
        prod2 = scale * trans
        prod3 = trans * 2
        prod4 = trans
        prod4 *= scale
        self.assertTrue(prod1 == mx.Matrix44(2, 0, 0, 0,
                                             0, 2, 0, 0,
                                             0, 0, 2, 0,
                                             2, 4, 6, 1))
        self.assertTrue(prod2 == mx.Matrix44(2, 0, 0, 0,
                                             0, 2, 0, 0,
                                             0, 0, 2, 0,
                                             1, 2, 3, 1))
        self.assertTrue(prod3 == mx.Matrix44(2, 0, 0, 0,
                                             0, 2, 0, 0,
                                             0, 0, 2, 0,
                                             2, 4, 6, 2))
        self.assertTrue(prod4 == prod1)

        # Matrix division
        quot1 = prod1 / scale
        quot2 = prod2 / trans
        quot3 = prod3 / 2
        quot4 = quot1
        quot4 /= trans
        self.assertTrue(quot1 == trans)
        self.assertTrue(quot2 == scale)
        self.assertTrue(quot3 == trans)
        self.assertTrue(quot4 == mx.Matrix44.IDENTITY)

        # 2D rotation
        rot1 = mx.Matrix33.createRotation(math.pi / 2)
        rot2 = mx.Matrix33.createRotation(math.pi)
        self.assertTrue((rot1 * rot1).isEquivalent(rot2, _epsilon))
        self.assertTrue(rot2.isEquivalent(
            mx.Matrix33.createScale(mx.Vector2(-1)), _epsilon))
        self.assertTrue((rot2 * rot2).isEquivalent(mx.Matrix33.IDENTITY, _epsilon))

        # 3D rotation
        rotX = mx.Matrix44.createRotationX(math.pi)
        rotY = mx.Matrix44.createRotationY(math.pi)
        rotZ = mx.Matrix44.createRotationZ(math.pi)
        self.assertTrue((rotX * rotY).isEquivalent(
            mx.Matrix44.createScale(mx.Vector3(-1, -1, 1)), _epsilon))
        self.assertTrue((rotX * rotZ).isEquivalent(
            mx.Matrix44.createScale(mx.Vector3(-1, 1, -1)), _epsilon))
        self.assertTrue((rotY * rotZ).isEquivalent(
            mx.Matrix44.createScale(mx.Vector3(1, -1, -1)), _epsilon))

        # Matrix copy
        trans2 = trans.copy()
        self.assertTrue(trans2 == trans)
        trans2[0, 0] += 1;
        self.assertTrue(trans2 != trans)

    def test_BuildDocument(self):
        # Create a document.
        doc = mx.createDocument()

        # Create a node graph with constant and image sources.
        nodeGraph = doc.addNodeGraph()
        self.assertTrue(nodeGraph)
        self.assertRaises(LookupError, doc.addNodeGraph, nodeGraph.getName())
        constant = nodeGraph.addNode('constant')
        image = nodeGraph.addNode('image')

        # Connect sources to outputs.
        output1 = nodeGraph.addOutput()
        output2 = nodeGraph.addOutput()
        output1.setConnectedNode(constant)
        output2.setConnectedNode(image)
        self.assertTrue(output1.getConnectedNode() == constant)
        self.assertTrue(output2.getConnectedNode() == image)
        self.assertTrue(output1.getUpstreamElement() == constant)
        self.assertTrue(output2.getUpstreamElement() == image)

        # Set constant node color.
        color = mx.Color3(0.1, 0.2, 0.3)
        constant.setInputValue('value', color)
        self.assertTrue(constant.getInputValue('value') == color)

        # Set image node file.
        file = 'image1.tif'
        image.setInputValue('file', file, 'filename')
        self.assertTrue(image.getInputValue('file') == file)

        # Create a custom nodedef.
        nodeDef = doc.addNodeDef('nodeDef1', 'float', 'turbulence3d')
        nodeDef.setInputValue('octaves', 3)
        nodeDef.setInputValue('lacunarity', 2.0)
        nodeDef.setInputValue('gain', 0.5)

        # Reference the custom nodedef.
        custom = nodeGraph.addNode('turbulence3d', 'turbulence1', 'float')
        self.assertTrue(custom.getInputValue('octaves') == 3)
        custom.setInputValue('octaves', 5)
        self.assertTrue(custom.getInputValue('octaves') == 5)

        # Test scoped attributes.
        nodeGraph.setFilePrefix('folder/')
        nodeGraph.setColorSpace('lin_rec709')
        self.assertTrue(image.getInput('file').getResolvedValueString() == 'folder/image1.tif')
        self.assertTrue(constant.getActiveColorSpace() == 'lin_rec709')

        # Create a simple shader interface.
        simpleSrf = doc.addNodeDef('', 'surfaceshader', 'simpleSrf')
        simpleSrf.setInputValue('diffColor', mx.Color3(1.0))
        simpleSrf.setInputValue('specColor', mx.Color3(0.0))
        roughness = simpleSrf.setInputValue('roughness', 0.25)
        self.assertTrue(roughness.getIsUniform() == False)
        roughness.setIsUniform(True);
        self.assertTrue(roughness.getIsUniform() == True)

        # Instantiate shader and material nodes.
        shaderNode = doc.addNodeInstance(simpleSrf)
        materialNode = doc.addMaterialNode('', shaderNode)

        # Bind the diffuse color input to the constant color output.
        shaderNode.setConnectedOutput('diffColor', output1)
        self.assertTrue(shaderNode.getUpstreamElement() == constant)

        # Bind the roughness input to a value.
        instanceRoughness = shaderNode.setInputValue('roughness', 0.5)
        self.assertTrue(instanceRoughness.getValue() == 0.5)
        self.assertTrue(instanceRoughness.getDefaultValue() == 0.25)

        # Create a look for the material.
        look = doc.addLook()
        self.assertTrue(len(doc.getLooks()) == 1)

        # Bind the material to a geometry string.
        matAssign1 = look.addMaterialAssign("matAssign1", materialNode.getName())
        matAssign1.setGeom("/robot1")
        self.assertTrue(matAssign1.getReferencedMaterial() == materialNode)
        self.assertTrue(len(mx.getGeometryBindings(materialNode, "/robot1")) == 1)
        self.assertTrue(len(mx.getGeometryBindings(materialNode, "/robot2")) == 0)

        # Bind the material to a collection.
        matAssign2 = look.addMaterialAssign("matAssign2", materialNode.getName())
        collection = doc.addCollection()
        collection.setIncludeGeom("/robot2")
        collection.setExcludeGeom("/robot2/left_arm")
        matAssign2.setCollection(collection)
        self.assertTrue(len(mx.getGeometryBindings(materialNode, "/robot2")) == 1)
        self.assertTrue(len(mx.getGeometryBindings(materialNode, "/robot2/right_arm")) == 1)
        self.assertTrue(len(mx.getGeometryBindings(materialNode, "/robot2/left_arm")) == 0)

        # Create a property assignment.
        propertyAssign = look.addPropertyAssign()
        propertyAssign.setProperty("twosided")
        propertyAssign.setGeom("/robot1")
        propertyAssign.setValue(True)
        self.assertTrue(propertyAssign.getProperty() == "twosided")
        self.assertTrue(propertyAssign.getGeom() == "/robot1")
        self.assertTrue(propertyAssign.getValue() == True)

        # Create a property set assignment.
        propertySet = doc.addPropertySet()
        propertySet.setPropertyValue('matte', False)
        self.assertTrue(propertySet.getPropertyValue('matte') == False)
        propertySetAssign = look.addPropertySetAssign()
        propertySetAssign.setPropertySet(propertySet)
        propertySetAssign.setGeom('/robot1')
        self.assertTrue(propertySetAssign.getPropertySet() == propertySet)
        self.assertTrue(propertySetAssign.getGeom() == '/robot1')

        # Create a variant set.
        variantSet = doc.addVariantSet()
        variantSet.addVariant("original")
        variantSet.addVariant("damaged")
        self.assertTrue(len(variantSet.getVariants()) == 2)

        # Validate the document.
        valid, message = doc.validate()
        self.assertTrue(valid, 'Document returned validation warnings: ' + message)

        # Disconnect outputs from sources.
        output1.setConnectedNode(None)
        output2.setConnectedNode(None)
        self.assertTrue(output1.getConnectedNode() == None)
        self.assertTrue(output2.getConnectedNode() == None)

    def test_TraverseGraph(self):
        # Create a document.
        doc = mx.createDocument()

        # Create a node graph with the following structure:
        #
        # [image1] [constant]     [image2]
        #        \ /                 |
        #    [multiply]          [contrast]         [noise3d]
        #             \____________  |  ____________/
        #                          [mix]
        #                            |
        #                         [output]
        #
        nodeGraph = doc.addNodeGraph()
        image1 = nodeGraph.addNode('image')
        image2 = nodeGraph.addNode('image')
        constant = nodeGraph.addNode('constant')
        multiply = nodeGraph.addNode('multiply')
        contrast = nodeGraph.addNode('contrast')
        noise3d = nodeGraph.addNode('noise3d')
        mix = nodeGraph.addNode('mix')
        output = nodeGraph.addOutput()
        multiply.setConnectedNode('in1', image1)
        multiply.setConnectedNode('in2', constant)
        contrast.setConnectedNode('in', image2)
        mix.setConnectedNode('fg', multiply)
        mix.setConnectedNode('bg', contrast)
        mix.setConnectedNode('mask', noise3d)
        output.setConnectedNode(mix)

        # Validate the document.
        valid, message = doc.validate()
        self.assertTrue(valid, 'Document returned validation warnings: ' + message)

        # Traverse the document tree (implicit iterator).
        nodeCount = 0
        for elem in doc.traverseTree():
            if elem.isA(mx.Node):
                nodeCount += 1
        self.assertTrue(nodeCount == 7)

        # Traverse the document tree (explicit iterator).
        nodeCount = 0
        maxElementDepth = 0
        treeIter = doc.traverseTree()
        for elem in treeIter:
            if elem.isA(mx.Node):
                nodeCount += 1
            maxElementDepth = max(maxElementDepth, treeIter.getElementDepth())
        self.assertTrue(nodeCount == 7)
        self.assertTrue(maxElementDepth == 3)

        # Traverse the document tree (prune subtree).
        nodeCount = 0
        treeIter = doc.traverseTree()
        for elem in treeIter:
            if elem.isA(mx.Node):
                nodeCount += 1
            if elem.isA(mx.NodeGraph):
                treeIter.setPruneSubtree(True)
        self.assertTrue(nodeCount == 0)

        # Traverse upstream from the graph output (implicit iterator).
        nodeCount = 0
        for edge in output.traverseGraph():
            upstreamElem = edge.getUpstreamElement()
            connectingElem = edge.getConnectingElement()
            downstreamElem = edge.getDownstreamElement()
            if upstreamElem.isA(mx.Node):
                nodeCount += 1
                if downstreamElem.isA(mx.Node):
                    self.assertTrue(connectingElem.isA(mx.Input))
        self.assertTrue(nodeCount == 7)

        # Traverse upstream from the graph output (explicit iterator).
        nodeCount = 0
        maxElementDepth = 0
        maxNodeDepth = 0
        graphIter = output.traverseGraph()
        for edge in graphIter:
            upstreamElem = edge.getUpstreamElement()
            connectingElem = edge.getConnectingElement()
            downstreamElem = edge.getDownstreamElement()
            if upstreamElem.isA(mx.Node):
                nodeCount += 1
            maxElementDepth = max(maxElementDepth, graphIter.getElementDepth())
            maxNodeDepth = max(maxNodeDepth, graphIter.getNodeDepth())
        self.assertTrue(nodeCount == 7)
        self.assertTrue(maxElementDepth == 3)
        self.assertTrue(maxNodeDepth == 3)

        # Traverse upstream from the graph output (prune subgraph).
        nodeCount = 0
        graphIter = output.traverseGraph()
        for edge in graphIter:
            upstreamElem = edge.getUpstreamElement()
            connectingElem = edge.getConnectingElement()
            downstreamElem = edge.getDownstreamElement()
            if upstreamElem.isA(mx.Node):
                nodeCount += 1
                if upstreamElem.getCategory() == 'multiply':
                    graphIter.setPruneSubgraph(True)
        self.assertTrue(nodeCount == 5)

        # Create and detect a cycle.
        multiply.setConnectedNode('in2', mix)
        self.assertTrue(output.hasUpstreamCycle())
        self.assertFalse(doc.validate()[0])
        multiply.setConnectedNode('in2', constant)
        self.assertFalse(output.hasUpstreamCycle())
        self.assertTrue(doc.validate()[0])

        # Create and detect a loop.
        contrast.setConnectedNode('in', contrast)
        self.assertTrue(output.hasUpstreamCycle())
        self.assertFalse(doc.validate()[0])
        contrast.setConnectedNode('in', image2)
        self.assertFalse(output.hasUpstreamCycle())
        self.assertTrue(doc.validate()[0])

    def test_Xmlio(self):
        # Read the standard library.
        libs = []
        for filename in _libraryFilenames:
            lib = mx.createDocument()
            mx.readFromXmlFile(lib, filename, _searchPath)
            libs.append(lib)

        # Declare write predicate for write filter test
        def skipLibraryElement(elem):
            return not elem.hasSourceUri()

        # Read and validate each example document.
        for filename in _exampleFilenames:
            doc = mx.createDocument()
            mx.readFromXmlFile(doc, filename, _searchPath)
            valid, message = doc.validate()
            self.assertTrue(valid, filename + ' returned validation warnings: ' + message)

            # Copy the document.
            copiedDoc = doc.copy()
            self.assertTrue(copiedDoc == doc)
            copiedDoc.addLook()
            self.assertTrue(copiedDoc != doc)

            # Traverse the document tree.
            valueElementCount = 0
            for elem in doc.traverseTree():
                if elem.isA(mx.ValueElement):
                    valueElementCount += 1
            self.assertTrue(valueElementCount > 0)

            # Serialize to XML.
            writeOptions = mx.XmlWriteOptions()
            writeOptions.writeXIncludeEnable = False
            xmlString = mx.writeToXmlString(doc, writeOptions)

            # Verify that the serialized document is identical.
            writtenDoc = mx.createDocument()
            mx.readFromXmlString(writtenDoc, xmlString)
            self.assertTrue(writtenDoc == doc)

            # Combine document with the standard library.
            doc2 = doc.copy()
            for lib in libs:
                doc2.importLibrary(lib)
            self.assertTrue(doc2.validate()[0])

            # Write without definitions
            writeOptions.writeXIncludeEnable = False
            writeOptions.elementPredicate = skipLibraryElement
            result = mx.writeToXmlString(doc2, writeOptions)
            doc3 = mx.createDocument()
            mx.readFromXmlString(doc3, result)    
            self.assertTrue(len(doc3.getNodeDefs()) == 0)   

        # Read the same document twice, and verify that duplicate elements
        # are skipped.
        doc = mx.createDocument()
        filename = 'StandardSurface/standard_surface_carpaint.mtlx'
        mx.readFromXmlFile(doc, filename, _searchPath)
        mx.readFromXmlFile(doc, filename, _searchPath)
        self.assertTrue(doc.validate()[0])

#--------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
