#!/usr/bin/env python
'''
Unit tests for shader generation in MaterialX Python.
'''

import os, unittest

import MaterialX as mx
import MaterialX.PyMaterialXGenShader as mx_gen_shader
import MaterialX.PyMaterialXGenOsl as mx_gen_osl

class TestGenShader(unittest.TestCase):
    def test_ShaderInterface(self):
        doc = mx.createDocument()
        searchPath = mx.getDefaultDataSearchPath()
        mx.loadLibraries(mx.getDefaultDataLibraryFolders(), searchPath, doc)

        exampleName = u"shader_interface"

        # Create a nodedef taking three color3 and producing another color3
        nodeDef = doc.addNodeDef("ND_foo", "color3", "foo")
        fooInputA = nodeDef.addInput("a", "color3")
        fooInputB = nodeDef.addInput("b", "color3")
        fooOutput = nodeDef.getOutput("out")
        fooInputA.setValue(mx.Color3(1.0, 1.0, 0.0))
        fooInputB.setValue(mx.Color3(0.8, 0.1, 0.1))

        # Create an implementation graph for the nodedef performing
        # a multiplication of the three colors.
        nodeGraph = doc.addNodeGraph("IMP_foo")
        nodeGraph.setAttribute("nodedef", nodeDef.getName())

        output = nodeGraph.addOutput(fooOutput.getName(), "color3")
        mult1 = nodeGraph.addNode("multiply", "mult1", "color3")
        in1 = mult1.addInput("in1", "color3")
        in1.setInterfaceName(fooInputA.getName())
        in2 = mult1.addInput("in2", "color3")
        in2.setInterfaceName(fooInputB.getName())
        output.setConnectedNode(mult1)

        doc.addNode("foo", "foo1", "color3")
        output = doc.addOutput("foo_test", "color3");
        output.setNodeName("foo1");
        output.setAttribute("output", "o");

        # Test for target
        targetDefs = doc.getTargetDefs()
        self.assertTrue(len(targetDefs))
        shadergen = mx_gen_osl.OslShaderGenerator.create()
        target = shadergen.getTarget()
        foundTarget = next((
            t for t in targetDefs
            if t.getName() == target), None)
        self.assertTrue(foundTarget)
        context = mx_gen_shader.GenContext(shadergen)
        context.registerSourceCodeSearchPath(searchPath)

        # Test generator with complete mode
        context.getOptions().shaderInterfaceType = mx_gen_shader.ShaderInterfaceType.SHADER_INTERFACE_COMPLETE;
        shader = shadergen.generate(exampleName, output, context);
        self.assertTrue(shader)
        self.assertTrue(len(shader.getSourceCode(mx_gen_shader.PIXEL_STAGE)) > 0)

        ps = shader.getStage(mx_gen_shader.PIXEL_STAGE);
        uniforms = ps.getUniformBlock(mx_gen_osl.OSL_UNIFORMS)
        self.assertTrue(uniforms.size() == 2)

        outputs = ps.getOutputBlock(mx_gen_osl.OSL_OUTPUTS)
        self.assertTrue(outputs.size() == 1)
        self.assertTrue(outputs[0].getName() == output.getName())

        file = open(shader.getName() + "_complete.osl", "w+")
        file.write(shader.getSourceCode(mx_gen_shader.PIXEL_STAGE))
        file.close()
        os.remove(shader.getName() + "_complete.osl");

        # Test generator with reduced mode
        context.getOptions().shaderInterfaceType = mx_gen_shader.ShaderInterfaceType.SHADER_INTERFACE_REDUCED;
        shader = shadergen.generate(exampleName, output, context);
        self.assertTrue(shader)
        self.assertTrue(len(shader.getSourceCode(mx_gen_shader.PIXEL_STAGE)) > 0)

        ps = shader.getStage(mx_gen_shader.PIXEL_STAGE);
        uniforms = ps.getUniformBlock(mx_gen_osl.OSL_UNIFORMS)
        self.assertTrue(uniforms.size() == 0)

        outputs = ps.getOutputBlock(mx_gen_osl.OSL_OUTPUTS)
        self.assertTrue(outputs.size() == 1)
        self.assertTrue(outputs[0].getName() == output.getName())

        file = open(shader.getName() + "_reduced.osl", "w+")
        file.write(shader.getSourceCode(mx_gen_shader.PIXEL_STAGE))
        file.close()
        os.remove(shader.getName() + "_reduced.osl");

        # Define a custom attribute
        customAttribute = doc.addAttributeDef("AD_attribute_node_name");
        self.assertIsNotNone(customAttribute)
        customAttribute.setType("string");
        customAttribute.setAttrName("node_name");
        customAttribute.setExportable(True);

        # Define a nodedef referencing the custom attribute.
        stdSurfNodeDef = doc.getNodeDef("ND_standard_surface_surfaceshader");
        self.assertIsNotNone(stdSurfNodeDef)
        stdSurfNodeDef.setAttribute("node_name", "Standard_Surface_Number_1");
        self.assertTrue(stdSurfNodeDef.getAttribute("node_name") == "Standard_Surface_Number_1")
        stdSurf1 = doc.addNodeInstance(stdSurfNodeDef, "standardSurface1");
        self.assertIsNotNone(stdSurf1)

        # Register shader metadata
        shadergen.registerShaderMetadata(doc, context);

        # Generate and test that attribute is in the code
        context.getOptions().shaderInterfaceType = mx_gen_shader.ShaderInterfaceType.SHADER_INTERFACE_COMPLETE;
        shader = shadergen.generate(stdSurf1.getName(), stdSurf1, context);
        self.assertIsNotNone(shader)
        code = shader.getSourceCode(mx_gen_shader.PIXEL_STAGE)
        self.assertTrue('Standard_Surface_Number_1' in code)
        self.assertTrue('node_name' in code)

        print()

if __name__ == '__main__':
    unittest.main()
