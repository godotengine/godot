import { expect } from 'chai';
import Module from './_build/JsMaterialXCore.js';
import { getMtlxStrings } from './testHelpers';

describe('Code Examples', () =>
{
    it('Building a MaterialX Document', async () =>
    {
        const mx = await Module();
        // Create a document.
        const doc = mx.createDocument();

        // Create a node graph with a single image node and output.
        const nodeGraph = doc.addNodeGraph();
        expect(doc.getNodeGraphs().length).to.equal(1);
        const image = nodeGraph.addNode('image');
        const nodes = nodeGraph.getNodes();
        expect(nodes.length).to.equal(1);
        expect(nodes[0]).to.eql(image);

        image.setInputValueString('file', 'image1.tif', 'filename');
        const input = image.getInput('file');
        expect(input).to.not.be.null;
        expect(input.getValue().getData()).to.equal('image1.tif');

        const output = nodeGraph.addOutput();
        const outputs = nodeGraph.getOutputs();
        expect(outputs.length).to.equal(1);
        expect(outputs[0]).to.eql(output);

        output.setConnectedNode(image);
        const connectedNode = output.getConnectedNode();
        expect(connectedNode).to.not.be.null;
        expect(connectedNode instanceof mx.Node).to.be.true;

        // Create a simple shader interface.
        const simpleSrf = doc.addNodeDef('ND_simpleSrf', 'surfaceshader', 'simpleSrf');
        const nodeDefs = doc.getNodeDefs();
        expect(nodeDefs.length).to.equal(1);
        expect(nodeDefs[0]).to.eql(simpleSrf);

        simpleSrf.setInputValueColor3('diffColor', new mx.Color3(1.0, 1.0, 1.0));
        let inputValue = simpleSrf.getInputValue('diffColor');
        expect(inputValue).to.not.be.null;
        expect(inputValue.getData()).to.eql(new mx.Color3(1.0, 1.0, 1.0));

        simpleSrf.setInputValueColor3('specColor', new mx.Color3(0.0, 0.0, 0.0));
        inputValue = simpleSrf.getInputValue('specColor');
        expect(inputValue).to.not.be.null;
        expect(inputValue.getData()).to.eql(new mx.Color3(0.0, 0.0, 0.0));

        const roughness = simpleSrf.setInputValueFloat('roughness', 0.25);
        inputValue = simpleSrf.getInputValue('roughness');
        expect(inputValue).to.not.be.null;
        expect(inputValue.getData()).to.equal(0.25);

        // // Create a material that instantiates the shader.
        // const material = doc.addMaterial();
        // const materials = doc.getMaterials();
        // expect(materials.length).to.equal(1);
        // expect(materials[0]).to.eql(material);
        // const refSimpleSrf = material.addShaderRef('SR_simpleSrf', 'simpleSrf');
        // const shaderRefs = material.getShaderRefs();
        // expect(shaderRefs.length).to.equal(1);
        // expect(shaderRefs[0]).to.eql(refSimpleSrf);
        // expect(shaderRefs[0].getName()).to.equal('SR_simpleSrf');

        // // Bind roughness to a new value within this material.
        // const bindInput = refSimpleSrf.addBindInput('roughness');
        // const bindInputs = refSimpleSrf.getBindInputs();
        // expect(bindInputs.length).to.equal(1);
        // expect(bindInputs[0]).to.eql(bindInput);
        // bindInput.setValuefloat(0.5);
        // expect(bindInput.getValue()).to.not.be.null;
        // expect(bindInput.getValue().getData()).to.equal(0.5);

        // // Validate the value of roughness in the context of this material.
        // expect(roughness.getBoundValue(material).getValueString()).to.equal('0.5');
    });

    it('Traversing a Document Tree', async () =>
    {
        const xmlStr = getMtlxStrings(
            ['standard_surface_greysphere_calibration.mtlx'],
            '../../resources/Materials/Examples/StandardSurface'
        )[0];
        const mx = await Module();

        // Read a document from disk.
        const doc = mx.createDocument();
        await mx.readFromXmlString(doc, xmlStr);

        // Traverse the document tree in depth-first order.
        const elements = doc.traverseTree();
        let elementCount = 0;
        let nodeCount = 0;
        let fileCount = 0;
        for (let elem of elements)
        {
            elementCount++;
            // Display the filename of each image node.
            if (elem.isANode('image'))
            {
                nodeCount++;
                const input = elem.getInput('file');
                if (input)
                {
                    fileCount++;
                    const filename = input.getValueString();
                    expect(elem.getName()).to.equal('image1');
                    expect(filename).to.equal('greysphere_calibration.png');
                }
            }
        }
        expect(elementCount).to.equal(21);
        expect(nodeCount).to.equal(1);
        expect(fileCount).to.equal(1);
    });

    it('Building a MaterialX Document', async () =>
    {
        const xmlStr = getMtlxStrings(['standard_surface_marble_solid.mtlx'], '../../resources/Materials/Examples/StandardSurface')[0];
        const mx = await Module();

        // Read a document from disk.
        const doc = mx.createDocument();
        await mx.readFromXmlString(doc, xmlStr);

        // let materialCount = 0;
        // let shaderInputCount = 0;
        // // Iterate through 1.37 materials for which there should be none
        // const materials = doc.getMaterials();
        // materials.forEach((material) => {
        //     materialCount++;

        //     // For each shader input, find all upstream images in the dataflow graph.
        //     const primaryShaderInputs = material.getPrimaryShaderInputs();
        //     primaryShaderInputs.forEach((input) => {
        //         const graphIter = input.traverseGraph(material);
        //         let edge = graphIter.next();
        //         while (edge) {
        //             shaderInputCount++;
        //             edge = graphIter.next();
        //         }
        //     });
        // });

        // expect(materialCount).to.equal(0);
        // expect(shaderInputCount).to.equal(0);
    });
});
