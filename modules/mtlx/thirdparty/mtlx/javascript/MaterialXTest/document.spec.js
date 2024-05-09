import { expect } from 'chai';
import Module from './_build/JsMaterialXCore.js';

describe('Document', () =>
{
    let mx, doc;
    before(async () =>
    {
        mx = await Module();
        // Create a document.
        doc = mx.createDocument();
    });

    function expectError(type, cb)
    {
        try
        {
            cb();
            throw new Error('Expected function to throw!');
        } catch (exceptionPtr)
        {
            const message = mx.getExceptionMessage(exceptionPtr);
            expect(message.indexOf(type) !== -1).to.be.true;
        }
    }

    let nodeGraph;
    it('Build document', () =>
    {
        // Create a node graph with constant and image sources.
        nodeGraph = doc.addNodeGraph();
        expect(nodeGraph).to.exist;
        expectError('Child name is not unique: nodegraph1', () =>
        {
            doc.addNodeGraph(nodeGraph.getName());
        });
        const constant = nodeGraph.addNode('constant');
        const image = nodeGraph.addNode('image');

        // Connect sources to outputs
        const output1 = nodeGraph.addOutput();
        const output2 = nodeGraph.addOutput();
        output1.setConnectedNode(constant);
        output2.setConnectedNode(image);
        expect(output1.getConnectedNode()).to.eql(constant);
        expect(output2.getConnectedNode()).to.eql(image);
        expect(output1.getUpstreamElement()).to.eql(constant);
        expect(output2.getUpstreamElement()).to.eql(image);

        // Set constant node color
        const color = new mx.Color3(0.1, 0.2, 0.3);
        constant.setInputValueColor3('value', color);
        expect(constant.getInputValue('value').getData()).to.eql(color);

        // Set image node file
        const file = 'image1.tif';
        image.setInputValueString('file', file, 'filename');
        expect(image.getInputValue('file').getData()).to.eql(file);

        // Create a custom nodedef
        const nodeDef = doc.addNodeDef('nodeDef1', 'float', 'turbulence3d');
        nodeDef.setInputValueInteger('octaves', 3);
        nodeDef.setInputValueFloat('lacunarity', 2.0);
        nodeDef.setInputValueFloat('gain', 0.5);

        // Reference the custom nodedef
        const custom = nodeGraph.addNode('turbulence3d', 'turbulence1', 'float');
        expect(custom.getInputValue('octaves').getData()).to.equal(3);
        custom.setInputValueInteger('octaves', 5);
        expect(custom.getInputValue('octaves').getData()).to.equal(5);

        // Test scoped attributes
        nodeGraph.setFilePrefix('folder/');
        nodeGraph.setColorSpace('lin_rec709');
        expect(image.getInput('file').getResolvedValueString()).to.equal('folder/image1.tif');
        expect(constant.getActiveColorSpace()).to.equal('lin_rec709');

        // Create a simple shader interface
        const simpleSrf = doc.addNodeDef('', 'surfaceshader', 'simpleSrf');
        simpleSrf.setInputValueColor3('diffColor', new mx.Color3(1.0, 1.0, 1.0));
        simpleSrf.setInputValueColor3('specColor', new mx.Color3(0.0, 0.0, 0.0));
        const roughness = simpleSrf.setInputValueFloat('roughness', 0.25);
        expect(roughness.getIsUniform()).to.be.false;
        roughness.setIsUniform(true);
        expect(roughness.getIsUniform()).to.be.true;

        // Instantiate shader and material nodes
        const shaderNode = doc.addNodeInstance(simpleSrf);
        const materialNode = doc.addMaterialNode('', shaderNode);
        expect(materialNode.getUpstreamElement().equals(shaderNode)).to.be.true;

        // Bind the diffuse color input to the constant color output
        shaderNode.setConnectedOutput('diffColor', output1);
        expect(shaderNode.getUpstreamElement().equals(constant)).to.be.true;

        // Bind the roughness input to a value
        const instanceRoughness = shaderNode.setInputValueFloat('roughness', 0.5);
        expect(instanceRoughness.getValue().getData()).to.equal(0.5);
        expect(instanceRoughness.getDefaultValue().getData()).to.equal(0.25);

        // Create a look for the material
        const look = doc.addLook();
        expect(doc.getLooks().length).to.equal(1);

        // Bind the material to a geometry string
        let matAssign1 = look.addMaterialAssign('matAssign1', materialNode.getName());
        matAssign1 = look.getMaterialAssign('matAssign1');
        expect(matAssign1);
        matAssign1.setGeom('/robot1');
        expect(matAssign1.getReferencedMaterial().equals(materialNode)).to.be.true;
        expect(mx.getGeometryBindings(materialNode, '/robot1').length).to.equal(1);
        expect(mx.getGeometryBindings(materialNode, '/robot2').length).to.equal(0);

        // Bind the material to a collection
        let matAssign2 = look.addMaterialAssign('matAssign2', materialNode.getName());
        matAssign2 = look.getMaterialAssign('matAssign1');
        expect(matAssign2);
        const collection = doc.addCollection();
        collection.setIncludeGeom('/robot2');
        collection.setExcludeGeom('/robot2/left_arm');
        matAssign2.setCollection(collection);
        expect(matAssign2.getReferencedMaterial().equals(materialNode)).to.be.true;
        expect(mx.getGeometryBindings(materialNode, '/robot2').length).to.equal(1);
        expect(mx.getGeometryBindings(materialNode, '/robot2/right_arm').length).to.equal(1);
        expect(mx.getGeometryBindings(materialNode, '/robot2/left_arm').length).to.equal(0);

        const materialAssigns = look.getMaterialAssigns();
        expect(materialAssigns.length).to.equal(2);

        // Create a property assignment
        const propertyAssign = look.addPropertyAssign();
        propertyAssign.setProperty('twosided');
        propertyAssign.setGeom('/robot1');
        propertyAssign.setValueBoolean(true);
        expect(propertyAssign.getProperty()).to.equal('twosided');
        expect(propertyAssign.getGeom()).to.equal('/robot1');
        expect(propertyAssign.getValue().getData()).to.equal(true);
        let propertyAssigns = look.getPropertyAssigns();
        expect(propertyAssigns.length).to.equal(1);

        // Create a property set assignment
        const propertySet = doc.addPropertySet();
        propertySet.setPropertyValueBoolean('matte', false);
        expect(propertySet.getPropertyValue('matte').getData()).to.equal(false);
        const propertySetAssign = look.addPropertySetAssign();
        propertySetAssign.setPropertySet(propertySet);
        propertySetAssign.setGeom('/robot1');
        expect(propertySetAssign.getPropertySet().equals(propertySet)).to.be.true;
        expect(propertySetAssign.getGeom()).to.equal('/robot1');

        // Create a variant set
        const variantSet = doc.addVariantSet();
        variantSet.addVariant('original');
        variantSet.addVariant('damaged');
        expect(variantSet.getVariants().length).to.equal(2);

        // Validate the document
        expect(doc.validate()).to.be.true;

        // Disconnect output from sources
        output1.setConnectedNode(null);
        output2.setConnectedNode(null);
        expect(output1.getConnectedNode()).to.equal(null);
        expect(output2.getConnectedNode()).to.equal(null);
    });
});
