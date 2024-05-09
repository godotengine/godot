import { expect } from 'chai';
import Module from './_build/JsMaterialXCore.js';

describe('Element', () =>
{
    let mx, doc, valueTypes;

    const primitiveValueTypes = {
        Integer: 10,
        Boolean: true,
        String: 'test',
        Float: 15,
        IntegerArray: [1, 2, 3, 4, 5],
        FloatArray: [12, 14], // Not using actual floats to avoid precision problems
        StringArray: ['first', 'second'],
        BooleanArray: [true, true, false],
    }

    before(async () =>
    {
        mx = await Module();
        doc = mx.createDocument();
        valueTypes = {
            Color3: new mx.Color3(1, 0, 0.5),
            Color4: new mx.Color4(0, 1, 0.5, 1),
            Vector2: new mx.Vector2(0, 1),
            Vector3: new mx.Vector3(0, 1, 2),
            Vector4: new mx.Vector4(0, 1, 2, 1),
            Matrix33: new mx.Matrix33(0, 1, 2, 3, 4, 5, 6, 7, 8),
            Matrix44: new mx.Matrix44(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        };
    });

    describe('value setters', () =>
    {
        const checkValue = (types, assertionCallback) =>
        {
            const elem = doc.addChildOfCategory('geomprop');
            Object.keys(types).forEach((typeName) =>
            {
                const setFn = `setValue${typeName}`;
                elem[setFn](types[typeName]);
                assertionCallback(elem.getValue().getData(), typeName);
            });
        };

        it('should work with expected type', () =>
        {
            checkValue(valueTypes, (returnedValue, typeName) =>
            {
                expect(returnedValue).to.be.an.instanceof(mx[`${typeName}`]);
                expect(returnedValue.equals(valueTypes[typeName])).to.equal(true);
            });
        });

        it('should work with expected primitive type', () =>
        {
            checkValue(primitiveValueTypes, (returnedValue, typeName) =>
            {
                expect(returnedValue).to.eql(primitiveValueTypes[typeName]);
            });
        });

        it('should fail for incorrect type', () =>
        {
            const elem = doc.addChildOfCategory('geomprop');
            expect(() => elem.Matrix33(true)).to.throw();
        });
    });

    describe('typed value setters', () =>
    {
        const checkTypes = (types, assertionCallback) =>
        {
            const elem = doc.addChildOfCategory('geomprop');
            Object.keys(types).forEach((typeName) =>
            {
                const setFn = `setTypedAttribute${typeName}`;
                const getFn = `getTypedAttribute${typeName}`;
                elem[setFn](typeName, types[typeName]);
                assertionCallback(elem[getFn](typeName), types[typeName]);
            });
        };

        it('should work with expected custom type', () =>
        {
            checkTypes(valueTypes, (returnedValue, originalValue) =>
            {
                expect(returnedValue.equals(originalValue)).to.equal(true);
            });
        });

        it('should work with expected primitive type', () =>
        {
            checkTypes(primitiveValueTypes, (returnedValue, originalValue) =>
            {
                expect(returnedValue).to.eql(originalValue);
            });
        });

        it('should fail for incorrect type', () =>
        {
            const elem = doc.addChildOfCategory('geomprop');
            expect(() => elem.setTypedAttributeColor3('wrongType', true)).to.throw();
        });
    });

    it('factory invocation should match specialized functions', () =>
    {
        // List based in source/MaterialXCore/Element.cpp
        const elemtypeArr = [
            'Backdrop',
            'Collection',
            'GeomInfo',
            'MaterialAssign',
            'PropertySetAssign',
            'Visibility',
            'GeomPropDef',
            'Look',
            'LookGroup',
            'PropertySet',
            'TypeDef',
            'AttributeDef',
            'NodeGraph',
            'Implementation',
            'Node',
            'NodeDef',
            'Variant',
            'Member',
            'TargetDef',
            'GeomProp',
            'Input',
            'Output',
            'Property',
            'PropertyAssign',
            'Unit',
            'UnitDef',
            'UnitTypeDef',
            'VariantAssign',
            'VariantSet',
        ];

        elemtypeArr.forEach((typeName) =>
        {
            const specializedFn = `addChild${typeName}`;
            const factoryName = typeName.toLowerCase();
            const type = mx[typeName];
            expect(doc[specializedFn]()).to.be.an.instanceof(type);
            expect(doc.addChildOfCategory(factoryName)).to.be.an.instanceof(type);
        });

        const specialElemType = {
            'MaterialX': mx.Document,
            'Comment': mx.CommentElement,
            'Generic': mx.GenericElement,
        };

        Object.keys(specialElemType).forEach((typeName) =>
        {
            const specializedFn = `addChild${typeName}`;
            const factoryName = typeName.toLowerCase();
            expect(doc[specializedFn]()).to.be.an.instanceof(specialElemType[typeName]);
            expect(doc.addChildOfCategory(factoryName)).to.be.an.instanceof(specialElemType[typeName]);
        });
    });
});
