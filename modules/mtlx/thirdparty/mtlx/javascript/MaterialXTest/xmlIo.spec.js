import Module from './_build/JsMaterialXCore.js';
import { expect } from 'chai';
import { getMtlxStrings } from './testHelpers';

const TIMEOUT = 60000;

describe('XmlIo', () =>
{
    let mx;

    // These should be relative to cwd
    const includeTestPath = 'data/includes';
    const libraryPath = '../../libraries/stdlib';
    const examplesPath = '../../resources/Materials/Examples';
    // TODO: Is there a better way to get these filenames than hardcoding them here?
    // The C++ tests load all files in the given directories. This would work in Node, but not in the browser.
    // Should we use a pre-test script that fetches the files and makes them available somehow?
    const libraryFilenames = ['stdlib_defs.mtlx', 'stdlib_ng.mtlx'];
    const exampleFilenames = [
        'StandardSurface/standard_surface_brass_tiled.mtlx',
        'StandardSurface/standard_surface_brick_procedural.mtlx',
        'StandardSurface/standard_surface_carpaint.mtlx',
        'StandardSurface/standard_surface_marble_solid.mtlx',
        'UsdPreviewSurface/usd_preview_surface_gold.mtlx',
        'UsdPreviewSurface/usd_preview_surface_plastic.mtlx',
    ];

    async function readStdLibrary(asString = false)
    {
        const libs = [];
        let iterable = libraryFilenames;
        if (asString)
        {
            const libraryMtlxStrings = getMtlxStrings(libraryFilenames, libraryPath);
            iterable = libraryMtlxStrings;
        }
        for (let file of iterable)
        {
            const lib = mx.createDocument();
            if (asString)
            {
                await mx.readFromXmlString(lib, file, libraryPath);
            } else
            {
                await mx.readFromXmlFile(lib, file, libraryPath);
            }
            libs.push(lib);
        };
        return libs;
    }

    async function readAndValidateExamples(examples, libraries, readFunc, searchPath = undefined)
    {
        for (let file of examples)
        {
            const doc = mx.createDocument();
            await readFunc(doc, file, searchPath);
            // Import stdlib into the current document and validate it.
            for (let lib of libraries)
            {
                doc.importLibrary(lib);
            }
            expect(doc.validate()).to.be.true;

            // Make sure the document does actually contain something.
            let valueElementCount = 0;
            const treeIter = doc.traverseTree();
            for (const elem of treeIter)
            {
                if (elem instanceof mx.ValueElement)
                {
                    valueElementCount++;
                }
            }
            expect(valueElementCount).to.be.greaterThan(0);
        };
    }

    before(async () =>
    {
        mx = await Module();
    });

    it('Read XML from file', async () =>
    {
        // Read the standard library
        const libs = await readStdLibrary(false);

        // Read and validate the example documents.
        await readAndValidateExamples(exampleFilenames, libs,
            async (document, file, sp) =>
            {
                await mx.readFromXmlFile(document, file, sp);
            }, examplesPath);

        // Read the same document twice, and verify that duplicate elements
        // are skipped.
        const doc = mx.createDocument();
        const filename = 'StandardSurface/standard_surface_carpaint.mtlx';
        await mx.readFromXmlFile(doc, filename, examplesPath);
        const copy = doc.copy();
        await mx.readFromXmlFile(doc, filename, examplesPath);
        expect(doc.validate()).to.be.true;
        expect(copy.equals(doc)).to.be.true;
    }).timeout(TIMEOUT);

    it('Read XML from string', async () =>
    {
        // Read the standard library
        const libs = await readStdLibrary(true);

        // Read and validate each example document.
        const examplesStrings = getMtlxStrings(exampleFilenames, examplesPath);
        await readAndValidateExamples(examplesStrings, libs,
            async (document, file) =>
            {
                await mx.readFromXmlString(document, file);
            });

        // Read the same document twice, and verify that duplicate elements
        // are skipped.
        const doc = mx.createDocument();
        const file = examplesStrings[exampleFilenames.indexOf('StandardSurface/standard_surface_carpaint.mtlx')];
        await mx.readFromXmlString(doc, file);
        const copy = doc.copy();
        await mx.readFromXmlString(doc, file);
        expect(doc.validate()).to.be.true;
        expect(copy.equals(doc)).to.be.true;
    }).timeout(TIMEOUT);

    it('Read XML with recursive includes', async () =>
    {
        const doc = mx.createDocument();
        await mx.readFromXmlFile(doc, includeTestPath + '/root.mtlx');
        expect(doc.getChild('paint_semigloss')).to.exist;
        expect(doc.validate()).to.be.true;
    });

    it('Locate XML includes via search path', async () =>
    {
        const searchPath = includeTestPath + ';' + includeTestPath + '/folder';
        const filename = 'non_relative_includes.mtlx';
        const doc = mx.createDocument();
        expect(async () => await mx.readFromXmlFile(doc, filename, includeTestPath)).to.throw;
        await mx.readFromXmlFile(doc, filename, searchPath);
        expect(doc.getChild('paint_semigloss')).to.exist;
        expect(doc.validate()).to.be.true;

        const doc2 = mx.createDocument();
        const mtlxString = getMtlxStrings([filename], includeTestPath);
        expect(async () => await mx.readFromXmlString(doc2, mtlxString[0])).to.throw;
        await mx.readFromXmlString(doc2, mtlxString[0], searchPath);
        expect(doc2.getChild('paint_semigloss')).to.exist;
        expect(doc2.validate()).to.be.true;
        expect(doc2.equals(doc)).to.be.true;
    });

    it('Locate XML includes via environment variable', async () =>
    {
        const searchPath = includeTestPath + ';' + includeTestPath + '/folder';
        const filename = 'non_relative_includes.mtlx';

        const doc = mx.createDocument();
        expect(async () => await mx.readFromXmlFile(doc, includeTestPath + '/' + filename)).to.throw;
        mx.setEnviron(mx.MATERIALX_SEARCH_PATH_ENV_VAR, searchPath);
        await mx.readFromXmlFile(doc, filename);
        mx.removeEnviron(mx.MATERIALX_SEARCH_PATH_ENV_VAR);
        expect(doc.getChild('paint_semigloss')).to.exist;
        expect(doc.validate()).to.be.true;

        const doc2 = mx.createDocument();
        const mtlxString = getMtlxStrings([filename], includeTestPath);
        expect(async () => await mx.readFromXmlString(doc2, mtlxString[0])).to.throw;
        mx.setEnviron(mx.MATERIALX_SEARCH_PATH_ENV_VAR, searchPath);
        await mx.readFromXmlString(doc2, mtlxString[0]);
        mx.removeEnviron(mx.MATERIALX_SEARCH_PATH_ENV_VAR);
        expect(doc2.getChild('paint_semigloss')).to.exist;
        expect(doc2.validate()).to.be.true;
        expect(doc2.equals(doc)).to.be.true;
    });

    it('Locate XML includes via absolute search paths', async () =>
    {
        let absolutePath;
        if (typeof window === 'object')
        {
            // We're in the browser
            const cwd = window.location.origin + window.location.pathname;
            absolutePath = cwd + '/' + includeTestPath;
        } else if (typeof process === 'object')
        {
            // We're in Node
            const nodePath = require('path');
            absolutePath = nodePath.resolve(includeTestPath);
        }
        const doc = mx.createDocument();
        await mx.readFromXmlFile(doc, 'root.mtlx', absolutePath);
    });

    it('Detect XML include cycles', async () =>
    {
        const doc = mx.createDocument();
        expect(async () => await mx.readFromXmlFile(doc, includeTestPath + '/cycle.mtlx')).to.throw;
    });

    it('Disabling XML includes', async () =>
    {
        const doc = mx.createDocument();
        const readOptions = new mx.XmlReadOptions();
        readOptions.readXIncludes = false;
        expect(async () => await mx.readFromXmlFile(doc, includeTestPath + '/cycle.mtlx', readOptions)).to.not.throw;
    });

    it('Write to XML string', async () =>
    {
        // Read all example documents and write them to an XML string
        const searchPath = libraryPath + ';' + examplesPath;
        for (let filename of exampleFilenames)
        {
            const doc = mx.createDocument();
            await mx.readFromXmlFile(doc, filename, searchPath);

            // Serialize to XML.
            const writeOptions = new mx.XmlWriteOptions();
            writeOptions.writeXIncludeEnable = false;
            const xmlString = mx.writeToXmlString(doc, writeOptions);

            // Verify that the serialized document is identical.
            const writtenDoc = mx.createDocument();
            await mx.readFromXmlString(writtenDoc, xmlString);
            expect(writtenDoc).to.eql(doc);
        };
    });

    it('Prepend include tag', () =>
    {
        const doc = mx.createDocument();
        const includePath = "SomePath";
        const writeOptions = new mx.XmlWriteOptions();
        mx.prependXInclude(doc, includePath);
        const xmlString = mx.writeToXmlString(doc, writeOptions);
        expect(xmlString).to.include(includePath);
    });

    // Node only, because we cannot read from a downloaded file in the browser
    it('Write XML to file', async () =>
    {
        const filename = '_build/testFile.mtlx';
        const includeRegex = /<xi:include href="(.*)"\s*\/>/g;
        const doc = mx.createDocument();
        await mx.readFromXmlFile(doc, 'root.mtlx', includeTestPath);

        // Write using includes
        mx.writeToXmlFile(doc, filename);
        // Read written document and compare with the original
        const doc2 = mx.createDocument();
        await mx.readFromXmlFile(doc2, filename, includeTestPath);
        expect(doc2.equals(doc));
        // Read written file content and verify that includes are preserved
        let fileString = getMtlxStrings([filename], '')[0];
        let matches = Array.from(fileString.matchAll(includeRegex));
        expect(matches.length).to.be.greaterThan(0);

        // Write inlining included content
        const writeOptions = new mx.XmlWriteOptions();
        writeOptions.writeXIncludeEnable = false;
        mx.writeToXmlFile(doc, filename, writeOptions);
        // Read written document and compare with the original
        const doc3 = mx.createDocument();
        await mx.readFromXmlFile(doc3, filename);
        expect(doc3.equals(doc));
        expect(doc.getChild('paint_semigloss')).to.exist;
        // Read written file content and verify that includes are inlined
        fileString = getMtlxStrings([filename], '')[0];
        matches = Array.from(fileString.matchAll(includeRegex));
        expect(matches.length).to.equal(0);
    });
});
