const path = require('path');
const fs = require('fs');
const CopyPlugin = require("copy-webpack-plugin");
const HtmlWebpackPlugin = require('html-webpack-plugin')

const stdSurfaceMaterials = "../../resources/Materials/Examples/StandardSurface";
const stdSurfaceMaterialsBaseURL = "Materials/Examples/StandardSurface";
let dirent = fs.readdirSync(stdSurfaceMaterials).filter(
    function (file) { if (file.lastIndexOf(".mtlx") > -1) return file; }
)
let materials = dirent
    .map((fileName) => ({ name: fileName, value: `${stdSurfaceMaterialsBaseURL}/${fileName}` }));

const usdSurfaceMaterials = "../../resources/Materials/Examples/UsdPreviewSurface";
const usdSurfaceMaterialsBaseURL = "Materials/Examples/UsdPreviewSurface";
dirent = fs.readdirSync(usdSurfaceMaterials).filter(
    function (file) { if (file.lastIndexOf(".mtlx") > -1) return file; }
)
let usdMaterials = dirent
    .map((fileName) => ({ name: fileName, value: `${usdSurfaceMaterialsBaseURL}/${fileName}` }));

const gltfSurfaceMaterials = "../../resources/Materials/Examples/GltfPbr";
const gltfSurfaceMaterialsBaseURL = "Materials/Examples/GltfPbr";
dirent = fs.readdirSync(gltfSurfaceMaterials).filter(
    function (file) { if (file.lastIndexOf(".mtlx") > -1) return file; }
)
let gltfMaterials = dirent
    .map((fileName) => ({ name: fileName, value: `${gltfSurfaceMaterialsBaseURL}/${fileName}` }));

materials = materials.concat(usdMaterials);
materials = materials.concat(gltfMaterials);

const geometryFiles = "../../resources/Geometry";
const geometryFilesURL = "Geometry";
dirent = fs.readdirSync(geometryFiles).filter(
    function (file) { if (file.lastIndexOf(".glb") > -1) return file; }
)
let geometry = dirent
    .map((fileName) => ({ name: fileName, value: `${geometryFilesURL}/${fileName}` }));

module.exports = {
    entry: './source/index.js',
    output: {
        filename: 'main.js',
        path: path.resolve(__dirname, 'dist')
    },
    mode: "development",
    plugins: [
        new HtmlWebpackPlugin({
            templateParameters: {
                materials,
                geometry
            },
            template: 'index.ejs'
        }),
        new CopyPlugin({
            patterns: [
                {
                    context: "../../resources/Images",
                    from: "*.*",
                    to: "Images",
                },
                {
                    context: "../../resources/Geometry/",
                    from: "*.glb",
                    to: "Geometry",
                },
                { from: "./public", to: 'public' },
                { context: "../../resources/Lights", from: "*.*", to: "Lights" },
                { context: "../../resources/Lights/irradiance", from: "*.*", to: "Lights/irradiance" },
                { from: stdSurfaceMaterials, to: stdSurfaceMaterialsBaseURL },
                { from: usdSurfaceMaterials, to: usdSurfaceMaterialsBaseURL },
                { from: gltfSurfaceMaterials, to: gltfSurfaceMaterialsBaseURL },
                { from: "../build/bin/JsMaterialXCore.wasm" },
                { from: "../build/bin/JsMaterialXCore.js" },
                { from: "../build/bin/JsMaterialXGenShader.wasm" },
                { from: "../build/bin/JsMaterialXGenShader.js" },
                { from: "../build/bin/JsMaterialXGenShader.data" },
            ],
        }),
    ]
};
