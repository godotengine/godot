import * as THREE from 'three';
import * as fflate from 'three/examples/jsm/libs/fflate.module.js';

const debugFileHandling = false;
let loadingCallback = null;
let sceneLoadingCallback = null;

export function setLoadingCallback(cb)
{
    loadingCallback = cb;
}

export function setSceneLoadingCallback(cb)
{
    sceneLoadingCallback = cb;
}

export function dropHandler(ev)
{
    if (debugFileHandling) console.log('File(s) dropped', ev.dataTransfer.items, ev.dataTransfer.files);

    // Prevent default behavior (Prevent file from being opened)
    ev.preventDefault();

    if (ev.dataTransfer.items)
    {
        const allEntries = [];

        let haveGetAsEntry = false;
        if (ev.dataTransfer.items.length > 0)
            haveGetAsEntry =
                ("getAsEntry" in ev.dataTransfer.items[0]) ||
                ("webkitGetAsEntry" in ev.dataTransfer.items[0]);

        // Useful for debugging file handling on platforms that don't support newer file system APIs
        // haveGetAsEntry = false;

        if (haveGetAsEntry)
        {
            for (var i = 0; i < ev.dataTransfer.items.length; i++)
            {
                let item = ev.dataTransfer.items[i];
                let entry = ("getAsEntry" in item) ? item.getAsEntry() : item.webkitGetAsEntry();
                allEntries.push(entry);
            }
            handleFilesystemEntries(allEntries);
            return;
        }

        for (var i = 0; i < ev.dataTransfer.items.length; i++)
        {
            let item = ev.dataTransfer.items[i];

            // API when there's no "getAsEntry" support
            console.log(item.kind, item);
            if (item.kind === 'file')
            {
                var file = item.getAsFile();
                testAndLoadFile(file);
            }
            // could also be a directory
            else if (item.kind === 'directory')
            {
                var dirReader = item.createReader();
                dirReader.readEntries(function (entries)
                {
                    for (var i = 0; i < entries.length; i++)
                    {
                        console.log(entries[i].name);
                        var entry = entries[i];
                        if (entry.isFile)
                        {
                            entry.file(function (file)
                            {
                                testAndLoadFile(file);
                            });
                        }
                    }
                });
            }
        }
    } else
    {
        for (var i = 0; i < ev.dataTransfer.files.length; i++)
        {
            let file = ev.dataTransfer.files[i];
            testAndLoadFile(file);
        }
    }
}

export function dragOverHandler(ev)
{
    ev.preventDefault();
}

async function getBufferFromFile(fileEntry)
{

    if (fileEntry instanceof ArrayBuffer) return fileEntry;
    if (fileEntry instanceof String) return fileEntry;

    const name = fileEntry.fullPath || fileEntry.name;
    const ext = name.split('.').pop();
    const readAsText = ext === 'mtlx';

    if (debugFileHandling) console.log("reading ", fileEntry, "as text?", readAsText);

    if (debugFileHandling) console.log("getBufferFromFile", fileEntry);
    const buffer = await new Promise((resolve, reject) =>
    {
        function readFile(file)
        {
            var reader = new FileReader();
            reader.onloadend = function (e)
            {
                if (debugFileHandling) console.log("loaded", "should be text?", readAsText, this.result);
                resolve(this.result);
            };

            if (readAsText)
                reader.readAsText(file);
            else
                reader.readAsArrayBuffer(file);
        }

        if ("file" in fileEntry)
        {
            fileEntry.file(function (file)
            {
                readFile(file);
            }, (e) =>
            {
                console.error("Error reading file ", e);
            });
        }
        else
        {
            readFile(fileEntry);
        }
    });
    return buffer;
}

async function handleFilesystemEntries(entries)
{
    const allFiles = [];
    const fileIgnoreList = [
        '.gitignore',
        'README.md',
        '.DS_Store',
    ]
    const dirIgnoreList = [
        '.git',
        'node_modules',
    ]

    let isGLB = false;
    let haveMtlx = false;
    for (let entry of entries)
    {
        if (debugFileHandling) console.log("file entry", entry)
        if (entry.isFile)
        {
            if (debugFileHandling)
                console.log("single file", entry);
            if (fileIgnoreList.includes(entry.name))
            {
                continue;
            }
            allFiles.push(entry);

            if (entry.name.endsWith('glb'))
            {
                isGLB = true;
                break;
            }
        }
        else if (entry.isDirectory)
        {
            if (dirIgnoreList.includes(entry.name))
            {
                continue;
            }
            const files = await readDirectory(entry);
            if (debugFileHandling) console.log("all files", files);
            for (const file of files)
            {
                if (fileIgnoreList.includes(file.name))
                {
                    continue;
                }
                allFiles.push(file);
            }
        }
    }

    const imageLoader = new THREE.ImageLoader();

    // unpack zip files first
    for (const fileEntry of allFiles)
    {
        // special case: zip archives
        if (fileEntry.fullPath.toLowerCase().endsWith('.zip'))
        {
            await new Promise(async (resolve, reject) =>
            {
                const arrayBuffer = await getBufferFromFile(fileEntry);

                // use fflate to unpack them and add the files to the cache
                fflate.unzip(new Uint8Array(arrayBuffer), (error, unzipped) =>
                {
                    // push these files into allFiles
                    for (const [filePath, buffer] of Object.entries(unzipped))
                    {

                        // mock FileEntry for easier usage downstream
                        const blob = new Blob([buffer]);
                        const newFileEntry = {
                            fullPath: "/" + filePath,
                            name: filePath.split('/').pop(),
                            file: (callback) =>
                            {
                                callback(blob);
                            },
                            isFile: true,
                        };
                        allFiles.push(newFileEntry);
                    }

                    resolve();
                });
            });
        }
    }

    // sort so mtlx files come first
    allFiles.sort((a, b) =>
    {
        if (a.name.endsWith('.mtlx') && !b.name.endsWith('.mtlx'))
        {
            return -1;
        }
        if (!a.name.endsWith('.mtlx') && b.name.endsWith('.mtlx'))
        {
            return 1;
        }
        return 0;
    });

    if (isGLB)
    {
        console.log("Load GLB file", allFiles[0]);

        const rootFile = allFiles[0];
        THREE.Cache.add(rootFile.fullPath, await getBufferFromFile(rootFile));

        if (debugFileHandling) console.log("CACHE", THREE.Cache.files);

        sceneLoadingCallback(rootFile);
        return;
    }

    if (!allFiles[0].name.endsWith('mtlx'))
    {
        console.log("No MaterialX files dropped. Skipping content.");
        return;
    }

    if (debugFileHandling)
    {
        console.log("- All files", allFiles);
    }

    // put all files in three' Cache
    for (const fileEntry of allFiles)
    {

        const allowedFileTypes = [
            'png', 'jpg', 'jpeg'
        ];

        const ext = fileEntry.fullPath.split('.').pop();
        if (!allowedFileTypes.includes(ext))
        {
            // console.log("skipping file", fileEntry.fullPath);
            continue;
        }

        const buffer = await getBufferFromFile(fileEntry);
        const img = await imageLoader.loadAsync(URL.createObjectURL(new Blob([buffer])));
        if (debugFileHandling) console.log("caching file", fileEntry.fullPath, img);
        THREE.Cache.add(fileEntry.fullPath, img);
    }

    // TODO we could also allow dropping of multiple MaterialX files (or folders with them inside) 
    // and seed the dropdown from that.
    // At that point, actually reading files and textures into memory should be deferred until they are actually used.
    if (allFiles.length > 0)
    {
        const rootFile = allFiles[0];
        THREE.Cache.add(rootFile.fullPath, await getBufferFromFile(rootFile));

        if (debugFileHandling) console.log("CACHE", THREE.Cache.files);

        loadingCallback(rootFile);
    }
    else
    {
        console.log('No files to add cache.')
    }
}

async function readDirectory(directory)
{
    let entries = [];
    let getEntries = async (directory) =>
    {
        let dirReader = directory.createReader();
        await new Promise((resolve, reject) =>
        {
            dirReader.readEntries(
                async (results) =>
                {
                    if (results.length)
                    {
                        // entries = entries.concat(results);
                        for (let entry of results)
                        {
                            if (entry.isDirectory)
                            {
                                await getEntries(entry);
                            }
                            else
                            {
                                entries.push(entry);
                            }
                        }
                    }
                    resolve();
                },
                (error) =>
                {
                    /* handle error — error is a FileError object */
                },
            )
        }
        )
    };

    await getEntries(directory);
    return entries;
}

async function testAndLoadFile(file)
{
    let ext = file.name.split('.').pop();
    if (debugFileHandling) console.log(file.name + ", " + file.size + ", " + ext);

    const arrayBuffer = await getBufferFromFile(file);
    console.log(arrayBuffer)

    // mock a fileEntry and pass through the same loading logic
    const newFileEntry = {
        fullPath: "/" + file.name,
        name: file.name.split('/').pop(),
        isFile: true,
        file: (callback) =>
        {
            callback(file);
        }
    };

    handleFilesystemEntries([newFileEntry]);
}