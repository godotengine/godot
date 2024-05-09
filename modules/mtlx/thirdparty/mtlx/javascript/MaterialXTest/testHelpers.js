var fs = require('fs');
var path = require('path');

export function getMtlxStrings(fileNames, subPath)
{
    const mtlxStrs = [];
    for (let i = 0; i < fileNames.length; i++)
    {
        const p = path.resolve(subPath, fileNames[parseInt(i, 10)]);
        const t = fs.readFileSync(p, 'utf8');
        mtlxStrs.push(t);
    }
    return mtlxStrs;
}
