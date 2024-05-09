#!/usr/bin/env python
'''
Reformat a folder of MaterialX documents in place, optionally upgrading
the documents to the latest version of the standard.
'''

import argparse
import os

import MaterialX as mx

def main():
    parser = argparse.ArgumentParser(description="Reformat a folder of MaterialX documents in place.")
    parser.add_argument("--yes", dest="yes", action="store_true", help="Proceed without asking for confirmation from the user.")
    parser.add_argument('--upgrade', dest='upgrade', action="store_true", help='Upgrade documents to the latest version of the standard.')
    parser.add_argument(dest="inputFolder", help="An input folder to scan for MaterialX documents.")
    opts = parser.parse_args()

    validDocs = dict()
    for root, dirs, files in os.walk(opts.inputFolder):
        for file in files:
            if file.endswith('.mtlx'):
                filename = os.path.join(root, file)
                doc = mx.createDocument()
                try:
                    readOptions = mx.XmlReadOptions()
                    readOptions.readComments = True
                    readOptions.readNewlines = True    
                    readOptions.upgradeVersion = opts.upgrade
                    try:
                        mx.readFromXmlFile(doc, filename, mx.FileSearchPath(), readOptions)
                    except Exception as err:
                        print('Skipping "' + file + '" due to exception: ' + str(err))
                        continue
                    validDocs[filename] = doc
                except mx.Exception:
                    pass

    if not validDocs:
        print('No MaterialX documents were found in "%s"' % (opts.inputFolder))
        return

    print('Found %s MaterialX files in "%s"' % (len(validDocs), opts.inputFolder))

    mxVersion = mx.getVersionIntegers()

    if not opts.yes:
        if opts.upgrade:
            question = 'Would you like to upgrade all %i documents to MaterialX v%i.%i in place (y/n)?' % (len(validDocs), mxVersion[0], mxVersion[1])
        else:
            question = 'Would you like to reformat all %i documents in place (y/n)?' % len(validDocs)
        answer = input(question)
        if answer != 'y' and answer != 'Y':
            return

    for (filename, doc) in validDocs.items():
        mx.writeToXmlFile(doc, filename)

    if opts.upgrade:
        print('Upgraded %i documents to MaterialX v%i.%i' % (len(validDocs), mxVersion[0], mxVersion[1]))
    else:
        print('Reformatted %i documents ' % len(validDocs))

if __name__ == '__main__':
    main()
