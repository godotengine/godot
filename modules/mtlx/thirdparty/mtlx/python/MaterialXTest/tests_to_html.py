#!/usr/bin/python

import sys
import os
import datetime
import argparse

try:
    # Use pip to install Pillow and Image to enable image diffs
    from PIL import Image, ImageChops
    DIFF_ENABLED = True
except Exception:
    DIFF_ENABLED = False

def createDiff(image1Path, image2Path, imageDiffPath):
    try:
        if os.path.exists(imageDiffPath):
            os.remove(imageDiffPath)
        
        if not os.path.exists(image1Path):
            print ("Image diff input missing: " + image1Path)
            return
        
        if not os.path.exists(image2Path):
            print ("Image diff input missing: " + image2Path)
            return

        image1 = Image.open(image1Path).convert('RGB')
        image2 = Image.open(image2Path).convert('RGB')
        diff = ImageChops.difference(image1, image2)
        diff.save(imageDiffPath)
    except Exception:
        if os.path.exists(imageDiffPath):
            os.remove(imageDiffPath)
        print ("Failed to create image diff between: " + image1Path + ", " + image2Path)

def main(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--inputdir1', dest='inputdir1', action='store', help='Input directory', default=".")
    parser.add_argument('-i2', '--inputdir2', dest='inputdir2', action='store', help='Second input directory', default=".")
    parser.add_argument('-i3', '--inputdir3', dest='inputdir3', action='store', help='Third input directory', default=".")
    parser.add_argument('-o', '--outputfile', dest='outputfile', action='store', help='Output file name', default="tests.html")
    parser.add_argument('-d', '--diff', dest='CREATE_DIFF', action='store_true', help='Perform image diff', default=False)
    parser.add_argument('-t', '--timestamp', dest='ENABLE_TIMESTAMPS', action='store_true', help='Write image timestamps', default=False)
    parser.add_argument('-w', '--imagewidth', type=int, dest='imagewidth', action='store', help='Set image display width', default=256)
    parser.add_argument('-ht', '--imageheight', type=int, dest='imageheight', action='store', help='Set image display height', default=256)
    parser.add_argument('-cp', '--cellpadding', type=int, dest='cellpadding', action='store', help='Set table cell padding', default=0)
    parser.add_argument('-tb', '--tableborder', type=int, dest='tableborder', action='store', help='Table border width. 0 means no border', default=3)
    parser.add_argument('-l1', '--lang1', dest='lang1', action='store', help='First target language for comparison. Default is glsl', default="glsl")
    parser.add_argument('-l2', '--lang2', dest='lang2', action='store', help='Second target language for comparison. Default is osl', default="osl")
    parser.add_argument('-l3', '--lang3', dest='lang3', action='store', help='Third target language for comparison. Default is empty', default="")

    args = parser.parse_args(args)

    fh = open(args.outputfile,"w+")
    fh.write("<html>\n")
    fh.write("<style>\n")
    fh.write("td {")
    fh.write("    padding: " + str(args.cellpadding) + ";")
    fh.write("    border: " + str(args.tableborder) + "px solid black;")
    fh.write("}")
    fh.write("table, tbody, th, .td_image {")
    fh.write("    border-collapse: collapse;")
    fh.write("    padding: 0;")
    fh.write("    margin: 0;")
    fh.write("}")
    fh.write("</style>")
    fh.write("<body>\n")
    dir1 = os.getcwd() if args.inputdir1 == "." else args.inputdir1
    dir2 = os.getcwd() if args.inputdir2 == "." else args.inputdir2
    dir3 = os.getcwd() if args.inputdir3 == "." else args.inputdir3
    
    useThirdLang = True if args.lang3 and (args.inputdir1 != args.inputdir3 or args.lang1 != args.lang3) else False
    
    if useThirdLang:
        fh.write("<h3>" + args.lang1 + " (in: " + dir1 + ") vs "+ args.lang2 + " (in: " + dir2 + ") vs "+ args.lang3 + " (in: " + dir3 + ")</h3>\n")
    else:
        fh.write("<h3>" + args.lang1 + " (in: " + dir1 + ") vs "+ args.lang2 + " (in: " + dir2 + ")</h3>\n")

    if not DIFF_ENABLED and args.CREATE_DIFF:
        print("--diff argument ignored. Diff utility not installed.")

    if not args.inputdir2:
        args.inputdir2 = args.inputdir1
    if useThirdLang and not args.inputdir3:
        args.inputdir3 = args.inputdir1

    # Get all source files
    langFiles1 = []
    langPaths1 = []
    for subdir, _, files in os.walk(args.inputdir1):
        for curFile in files:
            if curFile.endswith(args.lang1 + ".png"):
                langFiles1.append(curFile)
                langPaths1.append(subdir) 

    # Get all destination files, matching source files
    langFiles2 = []
    langPaths2 = []
    langFiles3 = []
    langPaths3 = []
    postFix = args.lang1 + ".png"
    for file1, path1 in zip(langFiles1, langPaths1):
        # Allow for just one language to be shown if source and dest are the same.
        # Otherwise add in equivalent name with dest language replacement if
        # pointing to the same directory 
        if args.inputdir1 != args.inputdir2 or args.lang1 != args.lang2:
            file2 = file1[:-len(postFix)] + args.lang2 + ".png"
            path2 = os.path.join(args.inputdir2, path1)
        else:
            file2 = ""
            path2 = None
        langFiles2.append(file2)
        langPaths2.append(path2)

        if useThirdLang:
            file3 = file1[:-len(postFix)] + args.lang3 + ".png"
            path3 = os.path.join(args.inputdir2, path1)
        else:
            file3 = ""
            path3 = None
        langFiles3.append(file3)
        langPaths3.append(path3)

    if langFiles1:
        curPath = ""
        for file1, file2, file3, path1, path2, path3 in zip(langFiles1, langFiles2, langFiles3, langPaths1, langPaths2, langPaths3):

            fullPath1 = os.path.join(path1, file1) if file1 else None
            fullPath2 = os.path.join(path2, file2) if file2 else None
            fullPath3 = os.path.join(path3, file3) if file3 else None
            diffPath1 = None
            diffPath2 = None
            diffPath3 = None

            if curPath != path1:
                if curPath != "":
                    fh.write("</table>\n")
                fh.write("<p>" + os.path.normpath(path1) + ":</p>\n")
                fh.write("<table>\n")
                curPath = path1

            if file1 and file2 and DIFF_ENABLED and args.CREATE_DIFF:
                diffPath1 = fullPath1[0:-8] + "_" + args.lang1 + "_vs_" + args.lang2 + "_diff.png"
                createDiff(fullPath1, fullPath2, diffPath1)

            if useThirdLang and file1 and file3 and DIFF_ENABLED and args.CREATE_DIFF:
                diffPath2 = fullPath1[0:-8] + "_" + args.lang1 + "_vs_" + args.lang3 + "_diff.png"
                createDiff(fullPath1, fullPath3, diffPath2)
                diffPath3 = fullPath1[0:-8] + "_" + args.lang2 + "_vs_" + args.lang3 + "_diff.png"
                createDiff(fullPath2, fullPath3, diffPath3)

            if os.path.isabs(args.outputfile):
                fileUri = 'file:///'
            else:
                fileUri =''

            fh.write("<tr>\n")
            if fullPath1:
                fh.write("<td class='td_image'><img src='" + fileUri + fullPath1 + "' height='" + str(args.imageheight) + "' width='" + str(args.imagewidth) + "' loading='lazy' style='background-color:black;'/></td>\n")
            if fullPath2:
                fh.write("<td class='td_image'><img src='" + fileUri + fullPath2 + "' height='" + str(args.imageheight) + "' width='" + str(args.imagewidth) + "' loading='lazy' style='background-color:black;'/></td>\n")
            if fullPath3:
                fh.write("<td class='td_image'><img src='" + fileUri + fullPath3 + "' height='" + str(args.imageheight) + "' width='" + str(args.imagewidth) + "' loading='lazy' style='background-color:black;'/></td>\n")
            if diffPath1:
                fh.write("<td class='td_image'><img src='" + fileUri + diffPath1 + "' height='" + str(args.imageheight) + "' width='" + str(args.imagewidth) + "' loading='lazy' style='background-color:black;'/></td>\n")
            if diffPath2:
                fh.write("<td class='td_image'><img src='" + fileUri + diffPath2 + "' height='" + str(args.imageheight) + "' width='" + str(args.imagewidth) + "' loading='lazy' style='background-color:black;'/></td>\n")
            if diffPath3:
                fh.write("<td class='td_image'><img src='" + fileUri + diffPath3 + "' height='" + str(args.imageheight) + "' width='" + str(args.imagewidth) + "' loading='lazy' style='background-color:black;'/></td>\n")
            fh.write("</tr>\n")

            fh.write("<tr>\n")
            if fullPath1:
                fh.write("<td align='center'>" + file1)
                if args.ENABLE_TIMESTAMPS and os.path.isfile(fullPath1):
                    fh.write("<br>(" + str(datetime.datetime.fromtimestamp(os.path.getmtime(fullPath1))) + ")")
                fh.write("</td>\n")
            if fullPath2:
                fh.write("<td align='center'>" + file2)
                if args.ENABLE_TIMESTAMPS and os.path.isfile(fullPath2):
                    fh.write("<br>(" + str(datetime.datetime.fromtimestamp(os.path.getmtime(fullPath2))) + ")")
                fh.write("</td>\n")
            if fullPath3:
                fh.write("<td align='center'>" + file3)
                if args.ENABLE_TIMESTAMPS and os.path.isfile(fullPath3):
                    fh.write("<br>(" + str(datetime.datetime.fromtimestamp(os.path.getmtime(fullPath3))) + ")")
                fh.write("</td>\n")
            if diffPath1:
                fh.write("<td align='center'>Difference " + args.lang1 + " vs. " + args.lang2 + " </td>\n")
            if diffPath2:
                fh.write("<td align='center'>Difference " + args.lang1 + " vs. " + args.lang3 + " </td>\n")
            if diffPath3:
                fh.write("<td align='center'>Difference " + args.lang2 + " vs. " + args.lang3 + " </td>\n")
            fh.write("</tr>\n")

    fh.write("</table>\n")
    fh.write("</body>\n")
    fh.write("</html>\n")

if __name__ == "__main__":
    main(sys.argv[1:])
