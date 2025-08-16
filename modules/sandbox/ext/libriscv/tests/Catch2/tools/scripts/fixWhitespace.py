#!/usr/bin/env python3

import os
from scriptCommon import catchPath

def isSourceFile( path ):
    return path.endswith( ".cpp" ) or path.endswith( ".h" ) or path.endswith( ".hpp" )

def fixAllFilesInDir( dir ):
    changedFiles = 0
    for f in os.listdir( dir ):
        path = os.path.join( dir,f )
        if os.path.isfile( path ):
            if isSourceFile( path ):
                if fixFile( path ):
                    changedFiles += 1
        else:
            fixAllFilesInDir( path )
    return changedFiles

def fixFile( path ):
    f = open( path, 'r' )
    lines = []
    changed = 0
    for line in f:
        trimmed = line.rstrip() + "\n"
        trimmed = trimmed.replace('\t', '    ')
        if trimmed != line:
            changed = changed +1
        lines.append( trimmed )
    f.close()
    if changed > 0:
        global changedFiles
        changedFiles = changedFiles + 1
        print( path + ":" )
        print( " - fixed " + str(changed) + " line(s)" )
        altPath = path + ".backup"
        os.rename( path, altPath )
        f2 = open( path, 'w' )
        for line in lines:
            f2.write( line )
        f2.close()
        os.remove( altPath )
        return True
    return False

changedFiles = fixAllFilesInDir(catchPath)
if changedFiles > 0:
    print( "Fixed " + str(changedFiles) + " file(s)" )
else:
    print( "No trailing whitespace found" )
