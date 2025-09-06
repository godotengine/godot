#!/usr/bin/env python3

import os
import sys
import shutil
import glob
from scriptCommon import catchPath

rootPath = os.path.join( catchPath, 'tests/SelfTest/Baselines' )

if len(sys.argv) > 1:
    files = [os.path.join( rootPath, f ) for f in sys.argv[1:]]
else:
    files = glob.glob( os.path.join( rootPath, "*.unapproved.txt" ) )


def approveFile( approvedFile, unapprovedFile ):
    justFilename = unapprovedFile[len(rootPath)+1:]
    if os.path.exists( unapprovedFile ):
        if os.path.exists( approvedFile ):
            os.remove( approvedFile )
        os.rename( unapprovedFile, approvedFile )
        print( "approved " + justFilename )
    else:
        print( "approval file " + justFilename + " does not exist" )

if files:
    for unapprovedFile in files:
        approveFile( unapprovedFile.replace( "unapproved.txt", "approved.txt" ), unapprovedFile )
else:
    print( "no files to approve" )
