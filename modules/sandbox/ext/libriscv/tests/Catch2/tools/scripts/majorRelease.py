#!/usr/bin/env python3

import releaseCommon

v = releaseCommon.Version()
v.incrementMajorVersion()
releaseCommon.performUpdates(v)

print( "Updated files to v{0}".format( v.getVersionString() ) )
