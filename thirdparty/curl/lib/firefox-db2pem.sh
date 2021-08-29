#!/bin/sh
# ***************************************************************************
# *                                  _   _ ____  _
# *  Project                     ___| | | |  _ \| |
# *                             / __| | | | |_) | |
# *                            | (__| |_| |  _ <| |___
# *                             \___|\___/|_| \_\_____|
# *
# * Copyright (C) 1998 - 2020, Daniel Stenberg, <daniel@haxx.se>, et al.
# *
# * This software is licensed as described in the file COPYING, which
# * you should have received as part of this distribution. The terms
# * are also available at https://curl.se/docs/copyright.html.
# *
# * You may opt to use, copy, modify, merge, publish, distribute and/or sell
# * copies of the Software, and permit persons to whom the Software is
# * furnished to do so, under the terms of the COPYING file.
# *
# * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
# * KIND, either express or implied.
# *
# ***************************************************************************
# This shell script creates a fresh ca-bundle.crt file for use with libcurl.
# It extracts all ca certs it finds in the local Firefox database and converts
# them all into PEM format.
#
db=`ls -1d $HOME/.mozilla/firefox/*default*`
out=$1

if test -z "$out"; then
  out="ca-bundle.crt" # use a sensible default
fi

currentdate=`date`

cat >$out <<EOF
##
## Bundle of CA Root Certificates
##
## Converted at: ${currentdate}
## These were converted from the local Firefox directory by the db2pem script.
##
EOF


certutil -L -h 'Builtin Object Token' -d $db | \
grep ' *[CcGTPpu]*,[CcGTPpu]*,[CcGTPpu]* *$' | \
sed -e 's/ *[CcGTPpu]*,[CcGTPpu]*,[CcGTPpu]* *$//' -e 's/\(.*\)/"\1"/' | \
sort | \
while read nickname; \
 do echo $nickname | sed -e "s/Builtin Object Token://g"; \
eval certutil -d $db -L -n "$nickname" -a ; \
done >> $out
