#!/usr/bin/env perl
##
##  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##


# ads2gas_apple.pl
# Author: Eric Fung (efung (at) acm.org)
#
# Convert ARM Developer Suite 1.0.1 syntax assembly source to GNU as format
#
# Usage: cat inputfile | perl ads2gas_apple.pl > outputfile
#

print "@ This file was created from a .asm file\n";
print "@  using the ads2gas_apple.pl script.\n\n";
print ".syntax unified\n";

my %macro_aliases;

my @mapping_list = ("\$0", "\$1", "\$2", "\$3", "\$4", "\$5", "\$6", "\$7", "\$8", "\$9");

my @incoming_array;

# Perl trim function to remove whitespace from the start and end of the string
sub trim($)
{
    my $string = shift;
    $string =~ s/^\s+//;
    $string =~ s/\s+$//;
    return $string;
}

while (<STDIN>)
{
    # Load and store alignment
    s/@/,:/g;

    # Comment character
    s/;/@/;

    # Convert ELSE to .else
    s/\bELSE\b/.else/g;

    # Convert ENDIF to .endif
    s/\bENDIF\b/.endif/g;

    # Convert IF to .if
    if (s/\bIF\b/.if/g) {
        s/=+/==/g;
    }

    # Convert INCLUDE to .INCLUDE "file"
    s/INCLUDE\s?(.*)$/.include \"$1\"/;

    # No AREA required
    # But ALIGNs in AREA must be obeyed
    s/^(\s*)\bAREA\b.*ALIGN=([0-9])$/$1.text\n$1.p2align $2/;
    # If no ALIGN, strip the AREA and align to 4 bytes
    s/^(\s*)\bAREA\b.*$/$1.text\n$1.p2align 2/;

    # Make function visible to linker.
    s/EXPORT\s+\|([\$\w]*)\|/.globl _$1/;

    # No vertical bars on function names
    s/^\|(\$?\w+)\|/$1/g;

    # Labels and functions need a leading underscore and trailing colon
    s/^([a-zA-Z_0-9\$]+)/_$1:/ if !/EQU/;

    # Branches need to call the correct, underscored, function
    s/^(\s+b[egln]?[teq]?\s+)([a-zA-Z_0-9\$]+)/$1 _$2/ if !/EQU/;

    # ALIGN directive
    s/\bALIGN\b/.balign/g;

    # Strip ARM
    s/\s+ARM//;

    # Strip REQUIRE8
    s/\s+REQUIRE8//;

    # Strip PRESERVE8
    s/\s+PRESERVE8//;

    # Strip PROC and ENDPROC
    s/\bPROC\b//g;
    s/\bENDP\b//g;

    # EQU directive
    s/(\S+\s+)EQU(\s+\S+)/.equ $1, $2/;

    # Begin macro definition
    if (/\bMACRO\b/) {
        # Process next line down, which will be the macro definition
        $_ = <STDIN>;
        s/^/.macro/;
        s/\$//g;             # Remove $ from the variables in the declaration
    }

    s/\$/\\/g;               # Use \ to reference formal parameters
    # End macro definition

    s/\bMEND\b/.endm/;       # No need to tell it where to stop assembling
    next if /^\s*END\s*$/;
    s/[ \t]+$//;
    print;
}
