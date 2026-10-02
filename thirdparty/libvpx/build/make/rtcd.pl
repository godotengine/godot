#!/usr/bin/env perl
##
##  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

no strict 'refs';
use warnings;
use Getopt::Long;
Getopt::Long::Configure("auto_help") if $Getopt::Long::VERSION > 2.32;

my %ALL_FUNCS = ();
my @ALL_ARCHS;
my @ALL_FORWARD_DECLS;
my @REQUIRES;

my %opts = ();
my %disabled = ();
my %required = ();

my @argv;
foreach (@ARGV) {
  $disabled{$1} = 1, next if /--disable-(.*)/;
  $required{$1} = 1, next if /--require-(.*)/;
  push @argv, $_;
}

# NB: use GetOptions() instead of GetOptionsFromArray() for compatibility.
@ARGV = @argv;
GetOptions(
  \%opts,
  'arch=s',
  'sym=s',
  'config=s',
);

foreach my $opt (qw/arch config/) {
  if (!defined($opts{$opt})) {
    warn "--$opt is required!\n";
    Getopt::Long::HelpMessage('-exit' => 1);
  }
}

foreach my $defs_file (@ARGV) {
  if (!-f $defs_file) {
    warn "$defs_file: $!\n";
    Getopt::Long::HelpMessage('-exit' => 1);
  }
}

open CONFIG_FILE, $opts{config} or
  die "Error opening config file '$opts{config}': $!\n";

my %config = ();
while (<CONFIG_FILE>) {
  next if !/^(?:CONFIG_|HAVE_)/;
  chomp;
  my @pair = split /=/;
  $config{$pair[0]} = $pair[1];
}
close CONFIG_FILE;

#
# Routines for the RTCD DSL to call
#
sub vpx_config($) {
  return (defined $config{$_[0]}) ? $config{$_[0]} : "";
}

sub specialize {
  if (@_ <= 1) {
    die "'specialize' must be called with a function name and at least one ",
        "architecture ('C' is implied): \n@_\n";
  }
  my $fn=$_[0];
  shift;
  foreach my $opt (@_) {
    eval "\$${fn}_${opt}=${fn}_${opt}";
  }
}

sub add_proto {
  my $fn = splice(@_, -2, 1);
  $ALL_FUNCS{$fn} = \@_;
  specialize $fn, "c";
}

sub require {
  foreach my $fn (keys %ALL_FUNCS) {
    foreach my $opt (@_) {
      my $ofn = eval "\$${fn}_${opt}";
      next if !$ofn;

      # if we already have a default, then we can disable it, as we know
      # we can do better.
      my $best = eval "\$${fn}_default";
      if ($best) {
        my $best_ofn = eval "\$${best}";
        if ($best_ofn && "$best_ofn" ne "$ofn") {
          eval "\$${best}_link = 'false'";
        }
      }
      eval "\$${fn}_default=${fn}_${opt}";
      eval "\$${fn}_${opt}_link='true'";
    }
  }
}

sub forward_decls {
  push @ALL_FORWARD_DECLS, @_;
}

#
# Include the user's directives
#
foreach my $f (@ARGV) {
  open FILE, "<", $f or die "cannot open $f: $!\n";
  my $contents = join('', <FILE>);
  close FILE;
  eval $contents or warn "eval failed: $@\n";
}

#
# Process the directives according to the command line
#
sub process_forward_decls() {
  foreach (@ALL_FORWARD_DECLS) {
    $_->();
  }
}

sub determine_indirection {
  vpx_config("CONFIG_RUNTIME_CPU_DETECT") eq "yes" or &require(@ALL_ARCHS);
  foreach my $fn (keys %ALL_FUNCS) {
    my $n = "";
    my @val = @{$ALL_FUNCS{$fn}};
    my $args = pop @val;
    my $rtyp = "@val";
    my $dfn = eval "\$${fn}_default";
    $dfn = eval "\$${dfn}";
    foreach my $opt (@_) {
      my $ofn = eval "\$${fn}_${opt}";
      next if !$ofn;
      my $link = eval "\$${fn}_${opt}_link";
      next if $link && $link eq "false";
      $n .= "x";
    }
    if ($n eq "x") {
      eval "\$${fn}_indirect = 'false'";
    } else {
      eval "\$${fn}_indirect = 'true'";
    }
  }
}

sub declare_function_pointers {
  foreach my $fn (sort keys %ALL_FUNCS) {
    my @val = @{$ALL_FUNCS{$fn}};
    my $args = pop @val;
    my $rtyp = "@val";
    my $dfn = eval "\$${fn}_default";
    $dfn = eval "\$${dfn}";
    foreach my $opt (@_) {
      my $ofn = eval "\$${fn}_${opt}";
      next if !$ofn;
      print "$rtyp ${ofn}($args);\n";
    }
    if (eval "\$${fn}_indirect" eq "false") {
      print "#define ${fn} ${dfn}\n";
    } else {
      print "RTCD_EXTERN $rtyp (*${fn})($args);\n";
    }
    print "\n";
  }
}

sub set_function_pointers {
  foreach my $fn (sort keys %ALL_FUNCS) {
    my @val = @{$ALL_FUNCS{$fn}};
    my $args = pop @val;
    my $rtyp = "@val";
    my $dfn = eval "\$${fn}_default";
    $dfn = eval "\$${dfn}";
    if (eval "\$${fn}_indirect" eq "true") {
      print "    $fn = $dfn;\n";
      foreach my $opt (@_) {
        my $ofn = eval "\$${fn}_${opt}";
        next if !$ofn;
        next if "$ofn" eq "$dfn";
        my $link = eval "\$${fn}_${opt}_link";
        next if $link && $link eq "false";
        my $cond = eval "\$have_${opt}";
        print "    if (${cond}) $fn = $ofn;\n"
      }
    }
  }
}

sub filter {
  my @filtered;
  foreach (@_) { push @filtered, $_ unless $disabled{$_}; }
  return @filtered;
}

#
# Helper functions for generating the arch specific RTCD files
#
sub common_top() {
  my $include_guard = uc($opts{sym})."_H_";
  my @time = localtime;
  my $year = $time[5] + 1900;
  print <<EOF;
/*
 *  Copyright (c) ${year} The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

// This file is generated. Do not edit.
#ifndef ${include_guard}
#define ${include_guard}

#ifdef RTCD_C
#define RTCD_EXTERN
#else
#define RTCD_EXTERN extern
#endif

EOF

process_forward_decls();
print <<EOF;

#ifdef __cplusplus
extern "C" {
#endif

EOF
declare_function_pointers("c", @ALL_ARCHS);

print <<EOF;
void $opts{sym}(void);

EOF
}

sub common_bottom() {
  my $include_guard = uc($opts{sym})."_H_";
  print <<EOF;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ${include_guard}
EOF
}

sub x86() {
  determine_indirection("c", @ALL_ARCHS);

  # Assign the helper variable for each enabled extension
  foreach my $opt (@ALL_ARCHS) {
    my $opt_uc = uc $opt;
    eval "\$have_${opt}=\"flags & HAS_${opt_uc}\"";
  }

  common_top;
  print <<EOF;
#ifdef RTCD_C
#include "vpx_ports/x86.h"
static void setup_rtcd_internal(void)
{
    int flags = x86_simd_caps();

    (void)flags;

EOF

  set_function_pointers("c", @ALL_ARCHS);

  print <<EOF;
}
#endif
EOF
  common_bottom;
}

sub arm() {
  determine_indirection("c", @ALL_ARCHS);

  # Assign the helper variable for each enabled extension
  foreach my $opt (@ALL_ARCHS) {
    my $opt_uc = uc $opt;
    # Enable neon assembly based on HAVE_NEON logic instead of adding new
    # HAVE_NEON_ASM logic
    if ($opt eq 'neon_asm') { $opt_uc = 'NEON' }
    eval "\$have_${opt}=\"flags & HAS_${opt_uc}\"";
  }

  common_top;
  print <<EOF;
#include "vpx_config.h"

#ifdef RTCD_C
#include "vpx_ports/arm.h"
static void setup_rtcd_internal(void)
{
    int flags = arm_cpu_caps();

    (void)flags;

EOF

  set_function_pointers("c", @ALL_ARCHS);

  print <<EOF;
}
#endif
EOF
  common_bottom;
}

sub mips() {
  determine_indirection("c", @ALL_ARCHS);

  # Assign the helper variable for each enabled extension
  foreach my $opt (@ALL_ARCHS) {
    my $opt_uc = uc $opt;
    eval "\$have_${opt}=\"flags & HAS_${opt_uc}\"";
  }

  common_top;

  print <<EOF;
#include "vpx_config.h"

#ifdef RTCD_C
#include "vpx_ports/mips.h"
static void setup_rtcd_internal(void)
{
    int flags = mips_cpu_caps();

    (void)flags;

EOF

  set_function_pointers("c", @ALL_ARCHS);

  print <<EOF;
#if HAVE_DSPR2
void vpx_dsputil_static_init();
#if CONFIG_VP8
void dsputil_static_init();
#endif

vpx_dsputil_static_init();
#if CONFIG_VP8
dsputil_static_init();
#endif
#endif
}
#endif
EOF
  common_bottom;
}

sub ppc() {
  determine_indirection("c", @ALL_ARCHS);

  # Assign the helper variable for each enabled extension
  foreach my $opt (@ALL_ARCHS) {
    my $opt_uc = uc $opt;
    eval "\$have_${opt}=\"flags & HAS_${opt_uc}\"";
  }

  common_top;
  print <<EOF;
#include "vpx_config.h"

#ifdef RTCD_C
#include "vpx_ports/ppc.h"
static void setup_rtcd_internal(void)
{
    int flags = ppc_simd_caps();
    (void)flags;
EOF

  set_function_pointers("c", @ALL_ARCHS);

  print <<EOF;
}
#endif
EOF
  common_bottom;
}

sub loongarch() {
  determine_indirection("c", @ALL_ARCHS);

  # Assign the helper variable for each enabled extension
  foreach my $opt (@ALL_ARCHS) {
    my $opt_uc = uc $opt;
    eval "\$have_${opt}=\"flags & HAS_${opt_uc}\"";
  }

  common_top;
  print <<EOF;
#include "vpx_config.h"

#ifdef RTCD_C
#include "vpx_ports/loongarch.h"
static void setup_rtcd_internal(void)
{
    int flags = loongarch_cpu_caps();

    (void)flags;
EOF

  set_function_pointers("c", @ALL_ARCHS);

  print <<EOF;
}
#endif
EOF
  common_bottom;
}

sub unoptimized() {
  determine_indirection "c";
  common_top;
  print <<EOF;
#include "vpx_config.h"

#ifdef RTCD_C
static void setup_rtcd_internal(void)
{
EOF

  set_function_pointers "c";

  print <<EOF;
}
#endif
EOF
  common_bottom;
}

# List of architectures in low-to-high preference order.
my @PRIORITY_ARCH = qw/
  c
  mmx sse sse2 sse3 ssse3 sse4_1 sse4_2 avx avx2 avx512
  arm_crc32 neon neon_dotprod neon_i8mm sve sve2
  rvv
  vsx
  dspr2 msa
/;
my %PRIORITY_INDEX;
for (my $i = 0; $i < @PRIORITY_ARCH; $i++) {
  $PRIORITY_INDEX{$PRIORITY_ARCH[$i]} = $i;
}

#
# Main Driver
#

&require("c");
&require(sort { $PRIORITY_INDEX{$a} <=> $PRIORITY_INDEX{$b} } keys %required);
if ($opts{arch} eq 'x86') {
  @ALL_ARCHS = filter(qw/mmx sse sse2 sse3 ssse3 sse4_1 avx avx2 avx512/);
  x86;
} elsif ($opts{arch} eq 'x86_64') {
  @ALL_ARCHS = filter(qw/mmx sse sse2 sse3 ssse3 sse4_1 avx avx2 avx512/);
  if (keys %required == 0) {
    @REQUIRES = filter(qw/mmx sse sse2/);
    &require(@REQUIRES);
  }
  x86;
} elsif ($opts{arch} eq 'mips32' || $opts{arch} eq 'mips64') {
  my $have_dspr2 = 0;
  my $have_msa = 0;
  my $have_mmi = 0;
  @ALL_ARCHS = filter("$opts{arch}");
  open CONFIG_FILE, $opts{config} or
    die "Error opening config file '$opts{config}': $!\n";
  while (<CONFIG_FILE>) {
    if (/HAVE_DSPR2=yes/) {
      $have_dspr2 = 1;
    }
    if (/HAVE_MSA=yes/) {
      $have_msa = 1;
    }
    if (/HAVE_MMI=yes/) {
      $have_mmi = 1;
    }
  }
  close CONFIG_FILE;
  if ($have_dspr2 == 1) {
    @ALL_ARCHS = filter("$opts{arch}", qw/dspr2/);
  } elsif ($have_msa == 1 && $have_mmi == 1) {
    @ALL_ARCHS = filter("$opts{arch}", qw/mmi msa/);
  } elsif ($have_msa == 1) {
    @ALL_ARCHS = filter("$opts{arch}", qw/msa/);
  } elsif ($have_mmi == 1) {
    @ALL_ARCHS = filter("$opts{arch}", qw/mmi/);
  } else {
    unoptimized;
  }
  mips;
} elsif ($opts{arch} =~ /armv7\w?/) {
  @ALL_ARCHS = filter(qw/neon_asm neon/);
  arm;
} elsif ($opts{arch} eq 'armv8' || $opts{arch} eq 'arm64' ) {
  @ALL_ARCHS = filter(qw/neon neon_dotprod neon_i8mm sve sve2/);
  if (keys %required == 0) {
    @REQUIRES = filter(qw/neon/);
    &require(@REQUIRES);
  }
  arm;
} elsif ($opts{arch} =~ /^ppc/ ) {
  @ALL_ARCHS = filter(qw/vsx/);
  ppc;
} elsif ($opts{arch} =~ /loongarch/ ) {
  @ALL_ARCHS = filter(qw/lsx lasx/);
  loongarch;
} else {
  unoptimized;
}

__END__

=head1 NAME

rtcd -

=head1 SYNOPSIS

Usage: rtcd.pl [options] FILE

See 'perldoc rtcd.pl' for more details.

=head1 DESCRIPTION

Reads the Run Time CPU Detections definitions from FILE and generates a
C header file on stdout.

=head1 OPTIONS

Options:
  --arch=ARCH       Architecture to generate defs for (required)
  --disable-EXT     Disable support for EXT extensions
  --require-EXT     Require support for EXT extensions
  --sym=SYMBOL      Unique symbol to use for RTCD initialization function
  --config=FILE     File with CONFIG_FOO=yes lines to parse
