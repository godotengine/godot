#!/usr/bin/perl

my $bigend;  # little/big endian
my $nxstack;

$nxstack = 0;

eval 'exec /usr/local/bin/perl -S $0 ${1+"$@"}'
    if $running_under_some_shell;

while ($ARGV[0] =~ /^-/) {
    $_ = shift;
  last if /^--/;
    if (/^-n/) {
    $nflag++;
    next;
    }
    die "I don't recognize this switch: $_\\n";
}
$printit++ unless $nflag;

$\ = "\n";      # automatically add newline on print
$n=0;

$thumb = 0;     # ARM mode by default, not Thumb.
@proc_stack = ();

LINE:
while (<>) {

    # For ADRLs we need to add a new line after the substituted one.
    $addPadding = 0;

    # First, we do not dare to touch *anything* inside double quotes, do we?
    # Second, if you want a dollar character in the string,
    # insert two of them -- that's how ARM C and assembler treat strings.
    s/^([A-Za-z_]\w*)[ \t]+DCB[ \t]*\"/$1:   .ascii \"/   && do { s/\$\$/\$/g; next };
    s/\bDCB\b[ \t]*\"/.ascii \"/                          && do { s/\$\$/\$/g; next };
    s/^(\S+)\s+RN\s+(\S+)/$1 .req r$2/                    && do { s/\$\$/\$/g; next };
    # If there's nothing on a line but a comment, don't try to apply any further
    #  substitutions (this is a cheap hack to avoid mucking up the license header)
    s/^([ \t]*);/$1@/                                     && do { s/\$\$/\$/g; next };
    # If substituted -- leave immediately !

    s/@/,:/;
    s/;/@/;
    while ( /@.*'/ ) {
      s/(@.*)'/$1/g;
    }
    s/\{FALSE\}/0/g;
    s/\{TRUE\}/1/g;
    s/\{(\w\w\w\w+)\}/$1/g;
    s/\bINCLUDE[ \t]*([^ \t\n]+)/.include \"$1\"/;
    s/\bGET[ \t]*([^ \t\n]+)/.include \"${ my $x=$1; $x =~ s|\.s|-gnu.S|; \$x }\"/;
    s/\bIMPORT\b/.extern/;
    s/\bEXPORT\b/.global/;
    s/^(\s+)\[/$1IF/;
    s/^(\s+)\|/$1ELSE/;
    s/^(\s+)\]/$1ENDIF/;
    s/IF *:DEF:/ .ifdef/;
    s/IF *:LNOT: *:DEF:/ .ifndef/;
    s/ELSE/ .else/;
    s/ENDIF/ .endif/;

    if( /\bIF\b/ ) {
      s/\bIF\b/ .if/;
      s/=/==/;
    }
    if ( $n == 2) {
        s/\$/\\/g;
    }
    if ($n == 1) {
        s/\$//g;
        s/label//g;
    $n = 2;
      }
    if ( /MACRO/ ) {
      s/MACRO *\n/.macro/;
      $n=1;
    }
    if ( /\bMEND\b/ ) {
      s/\bMEND\b/.endm/;
      $n=0;
    }

    # ".rdata" doesn't work in 'as' version 2.13.2, as it is ".rodata" there.
    #
    if ( /\bAREA\b/ ) {
        my $align;
        $align = "2";
        if ( /ALIGN=(\d+)/ ) {
            $align = $1;
        }
        if ( /CODE/ ) {
            $nxstack = 1;
        }
        s/^(.+)CODE(.+)READONLY(.*)/    .text/;
        s/^(.+)DATA(.+)READONLY(.*)/    .section .rdata/;
        s/^(.+)\|\|\.data\|\|(.+)/    .data/;
        s/^(.+)\|\|\.bss\|\|(.+)/    .bss/;
        s/$/;   .p2align $align/;
    }

    s/\|\|\.constdata\$(\d+)\|\|/.L_CONST$1/;       # ||.constdata$3||
    s/\|\|\.bss\$(\d+)\|\|/.L_BSS$1/;               # ||.bss$2||
    s/\|\|\.data\$(\d+)\|\|/.L_DATA$1/;             # ||.data$2||
    s/\|\|([a-zA-Z0-9_]+)\@([a-zA-Z0-9_]+)\|\|/@ $&/;
    s/^(\s+)\%(\s)/    .space $1/;

    s/\|(.+)\.(\d+)\|/\.$1_$2/;                     # |L80.123| -> .L80_123
    s/\bCODE32\b/.code 32/ && do {$thumb = 0};
    s/\bCODE16\b/.code 16/ && do {$thumb = 1};
    if (/\bPROC\b/)
    {
        my $prefix;
        my $proc;
        /^([A-Za-z_\.]\w+)\b/;
        $proc = $1;
        $prefix = "";
        if ($proc)
        {
            $prefix = $prefix.sprintf("\t.type\t%s, %%function; ",$proc);
            push(@proc_stack, $proc);
            s/^[A-Za-z_\.]\w+/$&:/;
        }
        $prefix = $prefix."\t.thumb_func; " if ($thumb);
        s/\bPROC\b/@ $&/;
        $_ = $prefix.$_;
    }
    s/^(\s*)(S|Q|SH|U|UQ|UH)ASX\b/$1$2ADDSUBX/;
    s/^(\s*)(S|Q|SH|U|UQ|UH)SAX\b/$1$2SUBADDX/;
    if (/\bENDP\b/)
    {
        my $proc;
        s/\bENDP\b/@ $&/;
        $proc = pop(@proc_stack);
        $_ = "\t.size $proc, .-$proc".$_ if ($proc);
    }
    s/\bSUBT\b/@ $&/;
    s/\bDATA\b/@ $&/;   # DATA directive is deprecated -- Asm guide, p.7-25
    s/\bKEEP\b/@ $&/;
    s/\bEXPORTAS\b/@ $&/;
    s/\|\|(.)+\bEQU\b/@ $&/;
    s/\|\|([\w\$]+)\|\|/$1/;
    s/\bENTRY\b/@ $&/;
    s/\bASSERT\b/@ $&/;
    s/\bGBLL\b/@ $&/;
    s/\bGBLA\b/@ $&/;
    s/^\W+OPT\b/@ $&/;
    s/:OR:/|/g;
    s/:SHL:/<</g;
    s/:SHR:/>>/g;
    s/:AND:/&/g;
    s/:LAND:/&&/g;
    s/CPSR/cpsr/;
    s/SPSR/spsr/;
    s/ALIGN$/.balign 4/;
    s/ALIGN\s+([0-9x]+)$/.balign $1/;
    s/psr_cxsf/psr_all/;
    s/LTORG/.ltorg/;
    s/^([A-Za-z_]\w*)[ \t]+EQU/ .set $1,/;
    s/^([A-Za-z_]\w*)[ \t]+SETL/ .set $1,/;
    s/^([A-Za-z_]\w*)[ \t]+SETA/ .set $1,/;
    s/^([A-Za-z_]\w*)[ \t]+\*/ .set $1,/;

    #  {PC} + 0xdeadfeed  -->  . + 0xdeadfeed
    s/\{PC\} \+/ \. +/;

    # Single hex constant on the line !
    #
    # >>> NOTE <<<
    #   Double-precision floats in gcc are always mixed-endian, which means
    #   bytes in two words are little-endian, but words are big-endian.
    #   So, 0x0000deadfeed0000 would be stored as 0x0000dead at low address
    #   and 0xfeed0000 at high address.
    #
    s/\bDCFD\b[ \t]+0x([a-fA-F0-9]{8})([a-fA-F0-9]{8})/.long 0x$1, 0x$2/;
    # Only decimal constants on the line, no hex !
    s/\bDCFD\b[ \t]+([0-9\.\-]+)/.double $1/;

    # Single hex constant on the line !
#    s/\bDCFS\b[ \t]+0x([a-f0-9]{8})([a-f0-9]{8})/.long 0x$1, 0x$2/;
    # Only decimal constants on the line, no hex !
#    s/\bDCFS\b[ \t]+([0-9\.\-]+)/.double $1/;
    s/\bDCFS[ \t]+0x/.word 0x/;
    s/\bDCFS\b/.float/;

    s/^([A-Za-z_]\w*)[ \t]+DCD/$1 .word/;
    s/\bDCD\b/.word/;
    s/^([A-Za-z_]\w*)[ \t]+DCW/$1 .short/;
    s/\bDCW\b/.short/;
    s/^([A-Za-z_]\w*)[ \t]+DCB/$1 .byte/;
    s/\bDCB\b/.byte/;
    s/^([A-Za-z_]\w*)[ \t]+\%/.comm $1,/;
    s/^[A-Za-z_\.]\w+/$&:/;
    s/^(\d+)/$1:/;
    s/\%(\d+)/$1b_or_f/;
    s/\%[Bb](\d+)/$1b/;
    s/\%[Ff](\d+)/$1f/;
    s/\%[Ff][Tt](\d+)/$1f/;
    s/&([\dA-Fa-f]+)/0x$1/;
    if ( /\b2_[01]+\b/ ) {
      s/\b2_([01]+)\b/conv$1&&&&/g;
      while ( /[01][01][01][01]&&&&/ ) {
        s/0000&&&&/&&&&0/g;
        s/0001&&&&/&&&&1/g;
        s/0010&&&&/&&&&2/g;
        s/0011&&&&/&&&&3/g;
        s/0100&&&&/&&&&4/g;
        s/0101&&&&/&&&&5/g;
        s/0110&&&&/&&&&6/g;
        s/0111&&&&/&&&&7/g;
        s/1000&&&&/&&&&8/g;
        s/1001&&&&/&&&&9/g;
        s/1010&&&&/&&&&A/g;
        s/1011&&&&/&&&&B/g;
        s/1100&&&&/&&&&C/g;
        s/1101&&&&/&&&&D/g;
        s/1110&&&&/&&&&E/g;
        s/1111&&&&/&&&&F/g;
      }
      s/000&&&&/&&&&0/g;
      s/001&&&&/&&&&1/g;
      s/010&&&&/&&&&2/g;
      s/011&&&&/&&&&3/g;
      s/100&&&&/&&&&4/g;
      s/101&&&&/&&&&5/g;
      s/110&&&&/&&&&6/g;
      s/111&&&&/&&&&7/g;
      s/00&&&&/&&&&0/g;
      s/01&&&&/&&&&1/g;
      s/10&&&&/&&&&2/g;
      s/11&&&&/&&&&3/g;
      s/0&&&&/&&&&0/g;
      s/1&&&&/&&&&1/g;
      s/conv&&&&/0x/g;
    }

    if ( /commandline/)
    {
        if( /-bigend/)
        {
            $bigend=1;
        }
    }

    if ( /\bDCDU\b/ )
    {
        my $cmd=$_;
        my $value;
        my $prefix;
        my $w1;
        my $w2;
        my $w3;
        my $w4;

        s/\s+DCDU\b/@ $&/;

        $cmd =~ /\bDCDU\b\s+0x(\d+)/;
        $value = $1;
        $value =~ /(\w\w)(\w\w)(\w\w)(\w\w)/;
        $w1 = $1;
        $w2 = $2;
        $w3 = $3;
        $w4 = $4;

        if( $bigend ne "")
        {
            # big endian
            $prefix = "\t.byte\t0x".$w1.";".
                      "\t.byte\t0x".$w2.";".
                      "\t.byte\t0x".$w3.";".
                      "\t.byte\t0x".$w4."; ";
        }
        else
        {
            # little endian
            $prefix = "\t.byte\t0x".$w4.";".
                      "\t.byte\t0x".$w3.";".
                      "\t.byte\t0x".$w2.";".
                      "\t.byte\t0x".$w1."; ";
        }
        $_=$prefix.$_;
    }

    if ( /\badrl\b/i )
    {
        s/\badrl\s+(\w+)\s*,\s*(\w+)/ldr $1,=$2/i;
        $addPadding = 1;
    }
    s/\bEND\b/@ END/;
} continue {
    printf ("%s", $_) if $printit;
    if ($addPadding != 0)
    {
        printf ("   mov r0,r0\n");
        $addPadding = 0;
    }
}
#If we had a code section, mark that this object doesn't need an executable
# stack.
if ($nxstack) {
    printf ("    .section\t.note.GNU-stack,\"\",\%\%progbits\n");
}
