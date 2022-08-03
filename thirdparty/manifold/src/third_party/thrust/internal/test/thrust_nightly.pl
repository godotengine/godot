#! /usr/bin/perl

###############################################################################
# Copyright (c) 2018 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

use strict;
use warnings;

print(`perl --version`);

use Getopt::Long;
use Cwd;
use Cwd "abs_path";
use Config; # For signal names and numbers.
use IPC::Open2;
use File::Temp;
use POSIX "strftime";

my $have_time_hi_res = 0;

if (eval { require Time::HiRes })
{
  printf("#### CONFIG timestamp `gettimeofday`\n");

  import Time::HiRes "gettimeofday";

  $have_time_hi_res = 1;
} else {
  printf("#### CONFIG timestamp `time`\n");
}

sub timestamp()
{
  if ($have_time_hi_res) {
    return gettimeofday();
  } else {
    return time();
  }
}

my %CmdLineOption;
my $arch                = "";
my $abi                 = "";
my $os                  = "";
my $build               = "release";
my $bin_path;
my $filecheck_path;
my $filecheck_data_path = "internal/test";
my $timeout_min         = 15;

# https://stackoverflow.com/questions/29862178/name-of-signal-number-2
my @sig_names;
@sig_names[ split ' ', $Config{sig_num} ] = split ' ', $Config{sig_name};
my %sig_nums;
@sig_nums{ split ' ', $Config{sig_name} } = split ' ', $Config{sig_num};

if (`uname` =~ m/CYGWIN/) {
  $os = "win32";
} elsif ($^O eq "MSWin32") {
  $os = "win32";
} else {
  $os = `uname`;
  chomp($os);
}

if ($os eq "win32") {
  $ENV{'PROCESSOR_ARCHITECTURE'} ||= "";
  $ENV{'PROCESSOR_ARCHITEW6432'} ||= "";

  if ((lc($ENV{PROCESSOR_ARCHITECTURE}) ne "x86") ||
      (lc($ENV{PROCESSOR_ARCHITECTURE}) eq "amd64") ||
      (lc($ENV{PROCESSOR_ARCHITEW6432}) eq "amd64")) {
    $arch = "x86_64";
  } else {
    $arch = "i686";
  }
} else {
  $arch = `uname -m`;
  chomp($arch);
}

sub usage()
{
  printf("Usage: thrust_nightly.pl <options>\n");
  printf("Options:\n");
  printf("  -help                         : Print help message\n");
  printf("  -forcearch <arch>             : i686|x86_64|ARMv7|aarch64 (default: $arch)\n");
  printf("  -forceabi <abi>               : Specify abi to be used for arm (gnueabi|gnueabihf)\n");
  printf("  -forceos <os>                 : win32|Linux|Darwin (default: $os)\n");
  printf("  -build <release|debug>        : (default: debug)\n");
  printf("  -bin-path <path>              : Specify location of test binaries\n");
  printf("  -filecheck-path <path>        : Specify location of filecheck binary\n");
  printf("  -filecheck-data-path <path>   : Specify location of filecheck data (default: $filecheck_data_path)\n");
  printf("  -timeout-min <min>            : timeout in minutes for each individual test\n");
}

GetOptions(\%CmdLineOption,
           'help' => sub { usage() and exit 0 },
           "forcearch=s" => \$arch,
           "forceabi=s" => \$abi,
           "forceos=s" => \$os,
           "build=s" => \$build,
           "bin-path=s" => \$bin_path,
           "filecheck-path=s" => \$filecheck_path,
           "filecheck-data-path=s" => \$filecheck_data_path,
           "timeout-min=i" => \$timeout_min,
          );

my $pwd = getcwd();
my $bin_path_root = abs_path ("${pwd}/..");

if ($arch eq "ARMv7") {
      if ($abi eq "") {
          $abi = "_gnueabi";  #Use default abi for arm if not specified
      }
      else {
          $abi = "_${abi}";
      }
}
else {
    $abi = "";                #Ignore abi for architectures other than arm
}

my $uname = "";
$uname = $arch;
chomp($uname);

if (not $bin_path) {
    $bin_path = "${bin_path_root}/bin/${uname}_${os}${abi}_${build}";
}

if (not $filecheck_path) {
    $filecheck_path = "${bin_path}/nvvm/tools";
}

sub process_return_code {
    my ($name, $ret, $msg) = @_;

    if ($ret != 0) {
        my $signal  = $ret & 127;
        my $app_exit = $ret >> 8;
        my $dumped_core = $ret & 0x80;
        if (($app_exit != 0) && ($app_exit != 0)) {
            if ($msg ne "") {
                printf("#### ERROR $name exited with return value $app_exit. $msg\n");
            } else {
                printf("#### ERROR $name exited with return value $app_exit.\n");
            }
        }
        if ($signal != 0) {
            if ($msg ne "") {
                printf("#### ERROR $name received signal SIG$sig_names[$signal] ($signal). $msg\n");
            } else {
                printf("#### ERROR $name received signal SIG$sig_names[$signal] ($signal).\n");
            }
            if ($sig_nums{'INT'} eq $signal) {
                die("Terminating testing due to SIGINT.");
            }
        }
        if ($dumped_core != 0) {
            if ($msg ne "") {
                printf("#### ERROR $name generated a core dump. $msg\n");
            } else {
                printf("#### ERROR $name generated a core dump.\n");
            }
        }
    }
}

my $have_filecheck = 1;

sub filecheck_smoke_test {
    my $filecheck_cmd = "$filecheck_path/FileCheck $filecheck_data_path/thrust.smoke.filecheck";

    my $filecheck_pid = open(my $filecheck_stdin, "|-", "$filecheck_cmd 2>&1");

    print $filecheck_stdin "SMOKE";

    my $filecheck_ret = 0;
    if (close($filecheck_stdin) == 0)
    {
      $filecheck_ret = $?;
    }

    if ($filecheck_ret == 0) {
      printf("&&&& PASSED FileCheck\n");
    } else {
      # Use a temporary file to send the output to
      # FileCheck so we can get the output this time,
      # because Perl and bidirectional pipes suck.
      my $tmp = File::Temp->new();
      my $tmp_filename = $tmp->filename;
      print $tmp "SMOKE";

      printf("********************************************************************************\n");
      print `$filecheck_cmd -input-file $tmp_filename`;
      printf("********************************************************************************\n");

      process_return_code("FileCheck Test", $filecheck_ret, "");
      printf("&&&& FAILED FileCheck\n");

      $have_filecheck = 0;
    }
}

# Wrapper for system that logs the commands so you can see what it did
sub run_cmd {
    my ($cmd) = @_;
    my $ret = 0;
    my @executable;
    my @output;
    my $syst_cmd;

    my $start = timestamp();
    eval {
        local $SIG{ALRM} = sub { die("Command timed out (received SIGALRM).\n") };
        alarm (60 * $timeout_min);
        $syst_cmd = $cmd;

        @executable = split(' ', $syst_cmd, 2);

        open(my $child, "-|", "$syst_cmd") or die("Could not execute $syst_cmd.\n");

        if ($child)
        {
          @output = <$child>;
        }

        if (close($child) == 0)
        {
          $ret = $?;
        }

        alarm 0;
    };
    my $elapsed = timestamp() - $start;

    if ($@) {
        printf("\n#### ERROR Command timeout reached, killing $executable[0].\n");
        system("killall ".$executable[0]);
        return ($sig_nums{'KILL'}, $elapsed, @output);
    }

    return ($ret, $elapsed, @output);
}

sub current_time
{
   return strftime("%x %X %Z", localtime());
}

my $failures = 0;
my $known_failures = 0;
my $errors = 0;
my $passes = 0;

sub run_examples {
    # Get list of tests in binary folder.
    my $dir = cwd();
    chdir $bin_path;
    my @examplelist;
    if ($os eq "win32")
    {
        @examplelist = glob('thrust.example.*.exe');
    } else {
        @examplelist = glob('thrust.example.*');
    }

    chdir $dir;

    my $test;
    foreach $test (@examplelist)
    {
        my $test_exe = $test;

        # Ignore FileCheck files.
        if ($test =~ /[.]filecheck$/)
        {
          next;
        }

        if ($os eq "win32")
        {
          $test =~ s/\.exe//g;
        }

        # Check the test actually exists.
        if (!-e "${bin_path}/${test_exe}")
        {
          next;
        }

        my $cmd = "${bin_path}/${test_exe} --verbose 2>&1";

        printf("&&&& RUNNING $test\n");
        printf("#### CURRENT_TIME " . current_time() . "\n");

        my ($ret, $elapsed, @output) = run_cmd($cmd);

        printf("********************************************************************************\n");
        print @output;
        printf("********************************************************************************\n");

        if ($ret != 0) {
            process_return_code($test, $ret, "Example crash?");
            printf("&&&& FAILED $test\n");
            printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
            $errors = $errors + 1;
        } else {
            printf("&&&& PASSED $test\n");
            printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
            $passes = $passes + 1;

            if ($have_filecheck) {
                # Check output with LLVM FileCheck.

                printf("&&&& RUNNING FileCheck $test\n");

                if (-f "${filecheck_data_path}/${test}.filecheck") {
                    # If the filecheck file is empty, don't use filecheck, just
                    # check if the output file is also empty.
                    if (-z "${filecheck_data_path}/${test}.filecheck") {
                        if (join("", @output) eq "") {
                            printf("&&&& PASSED FileCheck $test\n");
                            $passes = $passes + 1;
                        } else {
                            printf("#### ERROR Output received but not expected.\n");
                            printf("&&&& FAILED FileCheck $test\n");
                            $failures = $failures + 1;
                        }
                    } else {
                        my $filecheck_cmd = "$filecheck_path/FileCheck $filecheck_data_path/$test.filecheck";

                        my $filecheck_pid = open(my $filecheck_stdin, "|-", "$filecheck_cmd 2>&1");

                        print $filecheck_stdin @output;

                        my $filecheck_ret = 0;
                        if (close($filecheck_stdin) == 0)
                        {
                          $filecheck_ret = $?;
                        }

                        if ($filecheck_ret == 0) {
                          printf("&&&& PASSED FileCheck $test\n");
                          $passes = $passes + 1;
                        } else {
                          # Use a temporary file to send the output to
                          # FileCheck so we can get the output this time,
                          # because Perl and bidirectional pipes suck.
                          my $tmp = File::Temp->new();
                          my $tmp_filename = $tmp->filename;
                          print $tmp @output;

                          printf("********************************************************************************\n");
                          print `$filecheck_cmd -input-file $tmp_filename`;
                          printf("********************************************************************************\n");

                          process_return_code("FileCheck $test", $filecheck_ret, "");
                          printf("&&&& FAILED FileCheck $test\n");
                          $failures = $failures + 1;
                        }
                    }
                } else {
                    printf("#### ERROR $test has no FileCheck comparison.\n");
                    printf("&&&& FAILED FileCheck $test\n");
                    $errors = $errors + 1;
                }
            }
        }
        printf("\n");
    }
}

sub run_unit_tests {
    # Get list of tests in binary folder.
    my $dir = cwd();
    chdir $bin_path;
    my @unittestlist;
    if ($os eq "win32")
    {
        @unittestlist = glob('thrust.test.*.exe');
    } else {
        @unittestlist = glob('thrust.test.*');
    }
    chdir $dir;

    my $test;
    foreach $test (@unittestlist)
    {
        my $test_exe = $test;

        # Ignore FileCheck files.
        if ($test =~ /[.]filecheck$/)
        {
          next;
        }

        if ($os eq "win32")
        {
          $test =~ s/\.exe//g;
        }

        # Check the test actually exists.
        if (!-e "${bin_path}/${test_exe}")
        {
          next;
        }

        # Check the test actually exists
        next unless (-e "${bin_path}/${test_exe}");

        my $cmd = "${bin_path}/${test_exe} --verbose 2>&1";

        printf("&&&& RUNNING $test\n");
        printf("#### CURRENT_TIME " . current_time() . "\n");

        my ($ret, $elapsed, @output) = run_cmd($cmd);

        printf("********************************************************************************\n");
        print @output;
        printf("********************************************************************************\n");
        my $fail = 0;
        my $known_fail = 0;
        my $error = 0;
        my $pass = 0;
        my $found_totals = 0;
        foreach my $line (@output)
        {
            if (($fail, $known_fail, $error, $pass) = $line =~ /Totals: ([0-9]+) failures, ([0-9]+) known failures, ([0-9]+) errors, and ([0-9]+) passes[.]/igs) {
              $found_totals = 1;
              $failures = $failures + $fail;
              $known_failures = $known_failures + $known_fail;
              $errors = $errors + $error;
              $passes = $passes + $pass;
              last;
            } else {
              $fail = 0;
              $known_fail = 0;
              $error = 0;
              $pass = 0;
            }
        }
        if ($ret == 0) {
            if ($found_totals == 0) {
                $errors = $errors + 1;
                printf("#### ERROR $test returned 0 and no summary line was found. Invalid test?\n");
                printf("&&&& FAILED $test\n");
                printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
            }
            else {
                if ($fail != 0 or $error != 0) {
                    $errors = $errors + 1;
                    printf("#### ERROR $test returned 0 and had failures or errors. Test driver error?\n");
                    printf("&&&& FAILED $test\n");
                    printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
                } elsif ($known_fail == 0 and $pass == 0) {
                    printf("#### DISABLED $test returned 0 and had no failures, known failures, errors or passes.\n");
                    printf("&&&& PASSED $test\n");
                    printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
                } else {
                    printf("&&&& PASSED $test\n");
                    printf("#### WALLTIME $test %.2f [s]\n", $elapsed);

                    if ($have_filecheck) {
                        # Check output with LLVM FileCheck if the test has a FileCheck input.

                        if (-f "${filecheck_data_path}/${test}.filecheck") {
                            printf("&&&& RUNNING FileCheck $test\n");

                            # If the filecheck file is empty, don't use filecheck,
                            # just check if the output file is also empty.
                            if (! -z "${filecheck_data_path}/${test}.filecheck") {
                                if (@output) {
                                    printf("&&&& PASSED FileCheck $test\n");
                                    $passes = $passes + 1;
                                } else {
                                    printf("#### Output received but not expected.\n");
                                    printf("&&&& FAILED FileCheck $test\n");
                                    $failures = $failures + 1;
                                }
                            } else {
                                my $filecheck_cmd = "$filecheck_path/FileCheck $filecheck_data_path/$test.filecheck";

                                my $filecheck_pid = open(my $filecheck_stdin, "|-", "$filecheck_cmd 2>&1");

                                print $filecheck_stdin @output;

                                my $filecheck_ret = 0;
                                if (close($filecheck_stdin) == 0)
                                {
                                  $filecheck_ret = $?;
                                }

                                if ($filecheck_ret == 0) {
                                  printf("&&&& PASSED FileCheck $test\n");
                                  $passes = $passes + 1;
                                } else {
                                  # Use a temporary file to send the output to
                                  # FileCheck so we can get the output this time,
                                  # because Perl and bidirectional pipes suck.
                                  my $tmp = File::Temp->new();
                                  my $tmp_filename = $tmp->filename;
                                  print $tmp @output;

                                  printf("********************************************************************************\n");
                                  print `$filecheck_cmd -input-file $tmp_filename`;
                                  printf("********************************************************************************\n");

                                  process_return_code("FileCheck $test", $filecheck_ret, "");
                                  printf("&&&& FAILED FileCheck $test\n");
                                  $failures = $failures + 1;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            $errors = $errors + 1;
            process_return_code($test, $ret, "Test crash?");
            printf("&&&& FAILED $test\n");
            printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
        }
        printf("\n");
    }
}

sub dvs_summary {
    my $dvs_score = 0;
    my $denominator = $failures + $known_failures + $errors + $passes;
    if ($denominator == 0) {
       $dvs_score = 0;
    }
    else {
       $dvs_score = 100 * (($passes + $known_failures) / $denominator);
    }

    printf("\n");

    printf("%*%*%*%* FA!LUR3S       $failures\n");
    printf("%*%*%*%* KN0WN FA!LUR3S $known_failures\n");
    printf("%*%*%*%* 3RR0RS         $errors\n");
    printf("%*%*%*%* PASS3S         $passes\n");

    printf("\n");

    # We can't remove "sanity" here yet because DVS looks for this exact string.
    printf("CUDA DVS BASIC SANITY SCORE : %.1f\n", $dvs_score);

    if ($failures + $errors > 0) {
        exit(1);
    }
}

###############################################################################

printf("#### CONFIG arch `%s`\n", $arch);
printf("#### CONFIG abi `%s`\n", $abi);
printf("#### CONFIG os `%s`\n", $os);
printf("#### CONFIG build `%s`\n", $build);
printf("#### CONFIG bin_path `%s`\n", $bin_path);
printf("#### CONFIG have_filecheck `$have_filecheck`\n");
printf("#### CONFIG filecheck_path `%s`\n", $filecheck_path);
printf("#### CONFIG filecheck_data_path `%s`\n", $filecheck_data_path);
printf("#### CONFIG have_time_hi_res `$have_time_hi_res`\n");
printf("#### CONFIG timeout_min `%s`\n", $timeout_min);
printf("#### ENV PATH `%s`\n", defined $ENV{'PATH'} ? $ENV{'PATH'} : '');
printf("#### ENV LD_LIBRARY_PATH `%s`\n", defined $ENV{'LD_LIBRARY_PATH'} ? $ENV{'LD_LIBRARY_PATH'} : '');

printf("\n");

filecheck_smoke_test();

printf("\n");

my $START_TIME = current_time();

run_examples();
run_unit_tests();

my $STOP_TIME = current_time();

printf("#### START_TIME $START_TIME\n");
printf("#### STOP_TIME $STOP_TIME\n");

dvs_summary();

