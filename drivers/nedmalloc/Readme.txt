nedalloc v1.05 15th June 2008:
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

by Niall Douglas (http://www.nedprod.com/programs/portable/nedmalloc/)

Enclosed is nedalloc, an alternative malloc implementation for multiple
threads without lock contention based on dlmalloc v2.8.4. It is more
or less a newer implementation of ptmalloc2, the standard allocator in
Linux (which is based on dlmalloc v2.7.0) but also contains a per-thread
cache for maximum CPU scalability.

It is licensed under the Boost Software License which basically means
you can do anything you like with it. This does not apply to the malloc.c.h
file which remains copyright to others.

It has been tested on win32 (x86), win64 (x64), Linux (x64), FreeBSD (x64)
and Apple MacOS X (x86). It works very well on all of these and is very
significantly faster than the system allocator on all of these platforms.

By literally dropping in this allocator as a replacement for your system
allocator, you can see real world improvements of up to three times in normal
code!

To use:
-=-=-=-
Drop in nedmalloc.h, nedmalloc.c and malloc.c.h into your project.
Configure using the instructions in nedmalloc.h. Run and enjoy.

To test, compile test.c. It will run a comparison between your system
allocator and nedalloc and tell you how much faster nedalloc is. It also
serves as an example of usage.

Notes:
-=-=-=
If you want the very latest version of this allocator, get it from the
TnFOX SVN repository at svn://svn.berlios.de/viewcvs/tnfox/trunk/src/nedmalloc

Because of how nedalloc allocates an mspace per thread, it can cause
severe bloating of memory usage under certain allocation patterns.
You can substantially reduce this wastage by setting MAXTHREADSINPOOL
or the threads parameter to nedcreatepool() to a fraction of the number of
threads which would normally be in a pool at once. This will reduce
bloating at the cost of an increase in lock contention. If allocated size
is less than THREADCACHEMAX, locking is avoided 90-99% of the time and
if most of your allocations are below this value, you can safely set
MAXTHREADSINPOOL to one.

You will suffer memory leakage unless you call neddisablethreadcache()
per pool for every thread which exits. This is because nedalloc cannot
portably know when a thread exits and thus when its thread cache can
be returned for use by other code. Don't forget pool zero, the system pool.

For C++ type allocation patterns (where the same sizes of memory are
regularly allocated and deallocated as objects are created and destroyed),
the threadcache always benefits performance. If however your allocation
patterns are different, searching the threadcache may significantly slow
down your code - as a rule of thumb, if cache utilisation is below 80%
(see the source for neddisablethreadcache() for how to enable debug
printing in release mode) then you should disable the thread cache for
that thread. You can compile out the threadcache code by setting
THREADCACHEMAX to zero.

Speed comparisons:
-=-=-=-=-=-=-=-=-=
See Benchmarks.xls for details.

The enclosed test.c can do two things: it can be a torture test or a speed
test. The speed test is designed to be a representative synthetic
memory allocator test. It works by randomly mixing allocations with frees
with half of the allocation sizes being a two power multiple less than
512 bytes (to mimic C++ stack instantiated objects) and the other half
being a simple random value less than 16Kb. 

The real world code results are from Tn's TestIO benchmark. This is a
heavily multithreaded and memory intensive benchmark with a lot of branching
and other stuff modern processors don't like so much. As you'll note, the
test doesn't show the benefits of the threadcache mostly due to the saturation
of the memory bus being the limiting factor.

ChangeLog:
-=-=-=-=-=
v1.05 15th June 2008:
 * { 1042 } Added error check for TLSSET() and TLSFREE() macros. Thanks to
Markus Elfring for reporting this.
 * { 1043 } Fixed a segfault when freeing memory allocated using
nedindependent_comalloc(). Thanks to Pavel Vozenilek for reporting this.

v1.04 14th July 2007:
 * Fixed a bug with the new optimised implementation that failed to lock
on a realloc under certain conditions.
 * Fixed lack of thread synchronisation in InitPool() causing pool corruption
 * Fixed a memory leak of thread cache contents on disabling. Thanks to Earl
Chew for reporting this.
 * Added a sanity check for freed blocks being valid.
 * Reworked test.c into being a torture test.
 * Fixed GCC assembler optimisation misspecification

v1.04alpha_svn915 7th October 2006:
 * Fixed failure to unlock thread cache list if allocating a new list failed.
Thanks to Dmitry Chichkov for reporting this. Futher thanks to Aleksey Sanin.
 * Fixed realloc(0, <size>) segfaulting. Thanks to Dmitry Chichkov for
reporting this.
 * Made config defines #ifndef so they can be overriden by the build system.
Thanks to Aleksey Sanin for suggesting this.
 * Fixed deadlock in nedprealloc() due to unnecessary locking of preferred
thread mspace when mspace_realloc() always uses the original block's mspace
anyway. Thanks to Aleksey Sanin for reporting this.
 * Made some speed improvements by hacking mspace_malloc() to no longer lock
its mspace, thus allowing the recursive mutex implementation to be removed
with an associated speed increase. Thanks to Aleksey Sanin for suggesting this.
 * Fixed a bug where allocating mspaces overran its max limit. Thanks to
Aleksey Sanin for reporting this.

v1.03 10th July 2006:
 * Fixed memory corruption bug in threadcache code which only appeared with >4
threads and in heavy use of the threadcache.

v1.02 15th May 2006:
 * Integrated dlmalloc v2.8.4, fixing the win32 memory release problem and
improving performance still further. Speed is now up to twice the speed of v1.01
(average is 67% faster).
 * Fixed win32 critical section implementation. Thanks to Pavel Kuznetsov
for reporting this.
 * Wasn't locking mspace if all mspaces were locked. Thanks to Pavel Kuznetsov
for reporting this.
 * Added Apple Mac OS X support.

v1.01 24th February 2006:
 * Fixed multiprocessor scaling problems by removing sources of cache sloshing
 * Earl Chew <earl_chew <at> agilent <dot> com> sent patches for the following:
   1. size2binidx() wasn't working for default code path (non x86)
   2. Fixed failure to release mspace lock under certain circumstances which
      caused a deadlock

v1.00 1st January 2006:
 * First release
