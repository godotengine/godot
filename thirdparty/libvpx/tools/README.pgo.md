# Using Profile Guided Optimizations to identify compiler optimization failures

When using Clang, the `-Rpass-missed` flag enables the verbose log of failed
compiler optimizations. However, the extensive log messages can obscure
potential optimization opportunities.

Use the following steps to generate a more transparent optimization report
using a previously created PGO profile file. The report also includes code
hotness diagnostics:

```bash
$ ../libvpx/configure --use-profile=perf.profdata \
  --extra-cflags="-fsave-optimization-record -fdiagnostics-show-hotness"
```

Convert the generated YAML files into a detailed HTML report using the
[optviewer2](https://github.com/OfekShilon/optview2) tool:

```bash
$ opt-viewer.py --output-dir=out/ --source-dir=libvpx .
```

The HTML report displays each code line's relative hotness, cross-referenced
with the failed compiler optimizations.
