bin="$@"
sudo -E perf record --call-graph dwarf $bin
sudo perf report --call-graph
