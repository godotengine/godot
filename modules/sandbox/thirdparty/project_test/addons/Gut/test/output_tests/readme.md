These are a bunch of tests that you have to look at the output for.  They are not included in a normal run.  To run these tests use the following command:

```
gut -gconfig= -gdir test/output_tests
```

```
gut -gconfig= -gdir test/output_tests -gexit -gignore_pause
```

```
gut -gconfig= -gdir test/output_tests -gexit -gignore_pause -gselect test_bugs
```