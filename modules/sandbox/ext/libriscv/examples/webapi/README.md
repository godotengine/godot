## WebAPI

Compile and run RISC-V on a website

It's not very secure because it's not being run in a docker container anymore. I am not an expert on containers, even though I have been using this service inside a container before. Regardless, it should not be too hard to execute into a container instead of directly compiling files. Have a look at `sanitize.py` for the compiler arguments.

The API is created in a way where the Varnish cache will take most of the load by caching executables and even the results (for a shorter time). See the `varnish/cache.vcl` for more information on how this is configured. As a result the hosted page is snappy and reliable.


### Usage

1. Install Varnish
	- sudo apt install varnish
2. Start Varnish (tab 1)
	- cd varnish
	- varnishd -a :8080 -f $PWD/cache.vcl -F
	- If you need a custom working folder just add -n /tmp/varnishd
3. Build webapi (tab 2)
	- Check the [build.sh script](build.sh) for an example
4. Start the WebAPI server
	- ./build/webapi
5. Go to http://localhost:8080
	- Type some code and press Compile & Run
	- The program will be run in the background

### Benchmarking

While the benchmarking feature is kinda useless right now, it is possible to make it run the delineated code many times with some minor effort. At any rate, benchmarks are delinated using two EBREAK instructions.

Perhaps a better way would have been a system call that took a function pointer as argument, and then run that function X number of times to measure performance. But, I wanted something simple and here we are.
