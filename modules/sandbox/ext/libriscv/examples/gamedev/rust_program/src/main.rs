mod sysalloc;
mod dyncalls;
use std::alloc;
use std::thread;
use dyncalls::*;

const NTHREADS: u32 = 10;

// This is the `main` thread
fn main() {
    // Make a vector to hold the children which are spawned.
    let mut children = vec![];

    for i in 0..NTHREADS {
        // Spin up another thread
        children.push(thread::spawn(move || {
            println!("this is thread number {}", i);
        }));
    }

    for child in children {
        // Wait for the thread to finish. Returns a result.
        let _ = child.join();
    }

	let i = unsafe { dyncall1(0x12345678) };
	println!("dyncall1: {}", i);
}


#[no_mangle]
extern "C" fn test1(a: i32, b: i32, c: i32, d: i32) -> i32 {
	println!("test1 was called with {}, {}, {}, {}", a, b, c, d);
	return a + b + c + d;
}

#[no_mangle]
extern "C" fn test2() {
	// Benchmark global allocator
	let layout = std::alloc::Layout::from_size_align(1024, 8).expect("Invalid layout");
    unsafe {
        let raw: *mut u8 = alloc::alloc(layout);
		std::hint::black_box(&raw);
		std::alloc::dealloc(raw, layout);
	}
}

#[no_mangle]
extern "C" fn test3() {
	// TODO: Exception handling
}

#[derive(Debug)]
#[repr(C)]
struct Data {
	a: i32,
	b: i32,
	c: i32,
	d: i32,
	e: f32,
	f: f32,
	g: f32,
	h: f32,
	i: f64,
	j: f64,
	k: f64,
	l: f64,
	buffer: [u8; 32],
}

#[no_mangle]
extern "C" fn test4(d: Data) {
	println!("test4 {:?}", d);
}

#[no_mangle]
extern "C" fn test5() {
	println!("test5");
}

#[no_mangle]
extern "C" fn measure_overhead() { }

#[no_mangle]
extern "C" fn bench_dyncall_overhead() {
	unsafe { dyncall3() };
}
