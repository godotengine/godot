// https://play.rust-lang.org/?gist=4d6abc78a8c0d205da57a17c02201d7c&version=stable&mode=release&edition=2015

/// Returns the nth prime.
///
/// It uses a sieve internally, with a size of roughly
/// `n * (n.ln() + n.ln().ln()` bytes. As a result, its
/// runtime is also bound loglinear by the upper term.
///
pub fn nth_prime(n: u32) -> Option<u64> {
	if n < 1 {
		return None;
	}

	// The prime counting function is pi(x) which is approximately x/ln(x)
	// A good upper bound for the nth prime is ceil(x * ln(x * ln(x)))
	let x = if n <= 10 { 10.0 } else { n as f64 };
	let limit: usize = (x * (x * (x).ln()).ln()).ceil() as usize;
	let mut sieve = vec![true; limit];
	let mut count = 0;

	// Exceptional case for 0 and 1
	sieve[0] = false;
	sieve[1] = false;

	for prime in 2..limit {
		if !sieve[prime] {
			continue;
		}
		count += 1;
		if count == n {
			return Some(prime as u64);
		}

		for multiple in ((prime * prime)..limit).step_by(prime) {
			sieve[multiple] = false;
		}
	}
	None
}

#[no_mangle]
pub extern "C" fn __floatsitf(n: u32) -> f64 {
	n as f64
}
#[no_mangle]
pub	extern "C" fn __floatunsitf(n: u32) -> f64 {
	n as f64
}

fn main() {
	assert_eq!(nth_prime(0), None);
	assert_eq!(nth_prime(1), Some(2));
	assert_eq!(nth_prime(2), Some(3));
	assert_eq!(nth_prime(3), Some(5));

	println!("Sieve(256000) == {}\n", nth_prime(256000).unwrap());

	//let resp = reqwest::blocking::get("https://httpbin.org/ip").unwrap().text().unwrap();
	//println!("{:#?}", resp);
}
