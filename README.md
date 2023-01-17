# smallbitset
[![Crates.io](https://img.shields.io/crates/v/smallbitset)](https://crates.io/crates/smallbitset)
[![Documentation](https://img.shields.io/badge/Docs.rs-Latest-informational)](https://docs.rs/smallbitset/)
[![Build](https://github.com/xgillard/smallbitset/workflows/Build/badge.svg)](https://github.com/xgillard/smallbitset/actions?query=workflow%3A%22Build%22)
[![Tests](https://github.com/xgillard/smallbitset/workflows/Tests/badge.svg)](https://github.com/xgillard/smallbitset/actions?query=workflow%3A%22Tests%22)
[![codecov](https://codecov.io/gh/xgillard/smallbitset/branch/main/graph/badge.svg)](https://codecov.io/gh/xgillard/smallbitset)
[![Quality](https://github.com/xgillard/smallbitset/workflows/Quality%20Assurance/badge.svg)](https://github.com/xgillard/smallbitset/actions?query=workflow%3A%22Quality+Assurance%22)
![GitHub](https://img.shields.io/github/license/xgillard/smallbitset)

This crate provides a series of allocation free integers set capable of holding 
small integer values.

## Usage
In your `Cargo.toml`, you should add the following line to your dependencies
section.

```toml
[dependencies]
smallbitset = "0.6.0"
```

Then in your main code, you will simply use one of the available collections 
as shown below:

```rust
use smallbitset::Set32;

fn main() {
	let mut x = Set32::empty();

	x = x.insert(1);
	assert_eq!(Set32::singleton(1), x);
	assert!(x.contains(1));
	
	// and so on ... check the online documentation for the complete api details
	
}
```



