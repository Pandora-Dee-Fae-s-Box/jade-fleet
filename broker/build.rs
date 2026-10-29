// sqlite-vec is compiled into the binary (no ./vec0.so to ship or load) against the same bundled SQLite that
// rusqlite links, and registered for every connection at startup (see db.rs).
fn main() {
    let inc = std::env::var("DEP_SQLITE3_INCLUDE").expect("libsqlite3-sys (bundled) exposes its sqlite3.h");
    let mut b = cc::Build::new();
    b.file("vendor/sqlite-vec/sqlite-vec.c")
        .include("vendor/sqlite-vec")
        .include(&inc)
        .define("SQLITE_CORE", None)
        .define("SQLITE_VEC_STATIC", None)
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-variable")
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-sign-compare")
        .warnings(false)
        .opt_level(3);
    // SIMD distance functions: AVX on x86_64 (the OptiPlex 3020's Haswell has AVX2), NEON on aarch64 (Jetson)
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch == "x86_64" && std::env::var("JADE_FLEET_NO_AVX").is_err() {
        b.define("SQLITE_VEC_ENABLE_AVX", None).flag("-mavx").flag("-mavx2").flag("-mfma");
    } else if arch == "aarch64" {
        b.define("SQLITE_VEC_ENABLE_NEON", None);
    }
    b.compile("sqlite_vec");
    println!("cargo:rerun-if-changed=vendor/sqlite-vec/sqlite-vec.c");
    println!("cargo:rerun-if-env-changed=JADE_FLEET_NO_AVX");
}
