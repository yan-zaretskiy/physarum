use physarum::trig;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_cos_appr(c: &mut Criterion) {
    c.bench_function("Approximate cosine", |b| {
        b.iter(|| trig::cos(black_box(1.0)))
    });
}

fn bench_cos_exact(c: &mut Criterion) {
    c.bench_function("Exact cosine", |b| b.iter(|| f32::cos(black_box(1.0))));
}

criterion_group!(benches, bench_cos_appr, bench_cos_exact);
criterion_main!(benches);
