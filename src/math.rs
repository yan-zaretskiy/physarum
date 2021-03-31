// Previously from trig.rs
// From https://bits.stephan-brumme.com/absFloat.html
#[allow(dead_code)]
#[inline]
fn abs(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

// Previously from trig.rs
// Branchless floor implementation
#[allow(dead_code)]
#[inline]
fn floor(x: f32) -> f32 {
    let mut x_trunc = (x as i32) as f32;
    x_trunc -= (x < x_trunc) as i32 as f32;
    x_trunc
}

// Previously from trig.rs
// Approximates `cos(x)` in radians with the maximum error of `0.002`
// https://stackoverflow.com/posts/28050328/revisions
#[allow(dead_code)]
#[inline]
pub fn cos(mut x: f32) -> f32 {
    const ALPHA: f32 = 0.5 * std::f32::consts::FRAC_1_PI;
    x *= ALPHA;
    x -= 0.25_f32 + floor(x + 0.25_f32);
    x *= 16.0_f32 * (abs(x) - 0.5_f32);
    x += 0.225_f32 * x * (abs(x) - 1.0_f32);
    x
}

// Previously from trig.rs
// Approximates `sin(x)` in radians with the maximum error of `0.002`
#[allow(dead_code)]
#[inline]
pub fn sin(x: f32) -> f32 {
    cos(x - std::f32::consts::FRAC_PI_2)
}

