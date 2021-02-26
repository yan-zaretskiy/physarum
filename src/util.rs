#[inline(always)]
pub fn is_power_of_two(x: usize) -> bool {
    (x & (x - 1)) == 0
}

#[inline(always)]
pub fn wrap(x: f32, max: f32) -> f32 {
    x - max * ((x > max) as i32 as f32 - (x < 0.0_f32) as i32 as f32)
}
