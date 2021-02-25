/// Approximate 1D Gaussian filter of standard deviation sigma with N box filter passes. Each
/// element in the output array contains the radius of the box filter for the corresponding pass.
pub fn boxes_for_gaussian<const N: usize>(sigma: f32) -> ([usize; N]) {
    let w_ideal = (12.0 * sigma * sigma / N as f32 + 1.0).sqrt();
    let mut w = w_ideal as usize;
    w -= 1 - w & 1;

    let mut m = ((w * w + 4 * w + 3) * N) as f32;
    m -= 12.0 * sigma * sigma;
    m *= 0.25;
    m /= (w + 1) as f32;
    let m = m.round() as usize;

    let mut result = [0; N];
    for (i, value) in result.iter_mut().enumerate() {
        *value = (if i < m { w - 1 } else { w + 1 }) / 2;
    }
    result
}

/// Blur an image with 3 box filter passes. The result will be written to the src slice, while the
/// buf slice is used as a scratch space.
pub fn approximate_gauss_blur(
    src: &mut [f32],
    buf: &mut [f32],
    width: usize,
    height: usize,
    sigma: f32,
    decay: f32,
) {
    let boxes = boxes_for_gaussian::<3>(sigma);
    box_blur(src, buf, width, height, boxes[0], 1.0);
    box_blur(src, buf, width, height, boxes[1], 1.0);
    box_blur(src, buf, width, height, boxes[2], decay);
}

/// Perform one pass of the 2D box filter of the given radius. The result will be written to the src
/// slice, while the buf slice is used as a scratch space.
fn box_blur(
    src: &mut [f32],
    buf: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    decay: f32,
) {
    box_blur_h(src, buf, width, height, radius, 1.0);
    box_blur_v(buf, src, width, height, radius, decay);
}

/// Perform one pass of the 1D box filter of the given radius along x axis. Applies the decay factor
/// to the destination buffer.
fn box_blur_h(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    decay: f32,
) {
    let weight = decay / (2 * radius + 1) as f32;

    // TODO: Parallelize with rayon
    for i in 0..height {
        // First we build a value for the beginning of each row. We assume periodic boundary
        // conditions, so we need to push the left index to the opposite side of the row.
        let mut value = src[(i + 1) * width - radius - 1];
        for j in 0..radius {
            value += src[(i + 1) * width - radius + j] + src[i * width + j];
        }
        // At this point "value" contains the unweighted sum for the right-most row element.

        for current_id in i * width..(i + 1) * width {
            let left_id = ((current_id + width - radius - 1) & (width - 1)) + i * width;
            let right_id = ((current_id + radius) & (width - 1)) + i * width;
            value += src[right_id] - src[left_id];
            dst[current_id] = value * weight;
        }
    }
}

/// Perform one pass of the 1D box filter of the given radius along y axis. Applies the decay factor
/// to the destination buffer.
fn box_blur_v(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    decay: f32,
) {
    let weight = decay / (2 * radius + 1) as f32;

    // TODO: Parallelize with rayon
    for i in 0..width {
        // First we build a value for the beginning of each column. We assume periodic boundary
        // conditions, so we need to push the bottom index to the opposite side of the column.
        let mut value = src[i + (height - radius - 1) * width];
        for j in 0..radius {
            value += src[i + (height - radius + j) * width] + src[i + j * width];
        }
        // At this point "value" contains the unweighted sum for the top-most column element.

        for current_id in (i..i + height * width).step_by(width) {
            let bottom_id = (current_id + (height - radius - 1) * width) & (width * height - 1);
            let top_id = (current_id + radius * width) & (width * height - 1);
            value += src[top_id] - src[bottom_id];
            dst[current_id] = value * weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blur() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        box_blur_v(&src, &mut dst, 2, 4, 1, 1.0);
        println!("Out: {:?}", dst);
    }
}
