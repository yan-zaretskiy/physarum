use itertools::multizip;

#[derive(Debug)]
pub struct Blur {
    width: usize,
    height: usize,
    row_buffer: Vec<f32>,
}

impl Blur {
    pub fn new(width: usize, height: usize) -> Self {
        Blur {
            width,
            height,
            row_buffer: vec![0.0; width],
        }
    }

    /// Blur an image with 3 box filter passes. The result will be written to the src slice, while
    /// the buf slice is used as a scratch space.
    pub fn run(&mut self, src: &mut [f32], buf: &mut [f32], sigma: f32, decay: f32) {
        let boxes = Blur::boxes_for_gaussian::<3>(sigma);
        self.box_blur(src, buf, boxes[0], 1.0);
        self.box_blur(src, buf, boxes[1], 1.0);
        self.box_blur(src, buf, boxes[2], decay);
    }

    /// Approximate 1D Gaussian filter of standard deviation sigma with N box filter passes. Each
    /// element in the output array contains the radius of the box filter for the corresponding
    /// pass.
    fn boxes_for_gaussian<const N: usize>(sigma: f32) -> ([usize; N]) {
        let w_ideal = (12.0 * sigma * sigma / N as f32 + 1.0).sqrt();
        let mut w = w_ideal as usize;
        w -= 1 - w & 1;

        let mut m = 0.25 * (N * (w + 3)) as f32;
        m -= 3.0 * sigma * sigma / (w + 1) as f32;
        let m = m.round() as usize;

        let mut result = [0; N];
        for (i, value) in result.iter_mut().enumerate() {
            *value = (if i < m { w - 1 } else { w + 1 }) / 2;
        }
        result
    }

    /// Perform one pass of the 2D box filter of the given radius. The result will be written to the
    /// src slice, while the buf slice is used as a scratch space.
    fn box_blur(&mut self, src: &mut [f32], buf: &mut [f32], radius: usize, decay: f32) {
        self.box_blur_h(src, buf, radius);
        self.box_blur_v(buf, src, radius, decay);
    }

    /// Perform one pass of the 1D box filter of the given radius along x axis.
    fn box_blur_h(&mut self, src: &[f32], dst: &mut [f32], radius: usize) {
        let weight = 1.0 / (2 * radius + 1) as f32;
        let width = self.width;

        for (src_row, dst_row) in src.chunks_exact(width).zip(dst.chunks_exact_mut(width)) {
            // First we build a value for the beginning of each row. We assume periodic boundary
            // conditions, so we need to push the left index to the opposite side of the row.
            let mut value = src_row[width - radius - 1];
            for j in 0..radius {
                value += src_row[width - radius + j] + src_row[j];
            }

            for i in 0..width {
                let left = (i + width - radius - 1) & (width - 1);
                let right = (i + radius) & (width - 1);
                value += src_row[right] - src_row[left];
                dst_row[i] = value * weight;
            }
        }
    }

    /// Perform one pass of the 1D box filter of the given radius along y axis. Applies the decay
    /// factor to the destination buffer.
    fn box_blur_v(&mut self, src: &[f32], dst: &mut [f32], radius: usize, decay: f32) {
        let weight = decay / (2 * radius + 1) as f32;
        let (width, height) = (self.width, self.height);

        // We don't replicate the horizontal filter logic because of the cache-unfriendly memory
        // access patterns of sequential iteration over individual columns. Instead, we iterate over
        // rows via loop interchange.
        let offset = (height - radius - 1) * width;
        self.row_buffer
            .copy_from_slice(&src[offset..offset + width]);

        for j in 0..radius {
            let bottom_off = (height - radius + j) * width;
            let bottom_row = &src[bottom_off..bottom_off + width];
            let top_off = j * width;
            let top_row = &src[top_off..top_off + width];

            for (buf, bottom, top) in multizip((&mut self.row_buffer, bottom_row, top_row)) {
                *buf += bottom + top;
            }
        }

        for (i, dst_row) in dst.chunks_exact_mut(width).enumerate() {
            let bottom_off = ((i + height - radius - 1) & (height - 1)) * width;
            let bottom_row = &src[bottom_off..bottom_off + width];
            let top_off = ((i + radius) & (height - 1)) * width;
            let top_row = &src[top_off..top_off + width];

            for (dst, buf, bottom, top) in
                multizip((dst_row, &mut self.row_buffer, bottom_row, top_row))
            {
                *buf += top - bottom;
                *dst = *buf * weight;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blur() {
        let src: Vec<f32> = (1..17).map(|v| v as f32).collect();
        let mut dst = vec![0.0; 16];
        let mut blur = Blur::new(4, 4);

        blur.box_blur_h(&src, &mut dst, 1);
        assert_eq!(
            dst,
            [
                2.3333335, 2.0, 3.0, 2.6666667, 6.3333335, 6.0, 7.0, 6.666667, 10.333334, 10.0,
                11.0, 10.666667, 14.333334, 14.0, 15.0, 14.666667
            ]
        );

        blur.box_blur_v(&src, &mut dst, 1, 1.0);
        assert_eq!(
            dst,
            [
                6.3333335, 7.3333335, 8.333334, 9.333334, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                12.0, 7.666667, 8.666667, 9.666667, 10.666667
            ]
        )
    }
}
