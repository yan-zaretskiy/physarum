use itertools::multizip;
use rayon::prelude::*;

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
        w -= 1 - (w & 1);
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

        src.par_chunks_exact(width)
            .zip(dst.par_chunks_exact_mut(width))
            .for_each(|(src_row, dst_row)| {
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
            })
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

        // The outer loop cannot be parallelized because we need to use the buffer sequentially.
        for (i, dst_row) in dst.chunks_exact_mut(width).enumerate() {
            let bottom_off = ((i + height - radius - 1) & (height - 1)) * width;
            let bottom_row = &src[bottom_off..bottom_off + width];
            let top_off = ((i + radius) & (height - 1)) * width;
            let top_row = &src[top_off..top_off + width];

            (dst_row, &mut self.row_buffer, bottom_row, top_row)
                .into_par_iter()
                .for_each(|(dst, buf, bottom, top)| {
                    *buf += top - bottom;
                    *dst = *buf * weight;
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blur() {
        // The values from this test were obtained using Python's code:
        // a = np.random.rand(8, 8)
        // ndimage.uniform_filter1d(a, size=3, axis = 1, mode='wrap') # horizontal blur
        // ndimage.uniform_filter1d(a, size=3, axis = 0, mode='wrap') # vertical blur
        // ndimage.uniform_filter(a, size=3, mode='wrap') # 2D blur

        let mut src: Vec<f32> = vec![
            0.32352856, 0.06571674, 0.01939427, 0.06352045, 0.70852702, 0.61722172, 0.16638431,
            0.62840077, 0.55489392, 0.24007676, 0.32500995, 0.08515139, 0.67984092, 0.6975669,
            0.73623422, 0.55053085, 0.69222768, 0.22727048, 0.13594262, 0.10002105, 0.16099514,
            0.07719103, 0.23984282, 0.9083058, 0.64222744, 0.96893419, 0.74662715, 0.71562013,
            0.73654653, 0.70610344, 0.22101117, 0.75572186, 0.69195882, 0.83741404, 0.27583158,
            0.57257051, 0.68160597, 0.39237339, 0.33524343, 0.89396836, 0.60296932, 0.17130112,
            0.1733834, 0.77127819, 0.99537134, 0.91504964, 0.49312109, 0.43035202, 0.70297265,
            0.36734178, 0.4551964, 0.47104315, 0.60374777, 0.73872683, 0.5630592, 0.97440224,
            0.63368284, 0.84109297, 0.24447136, 0.750384, 0.16893725, 0.54225663, 0.43560783,
            0.41497124,
        ];
        let (width, height) = (8, 8);
        let mut dst = vec![0.0; width * height];
        let mut blur = Blur::new(width, height);

        blur.box_blur_h(&src, &mut dst, 1);
        let mut sol: Vec<f32> = vec![
            0.33921536, 0.13621319, 0.04954382, 0.26381392, 0.46308973, 0.49737768, 0.47066893,
            0.37277121, 0.44850051, 0.37332688, 0.21674603, 0.36333409, 0.48751974, 0.70454735,
            0.66144399, 0.61388633, 0.60926799, 0.35181359, 0.15441138, 0.1323196, 0.11273574,
            0.159343, 0.40844655, 0.61345877, 0.78896116, 0.78592959, 0.81039382, 0.73293127,
            0.71942337, 0.55455371, 0.56094549, 0.53965349, 0.80778041, 0.60173482, 0.56193871,
            0.51000269, 0.54884996, 0.46974093, 0.54052839, 0.64039021, 0.40154082, 0.31588461,
            0.37198757, 0.64667764, 0.89389972, 0.80118069, 0.61284092, 0.50881414, 0.68157222,
            0.50850361, 0.43119378, 0.50999577, 0.60450592, 0.63517794, 0.75872942, 0.74681136,
            0.62991568, 0.57308239, 0.61198277, 0.38793087, 0.48719263, 0.38226724, 0.46427857,
            0.49475397,
        ];
        for (v1, v2) in dst.iter().zip(sol) {
            assert!((v1 - v2).abs() < 1e-6);
        }

        blur.box_blur_v(&src, &mut dst, 1, 1.0);
        sol = vec![
            0.50403511, 0.38229549, 0.19629186, 0.29968528, 0.51910173, 0.61901508, 0.44607546,
            0.53130095, 0.52355005, 0.177688, 0.16011561, 0.08289763, 0.51645436, 0.46399322,
            0.38082045, 0.69574581, 0.62978301, 0.47876048, 0.40252657, 0.30026419, 0.5257942,
            0.49362046, 0.3990294, 0.73818617, 0.67547131, 0.67787291, 0.38613379, 0.46273723,
            0.52638255, 0.39188929, 0.26536581, 0.85266534, 0.64571853, 0.65921645, 0.39861405,
            0.68648961, 0.80450795, 0.67117549, 0.34979189, 0.69334741, 0.66596693, 0.45868565,
            0.30147046, 0.60496395, 0.76024169, 0.68204996, 0.46380791, 0.76624087, 0.6465416,
            0.45991196, 0.29101705, 0.66423511, 0.58935212, 0.73201103, 0.49726271, 0.60657516,
            0.55339468, 0.42471716, 0.23968734, 0.42831587, 0.49373735, 0.63273506, 0.38835045,
            0.67259142,
        ];
        for (v1, v2) in dst.iter().zip(sol) {
            assert!((v1 - v2).abs() < 1e-6);
        }

        blur.box_blur(&mut src, &mut dst, 1, 1.0);
        sol = vec![
            0.47254385, 0.36087415, 0.29275754, 0.33835963, 0.47926736, 0.52806409, 0.5321305,
            0.49380384, 0.46566129, 0.28711789, 0.14023375, 0.25315587, 0.3544484, 0.45375601,
            0.51351983, 0.5333721, 0.61557655, 0.50369002, 0.39385041, 0.40952832, 0.43989295,
            0.47281469, 0.54361201, 0.58899953, 0.73533652, 0.579826, 0.50891464, 0.45841785,
            0.46033635, 0.39454588, 0.50330681, 0.59783415, 0.66609413, 0.56784968, 0.58144004,
            0.62987053, 0.72072435, 0.60849178, 0.57143827, 0.56295261, 0.63029782, 0.47537435,
            0.45504002, 0.5555587, 0.68241853, 0.63536652, 0.63736624, 0.63200524, 0.57100958,
            0.46582354, 0.47172137, 0.5148681, 0.66186609, 0.60620862, 0.61194964, 0.58345982,
            0.55023442, 0.40593306, 0.36424013, 0.38724685, 0.51826276, 0.50494095, 0.56455897,
            0.53811218,
        ];
        for (v1, v2) in src.iter().zip(sol) {
            assert!((v1 - v2).abs() < 1e-6);
        }
    }

    #[test]
    fn test_boxes_for_gaussian() {
        let boxes = Blur::boxes_for_gaussian::<3>(1.5);
        assert_eq!(boxes, [1, 1, 1]);

        let boxes = Blur::boxes_for_gaussian::<3>(1.8);
        assert_eq!(boxes, [1, 1, 2]);

        let boxes = Blur::boxes_for_gaussian::<3>(2.5);
        assert_eq!(boxes, [2, 2, 2]);
    }
}
