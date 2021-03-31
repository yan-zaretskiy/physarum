use itertools::multizip;
use rayon::prelude::*;

#[derive(Debug)]
pub struct Blur {
    row_buffer: Vec<f32>,
}

impl Clone for Blur {
    fn clone(&self) -> Blur {
        Blur {
            row_buffer: self.row_buffer.clone(),
        }
    }
}

impl Blur {
    pub fn new(width: usize) -> Self {
        Blur {
            row_buffer: vec![0.0; width],
        }
    }

    // Blur an image with 2 box filter passes. The result will be written to the src slice, while the buf slice is used as a scratch space.
    pub fn run(
        &mut self,
        src: &mut [f32],
        buf: &mut [f32],
        width: usize,
        height: usize,
        sigma: f32,
        decay: f32,
    ) {
        let boxes = Blur::boxes_for_gaussian::<2>(sigma);
        self.box_blur(src, buf, width, height, boxes[0], 1.0);
        self.box_blur(src, buf, width, height, boxes[1], decay);
    }

    // Approximate 1D Gaussian filter of standard deviation sigma with N box filter passes. Each element in the output array contains the radius of the box filter for the corresponding pass.
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

    // Perform one pass of the 2D box filter of the given radius. The result will be written to the src slice, while the buf slice is used as a scratch space.
    fn box_blur(
        &mut self,
        src: &mut [f32],
        buf: &mut [f32],
        width: usize,
        height: usize,
        radius: usize,
        decay: f32,
    ) {
        self.box_blur_h(src, buf, width, radius);
        self.box_blur_v(buf, src, width, height, radius, decay);
    }

    // Perform one pass of the 1D box filter of the given radius along x axis.
    fn box_blur_h(&mut self, src: &[f32], dst: &mut [f32], width: usize, radius: usize) {
        let weight = 1.0 / (2 * radius + 1) as f32;

        src.par_chunks_exact(width)
            .zip(dst.par_chunks_exact_mut(width))
            .for_each(|(src_row, dst_row)| {
                // First we build a value for the beginning of each row. We assume periodic boundary conditions, so we need to push the left index to the opposite side of the row.
                let width_sub_radius = width - radius;
                let mut value = src_row[width - radius - 1];
                for j in 0..radius {
                    value += src_row[width_sub_radius + j] + src_row[j];
                }

                for (i, dst_elem) in dst_row.iter_mut().enumerate() {
                    let left = (i + width_sub_radius - 1) & (width - 1);
                    let right = (i + radius) & (width - 1);
                    value += src_row[right] - src_row[left];
                    *dst_elem = value * weight;
                }
            })
    }

    // Perform one pass of the 1D box filter of the given radius along y axis. Applies the decay factor to the destination buffer.
    fn box_blur_v(
        &mut self,
        src: &[f32],
        dst: &mut [f32],
        width: usize,
        height: usize,
        radius: usize,
        decay: f32,
    ) {
        let weight = decay / (2 * radius + 1) as f32;

        // We don't replicate the horizontal filter logic because of the cache-unfriendly memory  access patterns of sequential iteration over individual columns. Instead, we iterate over rows via loop interchange.
        let height_sub_radius = height - radius;
        let offset = (height_sub_radius - 1) * width;
        self.row_buffer
            .copy_from_slice(&src[offset..offset + width]);

        for j in 0..radius {
            let bottom_off = (height_sub_radius + j) * width;
            let bottom_row = &src[bottom_off..bottom_off + width];
            let top_off = j * width;
            let top_row = &src[top_off..top_off + width];

            for (buf, bottom, top) in multizip((&mut self.row_buffer, bottom_row, top_row)) {
                *buf += bottom + top;
            }
        }

        // The outer loop cannot be parallelized because we need to use the buffer sequentially.
        for (i, dst_row) in dst.chunks_exact_mut(width).enumerate() {
            let bottom_off = ((i + height_sub_radius - 1) & (height - 1)) * width;
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
        // The values from this test were obtained using Python's code:
        // a = np.random.rand(8, 8)
        // ndimage.uniform_filter1d(a, size=3, axis = 1, mode='wrap') # horizontal blur
        // ndimage.uniform_filter1d(a, size=3, axis = 0, mode='wrap') # vertical blur
        // ndimage.uniform_filter(a, size=3, mode='wrap') # 2D blur

        let mut src: Vec<f32> = vec![
            0.32352856, 0.06571674, 0.01939427, 0.06352045, 0.708_527, 0.617_221_7, 0.16638431,
            0.628_400_74, 0.554_893_9, 0.240_076_77, 0.325_009_94, 0.08515139, 0.679_840_9, 0.6975669,
            0.736_234_25, 0.55053085, 0.692_227_66, 0.22727048, 0.13594262, 0.10002105, 0.16099514,
            0.07719103, 0.23984282, 0.9083058, 0.642_227_4, 0.968_934_2, 0.74662715, 0.715_620_1,
            0.736_546_5, 0.70610344, 0.221_011_18, 0.755_721_87, 0.691_958_84, 0.837_414, 0.27583158,
            0.572_570_5, 0.681_606, 0.392_373_38, 0.33524343, 0.893_968_34, 0.602_969_35, 0.171_301_13,
            0.1733834, 0.771_278_2, 0.99537134, 0.915_049_6, 0.493_121_1, 0.430_352_03, 0.70297265,
            0.367_341_8, 0.4551964, 0.471_043_14, 0.603_747_8, 0.738_726_85, 0.5630592, 0.974_402_25,
            0.633_682_85, 0.841_092_94, 0.24447136, 0.750384, 0.16893725, 0.542_256_65, 0.435_607_82,
            0.414_971_23,
        ];
        let (width, height) = (8, 8);
        let mut dst = vec![0.0; width * height];
        let mut blur = Blur::new(width);

        blur.box_blur_h(&src, &mut dst, width, 1);
        let mut sol: Vec<f32> = vec![
            0.339_215_37, 0.136_213_18, 0.04954382, 0.263_813_9, 0.46308973, 0.497_377_7, 0.470_668_94,
            0.372_771_2, 0.448_500_5, 0.373_326_87, 0.21674603, 0.363_334_1, 0.48751974, 0.70454735,
            0.661_444, 0.613_886_36, 0.609_268, 0.351_813_58, 0.15441138, 0.1323196, 0.11273574,
            0.159343, 0.40844655, 0.613_458_75, 0.788_961_2, 0.785_929_56, 0.810_393_8, 0.732_931_26,
            0.719_423_35, 0.554_553_7, 0.560_945_5, 0.539_653_5, 0.807_780_4, 0.601_734_8, 0.561_938_7,
            0.510_002_7, 0.548_849_94, 0.46974093, 0.540_528_4, 0.640_390_2, 0.40154082, 0.315_884_62,
            0.371_987_58, 0.646_677_6, 0.893_899_74, 0.801_180_66, 0.612_840_9, 0.508_814_16, 0.681_572_2,
            0.508_503_6, 0.431_193_77, 0.509_995_76, 0.604_505_9, 0.635_177_9, 0.758_729_4, 0.746_811_33,
            0.629_915_65, 0.573_082_4, 0.611_982_76, 0.38793087, 0.48719263, 0.38226724, 0.464_278_58,
            0.494_753_96,
        ];
        for (v1, v2) in dst.iter().zip(sol) {
            assert!((v1 - v2).abs() < 1e-6);
        }

        blur.box_blur_v(&src, &mut dst, width, height, 1, 1.0);
        sol = vec![
            0.504_035_1, 0.382_295_5, 0.19629186, 0.299_685_27, 0.519_101_74, 0.619_015_1, 0.446_075_47,
            0.531_300_96, 0.523_550_03, 0.177688, 0.16011561, 0.08289763, 0.516_454_34, 0.46399322,
            0.38082045, 0.695_745_8, 0.629_783_03, 0.47876048, 0.402_526_56, 0.300_264_18, 0.5257942,
            0.49362046, 0.3990294, 0.738_186_2, 0.675_471_3, 0.677_872_9, 0.386_133_8, 0.46273723,
            0.526_382_57, 0.391_889_3, 0.265_365_8, 0.852_665_36, 0.645_718_5, 0.659_216_46, 0.39861405,
            0.686_489_6, 0.804_508, 0.671_175_5, 0.349_791_88, 0.693_347_4, 0.665_966_9, 0.458_685_64,
            0.30147046, 0.604_963_96, 0.760_241_7, 0.682_05, 0.463_807_9, 0.766_240_9, 0.6465416,
            0.459_911_97, 0.291_017_06, 0.664_235_1, 0.589_352_13, 0.732_011, 0.497_262_72, 0.606_575_13,
            0.553_394_7, 0.42471716, 0.23968734, 0.428_315_88, 0.493_737_34, 0.632_735_1, 0.388_350_46,
            0.672_591_45,
        ];
        for (v1, v2) in dst.iter().zip(sol) {
            assert!((v1 - v2).abs() < 1e-6);
        }

        blur.box_blur(&mut src, &mut dst, width, height, 1, 1.0);
        sol = vec![
            0.472_543_84, 0.36087415, 0.29275754, 0.338_359_62, 0.47926736, 0.528_064_1, 0.5321305,
            0.493_803_83, 0.465_661_3, 0.287_117_9, 0.140_233_76, 0.253_155_86, 0.3544484, 0.453_756,
            0.513_519_8, 0.5333721, 0.615_576_57, 0.503_69, 0.393_850_42, 0.40952832, 0.43989295,
            0.472_814_68, 0.543_612, 0.588_999_5, 0.735_336_54, 0.579826, 0.508_914_65, 0.458_417_86,
            0.460_336_36, 0.39454588, 0.503_306_8, 0.597_834_17, 0.666_094_1, 0.567_849_7, 0.581_440_03,
            0.62987053, 0.720_724_34, 0.608_491_8, 0.571_438_25, 0.562_952_64, 0.630_297_84, 0.475_374_34,
            0.455_04, 0.5555587, 0.682_418_5, 0.635_366_5, 0.63736624, 0.632_005_2, 0.571_009_6,
            0.465_823_53, 0.471_721_38, 0.5148681, 0.661_866_07, 0.606_208_6, 0.611_949_6, 0.583_459_8,
            0.550_234_44, 0.405_933_05, 0.364_240_14, 0.38724685, 0.518_262_74, 0.504_940_9, 0.564_559,
            0.538_112_16,
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
