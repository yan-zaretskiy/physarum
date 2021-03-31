use crate::blur::Blur;

use rand::{distributions::Uniform, Rng};

use std::fmt::{Display, Formatter};

// A population configuration.
#[derive(Debug)]
pub struct PopulationConfig {
    pub sensor_distance: f32,
    pub step_distance: f32,
    pub sensor_angle: f32,
    pub rotation_angle: f32,

    decay_factor: f32,
    deposition_amount: f32,
}

impl Clone for PopulationConfig {
    fn clone(&self) -> PopulationConfig {
        return PopulationConfig {
            sensor_distance: self.sensor_distance,
            step_distance: self.step_distance,
            sensor_angle: self.sensor_angle,
            rotation_angle: self.rotation_angle,
            decay_factor: self.decay_factor,
            deposition_amount: self.deposition_amount,
        };
    }
}

impl Display for PopulationConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n  Sensor Distance: {},\n  Step Distance: {},\n  Sensor Angle: {},\n  Rotation Angle: {},\n  Decay Factor: {},\n  Deposition Amount: {},\n}}",
            self.sensor_distance,
            self.step_distance,
            self.sensor_angle,
            self.rotation_angle,
            self.decay_factor,
            self.deposition_amount
        )
    }
}

impl PopulationConfig {
    const SENSOR_ANGLE_MIN: f32 = 0.0;
    const SENSOR_ANGLE_MAX: f32 = 120.0;
    const SENSOR_DISTANCE_MIN: f32 = 0.0;
    const SENSOR_DISTANCE_MAX: f32 = 64.0;
    const ROTATION_ANGLE_MIN: f32 = 0.0;
    const ROTATION_ANGLE_MAX: f32 = 120.0;
    const STEP_DISTANCE_MIN: f32 = 0.2;
    const STEP_DISTANCE_MAX: f32 = 2.0;
    const DEPOSITION_AMOUNT_MIN: f32 = 5.0;
    const DEPOSITION_AMOUNT_MAX: f32 = 5.0;
    const DECAY_FACTOR_MIN: f32 = 0.1;
    const DECAY_FACTOR_MAX: f32 = 0.1;

    // Construct a random configuration.
    pub fn new<R: Rng + ?Sized>(rng: &mut R) -> Self {
        PopulationConfig {
            sensor_distance: rng.gen_range(Self::SENSOR_DISTANCE_MIN..=Self::SENSOR_DISTANCE_MAX),
            step_distance: rng.gen_range(Self::STEP_DISTANCE_MIN..=Self::STEP_DISTANCE_MAX),
            decay_factor: rng.gen_range(Self::DECAY_FACTOR_MIN..=Self::DECAY_FACTOR_MAX),
            sensor_angle: rng
                .gen_range(Self::SENSOR_ANGLE_MIN..=Self::SENSOR_ANGLE_MAX)
                .to_radians(),
            rotation_angle: rng
                .gen_range(Self::ROTATION_ANGLE_MIN..=Self::ROTATION_ANGLE_MAX)
                .to_radians(),
            deposition_amount: rng
                .gen_range(Self::DEPOSITION_AMOUNT_MIN..=Self::DEPOSITION_AMOUNT_MAX),
        }
    }
}

// A 2D grid with a scalar value per each grid block. Each grid is occupied by a single population, hence we store the population config inside the grid.
#[derive(Debug)]
pub struct Grid {
    pub config: PopulationConfig,
    pub width: usize,
    pub height: usize,

    data: Vec<f32>,

    // Scratch space for the blur operation.
    buf: Vec<f32>,
    blur: Blur,
}

impl Clone for Grid {
    fn clone(&self) -> Grid {
        return Grid {
            config: self.config.clone(),
            width: self.width.clone(),
            height: self.height.clone(),
            data: self.data.clone(),
            buf: self.buf.clone(),
            blur: self.blur.clone(),
        };
    }
}

impl Grid {
    // Create a new grid filled with random floats in the [0.0..1.0) range.
    pub fn new<R: Rng + ?Sized>(width: usize, height: usize, rng: &mut R) -> Self {
        if !width.is_power_of_two() || !height.is_power_of_two() {
            panic!("Grid dimensions must be a power of two.");
        }
        let range = Uniform::from(0.0..1.0);
        let data = rng.sample_iter(range).take(width * height).collect();

        Grid {
            width,
            height,
            data,
            config: PopulationConfig::new(rng),
            buf: vec![0.0; width * height],
            blur: Blur::new(width),
        }
    }

    // Truncate x and y and return a corresponding index into the data slice.
    fn index(&self, x: f32, y: f32) -> usize {
        // x/y can come in negative, hence we shift them by width/height.
        let i = (x + self.width as f32) as usize & (self.width - 1);
        let j = (y + self.height as f32) as usize & (self.height - 1);
        j * self.width + i
    }

    // Get the buffer value at a given position. The implementation effectively treats data as periodic, hence any finite position will produce a value.
    pub fn get_buf(&self, x: f32, y: f32) -> f32 {
        self.buf[self.index(x, y)]
    }

    // Add a value to the grid data at a given position.
    pub fn deposit(&mut self, x: f32, y: f32) {
        let idx = self.index(x, y);
        self.data[idx] += self.config.deposition_amount;
    }

    // Diffuse grid data and apply a decay multiplier.
    pub fn diffuse(&mut self, radius: usize) {
        self.blur.run(
            &mut self.data,
            &mut self.buf,
            self.width,
            self.height,
            radius as f32,
            self.config.decay_factor,
        );
    }

    pub fn quantile(&self, fraction: f32) -> f32 {
        let index = if (fraction - 1.0_f32).abs() < f32::EPSILON {
            self.data.len() - 1
        } else {
            (self.data.len() as f32 * fraction) as usize
        };
        let mut sorted = self.data.clone();
        sorted
            .as_mut_slice()
            .select_nth_unstable_by(index, |a, b| a.partial_cmp(b).unwrap());
        sorted[index]
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

pub fn combine<T>(grids: &mut [Grid], attraction_table: &[T])
where
    T: AsRef<[f32]> + Sync,
{
    let datas: Vec<_> = grids.iter().map(|grid| &grid.data).collect();
    let bufs: Vec<_> = grids.iter().map(|grid| &grid.buf).collect();

    // We mutate grid buffers and read grid data. We use unsafe because we need shared/unique borrows on different fields of the same Grid struct.
    bufs.iter().enumerate().for_each(|(i, buf)| unsafe {
        let buf_ptr = *buf as *const Vec<f32> as *mut Vec<f32>;
        buf_ptr.as_mut().unwrap().fill(0.0);
        datas.iter().enumerate().for_each(|(j, other)| {
            let multiplier = attraction_table[i].as_ref()[j];
            buf_ptr
                .as_mut()
                .unwrap()
                .iter_mut()
                .zip(*other)
                .for_each(|(to, from)| *to += from * multiplier)
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_grid_new_panics() {
        let mut rng = rand::thread_rng();
        let _ = Grid::new(5, 5, &mut rng);
    }

    #[test]
    fn test_grid_new() {
        let mut rng = rand::thread_rng();
        let grid = Grid::new(8, 8, &mut rng);
        assert_eq!(grid.index(0.5, 0.6), 0);
        assert_eq!(grid.index(1.5, 0.6), 1);
        assert_eq!(grid.index(0.5, 1.6), 8);
        assert_eq!(grid.index(2.5, 0.6), 2);
        assert_eq!(grid.index(2.5, 1.6), 10);
        assert_eq!(grid.index(7.9, 7.9), 63);
        assert_eq!(grid.index(-0.5, -0.6), 0);
    }
}
