use rand::{distributions::Uniform, Rng};

use crate::blur::Blur;

/// A 2D grid with a scalar value per each grid block.
#[derive(Debug)]
pub struct Grid {
    width: usize,
    height: usize,
    data: Vec<f32>,

    // Scratch space for the blur operation.
    buf: Vec<f32>,
    blur: Blur,
}

impl Grid {
    /// Create a new grid filled with random floats in the [0.0..1.0) range.
    pub fn new(width: usize, height: usize) -> Self {
        if !width.is_power_of_two() || !height.is_power_of_two() {
            panic!("Grid dimensions must be a power of two.");
        }
        let rng = rand::thread_rng();
        let range = Uniform::from(0.0..1.0);
        let data = rng.sample_iter(range).take(width * height).collect();

        Grid {
            width,
            height,
            data,
            buf: vec![0.0; width * height],
            blur: Blur::new(width, height),
        }
    }

    /// Truncate x and y and return a corresponding index into the data slice.
    fn index(&self, x: f32, y: f32) -> usize {
        // x/y can come in negative, hence we shift them by width/height.
        let i = (x + self.width as f32) as usize & (self.width - 1);
        let j = (y + self.height as f32) as usize & (self.height - 1);
        j * self.width + i
    }

    /// Get the data value at a given position. The implementation effectively treats data as
    /// periodic, hence any finite position will produce a value.
    pub fn get(&self, x: f32, y: f32) -> f32 {
        self.data[self.index(x, y)]
    }

    /// Get the buffer value at a given position. The implementation effectively treats data as
    /// periodic, hence any finite position will produce a value.
    pub fn get_buf(&self, x: f32, y: f32) -> f32 {
        self.buf[self.index(x, y)]
    }

    /// Add a value to the grid data at a given position.
    pub fn add(&mut self, x: f32, y: f32, value: f32) {
        let idx = self.index(x, y);
        self.data[idx] += value
    }

    /// Diffuse grid data and apply a decay multiplier.
    pub fn diffuse(&mut self, radius: usize, decay_factor: f32) {
        self.blur
            .run(&mut self.data, &mut self.buf, radius as f32, decay_factor);
    }

    pub fn quantile(&self, fraction: f32) -> f32 {
        let index = if fraction == 1.0 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_grid_new_panics() {
        let _ = Grid::new(5, 5);
    }

    #[test]
    fn test_grid_new() {
        let grid = Grid::new(8, 8);
        assert_eq!(grid.index(0.5, 0.6), 0);
        assert_eq!(grid.index(1.5, 0.6), 1);
        assert_eq!(grid.index(0.5, 1.6), 8);
        assert_eq!(grid.index(2.5, 0.6), 2);
        assert_eq!(grid.index(2.5, 1.6), 10);
        assert_eq!(grid.index(7.9, 7.9), 63);
        assert_eq!(grid.index(-0.5, -0.6), 0);
    }
}
