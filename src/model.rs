use crate::grid::Grid;

use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;

/// A single Physarum agent. The x and y positions are continuous, hence we use floating point
/// numbers instead of integers.
#[derive(Debug)]
struct Agent {
    x: f32,
    y: f32,
    angle: f32,
}

impl Agent {
    /// Construct a new agent with random parameters.
    fn new<R: Rng + ?Sized>(width: usize, height: usize, rng: &mut R) -> Self {
        let (x, y, angle) = rng.gen::<(f32, f32, f32)>();
        Agent {
            x: x * width as f32,
            y: y * height as f32,
            angle: angle * std::f32::consts::TAU,
        }
    }

    /// Update agent's orientation angle and position on the grid.
    fn rotate_and_move(
        &mut self,
        direction: f32,
        rotation_angle: f32,
        step_distance: f32,
        width: usize,
        height: usize,
    ) {
        use crate::util::wrap;
        let delta_angle = rotation_angle * direction;
        self.angle = wrap(self.angle + delta_angle, std::f32::consts::TAU);
        self.x = wrap(self.x + step_distance * self.angle.cos(), width as f32);
        self.y = wrap(self.y + step_distance * self.angle.sin(), height as f32);
    }
}

/// A model configuration. We make it into a separate type, because we will eventually have multiple
/// configurations in one model.
#[derive(Debug)]
struct PopulationConfig {
    sensor_distance: f32,
    step_distance: f32,
    decay_factor: f32,
    sensor_angle: f32,
    rotation_angle: f32,
    deposition_amount: f32,
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

    /// Construct a random configuration.
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

/// Top-level simulation class.
#[derive(Debug)]
pub struct Model {
    // Physarum agents.
    agents: Vec<Agent>,

    // The grid they move on.
    grid: Grid,

    // Simulation parameters.
    diffusivity: usize,
    config: PopulationConfig,

    iteration: i32,
    width: usize,
    height: usize,
}

impl Model {
    /// Construct a new model with random initial conditions and random configuration.
    pub fn new(width: usize, height: usize, n_particles: usize, diffusivity: usize) -> Self {
        let mut rng = rand::thread_rng();

        Model {
            agents: (0..n_particles)
                .map(|_| Agent::new(width, height, &mut rng))
                .collect(),
            grid: Grid::new(width, height),
            diffusivity,
            config: PopulationConfig::new(&mut rng),
            iteration: 0,
            width,
            height,
        }
    }

    fn pick_direction<R: Rng + ?Sized>(center: f32, left: f32, right: f32, rng: &mut R) -> f32 {
        if (center > left) && (center > right) {
            0.0
        } else if (center < left) && (center < right) {
            *[-1.0, 1.0].choose(rng).unwrap()
        } else if left < right {
            1.0
        } else if right < left {
            -1.0
        } else {
            0.0
        }
    }

    /// Perform a single simulation step.
    pub fn step(&mut self) {
        // To avoid borrow-checker errors inside the parallel loop.
        let PopulationConfig {
            sensor_distance,
            sensor_angle,
            rotation_angle,
            step_distance,
            ..
        } = self.config;
        let (width, height) = (self.width, self.height);
        let grid = &self.grid;

        self.agents.par_iter_mut().for_each(|agent| {
            let mut rng = rand::thread_rng();
            let xc = agent.x + agent.angle.cos() * sensor_distance;
            let yc = agent.y + agent.angle.sin() * sensor_distance;
            let xl = agent.x + (agent.angle - sensor_angle).cos() * sensor_distance;
            let yl = agent.y + (agent.angle - sensor_angle).sin() * sensor_distance;
            let xr = agent.x + (agent.angle + sensor_angle).cos() * sensor_distance;
            let yr = agent.y + (agent.angle + sensor_angle).sin() * sensor_distance;

            // Sense
            let trail_c = grid.get(xc, yc);
            let trail_l = grid.get(xl, yl);
            let trail_r = grid.get(xr, yr);

            // Rotate and move
            let direction = Model::pick_direction(trail_c, trail_l, trail_r, &mut rng);
            agent.rotate_and_move(direction, rotation_angle, step_distance, width, height);
        });

        // Deposit
        for agent in self.agents.iter() {
            self.grid
                .add(agent.x, agent.y, self.config.deposition_amount);
        }
        // Diffuse + Decay
        self.grid
            .diffuse(self.diffusivity, self.config.decay_factor);
        self.iteration += 1;
    }

    /// Output the current trail layer as a grayscale image.
    pub fn save_to_image(&self) {
        let mut img = image::GrayImage::new(self.width as u32, self.height as u32);
        let max_value = self.grid.quantile(0.999);

        for (i, value) in self.grid.data().iter().enumerate() {
            let x = (i % self.width) as u32;
            let y = (i / self.width) as u32;
            let c = (value / max_value).clamp(0.0, 1.0) * 255.0;
            img.put_pixel(x, y, image::Luma([c as u8]));
        }
        img.save("out.png").unwrap();
    }
}
