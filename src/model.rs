use crate::grid::Grid;

use rand::{thread_rng, Rng};

/// A single Physarum agent. The x and y positions are continuous, hence we use floating point
/// numbers instead of integers.
#[derive(Debug)]
struct Agent {
    x: f32,
    y: f32,
    angle: f32,
}

impl Agent {
    fn new(width: usize, height: usize) -> Self {
        let mut rng = rand::thread_rng();
        let (x, y, angle) = rng.gen::<(f32, f32, f32)>();
        Agent {
            x: x * width as f32,
            y: y * height as f32,
            angle: angle * std::f32::consts::TAU,
        }
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

    /// Generate a random configuration.
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();

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
}

impl Model {
    pub fn new(width: usize, height: usize, n_particles: usize, diffusivity: usize) -> Self {
        Model {
            agents: (0..n_particles)
                .map(|_| Agent::new(width, height))
                .collect(),
            grid: Grid::new(width, height),
            diffusivity,
            config: PopulationConfig::new(),
            iteration: 0,
        }
    }
}
