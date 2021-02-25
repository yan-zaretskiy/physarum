use crate::grid::Grid;

/// A single Physarum agent. The x and y positions are continuous, hence we use floating point
/// numbers instead of integers.
#[derive(Debug)]
pub struct Agent {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
}

/// A model configuration. We make it into a separate type, because we will eventually have multiple
/// configurations in one model.
#[derive(Debug)]
pub struct PopulationConfig {
    pub sensor_angle: f32,
    pub sensor_distance: f32,
    pub rotation_angle: f32,
    pub step_distance: f32,
    pub decay_factor: f32,
}

impl PopulationConfig {}

/// Top-level simulation class.
#[derive(Debug)]
pub struct Model {
    grid: Grid,
    agents: Vec<Agent>,

    config: PopulationConfig,
}
