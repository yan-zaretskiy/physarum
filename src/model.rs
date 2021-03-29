use crate::{
    grid::{combine, Grid, PopulationConfig},
    palette::{random_palette, Palette},
    imgdata::ImgData,
};

use rand::{seq::SliceRandom, Rng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use itertools::multizip;
use std::f32::consts::TAU;
use std::time::{Instant};
use rayon::iter::{ParallelIterator,};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use std::path::Path;

/// A single Physarum agent. The x and y positions are continuous, hence we use floating point
/// numbers instead of integers.
#[derive(Debug)]
struct Agent {
    x: f32,
    y: f32,
    angle: f32,
    population_id: usize,
    i: usize,
}

impl Agent {
    /// Construct a new agent with random parameters.
    fn new<R: Rng + ?Sized>(width: usize, height: usize, id: usize, rng: &mut R, i: usize) -> Self {
        let (x, y, angle) = rng.gen::<(f32, f32, f32)>();
        Agent {
            x: x * width as f32,
            y: y * height as f32,
            angle: angle * TAU,
            population_id: id,
            i: i,
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
        self.angle = wrap(self.angle + delta_angle, TAU);
        self.x = wrap(self.x + step_distance * self.angle.cos(), width as f32);
        self.y = wrap(self.y + step_distance * self.angle.sin(), height as f32);
    }
}

impl Clone for Agent {
    fn clone(&self) -> Agent {
        return Agent {
            x: self.x,
            y: self.y,
            angle: self.angle,
            population_id: self.population_id,
            i: self.i,
        }
    }
}

impl PartialEq for Agent {
    fn eq(&self, other: &Self) -> bool {
        return self.x == other.x && self.y == other.y && self.angle == other.angle && self.population_id == other.population_id && self.i == other.i;
    }
}


/// Top-level simulation class.
pub struct Model {
    // Physarum agents.
    agents: Vec<Agent>,

    // The grid they move on.
    grids: Vec<Grid>,

    // Attraction table governs interaction across populations
    attraction_table: Vec<Vec<f32>>,

    // Global grid diffusivity.
    diffusivity: usize,

    // Current model iteration.
    iteration: i32,

    palette: Palette,

    // List of ImgData to be processed post-simulation into images
    img_data_vec: Vec<ImgData>,
}

impl Model {
    const ATTRACTION_FACTOR_MEAN: f32 = 1.0;
    const ATTRACTION_FACTOR_STD: f32 = 0.1;
    const REPULSION_FACTOR_MEAN: f32 = -1.0;
    const REPULSION_FACTOR_STD: f32 = 0.1;

    pub fn print_configurations(&self) {
        for (i, grid) in self.grids.iter().enumerate() {
            println!("Grid {}: {}", i, grid.config);
        }
        println!("Attraction table: {:#?}", self.attraction_table);
    }

    /// Construct a new model with random initial conditions and random configuration.
    pub fn new(
        width: usize,
        height: usize,
        n_particles: usize,
        n_populations: usize,
        diffusivity: usize,
    ) -> Self {
        let particles_per_grid = (n_particles as f64 / n_populations as f64).ceil() as usize;
        let n_particles = particles_per_grid * n_populations;

        let mut rng = rand::thread_rng();

        let attraction_distr =
            Normal::new(Self::ATTRACTION_FACTOR_MEAN, Self::ATTRACTION_FACTOR_STD).unwrap();
        let repulstion_distr =
            Normal::new(Self::REPULSION_FACTOR_MEAN, Self::REPULSION_FACTOR_STD).unwrap();

        let mut attraction_table = Vec::with_capacity(n_populations);
        for i in 0..n_populations {
            attraction_table.push(Vec::with_capacity(n_populations));
            for j in 0..n_populations {
                attraction_table[i].push(if i == j {
                    attraction_distr.sample(&mut rng)
                } else {
                    repulstion_distr.sample(&mut rng)
                });
            }
        }

        Model {
            agents: (0..n_particles)
                .map(|i| Agent::new(width, height, i / particles_per_grid, &mut rng, i))
                .collect(),
            grids: (0..n_populations)
                .map(|_| Grid::new(width, height, &mut rng))
                .collect(),
            attraction_table,
            diffusivity,
            iteration: 0,
            palette: random_palette(),
            img_data_vec: Vec::new(),
        }
    }

    fn pick_direction<R: Rng + ?Sized>(center: f32, left: f32, right: f32, rng: &mut R) -> f32 {
        if (center > left) && (center > right) {
            return 0.0;
        } else if (center < left) && (center < right) {
            return *[-1.0, 1.0].choose(rng).unwrap();
        } else if left < right {
            return 1.0;
        } else if right < left {
            return -1.0;
        }
        return 0.0;
    }

    /// Simulates `steps` # of steps
    pub fn run(&mut self, steps: usize) {
        let debug: bool = false;

        let pb = ProgressBar::new(steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} {percent}%, {per_sec})",
                )
                .progress_chars("#>-"),
        );

        let mut time_per_agent_list: Vec<f64> = Vec::new();
        let mut time_per_step_list: Vec<f64> = Vec::new();
        for i in 0..steps {
            if debug {println!("Starting tick for all agents...")};

            // Combine grids
            let grids = &mut self.grids;
            combine(grids, &self.attraction_table);
            let agents_tick_time = Instant::now();
            self.agents.par_iter_mut().for_each(|agent| {
                // let i: usize = agent.i;

                let grid = &grids[agent.population_id];
                let (width, height) = (grid.width, grid.height);
                let PopulationConfig {
                    sensor_distance,
                    sensor_angle,
                    rotation_angle,
                    step_distance,
                    ..
                } = grid.config;

                let xc = agent.x + agent.angle.cos() * sensor_distance;
                let yc = agent.y + agent.angle.sin() * sensor_distance;
                
                let agent_add_sens = agent.angle + sensor_angle;
                let agent_sub_sens = agent.angle - sensor_angle;

                let xl = agent.x + agent_sub_sens.cos() * sensor_distance;
                let yl = agent.y + agent_sub_sens.sin() * sensor_distance;
                let xr = agent.x + agent_add_sens.cos() * sensor_distance;
                let yr = agent.y + agent_add_sens.sin() * sensor_distance;

                // Sense. We sense from the buffer because this is where we previously combined data
                // from all the grid.
                let trail_c = grid.get_buf(xc, yc);
                let trail_l = grid.get_buf(xl, yl);
                let trail_r = grid.get_buf(xr, yr);

                // Rotate and move
                let mut rng = rand::thread_rng();
                let direction = Model::pick_direction(trail_c, trail_l, trail_r, &mut rng);
                agent.rotate_and_move(direction, rotation_angle, step_distance, width, height);
            });


            // Deposit
            for agent in self.agents.iter() {
                self.grids[agent.population_id].deposit(agent.x, agent.y);
            }

            // Diffuse + Decay
            let diffusivity = self.diffusivity;
            self.grids.par_iter_mut().for_each(|grid| {
                grid.diffuse(diffusivity);
            });

            self.save_image_data();

            let agents_tick_elapsed: f64 = agents_tick_time.elapsed().as_millis() as f64;
            let ms_per_agent: f64 = (agents_tick_elapsed as f64) / (self.agents.len() as f64);
            time_per_agent_list.push(ms_per_agent);
            time_per_step_list.push(agents_tick_elapsed);

            if debug {
                println!("Finished tick for all agents. took {}ms\nTime per agent: {}ms\n", agents_tick_elapsed, ms_per_agent);
            }

            self.iteration += 1;
            pb.set_position(i as u64);
        }
        pb.finish();

        let avg_per_step: f64 = time_per_step_list.iter().sum::<f64>() as f64 / time_per_step_list.len() as f64;
        let avg_per_agent: f64 = time_per_agent_list.iter().sum::<f64>() as f64 / time_per_agent_list.len() as f64;
        println!("Average time per step: {}ms\nAverage time per agent: {}ms", avg_per_step, avg_per_agent);
    }

    fn save_image_data(&mut self) {
        let grids = self.grids.clone();
        let img_data = ImgData::new(grids, self.palette, self.iteration);
        self.img_data_vec.push(img_data);
        if self.grids[0].width > 1024 && self.grids[0].height > 1024 {
            if self.img_data_vec.len() > 100 {
                self.render_all_imgdata();
                self.flush_image_data();
                return;
            }
        }
        
    }

    pub fn flush_image_data(&mut self) {
        self.img_data_vec.clear();
    }

    pub fn render_all_imgdata(&self) {
        if !Path::new("./tmp").exists() {
            std::fs::create_dir("./tmp");
        }

        let pb = ProgressBar::new(self.img_data_vec.len() as u64);
        pb.set_style(ProgressStyle::default_bar().template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, {percent}%, {per_sec})",
        ));

        /*
        for img in &self.img_data_vec {
            Self::save_to_image(img.to_owned());
            pb.inc(1);
        }
        pb.finish();
        */

        (&self.img_data_vec).par_iter().progress_with(pb)
            .for_each(|img| {
                Self::save_to_image(img.to_owned());
            });
    }

    pub fn save_to_image(imgdata: ImgData) {
        let (width, height) = (imgdata.grids[0].width, imgdata.grids[0].height);
        let mut img = image::RgbImage::new(width as u32, height as u32);

        let max_values: Vec<_> = imgdata
            .grids
            .iter()
            .map(|grid| grid.quantile(0.999) * 1.5)
            .collect();

        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let (mut r, mut g, mut b) = (0.0_f32, 0.0_f32, 0.0_f32);
                for (grid, max_value, color) in
                    multizip((&imgdata.grids, &max_values, &imgdata.palette.colors)) {
                    let mut t = (grid.data()[i] / max_value).clamp(0.0, 1.0);
                    t = t.powf(1.0 / 2.2); // gamma correction
                    r += color.0[0] as f32 * t;
                    g += color.0[1] as f32 * t;
                    b += color.0[2] as f32 * t;
                }
                r = r.clamp(0.0, 255.0);
                g = g.clamp(0.0, 255.0);
                b = b.clamp(0.0, 255.0);
                img.put_pixel(x as u32, y as u32, image::Rgb([r as u8, g as u8, b as u8]));
            }
        }

    
        img.save(format!("./tmp/out_{}.png", imgdata.iteration).as_str()).unwrap();
    }
}
