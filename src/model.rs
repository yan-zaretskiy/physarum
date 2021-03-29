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
use std::time::{Duration, Instant};
use rayon::iter::{ParallelIterator, IntoParallelIterator};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use arrayfire as af;
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
        let debug: bool = true;

        let pb = ProgressBar::new(steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} {percent}%, {per_sec})",
                )
                .progress_chars("#>-"),
        );

        for i in 0..steps {
            if debug {println!("Starting tick for all agents...")};

            // Combine grids
            let grids = &mut self.grids;
            combine(grids, &self.attraction_table);
            let agents_tick_time = Instant::now();
            self.agents.par_iter_mut().for_each(|agent| {
                let i: usize = agent.i;

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

            if debug {
                let agents_tick_elapsed = agents_tick_time.elapsed().as_millis();
                let ms_per_agent: f64 = (agents_tick_elapsed as f64) / (self.agents.len() as f64);
                println!("Finished tick for all agents. took {}ms\nTime per agent: {}ms\n", agents_tick_time.elapsed().as_millis(), ms_per_agent);
            }
            

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
            self.iteration += 1;
            pb.set_position(i as u64);
        }
        pb.finish();
    }

    // Currently VERY poorly implemented (allocates memory each iteration)
    // I need to learn more about gpu compute to tackle this one
    pub fn run_cl(&mut self, steps: usize) {
        let pb = ProgressBar::new(steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} {percent}%, {per_sec})",
                )
                .progress_chars("#>-"),
        );

        // Combine grids
        let grids = &mut self.grids;
        combine(grids, &self.attraction_table);

        let agents_list = &*self.agents.clone();


        let agent_num: usize = agents_list.len() as usize;
        let dims = af::Dim4::new(&[agent_num as u64, 1, 1, 1]);


        let mut sensor_distance_list: Vec<f32> = Vec::new();
        let mut sensor_angle_list: Vec<f32> = Vec::new();
        let mut rotation_angle_list: Vec<f32> = Vec::new();
        let mut step_distance_list: Vec<f32> = Vec::new();

        // Need to fix, super slow
        for agent in &*self.agents.clone() {
            let PopulationConfig {
                sensor_distance,
                sensor_angle,
                rotation_angle,
                step_distance,
                ..
            } = &grids.clone()[agent.population_id].config;
            sensor_distance_list.push(*sensor_distance);
            sensor_angle_list.push(*sensor_angle);
            rotation_angle_list.push(*rotation_angle);
            step_distance_list.push(*step_distance);
        }

        let sensor_distance = af::Array::new(&sensor_distance_list, dims);
        let sensor_angle = af::Array::new(&sensor_angle_list, dims);

        

        let mut agent_angles_list: Vec<f32> = Vec::new();
        let mut agent_x_list: Vec<f32> = Vec::new();
        let mut agent_y_list: Vec<f32> = Vec::new();

        for i in 0..steps {
            let grids = &mut self.grids;
            combine(grids, &self.attraction_table);

            println!("Starting tick for all agents...");
            let agents_tick_time = Instant::now();
            agent_angles_list = agents_list.iter().map(|agent| agent.angle).collect();
            agent_x_list = agents_list.iter().map(|agent| agent.x).collect();
            agent_y_list = agents_list.iter().map(|agent| agent.y).collect();


            let agent_x = af::Array::new(&agent_x_list, dims);
            let agent_y = af::Array::new(&agent_y_list, dims);
            let agent_angles = af::Array::new(&agent_angles_list, dims);

            let cos_angles = af::cos(&agent_angles);
            let sin_angles = af::sin(&agent_angles);

            let cos_angle_dis = af::mul(&cos_angles, &sensor_distance, false);
            let sin_angle_dis = af::mul(&sin_angles, &sensor_distance, false);

            let xc_array = &af::add(&agent_x, &cos_angle_dis, false);
            let yc_array = &af::add(&agent_y, &sin_angle_dis, false);

            let xc = Self::to_vec(xc_array);
            let yc = Self::to_vec(yc_array);
            
            let agent_add_sens = af::add(&agent_angles, &sensor_angle, false);
            let agent_sub_sens = af::sub(&agent_angles, &sensor_angle, false);

            let agent_add_sens_mul = af::mul(&agent_add_sens, &sensor_distance, false);
            let agent_sub_sens_mul = af::mul(&agent_sub_sens, &sensor_distance, false);

            let xl_array = &af::add(&agent_x, &af::sin(&agent_sub_sens_mul), false);
            let yl_array = &af::add(&agent_y, &af::sin(&agent_sub_sens_mul), false);
            let xr_array = &af::add(&agent_x, &af::sin(&agent_add_sens_mul), false);
            let yr_array = &af::add(&agent_y, &af::sin(&agent_add_sens_mul), false);

            let xl = Self::to_vec(xl_array);
            let yl = Self::to_vec(yl_array);
            let xr = Self::to_vec(xr_array);
            let yr = Self::to_vec(yr_array);

            
            self.agents.par_iter_mut().for_each(|agent| {
                let i: usize = agent.i;

                let rotation_angle = rotation_angle_list[i];
                let step_distance = rotation_angle_list[i];

                let xc = xc[i];
                let xl = xl[i];
                let xr = xr[i];
                let yc = yc[i];
                let yl = yl[i];
                let yr = yr[i];

                let grid = &grids[agent.population_id];
                let (width, height) = (grid.width, grid.height);
                
                let trail_c = grid.get_buf(xc, yc);
                let trail_l = grid.get_buf(xl, yl);
                let trail_r = grid.get_buf(xr, yr);

                let mut rng = rand::thread_rng();
                let direction = Model::pick_direction(trail_c, trail_l, trail_r, &mut rng);
                agent.rotate_and_move(direction, rotation_angle, step_distance, width, height);
            });

            let agents_tick_elapsed = agents_tick_time.elapsed().as_millis();
            let ms_per_agent: f64 = (agents_tick_elapsed as f64) / (self.agents.len() as f64);
            println!("Finished tick for all agents. took {}ms\nTime per agent: {}ms\n", agents_tick_time.elapsed().as_millis(), ms_per_agent);

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
            self.iteration += 1;
            pb.set_position(i as u64);
        }
        pb.finish();
    }
    
    fn to_vec<T:af::HasAfEnum+Default+Clone>(array: &af::Array<T>) -> Vec<T> {
        let mut vec = vec!(T::default();array.elements());
        array.host(&mut vec);
        return vec;
    }

    fn save_image_data(&mut self) {
        let grids = self.grids.clone();
        self.img_data_vec.push(ImgData::new(grids, self.palette, self.iteration));
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
