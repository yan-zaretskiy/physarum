use chrono::{DateTime, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use physarum::model;
use rand::Rng;

fn main() {
    // let n_iterations = 16384;
    let n_iterations = 10;
    // let n_iterations = 100;
    // let n_iterations = 10;

    let (width, height) = (256, 256);
    // let (width, height) = (512, 512);
    // let (width, height) = (1024, 1024);
    // let (width, height) = (2048, 2048);

    let n_particles = 1 << 22;
    // let n_particles = 1 << 10;
    // let n_particles = 1 << 20;
    // let n_particles = 100;
    println!("n_particles: {}", n_particles);
    let diffusivity = 1;
    let mut rng = rand::thread_rng();

    // let n_populations = 1 + rng.gen_range(1..4);
    let n_populations = 2;
    let mut model = model::Model::new(width, height, n_particles, n_populations, diffusivity);
    model.print_configurations();

    model.run(n_iterations);
    

    println!("Rendering all saved image data....");
    model.render_all_imgdata();
    model.flush_image_data();
    println!("Done!");
}