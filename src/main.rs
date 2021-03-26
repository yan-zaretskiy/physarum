use chrono::{DateTime, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use physarum::model;
use rand::Rng;

fn main() {
    // let n_iterations = 16384;
    let n_iterations = 4096;
    // let n_iterations = 10;

    // let (width, height) = (512, 512);
    let (width, height) = (1024, 1024);
    // let (width, height) = (2048, 2048);

    let n_particles = 1 << 22;
    println!("n_particles: {}", n_particles);
    let diffusivity = 1;
    let mut rng = rand::thread_rng();

    let pb = ProgressBar::new(n_iterations);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} {percent}%, {per_sec})",
            )
            .progress_chars("#>-"),
    );

    let n_populations = 1 + rng.gen_range(1..4);
    let mut model = model::Model::new(width, height, n_particles, n_populations, diffusivity);
    model.print_configurations();

    for i in 0..n_iterations {
        model.step();
        pb.set_position(i);
    }
    pb.finish();

    println!("Rendering all saved image data....");
    model.render_all_imgdata();
    model.flush_image_data();
    println!("Done!");
}
