use indicatif::{ProgressBar, ProgressStyle};
use physarum::model;
use rand::Rng;

fn main() {
    let n_iterations = 400;
    let pb = ProgressBar::new(n_iterations);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .progress_chars("#>-"),
    );

    let (width, height) = (1024, 1024);
    let n_particles = 1 << 22;
    let diffusivity = 1;
    let n_populations = 1 + rand::thread_rng().gen_range(1..4);
    let mut model = model::Model::new(width, height, n_particles, n_populations, diffusivity);
    model.print_configurations();

    for i in 0..n_iterations {
        model.step();
        pb.set_position(i);
    }
    pb.finish_with_message("Finished!");
    model.save_to_image("out.png");
}
