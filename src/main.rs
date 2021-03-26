use chrono::{DateTime, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use physarum::model;
use rand::Rng;
use arrayfire as af;

fn main() {
    backend_man();
    // af::set_backend(af::Backend::CPU);
    af::set_device(0);
    af::info();

    // let n_iterations = 16384;
    let n_iterations = 2024;
    // let n_iterations = 10;

    let (width, height) = (512, 512);
    // let (width, height) = (1024, 1024);
    // let (width, height) = (2048, 2048);

    // let n_particles = 1 << 22;
    let n_particles = 1 << 24;
    // let n_particles = 100;
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

    // let n_populations = 1 + rng.gen_range(1..4);
    let n_populations = 2;
    let mut model = model::Model::new(width, height, n_particles, n_populations, diffusivity);
    model.print_configurations();

    // let dims = af::Dim4::new(&[n_particles as u64, 1, 1, 1]);
    for i in 0..n_iterations {
        model.step();
        // model.step_cl(dims);
        pb.set_position(i);
    }
    pb.finish();

    println!("Rendering all saved image data....");
    model.render_all_imgdata();
    model.flush_image_data();
    println!("Done!");
}


fn backend_man() {
    let available = af::get_available_backends();

    if available.contains(&af::Backend::CUDA) {
        println!("Evaluating CUDA Backend...");
        af::set_backend(af::Backend::CUDA);
        println!("There are {} CUDA compute devices", af::device_count());
        return;
    }

    /*
    if available.contains(&af::Backend::OPENCL) {
        println!("Evaluating OpenCL Backend...");
        af::set_backend(af::Backend::OPENCL);
        println!("There are {} OpenCL compute devices", af::device_count());
        return;
    }
    */

    if available.contains(&af::Backend::CPU) {
        println!("Evaluating CPU Backend...");
        af::set_backend(af::Backend::CPU);
        println!("There are {} CPU compute devices", af::device_count());
        return;
    }
}
