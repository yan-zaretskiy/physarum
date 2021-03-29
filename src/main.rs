use chrono::{DateTime, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use physarum::model;
use rand::Rng;
use arrayfire as af;

fn main() {
    let gpu_compute: bool = false;
    if gpu_compute {
        backend_man();
        // af::set_backend(af::Backend::CPU);
        af::set_device(0);
        af::info();
    }

    // let n_iterations = 16384;
    let n_iterations = 100;
    // let n_iterations = 10;

    let (width, height) = (512, 512);
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
    let n_populations = 1;
    let mut model = model::Model::new(width, height, n_particles, n_populations, diffusivity);
    model.print_configurations();

    if gpu_compute {
        model.run_cl(n_iterations);
    } else {
        model.run(n_iterations);
    }
    

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
