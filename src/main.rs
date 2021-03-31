use physarum::model;

fn main() {
    // # of iterations to go through
    let n_iterations = 2048;

    // Size of grid and pictures
    // let (width, height) = (256, 256);
    let (width, height) = (1024, 1024);

    // # of agents
    let n_particles = 1 << 20;
    println!("n_particles: {}", n_particles);

    let diffusivity = 1;

    // `n_populations` is the # of types of agents
    let n_populations = 1;
    // let n_populations = 1 + rng.gen_range(1..4); // make # of populations between 2 and 5

    let mut model = model::Model::new(width, height, n_particles, n_populations, diffusivity); // Create the model

    model.print_configurations(); // Print config for model

    model.run(n_iterations); // Actually run the model

    // export saved image data
    println!("Rendering all saved image data....");
    model.render_all_imgdata();
    model.flush_image_data();
    println!("Done!");
}
