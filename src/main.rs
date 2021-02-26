mod blur;
mod grid;
mod model;
mod trig;

fn main() {
    let model = model::Model::new(4, 4, 20, 1);
    println!("{:#?}", model);
}
