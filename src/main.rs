use physarum::model;

fn main() {
    let mut model = model::Model::new(128, 128, 4096, 1);
    println!("{:#?}", model);
    model.step();
    model.step();
    model.save_to_image();
}
