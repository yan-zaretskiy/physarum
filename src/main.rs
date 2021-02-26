use physarum::model;

fn main() {
    let mut model = model::Model::new(4, 4, 20, 1);
    println!("{:#?}", model);
    model.step();
    println!("{:#?}", model);
}
