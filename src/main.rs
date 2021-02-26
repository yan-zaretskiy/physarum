use physarum::model;

fn main() {
    let model = model::Model::new(4, 4, 20, 1);
    println!("{:#?}", model);
}
