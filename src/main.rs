mod blur;
mod grid;
mod model;

fn main() {
    let boxes = blur::boxes_for_gaussian::<3>(2.5);
    println!("boxes: {:?}", boxes);
}
