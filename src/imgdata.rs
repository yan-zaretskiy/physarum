use crate::{grid::Grid, palette::Palette};

// Class for storing data that will be used to create images
pub struct ImgData {
    pub grids: Vec<Grid>,
    pub palette: Palette,
    pub iteration: i32,
}

impl Clone for ImgData {
    fn clone(&self) -> ImgData {
        return ImgData {
            grids: self.grids.clone(),
            palette: self.palette.clone(),
            iteration: self.iteration.clone(),
        };
    }
}

impl ImgData {
    pub fn new(in_grids: Vec<Grid>, in_palette: Palette, in_iteration: i32) -> Self {
        ImgData {
            grids: in_grids,
            palette: in_palette,
            iteration: in_iteration,
        }
    }
}
