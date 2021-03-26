use crate::{
    grid::{combine, Grid, PopulationConfig},
    palette::{random_palette, Palette},
};


use rayon::iter::{ParallelIterator, IntoParallelIterator};

// for file stuff
use std::fs;
use std::io::{BufRead, Write, BufReader};
use std::fs::File;
use std::path::Path;
use std::fs::OpenOptions;

/*
fn get_resumed_primes(file_path: &str) -> Vec<i32> {
    let path = Path::new(file_path);
    let lines = lines_from_file(path);

    let resumed_primes = lines.par_iter().map(|x| {
        return str::replace(&str::replace(x, "Invalid: ", ""), "Prime: ", "").parse::<i32>().unwrap();
    }).collect();
    return resumed_primes;
}
*/

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
        }
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