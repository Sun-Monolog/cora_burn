use rayon::iter::ParallelIterator;
use std::string::String;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufRead;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::data::dataset::transform::{Mapper, MapperDataset};
use rayon::prelude::IntoParallelIterator;
use crate::graph::data::GraphItem;

pub struct CoraDataset{
    dataset: InMemDataset<GraphItem>,
}

impl Dataset<GraphItem> for CoraDataset {
    fn get(&self, index: usize) -> Option<GraphItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl CoraDataset {
    pub fn new(path: &str) -> Self{
        let mut cites_path = path.to_string();
        cites_path.push_str(".cites");
        let mut content_path = path.to_string();
        content_path.push_str(".content");

        const CORA_NODE_NUM : usize = 2708;
        const CORA_EDGE_NUM : usize = 5429;
        const CORA_NODE_EMBEDDING_DIM : usize = 1433;

        let content_file = File::open(content_path).unwrap();
        let buf_content_file = std::io::BufReader::new(content_file);

        let cites_file = File::open(cites_path).unwrap();
        let buf_cites_file = std::io::BufReader::new(cites_file);

        let mut label_map = HashMap::new();
        let mut label_map_back = HashMap::new();
        let mut label_count = 0usize;
        let mut name_label_map: HashMap<String, i32> = Default::default();
        let mut node_neighbour_map: HashMap<usize, HashSet<usize>> = Default::default();
        let mut node_target_map: HashMap<usize, i32> = HashMap::new();
        let mut node_features_map: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut name_count = 0i32;

        buf_content_file.lines().for_each(|line| {
            let line = line.unwrap();
            let mut line: Vec<&str> = line.split_whitespace().collect();
            let label_name = line.remove(line.len() - 1).to_string();

            if !name_label_map.contains_key(&label_name) {
                name_label_map.insert(label_name.clone(), name_count);
                name_count += 1;
            }

            let name_label = name_label_map[&label_name];

            let node_label = line.first().unwrap().parse::<usize>().unwrap();
            label_map.insert(node_label, label_count);
            label_map_back.insert(label_count, node_label);
            line.remove(0);

            let node_features: Vec<f32> = line.iter().map(|x| x.parse::<f32>().unwrap()).collect();

            node_features_map.insert(label_count, node_features);
            node_target_map.insert(label_count, name_label);
            node_neighbour_map.insert(label_count, HashSet::new());

            label_count += 1;
        });

        buf_cites_file.lines().for_each(|line| {
            let line = line.unwrap();
            let line: Vec<&str> = line.split_whitespace().collect();
            let src_node = line[0].parse::<usize>().unwrap();
            let dst_node = line[1].parse::<usize>().unwrap();

            node_neighbour_map.get_mut(&label_map[&src_node]).unwrap().insert(label_map[&dst_node]);
        });

        let graph_item_vec = (0..CORA_NODE_NUM).into_par_iter().map(|i|{
            GraphItem{
                label: i,
                features: <[f32; 1433]>::try_from(node_features_map[&i].clone()).unwrap(),
                target: node_target_map[&i],
                neighbours: node_neighbour_map[&i].clone(),
            }
        }).collect::<Vec<GraphItem>>();

        Self{
            dataset: InMemDataset::new(graph_item_vec),
        }
    }
}