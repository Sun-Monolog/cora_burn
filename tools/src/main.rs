use rayon::iter::ParallelIterator;
use rayon::iter::IndexedParallelIterator;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::prelude::TensorData;
use burn::tensor::{Float, Tensor};
use rayon::prelude::ParallelSliceMut;
use cora_gnn::graph::data::GraphData;

fn main() {
    const CORA_NODE_NUM : usize = 2708;
    const CORA_EDGE_NUM : usize = 5429;
    const CORA_NODE_EMBEDDING_DIM : usize = 1433;

    let content_file = File::open("./data/cora.content").unwrap();
    let buf_content_file = std::io::BufReader::new(content_file);

    let cites_file = File::open("./data/cora.cites").unwrap();
    let buf_cites_file = std::io::BufReader::new(cites_file);

    let mut node_array: Vec<f32> = vec![0.0; CORA_NODE_EMBEDDING_DIM * CORA_NODE_NUM];

    let mut edge_vec: Vec<[usize; 2]> = Vec::new();

    let mut node_tensor: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut node_labels: Vec<usize> = Vec::new();

    let mut label_map = HashMap::new();
    let mut label_map_back = HashMap::new();
    let mut label_count = 0usize;

    buf_content_file.lines().for_each(|line| {
        let line = line.unwrap();
        let mut line: Vec<&str> = line.split_whitespace().collect();
        line.remove(line.len() - 1);
        let node_label = line.first().unwrap().parse::<usize>().unwrap();
        label_map.insert(node_label, label_count);
        label_map_back.insert(label_count, node_label);
        line.remove(0);

        let node_features: Vec<f32> = line.iter().map(|x| x.parse::<f32>().unwrap()).collect();

        node_tensor.insert(label_count, node_features);
        node_labels.push(node_label);

        label_count += 1;
    });

    buf_cites_file.lines().for_each(|line| {
        let line = line.unwrap();
        let line: Vec<&str> = line.split_whitespace().collect();
        let src_node = line[0].parse::<usize>().unwrap();
        let dst_node = line[1].parse::<usize>().unwrap();
        edge_vec.push([label_map[&src_node], label_map[&dst_node]]);
    });

    // node_tensor.iter().for_each(|(a, _)|{
    //     println!("{a}");
    // });

    node_array.par_chunks_mut(CORA_NODE_EMBEDDING_DIM).enumerate().for_each(|(idx, node)| {
        // println!("{:?}", idx);

        let node_features = node_tensor.get(&idx).unwrap().to_owned();
        node.copy_from_slice(node_features.as_slice());

    });

    let node_tensor: Tensor<Wgpu, 2, Float> = Tensor::from_data(TensorData::new(node_array, [CORA_NODE_NUM, CORA_NODE_EMBEDDING_DIM]), &WgpuDevice::default());
    let edge_tensor: Tensor<Wgpu, 2, Float> = Tensor::zeros([CORA_EDGE_NUM, CORA_NODE_EMBEDDING_DIM], &WgpuDevice::default());
    let global_tensor: Tensor<Wgpu, 1, Float> = Tensor::zeros([CORA_NODE_EMBEDDING_DIM], &WgpuDevice::default());
    // println!("{:#?}", node_tensor);
    let graph_data: GraphData<Wgpu> = GraphData::from_data(node_tensor, edge_tensor, global_tensor, edge_vec, CORA_NODE_NUM, CORA_NODE_EMBEDDING_DIM, CORA_NODE_EMBEDDING_DIM);

    graph_data.save("./data/cora.ghd".to_string())
}
