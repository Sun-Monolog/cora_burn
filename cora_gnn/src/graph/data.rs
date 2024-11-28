use burn::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use burn::data::dataloader::batcher::Batcher;

#[derive(Debug)]
pub struct GraphData<
    B,
> where
    B: Backend,
{
    pub node_tensor: Tensor<B, 2>,
    pub edge_tensor: Tensor<B, 2>,
    pub global_tensor: Tensor<B, 1>,
    edges: Vec<[usize; 2]>,
    node_neighbour_map: HashMap<usize, HashSet<usize>>,
    node_edge_map: HashMap<usize, HashSet<usize>>,
    node_len: usize,
    node_embedding_size: usize,
    edge_embedding_size: usize,
}

#[allow(dead_code)]
impl<
    B,
> GraphData<B>
where
    B: Backend,
    Tensor<B, 1>: std::ops::Add<Output = Tensor<B, 1>>,
    Tensor<B, 2>: std::ops::Add<Tensor<B, 2>, Output = Tensor<B, 2>>,
{
    pub fn from_data(
        node_tensor: Tensor<B, 2>,
        edge_tensor: Tensor<B, 2>,
        global_tensor: Tensor<B, 1>,
        edges: Vec<[usize; 2]>,
        node_len: usize,
        node_embedding_size: usize,
        edge_embedding_size: usize
    ) -> GraphData<B> {
        let mut node_neighbour_map: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut node_edge_map: HashMap<usize, HashSet<usize>> = HashMap::new();

        edges.iter().enumerate().for_each(|(i, pair)| {
            if node_neighbour_map.contains_key(&pair[0]) {
                node_neighbour_map
                    .get_mut(&pair[0])
                    .unwrap()
                    .insert(pair[1]);
            } else {
                let mut new_set: HashSet<usize> = HashSet::new();
                new_set.insert(pair[1]);
                node_neighbour_map.insert(pair[0], new_set);
            }
            if node_neighbour_map.contains_key(&pair[1]) {
                node_neighbour_map
                    .get_mut(&pair[1])
                    .unwrap()
                    .insert(pair[0]);
            } else {
                let mut new_set: HashSet<usize> = HashSet::new();
                new_set.insert(pair[0]);
                node_neighbour_map.insert(pair[1], new_set);
            }

            if node_edge_map.contains_key(&pair[0]) {
                node_edge_map.get_mut(&pair[0]).unwrap().insert(i);
            } else {
                let mut new_set = HashSet::new();
                new_set.insert(i);
                node_edge_map.insert(pair[0], new_set);
            }
            if node_edge_map.contains_key(&pair[1]) {
                node_edge_map.get_mut(&pair[1]).unwrap().insert(i);
            } else {
                let mut new_set = HashSet::new();
                new_set.insert(i);
                node_edge_map.insert(pair[1], new_set);
            }
        });

        GraphData {
            node_tensor,
            edge_tensor,
            global_tensor,
            edges,
            node_neighbour_map,
            node_edge_map,
            node_len,
            node_embedding_size,
            edge_embedding_size,
        }
    }

    pub fn get_node_len(&self) -> usize {
        self.node_len
    }

    pub fn get_node_embedding_size(&self) -> usize {
        self.node_embedding_size
    }

    pub fn get_edge_embedding_size(&self) -> usize {
        self.edge_embedding_size
    }

    pub fn temp_trans(&mut self) {
        println!("temp_trans:{:#?}", self.node_tensor.shape());

        let node_tensor_chunks = self.node_tensor.clone().iter_dim(0);

        let mut new_node_tensor_chunks: Vec<Tensor<B, 2>> = Vec::new();

        node_tensor_chunks.enumerate().for_each(|(index, chunk)| {
            let mut temp_neighbour_tensors: Vec<_> = Vec::new();

            temp_neighbour_tensors.push(chunk.clone());

            match self.node_neighbour_map.get(&index) {
                None => {}
                Some(temp_set) => {
                    // for i in temp_set {
                    //     temp_neighbour_tensors.push(self.node_tensor.clone().slice([*i..(*i+1), 0..NODE_EMBEDDING_SIZE]));
                    // }
                    let mut temp_vec = temp_set
                        .par_iter()
                        .map(|i| {
                            self.node_tensor
                                .clone()
                                .slice([*i..(*i + 1), 0..self.node_embedding_size])
                        })
                        .collect::<Vec<_>>();
                    temp_neighbour_tensors.append(&mut temp_vec);
                }
            }

            let neighbour_count = temp_neighbour_tensors.len();

            let mut result_tensor = temp_neighbour_tensors
                .into_iter()
                .reduce(|acc, x| acc + x)
                .unwrap();
            result_tensor = result_tensor / neighbour_count as f32;

            new_node_tensor_chunks.push(result_tensor);
        });

        self.node_tensor = Tensor::cat(new_node_tensor_chunks, 0);
    }

    pub fn node_to_node_trans(&mut self) {
        let node_tensor_chunks = self.node_tensor.clone().iter_dim(0);

        let mut new_node_tensor_chunks: Vec<Tensor<B, 2>> = Vec::new();

        node_tensor_chunks.enumerate().for_each(|(index, chunk)| {
            let mut temp_neighbour_tensors: Vec<_> = Vec::new();

            temp_neighbour_tensors.push(chunk.clone());

            match self.node_neighbour_map.get(&index) {
                None => {}
                Some(temp_set) => {
                    for i in temp_set {
                        temp_neighbour_tensors.push(
                            self.node_tensor
                                .clone()
                                .slice([*i..(*i + 1), 0..self.node_embedding_size]),
                        );
                    }
                }
            }

            let neighbour_count = temp_neighbour_tensors.len();

            let mut result_tensor = temp_neighbour_tensors
                .into_iter()
                .reduce(|acc, x| acc + x)
                .unwrap();
            result_tensor = result_tensor / neighbour_count as f32;

            new_node_tensor_chunks.push(result_tensor);
        });

        self.node_tensor = Tensor::cat(new_node_tensor_chunks, 0);
    }

    pub fn edge_to_node_trans(&mut self) {
        let node_tensor_chunks = self.node_tensor.clone().iter_dim(0);

        let mut new_node_tensor_chunks: Vec<Tensor<B, 2>> = Vec::new();

        node_tensor_chunks.enumerate().for_each(|(index, chunk)| {
            let mut temp_neighbour_tensors: Vec<_> = Vec::new();

            temp_neighbour_tensors.push(chunk.clone());

            if self.node_embedding_size != self.edge_embedding_size {
                todo!()
            } else {
                match self.node_edge_map.get(&index) {
                    None => {}
                    Some(temp_set) => {
                        for i in temp_set {
                            temp_neighbour_tensors.push(self.edge_tensor.clone().slice([*i..(*i + 1), 0..self.edge_embedding_size]));
                        }
                    }
                }

                let neighbour_count = temp_neighbour_tensors.len();

                let mut result_tensor = temp_neighbour_tensors
                    .into_iter()
                    .reduce(|acc, x| acc + x)
                    .unwrap();
                result_tensor = result_tensor / neighbour_count as f32;

                new_node_tensor_chunks.push(result_tensor);
            }
        });

        self.node_tensor = Tensor::cat(new_node_tensor_chunks, 0);
    }

    pub fn node_to_edge_trans(&mut self) {
        let edge_tensor_chunks = self.edge_tensor.clone().iter_dim(0);
        let mut new_edge_tensor_chunks: Vec<Tensor<B, 2>> = Vec::new();

        edge_tensor_chunks.enumerate().for_each(|(index, chunk)| {
            let mut new_edge_chunk = chunk.clone();

            let node_a_label = self.edges[index][0];
            let node_b_label = self.edges[index][1];

            let node_a_tensor = self
                .node_tensor
                .clone()
                .slice([node_a_label..(node_a_label + 1), 0..self.node_embedding_size]);
            let node_b_tensor = self
                .node_tensor
                .clone()
                .slice([node_b_label..(node_b_label + 1), 0..self.node_embedding_size]);

            if self.node_embedding_size != self.edge_embedding_size {
                todo!()
            }

            new_edge_chunk = new_edge_chunk + node_a_tensor;
            new_edge_chunk = new_edge_chunk + node_b_tensor;
            new_edge_chunk = new_edge_chunk / 3;
            new_edge_tensor_chunks.push(new_edge_chunk);
        });

        self.edge_tensor = Tensor::cat(new_edge_tensor_chunks, 0);
    }

    pub fn global_to_node_trans(&mut self) {
        let node_tensor_chunks = self.node_tensor.clone().iter_dim(0);

        let temp_reshaped_global_tensor = self.global_tensor.clone().reshape([1, self.node_embedding_size]);

        let mut new_node_tensor_chunks: Vec<Tensor<B, 2>> = Vec::new();

        node_tensor_chunks.for_each(|chunk| {
            let new_node_tensor = chunk + temp_reshaped_global_tensor.clone();

            new_node_tensor_chunks.push(new_node_tensor);
        });

        self.node_tensor = Tensor::cat(new_node_tensor_chunks, 0);
    }

    pub fn node_to_global_trans(&mut self) {
        let mut node_tenser_sum = self.node_tensor.clone().sum();

        node_tenser_sum = node_tenser_sum + self.global_tensor.clone();

        node_tenser_sum = node_tenser_sum / (self.node_len as f32 + 1.0);

        self.global_tensor = node_tenser_sum;
    }

    pub fn global_to_edge_trans(&mut self) {
        let edge_tensor_chunks = self.edge_tensor.clone().iter_dim(0);

        let mut new_edge_tensor_chunks: Vec<Tensor<B, 2>> = Vec::new();

        let temp_reshaped_global_tensor = self.global_tensor.clone().reshape([1, self.node_embedding_size]);

        if self.node_embedding_size != self.edge_embedding_size {
            todo!()
        } else {
            edge_tensor_chunks.for_each(|chunk| {
                let mut new_edge_tensor = chunk + temp_reshaped_global_tensor.clone();

                new_edge_tensor = new_edge_tensor / 2f32;

                new_edge_tensor_chunks.push(new_edge_tensor);
            });
        }

        self.edge_tensor = Tensor::cat(new_edge_tensor_chunks, 0);
    }

    pub fn edge_to_global_trnas(&mut self) {
        if self.node_embedding_size != self.edge_embedding_size {
            todo!();
        } else {
            let edge_count = self.edges.len() as f32;

            let mut edge_tensor_sum = self.edge_tensor.clone().sum();

            edge_tensor_sum = edge_tensor_sum + self.global_tensor.clone();

            edge_tensor_sum = edge_tensor_sum / (edge_count + 1.0);

            self.global_tensor = edge_tensor_sum;
        }
    }

    pub fn save(&self, path: String) {
        let mut file = std::fs::File::create(path).unwrap();

        let node_data = self.node_tensor.to_data();
        let edge_data = self.edge_tensor.to_data();
        let global_data = self.global_tensor.to_data();

        let node_bin = bincode::serialize(&node_data).unwrap();
        let edge_bin = bincode::serialize(&edge_data).unwrap();
        let global_bin = bincode::serialize(&global_data).unwrap();
        let edges_bin = bincode::serialize(&self.edges).unwrap();
        let node_neighbour_map_bin = bincode::serialize(&self.node_neighbour_map).unwrap();
        let node_edge_map_bin = bincode::serialize(&self.node_edge_map).unwrap();

        let node_len = node_bin.len() as u32;
        let edge_len = edge_bin.len() as u32;
        let global_len = global_bin.len() as u32;
        let edges_len = edges_bin.len() as u32;
        let neighbour_map_len = node_neighbour_map_bin.len() as u32;
        let node_edge_map_len = node_edge_map_bin.len() as u32;

        file.write(&self.node_len.to_le_bytes()).unwrap();

        let temp = self.node_embedding_size.to_le_bytes();
        file.write(&self.node_embedding_size.to_le_bytes()).unwrap();
        file.write(&self.edge_embedding_size.to_le_bytes()).unwrap();

        file.write(&node_len.to_le_bytes()).unwrap();
        file.write(&edge_len.to_le_bytes()).unwrap();
        file.write(&global_len.to_le_bytes()).unwrap();
        file.write(&edges_len.to_le_bytes()).unwrap();
        file.write(&neighbour_map_len.to_le_bytes()).unwrap();
        file.write(&node_edge_map_len.to_le_bytes()).unwrap();

        file.write(&node_bin).unwrap();
        file.write(&edge_bin).unwrap();
        file.write(&global_bin).unwrap();
        file.write(&edges_bin).unwrap();
        file.write(&node_neighbour_map_bin).unwrap();
        file.write(&node_edge_map_bin).unwrap();
    }

    pub fn load(
        path: String,
    ) -> GraphData<B> {
        let mut file = std::fs::File::open(path).unwrap();

        let mut node_len_bin = [0u8; 8];
        let mut node_embedding_size_bin = [0u8; 8];
        let mut edge_embedding_size_bin = [0u8; 8];

        let mut node_tensor_len_bin = [0u8; 4];
        let mut edge_tensor_len_bin = [0u8; 4];
        let mut global_tensor_len_bin = [0u8; 4];
        let mut edges_len_bin = [0u8; 4];
        let mut map_len_bin = [0u8; 4];
        let mut node_edge_map_len_bin = [0u8; 4];

        file.read(&mut node_len_bin).unwrap();
        file.read(&mut node_embedding_size_bin).unwrap();
        file.read(&mut edge_embedding_size_bin).unwrap();

        file.read(&mut node_tensor_len_bin).unwrap();
        file.read(&mut edge_tensor_len_bin).unwrap();
        file.read(&mut global_tensor_len_bin).unwrap();
        file.read(&mut edges_len_bin).unwrap();
        file.read(&mut map_len_bin).unwrap();
        file.read(&mut node_edge_map_len_bin).unwrap();

        let node_len = u64::from_le_bytes(node_len_bin) as usize;
        let node_embedding_size = u64::from_le_bytes(node_embedding_size_bin) as usize;
        let edge_embedding_size = u64::from_le_bytes(edge_embedding_size_bin) as usize;

        let node_tensor_len = u32::from_le_bytes(node_tensor_len_bin) as usize;
        let edge_tensor_len = u32::from_le_bytes(edge_tensor_len_bin) as usize;
        let global_tensor_len = u32::from_le_bytes(global_tensor_len_bin) as usize;
        let edges_len = u32::from_le_bytes(edges_len_bin) as usize;
        let map_len = u32::from_le_bytes(map_len_bin) as usize;
        let node_edge_map_len = u32::from_le_bytes(node_edge_map_len_bin) as usize;

        let mut node_bin = vec![0u8; node_tensor_len];
        let mut edge_bin = vec![0u8; edge_tensor_len];
        let mut global_bin = vec![0u8; global_tensor_len];
        let mut edges_bin = vec![0u8; edges_len];
        let mut map_bin = vec![0u8; map_len];
        let mut node_edge_map_bin = vec![0u8; node_edge_map_len];

        file.read(&mut node_bin).unwrap();
        file.read(&mut edge_bin).unwrap();
        file.read(&mut global_bin).unwrap();
        file.read(&mut edges_bin).unwrap();
        file.read(&mut map_bin).unwrap();
        file.read(&mut node_edge_map_bin).unwrap();

        let node_data: TensorData = bincode::deserialize(&node_bin).unwrap();
        let edge_data: TensorData = bincode::deserialize(&edge_bin).unwrap();
        let global_data: TensorData = bincode::deserialize(&global_bin).unwrap();
        let edges: Vec<[usize; 2]> = bincode::deserialize(&edges_bin).unwrap();
        let node_neighbour_map: HashMap<usize, HashSet<usize>> =
            bincode::deserialize(&map_bin).unwrap();
        let node_edge_map: HashMap<usize, HashSet<usize>> =
            bincode::deserialize(&node_edge_map_bin).unwrap();

        let node_tensor = Tensor::from_data(node_data, &B::Device::default());
        let edge_tensor = Tensor::from_data(edge_data, &B::Device::default());
        let global_tensor = Tensor::from_data(global_data, &B::Device::default());

        GraphData {
            node_tensor,
            edge_tensor,
            global_tensor,
            edges,
            node_neighbour_map,
            node_edge_map,
            node_len,
            node_embedding_size,
            edge_embedding_size,
        }
    }
}

impl <B: Backend> Clone for GraphData<B>{
    fn clone(&self) -> Self {
        Self{
            node_tensor: self.node_tensor.clone(),
            edge_tensor: self.node_tensor.clone(),
            global_tensor: self.global_tensor.clone(),
            edges: self.edges.clone(),
            node_neighbour_map: self.node_neighbour_map.clone(),
            node_edge_map: self.node_edge_map.clone(),
            node_len: self.node_len,
            node_embedding_size: self.node_embedding_size,
            edge_embedding_size: self.edge_embedding_size,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GraphItem{
    pub label: usize,
    pub features: [f32; 1433],
    pub target: i32,
    pub neighbours: HashSet<usize>
}


#[derive(Clone, Debug)]
pub struct GraphDataBatch<B: Backend> {
    pub graph_data: GraphData<B>,
    pub targets: Tensor<B, 1, Int>
}

#[derive(Clone)]
pub struct GraphDataBatcher<B: Backend>{
    device: B::Device,
}

impl <B: Backend> GraphDataBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self{ device }
    }
}


impl <B: Backend> Batcher<GraphItem, GraphDataBatch<B>> for GraphDataBatcher<B> {
    fn batch(&self, items: Vec<GraphItem>) -> GraphDataBatch<B> {
        let mut sorted_items: Vec<_> = items.into_iter().collect();
        sorted_items.sort_by(|a, b| a.label.cmp(&b.label));

        let node_data: Vec<_> = sorted_items.iter().map(|x| Tensor::<B, 1>::from_data(x.features, &B::Device::default())).collect();
        let node_tensor: Tensor<B, 2> = Tensor::stack(node_data, 0);
        let edge_tensor: Tensor<B, 2> = Tensor::zeros([5429, 1433], &B::Device::default());
        let global_tensor: Tensor<B, 1> = Tensor::zeros([1433], &B::Device::default());

        let targets_data: Vec<_> = sorted_items.iter().map(|x| Tensor::<B, 1, Int>::from_data([x.target], &B::Device::default())).collect();
        let targets = Tensor::cat(targets_data, 0);

        let mut edges: Vec<[usize; 2]> = Vec::new();

        sorted_items.iter().for_each(|x|{
            let current_label = x.label;

            x.neighbours.iter().for_each(|nei_label|{
                edges.push([current_label, *nei_label]);
            });
        });

        let graph_data = GraphData::from_data(node_tensor, edge_tensor, global_tensor, edges, 2708, 1433, 1433);

        println!("{:#?}", graph_data.node_tensor.shape());

        GraphDataBatch{
            graph_data,
            targets,
        }
    }
}