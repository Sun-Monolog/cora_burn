use crate::graph::data::{GraphData, GraphDataBatch};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};

#[derive(Module, Debug)]
pub struct GNN<B: Backend> {
    node_mlp: MLP<B>,
    edge_mlp: MLP<B>,
    global_mlp: MLP<B>,
    output_linear: Linear<B>,
}

impl<B: Backend> GNN<B> {
    pub fn forward(&self, mut graph_data: GraphData<B>) -> Tensor<B, 2> {
        let node_embedding_size = graph_data.get_node_embedding_size();

        for _ in 0..3 {
            graph_data.node_to_edge_trans();
            println!("node_to_edge");
            graph_data.edge_to_node_trans();
            println!("edge_to_node");

            graph_data.node_to_global_trans();
            println!("node_to_global");
            graph_data.global_to_node_trans();
            println!("global_to_node");

            graph_data.edge_to_global_trnas();
            println!("edge_to_global");
            graph_data.global_to_edge_trans();
            println!("global_to_edge");

            let node_tensor = graph_data.node_tensor.clone();
            let edge_tensor = graph_data.edge_tensor.clone();
            let global_tensor = graph_data.global_tensor.clone();

            let node_tensor = self.node_mlp.forward(node_tensor);
            let edge_tensor = self.edge_mlp.forward(edge_tensor);
            let global_tensor = self
                .global_mlp
                .forward(global_tensor.reshape([1, node_embedding_size]))
                .reshape([node_embedding_size]);

            graph_data.node_tensor = node_tensor;
            graph_data.edge_tensor = edge_tensor;
            graph_data.global_tensor = global_tensor;
        }

        let x = graph_data.clone();
        let x = self.output_linear.forward(x.node_tensor);
        let x = softmax(x, 0);
        println!("forward_now");
        x
    }

    pub fn forward_classification(
        &self,
        graph_data: GraphData<B>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(graph_data);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<GraphDataBatch<B>, ClassificationOutput<B>> for GNN<B> {
    fn step(&self, item: GraphDataBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item.graph_data, item.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<GraphDataBatch<B>, ClassificationOutput<B>> for GNN<B> {
    fn step(&self, item: GraphDataBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item.graph_data, item.targets)
    }
}

#[derive(Config, Debug)]
pub struct GNNConfig {
    node_size: usize,
    edge_size: usize,
    global_size: usize,
    class_num: usize,
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
}

impl GNNConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GNN<B> {
        GNN {
            node_mlp: MLPConfig::new(self.node_size, self.node_size, self.node_size).init(device),
            edge_mlp: MLPConfig::new(self.edge_size, self.edge_size, self.edge_size).init(device),
            global_mlp: MLPConfig::new(self.global_size, self.global_size, self.global_size)
                .init(device),
            output_linear: LinearConfig::new(self.node_size, self.class_num).init(device),
        }
    }
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear3.forward(input);
        let x = self.linear2.forward(x);
        let x = relu(x);
        self.linear3.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct MLPConfig {
    hidden_size: usize,
    input_size: usize,
    output_size: usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        MLP {
            linear1: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            linear3: LinearConfig::new(self.hidden_size, self.output_size).init(device),
        }
    }
}
