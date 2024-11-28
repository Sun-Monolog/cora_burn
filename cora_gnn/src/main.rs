use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;
use cora_gnn::graph::model::GNNConfig;
use cora_gnn::graph::train::{train, TrainingConfig};

fn main() {
    // const CORA_NODE_NUM : usize = 2708;
    // const CORA_EDGE_NUM : usize = 5429;
    const CORA_NODE_EMBEDDING_LEN: usize = 1433;

    type MyBacked = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBacked>;

    let device = WgpuDevice::default();
    let artifact_dir = "./tmp/cora";
    train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(GNNConfig::new(CORA_NODE_EMBEDDING_LEN, CORA_NODE_EMBEDDING_LEN, CORA_NODE_EMBEDDING_LEN, 7), AdamConfig::new()),
        device.clone()
    );
}