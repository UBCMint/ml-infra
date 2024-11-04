use prometheus::{Encoder, TextEncoder, register_int_gauge, IntGauge};
use std::sync::Arc;

// Define CPU and Memory metrics
pub struct Metrics {
    pub cpu_usage: IntGauge,
    pub memory_usage: IntGauge,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            cpu_usage: register_int_gauge!("node_cpu_usage", "Node CPU usage").unwrap(),
            memory_usage: register_int_gauge!("node_memory_usage", "Node memory usage").unwrap(),
        }
    }

    pub fn gather_metrics(&self) -> String {
        let encoder = TextEncoder::new();
        let mut buffer = vec![];
        let metrics = prometheus::gather();
        encoder.encode(&metrics, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}