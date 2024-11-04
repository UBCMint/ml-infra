use tauri::{Builder, generate_context};
use prometheus::{Encoder, IntGauge, Opts, Registry, TextEncoder};
use std::sync::Arc;
use std::sync::Mutex;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use warp::{Filter};

#[tokio::main]
async fn main() {
    // Initialize Prometheus registry
    let registry = Registry::new();

    // Register metrics
    let cpu_usage = IntGauge::with_opts(Opts::new("node_cpu_usage", "Node CPU usage")).unwrap();
    let memory_usage = IntGauge::with_opts(Opts::new("node_memory_usage", "Node memory usage")).unwrap();
    registry.register(Box::new(cpu_usage.clone())).unwrap();
    registry.register(Box::new(memory_usage.clone())).unwrap();

    // Wrap the registry in an Arc<Mutex> for safe sharing across threads
    let registry = Arc::new(Mutex::new(registry));

    // Set up HTTP server for Prometheus metrics
    let metrics_route = warp::path("metrics").map({
        let registry = Arc::clone(&registry);
        move || {
            let mut buffer = Vec::new();
            let encoder = TextEncoder::new();
            let registry = registry.lock().unwrap();
            encoder.encode(&registry.gather(), &mut buffer).unwrap();
            String::from_utf8(buffer).unwrap()
        }
    });

    // Run the HTTP server in a separate async task
    tokio::spawn(async move {
        let address = SocketAddr::from(([127, 0, 0, 1], 3000));
        warp::serve(metrics_route).run(address).await;
    });

    // Initialize the Tauri application
    Builder::default()
        .setup(|app| {
            // Access Tauri App handle for any setup-related configurations if needed
            let handle = app.handle();
            
            // Example: Simulating metrics updates (In a real app, use actual data)
            let cpu_usage = cpu_usage.clone();
            let memory_usage = memory_usage.clone();
            std::thread::spawn(move || {
                loop {
                    cpu_usage.set(rand::random::<i64>() % 100); // Dummy CPU usage
                    memory_usage.set(rand::random::<i64>() % 100); // Dummy memory usage
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
            });

            Ok(())
        })
        .run(generate_context!())
        .expect("error while running Tauri application");
}
