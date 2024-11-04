use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use std::sync::Arc;
use crate::metrics::Metrics;

async fn serve_metrics(req: Request<Body>, metrics: Arc<Metrics>) -> Result<Response<Body>, hyper::Error> {
    if req.uri().path() == "/metrics" {
        let metric_data = metrics.gather_metrics();
        Ok(Response::new(Body::from(metric_data)))
    } else {
        Ok(Response::new(Body::from("Not Found")))
    }
}

pub async fn start_http_server(metrics: Arc<Metrics>) {
    let addr = ([127, 0, 0, 1], 8000).into();
    let make_svc = make_service_fn(|_| {
        let metrics = metrics.clone();
        async move { Ok::<_, hyper::Error>(service_fn(move |req| serve_metrics(req, metrics.clone()))) }
    });
    let server = Server::bind(&addr).serve(make_svc);
    println!("Server running on http://{}", addr);
    if let Err(e) = server.await {
        eprintln!("Server error: {}", e);
    }
}