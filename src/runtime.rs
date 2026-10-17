use tokio::runtime::Runtime;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn;

/// The Tokio runtime shared with `pyo3-async-runtimes`, so the futures handed to
/// Python as awaitables and the sync store adapter use the same workers.
fn runtime() -> &'static Runtime {
    pyo3_async_runtimes::tokio::get_runtime()
}

pub struct TokioBlockOn(tokio::runtime::Handle);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

pub fn tokio_block_on() -> TokioBlockOn {
    TokioBlockOn(runtime().handle().clone())
}
