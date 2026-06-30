use std::future::Future;
use std::sync::OnceLock;
use tokio::runtime::Runtime;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn;

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create Tokio runtime"))
}

/// Drive `future` to completion on the shared multi-threaded Tokio runtime.
///
/// This blocks the calling thread until the future resolves and must therefore
/// be called from *outside* the runtime (e.g. a Python worker thread spawned via
/// `asyncio.to_thread`). Calling it from within a Tokio task would panic.
///
/// Unlike [`tokio::task::spawn`], `block_on` does not require the future to be
/// `'static` or `Send`: the future is polled on the current thread and cannot
/// outlive this call. That is exactly what lets [`crate::async_pipeline`] hand a
/// borrowed (non-`'static`) view of the output `numpy` buffer into the futures
/// that fill it.
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    runtime().block_on(future)
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
