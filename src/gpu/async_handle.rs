use anyhow::Result;

/// Result of GPU interpolation work
///
/// Contains the computed interpolation results.
/// In the current implementation, this is eagerly computed before returning.
/// Future versions may support true async/await with deferred computation.
#[allow(dead_code)] // Infrastructure for future async GPU operations
pub struct GpuInterpolationResult<T> {
	results: Vec<T>,
}

#[allow(dead_code)] // Infrastructure for future async GPU operations
impl<T: Clone> GpuInterpolationResult<T> {
	/// Create a new result from computed values
	pub const fn new(results: Vec<T>) -> Self {
		Self { results }
	}

	/// Get the results
	pub fn into_results(self) -> Vec<T> {
		self.results
	}

	/// Borrow the results
	pub fn results(&self) -> &[T] {
		&self.results
	}
}

/// Clone results
impl<T: Clone> Clone for GpuInterpolationResult<T> {
	fn clone(&self) -> Self {
		Self { results: self.results.clone() }
	}
}

/// Implement `IntoFuture` for async/await support
///
/// Currently this resolves immediately, but the infrastructure is in place
/// for future async GPU operations that could enable better CPU-GPU parallelism.
impl<T: Clone + Send + 'static> std::future::IntoFuture for GpuInterpolationResult<T> {
	type IntoFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send>>;
	type Output = Result<Vec<T>>;

	fn into_future(self) -> Self::IntoFuture {
		Box::pin(async move { Ok(self.results) })
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_result_creation() {
		let results = vec![1.0, 2.0, 3.0];
		let result = GpuInterpolationResult::new(results);
		assert_eq!(result.results(), &[1.0, 2.0, 3.0]);
	}

	#[test]
	fn test_result_into_results() {
		let results = vec![1.0, 2.0, 3.0];
		let result = GpuInterpolationResult::new(results.clone());
		assert_eq!(result.into_results(), results);
	}

	#[test]
	fn test_result_clone() {
		let results = vec![1.0, 2.0, 3.0];
		let result = GpuInterpolationResult::new(results);
		let cloned = result;
		assert_eq!(cloned.results(), &[1.0, 2.0, 3.0]);
	}
}
