pub mod cubic;
pub mod gpu_config;
pub mod linear;
pub mod plot;
pub mod polynomial;
pub mod quadratic;
pub mod regression;

#[cfg(test)]
pub use plot::plot::plot_terminal;
