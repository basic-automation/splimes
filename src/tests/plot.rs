use std::io;

use bigdecimal::ToPrimitive;

use crate::Point;

pub mod plot {
	use super::{io, Point, ToPrimitive};

	#[allow(dead_code)] // This function is used in tests but may not be called during normal clippy runs
	#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
	pub fn plot_terminal(title: &str, points: Vec<Point>) -> io::Result<()> {
		if points.is_empty() {
			return Ok(());
		}

		// Use a fixed width for the ASCII chart
		let chart_width: usize = 60;
		let chart_height: usize = 15;

		// Convert to chart data
		let data: Vec<f64> = points.iter().map(|p| p.value.to_f64().unwrap_or(0.0)).collect();

		// Find bounds
		let min_y = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
		let max_y = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
		let range = max_y - min_y;

		// Create ASCII chart
		println!("\n┌{}┐", "─".repeat(chart_width.saturating_sub(2)));
		println!("│{:^width$}│", title, width = chart_width.saturating_sub(2));
		println!("├{}┤", "─".repeat(chart_width.saturating_sub(2)));

		// Render chart lines
		for row in 0..chart_height {
			// Fix casting issue
			#[allow(clippy::cast_precision_loss)]
			let y_threshold = max_y - (range * row as f64 / chart_height as f64);
			let mut line = String::new();

			// Sample data points to fit chart width
			// Fix casting issue
			#[allow(clippy::cast_precision_loss)]
			let step = if data.len() > chart_width.saturating_sub(2) { data.len() as f64 / (chart_width.saturating_sub(2)) as f64 } else { 1.0 };

			for i in 0..(chart_width.saturating_sub(2)) {
				// Fix casting issue
				#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
				let data_index = (i as f64 * step) as usize;
				if data_index < data.len() {
					let value = data[data_index];
					if value >= y_threshold {
						line.push('█');
					} else {
						line.push(' ');
					}
				} else {
					line.push(' ');
				}
			}

			// Fix format string
			println!("│{line}│");
		}

		println!("└{}┘", "─".repeat(chart_width.saturating_sub(2)));
		println!("Min: {:.6}, Max: {:.6}, Points: {}", min_y, max_y, data.len());

		Ok(())
	}
}
