//! Resamples irregular sensor readings onto a one-second grid and prints the series with
//! each point's provenance, then the report of how it was computed.
//!
//! ```text
//! cargo run --example resample
//! ```
//!
//! Readings come as `RFC 3339 timestamp,value` lines on standard input when it is not a
//! terminal, in any order, or else from a built-in sample:
//!
//! ```text
//! printf '2026-01-01T00:00:00Z,10\n2026-01-01T00:00:07.5Z,14\n' | cargo run --example resample
//! ```

use std::io::{IsTerminal, Read};

use chrono::{DateTime, TimeDelta};
use splimes::{Error, Interpolator, PointKind, Resolution, Spline};

const SAMPLE: &str = "\
2026-01-01T00:00:00Z,10.0
2026-01-01T00:00:07Z,14.2
2026-01-01T00:00:19.250Z,11.1
2026-01-01T00:00:30Z,20.4
2026-01-01T00:00:31.5Z,19.8
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let mut text = String::new();
	if !std::io::stdin().is_terminal() {
		std::io::stdin().read_to_string(&mut text)?;
	}
	if text.trim().is_empty() {
		text = SAMPLE.to_owned();
	}
	let (mut timestamps, mut values) = (Vec::new(), Vec::new());
	for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
		let (t, v) = line.split_once(',').ok_or_else(|| format!("line {}: expected `timestamp,value`", n + 1))?;
		timestamps.push(DateTime::parse_from_rfc3339(t.trim())?.to_utc());
		values.push(v.trim().parse::<f64>()?);
	}
	let (Some(&first), Some(&last)) = (timestamps.iter().min(), timestamps.iter().max()) else {
		return Err("no readings".into());
	};

	// A cubic spline, a value every second, two seconds past the last reading. With
	// `exact(true)`, too few readings for a cubic is an error instead of a step down, and
	// `max_points` refuses a grid that would be unreasonably large before allocating it.
	let interpolator = Interpolator::new(Spline::Cubic, Resolution::Seconds).exact(true).max_points(1_000_000);
	let series = match interpolator.run_f64(&timestamps, &values, first, last + TimeDelta::seconds(2)) {
		Err(Error::InsufficientPoints { required, available, .. }) => {
			return Err(format!("a cubic spline needs {required} distinct readings; got {available}").into());
		}
		result => result?,
	};

	for (timestamp, value, kind) in series.iter() {
		let mark = match kind {
			PointKind::Raw => "  <- reading",
			PointKind::Extrapolated => "  (extrapolated)",
			PointKind::Interpolated => "",
		};
		println!("{} {value:>10.4}{mark}", timestamp.format("%H:%M:%S"));
	}
	let count = |k| series.kinds().iter().filter(|&&kind| kind == k).count();
	println!("\n{} points: {} raw, {} interpolated, {} extrapolated; {} on {} in {}", series.len(), count(PointKind::Raw), count(PointKind::Interpolated), count(PointKind::Extrapolated), series.spline(), series.backend(), series.precision(),);
	Ok(())
}
