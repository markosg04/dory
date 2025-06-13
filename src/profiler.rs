use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ProfileData {
    pub count: u64,
    pub total_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
}

impl ProfileData {
    fn new(duration: Duration) -> Self {
        Self {
            count: 1,
            total_duration: duration,
            min_duration: duration,
            max_duration: duration,
        }
    }

    fn update(&mut self, duration: Duration) {
        self.count += 1;
        self.total_duration += duration;
        if duration < self.min_duration {
            self.min_duration = duration;
        }
        if duration > self.max_duration {
            self.max_duration = duration;
        }
    }

    pub fn avg_duration(&self) -> Duration {
        if self.count > 0 {
            self.total_duration / self.count as u32
        } else {
            Duration::ZERO
        }
    }
}

static PROFILER: OnceLock<Mutex<HashMap<String, ProfileData>>> = OnceLock::new();

pub struct ProfilerGuard {
    label: String,
    start: Instant,
}

impl ProfilerGuard {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            start: Instant::now(),
        }
    }
}

impl Drop for ProfilerGuard {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        let profiler = PROFILER.get_or_init(|| Mutex::new(HashMap::new()));

        if let Ok(mut map) = profiler.lock() {
            match map.get_mut(&self.label) {
                Some(data) => data.update(duration),
                None => {
                    map.insert(self.label.clone(), ProfileData::new(duration));
                }
            }
        }
    }
}

pub fn profile<T>(label: impl Into<String>, f: impl FnOnce() -> T) -> T {
    let _guard = ProfilerGuard::new(label);
    f()
}

pub fn print_profile_report() {
    let profiler = PROFILER.get_or_init(|| Mutex::new(HashMap::new()));

    if let Ok(map) = profiler.lock() {
        if map.is_empty() {
            println!("No profiling data collected");
            return;
        }

        println!("\n=== PROFILING REPORT ===");
        println!(
            "{:<40} {:<8} {:<12} {:<12} {:<12} {:<12}",
            "Label", "Count", "Total (ms)", "Avg (ms)", "Min (ms)", "Max (ms)"
        );
        println!("{}", "-".repeat(108));

        let mut entries: Vec<_> = map.iter().collect();
        entries.sort_by(|a, b| b.1.total_duration.cmp(&a.1.total_duration));

        for (label, data) in entries {
            println!(
                "{:<40} {:<8} {:<12.3} {:<12.3} {:<12.3} {:<12.3}",
                label,
                data.count,
                data.total_duration.as_secs_f64() * 1000.0,
                data.avg_duration().as_secs_f64() * 1000.0,
                data.min_duration.as_secs_f64() * 1000.0,
                data.max_duration.as_secs_f64() * 1000.0
            );
        }
        println!("{}", "=".repeat(108));
    }
}

pub fn clear_profile_data() {
    let profiler = PROFILER.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut map) = profiler.lock() {
        map.clear();
    }
}

// Convenience macro for profiling
#[macro_export]
macro_rules! profile_span {
    ($label:expr, $code:block) => {
        $crate::profiler::profile($label, || $code)
    };
}
